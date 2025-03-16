# 1.1 Introduction
## What is a distributed system?  

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/be7c872991258a89f7ed59f598247208102411013e49267dee8e43a518ba5e97.jpg)  

Distributed system can also be considered as a subset of parallel system But, if we use the term parallel system, we typically focus on improving the performance of the system​. In contrast, a distributed system may not be faster than its single machine counterpart​ 

Why do we use a distributed system if it is even slower than a single machine system?  

Different types of Distributed Systems -- Categorized by the Type of Interconnect Network  

(1) Single-machine Parallel System (Not distributed)   
A single server Machine 
Communication Latency: Shared Memory $==$ DRAM latency $(\sim100\mathsf{n s}){=}>$ Inter-process communication (IPC such as pipe and socket) $==$ Syscall latency $(1\sim5\upmu s)$  

(2) Rack-scale Distributed Systems  
Typically less than 20 server machines interconnected by a single Top-of-rack Switch 
Communication Latency: CXL (ns-level) $=>$ RDMA $(\sim5\upmu v)$   $=>$ Ethernet TCP $(\sim50\upmu s)$ . 
Ethernet TCP will become a bottleneck that slows down the CPU (500x slower than DRAM latency). 
It should not be a surprise your distributed system is slower than a single machine solution if its implementation highly relies on network communication  

A picture of a rack of blade servers and top-of-rack switches connecting them 

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/e35284416c93a9647371dcaa9d72709f6de210fcb5e688f83eae098a3becbc24.jpg)  

(3) (Inter Data Center) Distributed Systems -- the typical setting when we use "Distributed System"  
Depending on the size of the data center. May contain hundreds of server machines 
Communication Latency: RDMA $(\sim5\mathsf{u s}$ , higher than rack-scale when going through the aggregation switch, i.e., more hops, is needed) $=>$ Ethernet TCP $(\sim50\upmu s)$ 

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/f8d6523170bba694f18d582d8533727b56d396be6ea855569df301031673222b.jpg)  


(4) Geo-distributed Systems 
Unlimited number of server machines (only limited by the budget)  

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/10b539e9961e8240e1e3eedd187a76fa88bada2778a1687676ac6f3146f3a757.jpg)  

Communication Latency: $=>$ wide-area network (WAN) latency $==1{\sim}10\mathrm{m}s$ 
It is hard to scale-out the performance if the work heavily relies on cross-data center communications  

### Example Distributed Systems  
Almost all the famous websites and apps are built on top of distributed systems   

This class mostly focuses on distributed infrastructure services
e.g., storage services, computation services, synchronization services. These services are the basic components and used to build upper-layer business services: The business services of WeChat and Weibo will be very different, but the data analysis system underlying both WeChat and Weibo will be very similar and resembles the MapReduce system we will teach in Lec 2. Similarly, the storage system underlying both WeChat and Weibo will be very similar and resembles the GFS system we will teach in Lec 3​  

## Why do people build distributed systems?  
### Performance  
More cores $=>$ More possible parallelism; Serial Processing $=>$ Parallel Processing $=>$ Distributed Processing​   
Ideal situation: Nx machines leads to Nx throughput V.S. Practical situation: **sublinear scalability** or even worse performance  

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/1bd09e1159a42bd8e0c45ece2cdfd47cc565bf12307c0a0d3d31e0f0be98c7fe.jpg)  

But the communication cost can become a bottleneck, especially when the communication latency is high​: COST

Paper: Scalability! But at what COST? 
Configuration that Outperforms a Single Thread (COST) -- How many cores a distributed system needs to make its performance better than a manually optimized single thread program  
COST of typical JAVA/Kernel-TCP based systems are $10\sim1000$ in 2015, what a surprise!  
This is the difficulty of building a distributed system! It would be very slow if it is not implemented cleverly.
The COST of state-of-the-art systems (especially those built with code generation, vectorizations, and RDMA) are reduced to **4-10** nowadays​  

Other problems: load imbalance, lack of scalability in certain parts of the program  

### Cost (in term of money)  
Mandatory: In the current big data era, the size of data can easily surpass the capacity of a single commodity machine  

Total Cost of Ownership $(\mathsf{T C O})$ of Scale-up (use a machine with $10\times$ cores, 10x memory, 10xstorage) is typically larger than Scale-out (use a cluster of 10 machines with the same size) 
>  scale up 的总成本一般比 scale out 的总成本高许多

This is especially true in old times: 
Banks (IBM Mainframe Servers $^+$ the same software) V.S. Google (hundreds of commodity machines $^+$ dedicated optimized distributed system) 
(The Google's distributed system implementation can be viewed as the starting point of modern distributed systems  )

On nowadays, it depends, especially when the size of data is only TB-level. On nowadays, the configuration of a typical commodity machine increases from "16 cores/15 GB memory/1TB HDD" to "64 cores / 64 GB memory / 10TB SSD". 

This is the art of trade-off. There is no silver bullet . It always changes with the status quo. Distributed systems are still mandatory for PB-level data  

### Fault Tolerance  
- Tolerate faults via replication (the main focus of many following lectures) 
- Rack-scale distributed systems can tolerate the failure of a single machine​
- (Inter data center) distributed systems can tolerate the failure of a whole rack​ (This is possible because of out of power or other problems​)
- Geo-distributed systems can tolerate the failure of a whole data center, also called disaster recovery​ (Actually the most common kind of "disaster" is the breaking of optical cable)

### Others 
(also important reasons, but not fully covered in this course)  

Security -- Achieve security via isolation

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/9d4e86909eedec47460d599b325a65fd2ed67ab93630f6ade861b89fd6ebc7e5.jpg)  

Locality -- End users are physically distributed on the earth (phones, sensors). A distributed system that matches the distribution of end users can lead to better latency/cost. 

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/9a6ec7a437eef8bcb1d093f08b1576d78a62436c3e013d1bcbefab70b53b53aa.jpg)  

Regulation -- required by law (GDRP, Law of network security)  

## The Complexity of Distributed Systems  
### Fault Tolerance V.S. Consistency  
**Fault Tolerance** 
High availability: service continues despite failures, i.e, hide these failures from the application 

Related concept: SLA (can be translated to the maximum allowed outage/down time​):
$99.99\%$ SLA $=$ 52m 9.8s down time per year  
$99.999\%$ SLA $=$ 5m 13s down time per year  

1000s of servers, big network ${->}$ always something broken:
P: the possibility of a single server's failure   
$1-(1-\mathsf{P})^{N}$ : the possibility of no failure in a cluster of N servers​   
$\mathsf{P}=0.001$ (three 9 SLA for a single server), $\mathsf{N}=1000,\mathsf{1}-(\mathtt{1}\mathrm{-P})^{\wedge}\mathsf{N}=0.63$ (zero 9 SLA for the cluster)​   

Big idea: **replicated servers**. 
If one server crashes, the system can still proceed using the other(s). 
However, replication leads to the problem of consistency​  

>  解决可用性的一个方法就是复制，但复制方法又会引出一致性的问题

### Consistency  
Consistency is an abstract concept that is hard to understand:
The real meaning of consistency depends on the current environment/situation/requirements and hence is not fixed   
The consistency of a distributed system is just like the "virtue" of a person -- the more the better, but hard to define, and frequently compromised due to the consideration of other interests (especially those interests related to money).  

Consistency means **a certain agreement on what are good behaviors at semantics level** 
e.g., "Get(k) yields the value from the most recent $\mathsf{P u t}(\mathsf{k},\mathsf{V})$ ."  This is the intention of the user -- we call it the consistency of a key-value store 
There are always multiple levels of consistency, from weak to strong  

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/ab7a7855ac7c0dae6899fac0155c328febc251c625007474012aa35e16927593.jpg)  

Note that even Read-your-writes consistency is surprising hard to implement  

Paper: FlightTracker: Consistency across Read-Optimized Online Stores at Facebook-2020-OSDI
Everything becomes complex when related to achieving consistency in a distributed system!  

Why not always the strongest level of consistency?  
Not achievable if you also want to achieve the strongest level of availability: **we need replicas to achieve fault tolerance, but it is hard to make sure that all the content of these replicas are identical** (the strongest level of consistency for replication). In fact, it is **theoretically impossible** in many scenarios.  

CAP theorem for replication system:
- Consistency: all nodes see the same data at the same time   
- Availability: every request receives a response about whether it succeeded or failed   
- Partition tolerance: the system continues to operate despite arbitrary partitioning due to network failures  

CAP theorem: U can only Pick 2!  

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/184e3d6ea536dd6bc39e1845e0c1900633be6379fb739444808bf08b8169cea7.jpg)  


Note: Consistency in CAP is a theoretical criteria that requires **all** nodes see the same data at the **same** time. 
Its availability is also not the same as the commonly used term "high availability" in practice. In CAP, all the nodes should be able to answer requests from users  

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/f70d5c15efe29fbd12f175bff0c67a84e7927210da675061046a5afa435a3fc2.jpg)  

We can tolerate some replicas returning inconsistent content as long as the client can identify this inconsistent (e.g, by reading from a quorum). Thus, the existence of Paxos and RAFT does not violate CAP 
Check [CAP理论中的P到底是个什么意思?](https://www.zhihu.com/question/54105974/answer/2144019181) for more details.  

FLP theorem for asynchronous application  

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/db3c360d0c5bff5a46f768db9a15a883d16f33d2021e76eaec35db101d239e78.jpg)  

"Agreement" here is actually a kind of consistency in asynchronous application (This is the ambiguity of consistency. We use different names in different scenarios​)
Both Paxos/RAFT try to achieve agreement in a fault tolerant way, and hence their termination is not guaranteed   
But it is fine because the **possibility** of termination is exponentially increasing through certain design (although it will never become 1)​  

### Practical Problems  
Tricky to realize performance potential $=>$ hard to be efficient $=>$ because of the slow network, load imbalance, etc   

Must cope with partial failure $=>$ hard to be correct​ 
Typically, we rely on an assumption called "fail-stop", i.e., a failed node will stop its execution
Note that this is not the real case, in real, some node may disappear for a while and somehow come back $\mathrm{-}\mathrm{>}$ this can be a problem  

Many concurrent parts, complex interactions $=>$ hard to model in a human's mind  

## The Essence of This Course  
This course will discuss some of the most important distributed systems. They are the foundation of our current digital world. During the course, I will focus on the following two topics.  

### Trade-off​  
System researchers are always making trade-offs!   
The most important spectrum of trade-off -- **fault-tolerance, consistency, and performance are enemies**  

![](https://cdn-mineru.openxlab.org.cn/extract/f765e853-b5fe-414b-8968-60cd64a953d6/3e48276ec85acd091176f4e14c81f1bcc3c9c8aef25327ea89567b9878e3ac6c.jpg)  

The spiral of NoSQL from "NO SQL" to "Not Only SQL"  

### Abstractions  
Distributed systems are complex, so let's make it simple via good and higher level of abstractions 
Example abstractions: **Programming models, Master-slave architectures, Replicated State Machines​** 
Reference implementations of these abstractions will also be discussed, which can be used directly in your future works​  

# 1.2 RPC
Remote Procedure Call (RPC)  

![](https://cdn-mineru.openxlab.org.cn/extract/f2456e30-5d4e-4846-ba6c-507ffe57fbaf/f20314a60f975f4d973617d3cf2ece0c3eab6aa205873fb0d8310430443a698d.jpg)  

A distributed system consists of multiple machines interconnected via network. RPC is the most common procedure for processes on different machines to communicate with each other. 

Consider a standard function call in a single program:

![](https://cdn-mineru.openxlab.org.cn/extract/f2456e30-5d4e-4846-ba6c-507ffe57fbaf/07b63e12323b81607c78d3f1c715ae3ca18f649e9ead8e1180295a3680b7fb29.jpg)  

The input argument is passed to (and returned from) the called function by registers/stacks
What if the "main" and the "foo" function are executed in different machines?  
RPC is the first and also **one of the most basic/useful abstraction** taught in this class​. It abstract away the complexity of cross machine communication to a similar level of executing a local function call. The network issues are hidden under this abstraction.

>  RPC 通过抽象掩盖了跨机器通讯的复杂性，使得远程过程调用的形式和本地过程调用类似

## An Example of Key-value RPC in Golang  
(Code refer to the original pdf)
The code written by the user is very simple because the complexity of cross-process/machine communication is hidden by the abstraction of RPC.   
The implementation of server part is also simplified, only two functions and the complexity of handling network request, etc. are hidden​  

## RPC Protocol  
**Marshalling & Unmarshalling**

![](https://cdn-mineru.openxlab.org.cn/extract/f2456e30-5d4e-4846-ba6c-507ffe57fbaf/63a238e1c9e43a5e5e4fdea6d2a7aeebda4b55f64511bb259c2e6cd36db38183.jpg)  

Marshalling: convert an object (with type info) into a standardized byte stream (XML, JSON, protobuf, custom binary format) for transforming.  

Problem: 97 is int 97 or char 'a'? Answer: Decided by the type information 
Marshalling will embed the type info and structure of obj into byte array  

![](https://cdn-mineru.openxlab.org.cn/extract/f2456e30-5d4e-4846-ba6c-507ffe57fbaf/5410085d0f75aee848f56582344e3641f5ffea167ccb44787b53c6ded06d5940.jpg)  

Other issues: e.g., gathering and flattening the linked objects (例如链表)

Unmarshalling: convert a byte stream to objects  

**RPC Protocol/Server**  
Basic functionality: Register and dispatch of remote call 
>  定义好函数后，直接在对应的对象上注册即可

Advance functionality: failure handling, throttling, etc. 
Example: gRPC by Google, Thrift by Meta  

>  throttling 即流量控制，系统承受过多访问时，必须告知一部分客户端进行等待

## Failure Handling in RPC  
**Problem:** client never sees a response but it does not know whether the server saw the request!
>  客户端在没有收到回复的时候，可能发生的故障有以下三种情况，而具体是哪一种情况则难以判断，这就是不确定性

![](https://cdn-mineru.openxlab.org.cn/extract/f2456e30-5d4e-4846-ba6c-507ffe57fbaf/b555f2b13d26780a567be980eb5cdd6fd5688f73dc557df4f676b701f69752f3.jpg)  

**Semantics** (different levels Consistency for RPC) 

![](https://cdn-mineru.openxlab.org.cn/extract/f2456e30-5d4e-4846-ba6c-507ffe57fbaf/a3a7db2bf0a2f9b448b52c1251e71f5d71411bce6f791bcbad367ad92407728a.jpg)  

A simple at-least-once RPC can be achieved by simply resending the RPC request if no response is received for a certain timeout threshold. But it only works for **idempotent** operations.  

Idempotent: multiple executions of the same operation will produce the same result of single execution​   
Get and put are naturally idempotent because they can be executed multiple times without changing the output, but other operations such as "append", "add-one" are not idempotent.

## A better RPC that is Almost Exactly Once  
(注意是 **almost** exactly once)
Please recall the Sliding Window Protocol implemented by TCP:

![](https://cdn-mineru.openxlab.org.cn/extract/f2456e30-5d4e-4846-ba6c-507ffe57fbaf/64f6eb7ceca048fc4646195e7a17f709479a6d0601539900f081621270121079.jpg)  

Add an unique ID for each RPC request at the sender side -- maybe (machine ID $^+$ thread ID $^+$ sequence number) to ensure uniqueness   
Remember all the received RPC ID and the corresponding response at the receiver side -- simply resend the response if a duplicate request is received (without re-executing the RPC)   
Client includes "seen all replies $<=\mathsf{X}$ with every RPC -- so that the receiver side can safely forget seen responses to save space​ (Underlying assumption: the underlying network protocol should be ordered e.g., TCP)

>  发送端为每个 RPC 请求添加一个唯一的 ID
>  接收端存储所收到的 RPC 请求的 ID，数据结构可以使用 map，收到重复的 ID 时，直接用 ID 索引缓存的结果返回 (而不需要重新执行调用)  
>  发送端在请求中添加“ $\le X$ 的回复已经收到” 的信息，接收端利用这一信息清除不再可能重新发送的缓存结果

**Many many corner cases** -- the complexity of distributed systems 
How to handle duplicate requests while original is still executing? 
-- pending flag

How to handle dead clients that leads to memory leaks? 
-- heartbeat 
>  如果一段时间没有收到 heartbeat, server 就认为 client dead，进而清除掉和其相关的缓存

What if the client somehow comes back after a heartbeat timeout?  
We assume a "fail-stop" model -- a node will not come back with the same machine ID and thread ID  
This model can be achieved by incrementing the thread ID after a heartbeat timeout.

What if the client crashed and hence did not remember the thread ID allocated?  
What if the thread ID rolled back because of the maximum limit of an integer?  

Crashed server -- a restarted server may process duplicate RPC if it stores the request ID in memory, which is lost after restarting (内存中信息丢失)
Solution: 
Store the information on disk -- may lead to performance problems
High available server that uses replication -- discussed later in lecture 4  

## How to achieve exactly-once RPC?  
unbounded retries $^+$ duplicate detection $^+$ fault-tolerant service 
Also discussed later in lecture 4  

# 2 MapReduce
## Problem Definition  
We also have multiple (even thousands) machines that can serve as computation nodes so that we want solve the problem parallelly to take full advantage of these machines.  

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/83edf4b929645e84371f890162e83a56b8c277db92350a05abc2b7d23d05e9ed.jpg)  

Computation Task Property: 
We have a huge computation task that conceptually can be parallelly solved by "Divide and Conquer". (分治) For example, building a search index and analyzing the structure of the web as Google. 
Why divide and conquer? -- massively parallel sub tasks, avoid complex communication and hence mitigate the problem of communication latency​   
Fortunately, many important problems can be solved with "Divide and Conquer" , e.g., the initial task of web indexing used in Google  

Assumption: All the computation nodes are somehow attached with the same distributed file systems (lec 3). Thus, the file written by computation node A can be read by another computation node B after flushing.  

## Before MapReduce -- Message Passing Interface (MPI)  
**Solve the Problem Via Directly Programming with Socket**  
Set up a coordinator and multiple workers 
At the beginning, the coordinator partitions the problem into worker-number of subtasks  (任务划分为 worker-number 数量的子任务)
Then the coordinator sends the input of each subtask to a specific worker via the socket After the processing of each subtask, the coordinator gathers all the results and merges them into the final results.   
This is actually the "master-slave" architecture where the coordinator is the master and the works are the salves​  

**MPI**
Why do we need MPI instead of directly using sockets? -- Abstraction!!!  
Different underlying network implementations (UDP, TCP, RDMA, etc.) provide different programming interfaces, and the optimization techniques of different implementations are also very different​.   
However, intentions of the upper layer application developers when processing data in a distributed environment are actually similar -- do message passing. 
Thus, MPI provides two basic operations: `MPI_Send/MPI_Recv` 

What is the difference between `MPI_Send/MPI_Recv` and UDP send/recv? 
The underlying connection may depend on other protocols (e.g., RDMA for faster speed), but the upper layer program does not need to change because this complexity is hidden by MPI's interface 

You can view MPI as a specialized form of RPC for distributed data processing​ 

`MPI_Send/MPI_Recv` also comes with lightweight marshaling mechanism because, in HPC scenario, the most common and important data passed in messages are arrays of int/double.  

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/5f9005e4d9903ded184761c67c1cc6633e56012e537c5c092d55c678e1b1905d.jpg)  

**Further Abstraction**  -- there are some advanced but still common patterns of message passing  
Asynchronous
there is also an asynchronous version of `MPI_Send/MPI_Recv` called `MPI_Isend/MPI_Irecv/MPI_Wait` . With asynchronous interface, one can overlap the computation with communication for a better performance​  

It would be very complex to implement an asynchronous interface directly based on socket: remembering all the sent messages, maintain the states, background threads, etc., many details and corner cases​  

Barrier
e.g., assuring the finishing of a former step before the executing of the next step  


![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/ce219f2c62f1879327d8b8523c8c179191eac5d941a8617cde031af4ea43f41e.jpg)  

Broadcast and Collective Communication

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/d4b8f360167a832a696b3945a3c5732c4b8b03014e58491267b20e2a5ec39d15.jpg)  

Implementing a "Divide and Conquer" with MPI's Broadcast and Collective Communication is straightforward: First, broad cast the configuration with MPI_Bcast and then the split subtasks with MPI_Scatter, use an MPI_Barrier for waiting the finish of all the subtasks, finally using an MPI_(All)Gather to gather the results​  

>  “分治” 的 MPI 实现：`MPI_Bcast` 广播配置，`MPI_Scatter` 划分子任务，`MPI_Barrier` 同步子任务的执行，`MPI_Gather` 收集结果

**Integrating communication with computation**
Even integrating some computation -- this can be further optimized, which is very important in current machine learning workloads and implemented in GPU-version MPI libraries such as NCCL. See $^1$ for more details.  

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/e9344d0719e6ec846ade379e985e4900da79e9bbd8c1ae8c5e6e98139b78c83b.jpg)  

> 通信可以进一步和计算集成  

Reduce+broadcast V.S. Butterfly AllReduce -- optimizing the bandwidth bottleneck

There are many different kinds of implementations of allreduce 
For example: Two-rounds v.s. Butterfly.  What is the difference?   
If the size of the message is large the communication will be bounded by the network bandwidth Bwd   
In contrast, if the size of the message is small the communication will be bounded by the network latency i.e., the round trip time (RTT)   

Thus different methods are used in different scenario  

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/32e6039c104b58a73e0862a52520ee5478b8517c041d6ccffddda69004d2aad8.jpg)  

If bandwidth bounded: $2^{\star}\mathsf{N}^{\star}$ Size/Bwd  
If bandwidth bounded: log_2N \* Size/Bwd  
If latency bounded: $2^{\star}$ RTT  
If latency bounded: log_2N \* RTT  

Insufficiency of the above methods: not able to utilize all the possible aggregated bandwidth simultaneously, especially both the inbound and outbound bandwidth of the current full-duplex network card​  

Ring AllReduce if there are multiple elements -- only brought to ML area from HPC by 2017​  

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/05f6437f2f278d44ce403f9be97086b049fae94279c4ef9e41d8c0f983cafdaa.jpg)  

If bandwidth bounded: $2^{\star}$ Size/Bwd: 2N rounds, Size/N/Bwd for each round  
If latency bounded: $2^{\star}\mathsf{N}^{\star}$ RTT  

Ring AllReduce is suitable for overlapping the computation and the communication 

What if the topology of switch does not provide full mesh connection? 
-- a lot of tradeoffs/complexities  

**Cons & Pros of MPI** (compare with mapreduce later)  
Pros:
Easy to understand and relatively easy to use (for system developers but not for application developers)  
Very efficient -- Implemented as only a thin layer (no complex, dynamic, nested objects need to be serialized) above network communication, almost the same speed as the underlying network​  

Cons:
Weak fault tolerance  
The whole computation task will fail even if only one computation node crashes.  
A full re-computation is needed without any manual checkpoint. Checkpoint can be a bottleneck. The complexity of constructing a Checkpoint is not hidden by the MPI

Manual cluster management 
Hand write and fixed IP addr $^+$ No resource allocation / isolation​  
Cluster management can be a hard problem in a multi-tenant environment and is also very important for achieving a better utilization of the cluster

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/4f9775be18936a968c5613e939d69161a9f4ddc45df36180da7f6ef06ce6373f.jpg)  

Manual task partitioning
The whole task is manually partitioned into fixed number of subtasks. No elastic scaling out/in.

**Conclusion**  
MPI is still widely used in HPC scenario. It is very useful if you intend to write a specialized distributed system that needs to be very fast (e.g., a research paper PoC and it is also widely used in ML scenarios)   
MPI is not enough for our current big data era, where every programmer (even those who only know how to write "Hello World" in Java/Python) should be able to write distributed computation program

## Programming Models: Map + Reduce  
### Why Do We Need Programming Models?  
The main functionality of programming model is **restricting** the capability provided to users.

Since MPI only abstracts the communication behavior, the programmers can write arbitrary computation logic on top of MPI. But, in system design, flexibility often comes at a cost of complexity. This is one of the most important trade-offs in system research. 
e.g., the complexity of fault tolerance is inevitable if the framework gives the full flexibility on computation logic -- the system cannot track the progress of computation automatically, recovery only by manual checkpointing and full restarting​.

>  MPI 仅抽象了通讯行为，故程序员可以基于 MPI 编写任意复杂的计算逻辑，但这一灵活性的代价就是更高的复杂性，程序员需要手动管理关于容错、负载均衡等复杂逻辑

The cons discussed above indicate that only experts can directly use MPI to write distributed applications that can scale to thousands of machines (where failure always occurs and manual cluster management is hard). But there are only a limited number of distributed system experts around the corner.   

When you face complexity that comes from flexibility, you need good abstractions!​ 
The most important abstraction in computation framework is its programming model. The restriction provided by programming models will enable the system to provide simplicity via automated fault tolerance, cluster management, etc.  

>  计算框架中最重要的抽象就是其编程模型，编程模型所提供的限制使得系统可以掩盖处理许多任务——例如容错、集群管理——的复杂性

### MapReduce's Programming Model  
MapReduce's process: \[split\] $^+$ (parallel) **map** $^+$ \[shuffle\] $^+$ (parallel) **reduce** 
The programmer just need to define Map and Reduce functions, which are often fairly simple **sequential** codes. Complexities of distributed processing are hidden by the system.  

A "Word Count" example -- counting the number of occurrences for each word  
- Input: a lot of files in the real-world, a file with multiple lines in the below example
- Input split: different files for different map tasks, different words for different map tasks in the below example​  

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/c041e6df2c8fa7d477e46469abbfe3fa72e44e7cdd9716ac31061b2fa1d3e779.jpg)  

Map function: note that the results of map function are a series of (key, value) pairs  
Reduce function: it is **guaranteed** by the system that all the (key, value) pairs with the same key will go through the same reduce task (through the **shuffling**)  

### Scalability of Map/Reduce  
Scalability of Map: N Map() worker computers (might) get you Nx throughput, since they do not interact with each other   
Scalability of Reduce: The same as scale-out the number of Reduce() workers, but with a problem of load imbalance caused by data skewness (the size of some hot keys may be much much larger than the others)  

## Example of Using MapReduce -- Search Engine  
**Problem:** Is it enough to use only map and reduce in real world scenarios? Is the restriction too strict?​  

**Demonstrate the capability of MapReduce** 
**-- Basic Search Engine:  Reverse Index $+$ PageRank**  

**Reverse Index**

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/ae960f7a7cc0a66e8765e1d290069bb7eeedba51f954a40d32e34c99eeb62da9.jpg)  

This is very similar to wordcount​. Instead of the count, we emit the index position in the map and use concat in the reduce task​  

**PageRank**
What is PageRank?  

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/9808e958f94953bd6b1d99864f2f170752af2fd934b49ce57cfdda048b22859b.jpg)  

The underlying assumption of PageRank is that more important websites are likely to receive more links from other websites.  That is, webpage A contains a hyperlink to webpage B represents an "acknowledgment" from A to B  

Paper: The PageRank Citation Ranking: Bringing Order to the Web-1998  

The PageRank algorithm models the web links as a graph, each web page is a vertex and each link represents an edge, and initialize $R_{u}~=~1$ for every vertex, then repeatedly do computing until it is stable.

It is mathematically proved that there will be such a stable point (the fixed point): “The vector of PageRank values of all web pages is the fixed point of a linear transformation derived from the World Wide Web's link structure.”

The final result can be viewed as the importance of the web page.  

This is only the most basic form of PageRank. There are many many further adaptions  

PageRank in MapReduce  
Map: for each vertex, input contains the current value of this vertex and the outgoing edge list  $=>$ output: <key: destination vertex, current value divided by the degree of the vertex>. 

Reduce: sum aggregation, whose result is the new value.  

Exemplary  formula refers to the original pdf.

**Limitations of MapReduce** 
(most in the system part rather than the model part)

No iteration $=>$ Spark (lec 14, uses DRAM to buffer intermediate results) $=>$ Graph Computation Frameworks   
No Streaming $=>$ Flink (lec 15, when latency matters)  

## What is hidden by the system? 
### (1) Parallelizing & Data Transferring  
The user-defined Map/Reduce function receives input data as function parameters and sends results as function returns (just like RPC), but the actual data transferring process is rather difficult.  

Map Inputs: the input data should be split into multiple **disjoint** subtasks that are loaded by parallel map tasks to achieve scalability 
Map Outputs $=>$ Reduce Inputs (also called the intermediate results): an efficient global shuffling process is implemented, essentially a distributed bucket sort (即哈希) according to the keys 
- Many system optimizations are hidden underneath this shuffling procedure (just like the allreduce procedure discussed above). 
- There is even a famous contest for competing who can achieve the best shuffle performance​ 

Reduce Outputs: parallelly flush to disk for persistency

Note that all the above steps are based on a distributed storage file system called Google File System (lec 3)​ 

More details:

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/dd78d085efc94d410764c8b1aae90008c7d1d4e9fba5a27d09b4a15534c06089.jpg)  

Maps write output (intermediate data) to local disk. It will execute a merge sort and then split the output, by hash of the key, into one file per Reduce task (A out-of-core merge sort can be implemented to avoid out-of-memory -- also a complex hidden by MapReduce)

The output of Map is stored on GFS and hence readable by the Reduce task, but it also causes the problem of excessive I/O, which is the main optimization of Spark.

Each Reduce task writes a separate output file on GFS  

### (2) Cluster Management & Fault Tolerance  
You have a cluster of 1000 machines and a series of computation tasks that require different numbers of parallelism. Which machines should be assigned to which task? What if several machines crash? How can i identify a crash?  

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/e7628f5916eed216fc496353711c97d21baa21107b6ad6f050d03fc24a4473a9.jpg)  

Overview  
**Standard master-slave architecture.** 
The resources (CPU, memory) of worker machines are partitioned into slots and allocated to tasks according to their requirements. No manual package distribution is needed.  

There are actually two kinds of masters. 
The master coordinates resource management (allocation, reclaim): All the resources (CPU, memory, etc.) are partitioned into slots. Each machine can contain multiple slots and each slot can contain XX cores YY GB memory​. The user program will demand ZZ slots from the master  

The user program acts as the coordinator of the current program (task partitioning, dispatching, tracking, etc.)  

**Fault Tolerance** -- The availability of data is guaranteed by GFS, thus currently we consider only the failure of computation tasks  
Detection: how does the system detect a failed task? -- heartbeats
Recovery -- recovery by **fine-grained re-execution**  
    The current processing progress (NotStart/Running/Finished/Failed of each Map/Reduce task) can be automatically tracked along with the heartbeat 
    This tracking is not available for MPI programs because there is no explicit task partitioning​   
    It would be good if the system could re-execute only the failed tasks  

**Re-execution**  
When can we do fine-grained re-execution？
- Restriction enforced by the programming model: only two kinds of tasks (i.e., map/reduce) ; automatic task partitioning, hence the master is able to track progress  
- Idempotence  
    Map/Reduce tasks are idempotent mainly because they are stateless computation tasks​  
    Thus, they are idempotent as long as the users do not involve non-idempotent logics in them, e.g., random  
    It is also possible that a map/reduce task will communicate with outside environment (e.g., a database), in that case the task is not idempotent and the original fault tolerance mechanism does not work in such scenario.  (How to tolerate this kind of scenario? -- we need transaction)
- Otherwise, a global checkpoint and a whole re-execution is needed.  

Recovery after Map/Reduce worker Crashes:
Master notices a worker X no longer responds to pings (heartbeats)  
    Heartbeats are maintained between the application master (the user program in the above figure) and each of the worker  
Master knows which Map/Reduce tasks ran on worker X. These Map/Reduce tasks are recreated on other live workers which re-generate the corresponding (intermediate) results.  

Support required from the master and the underlying storage that enable the above fault tolerance procedure  -- **corner cases​**:
The master should assign a unique ID for every Map/Reduce task​  
What if the coordinator gives two workers the same Map() task?  
    It is possible if the master incorrectly thinks one worker died (e.g., due to temporary network issues).  
    The Reduce workers consumes data from only one of them (identify through ID) . The Reduce worker never reads the intermediate results from a map task that is still not Finished  
What if the coordinator gives two workers the same Reduce() task?  
    They will both try to write the same output file on GFS! $_{->}$ may lead to inconsistent results due to concurrent appending the results $_{->}$ append is not idempotent 
    The underlying storage system needs to provide an **atomic commit** procedure​:
        Atomic: "All or nothing" 
        atomic rename by GFS in MapReduce; only one complete file will be visible. 
        Reducer 1 first stores its output to some path like /temp/uuid1/reduce1 
        Reducer 2 also stores its output to some other path like /temp/uuid2/reduce1 
        Only after the finish of output, both Reducer 1 and 2 try to rename their output path to the same final path -- idempotent as long as rename is atomic  
What if a worker computes incorrect output, due to broken h/w or s/w?  
    Lead to silent failure!   
    MapReduce assumes "fail-stop" CPUs and software 
        Fail-stop: the cpu will stop execution once after a failure. This is not the real-world scenario, thus one can build custom solutions, such as "checksum", to detect silent failure. 
        Such silent failure relates to the specific business logic of the program, thus ca not be automatically tolerated by a framework  
What if the master crashes?  
    The crashed workers are recovered by the master, what about the crashed master 
    The whole system will halt​
    You need a "high available" master -- which will be taught in Lec 4-7  

### (3) Others  
**Load Imbalance**  
Wasteful and slow if N-1 servers have to wait for 1 slow server to finish. 

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/3d042a6b09858cb01ddf6155fa95da879add44d6ef4db6205e0dc7e00b6ff643.jpg)  

The problem can be mitigated if the whole task is split into many more subtasks than workers -- hopefully no task is so big it dominates completion time  

The problem is still possible if the skewness of data is high -- salting $^+$ multi-layer reducing​ 
    For each hot key K, it is first split into many sub keys K #1 , K #2 , K3, ....  
    After reducing each K \# x , a second layer reduce is used to reduce their results  
The problem is still possible if there is a straggler in the cluster 
    Workers that are unusually slow (e.g., due to some hardware problem) are called stragglers, which can become bottleneck of the whole system  The problem is that these stragglers are slow but NOT DEAD, thus they maintain their heartbeats and hence cannot be killed  
    The master can start a second copy of the last few tasks that run on stragglers
        This is called **speculative execution**   
        A trade-off between more computational cost and lower load imbalance​   
        The correctness can be guaranteed with the same mechanism discussed above (two Map)​  

**Locality**  
MapReduce was designed twenty years ago. Network was the main bottleneck at that time. Especially during the all-to-all shuffle phase, where half of traffic goes through root switch: 

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/e24a9a20a21bf90097140ecb7844e45afd3cfcbaf024c4f75cdc33e5f92976a9.jpg)  

The aggregated bandwidth of core switches will become the bottleneck during the global shuffling​   
It can be largely mitigated if most of the traffic only goes through the lower level switches -- locality  

MapReduce paper's root switch: 100 to 200 gigabits/second, total 1800 machines, so 55 megabits/second/machine, which is much much less than disk or RAM speed.  

This assumption of low-speed networks is not always true nowadays  

The MapReduce system will cooperate with the underlying GFS to achieve locality 
    The map task often (not always) reads its input from local files
    The reduce task writes its output to local disk. This locality maximizes the system's throughput​  

Use local combiner to reduce the data volume of shuffle

![](https://cdn-mineru.openxlab.org.cn/extract/3c447368-9528-44dd-b6f1-601af98b7a01/9ceb51d563097ab06d320e7d829990eb10bd4660cce3ce4277aaa400e23ffc64.jpg)  

## Conclusion  
Although the original MapReduce system is outdated and is not used in Google, it has a huge impact and leads to the development of many following systems. In a sense, the release of MapReduce starts the Big Data era.  

User needs of processing big data in certain companies (e.g., Google)  
-  $=>$ The development and release the design of MapReduce   
-  $=>$ Open-source of Hadoop MapReduce (originally also used for building open source search engine)   
-  $=>$ Ease the complexity of developing distributed computation programs 
-  $=>$ Flourishing big data ecosystem (not only for search engine but many other data analysis tasks)   
-  $=>$ More data is collected by more companies and organizations   
-  $=>\mathsf{A}$ growing flywheel  

The importance of abstraction -- making trade-off between flexibility and complexity  

# 3 Google File System
## Problem Definition  
### Why Do We Need Distributed File System?
(1) All the reasons from "why we need a distributed system" and even more​  

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/06f9b08f697c802791d112aa2240b06fcac0208d114f15b3bdf650a7fcc6c97e.jpg)  

2 replicas for each of the 3 file distributed on 3 nodes leads to 2x aggregated bandwidth and higher availability and also an expansion of 1.5 times capacity  

>  分布式文件系统的 replicas 可以为文件读取提供分布式执行的可能性，带来更高的聚合带宽和更高的可用性

(2) Performance - Take advantage of the large aggregated bandwidth of the storage cluster  
>  高的聚合带宽 -> Performance

(3) Cost - Huge datasets that exceed the capacity of a single machine   

(4) Fault Tolerance - High availability so that the data will not be lost if some of the machine crashes   
>  高的可用性 -> 容错 (一个机器崩溃，数据也不丢失)

(5) Abstraction - Reduce the complexity of other distributed systems by providing a **global namespace** for data storage 
>  为分布式数据存储提供一个全局的命名空间，像使用平常文件系统一样使用分布式文件系统

We do not need to care about the distribution and the physical location of the data. You have already seen an example of this abstraction's usage: the use of GFS in MapReduce​.
This is the art of **decoupling**: computation is decoupled from storage via this global namespace​. Without decoupling (e.g., in MPI) the failure of a node leads to data loss and hence a full re-execution is needed. With a decoupled distributed storage system that can guarantee data high availability, only fine-grained re-execution is needed because the data of finished tasks is still available.

>  MapReduce 利用了分布式文件系统，通过 GFS 提供的全局命名空间，解耦了计算和存储
>  因为 GFS 确保了已经完成的任务的数据仍然保存 (高可用性)，MapReduce 进而不需要再出现问题时重新执行所有任务，只需要细粒度执行特定任务


### Why GFS
(1) An influential system whose main architecture is still widely used nowadays (e.g., Hadoop HDFS)​ 
(2) Again, a good abstraction that balances complexity brought by the flexibility of POSIX file APIs   
(3) Exemplary distributed system paper that touches on many themes of this course, including parallel performance, fault tolerance, replication, consistency, implementation optimization -- highly recommend to read the original paper  

## File System 101  
(1) File system provides **an abstraction of file** on top of storage devices (HDD, SSD)  

(Fig ref to original pdf)
<html><body><table><tr><td colspan="6">Filesystem: read/write files</td></tr><tr><td></td><td>Blocko</td><td>Block1</td><td></td><td>BlockN-1</td><td>BlockN</td></tr><tr><td colspan="6">BlockDevice: read/write data blocks</td></tr></table></body></html>  

The under-layer storage driver provides block-device abstraction, which is conceptually a large byte vector that can only be read from and written to in a granularity of block​.
In contrast, the upper-layer file system provides an abstraction of POSIX File API, which contains a tree structure of file paths, random read/write operations, and a lot of other metadata (e.g., security related policies). 

>  底层的存储设备提供了一层 block-device 抽象，设备上的数据以数组的形式呈现，并且可以按照 block 的粒度读写
>  上层的文件系统提供了一层 POSIX API 抽象，文件系统以树状结构呈现，并带有许多源数据 (例如读写权限)，支持随机读写操作

Info
POSIX stands for Portable Operating System Interface. It's a family of standards specified by IEEE for maintaining compatibility among different operating systems.  

(2) The most basic service of a file system is **mapping** file-level APIs to device-level APIs, which is achieved by managing a set of meta data that consists of the so-called directory tree  
>  文件系统提供了 file-level 的 API (POSIX API) 到 device-level API 的映射，为此，文件系统维护了一个命名空间，它本质是一个从路径到元数据的映射，例如，将路径 `/path/to/file` 映射到一个包含了 `device ID, start offset, size` 等数据的数据块，文件系统根据这个映射，为特定路径的文件在设备上索引其内容

Fig ref to the original pdf

A real file system will be much more complex than the above model (already taught in the OS course)​   

Allocation bitmap for space allocation: a bitmap records which parts of the block device is already allocated to files​   
>  设备空间的分配情况可以通过 bitmap 表示，bitmap 记录了 block device 的哪些部分已经分配给了文件

Directory metadata that records the file/directory list contained in this directory -- used in operations like "list a directory"   
>  文件系统还会维护目录元数据，它记录了该目录下包含的文件和目录列表

(3) Crash Consistency -- **atomic** write through logging as an example

Atomic: "all or nothing", all the writers are persisted in the storage or none of them is persisted.
Atomicity is not provided by the file system, write takes time -- if there is a failure in middle, partial write is persisted​. However, it can be implemented in an upper layer with logs.

Persistency Semantics: a write is only guaranteed to be persistent after an explicit flush -- enabling the optimization of buffering  
>  写入操作不一定会直接写入磁盘，而是会暂时留在缓存，只有缓存的数据写入磁盘后，这次写入操作的数据才保证了持久性

Atomic Semantics -- implement with undo log: 
If a failure happens after the actual write, we have all the data persisted -- All If a failure happens before the actual write, we have none of the data persisted -- None   
What if a failure happens just in between the actual write? -- uses undo log to revert the updates -- None​   
Fsync is needed to flush the memory buffer and guarantee the write order -- the hard part of achieving atomicity in the real-world  

>  文件系统本身没有提供原子性写入的语义，我们可以通过日志机制实现它
>  如果在写入后才发生故障，认为数据已经持久化保存
>  如果在写入前故障，认为没有数据写入
>  如果在写入时故障，使用日志回撤，将磁盘数据回撤到写入前的状态，认为数据没有写入

Undo-logging Pseudocode that Works Correctly in Linux File Systems  

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/b9c5e750e8d6feaeabd3498ab8a8de17cbd97098ce832932136f24c407f48035.jpg)  

## Main Ideas of Distributed File System  
### Main Ideas
(1) Use the same master-slave architecture as MapReduce   

(2) How to manage meta data?   
master-slave architecture with a centralized namenode 
namenode manages the mapping between the  "name" of the file to its actual address​ 
adding a machine id before the device id of the meta data  

>  元数据维护：
>  GFS 中，master 维护文件 “名称” 到其真实地址的映射，文件的真实地址包含了其机器 id

(3) How to achieve high performance?  
sharding the data (instead of the meta data) over many slaves servers (datanodes) 
Partition a file into chunks that are scattered on different machines  

>  真实数据维护：
>  文件划分为 chunks，分布在不同的机器上

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/74a9f8cfe01f1d233f4c8f53bd3f2c270caf252463ff94b475e6f11c8fa5a8c1.jpg)  

(4) How to tolerate failure? $=>$ multiple copies of each file 

Summary: the centralized namenode maintains a map from ("path/to/file", chunk ID) $=>$ to a list of (machine ID, device ID, offset, size) that each represents a copy of this specific chunk.

>  因此，master 实际上维护的映射是从 `/path/to/file + chunk ID` 到 `machine ID + device ID + offset + size` 
>  client 请求文件 `/path/to/file` 时，master 根据请求内容，确定 `chunk ID` ，通过映射，对文件进行索引

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/0e46542a0572d29ddeba95bb0781ce0a7019cb91f522ce4029395e68efb465d7.jpg)  

### Difficulty: how to manage the consistency between multiple copies
(1) Recall the CAP/FLP theorem 

(2) Ideal semantic/model: the same behavior as a single server 
executes client operations one at a time even if multiple clients issued operations concurrently
every reads reflect previous writes even if server crashes and restarts
all clients see the same data  

>  理想情况下，分布式模型应该提供和单个 server 相同的语义行为
>  为此，分布式系统需要在多个 clients 并发请求操作时，能够逐一处理 (Availability)、client 的每次读取能够反映上次写入的结果，即便 server 在期间崩溃或重启 (Fault Tolerance)、所有的 client 看到相同的数据 (Consistency)

(3) It is tricky to implement it right in a distributed environment where **failure/reorder** will happen​  

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/4c59e8f5403d44d4df8408cd830277a1a2a7cea0c095642fa930b2e2ee321238.jpg)  

(4) It may also lead to low performance -- this is a trade-off.  

Better consistency is usually achieved by more communication, which leads to lower performance. 
>  better consistency 往往需要更多的通信实现，进而导致 lower performance

It may be a surprise the GFS actually cannot handle the above reorder case  

(5) What model is proper if a weaker semantic is possible for more performance? 
A hard question that actually even the design of GFS is a failure in retrospect  

## Scenario that GFS is Aiming For  
(1) GFS is not a general purpose filesystem   
Different of workloads: \[small, middle, large\] x \[random, sequential\]   
GFS: optimized for large x sequential  

>  GFS 针对 large (文件大小) x sequential (文件读写类型) 的 workload 而设计

(2) Most of the file reads/writes are sequential accesses to huge files -- read or append  

<html><body><table><tr><td>Original</td><td>Appended</td></tr></table></body></html>  

Not a low-latency distributed database for small items   
Possible but not optimized for random read   
Modification of original written data is permitted but append-only is preferred 
e.g., MapReduce  

(3) A custom interface that is similar but not compatible with POSIX  

(4) Provides only a weak consistency model called **relaxed consistency** model, in which duplicate, reorder, missing are all possible.  

The client application should be aware of and tolerate these inconsistency  
How to tolerate? -- ignore in most case 
Web index can tolerate data inconsistency  

>  GFS 为了提高 performance，仅提供了弱的一致性模型，其中 reorder, duplicate, missing 导致的不一致情况都有可能出现

Why? -- to improve performance. At that time, the upper layer data analysis application (MapReduce) can tolerate results that are not fully accurate   

Leads to problems -- difficult to roll out to a wider range of scenarios. In fact, the open source version of GFS -- Hadoop HDFS provides a strong consistency model by disabling writes and supports only append.  

## Architecture of GFS  
GFS is also a typical master-slave architecture  

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/f8f3d85478ff1d6a0e8ee2c259da6b2c55dbe38a82d59b993d62f64834a24d8a.jpg)  

(1) Clients communicate with the master/slave via RPC (no POSIX filesystem)  

>  GFS 中，clients 通过 RPC 和 master/slave 通信 (而不是通过 POSIX API)

(2) Chunkserver is built on top of local file systems so that it does not need to manage low-level space allocations​  

Pro: the abstraction of local file reduce the complexity of implementing chunkserver 
Con: lower performance because the data needs to be read/write through multiple layers that leads to larger latency  

>  GFS 中，chunkservers 自身通过本地的文件系统管理本地文件，这减少了实现复杂度，但增加了读写开销

(3) Each file is partitioned into fixed-size 64MB chunks and each chunk has multiple replicas.

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/53f302036d016ad8b62b85b80f3f3873eb67b89f1169f59126b26463b88075a0.jpg)  

Different chunk scattered on different nodes   
Placement policy should guarantee that different replicas of the same chunk should not be placed on the same chunkserver​   
It is desirable to place at least one copy at a different rack​  

>  GFS 中，文件会被划分为 64 MB 的 chunks，每个 chunks 有多个副本，副本散布在不同的 chunkserver 上
>  理想情况下，最好每个机架上持有一个副本

(4) Discussion on design choices  
Fixed size: simplify the management of metadata 
64MB: only large chunk read/write to align with the characteristics of the underlying HDD for better performance​ 
Why one same-rack replica $+$ one different-rack replica: tolerate the failure of a whole rack $+$ mitigate the bandwidth bottleneck of aggregated switch  

>  固定大小的 chunk 设计简化了元数据的管理，64 MB 的 chunk 大小则仅适合于大文件 (否则碎片问题较严重)
>  chunk 的多机器、多机架分布提高了容错性，同时也提高了负载均衡能力

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/7e74bc82560065fbb2adcc14692441fbed8f86ee38329b1c1d4efcf5291fe457.jpg)  

(5) Master tracks the mapping from file names to the corresponding meta data
Each file name is mapped to an array of chunk handles (nv), which contains {list of chunkservers that sore this chunk (v) and the corresponding offset, which chunkserver is the primary (v) , the remaining lease time, version \#(nv)}  

>  master 维护从文件名到对应元数据的映射
>  每个文件的元数据是一个 chunk handle 数组，其中每个 chunk handle 包括的信息有：
>  1. 存储该 chunk replica 的 chunkservers
>  2. replica 在 chunkservers 上对应的 offset
>  3. 哪个 chunkserver 是 primary
>  4. 剩余的 lease time
>  5. chunk 版本号

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/4d0a88d2a45286ec48652cdc8e363ddef6d1ebf026cf6e791486d573aa08f965.jpg)  

Why do we need a primary of the replicas? 
In order to solve the problem of network reordering that is possible when concurrently modifying multiple replicas, we select only one replica as the primary and any further modifications should first ordered on this primary​  
>  primary 机制用于解决并发地修改多个 replicas 时，network reordering 导致的 replica 之间不一致的问题
>  我们选择一个 replica 为 primary, 连续的变更的执行顺序以 primary 收到的为准，其他 replica 基于 primary 定义的顺序进行变更

What if the primary is crashed?  
To tolerate the failure of primary, a lease mechanism is used which requires the remaining lease time and the version. A primary that does not renew its lease is considered dead an a newer primary is selected.
The version is used to determine which primary is newer. Version is the version of the primary (updated per primary change), not the version of the data (updated per data updates). Data version will lead to extremely high load on the master​.
>  为了解决 primary 可能崩溃的问题，需要租约机制，如果 primary 没有更新其租约，master 就认为 primary 故障，继而选择新的 primary
>  为了防止原来的 primary 恢复后和新的 primary 混淆，需要使用 primary 版本号区分最新的 primary, primary 版本号在每次 primary 改变时更新

## Primary  
Primary: one of the replicas will be assigned to be primary by the master. All concurrent writes will be redirected to this single primary for order decision.  

(1) Selected by the master, maintained by lease 
If client wants to write, but no primary, or primary lease expired and dead. Then the master will query chunkserver about what chunks/versions they have. 
A random replica that has the latest version will be selected by the master (error if there is none).
Increment version number, write to disk for persistency 

A certain period of time lease (e.g., 10s) is given so that this selected replica knows that it guaranteed to be the primary during this period of time. 
This information is sent to all replicas and those chunkservers will persist this information on disk.
The selected primary will renew its lease before it expires -- this renew will success in most times, so that the primary will not change very often (fail renew may be caused by machine failure or network issues) 

(2) What if the coordinator designates a new primary while old one is active? 
It is possible if there is network partitioning -- the original primary is still alive and serving client requests, but network between master and the original primary fails? 

If there are two active primaries​ 
C1 writes to R1 the old primary, C2 reads from R2 the new primary, doesn't seen C1's write! -- a disaster  
The **split brain** problem  

>  在原来的 primary 还活跃的时候，master 不应该指定新的 primary，这会导致 split brain 问题


![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/2c857b905857be757db4c59c437c0f95a6a8da1abfc879cb591edb8723820beb.jpg)  


Leases help prevent split brain: the master won't designate a new primary until the current one is guaranteed to have stopped acting as primary (after the original lease is expired).  
(Not A in CAP: during the waiting of lease experiment, this part of data is not available even though there is alive replica)
>  租约机制帮助防止了 split brain 问题，master 只有通过租约过期，确定原来的 primary 不活跃之后，才会指定新的 primary
>  但在原来的 primary 崩溃到它的租约到期的这段事件内，系统没有 primary，此时 availability 不能保证

The choice of lease  
Not too high -- long unavailable time for waiting​   
Not too low -- network variance may cause false positives and also cause higher burden on the single master  
>  租约过期时间过长，就有可能导致较长的 unavailable 事件
>  租约过期时间过短，就有可能出现假正例问题

(3) Stale Chunk  
Chunk replicas may become stale if a chunkserver fails and misses mutations to the chunk while it is down.   
Stale chunk can be identified by the version, which will be fixed with async clone   
The version will be attached for each client read, and hence a chunkserver can verify whether its local chunk is up to date or not  

>  chunkserver 在变更操作进行时故障，其存储的 replicas 就可能过期
>  可以通过版本识别过期的 replicas，识别到过期的 replicas 后，可以通过将其他 chunkserver 的 replica 克隆过来以更新
>  client 的读取请求中会包含 replica 的版本信息，因此 chunkserver 可以借此验证它的 replica 是否是最新的

## Recovery of Master  
(1) Split brain for master?  
Two active masters will be even a much larger disaster than two primary.
In GFS, there is no automatic switchover. A **stop-the-world** recovery should be invoked manually to avoid the split brain problem.   
Automatic switchover is possible with the techniques taught in later lectures (Manual switch-over is still frequently used in the real-world -- the failure of master (only a single machine) is rare​)

>  GFS 中，master 的出现故障后，需要手动重启

(2) Lost meta data? Information in memory will be lost after a crash  
Try to keep all the meta data in memory (e.g., 64PB/64MB = $10^9$, 128 bytes ${10}^9=$ 128GB)​, but the throughput of a single master can be a bottleneck in such a huge cluster.

Thus a persistent checkpoint $+$ log is needed for recovery   

Should log only the necessary information to save expensive disk IO

Only the above nv meta are logged (array of chunk handles, version of each chunk). The other meta information can be recovered through communicating with all the chunkservers at restart.  

Primary information is volatile and hence lost after a restart of master -- thus master simply waits for a lease period to make sure that all the primaries expire  

>  master 的元数据恢复通过检查点和日志机制实现
>  为了解决恢复时的磁盘 IO, 存盘的元数据只有 chunk handles 数组和每个 chunk 的版本，其他的数据可以通过重启后和 chunkservers 通信重新获取
>  primary 相关的信息会丢失，master 会等待所有的租约结束后，重新分配 primary

(3) Secondary master? 
Consume and replay the above redo log to its local state 
Manual switchover  

## Read Path of GFS   
The read path of GFS is straightforward   

(1) Communicate with the master for chunk meta data via RPC 
Client -> (/path/to/file, chunk index that is calculated by the offset/64MB) -> Master​ 
Client <- (Chunk Handle, List of Chunk Servers) <- Master​   

>  client 通过 RPC 向 master 询问 chunk 元数据

(2) Cache the meta data for further reuse​ 
Further reduce the load of master
What if the reading client has cached a stale server list for a chunk? $=>$ check at the server size and reset the connection 

>  client 会缓存元数据，减轻 master 的负担，因为只有一个 master

(3) Directly communicate with the **nearest** chunkserver for data (note, only one chunk server is read)​ 
Client  -> (chunk id, offset) -> Chunk Server 
Client <- (data) <- Chunk Server 

>  client 根据元数据，向最近 (根据网络拓扑结构) 的一个 chunkserver 通信，以获取数据

Chunkserver actually manages the chunks on it with a local file system with a specialized directory tree as the index​ 

Nearest -- decided by the network topology, priority communicate with the chunkserver within the same rack for example​   

As we can see, the master handles only meta data requests. The data stream, which may consume a lot of bandwidth, is handled directly by the chunkserver in parallel.​   

>  可以看到，master 仅处理元数据通信，不处理真实数据通信，真实的数据通信会由 chunkservers 并行处理

This can bring large aggregate throughput for read​.
In old times (the evaluation setting of GFS), the sequential throughput of one HDD disk was about 30 MB/s​ , and the bandwidth of a Network interface controller (NIC) was only about 10 MB/s.
For a single client, 6 MB/second per client is achieved by GFS, which is not very large​. However, for total 16 clients + 16 chunkservers, GFS achieved 94 MB/sec, which is close to saturating inter-switch link's 125 MB/sec​.  

Therefore, in GFS, the per-client performance is not huge, but multi-client scalability is good.

However, the situation has changed nowadays. Nowadays, the bandwidth of a high-end NIC can be as large as 400Gbps. Parallelism is needed even for only saturating a single client's bandwidth.

## Write Path of GFS  
### Append Record Operation

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/bce1f175be6929902a781bd8f2036ecacbf246f26ec9b941d2e15b37695d6d88.jpg)  

(1) Client asks master for the **location** of primary and the other replicas   

(2) Client sends append data to **all the replicas** concurrently for a better throughput
The received data is **stored only in a temporary location**. It is still not committed yet​.

(3) Client asks the primary to append the previous data (i.e., the control path and the data path are separated)   

>  chunkservers 收到 client 的 append 数据后，先暂时缓存它，client 会再次请求 primary 执行 append 操作

(4) The primary checks the validation of its lease and whether there is enough space, and then decides the offset that the data should write to  

All the concurrent writes to the same chunk will be ordered by the primary by assigning different and ordered offsets

>  primary 会对收到的并发的 append 请求的数据的 append 顺序排序

(5) Primary asks the other replicas to commit the data **at the same offset**, and wait for their success/error 

This "at the same offset" is critical.

(6) Primary return the final success/error to clients 

In case of errors, the write may have succeeded at the primary and an arbitrary subset of the secondary replicas. 
If it had failed at the primary, it would not have been assigned a serial number and forwarded. 

(7) The client **will retry** if an error/timeout occurs. 
Too simple? What problems?​   

(8) Two clients append at approximately the same time. Will they overwrite each others' records?​ 
No. They append operations will be ordered at the primary chunk

But there are many other potential positions of inconsistency, especially when there is a failure​.
Suppose one secondary replica doesn't hear the append command from the primary due to a temporary network failure.  What if reading client reads from that secondary replica? 
What if the primary crashes before sending append to all secondaries? Could a secondary that didn't see the append be chosen as the new primary?  

### What About Write Operation  
Most similar to append​ 
Writes that across the boundary of chunk (64MB) will be split to multiple write operations  

## Relaxed Consistency in GFS  
File namespace mutations in GFS (e.g., file creation) are atomic, because they are handled exclusively by the centralized master.  

(Table ref to original pdf)

**The above table describes the relaxed consistency model provided in GFS**  

(1) Write means "overwrite" data that has already been written before

(2) A file region is consistent if all clients always **see the same data**, regardless of which replicas they read from.​   

(3) A region is defined after a file data mutation if it is consistent and clients will **see what the mutation writes in its entirety**.  

What is consistent but undefined?  
All the replicas will contain the same content, but the content does not reflect any single modification from the client (e.g., the first half from A and the second half from B)  

>  GFS 对于 Write 操作，在并发 (多个 client 同时请求写入) 且都成功的前提下，提供的模型是 consistent but undefined，这主要是因为一个 client 跨越多个块的写入操作会被拆分为多个，这可能会与其他 client 的并发操作交错并被覆盖

What is defined interspersed with inconsistent?  
The final append result is stored in a certain offset that is both consistent and defined.
But there are also other parts of the file that will have inconsistent results due to the previous unsuccessful attempt of this append operation. 

>  defined interspersed with inconsistent 详见后面的例子

(4) Note that the data will become inconsistent if a failure (both the server and the client failure) happens during the write/append​  

Useless? -- for append operation, the client can keep retrying upon failures, which leads to defined (but interspersed with inconsistent) state. The result will remain to be inconsistent only when both the client and some of the chunkserver fail.  

**Example of consistent but undefined write**  
(1) Two clients A and B concurrently write the same region of data   

(2) The write region cross the boundary of chunk, thus each write operation are partitioned into two separate steps. 
Write A1 from client A to Chunk 1, Write A2 from client A to Chunk 2, Write B1 from client B to Chunk 1, Write B2 from client B to Chunk 2.   

(3) Write A1 arrives after Write B1 at the primary of chunk 1​   

(4) Write A2 arrives before Write B2 at the primary of chunk 2 -- due to network issues​  

(5) Both write success 

All the replicas have value A1 at chunk 1 and B2 at chunk 2, and hence is consistent.
The result is neither Write A entirely nor Write B entirely, and hence is undefined.  

**Example of Defined Append Interspersed with Inconsistent** 
(1) At first, a success append A to all replicas  

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/bfe5790c63b3c4d177a9d254baf52875edaa6a4b9252c160396669b95fe7162d.jpg)  

(2) A following append B, during which the operation to the third replica was **lost**. Then, the following append C success.  

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/03536cd3e88c7696726381cba6e809d8a9be48cdc4ffaa073f822b4ea8f15935.jpg)  

Note that data C on the third replica is not directly following data A, because the offset is calculated at the primary. That is, the start offset of C is the same in all replicas. This is not a typical "append" in the rightmost replica. This is because the offset is calculated only in the primary  

The current state is defined for append A and append C. 

Append B is currently not defined but the client can still retry.

(3) Retry of append B success  

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/f24120271147cc7d415ee4244403e4fbfb2f92dadfeaec92a19eed8daa46a532.jpg)  

Note that the retry of B uses a new offset (the old failure of append B is not remembered by the primary, it treats the retry as a new appending). (This is a very clever design because identifying and filling the gap is very complex)
The current state is defined (although there is duplicate) for append B. But interspersed with inconsistent because of the previous failure.  

What if the client of append B crashes before a success retry? -- inconsistent state  

**Workarounds**  (for the inconsistency provided by the relaxed model)
Use append instead of overwrite -- a crashed client during its writing will lead to inconsistent state 
Checkpoint -- but it is expensive for huge datasets   
Write self-validating checksum​   
Self-identifying record to cope with duplicates -- with a unique ID  

## How to Achieve Strong Consistency?  
**In a retrospective interview with GFS engineer**  

QUINLAN: At the time, it (the relaxed consistency model) must have seemed like a good idea, but in retrospect I think the consensus is that it proved to be more painful than it was worth. It just doesn't meet the expectations people have of a file system, so they end up getting surprised. Then they had to figure out work-arounds.  

**HDFS's Design** 
HDFS supports strong consistency, at a cost of:

(1) HDFS supports append operation only (actually it only supports atomic-create-and-then read-only at first)   

(2) Serialize all the append to the same file with a **pipeline append**   

(3) Built-in rollback (through recording the offset and size and rollback to the least length of downstream nodes) if a failure of datanode/client happens

This method called chain replication will be discussed with more details at Lec 7​  

![](https://cdn-mineru.openxlab.org.cn/extract/233d33c6-a6b4-48d5-ab0a-a6a0f41fc8cf/aaee3b6ffdb698105e8ced035fde426dd4e3c229c8efa0668278664a168a73ea.jpg)  


(4) Much less parallelism and longer critical path than GFS's model 
only suitable for large chunk appending, not small-size mutations
more sophisticated approaches will be given in the later lectures  

## Conclusion  
- Good ideas that are followed by many other systems 
    global cluster file system as universal infrastructure 
    separation of naming (coordinator/master) (i.e. metadata management) from storage (chunkserver)   
    sharding for parallel throughput​   
    huge files/chunks to reduce overheads 
    primary mechanism to achieve sequence writes​   
    leases to prevent split-brain chunkserver primaries​   
- Optimized by other systems​   
    single master that may run out of RAM and CPU for billions of files $=>$ federated/sharded master in HDFS   
    chunkservers not very efficient for small files $=>$ BigTable and other distributed K-V store (lec 12)   
    lack of automatic fail-over to coordinator replica maybe consistency was too relaxed  

# 4, 5 RAFT
## Problem Definition -- The Consensus Problem  
(1) In previous lectures, we always assume that there is a global coordinator that coordinates distributed execution.  

MapReduce distributes computation but relies on a single master to track the progress and do recovery​ 
GFS replicates data but relies on the master to manage all the metadata and pick primaries​ 

(2) The failure of this coordinator usually leads to unavailability or even inconsistency. 

Can we achieve a high available coordinator by using multiple machines that replicate the state of coordinator? 
We need a **conceptually single** master that is actually implemented as a highly-available multiple replicas 

(3) Problem 1: How to safely select a single coordinator within a set of candidates? -- **leader selection** problem 

Manual configuration is widely used but requires manual reconfiguration for every failure of coordinator.
How can we achieve automatic selection? 
Can we use a pre-defined priority order?  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/4327424d87e055c2c68fde8640530ad337a28e1dac0db4116868c83e3a14194c.jpg)  

(4) Problem 2: How to recover from a coordinator failure by replicating its state? -- **replication** problem​   

Can we first replicate the modification to all the followers and proceed only after receiving acks?   
How to handle failed followers?  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/86e443ddc41391d98fb9cecd65a26c9b9024f3d8ddfcf0c6a17e93bedb8ba24b.jpg)  

How can we know a follower is failed?  

(5) Both the above two problems can be reduced to the same **consensus** problem

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/d0699203c8f859b4eb766c6165d67cedd647afe9d4636f7207da1e4497879e36.jpg)  

Consensus by majority is natural in human society (at a cost of the “Tyranny of the majority”, but it is fine in computer science)  

The problem is from the unreliable and asynchronous network -- can be lost, duplicated, reordered, and even split -- most importantly, it can not finish in an atomic manner.
What if the results of the first round are: 2 coffee, 2 tea, 1 milk? -- a tie, some one(s) need to change their mind in order to break the tie and make progress.
If only one changes its mind, it will be fine, but what if two changes their mind concurrently? -- e.g., milk ${->}$ coffee, coffee ${->}$ tea.  
It is possible, because of the delay/reorder of the network, at some time A receives only "milk ${->}$ coffee" and considers there are three votes for coffee, but, simultaneously, another B  receives only " coffee ${->}$ tea" and considers there are three votes for tea  

A split brain!!!  

## The Split Brain Problem  
(1) A single coordinator that makes critical decisions is the most straightforward way to avoid the split brain problem  

It's easy to make a single entity always agree with itself​, but there is an availability problem of single coordinator
What is the problem of setting up a backup for the primary that only takes over after it detects the failure of the original coordinator? -- a **primary-backup** architecture  

(2) The following figure demonstrates a typical split brain situation that is **unavoidable** for a two nodes primary-backup system  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/0de5c59ebb79afda85e93e83b050feec68d0035043556645958617dda8ee1e4d.jpg)  

After a network partition, the backup master thinks it should become the primary master and serves client 2's requests   
But, at the same time, the original master is still alive and serving client 1's requests $=>$ a split brain that leads to inconsistent state​   
This is the reason why all the two nodes primary-backup system suffers from split brain problems if an automatic failover is enabled. For example, the keepalived  

>  所有的两节点 primary-backup 系统在启用了自动的故障切换功能时，都会存在 split-brain 问题

(3) The problem: computers cannot distinguish "server crashed" from "network broken"  

The symptom is the same: no response to a query over the network, but it may respond to other clients' query​   
A typical solution relies on a standalone arbiter node to decide which is the primary -- but what if this arbiter node is down? . Just like we use the master as the arbiter of choosing primary from replicas.
Manual reconfiguration is used for a long period of time because this problem seems insurmountable for a long period of time. But we indeed want an automatic scheme.  

## Consensus by Majority Voting -- Basic Paxos  
Solution proposed by Leslie Lamport in 1989, which leads to a Turing Award in 2013  

**Basic Idea: majority vote**  
(1) A cluster with at least 3 nodes   

(2) And it can end with a consensus acceptance of only one proposal if it gathered agreement votes from at least a majority set of nodes (2 out of 3, 3 out of 4, 3 out of 5, ...) 

We may have multiple rounds because the previous rounds may end in no results, but all these rounds are used to reach one consensus  

(3) Each round can start with multiple proposals  

May end with a consensus proposal   
May also end with no consensus if there is split votes   
Guaranteed to have no split results because two quorum will have at least one node in common. No need of an unanimous vote, quorum is enough  

(4) Wait and another round of voting starts if no proposal receives majority voting 

In a split vote where no proposal receives a majority, some voters must change their voting for making progress 
This change is very dangerous, which, without careful design, will lead to inconsistent problems​  
When is the right time that a voter can change its voting? 
Deciding by the voter along with a timeout? -- inconsistent problem (the above coffee/tea example)   
Deciding by the proposer by revoking its previous proposal? -- unavailability if the proposer fails​  

(5) The best property of majority (>50%): every two majority sets of nodes will have at least one node in common  

Nodes in the intersection can convey information about previous decisions -- but how? 
The majority is out of all servers, not just out of live ones​
In a partitioned network, the split that contains a majority of nodes can still proceed  

(6) More generally, $2\mathsf{f}{+}\mathsf{1}$ can tolerate f failed servers ◦ if more than f fail (or can't be contacted), no progress  

(7) Often called "quorum" systems  

### Basic Paxos  

Make only one consensus decision  

may contain multiple rounds if former rounds have not reached a consensus decision ◦ The result of consensus is a "value"  

▪ the explanation of this value is depended on the specific scenario e.g., in leader election, the value can be the node ID of the leader  

Three roles in the cluster [fig 2]  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/c1a992e6643386f978ad1d773c20a8db7ad052d7760a2f3bb4face75e4d0d46c.jpg)  

Proposer: propose candidate proposals --- multiple proposers for high availability  

Acceptor: decides whether to accept the proposal ▪ A proposal is accepted only after receiving a majority vote from all the acceptors Note it is a majority of acceptors, of a majority of all the nodes  

◦ Learner: does not participate in the decision, only learn accepted consensus decisions passively ◦ Each node can paly multiple roles (e.g., it can be proposer and acceptor simultaneously • The whole procedure is partitioned into two phases (1a prepare, 1b promise, 2a Propose, 2b Accept, c Learn) [fig 3]  

◦ A minimum of 2 RTT latency for decision (if there is no split vote in the first round), can be more if retry​   
◦ Two phase paradigm is common in distributed environment in order to achieve consensus/atomic/etc  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/5baf8a869aca6e27034437eebddbd7e69ab1f798aad950bcf849b3f87d8e5e1d.jpg)  

◦ Note that the proposer can notify the client its proposal has been accepted after receiving 2/3 Accepted messages  

## Basic Paxos  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/aaa4a716ec655d9b3f607d5da1443731517aff5eb79bc37de03b4d9aad7deff8.jpg)  

### Acceptors must record minProposal,acceptedProposal, and acceptedValue on stable storage (disk)  

What is promise? The acceptor promises not accepting a smaller proposal ID by maintaining a minProposal variable​   
◦ The proposal ID must be a unique and monotonically increasing  which can be obtained by $<a$ monotonically increasing counter, node ID> The above setting actually only achieves: 1) global unique; and 2) per node monotonical​   
For proposer A, its proposal should be something like 1.A, 2.A, 3.A.   
Similarly, for proposer A, its proposal should be something like 1.B, 2.B, 3.B.   
Why suffi $\langle\?\mathrm{~--~}1.mathsf{A}<1.\mathsf{B}<2.\mathsf{A}<2.\mathsf{B}<3.\mathsf{A}<3.\mathsf{B}$ ◦ Suffix of proposer ID can make the IDs global unique ◦ Prefix can be used to make sure that no single proposer will always triumph  

How is global monotonical achieved? -- asynchronously communicated via the return message in the above step 3/6  

◦ The most critical step is in (3), the returning of accepted value if the acceptor has already accepted a value before  

▪ The proposer's original proposed value is replaced by the accepted value returned by the acceptor   
▪ this is the way that Paxos conveys information about previous decisions through the nodes in the intersection of two majority sets of nodes  

Several Examples [fig 2]  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/238804e4aad3b10a2a8258eaf18c351d7a175c7d8beda7bf922a3d5e7dad8d57.jpg)  

### "Prepare proposal 3.1 (from s1)'  

◦ X is accepted by a and only a majority of acceptors   
The final accept value proposed by the green proposer is X, instead of its original value -- returned by acceptor s3 because it has already acc‘epted X.   
◦ Y can never be accepted after X is already accepted by a majority of acceptors because there is at least one acceptor in common will change the proposer's value to X​ ▪ What if P 4.5 happens before A 3.1 X in node S3? -- A 3.1 X will be rejected in this case because its proposer ID is lower than minProposal, see the example below  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/85cd9f1a3b13ab8f2c2fd94a336acb7252c5b522ee23800603b7864840e2c363.jpg)  

The situation where the final value is Y. The blue proposer cannot gather enough accept votes because the acceptor s3 has already promised the green proposer, which has a larger proposal ID.  

▪ The "promise" is only a promise of "no smaller" but can be overridden by a larger ID  

What happens to S1 and S2? -- they will change their value after receiving Y's accept message because it has a larger proposal ID  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/37d2df338d638861ba778a11a01af438ec314b122cbb723c5e5d9c29004191fa.jpg)  

• Similar to the first example. The final value is X even though it was not accepted by a majority at the time the green proposer broadcasted its proposal ID.  

## Basic Paxos and FLP  

FLP: only two can be achieved in (fault tolerance, termination, agreement)   
• Paxos can lead to livelock -- can be mitigated by certain solutions like randomization ◦ Also a good (bad?) example of how we increase the proposal ID to achieve "unique and monotonically increasing"  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/0a8e14097f32dc970a1e7179eb7778d6c24cc04d1d7fc9ac386522cb98c084a8.jpg)  

### Safety properties  

• Validity (or non-triviality): Only proposed values can be chosen and learned. Agreement (or consistency, or safety) No two distinct learners can learn different values (or there can't be more than one decided value) Termination (or liveness) If value C has been proposed, then eventually learner L will learn some value (if sufficient processors remain non-faulty). -- live lock is possible  

### What if​  

• What if the proposer crashes when its accepted proposal is only broadcast to a majority but not all the acceptors?  

◦ How can the remaining acceptors know the accepted proposal? Other proposers will learn the accepted value at their preparation phase and broadcast the value to other acceptors   
Thus, multiple proposers is enough for achieving high availability  

What if a proposer with a higher propose number overrides a previously accepted proposal Overriding is possible but it will be overridden with the same value, so that does not violate the "Agreement" property There must be one acceptor that is shared by two quorums. In this shared node  

▪ If the accept message of the former proposal comes before the prepare message of the later proposal​ • The later proposer will learn the accepted message and broadcast accept with this accepted message​   
▪ If the accept message of the former proposal comes later than the prepare message o the later proposal​ • This shared acceptor will update its minProposal and hence refuse to accept the accept message from the former proposal  

• What if we want to start another round of decision after the former is accepted?​ ◦ Can we directly increase our proposal number and start another round? No!!!! ▪ Or actually not possible in basic Paxos because, after an acceptance, any further proposal will be revert to the original proposal by the acceptor in step 3​ Basic Paxos is only for making one consensus decision  

## Replicated State Machine  

### Why Do We Need RSM?  

• Reaching consensus on one decision is not enough for implementing a high-available coordinator​ ◦ It needs a series of consensus, one for each modification​ Replicated state machine is a powerful abstraction that can be used to build all kinds of highavailable service as long as every operation of the service is deterministic.   
The centralized structure is a replicated log ◦ Each replica works as a state machine that reads the log for operation steps ◦ And then perform the step in order ◦ similar to the turing machine  

### Common Procedure of RSM  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/b2b3020c7f0587f7f58d27f8f0474944a07f6ff4ad2cb88e61b3c55c46555b4f.jpg)  

• Client sends "command A" to one replica of the service (may or may not need to be a chosen leader?) The received replica proposes a proposal on "command A" should be the X-th command executed, which is represented by storing "command A" in the X-th slot of the replicated log ◦ Transfer the semantic of "consensus on value" to "replicated log"  

• Somehow, the consensus module successfully committed the log "command A" at the Y-th slot​  

◦ A log is committed if 1) at least a majority of replicas reach agreement upon this log; 2) there will be no further rollback even if some (not all) of the agreed replicas crashes ▪ Thus there will be two states 1) replicated to a majority of replicas; and 2) committed. The first is still not safe.  

◦ The final position Y-th may not be identical to the original proposal X-th • The state machine on every replica continuously consumes committed logs and returns the result to the client only after it has applied the Y-th log​  

◦ This consumption can be an asynchronous procedure to the consensus procedure -- communicated via a count "commited_index"  

### Why replicated log?  

• The service keeps the state machine state. The log is an alternate representation of the same information! why both?  

Log stores the ordered history of committed commands ◦ to help replicas agree on a single execution order ◦ to help the leader ensure followers have identical logs Log can store tentative commands until committed -- otherwise , it is hard to rollback a command from the state machine  

◦ We'll see that they can temporarily have different entries on different replicas (that are not committed)   
These different entries eventually converge to be identical​   
◦ The commit mechanism ensures servers only execute stable entries  

Logs are helpful in recovery  

◦ the log stores commands in case leader must re-send to followers the log stores commands persistently for replay after reboot  

### How to Implement RSM with Paxos? -- Multi-Paxos  

Multiple rounds of basic-paxos for leader election and for replicating logs Very hard to understand and to use in practice $=>$ the main motivation of Raft​  

## Raft Model  

### Basic Procedure  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/1bbf8563a021383c36bfa6b9be394a4ba01f395366896c64b9e791d213b6323f.jpg)  

Raft directly provides a RSM model that basically contains only two procedures: 1) Electing a new leader; and 2) Ensuring identical logs despite failures  

In Raft, time is divided into terms  

Each term begins with an election.   
After a successful election, a single leader manages the cluster until the end of the term. $=>$ replicating and committing logs   
◦ Some elections fail, in which case the term ends without choosing a leader $=>$ re-election  

• Every server in the Raft cluster maintains a monotonically increasing currentTerm value ◦ This value should be persistent for recovering from a restart​  

### Why Strong Leader?  

Paxos can have multiple proposers, but only one leader in Raft can propose   
• A leader ensures all replicas execute the same commands, in the same order -- decided by the single leader   
• Raft uses a stronger form of leadership than other consensus algorithms. ◦ For example, log entries only flow from the leader to other servers. ◦ This simplifies the management of the replicated log and makes Raft easier to understand​   
• Leader can become a bottleneck in certain circumstances (e.g., geo-replicated systems) and lead to less parallelism and hence less throughput​ ◦ A trade-off between understandability and performance -- understandability wins  

### Leader Election in Raft  

State of Servers in Raft and the Transition  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/a440959b541e8f558934811962d59f663c4a02b5a415621355a73bc4844733a7.jpg)  

There are only three states of servers and two kinds of RPC calls in Raft (very elegant!)  

• A server starts as Follower that receives and replies AppendEntries RPC from the leader ◦ The RPC tries to append a log entry to the replicated log​ ◦ The RPC can also contain an empty log entry, which serves as a heartbeat from the lea • A follower timeout if not receiving valid AppendEntries RPC for a certain period of time. It becomes a candidate and broadcasts RequestVote RPC [fig 4]  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/13b05b10a30af9ad0680831382462af7056162c65d6bb86969606eca4814ecdc.jpg)  

◦ The candidate starts an election by increasing the currentTerm it stores and sending RequestVote RPC with this new term   
◦ A candidate becomes a new leader if it receives votes from a majority of servers   
◦ A candidate may notice that another server becomes a new leader (by receiving its AppendEntries RPC), and then it transfers to a follower   
◦ Sometimes no candidate receives enough votes, which leads to a new election   
A leader server X periodically broadcasts AppendEntries RPC to make sure that it is still the   
leader   
◦ leader must send heart-beats more often than the election timeout to suppress any new election​   
◦ But it is possible that this leader is temporally isolated from a majority number of servers (e.g., due to a network partition)   
◦ In that case, the other followers may timeout and elect a new leader Y   
◦ After reconnecting with the other servers, this (previous) leader X will receive AppendEntries RPC from the new leader Y (by comparing the term number) and becomes a follower ▪ Is there a split brain problem?  

### Asynchronous of Raft  

The transitions between terms may be observed at different times on different servers.  

It is guaranteed that there is at most one leader for each term ◦ so if you see AppendEntries with term T, you know who the leader for T is But, physically, there can be multiple servers that are both leaders at the same time because they are currently in different terms (have different currentTerm)​ How to solve this problem to ensure there is no split brain problem? ◦ New leader means a majority of servers have incremented currentTerm ◦ Old leader with an older currentTerm will not be able to gather enough requests for commit a log ▪ Every majority set will contain a server that has a larger currentTerm than the old leader​ ▪ Hence old leaders are harmless and there is no split brain  

◦ It will also eventually discover the new leader and become a follower ◦ But a minority may accept old leader's AppendEntries so logs may diverge at end of old term​  

▪ Fixed later -- we need some kind of rollback mechanism ▪ Again, the replicating of logs is not the same as the commitment of logs  

### Unsuccessful Election  

There are two possible reasons of unsuccessful election   
◦ Case 1: less than a majority of servers are reachable -- the whole cluster becomes unavailable in this case  

◦ Case 2: simultaneous candidates split the vote, none gets majority  

▪ Without special care, elections will often fail due to split vote   
▪ all election timers likely to go off at around the same time $=>$ every candidate votes for itself $=>$ no-one will vote for anyone else $=>$ time out at around the same time again $=>$ repeat​  

How to reduce the possibility of case 2?  

randomized election  

each server adds some randomness to its election timeout period ▪ The server with the lowest timeout has a good chance of becoming the new leader Indefinite re-election is still possible but with a very low possibility  

randomized delays are a common pattern in network protocols for resolving conflict without synchronization  

• Other solutions? A priority of leader selection -- possible but leads to many subtle issues and is abandoned by Raft designers  

## Malicious/manipulate Server  

In Raft, we assume that all the messages are sent according to the algorithm  

That is to say, a server will always believe the content of message it receives   
• This can not be true because of a bug or a malicious/manipulate server Algorithms that can tolerate (called Byzantine fault) are much more complex and require cryptography backgrounds​ ◦ The most well-known Byzantine system is bitcoin -- one can learn it from the material of the original MIT 6.824 course. It is not included in this class  

### Log Replication in Raft  

Replicating and Committing Logs Without Leadership Change  

Once a leader has been elected, it begins servicing client requests  

appends the command from the client to its local log as a new entry -- with a term Id   
(currentTerm of the leader) and a logIndex that represents its location in the log (i.e., the   
execution order)   
▪ Similar to GFS, the position of appending is decided by the leader, not directly append to the follower's log  

◦ Broadcasting the above log in parallel with AppendEntries RPC.  

◦ If followers crash or run slowly, or if network packets are lost, the leader retries AppendEntries RPCs indefinitely   
◦ A log entry is committed once the leader that \*\*created\*\* the entry has replicated it on a majority of the servers​   
◦ When the entry has been safely replicated (i.e., committed), the leader applies the entry to its state machine and returns the result of that execution to the client.   
◦ As for the other minority set of followers, the leader will still reties AppendEntries  even after it has responded to the client until all followers eventually store all log entries.  

Example  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/04e3402d1310691bd9c264bb02cf17b5d1489fcd1e4d392e9012ec1c21160e51.jpg)  

In the above example, entry 7 is committed because it is replicated to a majority by its corresponding leader ◦ Raft guarantees that committed entries are durable and will eventually be executed by al of the available state machines. ◦ Pay attention to the phrase "the leader that created the entry" $=>$ we will see later an example, in which, an entry is already replicated to a majority of the servers but still not committed, because it is not replicated by the leader who created the entry The leader keeps track of the highest index it knows to be committed, and it includes that index in future AppendEntries RPCs (including heartbeats) so that the other servers eventually find out. Once a follower learns that a log entry is committed, it applies the entry to its local state machine (in log order).  

◦ This information is inlined in AppendEntries so that we do need an additional round of RPC​  

The above algorithm is enough to converge the logs on all nodes if there is no failure  

◦ All the followers have a prefix of the leader's log in the above example, thus retry is enough   
But this is only a lucky special case   
There are complex divergences in real-world cases and we need some kind of roll back mechanism  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/c79bdc025bc860397928fc535d8a0b4cd6f47391b9d81f7399a1465752797c32.jpg)  

Discuss later  

To converge the difference, the leader must find the latest log entry where the two logs agree, delete any entries in the follower’s log after that point, and send the follower all of the leader’s entries after that point.  

The leader maintains a nextIndex for each follower.   
nextIndex is initialized to be the last index for a new leader   
After a rejection from the follower, the leader decrements nextIndex and retries Eventually nextIndex will reach a point where the leader and follower logs match In the above example, server f will rollback to log index 4  

### Log Matching Property  

• P1: If two entries in different logs have the same index and term, then they store the same command.  

straightforward, because only one master per term and it will choose only one content for every index​ ◦ Same index $^+$ different term $=$ different content  

• P2: If two entries in different logs have the same index and term, then the logs are identical in all preceding entries $=>$ even when these two entries are not committed Benefit: if we match the entries one by one from the end to front, once a match is found, we do not need to compare all the preceding entries because they are guaranteed to be the same ▪ Thus, the suffix is the divergent part that should be rolled back​ ◦ How to achieve this goal? -- roll back the divergent log entries $^{\circ}$ The situation becomes more complex when the leader fails, because the replicating of logs is executed parallel and the leader can commit a log for only a majority (not all) of servers reply  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/1cf7932bda40baa59843e1f43a58089b34dd6e444cd172caac67448d61005192.jpg)  

In the above example,  

▪ leaders in term 2 and 3 failed before committing any log entry but replicating several logs in server f​ • The current leader directly jumps from term 1 to term 4​   
▪ If the current leader of term 8 broadcasts a new log entry at index 11 and server f blindly accepts it, there will be a violation of P2  

◦ To resolve the above problem, when sending an AppendEntries RPC in raft, ▪ the leader includes the index and term of the entry in its log that immediately precedes the new entries.  

▪ If the follower does not find an entry in its log with the same index and term, then it refuses the new entries.   
▪ The consistency check acts as an induction step (prove by mathematical induction，数 学归纳法): • the initial empty state of the logs satisfies the P2 and the consistency check preserves the P2 whenever logs are extended. ◦ Assuming P2 is guaranteed with index i-1 ◦ for appending index i, the algorithm will check i-1 ◦ If the entries at index i-1 have the same term and same content​ ◦ Then, due to the assumption, all the entries before i-1 are also the same  

◦ To converge the difference, the leader must find the latest log entry where the two logs agree, delete any entries in the follower’s log after that point, and send the follower all of the leader’s entries after that point.  

▪ The leader maintains a nextIndex for each follower. nextIndex is initialized to be the last index for a new leader -- 11 in the above example   
After a rejection from the follower because the mismatch on the precede entry, the leader decrements nextIndex and retries Eventually, the nextIndex will reach a point where the leader and follower logs match - - at least they will agree on 0 In the above example, server f will rollback to log index 4  

How to assure that a committed log entry will never be rolled back?  

▪ P1 and P2 only guarantees the "matching/replicated", not related to "committed"  

## Safety  

The mechanisms described so far are not quite sufficient to ensure that each state machine executes exactly the same commands in the same order. -- can you image a counter example?  

### Election restriction  

A follower might be unavailable while the leader commits several log entries ◦ A lag follower does not violate the above Log Matching Property ◦ But if a lag follower is elected as a new leader, the committed commands will be overwritten   
Solution: selecting only the candidate that stores all of the committed log entries ◦ Follower a, c, d are also eligible for becoming the new leader in the above example  

Implementation: a RequestVote RPC is rejected if the sender candidate's log is staler than the receiver  

Again, we take advantage of the quorum property   
A log entry is committed only when it is replicated to at least a majority number of servers   
◦ A lag candidate can never receive agreement votes from any of these servers and hence cannot be elected as a new leader   
Raft determines which of two logs is more up-to-date by comparing the index and term of the last entries in the logs. If the logs have last entries with different terms, then the log with the later term is more up-to-date. ▪ If the logs end with the same term, then whichever log is longer is more up-to-date.  

Why not elect the server with the longest log as leader?  

### $^{\circ}$ term is more important than length  

◦ A failed leader may replicate a lot of log entries  to only a minority of nodes, which leads to long but stale nodes ◦ Node f in the above example is very long but with very small term number​  

### Committing entries from previous terms  

• Corner Case: a leader cannot immediately conclude that an entry from a previous term is committed once it is stored on a majority of servers  

not committed because it is not created by the current leader $^{\circ}$ It is replicated by the current leader, not the failed leader that creates it ◦ Thus, even when it is replicated to a majority of nodes by the new leader, the latest term of these nodes are still stale ◦ hence cannot guarantee that his replicated log entry will not be rolled back​  

▪ In contrast, if it is created by the current leader, its term is guaranteed to be the larges  

An example (the most important example)  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/7135f4ac6ddcd601a9ca1a4a5108a358b8e7402c3a8e89f95272fcb76f7efd0e.jpg)  

◦ S1 crashes at time c and S5 is elected as leader of term 5  

▪ S5 gets votes from S4 and S3 because the term of $\mathtt{S3^{\prime}S}$ last entry is 2  which is less than S5's 3​ ▪ 2 is rollbacked and overwritten by 3 although it has been replicated by a majority of servers at time c​ • If the 4 at logIndex 3 is also replicated to a majority, then it will never be rolled back any more Subfigure (e) is not a valid successive state of (d), it is a possible successive state of (c), in which the entry with term 4 is committed, and hence the entry with term 2 is also committed.   
• Raft never commits log entries from previous terms by counting replicas. Only log entries from the leader’s current term are committed by counting replicas; once an entry from the current term has been committed in this way, then all prior entrie are committed indirectly because of the Log Matching Property.​ ◦ A no-op (sync) is committed for every newly elected leader to commit all the previous logs   
• Why?​ ◦ New leader cannot roll back committed entries from a previous term, because of the leader election restriction ◦ But a new leader that replicates an uncommitted entry to a majority set of servers does not have the same guarantee, because the term of this uncommitted entry may be lower than the new leader ◦ In contrast, the log entry created by current leader and replicated to a majority of nodes has the newest term  

## Recovery  

### What would we like to happen after a server crashes?  

Raft can continue with one missing server but failed server must be repaired soon to avoid dipping below a majority Replace with a fresh (empty) server $=>$ long recovery time because it must catch up from the start​   
• Reboot crashed server, re-join with state intact, catch up $=>$ by persistence  

### Persistence  

### State  

### Persistent state on all servers:  

(Updated on stable storage before responding to RPCs)  

<html><body><table><tr><td>currentTerm votedFor</td><td>latest term server has seen (initialized to 0 on first boot, increases monotonically) candidateId that received vote in current</td></tr><tr><td>logll</td><td>term (or null if none) log entries; each entry contains command for state machine, and term when entry</td></tr><tr><td>Volatile state on all servers:</td><td>was received by leader (first index is 1)</td></tr><tr><td>commitIndex</td><td>index of highest log entry known to be committed (initialized to O, increases</td></tr><tr><td>lastApplied</td><td>monotonically) index of highest log entry applied to state machine (initialized to O, increases monotonically)</td></tr><tr><td colspan="2">Volatile state onleaders: (Reinitialized after election)</td></tr><tr><td>nextIndex</td><td>for each server, index of the next log entry to send to that server (initialized to leader</td></tr><tr><td>matchIndex</td><td>last log index + 1) for each server, index of highest log entry known to be replicated on server (initialized to O, increases monotonically)</td></tr></table></body></html>  

log[], persistent -- obvious votedFor, persistent -- to make sure that a server will not vote twice in a term  

currentTerm, persistent ◦ why not directly use the term of the last log entry? ◦ not correct, similar to votedFor a node may voted for a newer term but still have not received log entry from that term commitIndex, not persistent -- will receive it from the leader • lastApplied, not persistent -- it is not persistent in Raft because its persistency does not impact the property guaranteed by Raft, but its persistency should be considered by the implementation of the state machine ◦ Raft only guarantees the log is consistently replicated​ ◦ The state is always correct if the state machine always starts from initialization state an executes all the committed logs to restore the state after restart​ ◦ If the state machine is also recovered from a snapshot, the state machine should record the lastApplied at the checkpoint instead of the last lastApplied (discussed later)  

netIdenx/matchIndex, not persistent -- reinitialized after each election  

## Performance  

• Persistence is often the bottleneck for performance -- a hard disk write takes 10 ms, SSD write takes 0.1 ms so persistence limits us to 100 to 10,000 ops/second   
• Lots of tricks to cope with slowness of persistence ◦ batch many new log entries per disk write ◦ persist to battery-backed RAM, not disk $^{\circ}$ be lazy and risk loss of last few committed updates  

### Snapshot  

### Problem and Basic Idea  

• Problem: log will get to be huge -- much larger than state-machine state! It will take a long time to re-play on reboot or send to a new server   
• Idea: the executed part of the log is captured in the state and hence we can make a snapshot of state  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/0e4ebd251da09f11c7bfd7b418003a5bbef7d7744fc298b3401ddb50d9768cfc.jpg)  

With a persistent snapshot, a restart server can set the lastApplied to the last included index of the snapshot and go through the following log entries for recovery  

### Problem of Lag Follower  

• As discussed above, a leader must find the latest log entry where the two logs agree, delete any entries in the follower’s log after that point, and send the follower all of the leader’s entries after that point.   
It is possible that the latest log entry where the two logs agree is already deleted (compacted into the snapshot)   
• An additional InstallSnapshot RPC is used in this special case to install the snapshot of leader to this lag follower  

## Client Interaction (1) -- linearizability  

• Client operation lasts for a period of time, which starts from its invocation and ends when the leader returns the results  

Each operation appears to take effect atomically at some point between its invocation and completion. -- but what point?   
◦ Thus, the lasting period of two concurrent client operations may intersect with each other -- which one is executed earlier?​   
◦ The guarantee of the above problem is called a consistency model  

• Linearizability is the consistency model Raft achieves and it is the most common version of strong consistency model that can be achieved in a distributed environment​  

Definition: an execution history is linearizable if one can find a total order of all operations, that matches real-time (for nonoverlapping ops), and in which each read sees the value from the write preceding it in the orde  

Example 1: Linearizable as Wx1 Rx1 Wx2 Rx2 (although Rx2 actually starts before Rx1 but read a newer version of value)  

1 |-Wx1-| |-Wx2-|  
2 |---Rx2---|  
3 |-Rx1-|  

Example 2: Linearizable as Wx0 Wx2 Rx2 Wx1 Rx1  

1 |--Wx0--| |--Wx1--|  
2 |--Wx2--|  
3 |-Rx2-| |-Rx1-|  

Example 3: not linearizable  

1 |--Wx0--| |--Wx1--|  
2 |--Wx2--|  
3 C1: |-Rx2-| |-Rx1-|  
4 C2: |-Rx1-| |-Rx2-|  

In Raft, the log order is the linearizable order linearizability $\mathrel{\mathop:}=$ serializability  

$^{\circ}$ linearizability is a consistency model regarding the consistency among multiple replicas Serializability is an isolation level for database transactions -- concurrent transaction on even one database server needs to guarantee serializability   
◦ We will revisit this in later lectures  

## Client Interaction (2) -- Duplication caused by retrying  

"a committed operation" has three meanings in Raft:  

P1) the op cannot be lost, even due to (allowable) failures. In Raft, when a majority of servers persist it in their logs created by the current leader. This is the "commit point" of Raft ◦ P2) the system knows the op is committed -- the leader saw a majority. ◦ P3) the client knows the op is committed -- the client receives a reply from the leader  

• A client will resend the op if timeout, but the sent op may already pass P1 -- leads to a duplication​  

Basic Idea: duplicate RPC detection (Lec 1.2)  

client picks a unique ID for each request, sends in RPC same ID in re-sends of same RPC   
the service maintains a "duplicate table" indexed by ID. after executing, record reply   
content in duplicate table   
if 2nd RPC arrives with the same ID, it's a duplicate generate reply from the value in the   
table​  

Problem: how does a new leader get the duplicate table?  

◦ put ID in logged operations handed to Raft all replicas should update their duplicate tables as they execute so the information is already there if they become leader   
This mechanism also solves the problem of recovering the duplicate table after a resta  

• Another corner case: what if a duplicate request arrives before the original executes? ◦ could just call Start() (again) it will probably appear twice in the log (same client ID, same seq #)​ ◦ when cmd appears on applyCh, don't execute if table says already seen  

• Does it violate linearizability since the leader is returning an "old" state that may not be upto-date​  

1 C1: |-Wx10-| |-Wx20-|   
2 C2: |-Rx10-  

## Client Interaction (3) -- Read-only Optimization  

The above standardized procedure of executing an operation is expensive (at least two round-trips). Can we optimize read-only operations by directly returning from the state machine? Simple answer: No, at least not straightforward ◦ The leader who received the read-only operation might have recently lost an election, but not realized The new leader may already modify the value, which leads to a split brain problem that violates linearizability​ Possible Solution: lease ◦ define a lease period, e.g. 5 seconds after each time the leader gets an AppendEntries majority, it is entitled to respond to read-only requests for a lease period without adding read-only requests to the log, i.e. without sending AppendEntries. a new leader cannot execute Put()s until previous lease period has expired result: faster read-only operations, still linearizable.  

Another solution is also described in the paper  

## Cluster Membership Changes  

Problem: It isn’t possible to atomically switch all of the servers at once, so the cluster can potentially split into two independent majorities during the configuration transition  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/ca39b5daf48a687efbb4f13e2314c062db2e618240725b6110de82609b7901d0.jpg)  

• The state is first changed to a joint configuration Cold,new, during which all the decisions should reach majority consensus in both the two configurations   
Both Cold,new and Cnew are special log entries that are replicated and committed with the same mechanism of other operations  

A Visualization of Raft: https://thesecretlivesofdata.com/raft  