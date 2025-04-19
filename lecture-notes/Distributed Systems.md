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

>  通过多个 replicas 实现概念上单个高度可靠的 master

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

### Basic Idea: majority vote
(1) A cluster with at least 3 nodes   

(2) And it can end with a consensus acceptance of only one proposal if it gathered agreement votes from at least a majority set of nodes (2 out of 3, 3 out of 4, 3 out of 5, ...) 

We may have multiple rounds because the previous rounds may end in no results, but all these rounds are used to reach one consensus  

(3) Each round can start with multiple proposals  

May end with a consensus proposal   
May also end with no consensus if there is split votes   
Guaranteed to have no split results because two quorum (majority) will have at least one node in common. No need of an unanimous vote, quorum is enough  

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
(1) Make only one consensus decision  

The decision process may contain multiple rounds if former rounds have not reached a consensus decision 

The result of consensus is a "value" , and the explanation of this value is depended on the specific scenario, e.g., in leader election, the value can be the node ID of the leader.

(2) Three roles in the cluster

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/c1a992e6643386f978ad1d773c20a8db7ad052d7760a2f3bb4face75e4d0d46c.jpg)  

- Proposer: propose candidate proposals --- multiple proposers for high availability  
- Acceptor: decides whether to accept the proposal 
    A proposal is accepted only after receiving a majority vote from all the acceptors 
    Note it is a majority of acceptors, of a majority of all the nodes  
- Learner: does not participate in the decision, only learn accepted consensus decisions passively 

Each node can paly multiple roles (e.g., it can be proposer and acceptor simultaneously 

(3) The whole procedure is partitioned into two phases (1a prepare, 1b promise, 2a Propose, 2b Accept, c Learn) 

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/5baf8a869aca6e27034437eebddbd7e69ab1f798aad950bcf849b3f87d8e5e1d.jpg)  

Therefore, Paxos requires a minimum of 2 RTT latency for decision (if there is no split vote in the first round), and can be more if retry​   

Two phase paradigm is common in distributed environment in order to achieve consensus/atomic/etc  

Note that the proposer can notify the client its proposal has been accepted after receiving 2/3 Accepted messages  

(4) Illustration for Basic Paxos

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/aaa4a716ec655d9b3f607d5da1443731517aff5eb79bc37de03b4d9aad7deff8.jpg)  

Acceptors must record `minProposal`, `acceptedProposal`, and `acceptedValue` on stable storage (disk)  

What is promise? The acceptor **promises not accepting a smaller proposal ID** by maintaining a `minProposal` variable​   

The proposal ID must be a **unique and monotonically increasing** which can be obtained by a <monotonically increasing counter, node ID> 
    The above setting actually only achieves: 1) global unique; and 2) per node monotonical​   
    For proposer A, its proposal should be something like 1.A, 2.A, 3.A.    Similarly, for proposer B, its proposal should be something like 1.B, 2.B, 3.B.   
    Why suffix?
        e.g. `1.A < 1.B < 2.A < 2.B < 3.A < 3.B`
        Suffix of proposer ID can make the IDs global unique
        Prefix can be used to make sure that no single proposer will always triumph  
    How is global monotonical achieved? -- asynchronously communicated via the return message in the above step 3/6  

The most critical step is in (3), the returning of accepted value if the acceptor has already accepted a value before  
    The proposer's original proposed value is **replaced** by the accepted value returned by the acceptor   
    This is the way that Paxos **conveys information about previous decisions** through the nodes in the intersection of two majority sets of nodes  

(5) Several Examples

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/238804e4aad3b10a2a8258eaf18c351d7a175c7d8beda7bf922a3d5e7dad8d57.jpg)  

X is accepted by one and only one majority of acceptors   
The final accept value proposed by the green proposer is X, instead of its original value -- returned by acceptor s3 because it has already accepted X.

**Y can never be accepted after X is already accepted by a majority of acceptors because there is at least one acceptor in common will change the proposer's value to X​** 
    What if P 4.5 happens before A 3.1 X in node S3? 
    A 3.1 X will be rejected in this case because its proposer ID is lower than `minProposal`, see the example below  


![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/85cd9f1a3b13ab8f2c2fd94a336acb7252c5b522ee23800603b7864840e2c363.jpg)  

The situation where the final value is Y. The blue proposer cannot gather enough accept votes **because the acceptor s3 has already promised the green proposer, which has a larger proposal ID.**  
The "promise" is only a promise of "no smaller" but can be overridden by a larger ID  

What happens to S1 and S2? -- they will **change** their value after receiving Y's accept message because it has a larger proposal ID  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/37d2df338d638861ba778a11a01af438ec314b122cbb723c5e5d9c29004191fa.jpg)  

Similar to the first example. The final value is X even though it was not accepted by a majority at the time the green proposer broadcasted its proposal ID.  

**Basic Paxos and FLP**  
(1) FLP: only two can be achieved in (fault tolerance, termination, agreement)   
(2) Paxos can lead to livelock (no termination)-- can be mitigated by certain solutions like randomization 
Also a good (bad?) example of how we increase the proposal ID to achieve "unique and monotonically increasing"  


![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/0a8e14097f32dc970a1e7179eb7778d6c24cc04d1d7fc9ac386522cb98c084a8.jpg)  

**Safety properties**  
- Validity (or non-triviality): Only proposed values can be chosen and learned. 
- Agreement (or consistency, or safety) No two distinct learners can learn different values (or there can't be more than one decided value) 
- Termination (or liveness) If value C has been proposed, then **eventually** learner L will learn some value (if sufficient processors remain non-faulty). -- live lock is possible  

**What if​**  
(1) What if the proposer crashes when its accepted proposal is only broadcast to a majority but not all the acceptors?  

How can the remaining acceptors know the accepted proposal? 
**Other proposers will learn the accepted value** at their preparation phase and **broadcast the value** to other acceptors   
Thus, multiple proposers is enough for achieving high availability  

(2) What if a proposer with a higher propose number overrides a previously accepted proposal 

Overriding is possible but it will be overridden with the same value, so that does not violate the "Agreement" property 

There must be one acceptor that is shared by two quorums. In this shared node:
    If the accept message of the former proposal comes before the prepare message of the later proposal​.The later proposer will learn the accepted message and broadcast accept with this accepted message​   
    If the accept message of the former proposal comes later than the prepare message of the later proposal​. This shared acceptor will update its `minProposal` and hence refuse to accept the accept message from the former proposal  

(3) What if we want to start another round of decision after the former is accepted?​ 

Can we directly increase our proposal number and start another round? 
No!!!! Or actually not possible in basic Paxos. Because after an acceptance, any further proposal will be revert to the original proposal by the acceptor in step 3​ 
Basic Paxos is only for making one consensus decision. 

## Replicated State Machine  
### Why Do We Need RSM?  
(1) Reaching consensus on one decision is not enough for implementing a high-available coordinator​ 
It needs a series of consensus, one for each modification.​ 

(2) Replicated state machine is a powerful abstraction that can be used to build all kinds of **high available** service as long as every operation of the service is deterministic.   

(3) The centralized structure is a **replicated log** 
Each replica works as a state machine that reads the log for operation steps, and then perform the step in order, which is similar to the Turing machine.  

### Common Procedure of RSM  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/b2b3020c7f0587f7f58d27f8f0474944a07f6ff4ad2cb88e61b3c55c46555b4f.jpg)  

(1) Client sends "command A" to one replica of the service (may or may not need to be a chosen leader?) 

(2) The received replica proposes a proposal on "command A should be the X-th command executed", which is represented by storing "command A" in the X-th slot of the replicated log 

Here, we are transferring the semantic of "consensus on value" to "replicated log"  

(3) Somehow, the consensus module successfully committed the log "command A" at the Y-th slot​  

A log is committed if 
1) at least a majority of replicas reach agreement upon this log; 
2) there will be **no further rollback** even if some (not all) of the agreed replicas crashes 

Thus there will be two states 1) replicated to a majority of replicas; and 2) committed. The first is still not safe.  

>  要确保 committed，不仅要确保 replicated to a majority of replicas，还要确保 no further rollback

The final position Y-th may not be identical to the original proposal X-th 

(4) The state machine on every replica continuously consumes committed logs and returns the result to the client **only after it has applied** the Y-th log​  

This consumption can be an asynchronous procedure to the consensus procedure -- communicated via a count "`commited_index`"  

### Why replicated log?  
The service keeps the state machine state. The log is an alternate representation of the same information! why both?  

Reason:
(1) Log stores the ordered history of committed commands 
to help replicas agree on a single execution order 
to help the leader ensure followers have identical logs 

(2) Log can store tentative commands until committed -- otherwise , it is hard to rollback a command from the state machine  

We'll see that they can temporarily have different entries on different replicas (that are not committed)   
These different entries eventually converge to be identical​   
The commit mechanism ensures servers only execute stable entries  

(3) Logs are helpful in recovery  
the log stores commands in case leader must re-send to followers 
the log stores commands persistently for replay after reboot  

**How to Implement RSM with Paxos? -- Multi-Paxos**  
Multiple rounds of basic-Paxos for leader election and for replicating logs 
Very hard to understand and to use in practice $=>$ the main motivation of Raft​  

## Raft Model  
### Basic Procedure  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/1bbf8563a021383c36bfa6b9be394a4ba01f395366896c64b9e791d213b6323f.jpg)  

Raft directly provides a RSM model that basically contains only two procedures: 
1) Electing a new leader; and 
2) Ensuring identical logs despite failures  

In Raft, time is divided into **terms**  
    Each term begins with an election.   
    After a successful election, a single leader manages the cluster until the end of the term. $=>$ replicating and committing logs   
    Some elections fail, in which case the term ends without choosing a leader $=>$ re-election  

Every server in the Raft cluster maintains a **monotonically increasing currentTerm** value 
This value should be persistent for recovering from a restart​  

**Why Strong Leader?**  
Paxos can have multiple proposers, but only one leader in Raft can propose. A leader ensures all replicas execute the same commands, in the same order -- decided by the single leader   

Raft uses a stronger form of leadership than other consensus algorithms.
    For example, log entries only flow from the leader to other servers.
    This simplifies the management of the replicated log and makes Raft easier to understand​   

Leader can become a bottleneck in certain circumstances (e.g., geo-replicated systems) and lead to less parallelism and hence less throughput. This is a trade-off between understandability and performance -- understandability wins.

### Leader Election in Raft  
#### State of Servers in Raft and the Transition  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/a440959b541e8f558934811962d59f663c4a02b5a415621355a73bc4844733a7.jpg)  

(1) There are only three states of servers and two kinds of RPC calls in Raft (very elegant!)  

(2) A server starts as Follower that receives and replies `AppendEntries` RPC from the leader 
The RPC tries to append a log entry to the replicated log​
The RPC can also contain an empty log entry, which serves as a heartbeat from the leader

(3) A follower timeout if not receiving valid `AppendEntries` RPC for a certain period of time. It then becomes a candidate and broadcasts `RequestVote` RPC 

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/13b05b10a30af9ad0680831382462af7056162c65d6bb86969606eca4814ecdc.jpg)  

The candidate starts an election by increasing the `currentTerm` it stores and sending ` RequestVote ` RPC with this new term   

A candidate becomes a new leader if it receives votes from a majority of servers   
A candidate may notice that another server becomes a new leader (by receiving its `AppendEntries` RPC), and then it transfers to a follower   

Sometimes no candidate receives enough votes, which leads to a new election   

(4) A leader server X periodically broadcasts `AppendEntries` RPC to make sure that it is still the leader   

leader must send heart-beats more often than the election timeout to suppress any new election​   

But it is possible that this leader is temporally isolated from a majority number of servers (e.g., due to a network partition) . In that case, the other followers may timeout and elect a new leader Y.   

After reconnecting with the other servers, this (previous) leader X will receive `AppendEntries` RPC from the new leader Y (by comparing the term number) and becomes a follower 

Is there a split brain problem?  

#### Asynchronous of Raft  
The transitions between terms may be observed at different times on different servers.  

(1) It is guaranteed that there is at most one leader for **each term**  
So if you see `AppendEntries` with term T, you know who the leader for T is 

(2) But, physically, there can be multiple servers that are both leaders at the same time because they are currently in different terms (have different `currentTerm`)​ 

(3) How to solve this problem to ensure there is no split brain problem? 

New leader means a majority of servers have incremented `currentTerm` 
Old leader with an older `currentTerm` **will not** be able to gather enough requests for commit a log
- Every majority set will contain a server that has a larger `currentTerm` than the old leader​. 
- Hence old leaders are harmless and there is no split brain  

It will also eventually discover the new leader and become a follower
But a minority may accept old leader's `AppendEntries` so logs may diverge at end of old term​  
- Fixed later -- we need some kind of rollback mechanism
- Again, the replicating of logs is not the same as the commitment of logs  

#### Unsuccessful Election  
(1) There are two possible reasons of unsuccessful election   

Case 1: less than a majority of servers are reachable -- the whole cluster becomes unavailable in this case  

Case 2: simultaneous candidates split the vote, none gets majority  
- Without special care, elections will often fail due to split vote   
- all election timers likely to go off at around the same time $=>$ every candidate votes for itself $=>$ no-one will vote for anyone else $=>$ time out at around the same time again $=>$ repeat​  

(2) How to reduce the possibility of case 2?  

randomized election  
- Each server adds some randomness to its election timeout period
- The server with the lowest timeout has a good chance of becoming the new leader 
- Indefinite re-election is still possible but with a very low possibility  

randomized delays are a common pattern in network protocols for resolving conflict without synchronization  

(3) Other solutions? A priority of leader selection -- possible but leads to many subtle issues and is abandoned by Raft designers  

#### Malicious/manipulate Server  
In Raft, we assume that all the messages are sent according to the algorithm. 

That is to say, a server will always believe the content of message it receives.
This can not be true because of a bug or a malicious/manipulate server.

Algorithms that can tolerate (called Byzantine fault) are much more complex and require cryptography backgrounds​. The most well-known Byzantine system is bitcoin -- one can learn it from the material of the original MIT 6.824 course. It is not included in this class  

### Log Replication in Raft  
#### Replicating and Committing Logs Without Leadership Change  
(1) Once a leader has been elected, it begins servicing client requests  

The leader appends the command from the client to its local log as a new entry -- with a term Id (`currentTerm` of the leader) and a `logIndex` that represents its location in the log (i.e., the execution order)   

This is similar to GFS, where the position of appending is decided by the leader, not directly append to the follower's log.

The leader then broadcasting the above log in parallel with `AppendEntries` RPC.  
If followers crash or run slowly, or if network packets are lost, the leader retries `AppendEntries` RPCs indefinitely   

A log entry is committed once the leader that **created** the entry has replicated it on a majority of the servers​   

When the entry has been safely replicated (i.e., committed), the leader applies the entry to its state machine and returns the result of that execution to the client.   

As for the other minority set of followers, the leader will still reties `AppendEntries`  even after it has responded to the client until all followers eventually store all log entries.  

(2) Example  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/04e3402d1310691bd9c264bb02cf17b5d1489fcd1e4d392e9012ec1c21160e51.jpg)  

In the above example, entry 7 is committed because it is replicated to a majority by its **corresponding** leader
Raft guarantees that committed entries are durable and will eventually be executed by al of the available state machines. 

Pay attention to the phrase "the leader that created the entry" $=>$ we will see later an example, in which, an entry is **already replicated to a majority of the servers but still not committed**, because it is not replicated by the leader who created the entry.

(3) The leader keeps track of the highest index it knows to be committed, and it **includes that index** in future `AppendEntries` RPCs (including heartbeats) so that the other servers eventually find out. 

Once a follower learns that a log entry is committed, it applies the entry to its local state machine (in log order).  

This information is inlined in `AppendEntries` so that we do need an additional round of RPC​  

(4) The above algorithm is enough to converge the logs on all nodes if there is no failure  

All the followers have a prefix of the leader's log in the above example, thus retry is enough   
But this is only a lucky special case   
There are complex divergences in real-world cases and we need some kind of roll back mechanism  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/c79bdc025bc860397928fc535d8a0b4cd6f47391b9d81f7399a1465752797c32.jpg)  

(5) Discuss later  

To converge the difference, the leader must find the latest log entry **where the two logs agree**, delete any entries in the follower’s log after that point, and send the follower all of the leader’s entries after that point.  

The leader maintains a `nextIndex` for each follower.   
`nextIndex` is initialized to be the last index for a new leader   
After a rejection from the follower, the leader decrements `nextIndex` and retries.
Eventually `nextIndex` will reach a point where the leader and follower logs match In the above example, server f will rollback to log index 4  

#### Log Matching Property  
P1: If two entries in different logs have the same index and term, then they store the same command.  
- straightforward, because only one master per term and it will choose only one content for every index​
- same index $+$ different term $=$ different content  

P2: If two entries in different logs have the same index and term, then the logs are identical in all preceding entries $=>$ even when these two entries are not committed 
- Benefit: if we match the entries one by one from the end to front, once a match is found, we do not need to compare all the preceding entries because they are guaranteed to be the same. -> Thus, the suffix is the divergent part that should be rolled back​ 
- How to achieve this goal? -- roll back the divergent log entries
- The situation becomes more complex when the leader fails, because the replicating of logs is executed parallel and the leader can commit a log for only a majority (not all) of servers reply  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/1cf7932bda40baa59843e1f43a58089b34dd6e444cd172caac67448d61005192.jpg)  

In the above example,  

leaders in term 2 and 3 failed before committing any log entry but replicating several logs in server f​ 
The current leader directly jumps from term 1 to term 4​   
If the current leader of term 8 broadcasts a new log entry at index 11 and server f blindly accepts it, there will be a violation of P2  

To resolve the above problem, when sending an `AppendEntries` RPC in raft, the leader includes the index and term of the entry in its log that immediately precedes the new entries.  
- If the follower does not find an entry in its log with the same index and term, then it refuses the new entries.   
- The consistency check acts as an induction step (prove by mathematical induction，数学归纳法): 
    1: the initial empty state of the logs satisfies the P2 
    2: and the consistency check preserves the P2 whenever logs are extended. 
        Assuming P2 is guaranteed with index i-1 
        for appending index i, the algorithm will check i-1 
        If the entries at index i-1 have the same term and same content​
        Then, due to the assumption, all the entries before i-1 are also the same  

To converge the difference, the leader must find the latest log entry where the two logs agree, delete any entries in the follower’s log after that point, and send the follower all of the leader’s entries after that point.  
- The leader maintains a `nextIndex` for each follower. 
- `nextIndex` is initialized to be the last index for a new leader -- 11 in the above example   
- After a rejection from the follower because the mismatch on the precede entry, the leader decrements `nextIndex` and retries 
- Eventually, the `nextIndex` will reach a point where the leader and follower logs match - - at least they will agree on 0 In the above example, server f will rollback to log index 4  

How to assure that a committed log entry will never be rolled back?  
P1 and P2 only guarantees the "matching/replicated", not related to "committed"  

### Safety  
The mechanisms described so far are not quite sufficient to ensure that each state machine executes exactly the same commands in the same order. -- can you image a counter example?  

#### Election restriction  
A follower might be unavailable while the leader commits several log entries
- A lag follower does not violate the above Log Matching Property
- But if a lag follower is elected as a new leader, the committed commands will be overwritten   

Solution: selecting only the candidate that stores all of the committed log entries 
- Follower a, c, d are also eligible for becoming the new leader in the above example  

Implementation: a `RequestVote` RPC is rejected if the sender candidate's log is staler than the receiver  
- Again, we take advantage of the quorum property   
- A log entry is committed only when it is replicated to at least a majority number of servers   
- A lag candidate can never receive agreement votes from any of these servers and hence cannot be elected as a new leader   
- Raft determines which of two logs is more up-to-date by comparing the index and term of the last entries in the logs. 
    If the logs have last entries with different terms, then the log with the later term is more up-to-date.
    If the logs end with the same term, then whichever log is longer is more up-to-date.  

Why not elect the server with the longest log as leader?  
- term is more important than length  
- A failed leader may replicate a lot of log entries to only a minority of nodes, which leads to long but stale nodes 
- Node f in the above example is very long but with very small term number​  
#### Committing entries from previous terms  
Corner Case: a leader cannot immediately conclude that an entry **from a previous term** is committed once it is stored on a majority of servers  
- Not committed because it is not created by the current leader  
- It is replicated by the current leader, not the failed leader that creates it
- Thus, even when it is replicated to a majority of nodes by the new leader, **the latest term of these nodes are still stale**
- hence cannot guarantee that his replicated log entry will not be rolled back​. 

>  不是由 current leader 在 current term 确定的存储于 majority 的 entry 不能提交，即来自于 previous term 的 entry 即便确定了存储与 majority，也不能视作提交
>  这是因为这些 entry 存在被覆盖的可能性，它们的 term number 可能会落后于某些 term number
>  如果 entry 是当前 term 的，则其 term number 就是最大，故不存在被覆盖的可能性

In contrast, if it is created by the current leader, its term is guaranteed to be the larges  

An example (the most important example)  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/7135f4ac6ddcd601a9ca1a4a5108a358b8e7402c3a8e89f95272fcb76f7efd0e.jpg)  

S1 crashes at time c and S5 is elected as leader of term 5  
- S5 gets votes from S4 and S3 because the term of $\mathtt{S3^{\prime}S}$ last entry is 2 which is less than S5's 3​ 
- 2 is rollbacked and overwritten by 3 although it has been replicated by a majority of servers at time c​
    If the 4 at logIndex 3 is also replicated to a majority, then it will never be rolled back any more 
- Subfigure (e) is not a valid successive state of (d), it is a possible successive state of (c), in which the entry with term 4 is committed, and hence the entry with term 2 is also committed.  


Raft never commits log entries from **previous** terms by counting replicas. Only log entries from the leader’s current term are committed by counting replicas; 
- once an entry from the current term has been committed in this way, then all prior entries are committed **indirectly** because of the Log Matching Property.​ 
- A no-op (sync) is committed for every newly elected leader to commit all the previous logs   

Why?​ 
- New leader cannot roll back committed entries from a previous term, because of the leader election restriction 
- But a new leader that replicates an uncommitted entry to a majority set of servers does not have the same guarantee, because the term of this uncommitted entry may be lower than the new leader
- In contrast, the log entry created by current leader and replicated to a majority of nodes has the newest term  

### Recovery  
#### What would we like to happen after a server crashes?  
Raft can continue with one missing server but failed server must be repaired soon to avoid dipping below a majority 
Replace with a fresh (empty) server $=>$ long recovery time because it must catch up from the start​   
Reboot crashed server, re-join with state intact, catch up $=>$ by persistence  

#### Persistence  
Picture ref to the original pdf

`log[]`, persistent -- obvious 
`votedFor`, persistent -- to make sure that a server will not vote twice in a term  
`currentTerm`, persistent 
- why not directly use the term of the last log entry? 
- not correct, similar to `votedFor` a node may voted for a newer term but still have not received log entry from that term 

`commitIndex`, not persistent -- will receive it from the leader 
`lastApplied `, not persistent -- it is not persistent in Raft because its persistency does not impact the property guaranteed by Raft, but its persistency **should be considered** by the implementation of the state machine 
- Raft only guarantees the log is consistently replicated​ 
- The state is always correct if the state machine always starts from initialization state an executes all the committed logs to restore the state after restart​
- If the state machine is also recovered from a snapshot, the state machine should record the `lastApplied` at the checkpoint instead of the last ` lastApplied ` (discussed later)  
`nextIndex/matchIndex`, not persistent -- reinitialized after each election  

#### Performance  
Persistence is often the bottleneck for performance -- a hard disk write takes 10 ms, SSD write takes 0.1 ms. So persistence limits us to 100 to 10,000 ops/second   

Lots of tricks to cope with slowness of persistence
- batch many new log entries per disk write 
- persist to battery-backed RAM, not disk 
- be lazy and risk loss of last few committed updates  

#### Snapshot  
**Problem and Basic Idea**  
Problem: log will get to be huge -- much larger than state-machine state! It will take a long time to re-play on reboot or send to a new server   
Idea: the executed part of the log is captured in the state and hence we can make a snapshot of state  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/0e4ebd251da09f11c7bfd7b418003a5bbef7d7744fc298b3401ddb50d9768cfc.jpg)  

With a persistent snapshot, a restart server can set the `lastApplied` to the last included index of the snapshot and go through the following log entries for recovery  

**Problem of Lag Follower**  
As discussed above, a leader must find the latest log entry where the two logs agree, delete any entries in the follower’s log after that point, and send the follower all of the leader’s entries after that point.   

It is possible that the latest log entry where the two logs agree is **already deleted** (compacted into the snapshot)   

An additional `InstallSnapshot` RPC is used in this special case to install the snapshot of leader to this lag follower  

### Client Interaction
#### Client Interaction (1) -- linearizability  
Client operation **lasts for a period of time**, which starts from its invocation and ends when the leader returns the results  
- Each operation appears to take effect atomically at some point between its invocation and completion. -- but what point?   
- Thus, the **lasting period of two concurrent client operations** may intersect with each other -- which one is executed earlier?​   
- The guarantee of the above problem is called a consistency model.  

**Linearizability is the consistency model Raft achieves** and it is the most common version of **strong consistency** model that can be achieved in a distributed environment​  
- Definition: an execution history is linearizable if 
    one can find a total order of all operations, that matches real-time (for nonoverlapping ops), 
    and in which **each read sees the value from the write preceding it in the order**
- Example 1: Linearizable as Wx1 Rx1 Wx2 Rx2 (although Rx2 actually starts before Rx1 but read a newer version of value)  
    picture ref to the origin
- Example 2: Linearizable as Wx0 Wx2 Rx2 Wx1 Rx1  
    picture ref to the origin
- Example 3: not linearizable  
    picture ref to the origin

>  在对 operations 进行排序时，有 intersection 的两个 operations 之间的顺序可以任意

In Raft, **the log order is the linearizable order** 
linearizability $\ne$ serializability  
- linearizability is a consistency model regarding the consistency among multiple replicas 
- serializability is an isolation level for database transactions -- concurrent transaction on even one database server needs to guarantee serializability   
- We will revisit this in later lectures  

#### Client Interaction (2) -- Duplication caused by retrying  
"a committed operation" has three meanings in Raft:  
- P1) the op cannot be lost, even due to (allowable) failures. In Raft, when a majority of servers persist it in their logs created by the current leader. This is the "commit point" of Raft 
- P2) the system knows the op is committed -- the leader saw a majority.
- P3) the client knows the op is committed -- the client receives a reply from the leader  

A client will resend the op if timeout, but the sent op may already pass P1 -- leads to a duplication​  

Basic Idea: duplicate RPC detection (Lec 1.2)  
- client picks a unique ID for each request, sends in RPC same ID in re-sends of same RPC   
- the service maintains a "duplicate table" indexed by ID. after executing, record reply content in duplicate table   
- if 2nd RPC arrives with the same ID, it's a duplicate generate reply from the value in the table​  

Problem: how does a new leader get the duplicate table?  
- put ID in logged operations handed to Raft 
- all replicas should update their duplicate tables as they execute so the information is already there if they become leader   
- This mechanism also solves the problem of recovering the duplicate table after a restart  

>  request ID 也会被存储于 log 中，便于每个 server 维护各自的 duplicate table，故当前 leader 崩溃后，之后的 leader 将仍持有 duplicate table

Another corner case: what if a duplicate request arrives before the original executes?
- could just call Start() (again) it will probably appear twice in the log (same client ID, same seq #)​ 
- when cmd appears on applyCh, don't execute if table says already seen  

#### Client Interaction (3) -- Read-only Optimization  
The above standardized procedure of executing an operation is expensive (at least two round-trips). Can we optimize read-only operations by directly returning from the state machine? 

Simple answer: No, at least not straightforward
- The leader who received the read-only operation might have recently lost an election, but not realized 
- The new leader may already modify the value, which leads to a split brain problem that violates linearizability​ 

Possible Solution: lease 
- define a lease period, e.g. 5 seconds 
- after each time the leader gets an `AppendEntries` majority, it is entitled to respond to read-only requests for a lease period without adding read-only requests to the log, i.e. without sending `AppendEntries`. 
- a new leader cannot execute Put()s until previous lease period has expired result: faster read-only operations, still linearizable.  

Another solution is also described in the paper  

>  为了确保 leader 对于只读请求的返回是 up-to-date 的, leader 需要知道哪些 entries 是已经提交的最新信息，并且需要在处理只读请求之前，检查它是否已被替代

### Cluster Membership Changes  
Problem: It isn’t possible to atomically switch all of the servers at once, so the cluster can potentially split into two independent majorities during the configuration transition  

![](https://cdn-mineru.openxlab.org.cn/extract/281591fa-67f7-4729-a01a-26dee8ec29b6/ca39b5daf48a687efbb4f13e2314c062db2e618240725b6110de82609b7901d0.jpg)  

The state is first changed to a joint configuration $C_{old, new}$, during which all the decisions should reach majority consensus in both the two configurations   

Both $C_{old, new}$ and $C_{new}$ are special log entries that are replicated and committed with the same mechanism of other operations  

A Visualization of Raft: https://thesecretlivesofdata.com/raft  

# 6 ZooKeeper 
## Problem Definition  
(1) Replicated State Machine is a generalized framework that replicates computation on a cluster of machines -- states replication as a side-effect of deterministic computation replication 

In contrast, Zookeeper is directly based on state replication   

(2) It is possible but still complex to directly use RSM at application level
- Molding the whole business logic into a single state machine is very hard and also not efficient​ 
    RSM is easier than consensus but still not very similar to the concurrency programming in a multi-thread environment where we can directly use concurrent data structures such as concurrent queues, locks, etc.​ 
    Also, we may have certain parts of the program that can be naturally partitioned into disjoint processing units. There is no need to replicate them one by one 
- Most of the parts in the system can directly be executed in parallel and recovered by redo, while only certain points of the application need to be synchronized -- not all the steps should be ordered and executed step by step
- We need further abstraction to achieve ease-of-use! -- just some synchronization functionalities that we have used in a multi-thread program   

(3) Many common replicate (and hence highly available) services are already built -- each replicates a specific (but widely used) kind of computation 
- Lock service -- distributed lock, such as Google Chubby
- Queue service -- provide a FIFO queue 
- What about a LIFO queue? A Read-write lock? Do we really need to build these services one by one?   

(4) Can we provide a set of APIs that enable the users to design their own replicated service in about $10{\sim}100$ lines of code instead of thousands lines of code? 
- We need a **higher level abstraction** than RSM
- How to trade-off between flexibility and ease-of-use? -- similar to the computation model design problem of MapReduce  
- What kind of consistency guarantee should be provided -- how to trade-off between consistency and performance? -- RSM is essentially a serial model (linearizability) that can not scale  
    Also a relaxed consistency model is used in ZooKeeper but much more understandable than GFS   
    Socket $\rightarrow\sf R P C\rightarrow$ consensus Protocol -> RSM -> ZooKeeper -> Distributed Data Structure -> Distributed Infra -> Distributed App  

(5) Today we will emphasize on the abstraction of ZooKeeper and how to use it  
- ZooKeeper was originally developed at Yahoo! to streamline the processes running on big data clusters.   
- It started out as a sub-project of Hadoop, but became a standalone Apache Foundation project in 2008.   
- The debut of RAFT is 2013 later than ZooKeeper. Thus, the reason why ZooKeeper's Zab protocol is very similar to RAFT is actually because RAFT is leaning from Zab 
- There is a RAFT based version of ZooKeeper called etcd, which is also very (even more) popular today​  

## ZooKeeper's Main Architecture  

![](https://cdn-mineru.openxlab.org.cn/extract/release/17e8f0b9-bbf3-4205-bc33-a174909991be/a8918b76f7d3dbd6da822f304e952143da23c50720d915745dd3453d668c6d47.jpg)  

(1) Strong leader based replication  
- All requests that update ZooKeeper state are forwarded to the leader.   
- A raft like leader election algorithm is executed if the leader fails​   


![](https://cdn-mineru.openxlab.org.cn/extract/release/17e8f0b9-bbf3-4205-bc33-a174909991be/e3289e956b8e77ca9a42f04de6651173f6b39cc21dc21bb049525eb4176c08c1.jpg)  

(2) Replication via Zab (ZooKeeper Atomic Broadcast), an atomic broadcast protocol   
- discussed later​   
- Essentially broadcast one message and hence it is not as general as Raft's replicated log abstraction​  
- a single order for all writes -- ZK leader's log order. "zxid"  
    Even though there is no explicit form of "log" in ZK, this zxid is actually the logIndex in RAFT  
- all clients see writes appear in zxid order. including writes by other clients.  
>  Zab 中，由 leader 按照 `zxid` 组织所有 writes 的顺序
>  `zxid` 等价于 Raft 中的 `logindex`

![](https://cdn-mineru.openxlab.org.cn/extract/release/17e8f0b9-bbf3-4205-bc33-a174909991be/45fa87ea032817776820751ca812fb2407b72b606e58708b96d30826518fd3be.jpg)  

(3) Client can **directly** read from a follower server for read performance 
- A read may not see latest completed writes! -- **not linearizable** 
    A read to the lease-protected leader is needed for linearizability  
- As a compensation ZK provides two more guarantees:  
    **Sequential Consistency:** Updates from a client will be applied in the order that they were sent.  
    **Single System Image:** A client will see the same view of the service regardless of the server that it connects to. i.e., a client will never see an older view of the system even if the client fails over to a different server with the same session.  
        Client switches over to Follower 2 after its connection to Follower 1 is lost
        As long as the client maintain the same "session", the client is guaranteed to "never see an older view of the system",  even if Follower 1 is currently more up-to-date than Follower 2. 
- Implemented straightforwardly with the above monotonical zxid  
    The current read zxid is preserved in the current session 
    After reconnecting, the client will ask the connected follower to first sync to this zxid  
- Justified later  

>  ZooKeeper 为单个 client 提供了 Sequential Consistency 和 Single System Image 保证
>  Single System Image 保证通过 `zxid` 实现

(4) One sentence: a strong leader based replicated state machine that provides **linearizable write and allows non-linearizable but ordered read**.

## ZooKeeper's Hierarchical Namespace Abstraction 

![](https://cdn-mineru.openxlab.org.cn/extract/release/17e8f0b9-bbf3-4205-bc33-a174909991be/d269c567d306d5adfefbe522696a79ab517e4c75a4630d7646b57464e0637a88.jpg)  

(1) ZooKeeper provides to its clients the abstraction of a set of data nodes (**znodes**), organized according to a hierarchical namespace.  
- ZooKeeper's APIs set is similar to a combination of key-value store and file system, i.e., a key/value table with hierarchical keys (e.g., etcd)  
- Only **full** data reads (Get) and writes (Put) are allowed to a specific znode  
    - Compare to the flexible dynamic data structures supported by Redis  
        Redis Cluster does not guarantee strong consistency. In practical terms this means that under certain conditions it is possible that Redis Cluster **will lose writes** that were acknowledged by the system to the client. The first reason why Redis Cluster can lose writes is because it uses asynchronous replication. https://redis.io/docs/management/scaling/  
    - The main scenario of using Redis is cache, which can tolerate loss of write  
- Why hierarchical?  
    - Why not provide just flat key-value and the users use a special value that contains a list of keys to mimic a hierarchical structure?   
    - Simplicity: a useful abstraction because configurations in the real-world are naturally organized as a hierarchical structure   
    - Functionality: ZooKeeper provides **a sequential guarantee under a specific prefix** 
        As a compensation of the lack of linearizability, very elegant and useful!  

(2) `Create(path, data, flags)`
- Creates a znode with path name `path`, stores `data[]` in it, and returns the **name** of the new znode  
    Why return a name after given a "path" in the input? -- enable the append of id suffix 
- `flags` enables a client to select the type of znode: regular, ephemeral, and set the sequential/exclusive flag​  
    - Regular: Clients manipulate regular znodes by creating and deleting them explicitly 
    - Ephemeral: Clients create znodes that are either delete them explicitly, or let the system remove them **automatically** when the session that creates them terminates (deliberately or due to a failure)
        Heartbeat is automatically maintained between ZooKeeper and the client within the session​  
- Sequential flag (`Regular_Sequential` or `Ephemeral_Sequential`) 
    Nodes created with the sequential flag set have the value of a **monotonically** increasing counter appended to its name.  
    **\[IMPOARTANT\]** If n is the new znode and p is the parent znode, then the sequence value of n is **never smaller** than the value in the name of any other sequential znode ever created under p -- we will show how to use this guarantee later  
- **exclusive** -- only the first creation indicates success 

```
[zk: localhost:2181(coNNECTED) 9] create -e -s /test_znode/child_nodeA “this first is epemeral seq node data"   
Created/test_znode/child_nodeA0000000000   
[zk: localhost:2181(coNNECTED) 10] create -e -s /test_znode/child_nodeB "this second is epemeral seq node data"   
Created /test_znode/child_nodeB0000000001   
[zk:localhost:2181(CONNECTED) 11] ls /test_znode   
[child_nodeB0000000001,child_nodeA0000000000]   
[zk:localhost:2181(CONNECTED) 12]  
```

(3) `setData(path, data, version)` / `Delete (path, version)`
- Modify the value or delete the path  
- Version  
    - znode is associated with time stamps and version counters as meta-data 
    - Each time a znode's data changes, the version number increases.  
    - **\[IMPOARTANT\]** **Conditional updates:** if the version it supplies doesn't match the actual version of the data, the update will fail.  
        Similar to the usage of compare-and-swap instruction in multi-thread programming  

(4) `exists(path, watch)` / `getData(path, watch)` / `getChildren(path, watch)`  
- Return the existence / data (and meta data such as data version) / set of names of the children of path  
- `watch = true` enables a client to set a watch on the znode 
    - ZooKeeper's definition of a watch: a watch event is one-time trigger, sent to the client that set the watch, which occurs when the data for which the watch was set changes.  
    - One-time trigger -- only the first change will trigger a notification (needs to watch again if it is needed)  
    - Sent to the client  
        - Watches are sent **asynchronously** to watchers 
        - **\[IMPOARTANT\]** ZooKeeper provides an ordering guarantee: a client will never see a change for which it has set a watch until it **first sees the watch event.**  
            This order is guaranteed even though the client can only communicate with a non-leader replica for these read operations, e.g., read a non-leader replica A, the change is made on leader and still not replicated to A​  
    - The data for which the watch was set  
        - Different operations watch different events (see [here](https://zookeeper.apache.org/doc/r3.6.3/zookeeperProgrammers.html#ch_zkWatches) for more) 
        - Created event: Enabled with a call to `exists`.   
        - Deleted event: Enabled with a call to `exists`, ` getData `, and ` getChildren `.   
        - Changed event: Enabled with a call to `exists` and ` getData `.   
        - Child event: Enabled with a call to `getChildren`.  

(5) `sync(path)`: waits for all updates pending **at the start** of the operation to **propagate to the server that the client is connected to**  
- ZooKeeper does not guarantee that at every instance in time, two different clients will have identical views of ZooKeeper data.   
- If client A sets the value of a znode `/a` from 0 to 1, then tells client B to read `/a`, client B may read the old value of 0, depending on which server it is connected to.   
- Client B should call the `sync()` method from the ZooKeeper API method before it perform its read   
- How? `sync()` is essentially **a write operation** with empty write content. It will be **redirected to leader** and **force the current connecting replica to sync with the leader**  
    - In the above example, whether B reads 0 or 1 on `/a` depends on whether the sync operation of B is ordered before or later than the write operation from A​ 
    - Both cases are possible and they all guarantee linearizability  

## Examples of Using ZooKeeper's Abstraction 
### Distributed Lock​  
#### Exclusive Locks  
(1) Simple Approach   
- Concurrently create an **ephemeral** znode "/path/to/lock/file", only one will succeed and acquire the lock​ 
- Explicitly deletes the lock file for releasing or removed automatically because of network timeout​ 
    Network partition? -- again, use lease (already supported by the ephemeral znode)   

>  lock file 会在 network timeout 后被自动移除，如果 network timeout 只是因为 network partition 导致的，client 恢复后将不具备 lock
>  故 client 需要检查自己当前是否已有 lock，再确认自己是否有资格执行操作

- Others watch "/path/to/lock/file" for the liveness of the lock, try to create it again **if a delete notification is received**   
- **Herd Effect:** concurrent writing for every race but only one will success -- **a waste of network**  

(2) Complex Approach  

```
Lock  

1 n = Create(l + “/lock-"， EPHEMERALI | SEQUENTIAL)
2 C = getChildren(l, false)   
3 if n is lowest znode in C,exit   
4 p = znode in C ordered just before n  

Unlock  

1 delete(n)   
```

- Setting the sequential flag so that each child znode is attached with a unique sequence value   
- The child with smallest sequence value is the leader   
- Each child only watching the znode that **precedes** the client’s znode -- how to maintain  

#### Read Write Lock  

```
Write Lock  

1 n = Create(l + “/write-"， EPHEMERAL|SEQUENTIAL)   
2 C = getChildren(l, false)   
3 if n is lowest znode in C，exit   
4 if exists(p, true) znode in C ordered just before n   
5   
6 goto2  
```

```
Read Lock  

1 n = Create(l + “/read-"， EPHEMERAL|SEQUENTIAL)   
2 C = getChildren(l, false)   
3 if no write znodes lower than n in C, exit 
4 p = write znode in C ordered just before n   
5 if exits(p, true) wait for event 
6 goto3  
```

The flexibility of using the sequence value (only possible with the built-in hierarchical structure)  

**Corner Case: recoverable exceptions**  
- The above recipes employ sequential ephemeral nodes.   
- When creating a sequential ephemeral node there is an error case in which the `create()` **succeeds** on the server but the server **crashes before returning the name of the node to the client.**   
- When the client reconnects, its **session is still valid** and, thus, the node is not removed. 
- The implication is that it is difficult for the client to know if its node **was created or not**.   
- Solution 
    Assign a Global Unique ID (GUID) to each client and use this GUID as a prefix of the path "l + /guid-lock-" 
    If a recoverable error occurs calling create() the client should call `getChildren()` and check for a node containing the guid used in the path name  

>  为了在上述的 corner case 中让 client 确认 server 中关于它所请求的 znodes 的具体存在与否，需要为 client 赋予全局独立 id，将 client 所请求的 znodes 的路径都与该 id 关联
>  这样，client 可以通过 `getChildren()` ，确认 server 中存在的和它相关的 znodes 具体有哪些

**More​**  
More in https://zookeeper.apache.org/doc/r3.6.3/zookeeperTutorial.html and https://zookeeper.apache.org/doc/r3.6.3/recipes.html  

### Cluster Management  
A typical structure.

![](https://cdn-mineru.openxlab.org.cn/extract/release/17e8f0b9-bbf3-4205-bc33-a174909991be/d87fb7285bda55e81401156af6ec03d280cbec45877db469b1991a26a04cc0e8.jpg)  


**Leader Election:** similar to lock  

**Group Membership:**   
- Ephemeral znodes within the same specific parent znode 
- `getChildren(.., watch=true)` for membership changes  

**Configuration Management**  
- A specific path for configuration 
- Use watch to get notifications of configuration changes  

**Atomic Modification Problem**  
**One client** (maybe the master) wants to change a large number of configurations that are stored at multiple znodes and **publishes them atomically** (i.e., no other server sees partial configurations)  

Solution 1 -- communicated via only ZooKeeper  
- The master creates a "/path/to/**ready**" znode and the others watch on this znode   
- At the start of modifications, the master deletes "/path/to/ready" to indicate that the configuration znodes are currently not available to read​ 
- Then, the master can modify the configuration znodes in parallel 
- Finally, the master re-creates the "/path/to/ready" znode to indicate the other servers can fetch the new configuration
    Because of the **FIFO** guarantee, if ZooKeeper client sees "ready" znode, it will see **updates that preceded it​**.

What if a server first (1) checks the ready and then (2) read configuration, but the master deletes the ready just **in between** (1) and (2)? Is it possible that this server reads a partial state?  

```
Write order: 
1 
2
3
4 delete("ready")   
5 write f1   
6 write f2   
7  
8 create("ready")  
```

```
Read order:
1
2 exists("ready", watch=true)   
3 read f1   
4
5
6
7 read f2  
8
```

- Protected by the guaranteeing of watch order   
- The server (client) should check "/path/to/ready" with watch=true 
- Then the system guarantees that the deletion of ready **comes first before it can read any of the new configurations.**  

>  ZooKeeper 在修改某个 znode 时，先确保所有对该 znode 设置了 watch 的 clients 收到了通知，再执行修改

What if servers had their own communication channels other than the built-in watch?  
Use sync to assure that causes a server to apply all pending write requests before processing the read without the overhead of a full write  

## ZooKeeper Guarantees Revisit  
- Linearizable writes  
    - all requests that update the state of ZooKeeper are serializable and respect precedence 
    - a single order for all writes -- ZK leader's log order. "zxid" 
    - all clients see writes appear in zxid order. including writes by other clients.  
    - Sequential Consistency: Updates from a client will be applied in the order that they were sent.  
    - Atomicity : Updates either succeed or fail -- there are no partial results.  
        Only single node atomicity 
        Multi nodes atomicity can be achieved with a "ready" flag  
- FIFO read order  
    - Read is not linearizable but ordered 
    - all requests from a given client are executed in the order that they were sent by the client 
    - Single System Image: A client will see the same view of the service regardless of the server that it connects to. i.e., a client will never see an older view of the system even if the client fails over to a different server with the same session. 
    - Timeliness : The clients view of the system is guaranteed to be up-to-date within a certain time bound (on the order of tens of seconds). Either system changes will be seen by a client within this bound, or the client will detect a service outage.   
- Watch order: a client will never see a change for which it has set a watch until it first sees the watch event   
- Liveness: if a majority of ZooKeeper servers are active and communicating the service will be available   
- Durability: if the ZooKeeper service responds successfully to a change request, that change persists across any number of failures as long as a quorum of servers is eventually able to recover.  

## Advantage of ZooKeeper's Abstraction  
well tuned for concurrency and synchronization  
- exclusive file creation; exactly one concurrent create returns success​   
- `getData()` / `setData(x, version)` supports mini-transactions   
- sessions automate actions when clients fail (e.g. release lock on failure)   
- sequential files create order among multiple clients   
- watches avoid polling   
- read from followers for performance and watch/sync for a certain kind of consistency   
- clients of ZK launch many async operations without waiting​ 
    ZK processes them efficiently in a batch; fewer msgs, disk writes   
    client library numbers them, ZK executes them in that order  
- A-linearizability (asynchronous linearizability)  

## Zookeeper (Zab) V.S. Raft  
- Replica States: LEADING/FOLLOWING/LOOKING in Zab = Leader/Follower/Candidate in Raft​ 
- Replication  
    - Same: broadcast to all the $2\mathsf{n}+1$ replicas by leader, committed after receiving $\mathsf{n}+\mathsf{1}$ ack 
    - Not the same  
        Replicate state in Zab, Replicate Computation in Raft 
        Zab is essentially a primary-backup system, Raft is a replicated state machine 
        Raft is more flexible because of the arbitrary operation logic can be implemented and recorded in the replicated log -- only write operation is supported in Zab  

>  Zab 存储状态而 Raft 存储操作
>  回忆起在 ZooKeeper 中，master server 在接收到请求后，需要先根据请求计算出执行请求后会处于什么状态，再利用 Zab 广播该状态
>  如果是 Raft，则会直接为该请求执行的操作达成共识，具体的操作执行由各个确定性状态机分别执行
>  因此，Zab 更像是 primary-backup 系统，通过共识存储 master 所计算的状态，而 Raft 则服务于 replicated state machine，通过共识存储状态机所需要执行的操作

- Leader Election  
    - Raft directly uses a random timeout threshold to solve the problem of live lock  
    - Zab is more complex because it also compares the node ID  
        Thus, the newly selected leader may not contain the most recent committed data and hence a bi-direction (may read data from followers) recovery is needed 
    - Read: Zab is built-in with the read-only optimization that we have discussed in the Raft lecture and several more functionalities such as watch and sync (sync is similar to committing a no-op in Raft)  
        - How to guarantee the order in Zab? -- `zxid` 
            Client will remember the largest `zxid` it has seen and attach this `largest_zxid` in every read operation   
            The server will reject this read if its own committed `zxid` is smaller than `largest_zxid`   
            The server also needs to check and send watch notifications if there is such notifications in between `largest_zxid` and the server's current `zxid`  
- Snapshot: fuzzy snapshot in Zookeeper -- take snapshot without locking, possible because it is a state replication system and writes are idempotent​  
- Client liveness: ZooKeeper checks the liveness of client (for ephemeral znodes) by timeout  

# 7 Chain Replication
## Problem Definition
In a quorum based replication algorithm (e.g., Paxos, Raft, Zab), a replication cluster of 2N+1 nodes will stop working after losing N+1 nodes. 
- There are still N alive nodes but no progress -- a waste of resources 
- At least 3 replicas -- two is not enough

In contrast, in a primary/backup replication system
- a replication cluster of N nodes can still make progress after losing N-1 nodes  
- can start from 2 nodes​  
- the algorithm is also much simpler  
    Easier to be integrated into existing systems  
    Replication method is built-in supported by many existing systems  
    In contrast, using quorum based replication typically requires a refactoring of the original software

>  primary/backup 相较于 quorum-based 的优势在于故障容忍率更高，同时更加简单

What is the problem of primary/backup and how to mitigate it? How to choose?

## Primary/Backup
### Primary/Backup Revisit
Basic Replication Procedure
- One primary server and arbitrary number of backup server(s)  
- All write operations should be re-directed to the primary  
- After receiving a write request and the primary will​ 
    number the operation for ordering 
    remember the request ID for guaranteeing exact-once (the same algorithm we used in the duplication detecting mechanism of Raft) 
    replicating the request to all the backup servers (in parallel) 
    wait ack from all the backup servers (why not use a timeout?) 
    return to the client

![](https://cdn-mineru.openxlab.org.cn/extract/2855b925-4e88-422e-95f2-4795928a290b/5d84f5091c530ea56e5e0b3e82d9cd125fe7eb3855aec8721221ac10fb1e232d.jpg)


Read Procedure
- Primary can respond to reads without communicating with the backups (as long as it can assure that it is primary, similar to Raft's read-only optimization)
- How can we assure that the system never reveals uncommitted data
    i.e., data that might not survive a tolerated failure 
    a read is returned if **all the previous write is already committed** 
    A write is committed only after the primary receive ack from all the backup servers
- A backup can also return to read request (for better throughput) only if stale data is allowed

### Failure Recovery in Primary/Backup
Configuration service (CFG)
- All the nodes in a backup/primary cluster should not actively change their own role  
    It is **impossible** to make a consensus on who is the primary without a consensus algorithm  
    neither primary nor backup -- primary **cannot remove** a backup on its own after a timeout​ 
        Why? what if the primary crashes after it removes a backup by itself? The state of different backup will diverge and we do not want a complex roll-back based (like RAFT) mechanism to converge​
        Problem -- any single node's crash (not even the primary) will block the whole cluster
- Typically, there is a separate configuration service (CFG) manages failover 
    **configuration =¸​ identity of current primary and backups** 
    CFG is usually built on Paxos or Raft or ZooKeeper for high availability
    CFG pings all servers to detect failures and make new configuration 
    At least one replica from the previous configuration in any new configuration.
    CFG broadcasts the new configuration to all the nodes for a re-configuration

An example of the typical architecture
Patroni is a system for high-available PostgreSQL (a database) cluster

Patroni architecture:

![](https://cdn-mineru.openxlab.org.cn/extract/2855b925-4e88-422e-95f2-4795928a290b/355db03cdfe23996484edea4d4ba70f61fe049668da70935325bb2137de4a556.jpg)

- CFG of Patroni is based on etcd, which is based on Raft.
- There is also a load balancer that re-directs client requests (read backup if stale data is allowed)​ 
    - How to achieve high-availability of load balancer? 
    - Much simpler because it is a stateless service​   
    - A typical solution is sharing the same domain name (the basic method of a CDN)
- Why Raft for CFG and Primary/Backup for PostgreSQL?
    The data size and the throughput of CFG are both much **smaller** than the real database  
    3 replicas can be a large cost and hence **only one backup** is used in many real-world scenarios​  
    An efficient Raft-based replication requires modifications in the application/database services' internal logic. In contrast, a primary/backup is much more mature and widely supported 
        MySQL Group Replication is MySQL's official Paxos-based high-availability solution. Still not widely adopted by the industry

**Re-configuration**
- After a re-configuration, all replicas must **agree to achieve a true failure recovery​**  
- It is straightforward if only backup servers crash  
- In contrast, if the primary crashes, initially, some of the nodes may see the last msg from primary, but the others did not. 
    The new primary could merge all replicas' last few operations. 
    or, **elect** as primary the backup that received the highest # op.  
- New primary must start by forcing backups to agree with its decision
    May need to resend the last op
    The received backup server needs a detection mechanism to avoid duplication

**Split Brain Problem**
If a primary is **partitioned from** both some of the backup servers and the CFG and the CFG select a new primary. Is there a split brain problem?
- For write operations
    the old primary **cannot commit** a write because it needs ac**k from all the backup servers** to commit​ 
    so replicas must be careful not to reply to an old primary! -- with a sequence id
- For read operations
    It is possible because the primary do not need to communicate with backup for reads 
    Solution -- lease!

What if the primary can talk to all the backup servers but partitioned from the CFG?
- Typically, a re-configuration is executed by the CFG 
- Does not impact the correctness, but leads to performance fluctuations
- Better solution: CFG communicates with backup servers to check whether they can communicate with the primary. Postpone the re-configuration if all the backup servers say that they are ok.

### Problems of Broadcast based Primary/Backup
Impossible to fix without a consensus algorithm
- A re-configuration is needed for every node's failure
- A separate CFG is needed  

Can be mitigated by the following chain replication 
- primary has to do a lot of work​ 
- primary has to send a lot of network data for broadcasting to all the backup servers​
- re-sync after primary failure is complex, since backups may differ

## Chain Replication
### The basic Chain Replication Idea
We already saw it in Lecture 3 GFS
- GFS's chunkserver uses a non-strict primary/backup scheme and hence delivers only relaxed consistency  
- HDFS achieves strong consistency through using a chain replication (that is slightly different from the one taught today)

A chain of servers

![](https://cdn-mineru.openxlab.org.cn/extract/2855b925-4e88-422e-95f2-4795928a290b/f1d202c544487bd7704f29e448f47d6c67a95fb2b53b1d0946e0011581349b06.jpg)

- clients send updates requests to head  
- Head picks an order (assigns sequence numbers); updates local replica, sends it to the next server​  
- The next server updates its local replica and then sends to the next next server​  
- The last sever (the tail) updates local replica, sends response to client  

Updates (Write) / Queries (Read) 
- updates move along the chain in order: at each server, earlier updates delivered before later ones  
- Queries are directly sent to tail and the tail can read **local replica** and responds to client 
    Why?​ all the previous nodes are guaranteed to be newer than the tail​ Thus directly query the tail **will not unveil uncommitted data** -- but **may not the newest​**

Benefits
- head sends less network data than a primary
- **client interaction work is split** between head and tail

### Failure Recovery in Chain Replication
A separate CFG is also used in chain replication to detect failure and invoke failover

What if the head fails?
- CFG tells 2nd chain server to be new head and tells clients who the new head is
- There is no need to check the last op for selecting the newest replica -- simple!
- Will all (remaining) servers still be exact replicas?
    Each node will just miss the last few updates  
    Each server needs to compare notes with successor and just send those last few updates​  
    What if the original head restarted? Is there a split brain problem?
- Some client update requests **may be lost** if only the failed head knew about them 
    clients won't receive responses and will eventually re-send to new head

What if the tail fails?
- CFG tells next-to-last server to be new tail and tells clients, for read requests 
- for updates that new tail received but old tail didn't 
    system won't send responses to clients.
    clients will time out and re-send
    Section 2 of the paper says clients are responsible for checking whether timed-out operations have actually already been executed
        maybe with a duplication detection mechanism 
        Global unique client ID + monotonical sequence ID 
        Where to get this global unique client ID? 
        How to assure the monocity of sequence ID upon failures of client?

What if an intermediate server fails?
- CFG tells previous/next servers to talk to each other
- previous server may have to re-send some updates that it had already sent to failed server

Note that servers need to **remember updates even after forwarding**  
- in case a failure requires them to re-send  
- When to free? -- tail sends **ACKs back up** the chain as it receives updates when a server gets an ACK, it can **free all through that op**  
- Similar to HDFS

Partition situation is much as in primary/backup
- CFG makes all decisions  
- new head is old 2nd server, it should **ignore updates from the old head**, to cope with "what if old head is alive but CFG thinks it has failed"  
- CFG needs to **grant tail a lease to serve client reads**, and **not designate a new tail until lease has expired**

### Extend the chain
New server is added at the tail

The full procedure
- tell the old tail to stop processing updates  
- tell the old tail to send a complete copy of its data to the new tail  
- tell the old tail to start acting as an intermediate server, forwarding to the new tail  
- tell the new tail to start acting as the tail  
- tell clients about new tail

A long pause because of data cloning​
- snapshot can rescue -- no need to pause the system during snapshot cloning​
- Only the few last updates need to be forward during the pause

### Comparison
Primary/Backup versus Chain Replication?
- Primary/Backup may have lower latency (for small requests)  
- Chain head has less network load than primary, which **is important if data items are big** (as with GFS)  
- Chain splits work between head and tail  
- Chain has **simpler story for which server should take over if head fails**, and **how ensure servers get back in sync**

Chain (or p/b) versus Raft/Paxos/Zab (quorum)?
- p/b can tolerate N-1 of N failures, quorum only N/2  
- p/b simpler, maybe faster than quorum 
- p/b requires separate CFG, quorum self-contained 
- p/b must wait for reconfiguration after failure, quorum keeps going  
- **p/b slow if even one server slow, quorum tolerates temporary slow minority**
- p/b CFG's server failure detector hard to tune:
    any failed server stalls p/b , so want to find failed quickly! 
    but over-eager failure detector will waste time copying data to new server. 
    quorum system handles short/unclear failures more gracefully

### Sharding
What if you have **too much data to fit on a single replica group**? -- you need to "shard" across many "replica groups"

A not-so-great chain or p/b sharding arrangement -- load imbalance!

```
1 each set of three servers serves a single shard / chain  
2 shard A: S1 S2 S3  
3 shard B: S4 S5 S6
```

a better plan ("rndpar" in Section 5.4):
- split data into many **more shards than servers** (so each shard is much smaller than in previous arrangement)

```
1 each server is a replica in many shard groups  
2 shard A: S1 S2 S3  
3 shard B: S2 S3 S1  
4 shard C: S3 S1 S2
```

- for chain, a server is head in some, tail in others, middle in others 
- now request processing work is likely to be more balanced
- Repair?
    one server that **participated in M replica groups** fail 
    instead of designating a single replacement server, let's choose M replacement servers, a different one for each shard.
    now repair of the M shards can go on in parallel!
        The amount of data on the failed server is the same
        At what cost?
            M shard will pause due to the re-configuration may impact more requests  
            The re-configuration time is much smaller than the repairing time because of the data cloning in repair​  
            for the paper's setup, speed of reconstruction seems to be more important than simultaneous failures wiping out a chain. However, they assume MTBF of a single server of 24 hours, which does mean fast repair is crucial when repair can take hours but 24 hours seems unrealistically short.
- Also applicable in quorum based systems, e.g., the multi-raft in TiDB

![](https://cdn-mineru.openxlab.org.cn/extract/2855b925-4e88-422e-95f2-4795928a290b/85e63b1d38856b14f4a9be79c2e359fc1728c8d7a97c20a7c094e17682f5d353.jpg)

# 8 Distributed Transaction 101: 2PL + 2PC
## Problem Definition
We have learned how to achieve atomicity (all all nothing) and high availability in a distributed environment​ 
- Can be achieved via either **redo or undo log** in a **single-machine durable** environment​ 
- Can be achieved via **replicated state machines** in a **distributed** environment (**atomic broadcast**)  

Problem: atomic via replication naturally implies serial processing -- not able to scale-out​ 
- Solution: sharding -- split the data up over multiple replication groups 

![](https://cdn-mineru.openxlab.org.cn/extract/c11fdab6-010d-47fc-b08f-81d387adcfb5/dc0bfc10c869abf29ce6c1663ef038cd62e218fd3aee16368edc507efcf27b09.jpg)

- Further Problem: what if we want **an atomic operation that involves records in different shards?​**
    Further Solution: use **transaction**! -- **multiple operations** included in a pair of begin and end marks  
    Example Transaction

```
1  x and y are bank balances -- records in database tables  
2    that may on different servers/shards (maybe at different banks)
3    both start out as $10
4  T1 and T2 are transactions
5    T1: transfer $1 from y to x
6    T2: audit, to check that no money is lost
7  T1:        T2:
8  BEGIN-X    BEGIN-X
9    add(x, 1)    tmp1=get(x)
10   add(y, -1)   tmp2=get(y)
11 END-X          assert(tmp1+tmp2==20) -- semantic-level invariant
12            END-X
13
14 The following execution order will be excluded
15   T1:        T2:
16     add(x,1)   
17                tmp1=get(x)
18                tmp2=get(y)
19    add(y,-1)
```

- **Transaction = Concurrency Control + Atomic Commit**
- Caveats
    Transaction is needed **as long as there is concurrency**, no need to be distributed  
    In this 101 lecture, we will first focus on the single-machine transactional system (that processes concurrent transactions)

## ACID
Atomicity: guarantees that **each transaction** is treated as a single "**unit**", which either succeeds completely or fails completely  

Consistency: ensures that a transaction can only bring the database from one consistent state to another, preserving **database invariants**  

Isolation: ensures that concurrent execution of transactions leaves the database in the same state that **would have been obtained if** the transactions were executed sequentially. 
- We also use the term **serializable** to represent this kind of isolation
- There are also weaker versions of isolation. Actually, weaker versions are used more prevalently in the real-world because of their better performance. 
- Compare with linearizability later

Durability: guarantees that once a transaction has been committed, it will **remain committed** even in the case of a system failure
- DRAM is volatile and hence flush to disk is needed for durability
- In contrast, it is ambiguous to decide whether it is durable or not if there are three volatile copies in a distributed environment -- it is not durable in strict definition but it is high available​

Benefits of ACID Transaction: ACID transactions are magic! (the power of abstraction)
- programmer writes straightforward serial code 
- system automatically adds correct locking! 
- system automatically adds fault tolerance!

## Atomic/Durability in a Single Machine
Why atomic is still a problem even in a single-thread serial execution mode?

A transaction can "abort" if something goes wrong​  
- an abort un-does any record modifications -- result of abort should be as if never executed!!!  
- the transaction might voluntarily abort, e.g. if the account doesn't exist, or y's balance is <=0
- the system may force an abort, e.g. to break a locking deadlock  
- server failure can result in abort​  
- the application might (or might not) try an abort transaction again

**Rollback is needed for an aborted transaction** for atomicity
- Use write-ahead log 
- Method 1: write in-place + undo log 
- Method 2: Read redirect (redo log)

Durability can also be achieved by flushing the log to disk -- redo log 
- why not flush the data records?  
- the access pattern of log is mostly **sequential** (append), which leads to better performance than flushing the modified data records that may scattered on random locations of the disk​

## Consistency/Isolation in ACID V.S. Consistency in General
Be careful! very ambiguous discussion.
- Recap the metaphor/definition we give in Lecture 1
    The consistency of a distributed system is just like the "virtue" of a person -- the more the better, but hard to define, and frequently compromised due to the consideration of other interests (especially those interests related to money). 
    Consistency means a certain agreement on what good behaviors are at semantics-level​ 
- Both consistency in ACID and isolation in ACID are **a special form of the generalized concept of consistency​** 
- Consistency in ACID is a **higher semantics-level consistency** that relates to "preserving database invariants" 
    It means to obey application-specific invariants 
    "A Relational Model of Data for Large Shared Data Bank" by Edgar F. Codd @@ 1970 (another Turing-award paper) (pic refer to the original pdf)
- Z are user-defined constraints. For example, in MySQL, one can define the following types of "constraint"

The following constraints are commonly used in SQL:

- NOT NULL - Ensures that a column cannot have a NULL value  
- UNIQUE - Ensures that all values in a column are different  
- PRIMARY KEY - A combination of a NOT NULL and UNIQUE . Uniquely identifies each row in a table  
- FOREIGN KEY - Prevents actions that would destroy links between tables  
- CHEXK - Ensures that the values in a column satisfies a specific condition  
- DEFAULT - Sets a default value for a column if no value is specified  
- CREATE INDEX - Used to create and retrieve data from the database very quickly

```
1 CREATE TABLE Persons (

2 ID int NOT NULL,  
3 LastName varchar(255) NOT NULL,  
4 FirstName varchar(255),  
5 Age int,  
6 City varchar(255),  
7 CONSTRAINT CHK_Person CHECK (Age>=18 AND City =1=1 Sandnes')  
8 );
```

In contrast, isolation in ACID is actually more similar to the consistency in general we talked about before 
- but still many differences 
- discussed later by comparing with the consistency in CAP/Raft (Serializability vs. Linearizability)

## Serializable
Serializability is **a specific isolation level**

You execute some concurrent transactions, which yield results "results" means both output and changes in the DB. The results are serializable if:

There exists a serial execution order of the transactions that yields the same results as the actual execution  
◦ serial means one at a time -- no concurrent execution

Why serializability is popular: An easy model for programmers. They can write complex transactions while ignoring concurrency​

## Serializability v.s. Linearizability
Also Isolation in ACID (a form of consistency) V.S. Consistency in CAP/Raft

Linearizability recap

Definition: an execution history is linearizable if one can find a total order of all operations, that matches real-time (for nonoverlapping ops), and in which each read sees the value from the write preceding it in the order ◦ Example: Linearizable as Wx1 Rx1 Wx2 Rx2 (although Rx2 actually starts before Rx1 but read a newer version of value)

```
1 |-Wx1-| |-Wx2-|  
2   |- -Rx2- -|  
3     |-Rx1-|
```

Linearizability: single-operation, single-object, real-time order ◦ Real-time order: imprecisely, once a write completes, all later reads (where “later” is defined by wall-clock end time) should return the value of that write or the value of a later write.​

But what is time?

ambiguous concept in a distributed environment Linearizability is not possible without a definition of time

Example: The below example is not linearizable even if there is a possible order Wx1 Rx1, Wx2​

```
1 |-Wx1-| |-Wx2-|  
2                 |- Rx1 -|
```

To some extent, It is a property that can be viewed as a more strict version of atomicity!

Each operation, even if it accesses only a single object, is not executed "in no time", i lasts a period of time from start to end  
We use some mechanisms to make it work like each operation is executed "in no time", aka atomic, to achieve linearizability​  
Here, atomic means not only "all or nothing", it is just like an atomic instruction

Linearizability is the level of consistency in CAP/Raft

Serializability: multi-operation, multi-object, arbitrary total order  
◦ Arbitrary total order: serializability does not—by itself—impose any real-time constraints on the ordering of transactions.  
◦ For example, if T1 starts before T2 and the result is T2;T1 -- serializable but not linearizable Here we mean physical time by using the term "before"​

Strict Serializability =¸=¸​ Serializability ++ Linearizability

## Concurrency control -- The Method Used for Achieving Isolation
Concurrency control ensures that correct results for concurrent operations are generated, while getting those results as quickly as possible.

• Using only one replication group will lead to correct results, but not the fastest way​ Two transactions that do not conflict with each other can be processed concurrently ◦ but what does it mean when we use the term conflict? ◦ It depends on the level of isolation (consistency in general not in ACID) that is promised to the users​ Actually, it should be the level of isolation in ACID

▪ Actually, actually, the no different levels of isolation in ACID, it should be serializable in its original definition. But, actually, it is compromised in most of the real-world scenario

Concurrency control is mainly used for assuring isolation in ACID

Two classes of concurrency control for transactions:

Pessimistic (taught in this lecture)

An easy model: lock records before use Conflicts cause delays (waiting for locks)

◦ Optimistic (Optimistic Concurrency Control (OCC), taught in Lec 11)

use records without locking  
commit checks if reads/writes were serializable  
conflict causes abort+retry

Pessimistic is faster if conflicts are frequent; optimistic is faster if conflicts are rare

#### Strict Two-phase Locking -- A Concurrency Control that Achieves Serializability

The default pessimistic way of implementing serializable concurrency control.

Strict 2PL rules:

a transaction must acquire a record's lock before using it ◦ a transaction must hold its locks until after commit or abort [fig 3]

![](https://cdn-mineru.openxlab.org.cn/extract/c11fdab6-010d-47fc-b08f-81d387adcfb5/ccadb99413be6aefd67b1981c059911050d4570145c50a1e15e3b5c9244c3c11.jpg)

Default implementation

programmer doesn't explicitly lock, instead supplying BEGIN-X/END-X​ ◦ DB locks automatically, on first use of each record ◦ DB unlocks automatically, at transaction end -- END-X() releases all locks

DB may automatically abort to cure deadlock

Two-phase locking can produce deadlock ◦ Why deadlock? -- Tx1: Lock A, Lock B; Tx2: Lock B, Lock A

```
1 T1     T2  
2 get(x) get(y)  
3 get(y) get(x)
```

◦ The system must detect (cycles? timeout?) and abort a transaction ◦ How to avoid it? -- define an order of data records and only lock the records according to the order​

◦ Why is there still deadlock?

The above solution needs the transaction to declare its access set beforehand and locking in a specific order.  
not all the data records are available at the beginning of the transaction, e.g., interactive transactions

Why hold locks until after commit/abort?

2PL that is not S2PL

![](https://cdn-mineru.openxlab.org.cn/extract/c11fdab6-010d-47fc-b08f-81d387adcfb5/8ae3e8c69c32c4bfd85994782626e8efe917e747af62bc641caf68542b74d61f.jpg)

◦ 2PL can guarantee serializable but may lead to cascade rollback

After releasing a lock the updated value become visisble to other transactions But the current transaction is still not finished and hence may rolled back later If it is rolled back, all the transactions seeing its updates should also be rolled back

Could S2PL ever forbid a correct (serializable) execution?

```
1 T1     T2  
2 get(x)  
3         get(x)  
4         put(x,2)  
5 put(x,1)
```

◦ Serializable in the external view (still a cycle dependency but not exposed to external users)​  
◦ This is the reason why we develop other kinds of concurrency control, such as Timestamp based and OCC based.

### Weaker Level of Isolation By ANSI

Four levels of isolation from ANSI SQL-92 [fig 4] ◦ Dirty Read [fig 2]: read an uncommitted value that may be rolled back later ◦ Fuzzy Read (Non-repeatable Read) [fig 2]: read the same value but returns different values

Table 1. ANSi SQL Isolation Levels Defined in terms of the Three Original Phenomena


| Isolation Level      | P1 (or A1) Dirty Read | P2 (or A2) Fuzzy Read | P3 (or A3) Phantom |
| -------------------- | --------------------- | --------------------- | ------------------ |
| ANSIREADUNCOMMITTED  | Possible              | Possible              | Possible           |
| ANSIREAD COMMITTED   | Not Possible          | Possible              | Possible           |
| ANSIREPEATABLEREAD   | Not Possible          | Not Possible          | Possible           |
| ANOMALY SERIALIZABLE | Not Possible          | Not Possible          | Not Possible       |

![](https://cdn-mineru.openxlab.org.cn/extract/c11fdab6-010d-47fc-b08f-81d387adcfb5/ceb34dc065090de033c5d02120630c85732e673bbdb140d4c853a03f82dc4108.jpg)

![](https://cdn-mineru.openxlab.org.cn/extract/c11fdab6-010d-47fc-b08f-81d387adcfb5/cee93886fd44705d627f1ccb40477a413dbedd2c8e1fba0f9b9a350a56f009b0.jpg)

◦ Phantom Read [fig 2]: similar to fuzzy read but corresponding to a search range

![](https://cdn-mineru.openxlab.org.cn/extract/c11fdab6-010d-47fc-b08f-81d387adcfb5/0b9f54afbb1d1279001a7f1bd29de4f5dd27eaab090be06bc463873b1637ab89.jpg)

Row-level lock is enough to avoid fuzzy read.  
▪ A range-level lock (e.g., table-level lock, index lock, predict lock, prev/next-key lock) is needed to avoid phantom read

Implementation ◦ Long duration lock means release only at the end of transaction (S2PL)

Table 2. Degrees of Consistency and Locking Isolation Levels defined in terms of locks.

| Consistency Level = Locking Isolation Level | Read Locks on Data ltems and Predicates (the same unless noted)                            | Write Locks on Data Items and Predicates      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------- |
| Degree 0                                    | none required                                                                              | (always the same) Well-formed Writes          |
| Degree 1 = Locking READ UNCOMMITTED         | none required                                                                              | Well-formed Writes Long duration Write locks  |
| Degree 2 = Locking READ COMMITTED           | Well-formed Reads Short duration Read locks (both)                                         | Well-formed Writes, Long duration Write locks |
| Cursor Stability (see Section 4.1)          | Well-formed Reads Read locks held on current of cursor Short duration Read Predicate locks | Well-formed Writes, Long duration Write locks |
| Locking REPEATABLE READ                     | Well-formed Reads Long duration data-item Read locks                                       | Well-formed Writes, Long duration Write locks |
| Degree 3 = Locking SERIALIZABLE             | Short duration Read Predicate locks Well-formed Reads Long duration Read locks (both)      | Well-formed Writes, Long duration Write locks |

◦ Short duration lock means release as soon as the data is not used again (2PL)

Problems of ANSI's definition

- No Dirty Read ++ No Fuzzy Read ++ No Phantom Read is only a necessary but insufficient condition of serializable  
- There are many other kinds of anomaly phenomenal defined in [A Critique of ANSI SQL Isolation Levels](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-95-51.pdf)
    See more in [隔离级别的前世今生(三) 基于锁调度的隔离级别](https://mp.weixin.qq.com/s/0D1BKRiZAf0x0E87oPOg1Q)
    Also see [Generalized Isolation Level](https://www.pmg.csail.mit.edu/papers/icde00.pdf) Definitions 
        defined with dependency graph, not a collection of anomaly phenomenal 
        Check serializability by checking whether there is a circle in the dependency graph 
        This dependency graph is very useful. We can even learn a dependency graph in the training runs and use them to avoid concurrency bugs at deployed runs without explicitly fixing the bugs -- [my own paper](https://madsys.cs.tsinghua.edu.cn/publication/ai-a-lightweight-system-for-tolerating-concurrency-bugs/FSE2014-zhang.pdf) won a Distinguished Paper Award at FSE2014

The combination of these other kinds of anomaly phenomenal leads to other levels of isolation

a trade-off!  
The most important level of isolation not included in ANSI is snapshot isolation, which will be discussed in the next lecture.

### Distributed transaction = concurrency control + atomic commit
#### Atomic commit
An atomic commit point is a time point that: if a failure happens before this point, all the modifications are rolled back; otherwise, the modification is committed/applied once after this point.​  

It is actually a combination of atomic and durability in ACID.  

As discussed before, it can be achieved with undo/redo log in a single-machine environment

**Problem of Atomic Commit in a Distributed Environment**
Why not use replicated state machines? -- again, the accessed data may belong to different replicated groups for better performance. 

How to achieve atomic commit over multiple shards? -- it is possible that some of shards commit and the others fail, which will break the atomicity of the whole transaction

#### Two-phase commit
High-level Idea 
Use a coordinator to make sure that all the participants will all commit or all rollback.

A preparation phase that collects "votes" from participants -- all the participants should be able to commit, not a quorum

![](https://cdn-mineru.openxlab.org.cn/extract/c11fdab6-010d-47fc-b08f-81d387adcfb5/c66eaa195dd304ac1715c61756d4549c7e7c6d4e2e86d4df71398133da19f8b4.jpg)

**Serializability & 2PC**
2PC is a general approach to achieve atomic commit in a distributed environment​
- It is not bound to 2PL, it can be combined with different kinds of concurrency control to achieve different levels of isolation  

2PC can be combined with S2PL to achieve serializable distributed transaction 
- locks are required from the preparation phase
- and released only after receiving the final commit message from the coordinator

### Failure Handling in 2PC
**States** 

![](https://cdn-mineru.openxlab.org.cn/extract/c11fdab6-010d-47fc-b08f-81d387adcfb5/51cb88861b1c34fa729e20a43d0298499560c9c3a39afec51d17ee37bfd82ce0.jpg)

- A log record needs to be flushed to the persistent log for every state change

**Failure of Coordinator**  
The global commit point of 2PC is the time that the log record of "state change from prepare to commit" is flushed in the coordinator​ 
- even though some of the participants may have not received the commit message
- Even even though the coordinator fails after the flushing and before sending all the commit messages -- the restarted coordinator can use this persistent log to resend the commit messages
- When can the coordinator completely forget about a committed transaction? -- only after it sees an acknowledgement from every participant for the COMMIT.  

Participants must filter out duplicate COMMITs (using unique transaction ID).
- It can be implemented straightforwardly via remembering all the live transactions 
- A COMMIT for non-live transaction can be simply replied with ack without any other steps  

What if the coordinator has not received a prepared or aborted message from the participant
- It is possible because of the network failure
- Timeout can be used and the coordinator abort the whole transaction to release locks on other participants​
- Timeout is dangerous in a distributed environment, but in 2PC we assume only a single coordinator, thus it is fine for it to make its own decision

**Failure of Participant**
Similarly, a log record "state change from init to prepared" must be flushed to participant's persistent log before sending "can commit" vote to coordinator
- With this log, after the restarting of the participant, it can ask coordinator to commit or rollback the prepared transaction
- Meanwhile, the participant must continue to hold the prepared transaction's locks (acquire the lock during the restart procedure, before accepting any further requests). 

Otherwise, the participant can abort and release the locks if it decides to abort​ 

In contrast, the participant must block until receiving a commit/abort message from the coordinator if it has already sent prepared message to the coordinator
- Even if the participant never receives this message for a long period of time, a timeout should not be used​
- How to avoid this blocking?: implement coordinator as a highly-available service via replicated state machine

### Critique of the Original 2PC + S2PL
The original implementation of 2PC is slow for many reasons

- Two rounds of messages and lock even for read-only transactions
    locks are held during the prepare/commit exchanges; blocks other transactions
    A disk flush is needed for every state change
- Block due to coordinator failure
    2PC does not help availability since all servers must be up to get anything done
    In contrast, Raft does not ensure that all servers do something since only a majority have to be alive
    A combination of 2PC + S2PL ++ Raft can lead to highly-available distributed transaction

