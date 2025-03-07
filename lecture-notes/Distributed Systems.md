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
