# Abstract 
We offer a new metric for big data platforms, COST, or the Configuration that Outperforms a Single Thread. The COST of a given platform for a given problem is the hardware configuration required before the platform outperforms a competent single-threaded implementation. COST weighs a system’s scalability against the overheads introduced by the system, and indicates the actual performance gains of the system, without rewarding systems that bring substantial but parallelizable overheads. 
>  我们为大数据平台提出新的度量 COST (Configuration that Outperforms a Single Thread)，即优于单线程配置
>  对于给定的问题，给定平台的 COST 指在该平台上，能使得并行程序性能超过一个优秀的单线程实现所需要的硬件配置
>  COST 权衡了为了提高系统可拓展性而需要引入的开销，进而可以准确反映系统的实际性能提升，为了并行化而引入了显著开销的系统会有较高的 COST 

We survey measurements of data-parallel systems recently reported in SOSP and OSDI, and find that many systems have either a surprisingly large COST, often hundreds of cores, or simply underperform one thread for all of their reported configurations. 
>  我们调查了最近 SOSP 和 OSDI 上的数据并行系统，发现许多系统要么有非常高的 COST，通常为数百个核心，要么其性能在其所有报告的配置下实际上都低于优秀的单线程程序性能

# 1 Introduction 
“You can have a second computer once you’ve shown you know how to use the first one.”  –Paul Barham 

>  这句话的观点是：在获得更多资源和工具之前，首先考虑是否已经充分利用了现有的资源

The published work on big data systems has fetishized scalability as the most important feature of a distributed data processing platform. While nearly all such publications detail their system’s impressive scalability, few directly evaluate their absolute performance against reasonable benchmarks. To what degree are these systems truly improving performance, as opposed to parallelizing overheads that they themselves introduce? 
>  已发表的大数据系统将可拓展性视作分布式数据处理平台最重要的特性，但尽管几乎所有的工作都详细介绍了其系统优秀的可拓展性，但很少有工作将其系统的绝对性能和较为合理的基准进行比较
>  我们需要考虑：这些系统在多大程度上真正提高了其性能，而不是仅仅并行化它们自己引入的开销

![[pics/COST-Fig1.png]]

Contrary to the common wisdom that effective scaling is evidence of solid systems building, any system can scale arbitrarily well with a sufficient lack of care in its implementation. The two scaling curves in Figure 1 present the scaling of a Naiad computation before (system A) and after (system B) a performance optimization is applied. The optimization, which removes parallelizable overheads, damages the apparent scalability despite resulting in improved performance in all configurations. 
>  通常的观点是优秀的系统需要具有高效的拓展，但实际上，任意系统在不仔细考虑其实现时，都可以任意地良好拓展
>  Figure 1 展示了在应用性能优化之前，一个 Naiad 计算的拓展情况，可以看到，执行了优化后的 system B (由于移除了可以并行处理的开销)，尽管其性能提高了，但却损害了其表面上的可拓展性，而未优化的 system A 表面上具有很高的拓展性，但实际性能却不如 system B，因为其拓展只是更多地在并行处理额外开销

While this may appear to be a contrived example, we will argue that many published big data systems more closely resemble system A than they resemble system B. 
>  许多已发表的大数据系统更类似于 system A 而不是 system B

## 1.1 Methodology 
In this paper we take several recent graph processing papers from the systems literature and compare their reported performance against simple, single-threaded implementations on the same datasets using a high-end 2014 laptop. Perhaps surprisingly, many published systems have unbounded COST—i.e., no configuration outperforms the best single-threaded implementation—for all of the problems to which they have been applied. 
>  许多已发表的系统的 COST 是无上界的，即这些系统不存在可以胜过解决其针对的问题的最好的单线程实现的配置

The comparisons are neither perfect nor always fair, but the conclusions are sufficiently dramatic that some concern must be raised. In some cases the single-threaded implementations are more than an order of magnitude faster than published results for systems using hundreds of cores. We identify reasons for these gaps: some are intrinsic to the domain, some are entirely avoidable, and others are good subjects for further research. 
>  在一些情况下，单线程实现的速度比使用上百个内核的系统的速度快一个数量级以上，造成这些差距的原因中，一些是领域本身的特性，一些则完全可以避免，一些值得进一步研究

We stress that these problems lie not necessarily with the systems themselves, which may be improved with time, but rather with the measurements that the authors provide and the standard that reviewers and readers demand. Our hope is to shed light on this issue so that future research is directed toward distributed systems whose scalability comes from advances in system design rather than poor baselines and low expectations. 
>  这些问题出现在作者所提供的和审稿人所期望的度量标准上，即可拓展性
>  分布式系统的可拓展性应该来自于系统设计的进步，而不是糟糕的基准和低期望值

# 2 Basic Graph Computations 
Graph computation has featured prominently in recent SOSP and OSDI conferences, and represents one of the simplest classes of data-parallel computation that is not trivially parallelized. Conveniently, Gonzalez et al. [10] evaluated the latest versions of several graph-processing systems in 2014. We implement each of their tasks using single-threaded C# code, and evaluate the implementations on the same datasets they use (see Table 1). 
>  图计算是一类并非可以明显并行的最简单的数据并行计算模式，我们使用单线程 C# 实现多个图处理系统的任务，在和相比较系统系统使用的相同的数据集上评估

![[pics/COST-Table1.png]]

Our single-threaded implementations use a simple Boost-like graph traversal pattern. A `GraphIterator` type accepts actions on edges, and maps the action across all graph edges. The implementation uses unbuffered IO to read binary edge data from SSD and maintains per-node state in memory backed by large pages (2MB). 
>  我们的单线程实现使用了简单的类 Boost 的图遍历模式，`GraphIterator` 类型接受对边的操作，将该操作映射到图的所有边
>  该实现使用未缓冲的 IO 从 SSD 读取二进制边数据，通过大页 (2MB) 在内存中维护每个节点的状态

## 2.1 PageRank 
PageRank is an computation on directed graphs which iteratively updates a rank maintained for each vertex [19]. In each iteration a vertex’s rank is uniformly divided among its outgoing neighbors, and then set to be the accumulation of scaled rank from incoming neighbors. A dampening factor `alpha` is applied to the ranks, the lost rank distributed uniformly among all nodes. Figure 2 presents code for twenty PageRank iterations. 
>  PageRank 是一个在有向图上的计算，它迭代式更新每个节点上维护的秩
>  每次迭代中，每个节点上的秩会被均匀地分配给它的所有出边邻居节点 (乘上抑制因子 `alpha`)，然后再被设定为它的所有入边节点的更新后的秩的和

```rust
fn PageRank20(graph: GraphIterator, alpha: f32) {
    let mut a = vec![0f32; graph.nodes()]:
    let mut b = vec![0f32; graph.nodes()]:
    let mut d = vec![0f32; graph.nodes()]:

    graph.map_edges(|x, y| { d[x] += 1;});
    
    for iter in 0..20 {
        for i in 0..graph.nodes() {
            b[i] = alpha * a[i]/d[i]
            a[i] = lf32 - alpha;
        }
    }
    
    graph.map_edges(|x, y| {a[y] += b[x];});
}
```

Table 2 compares the reported times from several systems against a single-threaded implementations of PageRank, reading the data either from SSD or from RAM. Other than GraphChi and X-Stream, which reread edge data from disk, all systems partition the graph data among machines and load it in to memory. Other than GraphLab and GraphX, systems partition edges by source vertex; GraphLab and GraphX use more sophisticated partitioning schemes to reduce communication. 
>  Table 2 比较了 PageRank 的单线程实现和多个系统的实现，大多数系统将图数据划分给多个机器，然后将其载入内存

![[pics/COST-Table2.png]]

No scalable system in Table 2 consistently outperforms a single thread, even when the single thread repeatedly re-reads the data from external storage. Only GraphLab and GraphX outperform any single-threaded executions, although we will see in Section 3.1 that the single-threaded implementation outperforms these systems once it re-orders edges in a manner akin to the partitioning schemes these systems use. 
>  根据 Table 2，没有由于单线程性能的可拓展系统，即便单线程程序会重复地从外部存储中读取数据

## 2.2 Connected Components 
The connected components of an undirected graph are disjoint sets of vertices such that all vertices within a set are mutually reachable from each other. 
>  无向图的连通成分是互相不相交的节点集合，同一个连通成分中的节点互相两两可达

In the distributed setting, the most common algorithm for computing connectivity is label propagation [11] (Figure 3). In label propagation, each vertex maintains a label (initially its own ID), and iteratively updates its label to be the minimum of all its neighbors’ labels and its current label. The process propagates the smallest label in each component to all vertices in the component, and the iteration converges once this happens in every component. The updates are commutative and associative, and consequently admit a scalable implementation [7]. 
>  分布式设定下，计算连通性最常用的算法是标签传播，该算法中，每个节点维护一个标签 (初始值为其 ID)，然后迭代式地将其标签更新为其所有邻居的标签和其自己的标签中的最小值
>  该过程会将每个连通成分中的最小标签传播到连通成分中的所有节点上，当所有连通分量中都出现这种情况，算法收敛
>  该算法执行的更新是可交换和可结合的，因此可以进行可拓展的实现

```rust
fn LabelPropagation(graph: GraphIterator) {
    let mut label = (0..graph.nodes()).to_vec();
    let mut done = fales;
    
    while !done {
        done = true;
        graph.map_edges(|x, y| {
            if label[x] != label[y] {
                done = false;
                label[x] = min(label[x], label[y]);
                label[y] = min(label[x], label[y]);
            }
        });
    }
}
```

Table 3 compares the reported running times of label propagation on several data-parallel systems with a single-threaded implementation reading from SSD. Despite using orders of magnitude less hardware, single-threaded label propagation is significantly faster than any system above. 
>  如 Table 3 所示，单线程实现使用的硬件数量少了一个量级，但仍然显著快于其他系统

![[pics/COST-Table3.png]]

# 3 Better Baselines 
The single-threaded implementations we have presented were chosen to be the simplest, most direct implementations we could think of. There are several standard ways to improve them, yielding single-threaded implementations which strictly dominate the reported performance of the systems we have considered, in some cases by an additional order of magnitude. 
>  之前的单线程实现只是最简单的实现，通过进一步优化它们，其性能将由于所有其他系统，一些情况下会超出一个数量级

## 3.1 Improving graph layout 
Our single-threaded algorithms take as inputs edge iterators, and while they have no requirements on the order in which edges are presented, the order does affect performance. Up to this point, our single-threaded implementations have enumerated edges in vertex order, whereby all edges for one vertex are presented before moving on to the next vertex. Both GraphLab and GraphX instead partition the edges among workers, without requiring that all edges from a single vertex belong to the same worker, which enables those systems to exchange less data [9, 10]. 
>  我们的单线程算法以边迭代器作为输入，并没有考虑边的顺序，而顺序会影响性能
>  我们的单线程实现按照节点顺序枚举边，即处理下一个顶点之前，会枚举出当前顶点相关的所有边
>  GraphLab 和 GraphX 在 workers 中划分边，不要求来自于单个节点的所有边属于同一个 worker，这使得系统可以交换更少的数据

A single-threaded graph algorithm does not perform explicit communication, but edge ordering can have a pronounced effect on the cache behavior. For example, the edge ordering described by a Hilbert curve [2], akin to ordering edges $(a, b)$ by the interleaving of the bits of $a$ and $b$ , exhibits locality in both $a$ and $b$ rather than just $a$ as in the vertex ordering. Table 4 compares the running times of single-threaded PageRank with edges presented in Hilbert curve order against other implementations, where we see that it improves over all of them. 
>  单线程图算法不会执行显式的通信，但边顺序会对缓存行为有明显的影响，例如，由 Hilbert 曲线描述的边排序[2]，类似于通过交错 $a$ 和 $b$ 的位来排序边 $(a, b)$，在 $a$ 和 $b$ 中都表现出局部性，而不仅仅是像顶点排序中的 $a$。表 4 比较了以 Hilbert 曲线顺序呈现边的单线程 PageRank 的运行时间与其他实现的运行时间，在这里我们可以看到它比所有其他实现都要好。

![[pics/COST-Table4.png]]

Converting the graph data to a Hilbert curve order is an additional cost in pre-processing the graph. The process amounts to transforming pairs of node identifiers (edges) into an integer of twice as many bits, sorting these values, and then transforming back to pairs of node identifiers. Our implementation transforms the twitter rv graph in 179 seconds using one thread, which can be a performance win even if pre-processing is counted against the running time. 
>  将图数据转换为希尔伯特曲线顺序是图预处理所需要的额外成本。这个过程包括将节点标识符对（边）转换为位数翻倍的整数，对这些值进行排序，然后再转换回节点标识符对。我们的实现使用一个线程在 179 秒内转换 twitter rv 图，即使预处理时间计入运行时间，这也可能是一个性能上的优势。

## 3.2 Improving algorithms 
The problem of properly choosing a good algorithm lies at the heart of computer science. The label propagation algorithm is used for graph connectivity not because it is a good algorithm, but because it fits within the “think like a vertex” computational model [15], whose implementations scale well. Unfortunately, in this case (and many others) the appealing scaling properties are largely due to the algorithm’s sub-optimality; label propagation simply does more work than better algorithms. 
>  标签传播算法可以用于图连通性检测，不是因为它是一个好的算法，而是因为它符合 “像一个顶点一样思考” 的计算模型，这样的模型实现起来有良好的可拓展性
>  但这样优秀的可拓展性实际上来自于算法本身的次优性，标签传播算法所作的工作本质上比其他更好的算法要多

Consider the algorithmic alternative of Union-Find with weighted union [3], a simple $O (m\log n)$ algorithm which scans the graph edges once and maintains two integers for each graph vertex, as presented in Figure 4. Table 5 reports its performance compared with implementations of label propagation, faster than the fastest of them (the single-threaded implementation) by over an order of magnitude. 
>  考虑带权重合并的 Union-Find 算法变体，这是一个 $O(m\log n)$ 的算法，它扫描图的边一次，为每个图顶点维护两个整数
>  Union-Find 的单线程实现比标签传播算法的单线程实现快出一个数量级

```rust
fn UnionFind(graph: GraphIterator) {
    let mut root = (0..graph.nodes()).to_vec();
    let mut rank = [0u8; graph.nodes()];
    
    graph.map_edges(|mut x, mut y| {
        while(x != root[x]) {x = root[x];};
        while(y != root[y]) {y = root[y];};
        if x != y {
            match rank[x].cmp(&rank[y]) {
                Less => {root[x] = y;};
                Greater => {root[y] = x;};
                Equal => {root[y] = x; rank[x] += 1;};
            }
        }
    })
}
```

![[pics/COST-Table5.png]]

There are many other efficient algorithms for computing graph connectivity, several of which are parallelizable despite not fitting in the “think like a vertex” model. While some of these algorithms may not be the best fit for a given distributed system, they are still legitimate alternatives that must be considered. 

# 4 Applying COST to prior work 
Having developed single-threaded implementations, we now have a basis for evaluating the COST of systems. As an exercise, we retrospectively apply these baselines to the published numbers for existing scalable systems. 

## 4.1 PageRank 
Figure 5 presents published scaling information from PowerGraph [9], GraphX [10], and Naiad [16], as well as two single-threaded measurements as horizontal lines. The intersection with the upper line indicates the point at which the system out-performs a simple resource-constrained implementation, and is a suitable baseline for systems with similar limitations (e.g., GraphChi and X-Stream). The intersection with the lower line indicates the point at which the system out-performs a feature-rich implementation, including pre-processing and sufficient memory, and is a suitable baseline for systems with similar resources (e.g., GraphLab, Naiad, and GraphX). 
>  Fig 5 中，单线程度量是两条水平线，其他曲线和上部线条的交点表示了系统开始由于简单的资源受限的单线程实现的点，这个点是比较具有类似限制条件的系统的一个合适的点；其他曲线和下部线条的焦点表示了系统开始由于包括了预处理和充足内存的特性丰富的单线程实现的点，同样可以作为一个合适的比较基准

![[pics/COST-Fig5.png]]

From these curves we would say that Naiad has a COST of 16 cores for PageRanking the twitter rv graph. Although not presented as part of their scaling data, GraphLab reports a $3.6s$ measurement on 512 cores, and achieves a COST of 512 cores. GraphX does not intersect the corresponding single-threaded measurement, and we would say it has unbounded COST. 
>  根据该图，我们可以称 Naiad 的 COST 为 16 核，GraphLab 的 COST 为 512 核，GraphX 没有再和单线程度量相交，故其 COST 无界

## 4.2 Graph connectivity 
The published works do not have scaling information for graph connectivity, but given the absolute performance of label propagation on the scalable systems relative to single-threaded union-find we are not optimistic that such scaling data would have lead to a bounded COST. 

Instead, Figure 6 presents the scaling of two Naiad implementations of parallel union-find [14], the same examples from Figure 1. The two implementations differ in their storage of per-vertex state: the slower one uses hash tables where the faster one uses arrays. The faster implementation has a COST of 10 cores, while the slower implementation has a COST of roughly 100 cores. 

![[pics/COST-Fig6.png]]

The use of hash tables is the root cause of the factor of ten increase in COST, but it does provide some value: node identifiers need not lie in a compact set of integers. This evaluation makes the trade-off clearer to both system implementors and potential users. 
>  哈希表的使用是增大了上例中 Naiad 的 COST 的主要原因，但哈希表的使用也提供了价值：不需要在紧凑的整数数组中存储节点 ID

# 5 Lessons learned 
Several aspects of scalable systems design and implementation contribute to overheads and increased COST. The computational model presented by the system restricts the programs one may express. The target hardware may reflect different trade-offs, perhaps favoring capacity and throughput over high clock frequency. Finally, the implementation of the system may add overheads a single thread doesn’t require. Understanding each of these overheads is an important part of assessing the capabilities and contributions of a scalable system. 
>  可拓展系统在设计和实现的几个方面会导致更多的开销和 COST，系统所展现的计算模型限制了可以在系统上编写的程序
>  目标硬件也会反映不同的权衡，也许更侧重于容量和吞吐量而不是高的时钟频率
>  分布式系统的实现可能会增加单线程所不需要的开销，评估一个可拓展系统的贡献和能力是，需要理解这些开销的来源

To achieve scalable parallelism, big data systems restrict programs to models in which the parallelism is evident. These models may not align with the intent of the programmer, or the most efficient parallel implementations for the problem at hand. Map-Reduce intentionally precludes memory-resident state in the interest of scalability, leading to substantial overhead for algorithms that would benefit from it. Pregel’s “think like a vertex” model requires a graph computation to be cast as an iterated local computation at each graph vertex as a function of the state of its neighbors, which captures only a limited subset of efficient graph algorithms. Neither of these designs are the “wrong choice”, but it is important to distinguish “scalability” from “efficient use of resources”. 
>  为了达成可拓展的并行性，大数据系统将程序限制在了具有明显并行性的模型，这些模型并不一定符合程序员的预期，或者不是解决当前问题的最高效并行实现办法
>  MapReduce 为了实现可拓展性，故意排除了对内存驻留状态的考虑，这导致许多本应从内存驻留状态受益的算法产生了相当大的开销
>  Pregel 的 “像一个顶点一样思考” 模型要求图计算表示为在每个图顶点上的迭代局部计算，局部计算的形式即关于其邻居状态的函数，这一模型实际上仅能容纳高效图算法的一个有限子集
>  这两个设计都是错误的选择，但需要区分“可拓展性”和“资源的有效利用”

The cluster computing environment is different from the environment of a laptop. The former often values high capacity and throughput over latency, with slower cores, storage, and memory. The laptop now embodies the personal computer, with lower capacity but faster cores, storage, and memory. While scalable systems are often a good match to cluster resources, it is important to consider alternative hardware for peak performance. 
>  集群计算环境更注重高吞吐量和高容量，而不是低延迟，使用的是更慢的处理器、存储、内存，单机计算环境的容量低，但延迟低，使用更快的处理器、存储、内存
>  可拓展系统可以匹配集群资源，但要达到峰值性能，也需要考虑可以替代的更好的硬件

Finally, the implementation of the system may introduce overheads that conceal the performance benefits of a scalable system. High-level languages may facilitate development, but they can introduce performance issues (garbage collection, bounds checks, memory copies). It is especially common in a research setting to evaluate a new idea with partial or primitive implementations of other parts of the system (serialization, memory management, networking), asserting that existing techniques will improve the performance. While many of these issues might be improved with engineering effort that does not otherwise advance research, nonetheless it can be very difficult to assess whether the benefits the system claims will still manifest once the fat is removed. 
>  系统的实现会引入额外开销，掩盖了分布式系统的额外优势
>  高级语言促进开发，但存在性能问题 (垃圾收集、边界检查、内存拷贝)
>  在当前的研究环境中，研究员通常会使用部分或者简陋的其他系统组件 (序列化、内存管理、网络) 来评估比较其新思想，并声称其技术会提高性能

There are many good reasons why a system might have a high COST when compared with the fastest purpose-built single-threaded implementation. The system may target a different set of problems, be suited for a different deployment, or be a prototype designed to assess components of a full system. The system may also provide other qualitative advantages, including integration with an existing ecosystem, high availability, or security, that a simpler solution cannot provide. As Section 4 demonstrates, it is nonetheless important to evaluate the COST, both to explain whether a high COST is intrinsic to the proposed system, and because it can highlight avoidable inefficiencies and thereby lead to performance improvements for the system. 
>  和最快的专用单线程实现相比，分布式系统也存在具有高 COST 的合适理由，例如系统针对的是另一个问题集，适合不同的部署环境，或者是一个旨在评估完整系统组件的原型。该系统还可能提供其他质量优势，包括与现有生态系统集成、高可用性或安全性，而这些是简单解决方案无法提供的。
>  正如第 4 节所展示的，评估 COST 仍然是重要的，这不仅是为了说明高 COST 是否是系统固有的特性，并且还因为这可以突出避免的低效性，并因此推动系统的性能改进。 

# 6 Future directions (for the area) 
While this note may appear critical of research in distributed systems, we believe there is still good work to do, and our goal is to provide a framework for measuring and making the best forward progress. 

There are numerous examples of scalable algorithms and computational models; one only needs to look back to the parallel computing research of decades past. Borivka's algorithm [1] is nearly ninety years old, parallelizes cleanly, and solves a more general problem than label propagation. The Bulk Synchronous Parallel model [24] is surprisingly more general than most related work sections would have you believe. These algorithms and models are richly detailed, analyzed, and in many cases already implemented. 

Many examples of performant scalable systems exist. Both Galois [17] and Ligra [23] are shared-memory systems that significantly out-perform their distributed peers when run on single machines. Naiad [16] introduces a new general purpose dataflow model, and out-performs even specialized systems. Understanding what these systems did right and how to improve them is more important than re-hashing existing ideas in new domains compared against only the poorest of prior work. 

We are now starting to see performance studies of the current crop of scalable systems [18], challenging some conventional wisdom underlying their design principles. Similar such studies have come from previous generations of systems [22], including work explicitly critical of the absolute performance of scalable systems as compared with simpler solutions [20, 4, 25]. While it is surely valuable to understand and learn from the performance of popular scalable systems, we might also learn that we keep making, and publishing, the same mistakes. 

Fundamentally, a part of good research is making sure we are asking the right questions. “Can systems be made to scale well?” is trivially answered (in the introduction) and is not itself the right question. There is a substantial amount of good research to do, but identifying progress requires being more upfront about existing alternatives. The COST of a scalable system uses the simplest of alternatives, but is an important part of understanding and articulating progress made by research on these systems. 


