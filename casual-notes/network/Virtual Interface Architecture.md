>  https://en.wikipedia.org/wiki/Virtual_Interface_Architecture

The **Virtual Interface Architecture** (**VIA**) is an abstract model of a user-level [zero-copy](https://en.wikipedia.org/wiki/Zero-copy "Zero-copy") [network](https://en.wikipedia.org/wiki/Computer_network "Computer network"), and is the basis for [InfiniBand](https://en.wikipedia.org/wiki/InfiniBand "InfiniBand"), [iWARP](https://en.wikipedia.org/wiki/IWARP "IWARP") and [RoCE](https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet "RDMA over Converged Ethernet"). Created by [Microsoft](https://en.wikipedia.org/wiki/Microsoft "Microsoft"), [Intel](https://en.wikipedia.org/wiki/Intel "Intel"), and [Compaq](https://en.wikipedia.org/wiki/Compaq "Compaq"), the original VIA sought to standardize the interface for high-performance network technologies known as [System Area Networks](https://en.wikipedia.org/wiki/System_Area_Network "System Area Network") (SANs; not to be confused with [Storage Area Networks](https://en.wikipedia.org/wiki/Storage_Area_Network "Storage Area Network")).
>  虚拟接口架构是用户级零拷贝网络的抽象模型，是 InfiniBand, iWARP, RoCE 的基础，由微软，英特尔和康铂开发
>  最初的 VIA 致力于标准化高性能网络技术 (称为系统区域网络)

>  用户级指进程不需要切换到内核态
>  零拷贝指数据直接在进程与网络接口卡之间传输，减少内存拷贝开销

Networks are a shared resource. With traditional network APIs such as the [Berkeley socket API](https://en.wikipedia.org/wiki/Berkeley_sockets "Berkeley sockets"), the [kernel](https://en.wikipedia.org/wiki/Kernel_\(operating_system\) "Kernel (operating system)") is involved in every network communication. This presents a tremendous performance bottleneck when [latency](https://en.wikipedia.org/wiki/Latency_\(engineering\) "Latency (engineering)") is an issue.
>  网络是一种共享资源，在传统的网络 API 例如 Berkeley socket API 中，每次网络通信都需要内核的参与，会带来极大的性能瓶颈

>  内核的参与包括
>  - 数据包的发送和接收需要通过系统调用 (如 `send()` 和 `recv()`) 在用户态和内核态之间切换。
>  - 内核负责管理网络缓冲区、协议栈 (如 TCP/IP) 以及与硬件的交互

One of the classic developments in computing systems is [virtual memory](https://en.wikipedia.org/wiki/Virtual_memory "Virtual memory"), a combination of hardware and software that creates the illusion of private memory for each process. In the same school of thought, a virtual network interface protected across process boundaries could be accessed at the user level. With this technology, the "consumer" manages its own buffers and communication schedule while the "provider" handles the protection.
>  计算机系统中的一项经典技术是虚拟内存，它结合了软件和硬件，为每个进程创建了虚拟的私有内存
>  在同样的思路下，受进程边界保护的虚拟网络接口可以在用户态被访问 (每个进程在用户态直接管理自己的网络通信，不需要切换到内核态)，在该技术下，"消费者” (应用程序) 管理自己的缓冲区和通信调度，“提供者” (内核和网络接口卡) 负责资源保护 (如不同进程的虚拟接口隔离)

>  虚拟网络接口的优势在于
> - 用户态管理：每个进程或应用程序可以在用户态直接管理自己的网络通信，而不需要频繁地切换到内核态
> - 隔离性：虚拟网络接口为每个进程提供一个独立的网络资源，避免了进程之间的资源争用
>  虚拟网络接口的实现需要硬件支持，如网络接口卡的虚拟化功能

Thus, the [network interface card](https://en.wikipedia.org/wiki/Network_interface_controller "Network interface controller") (NIC) provides a "private network" for a process, and a process is usually allowed to have multiple such networks. The virtual interface (VI) of VIA refers to this network and is merely the destination of the user's communication requests. Communication takes place over a pair of VIs, one on each of the processing nodes involved in the transmission. In "kernel-bypass" communication, the user manages its own buffers.
>  通过虚拟网络接口，网络接口卡为进程提供了私有网络，且一个进程允许拥有多个这样的私有网络
>  虚拟接口架构的 “虚拟接口” 指的就是这样的私有网络，私有网络是用户进程通信的请求目标，是用户层发送和接收数据的 "目的地"
>  两个点之间的通信需要一对虚拟接口，在发送端 VI，用户进程将数据写入本地 VI 的缓冲区，接收端 VI 将数据传递给对应进程，VI 之间的通信可以通过物理网络或共享内存等实现
>  虚拟网络接口技术实现了 “内核旁路” 通信，内核仅负责管理资源分配，用户进程直接管理通信细节 (数据包收发、缓冲区管理)

Another facet of traditional networks is that arriving data is placed in a pre-allocated buffer and then copied to the user-specified final destination. Copying large messages can take a long time, and so eliminating this step is beneficial. Another classic development in computing systems is [direct memory access](https://en.wikipedia.org/wiki/Direct_memory_access "Direct memory access") (DMA), in which a device can access main memory directly while the CPU is free to perform other tasks.
>  传统网络的另一个缺点是到达数据会放在预分配好的缓冲区中，然后被拷贝到用户指定的最终区域，如果拷贝的数据量大，会花费大量时间，故最好消除这一步
>  为此，可以对照另一个经典的计算机技术: 直接内存访问，其中设备对内存的访问不需要通过 CPU

In a network with "remote direct memory access" ([RDMA](https://en.wikipedia.org/wiki/Remote_direct_memory_access "Remote direct memory access")), the sending NIC uses DMA to read data in the user-specified buffer and transmit it as a self-contained message across the network. The receiving NIC then uses DMA to place the data into the user-specified buffer. There is no intermediary copying and all of these actions occur without the involvement of the CPUs, which has an added benefit of lower CPU utilization.
>  在网络中，类似的技术称为远程直接内存访问，其中:
>  发送端 NIC 使用 DMA 从用户指定的缓冲区读取数据，并将其作为独立的消息传输给网络
>  接收端 NIC 使用 DMA 将数据放到用户指定的缓冲区
>  过程中没有中间的数据拷贝，这些操作都不需要 CPU 的参与

For the NIC to actually access the data through DMA, the user's page must be in memory. In VIA, the user must "pin-down" its buffers before transmission, so as to prevent the OS from swapping the page out to the disk. This action—one of the few that involve the kernel—ties the page to physical memory. To ensure that only the process that owns the registered memory may access it, the VIA NICs require permission keys known as "protection tags" during communication.
>  NIC 需要通过 DMA 访问数据时，用户进程的页必须驻留在内存中
>  VIA 中，用户需要在传输之前 “固定” 其缓冲区，避免操作系统将缓冲区的页换出到磁盘
>  为了确保注册内存的只能被其所有者进程访问，VIA NIC 在通信期间需要权限密钥，称为 "保护标签"

So essentially VIA is a standard that defines kernel bypassing and RDMA in a network. It also defines a programming library called "VIPL". It has been implemented, most notably in cLAN from Giganet (now [Emulex](http://www.emulex.com/)). Mostly though, VIA's major contribution has been in providing a basis for the [InfiniBand](https://en.wikipedia.org/wiki/InfiniBand "InfiniBand"), [iWARP](https://en.wikipedia.org/wiki/IWARP "IWARP") and RoCE standards.
>  因此，本质上虚拟接口架构是一个定义了网络中 kernel bypassing 和 RDMA 的标准
>  VIA 还定义了一个称为 VIPL 的编程库，已经被广泛实现，最著名的是 cLAN (Emulex)
>  VIA 的主要贡献在于为 InfiniBand, iWARP, RoCE 标准提供了基础

External links:

- [Usenix Notes On VIA](http://www.usenix.org/publications/library/proceedings/als00/2000papers/papers/full_papers/rangarajan/rangarajan_html/node3.html)
- [Distributed Enterprise Networks](https://makonetworks.com/markets/distributed-enterprises/)
- [Virtual Interface Architecture](https://noggin.intel.com/intelpress/categories/books/virtual-interface-architecture), a book from Intel

>  This page was last edited on 19 November 2024, at 20:52 (UTC).