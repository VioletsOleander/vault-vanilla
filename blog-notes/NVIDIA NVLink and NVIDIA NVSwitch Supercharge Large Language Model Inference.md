---
completed: true
---
Site: https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/
Date: 12 Aug 2024

[Large language models](https://www.nvidia.com/en-us/glossary/large-language-models/) (LLM) are getting larger, increasing the amount of compute required to process inference requests. To meet real-time latency requirements for serving today’s LLMs and do so for as many users as possible, multi-GPU compute is a must. Low latency improves the user experience.  High throughput reduces the cost of service.  Both are simultaneously important.
>  低延迟提高用户体验，高吞吐降低服务成本，二者都很重要

Even if a large model can fit in the memory of a single state-of-the-art GPU, the rate at which that GPU can generate tokens depends on the total compute available to process requests. By combining the compute capabilities of multiple cutting-edge GPUs, real-time user experiences on the latest models are possible. 

To understand the need for high tokens per second, the following GIFs show two scenarios: 

- **5 tokens/second:** Below typical human reading speed and not real-time.
- **50 tokens/second:** An excellent user experience. 

[![GIF displays three lines of a Shakespeare quote from Macbeth with words appearing one at a time.](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/5-tokens-second-macbeth.gif)](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/5-tokens-second-macbeth.gif)

_Figure 1. 5 tokens/second output example_

![GIF displays 20 lines of a Shakespeare quote from Macbeth with entire lines appearing quickly.](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/50-tokens-second-macbeth.gif)

_Figure 2. 50 tokens/second output example_

By using the combined compute performance of multiple GPUs with techniques such as tensor parallelism (TP) to run large models, inference requests can be processed quickly enough to enable real-time responses. By carefully selecting the number of GPUs used to run a model, cloud inference services can also simultaneously optimize both user experience and cost. 

For more information about parallelism techniques to balance user experience, see [Demystifying AI Inference Deployments for Trillion Parameter Large Language Models](https://developer.nvidia.com/blog/demystifying-ai-inference-deployments-for-trillion-parameter-large-language-models/). 

## Multi-GPU inference is communication-intensive
Multi-GPU TP inference works by splitting the calculation of each model layer across two, four, or even eight GPUs in a server. In theory, two GPUs could run a model 2x faster, four GPUs 4x faster, and eight GPUs 8x faster. 
>  多 GPU Tensor-Parallelism 推理将每个模型层的计算划分到 server 上的多个 GPU，理论上，有几个 GPU，就有几倍快

However, each GPU cannot complete their work independently. After each GPU completes the execution of its portion of the model layer, every GPU must send the results of the calculations to every other GPU, performing an all-to-all reduction. Only then can inference execution proceed to the next model layer.
>  但 GPU 无法独立完成任务，GPU 执行完自己的任务之后，需要进行 all-to-all reduce 通信，只有完成通信之后，才能进行下一层的推理执行

Minimizing the time spent communicating results between GPUs is critical, as during this communication, Tensor Cores often remain idle, waiting for data to continue processing. 
>  因此最小化 GPU 之间的通信开销非常重要，Tensor Cores 的处理速度很快，通常会等待数据传输

During this communication step, a large amount of data must be transferred. A single query to Llama 3.1 70B (8K input tokens and 256 output tokens) requires that up to 20 GB of TP synchronization data be transferred from each GPU. As multiple queries are processed in parallel through batching to improve inference throughput, the amount of data transferred increases by multiples. 
>  在这个通信步中，需要传输非常多的数据，对 Llama 3.1 70B 的单个 query (8K input tokens, 256 output tokens) 需要每个 GPU 传输高达 20GB 的 TB 同步数据
>  通过并行批处理多个 query 可以提高推理吞吐，但需要传输的数据量和成倍提高

This is why a high-bandwidth GPU-to-GPU interconnect is essential for multi-GPU inference. 
>  因此高带宽 GPU-GPU 互联对于多 GPU 推理是必要的

## NVSwitch is critical for fast multi-GPU LLM inference
For good multi-GPU scaling, an AI server first requires GPUs with excellent per-GPU interconnect bandwidth. It must also provide fast connectivity to enable all GPUs to exchange data with all other GPUs as quickly as possible.

The [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) GPU can communicate at 900 GB/s with fourth-generation NVLink. With the NVSwitch, every NVIDIA Hopper GPU in a server can communicate at 900 GB/s with any other NVIDIA Hopper GPU simultaneously.
>  NVIDIA Hopper 架构的 GPU 可以以 900GB/s 的速度互联，使用第四代 NVLink 和 NVSwitch

The peak rate does not depend on the number of GPUs that are communicating. That is, the NVSwitch is non-blocking. Every NVIDIA HGX H100 and NVIDIA HGX H200 system with eight GPUs features four third-generation NVSwitch chips. The total bidirectional bandwidth of each NVSwitch chip is a staggering 25.6 terabits per second.
>  峰值传输率不依赖于互联的 GPU 数量，即 NVSwitch 不是阻塞式的


![Picture of the NVIDIA Hopper Architecture GPU with a callout showing the four NVSwitch chips.](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/NVSwitch-image-fixed-1-625x281.png)

_Figure 3. HGX H200 8-GPU with four NVIDIA NVSwitch devices_

For comparison, consider a hypothetical server with eight H200 GPUs without NVSwitch that instead uses point-to-point connections on the server motherboard (Figure 4).

![Diagram shows 8 GPUs on the top, each with links going to every other GPU. On the bottom, 8 GPUs are connected to each other with a centralized NVSwitch.](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/gpu-to-gpu-bandwidth-nvswitch-comparison-b.png)

_Figure 4. G_PU-to-GPU bandwidth with and without NVSwitch all-to-all switch topology__

In the point-to-point design, though it is a lower system cost without four high-speed switches, each GPU must split the same 900 GB/s connectivity into seven dedicated 128 GB/s point-to-point connections, each connecting to one of the other GPUs in the system. This means that the speed at which GPUs can communicate depends on the number of GPUs that are communicating. 
>  如果没有 NVSwitch，每个 GPU 需要将连接带宽分割成 7 条专用的 128GB/s 点对点连接，这意味着 GPU 之间的通信速度取决于参与通信的 GPU 数量

| **GPU Count** | **Point-to-Point Bandwidth** | **NVSwitch Bandwidth** |
| ------------- | ---------------------------- | ---------------------- |
| 2             | 128 GB/s                     | 900 GB/s               |
| 4             | 3 x 128 GB/s                 | 900 GB/s               |
| 8             | 7 x 128 GB/s                 | 900 GB/s               |

_Table 1. GPU-to-GPU bandwidth comparison_

Table 1 shows a GPU-to-GPU bandwidth comparison between GPUs connected through a point-to-point interconnect and GPUs connected with NVSwitch.

For models that only require two GPUs for the best balance of user experience and cost, such as Llama 3.1 70B, a point-to-point architecture only provides 128 GB/s of bandwidth. 20 GB of data would consume 150 ms to perform just one of the many all-to-all reductions. With high communication overhead, Amdahl’s Law limits the speed-up possible with each additional GPU.

Meanwhile, the system using NVSwitch would provide the full 900 GB/s of bandwidth, taking only 22 ms to transfer 20 GB, dramatically reducing the time spent during GPU-to-GPU communication. This has a significant impact on overall inference throughput and user experience.

[![On the top of the diagram are two GPUs connected with a small green line, with an indicator that communication makes up a large portion of the execution time. On the bottom, two GPUs are connected via NVSwitch, with communication making up a small portion of the execution time.](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/multi-gpu-communication-nvswitch-comparison-625x611.png)](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/multi-gpu-communication-nvswitch-comparison.png)

_Figure 5. Multi-GPU communication with and without NVSwitch_

Cloud services often set fixed response time budgets for model serving, to provide good end-user experiences. This typically means being able to generate tokens faster than human reading speed. To maximize throughput and decrease serving costs, requests are batched as high as possible while maintaining the response time.

Table 2 shows the measured Llama 3.1 70B throughput at various real-time response time budgets from 30-50 tokens/s/user.

| **Real-time Response Budget** **tok/s/user** | **Throughput** **tok/s/GPU (batch size)** |                       |         | **NVSwitch**  **Benefit** |
| -------------------------------------------- | ----------------------------------------- | --------------------- | ------- | ------------------------- |
| **Single GPU** **TP=1**                      | **Point-to-Point** **TP=2**               | **NVSwitch** **TP=2** |         |                           |
| 30                                           | 67 (2)                                    | 80 (6)                | 115 (9) | **1.4x**                  |
| 35                                           | Does Not Meet                             | 74 (5)                | 104 (7) | **1.4x**                  |
| 40                                           | Does Not Meet                             | 67 (4)                | 87 (5)  | **1.3x**                  |
| 45                                           | Does Not Meet                             | 56 (3)                | 76 (4)  | **1.4x**                  |
| 50                                           | Does Not Meet                             | 43 (2)                | 63 (3)  | **1.5x**                  |

__Table 2. Throughput and NVSwitch benefit for Llama 3.1 70B inference at various real-time user experience targets with batch sizes__

_Throughput modeled using internal measurements. H200 GPU, ISL/OSL = 8k/256._ 

As Table 2 shows, a single GPU configuration (TP=1) is challenged to achieve real-time performance. Splitting the model using tensor parallel across two GPUs combines the compute resources of both GPUs to achieve high throughput across a wide range of real-time experience budgets. Real-time inference throughput on NVIDIA H200 GPUs with TP=2 and NVSwitch is up to 1.5x greater than a comparable GPU without NVSwitch.

To show how NVSwitch benefits scenarios with greater GPU-to-GPU communication traffic, Table 3 shows overall server throughput at fixed batch sizes. Larger batch sizes mean that requests from an increasing number of users can be processed at one time, improving overall server utilization and reducing cost per inference. 

| **Batch Size**     | **Throughput** **tok/s/GPU** |     | **NVSwitch** **Benefit** |
| ------------------ | ---------------------------- | --- | ------------------------ |
| **Point-to-Point** | **NVSwitch**                 |     |                          |
| 1                  | 25                           | 26  | **1.0x**                 |
| 2                  | 44                           | 47  | **1.1x**                 |
| 4                  | 66                           | 76  | **1.2x**                 |
| 8                  | 87                           | 110 | **1.3x**                 |
| 16                 | 103                          | 142 | **1.4x**                 |
| 32                 | 112                          | 168 | **1.5x**                 |

_Table 3. Throughput and NVSwitch benefit for Llama 3.1 70B inference at various fixed-batch sizes_

_Throughput modeled using internal measurements. H200 GPU, TP=2, ISL/OSL = 8K/256._ 

As batch size increases, GPU-to-GPU traffic increases, as does the benefit provided by NVSwitch compared to a point-to-point topology. However, even at relatively modest batch sizes, the gains can be significant.  

## Continued NVLink innovation for trillion-parameter model inference
NVLink and NVSwitch provide high bandwidth communication between GPUs based on the NVIDIA Hopper architecture and provide significant benefits for real-time, cost-effective large model inference today. 

As model sizes continue to grow, NVIDIA continues to innovate with both NVLink and NVSwitch to push the boundaries of real-time inference performance for even larger NVLink domains.

The NVIDIA Blackwell architecture features fifth-generation NVLink, which doubles per-GPU NVLink speeds to 1,800 GB/s. For Blackwell, a new NVSwitch chip and NVLink switch trays have also been introduced to enable even larger NVLink domain sizes.
>  Blackwell 架构具有第 5 代 NVLink，速度达到 1800 GB/s

The NVIDIA GB200 NVL72 system connects 36 NVIDIA Grace CPUs and 72 NVIDIA Blackwell GPUs in a rack-scale design, and with the fifth-generation NVLink, enables all 72 GPUs to act as a single GPU, enabling 30x faster real-time trillion-parameter inference compared to the prior generation.

## Related resources
- GTC session: [Accelerate Inference on NVIDIA GPUs](https://www.nvidia.com/gtc/session-catalog/?tab.catalogallsessionstab=1700692987788001F1cG&search=S72330&ncid=em-even-124008-vt33-23spring#/)
- GTC session: [Tencent HunYuan: Building a High-Performance Inference Engine for Large Models Based on NVIDIA TensorRT-LLM](https://www.nvidia.com/gtc/session-catalog/?tab.catalogallsessionstab=1700692987788001F1cG&search=S71563&ncid=em-even-124008-vt33-23spring#/)
- GTC session: [Blueprint for Supercharging LLM Inference With "PagedAttention over RDMA" (Presented by Weka)](https://www.nvidia.com/gtc/session-catalog/?tab.catalogallsessionstab=1700692987788001F1cG&search=S74226&ncid=em-even-124008-vt33-23spring#/)
- NGC Containers: [Phind-CodeLlama-34B-v2-Instruct](https://catalog.ngc.nvidia.com/orgs/nim/teams/phind/containers/phind-codellama-34b-v2-instruct?ncid=em-nurt-245273-vt33)
- NGC Containers: [Llama-3.1-Nemotron-70B-Instruct](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llama-3.1-nemotron-70b-instruct?ncid=em-nurt-245273-vt33)
- NGC Containers: [CodeLlama-70B-Instruct](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/codellama-70b-instruct?ncid=em-nurt-245273-vt33)