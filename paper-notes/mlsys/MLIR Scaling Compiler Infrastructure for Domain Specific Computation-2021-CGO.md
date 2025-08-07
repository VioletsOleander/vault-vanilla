# Abstract
This work presents MLIR, a novel approach to building reusable and extensible compiler infrastructure. MLIR addresses software fragmentation, compilation for heterogeneous hardware, significantly reducing the cost of building domain specific compilers, and connecting existing compilers together.
>  MLIR 是一种构建可重用和可拓展编译器基础设置的方法
>  MLIR 解决了软件碎片化问题，实现了对异构硬件的编译，显著降低了构建领域特定编译器的成本，并连接了现有的编译器

MLIR facilitates the design and implementation of code generators, translators and optimizers at different levels of abstraction and across application domains, hardware targets and execution environments. 
>  MLIR 促进在跨不同应用领域、硬件目标、执行环境的不同抽象层次上代码生成器、转换器、优化器的实现

The contribution of this work includes (1) discussion of MLIR as a research artifact, built for extension and evolution, while identifying the challenges and opportunities posed by this novel design, semantics, optimization specification, system, and engineering. (2) evaluation of MLIR as a generalized infrastructure that reduces the cost of building compilers --describing diverse use-cases to show research and educational opportunities for future programming languages, compilers, execution environments, and computer architecture. 
>  本文的贡献包括:
>  1. 将 MLIR 作为研究工具进行讨论，它旨在拓展和演进，同时识别由这种新设计、语义、优化规范、系统和工程所带来的挑战和机遇
>  2. 将 MLIR 作为一个通用的基础设施评估，MLIR 降低了构建编译器的成本

The paper also presents the rationale for MLIR, its original design principles, structures and semantics.
>  本文还阐述了 MLIR 的动机、原始设计原则、结构和语义

# 1 Introduction
Compiler design is a mature field with applications to code generation, static analysis, and more. The field has seen the development of a number of mature technology platforms which have enabled massive reuse, including systems like the LLVM compiler infrastructure [1], the Java Virtual Machine (JVM) [2], and many others. A common characteristic of these popular systems is their "one size fits all" approach—a single abstraction level to interface with the system: the LLVM Intermediate Representation (IR) is roughly "C with vectors", and JVM provides an "object-oriented type system with a garbage collector" abstraction. This "one size fits all" approach is incredibly valuable—and in practice, the mapping to these domains from ubiquitous source languages (C/C++ and Java respectively) is straightforward.
>  编译器设计是一个成熟的领域，应用于代码生成、静态分析等
>  该领域已经发展出许多成熟的技术平台，这些平台促进了大规模重用，包括了 LLVM 编译器设施，JVM 等
>  这些流行系统的一个共同特征是 “一刀切” 方法 —— 通过一层统一的抽象层来和系统交互: LLVM IR 大致是 "带有 vector 的 C", JVM 提供的是 "带有 GC 的面向对象的类型系统"

At the same time, many problems are better modeled at a higher- or lower-level abstraction, e.g. source-level analysis of  $C + +$  code is very difficult on LLVM IR. We observe that many languages (including e.g. Swift, Rust, Julia, Fortran) develop their own IR in order to solve domain-specific problems, like language/library-specific optimizations, flow-sensitive type checking (e.g. for linear types), and to improve the implementation of the lowering process. Similarly, machine learning systems typically use "ML graphs" as a domain-specific abstraction in the same way.
>  同时，许多问题更适合在更高或更低的抽象层次建模
>  例如 C++ 代码的源码分析在 LLVM IR 层级非常难以实现
>  我们发现许多语言开发了它们自己的 IR 以解决领域特定的问题，例如特定于语言/库的优化、流敏感的类型检查、以及改善下降过程的实现
>  类似地，ML 系统通常使用 ML graph 作为其领域特定的抽象

While the development of domain-specific IRs is a well studied art, their engineering and implementation cost remains high. The quality of the infrastructure is not always a first priority (or easy to justify) for implementers of these systems. Consequently, this can lead to lower quality compiler systems, including user-visible problems like slow compile times, buggy implementations, suboptimal diagnostic quality, poor debugging experience for optimized code, etc.
>  领域特定的 IR 的实现成本很高，且基础设施的质量不总是实现者考虑的首要事情

The MLIR project aims to directly tackle these programming language design and implementation challenges—by making it cheap to define and introduce new abstraction levels, and provide "in the box" infrastructure to solve common compiler engineering problems. MLIR does this by (1) standardizing the Static Single Assignment (SSA)-based IR data structures, (2) providing a declarative system for defining IR dialects, and (3) providing a wide range of common infrastructure including documentation, parsing and printing logic, location tracking, multithreaded compilation support, pass management, etc.
>  MLIR project 目标是直接解决这些编程语言的设计和实现挑战 —— 通过使得定义和引入新的抽象层次变得廉价，并提供 “内置” 的基础设施来解决常见的编译器工程问题
>  MLIR 通过以下的方式实现这一点:
>  1. 标准化基于 SSA 的 IR 数据结构
>  2. 提供了定义 IR 方言的声明式系统
>  3. 提供广泛的通用基础设施，包括文档、解析和打印逻辑、位置追踪、多线程编译支持、pass 管理等

This paper further presents the overarching principles underlying the design and implementation of MLIR. We will explore the essential design points of the system and how they relate to the overarching principles, sharing our experience applying MLIR to a number of compilation problems.

## A. Contributions
Most of the MLIR system is built out of well known concepts and algorithms. Yet the objectives and design are sufficiently novel that studying them offer vast opportunities for research, and even more so within the boundaries of the following overarching principles:
>  MLIR 没有发明新的算法，核心是在多个抽象层次之间进行建模和转化
>  三大核心设计原则如下:

*Parsimony*: Apply Occam's razor to builtin semantics, concepts, and programming interface. Harness both intrinsic and incidental complexity by abstracting properties of operations and types. Specify invariants once, but verify correctness throughout. Query properties in the context of a given compilation pass. With very little builtin, this opens the door to extensibility and customization.

>  简约性: 对内建语义、概念、编程接口应用奥卡姆剃刀原则，通过抽象 operation 和 type 的属性来驾驭内在复杂性和偶然复杂性
>  一次指定不变式，但在整个编译过程验证正确性
>  在给定的编译 pass 的上下文中查询属性
>  由于内建内容非常少，故促进了可拓展性和定制性

>  本质复杂性是问题本身无法避免的复杂，例如矩阵乘的计算量
>  偶然复杂性是工具和设计带来的复杂性，例如手动管理内存
>  MLIR 通过属性抽象来统一描述操作的行为，使得优化器统一根据声明式属性来做决策，而不是硬编码规则

>  不变式指定一次之后，后续的每个 pass 都会检查是否符合不变式，这个机制叫 verification
>  同时优化器可以在需要是查询操作是否符合特定的不变式，实现按需分析

*Traceability*: Retain rather than recover information. Declare rules and properties to enable transformation, rather than stepwise imperative specification. Extensibility comes with generic means to trace information, enforced by extensive verification. Composable abstractions stem from "glassboxing" their properties and separating their roles-type, control, data flow, etc.

>  可追溯性: 保留信息，而不是事后恢复信息 (传统编译器在下降时会丢弃高层语义，MLIR 确保可以往返)
>  通过声明规则和属性来实现转换，而不是一步一步地命令式描述 
>  提供了通用的信息追踪机制，并加以广泛验证，来提供可拓展性 (所有操作需要满足接口规范，确保统一的信息追踪可以使用)
>  可组合的抽象来自于 “透明化” 其属性，并分离其职责 —— 类型、控制流、数据流等 (MLIR 的操作可以实现接口，接口是正交的，可以自由组合)

*Progressivity*: Premature lowering is the root of all evil. Beyond representation layers, allow multiple transformation paths that lower individual regions on demand. Together with abstraction-independent principles and interfaces, this enables reuse across multiple domains.

>  渐进式降级: **过早的下降是一切问题的根源** (过早下降，丢失优化所需的必要语义)
>  MLIR 允许除了表示层次以外，有多个转换路径，按需下降单独的 region (代码块) (例如 CPU 控制逻辑先下降，GPU kernel 保留便于后续优化)，结合与抽象无关的原则和接口，实现跨多个领域的复用 (优化 pass，验证逻辑、内存分析等可以设计为不依赖于具体抽象层次)

>  Premature lowering is the root of all evil

While these principles are well established, one of them is often implemented at the expense of another; e.g., layering in network and operating system stacks aligns with the progressivity principle but breaks parsimony. This has also been the case in compilers with multiple layers of IR. Also, following these principles may hurt expressiveness and effectiveness; e.g., traceability in safety-critical and secure systems involves limiting optimizations and their aggressivity.
>  这些原则广为人知，但其中的一些原则往往在实现时会以牺牲另一些原则为代价
>  例如，在网络和 OS 中采用分层符合渐进性原则，但违背了简洁性原则，编译器中的多层 IR 也有类似的情况
>  此外，遵循这些原则可能会损害表达能力和效果，例如，在关注安全的系统中，可追溯性需要限制优化及其激进程度

In a nutshell, we identify design and engineering principles for compiler construction to thrive in a narrow middle that support an open semantics ecosystem. We discovered complexity can be tamed without restricting expressivity, allowing for fast IR design exploration and consolidation across domains, both of which are severely lacking in production systems.
>  简而言之，我们确定了编译器构建的工程和设计原则
>  我们发现，复杂性可以在不限制表达能力的情况下得到控制，从而实现快速的 IR 设计探索和跨领域整合，这两个能力在生产系统中严重缺失

The contributions of this paper are: (1) positioning the problem of building scalable and modular compiler systems in terms of proven design and engineering principles; (2) a description of a novel compiler infrastructure that follows these principles, with important industrial and research applications; (3) exploration of selected applications to diverse domains, illustrating the generality of the approach and sharing experience developing systems that build on the MLIR infrastructure.
>  本文的贡献包括:
>  1. 从已经验证的设计和工程原则触发，定位构建扩展的和模块化的编译系统的问题
>  2. 描述一种遵循这些原则的新编译器基础设施
>  3. 探索了该方法在多个领域的应用实例，展示该方法的通用性，并分享基于 MLIR 基础设施的开发系统经验

## B. Where Did MLIR Come From?
Work on MLIR began with a realization that modern machine learning frameworks are composed of many different compilers, graph technologies, and runtime systems (see Figure 1) which did not share a common infrastructure or design principles. This manifested in multiple user-visible ways, including poor error messages, failures in edge cases, unpredictable performance, and difficulty generalizing the stack to support new hardware.
>  MLIR 起源于我们发现现代 ML 框架由多个不同编译器、图技术、运行时系统组成，而它们都不共享一个共同的基础设施或设计原则，见 Figure1

We soon realized that the compiler industry as a whole has a similar problem: existing systems like LLVM are very successful at unifying and integrating work across a range of different languages, but high-level languages often end up building their own high-level IR and reinventing the same kind of technology for higher levels of abstraction (see Figure 2). At the same time, the LLVM community struggled with the representation of parallel constructs, and how to share frontend lowering infrastructure (e.g. for C calling conventions, or cross-language features like OpenMP), with no satisfactory solutions.
>  我们很快意识到整个编译器行业存在类似的问题: 像 LLVM 这样的现有系统在统一和整合多种不同语言的工作方面非常成功，但高级语言往往会构建自己的 IR，并重复发明相同的技术来实现更高层次的抽象，见 Figure2

![[pics/MLIR-Fig1.png]]

Faced with this challenge, given we could not afford to implement  $N$  improved compiler instances, we decided to go for a more general solution: investing in a high-quality infrastructure which would benefit multiple domains, progressively upgrading existing systems, making it easier to tackle pressing problems like heterogeneous compilation for specialized accelerators, and provide new research opportunities. 
>  因此，我们决定开发一个高质量的基础架构，该架构将惠及多个领域，逐步升级现有的系统，使得处理例如对专用加速设备的异构编译这样的问题更容易解决

Now that we gathered a significant amount of experience building and deploying MLIR-based systems, we are able to look back on its rationale and design and discuss why this direction was pursued.
>  我们现在已经积累了大量构建和部署基于 MLIR 的系统的经验，进而能够回顾其原理和设计，并讨论为何选择了这一方向

# 2 Design Principles
Let us now explore the requirements that guided the design of MLIR and their relation with the overarching principles.
>  我们来讨论指导了 MLIR 设计的需求以及它们和总体原则之间的关系

*Little Builtin, Everything Customizable \[Parsimony\]*: The system is based on a minimal number of fundamental concepts, leaving most of the intermediate representation fully customizable. A handful of abstractions—types, operations and attributes, which are the most common in IRs—should be used to express everything else, allowing fewer and more consistent abstractions that are easy to comprehend, extend and adopt. Broadly, customizability ensures the system can adapt to changing requirements and is more likely to be applicable to future problems. In that sense, we ought to build an IR as a rich infrastructure with reusable components and programming abstractions supporting the syntax and semantics of its intermediate language.
>  少量内建，全部可自定义 (简约性): 
>  系统基于最少的基本概念构建，IR 的大部分内容可以完全自定义
>  只有少量的抽象: types, operations, attributes，这三个抽象是 IR 中最常见的元素，这三个抽象可以用于表示所有其他内容
>  可自定义性则确保了系统能够适应不断变化的需求，并有可能适用于未来的问题

A success criterion for customization is the possibility to express a diverse set of abstractions including machine learning graphs, ASTs, mathematical abstractions such as polyhedral, Control Flow Graphs (CFGs) and instruction-level IRs such as LLVM IR, all without hard-coding concepts from these abstractions into the system.

Certainly, customizability creates a risk of internal fragmentation due to poorly compatible abstractions. While there is unlikely a purely technical solution, the system should encourage one to design reusable abstractions and assume they will be used outside of their initial scope.
>  可自定义性可能会由于不兼容的抽象而导致内部碎片化
>  虽然不存在技术解决方案，但系统鼓励设计可重用的抽象

*SSA and Regions \[Parimony\]:* The Static Single Assignment (SSA) form [3] is a widely used representation in compiler IRs. It provides numerous advantages including making dataflow analysis simple and sparse, is widely understood by the compiler community for its relation with continuation-passing style, and is established in major frameworks. As a result, the IR enforces the value-based semantics of SSA, its referential transparency and algorithmic efficiency, all considered essential to a modern compiler infrastructure. However, while many existing IRs use a flat, linearized CFG, representing higher level abstractions push introducing nested regions as a first-class concept in the IR. This goes beyond the traditional region formation to lift higher level abstractions (e.g., loop trees), speeding up the compilation process or extracting instruction, or SIMD parallelism [4], [5], [6]. To support heterogeneous compilation, the system has to support the expression of structured control flow, concurrency constructs, closures in source languages, and many other purposes. One specific challenge is to make CFG-based analyses and transformations compose over nested regions.
>  SSA and Regions (简约性): SSA 形式是编译器 IR 广泛使用的形式，它提供了诸多优势，包括使得数据流分析简单且稀疏
>  SSA 形式也被编译器社区广泛理解，并在主要框架中确立，现代编译器基础设施的关键要素就包括了 SSA 带来的基于值的语义、引用透明性和算法效率
>  此外，虽然许多现存的 IR 使用扁平的、线性化的控制流图，但为了表示更高层次的抽象，还是需要将 nested regions 作为 IR 的 first-class concept 引入，这是为了能够表示更高层次的抽象 (例如循环树)
>  为了支持异构编译，系统必须支持结构化控制流、并发构造、源语言的闭包等多个概念

In doing so, we agree to sacrifice the normalization, and sometimes the canonicalization properties of LLVM. Being able to lower a variety of data and control structures into a smaller collection of normalized representations is key to keeping compiler complexity under control. The canonical loop structure with its pre-header, header, latch, body, is a prototypical case of a linearized control flow representation of a variety of loop constructs in front-end languages. We aim at offering users a choice: depending on the compilation algorithm of interest, of the pass in the compilation flow, nested loops may be captured as nested regions, or as linearized control flow. By offering such a choice, we depart from the normalization-only orientation of LLVM while retaining the ability to deal with higher level abstractions when it matters. In turn, leveraging such choices raises questions about how to control the normalization of abstractions, which is the purpose of the next paragraph.
>  为此，我们同意牺牲 LLVM 中的规范化性质
>  能够将各种数据和控制结构降低为少量的规范化表示是控制编译器复杂性的关键
>  规范化的循环结构包含了 pre-header, header, latch, body，也就是线性化的控制流表示
>  但我们意在为用户提供更多选择，既可以用嵌套区域表示嵌套循环，也可以用线性化控制流表示嵌套循环，这偏离了 LLVM 仅关注规范化的方向，但保留了能够处理高层次抽象的能力

*Maintain Higher-Level Semantics \[Progressivity\]*: The system needs to retain the information and structure that are required for analysis or optimizing performance. Attempts to recover abstract semantics once lowered are fragile and shoehorning this information at low-level often invasive (e.g., all passes need to be revisited in the case of using debug information to record structure). Instead, the system should maintain the structure of computations and progressively lower to the hardware abstraction. The loss of structure is then conscious and happens only where the structure is no longer needed to match the underlying execution model. For example, the system should preserve the structured control flow such as loop structure throughout the relevant transformations; removing this structure, i.e. lowering to a CFG essentially means no further transformations will be performed that exploits the structure. The state of the art in modeling parallel computing constructs in a production compiler highlights how difficult the task may be in general [7], [8].
>  保持高层语义 (渐进性): 系统需要保留用于分析或优化性能所需的信息和结构
>  一旦降低到低层次，再试图恢复抽象语义会非常难
>  相反，系统应该保持计算的结构，并逐步降低到硬件抽象，结构的丢失应该是有意识的，并且仅再结构不需要在匹配底层的执行模型时发生
>  例如，系统应该将控制流例如循环结构保持，移除该结构 (例如降低为控制流图) 意味着不需要再执行任何利用该结构的转换

As a corollary, mixing different levels of abstraction and different concepts in the same IR is a key to allowing a part of the representation to remain in higher-level abstraction while another part is lowered. This would enable, for instance, a compiler for a custom accelerator to reuse some higher-level structure and abstractions defined by the system alongside with primitive scalar/vector instructions specific to the accelerator.
>  作为推论，将不同层次的抽象和不同概念混合在相同的 IR 中，是使得它能够同时表示高层和低层抽象的关键
>  例如，这将使得针对自定义加速设备的编译器可以复用一些 MLIR 系统定义的高级结构，同时使用该加速设备的原生标量/向量指令

Another corollary is that the system should support progressive lowering, from the higher-level representation down to the lowest-level, performed in small steps along multiple abstractions. The need for multiple levels of abstractions stems from the variety of platforms and programming models a compiler infrastructure has to support.
>  另一个推论是，系统应该支持渐进式降级，从高层逐步到低层
>  对多层抽象的需求源于编译器基础设施需要支持各种各样的平台和编程模型

Previous compilers have been introducing multiple fixed levels of abstraction in their pipeline—e.g. the Open64 WHIRL representation [9] has five levels, as does the Clang compiler which lowers from ASTs to LLVM IR, to SelectionDAG, to MachineInstr, and to MCInst. More flexible designs are required to support extensibility. This has deep implications on the phase ordering of transformations. As compiler experts started implementing more and more transformation passes, complex interactions between these passes started appearing. It was shown early on that combining optimization passes allows the compiler to discover more facts about the program. One of the first illustrations of the benefits of combining passes was to mix constant propagation, value numbering and unreachable code elimination [10].

*Declaration and Validation \[Parsimony and Traceability\]:* Defining representation modifiers should be as simple as introducing new abstractions; a compiler infrastructure is only as good as the transformations it supports. Common transformations should be implementable as rewrite rules expressed declaratively, in a machine-analyzable format to reason about properties of the rewrites such as complexity and completion. Rewriting systems have been studied extensively for their soundness and efficiency, and applied to numerous compilation problems, from type systems to instruction selection. Since we aim for unprecedented extensibility and incremental lowering capabilities, this opens numerous avenues for modeling program transformations as rewrite systems. It also raises interesting questions about how to represent the rewrite rules and strategies, and how to build machine descriptions capable of steering rewriting strategies through multiple levels of abstraction. The system needs to address these questions while preserving extensibility and enforcing monotonic and reproducible behavior.
>  声明和验证 (简约性和可追溯性):
>  定义 representation modifiers 应该像引入新抽象一样简单
>  编译器基础设置的价值取决于它所支持的 transformations，常见的 transformations 应该能够以声明式的方式表达为 rewrite rules
>  rewrite rules 已经被广泛研究，并应用到多个编译问题，从类型系统到指令选择，由于我们追求前所未有的可拓展性和逐步降级能力，我们就需要将程序转换建模为重写系统

The openness of the ecosystem also calls for an extensive validation mechanism. While verification and testing are useful to detect compiler bugs, and to capture IR invariants, the need for robust validation methodologies and tools is amplified in an extensible system. The mechanism should aim to make this easy to define and as declarative as practical, providing a single source of truth. A long term goal would be to reproduce the successes of translation validation [11], [12], [13], [14] and modern approaches to compiler testing [15]. Both are currently open problems in the context of an extensible compiler.
>  同样需要有一个全面的验证机制，该机制应该使得定义像声明一样容易，并提供一个单一的真相来源

*Source Location Tracking \[Traceability\]:* The provenance of an operation—including its original location and applied transformations—should be easily traceable within the system. This intends to address the lack-of-transparency problem, common to complex compilation systems, where it is virtually impossible to understand how the final representation was constructed from the original one.
>  源代码位置追踪 (可追溯性): operation 的来源应该在系统中易于追踪，这个性质旨在解决复杂编译系统中的透明度不足的问题

This is particularly problematic when compiling safety-critical and sensitive applications, where tracing lowering and optimization steps is an essential component of software certification procedures [16]. When operating on secure code such as cryptographic protocols or algorithms operating on privacy-sensitive data, the compiler often faces seemingly redundant or cumbersome computations that embed a security or privacy property not fully captured by the functional semantics of the source program: this code may prevent the exposure of side channels or harden the code against cyber or fault attacks. Optimizations may alter or completely invalidate such protections [17]; this lack of transparency is known as WYSINWYX [18] in secure compilation. One indirect goal of accurately propagating high-level information to the lower levels is to help support secure and traceable compilation.

# 3 IR Design
Our main contribution is to present an IR that follows the principles defined in the previous section. This is what MLIR does and we review its main design points in this section.
>  我们的主要贡献是提供一个遵循了之前定义的 principles 的 IR

MLIR has a generic textual representation (example in Figure 3) that supports MLIR's extensibility and fully reflects the in-memory representation, which is paramount for traceability, manual IR validation and testing. Extensibility comes with the burden of verbosity, which can be compensated by the custom syntax that MLIR supports; for example, Figure 7 illustrates the user-defined syntax for Figure 3.
>  MLIR 具有通用的文本形式表示，它可拓展，并完全反映内存中的表示形式

```
// Attribute aliases can be forward-declared.
#map1 = (d0, d1) -> (d0 + d1) 
#map3 = ()[s0] -> (s0)

// Ops may have regions attached. 
"affine.for"(%arg0) ({
// Regions consist of a CFG of blocks with arguments.
^bb0(%arg4: index):
  // Block are lists of operations. 
  "affine.for"(%arg0) ({ 
  ^bb0(%arg5: index):
    // Ops use and define typed values, which obey SSA.
    %0 = "affine.load"(%arg1, %arg4) {map = (d0) -> (d0)} : (memref<?xf32>, index) -> f32
    %1 = "affine.load"(%arg2, %arg5) {map = (d0) -> (d0)} : (memref<?xf32>, index) -> f32
    %2 = "std.mulf"(%0, %1) : (f32, f32) -> f32
    %3 = "affine.load"(%arg3, %arg4, %arg5) {map = #map1} : (memref<?xf32>, index, index) -> f32
    %4 = "std.addf"(%3, %2) : (f32, f32) -> f32
    "affine.store"(%4, %arg3, %arg4, %arg5) {map = #map1} : (f32, memref<?xf32>, index, index) -> ()
    // Blocks end with a terminator Op. 
    "affine.terminator"() : () -> ()
  // Ops have a list of attributes.
  }) {lower_bound = () -> (0), step = 1 : index, upper_bound = #map3} : (index) -> ()
"affine.terminator"() : () -> ()
}) {lower_bound = () -> (0), step = 1 : index, upper_bound = #map3} : (index) -> ()
```

Fig. 3. MLIR generic representation for polynomial multiplication using affine and std dialects. The same IR is displayed with the custom syntax Figure 7.

*Operations*: The unit of semantics in MLIR is an "operation", referred to as Op. Everything from "instruction" to "function" to "module" are modeled as Ops in this system. MLIR does not have a fixed set of Ops, but allows (and encourages) user-defined extensions, according to the parsimony and "everything customizable" principles. The infrastructure provides a declarative syntax for defining Ops based on TableGen [19], as illustrated in Figure 5.2
>  MLIR 的语义单位为 operation，称为 Op
>  Op 建模 “指令” “函数” "模块" 等概念
>  MLIR 不提供固定的一组 Ops
>  MLIR 提供定义 Ops 的声明式语法

```
%results:2 = "d.operation"(%arg0, %arg1) ({
  // Regions belong to Ops and can have multiple blocks.
  ^block(%argument: !d.type):
    // Ops have function types (expressing mapping).
    %value = "nested.operation"() ({ 
      // Ops can contain nested regions. 
      "d.op"() : () -> ()
  }) : () -> (!d.other_type)
  "consume.value"(%value) : (!d.other_type) -> ()
  ^other_block:
    "d.terminator"() [^block(%argument : !d.type)] : () -> ()
})
// Ops can have a list of attributes.
{attribute="value" : !d.type} : () -> (!d.type, !d.other_type)
```

Fig. 4. Operation (Op) is a main entity in MLIR; operations contain a list of regions, regions contain a list of blocks, blocks contains a list of Ops, enabling recursive structures

*Ops* (see Figure 4) have a unique opcode, a string identifying the operation and its dialect. Ops take and produce zero or more values, called operands and results respectively, and these are maintained in SSA form. Values represent data at runtime, and have a Type that encodes the compile-time knowledge about the data. In addition to an opcode, operands and results, Ops may also have Attributes, Regions, Successor Blocks, and Location Information. Figure 3 illustrates values and Ops, %-identifiers are (packs of) named values, with ": " specifying the number in a pack if more than one and "#" a particular value. In the generic textual representation, operation names are quoted string literals followed by operands in parentheses.
>  Ops 接收 value，输出 value，即 operands, results
>  values 都是 SSA 形式
>  values 表示运行时的数据, values 有 Type, Type 在编译时确定

Compiler passes treat unknown Ops conservatively, and MLIR has rich support for describing the semantics of Ops to passes through traits and interfaces as described in Section V-A. Op implementation has verifiers that enforce the Op invariants and participate in overall IR validation.
>  MLIR 通过 traits, interfaces 像 passes 传递 Ops 的语义
>  Op 可以实现 verifiers，用于在整个过程中描述 Op 需要遵守的不变式

*Attributes*: MLIR attributes contain compile-time information about operations, other than the opcode. Attributes are typed (e.g., integer, string), and each Op instance has an open key-value dictionary from string names to attribute values. In the generic syntax, attributes are found in a brace-enclosed comma-separated list of pairs. 
>  Attributes 存储 Ops 的编译时信息
>  Attributes 有 type
>  Op instance 使用 string-to-attribute values 的字典存储 attributes

Figure 3 uses attributes to define bounds of a loop that are known to be constant affine forms: `{lower_bound  = () ->  (0) step = 1: index upper_bound =  #map3 }` where `lower_bound` is an example of an attribute name. The  `() -> (0)`  notation is used for inline affine forms, in this case producing an affine function producing a constant 0 value. The ` #map3 ` notation is used for attribute aliases, which allow associate attribute values with a label upfront.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/59388ec6-3392-4edb-84d3-062dd0ab9a46/5b967b41f54c8db0a7f767646ff756c164e3976efd8d89a8eb3d3b30ad8705bf.jpg)  

Fig. 5. Operation Definition Syntax (ODS) provides a concise way of defining new Ops in MLIR. Here, one defines the LeakyRelu Op taking a tensor and a floating-point value, and returning a tensor of the same type as the input one.

Attributes derive their meaning either from the Op semantics or from the dialect (Section III) they are associated with. As with opcodes, there is no fixed set of attributes. Attributes may reference foreign data structures, which is useful for integrating with existing systems, e.g., the contents of (known at compile time) data storage in an ML system.
>  attributes 的意义和 Ops 语义或 dialect 语义相关
>  MLIR 不提供固定的一组 attributes
>  attributes 可以用于引用外部数据结构，例如系统中数据存储中的数据

*Location Information*: MLIR provides a compact representation for location information, and encourages the processing and propagation of this information throughout the system, following the traceability principle. It can be used to keep the source program stack trace that produced an Op, to generate debug information. It standardizes the way to emit diagnostics from the compiler, and is used by a wide range of testing tools.
>  MLIR 鼓励在处理和传递中保存位置信息，遵循可追溯性的原则

Location information is also extensible, allowing a compiler to refer to existing location tracking systems, high-level AST nodes, LLVM-style file-line-column address, DWARF debugging info, etc.

*Regions and Blocks*: An instance of an Op may have a list of attached regions. A region provides the nesting mechanism in MLIR: it contains a list of blocks, each of which contains a list of operations (that may contain further regions). As with attributes, the semantics of a region are defined by the operation they are attached to, however the blocks inside the region (if more than one) form a Control Flow Graph (CFG). For example, the affine.for operation in Figure 3 is a loop with the single-block body attached as a region, located between ({and}) delimiters. The Op specifies the flow of control across regions. In this example, the body is executed repeatedly until the upper bound is reached.
>  Op instance 可以有 a list of regions
>  regions 包含 a list of blocks
>  block 包含 a list of operations
>  region 内的 block 构成一个控制流图

The body of each region is a list of blocks, and each block ends with a terminator operation, that may have successor blocks to which the control flow may be transferred. Each terminator (e.g. "switch", "conditional branch" or "unwind") defines its own semantics. It may chose to transfer the control flow to another block in the same region, or return it to the Op enclosing the region. The graph of successors defines a CFG, allowing standard SSA-based control flow within a region.
>  block 以 terminator operation 结尾
>  terminator operation 可以有 successor block，将控制流转移

Instead of using  $\phi$  nodes, MLIR uses a functional form of SSA [20] where terminators pass values into block arguments defined by the successor block. Each block has a (potentially empty) list of typed block arguments, which are regular values and obey SSA. The semantics of terminator Ops defines what values the arguments of the block will take after the control is transferred. For the first (entry) block of the region, the values are defined by the semantics of the enclosing Op. For example, affine.for uses the entry block argument %arg4 as loop induction variable. Finally, this explicit graph design and the extensibility of Ops is reminiscent of the sea-of-nodes representation [21]: this connection is intentional and has been a major influence for the selection of MLIR's flavor of SSA.
>  terminator op 将 values 传递给 successor block 的 block arguments

*Value Dominance and Visibility*: Ops can only use values that are in scope, i.e. visible according to SSA dominance, nesting, and semantic restrictions imposed by enclosing operations. Values are visible within a CFG if they obey standard SSA dominance relationships, where control is guaranteed to pass through a definition before reaching a use.

Region-based visibility is defined based on simple nesting of regions: if the operand to an Op is outside the current region, then it must be defined lexically above and outside the region of the use. This is what allows Ops within an affine.for operation to use values defined in outer scopes.

MLIR also allows operations to be defined as isolated from above, indicating that the operation is a scope barrier—e.g. the "std.func" Op defines a function, and it is not valid for operations within the function to refer to values defined outside the function. In addition to providing useful semantic checking, a module containing isolated-from-above Ops may be processed in parallel by an MLIR compiler since no use-def chains may cross the isolation barriers. This is important for compilation to utilize multicore machines.

All these design choices highlight the progressivity principle, while erring on the side of parsimony when a concept does not appear to be generic and essential enough to be builtin.

*Symbols and Symbol Tables*: Ops can have a symbol table attached. This table is a standardized way of associating names, represented as strings, to IR objects, called symbols. The IR does not prescribe what symbols are used for, leaving it up to the Op definition. Symbols are most useful for named entities need that not obey SSA: they cannot be redefined within the same table, but they can be used prior to their definition. For example, global variables, functions or named modules can be represented as symbols. Without this mechanism, it would have been impossible to define, e.g., recursive function referring to themselves in their definition. Symbol tables can be nested if an Op with a symbol table attached has associated regions containing similar Ops. MLIR provides a mechanism to reference symbols from an Op, including nested symbols.
>  Ops 的 symbol table 将 string 和 IR object (symbols) 关联

*Dialects*: MLIR manages extensibility using Dialects, which provide a logical grouping of Ops, attributes and types under a unique namespace. Dialects themselves do not introduce any new semantics but serve as a logical grouping mechanism that provides common Op functionality (e.g., constant folding behavior for all ops in the dialect). They organize the ecosystem of language-and domain-specific semantics while following the parsimony principle. The dialect namespace appears as a dot-separated prefix in the opcode, e.g., Figure 3 uses affine and std dialects.
>  Dialect 组合了 Ops, attributes, types
>  dialect 本身不提供新语义，只是作为一个命名空间和容器

The separation of Ops, types and attributes into dialects is conceptual and is akin to designing a set of modular libraries. For example, a dialect can contain Ops and types for operating on hardware vectors (e.g., shuffle, insert/extract element, mask), and another dialect can contain Ops and types for operating on algebraic vectors (e.g. absolute value, dot product, etc.). Whether both dialects use the same vector type and where does this type belong are design decisions left to MLIR user.

While it is possible to put all Ops, types and attributes in a single dialect, it would quickly become unmanageable due to the large number of simultaneously present concepts and name conflicts, amongst other issues. Although each Op, type and attribute belongs to exactly one dialect, MLIR explicitly supports a mix of dialects to enable progressive lowering. Ops from different dialects can coexist at any level of the IR at any time, they can use types defined in different dialects, etc. Intermixing of dialects allows for greater reuse, extensibility and provides flexibility that otherwise would require developers to resort to all kinds of non-composable workarounds.
>  MLIR 允许不同 dialect 的 Ops 在同一层 IR 中存在

*Type System*: Every value in MLIR has a type, which is specified in the Op that produces the value or in the block that defines the value as an argument. Types encode compile-time information about a value. The type system in MLIR is user-extensible, and may, for example, refer to existing foreign type systems. MLIR enforces strict type equality checking and does not provide type conversion rules. Ops list their inputs and result types using trailing function-like syntax. In Figure 3, std.load maps from the memory reference and index types to the type of the value it loads.
>  values are typed
>  type 存储 value 的编译时信息，type system 也可以拓展
>  MLIR 不提供 type 转换规则，使用严格 type equality check

From the type theory point of view, MLIR only supports non-dependent types, including trivial, parametric, function, sum and product types. While it is possible to implement a dependent type system by combining Ops with symbols and user-defined types, such types will be opaque to the IR.

For convenience, MLIR provides a standardized set of commonly used types, including arbitrary precision integers, standard floating point types, and simple common containers—tuples, multi-dimensional vectors, and tensors. These types a merely a utility and their use is not required, illustrating parsimony.
>  MLIR 提供了标准的一组 types

*Functions and Modules*: Similarly to conventional IRs, MLIR is usually structured into functions and modules.
>  MLIR 通常组织为 functions, modules

However, these are not separate concepts in MLIR: they are implemented as Ops in the builtin dialect, again an illustration of parsimony in the design.

A module is an Op with a single region containing a single block, and terminated by a dummy Op that does not transfer the control flow. Like any block, its body contains a list of Ops, which may be functions, global variables, compiler metadata, or other top-level constructs. Modules may define a symbol in order to be referenced.
>  Op 中的 module 为 single region containing single block, terminated with a dummy Op

Similarly, a function is an Op with a single region that may contain zero (in case of declaration) or more blocks. Built-in functions are compatible with "call" and "return" operations of the std dialect, which transfer the control to and from the function, respectively. Other dialects are free to define their own function-like Ops.
>  Op 中的 function 为 single region containing blocks

# 4 Evaluation: Applications of MLIR
MLIR is a system that aims to generalize and drive a wide range of compiler projects, so our primary evaluation metric is to show that it is being adopted and used for diverse projects. By doing so we acknowledge the software engineering nature of the problem and contributions. We provide a summary of community activity and describe a few use cases in more detail to highlight the generality and extensibility of MLIR and evaluate how well compiler and domain experts experience the design principles of the IR.

Today, MLIR is a growing open source project with a community spanning academia and industry. For example, the first academic workshop about the use of MLIR in High-Performance Computing was attended by individuals from 16 universities and involved 4 national laboratories from 4 different countries. MLIR was also endorsed by 14 multinational companies and at the 2019 LLVM Developer Meeting more than 100 industry developers attended a roundtable event about MLIR. Community adoption and participation is a proxy measure for usability and need. More than 26 dialects are in development in public or private and 7 projects across different companies are replacing custom infrastructure with MLIR. We argue that this shows a real need for MLIR, as well as endorses its usability.

## A. TensorFlow Graphs
While the other discussed representations are familiar to most compiler developments, one of key use cases for MLIR is to support the development of machine learning frameworks. Their internal representations is often based on a data flow graph [22] with a dynamic execution semantics.
>  ML 框架的内部表示通常基于数据流图，带有动态执行语义

TensorFlow [23] is an example of such framework. Its representation is a high-level dataflow computation where the nodes are computations which can be placed on various devices, including specific hardware accelerators.
>  TensorFlow 的表示是高层数据流图，节点为计算，可以放置在任意设备上

MLIR is used in TensorFlow to model this internal representation and perform transformations for the use cases presented in Figure 1: from simple algebraic optimizations to retargeting graphs for parallel and distributed execution on data center clusters and asynchronous hardware acceleration, from lowering to a representation suitable for mobile deployment to generating efficient native code using domain-specific code generators like XLA [24]. 

The representation of a TensorFlow graph in MLIR is illustrated on Figure 6. It illustrates the modeling of asynchronous concurrency, where the dataflow graph is desynchronized via implicit futures and side-effecting Ops are serialized through explicit control signals (also following dataflow semantics). Despite the widely different abstractions, concurrency, asynchrony, delayed evaluation, MLIR offers the same infrastructure, analysis and transformation capabilities as for any other dialect or compiler pass. In particular, essential graph-level transformations implemented in Grappler are expressible in MLIR for both TensorFlow models and low level LLVM IR: dead code/node elimination, constant folding, canonicalization, loop-invariant code motion, common subexpression/subgraph elimination, instruction/device-specific-kernel selection, rematerialization, layout optimization; while other transformations may be domain-specific: optimizations for mixed precision, op fusion, shape arithmetic.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/59388ec6-3392-4edb-84d3-062dd0ab9a46/02928862fb07209367362ad2fb3d270eb1bcb2e22e2c3da2c6e816ccc64a4532.jpg)  

Fig. 6. SSA representation of a TensorFlow graph in MLIR.

## B. Polyhedral Code Generation
One of the original motivations for MLIR was the exploration of polyhedral code generation for accelerators. The affine dialect is a simplified polyhedral representation that was designed to enable progressive lowering. While a full exploration of the design points here is out of scope for this paper, we illustrate aspects of the affine dialect to show the modeling power of MLIR and contrast the affine dialect with past representations [25], [26], [27], [28], [29].
>  MLIR 的一个动机之一是为加速设备的代码生成利用多面体模型
>  affine dialect 为多面体表示 dialect

(1) *Similarities*: The MLIR affine dialect operates on a structured multi-dimensional type for all accesses to memory. In the default case, these structured types are injective: different indexings are guaranteed not to alias by construction, a common precondition for polyhedral dependence analyses.
>  affine dialect 对所有的内存访问使用结构化的多维类型表示

Affine modeling is split in two parts. Attributes are used to model affine maps and integer sets at compile-time and Ops are used to apply affine restrictions to the code. Namely, affine.for Op is a "for" loop with bounds expressed as affine maps of values required to be invariant in a function. Thus loops have static control flow. Similarly, affine.if is a conditional restricted by affine integer sets. The bodies of loops and conditionals are regions that use affine.load and affine.store to restrict indexing to affine forms of surrounding loop iterators. This enables exact affine dependence analysis while avoiding the need to infer affine forms from a lossy lower-level representation.
>  仿射建模分为两部分: attributes 在编译时建模仿射映射和整数集合，Ops 对代码应用仿射约束

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/59388ec6-3392-4edb-84d3-062dd0ab9a46/f260c7b89eab15a6b99d5fbdab54a540607723adc10d3685c37313a85d1aa71c.jpg)  
Fig. 7. Affine dialect representation of polynomial multiplication  ${C}(\mathbb{i} + \mathbb{j})$ $+ = \mathrm{A}(\mathrm{i})$  \*Bj.

(2) Differences with existing polyhedral: They are numerous:

1) Rich types: the MLIR structured memory reference type contains a layout map connecting the index space of the buffer to the actual address space. This separation of concerns makes loop and data transformations compose better: changes to data layout do not affect the code and do not pollute dependence analysis. Such mixes of transformations have been explored previously [30] but are uncommon.

2) Mix of abstractions: Bodies of affine loops in MLIR can be expressed with operations on typed SSA values. Therefore, all traditional compiler analyses and transformations remain applicable and can be interleaved with polyhedral transformations. On the contrary, polyhedral compilers often abstract such details away completely, making it challenging for a polyhedral compiler to manipulate, e.g., vector types.

3) Smaller representation gap: One of the key features of the polyhedral model is its ability to represent the order of loop iterations in the type system. In this system, a large number of loop transformations compose directly and can be reasoned about using simple mathematical abstractions [26]. However, polyhedral transformations require raising into a representation often drastically different from the original [31], [32]. Furthermore, the conversion from transformed polyhedra to loops is computationally hard [33]. MLIR-based representation maintains high-level loop structure around lower-level representation, removing the need for raising.

4) Compilation speed is a crucial goal for MLIR as discussed in Section V-D, but has not been a focus of most existing polyhedral approaches. These rely heavily on algorithms with exponential complexity: on integer linear programming to derive loop orderings automatically and on polyhedron scanning algorithms to convert the representation back to loops. The MLIR approach explicitly does not rely on polyhedron scanning since loops are preserved in the IR. In addition, code generation may take place ahead-of-time, e.g., when producing generic code for dynamic shapes, or just-in-time when specializing tensor operations on static shapes. The latter puts stricter constraints on available resources, and both scenarios are important.

Experience with the affine dialect shows that first-class affine abstractions facilitate the design and implementation of domain-specific code generators, including the linalg dialect, and declarative rewrite rules in RISE. These developments and the affine dialect itself represent important explorations that the MLIR design made practical.
>  first-class affine 抽象促进了领域特定的 code generator 的设计和实现

## C. Fortran IR (FIR)
The LLVM Fortran frontend "flang" is currently under major development, led by NVIDIA/PGI. Similar to Swift, Rust, and others, flang needs a specialized IR in order to support advanced transformations for high-performance Fortran codebase, and is using MLIR to support these Fortran-specific optimizations [34]. These high-level optimizations—advanced loop optimizations, array copy elimination, call specialization, devirtualization—would be hard implement using only LLVM.

For example, FIR is able to model Fortran virtual dispatch table as a first class concept (see Figure 8).

```
// Dispatch table for type(u) 
fir.dispatch_table @table_type_u { 
    fir.dt_entry "method", @u_method 
}

func @some_func() { 
    %uv = fir.alloca !fir.type<u> : !fir.ref<!fir.type<u>> fir.dispatch "method"(%uv) : (!fir.ref<!fir.type<u>=>) 
// ... 
}
```

Fig. 8. FIR has first class support for dynamic virtual function dispatch tables.

The ability to model the high-level semantics of the programming language in a structured IR is very powerful. For example, first-class modeling of the dispatch tables allows a robust devirtualization pass to be implemented. While this could have been implemented with a bespoke compiler IR, the use of MLIR allowed the flang developers to spend their engineering resources focused on the IR design for their domain instead of reimplementing basic infrastructure.

The choice of MLIR also unlocks the reusability of other dialects that are not specific to Fortran: a language-independent OpenMP dialect could be shared between Fortran and C language frontends. Similarly, targeting a heterogeneous platform using OpenACC becomes tractable within MLIR through the sharing and reuse of the GPU-oriented dialects and passes. This is straightforward thanks to MLIR begin specifically designed to support a mix of composable dialects.

## D. Domain-Specific Compilers
The applications above are within large workflows. But MLIR also helps building smaller domain specific compilers.
>  MLIR 帮助构建领域特定的编译器

A reusable and modular infrastructure makes these specialized paths feasible and relatively cheap to build.

*Optimizing MLIR Pattern Rewriting:* MLIR has an extensible system for pattern rewrites. In addition to statically declared patterns, we had applications where the rewrite patterns needed to be dynamically extensible at runtime, allowing hardware vendors to add new lowerings in drivers. The solution was to express MLIR pattern rewrites as an MLIR dialect itself, allowing us to use MLIR infrastructure to build and optimize efficient Finite State Machine (FSM) matcher and rewriters on the fly. This work includes FSM optimizations seen in other systems, such as the LLVM SelectionDAG and GlobalISel instruction selection systems.
>  MLIR 提供了可拓展的 pattern rewrites 系统

*Lattice Regression Compiler*: Lattice regression [35] is a machine learning technique renowned for fast evaluation times and interpretability. The predecessor of the compiler was implemented using  ${C} + +$  templates. This allowed for high-performance code with metaprogramming, but expressing general optimizations on the end-to-end models was not straightforward. This particular lattice regression system is used in applications with multiple millions of users and hence performance improvements are critical.

MLIR was used as the basis for a new compiler for this specialized area, which was driven by a specialized search approach—effectively resulting in a machine learning problem being solved during compilation. The resultant compiler was developed by investing a 3 person-month effort, and resulted in up to  $8\times$  performance improvement on a production model, while also improving transparency during compilation.

# 5 Consequences of The MLIR Design
The MLIR design facilitates the modeling of new language and compiler abstractions while reusing existing, generic ones. Effectively, the solution to many problems is to "add new ops, new types", possibly collected into "a new dialect". This is a significant design shift for compiler engineering. It produces new opportunities, challenges, and insights. This section explores a few of them.
>  MLIR 促进了新语言和新编译器抽象的建模，同时复用了现有的通用抽象
>  许多问题的解决方案变为添加新 ops, 新类型，这对编译器工程来说是一个设计转变

## A. Reusable Compiler Passes
The ability to represent multiple levels of abstraction in one IR incentivizes the passes that operate across these levels. MLIR handles extensibility by inverting the common approach: since there are more Ops than passes, it is easier for Ops to know about passes. This also improves modularity as the dialect-specific logic is implemented within the dialects instead of the core transformations. Since the passes rarely need to know all aspects of an Op, MLIR relies on the following mechanisms to implement generic passes.
>  MLIR 反转了常见的方法来处理可拓展性: 因为 Ops 比 passes 更多，故让 Ops 了解 passes 更加容易 (而不是 passes 了解 Ops)
>  这也提高了模块化，因为 dialect 特定的逻辑是在 dialect 内部实现的，而不是在核心转换中
>  MLIR 通过以下的机制来实现通用的 passes

*Operation Traits*: Many common "bread and butter" compiler passes, such as Dead Code or Common Subexpression Elimination, rely on simple properties like "is terminator" or "is commutative". We define such properties as Op Traits. An Op exhibits a trait unconditionally, e.g., a "standard branch" Op is always a terminator. For many passes, it is sufficient to know that an Op has a set of traits to operate on it, for example by swapping the operands or removing Ops with no side effects and no users.
>  Operation 特性: 许多常见的基础 compiler passes，例如死代码消除和公共子表达式消除，依赖于简单的属性，例如 “是否为终止操作”，“是否是可交换”
>  我们将这样的属性定义为 Op traits
>  Op 无条件地表示它的 traits
>  对于大多数 passes，只需要一个 Op 存在特定的 traits 就足以对它进行操作，例如交换 operands, 移除没有 side effects, users 的 Ops

Traits can serve as verification hooks allowing to share the logic across multiple Ops that have the trait. For example, the "isolated from above" trait verifies that no regions in the Op use values defined in the regions enclosing the Op. It allows for generic processing of functions, modules and other self-contained structures.
>  traits 可以作为验证方式，让拥有该 trait 的多个 Ops 共享逻辑
>  例如 isolated from above trait 验证了 Op 中没有 regions 使用了包含了该 Op 的 region 中定义的 values

*Interfaces*: When the unconditional, static behavior is insufficiently expressive, the processing can be parameterized through interfaces, a concept borrowed from object-oriented programming. An interface defines a view into the behavior of an IR object that abstracts away unnecessary details. Unlike traits, interfaces are implemented by IR objects, using arbitrary  $\mathbf{C} + +$  code that can produce different results for different objects. For example, the "call" Op implements a "call-like" interface, but different instances of the Op call different functions.
>  当无条件的静态行为 (traits) 表达能力不足时，可以通过接口对处理行为进行参数化
>  接口定义了对 IR object 行为的视图
>  接口需要由 IR objects 实现，使用 C++ 代码

MLIR passes can be implemented in terms of interfaces, establishing a contract with any Op that opts into being processed by a pass. Continuing the call-like example, consider the MLIR inlining pass that works on TensorFlow graphs, Flang functions, closures in a functional language etc. Such a pass needs to know: (1) whether it is valid to inline an operation into a given region, and (2) how to handle terminator operations that ended up in the middle of a block after inlining.
>  MLIR pass 可以基于接口实现
>  例如一个执行 inline 的 pass，它需要知道: 是否可以将 operation inline 到给定区域，如何处理 inline 后位于 block 中间的 terminator op

In order to query an Op about these properties, the pass defines a dedicated interface so that Ops may register their implementation with MLIR to benefit from inlining. The inlining pass will treat conservatively, i.e. ignore, any operation that does not implement the respective interface.
>  为此，这些 pass 定义了一个 interface
>  Op 需要将该 interface 实现

Constant folding is implemented through the same mechanism: each Op implements the "fold" interface by providing a function that may produce an attribute holding the value if the Op is foldable. 
>  constant folding 也是以相同的机制实现，Op 需要实现 "fold" interface，便于 pass 确定 Op 是否可以 fold

More generic canonicalization can be implemented similarly: an interface populates the list of canonicalization patterns amenable to pattern-rewriting. This design separates generic logic from Op-specific logic and puts the latter in the Op itself, reducing the well-known maintenance and complexity burden of "InstCombine", "PeepholeOptimizer" and the likes in LLVM.

Interfaces can be implemented by dialects rather than specific Ops, which enables shared behavior or delegation to the external logic, for example when constant folding TensorFlow Ops. Interfaces are also supported on types and attributes, for example an addition operation may support any type that self-declares as "integer-like" with queryable signedness semantics.

## B. Dialect-Specific Passes
Finally, it is valid and useful to define passes that are specific to particular dialects, which can be driven by full semantics of operations in the dialect(s) they are designed for. These passes are just as useful in the MLIR system as they are in other compiler systems. For example, code generators that want to do custom scheduling of machine instructions based on particular machine constraints or other tricks that do not fit into a broader framework. This is a simple and useful starting point for new transformations, where generalization isn't required.

## C. Mixing Dialects Together
One of the most profound (but also most difficult to grok) aspects of MLIR is that it allows and encourages mixing operations from different dialects together into a single program. While certain cases of this are reasonably easy to understand (e.g. holding host and accelerator computation in the same module) the most interesting cases occur when dialects are directly mixed—because this enables an entire class of reuse that we have not seen in other systems.
>  MLIR 最深刻，也最难理解的方面是允许并鼓励将来自不同 dialect 的 operation 混合到一个程序中

Consider the affine dialect described in Section IV-B. The definition of affine control flow and affine mappings are independent of the semantics of the operations that are contained in affine regions. In our case, we combine the affine dialect with the "standard" dialect that represents simple arithmetic in a target independent form like LLVM IR, with multiple target-specific machine instruction dialects for internal accelerators. Others have combined it with abstractions from other problem domains.

The ability to reuse generic polyhedral transformations (using Op interfaces to get semantics of operations in specific transformations) is a powerful (and exciting to us) way of factoring compiler infrastructure. Another example is that an OpenMP dialect could be used and reused across a wide variety of source-language IRs.

## D. Parallel Compilation
An important aspect of MLIR is the possibility to use multicore machines to increase the compilation speed. In particular, the "isolated from above" trait (Section V-A) allows Ops such as functions to opt into the concurrent IR traversal mechanism supported by MLIR's pass manager. Indeed this trait guarantees that SSA use-def chain cannot cross the region boundaries and can be processed in isolation. MLIR also does not feature whole-module use-def chains, but instead references global objects through symbol tables (Section III) and defines constants as operations with attributes (Section III).

## E. Interoperability
Our work involves interoperation with a large number of existing systems, e.g., machine learning graphs encoded as protocol buffers, compiler IRs including LLVM IR, proprietary instruction sets, etc. Often the representation has a number of suboptimal or unfortunate decisions that made sense in the context of an existing system, but capabilities of MLIR enable a more expressive representation. Because importers and exporters are notoriously difficult to test (test cases are often binary), we want to make sure their complexity is minimized.

The solution is to define a dialect that corresponds to the foreign system as directly as possible—allowing round tripping to-and-from that format in a simple and predictable way. Once the IR is imported into MLIR, it can be raised and lowered to a more convenient IR using all of the MLIR infrastructure, which allows those transformations to be tested similarly to all the other MLIR passes.

There are numerous examples of such dialects, including the LLVM dialect which maps LLVM IR into MLIR. This approach has worked well for us, and the MLIR tooling has also been useful to write tests for these foreign file formats.

## F. Unopinionated Design Provides New Challenges
While MLIR allows one to define almost arbitrary abstractions, it provides very little guidance on what should be done: what works better or worse in practice? We now have some experience with a number of engineers and researchers applying the techniques and technologies to new problem domains, and have realized that the "art" of compiler IR design and abstraction design is not well understood in the compiler and languages field-many people work within the constraints of established systems, but relatively few have had the opportunity define the abstractions themselves.

This is a challenge, but is also another set of opportunities for future research. The broader MLIR community is building expertise with these abstraction design trade-offs, and we expect this to be a fertile area of study over time.

## G. Looking Forward
The design of MLIR is different enough from other compiler infrastructures that we are still learning-even after building and applying it to many different systems. We believe that there is still a lot to discover, and several years of research will be required to better understand the design points and establish best practices. For example, the rise of out-of-tree dialects, increasing number of source language frontends using MLIR, possible application to Abstract Syntax Trees, and applications to structured data (like JSON, protocol buffers, etc) which are still very early and are likely to uncover interesting new challenges and opportunities. Better support for just-in-time compilation and precise garbage-collection would also be interesting, leveraging the modularity and programmability of the IR.

# 6 Related Work
MLIR is a project that overlaps with multiple different domains. While the composed infrastructure provides a novel system, individual components have analogs in the literature. For references and discussion directly related to the IR design itself, please refer to Section II.

MLIR is a compiler infrastructure akin to LLVM [1], but where LLVM has been a great boon to scalar optimizations and homogeneous compilation, MLIR aims to model a rich set of data structures and algorithms as first-class values and operations, including tensor algebra and algorithms, graph representations, as well as heterogeneous compilation. 
>  MLIR 是一个编译器基础设施，类似于 LLVM，但 LLVM 聚焦于标量优化和同构编译，MLIR 的目标是将丰富的数据结构和算法作为 first-class values, operations

MLIR allows mix-and-match optimization decomposing compilation passes into components and redefining lowering, cleanup roles. This is largely attributed to the pattern rewriting infrastructure, capturing full-fledged transformations as a composition of small local patterns and controlling which pattern rewrites are applied at the granularity of an individual operation. Extending, formalizing, and verifying the rewriting logic automatically would be an important next step [36], [37]. On the backend side, MLIR's DDR has an analogue to LLVM's instruction selection infrastructure, supporting extensible operations with multi-result patterns and specification as constraints [38].
>  MLIR 允许混合和匹配优化，这很大程度上归功于 pattern rewriting infrastructure
>  模式重写基础设施可以将完整的转换捕获为小、局部模式的组合，并控制在单个 operation 粒度上应用哪些 pattern rewrite

Numerous programming languages and models tackle hardware heterogeneity. Originally a homogeneous programming model, OpenMP added support for offloading tasks and parallel regions to accelerators [39], based on earlier proposals such as StarSs and OpenACC [40], [41].  $\mathrm{C + + }$  AMP, HCC and SyCL leverage a conventional Clang/LLVM flow and modern  $\mathrm{C + + }$  to provide a high-level abstraction for hardware acceleration [42]. Unfortunately, all these examples very quickly lower high-level constructs to calls to a runtime execution environment, relying on pre-existing optimizations in the host language (typically  $\mathrm{C + + }$  to alleviate the abstraction penalty. Far fewer efforts target the heterogeneous compilation process itself. Parallel intermediate representations extending LLVM IR address part of the issue but traditionally focus on the homogeneous setting [7], [8]. The most ambitious effort to date may be Liquid Metal [43], with a co-designed Domain Specific Language (DSL) and compilation flow converting managed object semantics into static, vector or reconfigurable hardware; yet most of the effort in its Lime compiler reside in fitting round objects into square hardware (paraphrasing Kou and Palsberg [44]). MLIR provides a direct embedding for high level languages embracing heterogeneity through extensible set of operations and types, while providing a common infrastructure for gradually lowering these constructs with maximal reuse of common components across the different targets.

Tackling language heterogeneity has been a long-term promise of metaprogramming systems, and of multistage programming in particular. Lightweight Modular Staging (LMS) [45] is a state of the art framework and runtime code generator, providing a library of core components for generating efficient code and embedding DSLs in Scala. Delite [46] promises dramatic productivity improvements for DSL developers, while supporting parallel and heterogeneous execution. This approach is complementary to MLIR, providing a higher-level of abstraction to embed DSLs and implement optimizations through generic metaprogramming constructs.

One step further up into the language syntax, ANTLR [47] is among a class of parser generators that aim to facilitate the development of compiler frontends. MLIR currently does not have a general parser generator, no AST construction or modeling functionality. Combining MLIR with a system such as ANTLR could expand reusability upstream all the way to frontends and development environments.

More narrowly construed by their application to machine learning, XLA [24], Glow [48] and TVM [49], address similar heterogeneous compilation objectives. These frameworks provide domain-specific code generation instances, starting from a graph abstraction and targeting multi-dimensional vector abstractions for accelerators. All of these could leverage MLIR as infrastructure, taking advantage of the common functionality while using their current code generation strategies. Similarly, the loop nest metaprogramming techniques from Halide [50] and TVM [49], earlier loop nest metaprogramming [26], [51], [52], [53], and automatic flows such as PolyMage [54], Tensor Comprehensions [29], Stripe [55], Diesel [56], Tiramisu [57] and their underlying polyhedral compilation techniques [25], [27], [58], [28] could co-exist as different code generation paths within an MLIR-based framework. This would greatly increase code reuse, defragmentation of the landscape, interoperability across domain, and portability. This is actually one of the motivations for the IREE project,8 building on MLIR at multiple levels of abstraction, from tensor algebra and operator graphs down to the low-level orchestration of asynchronous coroutines and code generation for multiple CPU and GPU architectures (within the Vulkan/SPiR-V standard).

Finally, interoperability formats, such as ONNX [59], have a different approach towards addressing the diversity of ML frontends by providing a common set of ops that different frameworks could map on to. ONNX would be a candidate as a dialect in MLIR to and from which ops could be converted.

# 7 Conclusion and Future Work
We presented MLIR, a concrete answer to the dual scientific and engineering challenge of designing a flexible and extensible infrastructure for compiler construction, ranging from backend code generation and orchestration of heterogeneous systems, to graph-level modeling for machine learning, and to the high-level language semantics of programming languages and domain-specific frameworks. We demonstrated its applicability to a range of domains and discussing research implications.

Motivated by the success of LLVM and looking ahead, we are eager to see how established communities in programming languages and high-performance computing, as well domain experts can benefit from the introduction of higher level, language-specific IRs. We also believe MLIR catalyzes new areas of research, as well as new approaches to teaching the art of compiler and IR design.

# Appendix
## A. Abstract
The artifact for this paper includes the MLIR system, instructions on how to download and build it and link to MLIR-related source code in TensorFlow.

## B. Artifact Check-List (Meta-Information)
Program: MLIR 
Compilation: LLVM  $\mathrm{C + + }$  toolchain 
Run-time environment: Recommended Linux 
Publicly available?: Yes 
Archived: DOI 10.5281/zenodo.4283090

## C. Description

1) How Delivered: To download MLIR please run git clone \ https://github.com/1lvm/1lvm-project.git

Instructions for downloading and building MLIR are also available at https://mlir.llvm.org/getting_started.

Additional information is available at mlir.llvm.org.

2) Software Dependencies: Downloading MLIR requires git. Building MLIR requires Ninja (https://ninja-build.org/) and a working  $\mathbf{C} + +$  toolchain including clang and lld.

## D. Installation
To build and test MLIR on Linux execute the following commands:

```
mkdir llvm-project/build 
cd llvm-project/build cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \ 
  -DLLVM_ENABLE_ASSERTIONS=ON \ 
  -DCMAKE_C_COMPILER=clang \ 
  -DCMAKE_CXX_COMPILER=clang++ \ 
  -DLLVM_ENABLE_LLD=ON
cmake --build . --target check-mlir
```

## E. Applications
MLIR use in TensorFlow can be observed in code located at https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir/. Tests located in the tensorflow/testes subdirectory contain MLIR snippets illustrating TensorFlow graph representation and transformations. Instructions for building TensorFlow from source are available at https://www.tensorflow.org/install/source.


