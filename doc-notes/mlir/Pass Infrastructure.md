Passes represent the basic infrastructure for transformation and optimization. This document provides an overview of the pass infrastructure in MLIR and how to use it.
>  Passes 表示了转换和优化的基本基础结构

See [MLIR specification](https://mlir.llvm.org/docs/LangRef/) for more information about MLIR and its core aspects, such as the IR structure and operations.

See [MLIR Rewrites](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/) for a quick start on graph rewriting in MLIR. If a transformation involves pattern matching operation DAGs, this is a great place to start.

## Operation Pass 
In MLIR, the main unit of abstraction and transformation is an [operation](https://mlir.llvm.org/docs/LangRef/#operations). As such, the pass manager is designed to work on instances of operations at different levels of nesting. In the following paragraphs, we refer to the operation that a pass operates on as the “current operation”.
>  MLIR 中，转换和抽象的主要单元是操作
>  pass manager 被设计于在不同的嵌套层次上针对不同的操作实例工作
>  我们称 pass 目前处理的操作为 “当前操作”

The structure of the [pass manager](https://mlir.llvm.org/docs/PassManagement/#pass-manager), and the concept of nesting, is detailed further below. All passes in MLIR derive from `OperationPass` and adhere to the following restrictions; any noncompliance will lead to problematic behavior in multithreaded and other advanced scenarios:

- Must not inspect the state of operations that are siblings of the current operation. Must neither access operations nested under those siblings.
    - Other threads may be modifying these operations in parallel.
    - Inspecting the state of ancestor/parent operations is permitted.
- Must not modify the state of operations other than the operations that are nested under the current operation. This includes adding, modifying or removing other operations from an ancestor/parent block.
    - Other threads may be operating on these operations simultaneously.
    - As an exception, the attributes of the current operation may be modified freely. This is the only way that the current operation may be modified. (I.e., modifying operands, etc. is not allowed.)
- Must not maintain mutable pass state across invocations of `runOnOperation`. A pass may be run on many different operations with no guarantee of execution order.
    - When multithreading, a specific pass instance may not even execute on all operations within the IR. As such, a pass should not rely on running on all operations.
- Must not maintain any global mutable state, e.g. static variables within the source file. All mutable state should be maintained by an instance of the pass.
- Must be copy-constructible
    - Multiple instances of the pass may be created by the pass manager to process operations in parallel.


>  MLIR 中，所有的 passes 继承自 `OperationPass` ，并遵循约束：
>  - 不能审查当前操作的 sibling 操作的状态，不能访问嵌套在 sibling 操作下的操作，原因是其他线程可能会并行地修改这些操作。审查父操作/祖先操作的状态是允许的。
>  - 除了嵌套在当前操作的操作外，其他操作的状态都不能修改，也不能从一个祖先块/父块中添加、修改、删除其他操作。当前操作的属性可以修改，其他的，例如操作数，不能修改。
>  - 不能在 `runOnOperation` 的调用中维护可变的 pass 状态，因为一个 pass 可能会在多个操作上执行，且执行顺序不保证。在多线程执行下，一个 pass 实例 (由一个线程执行) 可能不会在 IR 中的所有操作上都执行，因此 pass 不能依赖于它会运行于所有操作的假设。
>  - 不能维护任意全局可变状态，例如源文件中的静态变量，所有的可变状态应该都由一个 pass 实例维护
>  - 必须是可以拷贝构造的，因为 pass manager 会创建多个 pass 以并行处理操作。
