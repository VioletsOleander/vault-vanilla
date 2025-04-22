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

### Op-Agnostic Operation Passes 
By default, an operation pass is `op-agnostic`, meaning that it operates on the operation type of the pass manager that it is added to. This means a pass may operate on many different types of operations. Agnostic passes should be written such that they do not make assumptions on the operation they run on. 
>  operation pass 默认是操作无关的，这意味着该 pass 针对它所添加到的 pass manager 的 operation 类型进行处理，即该 pass 可以为许多不同类型的操作进行处理
>  编写操作无关的 pass 时，需要确保它们不对它们所处理的操作做出假设

Examples of this type of pass are [Canonicalization](https://mlir.llvm.org/docs/Passes/#-canonicalize) and [Common Sub-Expression Elimination](https://mlir.llvm.org/docs/Passes/#-cse).
>  操作无关的 pass 的例子有规范化和公共子表达式消除  

To create an agnostic operation pass, a derived class must adhere to the following:

- Inherit from the CRTP class `OperationPass`.
- Override the virtual `void runOnOperation()` method.

>  要创建一个操作无关的 pass，我们需要编写一个继承类，它需要
>  - 继承自 CRTP 类 `OperationPass`
>  - 覆盖虚方法 `void runOnOperation()`

A simple pass may look like:

```c++
/// Here we utilize the CRTP `PassWrapper` utility class to provide some
/// necessary utility hooks. This is only necessary for passes defined directly
/// in C++. Passes defined declaratively use a cleaner mechanism for providing
/// these utilities.
struct MyOperationPass : public PassWrapper<MyOperationPass, OperationPass<>> {
  void runOnOperation() override {
    // Get the current operation being operated on.
    Operation *op = getOperation();
    ...
  }
};
```

### Filtered Operation Pass 
If a pass needs to constrain its execution to specific types or classes of operations, additional filtering may be applied on top. This transforms a once `agnostic` pass into one more specific to a certain context. 
>  如果 pass 需要限制其执行仅针对特定类别的操作，可以在其上应用额外的过滤，这会将操作无关的 pass 转化为更针对特定上下文的 pass

There are various ways in which to filter the execution of a pass, and different contexts in which filtering may apply:

### Dependent Dialects 
Dialects must be loaded in the MLIRContext before entities from these dialects (operations, types, attributes, …) can be created. Dialects must also be loaded before starting the execution of a multi-threaded pass pipeline. 
> 在创建 Dialect 中的实体 (操作、类型、属性等) 之前，Dialect 必须被加载到 `MLIRContext`
   在启动多线程 pass pipeline 之前，Dialect 也必须被加载

To this end, a pass that may create an entity from a dialect that isn’t guaranteed to already be loaded must express this by overriding the `getDependentDialects()` method and declare this list of Dialects explicitly. See also the `dependentDialects` field in the [TableGen Specification](https://mlir.llvm.org/docs/PassManagement/#tablegen-specification).

> 因此，如果一个 pass 可能会从不保证已经被加载的 Dialect 中创建实体时，必须覆盖 `getDependentDialect()` 方法，并显式声明对应的 Dialect 列表

## Pass Manager
The above sections introduced the different types of passes and their invariants. This section introduces the concept of a PassManager, and how it can be used to configure and schedule a pass pipeline. 

There are two main classes related to pass management, the `PassManager` and the `OpPassManager`. The `PassManager` class acts as the top-level entry point, and contains various configurations used for the entire pass pipeline. The `OpPassManager` class is used to schedule passes to run at a specific level of nesting. The top-level `PassManager` also functions as an `OpPassManager`.
>  pass 管理主要和两个类相关：`PassManager, OpPassManager`
>  `PassManager` 类作为顶级入口点，包含了用于整个 pass pipeline 的多种配置
>  `OpPassManager` 类用于调度 pass 在特定嵌套级别运行
>  顶级 `PassManager` 继承自 `OpPassManager` ，因此也充当 `OpPassManager` 的功能

### OpPassManager 
An `OpPassManager` is essentially a collection of passes anchored to execute on operations at a given level of nesting. A pass manager may be `op-specific` (anchored on a specific operation type), or `op-agnostic` (not restricted to any specific operation, and executed on any viable operation type). 
>  `OpPassManager` 本质上是一组针对特定嵌套层级的操作所执行的 passes 的集合
>  pass manager 可以是操作相关的 (绑定到特定的操作类型)，也可以是操作无关的 (不限制具体操作，在任何可行的操作类型上执行)

Operation types that anchor pass managers must adhere to the following requirement:

- Must be registered and marked [`IsolatedFromAbove`](https://mlir.llvm.org/docs/Traits/#isolatedfromabove).
    
    - Passes are expected not to modify operations at or above the current operation being processed. If the operation is not isolated, it may inadvertently modify or traverse the SSA use-list of an operation it is not supposed to.

>  绑定 pass manager 的操作类型必须符合以下要求
>  - 必须被注册，并且标记为 `IsolatedFromAbove`
>     pass 不应该对与当前处理的操作处于同一层次或之上的操作进行修改，故如果操作没有被标记 `IsolatedFromAbove`，处理该操作的 pass 就有可能无意中修改或遍历它不应访问的操作的 SSA 使用列表

Passes can be added to a pass manager via `addPass`.
>  pass manager 通过 `addPass` 添加 passes

An `OpPassManager` is generally created by explicitly nesting a pipeline within another existing `OpPassManager` via the `nest<OpT>` or `nestAny` methods. The former method takes the operation type that the nested pass manager will operate on. The latter method nests an `op-agnostic` pass manager, that may run on any viable operation type. 
>  一个 `OpPassManager` 通常会通过在另一个现存的 `OpPassManager` 中使用 `nest<OpT>` 或 `nestAny` 方法显式嵌套一个 pipeline 来创建
>  `nest<OpT>` 方法接收嵌套的 pass manager 将处理的操作类型
>  `nestAny` 方法则是操作无关的，可以在任意可行的操作上执行

Nesting in this sense, corresponds to the [structural](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/) nesting within [Regions](https://mlir.llvm.org/docs/LangRef/#regions) of the IR.

For example, the following `.mlir`:

```mlir
module {
  spirv.module "Logical" "GLSL450" {
    func @foo() {
      ...
    }
  }
}
```

Has the nesting structure of:

```
`builtin.module`
  `spirv.module`
    `spirv.func`
```

Below is an example of constructing a pipeline that operates on the above structure:

```c++
// Create a top-level `PassManager` class.
auto pm = PassManager::on<ModuleOp>(ctx);

// Add a pass on the top-level module operation.
pm.addPass(std::make_unique<MyModulePass>());

// Nest a pass manager that operates on `spirv.module` operations nested
// directly under the top-level module.
OpPassManager &nestedModulePM = pm.nest<spirv::ModuleOp>();
nestedModulePM.addPass(std::make_unique<MySPIRVModulePass>());

// Nest a pass manager that operates on functions within the nested SPIRV
// module.
OpPassManager &nestedFunctionPM = nestedModulePM.nest<func::FuncOp>();
nestedFunctionPM.addPass(std::make_unique<MyFunctionPass>());

// Nest an op-agnostic pass manager. This will operate on any viable
// operation, e.g. func.func, spirv.func, spirv.module, builtin.module, etc.
OpPassManager &nestedAnyPM = nestedModulePM.nestAny();
nestedAnyPM.addPass(createCanonicalizePass());
nestedAnyPM.addPass(createCSEPass());

// Run the pass manager on the top-level module.
ModuleOp m = ...;
if (failed(pm.run(m)))
    ... // One of the passes signaled a failure.
```

The above pass manager contains the following pipeline structure:

```
OpPassManager<ModuleOp>
  MyModulePass
  OpPassManager<spirv::ModuleOp>
    MySPIRVModulePass
    OpPassManager<func::FuncOp>
      MyFunctionPass
    OpPassManager<>
      Canonicalizer
      CSE
```

These pipelines are then run over a single operation at a time. This means that, for example, given a series of consecutive passes on `func::FuncOp`, it will execute all on the first function, then all on the second function, etc. until the entire program has been run through the passes. 
>  以此定义的这些嵌套的 pipelines 将针对逐个操作执行
>  例如，给定针对 `func::FuncOP` 的一系列连续的 passes，这些 passes 会先对第一个函数执行所有 passes，然后对第二个函数执行所有 passes，以此类推

This provides several benefits:

- This improves the cache behavior of the compiler, because it is only touching a single function at a time, instead of traversing the entire program.
- This improves multi-threading performance by reducing the number of jobs that need to be scheduled, as well as increasing the efficiency of each job. An entire function pipeline can be run on each function asynchronously.

>  这样的好处在于
>  - 优化了编译器的缓存行为，因为一次仅处理一个函数，而不是遍历整个程序
>  - 优化了多线程性能，因为减少了需要调度的任务数量，且提高了每个任务的效率，每个函数可以异步地运行整个 pipeline

## Pass Instrumentation
MLIR provides a customizable framework to instrument pass execution and analysis computation, via the `PassInstrumentation` class. 
>  MLIR 的 `PassInstrumentation` 类提供了一个可定制的框架，用于记录 pass 的执行和分析计算过程

This class provides hooks into the PassManager that observe various events:

- `runBeforePipeline`
    - This callback is run just before a pass pipeline, i.e. pass manager, is executed.
- `runAfterPipeline`
    - This callback is run right after a pass pipeline has been executed, successfully or not.
- `runBeforePass`
    - This callback is run just before a pass is executed.
- `runAfterPass`
    - This callback is run right after a pass has been successfully executed. If this hook is executed, `runAfterPassFailed` will _not_ be.
- `runAfterPassFailed`
    - This callback is run right after a pass execution fails. If this hook is executed, `runAfterPass` will _not_ be.
- `runBeforeAnalysis`
    - This callback is run just before an analysis is computed.
    - If the analysis requested another analysis as a dependency, the `runBeforeAnalysis`/`runAfterAnalysis` pair for the dependency can be called from inside of the current `runBeforeAnalysis`/`runAfterAnalysis` pair.
- `runAfterAnalysis`
    - This callback is run right after an analysis is computed.


>  `PassInstrumentation` 类提供了对 `PassManager` 的各种钩子，用于观察各种事件，这些钩子包括：
>  - `runBeforePipeline`: 在执行 pass pipeline (pass manager) 之前运行
>  - `runAftrePipeline`: 在 pass pipeline 执行之后运行，无论执行是否成功
>  - `runBeforePass`: 在一个 pass 执行前运行
>  - `runAfterPass`: 在一个 pass 成功执行后运行
>  - `runAfterPassFailed`: 在一个 pass 执行失败后运行
>  - `runBeforeAnalysis`: 在计算分析前运行，如果当前分析依赖另一个分析，则另一个分析的 `runBeforeAnalysis/runAfterAnalysis` 会在当前分析的 `runBeforeAnalysis/runAfterAnalysis` 中被调用
>  - `runAfterAnalysis`: 在计算分析后运行

PassInstrumentation instances may be registered directly with a [PassManager](https://mlir.llvm.org/docs/PassManagement/#pass-manager) instance via the `addInstrumentation` method. Instrumentations added to the PassManager are run in a stack like fashion, i.e. the last instrumentation to execute a `runBefore*` hook will be the first to execute the respective `runAfter*` hook. 
>  `PassInstrumentation` 实例可以通过 `PassManager` 的 `addInstrumentation` 方法直接注册到 `PassManager` 中
>  添加到 `PassManager` 的 Instrumentations 会以栈的方式运行，即最后一个执行 `runBefore*` 钩子的 instrumentation 将是第一个执行 `runAfter` 钩子的 instrumentation (即 Instrumentation 执行完 `runBefore*` 之后入栈，故执行最后一个 `runBefroe*` 的 Intrumentation 将在栈顶)

The hooks of a `PassInstrumentation` class are guaranteed to be executed in a thread-safe fashion, so additional synchronization is not necessary. 
>  一个 `PassInstrumentation` 类的 hooks 保证以线程安全的方式执行，因此不需要额外的同步

Below in an example instrumentation that counts the number of times the `DominanceInfo` analysis is computed:

```c++
struct DominanceCounterInstrumentation : public PassInstrumentation {
  /// The cumulative count of how many times dominance has been calculated.
  unsigned &count;

  DominanceCounterInstrumentation(unsigned &count) : count(count) {}
  void runAfterAnalysis(llvm::StringRef, TypeID id, Operation *) override {
    if (id == TypeID::get<DominanceInfo>())
      ++count;
  }
};

MLIRContext *ctx = ...;
PassManager pm(ctx);

// Add the instrumentation to the pass manager.
unsigned domInfoCount;
pm.addInstrumentation(
    std::make_unique<DominanceCounterInstrumentation>(domInfoCount));

// Run the pass manager on a module operation.
ModuleOp m = ...;
if (failed(pm.run(m)))
    ...

llvm::errs() << "DominanceInfo was computed " << domInfoCount << " times!\n";
```