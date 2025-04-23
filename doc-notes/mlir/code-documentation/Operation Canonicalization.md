---
completed: true
---
Canonicalization is an important part of compiler IR design: it makes it easier to implement reliable compiler transformations and to reason about what is better or worse in the code, and it forces interesting discussions about the goals of a particular level of IR. Dan Gohman wrote [an article](https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html) exploring these issues; it is worth reading if you’re not familiar with these concepts.

Most compilers have canonicalization passes, and sometimes they have many different ones (e.g. instcombine, dag combine, etc in LLVM). Because MLIR is a multi-level IR, we can provide a single canonicalization infrastructure and reuse it across many different IRs that it represents. 
>  大多数编译器会有规范化 passes，有时它们会有多个不同的规范化 passes，例如 LLVM 中的 instcombine, dag combine 等 pass
>  因为 MLIR 是多层 IR，我们可以提供单个规范化基础设施，在多个不同 IR 中复用它

This document describes the general approach, global canonicalizations performed, and provides sections to capture IR-specific rules for reference.

## General Design
MLIR has a single canonicalization pass, which iteratively applies the canonicalization patterns of all loaded dialects in a greedy way. Canonicalization is best-effort and not guaranteed to bring the entire IR in a canonical form. It applies patterns until either fixpoint is reached or the maximum number of iterations/rewrites (as specified via pass options) is exhausted. This is for efficiency reasons and to ensure that faulty patterns cannot cause infinite looping.
>  MLIR 有一个规范化 pass，该 pass 以贪心地方式迭代应用所有已加载方言的规范化模式
>  该规范化 pass 是尽力而为的，不能保证将整个 IR 转化为规范形式，它会应用模式直到到达不动点 (再次应用规范化也得到相同的结果) 或者达到最大数量的迭代或重写次数 (通过 pass 选项指定)
>  尽力而为的设计是处于效率考虑，并确保有缺陷的模式不会导致无限循环

Canonicalization patterns are registered with the operations themselves, which allows each dialect to define its own set of operations and canonicalizations together.
>  规范化模式与操作本身进行注册，使得每种方言可以一起定义自己的操作集和规范化集

Some important things to think about w.r.t. canonicalization patterns:

- The goal of canonicalization is to make subsequent analyses and optimizations more effective. Therefore, performance improvements are not necessary for canonicalization.
- Pass pipelines should not rely on the canonicalizer pass for correctness. They should work correctly with all instances of the canonicalization pass removed.
- Repeated applications of patterns should converge. Unstable or cyclic rewrites are considered a bug: they can make the canonicalizer pass less predictable and less effective (i.e., some patterns may not be applied) and prevent it from converging.
- It is generally better to canonicalize towards operations that have fewer uses of a value when the operands are duplicated, because some patterns only match when a value has a single user. For example, it is generally good to canonicalize “x + x” into “x * 2”, because this reduces the number of uses of x by one.
- It is always good to eliminate operations entirely when possible, e.g. by folding known identities (like “x + 0 = x”).
- Pattens with expensive running time (i.e. have O(n) complexity) or complicated cost models don’t belong to canonicalization: since the algorithm is executed iteratively until fixed-point we want patterns that execute quickly (in particular their matching phase).
- Canonicalize shouldn’t lose the semantic of original operation: the original information should always be recoverable from the transformed IR.

>  定义规范化模式，需要考虑的事情包括
>  - 规范化的目标是让后续的分析和优化更高效，故规范化并不必要执行性能提升
>  - pass pipeline 的正确性不应该依赖于规范化 pass，应该在规范化 pass 的所有实例移除后，也可以正确运行
>  - 模式被重复应用后应该能够收敛，不稳定的或循环的重写都是 bug: 它们会使得规范化 pass 更不可预测和更低效 (即某些模式可能不会被应用)，并且阻碍收敛
>  - 当操作数有重复时，通常应将其朝着具有更少的 value uses 的方向规范化，因为一些模式仅在 value 仅有单个 user 时匹配。例如通常应将 `x + x` 规范化为 `x * 2`，因为这会减少 `x` 的一次使用
>  - 可能的话，最好完全消除某个操作，例如折叠已知的恒等式 (e.g. `x + 0 = x` )
>  - 运行时间长 (i.e. 具有 O(n) 复杂度) 或开销模型复杂的模式不应属于规范化模式: 因为规范化算法会迭代执行各个模式直到固定点，故我们希望模式可以被快速执行 (特别是匹配阶段能快速执行)
>  - 规范化不应该丢失原始操作的语义: 原始信息应该可以从转换后的 IR 恢复

For example, a pattern that transform

```
  %transpose = linalg.transpose
      ins(%input : tensor<1x2x3xf32>)
      outs(%init1 : tensor<2x1x3xf32>)
      dimensions = [1, 0, 2]
  %out = linalg.transpose
      ins(%transpose: tensor<2x1x3xf32>)
      outs(%init2 : tensor<3x1x2xf32>)
      permutation = [2, 1, 0]
```

to

```
  %out= linalg.transpose
      ins(%input : tensor<1x2x3xf32>)
      outs(%init2: tensor<3x1x2xf32>)
      permutation = [2, 0, 1]
```

is a good canonicalization pattern because it removes a redundant operation, making other analysis optimizations and more efficient.

## Globally Applied Rules 
These transformations are applied to all levels of IR:

- Elimination of operations that have no side effects and have no uses.
- Constant folding - e.g. “(addi 1, 2)” to “3”. Constant folding hooks are specified by operations.
- Move constant operands to commutative operators to the right side - e.g. “(addi 4, x)” to “(addi x, 4)”.
- `constant-like` operations are uniqued and hoisted into the entry block of the first parent barrier region. This is a region that is either isolated from above, e.g. the entry block of a function, or one marked as a barrier via the `shouldMaterializeInto` method on the `DialectFoldInterface`.

>  以下的转换被应用于所有层级的 IR
>  - 消除没有 side effects 和 uses 的操作
>  - 常量折叠 (例如 `addi 1, 2` 折叠为 `3`)，常量折叠钩子由操作指定
>  - 将可交换操作的常量操作数移动到右边 (例如 `addi 4, x` 转为 `addi x, 4`)
>  - 类常量的操作会被唯一化，并提升到第一个父级屏障区域的入口块处，所谓父级屏障区域指的是一个隔离的区域，它要么和之前隔离 (例如一个函数的入口快)，要么通过 `DialectFloadInterface` 的 `shouldMaterializeInto` 方法被标记为屏障

## Defining Canonicalizations 
Two mechanisms are available with which to define canonicalizations; general `RewritePattern` s and the `fold` method.
>  定义规范化有两种方法: `RewritePattern` , `fold` 

### Canonicalizing with `RewritePattern` s 
This mechanism allows for providing canonicalizations as a set of `RewritePattern` s, either imperatively defined in C++ or declaratively as [Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/). The pattern rewrite infrastructure allows for expressing many different types of canonicalizations. These transformations may be as simple as replacing a multiplication with a shift, or even replacing a conditional branch with an unconditional one.
>  第一种方法是通过一组 `RewritePattern` 提供规范化支持
>  `RewritePattern` 可以在 C++ 中命令式地定义，也可以用声明式重写规则声明式地定义
>  模式重写基础设置允许表达多种不同类型的规范化操作，这些转换可以是简单地将乘法替换为移位，也可以是用条件分支替换一个无条件分支

In [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/), an operation can set the `hasCanonicalizer` bit or the `hasCanonicalizeMethod` bit to generate a declaration for the `getCanonicalizationPatterns` method:

```tablegen
def MyOp : ... {
  // I want to define a fully general set of patterns for this op.
  let hasCanonicalizer = 1;
}

def OtherOp : ... {
  // A single "matchAndRewrite" style RewritePattern implemented as a method
  // is good enough for me.
  let hasCanonicalizeMethod = 1;
}
```

Canonicalization patterns can then be provided in the source file:

```c++
void MyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<...>(...);
}

LogicalResult OtherOp::canonicalize(OtherOp op, PatternRewriter &rewriter) {
  // patterns and rewrites go here.
  return failure();
}
```

See the [quickstart guide](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/) for information on defining operation rewrites.

### Canonicalizing with the `fold` method 
The `fold` mechanism is an intentionally limited, but powerful mechanism that allows for applying canonicalizations in many places throughout the compiler. For example, outside of the canonicalizer pass, `fold` is used within the [dialect conversion infrastructure](https://mlir.llvm.org/docs/DialectConversion/) as a legalization mechanism, and can be invoked directly anywhere with an `OpBuilder` via `OpBuilder::createOrFold`.
>  第二种方法是通过 `fold` 提供规范化支持
>  `fold` 机制是一个有意限制但功能强大的机制，允许在编译器的多个阶段应用规范化
>  例如，在规范化 pass 之外，`fold` 也可以在方言转化基础设施中用于合法化机制，并且可以在任何地方通过 `OpBuilder::createOrFold` 调用

`fold` has the restriction that no new operations may be created, and only the root operation may be replaced (but not erased). It allows for updating an operation in-place, or returning a set of pre-existing values (or attributes) to replace the operation with. This ensures that the `fold` method is a truly “local” transformation, and can be invoked without the need for a pattern rewriter.
>  `fold` 的限制是不能创建新的操作，并且只能替换根操作
>  `fold` 允许原地更新一个操作，或者返回一组预先存在的值 (或属性) 来替换该操作的值 (或属性)
>  这确保了 `fold` 方法是完全的 “局部” 转换，可以在不需要模式重写器的情况下被调用

In [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/), an operation can set the `hasFolder` bit to generate a declaration for the `fold` method. This method takes on a different form, depending on the structure of the operation.

```tablegen
def MyOp : ... {
  let hasFolder = 1;
}
```

If the operation has a single result the following will be generated:

```c++
/// Implementations of this hook can only perform the following changes to the
/// operation:
///
///  1. They can leave the operation alone and without changing the IR, and
///     return nullptr.
///  2. They can mutate the operation in place, without changing anything else
///     in the IR. In this case, return the operation itself.
///  3. They can return an existing value or attribute that can be used instead
///     of the operation. The caller will remove the operation and use that
///     result instead.
///
OpFoldResult MyOp::fold(FoldAdaptor adaptor) {
  ...
}
```

Otherwise, the following is generated:

```c++
/// Implementations of this hook can only perform the following changes to the
/// operation:
///
///  1. They can leave the operation alone and without changing the IR, and
///     return failure.
///  2. They can mutate the operation in place, without changing anything else
///     in the IR. In this case, return success.
///  3. They can return a list of existing values or attribute that can be used
///     instead of the operation. In this case, fill in the results list and
///     return success. The results list must correspond 1-1 with the results of
///     the operation, partial folding is not supported. The caller will remove
///     the operation and use those results instead.
///
/// Note that this mechanism cannot be used to remove 0-result operations.
LogicalResult MyOp::fold(FoldAdaptor adaptor,
                         SmallVectorImpl<OpFoldResult> &results) {
  ...
}
```

In the above, for each method a `FoldAdaptor` is provided with getters for each of the operands, returning the corresponding constant attribute. These operands are those that implement the `ConstantLike` trait. If any of the operands are non-constant, a null `Attribute` value is provided instead. For example, if MyOp provides three operands [`a`, `b`, `c`], but only `b` is constant then `adaptor` will return Attribute() for `getA()` and `getC()`, and b-value for `getB()`.
>  上例中，`fold` 方法都会接收一个 `FoldAdaptor` ，`FoldAdaptor` 为每个操作数提供了 getter 方法，对于实现了 `ConstantLike` 特性的操作数，getter 会返回对应的常量属性，如果操作数非常量，则会返回一个空 `Attribute` 值
>  例如，如果 `MyOp` 提供三个操作数 `a, b, c` ，其中仅 `b` 为常量，则 `adaptor.getA(), adaptor.getB(), adaptor.getC()` 将分别返回 `Attribute(), b, Attribute()`

Also above, is the use of `OpFoldResult`. This class represents the possible result of folding an operation result: either an SSA `Value`, or an `Attribute` (for a constant result). If an SSA `Value` is provided, it _must_ correspond to an existing value. The `fold` methods are not permitted to generate new `Value`s. There are no specific restrictions on the form of the `Attribute` value returned, but it is important to ensure that the `Attribute` representation of a specific `Type` is consistent.
>  上例中，第二个 `fold` 涉及了 `OpFoldResult` ，该类表示折叠一个操作结果所可能得到的结果：要么是一个 SSA `Value` ，要么是一个 `Attribute` (对于常量结果)
>  如果返回 SSA `Value` ，该 `Value` 一定要对应于一个现存的 `Value` ，因为 `fold` 方法不允许生成新 `Value`
>  `fold` 返回的 `Attribute` 值的形式没有具体限制，但重要的是确保特定 `Type` 的 `Attribute` 表示具有一致性

When the `fold` hook on an operation is not successful, the dialect can provide a fallback by implementing the `DialectFoldInterface` and overriding the fold hook.
>  方言可以实现 `DialectFoldInterface` ，以在操作的 `fold` 钩子失败时作为回退方案

#### Generating Constants from Attributes
When a `fold` method returns an `Attribute` as the result, it signifies that this result is “constant”. The `Attribute` is the constant representation of the value. Users of the `fold` method, such as the canonicalizer pass, will take these `Attribute` s and materialize constant operations in the IR to represent them. To enable this materialization, the dialect of the operation must implement the `materializeConstant` hook. This hook takes in an `Attribute` value, generally returned by `fold`, and produces a “constant-like” operation that materializes that value.
>  当 `fold` 方法返回一个 `Attribute` 时，这表明了该结果是 “常量”，该 `Attribute` 是该值的常量表示形式
>  `fold` 方法的使用者，例如规范化 pass，需要接收这些 `Attribute` ，并在 IR 中生成常量操作以表示它们
>  要让规范化 pass 能够执行这样的生成，对应操作的方言需要实现 `materializeConstant` 方法，该方法接收一个 `Attribute` 值 (通常由 `fold` 返回)，然后生成一个 “类常量” 的操作来具体化该值

In [ODS](https://mlir.llvm.org/docs/DefiningDialects/), a dialect can set the `hasConstantMaterializer` bit to generate a declaration for the `materializeConstant` method.

```tablegen
def MyDialect : ... {
  let hasConstantMaterializer = 1;
}
```

Constants can then be materialized in the source file:

```c++
/// Hook to materialize a single constant operation from a given attribute value
/// with the desired resultant type. This method should use the provided builder
/// to create the operation without changing the insertion position. The
/// generated operation is expected to be constant-like. On success, this hook
/// should return the value generated to represent the constant value.
/// Otherwise, it should return nullptr on failure.
Operation *MyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  ...
}
```

### When to use the `fold` method vs `RewriterPattern` s for canonicalizations 
A canonicalization should always be implemented as a `fold` method if it can be, otherwise it should be implemented as a `RewritePattern`.
>  尽可能将规范化实现为 `fold` 方法，其次的选择是实现为 `RewritePattern`

