---
completed:
---
This document describes how to define dialect [attributes](https://mlir.llvm.org/docs/LangRef/#attributes) and [types](https://mlir.llvm.org/docs/LangRef/#type-system).

## LangRef Refresher 
Before diving into how to define these constructs, below is a quick refresher from the [MLIR LangRef](https://mlir.llvm.org/docs/LangRef/).

### Attributes 
Attributes are the mechanism for specifying constant data on operations in places where a variable is never allowed - e.g. the comparison predicate of a [`arith.cmpi` operation](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-arithcmpiop), or the underlying value of a [`arith.constant` operation](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithconstant-arithconstantop). 
>  Attributes 存储了 operation 中固定不变的，在操作语义中直接编码的常量数据
>  Attributes 不是数据流的一部分，不能被其他操作的输出所替代，也不能在运行时被修改
>  例如，比较操作 `arith.cmpi` 的比较谓词 (`eq, ne, slt (signed less then), ult (unsigned less then)`) 就是该操作的属性，比较的方式是操作语义的一部分，它们是固定的，不会随着输入而变化
>  又比如，`arith.constant` 具体的常量值也是属性

Each operation has an attribute dictionary, which associates a set of attribute names to attribute values.
>  每个操作数都有一个属性字典，字典中存储操作的属性名到属性值的映射

### Types 
Every SSA value, such as operation results or block arguments, in MLIR has a type defined by the type system. 
>  MLIR 中的所有 SSA value (例如 operation results, block arguments) 都有相关的，由类型系统定义的类型

MLIR has an open type system with no fixed list of types, and there are no restrictions on the abstractions they represent. 
>  MLIR 的类型系统是开放的，用户可以自行拓展，并且类型的抽象级别不受限制 (不局限于传统的数据类型，可以表示任何数据级别的抽象，例如寄存器等低级硬件类型、稀疏张量等高级数据结构、以及更高层次的语义类型)

For example, take the following [Arithmetic AddI operation](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddi-arithaddiop):

```mlir
  %result = arith.addi %lhs, %rhs : i64
```

It takes two input SSA values (`%lhs` and `%rhs`), and returns a single SSA value (`%result`). The inputs and outputs of this operation are of type `i64`, which is an instance of the [Builtin IntegerType](https://mlir.llvm.org/docs/Dialects/Builtin/#integertype).

>  例如，`arith.addi` 接收两个输入 SSA value `%lhs, %rhs`，返回单个 SSA value `%result`，其输入和输出类型都是 `i64`，为内建类型

## Attributes and Types 
The C++ Attribute and Type classes in MLIR (like Ops, and many other things) are value-typed. This means that instances of `Attribute` or `Type` are passed around by-value, as opposed to by-pointer or by-reference. 
>  MLIR 的源码中，`Attribute, Type` 类是值类型 (就像 `Op` 类)，即它们通常直接值传递，不需要通过指针或引用 (可能因为它们本身也就是对指针的包装类)

The `Attribute` and `Type` classes act as wrappers around internal storage objects that are uniqued within an instance of an `MLIRContext`.
>  `Attribute, Type` 实际是内部存储对象的包装器，并且它们是任何 `MLIRContext` 中唯一的 (也就是任何的属性、类型本身只会在内存中存在一个实例)

The structure for defining Attributes and Types is nearly identical, with only a few differences depending on the context. As such, a majority of this document describes the process for defining both Attributes and Types side-by-side with examples for both. If necessary, a section will explicitly call out any distinct differences.
>  定义 Attributes, Types 的结构几乎一致，差异很小

One difference is that generating C++ classes from declarative TableGen definitions will require adding additional targets to your `CMakeLists.txt`. This is not necessary for custom types. The details are outlined further below.
>  一个差异是要从声明式 TableGen 定义中生成 Attribute 的 C++ 类需要在 `CMakeLists.txt` 中添加 target，而自定义类型不需要

