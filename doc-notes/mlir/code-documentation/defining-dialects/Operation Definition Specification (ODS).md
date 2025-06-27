In addition to specializing the `mlir::Op` C++ template, MLIR also supports defining operations and data types in a table-driven manner. This is achieved via [TableGen](https://llvm.org/docs/TableGen/index.html), which is both a generic language and its tooling to maintain records of domain-specific information. Facts regarding an operation are specified concisely into a TableGen record, which will be expanded into an equivalent `mlir::Op` C++ template specialization at compiler build time.
>  MLIR 支持在 C++ 中，定义 `mlir::Op` 的衍生类来定义 operation，也支持在 TableGen 中定义 operation
>  TableGen 中 operation 的定义在一个 record 中指定，该 record 会在编译时被生成为继承了 `mlir::Op` 的 C++ 类

This manual explains in detail all the available mechanisms for defining operations in such a table-driven manner. It aims to be a specification instead of a tutorial. Please refer to [Quickstart tutorial to adding MLIR graph rewrite](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/) for the latter.

In addition to detailing each mechanism, this manual also tries to capture best practices. They are rendered as quoted bullet points.

## Motivation 
MLIR allows pluggable dialects, and dialects contain, among others, a list of operations. This open and extensible ecosystem leads to the “stringly” type IR problem, e.g., repetitive string comparisons during optimization and analysis passes, unintuitive accessor methods (e.g., generic/error prone `getOperand(3)` vs self-documenting `getStride()`) with more generic return types, verbose and generic constructors without default arguments, verbose textual IR dumps, and so on.
>  MLIR 的开放性 (允许自定义方言、操作等) 导致了字符串化 IR 的问题，即实际的 C++ 开发中:
>  - 大量依赖字符串比较来识别 IR 元素、执行优化和分析 passes
>  - 会有不直观的访问方法
>  - 由于不知道 operation 的具体类型，返回值会很通用 (例如 `Value, Attribute`) 
>  - 由于构造函数没有默认值，故会很冗长和通用
>  - 由于文本 IR 格式需要表示所有底层细节，故文本 IR 会很冗长

 Furthermore, operation verification is:

1. best case: a central string-to-verification-function map,
2. middle case: duplication of verification across the code base, or
3. worst case: no verification functions.

>  此外，编写 operation 验证逻辑时
>  - 最好的情况: 维护一个中央映射表，将 operation 的名称映射到其 C++ 验证函数
>  - 中间情况: 验证逻辑分散在代码库的不同地方
>  - 最坏情况: 没有验证函数

The fix is to support defining ops in a table-driven manner. Then for each dialect, we can have a central place that contains everything you need to know about each op, including its constraints, custom assembly form, etc. This description is also used to generate helper functions and classes to allow building, verification, parsing, printing, analysis, and many more.
>  为此，MLIR 建议使用表驱动的定义方式，使用 TableGen 的声明式语法来描述 operation
>  对于每个方言，其所有 operation 的定义都集中在单个 `.td` 文件中，这个文件中包含了理解一个 operation 所有的信息来源，包括: 约束、自定义汇编格式、特性等
>  `.td` 文件中的定义可以被自动生成为构建、验证、解析、打印、分析等相关的辅助函数和类，这样声明都是集中的，且定义不需要自己编写

## Benefits 
Compared to the C++ template, this table-driven approach has several benefits including but not limited to:

- **Single source of truth**: We strive to encode all facts regarding an operation into the record, so that readers don’t need to jump among code snippets to fully understand an operation.
- **Removing boilerplate**: We can automatically generate operand/attribute/result getter methods, operation build methods, operation verify methods, and many more utilities from the record. This greatly reduces the boilerplate needed for defining a new op.
- **Facilitating auto-generation**: The usage of these operation information records are by no means limited to op definition itself. We can use them to drive the auto-generation of many other components, like computation graph serialization.

## TableGen Syntax 
We use TableGen as the language for specifying operation information. TableGen itself just provides syntax for writing records; the syntax and constructs allowed in a TableGen file (typically with the filename suffix `.td`) can be found [here](https://llvm.org/docs/TableGen/ProgRef.html).

- TableGen `class` is similar to C++ class; it can be templated and subclassed.
- TableGen `def` is similar to C++ object; it can be declared by specializing a TableGen `class` (e.g., `def MyDef : MyClass<...>;`) or completely independently (e.g., `def MyDef;`). It cannot be further templated or subclassed.
- TableGen `dag` is a dedicated type for directed acyclic graph of elements. A `dag` has one operator and zero or more arguments. Its syntax is `(operator arg0, arg1, argN)`. The operator can be any TableGen `def`; an argument can be anything, including `dag` itself. We can have names attached to both the operator and the arguments like `(MyOp:$op_name MyArg:$arg_name)`.

Please see the [language reference](https://llvm.org/docs/TableGen/ProgRef.html) to learn about all the types and expressions supported by TableGen.