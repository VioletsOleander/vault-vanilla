# 1 Toy Language and AST

## The Language 
This tutorial will be illustrated with a toy language that we’ll call “Toy” (naming is hard…). Toy is a tensor-based language that allows you to define functions, perform some math computation, and print results.
>  本教程定义了 Toy 语言
>  Toy 是基于 tensor 的语言，允许定义函数、执行数学计算、打印结果

Given that we want to keep things simple, the codegen will be limited to tensors of rank <= 2, and the only datatype in Toy is a 64-bit floating point type (aka ‘double’ in C parlance). As such, all values are implicitly double precision, `Values` are immutable (i.e. every operation returns a newly allocated value), and deallocation is automatically managed. But enough with the long description; nothing is better than walking through an example to get a better understanding:
>  codegen 的 tensor 的 rank 将不大于 2
>  Toy 仅支持 64 位浮点类型 (双精度)
>  `Values` 不可变，每次运算都返回新分配的值
>  存储释放自动执行

```toy
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new variables is the way to reshape tensors (element count must match).
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # transpose() and print() are the only builtin, the following will transpose
  # a and b and perform an element-wise multiplication before printing the result.
  print(transpose(a) * transpose(b));
}
```

>  该例通过 `var a = [[1,2,3], [4,5,6]]` 定义了张量 `a` ，张量的形状被自动推导
>  `b` 的定义 `var b<2,3> = [1,2,3,4,5,6]` 也是合法的，在形状给定的情况下，张量会被自动 reshape
>   `transpose(), print()` 为内建函数
>   `*` 执行 tensor 之间的逐元素乘法

Type checking is statically performed through type inference; the language only requires type declarations to specify tensor shapes when needed. Functions are generic: their parameters are unranked (in other words, we know these are tensors, but we don’t know their dimensions). They are specialized for every newly discovered signature at call sites. Let’s revisit the previous example by adding a user-defined function:
>  类型检查通过类型推导静态执行
>  Toy 语言仅要求必要的时候在类型声明中指定张量形状
>  函数为泛型，其参数是 unranked，即我们知道它们是张量，但不知道其维度
>  函数参数的具体维度在调用点具体指定

```toy
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);

  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return <3, 2>.
  var d = multiply_transpose(b, a);

  # A new call with <3, 2> (instead of <2, 3>) for both dimensions will trigger another specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);

  # Finally, calling into `multiply_transpose` with incompatible shapes (<2, 3> and <3, 2>) will trigger a shape inference error.
  var f = multiply_transpose(a, c);
}
```

>  本例中
>  函数 `def multiply_transpose(a, b)` 定义了泛型函数，实际参数的形状不具体指定
>  `var a, var b` 用之前提到的两种方式定义
>  调用 `var c = mutiply_transpose(a ,b)` 时，函数的两个参数形状根据传入参数指定为 `<2,3>`，`var c` 的形状被自动推导
>  调用 `var e = multiply_transpose(c, d)`，将函数的参数形状重新指定为 `<3, 2>`
>  调用 `var f = multiply_transpose(a ,c)` 时传入了形状不兼容的两个 tensor，会触发形状推理错误

## The AST 
The AST from the above code is fairly straightforward; here is a dump of it:

```
Module:
  Function 
    Proto 'multiply_transpose' @test/Examples/Toy/Ch1/ast.toy:4:1
    Params: [a, b]
    Block {
      Return
        BinOp: * @test/Examples/Toy/Ch1/ast.toy:5:25
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:10
            var: a @test/Examples/Toy/Ch1/ast.toy:5:20
          ]
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:25
            var: b @test/Examples/Toy/Ch1/ast.toy:5:35
          ]
    } // Block
  Function 
    Proto 'main' @test/Examples/Toy/Ch1/ast.toy:8:1
    Params: []
    Block {
      VarDecl a<> @test/Examples/Toy/Ch1/ast.toy:11:3
        Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/Examples/Toy/Ch1/ast.toy:11:11
      VarDecl b<2, 3> @test/Examples/Toy/Ch1/ast.toy:15:3
        Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/Examples/Toy/Ch1/ast.toy:15:17
      VarDecl c<> @test/Examples/Toy/Ch1/ast.toy:19:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:19:11
          var: a @test/Examples/Toy/Ch1/ast.toy:19:30
          var: b @test/Examples/Toy/Ch1/ast.toy:19:33
        ]
      VarDecl d<> @test/Examples/Toy/Ch1/ast.toy:22:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:22:11
          var: b @test/Examples/Toy/Ch1/ast.toy:22:30
          var: a @test/Examples/Toy/Ch1/ast.toy:22:33
        ]
      VarDecl e<> @test/Examples/Toy/Ch1/ast.toy:25:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:25:11
          var: c @test/Examples/Toy/Ch1/ast.toy:25:30
          var: d @test/Examples/Toy/Ch1/ast.toy:25:33
        ]
      VarDecl f<> @test/Examples/Toy/Ch1/ast.toy:28:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:11
          var: a @test/Examples/Toy/Ch1/ast.toy:28:30
          var: c @test/Examples/Toy/Ch1/ast.toy:28:33
        ]
    } // Block
```

You can reproduce this result and play with the example in the `examples/toy/Ch1/` directory; try running `path/to/BUILD/bin/toyc-ch1 test/Examples/Toy/Ch1/ast.toy -emit=ast`.

The code for the lexer is fairly straightforward; it is all in a single header: `examples/toy/Ch1/include/toy/Lexer.h`. The parser can be found in `examples/toy/Ch1/include/toy/Parser.h`; it is a recursive descent parser. If you are not familiar with such a Lexer/Parser, these are very similar to the LLVM Kaleidoscope equivalent that are detailed in the first two chapters of the [Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl02.html).

The [next chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/) will demonstrate how to convert this AST into MLIR.

# 2 Emitting Basic MLIR
Now that we’re familiar with our language and the AST, let’s see how MLIR can help to compile Toy.

## Introduction: Multi-Level Intermediate Representation 
Other compilers, like LLVM (see the [Kaleidoscope tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)), offer a fixed set of predefined types and (usually _low-level_ / RISC-like) instructions. It is up to the frontend for a given language to perform any language-specific type-checking, analysis, or transformation before emitting LLVM IR. For example, Clang will use its AST to perform not only static analysis but also transformations, such as C++ template instantiation through AST cloning and rewrite. Finally, languages with construction at a higher-level than C/C++ may require non-trivial lowering from their AST to generate LLVM IR.
>  LLVM 提供了一组预定义的类型和 (通常是低级的/类 RISC 的) 指令，即 LLVM IR
>  因此，对于给定的语言，需要由前端负责在生成 LLVM IR 之前执行所有特定于语言的类型检查、分析或转换
>  例如，Clang 使用其 AST 执行静态分析和转换，例如通过 AST 克隆和转换来实例化 C++ 模板
>  高于 C/C++ 级别的语言可能需要从其 AST 执行非平凡的降级以生成 LLVM IR

As a consequence, multiple frontends end up reimplementing significant pieces of infrastructure to support the need for these analyses and transformation. MLIR addresses this issue by being designed for extensibility. As such, there are few pre-defined instructions (_operations_ in MLIR terminology) or types.
>  因此，多个前端不得不重新实现大量  infrastructure 以支持这些分析和转换
>  MLIR 在设计上就考虑了可拓展性，以解决该问题
>  因此 MLIR 中只有很少的预定义的指令 (MLIR 称为操作) 和预定义的类型

## Interfacing with MLIR 
[Language Reference](https://mlir.llvm.org/docs/LangRef/)

MLIR is designed to be a completely extensible infrastructure; there is no closed set of attributes (think: constant metadata), operations, or types. MLIR supports this extensibility with the concept of [Dialects](https://mlir.llvm.org/docs/LangRef/#dialects). Dialects provide a grouping mechanism for abstraction under a unique `namespace`.
>  MLIR 被设计为一个完全可拓展的基础设施：MLIR 没有封闭的属性集合、操作、类型 
>  (在传统的编译器基础设施中，属性、操作、类型通常是预定义的，形成一个封闭集)
>  MLIR 通过方言的概念支持这种可拓展性，方言提供了一种独特命名空间下的分组机制，用于抽象 
>  (方言允许开发者定义自己的操作、属性、类型，并将它们组织在一个命名空间中，该命名空间就组织和抽象了特定的功能集，命名空间避免了冲突，并且使得不同领域的抽象概念更容易集成在一起)

In MLIR, [`Operations`](https://mlir.llvm.org/docs/LangRef/#operations) are the core unit of abstraction and computation, similar in many ways to LLVM instructions. Operations can have application-specific semantics and can be used to represent all of the core IR structures in LLVM: instructions, globals (like functions), modules, etc.
>  MLIR 中，操作是核心的抽象和计算单元，它类似于 LLVM 的指令
>  操作可以有特定于应用程序的语义
>  操作可以用于表示 LLVM 的所有核心 IR 结构，包括指令、全局变量 (如函数)、模块等

Here is the MLIR assembly for the Toy `transpose` operations:
>  MLIR 对于 Toy 中 `transpose` 运算的表示如下

```mlir
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

Let’s break down the anatomy of this MLIR operation:

- `%t_tensor`
    - The name given to the result defined by this operation (which includes [a prefixed sigil to avoid collisions](https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords)). An operation may define zero or more results (in the context of Toy, we will limit ourselves to single-result operations), which are SSA values. The name is used during parsing but is not persistent (e.g., it is not tracked in the in-memory representation of the SSA value).
>  `%t_tensor` 是该运算定义的结果的名字，前缀符号 `%` 用于避免命名冲突
>  一个操作可以定义零个或者多个结果 (Toy 中，我们仅关注单结果的操作)
>  这些结果是 SSA 值 (静态单赋值)
>  该名字会在解析过程中使用，但不是持久的 (例如，它不会在 SSA 值的内存表示中被追踪)
>  (意思是该名称会在解析过程中用来引用该结果值/SSA 值，但在内存表示中，可能会用其他的唯一表示符来追踪该结果值，而不是这个名字)

> [! info] SSA
> 静态单赋值 (Static Single Assignment) 是编译器 IR 中常用的一种形式，在该形式下，每个变量在生命周期内只能被赋值一次
> 该形式有助于简化编译器优化，同时使数据流分析更简单
> 例如，代码
> ```c
> x = 1
> x = x + 1
> ```
>  在 SSA 形式中，会被改写为
>  ```c
>  x_1 = 5
>  x_2 = x_1 + 1
>  ```
> 这样，每个变量 `x_1` , `x_2` 就仅被赋值一次

- `"toy.transpose"`
    - The name of the operation. It is expected to be a unique string, with the namespace of the dialect prefixed before the “`.`”. This can be read as the `transpose` operation in the `toy` dialect.
>  `"toy.transpose"` 是该操作的名字
>  `.` 之前是方言的命名空间名称 (`toy`)
>  该名字应该是一个唯一的字符串
>  该名字可以读为：`toy` 方言中的 `transpose` 操作

- `(%tensor)`
    - A list of zero or more input operands (or arguments), which are SSA values defined by other operations or referring to block arguments.
>  `(%tensor)` 表示包含了零个或者多个操作数的列表
>  这些操作数可以是其他操作定义的 SSA 值，或者直接引用块参数 (块参数是在基本块开始处定义的值，通常用于传递函数参数或全局变量)

- `{ inplace = true }`
    - A dictionary of zero or more attributes, which are special operands that are always constant. Here we define a boolean attribute named ‘inplace’ that has a constant value of true.
>  `{ ...=... }` 表示包含一个或者多个属性的字典，这些属性都是特殊的操作数，并且它们都是常量
>  本例中，我们定义了名为 `inplace` 的布尔属性，其值为常量值 `true`

- `(tensor<2x3xf64>) -> tensor<3x2xf64>`
    - This refers to the type of the operation in a functional form, spelling the types of the arguments in parentheses and the type of the return values afterward.
>  `(tensor<2x3xf64>) -> tensor<3x2xf64>` 是操作的功能形式类型
>  参数的类型处于 `()` 中，返回值的类型在 `->` 后

- `loc("example/file/path":12:1)`
    - This is the location in the source code from which this operation originated.
>  `loc(...)` 表示源码中，该操作出现的位置

Shown here is the general form of an operation. As described above, the set of operations in MLIR is extensible. Operations are modeled using a small set of concepts, enabling operations to be reasoned about and manipulated generically. These concepts are:

- A name for the operation.
- A list of SSA operand values.
- A list of [attributes](https://mlir.llvm.org/docs/LangRef/#attributes).
- A list of [types](https://mlir.llvm.org/docs/LangRef/#type-system) for result values.
- A [source location](https://mlir.llvm.org/docs/Diagnostics/#source-locations) for debugging purposes.
- A list of successors [blocks](https://mlir.llvm.org/docs/LangRef/#blocks) (for branches, mostly).
- A list of [regions](https://mlir.llvm.org/docs/LangRef/#regions) (for structural operations like functions).

>  这里展示的是 MLIR 的操作的一般形式
>  MLIR 的操作集合是可拓展的，操作通过一组概念建模，这些概念包括
>  - 操作的名称
>  - 一个 SSA 值列表
>  - 一个属性列表
>  - 一个结果值类型的列表
>  - 操作的源地址 (用于 debug)
>  - 一个后继块的列表 (主要用于分支操作，这些后继块会在分支操作完成后被执行)
>  - 一个区域的列表 (主要用于结构化操作，如函数)

In MLIR, every operation has a mandatory source location associated with it. Contrary to LLVM, where debug info locations are metadata and can be dropped, in MLIR, the location is a core requirement, and APIs depend on and manipulate it. Dropping a location is thus an explicit choice which cannot happen by mistake.
>  MLIR 中，每个操作都必须关联一个源位置
>  LLVM 中，debug 信息是元数据，可以丢弃；MLIR 中，源位置信息是核心要求，MLIR 的 API 依赖于该信息并且会使用操纵该信息

To provide an illustration: If a transformation replaces an operation by another, that new operation must still have a location attached. This makes it possible to track where that operation came from.
>  例如，如果一个转换操作将一个操作将替换为了另一个操作，新的操作要求附有源位置信息，这样才有可能追踪该操作的来源

It’s worth noting that the mlir-opt tool - a tool for testing compiler passes - does not include locations in the output by default. The `-mlir-print-debuginfo` flag specifies to include locations. (Run `mlir-opt --help` for more options.)
>  mlir-opt tool (测试编译器 passes 的工具) 默认不在输出中包含位置信息
>  `-mlir-print-debuginfo` 用于指定包含位置信息

### Opaque API 
MLIR is designed to allow all IR elements, such as attributes, operations, and types, to be customized. At the same time, IR elements can always be reduced to the above fundamental concepts. This allows MLIR to parse, represent, and [round-trip](https://mlir.llvm.org/getting_started/Glossary/#round-trip) IR for _any_ operation. 
>  MLIR 在设计上允许自定义所有的 IR 元素，例如属性、操作、类型
>  注意所有的 IR 元素都可以归结为之前的基本概念
>  通过自定义，MLIR 可以解析、表示、往返任何操作的 IR

For example, we could place our Toy operation from above into an `.mlir` file and round-trip through _mlir-opt_ without registering any `toy` related dialect:

```mlir
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
```

>  例如，我们可以在没有注册和 `toy` 相关的方言的情况下，将操作 `toy.transpose` 的表示放到一个 `.mlir` 文件中，然后用 mlir-opt 工具往返该操作

In the cases of unregistered attributes, operations, and types, MLIR will enforce some structural constraints (e.g. dominance, etc.), but otherwise they are completely opaque. For instance, MLIR has little information about whether an unregistered operation can operate on particular data types, how many operands it can take, or how many results it produces. This flexibility can be useful for bootstrapping purposes, but it is generally advised against in mature systems. Unregistered operations must be treated conservatively by transformations and analyses, and they are much harder to construct and manipulate.
>  对于未注册的属性、操作、类型，MLIR 会执行一些结构约束 (例如支配关系)
>  未注册的属性、操作、类型对于 MLIR 是完全不透明的，例如 MLIR 完全不了解未注册的操作是否可以处理特定的数据类型、需要接受的操作数的数量、产生的结果的数量
>  因此，未注册的不透明属性、操作、类型的灵活性在启动阶段是有用的，但一般不建议在成熟系统中使用
>  对未注册的操作进行分析和转换时，需要保守地对待

> [! info] Daminance
> 支配关系 (Dominance) 是一种在程序结构中定义的关系，主要用在控制流图中
> 在一个控制流图中，如果每一条从入口节点出发到达节点 $B$ 的路径都必须经过节点 $A$，就称节点 $A$ 支配节点 $B$

This handling can be observed by crafting what should be an invalid IR for Toy and seeing it round-trip without tripping the verifier:

```mlir
func.func @main() {
  %0 = "toy.print"() : () -> tensor<2x3xf64>
}
```

There are multiple problems here: the `toy.print` operation is not a terminator; it should take an operand; and it shouldn’t return any values.

>  例如，我们可以构造一个无效的 Toy 的 IR 如上
>  该 IR 的问题有： `toy.print` 操作不是一个终止符，它应该接受一个操作数，并且不应该返回值
>  虽然该 IR 有问题，但仍然可以正确往返，并不触发验证器的错误

 In the next section, we will register our dialect and operations with MLIR, plug into the verifier, and add nicer APIs to manipulate our operations.
 >  在下一小节，我们将把我们的方言和操作注册到 MLIR，接入验证器，并添加更友好的 API 来操纵这些操作

## Defining a Toy Dialect 
To effectively interface with MLIR, we will define a new Toy dialect. This dialect will model the structure of the Toy language, as well as provide an easy avenue for high-level analysis and transformation.
>  为了和 MLIR 高效交互，我们定义 Toy 方言
>  该方言将建模 Toy 语言的结构，同时提供一个便于进行高层次分析和转换的途径

```c++
/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types. It can
/// also override virtual methods to change some general behavior, which will be
/// demonstrated in later chapters of the tutorial.
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace.
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// An initializer called from the constructor of ToyDialect that is used to
  /// register attributes, operations, types, and more within the Toy dialect.
  void initialize();
};
```

>  该例给出了 Toy 方言的定义
>  方言在代码实现中就是一个类，它继承于 `mlir::Dialect` 类
>  方言在其类中注册属性、操作、类型
>  方言也可以在类中重载 `mlir::Dialect` 的虚方法
>  上例中：
>  `ToyDialect` 的构造函数接受一个 `mlir::MLIRContext` 的指针，构造函数声明为了 `explicit` 防止隐式类型转换
>  `getDialectNamespace()` 定义为静态方法，返回方言名称
>  `initialize()` 函数应该被构造函数调用，用以注册 Toy 方言的属性、操作、类型等等

This is the C++ definition of a dialect, but MLIR also supports defining dialects declaratively via [tablegen](https://llvm.org/docs/TableGen/ProgRef.html). Using the declarative specification is much cleaner as it removes the need for a large portion of the boilerplate when defining a new dialect. It also enables easy generation of dialect documentation, which can be described directly alongside the dialect. In this declarative format, the toy dialect would be specified as:
>  除了 C++ 定义，MLIR 还支持用 tablegen 声明式定义方言
>  声明式方法会更加简洁，它避免了写大量代码，同时声明式定义也便于自动生成方言文档
>  在声明式格式中，Toy 方言应该声明为：

```tablegen
// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  // The namespace of our dialect, this corresponds 1-1 with the string we
  // provided in `ToyDialect::getDialectNamespace`.
  let name = "toy";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for analyzing and optimizing the "
                "Toy language";

  // A much longer description of our dialect.
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }];

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "toy";
}
```

To see what this generates, we can run the `mlir-tblgen` command with the `gen-dialect-decls` action like so:
>  `mlir-tblgen` 接受我们的声明文件，生成方言定义代码

```shell
${build_root}/bin/mlir-tblgen -gen-dialect-decls ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

After the dialect has been defined, it can now be loaded into an MLIRContext:

```c++
  context.loadDialect<ToyDialect>();
```

By default, an `MLIRContext` only loads the [Builtin Dialect](https://mlir.llvm.org/docs/Dialects/Builtin/), which provides a few core IR components, meaning that other dialects, such as our `Toy` dialect, must be explicitly loaded.

>  定义好方言后，我们可以将该方言加载到一个 `MLIRContext` 中
>  `MLIRContext` 默认仅加载内建方言，MLIR 的内建方言仅提供少量的核心成分
>  其他的自定义方言，就需要调用 `loadDialect` 方法显式加载

> [!info] `mlir::MLIRContext`
> MLIR 中，上下文 `mlir::MLIRContext` 是一个全局对象，用于存储和管理各种类型的元数据
> 方言、属性、类型等都需要在上下文中注册，以便在后续编译过程中识别和处理
> `mlir::MLIRContext` 的 `loadDialect` 是一个模板方法，我们将定义好的方言类 `ToyDialect` 作为模板参数传入，该方法根据传入的模板参数确定要加载的方言类型

## Defining Toy Operations 
Now that we have a `Toy` dialect, we can start defining the operations. This will allow for providing semantic information that the rest of the system can hook into. As an example, let’s walk through the creation of a `toy.constant` operation. This operation will represent a constant value in the Toy language.
>  我们接着定义 `Toy` 方言的操作
>  作为示例，我们创建一个名为 `toy.constant` 的操作，该操作将表示 Toy 语言中的一个常量

```mlir
 %4 = "toy.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
```

This operation takes zero operands, a [dense elements](https://mlir.llvm.org/docs/Dialects/Builtin/#denseintorfpelementsattr) attribute named `value` to represent the constant value, and returns a single result of [RankedTensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype). 
>  在上面的定义中
>  `%4` 是操作结果的名称
>  `"toy.constant"` 是操作的名称
>  `()` 表示操作不接受参数
>  `{value = dense<1.0> : tensor<2x3xf64>}` 表示操作有一个名为 `value` 的属性，属性的值是 `dense<1.0>` ，它是一个形状为 `2x3` 的 `f64` 浮点数张量
>  `() -> tensor<2x3xf64>` 表示操作的结果类型是一个形状为 `2x3` 的 `f64` 浮点数张量

An operation class inherits from the [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) `mlir::Op` class which also takes some optional [_traits_](https://mlir.llvm.org/docs/Traits/) to customize its behavior. `Traits` are a mechanism with which we can inject additional behavior into an Operation, such as additional accessors, verification, and more. 
>  一个继承自 CRTP 类 `mlir::Op` 的操作类可以采用一些可选的特性来定制其行为
>  `Traits` 是一种机制，通过该机制，我们可以向一个操作中注入额外的行为，例如定义额外的访问器、验证逻辑等

> [!info] CRTP
> CRTP (Curiously Recurring Template Pattern) 奇异递归模板模式是一种 C++ 编程技术
> CRTP 中，基类是一个模板类，其模板参数可以是自己的派生类，这使得基类在编译时知道自己的派生类类型，以安全实现静态转换
> CRTP 用于在不使用虚函数的情况下实现编译时多态
> CRTP 的一个例子如下所示

```cpp
template <typename Derived>
class Base {
public:
    void inteface() {
        // 调用派生类的实现
        static_cast<Derived*>(this)->implementation();
    }
};

class Derived : public Base<Derived> {
private:
    void implementation() {
        // 具体实现
    }
}
```

Let’s look below at a possible definition for the constant operation that we have described above:

```c++
class ConstantOp : public mlir::Op<
                     /// `mlir::Op` is a CRTP class, meaning that we provide the
                     /// derived class as a template parameter.
                     ConstantOp,
                     /// The ConstantOp takes zero input operands.
                     mlir::OpTrait::ZeroOperands,
                     /// The ConstantOp returns a single result.
                     mlir::OpTrait::OneResult,
                     /// We also provide a utility `getType` accessor that
                     /// returns the TensorType of the single result.
                     mlir::OpTraits::OneTypedResult<TensorType>::Impl> {

 public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// Provide the unique name for this operation. MLIR will use this to register
  /// the operation and uniquely identify it throughout the system. The name
  /// provided here must be prefixed by the parent dialect namespace followed
  /// by a `.`.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Return the value of the constant by fetching it from the attribute.
  mlir::DenseElementsAttr getValue();

  /// Operations may provide additional verification beyond what the attached
  /// traits provide.  Here we will ensure that the specific invariants of the
  /// constant operation are upheld, for example the result type must be
  /// of TensorType and matches the type of the constant `value`.
  LogicalResult verifyInvariants();

  /// Provide an interface to build this operation from a set of input values.
  /// This interface is used by the `builder` classes to allow for easily
  /// generating instances of this operation:
  ///   mlir::OpBuilder::create<ConstantOp>(...)
  /// This method populates the given `state` that MLIR uses to create
  /// operations. This state is a collection of all of the discrete elements
  /// that an operation may contain.
  /// Build a constant with the given return type and `value` attribute.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  /// Build a constant and reuse the type from the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  /// Build a constant by broadcasting the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

>  该例定义了一个操作类 `ConstantOp`，该类继承自模板类 `mlir::Op` ，同时模板类 `mlir::Op` 使用了许多模板参数，包括：
>  `ConstantOp` 类本身、
>  `mlir::OpTrait::ZeroOperands` 表示该操作接受零个操作数、
>  `mlir::OpTrait::OneResult` 表示该操作返回一个结果、
>  `mlir::OpTrait::OneTypedResult<TensorType>::Impl` 用于提供一个工具方法 `getType` ，它返回结果的 `TensorType`
>  操作类的公有方法有：
>  `Op::Op` 基类的构造函数
>  `static llvm::StringRef getOperationName()` 返回操作的名称，MLIR 将使用该名称注册该操作，并唯一标识该操作，这里的名称需要由方言名称作为前缀
>  `mlir::DenseElementsAttr getValue();` 用于从属性取出常数值并返回
>  `LogicalResult verifyInvariants();` 用于提供对操作的验证，保证操作符合特定的不变式，例如返回类型必须是 `TensorType`
>  `static void build(mlir::OpBuilder &builder, mlir::OperationState &state, ...)` 用于从一组输入值构建该操作
>  `build` 方法会被 `builder` 类使用，用于生成该操作的实例，例如 `mlir::OpBuilder::create<ConstantOp>(...)`
>  这些 `build` 方法会填充其 `state` 参数，MLIR 会利用这些 `state` 来创建操作，该 `state` 实际上是一个操作可能包含的所有离散元素的集合

and we can register this operation in the `ToyDialect` initializer:

```c++
void ToyDialect::initialize() {
  addOperations<ConstantOp>();
}
```

>  我们在 `ToyDialect` 类的 `initialize()` 方法中调用 `addOperations` ，将 `ConstantOp` 类作为模板参数传入，以注册该操作

### Op vs Operation: Using MLIR Operations 
Now that we have defined an operation, we will want to access and transform it. In MLIR, there are two main classes related to operations: `Operation` and `Op`. The `Operation` class is used to generically model all operations. It is ‘opaque’, in the sense that it does not describe the properties of particular operations or types of operations. Instead, the `Operation` class provides a general API into an operation instance. On the other hand, each specific type of operation is represented by an `Op` derived class. For instance `ConstantOp` represents a operation with zero inputs, and one output, which is always set to the same value. 
>  定义好操作后，我们需要访问并且转化它
>  MLIR 中和操作相关的主要有两个类：`Operation` , `Op`
>  `Operation` 类用于泛型建模所有操作，它是不透明的，即不描述特定操作的属性或操作的类型
>  `Operation` 类用于提供进入一个操作实例的通用 API
>  所有特定类型的操作都是 `Op` 的衍生类，例如 `ConstantOp` 表示了一个没有输入、有一个始终是相同值的输出的操作

`Op` derived classes act as smart pointer wrapper around a `Operation*`, provide operation-specific accessor methods, and type-safe properties of operations. This means that when we define our Toy operations, we are simply defining a clean, semantically useful interface for building and interfacing with the `Operation` class. This is why our `ConstantOp` defines no class fields; all of the data for this operation is stored in the referenced `Operation`. A side effect of this design is that we always pass around `Op` derived classes “by-value”, instead of by reference or pointer (_passing by value_ is a common idiom in MLIR and applies similarly to attributes, types, etc). 
>  `Op` 派生类作为 `Operation*` 的智能指针包装器，提供操作特定的访问方法、类型安全的操作属性
>  即我们定义 Toy 操作时，实际上是定义一个清晰且语义上有用的接口，该接口用于构建以及与 `Operation` 类交互
>  因此我们在 `Op` 派生类中不定义类字段 (仅声明)，该操作的所有数据都存储在其引用的 `Operation` 中
>  因为 `Op` 派生类仅是轻量的包装器，MLIR 习惯按值传递它，而不是通过引用或指针

Given a generic `Operation*` instance, we can always get a specific `Op` instance using LLVM’s casting infrastructure:
>  给定一个通用的 `Operation*` 实例，我们可以通过 LLVM 的类型转换方法得到具体的 `Op` 实例

```c++
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);

  // This operation is not an instance of `ConstantOp`.
  if (!op)
    return;

  // Get the internal operation instance wrapped by the smart pointer.
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation &&
         "these operation instances are the same");
}
```

### Using the Operation Definition Specification (ODS) Framework 
In addition to specializing the `mlir::Op` C++ template, MLIR also supports defining operations in a declarative manner. This is achieved via the [Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/) framework. Facts regarding an operation are specified concisely into a TableGen record, which will be expanded into an equivalent `mlir::Op` C++ template specialization at compile time. 
>  除了在 C++ 中通过继承 `mlir::Op` 模板类来定义操作，MLIR 还支持以声明性的方式定义操作
>  这需要通过 Operation Definition Specification 框架实现，声明的操作的相关事实被记录在 TableGen 记录中，该记录会在编译时被转化为等价的 C++ 类

Using the ODS framework is the desired way for defining operations in MLIR given the simplicity, conciseness, and general stability in the face of C++ API changes.
>  ODS 框架是 MLIR 中定义操作的首选方式，因为在面对 C++ API 变化时具有简单、简洁和稳定性

Lets see how to define the ODS equivalent of our ConstantOp:

Operations in ODS are defined by inheriting from the `Op` class. To simplify our operation definitions, we will define a base class for operations in the Toy dialect.
>  ODS 中，定义操作时同样要从 `Op` 类继承
>  我们为 Toy 方言定义其操作的基类

```tablegen
// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

>  该例中，我们定义了模板类 `Toy_Op` 作为 Toy 方言操作的基类，它接受两个模板参数 `mnemonic` 和 `traits` ，表示操作的助记符和操作特性列表 (默认为空)，该模板参数实际上是要发送给其模板基类 `Op` 用于实例化模板基类
>  (只有模板基类具有全部的模板参数，它才能实例化为一个类，才能被继承，因此，模板基类的继承类都需要定义为模板类，且需要将其接受的模板参数传递给其模板基类)

With all of the preliminary pieces defined, we can begin to define the constant operation.

We define a toy operation by inheriting from our base ‘Toy_Op’ class above. Here we provide the mnemonic and a list of traits for the operation. The [mnemonic](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-name) here matches the one given in `ConstantOp::getOperationName` without the dialect prefix; `toy.`. Missing here from our C++ definition are the `ZeroOperands` and `OneResult` traits; these will be automatically inferred based upon the `arguments` and `results` fields we define later.

```tablegen
def ConstantOp : Toy_Op<"constant"> {
}
```

>  我们定义 `ConstantOp` 时，将它继承自基类 `Toy_Op<"constant>"` ，其中模板参数 `constant` 就是该操作的助记符
>  在这里我们没有定义 `ZeroOperands` 和 `OneResult` 特性，这些特性之后会根据我们定义的 `arguments` 和 `results` 字段自动推断

At this point you probably might want to know what the C++ code generated by TableGen looks like. Simply run the `mlir-tblgen` command with the `gen-op-decls` or the `gen-op-defs` action like so:

```shell
${build_root}/bin/mlir-tblgen -gen-op-defs ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

Depending on the selected action, this will print either the `ConstantOp` class declaration or its implementation. Comparing this output to the hand-crafted implementation is incredibly useful when getting started with TableGen.

>  `mlir-tblgen --gen-op-defs/--gen-op-decls` 用于根据我们的声明生成操作类的声明或实现

#### Defining Arguments and Results 
With the shell of the operation defined, we can now provide the [inputs](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments) and [outputs](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-results) to our operation. The inputs, or arguments, to an operation may be attributes or types for SSA operand values. The results correspond to a set of types for the values produced by the operation:
>  定义好操作的基本框架后，我们现在为操作提供输入和输出
>  一个操作的输入 (或参数) 可以是属性或者是 SSA 操作数值的类型
>  一个操作的结果定义于该操作所产生的值的类型

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

>  上例中，我们指定 `arguments` 为一个 `F64ElementsAttr`  ，也就是 64 位的浮点数元素属性，同时指定 `results` 为一个 `F64Tensor`，即 64 位浮点数张量类型

By providing a name to the arguments or results, e.g. `$value`, ODS will automatically generate a matching accessor: `DenseElementsAttr ConstantOp::value()`.

>  我们为输入属性指定了名字 `$value` ，则 ODS 会自动为我们的操作生成一个匹配的访问该属性的访问器方法 `DenseElementsAttr ConstantOp::value()`

#### Adding Documentation 
The next step after defining the operation is to document it. Operations may provide [`summary` and `description`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-documentation) fields to describe the semantics of the operation. This information is useful for users of the dialect and can even be used to auto-generate Markdown documents.
>  操作可以提供 `summary` 和 `description` 字段来描述操作的语义
>  这些字段可以用于自动生成文档

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

#### Verifying Operation Semantics 
At this point we’ve already covered a majority of the original C++ operation definition. The next piece to define is the verifier. Luckily, much like the named accessor, the ODS framework will automatically generate a lot of the necessary verification logic based upon the constraints we have given. This means that we don’t need to verify the structure of the return type, or even the input attribute `value`. 
>  我们接下来定义验证器
>  和访问器类似，ODS 框架基于我们给定的约束自动生成必要的验证逻辑
>  因此我们不需要验证返回类型的结果，以及输入属性 `value`

In many cases, additional verification is not even necessary for ODS operations. To add additional verification logic, an operation can override the [`verifier`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-verifier-code) field. The `verifier` field allows for defining a C++ code blob that will be run as part of `ConstantOp::verify`. This blob can assume that all of the other invariants of the operation have already been verified:
>  ODS 框架下，许多情况下我们都不需要添加额外的验证
>  要添加额外的验证逻辑，操作需要覆盖 `verifier` 字段，`verifier` 字段允许定义一段 C++ 代码段，该代码段会作为 `ConstantOp::veify` 的一部分被运行
>  该代码段可以假设操作的所有其他不变式都已经被验证 (因为它是最后验证/运行的)

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);

  // Add additional verification logic to the constant operation. Setting this bit
  // to `1` will generate a `::llvm::LogicalResult verify()` declaration on the
  // operation class that is called after ODS constructs have been verified, for
  // example the types of arguments and results. We implement additional verification
  // in the definition of this `verify` method in the C++ source file.
  let hasVerifier = 1;
}
```

>  我们指定 `hasVerifier` 字段为 `1`，这会在之后生成的操作类中生成一个 `::llvm::LogicalResult verify()` 声明，该 `verify()` 方法会在 ODS 自动提供的验证 (例如参数类型和结果类型的验证) 被调用完毕后被调用
>  我们可以在 C++ 源文件中在 `verify()` 方法中实现额外的验证逻辑

#### Attaching `build` Methods 
The final missing component here from our original C++ example are the `build` methods. ODS can generate some simple build methods automatically, and in this case it will generate our first build method for us. 
>  ODS 会自动生成一些简单的 `build` 方法

For the rest, we define the [`builders`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-builder-methods) field. This field takes a list of `OpBuilder` objects that take a string corresponding to a list of C++ parameters, as well as an optional code block that can be used to specify the implementation inline.
>  如果需要其他的 `build` 方法，我们需要定义 `builders` 字段
>  该字段接受一个 `OpBuilder` 对象列表，这些对象接受一个于 C++ 参数列表对应的字符串，以及一个可选的代码块，来指定内联实现

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  ...

  // Add custom build methods for the constant operation. These methods populate
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      // Call into an autogenerated `build` method.
      build(builder, result, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value. This builder
    // creates a declaration for `ConstantOp::build` with the given parameters.
    OpBuilder<(ins "double":$value)>
  ];
}
```

#### Specifying a Custom Assembly Format 
At this point we can generate our “Toy IR”. For example, the following:

```toy
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

Results in the following IR:

```mlir
module {
  "toy.func"() ({
  ^bb0(%arg0: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1)):
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    "toy.return"(%2) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  }) {sym_name = "multiply_transpose", type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  "toy.func"() ({
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    "toy.return"() : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  }) {sym_name = "main", type = () -> ()} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

One thing to notice here is that all of our Toy operations are printed using the generic assembly format. This format is the one shown when breaking down `toy.transpose` at the beginning of this chapter. MLIR allows for operations to define their own custom assembly format, either [declaratively](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format) or imperatively via C++. Defining a custom assembly format allows for tailoring the generated IR into something a bit more readable by removing a lot of the fluff that is required by the generic format. Let’s walk through an example of an operation format that we would like to simplify.

>  我们将 Toy example 转化为 IR，结果如上所示
>  注意 Toy example 的 IR 中，所有的操作都遵循通用汇编格式，也就是本章开头介绍的格式
>  MLIR 允许为操作定义自定义的汇编格式，可以通过 C++ 进行命令式定义，也可以通过声明式定义
>  自定义汇编格式可以去除通用格式的冗余，提高可读性

##### `toy.print` 
The current form of `toy.print` is a little verbose. There are a lot of additional characters that we would like to strip away. Let’s begin by thinking of what a good format of `toy.print` would be, and see how we can implement it. Looking at the basics of `toy.print` we get:

```mlir
toy.print %5 : tensor<*xf64> loc(...)
```

Here we have stripped much of the format down to the bare essentials, and it has become much more readable. 

>  我们考虑为操作 `toy.print` 定义自定义汇编格式
>  我们希望 `toy.print` 的自定义格式效果如上，它去除了通用格式的一些冗余

To provide a custom assembly format, an operation can either override the `hasCustomAssemblyFormat` field for a C++ format, or the `assemblyFormat` field for the declarative format. 
>  我们可以在 C++ 实现中覆盖 `hasCustomAssemblyFormat` 字段或者在声明式格式中覆盖 `assemblyFormat` 字段来实现自定义汇编格式

Let’s look at the C++ variant first, as this is what the declarative format maps to internally.

```tablegen
/// Consider a stripped definition of `toy.print` here.
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // Divert the printer and parser to `parse` and `print` methods on our operation,
  // to be implemented in the .cpp file. More details on these methods is shown below.
  let hasCustomAssemblyFormat = 1;
}
```

A C++ implementation for the printer and parser is shown below:

```c++
/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer << "toy.print " << op.input();
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand inputOperand;
  mlir::Type inputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType))
    return mlir::failure();

  // Resolve the input operand to the type we parsed in.
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return mlir::failure();

  return mlir::success();
}
```

>  上例中，`OpAsmPrinter` 类是一个流，用于格式化字符串、属性、操作数、类型等；`OpAsmParser` 类提供一系列方法用于解析各种标点符号、属性、操作数、类型等，这些方法都返回一个 `ParserResult` 类，它包装了 `LogicalResult` 。`LogicalResult` 可以被转化为布尔值 `true/false` 表示解析成功或失败。这便于我们将一系列解析规则串联起来

With the C++ implementation defined, let’s see how this can be mapped to the [declarative format](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format). 

The declarative format is largely composed of three different components:

- Directives
    - A type of builtin function, with an optional set of arguments.
- Literals
    - A keyword or punctuation surrounded by \` \`.
- Variables
    - An entity that has been registered on the operation itself, i.e. an argument(attribute or operand), result, successor, etc. In the `PrintOp` example above, a variable would be `$input`.

>  声明式格式主要由三个部分组成
>  - 指令：一类内建函数，带有一组可选参数
>  - 字面量：由 \`\` 包围的关键字或标点符号
>  - 变量：在操作本身上注册的实体，即参数 (属性或操作数)、结果、后继等

A direct mapping of our C++ format looks something like:

```tablegen
/// Consider a stripped definition of `toy.print` here.
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // In the following format we have two directives, `attr-dict` and `type`.
  // These correspond to the attribute dictionary and the type of a given
  // variable represectively.
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

The [declarative format](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format) has many more interesting features, so be sure to check it out before implementing a custom format in C++. After beautifying the format of a few of our operations we now get a much more readable:

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    toy.print %5 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

Above we introduce several of the concepts for defining operations in the ODS framework, but there are many more that we haven’t had a chance to: regions, variadic operands, etc. Check out the [full specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/) for more details.

## Complete Toy Example 
We can now generate our “Toy IR”. You can build `toyc-ch2` and try yourself on the above example: `toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo`. We can also check our RoundTrip: `toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo 2> codegen.mlir` followed by `toyc-ch2 codegen.mlir -emit=mlir`. You should also use `mlir-tblgen` on the final definition file and study the generated C++ code.

At this point, MLIR knows about our Toy dialect and operations. In the [next chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/), we will leverage our new dialect to implement some high-level language-specific analyses and transformations for the Toy language.

# 3 High-level Language-Specific Analysis and Transformation
Creating a dialect that closely represents the semantics of an input language enables analyses, transformations and optimizations in MLIR that require high-level language information and are generally performed on the language AST. For example, `clang` has a fairly [heavy mechanism](https://clang.llvm.org/doxygen/classclang_1_1TreeTransform.html) for performing template instantiation in C++.
>  创建一个和输入语言语义紧密匹配的方言，可以实现在 MLIR 中进行需要高级语言信息且通常需要在语言 AST 上执行的分析、转化和优化
>  例如，`clang` 为了在 C++ 中实现模板实例化，有相当复杂的机制

We divide compiler transformations into two categories: local and global. In this chapter, we focus on how to leverage the Toy Dialect and its high-level semantics to perform local pattern-match transformations that would be difficult in LLVM. For this, we use MLIR’s [Generic DAG Rewriter](https://mlir.llvm.org/docs/PatternRewriter/).
>  我们将编译器分为两类：局部和全局
>  本章中，我们关注如何利用 Toy Dialect 和其高级语义来执行 LLVM 中难以实现的局部模式匹配，我们将使用 MLIR 的通用 DAG 重写器

There are two methods that can be used to implement pattern-match transformations: 1. Imperative, C++ pattern-match and rewrite 2. Declarative, rule-based pattern-match and rewrite using table-driven [Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/) (DRR). Note that the use of DRR requires that the operations be defined using ODS, as described in [Chapter 2](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/).
>  实现模式匹配转换有两种方法：
>  1. 命令式：用 C++ 进行模式匹配和重写
>  2. 声明式：使用基于规则的模式匹配，使用 table-driven 的声明式重写规则重写，使用 DRR 要求使用 ODS 定义操作

## Optimize Transpose using C++ style pattern-match and rewrite
Let’s start with a simple pattern and try to eliminate a sequence of two transposes that cancel out: `transpose(transpose(X)) -> X`. Here is the corresponding Toy example:

```toy
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

Which corresponds to the following IR:

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %1 : tensor<*xf64>
}
```

>  考虑一段对张量连续转置的 Toy 代码，这段 Toy 代码对应的 IR 如上

This is a good example of a transformation that is trivial to match on the Toy IR but that would be quite hard for LLVM to figure. For example, today Clang can’t optimize away the temporary array, and the computation with the naive transpose is expressed with these loops:
>  对该连续转置进行优化 (转换) 在 Toy IR 上容易实现，但在 LLVM 上则不然
>  例如，目前 Clang 不能优化掉临时数组，而 Toy 代码中的连续转置计算需要用以下的循环来表达：

```c++
#define N 100
#define M 100

void sink(void *);
void double_transpose(int A[N][M]) {
  int B[M][N];
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       B[j][i] = A[i][j];
    }
  }
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       A[i][j] = B[j][i];
    }
  }
  sink(A);
}
```

For a simple C++ approach to rewrite, involving matching a tree-like pattern in the IR and replacing it with a different set of operations, we can plug into the MLIR `Canonicalizer` pass by implementing a `RewritePattern`:
>  使用 C++ 方法进行重写涉及到在 IR 中匹配树状模式，并将其替换为一组不同的操作
>  我们通过实现 `RewritePatter` 来接入 MLIR 的 `Caonicalizer` 过程

```c++
/// Fold transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  llvm::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

>  上述代码中
>  -  `SimplifyRedundantTranspose` 继承自 `mlir::OpRewritePattern<TransposeOp>` ，表示这是一个专门用于 `TransposeOp` 操作的重写模式
>  -  `SimplifyRedundantTranspose` 的构造函数调用了基类的构造函数，传入 `context` 和 `benefit` 参数，`benefit` 用于指示该重写模式的优先级，数值越高越有可能被优先处理
>  -  `matchAndRewrite` 方法接受一个 `Transpose` 操作和一个 `mlir::PatternRewriter` 对象作为参数，`PatternRewriter` 是 MLIR 提供的工具，用于执行 IR 的修改
> - 该方法首先获取当前转置操作的输入 `transposeInput` ，然后检查这个输入是否是由另一个转置操作生成的。如果不是，模式不匹配，返回。如果是，模式匹配成功，使用 `rewriter.replaceOp` 方法将当前转置操作替换为原始输入，然后返回 `successs()`

The implementation of this rewriter is in `ToyCombine.cpp`. The [canonicalization pass](https://mlir.llvm.org/docs/Canonicalization/) applies transformations defined by operations in a greedy, iterative manner. To ensure that the canonicalization pass applies our new transform, we set [hasCanonicalizer = 1](https://mlir.llvm.org/docs/DefiningDialects/Operations/#hascanonicalizer) and register the pattern with the canonicalization framework.
>  MLIR 的标准化过程会贪心且迭代地应用操作所定义的转换
>  为了确保标准化过程会应用我们定义的新转换，我们设定 `hasCanonicalizer=1`，然后将模式注册到标准化框架中：

>  MLIR 中，标准化过程会将 IR 转化为更规范的形式，标准化框架使用预定义的模式 `RewrittenPattern` 对 IR 进行优化

```c++
// Register our patterns for rewrite by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```

>  上述代码中
>  -  `getCanonicalizationPatterns` 是一个静态成员函数，其参数 `resutls` 是一个重写模式集合 `RewritePatternSet` ，存储所有要注册的重写模式，`context` 是上下文对象，提供环境信息
> - `results.add<SimplifyRedundantTranspose>` 将 `SimplifyRedundantTranspose` 模式添加到 `results` 中 

We also need to update our main file, `toyc.cpp`, to add an optimization pipeline. In MLIR, the optimizations are run through a `PassManager` in a similar way to LLVM:
>  我们需要更新主文件，添加优化管道，MLIR 中，优化会通过 `PassManager` 实现，和 LLVM 类似：

>  MLIR 中，优化是通过一系列 passes 实现的，每个 pass 负责执行特定的优化或转换，`PassManager` 是管理这些 passes 的工具，它按照一定顺序应用这些 passes

```c++
  mlir::PassManager pm(module->getName());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

>  上述代码中
>  - `mlir::PassManager pm(module->getName())` 创建了一个 `PassManager` 实例，它接受模块名称作为参数
>  - `pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass())` 将一个优化 pass 添加到 `PassManager` 中
>  - 其中 `mlir::createCanonicalizerPass()` 创建了一个标准化 pass，用于应用标准化规则，`addNestedPass<mlir::toy::FuncOp>` 表示这个 pass 会被嵌套在 `mlir::toy::FuncOp` 的上下文中，`mlir::toy::FuncOp` 是一个函数操作，表示优化将应用于函数级别

Finally, we can run `toyc-ch3 test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt` and observe our pattern in action:

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %arg0 : tensor<*xf64>
}
```

As expected, we now directly return the function argument, bypassing any transpose operation. However, one of the transposes still hasn’t been eliminated. That is not ideal! What happened is that our pattern replaced the last transform with the function input and left behind the now dead transpose input. 
>  优化后，有一个转置操作没有被消除，我们的模式将后一个转置替换为了函数输入，上一个转置被留下，且是多余的

The Canonicalizer knows to clean up dead operations; however, MLIR conservatively assumes that operations may have side-effects. We can fix this by adding a new trait, `Pure`, to our `TransposeOp`:

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {...}
```

>  标准化程序知道清除无用的操作，但 MLIR 假设了操作可能有副作用
>  我们可以为 `Transpose` 操作添加特性 `Pure` 表示没有副作用

Let’s retry now `toyc-ch3 test/transpose_transpose.toy -emit=mlir -opt`:

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  toy.return %arg0 : tensor<*xf64>
}
```

Perfect! No `transpose` operation is left - the code is optimal.

In the next section, we use DRR for pattern match optimizations associated with the Reshape op.

## Optimize Reshapes using DRR
Declarative, rule-based pattern-match and rewrite (DRR) is an operation DAG-based declarative rewriter that provides a table-based syntax for pattern-match and rewrite rules:
>  DRR 是一种基于 DAG 的声明式重写器，它为模式匹配和规则重写提供了基于 table 的语法：

```tablegen
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;
```

A redundant reshape optimization similar to `SimplifyRedundantTranspose` can be expressed more simply using DRR as follows:
>  类似 `SimplifyRedundantTranspose` 的冗余 reshape 优化可以用 DRR 表达如下：

```tablegen
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

The automatically generated C++ code corresponding to each of the DRR patterns can be found under `path/to/BUILD/tools/mlir/examples/toy/Ch3/ToyCombine.inc`.

DRR also provides a method for adding argument constraints when the transformation is conditional on some properties of the arguments and results. An example is a transformation that eliminates reshapes when they are redundant, i.e. when the input and output shapes are identical.
>  DRR 还提供了方法，用于在变换基于参数和结果的某些属性时添加参数约束

```tablegen
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

Some optimizations may require additional transformations on instruction arguments. This is achieved using NativeCodeCall, which allows for more complex transformations either by calling into a C++ helper function or by using inline C++. An example of such an optimization is FoldConstantReshape, where we optimize Reshape of a constant value by reshaping the constant in place and eliminating the reshape operation.
>  一些优化可能需要对指令参数执行额外的变换，可以通过 NativeCodeCall 实现，NativeCodeCall 允许调用 C++ 辅助函数或内联 C++ 实现更复杂的变换

```tablegen
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

We demonstrate these reshape optimizations using the following trivial_reshape.toy program:

```c++
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
```

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
    toy.print %3 : tensor<2x1xf64>
    toy.return
  }
}
```

We can try to run `toyc-ch3 test/Examples/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt` and observe our pattern in action:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

As expected, no reshape operations remain after canonicalization.

Further details on the declarative rewrite method can be found at [Table-driven Declarative Rewrite Rule (DRR)](https://mlir.llvm.org/docs/DeclarativeRewrites/).

In this chapter, we saw how to use certain core transformations through always available hooks. In the [next chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/), we will see how to use generic solutions that scale better through Interfaces.

# 4 Enabling Generic Transformation with Interfaces
## Background: Grappling with an Extensible IR 
Through dialects, MLIR allows for the representation of many different levels of abstraction; the Toy dialect that we have previously defined is one such example. Though these different dialects may represent different abstractions, there is often a set of common transformations and analyses that we would like to perform. The problem that arises is that naively implementing each transformation for each dialect leads to large amounts of code duplication, as the internal algorithms are generally very similar, if not the same. We would like to provide the ability for transformations to opaquely hook into dialects like Toy to get the information they need.
>  通过方言，MLIR 可以表示不同的抽象层次
>  虽然不同的方言表示不同的抽象，但我们通常会希望对方言执行一组通用的转换和解析
>  显然，为每个方言分别实现这些转换操作将导致大量代码重复，因为内部算法都是相似的，故我们希望实现共用的转换操作，并为转换操作提供一种能够不透明地接入方言的能力，便于它们获取所需要的信息

MLIR provides a set of always available-hooks for certain core transformations, as seen in the [previous chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/), where we registered some canonicalizations via a hook on our operations (`getCanonicalizationPatterns`). However, these types of hooks don’t really scale well. 
>  MLIR 为部分核心转换提供了一组始终可用的接入点
>  例如，在上一章中，我们通过操作上的接入点 `getCanonicalizationPatterns` 为操作注册了一些规范形式，但是，这些接入点并不具备良好的可拓展性

Therefore, a more generic solution was designed, in the form of [interfaces](https://mlir.llvm.org/docs/Interfaces/), to make the MLIR infrastructure as extensible as the representation. Interfaces provide a generic mechanism for dialects and operations to provide information to a transformation or analysis.
>  因此，MLIR 提供了一个更加通用的方案，即接口，它使得 MLIR infrastructure 可以像其 representation 一样拓展
>  接口提供了一种通用机制，便于方言和操作向转换和分析提供其信息

## Shape Inference: Preparing for Code Generation
Our Toy IR currently operates on generic tensors, meaning that we don’t know the shape of tensors other than during the initialization of constants. This complicates optimizations, as well as code generation. Fortunately, we can simply propagate the shapes through the computation until they are all known. 
>  Toy IR 目前在通用的张量上运算，意味着除了在初始化常量时，我们不知道张量的形状，而这会让优化和代码生成变得复杂
>  幸运的是，我们可以在计算过程中将张量的形状传播，直到所有参与计算的张量已知

The issue is how to handle calls to user-defined generic functions: every call site could deduce different shapes. One possibility would be to perform symbolic inference based on the argument types, but this would be hard to generalize if we were to introduce more control flow in the language. Another approach would be function specialization, where every call site with new argument shapes duplicates the called function and specializes it. 
>  此时，问题在于如何处理对用户定义的泛型函数的定义：泛型函数在每个调用点都可能推导出不同的形状
>  一种可能性是基于参数类型执行符号推理，但如果我们在语言中引入过多控制流，将难以 generalize
>  另一种方法是函数专用化，在每个具有新参数形状的调用点复制被调用的函数代码，并对其进行专用化

The approach we take for Toy is to inline all of the function calls, then perform intraprocedural shape propagation.
>  在 Toy 中，我们要采用的方法是内联所有的函数调用，然后执行过程内的形状传播

### Inlining 
Here we could write an inlining algorithm specifically designed for the Toy dialect, but that can become quite complicated depending on the level of complexity that we want. Disregarding cost modeling, the pure structural transformation is already complex to implement from scratch. 

Thankfully, MLIR provides a generic inliner algorithm that dialects can plug into. All we need to do in Toy is to provide the [interfaces](https://mlir.llvm.org/docs/Interfaces/) for the inliner to hook into.
>  MLIR 提供了一个通用的 inliner 算法，方言可以向 inliner 提供接口，使得它可以通过该接口接入方言

The first thing we need to do is to define the constraints on inlining operations in the Toy dialect. This information is provided through a [dialect interface](https://mlir.llvm.org/docs/Interfaces/#dialect-interfaces). This is essentially a class containing a set of virtual hooks which the dialect can override. In this case, the interface is `DialectInlinerInterface`.
>  我们首先要定义 Toy 方言中内联操作的约束条件，要定义这些约束条件，方言需要使用方言接口
>  方言接口本质是一个类，包含了一组虚钩子 (虚函数)，方言可以覆盖这些钩子，本例中，我们需要使用的方言接口是 ``

```c++
/// This class defines the interface for handling inlining with Toy operations.
/// We simplify inherit from the base interface class and override
/// the necessary methods.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// This hook checks to see if the given callable operation is legal to inline
  /// into the given call. For Toy this hook can simply return true, as the Toy
  /// Call operation is always inlinable.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// This hook checks to see if the given operation is legal to inline into the
  /// given region. For Toy this hook can simply return true, as all Toy
  /// operations are inlinable.
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }

  /// This hook cheks if the given 'src' region can be inlined into the 'dest'
  /// region. The regions here are the bodies of the callable functions. For
  /// Toy, any function can be inlined, so we simply return true.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  /// This hook is called when a terminator operation has been inlined. The only
  /// terminator that we have in the Toy dialect is the return
  /// operation(toy.return). We handle the return by replacing the values
  /// previously returned by the call operation with the operands of the
  /// return.
  void handleTerminator(Operation *op,
                        ValueRange valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
```

>  上述代码中
>  类 `ToyInlinerInterface` 用于为方言 Toy 处理 inlining 操作，它继承自 `DialcetInlinerInterface`
>  `using DialectInlinerInterface::DialectInlinerInterface` 表示 `ToyInlinerInterface` 使用基类的构造函数
>  该类中定义了四个重载方法 `isLegalInline`
>  第一个 `isLegalInline` 检查操作是否可以内联到指定的调用点，我们直接返回 `true` ，因为 Toy 中的调用操作总是可内联的
>  第二个 `isLegalInline` 检查操作是否可以内联到指定区域，我们直接返回 `true` ，因为 Toy 中的所有操作都是可内联的
>  第三个 `isLegalInline` 检查给定的 `src` 区域是否可以内联到 `dest` 区域，这里的区域指可调用函数的函数体，我们直接返回 `true` ，因为 Toy 中所有操作都是可内联的
>  `handleTerminator` 在终止操作 (例如 `return`) 被内联时被调用，Toy 方言中，唯一的终止操作是 `toy.return` ，我们将调用操作的返回值替换为返回操作的参数 (即内联函数的返回值)

Besides, the inliner will only discard private-visible unused function definitions. We also have to set the visibility of functions (except the main function) in the MLIR generator.
> 因为 inliner 只会丢弃可见性为 private 的未使用的函数定义，故我们还需要在 MLIR 生成器中设置函数的可见性为私有

```c++
/// Emit a new function and add it to the MLIR module.
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
  ...
  // If this function isn't main, then set the visibility to private.
  if (funcAST.getProto()->getName() != "main")
    function.setPrivate();

  return function;
}
```

We then register our dialect interface directly on the Toy dialect, similarly to how we did for operations.
>  我们在 `ToyDialect` 的 `initialize()` 方法中添加 `addInterface<ToyInlinerInterface>` 调用，将方言接口注册到 Toy 方言中

```c++
void ToyDialect::initialize() {
  addInterfaces<ToyInlinerInterface>();
}
```

Next, we need to provide a way for the inliner to know that `toy.generic_call` represents a call, and `toy.func` represents a function. 
>  我们已经通过方言接口让 inliner 知道了方言内内联操作的约束条件
>  接下来，我们需要让 inliner 知道 `toy.generic_call` 表示一个调用，`toy.func` 表示一个函数

MLIR provides [operation interfaces](https://mlir.llvm.org/docs/Interfaces/#attributeoperationtype-interfaces) that can be used to mark an operation as being “call-like” or “callable-like”. Unlike dialect interfaces, operation interfaces provide a more refined granularity of information that is specific and core to a single operation. 
>  MLIR 提供操作接口，用于标记一个操作是 "call-like" 还是 "callable-like"
>  相较于方言接口，操作接口提供了更精细的信息粒度，信息是针对单个操作的

The interfaces that we will be adding here is the `CallOpInterface` and `CallableOpInterface`.
>  我们要添加 `CallOpInterface, CallableOpInterface` 两个操作接口

To add this interface we just need to include the definition into our operation specification file (`Ops.td`):

```tablegen
include "mlir/Interfaces/CallInterfaces.td"
```

and add it to the traits list of `GenericCallOp`:

```tablegen
def FuncOp : Toy_Op<"func",
    [FunctionOpInterface, IsolatedFromAbove]> {
  ...
}

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
}
```

>  我们需要在我们的操作 `GenericCallOp, FuncOp` 的特性列表中添加指定的接口

In the above we also use the `DeclareOpInterfaceMethods` directive to auto-declare all of the interface methods in the class declaration of `GenericCallOp`. 
>  在 `GenericCallOp` 的声明中，我们还使用了 `DeclareOpInterfaceMethods` 指令以自动声明所有的接口方法

We have already provided the definition in the `extraClassDeclaration` field of the `FuncOp` class:
>  我们在 `FuncOp` 类中重载 `extraClassDeclaration` 方法，该方法被调用时，如果函数操作是 callable，它就返回函数操作的区域/函数体

```c++
/// Returns the region on the function operation that is callable.
Region *FuncOp::getCallableRegion() { return &getBody(); }

// ....

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}
```

>  此外，上述代码中
>  `GenericCallOp` 覆盖了 `getCallableForCallee() ` ，用于返回 callee
>  `GenericCallOp` 覆盖了 `setCallleeFromCallable()` ，用于设置 callee
>  `GenericCallOp` 覆盖了 `getArgOperands()` ，用于返回参数
>  `GenericCallOp` 覆盖了 `getArgOperandsMutable()` ，用于返回参数

Now that the inliner has been informed about the Toy dialect, we can add the inliner pass to the pass manager for Toy:

```c++
  pm.addPass(mlir::createInlinerPass());
```

>  inliner 可以通过接口了解 Toy 方言的信息后，我们将 inliner pass 添加到 Toy 的 pass manager

Now let’s look at a working example:

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
  %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
  %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %5 : tensor<*xf64>
  toy.return
}
```

We have two calls to multiply_transpose that we would like to inline into main, but if we look at the output nothing has changed. We are missing one last subtle piece: there is a hidden type conversion on the edge of the call. If we look at the above, the operands to the generic_call are of type `tensor<2x3xf64>`, while the inputs to the function expect `tensor<*xf64>`. To resolve this difference, the inliner expects an explicit cast operation to be inserted. 
>  上述 Toy 的 MLIR 表示中，包含了两个对 `multiply_transpose` 的调用，我们希望将其内联到 main 中
>  在进行 inline 之前，我们还需要处理操作调用的隐式类型转换，观察发现，在 `main` 中 `generic_call` 的输入参数的形状是 `tensor<2x3xf64>`，而在定义中，参数的形状是 `tensor<*xf64>`，故 inliner 期望会插入一个显式转换操作

For this, we need to add a new operation to the Toy dialect, `ToyCastOp` (toy.cast), to represent casts between two different shapes.
>  为此，我们需要为 Toy 方言添加一个新操作 `ToyCastOp` ，用于表示不同形状之间的转换

```tablegen
def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    Pure,
    SameOperandsAndResultShape]
  > {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked,
    then shape is required to match. The operation is invalid if converting
    to a mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
```

Note that the definition of this cast operation adds a `CastOpInterface` to the traits list. This interface provides several utilities for cast-like operation, such as folding identity casts and verification. 
>  `CastOp` 的定义如上，注意到它还添加了 `CastOpInterface` 作为特性，`CastOpInterface` 为类 cast 的操作提供了一些功能，例如折叠恒等转换和验证等

We hook into this interface by providing a definition for the `areCastCompatible` method:
>  我们在 `CastOp` 中覆盖 `areCastCompatible` 方法
>  它检查转换的两边类型是否兼容，如果兼容，返回 true

```c++
/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInterface` to verify
/// this operation and provide other additional utilities.
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // The inputs must be Tensors with the same element type.
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // The shape is required to match if both types are ranked.
  return !input.hasRank() || !output.hasRank() || input == output;
}
```

With a proper cast operation, we can now override the necessary hook on the `ToyInlinerInterface` to insert it for us when necessary:
>  之后，我们在 `ToyInlineInterface` 中覆盖 `materizlizeCallConversion` 方法，该方法会在发现方言中的调用和被调用区域出现类型不兼容时尝试生成一个操作，用于类型转换
>  如果没有生成转换操作，该方法会返回 nullptr

```c++
struct ToyInlinerInterface : public DialectInlinerInterface {
  ...

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};
```

If we run the working example through the pipeline again, we get the expected:

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
  %3 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
  %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
  %5 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
  %6 = toy.mul %4, %5 : tensor<*xf64>
  toy.print %6 : tensor<*xf64>
  toy.return
}
```

NOTE: The generic inliner will also perform simplifications, so the output may be a bit cleaner than expected.
>  inliner 也会执行简化

### Intraprocedural Shape Inference 
Now that we have inlined all of the functions, we are left with a main function containing a mix of static and dynamically shaped operations. We can now write a simple shape inference pass to propagate shapes intraprocedurally (within a single function). 
>  目前我们内联了所有函数，此时 `main` 函数中既有动态形状的操作，也有静态形状的操作，现在我们可以编写一个形状推理 pass，在过程内 (在单个函数内) 传播形状

We could write this as a pass that directly encodes the constraints of the operations within the Toy dialect, but this seems like a good candidate for a transformation that could be written generically. As a good rule of thumb, it is best to express a transformation as generically as possible, such that it can be extended to other dialects in the future. There is no telling how many other dialects may have similar needs or encounter the same problems.
>  我们希望为此编写一个尽可能通用的 transformation

For shape inference, if we break down the problem to its core, we really just want operations to tell us the expected outputs given a set of statically known inputs. (We can definitely get more complex than that, but for our needs we can keep it simple.) Given that this property is core to a specific operation, we can define an operation interface that can be specified on operations that need to have their result shapes inferred.
>  对于形状推理，我们实质上只需要操作在给定一组静态的、已知形状的输入时，能够给出期望的输出形状
>  为此，我们可以定义一个操作接口，如果操作需要对其结果形状进行推理，就接入该接口

Similarly to operations, we can also [define operation interfaces](https://mlir.llvm.org/docs/Interfaces/#attributeoperationtype-interfaces) using the operation definition specification (ODS) framework.

The interface is defined by inheriting from `OpInterface`, which takes the name to be given to the generated C++ interface class as a template argument. 

>  操作接口可以用 ODS 定义，自定义的新的操作接口需要继承 `OpInterface`，继承时其模板参数为自定义的操作接口的名字

For our purposes, we will simply name the generated class `ShapeInference`. We also provide a description for the interface.

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];
}
```

Next, we define the interface methods that the operations will need to provide. An interface method is comprised of: a description; a C++ return type in string form; a method name in string form; and a few optional components, depending on the need. See the [ODS documentation](https://mlir.llvm.org/docs/Interfaces/#attributeoperationtype-interfaces) for more information.
>  我们继而为该操作接口定义方法，这些方法应该是接入该接口的操作所需要实现的
>  一个接口方法包含了：描述、C++ 返回类型 (字符串形式)、方法名 (字符串形式)、一些可选组件

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  ...

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.", "void", "inferShapes">
  ];
}
```

Now that the interface is defined, we can add it to the necessary Toy operations in a similar way to how we added the `CallOpInterface` to the GenericCallOp:
>  定义好接口后，我们和之前一样，将其加入 `GenericCallOp` 的特性中即可

```tablegen
def MulOp : Toy_Op<"mul",
    [..., DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
}
```

Each of these operations will then need to provide a definition for the `inferShapes()` method. As an example, for the mul op, the result shape is inferred as the shape of the inputs.
>  接入了该接口的方法就需要实现我们所定义的 `inferShapes()` 方法，用于执行形状推理

```c++
/// Infer the output shape of the MulOp, this is required by the shape inference
/// interface.
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }
```

At this point, each of the necessary Toy operations provide a mechanism by which to infer their output shapes. 
>  为必要的 Toy 操作接入接口并定义好方法后，这些操作就具有了形状推理的能力

The ShapeInferencePass will operate on functions: it will run on each function in isolation. MLIR also supports general [OperationPasses](https://mlir.llvm.org/docs/PassManagement/#operation-pass) that run on any isolated operation, but here our module only contains functions, so there is no need to generalize to all operations.
>  我们继而需要实现一个自定义的 pass `ShapeInferencePass` ，它会单独运行每个函数，执行类型推理
>  MLIR 还提供了更通用的 OperationPasses，它会单独运行每个操作

Implementing such a pass is done by creating a class inheriting from `mlir::OperationPass<FuncOp>` and overriding the `runOnOperation()` method.

```c++
class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp function = getOperation();
    ...
  }
};
```

>  我们创建一个名为 `ShapeInferencePass` 的类，它需要继承 `mlir::OperationPass<FuncOp>` ，并重写 `runOnOpreation()` 方法

While at it, let’s also create a helper method for instantiating the pass:

```c++
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
```

The shape inference algorithm operates as follows:

1. Build a worklist containing all the operations that return a dynamically shaped tensor: these are the operations that need shape inference.
2. Iterate on the worklist:
    - find an operation to process: the next ready operation in the worklist has all of its arguments non-generic,
    - if no operation is found, break out of the loop,
    - remove the operation from the worklist,
    - infer the shape of its output from the argument types.
3. If the worklist is empty, the algorithm succeeded.

>  我们将形状推理算法定义为：
>  1. 构建 worklist，它包含所有会返回动态形状 tensor 的操作，这些操作将需要形状推理
>  2. 遍历 worklist，对每个参数是非泛型的操作执行形状推理，执行后，将其移除 worklist
>  3. 如果 worklist 为空，算法成功

When processing an operation like described, we query if it registered the `ShapeInference` interface, using this code snippet:

```c++
  // Ask the operation to infer its output shapes.
  LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");

  /// We check if an operation has a particular interface by casting.
  if (ShapeInference shapeOp = dyn_cast<ShapeInference>(op)) {
    shapeOp.inferShapes();
  } else {
    op->emitError("unable to infer shape of operation without shape "
                  "inference interface");
    return signalPassFailure();
  }
```

We can then add our pass to the pass manager:

```c++
  pm.addPass(mlir::createShapeInferencePass());
```

>  对 worklist 中的操作执行形状推理时，我们先检查该操作是否接入了 `ShapeInference` 接口，如果有，为它添加 `ShapeInferencePass`

If we rerun our original example, we now get the following:

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.mul %1, %1 : tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
}
```

You can build `toyc-ch4` and try yourself: `toyc-ch4 test/Examples/Toy/Ch4/codegen.toy -emit=mlir -opt`.

In the [next chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/), we will start the process of code generation by targeting a lower level dialect for optimizing some of the more compute-heavy Toy operations.

# 5 Partial Lowering to Lower-Level Dialects for Optimization
At this point, we are eager to generate actual code and see our Toy language take life. We will use LLVM to generate code, but just showing the LLVM builder interface here wouldn’t be very exciting. Instead, we will show how to perform progressive lowering through a mix of dialects coexisting in the same function.

To make it more interesting, in this chapter we will consider that we want to reuse existing optimizations implemented in a dialect optimizing affine transformations: `Affine`. This dialect is tailored to the computation-heavy part of the program and is limited: it doesn’t support representing our `toy.print` builtin, for instance, neither should it! Instead, we can target `Affine` for the computation heavy part of Toy, and in the [next chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/) directly target the `LLVM IR` dialect for lowering `print`. As part of this lowering, we will be lowering from the [TensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype) that `Toy` operates on to the [MemRefType](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype) that is indexed via an affine loop-nest. Tensors represent an abstract value-typed sequence of data, meaning that they don’t live in any memory. MemRefs, on the other hand, represent lower level buffer access, as they are concrete references to a region of memory.
>  我们将考虑复用 `Affine` 方言中的现存优化，以优化仿射变换，该方言专门为程序中的计算密集部分而设计，我们希望将 `Affine` 用于 Toy 中的计算密集部分
>  并且，我们会直接将 `print` lower to `LLVM IR` 方言，我们会将 `Toy` 所操作的 `TensorType` lower to 通过仿射嵌套循环索引的 `MemRefType` ，Tensor 表示抽象的值类型数据序列，它们并不驻留在任何内存中，而 MemRef 表示较低级别的缓存区访问，它们是对内存区域的具体引用 `

## Dialect Conversions
MLIR has many different dialects, so it is important to have a unified framework for [converting](https://mlir.llvm.org/getting_started/Glossary/#conversion) between them. This is where the `DialectConversion` framework comes into play. This framework allows for transforming a set of _illegal_ operations to a set of _legal_ ones. To use this framework, we need to provide two things (and an optional third):

- A [Conversion Target](https://mlir.llvm.org/docs/DialectConversion/#conversion-target)
    - This is the formal specification of what operations or dialects are legal for the conversion. Operations that aren’t legal will require rewrite patterns to perform [legalization](https://mlir.llvm.org/getting_started/Glossary/#legalization).
- A set of [Rewrite Patterns](https://mlir.llvm.org/docs/DialectConversion/#rewrite-pattern-specification)
    - This is the set of [patterns](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/) used to convert _illegal_ operations into a set of zero or more _legal_ ones.
- Optionally, a [Type Converter](https://mlir.llvm.org/docs/DialectConversion/#type-conversion).
    - If provided, this is used to convert the types of block arguments. We won’t be needing this for our conversion.

>  `DialectConversion` 框架用于将一组不合法的操作转化为一组合法的操作，该框架要求我们提供
>  - 一个转换目标：指定对于该转换来说，什么操作或方言是合法的，不合法的操作将会请求重写模式执行合法化
>  - 一组重写模式：用于将不合法的操作转化为一组零个或多个的合法操作
>  - 类型转换器 (Optional)：用于转换 block arguments 的类型

### Conversion Target 
For our purposes, we want to convert the compute-intensive `Toy` operations into a combination of operations from the `Affine`, `Arith`, `Func`, and `MemRef` dialects for further optimization. 
>  我们希望将计算密集的 `Toy` 操作转化为来自 `Affine, Arith, Func, MemRef` 方言的操作，便于进一步优化

To start off the lowering, we first define our conversion target:
>  我们定义 `ToyToAffineLoweringPass` 类，并定义 `runOnOperation()` 方法
>  方法中，我们为 lowering target 添加具体的合法方言或操作
>  同时，我们将不希望被 lower 的 Toy 操作设定为 legal，将希望被 lower 的 Toy 操作设定为 illegal

```c++
void ToyToAffineLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as *legal*. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return type.isa<TensorType>(); });
  });
  ...
}
```

Above, we first set the toy dialect to illegal, and then the print operation as legal. We could have done this the other way around. Individual operations always take precedence over the (more generic) dialect definitions, so the order doesn’t matter. See `ConversionTarget::getOpInfo` for the details.

### Conversion Patterns 
After the conversion target has been defined, we can define how to convert the _illegal_ operations into _legal_ ones. Similarly to the canonicalization framework introduced in [chapter 3](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/), the [`DialectConversion` framework](https://mlir.llvm.org/docs/DialectConversion/) also uses [RewritePatterns](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/) to perform the conversion logic. These patterns may be the `RewritePatterns` seen before or a new type of pattern specific to the conversion framework `ConversionPattern`. `ConversionPatterns` are different from traditional `RewritePatterns` in that they accept an additional `operands` parameter containing operands that have been remapped/replaced. This is used when dealing with type conversions, as the pattern will want to operate on values of the new type but match against the old. For our lowering, this invariant will be useful as it translates from the [TensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype) currently being operated on to the [MemRefType](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype). Let’s look at a snippet of lowering the `toy.transpose` operation:
>  定义完转化目标，我们继而需要定义转换模式
>  和 canonicalization 框架一样，`DialectConversion` 框架也使用 RewritePatterns 来执行转换逻辑，包括 `RewritePattern, ConversionPattern` `ConversionPattern` 和 `RewritePattern` 的差异在于它接受一个额外的 `operand` 参数，表示被重映射/重放置的参数，这会在类型转换时被使用，因为新操作的会有不同的值类型，但需要和旧的类型匹配

```c++
/// Lower the `toy.transpose` operation to an affine loop nest.
struct TransposeOpLowering : public mlir::ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TransposeOp::getOperationName(), 1, ctx) {}

  /// Match and rewrite the given `toy.transpose` operation, with the given
  /// operands that have been remapped from `tensor<...>` to `memref<...>`.
  llvm::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Call to a helper function that will lower the current operation to a set
    // of affine loops. We provide a functor that operates on the remapped
    // operands, as well as the loop induction variables for the inner most
    // loop body.
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::PatternRewriter &rewriter,
              ArrayRef<mlir::Value> memRefOperands,
              ArrayRef<mlir::Value> loopIvs) {
          // Generate an adaptor for the remapped operands of the TransposeOp.
          // This allows for using the nice named accessors that are generated
          // by the ODS. This adaptor is automatically provided by the ODS
          // framework.
          TransposeOpAdaptor transposeAdaptor(memRefOperands);
          mlir::Value input = transposeAdaptor.input();

          // Transpose the elements by generating a load from the reverse
          // indices.
          SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return rewriter.create<mlir::AffineLoadOp>(loc, input, reverseIvs);
        });
    return success();
  }
};
```

>  我们为 `TransposeOp` 专门定义了一个类 `TransposeOpLowering` ，它继承 `mlir::ConversionPattern`
>  其中 `matchAndRewrite` 方法负责重写该操作，该方法会调用 `lowerOpToLoops` 函数，将当前操作 lower 到一组仿射循环

Now we can prepare the list of patterns to use during the lowering process:

```c++
void ToyToAffineLoweringPass::runOnOperation() {
  ...

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<..., TransposeOpLowering>(&getContext());

  ...
```

### Partial Lowering 
Once the patterns have been defined, we can perform the actual lowering. The `DialectConversion` framework provides several different modes of lowering, but, for our purposes, we will perform a partial lowering, as we will not convert `toy.print` at this time.

```c++
void ToyToAffineLoweringPass::runOnOperation() {
  ...

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our *illegal*
  // operations were not converted successfully.
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}
```

#### Design Considerations With Partial Lowering
Before diving into the result of our lowering, this is a good time to discuss potential design considerations when it comes to partial lowering. In our lowering, we transform from a value-type, TensorType, to an allocated (buffer-like) type, MemRefType. However, given that we do not lower the `toy.print` operation, we need to temporarily bridge these two worlds. There are many ways to go about this, each with their own tradeoffs:

- Generate `load` operations from the buffer
    
    One option is to generate `load` operations from the buffer type to materialize an instance of the value type. This allows for the definition of the `toy.print` operation to remain unchanged. The downside to this approach is that the optimizations on the `affine` dialect are limited, because the `load` will actually involve a full copy that is only visible _after_ our optimizations have been performed.
    
- Generate a new version of `toy.print` that operates on the lowered type
    
    Another option would be to have another, lowered, variant of `toy.print` that operates on the lowered type. The benefit of this option is that there is no hidden, unnecessary copy to the optimizer. The downside is that another operation definition is needed that may duplicate many aspects of the first. Defining a base class in [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/) may simplify this, but you still need to treat these operations separately.
    
- Update `toy.print` to allow for operating on the lowered type
    
    A third option is to update the current definition of `toy.print` to allow for operating the on the lowered type. The benefit of this approach is that it is simple, does not introduce an additional hidden copy, and does not require another operation definition. The downside to this option is that it requires mixing abstraction levels in the `Toy` dialect.

>  在 lowering 过程中，我们从值类型 TensorType 转化为了已分配类型 MemRefType，然而，由于我们没有对 `toy.print` 操作进行降级，我们需要暂时在这两个世界之间架起桥梁
>  有多种方法可以实现这一点，每种方法都有其自身的权衡：
>  - 从缓冲区生成`load`操作
>  一种选择是从缓冲区类型生成 `load` 操作以实例化值类型的一个实例。这允许 `toy.print` 操作的定义保持不变。这种方法的缺点是，对 `affine` 方言的优化受到限制，因为 `load` 实际上涉及一个完整的复制，而这个复制仅在我们的优化执行之后才可见。
>  - 生成一个新的`toy.print`版本，该版本操作于降级后的类型
>  另一种选择是拥有另一个降级后的 `toy.print` 变体，该变体操作于降级后的类型。这种方法的好处是没有隐藏的、不必要的复制给优化器。缺点是需要定义另一个操作，这可能会使第一个操作的许多方面重复。在 [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/) 中定义基类可能会简化这一点，但你仍然需要将这些操作分开处理。
>  - 更新 `toy.print` 以允许操作于降级后的类型
>  第三种选择是更新当前的 `toy.print` 定义，使其能够操作于降级后的类型。这种方法的优点是简单，不会引入额外的隐藏复制，并且不需要另一个操作定义。这种方法的缺点是它要求在 `Toy` 方言中混合抽象层次。

For the sake of simplicity, we will use the third option for this lowering. This involves updating the type constraints on the PrintOp in the operation definition file:

```tablegen
def PrintOp : Toy_Op<"print"> {
  ...

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
}
```

### Complete Toy Example 
Let’s take a concrete example:

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

With affine lowering added to our pipeline, we can now generate:

```mlir
func.func @main() {
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %cst_1 = arith.constant 3.000000e+00 : f64
  %cst_2 = arith.constant 4.000000e+00 : f64
  %cst_3 = arith.constant 5.000000e+00 : f64
  %cst_4 = arith.constant 6.000000e+00 : f64

  // Allocating buffers for the inputs and outputs.
  %0 = memref.alloc() : memref<3x2xf64>
  %1 = memref.alloc() : memref<3x2xf64>
  %2 = memref.alloc() : memref<2x3xf64>

  // Initialize the input buffer with the constant values.
  affine.store %cst, %2[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %2[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %2[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %2[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %2[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %2[1, 2] : memref<2x3xf64>

  // Load the transpose value from the input buffer and store it into the
  // next input buffer.
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %2[%arg1, %arg0] : memref<2x3xf64>
      affine.store %3, %1[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Multiply and store into the output buffer.
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %4 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %5 = arith.mulf %3, %4 : f64
      affine.store %5, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Print the value held by the buffer.
  toy.print %0 : memref<3x2xf64>
  memref.dealloc %2 : memref<2x3xf64>
  memref.dealloc %1 : memref<3x2xf64>
  memref.dealloc %0 : memref<3x2xf64>
  return
}
```

### Taking Advantage of Affine Optimization 
Our naive lowering is correct, but it leaves a lot to be desired with regards to efficiency. For example, the lowering of `toy.mul` has generated some redundant loads. Let’s look at how adding a few existing optimizations to the pipeline can help clean this up. Adding the `LoopFusion` and `AffineScalarReplacement` passes to the pipeline gives the following result:
>  我们的 lowering 方法是正确的，但在效率方面还有很多不足之处。例如，`toy.mul` 的 lowering 生成了一些多余的 loads
>  让我们看看在 pipeline 中添加一些现有的优化如何可以改善这一点，将 `LoopFusion` 和 `AffineScalarReplacement` 传递到 pipeline 后，结果如下：

```mlir
func.func @main() {
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %cst_1 = arith.constant 3.000000e+00 : f64
  %cst_2 = arith.constant 4.000000e+00 : f64
  %cst_3 = arith.constant 5.000000e+00 : f64
  %cst_4 = arith.constant 6.000000e+00 : f64

  // Allocating buffers for the inputs and outputs.
  %0 = memref.alloc() : memref<3x2xf64>
  %1 = memref.alloc() : memref<2x3xf64>

  // Initialize the input buffer with the constant values.
  affine.store %cst, %1[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %1[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %1[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %1[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %1[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %1[1, 2] : memref<2x3xf64>

  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      // Load the transpose value from the input buffer.
      %2 = affine.load %1[%arg1, %arg0] : memref<2x3xf64>

      // Multiply and store into the output buffer.
      %3 = arith.mulf %2, %2 : f64
      affine.store %3, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Print the value held by the buffer.
  toy.print %0 : memref<3x2xf64>
  memref.dealloc %1 : memref<2x3xf64>
  memref.dealloc %0 : memref<3x2xf64>
  return
}
```

Here, we can see that a redundant allocation was removed, the two loop nests were fused, and some unnecessary `load` s were removed. You can build `toyc-ch5` and try yourself: `toyc-ch5 test/Examples/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine`. We can also check our optimizations by adding `-opt`.
>  在这里，我们可以看到多余的分配被移除了，两个循环嵌套被融合了，并且一些不必要的 `load` 也被移除了

In this chapter we explored some aspects of partial lowering, with the intent to optimize. In the [next chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/) we will continue the discussion about dialect conversion by targeting LLVM for code generation.

# 6 Lowering to LLVM and CodeGeneration
In the [previous chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/), we introduced the [dialect conversion](https://mlir.llvm.org/docs/DialectConversion/) framework and partially lowered many of the `Toy` operations to affine loop nests for optimization. In this chapter, we will finally lower to LLVM for code generation.

## Lowering to LLVM
For this lowering, we will again use the dialect conversion framework to perform the heavy lifting. However, this time, we will be performing a full conversion to the [LLVM dialect](https://mlir.llvm.org/docs/Dialects/LLVM/). Thankfully, we have already lowered all but one of the `toy` operations, with the last being `toy.print`. Before going over the conversion to LLVM, let’s lower the `toy.print` operation. We will lower this operation to a non-affine loop nest that invokes `printf` for each element. Note that, because the dialect conversion framework supports [transitive lowering](https://mlir.llvm.org/getting_started/Glossary/#transitive-lowering), we don’t need to directly emit operations in the LLVM dialect. By transitive lowering, we mean that the conversion framework may apply multiple patterns to fully legalize an operation. 
>  对于 lowering to LLVM，我们将再次使用方言转换框架来完成大部分工作，这次我们将进行全面转换到 [LLVM 方言](https://mlir.llvm.org/docs/Dialects/LLVM/)。
>  值得庆幸的是，我们已经 lower 了除 `toy.print` 之外的所有 `toy` 操作，在讨论到 LLVM 的转换之前，让我们先 lower `toy.print` 操作。我们将把这个操作 lower 为一个非仿射循环嵌套，该嵌套对每个元素调用 `printf`。
>  请注意，由于方言转换框架支持[传递性降低](https://mlir.llvm.org/getting_started/Glossary/#transitive-lowering)，我们不需要直接在 LLVM 方言中生成操作。transitive lowering 意味着转换框架可以应用多个模式以完全合法化一个操作。(先 lower 到一个方言，再从该方言 lower 到另一个方言)

In this example, we are generating a structured loop nest instead of the branch-form in the LLVM dialect. As long as we then have a lowering from the loop operations to LLVM, the lowering will still succeed.
>  在这个例子中，我们生成了一个结构化的循环嵌套，而不是 LLVM 方言中的分支形式。只要我们随后有从循环操作到 LLVM 的 lowering，整体的 lowering 仍然会成功。

During lowering we can get, or build, the declaration for printf as so:

```c++
/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module,
                                           LLVM::LLVMDialect *llvmDialect) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get("printf", context);

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI8PtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get("printf", context);
}
```

Now that the lowering for the printf operation has been defined, we can specify the components necessary for the lowering. These are largely the same as the components defined in the [previous chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/).
>  定义了 printf 操作的降低方式后，我们现在可以指定实现这一降低所需的组件。这些组件大多与前一章中定义的组件相同。(转换目标、转换模式、类型转换器)

### Conversion Target 
For this conversion, aside from the top-level module, we will be lowering everything to the LLVM dialect.
>  转换目标是 LLVM 方言

```c++
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();
```

### Type Converter 
This lowering will also transform the MemRef types which are currently being operated on into a representation in LLVM. To perform this conversion, we use a TypeConverter as part of the lowering. This converter specifies how one type maps to another. This is necessary now that we are performing more complicated lowerings involving block arguments. Given that we don’t have any Toy-dialect-specific types that need to be lowered, the default converter is enough for our use case.
>  这种降低也将把当前正在操作的 MemRef 类型转换为 LLVM 中的表示形式。为了执行这种转换，我们在降低过程中使用了 TypeConverter。该转换器指定了一个类型如何映射到另一个类型。既然我们现在正在进行涉及块参数的更复杂的降低，这 (定义类型转换器) 就变得必要了。
>  鉴于我们没有需要降低的特定于 Toy 方言的类型，默认转换器就足够满足我们的用例需求。

```c++
  LLVMTypeConverter typeConverter(&getContext());
```

### Conversion Patterns
Now that the conversion target has been defined, we need to provide the patterns used for lowering. At this point in the compilation process, we have a combination of `toy`, `affine`, `arith`, and `std` operations. Luckily, the `affine`, `arith`, and `std` dialects already provide the set of patterns needed to transform them into LLVM dialect. These patterns allow for lowering the IR in multiple stages by relying on [transitive lowering](https://mlir.llvm.org/getting_started/Glossary/#transitive-lowering).
>  转换目标已经定义后，我们需要提供用于降低的模式。
>  在编译过程的这个阶段，我们有一组合 `toy`、`affine`、`arith` 和 `std` 操作。幸运的是，`affine`、`arith` 和 `std` 这些方言已经提供了将它们转换为 LLVM 方言所需的模式集。这些模式通过依赖 [传递降低](https://mlir.llvm.org/getting_started/Glossary/#transitive-lowering) 允许分多个阶段降低 IR。

```c++
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
  mlir::cf::populateSCFToControlFlowConversionPatterns(patterns, &getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,
                                                          patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(patterns, &getContext());

  // The only remaining operation, to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.add<PrintOpLowering>(&getContext());
```

### Full Lowering 
We want to completely lower to LLVM, so we use a `FullConversion`. This ensures that only legal operations will remain after the conversion.
>  我们要完全转换到 LLVM，因此我们使用了一个 `FullConversion`。这确保了转换后只保留合法的操作。

```c++
  mlir::ModuleOp module = getOperation();
  if (mlir::failed(mlir::applyFullConversion(module, target, patterns)))
    signalPassFailure();
```

Looking back at our current working example:

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

We can now lower down to the LLVM dialect, which produces the following code:

```mlir
llvm.func @free(!llvm<"i8*">)
llvm.func @printf(!llvm<"i8*">, ...) -> i32
llvm.func @malloc(i64) -> !llvm<"i8*">
llvm.func @main() {
  %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %1 = llvm.mlir.constant(2.000000e+00 : f64) : f64

  ...

^bb16:
  %221 = llvm.extractvalue %25[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %222 = llvm.mlir.constant(0 : index) : i64
  %223 = llvm.mlir.constant(2 : index) : i64
  %224 = llvm.mul %214, %223 : i64
  %225 = llvm.add %222, %224 : i64
  %226 = llvm.mlir.constant(1 : index) : i64
  %227 = llvm.mul %219, %226 : i64
  %228 = llvm.add %225, %227 : i64
  %229 = llvm.getelementptr %221[%228] : (!llvm."double*">, i64) -> !llvm<"f64*">
  %230 = llvm.load %229 : !llvm<"double*">
  %231 = llvm.call @printf(%207, %230) : (!llvm<"i8*">, f64) -> i32
  %232 = llvm.add %219, %218 : i64
  llvm.br ^bb15(%232 : i64)

  ...

^bb18:
  %235 = llvm.extractvalue %65[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %236 = llvm.bitcast %235 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%236) : (!llvm<"i8*">) -> ()
  %237 = llvm.extractvalue %45[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %238 = llvm.bitcast %237 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%238) : (!llvm<"i8*">) -> ()
  %239 = llvm.extractvalue %25[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %240 = llvm.bitcast %239 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%240) : (!llvm<"i8*">) -> ()
  llvm.return
}
```

See [LLVM IR Target](https://mlir.llvm.org/docs/TargetLLVMIR/) for more in-depth details on lowering to the LLVM dialect.

## CodeGen: Getting Out of MLIR 
At this point we are right at the cusp of code generation. We can generate code in the LLVM dialect, so now we just need to export to LLVM IR and setup a JIT to run it.

### Emitting LLVM IR 
Now that our module is comprised only of operations in the LLVM dialect, we can export to LLVM IR. To do this programmatically, we can invoke the following utility:
>  现在我们的模块仅包含LLVM方言中的操作，我们可以导出为LLVM IR。
>  为了通过编程方式实现这一点，我们可以调用以下实用程序：

```c++
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module);
  if (!llvmModule)
    /* ... an error was encountered ... */
```

Exporting our module to LLVM IR generates:

```llvm
define void @main() {
  ...

102:
  %103 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %8, 0
  %104 = mul i64 %96, 2
  %105 = add i64 0, %104
  %106 = mul i64 %100, 1
  %107 = add i64 %105, %106
  %108 = getelementptr double, double* %103, i64 %107
  %109 = memref.load double, double* %108
  %110 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double %109)
  %111 = add i64 %100, 1
  cf.br label %99

  ...

115:
  %116 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %24, 0
  %117 = bitcast double* %116 to i8*
  call void @free(i8* %117)
  %118 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %16, 0
  %119 = bitcast double* %118 to i8*
  call void @free(i8* %119)
  %120 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %8, 0
  %121 = bitcast double* %120 to i8*
  call void @free(i8* %121)
  ret void
}
```

If we enable optimization on the generated LLVM IR, we can trim this down quite a bit:
>  如果我们对生成的LLVM IR启用优化，我们可以将其精简很多。

```llvm
define void @main()
  %0 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 1.000000e+00)
  %1 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 1.600000e+01)
  %putchar = tail call i32 @putchar(i32 10)
  %2 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 4.000000e+00)
  %3 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 2.500000e+01)
  %putchar.1 = tail call i32 @putchar(i32 10)
  %4 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 9.000000e+00)
  %5 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 3.600000e+01)
  %putchar.2 = tail call i32 @putchar(i32 10)
  ret void
}
```

The full code listing for dumping LLVM IR can be found in `examples/toy/Ch6/toy.cpp` in the `dumpLLVMIR()` function:

```c++

int dumpLLVMIR(mlir::ModuleOp module) {
  // Translate the module, that contains the LLVM dialect, to LLVM IR. Use a
  // fresh LLVM IR context. (Note that LLVM is not thread-safe and any
  // concurrent use of a context requires external locking.)
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}
```

### Setting up a JIT
Setting up a JIT to run the module containing the LLVM dialect can be done using the `mlir::ExecutionEngine` infrastructure. This is a utility wrapper around LLVM’s JIT that accepts `.mlir` as input. The full code listing for setting up the JIT can be found in `Ch6/toyc.cpp` in the `runJit()` function:
>  设置一个 JIT 来运行包含 LLVM 方言的模块可以使用 `mlir::ExecutionEngine` 基础设施来完成。这是一个围绕 LLVM 的 JIT 构建的实用包装器，接受 `.mlir` 作为输入。

```c++
int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(module,
      /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invoke("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```

You can play around with it from the build directory:

```shell
$ echo 'def main() { print([[1, 2], [3, 4]]); }' | ./bin/toyc-ch6 -emit=jit
1.000000 2.000000
3.000000 4.000000
```

You can also play with `-emit=mlir`, `-emit=mlir-affine`, `-emit=mlir-llvm`, and `-emit=llvm` to compare the various levels of IR involved. Also try options like [`--mlir-print-ir-after-all`](https://mlir.llvm.org/docs/PassManagement/#ir-printing) to track the evolution of the IR throughout the pipeline.

The example code used throughout this section can be found in test/Examples/Toy/Ch6/llvm-lowering.mlir.

So far, we have worked with primitive data types. In the [next chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/), we will add a composite `struct` type.

# 7 Adding a Composite Type to Toy
In the [previous chapter](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/), we demonstrated an end-to-end compilation flow from our Toy front-end to LLVM IR. In this chapter, we will extend the Toy language to support a new composite `struct` type.
>  在上一章中，我们展示了从我们的 Toy 前端到 LLVM IR 的端到端编译流程。在本章中，我们将扩展 Toy 语言以支持新的复合 `struct` 类型。

## Defining a `struct` in Toy 
The first thing we need to define is the interface of this type in our `toy` source language. The general syntax of a `struct` type in Toy is as follows:

```toy
# A struct is defined by using the `struct` keyword followed by a name.
struct MyStruct {
  # Inside of the struct is a list of variable declarations without initializers
  # or shapes, which may also be other previously defined structs.
  var a;
  var b;
}
```

Structs may now be used in functions as variables or parameters by using the name of the struct instead of `var`. The members of the struct are accessed via a `.` access operator. Values of `struct` type may be initialized with a composite initializer, or a comma-separated list of other initializers surrounded by `{}`. An example is shown below:

```toy
struct Struct {
  var a;
  var b;
}

# User defined generic function may operate on struct types as well.
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # We pass these arguments to functions like we do with variables.
  var c = multiply_transpose(value);
  print(c);
}
```

## Defining a `struct` in MLIR 
In MLIR, we will also need a representation for our struct types. MLIR does not provide a type that does exactly what we need, so we will need to define our own. We will simply define our `struct` as an unnamed container of a set of element types. The name of the `struct` and its elements are only useful for the AST of our `toy` compiler, so we don’t need to encode it in the MLIR representation.
>  定义好 Toy 中的结构体后，在 MLIR 中，我们也需要表示结构类型。MLIR 没有提供一个完全符合我们需要的类型，因此我们需要定义自己的类型。
>  我们将简单地将我们的 `struct` 定义为一组元素类型的无名容器。`struct` 及其元素的名称仅对我们的 `toy` 编译器的 AST 有用，因此我们不需要在 MLIR 表示中对其进行编码。

### Defining the Type Class 
#### Defining the Type Class 
As mentioned in [chapter 2](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/), [`Type`](https://mlir.llvm.org/docs/LangRef/#type-system) objects in MLIR are value-typed and rely on having an internal storage object that holds the actual data for the type. The `Type` class in itself acts as a simple wrapper around an internal `TypeStorage` object that is uniqued within an instance of an `MLIRContext`. When constructing a `Type`, we are internally just constructing and uniquing an instance of a storage class.
>  如在[第2章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)中所述，MLIR中的[`Type`](https://mlir.llvm.org/docs/LangRef/#type-system)对象是基于值的，并依赖于内部存储对象来保存类型的实际数据。
>  `Type` 类本身作为 `TypeStorage` 对象的一个简单包装器，该对象在 `MLIRContext` 实例中是唯一的。在构建一个 `Type` 时，我们实际上是构造并使存储类的实例唯一。

When defining a new `Type` that contains parametric data (e.g. the `struct` type, which requires additional information to hold the element types), we will need to provide a derived storage class. The `singleton` types that don’t have any additional data (e.g. the [`index` type](https://mlir.llvm.org/docs/Dialects/Builtin/#indextype)) don’t require a storage class and use the default `TypeStorage`.
>  当定义一个新的包含参数化数据的 `Type`（例如，需要额外信息来保存元素类型的 `struct` 类型）时，我们需要提供一个派生的存储类。
>  那些没有额外数据的“单例”类型（例如，[`index`类型](https://mlir.llvm.org/docs/Dialects/Builtin/#indextype)）不需要存储类，并使用默认的`TypeStorage`。

##### Defining the Storage Class 
Type storage objects contain all of the data necessary to construct and unique a type instance. Derived storage classes must inherit from the base `mlir::TypeStorage` and provide a set of aliases and hooks that will be used by the `MLIRContext` for uniquing. Below is the definition of the storage instance for our `struct` type, with each of the necessary requirements detailed inline:
>  类型存储对象包含了构造和唯一标识类型实例所需的所有数据。
>  派生的存储类必须从基础类 `mlir::TypeStorage` 继承，并提供一组别名和钩子，这些将在 `MLIRContext` 中用于唯一化。
>  以下是我们的 `struct` 类型的存储实例定义，其中详细列出了每个必要的要求：

```c++
/// This class represents the internal storage of the Toy `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself, see the `StructType::get` method further below.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::Type> elementTypes;
};
```

##### Defining the Type Class 
With the storage class defined, we can add the definition for the user-visible `StructType` class. This is the class that we will actually interface with.
>  定义好存储类后，我们继而定义面向用于的 `StructType` 类，我们会实际交互的就是这个类

```c++
/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");

    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. The first parameter is the context to unique in. The
    // parameters after are forwarded to the storage instance.
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
  }

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes() {
    // 'getImpl' returns a pointer to the internal storage instance.
    return getImpl()->elementTypes;
  }

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
```

We register this type in the `ToyDialect` initializer in a similar way to how we did with operations:

```c++
void ToyDialect::initialize() {
  addTypes<StructType>();
}
```

(An important note here is that when registering a type, the definition of the storage class must be visible.)
>  注意在注册一个类型时，其存储类的定义必须可见

With this we can now use our `StructType` when generating MLIR from Toy. See examples/toy/Ch7/mlir/MLIRGen.cpp for more details.

### Exposing to ODS 
After defining a new type, we should make the ODS framework aware of our Type so that we can use it in the operation definitions and auto-generate utilities within the Dialect. A simple example is shown below:

```tablegen
// Provide a definition for the Toy StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the StructType as belonging to the Toy dialect.
def Toy_StructType :
    DialectType<Toy_Dialect, CPred<"$_self.isa<StructType>()">,
                "Toy struct type">;

// Provide a definition of the types that are used within the Toy dialect.
def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;
```

### Parsing and Printing 
At this point we can use our `StructType` during MLIR generation and transformation, but we can’t output or parse `.mlir`. For this we need to add support for parsing and printing instances of the `StructType`. This can be done by overriding the `parseType` and `printType` methods on the `ToyDialect`. Declarations for these methods are automatically provided when the type is exposed to ODS as detailed in the previous section.
>  此时，我们可以在 MLIR 生成和转换过程中使用我们的 `StructType`，但我们不能输出或解析 `.mlir` 文件。
>  为此，我们需要添加对解析和打印 `StructType` 实例的支持。这可以通过在 `ToyDialect` 上覆盖 `parseType` 和 `printType` 方法来实现。当该类型被暴露给 ODS 时，这些方法的声明会自动生成，如前一节所述。

```c++
class ToyDialect : public mlir::Dialect {
public:
  /// Parse an instance of a type registered to the toy dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print an instance of a type registered to the toy dialect.
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};
```

These methods take an instance of a high-level parser or printer that allows for easily implementing the necessary functionality. Before going into the implementation, let’s think about the syntax that we want for the `struct` type in the printed IR. As described in the [MLIR language reference](https://mlir.llvm.org/docs/LangRef/#dialect-types), dialect types are generally represented as: `! dialect-namespace < type-data >`, with a pretty form available under certain circumstances. The responsibility of our `Toy` parser and printer is to provide the `type-data` bits. We will define our `StructType` as having the following form:
>  这些方法采用一个高层的解析器或打印器的实例，从而可以轻松实现所需的功能。在深入实现之前，让我们思考一下我们希望在打印出的 IR 中的 `struct` 类型所使用的语法。如在 [MLIR语言参考](https://mlir.llvm.org/docs/LangRef/#dialect-types)中所述，方言类型通常表示为：`!方言命名空间<类型数据>`，在某些情况下可以使用美观的形式。我们的 `Toy` 解析器和打印器的责任是提供 `类型数据` 部分。我们将定义我们的 `StructType` 具有以下形式：

```
  struct-type ::= `struct` `<` type (`,` type)* `>`
```

#### Parsing 
An implementation of the parser is shown below:

```c++
/// Parse an instance of a type registered to the toy dialect.
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // Check that the type is either a TensorType or another StructType.
    if (!elementType.isa<mlir::TensorType, StructType>()) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}
```

#### Printing 
An implementation of the printer is shown below:

```c++
/// Print an instance of a type registered to the toy dialect.
void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  StructType structType = type.cast<StructType>();

  // Print the struct type according to the parser format.
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
```

Before moving on, let’s look at a quick of example showcasing the functionality we have now:

```toy
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
}
```

Which generates the following:

```mlir
module {
  toy.func @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) {
    toy.return
  }
}
```

### Operating on `StructType` 
Now that the `struct` type has been defined, and we can round-trip it through the IR. The next step is to add support for using it within our operations.

#### Updating Existing Operations 
A few of our existing operations, e.g. `ReturnOp`, will need to be updated to handle `Toy_StructType`.

```tablegen
def ReturnOp : Toy_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  ...
  let arguments = (ins Variadic<Toy_Type>:$input);
  ...
}
```

#### Adding New `Toy` Operations 
In addition to the existing operations, we will be adding a few new operations that will provide more specific handling of `structs`.

##### `toy.struct_constant` 
This new operation materializes a constant value for a struct. In our current modeling, we just use an [array attribute](https://mlir.llvm.org/docs/Dialects/Builtin/#arrayattr) that contains a set of constant values for each of the `struct` elements.

```mlir
  %0 = toy.struct_constant [
    dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  ] : !toy.struct<tensor<*xf64>>
```

##### `toy.struct_access` 
This new operation materializes the Nth element of a `struct` value.

```mlir
  // Using %0 from above
  %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>> -> tensor<*xf64>
```

With these operations, we can revisit our original example:

```toy
struct Struct {
  var a;
  var b;
}

# User defined generic function may operate on struct types as well.
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # We pass these arguments to functions like we do with variables.
  var c = multiply_transpose(value);
  print(c);
}
```

and finally get a full MLIR module:

```mlir
module {
  toy.func @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64> {
    %0 = toy.struct_access %arg0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.struct_access %arg0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %3 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %4 = toy.mul %1, %3 : tensor<*xf64>
    toy.return %4 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.struct_constant [
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>,
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.generic_call @multiply_transpose(%0) : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    toy.print %1 : tensor<*xf64>
    toy.return
  }
}
```

#### Optimizing Operations on `StructType` 
Now that we have a few operations operating on `StructType`, we also have many new constant folding opportunities.

After inlining, the MLIR module in the previous section looks something like:

```mlir
module {
  toy.func @main() {
    %0 = toy.struct_constant [
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>,
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
    %3 = toy.struct_access %0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %4 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.mul %2, %4 : tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
```

We have several `toy.struct_access` operations that access into a `toy.struct_constant`. As detailed in [chapter 3](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/) (FoldConstantReshape), we can add folders for these `toy` operations by setting the `hasFolder` bit on the operation definition and providing a definition of the `*Op::fold` method.

```c++
/// Fold constants.
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return value(); }

/// Fold struct constants.
OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) {
  return value();
}

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor) {
  auto structAttr = adaptor.getInput().dyn_cast_or_null<mlir::ArrayAttr>();
  if (!structAttr)
    return nullptr;

  size_t elementIndex = index().getZExtValue();
  return structAttr[elementIndex];
}
```

To ensure that MLIR generates the proper constant operations when folding our `Toy` operations, i.e. `ConstantOp` for `TensorType` and `StructConstant` for `StructType`, we will need to provide an override for the dialect hook `materializeConstant`. This allows for generic MLIR operations to create constants for the `Toy` dialect when necessary.

```c++
mlir::Operation *ToyDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  if (type.isa<StructType>())
    return builder.create<StructConstantOp>(loc, type,
                                            value.cast<mlir::ArrayAttr>());
  return builder.create<ConstantOp>(loc, type,
                                    value.cast<mlir::DenseElementsAttr>());
}
```

With this, we can now generate code that can be generated to LLVM without any changes to our pipeline.

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.mul %1, %1 : tensor<3x2xf64>
    toy.print %2 : tensor<3x2xf64>
    toy.return
  }
}
```

You can build `toyc-ch7` and try yourself: `toyc-ch7 test/Examples/Toy/Ch7/struct-codegen.toy -emit=mlir`. More details on defining custom types can be found in [DefiningAttributesAndTypes](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/).