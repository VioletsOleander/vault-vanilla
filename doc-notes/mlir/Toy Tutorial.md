# Chapter 1: Toy Language and AST

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

# Chapter 2: Emitting Basic MLIR
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

> [! info]
> 静态单赋值 (Static Single Assignment) 是编译器 IR 中常用的一种形式，在该形式下，每个变量在生命周期内只能被赋值一次
> 该形式有助于简化编译器优化，同时是数据流分析更简单
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
>  mlir-opt tool (测试编译器传递的工具) 默认不再输出中包含位置信息
>  `-mlir-print-debuginfo` 用于指定包含位置信息

### Opaque API 
MLIR is designed to allow all IR elements, such as attributes, operations, and types, to be customized. At the same time, IR elements can always be reduced to the above fundamental concepts. This allows MLIR to parse, represent, and [round-trip](https://mlir.llvm.org/getting_started/Glossary/#round-trip) IR for _any_ operation. 
>  MLIR 在设计上允许自定义所有的 IR 元素，例如属性、操作、类型
>  注意所有的 IR 元素都可以按照之前的基本概念分类
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

> [! info]
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

> [!info]
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

> [!info]
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

##### `toy.print` 
The current form of `toy.print` is a little verbose. There are a lot of additional characters that we would like to strip away. Let’s begin by thinking of what a good format of `toy.print` would be, and see how we can implement it. Looking at the basics of `toy.print` we get:

```mlir
toy.print %5 : tensor<*xf64> loc(...)
```

Here we have stripped much of the format down to the bare essentials, and it has become much more readable. To provide a custom assembly format, an operation can either override the `hasCustomAssemblyFormat` field for a C++ format, or the `assemblyFormat` field for the declarative format. Let’s look at the C++ variant first, as this is what the declarative format maps to internally.

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

With the C++ implementation defined, let’s see how this can be mapped to the [declarative format](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format). The declarative format is largely composed of three different components:

- Directives
    - A type of builtin function, with an optional set of arguments.
- Literals
    - A keyword or punctuation surrounded by ``.
- Variables
    - An entity that has been registered on the operation itself, i.e. an argument(attribute or operand), result, successor, etc. In the `PrintOp` example above, a variable would be `$input`.

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