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

> 使用 TableGen 定义 operation 的优势在于
> - 单一事实来源: operation 所有相关的信息都编码在同一个 record 中
> - 消除样板代码: 从 record 中自动生成操作数/属性/结果的获取方法、操作构建方法、操作验证方法以及许多其他实用方法，减少了定义新操作所需的样板代码
> - 促进自动生成功能: 这些操作信息记录的用途远不止于操作定义本身，我们可以利用它们来驱动许多其他组件的自动生成，例如计算图序列化

## TableGen Syntax 
We use TableGen as the language for specifying operation information. TableGen itself just provides syntax for writing records; the syntax and constructs allowed in a TableGen file (typically with the filename suffix `.td`) can be found [here](https://llvm.org/docs/TableGen/ProgRef.html).

- TableGen `class` is similar to C++ class; it can be templated and subclassed.
- TableGen `def` is similar to C++ object; it can be declared by specializing a TableGen `class` (e.g., `def MyDef : MyClass<...>;`) or completely independently (e.g., `def MyDef;`). It cannot be further templated or subclassed.
- TableGen `dag` is a dedicated type for directed acyclic graph of elements. A `dag` has one operator and zero or more arguments. Its syntax is `(operator arg0, arg1, argN)`. The operator can be any TableGen `def`; an argument can be anything, including `dag` itself. We can have names attached to both the operator and the arguments like `(MyOp:$op_name MyArg:$arg_name)`.

>  - TableGen 的 `class` 类似 C++ class，它可以被模板化以及继承
>  - TableGen 的 `def` 类似于 C++ 对象，它可以被声明，并且继承 `class`
>  - TableGen 的 `dag` 是一个表示有向无环图的数据类型，`dag` 实例有一个 operation 以及零个或更多 arguments，其语法为 `(operator arg0, arg1, argN)`；`operator` 可以是任意的 TableGen `def`，包括 `dag` 本身

Please see the [language reference](https://llvm.org/docs/TableGen/ProgRef.html) to learn about all the types and expressions supported by TableGen.

## Operation Definition 
MLIR defines several common constructs to help operation definition and provide their semantics via a special [TableGen backend](https://llvm.org/docs/TableGen/BackEnds.html#introduction): [`OpDefinitionsGen`](https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp). 
>  MLIR 用 TableGen 定义了一些常见的结构，并提供了一个相应的 TableGen 后端 `OpDefinitionsGen`

These constructs are defined in [`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td). The main ones are:

- The `Op` class: It is the main construct for defining operations. All facts regarding the operation are specified when specializing this class, with the help of the following constructs.
- The `Dialect` class: Operations belonging to one logical group are placed in the same dialect. The `Dialect` class contains dialect-level information.
- The `OpTrait` class hierarchy: They are used to specify special properties and constraints of the operation, including whether the operation has side effect or whether its output has the same shape as the input.
- The `ins`/`outs` marker: These are two special markers builtin to the `OpDefinitionsGen` backend. They lead to the definitions of operands/attributes and results respectively.
- The `TypeConstraint` class hierarchy: They are used to specify the constraints over operands or results. A notable subclass hierarchy is `Type`, which stands for constraints for common C++ types.
- The `AttrConstraint` class hierarchy: They are used to specify the constraints over attributes. A notable subclass hierarchy is `Attr`, which stands for constraints for attributes whose values are of common types.
- The `Property` class hierarchy: They are used to specify non-attribute-backed properties that are inherent to operations. These properties can have constraints imposed on them using the `predicate` field or the `ConfinedProp` class. The `PropConstraint` superclass of `Property` is used to describe constraints on properties in rewrite patterns.

>  这些结构都定义在 `mlir/include/mlir/IR/OpBase.td` 中，主要包括
>  - `Op` class: 所有的 operation `def` 都必须继承/实例化该 `class`
>  - `Dialect` class: 包含方言级别的信息
>  - `OpTrait` class: 用于指定 operation 的特殊属性和约束，例如 operation 是否存在副作用，以及输出和输入形状是否相同
>  - `ins/outs` marker: 这是两个内建在 `OpDefinitionsGen` 后端的 marker，分别用于定义 operands/attributes 和 results
>  - `TypeConstraint` class: 用于指定对 operands, results 的约束
>  - `AttrConstraint` class: 用于指定对 attributes 的约束
>  - `Property` class: 用于指定内建于 operation 的 non-attribute-backed properties

An operation is defined by specializing the `Op` class with concrete contents for all the fields it requires. For example, `tf.AvgPool` is defined as

```tablegen
def TF_AvgPoolOp : TF_Op<"AvgPool", [NoMemoryEffect]> {
  let summary = "Performs average pooling on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins
    TF_FpTensor:$value,

    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvertDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}
```

In the following we describe all the fields needed. Please see the definition of the `Op` class for the complete list of fields supported.

### Operation name 
The operation name is a unique identifier for the operation within MLIR, e.g., `tf.Add` for addition operation in the TensorFlow dialect. This is the equivalent of the mnemonic in assembly language. It is used for parsing and printing in the textual format. It is also used for pattern matching in graph rewrites.
>  operation name 用于解析和打印 operation，以及用于 graph rewrite 中的模式匹配

The full operation name is composed of the dialect name and the op name, with the former provided via the dialect and the latter provided as the second template parameter to the `Op` class.

### Operation documentation 
This includes both a one-line `summary` and a longer human-readable `description`. They will be used to drive automatic generation of dialect documentation. They need to be provided in the operation’s definition body:

```tablegen
let summary = "...";

let description = [{
...
}];
```

`description` should be written in Markdown syntax.

Placing the documentation at the beginning is recommended since it helps in understanding the operation.

> - Place documentation at the beginning of the operation definition.
> - The summary should be short and concise. It should be a one-liner starting with a capital letter and without trailing punctuation. Put expanded explanation in the description.

### Operation arguments 
There are three kinds of arguments: operands, attributes, and properties. Operands are runtime values produced by other ops; while attributes and properties are compile-time known constant values, including two categories:

1. Natural attributes: these attributes affect the behavior of the operations (e.g., padding for convolution);
2. Derived attributes: these attributes are not needed to define the operation but are instead derived from information of the operation. E.g., the output shape of type. This is mostly used for convenience interface generation or interaction with other frameworks/translation.
    
    All derived attributes should be materializable as an Attribute. That is, even though they are not materialized, it should be possible to store as an attribute.


>  operation 的 arguments 分为三类: operands, attributes, properties
>  - operands 为运行时值，应该由其他 op 产生
>  - attributes, properties 为编译时常量，包括两类:
>      1. Natural attributes: 影响 operation 的行为
>      2. Derived attributes: 定义 operation 不需要它们，但它们会从 operation 的信息中被推导出来，例如输出的形状

Properties are similar to attributes, except that they are not stored within the MLIR context but are stored inline with the operation.
>  properties 类似于 attributes，但它们不存储在 MLIR context 中，而是内联存储在 operation 中

Operands, attributes, and properties are specified inside the `dag`-typed `arguments`, led by `ins`:

```tablegen
let arguments = (ins
  <type-constraint>:$<operand-name>,
  ...
  <attr-constraint>:$<attr-name>,
  ...
  <property>:$<property-name>,
);
```

>  operation, attributes, properties 都由 `dag` 类型的 `arguments` 指定，形式如上

Here `<type-constraint>` is a TableGen `def` from the `TypeConstraint` class hierarchy. Similarly, `<attr-constraint>` is a TableGen `def` from the `AttrConstraint` class hierarchy and `<property>` is a subclass of `Property` (constraints can be imposed onto it using its `predicate` field or the `ConfinedProp` subclass).
>  `<type-constraint>` 为 `TypeConstaint` class hierarchy 中的 TableGen `def`
>  `<attr-constraint>` 为 `AttrConstraint` class hierarchy 中的 TableGen `def`
>  `<prpoerty>` 为 `Property` 的子类

There are no requirements on the relative order of operands and attributes; they can mix freely. The relative order of operands themselves matters. From each named argument a named getter will be generated that returns the argument with the return type (in the case of attributes the return type will be constructed from the storage type, while for operands it will be `Value`). Each attribute’s raw value (e.g., as stored) can also be accessed via generated `<name>Attr` getters for use in transformation passes where the more user-friendly return type is less suitable.
>  operands, attributes 的相对顺序不影响，operands 之间的相对顺序影响
>  每个有名字的 arguments 都会生成一个 getter, 该 getter 具有相应的返回类型 (对于 attribute，其返回类型将根据存储类型构造，对于 operand，其返回类型为 `Value`)

All the arguments should be named to:

- provide documentation,
- drive auto-generation of getter methods, and
- provide a handle to reference for other places like constraints.

#### Variadic operands 
To declare a variadic operand, wrap the `TypeConstraint` for the operand with `Variadic<...>`.

Normally operations have no variadic operands or just one variadic operand. For the latter case, it is easy to deduce which dynamic operands are for the static variadic operand definition. However, if an operation has more than one variable length operands (either optional or variadic), it would be impossible to attribute dynamic operands to the corresponding static variadic operand definitions without further information from the operation. Therefore, either the `SameVariadicOperandSize` or `AttrSizedOperandSegments` trait is needed to indicate that all variable length operands have the same number of dynamic values.

#### VariadicOfVariadic operands 
To declare a variadic operand that has a variadic number of sub-ranges, wrap the `TypeConstraint` for the operand with `VariadicOfVariadic<..., "<segment-attribute-name>">`.

The second field of the `VariadicOfVariadic` is the name of a `DenseI32ArrayAttr` argument that contains the sizes of the variadic sub-ranges. This attribute will be used when determining the size of sub-ranges, or when updating the size of sub-ranges.

#### Optional operands 
To declare an optional operand, wrap the `TypeConstraint` for the operand with `Optional<...>`.
>  要声明可选的 operand，用 `Optional<>` 包围 `TypeConstraint`

Normally operations have no optional operands or just one optional operand. For the latter case, it is easy to deduce which dynamic operands are for the static operand definition. However, if an operation has more than one variable length operands (either optional or variadic), it would be impossible to attribute dynamic operands to the corresponding static variadic operand definitions without further information from the operation. Therefore, either the `SameVariadicOperandSize` or `AttrSizedOperandSegments` trait is needed to indicate that all variable length operands have the same number of dynamic values.

#### Optional attributes 
To declare an optional attribute, wrap the `AttrConstraint` for the attribute with `OptionalAttr<...>`.
>  要声明可选的 attribute，用 `OptinoalAttr<>` 包围 `AttrConstraint`

#### Attributes with default values 
To declare an attribute with a default value, wrap the `AttrConstraint` for the attribute with `DefaultValuedAttr<..., "...">`.

The second parameter to `DefaultValuedAttr` should be a string containing the C++ default value. For example, a float default value should be specified as like `"0.5f"`, and an integer array default value should be specified as like `"{1, 2, 3}"`.

>  要声明带有默认值的 attribute，用 `DefaultValuedAttr<>` 包围 `AttrConstraint`
>  `DefaultValuedAttr` 的第二个参数应该是包含了 C++ 默认值的字符串

The generated operation printing function will not print default-valued attributes when the attribute value is equal to the default.

### Operation results 
Similar to operands, results are specified inside the `dag` -typed `results`, led by `outs`:

```tablegen
let results = (outs
  <type-constraint>:$<result-name>,
  ...
);
```

>  results 通过 `dag` 类型的 `results` 指定

### Operation traits and constraints 
Traits are operation properties that affect syntax or semantics. MLIR C++ models various traits in the `mlir::OpTrait` namespace.

Both operation traits, [interfaces](https://mlir.llvm.org/docs/Interfaces/#utilizing-the-ods-framework), and constraints involving multiple operands/attributes/results are provided as the third template parameter to the `Op` class. They should be deriving from the `OpTrait` class. See [Constraints](https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints) for more information.
>  operation traits, interfaces, 以及涉及了多个 operands/attributes/results 的 constraints 都作为 `Op` class 的第三个模板参数提供
>  它们都应继承 `OpTrait` class

### Builder methods 
For each operation, there are a few builders automatically generated based on the arguments and returns types. For example, given the following op definition:

```tablegen
def MyOp : ... {
  let arguments = (ins
    I32:$i32_operand,
    F32:$f32_operand,
    ...,

    I32Attr:$i32_attr,
    F32Attr:$f32_attr,
    ...
    I32Prop:$i32_prop,
    ...
  );

  let results = (outs
    I32:$i32_result,
    F32:$f32_result,
    ...
  );
}
```

>  MLIR 会根据 operation 的 `arguments, results` 自动生成一些 builders

The following builders are generated:

```c++
// All result-types/operands/properties/discardable attributes have one
// aggregate parameter. `Properties` is the properties structure of
// `MyOp`.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  TypeRange resultTypes,
                  ValueRange operands,
                  Properties properties,
                  ArrayRef<NamedAttribute> discardableAttributes = {});

// All result-types/operands/attributes have one aggregate parameter.
// Inherent properties and discardable attributes are mixed together in the
//  `attributes` dictionary.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  TypeRange resultTypes,
                  ValueRange operands,
                  ArrayRef<NamedAttribute> attributes);

// Each result-type/operand/attribute has a separate parameter. The parameters
// for attributes are of mlir::Attribute types.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result, ...,
                  Value i32_operand, Value f32_operand, ...,
                  IntegerAttr i32_attr, FloatAttr f32_attr, ...,
                  int32_t i32_prop);

// Each result-type/operand/attribute has a separate parameter. The parameters
// for attributes are raw values unwrapped with mlir::Attribute instances.
// (Note that this builder will not always be generated. See the following
// explanation for more details.)
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result, ...,
                  Value i32_operand, Value f32_operand, ...,
                  APInt i32_attr, StringRef f32_attr, ...,
                  int32_t i32_prop, ...);

// Each operand/attribute has a separate parameter but result type is aggregate.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  TypeRange resultTypes,
                  Value i32_operand, Value f32_operand, ...,
                  IntegerAttr i32_attr, FloatAttr f32_attr, ...,
                  int32_t i32_prop, ...);

// All operands/attributes have aggregate parameters.
// Generated if return type can be inferred.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ValueRange operands,
                  Properties properties,
                  ArrayRef<NamedAttribute> discardableAttributes);

// All operands/attributes have aggregate parameters.
// Generated if return type can be inferred. Uses the legacy merged attribute
// dictionary.
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ValueRange operands, ArrayRef<NamedAttribute> attributes);

// (And manually specified builders depending on the specific op.)
```

The first two forms provide basic uniformity so that we can create ops using the same form regardless of the exact op. This is particularly useful for implementing declarative pattern rewrites.

The third and fourth forms are good for use in manually written code, given that they provide better guarantee via signatures.

The fourth form will be generated if any of the op’s attribute has different `Attr.returnType` from `Attr.storageType` and we know how to build an attribute from an unwrapped value (i.e., `Attr.constBuilderCall` is defined.) Additionally, for the third form, if an attribute appearing later in the `arguments` list has a default value, the default value will be supplied in the declaration. This works for `BoolAttr`, `StrAttr`, `EnumAttr` for now and the list can grow in the future. So if possible, the default-valued attribute should be placed at the end of the `arguments` list to leverage this feature. (This behavior is essentially due to C++ function parameter default value placement restrictions.) Otherwise, the builder of the third form will still be generated but default values for the attributes not at the end of the `arguments` list will not be supplied in the builder’s signature.

ODS will generate a builder that doesn’t require the return type specified if

- Op implements InferTypeOpInterface interface;
- All return types are either buildable types or are the same as a given operand (e.g., `AllTypesMatch` constraint between operand and result);

And there may potentially exist other builders depending on the specific op; please refer to the [generated C++ file](https://mlir.llvm.org/docs/DefiningDialects/Operations/#run-mlir-tblgen-to-see-the-generated-content) for the complete list.

### Generated C++ code 
[OpDefinitionsGen](https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp) processes the op definition spec file and generates two files containing the corresponding C++ code: one for declarations, the other for definitions. The former is generated via the `-gen-op-decls` command-line option, while the latter is via the `-gen-op-defs` option.
>  `OpDefinitionGen` 接收 op 定义规范文件，生成包含对应 C++ 代码的两个文件，一个用于声明，一个用于定义

The definition file contains all the op method definitions, which can be included and enabled by defining `GET_OP_CLASSES`. For each operation, OpDefinitionsGen generates an operation class and an [operand adaptor](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operand-adaptors) class. Besides, it also contains a comma-separated list of all defined ops, which can be included and enabled by defining `GET_OP_LIST`.
>  定义文件包含所有 op 方法的定义
>  `OpDefinitionGen` 会为每个 operation class 生成一个 operand adaptor class
>  此外，生成的文件中还包含了所有已定义 operation 的列表

#### Class name and namespaces 
For each operation, its generated C++ class name is the symbol `def` ed with TableGen with dialect prefix removed. The first `_` serves as the delimiter. For example, for `def TF_AddOp`, the C++ class name would be `AddOp`. We remove the `TF` prefix because it is for scoping ops; other dialects may as well define their own `AddOp`s.
>  对于所有的 operation，其生成的 C++ 类名会去除 `_` 之前方言前缀

The namespaces of the generated C++ class will come from the dialect’s `cppNamespace` field. For example, if a dialect’s `cppNamespace` is `A::B`, then an op of that dialect will be placed in `namespace A { namespace B { ... } }`. If a dialect does not specify a `cppNamespace`, we then use the dialect’s name as the namespace.
>  C++ 类的命名空间来自于方言的 `cppNamespace`

This means the qualified name of the generated C++ class does not necessarily match exactly with the operation name as explained in [Operation name](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-name). This is to allow flexible naming to satisfy coding style requirements.

#### Operand adaptors 
For each operation, we automatically generate an _operand adaptor_. This class solves the problem of accessing operands provided as a list of `Value` s without using “magic” constants. The operand adaptor takes a reference to an array of `Value` and provides methods with the same names as those in the operation class to access them. For example, for a binary arithmetic operation, it may provide `.lhs()` to access the first operand and `.rhs()` to access the second operand.
>  每个 operation 的 operand adaptor 用于解决在不使用 “魔法” 常量的情况下访问以 `Value` 的列表形式提供的 operands 的问题

The operand adaptor class lives in the same namespace as the operation class, and has the name of the operation followed by `Adaptor` as well as an alias `Adaptor` inside the op class.

Operand adaptors can be used in function templates that also process operations:

```c++
template <typename BinaryOpTy>
std::pair<Value, Value> zip(BinaryOpTy &&op) {
  return std::make_pair(op.lhs(), op.rhs());;
}

void process(AddOp op, ArrayRef<Value> newOperands) {
  zip(op);
  zip(Adaptor<AddOp>(newOperands));
  /*...*/
}
```

## Constraints 
Constraint is a core concept in table-driven operation definition: operation verification and graph operation matching are all based on satisfying constraints. So both the operation definition and rewrite rules specification significantly involve writing constraints. 
>  operation 验证和 graph rewrite 都必须满足 constraints，故 operation 定义和 rewrite rule 定义都涉及到定义 constraints

We have the `Constraint` class in [`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td) as the common base class for all constraints.

An operation’s constraint can cover different range; it may

- Only concern a single attribute (e.g. being a 32-bit integer greater than 5),
- Multiple operands and results (e.g., the 1st result’s shape must be the same as the 1st operand), or
- Intrinsic to the operation itself (e.g., having no side effect).

>  `OpBase.td` 中定义了所有 constraint 的基类: `Constraint`
>  operation 的 constraint 可以覆盖不同的方面，例如:
>  - 仅涉及单个 attribute -> single-entity constraint
>  - 涉及多个 operands 和 results -> multi-entity constraint
>  - 针对 operation 自身 -> traits

We call them as single-entity constraint, multi-entity constraint, and traits, respectively.

### Single-entity constraint 
Constraints scoped to a single operand, attribute, or result are specified at the entity’s declaration place as described in [Operation arguments](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments) and [Operation results](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-results).
>  single-entity constraint 即针对单个 operand, attribute, result 的约束
>  这类约束需要在 arguments, results 声明时指定

To help modelling constraints of common types, a set of `TypeConstraint` s are created; they are the `Type` subclass hierarchy. It includes `F32` for the constraints of being a float, `TensorOf<[F32]>` for the constraints of being a float tensor, and so on.
>  MLIR 预定义了一系列 `TypeConstraint` 类，用于表示对常见类型的约束
>  这些 `TypeConstraint` 类都是 `Type` 的子类，一些例子包括: `F32`, `Tensorof<[F32]>`

Similarly, a set of `AttrConstraint` s are created for helping modelling constraints of common attribute kinds. They are the `Attr` subclass hierarchy. It includes `F32Attr` for the constraints of being a float attribute, `F32ArrayAttr` for the constraints of being a float array attribute, and so on.
>  类似地，MLIR 预定义了一系列 `AttrConstraint` 类，用于表示对常见属性的约束，它们都是 `Attr` 的子类，一些例子包括 `F32Attr, F32ArrayAttr`

### Multi-entity constraint 
Constraints involving more than one operand/attribute/result are quite common on operations, like the element type and shape relation between operands and results. These constraints should be specified as the `Op` class template parameter as described in [Operation traits and constraints](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-traits-and-constraints).

Multi-entity constraints are modeled as `PredOpTrait` (a subclass of `OpTrait`) in [`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td).A bunch of constraint primitives are provided to help specification. See [`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td) for the complete list.

>  涉及多个 operand/attribute/result 的 constraint 称为 multi-entity constraint，这类约束主要涉及 operand/attribute/results 之间的类型和形状关系等
>  multi-entity constraint 应该通过 `Op` 类的模板参数指定 (即指定为 `OpTrait` 的子类 `PredOpTrait`)

### Trait 
Traits are intrinsic properties of the operation like having side effect or not, commutative or not, whether is a terminator, etc. These constraints should be specified as the `Op` class template parameter as described in [Operation traits and constraints](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-traits-and-constraints).

Traits are modeled as `NativeOpTrait` (a subclass of `OpTrait`) in [`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td). They are backed and will be translated into the corresponding C++ `mlir::OpTrait` classes.

>  Traits 表示内建于 operation 的属性，应该通过 `Op` 的第三个模板参数指定 (即指定为 `OpTrait` 的子类 `NaitiveOpTrait`)

## Attribute Definition 
An attribute is a compile-time known constant of an operation.
>  attribute 是 operation 的编译时已知常量

ODS provides attribute wrappers over C++ attribute classes. There are a few common C++ [attribute classes](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Attributes.h) defined in MLIR’s core IR library and one is free to define dialect-specific attribute classes. ODS allows one to use these attributes in TableGen to define operations, potentially with more fine-grained constraints. For example, `StrAttr` directly maps to `StringAttr`; `F32Attr` / `F64Attr` requires the `FloatAttr` to additionally be of a certain bitwidth.
>  MLIR 预定义了一些 C++ attribute 类
>  ODS 包装了这些 C++ attribute 类，通常是一一对应的，也可以有更细的粒度，例如 `StrAttr` 对应 `StringAttr`，`F32Attr/F64Attr` 对应 `FloatAttr`

ODS attributes are defined as having a storage type (corresponding to a backing `mlir::Attribute` that _stores_ the attribute), a return type (corresponding to the C++ _return_ type of the generated helper getters) as well as a method to convert between the internal storage and the helper method.
>  ODS 中的 attributes 具有存储类型和返回类型

### Attribute decorators 
There are a few important attribute adapters/decorators/modifiers that can be applied to ODS attributes to specify common additional properties like optionality, default values, etc.:

- `DefaultValuedAttr`: specifies the [default value](https://mlir.llvm.org/docs/DefiningDialects/Operations/#attributes-with-default-values) for an attribute.
- `OptionalAttr`: specifies an attribute as [optional](https://mlir.llvm.org/docs/DefiningDialects/Operations/#optional-attributes).
- `ConfinedAttr`: adapts an attribute with [further constraints](https://mlir.llvm.org/docs/DefiningDialects/Operations/#confining-attributes).
- `AllAttrOf`: adapts an attribute with [multiple constraints](https://mlir.llvm.org/docs/DefiningDialects/Operations/#combining-constraints).

>  一些常见的 attribute 修饰符如下，它们可以用于指定 attribute 的额外属性
>  - `DefaultValueAttr`
>  - `OptionalAttr`
>  - `ConfinedAttr`
>  - `AllAttrof`
