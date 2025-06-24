---
completed: true
---
This document will present a quickstart to adding graph rewrites. We shall start by defining an operation, showing multiple ways to define the rewrite using patterns, as well as defining the rewrite using a graph walker (note: using patterns and the rewrite engine is preferred, showing the walker is for demonstration purposes).

See [MLIR specification](https://mlir.llvm.org/docs/LangRef/) for more information about MLIR, the structure of the IR, operations, etc. See [Table-driven Operation Definition](https://mlir.llvm.org/docs/DefiningDialects/Operations/) and [Declarative Rewrite Rule](https://mlir.llvm.org/docs/DeclarativeRewrites/) for the detailed explanation of all available mechanisms for defining operations and rewrites in a table-driven manner.

## Adding operation 
An operation in MLIR is specified using a definition in [TableGen](https://llvm.org/docs/TableGen/index.html) file. TableGen is a modeling tool to specify the ops and the C++ code to interact with these operations are generated from. 
>  MLIR 中，Dialect 和 Dialect 的 operation 一般都通过 TableGen 定义

To define an operation one needs to specify:

- The operation name. This name is a unique identifier of the operation within MLIR. Most operations are within a dialect, so for example one could have `tfl.add` to represent the add operation in the TensorFlow Lite dialect. Instead of repeating the dialect in the op definition, a base class for the op dialect is commonly created that prepends the dialect namespace given an op name.
- The traits of the operation. These allow you to specify traits of the operation, such as whether it has side effects or whether it should be verified that the operands and result types are the same. These are backed by C++ traits that perform the verification.
- The arguments of the operation. These are the input operands (values at runtime produced by other ops) and attributes (compile time known constant values that affect the behavior of the op) that are the inputs of/define the behavior of the operation. The input operands may be named, the attributes must be named.
- The result(s) of the operation. These may again named or not.
- Documentation of the operation. This includes a one-line summary as well as a longer human-readable description of the operation.
- Dialect specific information. Additional information could be added to the operation definition that are only used by dialect specific drivers. These are ignored by the main op and doc generators, but could be used in, say, the translation from a dialect to another representation.

>  定义 operation 时，用户需要指定:
>  - 唯一的 operation 名称
>  - operation 的 `traits`，例如 operation 是否存在 side effect，是否应该验证其输入操作数和输出操作数的类型相同
>  - operation 的 `arguments`，包括了 input operands 和 attributes (其中 attributes 是编译时就已知的常数，用于影响 operation 的行为)，input operands 可以有名字，attributes 必须有名字
>  - operation 的 `results`
>  - operation 的文档，包括一行的 `summary` 和更长的 `description`
>  - Dialect 规范信息，一般用在 operation 到另一个 Dialect 的转换过程中

```tablegen
def TFL_LeakyReluOp: TFL_Op<TFL_Dialect, "leaky_relu",
                            [NoMemoryEffect, SameValueType]>,
                     Results<(outs Tensor)> {
  let arguments = (ins
    F32Tensor:$x,
    // Slope of the activation function at x < 0.
    F32Attr:$alpha
  );

  let summary = "Leaky ReLU operator";
  let description = [{
    Element-wise Leaky ReLU operator
      x -> x >= 0 ? x : (alpha * x)
  }];

  // TFLite specific attribute that is used when generating the output
  // flatbuffer.
  let hasOptions = 1;
}
```

Note in the above the result types and inputs are specified in different ways, one by way of trait and the other by way of let. It is possible to specify both in either way.
>  上例中特殊的一点是采用了 traits 来定义 `results`，使用 `let results = ...` 和 traits 来定义都是可以的，`arguments` 也是同理

Operations can also have custom parser, printer, builder, verifier, constant folder, or canonicalizer. These require specifying additional C++ methods to invoke for additional functionality. For example, if an operation is marked to have a folder, the constant folder also needs to be added, e.g.,:

```c++
OpFoldResult SpecificOp::fold(ArrayRef<Attribute> constOperands) {
  if (unable_to_fold)
    return {};
  ....
  return val;
}
```

>  除了上述标准的一些 fields 之外，operation 在定义中还可以自行指定自定义的 `parser, builder, printer, verifier, constatn foloer, canonicalizer`，它们的定义需要通过 inline 的 C++ 代码提供

## Adding patterns 
There are multiple forms of graph rewrite that can be performed in MLIR. One of the most common is DAG tile to DAG tile rewrite. Patterns provide a concise way to express this transformation as a pair of source pattern to match and resultant pattern. There are both the C++ classes to represent this transformation, as well as the patterns in TableGen from which these can be generated.
>  MLIR 的转换通过模式来定义
>  在 MLIR 中，需要转换时，需要定义一对源模式和结果模式，源模式用于匹配，结果模式用于生成
>  模式本质都是 C++ 类，抽象地表示了转换过程，模式也可以在 TableGen 中定义

### TableGen patterns 
Let us continue with LeakyRelu. To map from TensorFlow’s `LeakyRelu` to TensorFlow Lite’s `LeakyRelu`:

```tablegen
def : Pat<(TF_LeakyReluOp $arg, F32Attr:$a), (TFL_LeakyReluOp $arg, $a)>
```

The pattern is specified by instantiating a `Pat` with a source and result DAG. The arguments in the source pattern is captured and can be used in the result pattern. This is a simple pattern as we have a 1:1 mapping and the attribute does not need to be transformed (e.g., both have a floating point attribute for alpha). The names of the attributes specified in the pattern is for matching/referencing and need not match the original attribute name in the op definition but the order of arguments of the dags do need to match.

>  上述示例展示了 TableGen 定义下，从 `TF_LeakyReluOp` 到 `TFL_LeakyReluOp` 的转换模式
>  这样的转换模式适合简单的一对一映射，本质上就是把输入 op 的名称和参数名称替换为输出 op 的名称和参数名称

To specify a pattern, both the source and resultant ops need to be defined using TableGen.
>  在 TableGen 中定义转换模式要求源 op 和目的 op 也都定义在 TableGen 中

If this were a more advance pattern that the current framework could not express as destination then one could use a general native code fallback method. This consists of defining a pattern as well as adding a C++ function to perform the replacement:

```tablegen
def createTFLLeakyRelu : NativeCodeCall<
    "createTFLLeakyRelu($_builder, $0.getDefiningOp(), $1, $2)">;

def : Pat<(TF_LeakyReluOp:$old_value, $arg, F32Attr:$a),
          (createTFLLeakyRelu $old_value, $arg, $a)>;
```

```c++
static Value createTFLLeakyRelu(PatternRewriter &rewriter, Operation *op,
                                Value operand, Attribute attr) {
  return rewriter.create<mlir::TFL::LeakyReluOp>(
      op->getLoc(), operands[0].getType(), /*arg=*/operands[0],
      /*alpha=*/cast<FloatAttr>(attrs[0]));
}
```

This allows for arbitrarily complex builders. Input pattern side one can express multi-op patterns with constraints on input operands and attributes. But input patterns cannot yet express constraints across multiple operands/attributes.

>  如果转换需要一些逻辑，则需要 `Pat` 的输出 Op 替换为一个自定义 C++ 函数，在该函数中实现逻辑
>  C++ 函数有固定的签名，其主要功能应该是利用 `PatternRewirter &rewiter` 创建一个新的目标 op

### Register the pattern 
The file containing the patterns need to be processed using `mlir-tblgen` `-gen-rewriters` during compilation time. It can be invoked with the following configuration in CMake:

```cmake
set(LLVM_TARGET_DEFINITIONS <name-of-the-td-file>)
mlir_tablegen(<name-of-the-generated-inc-file> -gen-rewriters)
add_public_tablegen_target(<name-of-the-cmake-target>)
```

>  定义了转换模式的 TableGen 文件，通过 `mlir-tblgen -gen-rewriters` 就可以生成 C++ 代码

Then you can `#include` the generated file in any C++ implementation file you like. (You will also need to make sure the library depends on the CMake target defined in the above.) The generated file will have a `populateWithGenerated( RewritePatternSet &patterns)` function that you can use to collect all the generated patterns inside `patterns` and then use `patterns` in any pass you would like.

>  之后用户就可以 `#include` 生成的文件
>  通过 `mlir-tblgen -gen-rewriters` 生成的文件会定义一个 `populateWithGenerated(RewritePatternSet &patterns)` 函数，用于将文件中所有定义的模式加入到传入的 `patterns` 中
>  用户只需要自行声明好 `RewritePatternSet &patterns`，然后调用该函数，就可以初始化好 `patterns`

### Simple C++ `matchAndRewrite` style specifications 
Many simple rewrites can be expressed with a `matchAndRewrite` style of pattern, e.g. when converting a multiply by a power of two into a shift. For these cases, the you can define the pattern as a simple function:

```c++
static LogicalResult
convertTFLeakyRelu(TFLeakyReluOp op, PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<TFL::LeakyReluOp>(
      op, op->getResult(0).getType(), op->getOperand(0),
      /*alpha=*/op->getAttrOfType<FloatAttr>("alpha"));
  return success();
}

void populateRewrites(RewritePatternSet &patternSet) {
  // Add it to a pattern set.
  patternSet.add(convertTFLeakyRelu);
}
```

>  另一种风格是函数式风格，将转换逻辑封装在一个自定义的 C++ 函数中，或者说将这个转换函数视作之前的转换模式
>  这个转换函数同样有固定的签名，其签名风格和 `mathAndRewirte` 方法一致，接收 `op, rewiter`
>  `RewriterPatternSet` 的 `add` 方法不仅可以添加转换模式 (C++ 类)，也可以添加 C++ 函数，故定义好转换函数，同样可以将它加入模式集

ODS provides a simple way to define a function-style canonicalization for your operation. In the TableGen definition of the op, specify `let hasCanonicalizeMethod = 1;` and then implement the `canonicalize` method in your .cpp file:

```c++
// Example from the CIRCT project which has a variadic integer multiply.
LogicalResult circt::MulOp::canonicalize(MulOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  APInt value;

  // mul(x, c) -> shl(x, log2(c)), where c is a power of two.
  if (inputs.size() == 2 && matchPattern(inputs.back(), m_RConstant(value)) &&
      value.isPowerOf2()) {
    auto shift = rewriter.create<rtl::ConstantOp>(op.getLoc(), op.getType(),
                                                  value.exactLogBase2());
    auto shlOp =
        rewriter.create<comb::ShlOp>(op.getLoc(), inputs[0], shift);
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(),
                                       ArrayRef<Value>(shlOp));
    return success();
  }

  return failure();
}
```

>  如果想把转换作为 canonicalization 的一部分，可以将上述风格的转换函数定义为特定 op 的 `canonicalize` 方法，然后在 TableGen 中为该 op 声明 `hasCanonicalizeMethod=1` 即可

However, you may want the full generality of canonicalization patterns, for that you can specify an arbitrary list of `RewritePattern`s.

### Fully general C++ `RewritePattern` specifications 
In case ODS patterns and `matchAndRewrite` -style functions are not sufficient you can also specify rewrites as a general set of `RewritePattern` s:

```c++
struct ConvertTFLeakyRelu : public RewritePattern {
  ConvertTFLeakyRelu(MLIRContext *context)
      : RewritePattern("tf.LeakyRelu", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TFL::LeakyReluOp>(
        op, op->getResult(0).getType(), op->getOperand(0),
        /*alpha=*/op->getAttrOfType<FloatAttr>("alpha"));
    return success();
  }
};
```

In the C++ rewrite the static benefit of the rewrite pattern is specified at construction. While in the pattern generator a simple heuristic is currently employed based around the number of ops matched and replaced.

>  最后一种方法是直接在 C++ 中定义转换类，转换类应该继承 `RewriterPattern`，并定义 `mathAndRewrite` 方法
>  同样，这个转换类可以直接被添加到 `RewritePatternSet` 中

The above rule did not capture the matching operands/attributes, but in general the `match` function in a multi-step rewrite may populate and return a `PatternState` (or class derived from one) to pass information extracted during matching to the rewrite. A single-step rewrite with the `matchAndRewrite` function has the benefit of being able to directly use any values created when matching; removing the need for `PatternState`.

## Testing 
MLIR uses [lit](https://llvm.org/docs/CommandGuide/lit.html) (LLVM Integrated Testing) tool for performing testing. Testing is performed by way of creating the input IR file, running a transformation and then verifying the output IR. C++ unit tests are the exception, with the IR transformation serving as the core testing mechanism. This results in fewer binaries that need to be built (and linked) and forces to focus on the representation as an important piece.
>  MLIR 基于 LLVM Integrated Testing 工具执行测试
>  测试的方式是: 创建输入 IR 文件，执行转换，验证输出 IR，故核心关注点是 IR 转换

For the legalization transform above we would have a test (probably as part of the legalization pass test in TensorFlow Lite) such as:

```mlir
// RUN: mlir-opt -tfl-legalize-tf %s | FileCheck %s

func.func @LeakyRelu(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %2 = "tf.LeakyRelu"(%arg0) {alpha: 0.1} : (tensor<1xf32>) -> tensor<1xf32>
  return %2: tensor<1xf32>

// CHECK-LABEL: LeakyRelu
// CHECK:  %0 = "tfl.leaky_relu"(%arg0) {alpha: 1.000000e-01} : (tensor<1xf32>) -> tensor<1xf32>
}
```

The RUN command at the top results in running the `mlir-opt` binary (which is compiler writer tool to exercise different registered passes) to invoke the optimization pass this transform was added as part of on the current file and to verify its output using `FileCheck`. `FileCheck` is textual output verifier. In particular it uses the CHECK expressions to verify the given output is produced.
>  `// RUN` 这一行定义了应该运行什么命令
>  `// CHECK` 这一行定义了 `FileCheck` 应该如何验证输出 IR

There can be multiple RUN commands with different corresponding CHECK prefixes. And in addition multiple independent tests separated by `// -----` and `mlir-opt` invoked with `-split-input-file` flag. This is especially useful for error testing.
>  测试与测试之间用 `// -----` 分离

This results in very simple, directed testing without need to work around constant propagation or other, unrelated, optimization passes.

## Adding optimization pass 
Optimization passes that do not fit/are difficult to specify in the above structure can be specified as general iterations across modules/functions. See [Writing a Pass](https://mlir.llvm.org/docs/PassManagement/) for a general overview and introduction to optimization passes in MLIR.