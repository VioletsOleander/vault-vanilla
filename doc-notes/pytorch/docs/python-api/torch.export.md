> [!Warning]
> This feature is a prototype under active development and there WILL BE BREAKING CHANGES in the future.

## Overview
[`torch.export.export()`](https://docs.pytorch.org/docs/stable/export.html#torch.export.export "torch.export.export") takes a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") and produces a traced graph representing only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion, which can subsequently be executed with different outputs or serialized.
>  `torch.export.export()` 接收一个 `nn.Module` ，以提前编译的方式生成一个仅表示函数张量计算的追踪图 (`torch.export.ExportedProgram`)
>  追踪图可以用不同的输出执行或被序列化

```python
import torch
from torch.export import export

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

example_args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: torch.export.ExportedProgram = export(
    Mod(), args=example_args
)
print(exported_program)
```

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[10, 10]", y: "f32[10, 10]"):
            # code: a = torch.sin(x)
            sin: "f32[10, 10]" = torch.ops.aten.sin.default(x)

            # code: b = torch.cos(y)
            cos: "f32[10, 10]" = torch.ops.aten.cos.default(y)

            # code: return a + b
            add: f32[10, 10] = torch.ops.aten.add.Tensor(sin, cos)
            return (add,)

    Graph signature:
        ExportGraphSignature(
            input_specs=[
                InputSpec(
                    kind=<InputKind.USER_INPUT: 1>,
                    arg=TensorArgument(name='x'),
                    target=None,
                    persistent=None
                ),
                InputSpec(
                    kind=<InputKind.USER_INPUT: 1>,
                    arg=TensorArgument(name='y'),
                    target=None,
                    persistent=None
                )
            ],
            output_specs=[
                OutputSpec(
                    kind=<OutputKind.USER_OUTPUT: 1>,
                    arg=TensorArgument(name='add'),
                    target=None
                )
            ]
        )
    Range constraints: {}
```

`torch.export` produces a clean intermediate representation (IR) with the following invariants. More specifications about the IR can be found [here](https://docs.pytorch.org/docs/stable/export.ir_spec.html#export-ir-spec).

- **Soundness**: It is guaranteed to be a sound representation of the original program, and maintains the same calling conventions of the original program.
- **Normalized**: There are no Python semantics within the graph. Submodules from the original programs are inlined to form one fully flattened computational graph.
- **Graph properties**: The graph is purely functional, meaning it does not contain operations with side effects such as mutations or aliasing. It does not mutate any intermediate values, parameters, or buffers.
- **Metadata**: The graph contains metadata captured during tracing, such as a stacktrace from user’s code.

>  `torch.export` 生成具有以下不变式的 IR:
>  - 正确性: 保证是对源程序的正确表示，保留源程序的相同调用约定
>  - 规范化: 图中没有 Python 语义，原始程序中的子模块被内联，以形成要给完全展平的计算图
>  - 图属性: 图是完全函数式的，即它不包含具有 side effects 的 operation，例如 mutations or aliasing, operation 不会修改任何中间值、参数或缓冲区
>  - 元数据: 图包含在追踪过程中的元数据，例如用户代码的堆栈跟踪

Under the hood, `torch.export` leverages the following latest technologies:

- **TorchDynamo (`torch._dynamo`)** is an internal API that uses a CPython feature called the Frame Evaluation API to safely trace PyTorch graphs. This provides a massively improved graph capturing experience, with much fewer rewrites needed in order to fully trace the PyTorch code.
- **AOT Autograd** provides a functionalized PyTorch graph and ensures the graph is decomposed/lowered to the ATen operator set.
- **Torch FX (`torch.fx`)** is the underlying representation of the graph, allowing flexible Python-based transformations.

>  `torch.export` 利用了以下技术:
>  - TorchDynamo (`torch._dynamo`): 一个内部 API，利用 CPython 的 Frame Evaluation API 来追踪 PyTorch graph
>  - AOT Autograd: 提供了函数化的 PyTorch graph，并确保该图被分解/降级为 ATen 算子集
>  - Torch FX (`torch.fx`): graph 的底层表示形式，允许灵活的基于 Python 的转换

### Existing frameworks
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile "torch.compile") also utilizes the same PT2 stack as `torch.export`, but is slightly different:

- **JIT vs. AOT**: [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile "torch.compile") is a JIT compiler whereas which is not intended to be used to produce compiled artifacts outside of deployment.
- **Partial vs. Full Graph Capture**: When [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile "torch.compile") runs into an untraceable part of a model, it will “graph break” and fall back to running the program in the eager Python runtime. In comparison, `torch.export` aims to get a full graph representation of a PyTorch model, so it will error out when something untraceable is reached. Since `torch.export` produces a full graph disjoint from any Python features or runtime, this graph can then be saved, loaded, and run in different environments and languages.
- **Usability tradeoff**: Since [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile "torch.compile") is able to fallback to the Python runtime whenever it reaches something untraceable, it is a lot more flexible. `torch.export` will instead require users to provide more information or rewrite their code to make it traceable.

>  `torch.compile()` 也使用了和 `torch.export` 相同的 underlying technologies，但存在略微不同:
>  - JIT vs AOT: `torch.compile()` 是一个 JIT 编译器，其设计目的是在部署时使用，以生成 compiled artifacts (`torch.export()` 则是 AOT 编译器)
>  - Partial vs Full Graph Capture: 当 `torch.compile()` 遇到模型中不可追踪的部分时，它会中断图，然后回退到在 eager Python runtime 中执行程序; `torch.export()` 则目的是获取 PyTorch 模型的完整图表示，因此在遇到不可追踪的部分时，会直接报错退出，`torch.export()` 生成的是与任何 Python 特性和运行时无关的完整图，故该图可以被保存，并在不同的环境和语言下加载、运行
>  - Usability tradeoff: `torch.compile()` 在遇到不可追踪的内容会回退到 Python runtime，故会更加灵活; `torch.export()` 则要求用户提供更多信息，或者重写代码，使得代码可追踪

Compared to [`torch.fx.symbolic_trace()`](https://docs.pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace "torch.fx.symbolic_trace"), `torch.export` traces using TorchDynamo which operates at the Python bytecode level, giving it the ability to trace arbitrary Python constructs not limited by what Python operator overloading supports. 
>  和 `torch.fx.symbolic_trace()` 相比，`torch.export` 使用 TorchDynamo 进行追踪，故可以追踪任意的 Python 构造，而不受 Python 运算符重载支持的限制

Additionally, `torch.export` keeps fine-grained track of tensor metadata, so that conditionals on things like tensor shapes do not fail tracing. In general, `torch.export` is expected to work on more user programs, and produce lower-level graphs (at the `torch.ops.aten` operator level). Note that users can still use [`torch.fx.symbolic_trace()`](https://docs.pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace "torch.fx.symbolic_trace") as a preprocessing step before `torch.export`.
>  此外，`torch.export` 能细粒度追踪张量元数据，故像张量形状这样的条件不会导致跟踪失败
>  总体而言，`torch.export` 预计可以处理更多的用户程序，并生成更低级的图 (在 `torch.ops.aten` 算子级别)
>  用户仍然可以用 `torch.fx.symbolic_trace()` 作为 `torch.export` 之前的预处理步骤

Compared to [`torch.jit.script()`](https://docs.pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script "torch.jit.script"), `torch.export` does not capture Python control flow or data structures, but it supports more Python language features than TorchScript (as it is easier to have comprehensive coverage over Python bytecodes). The resulting graphs are simpler and only have straight line control flow (except for explicit control flow operators).
>  和 `torch.jit.script()` 相比，`torch.export` 不会捕获 Python 控制流或数据结构，但相较于 TorchScript，支持更多的 Python 语言特性 (因为它更容易全面覆盖 Python 字节码)
>  `torch.export` 生成的图更简单，且仅包含直线控制流 (除了显式的控制流算子)

Compared to [`torch.jit.trace()`](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace "torch.jit.trace"), `torch.export` is sound: it is able to trace code that performs integer computation on sizes and records all of the side-conditions necessary to show that a particular trace is valid for other inputs.
>  和 `torch.jit.trace()` 相比，`torch.export` 是可靠的: 它能够跟踪对大小进行整数运算的代码，并且记录所有必要的侧条件，以表明特定的追踪对于其他输入也是有效的

## Exporting a PyTorch Model
### An Example
The main entrypoint is through [`torch.export.export()`](https://docs.pytorch.org/docs/stable/export.html#torch.export.export "torch.export.export"), which takes a callable ([`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module"), function, or method) and sample inputs, and captures the computation graph into an [`torch.export.ExportedProgram`](https://docs.pytorch.org/docs/stable/export.html#torch.export.ExportedProgram "torch.export.ExportedProgram"). An example:

```python
import torch
from torch.export import export

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
        a = self.conv(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))

example_args = (torch.randn(1, 3, 256, 256),)
example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}

exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)
print(exported_program)
```

```
ExportedProgram:
    class GraphModule(torch.nn.Module):
    def forward(self, p_conv_weight: "f32[16, 3, 3, 3]", p_conv_bias: "f32[16]", x: "f32[1, 3, 256, 256]", constant: "f32[1, 16, 256, 256]"):
            # code: a = self.conv(x)
            conv2d: "f32[1, 16, 256, 256]" = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias, [1, 1], [1, 1])

            # code: a.add_(constant)
            add_: "f32[1, 16, 256, 256]" = torch.ops.aten.add_.Tensor(conv2d, constant)

            # code: return self.maxpool(self.relu(a))
            relu: "f32[1, 16, 256, 256]" = torch.ops.aten.relu.default(add_)
            max_pool2d: "f32[1, 16, 85, 85]" = torch.ops.aten.max_pool2d.default(relu, [3, 3], [3, 3])
            return (max_pool2d,)

Graph signature:
    ExportGraphSignature(
        input_specs=[
            InputSpec(
                kind=<InputKind.PARAMETER: 2>,
                arg=TensorArgument(name='p_conv_weight'),
                target='conv.weight',
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.PARAMETER: 2>,
                arg=TensorArgument(name='p_conv_bias'),
                target='conv.bias',
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.USER_INPUT: 1>,
                arg=TensorArgument(name='x'),
                target=None,
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.USER_INPUT: 1>,
                arg=TensorArgument(name='constant'),
                target=None,
                persistent=None
            )
        ],
        output_specs=[
            OutputSpec(
                kind=<OutputKind.USER_OUTPUT: 1>,
                arg=TensorArgument(name='max_pool2d'),
                target=None
            )
        ]
    )
Range constraints: {}
```

Inspecting the `ExportedProgram`, we can note the following:

- The [`torch.fx.Graph`](https://docs.pytorch.org/docs/stable/fx.html#torch.fx.Graph "torch.fx.Graph") contains the computation graph of the original program, along with records of the original code for easy debugging.
- The graph contains only `torch.ops.aten` operators found [here](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml) and custom operators, and is fully functional, without any inplace operators such as `torch.add_`.
- The parameters (weight and bias to conv) are lifted as inputs to the graph, resulting in no `get_attr` nodes in the graph, which previously existed in the result of [`torch.fx.symbolic_trace()`](https://docs.pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace "torch.fx.symbolic_trace").
- The [`torch.export.ExportGraphSignature`](https://docs.pytorch.org/docs/stable/export.html#torch.export.ExportGraphSignature "torch.export.ExportGraphSignature") models the input and output signature, along with specifying which inputs are parameters.
- The resulting shape and dtype of tensors produced by each node in the graph is noted. For example, the `convolution` node will result in a tensor of dtype `torch.float32` and shape `(1, 16, 256, 256)`.

>  `ExportedProgram` 中:
>  - `torch.fx.Graph` 包含了原始程序的计算图，以及原始代码的记录，易于调试
>  - 图中仅包含了 `torch.ops.aten` 算子和自定义算子，并且是纯函数式的，补办然任意的就地运算符例如 `torch.add_`
>  - 参数 (卷积的权重和偏置) 被提升为图的输入，因此图中不再有 `torch.fx.symbolic_trace()` 中会有的 `get_attr` 节点
>  - `torch.export.ExportGraphSignature` 模型化了输入和输出签名，并指定了那些输入是参数
>  - 图中每个节点生成的张量的形状和数据类型都被记录了下来，例如 `convolution` 节点将生成一个数据类型为 `torch.float32`，形状为 `(1, 16, 256, 256)` 的张量