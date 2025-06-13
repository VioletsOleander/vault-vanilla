## Overview
FX is a toolkit for developers to use to transform `nn.Module` instances. FX consists of three main components: a **symbolic tracer,** an **intermediate representation**, and **Python code generation**. 
>  FX 开发者用于转换 `torch.nn.Module` 的工具包
>  FX 包括三个主要组件: 1. 符号追踪器 2. 中间表示 3. Python 代码生成

A demonstration of these components in action:

```python
import torch

# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

module = MyModule()

from torch.fx import symbolic_trace

# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)

# High-level intermediate representation (IR) - Graph representation
print(symbolic_traced.graph)
"""
graph():
    %x : [num_users=1] = placeholder[target=x]
    %param : [num_users=1] = get_attr[target=param]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
    %linear : [num_users=1] = call_module[target=linear](args = (%add,), kwargs = {})
    %clamp : [num_users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
    return clamp
"""

# Code generation - valid Python code
print(symbolic_traced.code)
"""
def forward(self, x):
    param = self.param
    add = x + param;  x = param = None
    linear = self.linear(add);  add = None
    clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
    return clamp
"""
```

>  上述例子中
>  `troch.fx.symbolic_trace()` 接收 `nn.Module` ，返回 `torch.fx.GraphModule`，可以看到该函数直接将 `Module` 转化为了图表示 (High-level IR)
>  `torch.fx.GraphModule.graph` 获取图表示，`torch.fx.GraphModule.code` 获取 Python 代码
>  可以看到 `GraphModule.code` 和原来的代码不相同，但是计算流程是一致的

The **symbolic tracer** performs “symbolic execution” of the Python code. It feeds fake values, called Proxies, through the code. Operations on theses Proxies are recorded. More information about symbolic tracing can be found in the [`symbolic_trace()`](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace "torch.fx.symbolic_trace") and [`Tracer`](https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer "torch.fx.Tracer") documentation.
>  符号追踪器执行 Python 代码的 “符号执行”:
>  symbolic tracer 将称为 Proxies 的虚拟值传递给代码，然后记录对代码这些 Proxies 的 operations

The **intermediate representation** is the container for the operations that were recorded during symbolic tracing. It consists of a list of Nodes that represent function inputs, callsites (to functions, methods, or [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") instances), and return values. More information about the IR can be found in the documentation for [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph "torch.fx.Graph"). The IR is the format on which transformations are applied.
>  中间表示是 symbolic tracing 过程中，对 operations 的记录形式
>  中间表示由一组 Node 组成，这些 Node 表示函数输入、调用点 (指向函数、方法、或 `nn.Module` 实例)、返回值

**Python code generation** is what makes FX a Python-to-Python (or Module-to-Module) transformation toolkit. For each Graph IR, we can create valid Python code matching the Graph’s semantics. This functionality is wrapped up in [`GraphModule`](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule "torch.fx.GraphModule"), which is a [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") instance that holds a [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph "torch.fx.Graph") as well as a `forward` method generated from the Graph.
>  Python 代码生成是使 RX 称为 Python-to-Python (或 Module-to-Module) 转换工具包的关键部分
>  对于每个 Graph IR，我们都可以创建匹配 Graph 语义的 Python 代码
>  该功能被封装在 `GraphModule` 中，`GraphModule` 是一个 `torch.nn.Modules` 实例，它包含一个 `Graph` ，以及从 Graph 自动生成的 `forward` 方法

Taken together, this pipeline of components (symbolic tracing -> intermediate representation -> transforms -> Python code generation) constitutes the Python-to-Python transformation pipeline of FX. 
>  综合来看，这一组件管道: symbolic tracing -> intermediate representation -> transforms -> Python code generation 构成了 FX 的 Python-to-Python 转换流水线

In addition, these components can be used separately. For example, symbolic tracing can be used in isolation to capture a form of the code for analysis (and not transformation) purposes. Code generation can be used for programmatically generating models, for example from a config file. There are many uses for FX!
>  此外，这些组件可以单独使用

Several example transformations can be found at the [examples](https://github.com/pytorch/examples/tree/master/fx) repository.

## Writing Transformations
What is an FX transform? Essentially, it’s a function that looks like this.

```python
import torch
import torch.fx

def transform(m: nn.Module,
              tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
    # Step 1: Acquire a Graph representing the code in `m`

    # NOTE: torch.fx.symbolic_trace is a wrapper around a call to
    # fx.Tracer.trace and constructing a GraphModule. We'll
    # split that out in our transform to allow the caller to
    # customize tracing behavior.
    graph : torch.fx.Graph = tracer_class().trace(m)

    # Step 2: Modify this Graph or create a new one
    graph = ...

    # Step 3: Construct a Module to return
    return torch.fx.GraphModule(m, graph)
```

>  FX transform 本质上是一个函数，它接收 `nn.Module, torch.fx.Tracer`，返回 `nn.Module`
>  `transform` 会首先利用 `Tracer` 获取输入 ` nn.Module ` 的图表示 (`torch.fx.Graph`)；然后对图表示进行修改；最后从图表示构建一个 ` torch.fx.GraphModule ` 并返回
>  因此，本质上就是抓计算图 -> 修改图 -> 返回图 (以 `GraphModule` 的形式)

Your transform will take in a [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module"), acquire a [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph "torch.fx.Graph") from it, do some modifications, and return a new [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module"). 

You should think of the [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") that your FX transform returns as identical to a regular [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") – you can pass it to another FX transform, you can pass it to TorchScript, or you can run it. Ensuring that the inputs and outputs of your FX transform are a [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") will allow for composability.
>  我们应该将 FX transform 返回的 `torch.nn.Module` 视作与普通的 `torch.nn.Module` 完全相同 —— 可以将它传递给另一个 FX transform, 或传递给 TorchScript, 或者直接运行它

Note
It is also possible to modify an existing [`GraphModule`](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule "torch.fx.GraphModule") instead of creating a new one, like so:

```python
import torch
import torch.fx

def transform(m : nn.Module) -> nn.Module:
    gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)

    # Modify gm.graph
    # <...>

    # Recompile the forward() method of `gm` from its Graph
    gm.recompile()

    return gm
```

Note that you MUST call [`GraphModule.recompile()`](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.recompile "torch.fx.GraphModule.recompile") to bring the generated `forward()` method on the `GraphModule` in sync with the modified [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph "torch.fx.Graph").
>  注意，在修改了 `torch.fx.GraphModule` 之后，需要调用 `GraphModule.recompile()` 以将 `GraphModule` 的 `forward()` 方法和修改后的 `Graph` 同步

Given that you’ve passed in a [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") that has been traced into a [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph "torch.fx.Graph"), there are now two primary approaches you can take to building a new [`Graph`](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph "torch.fx.Graph").
