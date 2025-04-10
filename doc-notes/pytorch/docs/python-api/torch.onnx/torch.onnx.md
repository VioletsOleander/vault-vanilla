---
version: 2.6.0
completed: true
---
## Overview
[Open Neural Network eXchange (ONNX)](https://onnx.ai/) is an open standard format for representing machine learning models. The `torch.onnx` module captures the computation graph from a native PyTorch [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") model and converts it into an [ONNX graph](https://github.com/onnx/onnx/blob/main/docs/IR.md).
>  `torch.onnx` 模块捕获原生 PyTorch `torch.nn.Modeul` 模型的计算图，并将其转化为 ONNX graph

The exported model can be consumed by any of the many [runtimes that support ONNX](https://onnx.ai/supported-tools.html#deployModel), including Microsoft’s [ONNX Runtime](https://www.onnxruntime.ai/).
>  任意支持 ONNX 的 runtime 可以运行导出的 ONNX graph

**There are two flavors of ONNX exporter API that you can use, as listed below.** Both can be called through function [`torch.onnx.export()`](https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export "torch.onnx.export"). Next example shows how to export a simple model.
>  有两种 ONNX exporter API 可以使用，二者都通过 `torch.onnx.export()` 调用

```python
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 5)

    def forward(self, x):
        return torch.relu(self.conv1(x))

input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)

model = MyModel()

torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    "my_model.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamo=True             # True or False to select the exporter to use
)
```

Next sections introduces the two versions of the exporter.

## TorchDynamo-based ONNX Exporter
_The TorchDynamo-based ONNX exporter is the newest (and Beta) exporter for PyTorch 2.1 and newer_
>  PyTorch 2.1 及以上版本支持 TorchDynamo-based ONNX exporter

TorchDynamo engine is leveraged to hook into Python’s frame evaluation API and dynamically rewrite its bytecode into an FX Graph. The resulting FX Graph is then polished before it is finally translated into an ONNX graph.
>  TorchDynamo 将 Python bytecode 重写为 FX Graph
>  TorchDynamo-based exporter 将 FX Graph 优化，然后转换为 ONNX graph

The main advantage of this approach is that the [FX graph](https://pytorch.org/docs/stable/fx.html) is captured using bytecode analysis that preserves the dynamic nature of the model instead of using traditional static tracing techniques.
>  TorchDynamo-based exporter 的优势在于通过 bytecode analysis 捕获 FX graph ，保持了模型的动态性质

[Learn more about the TorchDynamo-based ONNX Exporter](https://pytorch.org/docs/stable/onnx_dynamo.html)

## TorchScript-based ONNX Exporter
_The TorchScript-based ONNX exporter is available since PyTorch 1.2.0_
>  PyTorch 1.2.0 及以上版本支持 TorchScript-based ONNX exporter

[TorchScript](https://pytorch.org/docs/stable/jit.html) is leveraged to trace (through [`torch.jit.trace()`](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace "torch.jit.trace")) the model and capture a static computation graph.
>  TorchScript-based exporter 利用 TorchScript (`torch.jit.trace`) 追踪模型计算，捕获静态计算图

As a consequence, the resulting graph has a couple limitations:

- It does not record any control-flow, like if-statements or loops;
- Does not handle nuances between `training` and `eval` mode;
- Does not truly handle dynamic inputs

>  如此得到的计算图的劣势有
>  - 不记录任何控制流
>  - 不处理 `training, eval` 模式的区别
>  - 不处理动态输入

As an attempt to support the static tracing limitations, the exporter also supports TorchScript scripting (through [`torch.jit.script()`](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script "torch.jit.script")), which adds support for data-dependent control-flow, for example. However, TorchScript itself is a subset of the Python language, so not all features in Python are supported, such as in-place operations.
>  TorchScript-based exporter 还支持 TorchScript scripting (`torch.jit.script()`)，以添加为数据依赖控制流的导出支持
>  但 TorchScript 本身是 Python 语言的子集，不支持所有 Python 特性，例如原地操作

[Learn more about the TorchScript-based ONNX Exporter](https://pytorch.org/docs/stable/onnx_torchscript.html)

## Contributing / Developing
The ONNX exporter is a community project and we welcome contributions. We follow the [PyTorch guidelines for contributions](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md), but you might also be interested in reading our [development wiki](https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter).
