---
completed: true
version: 2.6.0
---
> [!Note]
> This page describes an internal API which is not intended to be used outside of the PyTorch codebase and can be modified or removed without notice.

> [!Warning]
> The ONNX exporter for TorchDynamo is a rapidly evolving beta technology.

## Overview
The ONNX exporter leverages TorchDynamo engine to hook into Python’s frame evaluation API and dynamically rewrite its bytecode into an FX Graph. The resulting FX Graph is then polished before it is finally translated into an ONNX graph.
>  TorchDynamo 将 Python bytecode 重写为 FX Graph
>  TorchDynamo-based exporter 将 FX Graph 优化，然后转换为 ONNX graph

The main advantage of this approach is that the [FX graph](https://pytorch.org/docs/stable/fx.html) is captured using bytecode analysis that preserves the dynamic nature of the model instead of using traditional static tracing techniques.
>  TorchDynamo-based exporter 的优势在于通过 bytecode analysis 捕获 FX graph ，保持了模型的动态性质

In addition, during the export process, memory usage is significantly reduced compared to the TorchScript-enabled exporter. See the [documentation](https://pytorch.org/docs/stable/onnx_dynamo_memory_usage.html) for more information.
>  此外，导出过程中的内存使用显著低于 TorchScript-based exporter

## Dependencies
The ONNX exporter depends on extra Python packages:

 - [ONNX](https://onnx.ai/)
 - [ONNX Script](https://onnxscript.ai/)

They can be installed through [pip](https://pypi.org/project/pip/):

```
pip install --upgrade onnx onnxscript
```

[onnxruntime](https://onnxruntime.ai/) can then be used to execute the model on a large variety of processors.

## A simple example
See below a demonstration of exporter API in action with a simple Multilayer Perceptron (MLP) as example:

```python
import torch
import torch.nn as nn

class MLPModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc0 = nn.Linear(8, 8, bias=True)
      self.fc1 = nn.Linear(8, 4, bias=True)
      self.fc2 = nn.Linear(4, 2, bias=True)
      self.fc3 = nn.Linear(2, 2, bias=True)

  def forward(self, tensor_x: torch.Tensor):
      tensor_x = self.fc0(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      tensor_x = self.fc1(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      tensor_x = self.fc2(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      output = self.fc3(tensor_x)
      return output

model = MLPModel()
tensor_x = torch.rand((97, 8), dtype=torch.float32)
onnx_program = torch.onnx.export(model, (tensor_x,), dynamo=True)
```

As the code above shows, all you need is to provide [`torch.onnx.export()`](https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export "torch.onnx.export") with an instance of the model and its input. The exporter will then return an instance of [`torch.onnx.ONNXProgram`](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.ONNXProgram "torch.onnx.ONNXProgram") that contains the exported ONNX graph along with extra information.
>  导出的流程即向 `torch.onnx.export()` 提供模型实例及其输入
>  exporter 将返回 `torch.onnx.ONNXProgram` 实例，该实例包含了导出的 ONNX graph 和额外信息

`onnx_program.optimize()` can be called to optimize the ONNX graph with constant folding and elimination of redundant operators. The optimization is done in-place.

```python
onnx_program.optimize()
```

>  `onnx_program.optimize()` 对 ONNX graph 执行常数折叠和冗余操作消除，优化会原地执行 (故没有返回值)

The in-memory model available through `onnx_program.model_proto` , which is an `onnx.ModelProto` object in compliance with the [ONNX IR spec](https://github.com/onnx/onnx/blob/main/docs/IR.md). The ONNX model may then be serialized into a [Protobuf file](https://protobuf.dev/) using the [`torch.onnx.ONNXProgram.save()`](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.ONNXProgram.save "torch.onnx.ONNXProgram.save") API.

```python
onnx_program.save("mlp.onnx")
```

>  `onnx_program.model_proto` 是模型对应的符合 ONNX IR 规范的 `onnx.ModelProto` 对象
>  `onnx_program.save()` 方法将该对象序列化为 Protobuf 文件

Two functions exist to export the model to ONNX based on TorchDynamo engine. They slightly differ in the way they produce the [`torch.export.ExportedProgram`](https://pytorch.org/docs/stable/export.html#torch.export.ExportedProgram "torch.export.ExportedProgram"). [`torch.onnx.dynamo_export()`](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.dynamo_export "torch.onnx.dynamo_export") was introduced with PyTorch 2.1 and [`torch.onnx.export()`](https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export "torch.onnx.export") was extended with PyTorch 2.5 to easily switch from TorchScript to TorchDynamo. To call the former function, the last line of the previous example can be replaced by the following one.

```python
onnx_program = torch.onnx.dynamo_export(model, tensor_x)
```

> [!Note]
> [`torch.onnx.dynamo_export()`](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.dynamo_export "torch.onnx.dynamo_export") will be deprecated in the future. Please use [`torch.onnx.export()`](https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export "torch.onnx.export") with the parameter `dynamo=True` instead.


## Inspecting the ONNX model using GUI
You can view the exported model using [Netron](https://netron.app/).

[![MLP model as viewed using Netron](https://pytorch.org/docs/stable/_images/onnx_dynamo_mlp_model.png)](https://pytorch.org/docs/stable/_images/onnx_dynamo_mlp_model.png)

## When the conversion fails
Function [`torch.onnx.export()`](https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export "torch.onnx.export") should called a second time with parameter `report=True`. A markdown report is generated to help the user to resolve the issue.

## API Reference
Some function in `torch.onnx`:

```python
def export() -> ONNXProgram
    """Export a torch.nn.Module to an ONNX graph"
```

Some classes in `torch.onnx`:

```python
class ONNXProgram:
    """A class to represent an ONNX program that is callable with torch tensors."""
```
