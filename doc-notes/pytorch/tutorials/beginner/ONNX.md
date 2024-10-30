>2.5.0+cu124
# Introduction to ONNX
Authors: [Thiago Crepaldi](https://github.com/thiagocrepaldi),

[Open Neural Network eXchange (ONNX)](https://onnx.ai/) is an open standard format for representing machine learning models. The `torch.onnx` module provides APIs to capture the computation graph from a native PyTorch [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "(in PyTorch v2.5)") model and convert it into an [ONNX graph](https://github.com/onnx/onnx/blob/main/docs/IR.md).

The exported model can be consumed by any of the many [runtimes that support ONNX](https://onnx.ai/supported-tools.html#deployModel), including Microsoft’s [ONNX Runtime](https://www.onnxruntime.ai/).
> `torch.onnx` 模块提供 API 从原生 `torch.nn.Module` 模块捕获计算图，然后将其转化为 ONNX 图
> 导出的模型可以由 ONNX runtime 接受

Note
Currently, there are two flavors of ONNX exporter APIs, but this tutorial will focus on the `torch.onnx.dynamo_export`.

The TorchDynamo engine is leveraged to hook into Python’s frame evaluation API and dynamically rewrite its bytecode into an [FX graph](https://pytorch.org/docs/stable/fx.html). The resulting FX Graph is polished before it is finally translated into an [ONNX graph](https://github.com/onnx/onnx/blob/main/docs/IR.md).

The main advantage of this approach is that the [FX graph](https://pytorch.org/docs/stable/fx.html) is captured using bytecode analysis that preserves the dynamic nature of the model instead of using traditional static tracing techniques.
> 主要讨论的 ONNX 导出 API 为 `torch.onnx.dynamo_export`
> Dynamo 调用 Python 的帧评估 API，动态将字节码重写为 FX 图，FX 图经过微调，最终转化为 ONNX 图
> 该方法的优势在于：FX 图是通过字节码分析而捕获的，这保持了模型的动态性质

## Dependencies
PyTorch 2.1.0 or newer is required.

The ONNX exporter depends on extra Python packages:

> - [ONNX](https://onnx.ai/) standard library
> - [ONNX Script](https://onnxscript.ai/) library that enables developers to author ONNX operators, functions and models using a subset of Python in an expressive, and yet simple fashion
> - [ONNX Runtime](https://onnxruntime.ai/) accelerated machine learning library.

They can be installed through [pip](https://pypi.org/project/pip/):

```
pip install --upgrade onnx onnxscript onnxruntime
```

To validate the installation, run the following commands:

```python
import torch
print(torch.__version__)

import onnxscript
print(onnxscript.__version__)

from onnxscript import opset18  # opset 18 is the latest (and only) supported version for now

import onnxruntime
print(onnxruntime.__version__)
```

Each import must succeed without any errors and the library versions must be printed out.