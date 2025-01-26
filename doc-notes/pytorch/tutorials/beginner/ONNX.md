---
completed: 
version: 2.5.0+cu124
---
# Introduction to ONNX
Last Updated: Sep 05, 2024 | Last Verified: Nov 05, 2024
Authors: [Thiago Crepaldi](https://github.com/thiagocrepaldi),

[Open Neural Network eXchange (ONNX)](https://onnx.ai/) is an open standard format for representing machine learning models. The `torch.onnx` module provides APIs to capture the computation graph from a native PyTorch [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "(in PyTorch v2.5)") model and convert it into an [ONNX graph](https://github.com/onnx/onnx/blob/main/docs/IR.md).

The exported model can be consumed by any of the many [runtimes that support ONNX](https://onnx.ai/supported-tools.html#deployModel), including Microsoft’s [ONNX Runtime](https://www.onnxruntime.ai/).

> `torch.onnx` 模块提供 API 从原生 `torch.nn.Module` 模块捕获计算图，然后将其转化为 ONNX 图
> 导出的模型可以由 ONNX runtime 接受

> [!note] Note
> Currently, there are two flavors of ONNX exporter APIs, but this tutorial will focus on the `torch.onnx.dynamo_export`.

The TorchDynamo engine is leveraged to hook into Python’s frame evaluation API and dynamically rewrite its bytecode into an [FX graph](https://pytorch.org/docs/stable/fx.html). The resulting FX Graph is polished before it is finally translated into an [ONNX graph](https://github.com/onnx/onnx/blob/main/docs/IR.md).

The main advantage of this approach is that the [FX graph](https://pytorch.org/docs/stable/fx.html) is captured using bytecode analysis that preserves the dynamic nature of the model instead of using traditional static tracing techniques.

> 我们主要讨论的 ONNX 导出 API 为 `torch.onnx.dynamo_export`
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

# Export a PyTorch model to ONNX
Last Updated: Nov 06, 2023 | Last Verified: Nov 05, 2024
**Author**: [Thiago Crepaldi](https://github.com/thiagocrepaldi)

> [!note] Note
> As of PyTorch 2.1, there are two versions of ONNX Exporter.
> 
> - `torch.onnx.dynamo_export` is the newest (still in beta) exporter based on the TorchDynamo technology released with PyTorch 2.0
> - `torch.onnx.export` is based on TorchScript backend and has been available since PyTorch 1.2.0

In the [60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html), we had the opportunity to learn about PyTorch at a high level and train a small neural network to classify images. In this tutorial, we are going to expand this to describe how to convert a model defined in PyTorch into the ONNX format using TorchDynamo and the `torch.onnx.dynamo_export` ONNX exporter.
>  我们讨论如何使用 TorchDynamo 和 `torch.onnx.dynamo_export` 将 PyTorch 定义下的模型转化为 ONNX 格式

While PyTorch is great for iterating on the development of models, the model can be deployed to production using different formats, including [ONNX](https://onnx.ai/) (Open Neural Network Exchange)!

ONNX is a flexible open standard format for representing machine learning models which standardized representations of machine learning allow them to be executed across a gamut of hardware platforms and runtime environments from large-scale cloud-based supercomputers to resource-constrained edge devices, such as your web browser and phone.

In this tutorial, we’ll learn how to:

1. Install the required dependencies.
2. Author a simple image classifier model.
3. Export the model to ONNX format.
4. Save the ONNX model in a file.
5. Visualize the ONNX model graph using [Netron](https://github.com/lutzroeder/netron).
6. Execute the ONNX model with ONNX Runtime
7. Compare the PyTorch results with the ones from the ONNX Runtime.

## 1. Install the required dependencies
Because the ONNX exporter uses `onnx` and `onnxscript` to translate PyTorch operators into ONNX operators, we will need to install them.
>  ONNX 导出器使用 `onnx` 和 `onnxscript` 将 PyTorch 算子转化为 ONNX 算子

```
> pip install onnx
> pip install onnxscript
```

## 2. Author a simple image classifier model
Once your environment is set up, let’s start modeling our image classifier with PyTorch, exactly like we did in the [60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## 3. Export the model to ONNX format
Now that we have our model defined, we need to instantiate it and create a random 32x32 input. Next, we can export the model to ONNX format.

```python
torch_model = MyModel()
torch_input = torch.randn(1, 1, 32, 32)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
```

As we can see, we didn’t need any code change to the model. The resulting ONNX model is stored within `torch.onnx.ONNXProgram` as a binary protobuf file.

>  我们实例化模型，并创建一个随机的输入，然后直接调用 `torch.onnx.dynamo_export` 将模型导出为 ONNX 格式
>  导出得到的 ONNX 模型以二进制 protobuf 格式存储在 `torch.onnx.ONNXProgram` 中 (也就是我们得到了一个 `torch.onnx.ONNXProgram` 对象)

## 4. Save the ONNX model in a file
Although having the exported model loaded in memory is useful in many applications, we can save it to disk with the following code:

```python
onnx_program.save("my_image_classifier.onnx")
```

>  执行完上述代码后，导出的模型 `onnx_program` 已经存储在了内存中，我们可以进一步调用其 `save` 方法将它存储到磁盘中

You can load the ONNX file back into memory and check if it is well formed with the following code:

```python
import onnx
onnx_model = onnx.load("my_image_classifier.onnx")
onnx.checker.check_model(onnx_model)
```

>  `onnx.load` 函数可以用于加载磁盘中的 ONNX 模型到内存中
>  我们进一步使用 `onnx.checker.check_model` 检查模型的一致性

## 5. Visualize the ONNX model graph using Netron
Now that we have our model saved in a file, we can visualize it with [Netron](https://github.com/lutzroeder/netron). Netron can either be installed on macos, Linux or Windows computers, or run directly from the browser. Let’s try the web version by opening the following link: [https://netron.app/](https://netron.app/).

[![../../_images/netron_web_ui.png](https://pytorch.org/tutorials/_images/netron_web_ui.png)](https://pytorch.org/tutorials/_images/netron_web_ui.png)

Once Netron is open, we can drag and drop our `my_image_classifier.onnx` file into the browser or select it after clicking the **Open model** button.

[![../../_images/image_clossifier_onnx_modelon_netron_web_ui.png](https://pytorch.org/tutorials/_images/image_clossifier_onnx_modelon_netron_web_ui.png)](https://pytorch.org/tutorials/_images/image_clossifier_onnx_modelon_netron_web_ui.png)

And that is it! We have successfully exported our PyTorch model to ONNX format and visualized it with Netron.

## 6. Execute the ONNX model with ONNX Runtime
The last step is executing the ONNX model with ONNX Runtime, but before we do that, let’s install ONNX Runtime.

```
> pip install onnxruntime
```

The ONNX standard does not support all the data structure and types that PyTorch does, so we need to adapt PyTorch input’s to ONNX format before feeding it to ONNX Runtime. In our example, the input happens to be the same, but it might have more inputs than the original PyTorch model in more complex models.
>  ONNX 标准不支持 PyTorch 的所有数据结构和类型，因此在将 PyTorch 的输入提供给 ONNX Runtime 之前，需要先将其转化为 ONNX 支持的格式  (例如 Numpy 数组)

ONNX Runtime requires an additional step that involves converting all PyTorch tensors to Numpy (in CPU) and wrap them on a dictionary with keys being a string with the input name as key and the numpy tensor as the value.
>  并且，ONNX Runtime 要求将所有的 PyTorch 张量转化为 Numpy 张量 (CPU 上) 后，将它们包装在一个字典中，字典的键是输入名称的字符串，值是 Numpy 张量

Now we can create an _ONNX Runtime Inference Session_, execute the ONNX model with the processed input and get the output. In this tutorial, ONNX Runtime is executed on CPU, but it could be executed on GPU as well.
>  处理好输出后，我们可以创建一个 ONNX Runtime 推理会话，使用处理后的输入执行 ONNX 模型，得到输出
>  本教程中，ONNX Runtime 在 CPU 上执行，也可以在 GPU 上执行

```python
import onnxruntime

# 直接调用模型的方法转化 torch 输入
onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)
print(f"Input length: {len(onnx_input)}")
print(f"Sample input: {onnx_input}")

# 指定 CPU 作为计算设备
ort_session = onnxruntime.InferenceSession("./my_image_classifier.onnx", providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}

onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
```

>  `to_numpy()` 函数中：`detach()` 方法创建一个新的张量，它与原张量共享内存但不参与梯度计算，确保在转化过程中断开与计算图的连接，避免自动微分时不必要的计算和错误；`cpu()` 将张量从 GPU 迁移到 CPU 上；`numpy()` 方法将 torch 张量转化为 numpy 数组

## 7. Compare the PyTorch results with the ones from the ONNX Runtime
The best way to determine whether the exported model is looking good is through numerical evaluation against PyTorch, which is our source of truth.

For that, we need to execute the PyTorch model with the same input and compare the results with ONNX Runtime’s. Before comparing the results, we need to convert the PyTorch’s output to match ONNX’s format.
>  为了验证 ONNX 模型的正确性，我们需要比较 PyTorch 模型和 ONNX 模型的结果，在比较之前，需要将 PyTorch 的输出转变格式以匹配 ONNX 格式

```python
torch_outputs = torch_model(torch_input)
torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"Sample output: {onnxruntime_outputs}")
```

## Conclusion
That is about it! We have successfully exported our PyTorch model to ONNX format, saved the model to disk, viewed it using Netron, executed it with ONNX Runtime and finally compared its numerical results with PyTorch’s.

# Extending the ONNX Registry
 Last Updated: Jul 22, 2024 | Last Verified: Nov 05, 2024
**Authors:** Ti-Tai Wang ([titaiwang@microsoft.com](mailto:titaiwang%40microsoft.com))

## Overview
This tutorial is an introduction to ONNX registry, which empowers users to implement new ONNX operators or even replace existing operators with a new implementation.

During the model export to ONNX, the PyTorch model is lowered to an intermediate representation composed of [ATen operators](https://pytorch.org/docs/stable/torch.compiler_ir.html). While ATen operators are maintained by PyTorch core team, it is the responsibility of the ONNX exporter team to independently implement each of these operators to ONNX through [ONNX Script](https://onnxscript.ai/). The users can also replace the behavior implemented by the ONNX exporter team with their own implementation to fix bugs or improve performance for a specific ONNX runtime.
>  将模型导出到 ONNX 时，PyTorch 模型会先降级到一个由 ATen 算子组成的中间表示
>  ATen 算子由 PyTorch 核心团队维护，ONNX 导出器团队负责通过 ONNX Script 使用 ONNX 算子实现 ATen 算子
>  用户可以使用自己的实现替换 ONNX 导出器团队的实现，以针对特定 ONNX 运行时提高性能

The ONNX Registry manages the mapping between PyTorch operators and the ONNX operators counterparts and provides APIs to extend the registry.
>  ONNX 注册表管理 PyTorch 算子和 ONNX 算子之间的对应，并且提供了拓展注册表的 API

In this tutorial, we will cover three scenarios that require extending the ONNX registry with custom operators:

- Unsupported ATen operators
- Custom operators with existing ONNX Runtime support
- Custom operators without ONNX Runtime support

## Unsupported ATen operators
Although the ONNX exporter team does their best efforts to support all ATen operators, some of them might not be supported yet. In this section, we will demonstrate how you can add unsupported ATen operators to the ONNX Registry.
>  部分 ATen 算子可能尚没有 ONNX 格式的支持，本节介绍如何将尚未支持的 ATen 算子的 ONNX 实现添加到 ONNX 注册表

> [!Note]
> The steps to implement unsupported ATen operators are the same to replace the implementation of an existing ATen operator with a custom implementation. Because we don’t actually have an unsupported ATen operator to use in this tutorial, we are going to leverage this and replace the implementation of `aten::add.Tensor` with a custom implementation the same way we would if the operator was not present in the ONNX Registry.

>  实现不支持的 ATen 算子的流程和用自定义实现替换现存的支持的 ATen 算子的流程是一样的

When a model cannot be exported to ONNX due to an unsupported operator, the ONNX exporter will show an error message similar to:
>  当模型因为某个未支持的算子而无法导出到 ONNX 格式时，ONNX 导出器的报错信息会类似：

```
RuntimeErrorWithDiagnostic: Unsupported FX nodes: {'call_function': ['aten.add.Tensor']}.
```

The error message indicates that the fully qualified name of unsupported ATen operator is `aten::add.Tensor`. The fully qualified name of an operator is composed of the namespace, operator name, and overload following the format `namespace::operator_name.overload`.
>  错误信息包含了不支持 ATen 算子的完全限定名称 `aten::add.Tensor`，一个算子的完全限定名称包括命名空间名称、算子名称，以及重载 (例如 `Tensor` 表示该算子是针对张量的特定重载版本)，算子的完全限定名称格式为 ` namespace:: operator_name.overload `

To add support for an unsupported ATen operator or to replace the implementation for an existing one, we need:

- The fully qualified name of the ATen operator (e.g. `aten::add.Tensor`). This information is always present in the error message as show above.
- The implementation of the operator using [ONNX Script](https://github.com/microsoft/onnxscript). ONNX Script is a prerequisite for this tutorial. Please make sure you have read the [ONNX Script tutorial](https://github.com/microsoft/onnxscript/blob/main/docs/tutorial/index.md) before proceeding.

>  要注册自己实现的算子，我们需要提供：
>  - 要替代的 ATen 算子的完全限定名称 (例如 `aten::add.Tensor)
>  - 该算子使用 ONNX Script 的实现

Because `aten::add.Tensor` is already supported by the ONNX Registry, we will demonstrate how to replace it with a custom implementation, but keep in mind that the same steps apply to support new unsupported ATen operators.

This is possible because the `OnnxRegistry` allows users to override an operator registration. We will override the registration of `aten::add.Tensor` with our custom implementation and verify it exists.
>  `OnnxRegistry` 允许用户覆盖某个算子的注册，我们将用自定义实现覆盖 `aten::add.Tensor` 的注册

```python
import torch
import onnxruntime
import onnxscript
from onnxscript import opset18  # opset 18 is the latest (and only) supported version for now

class Model(torch.nn.Module):
    def forward(self, input_x, input_y):
        return torch.ops.aten.add(input_x, input_y)  # generates a aten::add.Tensor node

input_add_x = torch.randn(3, 4)
input_add_y = torch.randn(3, 4)
aten_add_model = Model()


# Now we create a ONNX Script function that implements ``aten::add.Tensor``.
# The function name (e.g. ``custom_aten_add``) is displayed in the ONNX graph, so we recommend to use intuitive names.
custom_aten = onnxscript.values.Opset(domain="custom.aten", version=1)

# NOTE: The function signature must match the signature of the unsupported ATen operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.
@onnxscript.script(custom_aten)
def custom_aten_add(input_x, input_y, alpha: float = 1.0):
    input_y = opset18.Mul(input_y, alpha)
    return opset18.Add(input_x, input_y)


# Now we have everything we need to support unsupported ATen operators.
# Let's register the ``custom_aten_add`` function to ONNX registry, and export the model to ONNX again.
onnx_registry = torch.onnx.OnnxRegistry()
onnx_registry.register_op(
    namespace="aten", op_name="add", overload="Tensor", function=custom_aten_add
    )
print(f"aten::add.Tensor is supported by ONNX registry: \
      {onnx_registry.is_registered_op(namespace='aten', op_name='add', overload='Tensor')}"
      )
export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry)
onnx_program = torch.onnx.dynamo_export(
    aten_add_model, input_add_x, input_add_y, export_options=export_options
    )
```

Now let’s inspect the model and verify the model has a `custom_aten_add` instead of `aten::add.Tensor`. The graph has one graph node for `custom_aten_add`, and inside of it there are four function nodes, one for each operator, and one for constant attribute.

```python
# graph node domain is the custom domain we registered
assert onnx_program.model_proto.graph.node[0].domain == "custom.aten"
assert len(onnx_program.model_proto.graph.node) == 1
# graph node name is the function name
assert onnx_program.model_proto.graph.node[0].op_type == "custom_aten_add"
# function node domain is empty because we use standard ONNX operators
assert {node.domain for node in onnx_program.model_proto.functions[0].node} == {""}
# function node name is the standard ONNX operator name
assert {node.op_type for node in onnx_program.model_proto.functions[0].node} == {"Add", "Mul", "Constant"}
```

This is how `custom_aten_add_model` looks in the ONNX graph using Netron:

[![../../_images/custom_aten_add_model.png](https://pytorch.org/tutorials/_images/custom_aten_add_model.png)](https://pytorch.org/tutorials/_images/custom_aten_add_model.png)

Inside the `custom_aten_add` function, we can see the three ONNX nodes we used in the function (`CastLike`, `Add`, and `Mul`), and one `Constant` attribute:

[![../../_images/custom_aten_add_function.png](https://pytorch.org/tutorials/_images/custom_aten_add_function.png)](https://pytorch.org/tutorials/_images/custom_aten_add_function.png)

This was all that we needed to register the new ATen operator into the ONNX Registry. As an additional step, we can use ONNX Runtime to run the model, and compare the results with PyTorch.

# Use ONNX Runtime to run the model, and compare the results with PyTorch
onnx_program.save("./custom_add_model.onnx")
ort_session = onnxruntime.InferenceSession(
    "./custom_add_model.onnx", providers=['CPUExecutionProvider']
    )

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_input = onnx_program.adapt_torch_inputs_to_onnx(input_add_x, input_add_y)
onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

torch_outputs = aten_add_model(input_add_x, input_add_y)
torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    [torch.testing.assert_close](https://pytorch.org/docs/stable/testing.html#torch.testing.assert_close "torch.testing.assert_close")(torch_output, [torch.tensor](https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor "torch.tensor")(onnxruntime_output))

## Custom operators with existing ONNX Runtime support
In this case, the user creates a model with standard PyTorch operators, but the ONNX runtime (e.g. Microsoft’s ONNX Runtime) can provide a custom implementation for that kernel, effectively replacing the existing implementation in the ONNX Registry. Another use case is when the user wants to use a custom implementation of an existing ONNX operator to fix a bug or improve performance of a specific operator. To achieve this, we only need to register the new implementation with the existing ATen fully qualified name.

In the following example, we use the `com.microsoft.Gelu` from ONNX Runtime, which is not the same `Gelu` from ONNX spec. Thus, we register the Gelu with the namespace `com.microsoft` and operator name `Gelu`.

Before we begin, let’s check whether `aten::gelu.default` is really supported by the ONNX registry.

onnx_registry = [torch.onnx.OnnxRegistry](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.OnnxRegistry "torch.onnx.OnnxRegistry")()
print(f"aten::gelu.default is supported by ONNX registry: \
    {onnx_registry.is_registered_op(namespace='aten', op_name='gelu', overload='default')}")

In our example, `aten::gelu.default` operator is supported by the ONNX registry, so `onnx_registry.is_registered_op()` returns `True`.

class CustomGelu([torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")):
    def forward(self, input_x):
        return torch.ops.aten.gelu(input_x)

# com.microsoft is an official ONNX Runtime namspace
custom_ort = onnxscript.values.Opset(domain="com.microsoft", version=1)

# NOTE: The function signature must match the signature of the unsupported ATen operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.
@onnxscript.script(custom_ort)
def custom_aten_gelu(input_x, approximate: str = "none"):
    # We know com.microsoft::Gelu is supported by ONNX Runtime
    # It's only not supported by ONNX
    return custom_ort.Gelu(input_x)

onnx_registry = [torch.onnx.OnnxRegistry](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.OnnxRegistry "torch.onnx.OnnxRegistry")()
onnx_registry.register_op(
    namespace="aten", op_name="gelu", overload="default", function=custom_aten_gelu)
export_options = [torch.onnx.ExportOptions](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.ExportOptions "torch.onnx.ExportOptions")(onnx_registry=onnx_registry)

aten_gelu_model = CustomGelu()
input_gelu_x = [torch.randn](https://pytorch.org/docs/stable/generated/torch.randn.html#torch.randn "torch.randn")(3, 3)

onnx_program = [torch.onnx.dynamo_export](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.dynamo_export "torch.onnx.dynamo_export")(
    aten_gelu_model, input_gelu_x, export_options=export_options
    )

Let’s inspect the model and verify the model uses op_type `Gelu` from namespace `com.microsoft`.

Note

`custom_aten_gelu()` does not exist in the graph because functions with fewer than three operators are inlined automatically.

# graph node domain is the custom domain we registered
assert onnx_program.model_proto.graph.node[0].domain == "com.microsoft"
# graph node name is the function name
assert onnx_program.model_proto.graph.node[0].op_type == "Gelu"

The following diagram shows `custom_aten_gelu_model` ONNX graph using Netron, we can see the `Gelu` node from module `com.microsoft` used in the function:

![../../_images/custom_aten_gelu_model.png](https://pytorch.org/tutorials/_images/custom_aten_gelu_model.png)

That is all we need to do. As an additional step, we can use ONNX Runtime to run the model, and compare the results with PyTorch.

onnx_program.save("./custom_gelu_model.onnx")
ort_session = onnxruntime.InferenceSession(
    "./custom_gelu_model.onnx", providers=['CPUExecutionProvider']
    )

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_input = onnx_program.adapt_torch_inputs_to_onnx(input_gelu_x)
onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

torch_outputs = aten_gelu_model(input_gelu_x)
torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    [torch.testing.assert_close](https://pytorch.org/docs/stable/testing.html#torch.testing.assert_close "torch.testing.assert_close")(torch_output, [torch.tensor](https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor "torch.tensor")(onnxruntime_output))

## Custom operators without ONNX Runtime support[](https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html#custom-operators-without-onnx-runtime-support)

In this case, the operator is not supported by any ONNX runtime, but we would like to use it as custom operator in ONNX graph. Therefore, we need to implement the operator in three places:

1. PyTorch FX graph
    
2. ONNX Registry
    
3. ONNX Runtime
    

In the following example, we would like to use a custom operator that takes one tensor input, and returns one output. The operator adds the input to itself, and returns the rounded result.

### Custom Ops Registration in PyTorch FX Graph (Beta)[](https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html#custom-ops-registration-in-pytorch-fx-graph-beta)

Firstly, we need to implement the operator in PyTorch FX graph. This can be done by using `torch._custom_op`.

# NOTE: This is a beta feature in PyTorch, and is subject to change.
from torch._custom_op import impl as custom_op

@custom_op.custom_op("mylibrary::addandround_op")
def addandround_op(tensor_x: [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")) -> [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor"):
    ...

@addandround_op.impl_abstract()
def addandround_op_impl_abstract(tensor_x):
    return [torch.empty_like](https://pytorch.org/docs/stable/generated/torch.empty_like.html#torch.empty_like "torch.empty_like")(tensor_x)

@addandround_op.impl("cpu")
def addandround_op_impl(tensor_x):
    return [torch.round](https://pytorch.org/docs/stable/generated/torch.round.html#torch.round "torch.round")(tensor_x + tensor_x)  # add x to itself, and round the result

torch._dynamo.allow_in_graph(addandround_op)

class CustomFoo([torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")):
    def forward(self, tensor_x):
        return addandround_op(tensor_x)

input_addandround_x = [torch.randn](https://pytorch.org/docs/stable/generated/torch.randn.html#torch.randn "torch.randn")(3)
custom_addandround_model = CustomFoo()

### Custom Ops Registration in ONNX Registry[](https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html#custom-ops-registration-in-onnx-registry)

For the step 2 and 3, we need to implement the operator in ONNX registry. In this example, we will implement the operator in ONNX registry with the namespace `test.customop` and operator name `CustomOpOne`, and `CustomOpTwo`. These two ops are registered and built in [cpu_ops.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/testdata/custom_op_library/cpu/cpu_ops.cc).

custom_opset = onnxscript.values.Opset(domain="test.customop", version=1)

# NOTE: The function signature must match the signature of the unsupported ATen operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.
@onnxscript.script(custom_opset)
def custom_addandround(input_x):
    # The same as opset18.Add(x, x)
    add_x = custom_opset.CustomOpOne(input_x, input_x)
    # The same as opset18.Round(x, x)
    round_x = custom_opset.CustomOpTwo(add_x)
    # Cast to FLOAT to match the ONNX type
    return opset18.Cast(round_x, to=1)

onnx_registry = [torch.onnx.OnnxRegistry](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.OnnxRegistry "torch.onnx.OnnxRegistry")()
onnx_registry.register_op(
    namespace="mylibrary", op_name="addandround_op", overload="default", function=custom_addandround
    )

export_options = [torch.onnx.ExportOptions](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.ExportOptions "torch.onnx.ExportOptions")(onnx_registry=onnx_registry)
onnx_program = [torch.onnx.dynamo_export](https://pytorch.org/docs/stable/onnx_dynamo.html#torch.onnx.dynamo_export "torch.onnx.dynamo_export")(
    custom_addandround_model, input_addandround_x, export_options=export_options
    )
onnx_program.save("./custom_addandround_model.onnx")

The `onnx_program` exposes the exported model as protobuf through `onnx_program.model_proto`. The graph has one graph nodes for `custom_addandround`, and inside `custom_addandround`, there are two function nodes, one for each operator.

assert onnx_program.model_proto.graph.node[0].domain == "test.customop"
assert onnx_program.model_proto.graph.node[0].op_type == "custom_addandround"
assert onnx_program.model_proto.functions[0].node[0].domain == "test.customop"
assert onnx_program.model_proto.functions[0].node[0].op_type == "CustomOpOne"
assert onnx_program.model_proto.functions[0].node[1].domain == "test.customop"
assert onnx_program.model_proto.functions[0].node[1].op_type == "CustomOpTwo"

This is how `custom_addandround_model` ONNX graph looks using Netron:

[![../../_images/custom_addandround_model.png](https://pytorch.org/tutorials/_images/custom_addandround_model.png)](https://pytorch.org/tutorials/_images/custom_addandround_model.png)

Inside the `custom_addandround` function, we can see the two custom operators we used in the function (`CustomOpOne`, and `CustomOpTwo`), and they are from module `test.customop`:

![../../_images/custom_addandround_function.png](https://pytorch.org/tutorials/_images/custom_addandround_function.png)

### Custom Ops Registration in ONNX Runtime[](https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html#custom-ops-registration-in-onnx-runtime)

To link your custom op library to ONNX Runtime, you need to compile your C++ code into a shared library and link it to ONNX Runtime. Follow the instructions below:

1. Implement your custom op in C++ by following [ONNX Runtime instructions](https://pytorch.org/tutorials/beginner/onnx/%60https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/reference/operators/add-custom-op.md).
    
2. Download ONNX Runtime source distribution from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).
    
3. Compile and link your custom op library to ONNX Runtime, for example:
    

$ gcc -shared -o libcustom_op_library.so custom_op_library.cc -L /path/to/downloaded/ort/lib/ -lonnxruntime -fPIC

4. Run the model with ONNX Runtime Python API and compare the results with PyTorch.
    

ort_session_options = onnxruntime.SessionOptions()

# NOTE: Link the custom op library to ONNX Runtime and replace the path
# with the path to your custom op library
ort_session_options.register_custom_ops_library(
    "/path/to/libcustom_op_library.so"
)
ort_session = onnxruntime.InferenceSession(
    "./custom_addandround_model.onnx", providers=['CPUExecutionProvider'], sess_options=ort_session_options)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_input = onnx_program.adapt_torch_inputs_to_onnx(input_addandround_x)
onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)

torch_outputs = custom_addandround_model(input_addandround_x)
torch_outputs = onnx_program.adapt_torch_outputs_to_onnx(torch_outputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    [torch.testing.assert_close](https://pytorch.org/docs/stable/testing.html#torch.testing.assert_close "torch.testing.assert_close")(torch_output, [torch.tensor](https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor "torch.tensor")(onnxruntime_output))

## Conclusion[](https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html#conclusion)

Congratulations! In this tutorial, we explored the `ONNXRegistry` API and discovered how to create custom implementations for unsupported or existing ATen operators using ONNX Script. Finally, we leveraged ONNX Runtime to execute the model and compare the results with PyTorch, providing us with a comprehensive understanding of handling unsupported operators in the ONNX ecosystem.