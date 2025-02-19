---
completed:
---
# Inference
## Tutorial
_ONNX Runtime_ provides an easy way to run machine learned models with high performance on CPU or GPU without dependencies on the training framework. Machine learning frameworks are usually optimized for batch training rather than for prediction, which is a more common scenario in applications, sites, and services. At a high level, you can:

1. Train a model using your favorite framework.
2. Convert or export the model into ONNX format. See [ONNX Tutorials](https://github.com/onnx/tutorials) for more details.
3. Load and run the model using _ONNX Runtime_.

In this tutorial, we will briefly create a pipeline with _scikit-learn_, convert it into ONNX format and run the first predictions.

### Step 1: Train a model using your favorite framework
We’ll use the famous iris datasets.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
clr = LogisticRegression()
clr.fit(X_train, y_train)
print(clr)
```

```
>>> LogisticRegression()
```

### Step 2: Convert or export the model into ONNX format
[ONNX](https://github.com/onnx/onnx) is a format to describe the machine learned model. It defines a set of commonly used operators to compose models. There are [tools](https://github.com/onnx/tutorials) to convert other model formats into ONNX. Here we will use [ONNXMLTools](https://github.com/onnx/onnxmltools).

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

### Step 3: Load and run the model using ONNX Runtime
We will use _ONNX Runtime_ to compute the predictions for this machine learning model.

```python
import numpy
import onnxruntime as rt

sess = rt.InferenceSession(
    "logreg_iris.onnx", providers=rt.get_available_providers())
input_name = sess.get_inputs()[0].name
pred_onx = sess.run(None, {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)
```

```
>>> [1 2 1 0 1 1 0 0 1 1 2 1 0 1 1 1 2 2 0 0 2 0 2 2 2 0 0 2 0 0 0 2 1 1 1 0 2 1]
```

The code can be changed to get one specific output by specifying its name into a list.

```python
import numpy
import onnxruntime as rt

sess = rt.InferenceSession(
    "logreg_iris.onnx", providers=rt.get_available_providers())
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)
```

```
>>> [1 2 1 0 1 1 0 0 1 1 2 1 0 1 1 1 2 2 0 0 2 0 2 2 2 0 0 2 0 0 0 2 1 1 1 0 2 1]
```

## API
### API Overview
_ONNX Runtime_ loads and runs inference on a model in ONNX graph format, or ORT format (for memory and disk constrained environments).
>  ONNX Runtime 加载并运行 ONNX 图格式的模型，或者 ORT 格式的模型 (在内存或磁盘空间受限的环境下)

The data consumed and produced by the model can be specified and accessed in the way that best matches your scenario.

#### Load and run a model
`InferenceSession` is the main class of ONNX Runtime. It is used to load and run an ONNX model, as well as specify environment and application configuration options.

```Python
session = onnxruntime.InferenceSession('model.onnx')

outputs = session.run([output names], inputs)
```

>  `InferenceSession` 是 ONNX Runtime 的主要类，`InferenceSession` 用于加载和运行 ONNX 模型，同时用于指定环境和应用配置选项

ONNX and ORT format models consist of a graph of computations, modeled as operators, and implemented as optimized operator kernels for different hardware targets. ONNX Runtime orchestrates the execution of operator kernels via execution providers. An execution provider contains the set of kernels for a specific execution target (CPU, GPU, IoT etc). 
>  ONNX 和 ORT 格式的模型表现为一个计算图，计算图中的各种计算被建模为算子，相同的计算针对不同的硬件目标实现为优化的算子内核
>  ONNX Runtime 通过执行提供程序组织算子内核的执行，一个执行提供程序包含了针对特定执行目标 (CPU, GPU, IoT 等) 的一组内核

Execution provides are configured using the providers parameter. Kernels from different execution providers are chosen in the priority order given in the list of providers. In the example below if there is a kernel in the CUDA execution provider ONNX Runtime executes that on GPU. If not the kernel is executed on CPU.
>  执行提供程序通过 `providers` 参数配置
>  不同执行提供程序的内核按照 `providers` 中给出的顺序选择，例如下面的例子中，如果 CUDA 执行提供程序中由所要的内核，ONNX Runtime 就在 GPU 上执行该内核，否则在 CPU 上执行 CPU 上的内核

```python
session = onnxruntime.InferenceSession(
        model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

The list of available execution providers can be found here: [Execution Providers](https://onnxruntime.ai/docs/execution-providers).

Since ONNX Runtime 1.10, you must explicitly specify the execution provider for your target. Running on CPU is the only time the API allows no explicit setting of the provider parameter. 
>  ONNX Runtime 1.10 之后，必须为特定的目标显示指定执行提供程序 (默认是 CPU)

In the examples that follow, the `CUDAExecutionProvider` and `CPUExecutionProvider` are used, assuming the application is running on NVIDIA GPUs. Replace these with the execution provider specific to your environment.

You can supply other session configurations via the session options parameter. For example, to enable profiling on the session:

```python
options = onnxruntime.SessionOptions()
options.enable_profiling=True
session = onnxruntime.InferenceSession(
        'model.onnx',
        sess_options=options,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
)
```

#### Data inputs and outputs
The ONNX Runtime Inference Session consumes and produces data using its `OrtValue` class.
>  ONNX Runtime 推理会话使用 `OrtValue` 类来包装输入和输出数据

##### Data on CPU
On CPU (the default), `OrtValues` can be mapped to and from native Python data structures: numpy arrays, dictionaries and lists of numpy arrays.
>  在 CPU 上 (默认情况)，`OrtValues` 可以和原生 Python 数据结构——numpy 数组、字典、numpy 数组列表——互相转换

```python
# X is numpy array on cpu
ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X)
ortvalue.device_name()  # 'cpu'
ortvalue.shape()        # shape of the numpy array X
ortvalue.data_type()    # 'tensor(float)'
ortvalue.is_tensor()    # 'True'
np.array_equal(ortvalue.numpy(), X)  # 'True'

# ortvalue can be provided as part of the input feed to a model
session = onnxruntime.InferenceSession(
        'model.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
)
results = session.run(["Y"], {"X": ortvalue})
```

By default, _ONNX Runtime_ always places input(s) and output(s) on CPU. Having the data on CPU may not optimal if the input or output is consumed and produced on a device other than CPU because it introduces data copy between CPU and the device.
>  ONNX Runtime 默认总是将输入和输出放在 CPU 上

##### Data on device
_ONNX Runtime_ supports a custom data structure that supports all ONNX data formats that allows users to place the data backing these on a device, for example, on a CUDA supported device. In ONNX Runtime, this called `IOBinding`.
>  ONNX Runtime 中的 `IOBinding` 支持所有 ONNX 数据格式，允许用户将这些数据结构的基础数据放在设备上

To use the `IOBinding` feature, replace ` InferenceSession.run() ` with ` InferenceSession.run_with_iobinding() `.

A graph is executed on a device other than CPU, for instance CUDA. Users can use `IOBinding` to copy the data onto the GPU.

```python
# X is numpy array on cpu
session = onnxruntime.InferenceSession(
        'model.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
)
io_binding = session.io_binding()
# OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
io_binding.bind_cpu_input('input', X)
io_binding.bind_output('output')
session.run_with_iobinding(io_binding)
Y = io_binding.copy_outputs_to_cpu()[0]
```

>  上例中
>  `io_binding.bind_cpu_input('input', X)` 将 CPU 上的 numpy 数组 `X` 绑定到名为 `input` 的输入节点上，如果模型的 `input` 节点需要在 GPU 上执行，ONNX Runtime 会自动将 `X` 从 CPU 拷贝到 GPU
>  `io_binding.bind_output('output')` 将模型的输出节点 `output` 和 `io_binding` 对象绑定，如果使用了 GPU，输出数据将在 GPU 上
>  `session.run_with_iobinding(io_binding)` 使用 `io_binding` 对象进行推理
>  `io_binding.copy_outputs_to_cpu()` 将输出数据拷贝到 CPU，它返回一个列表，其中每个元素对应一个输出节点

The input data is on a device, users directly use the input. The output data is on CPU.

```Python
# X is numpy array on cpu
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
session = onnxruntime.InferenceSession(
        'model.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
)
io_binding = session.io_binding()
io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
io_binding.bind_output('output')
session.run_with_iobinding(io_binding)
Y = io_binding.copy_outputs_to_cpu()[0]
```

>  上例中
>  `OrtValue.ortvalue_from_numpy(X, 'cuda', 0)` 将 `X` 转换为 ONNX Runtime 的 `OrtValue` 对象，并且将其放在 GPU 上，`OrtValue` 包装了数据的指针和元信息
>  `io_binding.bind_input()` 同样用于绑定数据到输入输出节点

The input data and output data are both on a device, users directly use the input and also place output on the device.

```python
#X is numpy array on cpu
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, 'cuda', 0)  # Change the shape to the actual shape of the output being bound
session = onnxruntime.InferenceSession(
        'model.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
)
io_binding = session.io_binding()
io_binding.bind_input(
        name='input',
        device_type=X_ortvalue.device_name(),
        device_id=0,
        element_type=np.float32,
        shape=X_ortvalue.shape(),
        buffer_ptr=X_ortvalue.data_ptr()
)
io_binding.bind_output(
        name='output',
        device_type=Y_ortvalue.device_name(),
        device_id=0,
        element_type=np.float32,
        shape=Y_ortvalue.shape(),
        buffer_ptr=Y_ortvalue.data_ptr()
)
session.run_with_iobinding(io_binding)
```

Users can request _ONNX Runtime_ to allocate an output on a device. This is particularly useful for dynamic shaped outputs. Users can use the _get_outputs()_ API to get access to the _OrtValue_ (s) corresponding to the allocated output(s). Users can thus consume the _ONNX Runtime_ allocated memory for the output as an _OrtValue_.
>  用户可以请求 ONNX Runtime 在设备上分配输出，并通过 `get_outputs()` 函数访问设备上分配的输出的 `OrtValue` 

```python
#X is numpy array on cpu
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
session = onnxruntime.InferenceSession(
        'model.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
)
io_binding = session.io_binding()
io_binding.bind_input(
        name='input',
        device_type=X_ortvalue.device_name(),
        device_id=0,
        element_type=np.float32,
        shape=X_ortvalue.shape(),
        buffer_ptr=X_ortvalue.data_ptr()
)
#Request ONNX Runtime to bind and allocate memory on CUDA for 'output'
io_binding.bind_output('output', 'cuda')
session.run_with_iobinding(io_binding)
# The following call returns an OrtValue which has data allocated by ONNX Runtime on CUDA
ort_output = io_binding.get_outputs()[0]
```

>  上例中
>  `io_binding.bind_output('output', 'cuda')` 将模型的输出节点 `output` 绑定到 `io_binding` 对象上，并请求 ONNX Runtime 在设备上为输出数据分配内存
>  `io_binding.get_outputs()` 返回包含所有输出节点的 `OrtValue` 对象的列表，此时输出数据仍然在设备上，不在 CPU 上

In addition, _ONNX Runtime_ supports directly working with _OrtValue_ (s) while inferencing a model if provided as part of the input feed.

Users can bind _OrtValue_ (s) directly.

```python
#X is numpy array on cpu
#X is numpy array on cpu
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, 'cuda', 0)  # Change the shape to the actual shape of the output being bound
session = onnxruntime.InferenceSession(
        'model.onnx',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
)
io_binding = session.io_binding()
io_binding.bind_ortvalue_input('input', X_ortvalue)
io_binding.bind_ortvalue_output('output', Y_ortvalue)
session.run_with_iobinding(io_binding)
```

>  上例中
>  `io_binding.bind_ortvalue_input/output()` 用于直接绑定 `OrtValue`

You can also bind inputs and outputs directly to a PyTorch tensor.

```python
# X is a PyTorch tensor on device
session = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
binding = session.io_binding()

X_tensor = X.contiguous()

binding.bind_input(
    name='X',
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=tuple(x_tensor.shape),
    buffer_ptr=x_tensor.data_ptr(),
    )

## Allocate the PyTorch tensor for the model output
Y_shape = ... # You need to specify the output PyTorch tensor shape
Y_tensor = torch.empty(Y_shape, dtype=torch.float32, device='cuda:0').contiguous()
binding.bind_output(
    name='Y',
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=tuple(Y_tensor.shape),
    buffer_ptr=Y_tensor.data_ptr(),
)

session.run_with_iobinding(binding)
```

You can also see code examples of this API in in the [ONNX Runtime inferences examples](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py).

Some onnx data type (like TensorProto.BFLOAT16, TensorProto.FLOAT8E4M3FN and TensorProto.FLOAT8E5M2) are not supported by Numpy. You can directly bind input or output with Torch tensor of corresponding data type (like torch.bfloat16, torch.float8_e4m3fn and torch.float8_e5m2) in GPU memory.

```python
x = torch.ones([3], dtype=torch.float8_e5m2, device='cuda:0')
y = torch.empty([3], dtype=torch.bfloat16, device='cuda:0')

binding = session.io_binding()
binding.bind_input(
    name='X',
    device_type='cuda',
    device_id=0,
    element_type=TensorProto.FLOAT8E5M2,
    shape=tuple(x.shape),
    buffer_ptr=x.data_ptr(),
    )
binding.bind_output(
    name='Y',
    device_type='cuda',
    device_id=0,
    element_type=TensorProto.BFLOAT16,
    shape=tuple(y.shape),
    buffer_ptr=y.data_ptr(),
    )
    session.run_with_iobinding(binding)
```

### API Details
#### Inference Session

```python
class onnxruntime.InferenceSession
```

This is the main class used to run the model.

#### Options
##### RunOptions

```python
class onnxruntime.RunOptions
```

Configuration Information for a single run.

##### SessionOptions

```python
class onnxruntime.SessionOptions
```

Configuration Information for a session.

```python
class onnxruntime.ExecutionMode
class onnxruntime.ExecutionOrder
class onnxruntime.GraphOptimizationLevel
class onnxruntime.OrtAllocatorType
class onnxruntime.OrtArenaCfg
class onnxruntime.OrtMemeoryInfo
class onnxruntime.OrtMemType
```

#### Functions
##### Allocators

```python
onnxruntime.create_and_register_allocator() -> None
onnxruntime.create_and_register_allocator_v2() -> None
```

##### Telemetry events

```python
onnxruntimme.disable_telemetry_events() -> None
onnxruntimme.enable_telemetry_events() -> None
```

##### Providers

```python
onnxruntime.get_all_providers() -> list[str]
onnxruntime.get_available_providers() -> list[str]
```

##### Build, Version

```python
onnxruntime.get_build_info() -> str
onnxruntime.get_version_string() -> str
onnxruntime.has_collecive_ops() -> bool
```

##### Device

```python
onnxruntime.get_device() -> str
```

Return the device used to compute the prediction (CPU, MKL, …)

#### Logging

```python
onnxruntime.set_default_logger_severity() -> None
```

Sets the default logging severity. 0: Verbose, 1: Info, 2: Warning, 3: Error, 4:Fatal

```python
onnxruntime.set_default_logger_verbosity() -> None
```

Sets the default logging verbosity level. To activate the verbose log, you need to set the default logging severity to 0: Verbose level.

#### Random

```python
onnxruntime.set_seed() -> None
```

Sets the seed used for random number generation in Onnxruntime.

#### Data
##### OrtValue

```python
class onnxruntime.OrtValue
```

A data structure that supports all ONNX data formats (tensors and non-tensors) that allows users to place the data backing these on a device, for example, on a CUDA supported device. This class provides APIs to construct and deal with OrtValues.

##### SparseTensor

```python
class onnxruntime.SparseTensor
```

#### Devices
##### IOBinding

```python
class onnxruntime.IOBinding
```

This class provides API to bind input/output to a specified device, e.g. GPU.

```python
class onnxruntime.SessionIOBinding
```

##### OrtDevice

```python
class onnxruntime.OrtDevice
```

#### Internal classes
These classes cannot be instantiated by users but they are returned by methods or functions of this library.

##### ModelMetaData

```python
class onnxruntime.ModelMetaData
```

##### NodeArg

```python
class onnxruntime.NodeArg
```

## Backend
In addition to the regular API which is optimized for performance and usability, _ONNX Runtime_ also implements the [ONNX backend API](https://github.com/onnx/onnx/blob/main/docs/ImplementingAnOnnxBackend.md) for verification of _ONNX_ specification conformance. The following functions are supported:
>  ONNX backend API 用于验证 ONNX 规范的一致性，包括以下函数

```python
onnxruntime.backend.is_compatible()
onnxruntime.backend.prepare()
onnxruntime.backend.run()
onnxruntime.backend.support_device()
```





