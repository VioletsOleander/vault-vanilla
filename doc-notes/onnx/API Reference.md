---
completed: 
version: 1.18.0
---
# Index
## Versioning
The following example shows how to retrieve onnx version, the onnx opset, the IR version. Every new major release increments the opset version (see [Opset Version](https://onnx.ai/onnx/api/defs.html#l-api-opset-version)).

```python
from onnx import __version__, IR_VERSION
from onnx.defs import onnx_opset_version
print(f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}")
```

```
onnx.__version__='1.18.0', opset=23, IR_VERSION=11
```

The intermediate representation (IR) specification is the abstract model for graphs and operators and the concrete format that represents them. Adding a structure, modifying one them increases the IR version.

The opset version increases when an operator is added or removed or modified. A higher opset means a longer list of operators and more options to implement an ONNX functions. An operator is usually modified because it supports more input and output type, or an attribute becomes an input.

## Data Structures
Every ONNX object is defined based on a [protobuf message](https://googleapis.dev/python/protobuf/latest/google/protobuf/message.html) and has a name ended with suffix `Proto`. For example, [NodeProto](https://onnx.ai/onnx/api/classes.html#l-nodeproto) defines an operator, [TensorProto](https://onnx.ai/onnx/api/classes.html#l-tensorproto) defines a tensor. 
>  每个 ONNX 对象都是基于 protobuf 消息定义的，其名称都以 `Proto` 后缀结尾
>  例如 `NodeProto` 定义了一个算子，`TensorProto` 定义了一个张量

Next page lists all of them.

- [Protos](https://onnx.ai/onnx/api/classes.html)
- [Serialization](https://onnx.ai/onnx/api/serialization.html)

## Functions
An ONNX model can be directly from the classes described in previous section but it is faster to create and verify a model with the following helpers.
>  以下的辅助函数帮助我们更快创建和验证模型

- [onnx.backend](https://onnx.ai/onnx/api/backend.html)
- [onnx.checker](https://onnx.ai/onnx/api/checker.html)
- [onnx.compose](https://onnx.ai/onnx/api/compose.html)
- [onnx._custom_element_types](https://onnx.ai/onnx/api/custom_element_types.html)
- [onnx.defs](https://onnx.ai/onnx/api/defs.html)
- [onnx.external_data_helper](https://onnx.ai/onnx/api/external_data_helper.html)
- [onnx.helper](https://onnx.ai/onnx/api/helper.html)
- [onnx.hub](https://onnx.ai/onnx/api/hub.html)
- [onnx.inliner](https://onnx.ai/onnx/api/inliner.html)
- [onnx.mapping](https://onnx.ai/onnx/api/mapping.html)
- [onnx.model_container](https://onnx.ai/onnx/api/model_container.html)
- [onnx.numpy_helper](https://onnx.ai/onnx/api/numpy_helper.html)
- [onnx.parser](https://onnx.ai/onnx/api/parser.html)
- [onnx.printer](https://onnx.ai/onnx/api/printer.html)
- [onnx.reference](https://onnx.ai/onnx/api/reference.html)
- [onnx.shape_inference](https://onnx.ai/onnx/api/shape_inference.html)
- [onnx.tools](https://onnx.ai/onnx/api/tools.html)
- [onnx.utils](https://onnx.ai/onnx/api/utils.html)
- [onnx.version_converter](https://onnx.ai/onnx/api/version_converter.html)

# Protos
This structures are defined with protobuf in files `onnx/*.proto`. It is recommended to use function in module [onnx.helper](https://onnx.ai/onnx/api/helper.html#l-mod-onnx-helper) to create them instead of directly instantiated them. Every structure can be printed with function `print` and is rendered as a json string.
>  所有的 `*Proto` 类都基于 protobuf 定义，位于文件 `onnx/*.proto` 中
>  推荐使用模块 `onnx.helper` 中的函数创建这些类，而不是直接实例化
>  所有的类都可以由 `print` 函数打印，会渲染为 json 格式的字符串

## AttributeProto
This class is used to define an attribute of an operator defined itself by a NodeProto. It is a named attribute containing either singular float, integer, string, graph, and tensor values, or repeated float, integer, string, graph, and tensor values. An AttributeProto MUST contain the name field, and _only one_ of the following content fields, effectively enforcing a C/C++ union equivalent.
>  `AttributeProto` 类用于定义一个算子 (算子由 `NodeProto` 定义) 的一个属性
>  属性有名字，可以包含 (重复的) 浮点数、整数、字符串、图、张量值
>  `AttributeProto` 必须包含一个名字字段，并且只能包含上述内容字段中的一个，类似 C/C++ 的联合体，即同一时刻仅存储一种类型的值

```python
class onnx.AttributeProto
```

## FunctionProto
This defines a function. It is not a model but can be used to define custom operators used in a model.
>  `FunctionProto` 定义一个函数
>  函数不是模型，但函数可以用于模型使用的自定义算子

```python
class onnx.FunctionProto
```

## GraphProto
This defines a graph or a set of nodes called from a loop or a test for example. A graph defines the computational logic of a model and is comprised of a parameterized list of nodes that form a directed acyclic graph based on their inputs and outputs. This is the equivalent of the _network_ or _graph_ in many deep learning frameworks.
>  `GraphProto` 定义一个图或一组节点，由 loop 或 test 所调用
>  图定义了模型的计算逻辑，图由一组参数化的节点组成，这些节点基于其输入和输出构成了有向无环图

```python
class onnx.GraphProto
```

## MapProto
This defines a map or a dictionary. It specifies an associative table, defined by keys and values. MapProto is formed with a repeated field of keys (of type INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, or STRING) and values (of type TENSOR, SPARSE_TENSOR, SEQUENCE, or MAP). Key types and value types have to remain the same throughout the instantiation of the MapProto.
>  `MapProto` 定义了一个映射或字典，它指定了一个关联表，该表由键和值组成

```python
class onnx.MapProto
```

## ModelProto
This defines a model. That is the type every converting library returns after converting a machine learned model. ModelProto is a top-level file/container format for bundling a ML model and associating its computation graph with metadata. The semantics of the model are described by the associated GraphProto’s.
>  `ModelProto` 定义了一个模型
>  所有的转换库在转换完机器学习模型后，返回的都是一个 `ModelProto`
>  `ModelProto` 是绑定了机器学习模型的元数据和其相关计算图的顶级文件/容器格式，模型的语义由它相关的 `GraphProto` 描述

```python
class onnx.ModelProto
```

## NodeProto
This defines an operator. A model is a combination of mathematical functions, each of them represented as an onnx operator, stored in a NodeProto. 
>  `NodeProto` 定义了一个算子
>  模型是由数学函数组合而成，而每个数学函数都可以表示为一个 onnx 算子，存储在一个 `NodeProto` 中

Computation graphs are made up of a DAG of nodes, which represent what is commonly called a _layer_ or _pipeline stage_ in machine learning frameworks. For example, it can be a node of type _Conv_ that takes in an image, a filter tensor and a bias tensor, and produces the convolved output.
>  计算图就是由节点组成的有向无环图，其中的节点表示了机器学习框架中的 “层” 或 “管道阶段“”
>  例如，一个类型为 “卷积” 的节点，接受一张图像、一个滤波器张量、一个偏置张量，生成卷积后的输出

```python
class onnx.NodeProto
```

## OperatorProto
This class is rarely used by users. An OperatorProto represents the immutable specification of the signature and semantics of an operator. Operators are declared as part of an OperatorSet, which also defines the domain name for the set. 
>  `OperatorProto` 表示算子的签名和语义的不可变规范
>  算子作为算子集 OperatorSet 的一部分被声明，OperatorSet 同时定义了该算子集的域名

Operators are uniquely identified by a three part identifier (domain, op_type, since_version) where

- _domain_ is the domain of an operator set that contains this operator specification.
- _op_type_ is the name of the operator as referenced by a NodeProto.op_type
- _since_version_ is the version of the operator set that this operator was initially declared in.

>  算子由一个三部分标识符唯一表示：
>  - `domain` 表示包含该算子规范的算子集的域
>  - `op_type` 表示由 `NodeProto.op_type` 引用的算子名称
>  - `since_version` 表示该算子最初声明时所在的算子集的版本

```python
class onnx.OperatorProto
```

## OperatorSetIdProto
This is the type of attribute `opset_import` of class ModelProto. This attribute specifies the versions of operators used in the model. Every operator or node belongs to a domain. All operators for the same domain share the same version.
>  `ModelProto` 的属性 `opset_import` 的类型就是 `OperatorSetIdProto`
>  该属性指定了模型使用的算子的版本，模型中每个算子/节点都属于一个域，相同域中的所有算子共享相同版本

```python
class onnx.OperatorSetIdProto
```

## OperatorSetProto
An OperatorSetProto represents an immutable set of immutable operator specifications. The domain of the set (OperatorSetProto.domain) is a reverse-DNS name that disambiguates operator sets defined by independent entities. The version of the set (opset_version) is a monotonically increasing integer that indicates changes to the membership of the operator set. Operator sets are uniquely identified by a two part identifier (domain, opset_version) Like ModelProto, OperatorSetProto is intended as a top-level file/wire format, and thus has the standard format headers in addition to the operator set information.
>  `OperatorSetProto` 表示了一个不可变的不可变算子规范集合
>  该集合的域为 `OperatorSetProto.domain`，版本为 `OperatorSetProto.opset_version`
>  算子集由两部分标识符 (`domain` , `opset_version`) 唯一标识，`OperatorSetProto` 和 `ModelProto` 类似，也是作为顶级的文件格式，包含了算子集本身以及额外的标准格式头

```python
class onnx.OperatorSetProto
```

## OptionalProto
Some input or output of a model are optional. This class must be used in this case. An instance of class OptionalProto may contain or not an instance of type TensorProto, SparseTensorProto, SequenceProto, MapProto and OptionalProto.
>  如果模型的一些输入或输出是可选的，就要使用 `OptionalProto` 类
>  一个 `OptionalProto` 的实例可以包含或者不包含 (即 optional ) 类型为 `TensorProto, SparseTensorProto, SequenceProto, MapProto, OptionalProto` 的实例

```python
class onnx.OptionalProto
```

## SequenceProto
This defines a dense, ordered, collection of elements that are of homogeneous types. Sequences can be made out of tensors, maps, or sequences. If a sequence is made out of tensors, the tensors must have the same element type (i.e. int32). In some cases, the tensors in a sequence can have different shapes. Whether the tensors can have different shapes or not depends on the type/shape associated with the corresponding `ValueInfo`. For example, `Sequence<Tensor<float, [M,N]>` means that all tensors have same shape. However, `Sequence<Tensor<float, [omitted,omitted]>` means they can have different shapes (all of rank 2), where _omitted_ means the corresponding dimension has no symbolic/constant value. Finally, `Sequence<Tensor<float, omitted>>` means that the different tensors can have different ranks, when the _shape_ itself is omitted from the tensor-type. For a more complete description, refer to [Static tensor shapes](https://github.com/onnx/onnx/blob/main/docs/IR.md#static-tensor-shapes).
>  `SequenceProto` 定义了一个密集的、有序的元素集合，这些元素类型相同
>   序列可以由张量、映射、或序列本身构成，如果是张量序列，所有张量的元素类型应该相同，但形状可以不同
>   序列中的张量是否可以有不同的形状取决于对应的 `ValueInfo` ，例如 `Sequence<Tensor<float, [M,N]>>` 表示序列中所有张量形状相同，`Sequence<Tensor<float, [ommitted, omitted]>>` 就不要求序列中所有张量形状相同，但是要求都是二维张量，`Sequence<Tensor<float, omitted>>` 则进一步不要求序列中张量维度相同

```python
class onnx.SequenceProto
```

## SparseTensorProto
This defines a sparse tensor. The sequence of non-default values are encoded as a tensor of shape `[NNZ]`. The default-value is zero for numeric tensors, and empty-string for string tensors. values must have a non-empty name present which serves as a name for SparseTensorProto when used in sparse_initializer list.
>  `SparseTensorProto` 定义了一个稀疏张量
>   稀疏张量中，非默认值的序列被编码为一个形状为 `[NNZ]` 的张量
>   稀疏张量中，数值张量的默认值为 0，字符串张量的默认值为空字符串
>  `SparseTensorProto` 的值必须有一个名称，用在 `sparse_initializer` 列表中指代该 `SparseTensorProto`

```python
class onnx.SparseTensorProto
```

## StringStringEntryProto
This is equivalent to a pair of strings. This is used to store metadata in ModelProto.
>  `StringStringEntryProto` 等价于字符串对，用在 `ModelProto` 中存储模型元信息

```python
class onnx.StringStringEntryProto
```

## TensorProto
This defines a tensor. A tensor is fully described with a shape (see ShapeProto), the element type (see TypeProto), and the elements themselves. All available types are listed in [onnx.mapping](https://onnx.ai/onnx/api/mapping.html#l-mod-onnx-mapping).
>  `TensorProto` 定义一个张量
>  一个张量由形状、元素类型、元素本身定义

```python
class onnx.TensorProto
    class Segment
```

## TensorShapeProto
This defines the shape of a tensor or a sparse tensor. It is a list of dimensions. A dimension can be either an integer value or a symbolic variable. A symbolic variable represents an unknown dimension.
>  `TensorShapeProto` 定义张量形状，它是一个 `Dimension` 列表
>  `Dimension` 可以是数字或符号变量，符号变量表示未知维度

```python
class onnx.TensorShapeProto
    class Dimension
```

## TrainingInfoProto
TrainingInfoProto stores information for training a model. In particular, this defines two functionalities: an initialization-step and a training-algorithm-step. Initialization resets the model back to its original state as if no training has been performed. Training algorithm improves the model based on input data.
>  `TrainingInfoProto` 存储了训练模型所用的信息，它定义了两种功能：初始化步骤和训练算法步骤
>  初始化步骤重置模型状态，训练算法步骤基于输入数据改进模型

 The semantics of the initialization-step is that the initializers in ModelProto. graph and in TrainingInfoProto. algorithm are first initialized as specified by the initializers in the graph, and then updated by the _initialization_binding_ in every instance in ModelProto. training_info. The field _algorithm_ defines a computation graph which represents a training algorithm’s step. After the execution of a TrainingInfoProto. algorithm, the initializers specified by _update_binding_ may be immediately updated. If the targeted training algorithm contains consecutive update steps (such as block coordinate descent methods), the user needs to create a TrainingInfoProto for each step.
 > 初始化步骤的语义是： `ModelProto.graph` 和 `TraningInfoProto.algorithm` 中的初始化值会首先由图指定的初始化值初始化，然后再由 `ModelProto.traning_info` 中每个实例中的 `initialization_binding` 更新
>  `TraingInfoProto.algorithm` 定义了表示训练算法的计算图
>  `TraningInfoProto.algorithm` 执行之后，由 `update_binding` 指定的初始化值会被更新
>  如果训练算法包括连续的更新步骤，例如块坐标下降方法，用户需要为其中的每一步创建 `TrainingInfoProto` 

```python
class onnx.TraningInfoProto
```

## TypeProto
This defines a type of a tensor which consists in an element type and a shape (ShapeProto).
>  `TypeProto` 定义了张量的类型

```python
class onnx.TypeProto
    class Map
    class Opaque
    class Optional
    class Sequence
    class SparseTensor
    class Tensor
```

## ValueInfoProto
This defines a input or output type of a GraphProto. It contains a name, a type (TypeProto), and a documentation string.
>  `ValueInfoProto` 定义了一个 `GraphProto` 的输入或输出类型
>  `ValueInforProto` 包含了一个名称、一个类型 (`TypeProto`)、一个文档字符串

```python
class onnx.ValueInfoProto
```

# Serialization
## Save a model and any Proto class
This ONNX graph needs to be serialized into one contiguous memory buffer. Method `SerializeToString` is available in every ONNX objects.

```python
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

This method has the following signature. 

(Refer to original doc)

Every Proto class implements method `SerializeToString`. Therefore the following code works with any class described in page [Protos](https://onnx.ai/onnx/api/classes.html#l-onnx-classes).
>  每个 `Proto` 都实现 `SerializeToString` 方法

```python
with open("proto.pb", "wb") as f:
    f.write(proto.SerializeToString())
```

Next example shows how to save a [NodeProto](https://onnx.ai/onnx/api/classes.html#l-nodeproto).

```python
from onnx import NodeProto

node = NodeProto()
node.name = "example-type-proto"
node.op_type = "Add"
node.input.extend(["X", "Y"])
node.output.extend(["Z"])

with open("node.pb", "wb") as f:
    f.write(node.SerializeToString())
```

## Load a model
Following function only automates the loading of a class [ModelProto](https://onnx.ai/onnx/api/classes.html#l-modelproto). Next sections shows how to restore any other proto class.

(Refer to original doc)

```python
from onnx import load

onnx_model = load("model.onnx")
```

Or:

```python
from onnx import load

with open("model.onnx", "rb") as f:
    onnx_model = load(f)
```

Next function does the same from a bytes array.

(Refer to original doc)

## Load a Proto
Proto means here any type containing data including a model, a tensor, a sparse tensor, any class listed in page [Protos](https://onnx.ai/onnx/api/classes.html#l-onnx-classes). The user must know the type of the data he needs to restore and then call method `ParseFromString`. [protobuf](https://developers.google.com/protocol-buffers) does not store any information about the class of the saved data. Therefore, this class must be known before restoring an object.
>  protobuf 并不存储数据的任何类信息，因此具体的信息必须在恢复该对象时提前知道，然后实例化该类，调用 `ParseFromString` 方法以恢复数据

(Refer to original doc)

Next example shows how to restore a [NodeProto](https://onnx.ai/onnx/api/classes.html#l-nodeproto).

```python
from onnx import NodeProto

tp2 = NodeProto()
with open("node.pb", "rb") as f:
    content = f.read()

tp2.ParseFromString(content)

print(tp2)
```

```
input: "X"
input: "Y"
output: "Z"
name: "example-type-proto"
op_type: "Add"
```

A shortcut exists for [TensorProto](https://onnx.ai/onnx/api/classes.html#l-tensorproto):

(Refer to original doc)