---
completed: true
version: 1.19.0
---
ONNX can be compared to a programming language specialized in mathematical functions. It defines all the necessary operations a machine learning model needs to implement its inference function with this language.
>  ONNX 可以比作一个专门用于数学函数的编程语言，ONNX 语言定义了机器学习模型实现其推理功能所必要的所有功能

A linear regression could be represented in the following way:

```python
def onnx_linear_regressor(X):
    "ONNX code for a linear regression"
    return onnx.Add(onnx.MatMul(X, coefficients), bias)
```

This example is very similar to an expression a developer could write in Python. It can be also represented as a graph that shows step-by-step how to transform the features to get a prediction. That’s why a machine-learning model implemented with ONNX is often referenced as an **ONNX graph**.
>  ONNX 所实现的机器学习模型也常称为一个 ONNX 图

![../_images/linreg1.png](https://onnx.ai/onnx/_images/linreg1.png)

ONNX aims at providing a common language any machine learning framework can use to describe its models. The first scenario is to make it easier to deploy a machine learning model in production. An ONNX interpreter (or **runtime**) can be specifically implemented and optimized for this task in the environment where it is deployed. With ONNX, it is possible to build a unique process to deploy a model in production and independent from the learning framework used to build the model. 
>  ONNX 的目的是提供一种通用语言，所有机器学习框架可以用该语言描述其模型
>  ONNX 的一个应用场景就是简化机器学习模型到生产环境的部署，要部署 ONNX 描述的模型到对应环境，我们只需要在部署环境专门实现和优化一个 ONNX 解释器 (或运行时) 即可。通过 ONNX，可以构建一个独立于用于构建模型的学习框架的，统一的生产部署流程

_onnx_ implements a python runtime that can be used to evaluate ONNX models and to evaluate ONNX ops. This is intended to clarify the semantics of ONNX and to help understand and debug ONNX tools and converters. It is not intended to be used for production and performance is not a goal (see [onnx.reference](https://onnx.ai/onnx/api/reference.html#l-reference-implementation)).
>  `onnx` 包实现了一个 python 运行时，可以用于评估 ONNX 模型和 ONNX 算子，其目的主要是用于清晰 ONNX 语义，帮助理解和 debug ONNX 工具和转换器，而不是用于生产部署，性能也不是其目标

## Input, Output, Node, Initializer, Attributes
Building an ONNX graph means implementing a function with the ONNX language or more precisely the [ONNX Operators](https://onnx.ai/onnx/operators/index.html#l-onnx-operators). 
>  构建 ONNX 图意味着使用 ONNX 语言 (更确切的说，是 ONNX 算子) 实现一个函数，

A linear regression would be written this way. The following lines do not follow python syntax. It is just a kind of pseudo-code to illustrate the model.

```
Input: float[M,K] x, float[K,N] a, float[N] c
Output: float[M, N] y

r = onnx.MatMul(x, a)
y = onnx.Add(r, c)
```

This code implements a function `f(x, a, c) -> y = x @ a + c`. And _x_, _a_, _c_ are the **inputs**, _y_ is the **output**. _r_ is an intermediate result. _MatMul_ and _Add_ are the **nodes**. They also have inputs and outputs. A node has also a type, one of the operators in [ONNX Operators](https://onnx.ai/onnx/operators/index.html#l-onnx-operators). This graph was built with the example in Section [A simple example: a linear regression](https://onnx.ai/onnx/intro/python.html#l-onnx-linear-regression-onnx-api).

>  上述代码实现了函数 `f(x, a, c) -> y = x @ a + c` ，其中 `x, a, c` 为输入，`y` 为输出
>  函数中，`r` 为中间结果，`MatMul` 和 `Add` 为节点
>  节点也有各自的输入输出，每个节点都有一个类型，该类型是 ONNX 算子中的一个

The graph could also have an **initializer**. When an input never changes such as the coefficients of the linear regression, it is most efficient to turn it into a constant stored in the graph.
>  ONNX 图中可以有一个初始化值，如果某些输入是不会改变的，我们可以将其转化为存储在图中的常量
>  例如上例中，可以将线性回归的系数转化为存储在图中的常量 (`Initializer` 部分)

```
Input: float[M,K] x
Initializer: float[K,N] a, float[N] c
Output: float[M, N] xac

xa = onnx.MatMul(x, a)
xac = onnx.Add(xa, c)
```

Visually, this graph would look like the following image. The right side describes operator _Add_ where the second input is defined as an initializer. This graph was obtained with this code [Initializer, default value](https://onnx.ai/onnx/intro/python.html#l-onnx-linear-regression-onnx-api-init).
>  该 ONNX 图视觉上如下图所示
>  右边描述了算子/节点 `Add` ，该算子的第二个输入是初始化值 (initializer)

![Snapshot of Netron](https://onnx.ai/onnx/_images/linreg2.png)

An **attribute** is a fixed parameter of an operator. Operator [Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html#l-onnx-doc-gemm) has four attributes, _alpha_, _beta_, _transA_, _transB_. Unless the runtime allows it through its API, once it has loaded the ONNX graph, these values cannot be changed and remain frozen for all the predictions.
>  属性指的是算子的固定参数，例如 Gemm 算子有四个属性：`alpha, beta, transA, transB` 
>  除非运行时允许通过其 API 修改属性，否则一旦运行时加载了 ONNX 图，算子的属性值就不能再修改，且对于所有的预测都保持不变

## Serialization with protobuf
The deployment of a machine-learned model into production usually requires replicating the entire ecosystem used to train the model, most of the time with a _docker_. Once a model is converted into ONNX, the production environment only needs a runtime to execute the graph defined with ONNX operators. This runtime can be developed in any language suitable for the production application, C, java, python, javascript, C#, Webassembly, ARM…
>  将机器学习模型部署到生产环境通常需要复制用于训练模型的整个生态系统，大多数情况下通过 docker 实现
>  但当模型转化为了 ONNX 之后，生产环境仅需要一个运行时来执行由 ONNX 算子定义的图，这一运行时可以由任意适合生产应用的语言开发

But to make that happen, the ONNX graph needs to be saved. ONNX uses _protobuf_ to serialize the graph into one single block (see [Parsing and Serialization](https://developers.google.com/protocol-buffers/docs/pythontutorial#parsing-and-serialization)). It aims at optimizing the model size as much as possible.
>  ONNX 使用 protobuf 来序列化 ONNX 图为一个单独的数据块，其目的是尽可能优化模型的大小

## Metadata
Machine learned models are continuously refreshed. It is important to keep track of the model version, the author of the model and how it was trained. ONNX offers the possibility to store additional data in the model itself.
>  机器学习模型会不断更新，因此需要记录模型的版本、作者以及模型的训练方式
>  ONNX 允许存储额外的模型元数据，如下所示

- **doc_string**: Human-readable documentation for this model.
    
    Markdown is allowed.
    
- **domain**: A reverse-DNS name to indicate the model namespace or domain,
    
    for example, ‘org.onnx’
    
- **metadata_props**: Named metadata as dictionary `map<string,string>`,
    
    `(values, keys)` should be distinct.
    
- **model_author**: A comma-separated list of names,
    
    The personal name of the author(s) of the model, and/or their organizations.
    
- **model_license**: The well-known name or URL of the license
    
    under which the model is made available.
    
- **model_version**: The version of the model itself, encoded in an integer.
    
- **producer_name**: The name of the tool used to generate the model.
    
- **producer_version**: The version of the generating tool.
    
- **training_info**: An optional extension that contains
    
    information for training (see [TrainingInfoProto](https://onnx.ai/onnx/api/classes.html#l-traininginfoproto))
    

## List of available operators and domains
The main list is described here: [ONNX Operators](https://onnx.ai/onnx/operators/index.html#l-onnx-operators). It merges standard matrix operators (Add, Sub, MatMul, Transpose, Greater, IsNaN, Shape, Reshape…), reductions (ReduceSum, ReduceMin, …) image transformations (Conv, MaxPool, …), deep neural networks layer (RNN, DropOut, …), activations functions (Relu, Softmax, …). It covers most of the operations needed to implement inference functions from standard and deep machine learning. ONNX does not implement every existing machine learning operator, the list of operator would be infinite.
>  ONNX 算子包括了标准矩阵算子 (Add, Sub, MatMul, Transpose, Greater, IsNaN, Shape, Reshape...)，归约算子 (ReduceSum, ReduceMin, ...)，图像变换 (Conv, MaxPool, ...)，深度神经网络算子 (RNN, Dropout, ...)，激活函数 (Relu, Softmax, ...) 等

The main list of operators is identified with a domain **ai.onnx**. A **domain** can be defined as a set of operators. A few operators in this list are dedicated to text but they hardly cover the needs. The main list is also missing tree based models very popular in standard machine learning. These are part of another domain **[ai.onnx.ml](http://ai.onnx.ml/)**, it includes tree bases models (TreeEnsemble Regressor, …), preprocessing (OneHotEncoder, LabelEncoder, …), SVM models (SVMRegressor, …), imputer (Imputer).
>  ONNX 的主要算子列表由域 `ai.onnx` 表示，域表示一组算子
>  ONNX 的主要算子列表没有覆盖基于树模型的相关算子，这些算子属于另一个域 `ai.onnx.ml` ，它包含了基于树的模型的相关算子、预处理算子、SVM 模型相关算子，以及插补器

ONNX only defines these two domains. But the library onnx supports any custom domains and operators (see [Extensibility](https://onnx.ai/onnx/intro/concepts.html#l-onnx-extensibility)).
>  ONNX 仅定义了这两个域，但库 onnx 支持自定义算子和域

## Supported Types
ONNX specifications are optimized for numerical computation with tensors. 
>  ONNX 规范对张量的数值计算进行了优化

A _tensor_ is a multidimensional array. It is defined by:

- a type: the element type, the same for all elements in the tensor
- a shape: an array with all dimensions, this array can be empty, a dimension can be null
- a contiguous array: it represents all the values

This definition does not include _strides_ or the possibility to define a view of a tensor based on an existing tensor. An ONNX tensor is a dense full array with no stride.

>  张量是一个多维数组，它由三个部分定义：
>  - 类型：张量的元素类型，张量中所有元素都是同类型的
>  - 形状：一个包含所有维度大小的数组，该数组可以为空，某个维度也可以为空
>  - 一个连续的数组：表示张量的所有元素值
>  该定义不包括步幅或基于现有张量定义张量视图的可能性
>  ONNX 张量是没有步幅的密集完整数组

### Element Type
ONNX was initially developed to help deploying deep learning model. That’s why the specifications were initially designed for floats (32 bits). The current version supports all common types. Dictionary [TENSOR_TYPE_MAP](https://onnx.ai/onnx/api/mapping.html#l-onnx-types-mapping) gives the correspondence between _ONNX_ and [`numpy`](https://numpy.org/doc/stable/reference/index.html#module-numpy "(in NumPy v2.2)").
>  ONNX 最初的开发目的是用于部署深度学习模型，因此其规范最初是为 32 位浮点数设计
>  ONNX 的目前版本支持所有常见的类型
>  字典 `TENSOR_TYPE_MAP` 给出了 ONNX 和 `numpy` 之间的对应关系

```python
import re
from onnx import TensorProto

reg = re.compile('^[0-9A-Z_]+$') 

values = {}
for att in sorted(dir(TensorProto)):
    if att in {'DESCRIPTOR'}: # 包含单个元素的集合
        continue
    if reg.match(att):
        values[getattr(TensorProto, att)] = att
for i, att in sorted(values.items()): # 根据属性值进行排序
    si = str(i) # 将属性值转化为字符串
    if len(si) == 1:
        si = " " + si
    print("%s: onnx.TensorProto.%s" % (si, att))
```

>  该例中
>  `onnx` 中的 `TensorProto ` 模块包含了张量的各种属性和常量
>  正则表达式 `reg` 用于匹配由数字、大写字母和下划线组成的字符串
>  `dir(TensorProto)` 返回 `TensorProto` 模块定义的所有名字，`sorted()` 对这些名字进行排序
>  `getattr(object, attribute: str[, default])` 函数用于获取对象特定属性的值，我们利用该函数构建字典 `values`，将值映射到属性名

```
 1: onnx.TensorProto.FLOAT
 2: onnx.TensorProto.UINT8
 3: onnx.TensorProto.INT8
 4: onnx.TensorProto.UINT16
 5: onnx.TensorProto.INT16
 6: onnx.TensorProto.INT32
 7: onnx.TensorProto.INT64
 8: onnx.TensorProto.STRING
 9: onnx.TensorProto.BOOL
10: onnx.TensorProto.FLOAT16
11: onnx.TensorProto.DOUBLE
12: onnx.TensorProto.UINT32
13: onnx.TensorProto.UINT64
14: onnx.TensorProto.COMPLEX64
15: onnx.TensorProto.COMPLEX128
16: onnx.TensorProto.BFLOAT16
17: onnx.TensorProto.FLOAT8E4M3FN
18: onnx.TensorProto.FLOAT8E4M3FNUZ
19: onnx.TensorProto.FLOAT8E5M2
20: onnx.TensorProto.FLOAT8E5M2FNUZ
21: onnx.TensorProto.UINT4
22: onnx.TensorProto.INT4
23: onnx.TensorProto.FLOAT4E2M1
```

ONNX is strongly typed and its definition does not support implicit cast. It is impossible to add two tensors or matrices with different types even if other languages does. That’s why an explicit cast must be inserted in a graph.
>  ONNX 是强类型语言，且不支持隐式转换，因此在 ONNX 中不可能将两个不同类型的 tensor 或矩阵相加，要这样做，必须要在图中插入显式转换

### Sparse Tensor
Sparse tensors are useful to represent arrays having many null coefficients. ONNX supports 2D sparse tensor. Class [SparseTensorProto](https://onnx.ai/onnx/api/classes.html#l-onnx-sparsetensor-proto) defines attributes `dims`, `indices` (int64) and `values`.
>  ONNX 支持二维稀疏张量，类 `SparseTensorProto` 定义了属性 `dims, indices, values`

### Other types
In addition to tensors and sparse tensors, ONNX supports sequences of tensors, map of tensors, sequences of map of tensors through types [SequenceProto](https://onnx.ai/onnx/api/classes.html#l-onnx-sequence-proto), [MapProto](https://onnx.ai/onnx/api/classes.html#l-onnx-map-proto). They are rarely used.
>  除张量和稀疏张量外，ONNX 还通过类型 `SequenceProto, MapProto` 支持张量序列，张量映射，张量映射序列

## What is an opset version?
The opset is mapped to the version of the _onnx_ package. It is incremented every time the minor version increases. Every version brings updated or new operators.
>  算子集和 onnx 包的版本对应，算子集版本号随着包的小版本号增加，新的小版本会更新算子集的旧算子以及添加新的算子

```python
import onnx
print(onnx.__version__, " opset=", onnx.defs.onnx_opset_version())

1.19.0  opset= 23
```

An opset is also attached to every ONNX graphs. It is a global information. It defines the version of all operators inside the graph. Operator _Add_ was updated in version 6, 7, 13 and 14. If the graph opset is 15, it means operator _Add_ follows specifications version 14. If the graph opset is 12, then operator _Add_ follows specifications version 7. An operator in a graph follows its most recent definition below (or equal) the global graph opset.
>  每个 ONNX 图也附有一个算子集，ONNX 图的算子集是全局信息，它定义了图中所有算子的版本
>  图中的算子遵循它在全局的图算子集版本之下 (或等于) 最近的版本。例如，算子 Add 在版本 6, 7, 13, 14 更新，如果图的算子集的版本是 15，说明图中 Add 算子的规范和版本 14 的一致，

A graph may include operators from several domains, `ai.onnx` and `ai.onnx.ml` for example. In that case, the graph must define a global opset for every domain. The rule is applied to every operators within the same domain.
>  一个 ONNX 图可以包含来自多个域的算子，例如 `ai.onnx` 以及 `ai.onnx.ml`
>  当 ONNX 图包含来自多个域的算子时，图必须为每个域定义一个全局算子集，表示图中属于该域的算子的版本

## Subgraphs, tests and loops
ONNX implements tests and loops. They all take another ONNX graphs as an attribute. These structures are usually slow and complex. It is better to avoid them if possible.
>  ONNX 实现了测试和循环结构，它们都以另一个 ONNX 图作为属性，这些结构通常缓慢且复杂，最好避免使用它们

### If
Operator [If](https://onnx.ai/onnx/operators/onnx__If.html#l-onnx-doc-if) executes one of the two graphs depending on the condition evaluation.
>  If 算子取决于条件评估，执行两个图中的一个

```
If(condition) then
    execute this ONNX graph (`then_branch`)
else
    execute this ONNX graph (`else_branch`)
```

Those two graphs can use any result already computed in the graph and must produce the exact same number of outputs. These outputs will be the output of the operator `If`.
>  这两个图可以使用主图中任意已经计算完的结果
>  这两个图必须产生相同数量的输出，这些输出就是 If 算子的输出

![../_images/dot_if.png](https://onnx.ai/onnx/_images/dot_if.png)

### Scan
Operator [Scan](https://onnx.ai/onnx/operators/onnx__Scan.html#l-onnx-doc-scan) implements a loop with a fixed number of iterations. It loops over the rows (or any other dimension) of the inputs and concatenates the outputs along the same axis. 
>  Scan 算子实现了固定数量迭代的循环，该算子遍历输入的所有行 (或者任意其他维度)，并且将输出沿着统一维度拼接

Let’s see an example which implements pairwise distances: $M(i,j) = \|X_i-X_j\|^2$. 

![../_images/dot_scan.png](https://onnx.ai/onnx/_images/dot_scan.png)

This loop is efficient even if it is still slower than a custom implementation of pairwise distances. It assumes inputs and outputs are tensors and automatically concatenate the outputs of every iteration into single tensors. The previous example only has one but it could have several.
>  该循环比自定义实现的算子慢，但也是足够高效的，它假设输入和输出都是张量，并且自动将每次迭代的输出拼接为单个张量

### Loop
Operator [Loop](https://onnx.ai/onnx/operators/onnx__Loop.html#l-onnx-doc-loop) implements a for and a while loop. It can do a fixed number of iterators and/or ends when a condition is not met anymore. Outputs are processed in two different ways. First one is similar to loop [Scan](https://onnx.ai/onnx/operators/onnx__Scan.html#l-onnx-doc-scan), outputs are concatenated into tensors (along the first dimension). This also means that these outputs must have compatible shapes. Second mechanism concatenates tensors into a sequence of tensors.
>  Loop 算子实现了 for 循环和 while 循环，因此可以进行固定次数的迭代或者在条件不满足时退出循环
>  其输出可以以两种不同方式处理，第一种类似 Scan 循环，将输出沿着第一维度拼接为一个张量，这要求这些输出的形状是相互兼容的，第二种是直接将张量连接为张量序列

## Extensibility
ONNX defines a list of operators as the standard: [ONNX Operators](https://onnx.ai/onnx/operators/index.html#l-onnx-operators). However, it is very possible to define your own operators under this domain or a new one. _onnxruntime_ defines custom operators to improve inference. 
>  ONNX 定义了一组标准算子，同时允许在该域下或者新域下自定义新算子
>  onnxruntime 定义用于改进推理的自定义算子 (加速后的自定义算子)

Every node has a type, a name, named inputs and outputs, and attributes. As long as a node is described under these constraints, a node can be added to any ONNX graph.
>  ONNX 图中，每个节点都有一个类型、一个名称、命名的输入和输出，以及属性，只要节点按照这些约束描述，自定义节点就可以被添加到 ONNX 图中

Pairwise distances can be implemented with operator Scan. However, a dedicated operator called CDist is proved significantly faster, significantly enough to make the effort to implement a dedicated runtime for it.

## Functions
Functions are one way to extend ONNX specifications. Some model requires the same combination of operators. This can be avoided by creating a function itself defined with existing ONNX operators. Once defined, a function behaves like any other operators. It has inputs, outputs and attributes.
>  函数是拓展 ONNX 规范的一种方式，一些模型需要相同的算子组合，故我们可以构建一个函数，函数本身由现存的多个 ONNX 算子定义
>  定义好函数后，函数可以像算子一样被使用，具有输入、输出和属性

There are two advantages of using functions. The first one is to have a shorter code and easier to read. The second one is that any onnxruntime can leverage that information to run predictions faster. The runtime could have a specific implementation for a function not relying on the implementation of the existing operators.
>  函数的优势有两点：1. 代码更短更易读 2. 任意 onnxruntime 可以利用函数的信息自定义加速改进的函数实现，且不依赖于现存的算子

## Shape (and Type) Inference
Knowing the shapes of results is not necessary to execute an ONNX graph but this information can be used to make it faster. 
>  执行 ONNX 图不必要需要知道结果的形状，但结果的形状这一信息可以帮助运行更快

If you have the following graph:

```
Add(x, y) -> z
Abs(z) -> w
```

If _x_ and _y_ have the same shape, then _z_ and _w_ also have the same shape. Knowing that, it is possible to reuse the buffer allocated for _z_, to compute the absolute value _w_ inplace. Shape inference helps the runtime to manage the memory and therefore to be more efficient.

>  例如对于上述的 ONNX 图，如果知道 x 和 y 有相同的形状，则 z 和 w 也会有相同的形状，我们进而可以复用为 z 分配的缓存，原地计算 w
>  因此，形状推理可以帮助运行时管理内存，进而提高推理效率

ONNX package can compute in most of the cases the output shape knowing the input shape for every standard operator. It cannot obviously do that for any custom operator outside of the official list.
>  ONNX 包在大多数情况下，对于标准算子，可以在已知输入形状的前提下计算输出形状
>  但对于其他自定义算子则一般不行

## Tools
[netron](https://netron.app/) is very useful to help visualize ONNX graphs. That’s the only one without programming. The first screenshot was made with this tool.

![../_images/linreg1.png](https://onnx.ai/onnx/_images/linreg1.png)

[onnx2py.py](https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/onnx2py.py) creates a python file from an ONNX graph. This script can create the same graph. It may be modified by a user to change the graph.

[zetane](https://github.com/zetane/viewer) can load onnx model and show intermediate results when the model is executed.

>  netron 用于可视化 ONNX 图
>  `onnx2py.py` 根据 ONNX 图创建 python 脚本，该脚本可以构建相同的图，用户可以修改该脚本以修改图
>  zetane 可以在 onnx 模型执行时加载模型并展示中间结果
