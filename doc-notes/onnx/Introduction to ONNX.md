---
completed: true
version: 1.18.0
---
# ONNX Concepts
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
>  构建 ONNX 图意味着使用 ONNX 语言 (更确切的说，是 ONNX 算子)实现一个函数，

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
>  ONNX 最初的开发目的是用于部署深度学习模型，因此其规范最初是为 32 位浮点数涉及
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

1.18.0  opset= 23
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

# ONNX with Python
Next sections highlight the main functions used to build an ONNX graph with the [Python API](https://onnx.ai/onnx/api/index.html#l-python-onnx-api) _onnx_ offers.
>  本节介绍使用 onnx 提供的 Python API 来构建 ONNX 图

## A simple example: a linear regression
The linear regression is the most simple model in machine learning described by the following expression $Y=XA+B$. We can see it as a function of three variables $Y=f(X,A,B)$ decomposed into `y = Add(MatMul(X, A), B)`. That’s what we need to represent with ONNX operators. 
>  考虑简单的线性回归模型，函数 $Y=f(X, A, B)$ 可以分解为 `y = Add(MatMul(X, A), B)` ，我们需要用 ONNX 算子表示它

The first thing is to implement a function with [ONNX operators](https://onnx.ai/onnx/operators/index.html#l-onnx-operators). ONNX is strongly typed. Shape and type must be defined for both input and output of the function. That said, we need four functions to build the graph among the [Helper functions to make ONNX graph components](https://onnx.ai/onnx/api/helper.html#l-onnx-make-function):

- `make_tensor_value_info`: declares a variable (input or output) given its shape and type
- `make_node`: creates a node defined by an operation (an operator type), its inputs and outputs
- `make_graph`: a function to create an ONNX graph with the objects created by the two previous functions
- `make_model`: a last function which merges the graph and additional metadata

>  因此我们首先需要用 ONNX 算子实现一个函数
>  ONNX 是强类型语言，函数的输入和输出的形状和类型都必须被定义
>  我们可以借助 ONNX 提供的辅助函数来构建 ONNX 图组件，我们目前需要其中的四个，包括：
>  - `make_tensor_value_info` : 给定形状和类型，声明一个变量 (输入或输出)
>  - `make_node`: 构建由一个算子 (算子类型) 和其输入输出定义的节点
>  - `make_graph`: 根据以上两个函数构建的成分构建 ONNX 图
>  - `make_model`: 将构建好的图和额外的元数据合并

All along the creation, we need to give a name to every input, output of every node of the graph. Input and output of the graph are defined by onnx objects, strings are used to refer to intermediate results. 
>  在构建过程中，我们需要为图中每个节点的输入和输出命名，ONNX 图中的输入和输出由 onnx 对象定义，字符串用于引用中间结果

This is how it looks like.

```python
# imports

from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info) # 利用圆括号的隐式行连接进行多行导入，避免使用反斜杠 \ 进行显式行连接
from onnx.checker import check_model

# inputs

# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

# outputs, the shape is left undefined

Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# nodes

# It creates a node defined by the operator type MatMul,
# 'X', 'A' are the inputs of the node, 'XA' the output.
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

# from nodes to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.

graph = make_graph([node1, node2],  # nodes
                    'lr',  # a name
                    [X, A, B],  # inputs
                    [Y])  # outputs

# onnx graph
# there is no metadata in this case.

onnx_model = make_model(graph)

# Let's check the model is consistent,
# this function is described in section
# Checker and Shape Inference.
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

>  上例中
>  `make_tensor_value_info` 定义了模型的输入和输出，第一个参数是名称，第二个参数是数据类型，第三个参数是形状，形状留 `None` 表示未定义
>  `make_node` 根据算子类型、输入输出定义节点，第一个参数是算子类型名称，第二个参数是输入，第三个参数是输出
>  `make_graph` 根据节点列表、输入列表、输出列表定义图
>  `make_model` 根据图、元数据定义模型
>  `check_model` 函数用于验证模型是否符合 ONNX 规范，确保所有算子都是有效的且输入输出类型匹配正确

```
ir_version: 11
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 23
}
```

![../_images/dot_linreg.png](https://onnx.ai/onnx/_images/dot_linreg.png)

An empty shape (`None`) means any shape, a shape defined as `[None, None]` tells this object is a tensor with two dimensions without any further precision. 
>  空形状 `None` 表示任意形状，`[None, None]` 仅仅表示对象是两个维度的张量

The ONNX graph can also be inspected by looking into the fields of each object of the graph.
>  可以通过查看 ONNX 图中的每个对象的字段来更细致地检查ONNX 图，如下所示

```python
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

# the list of inputs
print('** inputs **')
print(onnx_model.graph.input)

# in a more nicely format
print('** inputs **')
for obj in onnx_model.graph.input:
    print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))

# the list of outputs
print('** outputs **')
print(onnx_model.graph.output)

# in a more nicely format
print('** outputs **')
for obj in onnx_model.graph.output:
    print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))

# the list of nodes
print('** nodes **')
print(onnx_model.graph.node)

# in a more nicely format
print('** nodes **')
for node in onnx_model.graph.node:
    print("name=%r type=%r input=%r output=%r" % (
        node.name, node.op_type, node.input, node.output))
```

>  上例中
>  辅助函数 `shape2tuple` 用于将 ONNX 的形状 `shape` 转化为 Python 中的元组，`getattr(d, 'dim_value', 0)` 表示如果维度 `d` 没有名为 `dim_value` 的属性，则返回 `0`

```
** inputs **
[name: "X"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
      }
      dim {
      }
    }
  }
}
, name: "A"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
      }
      dim {
      }
    }
  }
}
, name: "B"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
      }
      dim {
      }
    }
  }
}
]
** inputs **
name='X' dtype=1 shape=(0, 0)
name='A' dtype=1 shape=(0, 0)
name='B' dtype=1 shape=(0, 0)
** outputs **
[name: "Y"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
      }
    }
  }
}
]
** outputs **
name='Y' dtype=1 shape=(0,)
** nodes **
[input: "X"
input: "A"
output: "XA"
op_type: "MatMul"
, input: "XA"
input: "B"
output: "Y"
op_type: "Add"
]
** nodes **
name='' type='MatMul' input=['X', 'A'] output=['XA']
name='' type='Add' input=['XA', 'B'] output=['Y']
```

The tensor type is an integer (= 1). 

The helper function [`onnx.helper.tensor_dtype_to_np_dtype()`](https://onnx.ai/onnx/api/helper.html#onnx.helper.tensor_dtype_to_np_dtype "onnx.helper.tensor_dtype_to_np_dtype") gives the corresponding type with numpy.

```python
from onnx import TensorProto
from onnx.helper import tensor_dtype_to_np_dtype, tensor_dtype_to_string

np_dtype = tensor_dtype_to_np_dtype(TensorProto.FLOAT)
print(f"The converted numpy dtype for {tensor_dtype_to_string(TensorProto.FLOAT)} is {np_dtype}.")
```

The converted numpy dtype for TensorProto.FLOAT is float32.

>  辅助函数 `onnx.helper.tensor.tensor_dtype_to_np_dtype` 将对应的 ONNX 张量类型转化为 Numpy 中的 dtype

## Serialization
ONNX is built on the top of protobuf. It adds the necessary definitions to describe a machine learning model and most of the time, ONNX is used to serialize or deserialize a model. First section addresses this need. Second section introduces the serialization and deserialization of data such as tensors, sparse tensors…
>  ONNX 基于 protobuf (Protocol Buffers)构建，它在 protobuf 上添加了必要的定义来描述机器学习模型，在大多数情况下，可以直接用 ONNX 来序列化或反序列化机器学习模型

### Model Serialization
The model needs to be saved to be deployed. ONNX is based on protobuf. It minimizes the space needed to save the graph on disk. Every object (see [Protos](https://onnx.ai/onnx/api/classes.html#l-onnx-classes)) in onnx can be serialized with method `SerializeToString`. That’s the case for the whole model.
>  ONNX 最小化了在磁盘上保存 ONNX 图所需要的空间
>  onnx 中的每个对象都可以用方法 `SerializeToString` 序列化，整个模型也是如此

```python
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

# The serialization
with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# display
print(onnx_model)
```

>  上例中，我们直接调用了 `onnx_model` 的 `SerializeToString` 方法将模型序列化为字节串形式，并写入文件 `linear_regression.onnx` 中
>  `f.write()` 用于将字符串或字节数据写入文件 `f`。如果文件是以文本形式打开，则该方法接受一个字符串参数，将字符串写入；如果是以二进制形式打开，则该方法接受一个字节序列参数，并将其写入

```
ir_version: 11
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 23
}
```

The graph can be restored with function `load`:
>  函数 `onnx.load` 可以直接将文件反序列化为 ONNX 图，其中文件以二进制模式打开

```python
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

# display
print(onnx_model)
```

```
ir_version: 11
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 23
}
```

It looks exactly the same. Any model can be serialized this way unless they are bigger than 2 Gb. protobuf is limited to size smaller than this threshold. Next sections will show how to overcome that limit.
>  任意模型都可以按照这样直接序列化，除非其大小超过两个 G，这是 Protocol Buffers 的阈值大小

### Data Serialization
The serialization of tensor usually happens like the following:
>  张量的序列化如下所示

```python
import numpy
from onnx.numpy_helper import from_array

numpy_tensor = numpy.array([0, 1, 4, 5, 3], dtype=numpy.float32)
print(type(numpy_tensor))

onnx_tensor = from_array(numpy_tensor)
print(type(onnx_tensor))

serialized_tensor = onnx_tensor.SerializeToString()
print(type(serialized_tensor))

with open("saved_tensor.pb", "wb") as f:
    f.write(serialized_tensor)
```

>  上例中
>  我们先构造了 Numpy ndarray，然后使用 `onnx.numpy_helper.from_array` 方法将其转化为 ONNX 张量，之后调用了其 `SerializeToString` 方法将其序列化为字节序列

```
<class 'numpy.ndarray'>
<class 'onnx.onnx_ml_pb2.TensorProto'>
<class 'bytes'>
```

And the deserialization like:

```python
from onnx import TensorProto
from onnx.numpy_helper import to_array

with open("saved_tensor.pb", "rb") as f:
    serialized_tensor = f.read()
print(type(serialized_tensor))

onnx_tensor = TensorProto()
onnx_tensor.ParseFromString(serialized_tensor)
print(type(onnx_tensor))

numpy_tensor = to_array(onnx_tensor)
print(numpy_tensor)
```

>  上例中使用了 `onnx.TensorProto` 的 `ParseFromString` 方法将字节序列反序列化为 ONNX 张量，并使用 `onnx.numpy_helper.to_array` 张量转化为了 Numpy ndarray

```
<class 'bytes'>
<class 'onnx.onnx_ml_pb2.TensorProto'>
[0. 1. 4. 5. 3.]
```

The same schema can be used for but not limited to [TensorProto](https://onnx.ai/onnx/api/classes.html#l-tensorproto):
>  不限于 `TensorProto` ，`onnx` 包中的以下类都可以使用该方法序列化和反序列化

```python
import onnx
import pprint
pprint.pprint([p for p in dir(onnx)
               if p.endswith('Proto') and p[0] != '_'])

['AttributeProto',
 'FunctionProto',
 'GraphProto',
 'MapProto',
 'ModelProto',
 'NodeProto',
 'OperatorProto',
 'OperatorSetIdProto',
 'OperatorSetProto',
 'OptionalProto',
 'SequenceProto',
 'SparseTensorProto',
 'StringStringEntryProto',
 'TensorProto',
 'TensorShapeProto',
 'TrainingInfoProto',
 'TypeProto',
 'ValueInfoProto']
```

This code can be simplified with function _load_tensor_from_string_ (see [Load a Proto](https://onnx.ai/onnx/api/serialization.html#l-onnx-load-data)).

```python
from onnx import load_tensor_from_string

with open("saved_tensor.pb", "rb") as f:
    serialized = f.read()
proto = load_tensor_from_string(serialized)
print(type(proto))
```

>  还可以用函数 `onnx.load_tensor_from_string` 进一步简化代码，该函数接受字节序列，直接反序列化为张量

```
<class 'onnx.onnx_ml_pb2.TensorProto'>
```

## Initializer, default value
The previous model assumed the coefficients of the linear regression were also input of the model. That’s not very convenient. They should be part of the model itself as constant or **initializer** to follow onnx semantic. 
>  之前的模型假设了线性回归的系数也是模型输入，但实际情况下它们应该作为常量成为模型的一部分，在 onnx 语义下就是指模型的初始化值

Next example modifies the previous one to change inputs `A` and `B` into initializers. 
>  我们考虑将之前的 `A, B` 从输入改为初始化值

The package implements two functions to convert from numpy into onnx and the other way around (see [array](https://onnx.ai/onnx/api/numpy_helper.html#l-numpy-helper-onnx-array)).

- `onnx.numpy_helper.to_array`: converts from onnx to numpy
- `onnx.numpy_helper.from_array`: converts from numpy to onnx

>  onnx 包实现了两个函数帮助进行 numpy 数组和 onnx 张量之间的转化
>  - `onnx.numpy_helper.to_array`
>  - `onnx.numpy_helper.from_array`

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# initializers
value = numpy.array([0.5, -0.6], dtype=numpy.float32)
A = numpy_helper.from_array(value, name='A')

value = numpy.array([0.4], dtype=numpy.float32)
C = numpy_helper.from_array(value, name='C')

# the part which does not change
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['AX'])
node2 = make_node('Add', ['AX', 'C'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
onnx_model = make_model(graph)
check_model(onnx_model)

print(onnx_model)
```

>  上例中先用 Numpy 数组定义好了张量的值，然后用 `from_array` 函数将数组转化为张量，此时的张量就有了初始化好的值，只需和之前一样利用这些张量构建节点即可 (没有额外的语义，和之前的差异仅在于把需要初始化的张量初始化了，初始化好的张量仍然作为算子输入的一部分)

```
ir_version: 11
graph {
  node {
    input: "X"
    input: "A"
    output: "AX"
    op_type: "MatMul"
  }
  node {
    input: "AX"
    input: "C"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  initializer {
    dims: 2
    data_type: 1
    name: "A"
    raw_data: "\000\000\000?\232\231\031\277"
  }
  initializer {
    dims: 1
    data_type: 1
    name: "C"
    raw_data: "\315\314\314>"
  }
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 23
}
```

![../_images/dot_linreg2.png](https://onnx.ai/onnx/_images/dot_linreg2.png)

Again, it is possible to go through the onnx structure to check how the initializers look like.
>  ONNX 图的初始化值可以通过属性 `graph.initializer` 访问

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# initializers
value = numpy.array([0.5, -0.6], dtype=numpy.float32)
A = numpy_helper.from_array(value, name='A')

value = numpy.array([0.4], dtype=numpy.float32)
C = numpy_helper.from_array(value, name='C')

# the part which does not change
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['AX'])
node2 = make_node('Add', ['AX', 'C'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
onnx_model = make_model(graph)
check_model(onnx_model)

print('** initializer **')
for init in onnx_model.graph.initializer:
    print(init)
```

```
** initializer **
dims: 2
data_type: 1
name: "A"
raw_data: "\000\000\000?\232\231\031\277"

dims: 1
data_type: 1
name: "C"
raw_data: "\315\314\314>"
```

The type is defined as integer as well with the same meaning. 
>  初始化值的类型同样被定义为整数，具体的整数对应具体的类型

In this second example, there is only one input left. Input `A` and `B` were removed. They could be kept. In that case, they are optional: every initializer sharing the same name as input is considered as a default value. It replaces the input if this one is not given.
>  本例中，模型仅需要一个输入，输入 `A, B` 被移除了
>  实际上，它们的状态变为了可选的，ONNX 将任何与输入张量相同名字的初始化值视作该输入张量的默认值，如果没有提供该输入张量，就使用该默认值

## Attributes
Some operators need attributes such as [Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html#l-onnx-doc-transpose) operator.
>  一些算子需要属性，例如 Transpose 算子

 Let’s build the graph for expression $y=XA'+B$ or `y = Add(MatMul(X, Transpose(A)) + B)`. Transpose needs an attribute defining the permutation of axes: `perm=[1, 0]`. It is added as a named attribute in function `make_node`.
 >  考虑为表达式和 $y = XA' + B$ 或 `y = Add(MatMul(X, Transpose(A)), B)` 构建图，其中的转置操作需要一个定义轴置换的属性 `perm=[1, 0]`
 >  算子的属性需要在函数 `make_node` 中添加，如下所示

```python
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# unchanged
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# added
node_transpose = make_node('Transpose', ['A'], ['tA'], perm=[1, 0])

# unchanged except A is replaced by tA
node1 = make_node('MatMul', ['X', 'tA'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

# node_transpose is added to the list
graph = make_graph([node_transpose, node1, node2],
                   'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

```
ir_version: 11
graph {
  node {
    input: "A"
    output: "tA"
    op_type: "Transpose"
    attribute {
      name: "perm"
      ints: 1
      ints: 0
      type: INTS
    }
  }
  node {
    input: "X"
    input: "tA"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 23
}
```

![../_images/dot_att.png](https://onnx.ai/onnx/_images/dot_att.png)

The whole list of _make_ functions is the following. Many of them are described in section [Helper functions to make ONNX graph components](https://onnx.ai/onnx/api/helper.html#l-onnx-make-function).
>  `onnx` 包中所有的 `make_` 函数如下所示

```python
import onnx
import pprint
pprint.pprint([k for k in dir(onnx.helper)
               if k.startswith('make')])

['make_attribute',
 'make_attribute_ref',
 'make_empty_tensor_value_info',
 'make_function',
 'make_graph',
 'make_map',
 'make_map_type_proto',
 'make_model',
 'make_model_gen_version',
 'make_node',
 'make_operatorsetid',
 'make_opsetid',
 'make_optional',
 'make_optional_type_proto',
 'make_sequence',
 'make_sequence_type_proto',
 'make_sparse_tensor',
 'make_sparse_tensor_type_proto',
 'make_sparse_tensor_value_info',
 'make_tensor',
 'make_tensor_sequence_value_info',
 'make_tensor_type_proto',
 'make_tensor_value_info',
 'make_training_info',
 'make_value_info']
```

## Opset and metadata
Let’s load the ONNX file previously created and check what kind of metadata it has.

```python
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

for field in ['doc_string', 'domain', 'functions',
              'ir_version', 'metadata_props', 'model_version',
              'opset_import', 'producer_name', 'producer_version',
              'training_info']:
    print(field, getattr(onnx_model, field))
```

>  之前定义的模型中，以上的这些属性表示了模型的元数据

```
doc_string 
domain 
functions []
ir_version 11
metadata_props []
model_version 0
opset_import [version: 23
]
producer_name 
producer_version 
training_info []
```

Most of them are empty because it was not filled when the ONNX graph was created. Two of them have a value:
>  可以看到在没有定义时，其中大多数属性都是空值
>  其中仅有 `ir_version` 和 `opset_import` 在没有定义时有值

```python
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

print("ir_version:", onnx_model.ir_version)
for opset in onnx_model.opset_import:
    print("opset domain=%r version=%r" % (opset.domain, opset.version))
```

```
ir_version: 11
opset domain='' version=23
```

`IR` defined the version of ONNX language. Opset defines the version of operators being used. Without any precision, ONNX uses the latest version available coming from the installed package. 
>  IR 定义了 ONNX 语言的版本，Opset 定义了 ONNX 图使用的算子的版本
>  没有指定时，ONNX 使用安装包可以提供的最新版本

Another one can be used.
>  也可以使用其他版本的算子集合

```python
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 14

for opset in onnx_model.opset_import:
    print("opset domain=%r version=%r" % (opset.domain, opset.version))
```

```
opset domain='' version=14
```

>  上述代码将 `opset_import` 中的条目删除，并添加一个新条目，设置该条目的域为 `''` (意味着它属于默认域)，并且设置版本为 `14`

Any opset can be used as long as all operators are defined the way ONNX specifies it. 

Version 5 of operator _Reshape_ defines the shape as an input and not as an attribute like in version 1. The opset tells which specifications is followed while describing the graph.

The other metadata can be used to store any information, to store information about the way the model was generated, a way to distinguish a model from another one with a version number.
>  可以手动定义其他元数据的值，定义模型的相关信息，如下所示

```python
from onnx import load, helper

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

onnx_model.model_version = 15
onnx_model.producer_name = "something"
onnx_model.producer_version = "some other thing"
onnx_model.doc_string = "documentation about this model"
prop = onnx_model.metadata_props

data = dict(key1="value1", key2="value2")
helper.set_model_props(onnx_model, data)

print(onnx_model)
```

```
ir_version: 11
producer_name: "something"
producer_version: "some other thing"
model_version: 15
doc_string: "documentation about this model"
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 23
}
metadata_props {
  key: "key1"
  value: "value1"
}
metadata_props {
  key: "key2"
  value: "value2"
}
```

Field `training_info` can be used to store additional graphs. See [training_tool_test.py](https://github.com/onnx/onnx/blob/main/onnx/test/training_tool_test.py) to see how it works.
>  字段 `training_info` 可以用于存储额外的图

## Subgraph: test and loops
They are usually grouped in a category called _control flow_. It is usually better to avoid them as they are not as efficient as the matrix operation are much faster and optimized.
>  test 和 loops 通常都归类为控制流，由于它们不如矩阵运算高效，因此最好避免使用它们

### If
A test can be implemented with operator [If](https://onnx.ai/onnx/operators/onnx__If.html#l-onnx-doc-if). It executes one subgraph or another depending on one boolean. This is not used very often as a function usually needs the result of many comparisons in a batch. 
>  test 可以通过算子 If 实现，该算子根据一个布尔值决定执行哪一个子图
>  由于函数通常需要一个 batch 内多个比较的结果，故该算子并不常用

The following example computes the sum of all floats in a matrix based on the sign, returns 1 or -1.

```python
import numpy
import onnx
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession

# initializers
value = numpy.array([0], dtype=numpy.float32)
zero = from_array(value, name='zero')

# Same as before, X is the input, Y is the output.
X = make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None])

# The node building the condition. The first one
# sum over all axes.
rsum = make_node('ReduceSum', ['X'], ['rsum'])
# The second compares the result to 0.
cond = make_node('Greater', ['rsum', 'zero'], ['cond'])

# Builds the graph is the condition is True.
# Input for then
then_out = make_tensor_value_info(
    'then_out', onnx.TensorProto.FLOAT, None)
# The constant to return.
then_cst = from_array(numpy.array([1]).astype(numpy.float32))

# The only node.
then_const_node = make_node(
    'Constant', inputs=[],
    outputs=['then_out'],
    value=then_cst, name='cst1')

# And the graph wrapping these elements.
then_body = make_graph(
    [then_const_node], 'then_body', [], [then_out])

# Same process for the else branch.
else_out = make_tensor_value_info(
    'else_out', onnx.TensorProto.FLOAT, [5])
else_cst = from_array(numpy.array([-1]).astype(numpy.float32))

else_const_node = make_node(
    'Constant', inputs=[],
    outputs=['else_out'],
    value=else_cst, name='cst2')

else_body = make_graph(
    [else_const_node], 'else_body',
    [], [else_out])

# Finally the node If taking both graphs as attributes.
if_node = onnx.helper.make_node(
    'If', ['cond'], ['Y'],
    then_branch=then_body,
    else_branch=else_body)

# The final graph.
graph = make_graph([rsum, cond, if_node], 'if', [X], [Y], [zero])
onnx_model = make_model(graph)
check_model(onnx_model)

# Let's freeze the opset.
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15
onnx_model.ir_version = 8

# Save.
with open("onnx_if_sign.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Let's see the output.
sess = InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])

x = numpy.ones((3, 2), dtype=numpy.float32)
res = sess.run(None, {'X': x})

# It works.
print("result", res)
print()

# Some display.
print(onnx_model)
```

>  上例中
>  `ReduceSum` 节点对输入张量的所有元素求和
>  then 分支是一个仅包含一个节点的子图，该节点直接返回值为 1 的常量张量，else 分支类似，返回值为 -1 的常量张量
>  `If` 节点接受条件 `cond` 作为输入，根据条件的结果选择执行哪个分支，同时返回一个结果
>  最后的图仅需要通过 `If` 节点构建即可
>  `onnxruntime.InferenceSession` 根据模型和要使用的执行提供程序 (这里是 CPU) 构建运行会话，会话的 `run` 方法运行模型

```
result [array([1.], dtype=float32)]
ir_version: 8
graph {
  node {
    input: "X"
    output: "rsum"
    op_type: "ReduceSum"
  }
  node {
    input: "rsum"
    input: "zero"
    output: "cond"
    op_type: "Greater"
  }
  node {
    input: "cond"
    output: "Y"
    op_type: "If"
    attribute {
      name: "else_branch"
      g {
        node {
          output: "else_out"
          name: "cst2"
          op_type: "Constant"
          attribute {
            name: "value"
            t {
              dims: 1
              data_type: 1
              raw_data: "\000\000\200\277"
            }
            type: TENSOR
          }
        }
        name: "else_body"
        output {
          name: "else_out"
          type {
            tensor_type {
              elem_type: 1
              shape {
                dim {
                  dim_value: 5
                }
              }
            }
          }
        }
      }
      type: GRAPH
    }
    attribute {
      name: "then_branch"
      g {
        node {
          output: "then_out"
          name: "cst1"
          op_type: "Constant"
          attribute {
            name: "value"
            t {
              dims: 1
              data_type: 1
              raw_data: "\000\000\200?"
            }
            type: TENSOR
          }
        }
        name: "then_body"
        output {
          name: "then_out"
          type {
            tensor_type {
              elem_type: 1
            }
          }
        }
      }
      type: GRAPH
    }
  }
  name: "if"
  initializer {
    dims: 1
    data_type: 1
    name: "zero"
    raw_data: "\000\000\000\000"
  }
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 15
}
```

The whole is easier to visualize with the following image.

![../_images/dot_if_py.png](https://onnx.ai/onnx/_images/dot_if_py.png)

Both else and then branches are very simple. Node _If_ could even be replaced with a node _Where_ and that would be faster. It becomes interesting when both branches are bigger and skipping one is more efficient.

### Scan
[Scan](https://onnx.ai/onnx/operators/onnx__Scan.html#l-onnx-doc-scan) seems quite complex when reading the specifications. It is useful to loop over one dimension of a tensor and store the results in a preallocated tensor.
>  Scan 算子用于循环遍历张量的一个维度，并将结果存储在预先分配的张量中

The following example implements a classic nearest neighbors for a regression problem. The first step consists in computing the pairwise distances between the input features _X_ and the training set _W_: $dist(X,W) = (M_{ij})=(\|X_i - W_j\|_2^2)$. It is followed by an operator [TopK](https://onnx.ai/onnx/operators/onnx__TopK.html#l-onnx-doc-topk) which extracts the _k_ nearest neighbors.
>  本例实现最近邻算法，它首先计算输入特征 $X$ 和训练集 $W$ 的成对距离，然后选出 topK 个最近邻

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# subgraph
initializers = []
nodes = []
inputs = []
outputs = []

value = make_tensor_value_info('next_in', 1, [None, 4])
inputs.append(value)
value = make_tensor_value_info('next', 1, [None])
inputs.append(value)

value = make_tensor_value_info('next_out', 1, [None, None])
outputs.append(value)
value = make_tensor_value_info('scan_out', 1, [None])
outputs.append(value)

node = make_node(
    'Identity', ['next_in'], ['next_out'],
    name='cdistd_17_Identity', domain='') # 将 next_in 直接赋值给 next_out
nodes.append(node)

node = make_node(
    'Sub', ['next_in', 'next'], ['cdistdf_17_C0'],
    name='cdistdf_17_Sub', domain='') # 计算 next_in - next
nodes.append(node)

node = make_node(
    'ReduceSumSquare', ['cdistdf_17_C0'], ['cdistdf_17_reduced0'],
    name='cdistdf_17_ReduceSumSquare', axes=[1], keepdims=0, domain='') # 在第一轴上求平方和，不保留维度
nodes.append(node)

node = make_node(
    'Identity', ['cdistdf_17_reduced0'],
    ['scan_out'], name='cdistdf_17_Identity', domain='') # 将平方和结果赋值给 scan_out
nodes.append(node)

graph = make_graph(nodes, 'OnnxIdentity',
                   inputs, outputs, initializers) # inputs 包括 next_in, next, outputs 包括 next_out, scan_out, 初始化值为空

# main graph
initializers = []
nodes = []
inputs = []
outputs = []

opsets = {'': 15, 'ai.onnx.ml': 15}
target_opset = 15  # subgraphs

# initializers
list_value = [23.29599822460675, -120.86516699239603, -144.70495899914215, -260.08772982740413,
              154.65272105889147, -122.23295157108991, 247.45232560871727, -182.83789715805776,
              -132.92727431421793, 147.48710175784703, 88.27761768038069, -14.87785569894749,
              111.71487894705504, 301.0518319089629, -29.64235742280055, -113.78493504731911,
              -204.41218591022718, 112.26561056133608, 66.04032954135549,
              -229.5428380626701, -33.549262642481615, -140.95737409864623, -87.8145187836131,
              -90.61397011283958, 57.185488100413366, 56.864151796743855, 77.09054590340892,
              -187.72501631246712, -42.779503579806025, -21.642642730674076, -44.58517761667535,
              78.56025104939847, -23.92423223842056, 234.9166231927213, -73.73512816431007,
              -10.150864499514297, -70.37105466673813, 65.5755688281476, 108.68676290979731, -78.36748960443065]
value = numpy.array(list_value, dtype=numpy.float64).reshape((2, 20))
tensor = numpy_helper.from_array(
    value, name='knny_ArrayFeatureExtractorcst')
initializers.append(tensor)

list_value = [1.1394007205963135, -0.6848101019859314, -1.234825849533081, 0.4023416340351105,
              0.17742614448070526, 0.46278226375579834, -0.4017809331417084, -1.630198359489441,
              -0.5096521973609924, 0.7774903774261475, -0.4380742907524109, -1.2527953386306763,
              -1.0485529899597168, 1.950775384902954, -1.420017957687378, -1.7062702178955078,
              1.8675580024719238, -0.15135720372200012, -0.9772778749465942, 0.9500884413719177,
              -2.5529897212982178, -0.7421650290489197, 0.653618574142456, 0.8644362092018127,
              1.5327792167663574, 0.37816253304481506, 1.4693588018417358, 0.154947429895401,
              -0.6724604368209839, -1.7262825965881348, -0.35955315828323364, -0.8131462931632996,
              -0.8707971572875977, 0.056165341287851334, -0.5788496732711792, -0.3115525245666504,
              1.2302906513214111, -0.302302747964859, 1.202379822731018, -0.38732680678367615,
              2.269754648208618, -0.18718385696411133, -1.4543657302856445, 0.04575851559638977,
              -0.9072983860969543, 0.12898291647434235, 0.05194539576768875, 0.7290905714035034,
              1.4940791130065918, -0.8540957570075989, -0.2051582634449005, 0.3130677044391632,
              1.764052391052246, 2.2408931255340576, 0.40015721321105957, 0.978738009929657,
              0.06651721894741058, -0.3627411723136902, 0.30247190594673157, -0.6343221068382263,
              -0.5108051300048828, 0.4283318817615509, -1.18063223361969, -0.02818222902715206,
              -1.6138978004455566, 0.38690251111984253, -0.21274028718471527, -0.8954665660858154,
              0.7610377073287964, 0.3336743414402008, 0.12167501449584961, 0.44386324286460876,
              -0.10321885347366333, 1.4542734622955322, 0.4105985164642334, 0.14404356479644775,
              -0.8877857327461243, 0.15634897351264954, -1.980796456336975, -0.34791216254234314]
value = numpy.array(list_value, dtype=numpy.float32).reshape((20, 4))
tensor = numpy_helper.from_array(value, name='Sc_Scancst')
initializers.append(tensor)

value = numpy.array([2], dtype=numpy.int64)
tensor = numpy_helper.from_array(value, name='To_TopKcst')
initializers.append(tensor)

value = numpy.array([2, -1, 2], dtype=numpy.int64)
tensor = numpy_helper.from_array(value, name='knny_Reshapecst')
initializers.append(tensor)

# inputs
value = make_tensor_value_info('input', 1, [None, 4])
inputs.append(value)

# outputs
value = make_tensor_value_info('variable', 1, [None, 2])
outputs.append(value)

# nodes

node = make_node(
    'Scan', ['input', 'Sc_Scancst'], ['UU032UU', 'UU033UU'],
    name='Sc_Scan', body=graph, num_scan_inputs=1, domain='')
nodes.append(node)

node = make_node(
    'Transpose', ['UU033UU'], ['Tr_transposed0'],
    name='Tr_Transpose', perm=[1, 0], domain='')
nodes.append(node)

node = make_node(
    'Sqrt', ['Tr_transposed0'], ['Sq_Y0'],
    name='Sq_Sqrt', domain='')
nodes.append(node)

node = make_node(
    'TopK', ['Sq_Y0', 'To_TopKcst'], ['To_Values0', 'To_Indices1'],
    name='To_TopK', largest=0, sorted=1, domain='')
nodes.append(node)

node = make_node(
    'Flatten', ['To_Indices1'], ['knny_output0'],
    name='knny_Flatten', domain='')
nodes.append(node)

node = make_node(
    'ArrayFeatureExtractor',
    ['knny_ArrayFeatureExtractorcst', 'knny_output0'], ['knny_Z0'],
    name='knny_ArrayFeatureExtractor', domain='ai.onnx.ml')
nodes.append(node)

node = make_node(
    'Reshape', ['knny_Z0', 'knny_Reshapecst'], ['knny_reshaped0'],
    name='knny_Reshape', allowzero=0, domain='')
nodes.append(node)

node = make_node(
    'Transpose', ['knny_reshaped0'], ['knny_transposed0'],
    name='knny_Transpose', perm=[1, 0, 2], domain='')
nodes.append(node)

node = make_node(
    'Cast', ['knny_transposed0'], ['Ca_output0'],
    name='Ca_Cast', to=TensorProto.FLOAT, domain='')
nodes.append(node)

node = make_node(
    'ReduceMean', ['Ca_output0'], ['variable'],
    name='Re_ReduceMean', axes=[2], keepdims=0, domain='')
nodes.append(node)

# graph
graph = make_graph(nodes, 'KNN regressor', inputs, outputs, initializers)

# model
onnx_model = make_model(graph)
onnx_model.ir_version = 8
onnx_model.producer_name = 'skl2onnx'
onnx_model.producer_version = ''
onnx_model.domain = 'ai.onnx'
onnx_model.model_version = 0
onnx_model.doc_string = ''
set_model_props(onnx_model, {})

# opsets
del onnx_model.opset_import[:]
for dom, value in opsets.items():
    op_set = onnx_model.opset_import.add()
    op_set.domain = dom
    op_set.version = value

check_model(onnx_model)
with open("knnr.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print(onnx_model)
```

```
ir_version: 8
producer_name: "skl2onnx"
producer_version: ""
domain: "ai.onnx"
model_version: 0
doc_string: ""
graph {
  node {
    input: "input"
    input: "Sc_Scancst"
    output: "UU032UU"
    output: "UU033UU"
    name: "Sc_Scan"
    op_type: "Scan"
    attribute {
      name: "body"
      g {
        node {
          input: "next_in"
          output: "next_out"
          name: "cdistd_17_Identity"
          op_type: "Identity"
          domain: ""
        }
        node {
          input: "next_in"
          input: "next"
          output: "cdistdf_17_C0"
          name: "cdistdf_17_Sub"
          op_type: "Sub"
          domain: ""
        }
        node {
          input: "cdistdf_17_C0"
          output: "cdistdf_17_reduced0"
          name: "cdistdf_17_ReduceSumSquare"
          op_type: "ReduceSumSquare"
          attribute {
            name: "axes"
            ints: 1
            type: INTS
          }
          attribute {
            name: "keepdims"
            i: 0
            type: INT
          }
          domain: ""
        }
        node {
          input: "cdistdf_17_reduced0"
          output: "scan_out"
          name: "cdistdf_17_Identity"
          op_type: "Identity"
          domain: ""
        }
        name: "OnnxIdentity"
        input {
          name: "next_in"
          type {
            tensor_type {
              elem_type: 1
              shape {
                dim {
                }
                dim {
                  dim_value: 4
                }
              }
            }
          }
        }
        input {
          name: "next"
          type {
            tensor_type {
              elem_type: 1
              shape {
                dim {
                }
              }
            }
          }
        }
        output {
          name: "next_out"
          type {
            tensor_type {
              elem_type: 1
              shape {
                dim {
                }
                dim {
                }
              }
            }
          }
        }
        output {
          name: "scan_out"
          type {
            tensor_type {
              elem_type: 1
              shape {
                dim {
                }
              }
            }
          }
        }
      }
      type: GRAPH
    }
    attribute {
      name: "num_scan_inputs"
      i: 1
      type: INT
    }
    domain: ""
  }
  node {
    input: "UU033UU"
    output: "Tr_transposed0"
    name: "Tr_Transpose"
    op_type: "Transpose"
    attribute {
      name: "perm"
      ints: 1
      ints: 0
      type: INTS
    }
    domain: ""
  }
  node {
    input: "Tr_transposed0"
    output: "Sq_Y0"
    name: "Sq_Sqrt"
    op_type: "Sqrt"
    domain: ""
  }
  node {
    input: "Sq_Y0"
    input: "To_TopKcst"
    output: "To_Values0"
    output: "To_Indices1"
    name: "To_TopK"
    op_type: "TopK"
    attribute {
      name: "largest"
      i: 0
      type: INT
    }
    attribute {
      name: "sorted"
      i: 1
      type: INT
    }
    domain: ""
  }
  node {
    input: "To_Indices1"
    output: "knny_output0"
    name: "knny_Flatten"
    op_type: "Flatten"
    domain: ""
  }
  node {
    input: "knny_ArrayFeatureExtractorcst"
    input: "knny_output0"
    output: "knny_Z0"
    name: "knny_ArrayFeatureExtractor"
    op_type: "ArrayFeatureExtractor"
    domain: "ai.onnx.ml"
  }
  node {
    input: "knny_Z0"
    input: "knny_Reshapecst"
    output: "knny_reshaped0"
    name: "knny_Reshape"
    op_type: "Reshape"
    attribute {
      name: "allowzero"
      i: 0
      type: INT
    }
    domain: ""
  }
  node {
    input: "knny_reshaped0"
    output: "knny_transposed0"
    name: "knny_Transpose"
    op_type: "Transpose"
    attribute {
      name: "perm"
      ints: 1
      ints: 0
      ints: 2
      type: INTS
    }
    domain: ""
  }
  node {
    input: "knny_transposed0"
    output: "Ca_output0"
    name: "Ca_Cast"
    op_type: "Cast"
    attribute {
      name: "to"
      i: 1
      type: INT
    }
    domain: ""
  }
  node {
    input: "Ca_output0"
    output: "variable"
    name: "Re_ReduceMean"
    op_type: "ReduceMean"
    attribute {
      name: "axes"
      ints: 2
      type: INTS
    }
    attribute {
      name: "keepdims"
      i: 0
      type: INT
    }
    domain: ""
  }
  name: "KNN regressor"
  initializer {
    dims: 2
    dims: 20
    data_type: 11
    name: "knny_ArrayFeatureExtractorcst"
    raw_data: ",\&\212\306K7@\333z`\345^7^\300\304\312,\006\217\026b\300Z9dWgAp\300.+F\027\343Tc@\203\330\264\255\350\216^\300\260\022\216sy\356n@\237h\263\r\320\332f\300\224\277.;\254\235`\300\336\370lV\226ob@\261\201\362|\304\021V@c,[Mv\301-\300\322\214\240\223\300\355[@)\036\262M\324\320r@nE;\211q\244=\300\021n5`<r\\300\207\211\201\2400\215i\300H\232p\303\377\020\@\317K[\302\224\202P@&\306\355\355^\261l\300\301/\377<N\306@\300#w\001\317\242\236a\300$fd\023!\364U\300\204\327LIK\247V\300J\211\366\022\276\227L@\262\345\254\206\234nL@f{\013\201\313ES@\234\343hU3wg\300\3370\367\305\306cE\300\336A\347;\204\2445\300f\374\242\031\347JF\300\325\2557'\333\243S@\331\354\345{\232\3547\300\307o)\372T]m@#\005\000W\014oR\300'\025\227\034>M$\300\310\252\022\\277\227Q\300l_\243\036\326dP@\333kk\354\363+[@\223)\036\363\204\227S\300"
  }
  initializer {
    dims: 20
    dims: 4
    data_type: 1
    name: "Sc_Scancst"
    raw_data: "\342\327\221?\267O/\277\306\016\236\277\271\377\315>3\2575>\314\361\354>;\266\315\276W\252\320\277\221x\002\277\234\tG?FK\340\276\231[\240\277\3746\206\277\002\263\371?&\303\265\277\020g\332\277$\014\357?b\375\032\276\342.z\277\3778s?/d#\300\207\376=\277\214S'?\261K]?\0342\304?\205\236\301>\363\023\274?\212\252\036>^&,\277\324\366\334\277Z\027\270\276[*P\277\220\354^\277\241\rf=~/\024\277\320\203\237\276*z\235?m\307\232\276\225\347\231?\263O\306\276\251C\021@ \255?\276\250(\272\277Hm;=\265Dh\277\031\024\004>\262\304T=\256\245:?\374=\277?\005\246Z\277\002\025R\276iJ\240>x\314\341?\313j\017@h\341\314>\223\216z?.:\210=6\271\271\276\231\335\232>\357b"\277 \304\002\277QN\333>\365\036\227\277k\336\346\2744\224\316\277\026\030\306>\227\330Y\276L=e\277^\323B?]\327\252>\3000\371=\013B\343>hd\323\275\242%\272?\3709\322>(\200\023>\355Ec\277\362\031 >\275\212\375\277\213!\262\276"
  }
  initializer {
    dims: 1
    data_type: 7
    name: "To_TopKcst"
    raw_data: "\002\000\000\000\000\000\000\000"
  }
  initializer {
    dims: 3
    data_type: 7
    name: "knny_Reshapecst"
    raw_data: "\002\000\000\000\000\000\000\000\377\377\377\377\377\377\377\377\002\000\000\000\000\000\000\000"
  }
  input {
    name: "input"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  output {
    name: "variable"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 15
}
opset_import {
  domain: "ai.onnx.ml"
  version: 15
}
```

Visually it looks like the following:

![../_images/dot_scan_py.png](https://onnx.ai/onnx/_images/dot_scan_py.png)

The subgraph is executed by operator [Scan](https://onnx.ai/onnx/operators/onnx__Scan.html#l-onnx-doc-scan). In this case, there is one _scan_ input meaning the operator only builds one output.
>  子图由算子 Scan 执行，本例中，`num_scan_inputs=1` 表示只有一个 scan 输入，这也意味着算子仅生成一个输出 (对应于该输入的输出)

```python
node = make_node(
    'Scan', ['X1', 'X2'], ['Y1', 'Y2'],
    name='Sc_Scan', body=graph, num_scan_inputs=1, domain='')
```

At the first iteration, the subgraph gets _X1_ and the first row of _X2_. The graph produces two outputs. The first one replaces _X1_ in the next iteration, the second one is store in a container to form _Y2_. At the second iteration, second input of the subgraph is the second row of _X2_. 
>  在第一次迭代中，子图获取 X1, X2 的第一行，子图产生两个输出，第一个输出替换下次迭代中的 X1，第二个输出存储在一个容器中以形成 Y2
>  第二次迭代中，子图的第二个输入是 X2 的第二行

Here is a short summary. Green is the first iteration, blue the second.

[![../_images/scanop.png](https://onnx.ai/onnx/_images/scanop.png)](https://onnx.ai/onnx/_images/scanop.png)

## Functions
As mentioned in previous chapter, functions can be used to shorten the code to build the model and offer more possibilities to the runtime running predictions to be faster if there exists a specific implementation of this function. If it is not the case, the runtime can still use the default implementation based on existing operators.
>  如果运行时存在函数的特定实现，则会采用该特定实现加速推理，如果不存在，仍然可以使用基于现存算子的默认实现

Function `make_function` is used to define a function. It works like a graph with less types. It is more like a template. This API may evolve. It does not include initializers either.
>  `make_function` 用于定义函数，函数的工作原理类似类型较少的图，更像是一个模板

### A function with no attribute
That’s the more simple case. Every input of the function is a dynamic object known at execution time.
>  下面的例子中，函数的每个输入都是在执行时已知的动态对象

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid,
    make_function)
from onnx.checker import check_model

new_domain = 'custom'
opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)] # 包含默认域的第 14 版和自定义域的第 1 版

# Let's define a function for a linear regression

node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

linear_regression = make_function(
    new_domain,            # domain name
    'LinearRegression',     # function name
    ['X', 'A', 'B'],        # input names
    ['Y'],                  # output names
    [node1, node2],         # nodes
    opset_imports,          # opsets
    [])                     # attribute names

# Let's use it in a graph.

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

graph = make_graph(
    [make_node('LinearRegression', ['X', 'A', 'B'], ['Y1'], domain=new_domain), # 用函数构造节点
     make_node('Abs', ['Y1'], ['Y'])],
    'example',
    [X, A, B], [Y])

onnx_model = make_model(
    graph, opset_imports=opset_imports,
    functions=[linear_regression])  # functions to add)
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

```
ir_version: 11
graph {
  node {
    input: "X"
    input: "A"
    input: "B"
    output: "Y1"
    op_type: "LinearRegression"
    domain: "custom"
  }
  node {
    input: "Y1"
    output: "Y"
    op_type: "Abs"
  }
  name: "example"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 14
}
opset_import {
  domain: "custom"
  version: 1
}
functions {
  name: "LinearRegression"
  input: "X"
  input: "A"
  input: "B"
  output: "Y"
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  opset_import {
    domain: ""
    version: 14
  }
  opset_import {
    domain: "custom"
    version: 1
  }
  domain: "custom"
}
```

### A function with attributes
The following functions are equivalent to the previous one except one input, _B_, was converted into an argument named _bias_. The code is almost the same except the bias is now a constant. Inside the function definition, a node _Constant_ is created to insert the argument as a result. It is linked to the argument with the attribute `ref_attr_name`.
>  下面的函数和之前的函数等效，除了一个输入 B 被转化为了参数 bias
>  此时 bias 是一个常量，我们在函数内部创建 Constant 节点，将参数作为结果插入，该 Constant 节点通过其属性的 `ref_attr_name` 字段获取参数值

```python
import numpy
from onnx import numpy_helper, TensorProto, AttributeProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid,
    make_function)
from onnx.checker import check_model

new_domain = 'custom'
opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

# Let's define a function for a linear regression
# The first step consists in creating a constant
# equal to the input parameter of the function.
cst = make_node('Constant',  [], ['B'])

att = AttributeProto()
att.name = "value"

# This line indicates the value comes from the argument
# named 'bias' the function is given.
att.ref_attr_name = "bias"
att.type = AttributeProto.TENSOR
cst.attribute.append(att)

node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

linear_regression = make_function(
    new_domain,            # domain name
    'LinearRegression',     # function name    
    ['X', 'A'],             # input names
    ['Y'],                  # output names
    [cst, node1, node2],    # nodes
    opset_imports,          # opsets
    ["bias"])               # attribute names

# Let's use it in a graph.

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

graph = make_graph(
    [make_node('LinearRegression', ['X', 'A'], ['Y1'], domain=new_domain,
               # bias is now an argument of the function and is defined as a tensor
               bias=make_tensor('former_B', TensorProto.FLOAT, [1], [0.67])),
     make_node('Abs', ['Y1'], ['Y'])],
    'example',
    [X, A], [Y])

onnx_model = make_model(
    graph, opset_imports=opset_imports,
    functions=[linear_regression])  # functions to add)
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

```
ir_version: 11
graph {
  node {
    input: "X"
    input: "A"
    output: "Y1"
    op_type: "LinearRegression"
    attribute {
      name: "bias"
      t {
        dims: 1
        data_type: 1
        float_data: 0.67
        name: "former_B"
      }
      type: TENSOR
    }
    domain: "custom"
  }
  node {
    input: "Y1"
    output: "Y"
    op_type: "Abs"
  }
  name: "example"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 14
}
opset_import {
  domain: "custom"
  version: 1
}
functions {
  name: "LinearRegression"
  input: "X"
  input: "A"
  output: "Y"
  attribute: "bias"
  node {
    output: "B"
    op_type: "Constant"
    attribute {
      name: "value"
      type: TENSOR
      ref_attr_name: "bias"
    }
  }
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  opset_import {
    domain: ""
    version: 14
  }
  opset_import {
    domain: "custom"
    version: 1
  }
  domain: "custom"
}
```

## Parsing
Module onnx provides a faster way to define a graph and is lot easier to read. That’s easy to use when the graph is built in a single function, less easy when the graph is built from many different functions converting each piece of a machine learning pipeline.
>  onnx 模块提供了一种更快定义图的方式，如果图仅包含单个函数时，使用起来会十分方便

```python
import onnx.parser
from onnx.checker import check_model

input = '''
    <
        ir_version: 8,
        opset_import: [ "" : 15]
    >
    agraph (float[I,J] X, float[I] A, float[I] B) => (float[I] Y) {
        XA = MatMul(X, A)
        Y = Add(XA, B)
    }
    '''
onnx_model = onnx.parser.parse_model(input)
check_model(onnx_model)

print(onnx_model)
```

```
ir_version: 8
graph {
node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
    domain: ""
}
node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
    domain: ""
}
name: "agraph"
input {
    name: "X"
    type {
    tensor_type {
        elem_type: 1
        shape {
        dim {
            dim_param: "I"
        }
        dim {
            dim_param: "J"
        }
        }
    }
    }
}
input {
    name: "A"
    type {
    tensor_type {
        elem_type: 1
        shape {
        dim {
            dim_param: "I"
        }
        }
    }
    }
}
input {
    name: "B"
    type {
    tensor_type {
        elem_type: 1
        shape {
        dim {
            dim_param: "I"
        }
        }
    }
    }
}
output {
    name: "Y"
    type {
    tensor_type {
        elem_type: 1
        shape {
        dim {
            dim_param: "I"
        }
        }
    }
    }
}
}
opset_import {
domain: ""
version: 15
}
```

This way is used to create small models but it is rarely used in converting libraries.
>  这种方法主要用于创建小模型，在转换库时很少使用

## Checker and Shape Inference
onnx provides a function to check the model is valid. It checks input type or shapes whenever it can detect inconsistency. 
>  onnx 提供了 `checker.checker_model` 函数用于检查模型是否有效，该函数检查模型中的输入类型和形状是否和输出一致

The following example adds two matrices of different types which is not allowed.

```python
import onnx.parser
import onnx.checker

input = '''
    <
        ir_version: 8,
        opset_import: [ "" : 15]
    >
    agraph (float[I,4] X, float[4,2] A, int[4] B) => (float[I] Y) {
        XA = MatMul(X, A)
        Y = Add(XA, B)
    }
    '''
try:
    onnx_model = onnx.parser.parse_model(input)
    onnx.checker.check_model(onnx_model)
except Exception as e:
    print(e)
```

```
b'[ParseError at position (line: 6 column: 44)]\nError context:     agraph (float[I,4] X, float[4,2] A, int[4] B) => (float[I] Y) {\nExpected character ) not found.'
```

`check_model` raises an error due to that inconsistency. This work for all operators defined in the main domain or the ML domain. It remains silent for any custom operator not defined in any specification.
>  检测到不一致性时，函数会抛出错误，这适用于主域和 ML 域中定义的所有算子
>  对于未在任何规范中定义的自定义算子则无法检测

Shape inference serves one purpose: estimate the shape and the type of intermediate results. If known, the runtime can estimate the memory consumption beforehand and optimize the computation. It can fuse some operators, it can do the computation inplace…
>  形状推理 `onnx.shape_inference` 只有一个目的：估计中间结果的形状和类型，如果可以确定，运行时可以估计内存消耗并优化计算，例如融合某些算子、进行原地计算等

```python
import onnx.parser
from onnx import helper, shape_inference

input = '''
    <
        ir_version: 8,
        opset_import: [ "" : 15]
    >
    agraph (float[I,4] X, float[4,2] A, float[4] B) => (float[I] Y) {
        XA = MatMul(X, A)
        Y = Add(XA, B)
    }
    '''
onnx_model = onnx.parser.parse_model(input)
inferred_model = shape_inference.infer_shapes(onnx_model)

print(inferred_model)
```

```
ir_version: 8
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
    domain: ""
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
    domain: ""
  }
  name: "agraph"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "I"
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 4
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "I"
          }
        }
      }
    }
  }
  value_info {
    name: "XA"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "I"
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 15
}
```

There is a new attribute `value_info` which stores the inferred shapes. Letter `I` in `dim_param: "I"` can be seen as a variable. It depends on the inputs but the function is able to tell which intermediate result will share the same dimension. 
>  形状推理后得到的 `inferred_model` 新增了一个属性 `value_info` 用于存储推断出的形状
>  `dim_param: "I"` 中的 `I` 可以视作一个变量，它取决于输入，但函数可以确定哪个中间结果将共享相同的维度

Shape inference does not work all the time. For example, a Reshape operator. Shape inference only works if the shape is constant. If not constant, the shape cannot be easily inferred unless the following nodes expect specific shape.

## Evaluation and Runtime
The ONNX standard allows frameworks to export trained models in ONNX format, and enables inference using any backend that supports the ONNX format. _onnxruntime_ is one efficient option. It is available in many platforms. It is optimized for fast inference. Its coverage can be tracked on [ONNX Backend Dashboard](https://onnx.ai/backend-scoreboard/). _onnx_ implements a python runtime useful to help understand a model. It is not intended to be used for production and performance is not a goal.
>  `onnxruntime` 是运行 ONNX 格式模型的一个高效后端，它可以在多个平台上使用
>  `onnx` Python 库实现了一个 Python 运行时，帮助理解模型

### Evaluation of a linear regression
Full API is described at [onnx.reference](https://onnx.ai/onnx/api/reference.html#l-reference-implementation). It takes a model (a _ModelProto_, a filename, …). Method `run` returns the outputs for a given set of inputs specified in a dictionary.

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

sess = ReferenceEvaluator(onnx_model)

x = numpy.random.randn(4, 2).astype(numpy.float32)
a = numpy.random.randn(2, 1).astype(numpy.float32)
b = numpy.random.randn(1, 1).astype(numpy.float32)
feeds = {'X': x, 'A': a, 'B': b}

print(sess.run(None, feeds))

[array([[-1.6434419 ],
       [ 0.31556654],
       [-0.00626087],
       [ 0.1317825 ]], dtype=float32)]
```

### Evaluation of a node
The evaluator can also evaluate a simple node to check how an operator behaves on a specific input.

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import make_node

from onnx.reference import ReferenceEvaluator

node = make_node('EyeLike', ['X'], ['Y'])

sess = ReferenceEvaluator(node)

x = numpy.random.randn(4, 2).astype(numpy.float32)
feeds = {'X': x}

print(sess.run(None, feeds))

[array([[1., 0.],
       [0., 1.],
       [0., 0.],
       [0., 0.]], dtype=float32)]
```

Similar code would also work on _GraphProto_ or _FunctionProto_.

### Evaluation Step by Step
A converting library takes an existing model trained with a machine learning framework (_pytorch_, _scikit-learn_, …) and converts the model into an ONNX graph. Complex models usually do not work on the first try and seeing intermediate results may help to find the part incorrectly converted. Parameter `verbose` displays information about intermediate results.
>  转换库将机器学习框架训练的现有模型转换为 ONNX 图，复杂的模型通常不会一次性正常工作，查看中介结果可以有助于找到转换错误的部分
>  `ReferenceEvaluator` 的 `verbose` 参数可以用于控制显示关于中间结果的信息

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

for verbose in [1, 2, 3, 4]:
    print()
    print(f"------ verbose={verbose}")
    print()
    sess = ReferenceEvaluator(onnx_model, verbose=verbose)

    x = numpy.random.randn(4, 2).astype(numpy.float32)
    a = numpy.random.randn(2, 1).astype(numpy.float32)
    b = numpy.random.randn(1, 1).astype(numpy.float32)
    feeds = {'X': x, 'A': a, 'B': b}

    print(sess.run(None, feeds))
```

```
------ verbose=1

[array([[ 0.08636874],
       [ 0.8525568 ],
       [-0.45350397],
       [-1.323355  ]], dtype=float32)]

------ verbose=2

MatMul(X, A) -> XA
Add(XA, B) -> Y
[array([[ 0.6552129 ],
       [-0.07477209],
       [ 0.52775025],
       [ 0.5075038 ]], dtype=float32)]

------ verbose=3

 +I X: float32:(4, 2) in [-0.6147719025611877, 0.9176718592643738]
 +I A: float32:(2, 1) in [0.11456892639398575, 0.6806508898735046]
 +I B: float32:(1, 1) in [-0.6908984780311584, -0.6908984780311584]
MatMul(X, A) -> XA
 + XA: float32:(4, 1) in [-0.3411978483200073, 0.6344169974327087]
Add(XA, B) -> Y
 + Y: float32:(4, 1) in [-1.0320963859558105, -0.05648148059844971]
[array([[-0.15087938],
       [-0.6015661 ],
       [-0.05648148],
       [-1.0320964 ]], dtype=float32)]

------ verbose=4

 +I X: float32:(4, 2):1.1490157842636108,-0.48906397819519043,-0.6849649548530579,0.4220026731491089,0.17349936068058014...
 +I A: float32:(2, 1):[-1.2879788875579834, 0.4442490339279175]
 +I B: float32:(1, 1):[-0.569837749004364]
MatMul(X, A) -> XA
 + XA: float32:(4, 1):[-1.697174310684204, 1.0696946382522583, 0.8287805318832397, -0.028250157833099365]
Add(XA, B) -> Y
 + Y: float32:(4, 1):[-2.267012119293213, 0.4998568892478943, 0.25894278287887573, -0.5980879068374634]
[array([[-2.2670121 ],
       [ 0.4998569 ],
       [ 0.25894278],
       [-0.5980879 ]], dtype=float32)]
```

### Evaluate a custom node
The following example still implements a linear regression but adds the identity matrix to _A_: $Y=X(A+I)+B$.

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info)
from onnx.checker import check_model
from onnx.reference import ReferenceEvaluator

# 创建张量
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# 根据张量和计算需求创建节点
node0 = make_node('EyeLike', ['A'], ['Eye'])
node1 = make_node('Add', ['A', 'Eye'], ['A1'])
node2 = make_node('MatMul', ['X', 'A1'], ['XA1'])
node3 = make_node('Add', ['XA1', 'B'], ['Y'])

# 根据张量和节点创建图
graph = make_graph([node0, node1, node2, node3], 'lr', [X, A, B], [Y])

# 根据图创建模型
onnx_model = make_model(graph)
check_model(onnx_model)
with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# 运行图
sess = ReferenceEvaluator(onnx_model, verbose=2)

x = numpy.random.randn(4, 2).astype(numpy.float32)
a = numpy.random.randn(2, 2).astype(numpy.float32) / 10
b = numpy.random.randn(1, 2).astype(numpy.float32)
feeds = {'X': x, 'A': a, 'B': b}

print(sess.run(None, feeds))
```

```
EyeLike(A) -> Eye
Add(A, Eye) -> A1
MatMul(X, A1) -> XA1
Add(XA1, B) -> Y
[array([[-0.28101844,  0.7672609 ],
       [-0.8608558 ,  1.3707852 ],
       [-0.8636193 ,  2.132121  ],
       [-0.87918663, -0.10823274]], dtype=float32)]
```

What if we combine operators _EyeLike_ and _Add_ into _AddEyeLike_ to make it more efficient. Next example replaces these two operators by a single one from domain `'optimized'`.

```python
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid)
from onnx.checker import check_model

# 创建张量
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# 创建节点
node01 = make_node('AddEyeLike', ['A'], ['A1'], domain='optimized')

node2 = make_node('MatMul', ['X', 'A1'], ['XA1'])
node3 = make_node('Add', ['XA1', 'B'], ['Y'])

# 创建图
graph = make_graph([node01, node2, node3], 'lr', [X, A, B], [Y])

# 创建模型
onnx_model = make_model(graph, opset_imports=[
    make_opsetid('', 18), make_opsetid('optimized', 1)
])

check_model(onnx_model)
with open("linear_regression_improved.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

We need to evaluate this model is equivalent to the first one. This requires an implementation for this particular node.

```python
import numpy
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

# 实现自定义节点
class AddEyeLike(OpRun):

    op_domain = "optimized"

    def _run(self, X, alpha=1.):
        assert len(X.shape) == 2
        assert X.shape[0] == X.shape[1]
        X = X.copy()
        ind = numpy.diag_indices(X.shape[0])
        X[ind] += alpha
        return (X,)

# 传入实现的自定义节点
sess = ReferenceEvaluator("linear_regression_improved.onnx", verbose=2, new_ops=[AddEyeLike])

x = numpy.random.randn(4, 2).astype(numpy.float32)
a = numpy.random.randn(2, 2).astype(numpy.float32) / 10
b = numpy.random.randn(1, 2).astype(numpy.float32)
feeds = {'X': x, 'A': a, 'B': b}

print(sess.run(None, feeds))

# Let's check with the previous model.

sess0 = ReferenceEvaluator("linear_regression.onnx",)
sess1 = ReferenceEvaluator("linear_regression_improved.onnx", new_ops=[AddEyeLike])

y0 = sess0.run(None, feeds)[0]
y1 = sess1.run(None, feeds)[0]
print(y0)
print(y1)
print(f"difference: {numpy.abs(y0 - y1).max()}")
```

```
AddEyeLike(A) -> A1
MatMul(X, A1) -> XA1
Add(XA1, B) -> Y
[array([[-2.3503592 ,  2.368486  ],
       [-1.1118807 ,  1.1540793 ],
       [ 0.57669586,  1.410782  ],
       [ 0.63317615,  1.2467988 ]], dtype=float32)]
[[-2.3503592   2.368486  ]
 [-1.1118807   1.1540793 ]
 [ 0.57669586  1.410782  ]
 [ 0.63317615  1.2467988 ]]
[[-2.3503592   2.368486  ]
 [-1.1118807   1.1540793 ]
 [ 0.57669586  1.410782  ]
 [ 0.63317615  1.2467988 ]]
difference: 0.0
```

Predictions are the same. Let’s compare the performance on a matrix big enough to see a significant difference.

```python
import timeit
import numpy
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

class AddEyeLike(OpRun):

    op_domain = "optimized"

    def _run(self, X, alpha=1.):
        assert len(X.shape) == 2
        assert X.shape[0] == X.shape[1]
        X = X.copy()
        ind = numpy.diag_indices(X.shape[0])
        X[ind] += alpha
        return (X,)

sess = ReferenceEvaluator("linear_regression_improved.onnx", verbose=2, new_ops=[AddEyeLike])

x = numpy.random.randn(4, 100).astype(numpy.float32)
a = numpy.random.randn(100, 100).astype(numpy.float32) / 10
b = numpy.random.randn(1, 100).astype(numpy.float32)
feeds = {'X': x, 'A': a, 'B': b}

sess0 = ReferenceEvaluator("linear_regression.onnx")
sess1 = ReferenceEvaluator("linear_regression_improved.onnx", new_ops=[AddEyeLike])

y0 = sess0.run(None, feeds)[0]
y1 = sess1.run(None, feeds)[0]
print(f"difference: {numpy.abs(y0 - y1).max()}")
print(f"time with EyeLike+Add: {timeit.timeit(lambda: sess0.run(None, feeds), number=1000)}")
print(f"time with AddEyeLike: {timeit.timeit(lambda: sess1.run(None, feeds), number=1000)}")
```

```
difference: 0.0
time with EyeLike+Add: 0.09263075000001209
time with AddEyeLike: 0.07530906099998447
```

It seems worth adding an optimized node in this case. This kind of optimization is usually called _fusion_. Two consecutive operators are fused into an optimized version of both. Production usually relies on _onnxruntime_ but since the optimization uses basic matrix operation, it should bring the same performance gain on any other runtime.
>  本例中涉及的优化称为融合，两个连续的算子被融合为一个经过优化的版本

## Implementation details
### Python and C++
onnx relies on protobuf to define its type. You would assume that a python object is just a wrapper around a C pointer on the internal structure. Therefore, it should be possible to access internal data from a function receiving a python object of type `ModelProto`. But it is not. According to [Protobuf 4, changes](https://developers.google.com/protocol-buffers/docs/news/2022-05-06), this is no longer possible after version 4 and it is safer to assume the only way to get a hold on the content is to serialize the model into bytes, give it to the C function, then deserialize it. Functions like `check_model` or `shape_inference` are calling `SerializeToString` then `ParseFromString` before checking the model with a C code.
>  `onnx` 使用 protobuf 来定义其类型
>  在 protobuf 4 之前，可以认为 `onnx` 中的一个 python 对象 (例如 `ModelProto` ) 仅仅是对内部结构的 C 指针的封装，因此可以直接通过 python 对象访问其内部数据
>  在 protobuf 4 之后，则不能再通过 python 对象直接访问和修改 C 内部的数据，任何需要将 python 对象传递给 C 函数的操作都需要先将对象序列化为字节流，然后传递给 C 函数，然后再反序列化
>  因此，ONNX 的许多工具链多了序列化和反序化的操作，例如 `check_model` 或 `shape_inference` 调用底层 C 代码检查模型前，会首先调用 `SerializeToString` 将模型序列化为字节流，然后将字节流传递给 C 函数，在 C 函数执行完毕后，再调用 `ParseFromString` 反序列化字节流

### Attributes and inputs
There is a clear distinction between the two. Inputs are dynamic and may change at every execution. Attributes never changes and an optimizer can improve the execution graph assuming it never changes. Therefore, it is impossible to turn an input into an attribute. And the operator _Constant_ is the only operator changing an attribute into an input.
>  属性和输入之间具有明确的区别，输入是动态的，每次执行都有可能改变，属性则永远不变，优化器在优化时可以假设属性是不变的
>  因此不可能将输入转化为属性，算子 Constant 则是唯一将属性转化为输入的算子

### Shape or no shape
onnx usually expects a shape for every input or output assuming the rank (or the number of dimensions) is known. What if we need to create a valid graph for every dimension? This case is still puzzling.

```python
import numpy
from onnx import numpy_helper, TensorProto, FunctionProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid,
    make_function)
from onnx.checker import check_model
from onnxruntime import InferenceSession

def create_model(shapes):
    new_domain = 'custom'
    opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'A'], ['Y'])

    X = make_tensor_value_info('X', TensorProto.FLOAT, shapes['X'])
    A = make_tensor_value_info('A', TensorProto.FLOAT, shapes['A'])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, shapes['Y'])

    graph = make_graph([node1, node2], 'example', [X, A], [Y])

    onnx_model = make_model(graph, opset_imports=opset_imports)
    # Let models runnable by onnxruntime with a released ir_version
    onnx_model.ir_version = 8

    return onnx_model

print("----------- case 1: 2D x 2D -> 2D")
onnx_model = create_model({'X': [None, None], 'A': [None, None], 'Y': [None, None]})
check_model(onnx_model)
sess = InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])
res = sess.run(None, {
    'X': numpy.random.randn(2, 2).astype(numpy.float32),
    'A': numpy.random.randn(2, 2).astype(numpy.float32)})
print(res)

print("----------- case 2: 2D x 1D -> 1D")
onnx_model = create_model({'X': [None, None], 'A': [None], 'Y': [None]})
check_model(onnx_model)
sess = InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])
res = sess.run(None, {
    'X': numpy.random.randn(2, 2).astype(numpy.float32),
    'A': numpy.random.randn(2).astype(numpy.float32)})
print(res)

print("----------- case 3: 2D x 0D -> 0D")
onnx_model = create_model({'X': [None, None], 'A': [], 'Y': []})
check_model(onnx_model)
try:
    InferenceSession(onnx_model.SerializeToString(),
                     providers=["CPUExecutionProvider"])
except Exception as e:
    print(e)

print("----------- case 4: 2D x None -> None")
onnx_model = create_model({'X': [None, None], 'A': None, 'Y': None})
try:
    check_model(onnx_model)
except Exception as e:
    print(type(e), e)
sess = InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])
res = sess.run(None, {
    'X': numpy.random.randn(2, 2).astype(numpy.float32),
    'A': numpy.random.randn(2).astype(numpy.float32)})
print(res)
print("----------- end")
```

```
----------- case 1: 2D x 2D -> 2D
[array([[ 0.00577903, -0.43896937],
       [ 0.93004453, -0.5679703 ]], dtype=float32)]
----------- case 2: 2D x 1D -> 1D
[array([-0.48937678,  0.758713  ], dtype=float32)]
----------- case 3: 2D x 0D -> 0D
[ONNXRuntimeError] : 1 : FAIL : Node () Op (MatMul) [ShapeInferenceError] Input tensors of wrong rank (0).
----------- case 4: 2D x None -> None
<class 'onnx.onnx_cpp2py_export.checker.ValidationError'> Field 'shape' of 'type' is required but missing.
[array([-0.15104821, -2.5617983 ], dtype=float32)]
----------- end
```

# Converters
Using ONNX in production means the prediction function of a model can be implemented with ONNX operators. A runtime must be chosen, one available on the platform the model is deployed. Discrepancies are checked and finally, the latency is measured.
>  在生产中使用 ONNX 意味着模型的预测函数需要使用 ONNX 算子实现，并且模型部署的平台上需要实现一个 ONNX 运行时
>  然后我们要测量使用 ONNX 部署的模型的预测是否会存在差异，并且测量延迟

The first step of the model conversion can be easy if there exists a converting library for this framework supporting all the pieces of the model. If it is not the case, the missing parts must be implemented in ONNX. That may be very time consuming.
>  如果框架存在支持模型所有部分的转换库，则模型转换的第一步就很简单
>  如果没有，则缺失的部分必须用 ONNX 实现，这可能会非常耗时

## What is a converting library?
[sklearn-onnx](https://onnx.ai/sklearn-onnx/) converts [scikit-learn](https://scikit-learn.org/stable/) models into ONNX. It rewrites the prediction function of a model, whatever it is, with ONNX operators using the API introduced above. It ensures that the predictions are equal or at least very close to the expected predictions computed with the original model.
>  `sklearn-onnx` 将 `scikit-learn` 模型转化为 ONNX 格式，它使用 ONNX 算子重写模型的预测函数，并且确保预测结果和原来模型的结果相等或者至少非常接近

Machine learning libraries usually have their own design. That’s why there exists a specific converting library for each of them. Many of them are listed there: [Converting to ONNX format](https://github.com/onnx/tutorials#converting-to-onnx-format). Here is a short list:

- [sklearn-onnx](https://onnx.ai/sklearn-onnx/): converts models from [scikit-learn](https://scikit-learn.org/stable/),
- [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx): converts models from [tensorflow](https://www.tensorflow.org/),
- [onnxmltools](https://github.com/onnx/onnxmltools): converts models from [lightgbm](https://lightgbm.readthedocs.io/), [xgboost](https://xgboost.readthedocs.io/en/stable/), [pyspark](https://spark.apache.org/docs/latest/api/python/), [libsvm](https://github.com/cjlin1/libsvm)
- [torch.onnx](https://pytorch.org/docs/master/onnx.html): converts model from [pytorch](https://pytorch.org/).

>  每个机器学习库都有特定的转换为 ONNX 的库，包括
>  - `skleran-onnx`
>  - `tensorflow-onnx`
>  - `onnxmltools`
>  - `torch.onnx`

The main challenge for all these libraries is to keep up the rhythm. They must be updated everytime ONNX or the library they support have a new released version. That means three to five new releases per year.
>  这些库需要随着 ONNX 或者支持它们的库发行新版本时进行更新，这意味着每年大约进行 3-5 次新发布

Converting libraries are not compatible among each others. [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) is dedicated to tensorflow and only tensorflow. The same goes for sklearn-onnx specialized into scikit-learn.

One challenge is customization. It is difficult to support custom pieces in a machine learned model. They have to write the specific converter for this piece. Somehow, it is like implementing twice the prediction function. There is one easy case: deep learning frameworks have their own primitives to ensure the same code can be executed on different environments. As long as a custom layer or a subpart is using pieces of pytorch or tensorflow, there is not much to do. It is a different story for scikit-learn. This package does not have its own addition or multiplication, it relies on numpy or scipy. The user must implement its transformer or predictor with ONNX primitives, whether or not it was implemented with numpy.
>  深度学习框架拥有自己的原始操作/原语，这些操作已经针对不同环境进行了优化，并且可以直接转换为 ONNX 格式，如果机器学习模型使用的是这些操作，就不需要额外编写转换器
>  sklearn 没有自己的数学运算实现，而是依赖于外部库 numpy 或 scipy，需要用户额外用 ONNX 原语实现转换器

## Alternatives
One alternative for implementing ONNX export capability is to leverage standard protocols such as the [Array API standard](https://data-apis.org/array-api/latest/), which standardizes a common set of array operations. It enables code reuse across libraries like NumPy, JAX, PyTorch, CuPy and more. [ndonnx](https://github.com/Quantco/ndonnx) enables execution with an ONNX backend and instant ONNX export for Array API compliant code. This diminishes the need for dedicated converter library code since the same code used to implement most of a library can reused in ONNX conversion. It also provides a convenient primitive for converter authors looking for a NumPy-like experience when constructing ONNX graphs.
>  实现 ONNX 导出功能的一种替代方法是利用标准协议，例如 Array API 标准，该标准统一了一组常见的数组操作
>  `ndonxx` 为符合 Array API 的代码实现了即时的 ONNX 导出，这使得各个框架的专用转换库代码的大部分可以进行重用
>  `ndonnx` 还为转换器作者提供了一个方便的原语，以便在构建 ONNX 图时获得类似 Numpy 的体验

## Opsets
ONNX releases packages with version numbers like `major.minor.fix`. Every minor update means the list of operators is different or the signature has changed. It is also associated to an opset, version `1.10` is opset 15, `1.11` will be opset 16. 
>  ONNX 发布包的版本号格式为 `major.minor.fix`
>  每个 minor 更新意味着新的算子列表或者部分算子的签名改变，每个 minor 版本都和一个算子集相关联，例如 `1.10` 版本对应算子集 15，`1.11` 版本对应算子集 16

Every ONNX graph should define the opset it follows. Changing this version without updating the operators could make the graph invalid. If the opset is left unspecified, ONNX will consider that the graph is valid for the latest opset.
>  每个 ONNX 图都需要定义它遵循的算子集版本，在没有更新算子集时直接改变版本会使得图无效
>  如果算子集没有指定，ONNX 会使用使得图有效的最新的算子集版本

New opsets usually introduce new operators. A same inference function could be implemented differently, usually in a more efficient way. However, the runtime the model is running on may not support newest opsets or at least not in the installed version. That’s why every converting library offers the possibility to create an ONNX graph for a specific opset usually called `target_opset`. ONNX language describes simple and complex operators. Changing the opset is similar to upgrading a library. onnx and onnx runtimes must support backward compatibility.
>  新的算子集一般会引入新的算子，因此更新后相同的推理函数可以利用新算子更高效地实现
>  但运行模型的运行时也需要支持图使用的算子集，因此每个转换库都提供了针对特定算子集 (称为 `target_opset` ) 创建 ONNX 图的功能
>  ONNX 语言用于描述简单和复杂的算子，更新算子集类似于更新库，`onnx` 和 `onnx` 运行时必须支持向后兼容性，即新的运行时可以运行旧的算子集

## Other API
Examples in previous sections show that onnx API is very verbose. It is also difficult to get a whole picture of a graph by reading the code unless it is a small one. Almost every converting library has implemented a different API to create a graph, usually more simple, less verbose than the API of onnx package. All API automate the addition of initializers, hide the creation of a name of every intermediate result, deal with different version for different opset.
>  ONNX 的 API 非常冗长，除非要创建的图很小，否则通过阅读代码很难了解整个图
>  几乎所有的转换库都实现了一个不同的 API 来创建图，通常比 `onnx` 包的 API 更简单，这类 API 会自动添加初始值，隐藏对中间结果名称的创建，并处理不同版本的算子集

### A class Graph with a method add_node
`tensorflow-onnx` implements a class graph. It rewrites tensorflow function with ONNX operator when ONNX does not have a similar function (see [Erf](https://github.com/onnx/tensorflow-onnx/blob/master/tf2onnx/onnx_opset/math.py#L414).

sklearn-onnx defines two different API. The first one introduced in that example [Implement a converter](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_jcustom_syntax.html) follows a similar design that tensorflow-onnx follows. The following lines are extracted from the converter of a linear classifier.

```python
# initializer

coef = scope.get_unique_variable_name('coef')
model_coef = np.array(
    classifier_attrs['coefficients'], dtype=np.float64)
model_coef = model_coef.reshape((number_of_classes, -1)).T
container.add_initializer(
    coef, proto_dtype, model_coef.shape, model_coef.ravel().tolist())

intercept = scope.get_unique_variable_name('intercept')
model_intercept = np.array(
    classifier_attrs['intercepts'], dtype=np.float64)
model_intercept = model_intercept.reshape((number_of_classes, -1)).T
container.add_initializer(
    intercept, proto_dtype, model_intercept.shape,
    model_intercept.ravel().tolist())

# add nodes

multiplied = scope.get_unique_variable_name('multiplied')
container.add_node(
    'MatMul', [operator.inputs[0].full_name, coef], multiplied,
    name=scope.get_unique_operator_name('MatMul'))

# [...]

argmax_output_name = scope.get_unique_variable_name('label')
container.add_node('ArgMax', raw_score_name, argmax_output_name,
                   name=scope.get_unique_operator_name('ArgMax'),
                   axis=1)
```

### Operator as function
The second API shown in [Implement a new converter](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_icustom_converter.html) is more compact and defines every ONNX operator as composable functions. The syntax looks like this for [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), less verbose and easier to read.

```python
rs = OnnxReduceSumSquare(
    input_name, axes=[1], keepdims=1, op_version=opv)

gemm_out = OnnxMatMul(
    input_name, (C.T * (-2)).astype(dtype), op_version=opv)

z = OnnxAdd(rs, gemm_out, op_version=opv)
y2 = OnnxAdd(C2, z, op_version=opv)
ll = OnnxArgMin(y2, axis=1, keepdims=0, output_names=out[:1],
                op_version=opv)
y2s = OnnxSqrt(y2, output_names=out[1:], op_version=opv)
```

## Tricks learned from experience
### Discrepancies
ONNX is strongly typed and optimizes for float32, the most common type in deep learning. Libraries in standard machine learning use both float32 and float64. numpy usually cast to the most generic type, float64. It has no significant impact when the prediction function is contiguous. When it is not, the right type must be used. Example [Issues when switching to float](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_ebegin_float_double.html) gives more insights on that topic.
>  ONNX 是强类型的，并且针对深度学习中最常见的类型 float32 进行了优化

Parallelization changes the order of computation. It is usually not significant but it may explain some weird discrepancies. `1 + 1e17 - 1e17 = 0` but `1e17 - 1e17 + 1 = 1`. High order of magnitude are rare but not so rare when a model uses the inverse of a matrix.

### IsolationForest Trick
ONNX only implements a [TreeEnsembleRegressor](https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html#l-onnx-docai-onnx-ml-treeensembleregressor) but it does not offer the possibility to retrieve any information about the path the decision followed or statistics to the graph. The trick is to used one forest to predict the leave index and map this leave index one or multiple times with the information needed.

![../_images/iff.png](https://onnx.ai/onnx/_images/iff.png)

### Discretization
Looking in which interval a feature falls into. That’s easy to do with numpy but not so easy to do efficiently with ONNX. The fastest way is to use a TreeEnsembleRegressor, a binary search, which outputs the interval index. That’s what this example implements: [Converter for WOE](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_woe_transformer.html).

### Contribute
[onnx repository](https://github.com/onnx/onnx) must be forked and cloned.

### Build
The windows build requires conda. The following steps might not be up to date. Folder [onnx/.github/workflows](https://github.com/onnx/onnx/tree/main/.github/workflows) contains the latest instructions.

**Windows**
The build is easier with Anaconda. First: create an environment. It must be done only once.

```
conda create --yes --quiet --name py3.9 python=3.9
conda install -n py3.9 -y -c conda-forge numpy libprotobuf=3.16.0
```

Then build the package:

```
git submodule update --init --recursive
set ONNX_BUILD_TESTS=1
set ONNX_ML=$(onnx_ml)
set CMAKE_ARGS=-DONNX_USE_PROTOBUF_SHARED_LIBS=ON -DONNX_USE_LITE_PROTO=ON -DONNX_WERROR=ON

python -m build --wheel
```

The package can now be installed.

**Linux**
After cloning the repository, the following instructions can be run:

```
python -m build --wheel
```

### Build the markdown documentation
The package must be built first (see previous section).

```
set ONNX_BUILD_TESTS=1
set ONNX_ML=$(onnx_ml)
set CMAKE_ARGS=-DONNX_USE_PROTOBUF_SHARED_LIBS=ON -DONNX_USE_LITE_PROTO=ON -DONNX_WERROR=ON

python onnx\gen_proto.py -l
python onnx\gen_proto.py -l --ml
pip install -e .
python onnx\backend\test\cmd_tools.py generate-data
python onnx\backend\test\stat_coverage.py
python onnx\defs\gen_doc.py
set ONNX_ML=0
python onnx\defs\gen_doc.py
set ONNX_ML=1
```

### Update an existing operator
All operators are defined in folder [onnx/onnx/defs](https://github.com/onnx/onnx/tree/main/onnx/defs). There are two files in every subfolder, one called `defs.cc` and another one called `old.cc`.

- `defs.cc`: contains the most recent definition for every operator
- `old.cc`: contains the deprecated version of the operators in previous opset

>  所有的算子都定义在目录 `onnx/onnx/defs` 其中每个子目录都有两个文件，一个称为 `defs.cc` ，包含了每个算子最新的定义，另一个是 `old.cc` 包含了以前算子集中已经弃用的算子定义

Updating an operator means copying the definition from `defs.cc` to `old.cc` and updating the existing one in `defs.cc`.
>  更新算子意味着将 `defs.cc` 中的算子定义拷贝到 `old.cc` 中，并且更新 `defs.cc` 中的算子定义

One file following the pattern `onnx/defs/operator_sets*.h` must be modified. These headers registers the list of existing operators.
>  更新算子时，模式为 `onnx/defs/operator_sets*.h` 的文件需要被定义，这些头文件为现存的算子列表进行注册

File [onnx/defs/schema.h](https://github.com/onnx/onnx/blob/main/onnx/defs/schema.h) contains the latest opset version. It must be updated too if one opset was upgraded.
>  `onnx/defs/schama.h` 包含了最新的算子集版本，如果算子集更新了，该文件也需要更新

File [onnx/version_converter/convert.h](https://github.com/onnx/onnx/blob/main/onnx/version_converter/convert.h) contains rules to apply when converter a node from an opset to the next one. This file may be updated too.
>  `onnx/version_converter/convert.h` 包含了将节点从一个算子集转化到另一个算子集应用的规则，该文件也需要更新

The package must be compiled and the documentation must be generated again to automatically update the markdown documentation and it must be included in the PR.

Then unit test must be updated.

**Summary**

- Modify files `defs.cc`, `old.cc`, `onnx/defs/operator_sets*.h`, `onnx/defs/schema.h`
- Optional: modify file `onnx/version_converter/convert.h`
- Build onnx.
- Build the documentation.
- Update unit test.

The PR should include the modified files and the modified markdown documentation, usually a subset of `docs/docs/Changelog-ml.md`, `docs/Changelog.md`, `docs/Operators-ml.md`, `docs/Operators.md`, `docs/TestCoverage-ml.md`, `docs/TestCoverage.md`.