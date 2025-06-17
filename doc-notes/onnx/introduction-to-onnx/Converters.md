---
completed: true
version: 1.19.0
---
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