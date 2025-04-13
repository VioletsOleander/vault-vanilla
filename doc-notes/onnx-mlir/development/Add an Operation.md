# Import ONNX Definitions and Support Operations

# Overview
ONNX-MLIR defines an ONNX dialect to represent operations specified by ONNX. The ONNX dialect is created with MLIR table gen tool. The definition of each operation is transferred from ONNX automatically with a python script, [utils/gen_onnx_mlir.py](https://onnx.ai/utils/gen_onnx_mlir.py). This script retrieves operation definition from ONNX package to generate ONNXOps.td.inc for dialect table gen and OpBuilderTable.inc for ONNX model importer in ONNX-MLIR. 
>  ONNX-MLIR 定义了 ONNX dialect 表示 ONNX 中的 operation
>  ONNX dialect 基于 MLIR table gen 工具构建，其中每个 operation 的定义都是使用脚本 `utils/gen_onnx_mlir.py` 自动从 ONNX 转化而来
>  `utils/gen_onnx_mlir.py` 从 ONNX 包中查找 operation 定义，生成 `ONNXOps.td.inc` 以及 `OpBuilderTable.inc`

The following sections will describe how to use gen_onnx_mlir.py to add an operation into ONNX dialect in ONNX-MLIR and how to refine the definition of the operation.
>  本文档介绍如何使用 `gen_onnx_mlir.py` 为 ONNX dialect 添加 operation，以及如何微调 operation 定义

# Add an Operation
To generate an operation for ONNX dialect, add this operation into the dictionary, `version_dict`, in `gen_onnx_mlir.py`. The key of this directory is the operation name and the value is the list of opset for this operation. Usually only the top version opset of this operation (in `onnx-mlir/third_party/onnx`) is supported. Details about versioning can be found in [version section](https://onnx.ai/onnx-mlir/ImportONNXDefs.html#version). With this entry, the script will generate the operation definition for ONNX dialect.
>  operation 需要先添加到 `gen_onnx_mlir.py` 中的 `version_dict` 中，`version_dict` 的 key 为 operation 名称，value 为该 operation 的 opset 列表
>  通常仅支持该 operation 的最高版本 opset
>  添加后，`gen_onnx_mlir.py` 就会生成该 operation 在 ONNX dialect 中的定义

# Customization
## Add Interface and Trait

- By default, all operation has shape inference interface and `Pure` trait.
- If an operation has `ResultTypeInferenceOpInterface`, add it to dictionary `OpsWithResultTypeInference`. This interface infers the type of result tensor, not shape.
- If an operation has subgraph, it will has interface `HasOnnxSubgraphOpInterface`. This attribute is inferred from the ONNX operation definition.
- You can define helper function for an operation with dictionary `OpsWithHelpers`.

>  - 所有 operation 默认都有 shape inference 接口和 `Pure` trait
>  - 如果 operation 有 `ResultTypeInferenceOpInterface`，需要将其加入 `OpsWithResultTypeInference` list 中
>  - 如果 operation 有 subgraph，它将有 `HasOnnxSubgraphOpInterface` 接口
>  - 可以在字典 `OpsWithHelpers` 中为 operation 定义 helper 函数

## Add canonicalization interface
If a transformation should be applied locally to an operation across passes, canonicalization interface can be used for this transformation. 

To enable the canonicalization for an operation, add the name of this operation into this list of `OpsWithCanonicalizer` and then the operation will have `hasCanonicalizer = 1;` in its definition.
>  要为 operation 启用规范化功能，需要将 operation 名称添加到 `OpsWithCnonicalizer` list 中，这样 operation 的定义中就会有 `hasCanonicalizer = 1`

## Customize builder
The default builders for an operation require the type of results as a parameter. However, the type of results can be inferred. A customize builder may be a useful to simplify the code. Based on the type of inference, there are two kinds builder, unranked type and broadcast type. To enable the special builder for an operation, you can add its name into `custom_builder_unranked_ops_list` and `custom_builder_broadcast_ops_list` respectively.
>  operation 的默认 builder 需要结果类型作为参数
>  可以利用类型推断构建自定义 builder，基于推断的种类，有两类 builder, unranked 和 broadcast
>  要为 operation 启用自定义 builder，需要将 operation 名称添加到 `custom_builder_unranked_ops_list` 或 `custom_builder_broadcast_ops_list` 中

Please note that the need of special builder in rewriting rules can be avoided with the use of `returnType`. Refer to [MLIR doc](https://mlir.llvm.org/docs/DeclarativeRewrites/) or the [example in ONNX-MLIR](https://onnx.ai/src/Transform/ONNX/Decompose.td). It may be a better solution to just move such type inference code into ONNXOpHelper.cpp and get rid of customize builder.

## Customize verifier
The operation description for an operation lists out the allowed types of each input/output and attribute. The table gen will generate a default verifier to check IR for the allowed types. If an operation has extra constraints, a customized verifier should be defined to enhance error detection. For example, two inputs of an operation may require the same element type or same rank. Such information can be found in the ONNX operation definition, but can not be expressed with the dialect definition. The best way to test for these constraints are in a verifier. To add the interface of customized verifier to an operation, locate the array below in `gen_onnx_mlir.py` and add your operation in it.

```
OpsWithVerifier = ['AveragePool', 'Conv', 'InstanceNormalization', 'Mod']
```

Then you will find the following line in operation definition in ONNXOps.td.inc:

```
let verifier = [{ return ::verify(*this); }];
```

You will need to add the implementation code in the `src/Dialect/ONNX/ONNXOps.cpp` when the new op was declared as using a customized verifier. Best is to look at other operations to get the general pattern, by searching for [static LogicalResult verify(ONNXInstanceNormalizationOp op)](https://onnx.ai/src/Dialect/ONNX/ONNXOps.cpp), for example. Note that a verifier will execute each time that one such op is created. So you will need to ensure that it can work with tensors and MemRefs, and possibly unranked tensors. So guard each of your tests to the proper circumstances. For examples, once a tensor is ranked, you may then verify that the rank is within the approved range (if there is such a constraint); before it is ranked, do not perform this test yet.

Tips:

- Use `operandAdaptor` object to get the inputs (must use `operandAdaptor` to get the current values of the inputs) and the `op` object to get the attributes (can use `op` because attributes are typically immutable).
- Use `hasShapeAndRank(X)` to test if `X` input is currently shaped and ranked. If not, return success as we will get a chance later to test the operation with this info. Note that some inputs may be scalar too, in which case they may or may not be encoded as a shape type.
- You can then use MLIR call `mlir::cast<ShapedType>(X.getType())` to get a shape types, for which you can get the rank and the dimensions. At this time, we only check dimension validity for values known at runtime. Unknown dimensions are encoded as a negative number. Please only use the cast when you are sure that it will not assert, i.e. the type is indeed a `ShapedType`.
- When you find an error, report it with a friendly error message using `op->emitError(msg)`.

## Customize importer
`special_op_handler`: creates special import function in frontend_dialect_transformer.cpp. Currently, a special handler is used for operations with operational arguments

## Arbitrary extra definition
If the definition of an operation needs extra code other than described above, you can put the code in the dictionary `custom_definition_misc`. The key is the operation name and the value is the code.

# Build
In order to run gen_onnx_mlir.py, ONNX has to be installed. Refer to Readme. In your build directory, execute the following command.

```
 make OMONNXOpsIncTranslation
```

This command will generate those two files (src/Dialect/ONNX/ONNXOps.td.inc and OpBuilderTable.inc), and copy them to the right place in src directory. If you modified gen_onnx_mlir.py, you need to check in two generated files too. They are treated source file in ONNX-MLIR build so that user of ONNX-MLIR does not need to install the particular version of ONNX. Do not modify these files directly. 

>  运行 `gen_onnx_mlir.py` 之前，需要安装 ONNX
>  `make OMONNXOpsIncTranslation` 将生成 `src/Dialect/ONNX/ONNXOps.td.inc` 和 `OpBuilderTable.inc`
>  这两个文件是 ONNX-MLIR 构建的源文件，以便 ONNX-MLIR 的用户不需要安装特定版本的 ONNX

You can also run the script directly with the files generated in utils directory. `python ../utils/gen_onnx_mlir.py`.
>  也可以直接运行 `utils/gen_onnx_mlir.py` 脚本，则两个文件会生成在 `utils` 目录下

## Update the documentation
When adding a new op version or making changes to the ONNX version, we would like to also reflect these changes in the ONNX documentation of our supported operations. While the latest [ONNX specs](https://github.com/onnx/onnx/blob/main/docs/Operators.md) are always available, the specs that we support are often a bit back, plus we support older versions under the versioned name as mentioned in the previous section.

There is a convenient command to update both the ONNX and Krnl dialect, as shown below.

```
make onnx-mlir-docs
```

The above command is run in the usual `build` directory and it will install the new dialect md files directly into the `docs/Dialects` directory.

The same command should be used when adding operations/making changes to the Krnl dialect.

# Operation Version
ONNX-MLIR project started when ONNX was at version 1.7.0 and does not intended to be backward compatible. We relies on onnx/converter to convert the model to the version which ONNX-MLIR supports. As ONNX version is evolving, ONNX-MLIR tries to follow but may be behind the latest version.
>  我们依赖 onnx/converter 将模型转换为 ONNX-MLIR 支持的版本

## Version of Operations
As stated previous, we try to support the latest version of ONNX operations. The version of each operation currently supported is recorded in [utils/gen_onnx_mlir.py](https://onnx.ai/utils/gen_onnx_mlir.py). This mechanism provides some stability in version. To check the changes in version, run gen_onnx_mlir.py with flag “–check-version” and the changes will be reported. To move to a newer version, manually update the version dictionary in the script.
>  每个 operation 目前支持的版本都记录在 `utils/gen_onnx_mlir.py` 中
>  运行 `python utils/gen_onnx_mlir.py -check-version` 可以检查版本变化

## Support Multiple versions
To support multiple versions of an op, the selected version should be added in the version dictionary in [utils/gen_onnx_mlir.py](https://onnx.ai/utils/gen_onnx_mlir.py). For example, there are two versions (opset), 11 and 13, forReduceSum that are supported. The corresponding entry in version_dic is `'ReduceSum': [13, 11]`.

In ONNX dialect, the op for the top version has no version in the op name, while other version with name followed by ‘V’ and version number. For example, ReduceSum of opset 13 will be `ONNXReduceSumOp`, while ReduceSum of opset 11 is ‘ONNXReduceSumV11Op`. Since most of ONNX op are compatible when upgraded to higher version, we can keep the name of the operation in the dialect and just update version_dict in gen_onnx_mlir.py without touching the code in ONNX-MLIR.

When a model is imported, the highest version which is not higher than the next available version is used. For the example of ReduceSum, if the opset is 12, ONNXReduceSumV11Op is chosen.

## Migrating
To migrate a new version ONNX, first the third_part/onnx should be upgraded and your installation of ONNX. Then you can run gen_onnx_mlir.py with flag `--check_operation_version`. The top version for all operation will be outputted as a new `version_dict`. If the interface of an operation remains the same (from the change document of ONNX), you can just use the new version. If the interface does change, you can insert the new version as the first in the version list. For the existing code, all the corresponding code has to be changed. For example, when ReduceSum is moved from version 11 to 13, ONNXReduceSumOp is replaced with ONNXReduceSumOpV11 first. Then the code for version 13 will use ONNXReduceSumOp. The reason for such design is that most of ONNX changes do not change the interface. We do not want to put burden on developer to remember which version of operation is used unless absolutely necessary. It is not always needed to keep the code for an older version, which may be rewritten into the new operation. Thus, we just need to have the dialect definition, but not the code for inference or lowering.