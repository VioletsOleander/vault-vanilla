# Topic : Understanding the conversion process of onnx-mlir 
Date : 2025.4.19-2025.4.21

## Frontend Process
(1) `main` function
Defined in `src/onnx-mlir.cpp`.

The entry point of `onnx-mlir` is the `main` function in `onnx-mlir.cpp`.

Declare variable `mlir::OwningOpRef<mlir::ModuleOP> module`.

(2) `processInputFile` function
Defined in `src/Compiler/CompilerUtils`.

Classify the types of the input file. For `.onnx` file, it invokes `ImportFrontendModelFile` function, and passes `module` as one of its arguments.

(3) `ImportFrontendModelFile` function
Defined in `src/Compiler/CompilerUtils`.

description: responsible for importing an ONNX model file into the ONNX Dialect. This function takes three arguments:

1. `model_fname`: file name pointing to the onnx model protobuf.
2. `module`: `MLIR::module` generated for the ONNX model.

returns:
-  `0` on success, error number of failure.

Declares variable `onnx::ModelProto model`, and then invoke the `ParseFromIstream` method of `model`, to parse the protobuf file into `onnx::ModelProto`.

Invoke the `ImportFrontendModelInternal` function.

(4) `ImportFrontendModelInternal` function
Defined in `src/Compiler/CompilerUtils`.

Invoke the `ImportFrontendModel` function.

(5) `ImportFrontendModel` function
Defined in `src/Compiler/CompilerUtils.cpp`.

Initialize `detail::FrontendGenImpl myONNXGen`, and then invoke the `ImportONNXModel` method to finally convert the onnx model to mlir module.

(5) `FrontendGenImpl` class
Defined in `src/Builder/FrontendDialectTransformer.cpp`

In namespace `detail`.

-> Attributes

--> Private attributes
-   `onnx_mlir::ImportOptions options_`
-   `mlir::MLIRContext &context_`
-   `mlir::ModuleOp module_`
-   `mlir::OpBuilder builder_`

<--

<-

-> Methods

--> Public methods
---> `ImportONNXModel`

arguments:
-   `onnx::ModelProto &model`
-   `ImportOptions options`

Invoke method `importGraph` with argument `model.graph()`

<--- 

<-- 

--> Private methods
---> `importGraph` (overload 1)

description: Import ONNX main computation graph

arguments:
-   `onnx::GraphProto &graph`: onnx graph proto

returns:
-   `mlir::func::FuncOp`: A function corresponding to the imported computation graph

Create a `mlir::func::FuncOp mainFunc`, and `push_back` `mainFunc` into `module_`.

`module_` is a `ModuleOp`, and now has a single block, with a `FuncOp` in it.

Invoke `getBody()` method of `mainFunc` and `push_back` a new `mlir::Block` into it. `mainFunc` is a `FuncOp`, and now has a region, and a block in it.

Invoke `importGraph` (overload 2), pass in the graph, the region of `mainFunc` as the region, the `mainFunc` as the operation.

That is, the whole graph will be considered as a subgraph of the `mainFunc` operation.

<--- `importGraph` (overload 1)

---> `importGraph` (overload 2)

description: an alternative graph importing procedure for importing ONNX subgraphs. ONNX subgraphs, unlike the main computation graph, are imported as regions nested within the association operations (e.g., the loop body subgraph associated with Loop operation)

arguments:

-   `onnx:GraphProto &graph`: sub-computation graph to import
-   `mlir::Region &region`: region to import computation graph to
-   `mlir::Operation`: operations whose attributes will be updated to contain input/output names
-   `bool useReturn`: if set to true, will emit `ONNXReturnOp` as terminator, otherwise, will use `ONNXYieldOp` as terminator.

Declare `std::unordered_set<std::string> initializerNames`. Iterate over `graph.initializer()` , invoking `BindOnnxName` for each `initializer.name()` and `ImportTensor(initializer)` , updating `initializerNames`.

|> 
`ImporTensor` function
Defined in `src/Builder/FrontendDialectTransformer.cpp`

arguments:
- `onnx::TensorProto &tensor`

returns:
- `mlir::Value`

Invoke `onnxTensorProtoToElmAttr` to maintain a mapping between the parameter and its initializer.

||> 
`onnxTensorProtoToElmAttr` function
Defined in `src/Builder/FrontendDialectHelper.cpp`

Extract tensor type (includes tensor dims and element type), pass tensor type and `onnx::TensorProto` to `createElmAttr`.

|||> 
`createElmAttr` function
Defined in `src/Builder/FrontendDialectHelper.cpp`

For `onnx::TensorProto` that `has_data_location` and `data_location() == onnx::TensorProto::EXTERNAL` (that is, when exporting onnx model, we store the parameters in a separate file), invoke `createElementsAttrFromMemoryBuffer_LE` to construct `mlir::ElementsAttr` with `onnx::TensorProto` 's data.

For `onnx::TensorProto` that `has_raw_data` (that is, the parameters are directly stored in the onnx graph), invoke `createElmAttrFromRawBytes_LE` to construct `mlir::ElementsAttr` with `onnx::TensorProto` 's data.

<|||

<||

<|

Then, create a function for the graph. This process includes:
- Iterate `graph.input(), graph.value_info(), graph.output()` , extracting information.
- Add arguments to the region's entry block, and map graph inputs to the entry block arguments.
- Iterate `graph.node()` , invoke `ImportNode` for each node.

|> `ImportNode` function
Defined in `src/Builder/FrontendDialectTransformer.cpp`

arguments:
- `onnx::NodeProto &node` 

Initialize `std::string opName` with `node.op_type()` and node import version.

Using `ModelLocalFunctionsMap in_model_functions_` to find the model function corresponding to the node. `ModeLocalFunctionsMap` is defined as `using ModelLocalFunctionsMap = std::unordered_map<std::string, const FunctionProto*>;`. That is, `ModelLocalFunctionsMap` is a `std::unordered_map` which maps `std::string` to `onnx::FunctionProto*`.

Therefore, the `model_function` found should have type `onnx::FunctionProto*`.

Invoke `ImportFunctionCallNode` with `onnx::NodeProto` , and `model_function`

||> `ImportFunctionCallNode` function

arguments:
- `onnx::NodeProto &node` 
- `onnx::OpSchema *schame`
- `onnx::FunctionProto *modelLocalFunction`

Iterate `node.input()` to collect the input values and their onnx types into `std::vector<Value> inputs` and `std::vector<onnx::TypeProto> inputOnnxTypes`.

<||

<|

<---

<--

<-

The frontend process convert the `.onnx` file into a `mlir::OwningOpRef<mlir::ModuleOp>` object names `module`. Instead of involving the actual computation logic of each operation, the conversion process is actually just extract each operation's name, arguments, parameters to construct a corresponding MLIR Op.

The resulting `ModuleOp` is actually in ONNX dialect, the ops in ONNX dialect corresponds to the ops in ONNX graph one by one.

## Module Compilation
(1) `main` function
Defined in `src/onnx-mlir.cpp`.

(2) `compileModule` function
Defined in `src/Compiler/CompilerUtils.cpp`

description: process the input module into an output file according to the emission target type. 

arguments:
- `mlir::OwningOpRef<mlir::ModuleOP> &module`
- `mlir::MLIRContext &context`
- `std::string outputNameNoExt` 
- `onnx_milr::EmissionTargetType emissionTarget`

returns: 0 on success, error code on error

Invoke `setupModule` function.

Initialize `mlir::PassManger pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit)`.

Invoke `addPasses` function

|>
`addPasses` function
Defined in `src/Compiler/CompilerPasses.cpp`

arguments:
- `mlir::OwningOpRef<ModuleOp> &module`
- `mlir::PassManager &pm`
- `onnx_mlir::EmissionTargetType emissionTarget`
- `std::string outputNameNoExt`

returns: `void`

Invoke `determineInputIRLevel` function with `module`, to determine the `inputIRLevel`.

||>
`determineInputIRLevel` function
Defined in `src/Compiler/CompilerPasses.cpp`

Initialize `mlir::Operation *moduleOp = module->getOperation()`.

Walk the operations to collect the dialect namespace, and if there are ONNX ops (namespace is `onnx`), will determine the input level to be ONNX, and then return `onnx_mlir::ONNXLevel`.

<||

For our case, it will invoke `addONNXToMLIRPasses` , with target CPU.

||>
`addONNXToMLIRPasses` function
Defined in `src/Compiler/CompilerPasses.cpp`

The passes added includes:
- `DecomposeONNXToONNXPass`
- `RecomposeONNXToONNXPass`
- `ONNXHybridTransformPass`
- `ConvOptONNXToONNXPass`
- `ONNXHybridTransformPass`
- `SimplifyShapeRelatedOpsPass`
- `StandardFuncReturnPass`
- `SymbolDCEPass`
- `ScrubDisposablePass`
- `SetONNXNodeNamePass`

<||

<|

Invoke `run` method

(3) `run` method (class `mlir::PassManager`)
Defined in `llvm-project/mlir/lib/Pass/Pass.cpp` 

arguments:
- `mlir::Operatoin *op`

The run code is surrounded by `context->enterMultiThreadedExecution()` and `context->exitMultiThreadedExecution()`

Initialize `mlir::ModuleAnalysisMnager am` as the top level analysis manager for the pipeline.
 
Invoke `runPasses` function with `op, am`

|>
`runPasses` method
Defined in `llvm-project/mlir/lib/Pass/Pass.cpp`

Invoke `mlir::detail::OpToOpPassAdapater::runPipeline` function.

||>
`runPipeline` function
Defined in `llvm-project/mlir/lib/Pass/Pass.cpp`

arguments:
- `mlir::OpPassManager &pm`
- `mlir::Operation *op`
- `mlir::AnalysisManager am`
- `bool verifyPasses`
- `unsigned parentInitGeneration`
- `mlir::PassInstrumentor *instrumentor`
- `mlir::PassInstrumentation::PipelineParentInfo *parentInfo`

Check if `*instrumentor` it not null. If not, invoke its `runBeforePipeline` method.

Iterate passes from `pm.getPasses()` , and invoke the `run` method for each pass.

Check if `*instrumentor` it not null. If not, invoke its `runAfterPipeline` method.

<||

<|

(4) `run` method (class `mlir::Detail::OpToOpPassAdaptor)
Defined in `llvm-project/mlir/lib/Passes/Pass.cpp`

arguments:
- `mlir::Pass *pass`
- `mlir::Operation *op`
- `mlir::AnalysisManager am`
- `bool verifyPasses`
- `unsigned parentIntGeneration`

Check if the operation is registered and `IsIsolatedFromAbove`.

Invoke the `executeAction` method of the `MLIRContext`, where the action will initialize an `adaptor` . If the `adaptor` is not null, it will invoke the ` runOnOperation ` method of the ` OptoOpPassAdaptor ` class, otherwise, it will directly invoke the `runOnOperation` method of the `pass`. (to end the recursion)

If multi-thread is disabled, the `runOnOperation` method will invoke the `runOnOperationImpl` method of the `OptoOpPassAdapator` class.

The `runOnOperationImpl` method iterates over `region, block, operation`, and invoke `runPipeline` method for each operation.

In effect, all those code will finally apply each pass on each operations one by one.

Therefore, next we should focus on each passes' `runOnOpeartion` method, to see what actually happened in each passes.
(`runOnOperation` is a pure virtual function of `mlir::Pass` , each specific passes that inherited `mlir::Pass` will implement  its own `runOnOperation` method)

### Passes
(1)  `ONNXHybridTransformPass` class
Defined in `src/Dialect/ONNX/Transforms/ONNXHybridTransformPass.cpp`

description: hybrid ONNX transformation pass that combines conversion patterns for shape inference, canonicalization, constant propagation, and decomposition.

-> Methods

--> `initialize` method

Initialize `RewritePatternSet cumulativePattern`.

Add `shapeInference, canonicalization, constantPropagation, decomposition, recomposition` into the `cumulativePattern` set.

Use `cumulativePattern` to initialize `FrozenRewritePatternSet patterns`.

<--

--> `runOnOperation` method

Invoke `getOperation()` to initialize `mlir::func::FuncOp f` . Invoke `f.getBody()` to initialize `milr::Region &body`.

Invoke `applyPatternAndFoldGreedily` with `body` and `patterns`. `applyPatternAndFoldGreedily` will set `config.fold = true`, and in turn invoke `applyPatternGreedily`

The `applyPatternGreedily` function will initialize a `RegionPatternRewriteDriver driver` , and invoke its the `simplify()` method.

|>
`simplify` method
Defined in `llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp`

Enter a loop, when the iteration number does not achieve `config.maxIterations`.

In the loop: first clear the worklist by `worklist.clear()` , and then add `walk` the region to add all operations to the `worklist`.

Invoke `executeAction` method of the context, which in turn invoke the `processworklist` method

||>
`processWorklist` method of `GreedyPatternRewriteDriver`
Defined in `llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp`

Iteratively `pop` operations from the `worklist`.

If `op->hasTrait<OpTrait::ConstantLike>()`, then try to fold this op to an Attribute by invoking `op->fold()`, and then materialize the resulting Attributes as an SSA value.

Invoke the `matchAndRewrite` method of `mlir::PatternApplicator` , which in turn invoke the `mathchAndRewtire` method of the `pattern`

<||

<|

<--

<-

# Topic: Issue solving
Date: 2025.4.24, 4.28
See [issue](https://github.com/onnx/onnx-mlir/issues/2989)

The pattern that caused the problem is `FuseAddConvPattern`, which is defined originally in `src/Dialect/ONNX/ONNXOps/ONNXCanonicalize.td`, and will generate `bulid/src/Dialect/ONNX/ONNXOps/ONNXCanonicalize.inc`.

Error occurs in the 181-th invocation of `FuseAddConvPattern::matchAndRewrite`, line 688. 

Before the erroneous invocation, the previous 180 invocations of `FuseAddConvPattern::matchAndRewrite` all can not reach line 688 in `build/src/Dialect/ONNX/ONNXOps/ONNXCanonicalize.inc` , i.e. , they all fail previous checks and can not finish the rewriting.

|>
Function `FuseAddConvPattern::matchAndRewrite`
Defined in `build/src/Dialect/ONNX/ONNXOps/ONNXCanonicalize.inc`

arguments:
- `mlir::Operation *op0`
- `mlir::PatternRewriter &rewriter`

returns:
- `llvm::LogicalResult`

<|

The information about `mlir::Operation *op0` is:

```
%2836 = "onnx.Add" (%2824, %2835) {onnx_node_name = "node_Add_8121"} : (tensor<1x1024x16x16xf32>, tensor<1x1024x16x16xf32>) -> tensor<1x1024x16x16xf32>
```

|>
class `mlir::Operation`

-> Public methods
--> `dump()` 
--> `dumpPreety()` 

<-

<|

Declare `mlir::ONNXConstantOp y`

Cast `op0` to `mlir::ONNXAddOp castedOp0`

|>
class  `mlir::ONNXAddOp`
Defined in `bulid/src/Dialect/ONNX/ONNXOps.cpp.inc`

<|

The statement in line 688 is:

```cpp
if (!((llvm::all_of(
    ArrayRef<int64_t>(mlir::cast<ShapedType>((*y.getODSResults(0).begin()).getType()).getShape().begin() + 1,                  mlir::cast<ShapedType>((*y.getODSResults(0).begin()).getType()).getShape().end()),[](int64_t val) { return (val == 1);}
)))){ return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
    diag << "entities '' failed to satisfy constraint: 'All dimensions from axis to the end are val'";
  });
}
```

It invokes `llvm::all_of` with two arguments. The first argument is:

```cpp
ArrayRef<int64_t>(mlir::cast<ShapedType>((*y.getODSResults(0).begin()).getType()).getShape().begin() + 1,                  mlir::cast<ShapedType>((*y.getODSResults(0).begin()).getType()).getShape().end())
```

which is a constructor of `ArrayRef<int64_t>`.

There are two arguments of the constructor. The first argument is

```cpp
mlir::cast<ShapedType>((*y.getODSResults(0).begin()).getType()).getShape().begin() + 1,  
```

The second argument is

```cpp
mlir::cast<ShapedType>((*y.getODSResults(0).begin()).getType()).getShape().end())
```

Both arguments invoke the `getODSResults(0)` method on `y`

|>
Method `getODSResults`
Defined in `build/src/Dialect/ONNX/ONNXOps.cpp.inc`

arguments:
- `unsigned index`

returns:
- `mlir::ResultRange`

Invoke `getODSResultIndexAndLength(index)` to initialize `valueRange`.  In this case, `valueRange`  will be `std::pair<unsigned, unsigned>{0, 1}` 

Use  `valueRange` to construct the return value:

```cpp
{std::next(getOperation()->result_begin(), valueRange.first),
 std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
```

with `valueRange.first = 0, valueRange.second = 1`, the expression should be

```cpp
{std::next(getOperation()->result_begin(), 0),
 std::next(getOperation()->result_begin(), 1)};
```

||>
class `mlir::ResultRange`
Defined in `llvm-project/mlir/include/IR/valueRange.h` and `llvm-project/mlir/lib/IR/valueRange.cpp`

`mlir::ResultRange` is a subclass of 

```cpp
llvm::detail::indexed_accessor_range_base<
ResultRange, 
detail::OpRedultImpl *,
OpResult,
OpResult,
OpResult>
```

which defines method `begin()` and `end()` as

```cpp
iterator begin() const { return iterator(base, 0); }
iterator end() const { return iterator(base, count); }
```

with return type `iterator` (an inner class defined in class `llvm::detail::indexed_accessor_range_base`)

The inner class `iterator` overloads `operator*()`, which return `mlir::OpResult`.

Therefore, expression `((*y.getODSResults(0).begin())` will evaluate as a `mlir::OpResult`.

<||

<|

Expression `((*y.getODSResults(0).begin())` will evaluate as a `mlir::OpResult`.

|>
class `mlir::OpResult`
Defined in `llvm-project/mlir/include/mlir/IR/Value.h` and `llvm-project/mlir/lib/IR/Value.cpp`

Subclass `mlir::Value`


||>
class `mlir::Value`
Defined in `llvm-project/mlir/include/mlir/IR/Value.h` and `llvm-project/mlir/lib/IR/Value.cpp`

Method:

```cpp
mlir::Type getType() const { return impl->getType(); }
```

|||>
class `mlir::Type`
Defined in `llvm-project/mlir/include/mlir/IR/Types.h` and `llvm-project/mlir/lib/IR/Types.cpp`

Method:
`dump()`

<|||

<||

<|

Expression `((*y.getODSResults(0).begin()).getType())` will invoke the `getType()` method of `mlir::OpResult`.

If we invoke `((*y.getODSResults(0).begin()).getType()).dump()` , we will get output `tensor<f32>`.

|>
class `mlir::ShapedType`
Defined in `llvm-project/build/tools/mlir/include/mlir/IR/BuiltinTypeInterfaces.h.inc` and `llvm-project/build/tools/mlir/include/mlir/IR/BulitinTypeInterface.cpp.inc`

Method:

```cpp
/// Returns the shape of this type if it is ranked, otherwise asserts.
::llvm::ArrayRef<int64_t> mlir::ShapedType::getShape() const {
      return getImpl()->getShape(getImpl(), *this);
}
```

||>
class `llvm::ArrayRef`
Defined in `llvm-project/llvm/include/ADT/ArrayRef.h`

description: Represent a constant reference to an array (0 or more elements consecutively in memory), i.e. a start pointer and a length. 

Methods:

```cpp
iterator begin() const { return Data; }
iterator end() const { return Data + Length; }
```

where the `iterator` type is essentially an alias of the pointer type `*T` (`T` is the type of data the array stored)

Because in the case `T` is `int64_t`, therefore the `iterator` will be type `*int64_t`

<||

<|

Then, the expression invoke the `getShape()` method of the converted `mlir::ShapedType`, the returned result will be

```
Value returned is $57 = {
  Data = 0x0,
  Length = 0
}
```

which means the returned `ArrayRef` is just an empty `ArrayRef`.

Obviously, the second argument expression in the original `ArrayRef` constructor will evaluate to the same empty `ArrayRef`.

The, the expression invoke `begin()` and `end()` method of the empty `ArrayRef`, because the `ArrayRef` is empty, therefore `begin()` and `end()` will both return `0x0`. (i.e. `nullptr` )

Till now, every thing is fine, but the expression calculate `begin() + 1`, which in turn yield the first argument finally to be `0x8` (the pointed type is `int64_t`, so the pointer will be forwarded 64 bit = 8 byte).

Then,  the `ArrayRef` constructor will get `begin = 0x8` and `end = 0x0`, which makes assertion statement `assert(begin <= end)` fail.

To conclude, it is the error in the original code of onnx-mlir (at least the code generation process of the td definition). The check in line 688 does not consider the situation the `ONNXConstantOP y` has just dimension 1 (i.e. `y` is just a scalar). 
