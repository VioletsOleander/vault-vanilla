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

Responsible for importing an ONNX model file into the ONNX Dialect. This function takes three arguments:

1. `model_fname`: file name pointing to the onnx model protobuf.
2. `module`: `MLIR::module` generated for the ONNX model.

Returns `0` on success, error number of failure.

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

(5.1) Attributes

(5.1.1) Public attributes:

(5.1.2) Private attributes:
-   `onnx_mlir::ImportOptions options_`
-   `mlir::MLIRContext &context_`
-   `mlir::ModuleOp module_`
-   `mlir::OpBuilder builder_`

(5.2) Methods

(5.2.1) Public method
`ImportONNXModel`:

args:

-   `onnx::ModelProto &model`
-   `ImportOptions options`

Invoke method `importGraph` with argument `model.graph()`

(5.2.2) Private method

`importGraph` (overload 1):

description: Import ONNX main computation graph

args:

-   `onnx::GraphProto &graph`: onnx graph proto

returns:

-   `mlir::func::FuncOp`: A function corresponding to the imported computation graph

Create a `mlir::func::FuncOp mainFunc`, and `push_back` `mainFunc` into `module_`.

`module_` is a `ModuleOp`, and now has a single block, with a `FuncOp` in it.

Invoke `getBody()` method of `mainFunc` and `push_back` a new `mlir::Block` into it. `mainFunc` is a `FuncOp`, and now has a region, and a block in it.

Invoke `importGraph` (overload 2), pass in the graph, the region of `mainFunc` as the region, the `mainFunc` as the operation.

That is, the whole graph will be considered as a subgraph of the `mainFunc` operation.

`importGraph` (overload 2):

description: an alternative graph importing procedure for importing ONNX subgraphs. ONNX subgraphs, unlike the main computation graph, are imported as regions nested within the association operations (e.g., the loop body subgraph associated with Loop operation)

args:

-   `onnx:GraphProto &graph`: sub-computation graph to import
-   `mlir::Region &region`: region to import computation graph to
-   `mlir::Operation`: operations whose attributes will be updated to contain input/output names
-   `bool useReturn`: if set to true, will emit `ONNXReturnOp` as terminator, otherwise, will use `ONNXYieldOp` as terminator.

 Declare `std::unordered_set<std::string> initializerNames`. Iterate over `graph.initializer()` , invoking `BindOnnxName` for each `initializer.name()` and `ImportTensor(initializer)` , updating `initializerNames`.

`ImporTensor` function:
Defined in `src/Builder/FrontendDialectTransformer.cpp`

args:
- `onnx::TensorProto &tensor`

returns:
- `mlir::Value`

Invoke `onnxTensorProtoToElmAttr` to maintain a mapping between the parameter and its initializer.

|>
`onnxTensorProtoToElmAttr` function:
Defined in `src/Builder/FrontendDialectHelper.cpp`

Extract tensor type (includes tensor dims and element type), pass tensor type and `onnx::TensorProto` to `createElmAttr`.

<|

|>
`createElmAttr` function: 
Defined in `src/Builder/FrontendDialectHelper.cpp`

For `onnx::TensorProto` that `has_data_location` and `data_location() == onnx::TensorProto::EXTERNAL` (that is, when exporting onnx model, we store the parameters in a separate file), invoke `createElementsAttrFromMemoryBuffer_LE` to construct `mlir::ElementsAttr` with `onnx::TensorProto` 's data.

For `onnx::TensorProto` that `has_raw_data` (that is, the parameters are directly stored in the onnx graph), invoke `createElmAttrFromRawBytes_LE` to construct `mlir::ElementsAttr` with `onnx::TensorProto` 's data.

<|

Then, create a function for the graph. This process includes:
- Iterate `graph.input(), graph.value_info(), graph.output()` , extracting information.
- Add arguments to the region's entry block, and map graph inputs to the entry block arguments.
- Iterate `graph.node()` , invoke `ImportNode` for each node.

|>
`ImportNode` function:
Defined in `src/Builder/FrontendDialectTransformer.cpp`

args:
- `onnx::NodeProto &node` 

Initialize `std::string opName` with `node.op_type()` and node import version.

Using `ModelLocalFunctionsMap in_model_functions_` to find the model function corresponding to the node. `ModeLocalFunctionsMap` is defined as `using ModelLocalFunctionsMap = std::unordered_map<std::string, const FunctionProto*>;`. That is, `ModelLocalFunctionsMap` is a `std::unordered_map` which maps `std::string` to `onnx::FunctionProto*`.

Therefore, the `model_function` found should have type `onnx::FunctionProto*`.

Invoke `ImportFunctionCallNode` with `onnx::NodeProto` , and `model_function`

<|

|>
`ImportFunctionCallNode` function

args:
- `onnx::NodeProto &node` 
- `onnx::OpSchema *schame`
- `onnx::FunctionProto *modelLocalFunction`

Iterate `node.input()` to collect the input values and their onnx types into `std::vector<Value> inputs` and `std::vector<onnx::TypeProto> inputOnnxTypes`.

<|

The frontend process convert the `.onnx` file into a `mlir::OwningOpRef<mlir::ModuleOp>` object names `module`. Instead of involving the actual computation logic of each operation, the conversion process is actually just extract each operation's name, arguments, parameters to construct a corresponding MLIR Op.

The resulting `ModuleOp` is actually in ONNX dialect, the ops in ONNX dialect corresponds to the ops in ONNX graph one by one.

## Module Compilation
(1) `main` function
Defined in `src/onnx-mlir.cpp`.

(2) `compileModule` function
Defined in `src/Compiler/CompilerUtils.cpp`

description: process the input module into an output file according to the emission target type. 

args:
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

args:
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

(3)
`run` method (class `mlir::PassManager`)
Defined in `llvm-project/mlir/lib/Pass/Pass.cpp` 

args:
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

args:
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

args:
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

### Passes
(1)  `ONNXHybridTransformPass` class
Defined in `src/Dialect/ONNX/Transforms/ONNXHybridTransformPass.cpp`

description: hybrid ONNX transformation pass that combines conversion patterns for shape inference, canonicalization, constant propagation, and decomposition.

