# Topic: Understanding the process of `torch.onnx.export`
Date: 2025.6.13

All related code are in `torch/onnx` directory, and grouped into `torch.onnx` package.

## `__init__.py`
(1) the definition of special variable `__all__`, which declared all the public API that  `torch.onnx` would export.

some familiar names in `__all__`:

```python
__all__ = [
    ...
    # Public functions
    "export",
    "is_in_onnx_export",
    "select_model_mode_for_export",
    "register_custom_op_symbolic",
    "unregister_custom_op_symbolic",
    # Base error
    "OnnxExporterError",
    "ExportOptions",
    "ONNXProgram",
    "dynamo_export",
    "enable_fake_mode",
    ...
]
```

(2) many `import` statements. 
Notice that some of them import from the package `_internal` defined in `torch/onnx/_internal/`. For example:

```python
from _internal.exporter._onnx_program import ONNXProgram
```

The above statement imports class `ONNXProgram` defined in `torch/onnx/_internal/exporter/_onnx_program` .

The statement

```python
ONNXProgram.__module__ = "torch.onnx"
```

assign `"torch.onnx"` to the special attribute of object (objects including class, function, instance) `ONNXProgram`. The intention is just promoting the namespace of `ONNXProgram` to the public to avoid the user getting confused when debugging or writing related programs. (If not do that, users may see that `ONNXProgram` is actually defined in the `_internal` packages, so they will get confused.)

(3) `export` function
- arguments: a long list, omit here
- returns: `ONNXProgram | None`

It has a long docstring, which exactly matches the description in the documentation.

If argument `dynamo` is `True`, `export` directly invoke function `_compat.export_compat` .
Module `_compat` is defined in `torch/onnx/_internal/exporter/`

## `_internal/exporter/_compact.py`
docstring: "Compatibility functions for the `torch.onnx.export` API"

(1) `export_compat` function
- arguments: a long list, almost the same as the `export` function defined in `__init__.py`
- returns: `ONNXProgram`

```python
if opset_version is None:
    opset_version = _constant.TORCHLIB_OPSET
```

(2) `registry` initialization

```python
registrpy = _registration.ONNXRegistry().from_torchlib(opset_version=opset_version)
```

|>
`_internal/exporter/_registration.py`
docstring: "Module for handling ATen to ONNX functions registration"

(1) `ONNXRegistry` class
docstring: "Registry for ONNX functions"

-> Methods
--> `from_torchlib`
- docstring: "Populate the registry with ATen functions from torchlib"
- arguments: `opset_version`
- returns: `ONNXRegistry`

<--

<|

After some checks, invoke `_core.export()` with `onnx_program = _core.export(...)` .

|>
 `_internal/exporter/_core.py`

(1) `export` function
- docstring: "Exports a PyTorch model to ONNXProgram"
- returns: `ONNXProgram`

Convert `nn.Module` to `torch.export.ExportedProgram`, trying every capture strategies.

```python
for strategy_class in _capture_strategies.CAPTURE_STRATEGIES:
    strategy = strategy_class(  # type: ignore[abstract]
        verbose=verbose is not False,  # Treat None as verbose
        dump=dump_exported_program,
        artifacts_dir=artifacts_dir,
        timestamp=timestamp,
    )

result = strategy(model, args, kwargs, dynamic_shapes=dynamic_shapes)
```

All capture strategies are encapsulated as a class, and are defined in `_internal/exporter/_capture_strategies.py`

The `result` will be an instance of `torch.exporter.ExportedProgram`, which contains an `torch.fx.Graph` that represents the tensor computation, a `static_dict` containing tensor values of all lifted parameters, buffers, and various metadata.

The essential capture process in these capture strategies are implemented by calling `torch.export.export()`.

```python
if result.exception is not None:
    failed_results.append(result)
if result.success:
    assert result.exported_program is not None
    program = result.exported_program
    break
```

If one strategy success, then assign `program = result.exported_program`, and `break`.

<|



