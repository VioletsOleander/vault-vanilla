## ScatterElements - 18
### Version
- **name**: [ScatterElements (GitHub)](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements)
- **domain**: `main`
- **since_version**: `18`
- **function**: `False`
- **support_level**: `SupportType.COMMON`
- **shape inference**: `True`

This version of the operator has been available **since version 18**.

### Summary
ScatterElements takes three inputs `data`, `updates`, and `indices` of the same rank r >= 1 and an optional attribute axis that identifies an axis of `data` (by default, the outer-most axis, that is axis 0). The output of the operation is produced by creating a copy of the input `data`, and then updating its value to values specified by `updates` at specific index positions specified by `indices`. Its output shape is the same as the shape of `data`.
>  `ScatterElements` 接收三个输入 `data, updates, indices` 三个输入的维度应该都相同，且大于 1 维
>  `ScatterElements` 接收一个可选的 `axis`，指定 `data` 的一个维度 (默认是最外的维度 axis 0)
>  `ScatterElements` 先拷贝一份 `data`，然后使用 ` updates ` 指定的值，更新 ` data ` 中在 ` indices ` 中指定的坐标处的值，其输出的形状和 ` data ` 相同

For each entry in `updates`, the target index in `data` is obtained by combining the corresponding entry in `indices` with the index of the entry itself: the index-value for dimension = axis is obtained from the value of the corresponding entry in `indices` and the index-value for dimension != axis is obtained from the index of the entry itself.
>  对于 `updates` 中的每一项，其在 `data` 的更新目标的索引是通过结合 `indices` 中的对应项和它自己的索引得到的: 在 `axis` 轴上的索引值通过 `indices` 得到，在 `axis` 外的索引通过它自己的索引得到

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates` tensor into `output` at the specified `indices`. In cases where `reduction` is set to “none”, indices should not have duplicate entries: that is, if idx1 != idx2, then `indices[idx1] != indices[idx2]`. 
>  `reduction` 可以指定一个可选的规约函数
>  规约函数会对 `updates` 中的所有值进行规约，然后将该规约值放入 `output` 中 `indices` 中指定的位置
>  如果 `reduction=None`，`indices` 中不允许有重复的项

For instance, in a 2-D tensor case, the update corresponding to the `[i][j]` entry is performed as below:

```
output[indices[i][j]][j] = updates[i][j] if axis = 0,
output[i][indices[i][j]] = updates[i][j] if axis = 1,
```

When `reduction` is set to some reduction function `f`, the update corresponding to the `[i][j]` entry is performed as below:

```
output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0,
output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1,
```

where the `f` is `+`, `*`, `max` or `min` as specified.

This operator is the inverse of GatherElements. It is similar to Torch’s Scatter operation.
>  该算子是 `GatherElements` 的反向操作，该算子类似于 Torch 的 `Scatter` 操作

(Opset 18 change): Adds max/min to the set of allowed reduction ops.

Example 1:

```
data = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]
indices = [
    [1, 0, 2],
    [0, 2, 1],
]
updates = [
    [1.0, 1.1, 1.2],
    [2.0, 2.1, 2.2],
]
output = [
    [2.0, 1.1, 0.0]
    [1.0, 0.0, 2.2]
    [0.0, 2.1, 1.2]
]
```

Example 2:

```
data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
indices = [[1, 3]]
updates = [[1.1, 2.1]]
axis = 1
output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```

### Attributes
- **axis - INT** (default is `'0'`):
    
    Which axis to scatter on. Negative value means counting dimensions from the back. Accepted range is `[-r, r-1]` where `r = rank(data)`.
    
- **reduction - STRING** (default is `'none'`):
    
    Type of reduction to apply: none (default), add, mul, max, min. ‘none’: no reduction applied. ‘add’: reduction using the addition operation. ‘mul’: reduction using the multiplication operation.‘max’: reduction using the maximum operation.‘min’: reduction using the minimum operation.

### Inputs
- **data** (heterogeneous) - **T**:
    
    Tensor of rank r >= 1.
    
- **indices** (heterogeneous) - **Tind**:
    
    Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index values are expected to be within bounds `[-s, s-1]` along axis of size s. It is an error if any of the index values are out of bounds.
    
- **updates** (heterogeneous) - **T**:
    
    Tensor of rank r >=1 (same rank and shape as indices)
    

### Outputs
- **output** (heterogeneous) - **T**:
    
    Tensor of rank r >= 1 (same rank as input).
    

### Type Constraints
- **T** in ( `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)` ):
    
    Input and output types can be of any tensor type.
    
- **Tind** in ( `tensor(int32)`, `tensor(int64)` ):
    
    Constrain indices to integer types
    

- [ScatterElements - 16 vs 18](https://onnx.ai/onnx/operators/text_diff_ScatterElements_16_18.html)