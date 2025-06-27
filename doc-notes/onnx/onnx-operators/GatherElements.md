## GatherElements - 13
### Version
- **name**: [GatherElements (GitHub)](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements)
- **domain**: `main`
- **since_version**: `13`
- **function**: `False`
- **support_level**: `SupportType.COMMON`
- **shape inference**: `True`

This version of the operator has been available **since version 13**.

### Summary
GatherElements takes two inputs `data` and `indices` of the same rank r >= 1 and an optional attribute `axis` that identifies an axis of `data` (by default, the outer-most axis, that is axis 0). It is an indexing operation that produces its output by indexing into the input data tensor at index positions determined by elements of the `indices` tensor. Its output shape is the same as the shape of `indices` and consists of one value (gathered from the `data`) for each element in `indices`.
>  `GatherElements` 接收两个相同维度的输入 `data, indices` 和一个可选的属性 `axis` (默认为最外的维度 `axis=0`)
>  `GatherElements` 根据 `indices` 来索引 `data` 中的值，构成它的输出
>  其输出形状和 `indices` 相同，也就是 `indices` 中的每一个元素都对应到了 `data` 中的一个值

For instance, in the 3-D case (r = 3), the output produced is determined by the following equations:

```
out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch’s gather operation.

Example 1:

```
data = [
    [1, 2],
    [3, 4],
]
indices = [
    [0, 0],
    [1, 0],
]
axis = 1
output = [
    [1, 1],
    [4, 3],
]
```

Example 2:

```
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
indices = [
    [1, 2, 0],
    [2, 0, 0],
]
axis = 0
output = [
    [4, 8, 3],
    [7, 2, 3],
]
```

### Attributes
- **axis - INT** (default is `'0'`):
    
    Which axis to gather on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
    

### Inputs
- **data** (heterogeneous) - **T**:
    
    Tensor of rank r >= 1.
    
- **indices** (heterogeneous) - **Tind**:
    
    Tensor of int32/int64 indices, with the same rank r as the input. All index values are expected to be within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.
    

### Outputs
- **output** (heterogeneous) - **T**:
    
    Tensor of the same shape as indices.
    

### Type Constraints
- **T** in ( `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)` ):
    
    Constrain input and output types to any tensor type.
    
- **Tind** in ( `tensor(int32)`, `tensor(int64)` ):
    
    Constrain indices to integer types
    

- [GatherElements - 11 vs 13](https://onnx.ai/onnx/operators/text_diff_GatherElements_11_13.html)