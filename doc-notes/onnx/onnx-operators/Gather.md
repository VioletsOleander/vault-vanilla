## Gather - 13
### Version
- **name**: [Gather (GitHub)](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather)
- **domain**: `main`
- **since_version**: `13`
- **function**: `False`
- **support_level**: `SupportType.COMMON`
- **shape inference**: `True`

This version of the operator has been available **since version 13**.

### Summary
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates them in an output tensor of rank q + (r - 1).
>  `Gather` 的输入是维度 `r >= 1` 的张量 `data` 和维度为 `q` 的张量 `indices`
>  `Gather` 根据 `indices` 收集 ` data ` 的 ` axis ` 维度中特定索引的数据 (默认为最外面的维度 ` axis=0 `)，将这些数据拼接，得到维度为 `q+(r-1)` 的输出张量

It is an indexing operation that indexes into the input `data` along a single (specified) axis. Each entry in `indices` produces a `r-1` dimensional slice of the input tensor. The entire operation produces, conceptually, a `q` -dimensional tensor of `r-1` dimensional slices, which is arranged into a `q + (r-1)` -dimensional tensor, with the `q` dimensions taking the place of the original `axis` that is being indexed into.
>  `Gather` 本质是一个沿着单个 (指定的) `axis` 对输入 `data` 进行索引的操作，`indices` 中的每一项都生成输入张量的 `r-1` 维切片
>  因此整个 `Gather` 操作就会生成一个概念上的 `q` 维张量，张量中的每一个元素都是一个 `r-1` 维切片，故其结果实际上组织为一个 `q + (r-1)` 维张量，其中 `q` 维占据了原来的 `axis` 所在的维度

> 简单地说: `Gather` 将 `indices` 中的每一个 entry 都映射为了 `data` 中对应的一个 slice

>  更简单的说: `Gather` 是取数据，`Scatter` 是发数据

The following few examples illustrate how `Gather` works for specific shapes of `data`, `indices`, and given value of `axis`:

| data shape | indices shape  | axis | output shape | output equation                            |
| ---------- | -------------- | ---- | ------------ | ------------------------------------------ |
| (P, Q)     | ( ) (a scalar) | 0    | (Q)          | output[q] = data[indices, q]               |
| (P, Q, R)  | ( ) (a scalar) | 1    | (P, R)       | output[p, r] = data[p, indices, r]         |
| (P, Q)     | (R, S)         | 0    | (R, S, Q)    | output[r, s, q] = data[ [indices[r, s], q] |
| (P, Q)     | (R, S)         | 1    | (P, R, S)    | output[p, r, s] = data[ p, indices[r, s]]  |

More generally, if `axis = 0`, let `k = indices[i_{0}, ..., i_{q-1}]` then `output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]`:

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]
indices = [
    [0, 1],
    [1, 2],
]
output = [
    [
        [1.0, 1.2],
        [2.3, 3.4],
    ],
    [
        [2.3, 3.4],
        [4.5, 5.7],
    ],
]
```

If `axis = 1`, let `k = indices[i_{0}, ..., i_{q-1}]` then `output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]`:

```
data = [
    [1.0, 1.2, 1.9],
    [2.3, 3.4, 3.9],
    [4.5, 5.7, 5.9],
]
indices = [
    [0, 2],
]
axis = 1,
output = [
        [[1.0, 1.9]],
        [[2.3, 3.9]],
        [[4.5, 5.9]],
]
```

### Attributes
- **axis - INT** (default is `'0'`):
    
    Which axis to gather on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
    

### Input
- **data** (heterogeneous) - **T**:
    
    Tensor of rank r >= 1.
    
- **indices** (heterogeneous) - **Tind**:
    
    Tensor of int32/int64 indices, of any rank q. All index values are expected to be within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.
    

### Outputs
- **output** (heterogeneous) - **T**:
    
    Tensor of rank q + (r - 1).
    

### Type Constraints
- **T** in ( `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)` ):
    
    Constrain input and output types to any tensor type.
    
- **Tind** in ( `tensor(int32)`, `tensor(int64)` ):
    
    Constrain indices to integer types
    

- [Gather - 11 vs 13](https://onnx.ai/onnx/operators/text_diff_Gather_11_13.html)