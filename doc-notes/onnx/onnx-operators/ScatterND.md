## ScatterND - 18
### Version
- **name**: [ScatterND (GitHub)](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND)
- **domain**: `main`
- **since_version**: `18`
- **function**: `False`
- **support_level**: `SupportType.COMMON`
- **shape inference**: `True`

This version of the operator has been available **since version 18**.

### Summary
ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1, and `updates` tensor of rank `q + r - indices.shape[-1] - 1`. 
>  ScatterND 接收三个 tensor 输入，前两个 tensor 输入: `data` 和 `indices` 都要求 `rank >= 1`，最后一个 tensor 输入 `updates` 要求 `rank = q + r - indices.shape[-1] - 1`

The output of the operation is produced by creating a copy of the input ` data `, and then updating its value to values specified by ` updates ` at specific index positions specified by ` indices `. Its output shape is the same as the shape of ` data `.
>  该算子的计算过程是先将 `data` 拷贝，然后根据 `indices` 中指定的索引，将对应的值更新为 `updates` 中指定的值
>  故输出的形状和 `data` 的形状是相同的

`indices` is an integer tensor. Let k denote `indices.shape[-1]`, the last dimension in the shape of ` indices `. ` indices ` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into ` data `. 
>  `indices` 的数据类型为 integer
>  记 `indices` 的最后一个维度形状 `indices.shape[-1]` 为 `k`，记 `indices` 的总维度数量为 `q`，ScatterND 会将 ` indices ` 概念性地视作一个 `q-1` 维的结构，这个结构中的单位元素不是单个整数，而是由 `k` 个整数组成的元组，每个元组都是对 `data` 的一个部分索引，用于定位 `data` 中的某个元素或切片

Hence, k can be a value at most the rank of ` data `. When k equals rank(data), each update entry specifies an update to a single element of the tensor. When k is less than rank(data) each update entry specifies an update to a slice of the tensor. Index values are allowed to be negative, as per the usual convention for counting backwards from the end, but are expected in the valid range.
>  因此 `k` 不能超过 `data` 的维度数量 (`rank(data)`)
>  如果 `k = rank(data)`，`k` 维元组就提供了一个完整的索引，则 ` update ` 中的每一项都针对 ` data ` 中的一个元素
>  如果 `k < rank(data)`，则更新对象就是 `data` 中的切片
>  索引值允许为负数，表示从数组末尾开始计数

`updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the first (q-1) dimensions of `updates.shape` must match the first (q-1) dimensions of `indices.shape`. 
>  `update` 同样会被概念性地视作一个 `q-1` 维的张量，这个张量中的每个元素本身都是一个替换切片值
>  因此，`update.shape` 的前 `q-1` 个维度必须和 `indices.shape` 的前 `q-1` 个维度匹配 (`indices.shape` 的前 `q-1` 个维度指定了有多少个独立的索引操作，`update.shape` 的前 `q-1` 个维度指定了有多少个对应的替换值，故二者需要一一对应)

The remaining dimensions of ` updates ` correspond to the dimensions of the replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor, corresponding to the trailing (r-k) dimensions of ` data `. 
>  `update` 的剩余维度 (去掉前 `q-1` 维) 会被视作替换切片值的维度
>  因为 `update` 的总维度是 `q-1 + r - k`，故每个替换切片值的维度都是 `r-k` (正好对应了 `k` 维的索引对 `r` 维的 `data` 索引后，得到的数据的维度)

Thus, the shape of ` updates ` must equal `indices.shape[0: q-1] ++ data.shape[k: r-1]`, where ++ denotes the concatenation of shapes.

The `output` is calculated via the following equation:

```python
output = np.copy(data)
update_indices = indices.shape[:-1]
for idx in np.ndindex(update_indices):
    output[indices[idx]] = updates[idx]
```

>  输出计算的逻辑如上:
>  先拷贝数据
>  然后遍历 `indices` 指定的所有的 `k` 维索引，将其赋值为 `updates` 中指定的 (`r-k` 维的) 值

The order of iteration in the above loop is not specified. In particular, indices should not have duplicate entries: that is, if `idx1 != idx2`, then ` indices[idx1] != indices[idx2]`. This ensures that the output value does not depend on the iteration order.
>  迭代的顺序未定义
>  故 `indices` 中不应该有重复的条目，确保输出值不依赖于迭代顺序

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates` tensor into `output` at the specified `indices`. 
>  可选参数 `reduction` 代表用户指定的规约函数
>  当 `updates` 中的所有值需要写到 `output` 中的同一个元素/切片中时 (此时 `indices` 中会有重复的条目)，该规约函数会被执行

In cases where `reduction` is set to “none”, indices should not have duplicate entries: that is, if idx1 != idx2, then `indices[idx1] != indices[idx2]`. This ensures that the output value does not depend on the iteration order. 
>  注意，如果 `reduction=None`，`indices` 中不能有重复项

When ` reduction ` is set to some reduction function ` f `, ` output ` is calculated as follows:

```python
output = np.copy(data)
update_indices = indices.shape[:-1]
for idx in np.ndindex(update_indices):
    output[indices[idx]] = f(output[indices[idx]], updates[idx])
```

where the `f` is `+`, `*`, `max` or `min` as specified.

This operator is the inverse of GatherND.

(Opset 18 change): Adds max/min to the set of allowed reduction ops.

```
Example 1:

data    = [1, 2, 3, 4, 5, 6, 7, 8]
indices = [[4], [3], [1], [7]]
updates = [9, 10, 11, 12]
output  = [1, 11, 3, 10, 9, 6, 7, 12]

Example 2:

data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
indices = [[0], [2]]
updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
```

### Attributes
- **reduction - STRING** (default is `'none'`):
    Type of reduction to apply: none (default), add, mul, max, min. ‘none’: no reduction applied. ‘add’: reduction using the addition operation. ‘mul’: reduction using the addition operation. ‘max’: reduction using the maximum operation. ‘min’: reduction using the minimum operation.

### Inputs
- **data** (heterogeneous) - **T**:
    Tensor of rank r >= 1.
    
- **indices** (heterogeneous) - **tensor(int64)**:
    Tensor of rank q >= 1.
    
- **updates** (heterogeneous) - **T**:
    Tensor of rank `q + r - indices_shape[-1] - 1`.

### Outputs
- **output** (heterogeneous) - **T**:
    Tensor of rank r >= 1.

### Type Constraints
- **T** in ( `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)` ):
    
    Constrain input and output types to any tensor type.

- [ScatterND - 16 vs 18](https://onnx.ai/onnx/operators/text_diff_ScatterND_16_18.html)