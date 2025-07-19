## InstanceNormalization - 22
### Version
- **name**: [InstanceNormalization (GitHub)](https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization)
- **domain**: `main`
- **since_version**: `22`
- **function**: `False`
- **support_level**: `SupportType.COMMON`
- **shape inference**: `True`

This version of the operator has been available **since version 22**.

### Summary
Carries out instance normalization as described in the paper [https://arxiv.org/abs/1607.08022](https://arxiv.org/abs/1607.08022).

y = scale * (x - mean) / sqrt(variance + epsilon) + B, where mean and variance are computed per instance per channel.

### Attributes
- **epsilon - FLOAT** (default is `'1e-05'`):
    
    The epsilon value to use to avoid division by zero.

>  epsilon 值用于避免除零异常

### Inputs
- **input** (heterogeneous) - **T**:
    
    Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size.
    
- **scale** (heterogeneous) - **T**:
    
    The input 1-dimensional scale tensor of size C.
    
- **B** (heterogeneous) - **T**:
    
    The input 1-dimensional bias tensor of size C.
    

### Outputs
- **output** (heterogeneous) - **T**:
    
    The output tensor of the same shape as input.
    

### Type Constraints
- **T** in ( `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)` ):
    
    Constrain input and output types to float tensors.
    

- [InstanceNormalization - 6 vs 22](https://onnx.ai/onnx/operators/text_diff_InstanceNormalization_6_22.html)