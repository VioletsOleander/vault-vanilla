## GridSample - 22
### Version
- **name**: [GridSample (GitHub)](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GridSample)
- **domain**: `main`
- **since_version**: `22`
- **function**: `False`
- **support_level**: `SupportType.COMMON`
- **shape inference**: `True`

This version of the operator has been available **since version 22**.

### Summary
Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`. 
>  `GridSample` 接收输入 `X` 和流场 `grid`，使用 `X` 的值和 `grid` 中的像素位置计算输出 `Y`

For spatial input `X` with shape (N, C, H, W), the `grid` will have shape (N, H_out, W_out, 2), the output `Y` will have shape (N, C, H_out, W_out). 
>  对于形状为 `(N, C, H, W)` 的空间输入 `X`，形状为 `(N, H_out, W_out, 2) ` 的 `grid`，输出 `Y` 的形状应该为 `(N, C, H_out, W_out)`

For volumetric input `X` with shape (N, C, D, H, W), the `grid` will have shape (N, D_out, H_out, W_out, 3), the output `Y` will have shape (N, C, D_out, H_out, W_out). 
>  对于形状为 `(N, C, D, H, W)` 的体积分量输入 `X`，形状为 `(N, D_out, H_out, W_out, 3)` 的 `grid`，输出 `Y` 的形状应该为 `(N, C, D_out, H_out, W_out)`

More generally, for an input `X` of rank r+2 with shape (N, C, d1, d2, …, dr), the `grid` will have shape (N, D1_out, D2_out, …, Dr_out, r), the output `Y` will have shape (N, C, D1_out, D2_out, …, Dr_out).
>  更一般地说，对于维度为 `r+2` 的输入 `X` (形状为 `(N, C, d1, d2, ..., dr)`)，以及维度为 `r+2` 的输入 `grid` (形状为 `(N, D1_out, D2_out, ... , Dr_out, r)`)，输出 `Y` 的形状应该为 `(N, C, D1_out, D2_out, ... , Dr_out)`

The tensor `X` contains values at centers of square pixels (voxels, etc) locations such as (n, c, d1_in, d2_in, …, dr_in). The (n, d1_out, d2_out, …, dr_out, :) values from the tensor `grid` are the normalized positions for interpolating the values at the (n, c, d1_out, d2_out, …, dr_out) locations from the output tensor `Y` using a specified interpolation method (the mode) and a padding mode (for `grid` positions falling outside the 2-dimensional image).
>  输入张量 `X` 的值是方形像素 (或体积像素等) 中心位置的值，例如 `(n, c, d1_in, d2_in, ... , dr_in)` (也就是每个数值都对应一个离散的、规则网格上的中心点)
>  输入张量 `grid` 的值是用于插值的归一化位置，`grid` 中的每一个维度为 `r` 的 slice `(n, c, d1_out, d2_out, ..., dr_out)` 都对应输出张量中相应位置的一个值，这个值就根据 `grid` 指定的归一化坐标，利用 `X` 中相应的值计算得到
>  interpolation method 指定了插值方法，padding mode 指定了 `grid` 中的某个坐标指向了 `X` 的边界之外的情况 (`padding_mode=zeros` 表示用 0 填充边界外区域，为默认值，`padding_mode=border` 表示用边界像素值填充，`padding_mode=reflection` 表示镜像反射填充)

For example, the values in `grid[n, h_out, w_out, :]` are size-2 vectors specifying normalized positions in the 2-dimensional space of `X`. They are used to interpolate output values of `Y[n, c, h_out, w_out]`.
>  例如，`grid[n, h_out, w_out, :]` 的值是一个 2 维的 tuple，指定了 `X` 中的一个规范化位置
>  这个位置的值会通过插值计算，然后赋值到 `Y[n, c, h_out, w_out]`

The GridSample operator is often used in doing grid generator and sampler in the [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025). See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).

### Attributes
- **align_corners - INT** (default is `'0'`):
    
    If align_corners=1, the extrema (-1 and 1) are considered as referring to the center points of the input’s corner pixels (voxels, etc.). If align_corners=0, they are instead considered as referring to the corner points of the input’s corner pixels (voxels, etc.), making the sampling more resolution agnostic.
    
- **mode - STRING** (default is `'linear'`):
    
    Three interpolation modes: linear (default), nearest and cubic. The “linear” mode includes linear and N-linear interpolation modes depending on the number of spatial dimensions of the input tensor (i.e. linear for 1 spatial dimension, bilinear for 2 spatial dimensions, etc.). The “cubic” mode also includes N-cubic interpolation modes following the same rules. The “nearest” mode rounds to the nearest even index when the sampling point falls halfway between two indices.
    
- **padding_mode - STRING** (default is `'zeros'`):
    
    Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`. zeros: use 0 for out-of-bound grid locations, border: use border values for out-of-bound grid locations, reflection: use values at locations reflected by the border for out-of-bound grid locations. If index 0 represents the margin pixel, the reflected value at index -1 will be the same as the value at index 1. For location far away from the border, it will keep being reflected until becoming in bound. If pixel location x = -3.5 reflects by border -1 and becomes x’ = 1.5, then reflects by border 1 and becomes x’’ = 0.5.
    

### Inputs
- **X** (heterogeneous) - **T1**:
    
    Input tensor of rank r+2 that has shape (N, C, D1, D2, …, Dr), where N is the batch size, C is the number of channels, D1, D2, …, Dr are the spatial dimensions.
    
- **grid** (heterogeneous) - **T2**:
    
    Input offset of shape (N, D1_out, D2_out, …, Dr_out, r), where D1_out, D2_out, …, Dr_out are the spatial dimensions of the grid and output, and r is the number of spatial dimensions. Grid specifies the sampling locations normalized by the input spatial dimensions. Therefore, it should have most values in the range of [-1, 1]. If the grid has values outside the range of [-1, 1], the corresponding outputs will be handled as defined by padding_mode. Following computer vision convention, the coordinates in the length-r location vector are listed from the innermost tensor dimension to the outermost, the opposite of regular tensor indexing.
    

### Outputs
- **Y** (heterogeneous) - **T1**:
    
    Output tensor of rank r+2 that has shape (N, C, D1_out, D2_out, …, Dr_out) of the sampled values. For integer input types, intermediate values are computed as floating point and cast to integer at the end.
    

### Type Constraints
- **T1** in ( `tensor(bfloat16)`, `tensor(bool)`, `tensor(complex128)`, `tensor(complex64)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int16)`, `tensor(int32)`, `tensor(int64)`, `tensor(int8)`, `tensor(string)`, `tensor(uint16)`, `tensor(uint32)`, `tensor(uint64)`, `tensor(uint8)` ):
    
    Constrain input `X` and output `Y` types to all tensor types.
    
- **T2** in ( `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)` ):
    
    Constrain grid types to float tensors.
    

- [GridSample - 20 vs 22](https://onnx.ai/onnx/operators/text_diff_GridSample_20_22.html)