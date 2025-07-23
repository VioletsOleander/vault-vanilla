---
completed: true
version: "3.10"
---
# oneDNN API Basic Workflow Tutorial
This C++ API example demonstrates the basics of the oneDNN programming model.

Example code: [getting_started.cpp](https://uxlfoundation.github.io/oneDNN/example_getting_started.cpp.html#doxid-getting-started-8cpp-example)

```cpp
/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/


#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

#include "example_utils.hpp"

using namespace dnnl;
// [Prologue]


// [Prologue]

void getting_started_tutorial(engine::kind engine_kind) {
    // [Initialize engine]
    engine eng(engine_kind, 0);
    // [Initialize engine]

    // [Initialize stream]
    stream engine_stream(eng);
    // [Initialize stream]


    // [Create user's data]
    const int N = 1, H = 13, W = 13, C = 3;

    // Compute physical strides for each dimension
    const int stride_N = H * W * C;
    const int stride_H = W * C;
    const int stride_W = C;
    const int stride_C = 1;

    // An auxiliary function that maps logical index to the physical offset
    auto offset = [=](int n, int h, int w, int c) {
        return n * stride_N + h * stride_H + w * stride_W + c * stride_C;
    };

    // The image size
    const int image_size = N * H * W * C;

    // Allocate a buffer for the image
    std::vector<float> image(image_size);

    // Initialize the image with some values
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c) {
                    int off = offset(
                            n, h, w, c); // Get the physical offset of a pixel
                    image[off] = -std::cos(off / 10.f);
                }
    // [Create user's data]

    // [Init src_md]
    auto src_md = memory::desc(
            {N, C, H, W}, // logical dims, the order is defined by a primitive
            memory::data_type::f32, // tensor's data type
            memory::format_tag::nhwc // memory format, NHWC in this case
    );
    // [Init src_md]


    // [Init alt_src_md]
    auto alt_src_md = memory::desc(
            {N, C, H, W}, // logical dims, the order is defined by a primitive
            memory::data_type::f32, // tensor's data type
            {stride_N, stride_C, stride_H, stride_W} // the strides
    );

    // Sanity check: the memory descriptors should be the same
    if (src_md != alt_src_md)
        throw std::logic_error("Memory descriptor initialization mismatch.");
    // [Init alt_src_md]


    // [Create memory objects]
    // src_mem contains a copy of image after write_to_dnnl_memory function
    auto src_mem = memory(src_md, eng);
    write_to_dnnl_memory(image.data(), src_mem);

    // For dst_mem the library allocates buffer
    auto dst_mem = memory(src_md, eng);
    // [Create memory objects]

    // [Create a ReLU primitive]
    // ReLU primitive descriptor, which corresponds to a particular
    // implementation in the library
    auto relu_pd = eltwise_forward::primitive_desc(
            eng, // an engine the primitive will be created for
            prop_kind::forward_inference, algorithm::eltwise_relu,
            src_md, // source memory descriptor for an operation to work on
            src_md, // destination memory descriptor for an operation to work on
            0.f, // alpha parameter means negative slope in case of ReLU
            0.f // beta parameter is ignored in case of ReLU
    );

    // ReLU primitive
    auto relu = eltwise_forward(relu_pd); // !!! this can take quite some time
    // [Create a ReLU primitive]


    // [Execute ReLU primitive]
    // Execute ReLU (out-of-place)
    relu.execute(engine_stream, // The execution stream
            {
                    // A map with all inputs and outputs
                    {DNNL_ARG_SRC, src_mem}, // Source tag and memory obj
                    {DNNL_ARG_DST, dst_mem}, // Destination tag and memory obj
            });

    // Wait the stream to complete the execution
    engine_stream.wait();
    // [Execute ReLU primitive]

    // [Execute ReLU primitive in-place]
    // Execute ReLU (in-place)
    // relu.execute(engine_stream,  {
    //          {DNNL_ARG_SRC, src_mem},
    //          {DNNL_ARG_DST, src_mem},
    //         });
    // [Execute ReLU primitive in-place]

    // [Check the results]
    // Obtain a buffer for the `dst_mem` and cast it to `float *`.
    // This is safe since we created `dst_mem` as f32 tensor with known
    // memory format.
    std::vector<float> relu_image(image_size);
    read_from_dnnl_memory(relu_image.data(), dst_mem);
    /*
    // Check the results
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c) {
                    int off = offset(
                            n, h, w, c); // get the physical offset of a pixel
                    float expected = image[off] < 0
                            ? 0.f
                            : image[off]; // expected value
                    if (relu_image[off] != expected) {
                        std::cout << "At index(" << n << ", " << c << ", " << h
                                  << ", " << w << ") expect " << expected
                                  << " but got " << relu_image[off]
                                  << std::endl;
                        throw std::logic_error("Accuracy check failed.");
                    }
                }
    // [Check the results]
    */
}

// [Main]
int main(int argc, char **argv) {
    int exit_code = 0;

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    try {
        getting_started_tutorial(engine_kind);
    } catch (dnnl::error &e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch (std::string &e) {
        std::cout << "Error in the example: " << e << "." << std::endl;
        exit_code = 2;
    } catch (std::exception &e) {
        std::cout << "Error in the example: " << e.what() << "." << std::endl;
        exit_code = 3;
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
              << engine_kind2str_upper(engine_kind) << "." << std::endl;
    finalize();
    return exit_code;
}
// [Main]
```

- How to create oneDNN memory objects.
    - How to get data from the user’s buffer into a oneDNN memory object.
    - How a tensor’s logical dimensions and memory object formats relate.
- How to create oneDNN primitives.
- How to execute the primitives.

The example uses the ReLU operation and comprises the following steps:

1. Creating [Engine and stream](https://uxlfoundation.github.io/oneDNN/page_getting_started_cpp.html#doxid-getting-started-cpp-1getting-started-cpp-sub1) to execute a primitive.
2. Performing [Data preparation (code outside of oneDNN)](https://uxlfoundation.github.io/oneDNN/page_getting_started_cpp.html#doxid-getting-started-cpp-1getting-started-cpp-sub2).
3. [Wrapping data into a oneDNN memory object](https://uxlfoundation.github.io/oneDNN/page_getting_started_cpp.html#doxid-getting-started-cpp-1getting-started-cpp-sub3) (using different flavors).
4. [Creating a ReLU primitive](https://uxlfoundation.github.io/oneDNN/page_getting_started_cpp.html#doxid-getting-started-cpp-1getting-started-cpp-sub4).
5. [Executing the ReLU primitive](https://uxlfoundation.github.io/oneDNN/page_getting_started_cpp.html#doxid-getting-started-cpp-1getting-started-cpp-sub5).
6. [Obtaining the result and validation](https://uxlfoundation.github.io/oneDNN/page_getting_started_cpp.html#doxid-getting-started-cpp-1getting-started-cpp-sub6) (checking that the resulting image does not contain negative values).

>  示例中的代码总体包含了以下步骤:
>  - 创建引擎和流以执行原语
>  - 执行数据准备工作
>  - 将数据封装在 oneDNN 内存对象中
>  - 创建 ReLU 原语
>  - 执行 ReLU 原语
>  - 获取结果并验证

These steps are implemented in the [getting_started_tutorial() function](https://uxlfoundation.github.io/oneDNN/page_getting_started_cpp.html#doxid-getting-started-cpp-1getting-started-cpp-tutorial), which in turn is called from [main() function](https://uxlfoundation.github.io/oneDNN/page_getting_started_cpp.html#doxid-getting-started-cpp-1getting-started-cpp-main) (which is also responsible for error handling).

## Public headers
To start using oneDNN we must first include the `dnnl.hpp` header file in the program. We also include `dnnl_debug.h` in `example_utils.hpp`, which contains some debugging facilities like returning a string representation for common oneDNN C types.
>  头文件: `dnnl.hpp, dnnl_debug.h`

## `getting_started_tutorial()` function
### Engine and stream
All oneDNN primitives and memory objects are attached to a particular [dnnl::engine](https://uxlfoundation.github.io/oneDNN/struct_dnnl_engine-2.html#doxid-structdnnl-1-1engine), which is an abstraction of a computational device (see also [Basic Concepts](https://uxlfoundation.github.io/oneDNN/dev_guide_basic_concepts.html#doxid-dev-guide-basic-concepts)). The primitives are created and optimized for the device they are attached to and the memory objects refer to memory residing on the corresponding device. In particular, that means neither memory objects nor primitives that were created for one engine can be used on another.
>  所有的 oneDNN 原语和内存对象都附加在特定的 `dnnl::engine` 上，engine 作为特定计算设备的抽象
>  原语会针对设备创建并优化，内存对象也指向对应设备的内存
>  这也意味着为一个 engine 创建的原语和内存对象不能用于另一个 engine

To create an engine, we should specify the [dnnl::engine::kind](https://uxlfoundation.github.io/oneDNN/enum_dnnl_engine_kind.html#doxid-structdnnl-1-1engine-1a2635da16314dcbdb9bd9ea431316bb1a) and the index of the device of the given kind.

```cpp
engine eng(engine_kind, 0);
```

>  engine 的创建需要指定 `dnnl::engine::kind` 和对应 kind 的设备索引

In addition to an engine, all primitives require a [dnnl::stream](https://uxlfoundation.github.io/oneDNN/struct_dnnl_stream-2.html#doxid-structdnnl-1-1stream) for the execution. The stream encapsulates an execution context and is tied to a particular engine.

The creation is pretty straightforward:

```cpp
stream engine_stream(eng);
```

>  原语还需要 `dnnl::stream` 封装其执行环境，`dnnl::stream` 同样绑定到特定的 engine
>  创建 `dnnl::stream` 时，传入 engine 即可

In the simple cases, when a program works with one device only (e.g. only on CPU), an engine and a stream can be created once and used throughout the program. Some frameworks create singleton objects that hold oneDNN engine and stream and use them throughout the code.

### Data preparation (code outside of oneDNN)
Now that the preparation work is done, let’s create some data to work with. We will create a 4D tensor in NHWC format, which is quite popular in many frameworks.

Note that even though we work with one image only, the image tensor is still 4D. The extra dimension (here N) corresponds to the batch, and, in case of a single image, is equal to 1. It is pretty typical to have the batch dimension even when working with a single image.

In oneDNN, all CNN primitives assume that tensors have the batch dimension, which is always the first logical dimension (see also [Naming Conventions](https://uxlfoundation.github.io/oneDNN/dev_guide_conventions.html#doxid-dev-guide-conventions)).
>  oneDNN 的所有 CNN 原语都假设张量具有 batch 维度，batch 维度总是第一个逻辑维度

```cpp
const int N = 1, H = 13, W = 13, C = 3;

// Compute physical strides for each dimension
const int stride_N = H * W * C;
const int stride_H = W * C;
const int stride_W = C;
const int stride_C = 1;

// An auxiliary function that maps logical index to the physical offset
auto offset = [=](int n, int h, int w, int c) {
    return n * stride_N + h * stride_H + w * stride_W + c * stride_C;
};

// The image size
const int image_size = N * H * W * C;

// Allocate a buffer for the image
std::vector<float> image(image_size);

// Initialize the image with some values
for (int n = 0; n < N; ++n)
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            for (int c = 0; c < C; ++c) {
                int off = offset(
                        n, h, w, c); // Get the physical offset of a pixel
                image[off] = -std::cos(off / 10.f);
            }
```

### Wrapping data into a oneDNN memory object
Now, having the image ready, let’s wrap it in a [dnnl::memory](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory-2.html#doxid-structdnnl-1-1memory) object to be able to pass the data to oneDNN primitives.

Creating [dnnl::memory](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory-2.html#doxid-structdnnl-1-1memory) comprises two steps:
1. Initializing the [dnnl::memory::desc](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory_desc-2.html#doxid-structdnnl-1-1memory-1-1desc) struct (also referred to as a memory descriptor), which only describes the tensor data and doesn’t contain the data itself. Memory descriptors are used to create [dnnl::memory](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory-2.html#doxid-structdnnl-1-1memory) objects and to initialize primitive descriptors (shown later in the example).
2. Creating the [dnnl::memory](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory-2.html#doxid-structdnnl-1-1memory) object itself (also referred to as a memory object), based on the memory descriptor initialized in step 1, an engine, and, optionally, a handle to data. The memory object is used when a primitive is executed.

>  我们将数据对象封装在 `dnnl::memory` 中，分为两步:
>  1. 初始化 `dnnl::memory::desc` 结构，它仅描述数据，不包含数据本身，内存描述符用于创建内存对象和原语描述符
>  2. 使用内存描述符创建 `dnnl::memory`

Thanks to the [list initialization](https://en.cppreference.com/w/cpp/language/list_initialization) introduced in C++11, it is possible to combine these two steps whenever a memory descriptor is not used anywhere else but in creating a [dnnl::memory](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory-2.html#doxid-structdnnl-1-1memory) object.

However, for the sake of demonstration, we will show both steps explicitly.

#### Memory descriptor
To initialize the [dnnl::memory::desc](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory_desc-2.html#doxid-structdnnl-1-1memory-1-1desc), we need to pass:

1. The tensor’s dimensions, the semantic order of which is defined by the primitive that will use this memory (descriptor).
    
    Warning
    Memory descriptors and objects are not aware of any meaning of the data they describe or contain.
    
2. The data type for the tensor ([dnnl::memory::data_type](https://uxlfoundation.github.io/oneDNN/enum_dnnl_memory_data_type.html#doxid-structdnnl-1-1memory-1a8e83474ec3a50e08e37af76c8c075dce)).
3. The memory format tag ([dnnl::memory::format_tag](https://uxlfoundation.github.io/oneDNN/enum_dnnl_memory_format_tag.html#doxid-structdnnl-1-1memory-1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f)) that describes how the data is going to be laid out in the device’s memory. The memory format is required for the primitive to correctly handle the data.

>  初始化内存描述符需要:
>  - 张量维度
>  - 数据类型 (`dnnl::memory::data_type`)
>  - 内存格式标签 (`dnnl::memory::format_tag`)，描述数据的内存布局

The code:

```cpp
auto src_md  = memory::desc(
        {N, C, H, W}, // logical dims, the order is defined by a primitive
        memory::data_type::f32, // tensor's data type
        memory::format_tag::nhwc // memory format, NHWC in this case
);
```

The first thing to notice here is that we pass dimensions as `{N, C, H, W}` while it might seem more natural to pass `{N, H, W, C}`, which better corresponds to the user’s code. This is because oneDNN CNN primitives like ReLU always expect tensors in the following form:
>  创建描述符时，我们传入的 (逻辑) 维度为 NCHW，这是为了匹配 oneDNN CNN 原语 (例如 ReLU) 所期望的形状

| Spatial dim | Ten               |
| :---------- | :---------------- |
| 0D          | N x C             |
| 1D          | N x C x W         |
| 2D          | N x C x H x W     |
| 3D          | N x C x D x H x W |

where:

- N is a batch dimension (discussed above),
- C is channel (aka feature maps) dimension, and
- D, H and W are spatial dimensions.

Now that the logical order of dimension is defined, we need to specify the memory format (the third parameter), which describes how logical indices map to the offset in memory. This is the place where the user’s format NHWC comes into play. oneDNN has different [dnnl::memory::format_tag](https://uxlfoundation.github.io/oneDNN/enum_dnnl_memory_format_tag.html#doxid-structdnnl-1-1memory-1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f) values that cover the most popular memory formats like NCHW, NHWC, CHWN, and some others.
>  定义好了逻辑维度，我们需要指定内存格式
>  内存格式描述了逻辑索引如何映射到内存 offset，我们指定为 `dnnl::memory::nhwc`，表示实际的数据格式

The memory descriptor for the image is called `src_md`. The `src` part comes from the fact that the image will be a source for the ReLU primitive (that is, we formulate memory names from the primitive perspective; hence we will use `dst` to name the output memory). The `md` is an initialism for Memory Descriptor.

##### Alternative way to create a memory descriptor
Before we continue with memory creation, let us show the alternative way to create the same memory descriptor: instead of using the [dnnl::memory::format_tag](https://uxlfoundation.github.io/oneDNN/enum_dnnl_memory_format_tag.html#doxid-structdnnl-1-1memory-1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f), we can directly specify the strides of each tensor dimension:

```cpp
auto alt_src_md = memory::desc(
        {N, C, H, W}, // logical dims, the order is defined by a primitive
        memory::data_type::f32, // tensor's data type
        {stride_N, stride_C, stride_H, stride_W} // the strides
);

// Sanity check: the memory descriptors should be the same
if (src_md != alt_src_md)
    throw std::logic_error("Memory descriptor initialization mismatch.");
```

Just as before, the tensor’s dimensions come in the `N, C, H, W` order as required by CNN primitives. To define the physical memory format, the strides are passed as the third parameter. Note that the order of the strides corresponds to the order of the tensor’s dimensions.

> 我们也可以手动指定内存中的步长，用于定义张量在内存中如何存储
> 步长表示在每个维度上移动一个单位所需的字节数 (或元素数)，例如：
> - 如果是 `NCHW` 格式，那么：
    - `stride_N = 1`
    - `stride_C = H * W`
    - `stride_H = W`
    - `stride_W = 1`
> - 如果是 `NHWC` 格式：
    - `stride_N = 1`
    - `stride_H = W * C`
    - `stride_W = C`
    - `stride_C = 1`

Warning
Using the wrong order might lead to incorrect results or even a crash.

#### Creating a memory object
Having a memory descriptor and an engine prepared, let’s create input and output memory objects for a ReLU primitive.

```cpp
// src_mem contains a copy of image after write_to_dnnl_memory function
auto src_mem = memory(src_md, eng);
write_to_dnnl_memory(image.data(), src_mem);

// For dst_mem the library allocates buffer
auto dst_mem = memory(src_md, eng);
```

We already have a memory buffer for the source memory object. We pass it to the [dnnl::memory::memory(const dnnl::memory::desc &, const dnnl::engine &, void *)](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory-2.html#doxid-structdnnl-1-1memory-1a7463ff54b529ec2b5392230861212a09) constructor that takes a buffer pointer as its last argument.

>  我们已经分配了数据内存，故创建 `dnnl::memory` 时，传入指针和描述符即可

Let’s use a constructor that instructs the library to allocate a memory buffer for the `dst_mem` for educational purposes.

The key difference between these two are:

1. The library will own the memory for `dst_mem` and will deallocate it when `dst_mem` is destroyed. That means the memory buffer can be used only while `dst_mem` is alive.
2. Library-allocated buffers have good alignment, which typically results in better performance.

>  也可以不传入指针，那么 oneDNN 就会自己分配内存，并且会在内存对象被销毁后自己释放内存
>  oneDNN 分配的内存会注意对齐方式，故通常性能会更好

Note
Memory allocated outside of the library and passed to oneDNN should have good alignment for better performance.

In the subsequent section we will show how to get the buffer (pointer) from the `dst_mem` memory object.

### Creating a ReLU primitive
Let’s now create a ReLU primitive.

The library implements ReLU primitive as a particular algorithm of a more general [Eltwise](https://uxlfoundation.github.io/oneDNN/dev_guide_eltwise.html#doxid-dev-guide-eltwise) primitive, which applies a specified function to each and every element of the source tensor.
>  oneDNN 将 ReLU 原语实现为 Eltwise 原语的一种特定形式
>  Eltwise 原语将特定的函数为 source tensor 中的每个元素应用

Just as in the case of [dnnl::memory](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory-2.html#doxid-structdnnl-1-1memory), a user should always go through (at least) two creation steps (which however, can be sometimes combined thanks to C++11):

1. Create an operation primitive descriptor (here [dnnl::eltwise_forward::primitive_desc](https://uxlfoundation.github.io/oneDNN/struct_dnnl_eltwise_forward_primitive_desc.html#doxid-structdnnl-1-1eltwise-forward-1-1primitive-desc)) that defines operation parameters and is a lightweight descriptor of the actual algorithm that implements the given operation. The user can query different characteristics of the chosen implementation such as memory consumptions and some others that will be covered in the next topic ([Memory Format Propagation](https://uxlfoundation.github.io/oneDNN/page_memory_format_propagation_cpp.html#doxid-memory-format-propagation-cpp)).
2. Create a primitive (here [dnnl::eltwise_forward](https://uxlfoundation.github.io/oneDNN/struct_dnnl_eltwise_forward.html#doxid-structdnnl-1-1eltwise-forward)) that can be executed on memory objects to compute the operation.

>  创建原语的步骤为:
>  1. 创建原语描述符，这里为 `dnnl::eltwise_forward::primitive_desc`，它定义了运算参数、实际算法等
>  2. 创建原语，这里为 `dnnl::eltwise_fowrad`，原语是可以实际在内存对象上执行计算的抽象

oneDNN separates steps 2 and 3 to enable the user to inspect details of a primitive implementation prior to creating the primitive. This may be expensive, because, for example, oneDNN generates the optimized computational code on the fly.

Note
Primitive creation might be a very expensive operation, so consider creating primitive objects once and executing them multiple times.

>  原语创建的较为昂贵的操作，故考虑创建原语一次，执行它多次

The code:

```cpp
// ReLU primitive descriptor, which corresponds to a particular
// implementation in the library
auto relu_pd = eltwise_forward::primitive_desc(
        eng, // an engine the primitive will be created for
        prop_kind::forward_inference, algorithm::eltwise_relu,
        src_md, // source memory descriptor for an operation to work on
        src_md, // destination memory descriptor for an operation to work on
        0.f, // alpha parameter means negative slope in case of ReLU
        0.f // beta parameter is ignored in case of ReLU
);

// ReLU primitive
auto relu = eltwise_forward(relu_pd); // !!! this can take quite some time
```

A note about variable names. Similar to the `_md` suffix used for memory descriptors, we use `_pd` for the primitive descriptors, and no suffix for primitives themselves.

It is worth mentioning that we specified the exact tensor and its memory format when we were initializing the `relu_pd`. That means `relu` primitive would perform computations with memory objects that correspond to this description. This is the one and only one way of creating non-compute-intensive primitives like [Eltwise](https://uxlfoundation.github.io/oneDNN/dev_guide_eltwise.html#doxid-dev-guide-eltwise), [Batch Normalization](https://uxlfoundation.github.io/oneDNN/dev_guide_batch_normalization.html#doxid-dev-guide-batch-normalization), and others.

Compute-intensive primitives (like [Convolution](https://uxlfoundation.github.io/oneDNN/dev_guide_convolution.html#doxid-dev-guide-convolution)) have an ability to define the appropriate memory format on their own. This is one of the key features of the library and will be discussed in detail in the next topic: [Memory Format Propagation](https://uxlfoundation.github.io/oneDNN/page_memory_format_propagation_cpp.html#doxid-memory-format-propagation-cpp).

>  非计算密集的原语可以在初始化时由用户指定张量和内存格式，原语直接按照指定的方式执行
>  计算密集的原语可以自行定义合适的内存格式，不需要用户管

### Executing the ReLU primitive
Finally, let’s execute the primitive and wait for its completion.

The input and output memory objects are passed to the `execute()` method using a <tag, memory> map. Each tag specifies what kind of tensor each memory object represents. 
>  原语的执行通过调用 `execute` 方法进行，调用时要以 `<tag, memory>` 的形式传入内存对象，其中每个 tag 都指定了相应内存对象标识的 tensor 类型

All [Eltwise](https://uxlfoundation.github.io/oneDNN/dev_guide_eltwise.html#doxid-dev-guide-eltwise) primitives require the map to have two elements: a source memory object (input) and a destination memory (output).
>  所有的 `Eltwise` 原语要求 <tag, memory> map 具有两个元素: 一个源内存对象 (输入)，一个目标内存对象 (输出)

A primitive is executed in a stream (the first parameter of the `execute()` method). Depending on a stream kind, an execution might be blocking or non-blocking. This means that we need to call [dnnl::stream::wait](https://uxlfoundation.github.io/oneDNN/struct_dnnl_stream-2.html#doxid-structdnnl-1-1stream-1a59985fa8746436057cf51a820ef8929c) before accessing the results.

```cpp
// Execute ReLU (out-of-place)
relu.execute (engine_stream, // The execution stream
        {
                // A map with all inputs and outputs
                {DNNL_ARG_SRC, src_mem}, // Source tag and memory obj
                {DNNL_ARG_DST, dst_mem}, // Destination tag and memory obj
        });

// Wait the stream to complete the execution
engine_stream.wait ();
```

The [Eltwise](https://uxlfoundation.github.io/oneDNN/dev_guide_eltwise.html#doxid-dev-guide-eltwise) is one of the primitives that support in-place operations, meaning that the source and destination memory can be the same. To perform in-place transformation, the user must pass the same memory object for both the `DNNL_ARG_SRC` and `DNNL_ARG_DST` tags:
>  Eltwise 原语支持原地运算，也就是源和目标内存对象可以相同

```cpp
// Execute ReLU (in-place)
// relu.execute(engine_stream,  {
//          {DNNL_ARG_SRC, src_mem},
//          {DNNL_ARG_DST, src_mem},
//         });
```

### Obtaining the result and validation
Now that we have the computed result, let’s validate that it is actually correct. The result is stored in the `dst_mem` memory object. So we need to obtain the C++ pointer to a buffer with data via [dnnl::memory::get_data_handle()](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory-2.html#doxid-structdnnl-1-1memory-1a24aaca8359e9de0f517c7d3c699a2209) and cast it to the proper data type as shown below.
>  完成计算后，结果就存在 `dst_mem` 内存对象中，我们通过 `dnnl::memory::get_data_handler()` 获取数据指针

Warning
The [dnnl::memory::get_data_handle()](https://uxlfoundation.github.io/oneDNN/struct_dnnl_memory-2.html#doxid-structdnnl-1-1memory-1a24aaca8359e9de0f517c7d3c699a2209) returns a raw handle to the buffer, the type of which is engine specific. For the CPU engine the buffer is always a pointer to `void`, which can safely be used. However, for engines other than CPU the handle might be runtime-specific type, such as `cl_mem` in case of GPU/OpenCL.

>  注意，`dnnl::memory::get_data_handler()` 返回的是对缓冲区的原始句柄，其类型取决于具体的引擎
>  对于 CPU ，句柄始终是指向 `void` 的指针，可以安全使用
>  对于 GPU/OpenCL，句柄可能是特定于运行时的类型，例如 `cl_mem`

```cpp
// Obtain a buffer for the `dst_mem` and cast it to `float *`.
// This is safe since we created `dst_mem` as f32 tensor with known
// memory format.
std::vector<float> relu_image(image_size);
read_from_dnnl_memory(relu_image.data(), dst_mem);
/*
// Check the results
for (int n = 0; n < N; ++n)
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            for (int c = 0; c < C; ++c) {
                int off = offset(
                        n, h, w, c); // get the physical offset of a pixel
                float expected = image[off] < 0
                        ? 0.f
                        : image[off]; // expected value
                if (relu_image[off] != expected) {
                    std::cout << "At index(" << n << ", " << c << ", " << h
                              << ", " << w << ") expect " << expected
                              << " but got " << relu_image[off]
                              << std::endl;
                    throw std::logic_error("Accuracy check failed.");
                }
            }
```

## `main()` function
We now just call everything we prepared earlier.

Because we are using the oneDNN C++ API, we use exceptions to handle errors (see [API](https://uxlfoundation.github.io/oneDNN/dev_guide_c_and_cpp_apis.html#doxid-dev-guide-c-and-cpp-apis)). The oneDNN C++ API throws exceptions of type [dnnl::error](https://uxlfoundation.github.io/oneDNN/struct_dnnl_error.html#doxid-structdnnl-1-1error), which contains the error status (of type [dnnl_status_t](https://uxlfoundation.github.io/oneDNN/enum_dnnl_status_t.html#doxid-group-dnnl-api-utils-1gad24f9ded06e34d3ee71e7fc4b408d57a)) and a human-readable error message accessible through regular `what()` method.

```cpp
int main(int argc, char **argv) {
    int exit_code = 0;

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    try {
        getting_started_tutorial(engine_kind);
    } catch (dnnl::error &e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch (std::string &e) {
        std::cout << "Error in the example: " << e << "." << std::endl;
        exit_code = 2;
    } catch (std::exception &e) {
        std::cout << "Error in the example: " << e.what() << "." << std::endl;
        exit_code = 3;
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
              << engine_kind2str_upper(engine_kind) << "." << std::endl;
    finalize();
    return exit_code;
}
```

Upon compiling and run the example the output should be just:

```
Example passed.
```

Users are encouraged to experiment with the code to familiarize themselves with the concepts. In particular, one of the changes that might be of interest is to spoil some of the library calls to check how error handling happens. For instance, if we replace

```cpp
relu.execute(engine_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem},
    });
```

with

```cpp
relu.execute (engine_stream, 
        {DNNL_ARG_SRC, src_mem},
        // {DNNL_ARG_DST, dst_mem}, // Oops, forgot about this one
    });
```

we should get the following output:

```
oneDNN error caught:
        Status: invalid_arguments
        Message: could not execute a primitive
Example failed.
```