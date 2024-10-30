# 1 Introduction
本课程讨论如何将机器学习从开发阶段(development phase)引入到生产环境(production environment)，包括了一系列促进机器学习算法落地部署(productionisation)的方法
## 1.1 What is ML Compilation
机器学习编译(machine learning compilation/MLC)指通过变换和优化(transforming and optimizing)机器学习算法使其从开发形式(development form)到部署形式(deployment form)的过程
- Development form(开发形式)
	指我们在开发机器学习模型时使用的一系列元素(elements)
	典型的开发形式包括用PyTorch、TensorFlow、JAX等通用框架编写的模型描述(model descriptions)和与之相关的权重(weights)
- Depolyment form(部署形式)
	指执行机器学习应用程序所需要的一系列元素(elements)
	它通常涉及机器学习模型的每个步骤的支撑代码(例如库函数)、管理资源(例如内存)的例程(routines to manage resources)、与应用程序开发环境(application development environment)的接口(例如用于android app的java API) 

机器学习编译通常有以下几个目标：
- Integration and dependency minimization(集成与最小化依赖)
	部署通常涉及集成(Integration)——将必要的元素组合在一起以用于部署程序
	代码集成、最小化依赖能够减小应用的大小，并可以使应用程序能部署到更多的环境
- Leveraging hardware native acceleration(利用硬件加速)
	每个部署环境都有自己的原生加速技术(native accelaration techniques)，我们可以利用硬件本身的特性进行加速，例如构建调用原生加速库(invoke native accelaration libraries)的部署代码或生成利用原生指令(leverage native instrutions such as TensorCore)的部署代码
- Optimization in general(通用优化)
	运行(run)同一个模型一般有多种等效的方法，在不改变程序语义的情况下可以通过机器学习编译对其进行不同形式的优化，例如以最小化内存使用为目标的优化或以提高执行效率的为目标的优化

这些目标没有严格的界限，例如集成和硬件加速也可以被视为通用优化，具体目标取决于应用场景

许多机器学习编译实践涉及与来自不同背景的开发人员的合作，硬件开发人员需要支持他们最新的硬件加速，机器学习工程师实现额外的优化，同时算法工程师引入新模型

## 1.2 Why Study ML Compliation
## 1.3 Key Elements of ML Compliation
- Tensor(张量)
	张量是表示神经网络模型执行(neural network execution)的输入、输出和中间结果的多维数组(multidimensional array)
- Tensor functions(张量函数)
	神经网络的“知识”被编码在权重(weights)和接受张量和输出张量的计算序列(the sequence of computations that takes in tensors and output tensors)中，这些计算被称为张量函数
	一个张量函数不一定需要对应神经网络计算的单个步骤，部分计算或整个端到端(end-to-end)计算也可以视为一个张量函数
## 1.4 Summary
- 机器学习编译的目标
    集成与最小化依赖
    利用硬件加速
    通用优化
- 为什么学习机器学习编译
    构建机器学习部署解决方案
    深入了解现有机器学习框架
    为新兴硬件建立软件栈
- 机器学习编译的关键要素
	张量和张量函数
    抽象和实现是值得思考的工具

# 2 Tensor Program Abstraction
## 2.1 Primitive Tensor Function
一个典型的模型执行(model execution)包含多个将输入张量之间转化为输出张量的计算步骤(compuation steps)，其中的每一个单元步骤(each unit step)都被称为元张量函数(primitive tensor function)

相同的元张量函数可以有许多不同的实现
例如元张量函数 `add` 的三个不同实现：
```python
torch.add
```

```python
def add(a, b, c):
	for i in range(128):
		c[i] = a[i] + b[i]
```

```c
void add(float *a, float *b, float *c){
	for(int i = 0; i < 128; ++i){
		c[i] = a[i] + b[i];
	}
}
```

许多机器学习框架都提供了可以将元张量函数变换为更加专门的、针对特定工作和部署环境(particular workload and deployment environment)的函数的机器学习编译过程(machine learning compilation precedures)
例如将元张量函数 `add` 的实现
```python
for x in range(128):
	C[x] = A[x] + B[x]
```
变化为
```python
parallel for xo in range(32):
	C[xo*4:xo*4+4] = f32x4.add(A[xo*4:xo*4+4], B[xo*4:xo*4+4])
```
其中 `f32x4.add` 是一个特殊的执行向量加法计算的函数`
## 2.2 Tensor Program Abstraction
张量程序是用于表示元张量函数的一类有效抽象，它一般包括了：
- 存储数据的多维数组(multi-dimensional buffer)
- 驱动张量计算的循环嵌套(loop nests that drive the tensor computations)
- 计算部分的语句(computation statements)
```python
from tvm.script import tir as T
@T.prim_func
def main(A: T.Buffer[128, "float32"], # buffers that holds data
		 B: T.Buffer[128, "float32"],
		 C: T.Buffer[128, "float32"]):
	for i in range(128): # loop nests that drive compute iterations
		with T.block("C"):
			vi = T.axis.spatial(128, i) # extra info about iteration
			C[vi] = A[vi] + B[vi] # computation statements
```
称这类抽象为张量程序抽象(tensor program abstraction)
张量程序的一个重要性质是，它们能够通过一系列有效的变化操作(如循环拆分、并行化、向量化)进行改变

但我们不能任意地对程序进行变换，例如只有当计算在循环迭代之间保持独立性的情况下我们才可以将循环并行化，而这类额外信息也可以体现在张量程序中

例如，示例程序中就包含额外的 `T.axis.spatial` 标注，表明 `vi` 这个特定的变量被映射到循环变量 `i`，并且所有的迭代都是独立的
这个信息对于执行这个程序而言并非必要，但会使得我们在变换这个程序时更加方便，在此例中，我们知道可以安全地并行或者重新排序所有与 `vi` 有关的循环

## 2.3 Summary
- 元张量函数表示模型执行中的单个单元计算(single unit of computation)
	 机器学习编译的过程即有选择地转换元张量函数的实现
- 张量程序是表示元张量函数的有效抽象
	关键成分包括: 多维数组，循环嵌套，计算语句
	可以利用程序变换得到更高效的张量程序
    张量程序中额外的结构能够为程序变换提供更多的信息

## 2.4 TensorIR: Tensor Program Abstraction Case Study

### 2.4.1 TensorIR
TensorIR 是标准机器学习编译框架Apache TVM中使用的张量程序抽象

具体地，对于两个大小为$128\times128$的矩阵$A$和$B$，进行如下两步的张量计算：
$Y_{i,j} = \sum_k A_{i,k}\times B_{k,j}$
$C_{i,j} = relu(Y_{i,j}) = max(Y_{i,j},0)$

可以使用Numpy中的数组计算实现：
```python
import numpy as np
dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
c_mm_rule = np.maximum(a_np @ b_np, 0)
```
在底层，Numpy会调用库(例如OpenBLAS)和它自己的C语言实现来执行这些计算

也可以使用NumPy API的一个受限子集实现：
(称之为低级 NumPy，它遵守以下的约定：使用循环而不是数组函数来展示可能的循环计算，尽量通过 `numpy.empty` 显式地分配数组并传递它们)
```python
def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)
```
它包含了我们将在张量计算的实际实现中会使用的所有可能元素：
多维缓冲区/数组、在数组维度上的循环、在循环下执行的计算语句

也可以使用TensorIR实现：
```python
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
```

在 NumPy和TensorIR之间具有直接对应关系的元素有：
**函数参数与缓冲区**
```python
# TensorIR
def mm_relu(A: T.Buffer[(128, 128), "float32"],
            B: T.Buffer[(128, 128), "float32"],
            C: T.Buffer[(128, 128), "float32"]):
    ...
# numpy
def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    ...
```
这里A、B和C采用名为 `T.Buffer` 的类型，其形状参数为 `(128, 128)`，数据类型为 `float32`
这些附加信息有助于可能的机器学习编译过程生成专门针对形状和数据类型的代码

TensorIR 在中间结果分配中也使用了 `T.buffer` 类型：
```python
# TensorIR
Y = T.alloc_buffer((128, 128), dtype="float32")
# numpy
Y = np.empty((128, 128), dtype="float32")
```

**For循环迭代**
```python
# TensorIR
for i, j, k in T.grid(128, 128, 128):
# numpy
for i in range(128):
    for j in range(128):
        for k in range(128):
```
其中 `T.grid` 是 TensorIR 中的语法糖，供我们书写多个嵌套的迭代器(multiple nested iterators)

**计算语句**
```python
# TensorIR
with T.block("Y"):
    vi = T.axis.spatial(128, i)
    vj = T.axis.spatial(128, j)
    vk = T.axis.reduce(128, k)
    with T.init():
        Y[vi, vj] = T.float32(0)
    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

# coressponding numpy code
vi, vj, vk = i, j, k
if vk == 0:
    Y[vi, vj] = 0
Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
```
TensorIR的实现中，包含了一个名为 `T.block` 的额外结构
块(block)是TensorIR中的基本计算单位
TensorIR中的计算块一般比NumPy的计算语句包含更多的信息
TensorIR中，一个块包含一组块轴（`vi、vj、vk`）和围绕它们定义的计算：
```python
vi = T.axis.spatial(128, i)
vj = T.axis.spatial(128, j)
vk = T.axis.reduce(128, k)
```
这三行声明了关于块轴的关键性质，语法如下：
```python
[block_axis] = T.axis.[axis_type]([axis_range], [mapped_value])
```
这三行包含以下信息：
- 定义了 `vi`、`vj`、`vk` 应被绑定到的位置(`mapped_value` )
	在本例中为 `i`、`j` 和 `k` 
- 声明了 `vi`、`vj`、`vk` 的原始范围( `axis_range` )
	在本例中 `vi = T.axis.spatial(128, i)` 中的 `128` 即表示 `vi` 对应的范围为 `range(0, 128)`
- 声明了块轴的属性( `axis_type` )
	在本例中为 `spatial` 或 `reduce`

在每个块轴都直接映射到外部循环迭代器的情况下，可以使用 `T.axis.remap` 在一行中声明所有块轴：
```python
# SSR means the properties of each axes are "spatial", "spatial", "reduce"
vi, vj, vk = T.axis.remap("SSR", [i, j, k])
```
它等价于
```python
vi = T.axis.spatial(range_of_i, i)
vj = T.axis.spatial(range_of_j, j)
vk = T.axis.reduce(range_of_k, k)
```

示例中的TVMScript的其余元素有：
**函数属性和装饰器**
函数的属性信息包含了关于函数的额外信息
```python
T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
```
 示例中的 `global_symbol` 对应函数名，`tir.noalias` 是一个属性，表示所有的缓冲区域不重叠
 
`@tvm.script.ir_module` 和 `@T.prim_func` 这两个装饰器用于表示对应部分的类型
`@tvm.script.ir_module` 表示 `MyModule` 是一个IRModule，IRModule是在机器学习编译中保存张量函数集合的容器对象
`@T.prim_func` 表示 `mm_relu` 是一个元张量函数

### 2.4.2 Transformation
机器学习编译工作流的主要成分就是关于元张量函数的变换

`mm_relu` 的一个稍微不同的变体：
```python
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
	Y = np.empty((128, 128), dtype="float32")
	for i in range(128):
		for j0 in range(32):
			for k in range(128):
				for j1 in range(4):
					j = j0*4 + j1
					if k == 0:
						Y[i, j] = 0
					Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
	for i in range(128):
		for j in range(128):
			C[i, j] = max(Y[i, j], 0)
```
它用两个循环 `j0` 和 `j1` 替换了 `j` 循环，并且迭代顺序略有变化

TensorIR引入了一个名为Schedule的辅助结构用于对原张量函数进行变换
首先创建一个以给定的 `MyModule` 作为输入的Schedule辅助类
```python
sch = tvm.tir.Schedule(MyModule)
```
然后获得对块 `Y` 和相应循环的引用
```python
block_Y = sch.get_block("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)
```

要执行将循环 `j` 分成两个循环，其中内部循环的长度为4的变换，则运行
```python
j0, j1 = sch.split(j, factors=[None, 4])
```
我们得到了 `j_0` 和 `j_1`两个新的循环，范围分别为32和4

要重新排序循环，则运行
```python
sch.reorder(j0, k, j1)
```

最后得到变换后的程序
```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        # with T.block("root"):
        Y = T.alloc_buffer((128, 128))
        for i, j_0, k, j_1 in T.grid(128, 32, 128, 4):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j_0 * 4 + j_1)
                vk = T.axis.reduce(128, k)
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(Y[vi, vj])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(Y[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
```

 要继续将块 `C` 移动到 `Y` 的内循环里，可以使用名为 `reverse_compute_at` 原语
 ```python
block_C = sch.get_block("C", "mm_relu")
sch.reverse_compute_at(block_C, j0)
```
得到
```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        # with T.block("root"):
        Y = T.alloc_buffer((128, 128))
        for i, j_0 in T.grid(128, 32):
            for k, j_1 in T.grid(128, 4):
                with T.block("Y"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 4 + j_1)
                    vk = T.axis.reduce(128, k)
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(Y[vi, vj])
                    with T.init():
                        Y[vi, vj] = T.float32(0)
                    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
            for ax0 in range(4):
                with T.block("C"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 4 + ax0)
                    T.reads(Y[vi, vj])
                    T.writes(C[vi, vj])
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
```

### 2.4.3 Build and Run
要运行IRModule中的程序，首先需要调用 `build` 函数将IRModule变换为 `runtime.Module` ，它表示可运行函数的集合
```python
rt_lib = tvm.build(MyModule, target="llvm")
```
其中 `target` 指定了部署环境的详细信息，当针对不同的平台(例如Android)或具有特殊说明的平台(例如Intel Skylake)时，需要相应地调整 `target`

然后创建三个用于保存输入和输出的TVM NDArray
```python
a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((128, 128), dtype="float32")
```

然后就可以从 `rt_lib` 中获取可运行函数，并传递三个数组参数来执行它
```python
func_mm_relu = rt_lib["mm_relu"]
func_mm_relu(a_nd, b_nd, c_nd)
```

在 `build` 函数中传入变换之后的IRModule则可以对变换后的程序构建并运行
```python
rt_lib_after = tvm.build(sch.mod, target="llvm")
rt_lib_after["mm_relu"](a_nd, b_nd, c_nd)
```

### 2.4.4 Summary
- TensorIR抽象
    包含循环、多维缓冲区等常用元素
    引入封装了循环计算要求(loop computation requirements)的新结构块
    可以在 Python AST中构建(通过 TVMScript)
- 可以使用变换来创建不同的TensorIR变体(variants)
- 通用 MLC 流程：开发、变换、构建(develop, transform, build)
# 3 End to End Model Execution
## 3.1 End to End Model Integration
一个简单的端到端模型，由两个全连接层和一个relu激活层组成
![[MLC-Fig1.png]]

模型的高层Numpy实现
```python
def numpy_mlp(data, w0, b0, w1, b1):
	lv0 = data @ w0.T + b0
	lv1 = np.maximum(lv0, 0)
	lv2 = lv1 @ w1.T + b1
	return lv2
```

模型的底层Numpy实现
```python
def lnumpy_linear0(X: np.nadrray, W: np.ndarray, B: np.ndarray, Z: np.ndarray)
	Y = np.empty((1, 128), dtype="float32")
	for i in range(1):
		for j in range(128):
			for k in range(784):
				if k == 0:
					Y[i, j] = 0
				Y[i, j] = Y[i, j] + X[i, k] * W[j, k]
	for i in range(1):
		for j in range(128):
			Z[i, j] = Y[i, j] + B[j]

def lnumpy_relu0(X: np.ndarray, Y: np.ndarray):
	for i in range(1):
		for j in range(128):
			Y[i, j] = np.maximum(X[i, j], 0)

def lnumpy_linear1(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
	Y = np.empty((1, 10), dtype="float32")
	for i in range(1):
		for j in range(10):
			for k in range(128):
				if k == 0:
					Y[i, j] = 0
				Y[i, j] = Y[i, j] + X[i, k] * W[j, k]
	for i in range(1):
		for j in range(10):
			Z[i, j] = Y[i, j] + B[j]

def lnumpy_mlp(data, w0, b0, w1, b1):
	lv0 = np.empty((1, 128), dtype="float32")
	lnumpy_linear0(data, w0, b0, lv0)

	lv1 = np.empty((1, 128), dtype="float32")
	lnumpy_relu0(lv0, lv1)

	out = np.empty((1, 10), dtype="float32")
	lnumpy_linear1(lv1, w1, b1, out)
	return out
```

## 3.2 Constructing an End to End IRModule in TVMScript
模型TVMScript实现
```python
import tvm
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T
from tvm import relax

@tvm.script.ir_module
class MyModule:
	@T.prim_func
	def relu0(x: T.handle, y: T.handle):
		n = T.int64()
		X = T.match_buffer(x, (1, n), "float32")
		Y = T.match_buffer(y, (1, n), "float32")
		for i, j in T.grid(1, n):
			with T.block("Y"):
				vi, vj = T.axis.remap("SS", [i, j])
				Y[vi, vj] = T.max(X[vi, vj], T.float32(0))

	@T.prim_func
	def linear0(x: T.handle,
				w: T.handle,
				b: T.handle,
				z: T.handle):
		m, n, k = T.int64(), T.int64(), T.int64()
		X = T.match_buffer(x, (1, m), "float32")
        W = T.match_buffer(w, (n, m), "float32")
        B = T.match_buffer(b, (n, ), "float32")
        Z = T.match_buffer(z, (1, n), "float32")
        Y = T.alloc_buffer((1, n), "float32")
        for i, j, k in T.grid(1, n, m):
	        with T.block("Y"):
		        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
		        with T.init():
			        Y[vi, vj] = T.float32(0)
			    Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
		for i, j in T.grid(1, n):
			with T.block("Z")
				vi, vj = T.axis.remap("SS", [i, j])
				Z[vi, vj] = Y[vi ,vj] + B[vj]

	@R.function
    def main(x: R.Tensor((1, "m"), "float32"),
             w0: R.Tensor(("n", "m"), "float32"),
             b0: R.Tensor(("n", ), "float32"),
             w1: R.Tensor(("k", "n"), "float32"),
             b1: R.Tensor(("k", ), "float32")):
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("relu0", (lv0, ), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed("linear0", (lv1, w1, b1), R.Tensor((1, k), "float32"))
            R.output(out)
        return out
```
代码包含了元张量函数 `T.prim_func` 和松弛函数 `R.function`
松弛函数(relax function)是用于表示高层神经网络执行(high-level neural network execution)的抽象

#### 3.2.1 Computational Graph View
`main` 函数的计算图
![[MLC-Fig2.png]]
- 图中的每个框对应于计算操作(computation operations)
- 图中的箭头对应于中间张量(intermediate tensor)的输入输出

#### 3.2.2 `call_dps_packed` Construct
`R.call_dps_packed` 是引入元张量函数的过程，例如
```python
lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), dtype="float32"))
```

底层Numpy与之等效的操作为
```python
def lnumpy_call_dps_packed(prim_func, inputs, shape, dtype):
    res = np.empty(shape, dtype=dtype)
    prim_func(*inputs, res)
    return res
```
`call_dps_packed` 接受一个元张量函数( `prim_func` )的输入列表，并分配一个输出张量`res`，然后将输入和输出都传递给`prim_func` 
`prim_func` 执行后，结果会填充到 `res` 中，然后 `call_dps_packed` 返回结果

元张量函数的实现一般都采用以下约定：
```python
def low_level_prim_func(in0, in1, ..., out):
    # implementations
```
该约定称为目标传递(destination passing)
其中的想法就是输入和输出都在函数外部分配，然后传递给函数
该策略常用于底层库设计，以便高层框架处理内存分配决策

在一个高层函数内通过显式分配中间结果的方式可以将采用目标传递实现的底层函数组装到一起：
```python
def lnumpy_mlp(data, w0, b0, w1, b1):
    lv0 = np.empty((1, 128), dtype="float32")
    lnumpy_linear0(data, w0, b0, lv0)

    lv1 = np.empty((1, 128), dtype="float32")
    lnumpy_relu0(lv0, lv1)

    out = np.empty((1, 10), dtype="float32")
    lnumpy_linear1(lv1, w1, b1, out)
    return out
```
其对应的“计算图”为
![[MLC-Fig3.png]]
该图实际上失去了计算图所要求具备的一些性质
一般来说，计算图要求具备三个性质：
- 框的输入边只能对应于运算的输入
- 框的输出边只能对应于运算的输出
- 在满足边的拓扑排序的范围内，运算可以任意排序

如果一个函数只从其输入(inputs)中读取数据并通过其输出(outputs)返回结果，不会改变程序的其他部分(如递增全局计数器)，那么它是pure或side-effect free的
计算图内的每个运算都要求是side-effect free的

`call_dps_packed` 的意图就在于隐藏了调用底层元张量函数的细节，使得其对于上层展现出满足计算图性质的抽象

对于底层Numpy实现也可以运用该思想：
```python
def lnumpy_mlp_with_call_dps_packed(data, w0, b0, w1, b1):
    lv0 = lnumpy_call_dps_packed(lnumpy_linear0, (data, w0, b0), (1, 128), dtype="float32")
    lv1 = lnumpy_call_dps_packed(lnumpy_relu0, (lv0, ), (1, 128), dtype="float32")
    out = lnumpy_call_dps_packed(lnumpy_linear1, (lv1, w1, b1), (1, 10), dtype="float32")
    return out
```

#### 3.2.3 Dataflow Block
松弛函数中的另一个元素是 `R.dataflow()` 范围标注：
```python
with R.dataflow():
    lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
    lv1 = R.call_dps_packed("relu0", (lv0, ), R.Tensor((1, n), "float32"))
    out = R.call_dps_packed("linear0", (lv1, w1, b1), R.Tensor((1, k), "float32"))
    R.output(out)
```

数据流块(dataflow block)是用于标记程序计算图区域的一种方式
在数据流块中，要求所有的运算都是side-effect free的，而在数据流快之外，允许运算不是side-effect free的

#### 3.2.4 Section Checkpoint
到目前为止，我们已经完成了一个Relax程序的示例，涵盖了：
- 计算图
- `call_dps_packed`
- 数据流块

## 3.3 Build and Run the Model
调用 `relax.build` 来构建这个函数
```python
ex = relax.build(MyModule, target="llvm")
type(ex)
```
`build` 函数返回一个可执行文件(其并非传统操作系统中的可执行文件，不能直接在系统中运行，而是针对Relax VM设计的一种文件格式)

我们可以初始化一个虚拟机执行器(executor)，使我们能够运行该函数，此外，我们将传入第二个参数以指示要在哪个设备上运行该端到端执行
```python
vm = relax.VirtualMachine(ex, tvm.cpu())
```

构建包含输入数据和权重的 tvm NDArray，然后可传入输入参数和权重来运行 `main` 函数：
```python
data_nd = tvm.nd.array(img.reshape(1, 784))
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

nd_res = vm["main"](data_nd,
                    nd_params["w0"],
                    nd_params["b0"],
                    nd_params["w1"],
                    nd_params["b1"])
```

## 3.4 Integrate Existing Libraries in the Environment
可以将现有的库函数集成到MLC过程中：
```python
@tvm.script.ir_module
class MyModuleWithExternCall:
    @R.function
    def main(x: R.Tensor((1, "m"), "float32"),
             w0: R.Tensor(("n", "m"), "float32"),
             b0: R.Tensor(("n", ), "float32"),
             w1: R.Tensor(("k", "n"), "float32"),
             b1: R.Tensor(("k", ), "float32")):
        # block 0
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("env.linear", (x, w0, b0), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0, ), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed("env.linear", (lv1, w1, b1), R.Tensor((1, k), "float32"))
            R.output(out)
        return out
```

在示例代码中，我们在 `call_dps_packed` 中传入字符串
```python
R.call_dps_packed("env.linear", (x, w0, b0), R.Tensor((1, n), "float32"))
```
这些字符串即我们需要的在模型执行期间的存在的运行时函数(runtime function)的函数名称

#### 3.4.1 Register Runtime Function
要调用外部函数，需要注册相应的函数，例如：
```python
@tvm.register_func("env.linear", override=True)
def torch_linear(x: tvm.nd.NDArray,
                 w: tvm.nd.NDArray,
                 b: tvm.nd.NDArray,
                 out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)
    torch.mm(x_torch, w_torch.T, out=out_torch)
    torch.add(out_torch, b_torch, out=out_torch)

@tvm.register_func("env.relu", override=True)
def lnumpy_relu(x: tvm.nd.NDArray,
                out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    out_torch = torch.from_dlpack(out)
    torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)
```
在上面的代码中，我们使用 `from_dlpack` 将TVM NDArray 转换为Torch NDArray这是一个零拷贝转换，这意味着Torch NDArray与TVM NDArray共享底层内存

DLPack是一种通用的交换标准，允许在不同的框架之间转换张量/多维数组(Tensor/NDArray)而无需数据复制
`from_dlpack` API由多个框架支持，也是Python数组API标准的一部分

在这个特定的函数中，我们只是简单地调用 PyTorch 的实现
在真实的应用场景中，我们可以使用类似的机制将调用重定向到特定的库，例如 cuDNN或我们自己的库实现，也可以用不同的语言(例如 C++)注册没有 Python 依赖的函数

#### 3.4.2 Build and Run
```python
ex = relax.build(MyModuleWithExternCall, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd,
                    nd_params["w0"],
                    nd_params["b0"],
                    nd_params["w1"],
                    nd_params["b1"])
```

## 3.5 Mixing TensorIR Code and Libraries
```python
lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
lv1 = R.call_dps_packed("env.relu", (lv0, ), R.Tensor((1, n), "float32"))
out = R.call_dps_packed("env.linear", (lv1, w1, b1), R.Tensor((1, k), "float32"))
R.output(out)
```

## 3.6 Bind Parameters to IRModule
除了显式传递参数来构造主函数外，将参数绑定为附加到 IRModule 的常量通常会降低API的复杂程度 
```python
MyModuleWithParams = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
```
该语句会将参数名称(name)与 `nd_params` 中的键(key)匹配来创建绑定

得到
```python
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def linear0(x: T.handle, w: T.handle, b: T.handle, z: T.handle):
        m = T.int64()
        X = T.match_buffer(x, (1, m))
        n = T.int64()
        W = T.match_buffer(w, (n, m))
        B = T.match_buffer(b, (n,))
        Z = T.match_buffer(z, (1, n))
        # with T.block("root"):
        Y = T.alloc_buffer((1, n))
        for i, j, k in T.grid(1, n, m):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(X[vi, vk], W[vj, vk])
                T.writes(Y[vi, vj])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, n):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(Y[vi, vj], B[vj])
                T.writes(Z[vi, vj])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, metadata["relax.expr.Constant"][0], metadata["relax.expr.Constant"][1]), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            out = R.call_dps_packed("env.linear", (lv1, metadata["relax.expr.Constant"][2], metadata["relax.expr.Constant"][3]), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(out)
        return out

# Metadata omitted. Use show_meta=True in script() method to show it.
```
在上面的脚本中，`meta[relay.Constant][0]` (目前 `Relax` 的常量表达依然继承自 `Relay` ，未来该API 可能会更改)对应于一个存储常量的隐式字典(它没有显示为脚本的一部分，但仍然是 IRModule 的一部分)

现在可以仅传入输入数据来调用该函数
```python
ex = relax.build(MyModuleWithParams, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd)
```

## 3.7 Summary
- 计算图抽象有助于将元张量函数拼接在一起以进行端到端执行
- Relax抽象的关键要素包括
    - call_dps_packed 构造，将目标传递规范的元函数嵌入到计算图中
    - Dataflow block
- 计算图允许调用环境库函数和 `TensorIR` 函数
# 4 Automatic Program Optimization
## 4.1 Preclude
在过去的章节中，我们学习了如何构建元张量函数并将它们连接起来以进行端到端的模型执行，到目前为止，我们使用了三种主要的抽象类型：
- 驱动高层执行的计算图抽象
- 元张量函数的抽象
- 通过注册环境函数从而能被调用的库函数
所有这些元素都封装在一个 IRModule 中，大多数 MLC 过程可以看作是元张量函数之间的变换

## 4.2 Recap: Transform a Primitive Tensor Function
`tvm.tir.Schedule` 提供了名为 `trace` 的数据结构，它包含了 IRModule 在变换过程中所涉及的步骤，可以使用 `print(sch.trace)` 打印

## 4.3 Stochastic Schedule Transformation
在选择对原始TensorIR程序进行哪些变换时，许多选择基于我们对底层环境的理解，例如缓存和硬件单元
而在实践中，我们可能无法准确地决定每一个细节，因而我们想仅指定什么是变换程序的可能方法，省略(leave out)一些细节

实现目标的一种自然方法是在我们的变换中添加一些随机元素，例如
```python
def stochastic_schedule_mm(sch: tvm.tir.Schedule):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_factors = sch.sample_perfect_tile(loop=j, n=2)
    j_0, j_1 = sch.split(loop=j, factors=j_factors)
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch
```
![[MLC-Fig4.png]]
对比 `stochastic_schedule_mm` 和 `schedule_mm` 可以发现，它们唯一的区别是指定 `j_factors` 的方式
在 `schedule_mm` 中， `j_factors` 作为我们指定的参数传入
在 `stochastic_schedule_mm` 中，它来自 `sch.sample_perfect_tile`

`sch.sample_perfect_tile` 尝试使用随机数来作为 `j_factors` 的值，它在输入循环的长度的因子(factors)中进行采样，以便采样结果能完美地分割循环
例如，当原始循环长度为 `128` 时，拆分循环的可能方式包括：`[8, 16]`、`[32, 4]`、`[2, 64]`($8 * 16 = 32 * 4 = 2 * 64 = 128$)

每次运行 `stochastic_schedule_mm` 时，它都会随机采样一组不同的 `j_factors` ，`j_1` 的循环边界都会发生变化
```python
sch = tvm.tir.Schedule(MyModule)
sch = stochastic_schedule_mm(sch)
```
可以打印出最新的历史轨迹，以查看采样中做出的决定
```python
print(sch.trace)
```
得到
```python
# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  l1, l2, l3 = sch.get_loops(block=b0)
  v4, v5 = sch.sample_perfect_tile(loop=l2, n=2, max_innermost_factor=16, decision=[32, 4])
  l6, l7 = sch.split(loop=l2, factors=[v4, v5], preserve_unit_iters=True)
  sch.reorder(l1, l6, l3, l7)
  b8 = sch.decompose_reduction(block=b0, loop=l3)
```
注意 `sample_perfect_tile` 的 `decision=[...]` 部分对应于我们上次调用 `stochastic_schedule_mm` 时 `sampling_perfect_tile` 返回的值

### 4.3.1 Deep Dive into Stochastic Transformation
随机调度变换中发生的事情是原始确定性变换的简单泛化，包含两个附加元素：
- 采样操作得到随机变量，例如来自 `sample_perfect_tile` 的随机变量
- 利用随机变量进行的后续变换操作
让我们尝试逐步运行随机变换
```python
sch = tvm.tir.Schedule(MyModule)
block_C = sch.get_block("C", "main")
i, j, k = sch.get_loops(block=block_C)
j_factors = sch.sample_perfect_tile(loop=j, n=2)
```
运行
```python
type(j_factors[0])
```
得到
```text
tvm.tir.expr.Var
```
说明 `j_factors` 中的元素并不是实际的整数，它们是表示被采样的随机变量的符号变量(symbolic variables)，符号变量被传递给transofmation API从而指定诸如因子值之类的选择

此时查看 `sch.trace` ，可以在 `decisions` 字段中看到这些符号变量的选择
```python
# from tvm import tir
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  l1, l2, l3 = sch.get_loops(block=b0)
  v4, v5 = sch.sample_perfect_tile(loop=l2, n=2, max_innermost_factor=16, decision=[8, 16])
```

此时查看 `sch.mod.script()` ，可以发现IRModule保持不变，因为我们只对随机变量进行了采样，但还没有进行任何变换操作

然后进行变换
```python
j_0, j_1 = sch.split(loop=j, factors=j_factors)
sch.reorder(i, j_0, k, j_1)
```

此时查看 `sch.trace` ，可以看到这些变化记录
此时查看 `sch.mod.script()` ，可以看到IRModule发生了变化

然后可以做进一步的变化
```python
sch.reorder(i, j_0, k, j_1)
sch.decompose_reduction(block_C, k)
```

## 4.4 Search Over Stochastic Transformations
`stochastic_schedule_mm` 创建了一个可能程序的搜索空间(search space of possible programs)，最终结果具体取决于在每个采样步骤中做出的具体决定
![[MLC-Fig5.png]]
`stochastic_schedule_mm` 为我们提供了一组可能的(possible)程序而不是一个程序


要得到最佳选择，我们需要一个搜索算法
我们首先在下面的代码块中尝试最直接的搜索算法——随机搜索，它尝试重复运行 `stochastic_schedule_mm`，获取转换后的模块，运行测试，然后保留历史上最好(用时最短)的模块
```python
def random_search(mod: tvm.IRModule, num_trials=5):
    best_result = None
    best_sch = None

    for i in range(num_trials):
        sch = stochastic_schedule_mm(tvm.tir.Schedule(mod))
        lib = tvm.build(sch.mod, target="llvm")
        f_timer_after = lib.time_evaluator("main", tvm.cpu())
        result = f_timer_after(a_nd, b_nd, c_nd).mean

        print("=====Attempt %d, time-cost: %.3f ms====" % (i, result * 1000))
        print(sch.trace)

        # book keep the best result so far
        if best_result is None or result < best_result:
            best_result = result
            best_sch = sch

    return best_sch

sch = random_search(MyModule)
```
运行代码，我们会发现它经过了几个选择，然后在五次试验中返回了最佳运行

在实践中，我们需要使用更智能的算法，有时我们还需要提供额外的工具，例如远程设备上的基准测试
TVM的Meta-Schedule API提供了这些附加功能

`meta_schedule` 是一个支持在可能变换空间进行搜索操作的命名空间(namespace)
Meta-Schedule在幕后可以做很多事情，例如：
- 跨越多个进程的并行基准测试(parallel benchmarking)
- 使用代价模型(cost model)来避免每次都进行基准测试
- 基于历史轨迹进行遗传搜索(evolutionary search)，而不是每次都随机采样

尽管有这些工具，但关键思想是一致的：使用随机变换(stochastic transformation)来指定好的程序的搜索空间，使用 ``tune_tir`` API帮助在搜索空间内搜索并找到最优的变换
```python
from tvm import meta_schedule as ms

database = ms.tune_tir(
    mod=MyModule,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
    work_dir="./tune_tmp",
    task_name="main"
)

sch = ms.tir_integration.compile_tir(database, MyModule, "llvm --num-cores=1")
```

### 4.4.1 Leveraging Default AutoScheduling
在之前，我们通过 
```python
space=ms.space_generator.ScheduleFn(stochastic_schedule_mm)
```
指定了我们需要的随机变换

而Meta-Schedule带有内置通用随机变换集合(set of generic stochastic transformations)，能够适用于广泛的TensorIR计算，这种方法称为自动调度(auto-scheduling)，搜索空间是由系统生成的
删除行 `space=ms.space_generator.ScheduleFn(stochastic_schedule_mm)` 即可运行自动调度

在底层，Meta-Schedule分析每个TensorIR block的数据访问和循环模式，提出对程序的随机变换方式，实际上它们也只是随机转换加上代码分析而已
```python
database = ms.tune_tir(
    mod=MyModule,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    work_dir="./tune_tmp",
    task_name="main",
)
sch = ms.tir_integration.compile_tir(database, MyModule, "llvm --num-cores=1")
```
运行后会发现结果比我们的原始代码快得多，我们可以查看历史轨迹和最终代码，会发现相较于之前较简单的变换，此时的历史轨迹包含：
- 更多级的循环转换
- 中间计算的矢量化
- 并行化和循环展开

### 4.4.2 Section Checkpoint
- 随机调度允许我们表示“可能的变换是什么”
- Meta-Schedule的 `tune_tir` API帮助我们在搜索空间内找到一个好的解决方案
- Meta-Schedule带有一组默认的内置随机变换，涵盖了广泛的搜索空间

## 4.5 Putting Things Back to End to End Model Execution
到目前为止，我们已经了解了自动优化单个元张量函数，进一步，我们将它改进到我们的端到端模型执行
从MLC的角度来看，自动搜索是一个模块化的步骤，我们只需要用调优结果提供的新的元张量函数实现替换原始的元张量函数实现即可

考虑上一章中的两层 MLP 示例，我们这次使用一个混合 IRModule，其中大多数步骤都调用环境函数，同时带有一个TensorIR 函数 `linear0`
```python
@tvm.script.ir_module
class MyModuleMixture:
    @T.prim_func
    def linear0(X: T.Buffer((1, 784), "float32"),
                W: T.Buffer((128, 784), "float32"),
                B: T.Buffer((128,), "float32"),
                Z: T.Buffer((1, 128), "float32")):
        T.func_attr({"global_symbol": "linear0", "tir.noalias": True})
        Y = T.alloc_buffer((1, 128), "float32")
        for i, j, k in T.grid(1, 128, 784):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]

        for i, j in T.grid(1, 128):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] =  Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, 784), "float32"),
             w0: R.Tensor((128, 784), "float32"),
             b0: R.Tensor((128,), "float32"),
             w1: R.Tensor((10, 128), "float32"),
             b1: R.Tensor((10,), "float32")):
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0,), R.Tensor((1, 128), dtype="float32"))
            out = R.call_dps_packed("env.linear", (lv1, w1, b1), R.Tensor((1, 10), dtype="float32"))
            R.output(out)
        return out
```

环境函数注册：
```python
@tvm.register_func("env.linear", override=True)
def torch_linear(x: tvm.nd.NDArray,
                 w: tvm.nd.NDArray,
                 b: tvm.nd.NDArray,
                 out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)
    torch.mm(x_torch, w_torch.T, out=out_torch)
    torch.add(out_torch, b_torch, out=out_torch)

@tvm.register_func("env.relu", override=True)
def lnumpy_relu(x: tvm.nd.NDArray,
                out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    out_torch = torch.from_dlpack(out)
    torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)
```

 我们希望调整 `linear0`，下图总结了我们的整个流程：
![[MLC-Fig6.png]]

而目前调优API只接受带有一个 `main` 函数的IRModule，所以我们首先将 `linear0` 取出到另一个模块的 main函数中
```python
mod_linear = tvm.IRModule.from_expr(MyModuleMixture["linear0"].with_attr("global_symbol", "main"))
```
得到 `mod_linera.script()`
```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(X: T.Buffer((1, 784), "float32"), W: T.Buffer((128, 784), "float32"), B: T.Buffer((128,), "float32"), Z: T.Buffer((1, 128), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        Y = T.alloc_buffer((1, 128))
        for i, j, k in T.grid(1, 128, 784):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(X[vi, vk], W[vj, vk])
                T.writes(Y[vi, vj])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, 128):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(Y[vi, vj], B[vj])
                T.writes(Z[vi, vj])
                Z[vi, vj] = Y[vi, vj] + B[vj]
```
我们将其传递给 `tune_tir` ：
```python
database = ms.tune_tir(
    mod=mod_linear,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    work_dir="./tune_tmp",
    task_name="main",
)
sch = ms.tir_integration.compile_tir(database, mod_linear, "llvm --num-cores=1")
```

现在我们用调优后的新函数替换原来的 `linear0`，我们首先获得一个 `global_var`(一个指向IRModule中函数的 `pointer` 引用)，然后调用 `update_func` 用新的函数替换原本的函数
```python
MyModuleWithParams2 = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
new_func = sch.mod["main"].with_attr("global_symbol", "linear0")
gv = MyModuleWithParams2.get_global_var("linear0")
MyModuleWithParams2.update_func(gv, new_func)
```
得到 `MyModuleWithParams2.script()`
```python
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def linear0(X: T.Buffer((1, 784), "float32"), W: T.Buffer((128, 784), "float32"), B: T.Buffer((128,), "float32"), Z: T.Buffer((1, 128), "float32")):
        T.func_attr({"global_symbol": "linear0", "tir.noalias": True})
        # with T.block("root"):
        Y = T.alloc_buffer((1, 128))
        for i_0, j_0 in T.grid(1, 1):
            for i_1, j_1 in T.grid(1, 8):
                for i_2_init, j_2_init, i_3_init in T.grid(1, 2, 1):
                    for j_3_fused_init in T.vectorized(8):
                        with T.block("Y_init"):
                            vi = T.axis.spatial(1, i_0 + i_1 + i_2_init + i_3_init)
                            vj = T.axis.spatial(128, j_0 * 128 + j_1 * 16 + j_2_init * 8 + j_3_fused_init)
                            T.reads()
                            T.writes(Y[vi, vj])
                            T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            Y[vi, vj] = T.float32(0)
                for k_0, i_2, j_2, k_1, i_3 in T.grid(14, 1, 2, 56, 1):
                    for j_3_fused in T.vectorized(8):
                        with T.block("Y_update"):
                            vi = T.axis.spatial(1, i_0 + i_1 + i_2 + i_3)
                            vj = T.axis.spatial(128, j_0 * 128 + j_1 * 16 + j_2 * 8 + j_3_fused)
                            vk = T.axis.reduce(784, k_0 * 56 + k_1)
                            T.reads(Y[vi, vj], X[vi, vk], W[vj, vk])
                            T.writes(Y[vi, vj])
                            T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
            for ax0, ax1 in T.grid(1, 128):
                with T.block("Z"):
                    vi, vj = T.axis.remap("SS", [ax0, ax1])
                    T.reads(Y[vi, vj], B[vj])
                    T.writes(Z[vi, vj])
                    Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, metadata["relax.expr.Constant"][0], metadata["relax.expr.Constant"][1]), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            out = R.call_dps_packed("env.linear", (lv1, metadata["relax.expr.Constant"][2], metadata["relax.expr.Constant"][3]), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(out)
        return out

# Metadata omitted. Use show_meta=True in script() method to show it.
```
我们可以发现上面代码中的 `linear0` 已经被替换了
再次运行代码，可以发现时间明显减少，这主要归功于新的 `linear0` 函数

## 4.6 Summary
-  随机变换帮助我们指定可能程序的搜索空间
- Meta-Schedule在搜索空间中搜索，并找到优化后的程序
- 我们可以使用变换，将初始的元张量函数替换为优化后的函数，并更新端到端执行流程

# 5 Intergration with Machine Learning Frameworks
## 5.1 Preclude
本章将讨论如何将机器学习模型从现有的机器学习框架引入MLC流程

## 5.2 Build an IRModule Through a Builder
在过去的章节中，我们一直通过直接编写TVMScript来构建IRModule，随着模型变得越来越大，我们需要一种编程方式来构建IRModule

#### 5.2.1 Tensor Expression for TensorIR Creation
张量表达式(Tensor Expression/TE)这一领域特定语言可以用来构建TensorIR函数

我们首先创建一个placeholder对象，它表示TensorIR函数的输入
```python
from tvm import te
A = te.placeholder((128, 128), name="A", dtype="float32")
B = te.placeholder((128, 128), name="B", dtype="float32")
```
得到的 `A` 和 `B` 为 `te.Tensor 对象
每个 `te.Tensor` 对象都有一个 `shape` 字段和 `dtype` 字段，用于记录计算的shape和数据类型，例如 `A.shape` ，`A.dtype`

我们通过一系列张量表达式来描述计算，例如矩阵乘法计算
```python
def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor
	assert A.shape[1] == B.shape[0]
	n = A.shape[0]
	m = B.shape[1]
	k = te.reduce_axis((0, A.shape[1], name="k"))
	return te.compute((n, m), lambda i,j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")
```
`te.compute(output_shape, fcompute)` 中，`output_shape` 指定输出形状，`fcompute` 指定计算函数，计算函数描述了要如何计算给定索引的每个元素 `output[i, j]` 的值

`te_matmul` 函数接受一个 `te.Tensor` 类型的对象，并返回矩阵乘法结果

向 `te_matmul()` 传入 `A,B` 即可获得计算结果
```python
C = te_matmul(A, B)
```

用张量表达式描述了计算后，我们可以调用 `te.create_prim_func` 并传入输入和输出值以创建TensorIR函数
```python
te.create_prim_func([A, B, C]).show()
```
得到
```python
# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), matmul: T.Buffer((128, 128), "float32")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # with T.block("root"):
    for i, j, k in T.grid(128, 128, 128):
        with T.block("matmul"):
            v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
            T.reads(A[v_i, v_k], B[v_k, v_j])
            T.writes(matmul[v_i, v_j])
            with T.init():
                matmul[v_i, v_j] = T.float32(0)
            matmul[v_i, v_j] = matmul[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
```

同样可以为ReLU计算创建张量表达式
```python
def te_relu(A: te.Tensor) -> te.Tensor:
	return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")
```

`te` API允许我们组合运算并创建“融合(fused)”算子，例如，我们可以将 `matmul` 的结果再次应用 `relu`
```python
C = te_matmul(A, B)
D = te_relu(C)
```

可以通过只传递感兴趣的输入和输出值，跳过中间值来创建一个TensorIR函数
而 `matmul` 的结果将被分配为TensorIR函数中的临时空间
```python
te.create_prim_func([A, B, D]).show()
```
得到
```python
# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), relu: T.Buffer((128, 128), "float32")):
    T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    matmul = T.alloc_buffer((128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.block("matmul"):
            v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
            T.reads(A[v_i, v_k], B[v_k, v_j])
            T.writes(matmul[v_i, v_j])
            with T.init():
                matmul[v_i, v_j] = T.float32(0)
            matmul[v_i, v_j] = matmul[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
    for i0, i1 in T.grid(128, 128):
        with T.block("relu"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(matmul[v_i0, v_i1])
            T.writes(relu[v_i0, v_i1])
            relu[v_i0, v_i1] = T.max(matmul[v_i0, v_i1], T.float32(0))
```
我们还可以将中间结果 `C` 传递到参数列表中，在这种情况下，TensorIR函数会希望我们也从调用方传入 `C`，但通常建议只传入输入和输出即可

#### 5.2.2 Use BlockBuilder to Create an IRModule
到目前为止，我们已经创建了一个TensorIR函数，为了构建端到端的模型执行，我们还需要能够通过计算图连接多个TensorIR函数

我们通过创建block builder和一系列元张量函数来构造Relax函数
```python
A = relax.Var("A", relax.TensorStructInfo((128, 128), "float32"))
B = relax.Var("B", relax.TensorStructInfo((128, 128), "float32"))

bb = relax.BlockBuilder()

with bb.function("main"):
	with bb.dataflow():
		C = bb.emit_te(te_matmul, A, B)
        D = bb.emit_te(te_relu, C)
        R = bb.emit_output(D)
    bb.emit_func_output(R, params=[A, B])

MyModule = bb.get()
MyModule.show()
```
得到
```python
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def te_matmul(rxplaceholder: T.Buffer((T.int64(128), T.int64(128)), "float32"), rxplaceholder_1: T.Buffer((T.int64(128), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for i, j, k in T.grid(T.int64(128), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(rxplaceholder[v_i, v_k], rxplaceholder_1[v_k, v_j])
                T.writes(matmul[v_i, v_j])
                with T.init():
                    matmul[v_i, v_j] = T.float32(0)
                matmul[v_i, v_j] = matmul[v_i, v_j] + rxplaceholder[v_i, v_k] * rxplaceholder_1[v_k, v_j]

    @T.prim_func
    def te_relu(rxplaceholder: T.Buffer((T.int64(128), T.int64(128)), "float32"), relu: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(128), T.int64(128)):
            with T.block("relu"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(rxplaceholder[v_i0, v_i1])
                T.writes(relu[v_i0, v_i1])
                relu[v_i0, v_i1] = T.max(rxplaceholder[v_i0, v_i1], T.float32(0))

    @R.function
    def main(A: R.Tensor((128, 128), dtype="float32"), B: R.Tensor((128, 128), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.te_matmul, (A, B), out_sinfo=R.Tensor((128, 128), dtype="float32"))
            lv1 = R.call_tir(cls.te_relu, (lv,), out_sinfo=R.Tensor((128, 128), dtype="float32"))
            gv: R.Tensor((128, 128), dtype="float32") = lv1
            R.output(gv)
        return gv
```

#### 5.2.3 Deep Dive into Block Builder APIs
BlockBuilder代码和生成的IRModule：
![[MLC-Fig7.png]]
BlockBuilder与Relax函数中存在相对应的作用域
例如，`bb.dataflow()` 创建一个dataflow block，`bb.dataflow()` 中所有的BlockBuilder调用在Relax函数相对应的调用在Relax函数中也会处在dataflow block的作用域中
```python
with bb.function("main"):
    with bb.dataflow():
        # every emit call generates a variable inside a dataflow block.
```

BlockBuilder中每个结果都是一个 `relax.Var`，对应Relax函数中一个存储计算结果的变量
```python
isinstance(C, relax.Var)
> True
type(C)
> tvm.relax.expr.DataflowVar
```
(`DataflowVar` 表明了该变量是dataflow block和计算图内的中间结果)

Relax函数中的每一行都是由 `emit_te` 调用生成的，例如
```python
lv = R.call_dps_packed(te_matmul, (A, B), (128, 128), dtype="float32")
```
由
```python
C = bb.emit_te(te_matmul, A, B).
```
生成

在幕后，`bb.emit_te` 做了以下事情：
- 为A和B创建一个输入 `te.placeholder`
- 通过 `te_matmul` 函数运行它们
- 调用 `te.create_prim_func` 来创建一个TensorIR函数
- 通过 `call_dps_packed` 生成对TensorIR函数的调用

我们通过 `bb.emit_output` 创建每个dataflow block的输出变量
```python
with bb.dataflow():
    ...
    R = bb.emit_output(D)
```
上面的代码标志着 `D` 是一个可以在dataflow block之外引用的变量

最后，函数输出由 `bb.emit_func_output` 标记，我们只能在每个函数作用域(function scope)内调用一次 `emit_func_output`

值得注意的是，我们可以在输出阶段指定函数的参数列表
```python
with bb.function("main"):
    ...
    # specify parameters in the end
    bb.emit_func_output(R, params=[A, B])
```
或者我们也可以在函数范围的开头指定参数列表
```python
# specify parameters in the beginning.
with bb.function("main", params=[A, B]):
    ...
    bb.emit_func_output(R)
```

## 5.3 Import Model From PyTroch
大多数机器学习框架都带有计算图抽象，其中每个节点对应一个操作，边对应它们之间的依赖关系
我们将采用PyTorch模型，获取PyTorch原生格式的计算图，并将其转换为 IRModule

我们首先在PyTorch中定义一个模型
```python
class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		self.weight = nn.Parameter(torch.randn(128, 128))

	def forward(self, x):
		x = torch.matmul(x, self.weight)
		x = torch.relu(x)
		return x
```

### 5.3.1 Create TorchFX GraphModule
我们使用TorchFX来表示来自PyTorch的模型的计算图
```python
model = MyModel()
fx_module = fx.symbolic_trace(model)
type(fx_module)

> torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl
```
`fx_module` 包含一个简单的计算图，可以打印成表格便于查看
```python
fx_module.graph.print_tabular()
```
![[MLC_Fig7.png]]
我们的目标是将此图转换为IRModule

### 5.3.2 Create Map Function
整体的翻译逻辑的主要流程如下：
- 创建一个 `node_map`，将 `fx.Node` 映射到相应的 `relax.Var`，该 `relax.Var` 代表IRModule中的已翻译节点
- 以拓扑顺序迭代FX图中的节点
- 给定映射输入(mapped inputs)，获取节点的映射输出(mapped outputs)
```python
def map_param(param: nn.Parameter):
    return relax.const(
        param.data.cpu().numpy(), relax.TensorStructInfo(param.data.shape, "float32")
    )

def fetch_attr(fx_mod, target: str):
    """Helper function to fetch an attr"""
    target_atoms = target.split('.')
    attr_itr = fx_mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr

def from_fx(fx_mod, input_shapes, call_function_map, call_module_map):
    input_index = 0
    node_map = {}
    named_modules = dict(fx_mod.named_modules())

    bb = relax.BlockBuilder()

    fn_inputs = []
    fn_output = None
    with bb.function("main"):
        with bb.dataflow():
            for node in fx_mod.graph.nodes:
                if node.op == "placeholder":
                    # create input placeholder
                    shape = input_shapes[input_index]
                    input_index += 1
                    input_var = relax.Var(
                        node.target, relax.TensorStructInfo(shape, "float32")
                    )
                    fn_inputs.append(input_var)
                    node_map[node] = input_var
                elif node.op == "get_attr":
                    node_map[node] = map_param(fetch_attr(fx_mod, node.target))
                elif node.op == "call_function":
                    node_map[node] = call_function_map[node.target](bb, node_map, node)
                elif node.op == "call_module":
                    named_module = named_modules[node.target]
                    node_map[node] = call_module_map[type(named_module)](bb, node_map, node, named_module)
                elif node.op == "output":
                    output = node_map[node.args[0]]
                    assert fn_output is None
                    fn_output = bb.emit_output(output)
        # output and finalize the function
        bb.emit_func_output(output, fn_inputs)
    return bb.get()
```

函数映射(function map)函数定义在外部，为torch function提供翻译规则，然后通过参数传入 `from_fx` 
以下代码块显示了我们如何通过 `emit_te` API做到这一点
```python
def map_matmul(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    B = node_map[node.args[1]]
    return bb.emit_te(te_matmul, A, B)

def map_relu(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    return bb.emit_te(te_relu, A)

MyModule = from_fx(
    fx_module,
    input_shapes = [(1, 128)],
    call_function_map = {
      torch.matmul: map_matmul,
      torch.relu: map_relu,
    },
    call_module_map={},
)
```

## 5.4 Summary
- 张量表达式API允许我们创建原始的TensorIR函数
- BlockBuilder API通过 `emit_te` 和其他函数创建IRModule
- 通过将模型转换为IRModule，可以实现MLC与现有的机器学习框架的整合

# 6 GPU and Hardware Acceleartion
## Part1
### 6.1 GPU Architecture
典型的GPU包含一组流处理器(stream multi-processors/SM)，每个流处理器都有许多核心，GPU设备是大规模并行的，允许我们同时执行许多任务
![[MLC-Fig8.png]]
要对GPU进行编程，我们需要创建一组线程块(thread blocks)，每个 thread 映射到单个核心，而每个block映射到单个流式多处理器(SM)
![[MLC-Fig9.png]]

让我们使用向量相加示例开始GPU编程，以下 TensorIR 程序采用两个向量 `A` 和 `B`，执行元素相加，并将结果存储在 `C` 中
```python
@tvm.sciprt.ir_module
class MyModuleVecAdd:
	@T.prim_func
	def main(A: T.Buffer((1024,), "float32"),
             B: T.Buffer((1024,), "float32"),
             C: T.Buffer((1024,), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.grid(1024):
	        with T.block("C"):
		        vi = T.axis.remap("S", [i])
		        C[vi] = A[vi] + B[vi]
```
我们首先将循环 `i` 拆分成两个循环
```python
sch = tvm.tir.Schedule(MyModuleVecAdd)
block_C = sch.get_block("C")
i, = sch.get_loops(block=block_C)
i0, i1 = sch.split(i, [None, 128])
sch.mod.show()
```
得到
```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024,), "float32"), B: T.Buffer((1024,), "float32"), C: T.Buffer((1024,), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # with T.block("root"):
        for i_0, i_1 in T.grid(8, 128):
            with T.block("C"):
                vi = T.axis.spatial(1024, i_0 * 128 + i_1)
                T.reads(A[vi], B[vi])
                T.writes(C[vi])
                C[vi] = A[vi] + B[vi]
```
#### 6.1.1 GPU Thread Blocks
然后我们将迭代器绑定到GPU线程块，每个线程由 `threadIdx.x` 和 `blockIdx.x` 两个索引进行表示 
在实际应用中，我们可以有多维线程索引，但这里我们为了简化问题，将它们固定为一维表示
```python
sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")
sch.mod.show()
```
得到
```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024,), "float32"), B: T.Buffer((1024,), "float32"), C: T.Buffer((1024,), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # with T.block("root"):
        for i_0 in T.thread_binding(8, thread="blockIdx.x"):
            for i_1 in T.thread_binding(128, thread="threadIdx.x"):
                with T.block("C"):
                    vi = T.axis.spatial(1024, i_0 * 128 + i_1)
                    T.reads(A[vi], B[vi])
                    T.writes(C[vi])
                    C[vi] = A[vi] + B[vi]
```

#### 6.1.2 Build and Run the TensorIR Function on GPU
我们可以在GPU上构建和测试生成的函数
```python
rt_mod = tvm.build(sch.mod, target="cuda")

A_np = np.random.uniform(size=(1024,)).astype("float32")
B_np = np.random.uniform(size=(1024,)).astype("float32")
A_nd = tvm.nd.array(A_np, tvm.cuda(0))
B_nd = tvm.nd.array(B_np, tvm.cuda(0))
C_nd = tvm.nd.array(np.zeros((1024,), dtype="float32"), tvm.cuda(0))

rt_mod["main"](A_nd, B_nd, C_nd)
print(A_nd)
print(B_nd)
print(C_nd)
```

### 6.2 Example: Window Sum
现在，让我们继续看另一个例子——窗口总和
这个程序可以被视为具有预定义权重 `[1,1,1]` 的“卷积“的基本版本，我们对输入进行滑动并将三个相邻值相加
![[MLC-Fig10.png]]
```python
@tvm.script.ir_module
class MyModuleWindowSum:
    @T.prim_func
    def main(A: T.Buffer[(1027,), "float32"],
             B: T.Buffer[(1024,), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.grid(1024):
            with T.block("C"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi] + A[vi + 1] + A[vi + 2]
```
首先，我们可以将循环绑定到GPU线程
```python
sch = tvm.tir.Schedule(MyModuleWindowSum)
nthread = 128
block_C = sch.get_block("C")
i,  = sch.get_loops(block=block_C)
i0, i1 = sch.split(i, [None, nthread])
sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")
sch.mod.show()
```
得到
```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1027,), "float32"), B: T.Buffer((1024,), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # with T.block("root"):
        for i_0 in T.thread_binding(8, thread="blockIdx.x"):
            for i_1 in T.thread_binding(128, thread="threadIdx.x"):
                with T.block("C"):
                    vi = T.axis.spatial(1024, i_0 * 128 + i_1)
                    T.reads(A[vi:vi + 3])
                    T.writes(B[vi])
                    B[vi] = A[vi] + A[vi + 1] + A[vi + 2]
```
在这种情况下，有数据复用的机会(reuse opportunities)，每个GPU线程块都包含所有线程都可以在块内访问的共享内存(shared memory)
我们使用`cache_read`添加一个中间阶段，将部分数据缓存到共享内存上，缓存完成后，线程可以从共享内存中读取数据
```python
A_shared = sch.cache_read(block_C, read_buffer_index=0, storage_scope="shared")
sch.compute_at(A_shared, i1)
sch.mod.show()
```
得到
```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1027,), "float32"), B: T.Buffer((1024,), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # with T.block("root"):
        A_shared = T.alloc_buffer((1027,), scope="shared")
        for i_0 in T.thread_binding(8, thread="blockIdx.x"):
            for i_1 in T.thread_binding(128, thread="threadIdx.x"):
                for ax0 in range(130):
                    with T.block("A_shared"):
                        v0 = T.axis.spatial(1027, i_0 * 128 + ax0)
                        T.reads(A[v0])
                        T.writes(A_shared[v0])
                        A_shared[v0] = A[v0]
                with T.block("C"):
                    vi = T.axis.spatial(1024, i_0 * 128 + i_1)
                    T.reads(A_shared[vi:vi + 3])
                    T.writes(B[vi])
                    B[vi] = A_shared[vi] + A_shared[vi + 1] + A_shared[vi + 2]
```
因为内存是跨线程共享的，所以我们需要重新拆分循环并将获取过程(fetching process)的内部迭代器绑定到线程索引上，这种技术称为cooperative fetching，即多个线程一起工作以将数据带到共享内存中
```python
ax = sch.get_loops(A_shared)[-1]
ax0, ax1 = sch.split(ax, [None, nthread])
sch.bind(ax1, "threadIdx.x")
sch.mod.show()
```
得到
```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1027,), "float32"), B: T.Buffer((1024,), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_shared = T.alloc_buffer((1027,), scope="shared")
        for i_0 in T.thread_binding(8, thread="blockIdx.x"):
            for i_1 in T.thread_binding(128, thread="threadIdx.x"):
                for ax0_0 in range(2):
                    for ax0_1 in T.thread_binding(128, thread="threadIdx.x"):
                        with T.block("A_shared"):
                            v0 = T.axis.spatial(1027, i_0 * 128 + (ax0_0 * 128 + ax0_1))
                            T.where(ax0_0 * 128 + ax0_1 < 130)
                            T.reads(A[v0])
                            T.writes(A_shared[v0])
                            A_shared[v0] = A[v0]
                with T.block("C"):
                    vi = T.axis.spatial(1024, i_0 * 128 + i_1)
                    T.reads(A_shared[vi:vi + 3])
                    T.writes(B[vi])
                    B[vi] = A_shared[vi] + A_shared[vi + 1] + A_shared[vi + 2]
```
我们可以检查相应的底层代码(CUDA)中，生成的代码包含两部分：
- 在主机(CPU)上的调用GPU程序的部分
- 执行相应计算的CUDA内核
我们可以使用以下代码打印出 CUDA 内核，我们仍然需要主机和内核代码来运行程序，因此它只是一种快速检查最终代码生成结果的方法。
值得注意的是，构建过程会自动压缩共享内存阶段以使用线程块中使用的最小区域
```python
rt_mod = tvm.build(sch.mod, target="cuda")
print(rt_mod.imported_modules[0].get_source())
```

### 6.3 Matrix Multiplication
现在让我们来处理一些稍微复杂的事情，并尝试在GPU上优化矩阵乘法，我们将介绍两种用于GPU性能优化的常用技术
```python
@tvm.script.ir_module
class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"),
             B: T.Buffer((1024, 1024), "float32"),
             C: T.Buffer((1024, 1024), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

#### 6.3.1 Local Blocking
![[MLC-Fig11.png]]
我们可以进行循环拆分(tile the loops)，来增加整体内存复用
我们引入了局部切分(local tiles)，这样我们只需要从 `A` 和 `B` 加载一次条形数据(上图中的灰色部分)，然后使用它们来计算$V*V$的矩阵乘法结果

这种本地存储的切分有助于减少内存压力，因为条形数据块的每个元素都被重用了 `V` 次
```python
def blocking(sch,
             tile_local_y,
             tile_local_x,
             tile_block_y,
             tile_block_x,
             tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    sch.decompose_reduction(block_C, k0)

    return sch

sch = tvm.tir.Schedule(MyModuleMatmul)
sch = blocking(sch, 8, 8, 8, 8, 4)
sch.mod.show()
```
```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1 in T.thread_binding(8, thread="threadIdx.y"):
                    for j_1 in T.thread_binding(8, thread="threadIdx.x"):
                        for i_2_init, j_2_init in T.grid(8, 8):
                            with T.block("C_init"):
                                vi = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2_init)
                                vj = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + j_2_init)
                                T.reads()
                                T.writes(C_local[vi, vj])
                                C_local[vi, vj] = T.float32(0)
                        for k_0 in range(256):
                            for k_1 in T.unroll(4):
                                for i_2, j_2 in T.grid(8, 8):
                                    with T.block("C_update"):
                                        vi = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2)
                                        vj = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + j_2)
                                        vk = T.axis.reduce(1024, k_0 * 4 + k_1)
                                        T.reads(C_local[vi, vj], A[vi, vk], B[vk, vj])
                                        T.writes(C_local[vi, vj])
                                        C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
                        for ax0, ax1 in T.grid(8, 8):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + ax0)
                                v1 = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + ax1)
                                T.reads(C_local[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_local[v0, v1]
```
```python
rt_mod = tvm.build(sch.mod, target="cuda")
dev = tvm.cuda(0)
A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
B_np = np.random.uniform(size=(1024, 1024)).astype("float32")
A_nd = tvm.nd.array(A_np, dev)
B_nd = tvm.nd.array(B_np, dev)
C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)

num_flop = 2 * 1024 * 1024 * 1024
evaluator = rt_mod.time_evaluator("main", dev, number=10)

print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
```

#### 6.3.2 Shared Memory Blocking
![[MLC-Fig12.png]]
我们的第一次尝试没有考虑位于同一个 GPU 线程块中的相邻线程，我们可以将它们需要的数据加载到一块共享内存(shared memory)

### 6.4 Leveraging Automatic Program Optimization
到目前为止，我们一直在手动编写变换来优化 GPU 上的 TensorIR 程序。我们可以利用自动程序优化框架来调整相同的程序
```python
from tvm import meta_schedule as ms

database = ms.tune_tir(
    mod=MyModuleMatmul,
    target="nvidia/tesla-p100",
    max_trials_global=64,
    num_trials_per_iter=64,
    work_dir="./tune_tmp",
    task_name="main"
)
sch = ms.tir_integration.compile_tir(database, MyModuleMatmul, "nvidia/tesla-p100")
sch.mod.show()
```

```python
rt_mod = tvm.build(sch.mod, target="nvidia/tesla-p100")
dev = tvm.cuda(0)
evaluator = rt_mod.time_evaluator("main", dev, number=10)

print("MetaSchedule: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
```

### 6.5 Summary
本章研究 MLC 的另一个维度，即我们如何变换我们的程序以实现硬件加速
MLC 过程帮助我们将输入模型连接到不同的GPU编程模型和环境
- 典型的 GPU 包含两级层次结构。 每个线程由(在 CUDA 术语中) `threadIdx.x` 和 `blockIdx.x` 索引(也可以有多个维度索引，但它们可以融合为一个)
- 共享内存有助于缓存同一块内的线程中常用的数据
- 在 GPU 优化期间鼓励内存重用

## Part 2
### 6.6 Hardware Specialization Trend
机器学习硬件领域最近一个新兴的主题是专业化
传统上，我们在通用标量处理器(generic scalar processors)上构建我们的解决方案：一次在一个浮点数上执行操作
AVX和ARM/Neon等向量指令集提供了加速程序的有效方法，但也给我们编写程序的方式带来了一些复杂性

最新的机器学习加速器引入了用于张量计算的专用单元，以及用于多维数据复制和矩阵/张量计算的专用指令

#### 6.6.1 Key Elements of Specialized Code
为了帮助我们更好地理解专业硬件编程的元素，让我们首先研究以下low-level NumPy代码，虽然这段代码仍然在 Python 中运行，但它类似于一组可能发生在专用硬件后端的操作
```python
def accel_fill_zero(C):
    C[:] = 0

def accel_tmm_add(C, A, B):
    C[:] += A @ B.T

def accel_dma_copy(reg, dram):
    reg[:] = dram[:]

def lnumpy_tmm(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    # a special accumulator memory
    C_accumulator = np.empty((16, 16), dtype="float32")
    A_reg = np.empty((16, 16), dtype="float32")
    B_reg = np.empty((16, 16), dtype="float32")

    for i in range(64):
        for j in range(64):
            accel_fill_zero(C_accumulator[:,:])
            for k in range(64):
                accel_dma_copy(A_reg[:], A[i * 16 : i * 16 + 16, k * 16 : k * 16 + 16])
                accel_dma_copy(B_reg[:], B[j * 16 : j * 16 + 16, k * 16 : k * 16 + 16])
                accel_tmm_add(C_accumulator[:,:], A_reg, B_reg)
            accel_dma_copy(C[i * 16 : i * 16 + 16, j * 16 : j * 16 + 16], C_accumulator[:,:])
```
![[MLC-Fig13.png]]
上面的低级 NumPy 程序包含以下关键元素：
- 计算的基本单位是$16\times 16$矩阵乘法 (`accel_tmm_add`)
- `accel_tmm_add` 接受两个输入 —— `A_reg` 和 `B_reg` 并累加到累加器内存中
- 使用特殊功能 (`accel_dma_copy`) 执行数据复制

在现实世界的硬件后端中，我们通常期望 `A_reg`、`B_reg` 和 `C_accumulator` 映射到硬件中的特殊内存区域(或寄存器)，即特殊内存层级(special memory scopes) 
此外，我们可以对这些设置执行一组有限的硬件加速操作，诸如 `accel_tmm_add` 之类的操作可以映射到真正的硬件指令或供应商提供的高效内核函数实现

#### 6.6.2 A Block with Tensorized Computation
到目前为止，我们运行的大多数TensorIR码都包含一个 block，用于计算输出张量中的单个元素，专用加速器代码不是以标量计算为单位构建的，许多专门的加速器在张量区域上运行计算，TensorIR中的block结构帮助我们对此类相关计算进行分组
```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # with T.block("root"):
        for i0, j0, k0 in T.grid(64, 64, 64):
            with T.block("tmm-16x16"):
                vi0, vj0, vk0 = T.axis.remap("SSR", [i0, j0, k0])
                T.reads(A[vi0 * 16:vi0 * 16 + 16, vk0 * 16:vk0 * 16 + 16], B[vj0 * 16:vj0 * 16 + 16, vk0 * 16:vk0 * 16 + 16])
                T.writes(C[vi0 * 16:vi0 * 16 + 16, vj0 * 16:vj0 * 16 + 16])
                with T.init():
                    for i1, j1 in T.grid(16, 16):
                        with T.block("tmm_init"):
                            vi1, vj1 = T.axis.remap("SS", [i1, j1])
                            T.reads()
                            T.writes(C[vi0 * 16 + vi1, vj0 * 16 + vj1])
                            C[vi0 * 16 + vi1, vj0 * 16 + vj1] = T.float32(0)
                for i1, j1, k1 in T.grid(16, 16, 16):
                    with T.block("tmm"):
                        vi1, vj1, vk1 = T.axis.remap("SSR", [i1, j1, k1])
                        T.reads(C[vi0 * 16 + vi1, vj0 * 16 + vj1], A[vi0 * 16 + vi1, vk0 * 16 + vk1], B[vj0 * 16 + vj1, vk0 * 16 + vk1])
                        T.writes(C[vi0 * 16 + vi1, vj0 * 16 + vj1])
                        C[vi0 * 16 + vi1, vj0 * 16 + vj1] = C[vi0 * 16 + vi1, vj0 * 16 + vj1] + A[vi0 * 16 + vi1, vk0 * 16 + vk1] * B[vj0 * 16 + vj1, vk0 * 16 + vk1]
```
我们进一步观察下面这个block：
```python
with T.block("tmm-16x16"):
    T.reads(A[vi0 * 16 : vi0 * 16 + 16, vk0 * 16 : vk0 * 16 + 16], B[vj0 * 16 : vj0 * 16 + 16, vk0 * 16 : vk0 * 16 + 16])
    T.writes(C[vi0 * 16 : vi0 * 16 + 16, vj0 * 16 : vj0 * 16 + 16])
    ...
```
这个block从 `A` 和 `B` 的$16\times16$区域读取，并写入 `C` 的$16\times 16$区域
在这种情况下，block的内容包含有关子区域计算的特定实现的更多细节，我们将此 block称为张量化block，因为它们包含在张量子区域上的计算

#### 6.6.3 Transforming Loops Around Tensorized Block
我们在这里可以做的一件事是变换张量计算block周围的循环，这些循环变换可以帮助我们重新组织周围的迭代方式，从而使得不同张量程序变体成为可能
```python
sch = tvm.tir.Schedule(MatmulBlockModule)

block_mm = sch.get_block("tmm-16x16")
i, j, k = sch.get_loops(block_mm)

i0, i1 = sch.split(i, [None, 4])

sch.reorder(i0, j, i1, k)
sch.mod.show()
```

#### 6.6.4 Blockization – Creating Tensorized Blocks
在大多数情况下，我们从带有标量计算的循环开始，TensorIR提供了一种变换原语 blockization来将循环的子区域组合在一起以形成张量化的计算 block
```python
@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
        A: T.Buffer((1024, 1024), "float32"),
        B: T.Buffer((1024, 1024), "float32"),
        C: T.Buffer((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] += A[vi, vk] * B[vj, vk]
```
```python
sch = tvm.tir.Schedule(MatmulModule)
i, j, k = sch.get_loops("matmul")
i, ii = sch.split(i, factors=[None, 16])
j, ji = sch.split(j, factors=[None, 16])
k, ki = sch.split(k, factors=[None, 16])
sch.reorder(i, j, k, ii, ji, ki)
sch.mod.show()
```
```python
block_mm = sch.blockize(ii)
sch.mod.show()
```

#### 6.6.5 Transforming TensorIR to Introduce Special Memory Scope
正如我们在低级 NumPy 代码中所指出的，之前低级TensorIR的一个关键元素是加速期间使用的特殊内存层级
我们可以使用 `cache_read` 和 `cache_write` 来创建中间内存阶段
```python
A_reg = sch.cache_read(block_mm, 0, storage_scope="global.A_reg")
B_reg = sch.cache_read(block_mm, 1, storage_scope="global.B_reg")
sch.compute_at(A_reg, k)
sch.compute_at(B_reg, k)

write_back_block = sch.cache_write(block_mm, 0, storage_scope="global.accumulator")
sch.reverse_compute_at(write_back_block, j)
sch.mod.show()
```
这里 `global.A_reg` 包含两个部分
`global` 表示所有线程都可以全局访问内存，而 `A_reg` 是内存的层级标签(scope tag)，为后续编译映射到寄存器等特殊区域提供了机会

### 6.7 Tensorization
现在我们已经创建了一组映射到TensorIR中相应计算阶段的 block，剩下的步骤是映射一些block以使用映射到硬件加速指令的特定实现。 此映射过程称为张量化
为了准备张量化，我们首先注册一个张量intrinsic(TensorIntrin)，其中包含计算和实现的描述
系统将使用描述找到与计算匹配的相关区域，而实现将计算映射到加速硬件指令

作为准备步骤，我们首先将归约分解为一个初始化block和一个更新block
```python
sch.decompose_reduction(block_mm, k)
sch.mod.show()
```

然后我们可以调用 `tensorize`，将 `block_mm` (对应于 `matmul_o_update` block) 映射到使用 `tmm16` 的实现
```python
sch.tensorize(block_mm, "tmm16")
```

这里我们使用 `T.call_extern` 来调用环境中的外部函数，下游编译步骤可以轻松地将实现映射到实现操作的指令

### 6.8 Summary
- 硬件专业化向张量计算的总体趋势
- 带有张量化block的TensorIR变换
- 张量化：将循环计算 block 映射到专门实现的过程


# 7 Computational Graph Optimization
## 7.1 Pattern Match and Rewriting
首先，让我们从以下示例开始
```python
@tvm.script.ir_module
class MyModule:
    @R.function
    def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
        with R.dataflow():
            lv0 = relax.op.multiply(x, y)
            gv0 = relax.op.add(lv0, y)
            R.output(gv0)
        return gv0
```
`MyModule` 包含一个带有两个图层次算子的 relax 函数，其中包含 `relax.op.multiply` 和`relax.op.add`
我们的目标是找到这两个运算符并将它们替换为对一个 `relax.op.ewise_fma` 运算符的调用

在我们研究如何准确地做到这一点之前，让我们首先检查构成 `MyModule` 的数据结构： 每个 `IRModule` 都包含一组函数，函数体由一组称为抽象语法树(AST)的数据结构组成
```python
relax_func = MyModule["main"]
```

每个 relax 函数都由一个 `elax.expr.Function` 节点表示
```python
type(relax_func)
> tvm.relax.expr.Function
```
函数包含一系列参数
```python
relax_func.params
> [x, y]
```
函数包含一个返回值表达式 
```python
func_body = relax_func.body
type(func_body)
> tvm.relax.expr.SeqExpr
```
函数主体 `SeqExpr` 包含一系列 binding blocks
```python
func_body.blocks

> [x: R.Tensor((3, 4), dtype="float32")
  y: R.Tensor((3, 4), dtype="float32")
  with R.dataflow():
	  lv0: R.Tensor((3, 4), dtype="float32") = R.multiply(x, y)
	  gv0: R.Tensor((3, 4), dtype="float32") = R.add(lv0, y)
	  R.output(gv0)]
```
每个binding blocks包含一系列binding，在我们的特定情况下，我们有一个数据流块，其中包含两个 binding
```python
dataflow_block = func_body.blocks[0]
dataflow_block.bindings
> [x: R.Tensor((3, 4), dtype="float32")
y: R.Tensor((3, 4), dtype="float32")
lv0: R.Tensor((3, 4), dtype="float32") = R.multiply(x, y), lv0: R.Tensor((3, 4), dtype="float32")
y: R.Tensor((3, 4), dtype="float32")
gv0: R.Tensor((3, 4), dtype="float32") = R.add(lv0, y)]
```
对应于
```python
lv0 = relax.op.multiply(x, y)
gv0 = relax.op.add(lv0, y)
```
每个 binding 都有一个对应于 binding 左侧的 var (`lv0`、`gv0`)
```python
binding = dataflow_block.bindings[0]
binding.var
> lv0
```
并且每个 binding 的右侧是他的 value，每个 value 对应一个 `relax.Call` 节点，表示对元函数的调用
```python
binding.value
> R.multiply(x, y)
```
![[MLC-Fig14.png]]

改写程序可以通过递归遍历 MyModule 的 AST ，并生成转换后的 AST 来实现
我们当然可以直接使用构建AST的 python API 来做到这一点，但是，我们可以使用额外的工具支持来简化流程，下面的代码块遵循一种称为访问者模式(visitor pattern)的设计模式，它允许我们访问每个 AST 节点并将它们重写为转换后的版本
```python
@relax.expr_functor.mutator
class EwiseFMARewriter(relax.PyExprMutator):
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        add_op = tvm.ir.Op.get("relax.add")
        multiply_op = tvm.ir.Op.get("relax.multiply")
        ewise_fma_op = tvm.ir.Op.get("relax.ewise_fma")

        if call.op != add_op:
            return call

        value = self.lookup_binding(call.args[0])
        if not isinstance(value, relax.Call) or value.op != multiply_op:
            return call

        fma_call = relax.Call(
            ewise_fma_op, [value.args[0], value.args[1], call.args[1]], None, None
        )
        return fma_call

updated_fn = EwiseFMARewriter().visit_expr(MyModule["main"])
updated_fn.show()
```
```python
# from tvm.script import relax as R

@R.function
def main(x: R.Tensor((3, 4), dtype="float32"), y: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((3, 4), dtype="float32"):
    with R.dataflow():
        lv0: R.Tensor((3, 4), dtype="float32") = R.multiply(x, y)
        gv0: R.Tensor((3, 4), dtype="float32") = R.ewise_fma(x, y, y)
        R.output(gv0)
    return gv0
```
请注意，结果将 `gv0` 重写为融合运算符，但将 `lv0` 留在代码中。 我们可以使用 `remove_all_unused` 来进一步简化代码块
```python
relax.analysis.remove_all_unused(updated_fn).show()
```

```python
# from tvm.script import relax as R

@R.function
def main(x: R.Tensor((3, 4), dtype="float32"), y: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((3, 4), dtype="float32"):
    with R.dataflow():
        gv0: R.Tensor((3, 4), dtype="float32") = R.ewise_fma(x, y, y)
        R.output(gv0)
    return gv0
```

## 7.2 Fuse Linear and ReLU
现在我们对计算图改写有了基本的了解，让我们在端到端模型上进行尝试
以下代码重新构建了我们在过去章节中使用的FashionMNIST MLP模型，为了简化过程，我们直接使用高级运算符构建模型，例如 `relax.op.add` 和 `relax.op.matmul`
```python
def create_model():
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((1, 784), "float32"))
    w0 = relax.const(mlp_params["w0"], "float32")
    b0 = relax.const(mlp_params["b0"], "float32")
    w1 = relax.const(mlp_params["w1"], "float32")
    b1 = relax.const(mlp_params["b1"], "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.matmul(x, relax.op.permute_dims(w0)))
            lv1 = bb.emit(relax.op.add(lv0, b0))
            lv2 = bb.emit(relax.op.nn.relu(lv1))
            lv3 = bb.emit(relax.op.matmul(lv2, relax.op.permute_dims(w1)))
            lv4 = bb.emit(relax.op.add(lv3, b1))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)

    return bb.get()

MLPModel = create_model()
MLPModel.show()
```

```python
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][0], axes=None)
            lv1: R.Tensor((1, 128), dtype="float32") = R.matmul(x, lv, out_dtype="void")
            lv2: R.Tensor((1, 128), dtype="float32") = R.add(lv1, metadata["relax.expr.Constant"][1])
            lv3: R.Tensor((1, 128), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][2], axes=None)
            lv5: R.Tensor((1, 10), dtype="float32") = R.matmul(lv3, lv4, out_dtype="void")
            lv6: R.Tensor((1, 10), dtype="float32") = R.add(lv5, metadata["relax.expr.Constant"][3])
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return gv

# Metadata omitted. Use show_meta=True in script() method to show it.
```

我们的目标是“融合” `matmul` 和 `add` 算子到一起，以下代码通过以下步骤实现：
- 识别 `matmul` 和 `add` 算子
- 生成另一个调用 `matmul` 和 `add` 算子的子函数
- 将 `matmul` 和 `add` 替换为融合后的子函数
```python
@relax.expr_functor.mutator
class MatmulAddFusor(relax.PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__()
        self.mod_ = mod
        # cache pre-defined ops
        self.add_op = tvm.ir.Op.get("relax.add")
        self.matmul_op = tvm.ir.Op.get("relax.matmul")
        self.counter = 0

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            # avoid already fused primitive functions
            if func.attrs is not None and "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
                continue
            updated_func = self.visit_expr(func)
            updated_func = relax.analysis.remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        def match_call(node, op):
            if not isinstance(node, relax.Call):
                return False
            return node.op == op

        # pattern match matmul => add
        if not match_call(call, self.add_op):
            return call

        value = self.lookup_binding(call.args[0])
        if value is None:
            return call

        if not match_call(value, self.matmul_op):
            return call

        x = value.args[0]
        w = value.args[1]
        b = call.args[1]

        # construct a new fused primitive function
        param_x = relax.Var("x" ,relax.TensorStructInfo(x.struct_info.shape, x.struct_info.dtype))
        param_w = relax.Var("w" ,relax.TensorStructInfo(w.struct_info.shape, w.struct_info.dtype))
        param_b = relax.Var("b" ,relax.TensorStructInfo(b.struct_info.shape, b.struct_info.dtype))

        bb = relax.BlockBuilder()

        fn_name = "fused_matmul_add%d" % (self.counter)
        self.counter += 1
        with bb.function(fn_name, [param_x, param_w, param_b]):
            with bb.dataflow():
                lv0 = bb.emit(relax.op.matmul(param_x, param_w))
                gv = bb.emit_output(relax.op.add(lv0, param_b))
            bb.emit_func_output(gv)

        # Add Primitive attribute to the fused funtions
        fused_fn = bb.get()[fn_name].with_attr("Primitive", 1)
        global_var = self.builder_.add_func(fused_fn, fn_name)

        # construct call into the fused function
        return relax.Call(global_var, [x, w, b], None, None)

@tvm.ir.transform.module_pass(opt_level=2, name="MatmulAddFuse")
class FuseDenseAddPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return MatmulAddFusor(mod).transform()

MLPFused = FuseDenseAddPass()(MLPModel)
MLPFused.show()
```

## 7.3 Map to TensorIR Calls
融合后的 IRModule 仅包含对图层次 op 的调用，为了进一步进行底层优化和代码生成，我们需要将这些高级原语运算转换为相应的 TensorIR 函数(或调用库函数)

以下代码将图层算子重新映射到相应的 TensorIR 函数，在这里，我们利用 Mutator 中的内部 block builder 并使用 `call_te` 返回转换后的值
```python
@relax.expr_functor.mutator
class LowerToTensorIR(relax.PyExprMutator):
    def __init__(self, mod: IRModule, op_map) -> None:
        super().__init__()
        self.mod_ = mod
        self.op_map = {
            tvm.ir.Op.get(k): v for k, v in op_map.items()
        }

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        if call.op in self.op_map:
            return self.op_map[call.op](self.builder_, call)
        return call

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()

def map_matmul(bb, call):
    x, w = call.args
    return bb.call_te(topi.nn.matmul, x, w)

def map_add(bb, call):
    a, b = call.args
    return bb.call_te(topi.add, a, b)

def map_relu(bb, call):
    return bb.call_te(topi.nn.relu, call.args[0])

def map_transpose(bb, call):
    return bb.call_te(topi.transpose, call.args[0], )

op_map = {
  "relax.matmul": map_matmul,
  "relax.add": map_add,
  "relax.nn.relu": map_relu,
  "relax.permute_dims": map_transpose
}

@tvm.ir.transform.module_pass(opt_level=0, name="LowerToTensorIR")
class LowerToTensorIRPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return LowerToTensorIR(mod, op_map).transform()

MLPModelTIR = LowerToTensorIRPass()(MLPFused)
MLPModelTIR.show()
```
此时得到的 `fused_matmul_add0` 和 `fused_matmul_add1` 仍然是上层 relax 函数，它们调用相应的 TensorIR `matmul` 和 `add` 函数，我们可以将它们变成一个单一的 TensorIR 函数，然后可以用于后续优化和代码生成阶段
```python
MLPModelFinal = relax.transform.FuseTIR()(MLPModelTIR)
MLPModelFinal.show()
```
## 7.4 Build and Run
## 7.5 Summary
- 我们可以通过改写计算图数据结构来优化模型
- 使用访问者模式改写调用节点
- 我们可以进行计算图转换，例如融合和循环级代码生成