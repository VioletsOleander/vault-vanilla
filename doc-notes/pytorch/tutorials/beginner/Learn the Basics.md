> 2.5.0+cu124
# 0 Quickstart
This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.

## Working with data
PyTorch has two [primitives to work with data](https://pytorch.org/docs/stable/data.html): `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`.  `Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset`.
>处理数据：`torch.utils.data.DataLoader/Dataset`
> `Dataset` ： (样本，标签)；`DataLoader`：包装 `Dataset` 的迭代器

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

PyTorch offers domain-specific libraries such as [TorchText](https://pytorch.org/text/stable/index.html), [TorchVision](https://pytorch.org/vision/stable/index.html), and [TorchAudio](https://pytorch.org/audio/stable/index.html), all of which include datasets. For this tutorial, we will be using a TorchVision dataset.
> torch 的领域特定库：torchtext, torchvision, torchaudio

The `torchvision.datasets` module contains `Dataset` objects for many real-world vision data like CIFAR, COCO ([full list here](https://pytorch.org/vision/stable/datasets.html)). In this tutorial, we use the FashionMNIST dataset. Every TorchVision `Dataset` includes two arguments: `transform` and `target_transform` to modify the samples and labels respectively.
> `torchvision.datasets` 模块包含了多个 `Dataset` 的子类，例如 `CIFAR/COCO`
> 每个 `Dataset` 的子类的构造函数包含两个参数：`transform/target_transform` 用于变换样本/标签

```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 65.5k/26.4M [00:00<01:12, 366kB/s]
  1%|          | 229k/26.4M [00:00<00:38, 684kB/s]
  4%|3         | 950k/26.4M [00:00<00:11, 2.19MB/s]
 15%|#4        | 3.83M/26.4M [00:00<00:02, 7.62MB/s]
 37%|###7      | 9.83M/26.4M [00:00<00:00, 16.8MB/s]
 60%|######    | 15.9M/26.4M [00:01<00:00, 22.5MB/s]
 83%|########3 | 22.0M/26.4M [00:01<00:00, 26.1MB/s]
100%|##########| 26.4M/26.4M [00:01<00:00, 19.4MB/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|##########| 29.5k/29.5k [00:00<00:00, 324kB/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|1         | 65.5k/4.42M [00:00<00:12, 360kB/s]
  4%|4         | 197k/4.42M [00:00<00:07, 572kB/s]
 10%|9         | 426k/4.42M [00:00<00:04, 886kB/s]
 16%|#5        | 688k/4.42M [00:00<00:03, 1.10MB/s]
 22%|##2       | 983k/4.42M [00:00<00:02, 1.29MB/s]
 29%|##8       | 1.28M/4.42M [00:01<00:02, 1.52MB/s]
 33%|###3      | 1.47M/4.42M [00:01<00:01, 1.51MB/s]
 41%|####1     | 1.84M/4.42M [00:01<00:01, 1.67MB/s]
 50%|#####     | 2.23M/4.42M [00:01<00:01, 1.84MB/s]
 61%|######    | 2.69M/4.42M [00:01<00:00, 2.06MB/s]
 72%|#######1  | 3.18M/4.42M [00:01<00:00, 2.26MB/s]
 84%|########3 | 3.70M/4.42M [00:02<00:00, 2.45MB/s]
 97%|#########7| 4.29M/4.42M [00:02<00:00, 2.69MB/s]
100%|##########| 4.42M/4.42M [00:02<00:00, 1.94MB/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|##########| 5.15k/5.15k [00:00<00:00, 31.8MB/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

We pass the `Dataset` as an argument to `DataLoader`. This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element in the dataloader iterable will return a batch of 64 features and labels.
> `DataLoader` 包装 `Dataset`，支持自动 batching、采样、shuffling、多进程数据装载
> `batch_size` 是多少，`for-in` loop 每次迭代时 `DataLoader` 返回的 (样本，标签) 对的数量就是多少

```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

```
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```

Read more about [loading data in PyTorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

---

## Creating Models
To define a neural network in PyTorch, we create a class that inherits from [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). We define the layers of the network in the `__init__` function and specify how data will pass through the network in the `forward` function. To accelerate operations in the neural network, we move it to the GPU or MPS if available.
> 模型类继承于 `nn.Module`
> `__init__` 函数定义网络的层
> `forward` 定义模型如何处理数据
> `to()` 方法将模型移动到设备

```python
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```


```
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

Read more about [building neural networks in PyTorch](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).

---

## Optimizing the Model Parameters
To train a model, we need a [loss function](https://pytorch.org/docs/stable/nn.html#loss-functions) and an [optimizer](https://pytorch.org/docs/stable/optim.html).

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.
> 训练 (batchwise)：
> - 模型预测/前向传播
> - 计算损失/误差
> - 反向传播
> - 更新参数
>
> `model.train()`
> `optimizer.zero_grad()` 防止不同 batch 的梯度累积
> `Dataloader` 返回的数据也通过 `to()` 方法移动到设备

```python
def train(dataloader, model, loss_fn, optimizer):
size = len(dataloader.dataset)
model.train()
for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

We also check the model’s performance against the test dataset to ensure it is learning.
> 测试：
> - 前向传播 (batchwise)
> - 计算并累积误差/正确样本数
>
> `model.eval()`
> `with torch.no_grad()`
> `test_loss` 随着 batch 累积，最后 `test_loss/=num_batches` 计算平均值
> `correct` 随着 batch 累积，最后 `correct/=size` 计算整体正确率

```python
def test(dataloader, model, loss_fn):
size = len(dataloader.dataset)
num_batches = len(dataloader)
model.eval()
test_loss, correct = 0, 0
with torch.no_grad():
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
test_loss /= num_batches
correct /= size
print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

The training process is conducted over several iterations (_epochs_). During each epoch, the model learns parameters to make better predictions. We print the model’s accuracy and loss at each epoch; we’d like to see the accuracy increase and the loss decrease with every epoch.
> 训练过程以 epoch 为单位重复，每个 epoch 遍历一遍整个数据集
> 一般每个 epoch 之后对模型进行一次测试，由此了解 epoch 之间模型的进步

```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

```
Epoch 1
-------------------------------
loss: 2.303494  [   64/60000]
loss: 2.294637  [ 6464/60000]
loss: 2.277102  [12864/60000]
loss: 2.269977  [19264/60000]
loss: 2.254235  [25664/60000]
loss: 2.237146  [32064/60000]
loss: 2.231055  [38464/60000]
loss: 2.205037  [44864/60000]
loss: 2.203240  [51264/60000]
loss: 2.170889  [57664/60000]
Test Error:
 Accuracy: 53.9%, Avg loss: 2.168588

Epoch 2
-------------------------------
loss: 2.177787  [   64/60000]
loss: 2.168083  [ 6464/60000]
loss: 2.114910  [12864/60000]
loss: 2.130412  [19264/60000]
loss: 2.087473  [25664/60000]
loss: 2.039670  [32064/60000]
loss: 2.054274  [38464/60000]
loss: 1.985457  [44864/60000]
loss: 1.996023  [51264/60000]
loss: 1.917241  [57664/60000]
Test Error:
 Accuracy: 60.2%, Avg loss: 1.920374

Epoch 3
-------------------------------
loss: 1.951705  [   64/60000]
loss: 1.919516  [ 6464/60000]
loss: 1.808730  [12864/60000]
loss: 1.846550  [19264/60000]
loss: 1.740618  [25664/60000]
loss: 1.698733  [32064/60000]
loss: 1.708889  [38464/60000]
loss: 1.614436  [44864/60000]
loss: 1.646475  [51264/60000]
loss: 1.524308  [57664/60000]
Test Error:
 Accuracy: 61.4%, Avg loss: 1.547092

Epoch 4
-------------------------------
loss: 1.612695  [   64/60000]
loss: 1.570870  [ 6464/60000]
loss: 1.424730  [12864/60000]
loss: 1.489542  [19264/60000]
loss: 1.367256  [25664/60000]
loss: 1.373464  [32064/60000]
loss: 1.376744  [38464/60000]
loss: 1.304962  [44864/60000]
loss: 1.347154  [51264/60000]
loss: 1.230661  [57664/60000]
Test Error:
 Accuracy: 62.7%, Avg loss: 1.260891

Epoch 5
-------------------------------
loss: 1.337803  [   64/60000]
loss: 1.313278  [ 6464/60000]
loss: 1.151837  [12864/60000]
loss: 1.252142  [19264/60000]
loss: 1.123048  [25664/60000]
loss: 1.159531  [32064/60000]
loss: 1.175011  [38464/60000]
loss: 1.115554  [44864/60000]
loss: 1.160974  [51264/60000]
loss: 1.062730  [57664/60000]
Test Error:
 Accuracy: 64.6%, Avg loss: 1.087374

Done!
```

Read more about [Training your model](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

---

## Saving Models
A common way to save a model is to serialize the internal state dictionary (containing the model parameters).
> `torch.save(model.state_dict(), 'model.pth')` ：序列化模型的状态字典

```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

```
Saved PyTorch Model State to model.pth
```

## Loading Models
The process for loading a model includes re-creating the model structure and loading the state dictionary into it.
> 装载模型：
> - 重建模型结构
> - `model.load_state_dict(torch.load('model.pth'))` 装载状态字典

```python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

```
<All keys matched successfully>
```

This model can now be used to make predictions.

```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

```
Predicted: "Ankle boot", Actual: "Ankle boot"
```

Read more about [Saving & Loading your model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).

# 1 Tensors
Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

Tensors are similar to [NumPy’s](https://numpy.org/) ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data (see [Bridge with NumPy](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#bridge-to-np-label)). Tensors are also optimized for automatic differentiation (we’ll see more about that later in the [Autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) section). If you’re familiar with ndarrays, you’ll be right at home with the Tensor API. If not, follow along!
> tensor 和 ndarray 可以共享相同的内存对象
> tensor 可以运行于加速设备
> tensor 可以自动微分

```python
import torch
import numpy as np
```

## Initializing a Tensor
Tensors can be initialized in various ways. Take a look at the following examples:

**Directly from data**
Tensors can be created directly from data. The data type is automatically inferred.

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

**From a NumPy array**
Tensors can be created from NumPy arrays (and vice versa - see [Bridge with NumPy](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#bridge-to-np-label)).

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

**From another tensor:**
The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```

```
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.8823, 0.9150],
        [0.3829, 0.9593]])
```

**With random or constant values:**
`shape` is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

```
Random Tensor:
 tensor([[0.3904, 0.6009, 0.2566],
        [0.7936, 0.9408, 0.1332]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

---

## Attributes of a Tensor
Tensor attributes describe their shape, datatype, and the device on which they are stored.

```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

```
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

---

## Operations on Tensors
Over 100 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing), sampling and more are comprehensively described [here](https://pytorch.org/docs/stable/torch.html).

Each of these operations can be run on the GPU (at typically higher speeds than on a CPU). If you’re using Colab, allocate a GPU by going to Runtime > Change runtime type > GPU.

By default, tensors are created on the CPU. We need to explicitly move tensors to the GPU using `.to` method (after checking for GPU availability). Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!
> tensor 默认创建于 CPU
> `to()` 方法移动到设备
> 注意设备间移动 tensor 是有考校的

```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```

Try out some of the operations from the list. If you’re familiar with the NumPy API, you’ll find the Tensor API a breeze to use.

**Standard numpy-like indexing and slicing:**

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

```
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

**Joining tensors** You can use `torch.cat` to concatenate a sequence of tensors along a given dimension. See also [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html), another tensor joining operator that is subtly different from `torch.cat`.

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

```
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

**Arithmetic operations**

```python
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

**Single-element tensors** If you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using `item()`:

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

```
12.0 <class 'float'>
```

**In-place operations** Operations that store the result into the operand are called in-place. They are denoted by a `_` suffix. For example: `x.copy_(y)`, `x.t_()`, will change `x`.
> 原地的操作其名称都带有后缀 `_`

```python
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```

```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

Note
In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.
> 原地操作会导致计算微分时出问题，故不推荐使用

---

## Bridge with NumPy
Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

### Tensor to NumPy array

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

```
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

A change in the tensor reflects in the NumPy array.
> CPU 上， `tensor.numpy()` 返回的 ndarray 会共享 tensor 的内存对象

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

```
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

### NumPy array to Tensor

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

Changes in the NumPy array reflects in the tensor.
> `torch.from_numpy(ndarray)` 返回的 tensor 同样共享 ndarray 的内存对象

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

```
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```

# 2 Datasets & DataLoaders
Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability and modularity. PyTorch provides two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset` that allow you to use pre-loaded datasets as well as your own data. `Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset` to enable easy access to the samples.
> torch 意在解耦模型训练的代码和数据集操作的代码

PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that subclass `torch.utils.data.Dataset` and implement functions specific to the particular data. They can be used to prototype and benchmark your model. You can find them here: [Image Datasets](https://pytorch.org/vision/stable/datasets.html), [Text Datasets](https://pytorch.org/text/stable/datasets.html), and [Audio Datasets](https://pytorch.org/audio/stable/datasets.html)
> torch 领域特定库提供的预装载数据集都继承自 `torch.utils.data.Dataste` ，并针对数据重载/实现了特定函数

## Loading a Dataset
Here is an example of how to load the [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) dataset from TorchVision. Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples. Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

We load the [FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist) with the following parameters:

- `root` is the path where the train/test data is stored,
- `train` specifies training or test dataset,
- `download=True` downloads the data from the internet if it’s not available at `root`.
- `transform` and `target_transform` specify the feature and label transformations

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 65.5k/26.4M [00:00<01:11, 367kB/s]
  1%|          | 229k/26.4M [00:00<00:38, 687kB/s]
  4%|3         | 950k/26.4M [00:00<00:11, 2.20MB/s]
 15%|#4        | 3.83M/26.4M [00:00<00:02, 7.66MB/s]
 38%|###7      | 9.96M/26.4M [00:00<00:00, 17.2MB/s]
 61%|######1   | 16.1M/26.4M [00:01<00:00, 23.0MB/s]
 83%|########2 | 21.9M/26.4M [00:01<00:00, 30.3MB/s]
 96%|#########5| 25.3M/26.4M [00:01<00:00, 26.7MB/s]
100%|##########| 26.4M/26.4M [00:01<00:00, 19.5MB/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|##########| 29.5k/29.5k [00:00<00:00, 328kB/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|1         | 65.5k/4.42M [00:00<00:12, 363kB/s]
  5%|5         | 229k/4.42M [00:00<00:06, 682kB/s]
 21%|##1       | 950k/4.42M [00:00<00:01, 2.19MB/s]
 87%|########6 | 3.83M/4.42M [00:00<00:00, 7.62MB/s]
100%|##########| 4.42M/4.42M [00:00<00:00, 6.10MB/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|##########| 5.15k/5.15k [00:00<00:00, 36.9MB/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## Iterating and Visualizing the Dataset
We can index `Datasets` manually like a list: `training_data[index]`. We use `matplotlib` to visualize some samples in our training data.
> `Datasets` 对象可以直接 `[]`  索引

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

![Ankle Boot, Shirt, Bag, Ankle Boot, Trouser, Sandal, Coat, Sandal, Pullover](https://pytorch.org/tutorials/_images/sphx_glr_data_tutorial_001.png)

---

## Creating a Custom Dataset for your files
A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. Take a look at this implementation; the FashionMNIST images are stored in a directory `img_dir`, and their labels are stored separately in a CSV file `annotations_file`.
> 自定义数据集需要继承 `Dataset` ，并实现 `__init__/__len__/__getitem__`

In the next sections, we’ll break down what’s happening in each of these functions.

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

### `__init__` 
The __init__ function is run once when instantiating the Dataset object. We initialize the directory containing the images, the annotations file, and both transforms (covered in more detail in the next section).
> `__init__` 初始化图片目录路径、标记文件、transform 函数

The labels.csv file looks like:

```
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

### `__len__`
The __len__ function returns the number of samples in our dataset.

Example:

```python
def __len__(self):
    return len(self.img_labels)
```

### `__getitem__`
The __getitem__ function loads and returns a sample from the dataset at the given index `idx`. Based on the index, it identifies the image’s location on disk, converts that to a tensor using `read_image`, retrieves the corresponding label from the csv data in `self.img_labels`, calls the transform functions on them (if applicable), and returns the tensor image and corresponding label in a tuple.
> `__getitem__` 返回 `idx` 处的图像(tensor)和标签
> `read_image` 将图像转化为 tensor

```python
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label
```

---

## Preparing your data for training with DataLoaders
The `Dataset` retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s `multiprocessing` to speed up data retrieval.

`DataLoader` is an iterable that abstracts this complexity for us in an easy API.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

> `Dataloader` 负责 batch 化样本、在每个 epoch reshuffle 数据集 (减少模型过拟合)、使用 `multiprocessing` 多进程装载数据
> `DataLoader` 是可迭代对象，每次迭代返回一个 batch

## Iterate through the DataLoader
We have loaded that dataset into the `DataLoader` and can iterate through the dataset as needed. Each iteration below returns a batch of `train_features` and `train_labels` (containing `batch_size=64` features and labels respectively). Because we specified `shuffle=True`, after we iterate over all batches the data is shuffled (for finer-grained control over the data loading order, take a look at [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)).
> `shuffle=True` 时，`DataLoader` 迭代完全部的 batch 之后，会 shuffle 数据集

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

![data tutorial](https://pytorch.org/tutorials/_images/sphx_glr_data_tutorial_002.png)

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 5
```

# 3 Transforms
Data does not always come in its final processed form that is required for training machine learning algorithms. We use **transforms** to perform some manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters - `transform` to modify the features and `target_transform` to modify the labels - that accept callables containing the transformation logic. The [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) module offers several commonly-used transforms out of the box.
> `transform/targe_transform` 应该是可调用对象
> `torchvision.transforms` 模块提供了常用的 transform 函数

The FashionMNIST features are in PIL Image format, and the labels are integers. For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. To make these transformations, we use `ToTensor` and `Lambda`.
> 例如 `torch.vision.transforms.ToTensor()` 将 PIL 图像格式转化为规范化的 tensor；`torch.vision.transforms.Lambda(...) ` 将整数转化为 one-hot tensor

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 65.5k/26.4M [00:00<01:12, 364kB/s]
  1%|          | 229k/26.4M [00:00<00:38, 685kB/s]
  4%|3         | 950k/26.4M [00:00<00:11, 2.20MB/s]
 14%|#4        | 3.70M/26.4M [00:00<00:02, 8.94MB/s]
 26%|##6       | 6.91M/26.4M [00:00<00:01, 12.8MB/s]
 47%|####7     | 12.5M/26.4M [00:00<00:00, 23.5MB/s]
 60%|######    | 15.9M/26.4M [00:01<00:00, 22.2MB/s]
 80%|########  | 21.3M/26.4M [00:01<00:00, 29.7MB/s]
 95%|#########4| 25.0M/26.4M [00:01<00:00, 26.9MB/s]
100%|##########| 26.4M/26.4M [00:01<00:00, 19.5MB/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|##########| 29.5k/29.5k [00:00<00:00, 326kB/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|1         | 65.5k/4.42M [00:00<00:12, 363kB/s]
  5%|5         | 229k/4.42M [00:00<00:06, 682kB/s]
 21%|##1       | 950k/4.42M [00:00<00:01, 2.19MB/s]
 87%|########6 | 3.83M/4.42M [00:00<00:00, 7.61MB/s]
100%|##########| 4.42M/4.42M [00:00<00:00, 6.09MB/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|##########| 5.15k/5.15k [00:00<00:00, 33.0MB/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## ToTensor()
[ToTensor](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor) converts a PIL image or NumPy `ndarray` into a `FloatTensor`. and scales the image’s pixel intensity values in the range [0., 1.]
> `ToTensor()` 将 PIL 图像或 ndarray 转化为 `FloatTensor` (按像素)，tensor 中元素范围在 `[0., 1.]`

## Lambda Transforms
Lambda transforms apply any user-defined lambda function. Here, we define a function to turn the integer into a one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls [scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html) which assigns a `value=1` on the index as given by the label `y`.

```python
target_transform = Lambda(lambda y: torch.zeros(
10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

# 4 Build the Neural Network
Neural networks comprise of layers/modules that perform operations on data. The [torch.nn](https://pytorch.org/docs/stable/nn.html) namespace provides all the building blocks you need to build your own neural network. Every module in PyTorch subclasses the [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). A neural network is a module itself that consists of other modules (layers). This nested structure allows for building and managing complex architectures easily.
> `torch.nn` 提供构建 NN 所需的模块
> torch 中，所有的模块都是 `nn.Module` 的子类
> NN 由模块 (层) 构成，且本身也是模块

In the following sections, we’ll build a neural network to classify images in the FashionMNIST dataset.

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

## Get Device for Training
We want to be able to train our model on a hardware accelerator like the GPU or MPS, if available. Let’s check to see if [torch.cuda](https://pytorch.org/docs/stable/notes/cuda.html) or [torch.backends.mps](https://pytorch.org/docs/stable/notes/mps.html) are available, otherwise we use the CPU.

```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

```
Using cuda device
```

## Define the Class
We define our neural network by subclassing `nn.Module`, and initialize the neural network layers in `__init__`. Every `nn.Module` subclass implements the operations on input data in the `forward` method.
> `nn.Module` 必须定义的方法：`__init__,` `forward `

```python
class NeuralNetwork(nn.Module):
def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )

def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
```

```
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

To use the model, we pass it the input data. This executes the model’s `forward`, along with some [background operations](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866). Do not call `model.forward()` directly!
> 直接调用 `nn.Module` 对象即执行 `forward` ，以及部分额外操作

Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of 10 raw predicted values for each class, and dim=1 corresponding to the individual values of each output. We get the prediction probabilities by passing it through an instance of the `nn.Softmax` module.
> 模型的 `forward` 是 samplewise 定义的，但支持直接传入第零维为 batch_size 的样本 batch，对应的返回值的第零维也是 batch_size，之后的维度为各个 sample 的前向传播结果

> 模型中，`nn.Flatten, nn.ReLU, nn.Linear` 等都默认第零维为 batch_size，故而保持第零维，这点对于其他 `nn` 模块同样成立

```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

```
Predicted class: tensor([7], device='cuda:0')
```

---

## Model Layers
Let’s break down the layers in the FashionMNIST model. To illustrate it, we will take a sample minibatch of 3 images of size 28x28 and see what happens to it as we pass it through the network.

```python
input_image = torch.rand(3,28,28)
print(input_image.size())
```

```
torch.Size([3, 28, 28])
```

### nn.Flatten
We initialize the [nn.Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values ( the minibatch dimension (at dim=0) is maintained).

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

```
torch.Size([3, 784])
```

### nn.Linear
The [linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) is a module that applies a linear transformation on the input using its stored weights and biases.

```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

```
torch.Size([3, 20])
```

### nn.ReLU
Non-linear activations are what create the complex mappings between the model’s inputs and outputs. They are applied after linear transformations to introduce _nonlinearity_, helping neural networks learn a wide variety of phenomena.

In this model, we use [nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) between our linear layers, but there’s other activations to introduce non-linearity in your model.

```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

```
Before ReLU: tensor([[ 0.4158, -0.0130, -0.1144,  0.3960,  0.1476, -0.0690, -0.0269,  0.2690,
          0.1353,  0.1975,  0.4484,  0.0753,  0.4455,  0.5321, -0.1692,  0.4504,
          0.2476, -0.1787, -0.2754,  0.2462],
        [ 0.2326,  0.0623, -0.2984,  0.2878,  0.2767, -0.5434, -0.5051,  0.4339,
          0.0302,  0.1634,  0.5649, -0.0055,  0.2025,  0.4473, -0.2333,  0.6611,
          0.1883, -0.1250,  0.0820,  0.2778],
        [ 0.3325,  0.2654,  0.1091,  0.0651,  0.3425, -0.3880, -0.0152,  0.2298,
          0.3872,  0.0342,  0.8503,  0.0937,  0.1796,  0.5007, -0.1897,  0.4030,
          0.1189, -0.3237,  0.2048,  0.4343]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.4158, 0.0000, 0.0000, 0.3960, 0.1476, 0.0000, 0.0000, 0.2690, 0.1353,
         0.1975, 0.4484, 0.0753, 0.4455, 0.5321, 0.0000, 0.4504, 0.2476, 0.0000,
         0.0000, 0.2462],
        [0.2326, 0.0623, 0.0000, 0.2878, 0.2767, 0.0000, 0.0000, 0.4339, 0.0302,
         0.1634, 0.5649, 0.0000, 0.2025, 0.4473, 0.0000, 0.6611, 0.1883, 0.0000,
         0.0820, 0.2778],
        [0.3325, 0.2654, 0.1091, 0.0651, 0.3425, 0.0000, 0.0000, 0.2298, 0.3872,
         0.0342, 0.8503, 0.0937, 0.1796, 0.5007, 0.0000, 0.4030, 0.1189, 0.0000,
         0.2048, 0.4343]], grad_fn=<ReluBackward0>)
```

### nn.Sequential
[nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) is an ordered container of modules. The data is passed through all the modules in the same order as defined. You can use sequential containers to put together a quick network like `seq_modules`.
> `nn.Sequential` 为有序模块容器，其 `forward` 就是调用其中模块的 `forward` 并传递相应数据

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

### nn.Softmax
The last linear layer of the neural network returns logits - raw values in [-infty, infty] - which are passed to the [nn.Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) module. The logits are scaled to values [0, 1] representing the model’s predicted probabilities for each class. `dim` parameter indicates the dimension along which the values must sum to 1.

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

## Model Parameters
Many layers inside a neural network are _parameterized_, i.e. have associated weights and biases that are optimized during training. Subclassing `nn.Module` automatically tracks all fields defined inside your model object, and makes all parameters accessible using your model’s `parameters()` or `named_parameters()` methods.
> 模型的 `parameters/name_parameters` 方法用于访问其各层参数，方法返回字典

In this example, we iterate over each parameter, and print its size and a preview of its values.

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

```
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0273,  0.0296, -0.0084,  ..., -0.0142,  0.0093,  0.0135],
        [-0.0188, -0.0354,  0.0187,  ..., -0.0106, -0.0001,  0.0115]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0155, -0.0327], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0116,  0.0293, -0.0280,  ...,  0.0334, -0.0078,  0.0298],
        [ 0.0095,  0.0038,  0.0009,  ..., -0.0365, -0.0011, -0.0221]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0148, -0.0256], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0147, -0.0229,  0.0180,  ..., -0.0013,  0.0177,  0.0070],
        [-0.0202, -0.0417, -0.0279,  ..., -0.0441,  0.0185, -0.0268]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([ 0.0070, -0.0411], device='cuda:0', grad_fn=<SliceBackward0>)
```

# 5 Automatic Differentiation with `torch.autograd`
When training neural networks, the most frequently used algorithm is **back propagation**. In this algorithm, parameters (model weights) are adjusted according to the **gradient** of the loss function with respect to the given parameter.

To compute those gradients, PyTorch has a built-in differentiation engine called `torch.autograd`. It supports automatic computation of gradient for any computational graph.
> 反向传播时，模型所有参数根据它相对 loss 的梯度更新
> torch 内建微分引擎 `torch.autograd` 支持任意计算图的自动微分计算

Consider the simplest one-layer neural network, with input `x`, parameters `w` and `b`, and some loss function. It can be defined in PyTorch in the following manner:

```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

## Tensors, Functions and Computational graph
This code defines the following **computational graph**:


![](https://pytorch.org/tutorials/_images/comp-graph.png)

In this network, `w` and `b` are **parameters**, which we need to optimize. Thus, we need to be able to compute the gradients of loss function with respect to those variables. In order to do that, we set the `requires_grad` property of those tensors.
> torch 随着前向计算而构建计算图，图的起始节点为输入，最终节点为 loss
> 图中需要计算梯度的 tensor，其 `requires_grad = True`

Note
You can set the value of `requires_grad` when creating a tensor, or later by using `x.requires_grad_(True)` method.

A function that we apply to tensors to construct computational graph is in fact an object of class `Function`. This object knows how to compute the function in the _forward_ direction, and also how to compute its derivative during the _backward propagation_ step. A reference to the backward propagation function is stored in `grad_fn` property of a tensor. You can find more information of `Function` [in the documentation](https://pytorch.org/docs/stable/autograd.html#function).
> 要构建计算图，对 tensor 应用的函数应该是类 `Function` 的对象
> 计算图的构建、微分计算由该 `Function` 对象支持
> tensor 的 `grad_fn` 指向其反向传播函数对象

```python
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

```
Gradient function for z = <AddBackward0 object at 0x7f019c2cffd0>
Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x7f019c2cfb50>
```

## Computing Gradients
To optimize weights of parameters in the neural network, we need to compute the derivatives of our loss function with respect to parameters, namely, we need $\frac {\partial loss}{\partial w}$ ​and $\frac {\partial loss}{\partial w}$ under some fixed values of `x` and `y`. To compute those derivatives, we call `loss.backward()`, and then retrieve the values from `w.grad` and `b.grad`:
> `loss.backward()` 直接根据计算图计算图中 `requries_grad=True` 的 tensor (叶节点) 相对于 loss 的梯度
> tensor 的梯度存储于 `grad` 属性

```python
loss.backward()
print(w.grad)
print(b.grad)
```

```
tensor([[0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530]])
tensor([0.3313, 0.0626, 0.2530])
```

Note

- We can only obtain the `grad` properties for the leaf nodes of the computational graph, which have `requires_grad` property set to `True`. For all other nodes in our graph, gradients will not be available.
- We can only perform gradient calculations using `backward` once on a given graph, for performance reasons. If we need to do several `backward` calls on the same graph, we need to pass `retain_graph=True` to the `backward` call.

> 给定计算图，`backward` 仅能调用一次，之后图被清理
> 如果需要对同一张图调用多次 `backward` ，传入 `retain_graph=True`

## Disabling Gradient Tracking
By default, all tensors with `requires_grad=True` are tracking their computational history and support gradient computation. However, there are some cases when we do not need to do that, for example, when we have trained the model and just want to apply it to some input data, i.e. we only want to do _forward_ computations through the network. We can stop tracking computations by surrounding our computation code with `torch.no_grad()` block:
> 所有 tensor 默认 `requires_grad=True` ，因此会追踪计算历史帮助构建计算图以支持梯度计算
> `torch.no_grad()` 环境中，不会进行计算追踪和图构建

```python
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

```
True
False
```

Another way to achieve the same result is to use the `detach()` method on the tensor:
> `detach()` 方法将 tensor 从图中脱离，其 `requries_grad` 被自动设为 `False`

```python
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```

```
False
```

There are reasons you might want to disable gradient tracking:

- To mark some parameters in your neural network as **frozen parameters**.
- To **speed up computations** when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.

## More on Computational Graphs
Conceptually, autograd keeps a record of data (tensors) and all executed operations (along with the resulting new tensors) in a directed acyclic graph (DAG) consisting of [Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) objects. In this DAG, leaves are the input tensors, roots are the output tensors. By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule.
> 计算图是有向无环图，由 `Function` 对象构成，记录了计算过程中涉及的 tensor 和运算
> 图中的叶子为输入 tensor，根为输出 tensor
> 梯度的计算通过从根到叶子的链式法则实现

In a forward pass, autograd does two things simultaneously:

- run the requested operation to compute a resulting tensor
- maintain the operation’s _gradient function_ in the DAG.

> 前向过程中 autograd 的任务：计算、维护算子在图中的梯度函数

The backward pass kicks off when `.backward()` is called on the DAG root. `autograd` then:

- computes the gradients from each `.grad_fn`,
- accumulates them in the respective tensor’s `.grad` attribute
- using the chain rule, propagates all the way to the leaf tensors.

> 调用根的 `backward()` 开始反向传播
> autograd 的任务：为每个 `grad_fn` 计算梯度、将梯度累积到 tensor 的 `grad` 、由此链式传播到叶节点

Note
**DAGs are dynamic in PyTorch** An important thing to note is that the graph is recreated from scratch; after each `.backward()` call, autograd starts populating a new graph. This is exactly what allows you to use control flow statements in your model; you can change the shape, size and operations at every iteration if needed.
> torch 中，每次 `backward()` 调用后，autograd 都清理旧图，下一次前向计算会创建新图，即计算图是动态的
> 因此模型中允许控制流语句

## Optional Reading: Tensor Gradients and Jacobian Products
In many cases, we have a scalar loss function, and we need to compute the gradient with respect to some parameters. However, there are cases when the output function is an arbitrary tensor. In this case, PyTorch allows you to compute so-called **Jacobian product**, and not the actual gradient.

For a vector function $\vec y = f(\vec x)$, where $\vec x = \langle x_1, \dots, x_n \rangle$ and $\vec y = \langle y_1, \dots, y_m\rangle$, a gradient of $\vec y$ ​ with respect to $\vec x$ is given by **Jacobian matrix**:
> 向量对向量的梯度是 Jacobian 矩阵：

$$
\begin{align}
J = \begin{bmatrix}
\frac {\partial y_1}{\partial x_1} & \cdots & \frac {\partial y_1}{\partial x_n}\\
\vdots & \ddots & \vdots\\
\frac {\partial y_m}{\partial x_1} & \cdots & \frac {\partial y_m}{\partial x_n}
\end{bmatrix}
\end{align}
$$

Instead of computing the Jacobian matrix itself, PyTorch allows you to compute **Jacobian Product** $v^T\cdot J$ for a given input vector $v = (v_1, \dots, v_m)$. This is achieved by calling `backward` with $v$ as an argument. The size of $v$ should be the same as the size of the original tensor, with respect to which we want to compute the product:
> 对于向量值 loss，其 `backward` 方法可以传入一个向量 $v$
> 要求形状和计算图的输出向量相同
> 此时输入向量的 `grad` 存储的是 Jacobian 积 $v^T \cdot J$

```python
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
```

```
First call
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])

Second call
tensor([[8., 4., 4., 4., 4.],
        [4., 8., 4., 4., 4.],
        [4., 4., 8., 4., 4.],
        [4., 4., 4., 8., 4.]])

Call after zeroing gradients
tensor([[4., 2., 2., 2., 2.],
        [2., 4., 2., 2., 2.],
        [2., 2., 4., 2., 2.],
        [2., 2., 2., 4., 2.]])
```

Notice that when we call `backward` for the second time with the same argument, the value of the gradient is different. This happens because when doing `backward` propagation, PyTorch **accumulates the gradients**, i.e. the value of computed gradients is added to the `grad` property of all leaf nodes of computational graph. If you want to compute the proper gradients, you need to zero out the `grad` property before. In real-life training an _optimizer_ helps us to do this.

Note
Previously we were calling `backward()` function without parameters. This is essentially equivalent to calling `backward(torch.tensor(1.0))`, which is a useful way to compute the gradients in case of a scalar-valued function, such as loss during neural network training.

# 6 Optimizing Model Parameters
Now that we have a model and data it’s time to train, validate and test our model by optimizing its parameters on our data. Training a model is an iterative process; in each iteration the model makes a guess about the output, calculates the error in its guess (_loss_), collects the derivatives of the error with respect to its parameters (as we saw in the [previous section](https://pytorch.org/tutorials/beginner/basics/autograd_tutorial.html)), and **optimizes** these parameters using gradient descent. For a more detailed walkthrough of this process, check out this video on [backpropagation from 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8).

## Prerequisite Code
We load the code from the previous sections on [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and [Build Model](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 65.5k/26.4M [00:00<01:12, 362kB/s]
  1%|          | 197k/26.4M [00:00<00:33, 772kB/s]
  2%|1         | 492k/26.4M [00:00<00:20, 1.27MB/s]
  6%|6         | 1.67M/26.4M [00:00<00:05, 4.41MB/s]
 15%|#4        | 3.83M/26.4M [00:00<00:02, 7.92MB/s]
 33%|###3      | 8.75M/26.4M [00:00<00:00, 18.5MB/s]
 46%|####6     | 12.3M/26.4M [00:00<00:00, 19.4MB/s]
 66%|######6   | 17.6M/26.4M [00:01<00:00, 27.4MB/s]
 80%|########  | 21.2M/26.4M [00:01<00:00, 25.3MB/s]
100%|#########9| 26.3M/26.4M [00:01<00:00, 31.1MB/s]
100%|##########| 26.4M/26.4M [00:01<00:00, 19.2MB/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|##########| 29.5k/29.5k [00:00<00:00, 326kB/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|1         | 65.5k/4.42M [00:00<00:12, 361kB/s]
  4%|3         | 164k/4.42M [00:00<00:09, 467kB/s]
  6%|5         | 262k/4.42M [00:00<00:08, 501kB/s]
  8%|8         | 360k/4.42M [00:00<00:07, 516kB/s]
 11%|#1        | 492k/4.42M [00:00<00:06, 589kB/s]
 14%|#4        | 623k/4.42M [00:01<00:05, 634kB/s]
 18%|#7        | 786k/4.42M [00:01<00:05, 721kB/s]
 21%|##1       | 950k/4.42M [00:01<00:04, 778kB/s]
 26%|##5       | 1.15M/4.42M [00:01<00:03, 872kB/s]
 30%|###       | 1.34M/4.42M [00:01<00:03, 938kB/s]
 35%|###4      | 1.54M/4.42M [00:01<00:02, 1.07MB/s]
 38%|###7      | 1.67M/4.42M [00:02<00:02, 1.04MB/s]
 44%|####3     | 1.93M/4.42M [00:02<00:02, 1.17MB/s]
 50%|#####     | 2.23M/4.42M [00:02<00:01, 1.32MB/s]
 57%|#####7    | 2.52M/4.42M [00:02<00:01, 1.42MB/s]
 65%|######5   | 2.88M/4.42M [00:02<00:00, 1.59MB/s]
 74%|#######4  | 3.28M/4.42M [00:03<00:00, 1.76MB/s]
 83%|########2 | 3.67M/4.42M [00:03<00:00, 2.05MB/s]
 89%|########8 | 3.93M/4.42M [00:03<00:00, 2.01MB/s]
 99%|#########9| 4.39M/4.42M [00:03<00:00, 2.18MB/s]
100%|##########| 4.42M/4.42M [00:03<00:00, 1.28MB/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|##########| 5.15k/5.15k [00:00<00:00, 41.8MB/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## Hyperparameters
Hyperparameters are adjustable parameters that let you control the model optimization process. Different hyperparameter values can impact model training and convergence rates ([read more](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html) about hyperparameter tuning)

We define the following hyperparameters for training:

- **Number of Epochs** - the number times to iterate over the dataset
- **Batch Size** - the number of data samples propagated through the network before the parameters are updated
- **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## Optimization Loop
Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each iteration of the optimization loop is called an **epoch**.

Each epoch consists of two main parts:

- **The Train Loop** - iterate over the training dataset and try to converge to optimal parameters.
- **The Validation/Test Loop** - iterate over the test dataset to check if model performance is improving.

> optimization 循环即 epoch 循环
> 每个 epoch 包括：
> - 训练循环
> - 验证/测试循环

Let’s briefly familiarize ourselves with some of the concepts used in the training loop. Jump ahead to see the [Full Implementation](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-impl-label) of the optimization loop.

### Loss Function
When presented with some training data, our untrained network is likely not to give the correct answer. **Loss function** measures the degree of dissimilarity of obtained result to the target value, and it is the loss function that we want to minimize during training. To calculate the loss we make a prediction using the inputs of our given data sample and compare it against the true data label value.

Common loss functions include [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) (Mean Square Error) for regression tasks, and [nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) (Negative Log Likelihood) for classification. [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) combines `nn.LogSoftmax` and `nn.NLLLoss`.
> `nn.MSELoss` for regression
> `nn.NLLoss` for  classification
> `nn.CrossEntropyLoss` = `nn.LogSoftmax` + `nn.NLLLoss`

We pass our model’s output logits to `nn.CrossEntropyLoss`, which will normalize the logits and compute the prediction error.

```python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

### Optimizer
Optimization is the process of adjusting model parameters to reduce model error in each training step. **Optimization algorithms** define how this process is performed (in this example we use Stochastic Gradient Descent). All optimization logic is encapsulated in the `optimizer` object. Here, we use the SGD optimizer; additionally, there are many [different optimizers](https://pytorch.org/docs/stable/optim.html) available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models and data.

We initialize the optimizer by registering the model’s parameters that need to be trained, and passing in the learning rate hyperparameter.
> `optimizer` 对象封装了全部优化逻辑
> 初始化时，传入模型的参数，指定学习率

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Inside the training loop, optimization happens in three steps:

- Call `optimizer.zero_grad()` to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
- Backpropagate the prediction loss with a call to `loss.backward()`. PyTorch deposits the gradients of the loss w.r.t. each parameter.
- Once we have our gradients, we call `optimizer.step()` to adjust the parameters by the gradients collected in the backward pass.

> `optimizer.zero_grad()` 清空注册的所有参数的 `grad`
> `optimizer.step()` 根据自身逻辑、学习率等超参数，注册的所有参数的 `grad` 来更新注册参数的值

## Full Implementation
We define `train_loop` that loops over our optimization code, and `test_loop` that evaluates the model’s performance against our test data.

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

```
Epoch 1
-------------------------------
loss: 2.298730  [   64/60000]
loss: 2.289123  [ 6464/60000]
loss: 2.273286  [12864/60000]
loss: 2.269406  [19264/60000]
loss: 2.249603  [25664/60000]
loss: 2.229407  [32064/60000]
loss: 2.227368  [38464/60000]
loss: 2.204261  [44864/60000]
loss: 2.206193  [51264/60000]
loss: 2.166651  [57664/60000]
Test Error:
 Accuracy: 50.9%, Avg loss: 2.166725

Epoch 2
-------------------------------
loss: 2.176750  [   64/60000]
loss: 2.169595  [ 6464/60000]
```

# 7 Save and Load the Model
In this section we will look at how to persist model state with saving, loading and running model predictions.

```python
import torch
import torchvision.models as models
```

## Saving and Loading Model Weights
PyTorch models store the learned parameters in an internal state dictionary, called `state_dict`. These can be persisted via the `torch.save` method:

```python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

```
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /var/lib/ci-user/.cache/torch/hub/checkpoints/vgg16-397923af.pth

  0%|          | 0.00/528M [00:00<?, ?B/s]
  4%|3         | 20.2M/528M [00:00<00:02, 212MB/s]
  8%|7         | 41.0M/528M [00:00<00:02, 215MB/s]
 12%|#1        | 61.8M/528M [00:00<00:02, 216MB/s]
 16%|#5        | 82.5M/528M [00:00<00:02, 216MB/s]
 20%|#9        | 103M/528M [00:00<00:02, 217MB/s]
 23%|##3       | 124M/528M [00:00<00:01, 216MB/s]
 27%|##7       | 145M/528M [00:00<00:01, 217MB/s]
 31%|###1      | 166M/528M [00:00<00:01, 217MB/s]
 35%|###5      | 186M/528M [00:00<00:01, 217MB/s]
 39%|###9      | 207M/528M [00:01<00:01, 217MB/s]
 43%|####3     | 228M/528M [00:01<00:01, 216MB/s]
 47%|####7     | 249M/528M [00:01<00:01, 216MB/s]
 51%|#####1    | 269M/528M [00:01<00:01, 216MB/s]
 55%|#####4    | 290M/528M [00:01<00:01, 217MB/s]
 59%|#####8    | 311M/528M [00:01<00:01, 217MB/s]
 63%|######2   | 332M/528M [00:01<00:00, 217MB/s]
 67%|######6   | 352M/528M [00:01<00:00, 217MB/s]
 71%|#######   | 373M/528M [00:01<00:00, 217MB/s]
 75%|#######4  | 394M/528M [00:01<00:00, 217MB/s]
 79%|#######8  | 415M/528M [00:02<00:00, 217MB/s]
 83%|########2 | 436M/528M [00:02<00:00, 217MB/s]
 86%|########6 | 456M/528M [00:02<00:00, 217MB/s]
 90%|######### | 477M/528M [00:02<00:00, 217MB/s]
 94%|#########4| 498M/528M [00:02<00:00, 217MB/s]
 98%|#########8| 518M/528M [00:02<00:00, 217MB/s]
100%|##########| 528M/528M [00:02<00:00, 217MB/s]
```

To load model weights, you need to create an instance of the same model first, and then load the parameters using `load_state_dict()` method.

In the code below, we set `weights_only=True` to limit the functions executed during unpickling to only those necessary for loading weights. Using `weights_only=True` is considered a best practice when loading weights.
> 使用 `torch.load()` 时，`weights_only=True` 限制 unpickling 时执行的函数仅为装载权重所必要的函数

```python
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
```

```
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

Note
be sure to call `model.eval()` method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.
> `model.eval()` ：设定 dropout 和 bach normalization 层的参数固定，保持推理结果一致

## Saving and Loading Models with Shapes
When loading model weights, we needed to instantiate the model class first, because the class defines the structure of a network. We might want to save the structure of this class together with the model, in which case we can pass `model` (and not `model.state_dict()`) to the saving function:
> 要存储模型结构+模型状态字典，则直接 `torch.save(model)`

```python
torch.save(model, 'model.pth')
```

We can then load the model as demonstrated below.

As described in [Saving and loading torch.nn.Modules](https://pytorch.org/docs/main/notes/serialization.html#saving-and-loading-torch-nn-modules), saving `state_dict` is considered the best practice. However, below we use `weights_only=False` because this involves loading the model, which is a legacy use case for `torch.save`.

```python
model = torch.load('model.pth', weights_only=False),
```

Note
This approach uses Python [pickle](https://docs.python.org/3/library/pickle.html) module when serializing the model, thus it relies on the actual class definition to be available when loading the model.