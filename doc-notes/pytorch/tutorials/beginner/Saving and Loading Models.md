---
completed: true
version: 2.6.0+cu124
---
Last Updated: Sep 10, 2024 | Last Verified: Nov 05, 2024
**Author:** [Matthew Inkawhich](https://github.com/MatthewInkawhich)

This document provides solutions to a variety of use cases regarding the saving and loading of PyTorch models. Feel free to read the whole document, or just skip to the code you need for a desired use case.

When it comes to saving and loading models, there are three core functions to be familiar with:

1. [torch.save](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save): Saves a serialized object to disk. This function uses Python’s [pickle](https://docs.python.org/3/library/pickle.html) utility for serialization. Models, tensors, and dictionaries of all kinds of objects can be saved using this function.
2. [torch.load](https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load): Uses [pickle](https://docs.python.org/3/library/pickle.html) ’s unpickling facilities to deserialize pickled object files to memory. This function also facilitates the device to load the data into (see [Saving & Loading Model Across Devices](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices)).
3. [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict): Loads a model’s parameter dictionary using a deserialized _state_dict_. For more information on _state_dict_, see [What is a state_dict?](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

>  涉及保存和加载模型和核心函数包括：
>  1. `torch.save` : 将序列化的对象存储到磁盘，该函数使用 Python 的 `pickle` 模块将对象序列化。该函数可以用于保存模型、张量和各种对象的字典
>  2. `torch.load` : 使用 `pickle` 的反序列化功能将序列化的对象文件反序列化到内存中。该函数可以指定需要加载数据到哪个设备
>  3. `torch.nn.Module.load_state_dict` : 使用反序列化的 `state_dict` 加载模型的参数字典

## What is a `state_dict`?
In PyTorch, the learnable parameters (i.e. weights and biases) of an `torch.nn.Module` model are contained in the model’s _parameters_ (accessed with `model.parameters()`). A _state_dict_ is simply a Python dictionary object that maps each layer to its parameter tensor. Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) and registered buffers (batchnorm’s running_mean) have entries in the model’s _state_dict_. 
>  PyTorch 中，`torch.nn.Module` 模型的可学习参数 (例如权重和偏置) 都存储于模型的 `parameters` 中 (通过 `model.parameters()` 访问)
>  `state_dict` 本质上是一个 Python 字典对象，它将模型的每层映射到对应的参数张量，注意只有带有可学习参数的层 (例如卷积层、线性层) 和已注册的缓存 (例如 batchnorm 的 running_mean) 才会在模型的 `state_dict` 中有条目

Optimizer objects (`torch.optim`) also have a _state_dict_, which contains information about the optimizer’s state, as well as the hyperparameters used.
>  优化器对象 (`torch.optim`) 也有 `state_dict` ，包含了关于优化器状态、使用的超参数的信息

Because _state_dict_ objects are Python dictionaries, they can be easily saved, updated, altered, and restored, adding a great deal of modularity to PyTorch models and optimizers.
>  `state_dict` 对象本质是 Python 字典，故易于存储、更新、修改、恢复

### Example:
Let’s take a look at the _state_dict_ from the simple model used in the [Training a classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) tutorial.

```python
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

```
**Output:**

Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias   torch.Size([16])
fc1.weight   torch.Size([120, 400])
fc1.bias     torch.Size([120])
fc2.weight   torch.Size([84, 120])
fc2.bias     torch.Size([84])
fc3.weight   torch.Size([10, 84])
fc3.bias     torch.Size([10])

Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
```

## Saving & Loading Model for Inference

### Save/Load `state_dict` (Recommended)

**Save:**

```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()
```

> [! Note]
> The 1.6 release of PyTorch switched `torch.save` to use a new zip file-based format. `torch.load` still retains the ability to load files in the old format. If for any reason you want `torch.save` to use the old format, pass the `kwarg` parameter `_use_new_zipfile_serialization=False`.

When saving a model for inference, it is only necessary to save the trained model’s learned parameters. Saving the model’s _state_dict_ with the `torch.save()` function will give you the most flexibility for restoring the model later, which is why it is the recommended method for saving models.
>  保存用于推理的模型时，仅需要保存训练好的模型参数
>  因此用 `torch.save()` 仅保存模型的 `state_dict` 可以为之后恢复模型提高更多的灵活性

A common PyTorch convention is to save models using either a `.pt` or `.pth` file extension.

Remember that you must call `model.eval()` to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.
>  注意在运行推理之前必须调用 `model.eval()` 将 dropout 和 batch normalization 层设定为评估模式，否则推理结果将不一致

> [! Note]
> Notice that the `load_state_dict()` function takes a dictionary object, NOT a path to a saved object. This means that you must deserialize the saved _state_dict_ before you pass it to the `load_state_dict()` function. For example, you CANNOT load using `model.load_state_dict(PATH)`.

>  注意 `load_state_dict()` 函数接受字典对象，我们需要先反序列化存储好的状态字典，再将其传递给 `load_state_dict()`

> [! Note]
> If you only plan to keep the best performing model (according to the acquired validation loss), don’t forget that `best_model_state = model.state_dict()` returns a reference to the state and not its copy! You must serialize `best_model_state` or use `best_model_state = deepcopy(model.state_dict())` otherwise your best `best_model_state` will keep getting updated by the subsequent training iterations. As a result, the final model state will be the state of the overfitted model.

>  注意 `model.state_dict()` 返回的是对模型状态字典的引用而非拷贝，如果我们需要在验证过程中保存最优的模型，需要使用 `best_model_state = deepcopy(model.state_dict())` 进行拷贝，因为后续的迭代会更新模型的状态字典

### Save/Load Entire Model

**Save:**

```Python
torch.save(model, PATH)
```

**Load:**

```python
# Model class must be defined somewhere
model = torch.load(PATH, weights_only=False)
model.eval()
```

This save/load process uses the most intuitive syntax and involves the least amount of code. Saving a model in this way will save the entire module using Python’s [pickle](https://docs.python.org/3/library/pickle.html) module. The disadvantage of this approach is that the serialized data is bound to the specific classes and the exact directory structure used when the model is saved. The reason for this is because pickle does not save the model class itself. Rather, it saves a path to the file containing the class, which is used during load time. Because of this, your code can break in various ways when used in other projects or after refactors.
>  直接保存模型将使用 `pickle` 模块保存整个模块
>  这种方法的缺点在于序列化后的数据和保存模型时使用的具体类和具体目录结构绑定在一起。这是因为 `pickle` 不会保存模型类本身，而是保存导向包含了模型类的文件的路径，这一路径会在加载时被使用

A common PyTorch convention is to save models using either a `.pt` or `.pth` file extension.

Remember that you must call `model.eval()` to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

### Export/Load Model in TorchScript Format
One common way to do inference with a trained model is to use [TorchScript](https://pytorch.org/docs/stable/jit.html), an intermediate representation of a PyTorch model that can be run in Python as well as in a high performance environment like C++. TorchScript is actually the recommended model format for scaled inference and deployment.
>  使用训练好的模型推理的一种常见方式是使用 TorchScript, TorchScript 是 PyTorch 模型的中间表示形式，既可以在 Python 中，也可以在像 C++ 这样的高性能环境中运行
>  TorchScript 是大规模推理和部署的推荐模型形式

> [! Note]
> Using the TorchScript format, you will be able to load the exported model and run inference without defining the model class.

**Export:**

```python
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save
```

**Load:**

```python
model = torch.jit.load('model_scripted.pt')
model.eval()
```

Remember that you must call `model.eval()` to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

For more information on TorchScript, feel free to visit the dedicated [tutorials](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html). You will get familiar with the tracing conversion and learn how to run a TorchScript module in a [C++ environment](https://pytorch.org/tutorials/advanced/cpp_export.html).

## Saving & Loading a General Checkpoint for Inference and/or Resuming Training
### Save:

```python
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
```

### Load:

```python
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

When saving a general checkpoint, to be used for either inference or resuming training, you must save more than just the model’s _state_dict_. It is important to also save the optimizer’s _state_dict_, as this contains buffers and parameters that are updated as the model trains. Other items that you may want to save are the epoch you left off on, the latest recorded training loss, external `torch.nn.Embedding` layers, etc. As a result, such a checkpoint is often 2~3 times larger than the model alone.
>  保存一个通用的检查点时，除了模型的 `state_dict` 之外的更多内容，还需要保存优化器的 `state_dict` ，因为它也包含了随模型训练更新的缓存和参数
>  还需要保存的其他内容包括：停止时的 epoch 数、最后记录的训练损失、外部的 `torch.nn.Embedding` 层等
>  因此通用的检查点一般比单独的模型大 2-3 倍

To save multiple components, organize them in a dictionary and use `torch.save()` to serialize the dictionary. A common PyTorch convention is to save these checkpoints using the `.tar` file extension.
>  我们将其组织为一个字典，并使用 `torch.save()` 序列化该字典
>  惯例将检查点的文件拓展名记作 `.tar`

To load the items, first initialize the model and optimizer, then load the dictionary locally using `torch.load()`. From here, you can easily access the saved items by simply querying the dictionary as you would expect.
>  加载检查点时，先初始化模型和优化器，然后使用 `torch.load()` 反序列化字典

Remember that you must call `model.eval()` to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results. If you wish to resuming training, call `model.train()` to ensure these layers are in training mode.

## Saving Multiple Models in One File
### Save:

```python
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)
```

### Load:

```python
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH, weights_only=True)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()
```

When saving a model comprised of multiple `torch.nn.Modules`, such as a GAN, a sequence-to-sequence model, or an ensemble of models, you follow the same approach as when you are saving a general checkpoint. In other words, save a dictionary of each model’s _state_dict_ and corresponding optimizer. As mentioned before, you can save any other items that may aid you in resuming training by simply appending them to the dictionary.
>  对于由多个 `nn.Module` 构成的模型，例如 GAN、序列到序列的模型、集成模型等，我们同样自行构造一个字典，存储各个模型的 `state_dict` 和对应的优化器状态

A common PyTorch convention is to save these checkpoints using the `.tar` file extension.
>  PyTorch 惯例是将这类检查点的后缀记作 `.tar`

To load the models, first initialize the models and optimizers, then load the dictionary locally using `torch.load()`. From here, you can easily access the saved items by simply querying the dictionary as you would expect.

Remember that you must call `model.eval()` to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results. If you wish to resuming training, call `model.train()` to set these layers to training mode.

## Warmstarting Model Using Parameters from a Different Model
### Save:

```python
torch.save(modelA.state_dict(), PATH)
```

### Load:

```python
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH, weights_only=True), strict=False)
```

Partially loading a model or loading a partial model are common scenarios when transfer learning or training a new complex model. Leveraging trained parameters, even if only a few are usable, will help to warmstart the training process and hopefully help your model converge much faster than training from scratch.

Whether you are loading from a partial _state_dict_, which is missing some keys, or loading a _state_dict_ with more keys than the model that you are loading into, you can set the `strict` argument to **False** in the `load_state_dict()` function to ignore non-matching keys.
>  部分加载一个模型时，`state_dict` 中可能会有缺失的键，又或者 `state_dict` 中会有更多的键，我们可以令关键字参数 `strict=True` 以忽略不匹配的键

If you want to load parameters from one layer to another, but some keys do not match, simply change the name of the parameter keys in the _state_dict_ that you are loading to match the keys in the model that you are loading into.
>  如果要将一层的参数加载到另一层，但键不匹配，此时可以考虑直接手动改变 `state_dict` 中对应的键

## Saving & Loading Model Across Devices
### Save on GPU, Load on CPU

**Save:**

```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```python
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device, weights_only=True))
```

When loading a model on a CPU that was trained with a GPU, pass `torch.device('cpu')` to the `map_location` argument in the `torch.load()` function. In this case, the storages underlying the tensors are dynamically remapped to the CPU device using the `map_location` argument.
>  将 GPU 模型加载到 CPU 时，向 `torch.load()` 传递 `map_location=torch.device('cpu')` 即可，这会将模型中的张量都动态重新映射到 CPU 设备上

### Save on GPU, Load on GPU
**Save:**

```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```python
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, weights_only=True))
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```

When loading a model on a GPU that was trained and saved on GPU, simply convert the initialized `model` to a CUDA optimized model using `model.to(torch.device('cuda'))`. Also, be sure to use the `.to(torch.device('cuda'))` function on all model inputs to prepare the data for the model. 

Note that calling `my_tensor.to(device)` returns a new copy of `my_tensor` on GPU. It does NOT overwrite `my_tensor`. Therefore, remember to manually overwrite tensors: `my_tensor = my_tensor.to(torch.device('cuda'))`.
>  调用 `my_tensor.to(device)` 返回的是 `my_tensor` 在 GPU 上的一个拷贝，不会覆盖 `my_tensor`，要覆盖，需要写 `my_tensor = my_tensor.to(torch.device('cuda'))` 

### Save on CPU, Load on GPU

**Save:**

```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```python
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, weights_only=True, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```

When loading a model on a GPU that was trained and saved on CPU, set the `map_location` argument in the `torch.load()` function to `cuda:device_id`. This loads the model to a given GPU device. Next, be sure to call `model.to(torch.device('cuda'))` to convert the model’s parameter tensors to CUDA tensors. Finally, be sure to use the `.to(torch.device('cuda'))` function on all model inputs to prepare the data for the CUDA optimized model. 

Note that calling `my_tensor.to(device)` returns a new copy of `my_tensor` on GPU. It does NOT overwrite `my_tensor`. Therefore, remember to manually overwrite tensors: `my_tensor = my_tensor.to(torch.device('cuda'))`.

### Saving `torch.nn.DataParallel` Models
**Save:**

```python
torch.save(model.module.state_dict(), PATH)
```

**Load:**

```python
# Load to whatever device you want
```

`torch.nn.DataParallel` is a model wrapper that enables parallel GPU utilization. To save a `DataParallel` model generically, save the `model.module.state_dict()`. This way, you have the flexibility to load the model any way you want to any device you want.
>  `torch.nn.DataParallel` 是一个模型包装器，用于实现并行 GPU 使用