---
version: 4.55.4
completed: true
---
# Quickstart
Transformers is designed to be fast and easy to use so that everyone can start learning or building with transformer models.

The number of user-facing abstractions is limited to only three classes for instantiating a model, and two APIs for inference or training. This quickstart introduces you to Transformers’ key features and shows you how to:

- load a pretrained model
- run inference with [Pipeline](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/pipelines#transformers.Pipeline)
- fine-tune a model with [Trainer](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer#transformers.Trainer)

>  `transformers` 面向用户的抽象仅有三个用于实例化一个模型的类，以及两个用户推理和训练的 API

## Set up
To start, we recommend creating a Hugging Face [account](https://hf.co/join). An account lets you host and access version controlled models, datasets, and [Spaces](https://hf.co/spaces) on the Hugging Face [Hub](https://hf.co/docs/hub/index), a collaborative platform for discovery and building.

Create a [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) and log in to your account.

Paste your User Access Token into [notebook_login](https://huggingface.co/docs/huggingface_hub/v0.35.0.rc0/en/package_reference/authentication#huggingface_hub.notebook_login) when prompted to log in.

Make sure the [huggingface_hub[cli]](https://huggingface.co/docs/huggingface_hub/guides/cli#getting-started) package is installed and run the command below. Paste your User Access Token when prompted to log in.

```
hf auth login
```

>  为了访问 HuggingFace Hub 资源，需要首先登陆账号

Install a machine learning framework.

```
!pip install torch
```

Then install an up-to-date version of Transformers and some additional libraries from the Hugging Face ecosystem for accessing datasets and vision models, evaluating training, and optimizing training for large models.

```
!pip install -U transformers datasets evaluate accelerate timm
```

## Pretrained models
Each pretrained model inherits from three base classes.

| **Class**                                                                                                                        | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [PretrainedConfig](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/configuration#transformers.PretrainedConfig) | A file that specifies a models attributes such as the number of attention heads or vocabulary size.                                                                                                                                                                                                                                                                                                                                                                                                                |
| [PreTrainedModel](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/model#transformers.PreTrainedModel)           | A model (or architecture) defined by the model attributes from the configuration file. A pretrained model only returns the raw hidden states. For a specific task, use the appropriate model head to convert the raw hidden states into a meaningful result (for example, [LlamaModel](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/llama#transformers.LlamaModel) versus [LlamaForCausalLM](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/llama#transformers.LlamaForCausalLM)). |
| Preprocessor                                                                                                                     | A class for converting raw inputs (text, images, audio, multimodal) into numerical inputs to the model. For example, [PreTrainedTokenizer](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) converts text into tensors and [ImageProcessingMixin](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/image_processor#transformers.ImageProcessingMixin) converts pixels into tensors.                                                    |

>  每个预训练模型都继承三个基类:
>  - `PretrainedConfig`: 指定模型属性的文件，例如 attention heads 数量或词袋大小
>  - `PreTrainedModel`: 由模型属性定义的模型结构，模型仅返回原始隐藏状态，针对特定任务需要使用特定的 head 将状态转化为有意义的结果，例如 `LlamaModel` + `LlamaForCausalLM`
>  - `Preprocessor`: 将原始输入 (文本、图像、语音、多模态) 转化为模型的数字输入的类，例如 `PretrainedTokenizer` 将文本转化为张量

We recommend using the [AutoClass](https://huggingface.co/docs/transformers/model_doc/auto) API to load models and preprocessors because it automatically infers the appropriate architecture for each task and machine learning framework based on the name or path to the pretrained weights and configuration file.
>  推荐用 `AutoClass` API 来加载模型并进行预处理，它会基于预训练权重的名称和配置文件的名称，自动推断针对各种任务适合的模型架构以及 ML 框架

Use [from_pretrained()](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) to load the weights and configuration file from the Hub into the model and preprocessor class.
>  model, preprocessor 类都有 `from_pretrained()` 方法来从 Hub 加载已有的权重和配置文件

When you load a model, configure the following parameters to ensure the model is optimally loaded.

- `device_map="auto"` automatically allocates the model weights to your fastest device first, which is typically the GPU.
- `torch_dtype="auto"` directly initializes the model weights in the data type they’re stored in, which can help avoid loading the weights twice (PyTorch loads weights in `torch.float32` by default).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

>  上述代码加载了模型和 tokenizer

Tokenize the text and return PyTorch tensors with the tokenizer. Move the model to a GPU if it’s available to accelerate inference.

```python
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")
```

>  上述代码执行 tokenizer 得到 PyTorch 张量作为输入

The model is now ready for inference or training.

For inference, pass the tokenized inputs to [generate()](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/text_generation#transformers.GenerationMixin.generate) to generate text. Decode the token ids back into text with [batch_decode()](https://huggingface.co/docs/transformers/v4.55.4/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode).

```python
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
'<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'
```

>  上述代码使用 `generate` 方法进行推理，使用 `batch_decode` 将模型输出的 token ids 解码为文本

Skip ahead to the [Trainer](https://huggingface.co/docs/transformers/quicktour#trainer-api) section to learn how to fine-tune a model.

## Pipeline
The [Pipeline](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/pipelines#transformers.Pipeline) class is the most convenient way to inference with a pretrained model. It supports many tasks such as text generation, image segmentation, automatic speech recognition, document question answering, and more.

Refer to the [Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) API reference for a complete list of available tasks.

Create a [Pipeline](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/pipelines#transformers.Pipeline) object and select a task. By default, [Pipeline](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/pipelines#transformers.Pipeline) downloads and caches a default pretrained model for a given task. Pass the model name to the `model` parameter to choose a specific model.

Set `device="cuda"` to accelerate inference with a GPU.

```python
from transformers import pipeline

pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device="cuda")
```

Prompt [Pipeline](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/pipelines#transformers.Pipeline) with some initial text to generate more text.

```python
pipeline("The secret to baking a good cake is ", max_length=50)
[{'generated_text': 'The secret to baking a good cake is 100% in the batter. The secret to a great cake is the icing.\nThis is why we’ve created the best buttercream frosting reci'}]
```

>  如果要用预训练的模型做推理，`pipeline` 类更加方便，只需要指定任务和模型名称即可

## Trainer
[Trainer](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer#transformers.Trainer) is a complete training and evaluation loop for PyTorch models. It abstracts away a lot of the boilerplate usually involved in manually writing a training loop, so you can start training faster and focus on training design choices. You only need a model, dataset, a preprocessor, and a data collator to build batches of data from the dataset.
>  `Trainer` 类抽象了 PyTorch 模型的训练和评估循环过程，只需要指定模型、数据集、预处理器、数据收集器即可

Use the [TrainingArguments](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer#transformers.TrainingArguments) class to customize the training process. It provides many options for training, evaluation, and more. Experiment with training hyperparameters and features like batch size, learning rate, mixed precision, torch.compile, and more to meet your training needs. You could also use the default training parameters to quickly produce a baseline.
>  `TrainingArguments` 类抽象了各种训练超参数的指定，例如 batch size, lr, 混合精度、`torch.compile` 等
>  要快速开始，使用默认情况即可

Load a model, tokenizer, and dataset for training.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("rotten_tomatoes")
```

>  上述代码加载了模型、预处理器/tokenizer、数据集

Create a function to tokenize the text and convert it into PyTorch tensors. Apply this function to the whole dataset with the [map](https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.Dataset.map) method.

```python
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset, batched=True)
```

>  上述代码将预处理器通过数据集的 `map` 方法注册到数据集上

Load a data collator to create batches of data and pass the tokenizer to it.

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

>  上述代码实例化了数据收集器，并为它注册了 tokenizer

Next, set up [TrainingArguments](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer#transformers.TrainingArguments) with the training features and hyperparameters.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)
```

Finally, pass all these separate components to [Trainer](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer#transformers.Trainer) and call [train()](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer#transformers.Trainer.train) to start.

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

>  上述代码实例化了 `TrainingArgument`，并设定了各种超参数，然后实例化了 `Trainer`，将这些东西都传进去，调用 `train` 即可

Share your model and tokenizer to the Hub with [push_to_hub()](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer#transformers.Trainer.push_to_hub).

```python
trainer.push_to_hub()
```

Congratulations, you just trained your first model with Transformers!

### TensorFlow
Not all pretrained models are available in TensorFlow. Refer to a models API doc to check whether a TensorFlow implementation is supported.

[Trainer](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer#transformers.Trainer) doesn’t work with TensorFlow models, but you can still train a Transformers model implemented in TensorFlow with [Keras](https://keras.io/). Transformers TensorFlow models are a standard [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model), which is compatible with Keras’ [compile](https://keras.io/api/models/model_training_apis/#compile-method) and [fit](https://keras.io/api/models/model_training_apis/#fit-method) methods.

Load a model, tokenizer, and dataset for training.

```python
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

Create a function to tokenize the text and convert it into TensorFlow tensors. Apply this function to the whole dataset with the [map](https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.Dataset.map) method.

```python
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset)
```

Transformers provides the [prepare_tf_dataset()](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset) method to collate and batch a dataset.

```python
tf_dataset = model.prepare_tf_dataset(
    dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
)
```

Finally, call [compile](https://keras.io/api/models/model_training_apis/#compile-method) to configure the model for training and [fit](https://keras.io/api/models/model_training_apis/#fit-method) to start.

```python
from tensorflow.keras.optimizers import Adam

model.compile(optimizer="adam")
model.fit(tf_dataset)
```

## Next steps
Now that you have a better understanding of Transformers and what it offers, it’s time to keep exploring and learning what interests you the most.

- **Base classes**: Learn more about the configuration, model and processor classes. This will help you understand how to create and customize models, preprocess different types of inputs (audio, images, multimodal), and how to share your model.
- **Inference**: Explore the [Pipeline](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/pipelines#transformers.Pipeline) further, inference and chatting with LLMs, agents, and how to optimize inference with your machine learning framework and hardware.
- **Training**: Study the [Trainer](https://huggingface.co/docs/transformers/v4.55.4/en/main_classes/trainer#transformers.Trainer) in more detail, as well as distributed training and optimizing training on specific hardware.
- **Quantization**: Reduce memory and storage requirements with quantization and speed up inference by representing weights with fewer bits.
- **Resources**: Looking for end-to-end recipes for how to train and inference with a model for a specific task? Check out the task recipes!