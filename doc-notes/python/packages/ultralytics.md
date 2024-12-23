---
completed: true
version: 8.3.6
---
# Quickstart
## Install Ultralytics

## Use Ultralytics with CLI
The Ultralytics command line interface (CLI) allows for simple single-line commands without the need for a Python environment. CLI requires no customization or Python code. You can simply run all tasks from the terminal with the `yolo` command. Check out the [CLI Guide](https://docs.ultralytics.com/usage/cli/) to learn more about using YOLO from the command line.
> Ultralytics 提供了命令行接口 (`yolo` 命令)

## Use Ultralytics with Python
YOLO's Python interface allows for seamless integration into your Python projects, making it easy to load, run, and process the model's output. Designed with simplicity and ease of use in mind, the Python interface enables users to quickly implement [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and classification in their projects. This makes YOLO's Python interface an invaluable tool for anyone looking to incorporate these functionalities into their Python projects.

For example, users can load a model, train it, evaluate its performance on a validation set, and even export it to ONNX format with just a few lines of code. Check out the [Python Guide](https://docs.ultralytics.com/usage/python/) to learn more about using YOLO within your Python projects.

Example:

```python
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")
```

## Ultralytics Settings
The Ultralytics library provides a powerful settings management system to enable fine-grained control over your experiments. By making use of the `SettingsManager` housed within the `ultralytics.utils` module, users can readily access and alter their settings. These are stored in a JSON file in the environment user configuration directory, and can be viewed or modified directly within the Python environment or via the Command-Line Interface (CLI).
> 设定通过 `ultralytics.utils` 模块中的 `SettingsManager` 修改
> 设定以 JSON 存储

### Inspecting Settings
To gain insight into the current configuration of your settings, you can view them directly:

View settings:
You can use Python to view your settings. Start by importing the `settings` object from the `ultralytics` module. Print and return settings using the following commands:
> `settings` 对象支持直接打印以及索引访问

```python
from ultralytics import settings

# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"]
```

### Modifying Settings
Ultralytics allows users to easily modify their settings. Changes can be performed in the following ways:

Update settings:
Within the Python environment, call the `update` method on the `settings` object to change your settings:
> 使用 `settings` 对象的 `update` 方法更新设定

```python
from ultralytics import settings

# Update a setting
settings.update({"runs_dir": "/path/to/runs"})

# Update multiple settings
settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

# Reset settings to default values
settings.reset()
```

### Understanding Settings
The table below provides an overview of the settings available for adjustment within Ultralytics. Each setting is outlined along with an example value, the data type, and a brief description.

| Name               | Example Value         | Data Type | Description                                                                                                                                                            |
| ------------------ | --------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `settings_version` | `'0.0.4'`             | `str`     | Ultralytics _settings_ version (different from Ultralytics [pip](https://pypi.org/project/ultralytics/) version)                                                       |
| `datasets_dir`     | `'/path/to/datasets'` | `str`     | The directory where the datasets are stored                                                                                                                            |
| `weights_dir`      | `'/path/to/weights'`  | `str`     | The directory where the model weights are stored                                                                                                                       |
| `runs_dir`         | `'/path/to/runs'`     | `str`     | The directory where the experiment runs are stored                                                                                                                     |
| `uuid`             | `'a1b2c3d4'`          | `str`     | The unique identifier for the current settings                                                                                                                         |
| `sync`             | `True`                | `bool`    | Whether to sync analytics and crashes to HUB                                                                                                                           |
| `api_key`          | `''`                  | `str`     | Ultralytics HUB [API Key](https://hub.ultralytics.com/settings?tab=api+keys)                                                                                           |
| `clearml`          | `True`                | `bool`    | Whether to use [ClearML](https://docs.ultralytics.com/integrations/clearml/) logging                                                                                   |
| `comet`            | `True`                | `bool`    | Whether to use [Comet ML](https://bit.ly/yolov8-readme-comet) for experiment tracking and visualization                                                                |
| `dvc`              | `True`                | `bool`    | Whether to use [DVC for experiment tracking](https://dvc.org/doc/dvclive/ml-frameworks/yolo) and version control                                                       |
| `hub`              | `True`                | `bool`    | Whether to use [Ultralytics HUB](https://hub.ultralytics.com/) integration                                                                                             |
| `mlflow`           | `True`                | `bool`    | Whether to use [MLFlow](https://docs.ultralytics.com/integrations/mlflow/) for experiment tracking                                                                     |
| `neptune`          | `True`                | `bool`    | Whether to use [Neptune](https://neptune.ai/) for experiment tracking                                                                                                  |
| `raytune`          | `True`                | `bool`    | Whether to use [Ray Tune](https://docs.ultralytics.com/integrations/ray-tune/) for [hyperparameter tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning) |
| `tensorboard`      | `True`                | `bool`    | Whether to use [TensorBoard](https://docs.ultralytics.com/integrations/tensorboard/) for visualization                                                                 |
| `wandb`            | `True`                | `bool`    | Whether to use [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/) logging                                                                   |
| `vscode_msg`       | `True`                | `bool`    | When VS Code terminal detected, enables prompt to download [Ultralytics-Snippets](https://docs.ultralytics.com/integrations/vscode/) extension.                        |

As you navigate through your projects or experiments, be sure to revisit these settings to ensure that they are optimally configured for your needs.

# Usage
## Python Usage
Welcome to the YOLO11 Python Usage documentation! This guide is designed to help you seamlessly integrate YOLO11 into your Python projects for [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and classification. Here, you'll learn how to load and use pretrained models, train new models, and perform predictions on images. The easy-to-use Python interface is a valuable resource for anyone looking to incorporate YOLO11 into their Python projects, allowing you to quickly implement advanced object detection capabilities. Let's get started!

For example, users can load a model, train it, evaluate its performance on a validation set, and even export it to ONNX format with just a few lines of code.

```python
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")
```

### [Train](https://docs.ultralytics.com/modes/train/)
Train mode is used for training a YOLO11 model on a custom dataset. In this mode, the model is trained using the specified dataset and hyperparameters. The training process involves optimizing the model's parameters so that it can accurately predict the classes and locations of objects in an image.

Train from pretrained:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # pass any model type
results = model.train(epochs=5)
```

### [Val](https://docs.ultralytics.com/modes/val/)
Val mode is used for validating a YOLO11 model after it has been trained. In this mode, the model is evaluated on a validation set to measure its [accuracy](https://www.ultralytics.com/glossary/accuracy) and generalization performance. This mode can be used to tune the hyperparameters of the model to improve its performance.

Val after training:

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11n.yaml")

# Train the model
model.train(data="coco8.yaml", epochs=5)

# Validate on training data
model.val()
```

### [Predict](https://docs.ultralytics.com/modes/predict/)
Predict mode is used for making predictions using a trained YOLO11 model on new images or videos. In this mode, the model is loaded from a checkpoint file, and the user can provide images or videos to perform inference. The model predicts the classes and locations of objects in the input images or videos.

Predict from source:

```python
import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("model.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="0")
results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("bus.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
im2 = cv2.imread("bus.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
results = model.predict(source=[im1, im2])
```

### [Export](https://docs.ultralytics.com/modes/export/)
Export mode is used for exporting a YOLO11 model to a format that can be used for deployment. In this mode, the model is converted to a format that can be used by other software applications or hardware devices. This mode is useful when deploying the model to production environments.
> export 用于将 YOLO 模型转化为用于部署的格式

Export an official YOLO11n model to ONNX with dynamic batch-size and image-size:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="onnx", dynamic=True)
```

### [Track](https://docs.ultralytics.com/modes/track/)
Track mode is used for tracking objects in real-time using a YOLO11 model. In this mode, the model is loaded from a checkpoint file, and the user can provide a live video stream to perform real-time object tracking. This mode is useful for applications such as surveillance systems or self-driving cars.
> track 模式用于实时对象追踪

Track

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official detection model
model = YOLO("yolo11n-seg.pt")  # load an official segmentation model
model = YOLO("path/to/best.pt")  # load a custom model

# Track with the model
results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
```

### [Benchmark](https://docs.ultralytics.com/modes/benchmark/)
Benchmark mode is used to profile the speed and accuracy of various export formats for YOLO11. The benchmarks provide information on the size of the exported format, its `mAP50-95` metrics (for object detection and segmentation) or `accuracy_top5` metrics (for classification), and the inference time in milliseconds per image across various export formats like ONNX, OpenVINO, TensorRT and others. This information can help users choose the optimal export format for their specific use case based on their requirements for speed and accuracy.
> benchmark 模型用于测试 YOLO11 不同 export 格式的速度和准确率

Benchmark an official YOLO11n model across all export formats.

```python
from ultralytics.utils.benchmarks import benchmark

# Benchmark
benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
```

### Explorer
Explorer API can be used to explore datasets with advanced semantic, vector-similarity and SQL search among other features. It also enabled searching for images based on their content using natural language by utilizing the power of LLMs. The Explorer API allows you to write your own dataset exploration notebooks or scripts to get insights into your datasets.
> explorer API 用于探索数据集，例如使用向量相似度和 SQL 搜索
> exporer API 还支持文搜图

Semantic Search Using Explorer

```python
from ultralytics import Explorer

# create an Explorer object
exp = Explorer(data="coco8.yaml", model="yolo11n.pt")
exp.create_embeddings_table()

similar = exp.get_similar(img="https://ultralytics.com/images/bus.jpg", limit=10)
print(similar.head())

# Search using multiple indices
similar = exp.get_similar(
    img=["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/bus.jpg"], limit=10
)
print(similar.head())
```

### Using Trainers
`YOLO` model class is a high-level wrapper on the Trainer classes. Each YOLO task has its own trainer that inherits from `BaseTrainer`.
> `YOLO` 模型是 Trainer 类的高层包装，每个 YOLO 任务都有自己的继承自 `BaseTrainer` 的任务

Detection Trainer Example

```python
from ultralytics.models.yolo import DetectionPredictor, DetectionTrainer, DetectionValidator

# trainer
trainer = DetectionTrainer(overrides={})
trainer.train()
trained_model = trainer.best

# Validator
val = DetectionValidator(args=...)
val(model=trained_model)

# predictor
pred = DetectionPredictor(overrides={})
pred(source=SOURCE, model=trained_model)

# resume from last weight
overrides["resume"] = trainer.last
trainer = detect.DetectionTrainer(overrides=overrides)
```

You can easily customize Trainers to support custom tasks or explore R&D ideas. Learn more about Customizing `Trainers`, `Validators` and `Predictors` to suit your project needs in the Customization Section.

## Callbacks
Ultralytics framework supports callbacks as entry points in strategic stages of train, val, export, and predict modes. Each callback accepts a `Trainer`, `Validator`, or `Predictor` object depending on the operation type. All properties of these objects can be found in Reference section of the docs.
> Ultralytics 支持指定训练、验证、导出、预测模式中的回调函数，这些回调函数在训练、验证、导出、预测的特定阶段被调用，以提供一些额外的功能
> 回调函数应该接受 `Trainer` , `Validator` 或 `Predictor` 对象，具体决定于其操作类型

### Examples
#### Returning additional information with Prediction
In this example, we want to return the original frame with each result object. Here's how we can do that

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """Handle prediction batch end by combining results with corresponding frames; modifies predictor results."""
    _, image, _, _ = predictor.batch

    # Ensure that image is a list
    image = image if isinstance(image, list) else [image]

    # Combine the prediction results with the corresponding frames
    predictor.results = zip(predictor.results, image)


# Create a YOLO model instance
model = YOLO("yolo11n.pt")

# Add the custom callback to the model
model.add_callback("on_predict_batch_end", on_predict_batch_end)

# Iterate through the results and frames
for result, frame in model.predict():  # or model.track()
    pass
```

### All callbacks 
Here are all supported callbacks. See callbacks [source code](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py) for additional details.

#### Trainer Callbacks

|Callback|Description|
|---|---|
|`on_pretrain_routine_start`|Triggered at the beginning of pre-training routine|
|`on_pretrain_routine_end`|Triggered at the end of pre-training routine|
|`on_train_start`|Triggered when the training starts|
|`on_train_epoch_start`|Triggered at the start of each training [epoch](https://www.ultralytics.com/glossary/epoch)|
|`on_train_batch_start`|Triggered at the start of each training batch|
|`optimizer_step`|Triggered during the optimizer step|
|`on_before_zero_grad`|Triggered before gradients are zeroed|
|`on_train_batch_end`|Triggered at the end of each training batch|
|`on_train_epoch_end`|Triggered at the end of each training epoch|
|`on_fit_epoch_end`|Triggered at the end of each fit epoch|
|`on_model_save`|Triggered when the model is saved|
|`on_train_end`|Triggered when the training process ends|
|`on_params_update`|Triggered when model parameters are updated|
|`teardown`|Triggered when the training process is being cleaned up|

#### Validator Callbacks

|Callback|Description|
|---|---|
|`on_val_start`|Triggered when the validation starts|
|`on_val_batch_start`|Triggered at the start of each validation batch|
|`on_val_batch_end`|Triggered at the end of each validation batch|
|`on_val_end`|Triggered when the validation ends|

#### Predictor Callbacks

|Callback|Description|
|---|---|
|`on_predict_start`|Triggered when the prediction process starts|
|`on_predict_batch_start`|Triggered at the start of each prediction batch|
|`on_predict_postprocess_end`|Triggered at the end of prediction postprocessing|
|`on_predict_batch_end`|Triggered at the end of each prediction batch|
|`on_predict_end`|Triggered when the prediction process ends|

#### Exporter Callbacks

|Callback|Description|
|---|---|
|`on_export_start`|Triggered when the export process starts|
|`on_export_end`|Triggered when the export process ends|

### FAQ

#### What are Ultralytics callbacks and how can I use them?
**Ultralytics callbacks** are specialized entry points triggered during key stages of model operations like training, validation, exporting, and prediction. These callbacks allow for custom functionality at specific points in the process, enabling enhancements and modifications to the workflow. Each callback accepts a `Trainer`, `Validator`, or `Predictor` object, depending on the operation type. For detailed properties of these objects, refer to the [Reference section](https://docs.ultralytics.com/reference/cfg/__init__/).

To use a callback, you can define a function and then add it to the model with the `add_callback` method. Here's an example of how to return additional information during prediction:

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """Handle prediction batch end by combining results with corresponding frames; modifies predictor results."""
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)


model = YOLO("yolo11n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)
for result, frame in model.predict():
    pass
```

#### How can I customize Ultralytics training routine using callbacks?
To customize your Ultralytics training routine using callbacks, you can inject your logic at specific stages of the training process. Ultralytics YOLO provides a variety of training callbacks such as `on_train_start`, `on_train_end`, and `on_train_batch_end`. These allow you to add custom metrics, processing, or logging.

Here's an example of how to log additional metrics at the end of each training epoch:

```python
from ultralytics import YOLO


def on_train_epoch_end(trainer):
    """Custom logic for additional metrics logging at the end of each training epoch."""
    additional_metric = compute_additional_metric(trainer)
    trainer.log({"additional_metric": additional_metric})


model = YOLO("yolo11n.pt")
model.add_callback("on_train_epoch_end", on_train_epoch_end)
model.train(data="coco.yaml", epochs=10)
```

Refer to the [Training Guide](https://docs.ultralytics.com/modes/train/) for more details on how to effectively use training callbacks.

#### Why should I use callbacks during validation in Ultralytics YOLO?
Using **callbacks during validation** in Ultralytics YOLO can enhance model evaluation by allowing custom processing, logging, or metrics calculation. Callbacks such as `on_val_start`, `on_val_batch_end`, and `on_val_end` provide entry points to inject custom logic, ensuring detailed and comprehensive validation processes.

For instance, you might want to log additional validation metrics or save intermediate results for further analysis. Here's an example of how to log custom metrics at the end of validation:

```python
from ultralytics import YOLO


def on_val_end(validator):
    """Log custom metrics at end of validation."""
    custom_metric = compute_custom_metric(validator)
    validator.log({"custom_metric": custom_metric})


model = YOLO("yolo11n.pt")
model.add_callback("on_val_end", on_val_end)
model.val(data="coco.yaml")
```

Check out the [Validation Guide](https://docs.ultralytics.com/modes/val/) for further insights on incorporating callbacks into your validation process.

#### How do I attach a custom callback for the prediction mode in Ultralytics YOLO?
To attach a custom callback for the **prediction mode** in Ultralytics YOLO, you define a callback function and register it with the prediction process. Common prediction callbacks include `on_predict_start`, `on_predict_batch_end`, and `on_predict_end`. These allow for modification of prediction outputs and integration of additional functionalities like data logging or result transformation.

Here is an example where a custom callback is used to log predictions:

```python
from ultralytics import YOLO


def on_predict_end(predictor):
    """Log predictions at the end of prediction."""
    for result in predictor.results:
        log_prediction(result)


model = YOLO("yolo11n.pt")
model.add_callback("on_predict_end", on_predict_end)
results = model.predict(source="image.jpg")
```

For more comprehensive usage, refer to the [Prediction Guide](https://docs.ultralytics.com/modes/predict/) which includes detailed instructions and additional customization options.

#### What are some practical examples of using callbacks in Ultralytics YOLO?
Ultralytics YOLO supports various practical implementations of callbacks to enhance and customize different phases like training, validation, and prediction. Some practical examples include:

1. **Logging Custom Metrics**: Log additional metrics at different stages, such as the end of training or validation epochs.
2. **[Data Augmentation](https://www.ultralytics.com/glossary/data-augmentation)**: Implement custom data transformations or augmentations during prediction or training batches.
3. **Intermediate Results**: Save intermediate results such as predictions or frames for further analysis or visualization.

Example: Combining frames with prediction results during prediction using `on_predict_batch_end`:

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """Combine prediction results with frames."""
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)


model = YOLO("yolo11n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)
for result, frame in model.predict():
    pass
```

Explore the [Complete Callback Reference](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py) to find more options and examples.

## Configuration
YOLO settings and hyperparameters play a critical role in the model's performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy). These settings and hyperparameters can affect the model's behavior at various stages of the model development process, including training, validation, and prediction.

Ultralytics commands use the following syntax:

```python
from ultralytics import YOLO

# Load a YOLO11 model from a pre-trained weights file
model = YOLO("yolo11n.pt")

# Run MODE mode using the custom arguments ARGS (guess TASK)
model.MODE(ARGS)
```

Where:

- `TASK` (optional) is one of ([detect](https://docs.ultralytics.com/tasks/detect/), [segment](https://docs.ultralytics.com/tasks/segment/), [classify](https://docs.ultralytics.com/tasks/classify/), [pose](https://docs.ultralytics.com/tasks/pose/), [obb](https://docs.ultralytics.com/tasks/obb/))
- `MODE` (required) is one of ([train](https://docs.ultralytics.com/modes/train/), [val](https://docs.ultralytics.com/modes/val/), [predict](https://docs.ultralytics.com/modes/predict/), [export](https://docs.ultralytics.com/modes/export/), [track](https://docs.ultralytics.com/modes/track/), [benchmark](https://docs.ultralytics.com/modes/benchmark/))
- `ARGS` (optional) are `arg=value` pairs like `imgsz=640` that override defaults.

Default `ARG` values are defined on this page from the `cfg/defaults.yaml` [file](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml).

#### Tasks
YOLO models can be used for a variety of tasks, including detection, segmentation, classification and pose. These tasks differ in the type of output they produce and the specific problem they are designed to solve.

- **Detect**: For identifying and localizing objects or regions of interest in an image or video.
- **Segment**: For dividing an image or video into regions or pixels that correspond to different objects or classes.
- **Classify**: For predicting the class label of an input image.
- **Pose**: For identifying objects and estimating their keypoints in an image or video.
- **OBB**: Oriented (i.e. rotated) bounding boxes suitable for satellite or medical imagery.

|Argument|Default|Description|
|---|---|---|
|`task`|`'detect'`|Specifies the YOLO task to be executed. Options include `detect` for [object detection](https://www.ultralytics.com/glossary/object-detection), `segment` for segmentation, `classify` for classification, `pose` for pose estimation and `obb` for oriented bounding boxes. Each task is tailored to specific types of output and problems within image and video analysis.|
> 每个任务类型都有自己特定格式的输出

#### Modes
YOLO models can be used in different modes depending on the specific problem you are trying to solve. These modes include:

- **Train**: For training a YOLO11 model on a custom dataset.
- **Val**: For validating a YOLO11 model after it has been trained.
- **Predict**: For making predictions using a trained YOLO11 model on new images or videos.
- **Export**: For exporting a YOLO11 model to a format that can be used for deployment.
- **Track**: For tracking objects in real-time using a YOLO11 model.
- **Benchmark**: For benchmarking YOLO11 exports (ONNX, TensorRT, etc.) speed and accuracy.

|Argument|Default|Description|
|---|---|---|
|`mode`|`'train'`|Specifies the mode in which the YOLO model operates. Options are `train` for model training, `val` for validation, `predict` for inference on new data, `export` for model conversion to deployment formats, `track` for object tracking, and `benchmark` for performance evaluation. Each mode is designed for different stages of the model lifecycle, from development through deployment.|

### Train Settings
The training settings for YOLO models encompass various hyperparameters and configurations used during the training process. These settings influence the model's performance, speed, and accuracy. 
Key training settings include batch size, [learning rate](https://www.ultralytics.com/glossary/learning-rate), momentum, and weight decay. Additionally, the choice of optimizer, [loss function](https://www.ultralytics.com/glossary/loss-function), and training dataset composition can impact the training process. Careful tuning and experimentation with these settings are crucial for optimizing performance.

| Argument          | Default  | Description                                                                                                                                                                                                                                                  |
| ----------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model`           | `None`   | Specifies the model file for training. Accepts a path to either a `.pt` pretrained model or a `.yaml` configuration file. Essential for defining the model structure or initializing weights.                                                                |
| `data`            | `None`   | Path to the dataset configuration file (e.g., `coco8.yaml`). This file contains dataset-specific parameters, including paths to training and [validation data](https://www.ultralytics.com/glossary/validation-data), class names, and number of classes.    |
| `epochs`          | `100`    | Total number of training epochs. Each [epoch](https://www.ultralytics.com/glossary/epoch) represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.                                           |
| `time`            | `None`   | Maximum training time in hours. If set, this overrides the `epochs` argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.                                                          |
| `patience`        | `100`    | Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent [overfitting](https://www.ultralytics.com/glossary/overfitting) by stopping training when performance plateaus.                         |
| `batch`           | `16`     | [Batch size](https://www.ultralytics.com/glossary/batch-size), with three modes: set as an integer (e.g., `batch=16`), auto mode for 60% GPU memory utilization (`batch=-1`), or auto mode with specified utilization fraction (`batch=0.70`).               |
| `imgsz`           | `640`    | Target image size for training. All images are resized to this dimension before being fed into the model. Affects model [accuracy](https://www.ultralytics.com/glossary/accuracy) and computational complexity.                                              |
| `save`            | `True`   | Enables saving of training checkpoints and final model weights. Useful for resuming training or [model deployment](https://www.ultralytics.com/glossary/model-deployment).                                                                                   |
| `save_period`     | `-1`     | Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.                                                                                             |
| `cache`           | `False`  | Enables caching of dataset images in memory (`True`/`ram`), on disk (`disk`), or disables it (`False`). Improves training speed by reducing disk I/O at the cost of increased memory usage.                                                                  |
| `device`          | `None`   | Specifies the computational device(s) for training: a single GPU (`device=0`), multiple GPUs (`device=0,1`), CPU (`device=cpu`), or MPS for Apple silicon (`device=mps`).                                                                                    |
| `workers`         | `8`      | Number of worker threads for data loading (per `RANK` if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.                                                                  |
| `project`         | `None`   | Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.                                                                                                                                       |
| `name`            | `None`   | Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.                                                                                                                            |
| `exist_ok`        | `False`  | If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation without needing to manually clear previous outputs.                                                                                                  |
| `pretrained`      | `True`   | Determines whether to start training from a pretrained model. Can be a boolean value or a string path to a specific model from which to load weights. Enhances training efficiency and model performance.                                                    |
| `optimizer`       | `'auto'` | Choice of optimizer for training. Options include `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp` etc., or `auto` for automatic selection based on model configuration. Affects convergence speed and stability.                                         |
| `verbose`         | `False`  | Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process.                                                                                                      |
| `seed`            | `0`      | Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.                                                                                                                                             |
| `deterministic`   | `True`   | Forces deterministic algorithm use, ensuring reproducibility but may affect performance and speed due to the restriction on non-deterministic algorithms.                                                                                                    |
| `single_cls`      | `False`  | Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.                                                                         |
| `rect`            | `False`  | Enables rectangular training, optimizing batch composition for minimal padding. Can improve efficiency and speed but may affect model accuracy.                                                                                                              |
| `cos_lr`          | `False`  | Utilizes a cosine [learning rate](https://www.ultralytics.com/glossary/learning-rate) scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.                                   |
| `close_mosaic`    | `10`     | Disables mosaic [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.                                                                |
| `resume`          | `False`  | Resumes training from the last saved checkpoint. Automatically loads model weights, optimizer state, and epoch count, continuing training seamlessly.                                                                                                        |
| `amp`             | `True`   | Enables Automatic [Mixed Precision](https://www.ultralytics.com/glossary/mixed-precision) (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.                                                           |
| `fraction`        | `1.0`    | Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.                                                                                        |
| `profile`         | `False`  | Enables profiling of ONNX and TensorRT speeds during training, useful for optimizing model deployment.                                                                                                                                                       |
| `freeze`          | `None`   | Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or [transfer learning](https://www.ultralytics.com/glossary/transfer-learning).                                    |
| `lr0`             | `0.01`   | Initial learning rate (i.e. `SGD=1E-2`, `Adam=1E-3`) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.                                                                                      |
| `lrf`             | `0.01`   | Final learning rate as a fraction of the initial rate = (`lr0 * lrf`), used in conjunction with schedulers to adjust the learning rate over time.                                                                                                            |
| `momentum`        | `0.937`  | Momentum factor for SGD or beta1 for [Adam optimizers](https://www.ultralytics.com/glossary/adam-optimizer), influencing the incorporation of past gradients in the current update.                                                                          |
| `weight_decay`    | `0.0005` | L2 [regularization](https://www.ultralytics.com/glossary/regularization) term, penalizing large weights to prevent overfitting.                                                                                                                              |
| `warmup_epochs`   | `3.0`    | Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.                                                                                              |
| `warmup_momentum` | `0.8`    | Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.                                                                                                                                                           |
| `warmup_bias_lr`  | `0.1`    | Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.                                                                                                                                           |
| `box`             | `7.5`    | Weight of the box loss component in the [loss function](https://www.ultralytics.com/glossary/loss-function), influencing how much emphasis is placed on accurately predicting [bounding box](https://www.ultralytics.com/glossary/bounding-box) coordinates. |
| `cls`             | `0.5`    | Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.                                                                                                             |
| `dfl`             | `1.5`    | Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.                                                                                                                                                        |
| `pose`            | `12.0`   | Weight of the pose loss in models trained for pose estimation, influencing the emphasis on accurately predicting pose keypoints.                                                                                                                             |
| `kobj`            | `2.0`    | Weight of the keypoint objectness loss in pose estimation models, balancing detection confidence with pose accuracy.                                                                                                                                         |
| `label_smoothing` | `0.0`    | Applies label smoothing, softening hard labels to a mix of the target label and a uniform distribution over labels, can improve generalization.                                                                                                              |
| `nbs`             | `64`     | Nominal batch size for normalization of loss.                                                                                                                                                                                                                |
| `overlap_mask`    | `True`   | Determines whether segmentation masks should overlap during training, applicable in [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) tasks.                                                                               |
| `mask_ratio`      | `4`      | Downsample ratio for segmentation masks, affecting the resolution of masks used during training.                                                                                                                                                             |
| `dropout`         | `0.0`    | Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.                                                                                                                                  |
| `val`             | `True`   | Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.                                                                                                                                             |
| `plots`           | `False`  | Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.                                                                                     |


Note on Batch-size Settings

The `batch` argument can be configured in three ways:

- **Fixed [Batch Size](https://www.ultralytics.com/glossary/batch-size)**: Set an integer value (e.g., `batch=16`), specifying the number of images per batch directly.
- **Auto Mode (60% GPU Memory)**: Use `batch=-1` to automatically adjust batch size for approximately 60% CUDA memory utilization.
- **Auto Mode with Utilization Fraction**: Set a fraction value (e.g., `batch=0.70`) to adjust batch size based on the specified fraction of GPU memory usage.

### Predict Settings
The prediction settings for YOLO models encompass a range of hyperparameters and configurations that influence the model's performance, speed, and accuracy during inference on new data. Careful tuning and experimentation with these settings are essential to achieve optimal performance for a specific task. 
Key settings include the confidence threshold, Non-Maximum Suppression (NMS) threshold, and the number of classes considered. 
Additional factors affecting the prediction process are input data size and format, the presence of supplementary features such as masks or multiple labels per box, and the particular task the model is employed for.

Inference arguments:

|Argument|Type|Default|Description|
|---|---|---|---|
|`source`|`str`|`'ultralytics/assets'`|Specifies the data source for inference. Can be an image path, video file, directory, URL, or device ID for live feeds. Supports a wide range of formats and sources, enabling flexible application across [different types of input](https://docs.ultralytics.com/modes/predict/#inference-sources).|
|`conf`|`float`|`0.25`|Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.|
|`iou`|`float`|`0.7`|[Intersection Over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.|
|`imgsz`|`int or tuple`|`640`|Defines the image size for inference. Can be a single integer `640` for square resizing or a (height, width) tuple. Proper sizing can improve detection [accuracy](https://www.ultralytics.com/glossary/accuracy) and processing speed.|
|`half`|`bool`|`False`|Enables half-[precision](https://www.ultralytics.com/glossary/precision) (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.|
|`device`|`str`|`None`|Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.|
|`max_det`|`int`|`300`|Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes.|
|`vid_stride`|`int`|`1`|Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames.|
|`stream_buffer`|`bool`|`False`|Determines whether to queue incoming frames for video streams. If `False`, old frames get dropped to accomodate new frames (optimized for real-time applications). If `True', queues new frames in a buffer, ensuring no frames get skipped, but will cause latency if inference FPS is lower than stream FPS.|
|`visualize`|`bool`|`False`|Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation.|
|`augment`|`bool`|`False`|Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.|
|`agnostic_nms`|`bool`|`False`|Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.|
|`classes`|`list[int]`|`None`|Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.|
|`retina_masks`|`bool`|`False`|Uses high-resolution segmentation masks if available in the model. This can enhance mask quality for segmentation tasks, providing finer detail.|
|`embed`|`list[int]`|`None`|Specifies the layers from which to extract feature vectors or [embeddings](https://www.ultralytics.com/glossary/embeddings). Useful for downstream tasks like clustering or similarity search.|

Visualization arguments:

|Argument|Type|Default|Description|
|---|---|---|---|
|`show`|`bool`|`False`|If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing.|
|`save`|`bool`|`False` or `True`|Enables saving of the annotated images or videos to file. Useful for documentation, further analysis, or sharing results. Defaults to True when using CLI & False when used in Python.|
|`save_frames`|`bool`|`False`|When processing videos, saves individual frames as images. Useful for extracting specific frames or for detailed frame-by-frame analysis.|
|`save_txt`|`bool`|`False`|Saves detection results in a text file, following the format `[class] [x_center] [y_center] [width] [height] [confidence]`. Useful for integration with other analysis tools.|
|`save_conf`|`bool`|`False`|Includes confidence scores in the saved text files. Enhances the detail available for post-processing and analysis.|
|`save_crop`|`bool`|`False`|Saves cropped images of detections. Useful for dataset augmentation, analysis, or creating focused datasets for specific objects.|
|`show_labels`|`bool`|`True`|Displays labels for each detection in the visual output. Provides immediate understanding of detected objects.|
|`show_conf`|`bool`|`True`|Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection.|
|`show_boxes`|`bool`|`True`|Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames.|
|`line_width`|`None` or `int`|`None`|Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity.|

### Validation Settings
The val (validation) settings for YOLO models involve various hyperparameters and configurations used to evaluate the model's performance on a validation dataset. These settings influence the model's performance, speed, and accuracy. 
Common YOLO validation settings include batch size, validation frequency during training, and performance evaluation metrics. Other factors affecting the validation process include the validation dataset's size and composition, as well as the specific task the model is employed for.

| Argument      | Type    | Default | Description                                                                                                                                                                                                                           |
| ------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`        | `str`   | `None`  | Specifies the path to the dataset configuration file (e.g., `coco8.yaml`). This file includes paths to [validation data](https://www.ultralytics.com/glossary/validation-data), class names, and number of classes.                   |
| `imgsz`       | `int`   | `640`   | Defines the size of input images. All images are resized to this dimension before processing.                                                                                                                                         |
| `batch`       | `int`   | `16`    | Sets the number of images per batch. Use `-1` for AutoBatch, which automatically adjusts based on GPU memory availability.                                                                                                            |
| `save_json`   | `bool`  | `False` | If `True`, saves the results to a JSON file for further analysis or integration with other tools.                                                                                                                                     |
| `save_hybrid` | `bool`  | `False` | If `True`, saves a hybrid version of labels that combines original annotations with additional model predictions.                                                                                                                     |
| `conf`        | `float` | `0.001` | Sets the minimum confidence threshold for detections. Detections with confidence below this threshold are discarded.                                                                                                                  |
| `iou`         | `float` | `0.6`   | Sets the [Intersection Over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for Non-Maximum Suppression (NMS). Helps in reducing duplicate detections.                                       |
| `max_det`     | `int`   | `300`   | Limits the maximum number of detections per image. Useful in dense scenes to prevent excessive detections.                                                                                                                            |
| `half`        | `bool`  | `True`  | Enables half-[precision](https://www.ultralytics.com/glossary/precision) (FP16) computation, reducing memory usage and potentially increasing speed with minimal impact on [accuracy](https://www.ultralytics.com/glossary/accuracy). |
| `device`      | `str`   | `None`  | Specifies the device for validation (`cpu`, `cuda:0`, etc.). Allows flexibility in utilizing CPU or GPU resources.                                                                                                                    |
| `dnn`         | `bool`  | `False` | If `True`, uses the [OpenCV](https://www.ultralytics.com/glossary/opencv) DNN module for ONNX model inference, offering an alternative to [PyTorch](https://www.ultralytics.com/glossary/pytorch) inference methods.                  |
| `plots`       | `bool`  | `False` | When set to `True`, generates and saves plots of predictions versus ground truth for visual evaluation of the model's performance.                                                                                                    |
| `rect`        | `bool`  | `False` | If `True`, uses rectangular inference for batching, reducing padding and potentially increasing speed and efficiency.                                                                                                                 |
| `split`       | `str`   | `val`   | Determines the dataset split to use for validation (`val`, `test`, or `train`). Allows flexibility in choosing the data segment for performance evaluation.                                                                           |

Careful tuning and experimentation with these settings are crucial to ensure optimal performance on the validation dataset and detect and prevent [overfitting](https://www.ultralytics.com/glossary/overfitting).

### Export Settings
Export settings for YOLO models encompass configurations and options related to saving or exporting the model for use in different environments or platforms. These settings can impact the model's performance, size, and compatibility with various systems. 
Key export settings include the exported model file format (e.g., ONNX, TensorFlow SavedModel), the target device (e.g., CPU, GPU), and additional features such as masks or multiple labels per box. The export process may also be affected by the model's specific task and the requirements or constraints of the destination environment or platform.

|Argument|Type|Default|Description|
|---|---|---|---|
|`format`|`str`|`'torchscript'`|Target format for the exported model, such as `'onnx'`, `'torchscript'`, `'tensorflow'`, or others, defining compatibility with various deployment environments.|
|`imgsz`|`int` or `tuple`|`640`|Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.|
|`keras`|`bool`|`False`|Enables export to Keras format for [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) SavedModel, providing compatibility with TensorFlow serving and APIs.|
|`optimize`|`bool`|`False`|Applies optimization for mobile devices when exporting to TorchScript, potentially reducing model size and improving performance.|
|`half`|`bool`|`False`|Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.|
|`int8`|`bool`|`False`|Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for edge devices.|
|`dynamic`|`bool`|`False`|Allows dynamic input sizes for ONNX, TensorRT and OpenVINO exports, enhancing flexibility in handling varying image dimensions.|
|`simplify`|`bool`|`True`|Simplifies the model graph for ONNX exports with `onnxslim`, potentially improving performance and compatibility.|
|`opset`|`int`|`None`|Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version.|
|`workspace`|`float`|`4.0`|Sets the maximum workspace size in GiB for TensorRT optimizations, balancing memory usage and performance.|
|`nms`|`bool`|`False`|Adds Non-Maximum Suppression (NMS) to the CoreML export, essential for accurate and efficient detection post-processing.|
|`batch`|`int`|`1`|Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.|

It is crucial to thoughtfully configure these settings to ensure the exported model is optimized for the intended use case and functions effectively in the target environment.


### Augmentation Settings
Augmentation techniques are essential for improving the robustness and performance of YOLO models by introducing variability into the [training data](https://www.ultralytics.com/glossary/training-data), helping the model generalize better to unseen data. The following table outlines the purpose and effect of each augmentation argument:
> 数据增强

|Argument|Type|Default|Range|Description|
|---|---|---|---|---|
|`hsv_h`|`float`|`0.015`|`0.0 - 1.0`|Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.|
|`hsv_s`|`float`|`0.7`|`0.0 - 1.0`|Alters the saturation of the image by a fraction, affecting the intensity of colors. Useful for simulating different environmental conditions.|
|`hsv_v`|`float`|`0.4`|`0.0 - 1.0`|Modifies the value (brightness) of the image by a fraction, helping the model to perform well under various lighting conditions.|
|`degrees`|`float`|`0.0`|`-180 - +180`|Rotates the image randomly within the specified degree range, improving the model's ability to recognize objects at various orientations.|
|`translate`|`float`|`0.1`|`0.0 - 1.0`|Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.|
|`scale`|`float`|`0.5`|`>=0.0`|Scales the image by a gain factor, simulating objects at different distances from the camera.|
|`shear`|`float`|`0.0`|`-180 - +180`|Shears the image by a specified degree, mimicking the effect of objects being viewed from different angles.|
|`perspective`|`float`|`0.0`|`0.0 - 0.001`|Applies a random perspective transformation to the image, enhancing the model's ability to understand objects in 3D space.|
|`flipud`|`float`|`0.0`|`0.0 - 1.0`|Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.|
|`fliplr`|`float`|`0.5`|`0.0 - 1.0`|Flips the image left to right with the specified probability, useful for learning symmetrical objects and increasing dataset diversity.|
|`bgr`|`float`|`0.0`|`0.0 - 1.0`|Flips the image channels from RGB to BGR with the specified probability, useful for increasing robustness to incorrect channel ordering.|
|`mosaic`|`float`|`1.0`|`0.0 - 1.0`|Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.|
|`mixup`|`float`|`0.0`|`0.0 - 1.0`|Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.|
|`copy_paste`|`float`|`0.0`|`0.0 - 1.0`|Copies objects from one image and pastes them onto another, useful for increasing object instances and learning object occlusion.|
|`copy_paste_mode`|`str`|`flip`|-|Copy-Paste augmentation method selection among the options of (`"flip"`, `"mixup"`).|
|`auto_augment`|`str`|`randaugment`|-|Automatically applies a predefined augmentation policy (`randaugment`, `autoaugment`, `augmix`), optimizing for classification tasks by diversifying the visual features.|
|`erasing`|`float`|`0.4`|`0.0 - 0.9`|Randomly erases a portion of the image during classification training, encouraging the model to focus on less obvious features for recognition.|
|`crop_fraction`|`float`|`1.0`|`0.1 - 1.0`|Crops the classification image to a fraction of its size to emphasize central features and adapt to object scales, reducing background distractions.|

These settings can be adjusted to meet the specific requirements of the dataset and task at hand. Experimenting with different values can help find the optimal augmentation strategy that leads to the best model performance.

### Logging, Checkpoints and Plotting Settings
Logging, checkpoints, plotting, and file management are important considerations when training a YOLO model.

- Logging: It is often helpful to log various metrics and statistics during training to track the model's progress and diagnose any issues that may arise. This can be done using a logging library such as TensorBoard or by writing log messages to a file.
- Checkpoints: It is a good practice to save checkpoints of the model at regular intervals during training. This allows you to resume training from a previous point if the training process is interrupted or if you want to experiment with different training configurations.
- Plotting: Visualizing the model's performance and training progress can be helpful for understanding how the model is behaving and identifying potential issues. This can be done using a plotting library such as matplotlib or by generating plots using a logging library such as TensorBoard.
- File management: Managing the various files generated during the training process, such as model checkpoints, log files, and plots, can be challenging. It is important to have a clear and organized file structure to keep track of these files and make it easy to access and analyze them as needed.

Effective logging, checkpointing, plotting, and file management can help you keep track of the model's progress and make it easier to debug and optimize the training process.

| Argument   | Default  | Description                                                                                                                                                                                                                                                                                                                         |
| ---------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project`  | `'runs'` | Specifies the root directory for saving training runs. Each run will be saved in a separate subdirectory within this directory.                                                                                                                                                                                                     |
| `name`     | `'exp'`  | Defines the name of the experiment. If not specified, YOLO automatically increments this name for each run, e.g., `exp`, `exp2`, etc., to avoid overwriting previous experiments.                                                                                                                                                   |
| `exist_ok` | `False`  | Determines whether to overwrite an existing experiment directory if one with the same name already exists. Setting this to `True` allows overwriting, while `False` prevents it.                                                                                                                                                    |
| `plots`    | `False`  | Controls the generation and saving of training and validation plots. Set to `True` to create plots such as loss curves, [precision](https://www.ultralytics.com/glossary/precision)-[recall](https://www.ultralytics.com/glossary/recall) curves, and sample predictions. Useful for visually tracking model performance over time. |
| `save`     | `False`  | Enables the saving of training checkpoints and final model weights. Set to `True` to periodically save model states, allowing training to be resumed from these checkpoints or models to be deployed.                                                                                                                               |

### FAQ
#### How do I improve the performance of my YOLO model during training?
Improving YOLO model performance involves tuning hyperparameters like batch size, learning rate, momentum, and weight decay. Adjusting augmentation settings, selecting the right optimizer, and employing techniques like early stopping or [mixed precision](https://www.ultralytics.com/glossary/mixed-precision) can also help. For detailed guidance on training settings, refer to the [Train Guide](https://docs.ultralytics.com/modes/train/).

#### What are the key hyperparameters to consider for YOLO model accuracy?
Key hyperparameters affecting YOLO model accuracy include:

- **Batch Size (`batch`)**: Larger batch sizes can stabilize training but may require more memory.
- **Learning Rate (`lr0`)**: Controls the step size for weight updates; smaller rates offer fine adjustments but slow convergence.
- **Momentum (`momentum`)**: Helps accelerate gradient vectors in the right directions, dampening oscillations.
- **Image Size (`imgsz`)**: Larger image sizes can improve accuracy but increase computational load.

Adjust these values based on your dataset and hardware capabilities. Explore more in the [Train Settings](https://docs.ultralytics.com/usage/cfg/#train-settings) section.
#### How do I set the learning rate for training a YOLO model?
The learning rate (`lr0`) is crucial for optimization. A common starting point is `0.01` for SGD or `0.001` for Adam. It's essential to monitor training metrics and adjust if necessary. Use cosine learning rate schedulers (`cos_lr`) or warmup techniques (`warmup_epochs`, `warmup_momentum`) to dynamically modify the rate during training. Find more details in the [Train Guide](https://docs.ultralytics.com/modes/train/).

#### What are the default inference settings for YOLO models?
Default inference settings include:

- **Confidence Threshold (`conf=0.25`)**: Minimum confidence for detections.
- **IoU Threshold (`iou=0.7`)**: For Non-Maximum Suppression (NMS).
- **Image Size (`imgsz=640`)**: Resizes input images prior to inference.
- **Device (`device=None`)**: Selects CPU or GPU for inference. For a comprehensive overview, visit the [Predict Settings](https://docs.ultralytics.com/usage/cfg/#predict-settings) section and the [Predict Guide](https://docs.ultralytics.com/modes/predict/).

#### Why should I use mixed precision training with YOLO models?
Mixed precision training, enabled with `amp=True`, helps reduce memory usage and can speed up training by utilizing the advantages of both FP16 and FP32. This is beneficial for modern GPUs, which support mixed precision natively, allowing more models to fit in memory and enable faster computations without significant loss in accuracy. Learn more about this in the [Train Guide](https://docs.ultralytics.com/modes/train/).

## Simple Utilities
The `ultralytics` package comes with a myriad of utilities that can support, enhance, and speed up your workflows. There are many more available, but here are some that will be useful for most developers. They're also a great reference point to use when learning to program.
> `ultralytics` 还提供了一系列实用程序

### Data
#### YOLO Data Explorer
[YOLO Explorer](https://docs.ultralytics.com/datasets/explorer/) was added in the `8.1.0` anniversary update and is a powerful tool you can use to better understand your dataset. One of the key functions that YOLO Explorer provides, is the ability to use text queries to find object instances in your dataset.
> YOLO Explorer 主要用于帮助理解数据集，支持使用文本 query 以在数据集中寻找对象实例

#### Auto Labeling / Annotations
Dataset annotation is a very resource intensive and time-consuming process. If you have a YOLO [object detection](https://www.ultralytics.com/glossary/object-detection) model trained on a reasonable amount of data, you can use it and [SAM](https://docs.ultralytics.com/models/sam/) to auto-annotate additional data (segmentation format).
> `ultralyltics` 支持使用训练好的 YOLO 目标检测模型和 SAM 来自动标注数据（分割格式）

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(  
    data="path/to/new/data",
    det_model="yolo11n.pt",
    sam_model="mobile_sam.pt",
    device="cuda",
    output_dir="path/to/save_labels",
)
```

#### Convert Segmentation Masks into YOLO Format

![Segmentation Masks to YOLO Format](https://github.com/ultralytics/docs/releases/download/0/segmentation-masks-to-yolo-format.avif)

Use to convert a dataset of segmentation mask images to the `YOLO` segmentation format. This function takes the directory containing the binary format mask images and converts them into YOLO segmentation format.
> `ultralytics` 支持将分割 mask 图像格式的数据集转化为 YOLO 分割格式
> 函数的输入是包含了二进制格式 mask 图像的目录

The converted masks will be saved in the specified output directory.

```python
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
convert_segment_masks_to_yolo_seg(masks_dir="path/to/masks_dir", output_dir="path/to/output_dir", classes=80)
```

#### Convert COCO into YOLO Format
Use to convert COCO JSON annotations into proper YOLO format. For object detection (bounding box) datasets, `use_segments` and `use_keypoints` should both be `False`
> 将 COCO JSON 标注转化为 YOLO 格式

```python
from ultralytics.data.converter import convert_coco

convert_coco(  
    "../datasets/coco/annotations/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
)
```

For additional information about the `convert_coco` function, [visit the reference page](https://docs.ultralytics.com/reference/data/converter/#ultralytics.data.converter.convert_coco)

#### Get [Bounding Box](https://www.ultralytics.com/glossary/bounding-box) Dimensions

```python
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')  # Load pretrain or fine-tune model

# Process the image
source = cv2.imread('path/to/image.jpg')
results = model(source)

# Extract results
annotator = Annotator(source, example=model.names)

for box in results[0].boxes.xyxy.cpu():
    width, height, area = annotator.get_bbox_dimension(box)
    print("Bounding Box Width {}, Height {}, Area {}".format(
        width.item(), height.item(), area.item()))
```

#### Convert Bounding Boxes to Segments
With existing `x y w h` bounding box data, convert to segments using the `yolo_bbox2segment` function. The files for images and annotations need to be organized like this:

```
data
|__ images
    ├─ 001.jpg
    ├─ 002.jpg
    ├─ ..
    └─ NNN.jpg
|__ labels
    ├─ 001.txt
    ├─ 002.txt
    ├─ ..
    └─ NNN.txt
```

```python
from ultralytics.data.converter import yolo_bbox2segment

yolo_bbox2segment(  
    im_dir="path/to/images",
    save_dir=None,  # saved to "labels-segment" in images directory
    sam_model="sam_b.pt",
)
```

[Visit the `yolo_bbox2segment` reference page](https://docs.ultralytics.com/reference/data/converter/#ultralytics.data.converter.yolo_bbox2segment) for more information regarding the function.

#### Convert Segments to Bounding Boxes
If you have a dataset that uses the [segmentation dataset format](https://docs.ultralytics.com/datasets/segment/) you can easily convert these into up-right (or horizontal) bounding boxes (`x y w h` format) with this function.

```python
import numpy as np

from ultralytics.utils.ops import segments2boxes

segments = np.array(
    [
        [805, 392, 797, 400, ..., 808, 714, 808, 392],
        [115, 398, 113, 400, ..., 150, 400, 149, 298],
        [267, 412, 265, 413, ..., 300, 413, 299, 412],
    ]
)

segments2boxes([s.reshape(-1, 2) for s in segments])
# >>> array([[ 741.66, 631.12, 133.31, 479.25],
#           [ 146.81, 649.69, 185.62, 502.88],
#           [ 281.81, 636.19, 118.12, 448.88]],
#           dtype=float32) # xywh bounding boxes
```

To understand how this function works, visit the [reference page](https://docs.ultralytics.com/reference/utils/ops/#ultralytics.utils.ops.segments2boxes)

### Utilities
#### Image Compression
Compresses a single image file to reduced size while preserving its aspect ratio and quality. If the input image is smaller than the maximum dimension, it will not be resized.
> 将单张图像压缩到指定大小，保持其长宽比和质量

```python
from pathlib import Path

from ultralytics.data.utils import compress_one_image

for f in Path("path/to/dataset").rglob("*.jpg"):
    compress_one_image(f)
```

#### Auto-split Dataset
Automatically split a dataset into `train` / `val` / `test` splits and save the resulting splits into `autosplit_*.txt` files. This function will use random sampling, which is not included when using [`fraction` argument for training](https://docs.ultralytics.com/modes/train/#train-settings).
> 自动将数据集随机划分训练、验证、测试

```python
from ultralytics.data.utils import autosplit

autosplit(  
    path="path/to/images",
    weights=(0.9, 0.1, 0.0),  # (train, validation, test) fractional splits
    annotated_only=False,  # split only images with annotation file when True
)
```

See the [Reference page](https://docs.ultralytics.com/reference/data/utils/#ultralytics.data.utils.autosplit) for additional details on this function.

#### Segment-polygon to Binary Mask
Convert a single polygon (as list) to a binary mask of the specified image size. Polygon in the form of `[N, 2]` with `N` as the number of `(x, y)` points defining the polygon contour.

Warning
`N` **must always** be even.

```python
import numpy as np

from ultralytics.data.utils import polygon2mask

imgsz = (1080, 810)
polygon = np.array([805, 392, 797, 400, ..., 808, 714, 808, 392])  # (238, 2)

mask = polygon2mask(
    imgsz,  # tuple
    [polygon],  # input as list
    color=255,  # 8-bit binary
    downsample_ratio=1,
)
```

### Bounding Boxes
#### Bounding Box (horizontal) Instances
To manage bounding box data, the `Bboxes` class will help to convert between box coordinate formatting, scale box dimensions, calculate areas, include offsets, and more!

```python
import numpy as np

from ultralytics.utils.instance import Bboxes

boxes = Bboxes(
    bboxes=np.array(
        [
            [22.878, 231.27, 804.98, 756.83],
            [48.552, 398.56, 245.35, 902.71],
            [669.47, 392.19, 809.72, 877.04],
            [221.52, 405.8, 344.98, 857.54],
            [0, 550.53, 63.01, 873.44],
            [0.0584, 254.46, 32.561, 324.87],
        ]
    ),
    format="xyxy",
)

boxes.areas()
# >>> array([ 4.1104e+05,       99216,       68000,       55772,       20347,      2288.5])

boxes.convert("xywh")
print(boxes.bboxes)
# >>> array(
#     [[ 413.93, 494.05,  782.1, 525.56],
#      [ 146.95, 650.63,  196.8, 504.15],
#      [  739.6, 634.62, 140.25, 484.85],
#      [ 283.25, 631.67, 123.46, 451.74],
#      [ 31.505, 711.99,  63.01, 322.91],
#      [  16.31, 289.67, 32.503,  70.41]]
# )
```

See the [`Bboxes` reference section](https://docs.ultralytics.com/reference/utils/instance/#ultralytics.utils.instance.Bboxes) for more attributes and methods available.

Tip
Many of the following functions (and more) can be accessed using the [`Bboxes` class](https://docs.ultralytics.com/usage/simple-utilities/#bounding-box-horizontal-instances) but if you prefer to work with the functions directly, see the next subsections on how to import these independently.

#### Scaling Boxes
When scaling and image up or down, corresponding bounding box coordinates can be appropriately scaled to match using `ultralytics.utils.ops.scale_boxes`.

```python
import cv2 as cv
import numpy as np

from ultralytics.utils.ops import scale_boxes

image = cv.imread("ultralytics/assets/bus.jpg")
h, w, c = image.shape
resized = cv.resize(image, None, (), fx=1.2, fy=1.2)
new_h, new_w, _ = resized.shape

xyxy_boxes = np.array(
    [
        [22.878, 231.27, 804.98, 756.83],
        [48.552, 398.56, 245.35, 902.71],
        [669.47, 392.19, 809.72, 877.04],
        [221.52, 405.8, 344.98, 857.54],
        [0, 550.53, 63.01, 873.44],
        [0.0584, 254.46, 32.561, 324.87],
    ]
)

new_boxes = scale_boxes(
    img1_shape=(h, w),  # original image dimensions
    boxes=xyxy_boxes,  # boxes from original image
    img0_shape=(new_h, new_w),  # resized image dimensions (scale to)
    ratio_pad=None,
    padding=False,
    xywh=False,
)

print(new_boxes)  
# >>> array(
#     [[  27.454,  277.52,  965.98,   908.2],
#     [   58.262,  478.27,  294.42,  1083.3],
#     [   803.36,  470.63,  971.66,  1052.4],
#     [   265.82,  486.96,  413.98,    1029],
#     [        0,  660.64,  75.612,  1048.1],
#     [   0.0701,  305.35,  39.073,  389.84]]
# )
```

#### Bounding Box Format Conversions
##### XYXY → XYWH
Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

```python
import numpy as np

from ultralytics.utils.ops import xyxy2xywh

xyxy_boxes = np.array(
    [
        [22.878, 231.27, 804.98, 756.83],
        [48.552, 398.56, 245.35, 902.71],
        [669.47, 392.19, 809.72, 877.04],
        [221.52, 405.8, 344.98, 857.54],
        [0, 550.53, 63.01, 873.44],
        [0.0584, 254.46, 32.561, 324.87],
    ]
)
xywh = xyxy2xywh(xyxy_boxes)

print(xywh)
# >>> array(
#     [[ 413.93,  494.05,   782.1, 525.56],
#     [  146.95,  650.63,   196.8, 504.15],
#     [   739.6,  634.62,  140.25, 484.85],
#     [  283.25,  631.67,  123.46, 451.74],
#     [  31.505,  711.99,   63.01, 322.91],
#     [   16.31,  289.67,  32.503,  70.41]]
# )
```

#### All Bounding Box Conversions

```python
from ultralytics.utils.ops import (
    ltwh2xywh,
    ltwh2xyxy,
    xywh2ltwh,  # xywh → top-left corner, w, h
    xywh2xyxy,
    xywhn2xyxy,  # normalized → pixel
    xyxy2ltwh,  # xyxy → top-left corner, w, h
    xyxy2xywhn,  # pixel → normalized
)

for func in (ltwh2xywh, ltwh2xyxy, xywh2ltwh, xywh2xyxy, xywhn2xyxy, xyxy2ltwh, xyxy2xywhn):
    print(help(func))  # print function docstrings
```

See docstring for each function or visit the `ultralytics.utils.ops` [reference page](https://docs.ultralytics.com/reference/utils/ops/) to read more about each function.

### Plotting
#### Drawing Annotations
Ultralytics includes an Annotator class that can be used to annotate any kind of data. It's easiest to use with [object detection bounding boxes](https://docs.ultralytics.com/modes/predict/#boxes), [pose key points](https://docs.ultralytics.com/modes/predict/#keypoints), and [oriented bounding boxes](https://docs.ultralytics.com/modes/predict/#obb).
> `ultralytics` 提供了 Annotator 类，用于标注各种类型的数据

##### Horizontal Bounding Boxes

```python
import cv2 as cv
import numpy as np

from ultralytics.utils.plotting import Annotator, colors

names = {  
    0: "person",
    5: "bus",
    11: "stop sign",
}

image = cv.imread("ultralytics/assets/bus.jpg")
ann = Annotator(
    image,
    line_width=None,  # default auto-size
    font_size=None,  # default auto-size
    font="Arial.ttf",  # must be ImageFont compatible
    pil=False,  # use PIL, otherwise uses OpenCV
)

xyxy_boxes = np.array(
    [
        [5, 22.878, 231.27, 804.98, 756.83],  # class-idx x1 y1 x2 y2
        [0, 48.552, 398.56, 245.35, 902.71],
        [0, 669.47, 392.19, 809.72, 877.04],
        [0, 221.52, 405.8, 344.98, 857.54],
        [0, 0, 550.53, 63.01, 873.44],
        [11, 0.0584, 254.46, 32.561, 324.87],
    ]
)

for nb, box in enumerate(xyxy_boxes):
    c_idx, *box = box
    label = f"{str(nb).zfill(2)}:{names.get(int(c_idx))}"
    ann.box_label(box, label, color=colors(c_idx, bgr=True))

image_with_bboxes = ann.result()
```

##### Oriented Bounding Boxes (OBB)

```python
import cv2 as cv
import numpy as np

from ultralytics.utils.plotting import Annotator, colors

obb_names = {10: "small vehicle"}
obb_image = cv.imread("datasets/dota8/images/train/P1142__1024__0___824.jpg")
obb_boxes = np.array(
    [
        [0, 635, 560, 919, 719, 1087, 420, 803, 261],  # class-idx x1 y1 x2 y2 x3 y2 x4 y4
        [0, 331, 19, 493, 260, 776, 70, 613, -171],
        [9, 869, 161, 886, 147, 851, 101, 833, 115],
    ]
)
ann = Annotator(
    obb_image,
    line_width=None,  # default auto-size
    font_size=None,  # default auto-size
    font="Arial.ttf",  # must be ImageFont compatible
    pil=False,  # use PIL, otherwise uses OpenCV
)
for obb in obb_boxes:
    c_idx, *obb = obb
    obb = np.array(obb).reshape(-1, 4, 2).squeeze()
    label = f"{obb_names.get(int(c_idx))}"
    ann.box_label(
        obb,
        label,
        color=colors(c_idx, True),
        rotated=True,
    )

image_with_obb = ann.result()
```

##### Bounding Boxes Circle Annotation [Circle Label](https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Annotator.circle_label)

```python
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("yolo11s.pt")
names = model.names
cap = cv2.VideoCapture("path/to/video/file.mp4")

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
writer = cv2.VideoWriter("Ultralytics circle annotation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    annotator = Annotator(im0)
    results = model.predict(im0)
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()

    for box, cls in zip(boxes, clss):
        annotator.circle_label(box, label=names[int(cls)])

    writer.write(im0)
    cv2.imshow("Ultralytics circle annotation", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
```

##### Bounding Boxes Text Annotation [Text Label](https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Annotator.text_label)

```python
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("yolo11s.pt")
names = model.names
cap = cv2.VideoCapture("path/to/video/file.mp4")

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
writer = cv2.VideoWriter("Ultralytics text annotation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    annotator = Annotator(im0)
    results = model.predict(im0)
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()

    for box, cls in zip(boxes, clss):
        annotator.text_label(box, label=names[int(cls)])

    writer.write(im0)
    cv2.imshow("Ultralytics text annotation", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
```

See the [`Annotator` Reference Page](https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Annotator) for additional insight.

### Miscellaneous
#### Code Profiling
Check duration for code to run/process either using `with` or as a decorator.

```python
from ultralytics.utils.ops import Profile

with Profile(device="cuda:0") as dt:
    pass  # operation to measure

print(dt)
# >>> "Elapsed time is 9.5367431640625e-07 s"
```

#### Ultralytics Supported Formats
Want or need to use the formats of [images or videos types supported](https://docs.ultralytics.com/modes/predict/#image-and-video-formats) by Ultralytics programmatically? Use these constants if you need.

```python
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

print(IMG_FORMATS)
# {'tiff', 'pfm', 'bmp', 'mpo', 'dng', 'jpeg', 'png', 'webp', 'tif', 'jpg'}

print(VID_FORMATS)
# {'avi', 'mpg', 'wmv', 'mpeg', 'm4v', 'mov', 'mp4', 'asf', 'mkv', 'ts', 'gif', 'webm'}
```

### Make Divisible
Calculates the nearest whole number to `x` to make evenly divisible when divided by `y`.

```python
from ultralytics.utils.ops import make_divisible

make_divisible(7, 3)
# >>> 9
make_divisible(7, 2)
# >>> 8
```

## Advanced Customization
Both the Ultralytics YOLO command-line and Python interfaces are simply a high-level abstraction on the base engine executors. Let's take a look at the Trainer engine.
> Ultralytics 的 YOLO 命令行和 Python 接口都仅仅是基础引擎处理器的高层抽象

### BaseTrainer
BaseTrainer contains the generic boilerplate training routine. It can be customized for any task based over overriding the required functions or operations as long the as correct formats are followed. For example, you can support your own custom model and dataloader by just overriding these functions:
> BaseTrainer 包含了通用的训练例程，可以被自定义以用于任意的任务

- `get_model(cfg, weights)` - The function that builds the model to be trained
- `get_dataloader()` - The function that builds the dataloader More details and source code can be found in [`BaseTrainer` Reference](https://docs.ultralytics.com/reference/engine/trainer/)

### DetectionTrainer
Here's how you can use the YOLO11 `DetectionTrainer` and customize it.

```python
from ultralytics.models.yolo.detect import DetectionTrainer

trainer = DetectionTrainer(overrides={...})
trainer.train()
trained_model = trainer.best  # get best model
```

#### Customizing the DetectionTrainer
Let's customize the trainer **to train a custom detection model** that is not supported directly. You can do this by simply overloading the existing the `get_model` functionality:

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Loads a custom detection model given configuration and weight files."""
        ...


trainer = CustomTrainer(overrides={...})
trainer.train()
```

You now realize that you need to customize the trainer further to:

- Customize the `loss function`.
- Add `callback` that uploads model to your Google Drive after every 10 `epochs` Here's how you can do it:

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel


class MyCustomModel(DetectionModel):
    def init_criterion(self):
        """Initializes the loss function and adds a callback for uploading the model to Google Drive every 10 epochs."""
        ...


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Returns a customized detection model instance configured with specified config and weights."""
        return MyCustomModel(...)


# callback to upload model weights
def log_model(trainer):
    """Logs the path of the last model weight used by the trainer."""
    last_weight_path = trainer.last
    print(last_weight_path)


trainer = CustomTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
trainer.train()
```

To know more about Callback triggering events and entry point, checkout our [Callbacks Guide](https://docs.ultralytics.com/usage/callbacks/)

### Other engine components
There are other components that can be customized similarly like `Validators` and `Predictors`. See Reference section for more information on these.