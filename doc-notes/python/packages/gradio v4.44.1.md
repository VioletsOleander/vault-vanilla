# 1 Getting Started
## 1.1 Quickstart
Gradio is an open-source Python package that allows you to quickly **build a demo** or web application for your machine learning model, API, or any arbitary Python function. You can then **share your demo** with a a public link in seconds using Gradio's built-in sharing features. _No JavaScript, CSS, or web hosting experience needed!_
> Gradio: 开源 Python 包，用于快速为 ML 模型/API/任意 Python 函数快速构建 demo 或 web 应用
> 构建之后，可以使用 Gradio 内建的共享特性直接共享一个 public link，不需要自行使用 JS/CSS 等构建网页客户端

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/lcm-screenshot-3.gif)

It just takes a few lines of Python to create a demo like the one above, so let's get started.

### Installation
**Prerequisite**: Gradio requires [Python 3.8 or higher](https://www.python.org/downloads/).

We recommend installing Gradio using `pip`, which is included by default in Python. Run this in your terminal or command prompt:

```bash
pip install gradio
```

**Tip:** it is best to install Gradio in a virtual environment. Detailed installation instructions for all common operating systems [are provided here](https://www.gradio.app/main/guides/installing-gradio-in-a-virtual-environment).

### Building Your First Demo
You can run Gradio in your favorite code editor, Jupyter notebook, Google Colab, or anywhere else you write Python. Let's write your first Gradio app:

```python
import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()
```

**Tip:** We shorten the imported name from `gradio` to `gr`. This is a widely adopted convention for better readability of code.

Now, run your code. If you've written the Python code in a file named `app.py`, then you would run `python app.py` from the terminal.

The demo below will open in a browser on [http://localhost:7860](http://localhost:7860/) if running from a file. If you are running within a notebook, the demo will appear embedded within the notebook.
> 以文件运行时，demo 默认在 localhost: 7860 开启一个客户端

Type your name in the textbox on the left, drag the slider, and then press the Submit button. You should see a friendly greeting on the right.

**Tip:** When developing locally, you can run your Gradio app in **hot reload mode**, which automatically reloads the Gradio app whenever you make changes to the file. To do this, simply type in `gradio` before the name of the file instead of `python`. In the example above, you would type: `gradio app.py` in your terminal. Learn more in the [Hot Reloading Guide](https://www.gradio.app/guides/developing-faster-with-reload-mode).
> 本地开发时，可以以 hot reload 模型运行 gradio 程序，即每当我们修改源码时，都重新加载
> 要启用 hot reload 模式，将 `gradio` 替换 `python` 即可，例如 `gradio app.py`

**Understanding the `Interface` Class**
You'll notice that in order to make your first demo, you created an instance of the `gr.Interface` class. The `Interface` class is designed to create demos for machine learning models which accept one or more inputs, and return one or more outputs.
> 上例中，我们创建了一个 `gr.Interface` 实例，`Interface` 类用于创建 demo，接受输入，返回输出

The `Interface` class has three core arguments:

- `fn`: the function to wrap a user interface (UI) around
- `inputs`: the Gradio component(s) to use for the input. The number of components should match the number of arguments in your function.
- `outputs`: the Gradio component(s) to use for the output. The number of components should match the number of return values from your function.

> `Interface` 类的三个核心参数：
> - `fn`： 会被 UI 包装的执行函数
> - `inputs`：用于输入的 Gradio 组件，组件的数量应该和函数的参数数量匹配
> - `outputs`：用于输出的 Gradio 组件，组件的数量应该和函数的返回值数量匹配

The `fn` argument is very flexible -- you can pass _any_ Python function that you want to wrap with a UI. In the example above, we saw a relatively simple function, but the function could be anything from a music generator to a tax calculator to the prediction function of a pretrained machine learning model.
> `fn` 可以是任意 Python 函数

The `inputs` and `outputs` arguments take one or more Gradio components. As we'll see, Gradio includes more than [30 built-in components](https://www.gradio.app/docs/gradio/introduction) (such as the `gr.Textbox()`, `gr.Image()`, and `gr.HTML()` components) that are designed for machine learning applications.
> `inputs/outputs` 接受一个或多个 Gradio 组件，Gradio 包括了30多个内建组件
> 我们可以以字符串形式传入组件的名字，例如 `textbox` ，也可以传入组件实例 `gr.Textbox()`

**Tip:** For the `inputs` and `outputs` arguments, you can pass in the name of these components as a string (`"textbox"`) or an instance of the class (`gr.Textbox()`).

If your function accepts more than one argument, as is the case above, pass a list of input components to `inputs`, with each input component corresponding to one of the arguments of the function, in order. The same holds true if your function returns more than one value: simply pass in a list of components to `outputs`. This flexibility makes the `Interface` class a very powerful way to create demos.
> 函数接受多个参数时，`inputs` 接受一个组件列表，返回同理

We'll dive deeper into the `gr.Interface` on our series on [building Interfaces](https://www.gradio.app/main/guides/the-interface-class).

### Sharing Your Demo
What good is a beautiful demo if you can't share it? Gradio lets you easily share a machine learning demo without having to worry about the hassle of hosting on a web server. Simply set `share=True` in `launch()`, and a publicly accessible URL will be created for your demo. Let's revisit our example demo, but change the last line as follows:
> 在 `launch()` 中令 `shared=True` 可以生成可公共访问的 URL

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share your demo with just 1 extra parameter 🚀
```

When you run this code, a public URL will be generated for your demo in a matter of seconds, something like:

👉   `https://a23dsf231adb.gradio.live`

Now, anyone around the world can try your Gradio demo from their browser, while the machine learning model and all computation continues to run locally on your computer.

To learn more about sharing your demo, read our dedicated guide on [sharing your Gradio application](https://www.gradio.app/guides/sharing-your-app).

### Core Gradio Classes
So far, we've been discussing the `Interface` class, which is a high-level class that lets to build demos quickly with Gradio. But what else does Gradio include?

#### Custom Demos with `gr.Blocks`
Gradio offers a low-level approach for designing web apps with more customizable layouts and data flows with the `gr.Blocks` class. Blocks supports things like controlling where components appear on the page, handling multiple data flows and more complex interactions (e.g. outputs can serve as inputs to other functions), and updating properties/visibility of components based on user interaction — still all in Python.
> `gr.Blocks` 类用于自定义页面布局和数据流，以及基于用户交互更新组件的属性/可视性

You can build very custom and complex applications using `gr.Blocks()`. For example, the popular image generation [Automatic1111 Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) is built using Gradio Blocks. We dive deeper into the `gr.Blocks` on our series on [building with Blocks](https://www.gradio.app/guides/blocks-and-event-listeners).

#### Chatbots with `gr.ChatInterface`
Gradio includes another high-level class, `gr.ChatInterface`, which is specifically designed to create Chatbot UIs. Similar to `Interface`, you supply a function and Gradio creates a fully working Chatbot UI. If you're interested in creating a chatbot, you can jump straight to [our dedicated guide on `gr.ChatInterface`](https://www.gradio.app/guides/creating-a-chatbot-fast).
> `gr.ChatInterface` 用于构建 Chatbot UI，使用和 `Interface` 类似

#### The Gradio Python & JavaScript Ecosystem
That's the gist of the core `gradio` Python library, but Gradio is actually so much more! Its an entire ecosystem of Python and JavaScript libraries that let you build machine learning applications, or query them programmatically, in Python or JavaScript. 

Here are other related parts of the Gradio ecosystem:

- [Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`gradio_client`): query any Gradio app programmatically in Python.
 - [Gradio JavaScript Client](https://www.gradio.app/guides/getting-started-with-the-js-client) (`@gradio/client`): query any Gradio app programmatically in JavaScript.
 - [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite`): write Gradio apps in Python that run entirely in the browser (no server needed!), thanks to Pyodide.
 - [Hugging Face Spaces](https://huggingface.co/spaces): the most popular place to host Gradio applications — for free!

# 2 Building Interfaces
## 2.1 The `Interface` class
As mentioned in the [Quickstart](https://www.gradio.app/main/guides/quickstart), the `gr.Interface` class is a high-level abstraction in Gradio that allows you to quickly create a demo for any Python function simply by specifying the input types and the output types. Revisiting our first demo:

```python
import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()
```

We see that the `Interface` class is initialized with three required parameters:

- `fn`: the function to wrap a user interface (UI) around
 - `inputs`: which Gradio component(s) to use for the input. The number of components should match the number of arguments in your function.
 - `outputs`: which Gradio component(s) to use for the output. The number of components should match the number of return values from your function.

In this Guide, we'll dive into `gr.Interface` and the various ways it can be customized, but before we do that, let's get a better understanding of Gradio components.

### Gradio Components
Gradio includes more than 30 pre-built components (as well as many [community-built _custom components_](https://www.gradio.app/custom-components/gallery)) that can be used as inputs or outputs in your demo. These components correspond to common data types in machine learning and data science, e.g. the `gr.Image` component is designed to handle input or output images, the `gr.Label` component displays classification labels and probabilities, the `gr.LinePlot` component displays line plots, and so on.
> Gradio 的组件对应于 ML 和数据科学中常见的数据类型，例如 `gr.Image` 用于处理输入、输出图像，`gr.Label` 用于处理分类标签和概率，`gr.LinePlot` 用于处理折线图

### Components Attributes
We used the default versions of the `gr.Textbox` and `gr.Slider`, but what if you want to change how the UI components look or behave?

Let's say you want to customize the slider to have values from 1 to 10, with a default of 2. And you wanted to customize the output text field — you want it to be larger and have a label.

If you use the actual classes for `gr.Textbox` and `gr.Slider` instead of the string shortcuts, you have access to much more customizability through component attributes.
> 构建组件实例可以帮助我们自定义组件的行为和属性

```python
import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * intensity

demo = gr.Interface(
    fn=greet,
    inputs=["text", gr.Slider(value=2, minimum=1, maximum=10, step=1)],
    outputs=[gr.Textbox(label="greeting", lines=3)],
)

demo.launch()
```

### Multiple Input and Output Components
Suppose you had a more complex function, with multiple outputs as well. In the example below, we define a function that takes a string, boolean, and number, and returns a string and number.

```python
import gradio as gr

def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "checkbox", gr.Slider(0, 100)],
    outputs=["text", "number"],
)
demo.launch()
```

Just as each component in the `inputs` list corresponds to one of the parameters of the function, in order, each component in the `outputs` list corresponds to one of the values returned by the function, in order.

### An Image Example
Gradio supports many types of components, such as `Image`, `DataFrame`, `Video`, or `Label`. Let's try an image-to-image function to get a feel for these!

```python
import numpy as np
import gradio as gr

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

demo = gr.Interface(sepia, gr.Image(), "image")
demo.launch()
```

When using the `Image` component as input, your function will receive a NumPy array with the shape `(height, width, 3)`, where the last dimension represents the RGB values. We'll return an image as well in the form of a NumPy array.
> 使用 `Image` 组件作为输入时，我们的函数会接受一个 Numpy 数组，其形状为 `(height, width, 3)` ，最后一个维度是 RGB 值
> 作为输出时，我们也应该返回一个 Numpy 数组

Gradio handles the preprocessing and postprocessing to convert images to NumPy arrays and vice versa. You can also control the preprocessing performed with the `type=` keyword argument. For example, if you wanted your function to take a file path to an image instead of a NumPy array, the input `Image` component could be written as:
> Gradio 会负责将输入图片转化为 Numpy 数组传给函数，以及将函数返回的 Numpy 数组转化为输出图片
> 可以指定 `gr.Image()` 实例中的 `type` 来指定 Gradio 如何处理输入图片，例如函数可以接受一个指向输入图片的路径

```python
gr.Image(type="filepath")
```

You can read more about the built-in Gradio components and how to customize them in the [Gradio docs](https://gradio.app/docs).

### Example Inputs
You can provide example data that a user can easily load into `Interface`. This can be helpful to demonstrate the types of inputs the model expects, as well as to provide a way to explore your dataset in conjunction with your model. To load example data, you can provide a **nested list** to the `examples=` keyword argument of the Interface constructor. Each sublist within the outer list represents a data sample, and each element within the sublist represents an input for each input component. The format of example data for each component is specified in the [Docs](https://gradio.app/docs#components).
> 可以提供 `Interface` 接受的示例数据
> 通过对 `examples` 关键字参数提供一个嵌套列表即可，嵌套列表中的每一个列表表示一个数据样本，数据样本中每一个元素表示一个输入组件对应的输入

```python
import gradio as gr

def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        if num2 == 0:
            raise gr.Error("Cannot divide by zero!")
        return num1 / num2

demo = gr.Interface(
    calculator,
    [
        "number",
        gr.Radio(["add", "subtract", "multiply", "divide"]),
        "number"
    ],
    "number",
    examples=[
        [45, "add", 3],
        [3.14, "divide", 2],
        [144, "multiply", 2.5],
        [0, "subtract", 1.2],
    ],
    title="Toy Calculator",
    description="Here's a sample toy calculator.",
)

demo.launch()
```

You can load a large dataset into the examples to browse and interact with the dataset through Gradio. The examples will be automatically paginated (you can configure this through the `examples_per_page` argument of `Interface`).

Continue learning about examples in the [More On Examples](https://gradio.app/guides/more-on-examples) guide.

### Descriptive Content
In the previous example, you may have noticed the `title=` and `description=` keyword arguments in the `Interface` constructor that helps users understand your app.

There are three arguments in the `Interface` constructor to specify where this content should go:

- `title`: which accepts text and can display it at the very top of interface, and also becomes the page title.
- `description`: which accepts text, markdown or HTML and places it right under the title.
- `article`: which also accepts text, markdown or HTML and places it below the interface.

> `Interface` 的描述性内容参数：
> - `title`: 接受文本，在界面的最上端展示，也是页面标题
> - `description`: 接受文本、markdown、HTML，放在标题下
> - `article`: 接受文本、markdown、HTML，放在界面下

![annotated](https://github.com/gradio-app/gradio/blob/main/guides/assets/annotated.png?raw=true)


Another useful keyword argument is `label=`, which is present in every `Component`. This modifies the label text at the top of each `Component`. You can also add the `info=` keyword argument to form elements like `Textbox` or `Radio` to provide further information on their usage.
> 每个组件有关键字参数 `label` ，表示组件的标签
> `info` 参数会为该组件附加信息

```python
gr.Number(label='Age', info='In years, must be greater than 0')
```

### Additional Inputs within an Accordion
If your prediction function takes many inputs, you may want to hide some of them within a collapsed accordion to avoid cluttering the UI. The `Interface` class takes an `additional_inputs` argument which is similar to `inputs` but any input components included here are not visible by default. The user must click on the accordion to show these components. The additional inputs are passed into the prediction function, in order, after the standard inputs.
> `Interface` 的 `additional_inputs` 类似 `inputs`，但其包含的输入组件默认不可见，`additional_inputs` 中的输入组件也会传入给函数，顺序在 `inputs` 之后

You can customize the appearance of the accordion by using the optional `additional_inputs_accordion` argument, which accepts a string (in which case, it becomes the label of the accordion), or an instance of the `gr.Accordion()` class (e.g. this lets you control whether the accordion is open or closed by default).
> `additional_inputs_accordion` 参数接受字符串或者 `gr.Accordion()` 实例，用于自定义 accordion

Here's an example:

```python
import gradio as gr

def generate_fake_image(prompt, seed, initial_image=None):
    return f"Used seed: {seed}", "https://dummyimage.com/300/09f.png"

demo = gr.Interface(
    generate_fake_image,
    inputs=["textbox"],
    outputs=["textbox", "image"],
    additional_inputs=[
        gr.Slider(0, 1000),
        "image"
    ]
)

demo.launch()
```

## 2.2 More on Examples
In the [previous Guide](https://www.gradio.app/main/guides/the-interface-class), we discussed how to provide example inputs for your demo to make it easier for users to try it out. Here, we dive into more details.

### Providing Examples
Adding examples to an Interface is as easy as providing a list of lists to the `examples` keyword argument. Each sublist is a data sample, where each element corresponds to an input of the prediction function. The inputs must be ordered in the same order as the prediction function expects them.

If your interface only has one input component, then you can provide your examples as a regular list instead of a list of lists.
> 如果 interface 仅有一个输入组件，则可以不用提供嵌套列表作为样例，提供列表即可

#### Loading Examples from a Directory
You can also specify a path to a directory containing your examples. If your Interface takes only a single file-type input, e.g. an image classifier, you can simply pass a directory filepath to the `examples=` argument, and the `Interface` will load the images in the directory as examples. In the case of multiple inputs, this directory must contain a log.csv file with the example values. In the context of the calculator demo, we can set `examples='/demo/calculator/examples'` and in that directory we include the following `log.csv` file:
> 可以指定指向样例文件的路径
> 例如，如果我们的 Interface 仅接受单个文件类型的输入，我们可以直接给 `examples=` 传入一个目录路径，Interface 会将目录中的文件装载为样例
> 如果 Interface 接受多个输入，也可以传入目录路径，但目录中必须包含一个 `log.csv` 文件，文件中包含样例值

```csv
num,operation,num2
5,"add",3
4,"divide",2
5,"multiply",3
```

This can be helpful when browsing flagged data. Simply point to the flagged directory and the `Interface` will load the examples from the flagged data.

#### Providing Partial Examples
Sometimes your app has many input components, but you would only like to provide examples for a subset of them. In order to exclude some inputs from the examples, pass `None` for all data samples corresponding to those particular components.
> 如果样例中没有全部信息，在全部的数据样本中，对于这些缺少的组件都传入 `None` 即可，这可以将这些输入组件排除出样例

### Caching examples
You may wish to provide some cached examples of your model for users to quickly try out, in case your model takes a while to run normally. If `cache_examples=True`, your Gradio app will run all of the examples and save the outputs when you call the `launch()` method. This data will be saved in a directory called `gradio_cached_examples` in your working directory by default. You can also set this directory with the `GRADIO_EXAMPLES_CACHE` environment variable, which can be either an absolute path or a relative path to your working directory.
> 如果我们的模型需要一段时间运行，我们可以令 `cache_examples=True`，这样，我们调用 `launch` 时，Gradio app 会事先运行好所有的示例，然后缓存结果，这些数据默认会存储于 `gradio_cached_examples` 目录中
> 可以通过 `GRADIO_EXAMPLES_CACHE` 环境变量改变默认目录，可以是绝对路径，也可以是相对于工作目录的相对路径

Whenever a user clicks on an example, the output will automatically be populated in the app now, using data from this cached directory instead of actually running the function. This is useful so users can quickly try out your model without adding any load!

Alternatively, you can set `cache_examples="lazy"`. This means that each particular example will only get cached after it is first used (by any user) in the Gradio app. This is helpful if your prediction function is long-running and you do not want to wait a long time for your Gradio app to start.
> `cache_examples` 还可以设定为 `"lazy"` ，意思是特定的样例仅在被用户第一次使用之后才缓存，这可以避免我们调用 `launch()` 之后应用花费太多时间跑所有样例

Keep in mind once the cache is generated, it will not be updated automatically in future launches. If the examples or function logic change, delete the cache folder to clear the cache and rebuild it with another `launch()`.
> 当缓存数据生成后，今后的 `launch` 调用不会自动更新它
> 因此，如果我们的函数逻辑有变化，需要删除 cache 目录，然后重跑

## 2.3 Flagging
You may have noticed the "Flag" button that appears by default in your `Interface`. When a user using your demo sees input with interesting output, such as erroneous or unexpected model behaviour, they can flag the input for you to review. Within the directory provided by the `flagging_dir=` argument to the `Interface` constructor, a CSV file will log the flagged inputs. If the interface involves file data, such as for Image and Audio components, folders will be created to store those flagged data as well.
> `Interface` 中默认有一个 Flag 按钮
> 如果使用我们 demo 的用户发现对于特定输入，输出有误或模型行为错误，它们可以 flag 该输入供我们回顾
> flag 的输入数据存储在由 `flagging_dir` 指定的目录(默认为 `flagged` )下的一个 CSV 文件(默认为 `logs.csv` )中，图像输入和音频输入等也会存在该目录下的子目录中，`logs.csv` 中就存储指向它们的相对路径

For example, with the calculator interface shown above, we would have the flagged data stored in the flagged directory shown below:

```directory
+-- calculator.py
+-- flagged/
|   +-- logs.csv
```

_flagged/logs.csv_

```csv
num1,operation,num2,Output
5,add,7,12
6,subtract,1.5,4.5
```

With the sepia interface shown earlier, we would have the flagged data stored in the flagged directory shown below:

```directory
+-- sepia.py
+-- flagged/
|   +-- logs.csv
|   +-- im/
|   |   +-- 0.png
|   |   +-- 1.png
|   +-- Output/
|   |   +-- 0.png
|   |   +-- 1.png
```

_flagged/logs.csv_

```csv
im,Output
im/0.png,Output/0.png
im/1.png,Output/1.png
```

If you wish for the user to provide a reason for flagging, you can pass a list of strings to the `flagging_options` argument of Interface. Users will have to select one of the strings when flagging, which will be saved as an additional column to the CSV.
> `flagging_options` 参数用于让用户指定 flagging 原因，用户需要在 flagging 中从中选择一个原因，该原因也会被存储在 `logs.csv` 中

## 2.4 Interface State
So far, we've assumed that your demos are _stateless_: that they do not persist information beyond a single function call. What if you want to modify the behavior of your demo based on previous interactions with the demo? There are two approaches in Gradio: _global state_ and _session state_.
> 目前为止，我们都假设我们的 demo 是无状态的，即不会在函数调用之间保存信息，如果我们需要基于之前和 demo 的交互改变 demo 的行为，有两种方法：全局状态和会话状态

### Global State
If the state is something that should be accessible to all function calls and all users, you can create a variable outside the function call and access it inside the function. For example, you may load a large model outside the function and use it inside the function so that every function call does not need to reload the model.
> 如果状态需要被所有函数调用和所有用户访问，我们可以在函数调用外创造变量，然后在函数内访问，例如在函数外加载一个大型模型，方便所有函数访问

```python
import gradio as gr

scores = []

def track_score(score):
    scores.append(score)
    top_scores = sorted(scores, reverse=True)[:3]
    return top_scores

demo = gr.Interface(
    track_score,
    gr.Number(label="Score"),
    gr.JSON(label="Top Scores")
)
demo.launch()
```

In the code above, the `scores` array is shared between all users. If multiple users are accessing this demo, their scores will all be added to the same list, and the returned top 3 scores will be collected from this shared reference.
> 上例中，`scores` 列表由所有的用户共享，如果多个用户访问该 demo，它们看到的是相同的输出

### Session State
Another type of data persistence Gradio supports is session state, where data persists across multiple submits within a page session. However, data is _not_ shared between different users of your model. 
> Gradio 还支持通过会话状态保持数据，即数据在一个页面会话中的多个提交中会保持，但此时数据不是用户之间共享的

To store data in a session state, you need to do three things:

1. Pass in an extra parameter into your function, which represents the state of the interface.
 3. At the end of the function, return the updated value of the state as an extra return value.
 5. Add the `'state'` input and `'state'` output components when creating your `Interface`

>  要在会话状态中保存数据，我们需要：
>  - 向我们的函数传递一个额外的参数，它用以表示界面的状态
>  - 在函数末尾，额外返回该状态更新的值
>  - 对 `Interface` 添加 `state` 输入和输出组件

Here's a simple app to illustrate session state - this app simply stores users previous submissions and displays them back to the user:

```python
import gradio as gr

def store_message(message: str, history: list[str]):  
    output = {
        "Current messages": message,
        "Previous messages": history[::-1]
    }
    history.append(message)
    return output, history

demo = gr.Interface(fn=store_message,
                    inputs=["textbox", gr.State(value=[])],
                    outputs=["json", gr.State()])

demo.launch()
```

Notice how the state persists across submits within each page, but if you load this demo in another tab (or refresh the page), the demos will not share chat history. Here, we could not store the submission history in a global variable, otherwise the submission history would then get jumbled between different users.
> 在页面中的多次提交会保持这些状态，但如果在另一个页面打开 demo，或者刷新页面，数据就会消失

The initial value of the `State` is `None` by default. If you pass a parameter to the `value` argument of `gr.State()`, it is used as the default value of the state instead.

Note: the `Interface` class only supports a single session state variable (though it can be a list with multiple elements). For more complex use cases, you can use Blocks, [which supports multiple `State` variables](https://www.gradio.app/guides/state-in-blocks/). Alternatively, if you are building a chatbot that maintains user state, consider using the `ChatInterface` abstraction, [which manages state automatically](https://www.gradio.app/guides/creating-a-chatbot-fast).
> `Interface` 类仅支持单个会话状态变量

## 2.5 Reactive Interfaces
Finally, we cover how to get Gradio demos to refresh automatically or continuously stream data.

### Live Interfaces
You can make interfaces automatically refresh by setting `live=True` in the interface. Now the interface will recalculate as soon as the user input changes.
> 设定 `live=True` 可以让界面自动刷新，此时界面没有 Submit 按钮，界面会在输入变化之后自动运行函数

```python
import gradio as gr

def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2

demo = gr.Interface(
    calculator,
    [
        "number",
        gr.Radio(["add", "subtract", "multiply", "divide"]),
        "number"
    ],
    "number",
    live=True,
)
demo.launch()
```

Note there is no submit button, because the interface resubmits automatically on change.

### Streaming Components
Some components have a "streaming" mode, such as `Audio` component in microphone mode, or the `Image` component in webcam mode. Streaming means data is sent continuously to the backend and the `Interface` function is continuously being rerun.
> 一些组件具有“流”模式，例如 `Audio` 在麦克风模式下，`Image` 在摄像头模式下
> 流模式即数据是连续地被发送到后端，以及 `Interface` 函数也会连续的被重运行

The difference between `gr.Audio(source='microphone')` and `gr.Audio(source='microphone', streaming=True)`, when both are used in `gr.Interface(live=True)`, is that the first `Component` will automatically submit data and run the `Interface` function when the user stops recording, whereas the second `Component` will continuously send data and run the `Interface` function _during_ recording.
> 没有 `streaming=True` ，界面在用户停止记录时再自动提交数据，如果有，则在用户记录时就会提交数据

Here is example code of streaming images from the webcam.

```python
import gradio as gr
import numpy as np

def flip(im):
    return np.flipud(im)

demo = gr.Interface(
    flip,
    gr.Image(sources=["webcam"], streaming=True),
    "image",
    live=True
)
demo.launch()
```

Streaming can also be done in an output component. A `gr.Audio(streaming=True)` output component can take a stream of audio data yielded piece-wise by a generator function and combines them into a single audio file. For a detailed example, see our guide on performing [automatic speech recognition](https://www.gradio.app/guides/real-time-speech-recognition) with Gradio.
> 输出也可以流式输出，例如 `gr.Audio(streaming=True)` 输出组件会接受由生成函数一段一段生成的数据流，然后将它们结合为一个 audio 文件

## 2.6 The 4 Kinds of Gradio Interfaces
So far, we've always assumed that in order to build an Gradio demo, you need both inputs and outputs. But this isn't always the case for machine learning demos: for example, _unconditional image generation models_ don't take any input but produce an image as the output.
> 实际上，一些模型并不需要输入或者输出，例如无条件图像生成模型

It turns out that the `gradio.Interface` class can actually handle 4 different kinds of demos:

1. **Standard demos**: which have both separate inputs and outputs (e.g. an image classifier or speech-to-text model)
3. **Output-only demos**: which don't take any input but produce on output (e.g. an unconditional image generation model)
5. **Input-only demos**: which don't produce any output but do take in some sort of input (e.g. a demo that saves images that you upload to a persistent external database)
7. **Unified demos**: which have both input and output components, but the input and output components _are the same_. This means that the output produced overrides the input (e.g. a text autocomplete model)

> Interface 类可以处理四类不同的 demo：
> - 标准 demo：有分离的输入输出
> - 仅输出 demo：不接受输入，直接输出
> - 仅输入 demo：不产生输出，接受输入
> - 联合的 demo：输入和输出组件相同，即输出会覆盖输出（例如文本自动补全模型）

Depending on the kind of demo, the user interface (UI) looks slightly different:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/interfaces4.png)

Let's see how to build each kind of demo using the `Interface` class, along with examples:

### Standard demos
To create a demo that has both the input and the output components, you simply need to set the values of the `inputs` and `outputs` parameter in `Interface()`. Here's an example demo of a simple image filter:

```python
import numpy as np
import gradio as gr

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

demo = gr.Interface(sepia, gr.Image(), "image")
demo.launch()
```

### Output-only demos
What about demos that only contain outputs? In order to build such a demo, you simply set the value of the `inputs` parameter in `Interface()` to `None`. Here's an example demo of a mock image generation model:
> 仅输出的 demo：将 `inputs` 参数设定为 `None`

```python
import time

import gradio as gr

def fake_gan():
    time.sleep(1)
    images = [
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=387&q=80",
            "https://images.unsplash.com/photo-1554151228-14d9def656e4?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=386&q=80",
            "https://images.unsplash.com/photo-1542909168-82c3e7fdca5c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8aHVtYW4lMjBmYWNlfGVufDB8fDB8fA%3D%3D&w=1000&q=80",
    ]
    return images

demo = gr.Interface(
    fn=fake_gan,
    inputs=None,
    outputs=gr.Gallery(label="Generated Images", columns=[2]),
    title="FD-GAN",
    description="This is a fake demo of a GAN. In reality, the images are randomly chosen from Unsplash.",
)

demo.launch()
```

### Input-only demos
Similarly, to create a demo that only contains inputs, set the value of `outputs` parameter in `Interface()` to be `None`. Here's an example demo that saves any uploaded image to disk:

```python
import random
import string
import gradio as gr

def save_image_random_name(image):
    random_string = ''.join(random.choices(string.ascii_letters, k=20)) + '.png'
    image.save(random_string)
    print(f"Saved image to {random_string}!")

demo = gr.Interface(
    fn=save_image_random_name,
    inputs=gr.Image(type="pil"),
    outputs=None,
)
demo.launch()
```

### Unified demos
A demo that has a single component as both the input and the output. It can simply be created by setting the values of the `inputs` and `outputs` parameter as the same component. Here's an example demo of a text generation model:
> 设定 `inputs/outputs` 为相同的组件

```python
import gradio as gr
from transformers import pipeline

generator = pipeline('text-generation', model = 'gpt2')

def generate_text(text_prompt):
  response = generator(text_prompt, max_length = 30, num_return_sequences=5)
  return response[0]['generated_text']  

textbox = gr.Textbox()

demo = gr.Interface(generate_text, textbox, textbox)

demo.launch()
```

It may be the case that none of the 4 cases fulfill your exact needs. In this case, you need to use the `gr.Blocks()` approach!