---
completed: true
version: 0.54.2
---
# Basic Usage
In this tutorial, we will show you how to create a simple Python project with pixi. We will show some of the features that Pixi provides, that are currently not a part of `pdm`, `poetry` etc.

## Why is this useful?
Pixi builds upon the conda ecosystem, which allows you to create a Python environment with all the dependencies you need. This is especially useful when you are working with multiple Python interpreters and bindings to C and C++ libraries. For example, GDAL from PyPI does not have binary C dependencies, but the conda package does. On the other hand, some packages are only available through PyPI, which `pixi` can also install for you. Best of both world, let's give it a go!
>  pixi 基于 conda 生态系统构建，允许我们创建一个包含所有依赖项的 Python 环境
>  这在同时使用多个 Python 解释器，并且需要绑定 C/C++ 库的时候非常有用
>  例如 PyPI 上的 GDAL 包没有预编译的 C 依赖项，但它的 conda 包有
>  另一方面，一些包只能从 PyPI 上下载
>  pixi 集两者之长，两边都可以下载

## `pixi.toml` and `pyproject.toml`
We support two manifest formats: `pyproject.toml` and `pixi.toml`. In this tutorial, we will use the `pyproject.toml` format because it is the most common format for Python projects.
>  pixi 支持两种清单文件格式: `pyproject.toml, pixi.toml`
>  本教程使用 `pyproject.toml`

## Let's get started
Let's start out by creating a new project that uses a `pyproject.toml` file.

```
pixi init pixi-py --format pyproject
```

This creates a project directory with the following structure:

```
pixi-py
├── pyproject.toml
└── src
    └── pixi_py
        └── __init__.py
```

>  `pixi init xxx --format pyproject` 会创建 Python 项目，格式如上

The `pyproject.toml` for the project looks like this:

```toml
[project]
dependencies = []
name = "pixi-py"
requires-python = ">= 3.11"
version = "0.1.0"

[build-system]
build-backend = "hatchling.build"
requires = ["hatchling"]

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["osx-arm64"]

[tool.pixi.pypi-dependencies]
pixi_py = { path = ".", editable = true }

[tool.pixi.tasks]
```

This project uses a src-layout, but Pixi supports both [flat- and src-layouts](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/#src-layout-vs-flat-layout).
>  pixi 支持 src-layout 和 flat-layout

### What's in the `pyproject.toml`?
Okay, so let's have a look at what sections have been added and how we can modify the `pyproject.toml`.

These first entries were added to the `pyproject.toml` file:

```toml
# Main pixi entry
[tool.pixi.workspace]
channels = ["conda-forge"]
# This is your machine platform by default
platforms = ["osx-arm64"]
```

The `channels` and `platforms` are added to the `[tool.pixi.workspace]` section. Channels like `conda-forge` manage packages similar to PyPI but allow for different packages across languages. The keyword `platforms` determines what platform the project supports.
>  `tool.pixi.workspace` 中添加了 `channels, platforms`
>  例如 `conda-forge` 这样的 channel 和 PyPI 管理包的方式类似，但是允许不同语言的包
>  `platforms` 决定了项目支持的平台

The `pixi_py` package itself is added as an `editable` dependency. This means that the package is installed in editable mode, so you can make changes to the package and see the changes reflected in the environment, without having to re-install the environment.
>  我们创建的包本身被添加为 `editable` 依赖 (在 `[tool.pixi.pypi-dependencies]` 中)
>  这意味着该包可以以可编辑模式安装，这便于我们开发，无需重新安装环境就能看到改变

```toml
# Editable installs
[tool.pixi.pypi-dependencies]
pixi-py = { path = ".", editable = true }
```

In pixi, unlike other package managers, this is explicitly stated in the `pyproject.toml` file. The main reason being so that you can choose which environment this package should be included in.
>  pixi 不像其他的包管理器，它显式地将工作空间包是否可编辑写入了 `pyproject.toml` 文件 (其他包管理器则不显式支持，需要以 `pip install -e` 手动控制)，原因就是方便我们为不同的环境定义应该是可编辑还是普通安装，例如开发环境就是可编辑，生产环境就使用普通安装，避免意外修改影响部署

### Managing both conda and PyPI dependencies in pixi
Our projects usually depend on other packages.

```
cd pixi-py # Move into the project directory
pixi add black
```

This will add the `black` package as a Conda package to the `pyproject.toml` file. Which will result in the following addition to the `pyproject.toml`:
>  `pixi add xxx` 将 conda 包 `xxx` 添加为依赖
>  conda 依赖会位于 `pyproject.toml` 中 `[tool.pixi.dependencies]` 下

```toml
[tool.pixi.dependencies]
black = ">=25.1.0,<26"
```

But we can also be strict about the version that should be used.

```
pixi add black=25
```

>  可以明确指定版本号

resulting in:

```toml
[tool.pixi.dependencies]
black = "25.*"
```

Sometimes there are packages that aren't available on conda channels but are published on PyPI.

```
pixi add black --pypi
```

which results in the addition to the `dependencies` key in the `pyproject.toml`

```toml
dependencies = ["black"]
```

>  PyPI 依赖则直接添加到 `dependencies` key 中

When using the `pypi-dependencies` you can make use of the `optional-dependencies` that other packages make available as extras. For example, `flask` makes the `async` dependencies option, which can be added with the `--pypi` keyword:

```
pixi add "flask[async]==3.1.0" --pypi
```

which updates the `dependencies` entry to

```toml
dependencies = ["black", "flask[async]==3.1.0"]
```

Extras in `pixi.toml`
This tutorial focuses on the use of the `pyproject.toml`, but in case you're curious, the `pixi.toml` would contain the following entry after the installation of a PyPI package including an optional dependency:

```toml
[pypi-dependencies]
flask = { version = "==3.1.0", extras = ["async"] }
```

>  `pixi.toml` 中则会在 `[pypi-dependencies]` 中指定 PyPI 依赖

### Installation: `pixi install
Pixi always ensures the environment is up-to-date with the `pyproject.toml` file when running the environment. If you want to do it manually, you can run:
>  pixi 会在运行环境时确保环境和 `pyproject.toml` 保持同步
>  如果需要手动同步环境，可以 `pixi install`

```
pixi install
```

We now have a new directory called `.pixi` in the project root. The environment is a Conda environment with all the Conda and PyPI dependencies installed into it.
>  pixi 会在项目根目录下创建 `.pixi` 目录
>  环境是一个 conda 环境，所有的 conda 和 PyPI 依赖都会被安装到里面

The environment is always a result of the `pixi.lock` file, which is generated from the `pyproject.toml` file. This file contains the exact versions of the dependencies that were installed in the environment across platforms.
>  环境总是 `pixi.lock` 文件的结果，该文件由 `pyproject.toml` 文件生成
>  它包含了不同平台上所有安装在环境中的依赖的精确版本

## What's in the environment?
Using `pixi list`, you can see what's in the environment, this is essentially a nicer view on the lock file (`pixi.lock`):
>  `pixi list` 可以查看环境中有什么

```
Package          Version     Build               Size       Kind   Source
asgiref          3.8.1                           68.5 KiB   pypi   asgiref-3.8.1-py3-none-any.whl
black            24.10.0     py313h8f79df9_0     388.7 KiB  conda  black
blinker          1.9.0                           23.9 KiB   pypi   blinker-1.9.0-py3-none-any.whl
bzip2            1.0.8       h99b78c6_7          120 KiB    conda  bzip2
ca-certificates  2024.12.14  hf0a4a13_0          153.4 KiB  conda  ca-certificates
click            8.1.8       pyh707e725_0        82.7 KiB   conda  click
flask            3.1.0                           335.9 KiB  pypi   flask-3.1.0-py3-none-any.whl
itsdangerous     2.2.0                           45.8 KiB   pypi   itsdangerous-2.2.0-py3-none-any.whl
jinja2           3.1.5                           484.8 KiB  pypi   jinja2-3.1.5-py3-none-any.whl
libexpat         2.6.4       h286801f_0          63.2 KiB   conda  libexpat
libffi           3.4.2       h3422bc3_5          38.1 KiB   conda  libffi
liblzma          5.6.3       h39f12f2_1          96.8 KiB   conda  liblzma
libmpdec         4.0.0       h99b78c6_0          67.6 KiB   conda  libmpdec
libsqlite        3.48.0      h3f77e49_1          832.8 KiB  conda  libsqlite
libzlib          1.3.1       h8359307_2          45.3 KiB   conda  libzlib
markupsafe       3.0.2                           73 KiB     pypi   markupsafe-3.0.2-cp313-cp313-macosx_11_0_arm64.whl
mypy_extensions  1.0.0       pyha770c72_1        10.6 KiB   conda  mypy_extensions
ncurses          6.5         h5e97a16_3          778.3 KiB  conda  ncurses
openssl          3.4.0       h81ee809_1          2.8 MiB    conda  openssl
packaging        24.2        pyhd8ed1ab_2        58.8 KiB   conda  packaging
pathspec         0.12.1      pyhd8ed1ab_1        40.1 KiB   conda  pathspec
pixi_py          0.1.0                                      pypi    (editable)
platformdirs     4.3.6       pyhd8ed1ab_1        20 KiB     conda  platformdirs
python           3.13.1      h4f43103_105_cp313  12.3 MiB   conda  python
python_abi       3.13        5_cp313             6.2 KiB    conda  python_abi
readline         8.2         h92ec313_1          244.5 KiB  conda  readline
tk               8.6.13      h5083fa2_1          3 MiB      conda  tk
tzdata           2025a       h78e105d_0          120 KiB    conda  tzdata
werkzeug         3.1.3                           743 KiB    pypi   werkzeug-3.1.3-py3-none-any.whl
```

Here, you can see the different conda and Pypi packages listed. As you can see, the `pixi-py` package that we are working on is installed in editable mode. Every environment in Pixi is isolated but reuses files that are hard-linked from a central cache directory. This means that you can have multiple environments with the same packages but only have the individual files stored once on disk.
>  Pixi 的每个环境都是独立的，但会复用从中心缓存目录硬链接的文件，因此多个环境中相同的包仅会在磁盘上留下一个拷贝

Why does the environment have a Python interpreter?
The Python interpreter is also installed in the environment. This is because the Python interpreter version is read from the `requires-python` field in the `pyproject.toml` file. This is used to determine the Python version to install in the environment. This way, Pixi automatically manages/bootstraps the Python interpreter for you, so no more `brew`, `apt` or other system install steps.
>  Pixi 也会根据 `pyproject.toml` 中的 `requires-python` 安装符合版本的 Python 解释器到环境中

How to use the Free-threaded interpreter?
If you want to use a free-threaded Python interpreter, you can add the `python-freethreading` dependency with:

```
pixi add python-freethreading
```

This ensures that a free-threaded version of Python is installed in the environment. This might not work with other packages that are not thread-safe yet. You can read more about free-threaded Python [here](https://docs.python.org/3/howto/free-threading-python.html).

### Multiple environments
Pixi can also create multiple environments, this works well together with the `dependency-groups` feature in the `pyproject.toml` file.
>  Pixi 也可以创建多个环境，和 `pyproject.toml` 的 `dependency-groups` 兼容

Let's add a dependency-group, which Pixi calls a `feature`, named `test`. And add the `pytest` package to this group.

```
pixi add --pypi --feature test pytest
```

This results in the package being added to the `dependency-groups` following the [PEP 735](https://peps.python.org/pep-0735/).

```toml
[dependency-groups]
test = ["pytest"]
```

After we have added the `dependency-groups` to the `pyproject.toml`, Pixi sees these as a [`feature`](https://pixi.sh/v0.53.0/reference/pixi_manifest/#the-feature-and-environments-tables), which can contain a collection of `dependencies`, `tasks`, `channels`, and more.
>  Pixi 会将 `dependency-groups` 中的 entry 视作 feature
>  feature 可以包含一系列 `dependencies, tasks, channels`

```
pixi workspace environment add default --solve-group default --force
pixi workspace environment add test --feature test --solve-group default
```

Which results in:

```toml
[tool.pixi.environments]
default = { solve-group = "default" }
test = { features = ["test"], solve-group = "default" }
```

Solve Groups
Solve groups are a way to group dependencies together. This is useful when you have multiple environments that share the same dependencies. For example, maybe `pytest` is a dependency that influences the dependencies of the `default` environment. By putting these in the same solve group, you ensure that the versions in `test` and `default` are exactly the same.
>  解决组是一种将依赖项分组的方式，当我们有多个共享相同依赖项的环境时，解决组十分有用
>  例如，假设 `pytest` 是一个影响了 `default` 环境的依赖项，将它们放在同一个解决组中，可以确保 `test, default` 的版本完全相同

Without specifying the environment name, Pixi will default to the `default` environment. If you want to install or run the `test` environment, you can specify the environment with the `--environment` flag.
>  不指定环境名称时，pixi 默认使用 `default` 环境
>  如果我们要安装并运行 `test` 环境，可以通过 `--environment` 指定环境

```
pixi install --environment test
pixi run --environment test pytest
```

## Getting code to run
Let's add some code to the `pixi_py` package. We will add a new function to the `src/pixi_py/__init__.py` file:

```python
from rich import print

def hello():
    return "Hello, [bold magenta]World[/bold magenta]!", ":vampire:"

def say_hello():
    print(*hello())
```

Now add the `rich` dependency from PyPI

```
pixi add --pypi rich
```

Let's see if this works by running:

```
pixi run python -c 'import pixi_py; pixi_py.say_hello()'
```

Which should output:

```
Hello, World! 🧛
```

Slow?
This might be slow the first time because Pixi installs the project, but it will be near instant the second time.

Pixi runs the self installed Python interpreter. Then, we are importing the `pixi_py` package, which is installed in editable mode. The code calls the `say_hello` function that we just added. And it works! Cool!

## Testing this code
Okay, so let's add a test for this function. Let's add a `tests/test_me.py` file in the root of the project.
>  我们在项目根目录下添加 `tests/test_me.py`

Giving us the following project structure:

```
.
├── pixi.lock
├── src
│   └── pixi_py
│       └── __init__.py
├── pyproject.toml
└── tests/test_me.py
```

```python
from pixi_py import hello

def test_pixi_py():
    assert hello() == ("Hello, [bold magenta]World[/bold magenta]!", ":vampire:")
```

Let's add an easy task for running the tests.

```
pixi task add --feature test test "pytest"
```

So Pixi has a task system to make it easy to run commands. Similar to `npm` scripts or something you would specify in a `Justfile`.
>  我们可以将 `test` 添加为一个 task

Pixi tasks
Tasks are a cool Pixi feature that is powerful and runs in a cross-platform shell. You can do caching, dependencies and more. Read more about tasks in the [tasks](https://pixi.sh/v0.53.0/workspace/advanced_tasks/) section.

```
pixi run test
```

results in the following output:

```
✨ Pixi task (test): pytest .
================================================================================================= test session starts =================================================================================================
platform darwin -- Python 3.12.2, pytest-8.1.1, pluggy-1.4.0
rootdir: /private/tmp/pixi-py
configfile: pyproject.toml
collected 1 item

test_me.py .                                                                                                                                                                                                    [100%]

================================================================================================== 1 passed in 0.00s =================================================================================================
```

>  task 是一个 pixi 提供的在跨平台 shell 上运行任务的特性

Why didn't I have to specify the environment?
The `test` task was added to the `test` feature/environment. When you run the `test` task, Pixi automatically switches to the `test` environment. Because that is the only environment that has the task.

>  task 会被添加到特定的 feature/environment，因此我们运行 task 时，pixi 会自动切换到该环境

Neat! It seems to be working!

### Test vs Default environment
Let's compare the output of the test and default environments. We add the `--explicit` flag to show the explicit dependencies in the environment.

```
pixi list --explicit --environment test
# vs. default environment
pixi list --explicit
```

We see that the `test` environment has:

```
package          version       build               size       kind   source
...
pytest           8.1.1                             1.1 mib    pypi   pytest-8.1.1-py3-none-any.whl
...
```

However, the default environment is missing the `pytest` package. This way, you can finetune your environments to only have the packages that are needed for that environment. E.g. you could also have a `dev` environment that has `pytest` and `ruff` installed, but you could omit these from the `prod` environment. There is a [docker](https://github.com/prefix-dev/pixi/tree/main/examples/docker) example that shows how to set up a minimal `prod` environment and copy from there.
>  我们可以让 `dev` 环境安装 `pytest, ruff`，但 `prod` 环境则没必要

## Replacing PyPI packages with conda packages
Last thing, Pixi provides the ability for `pypi` packages to depend on `conda` packages. Let's confirm this with:

```
pixi list pygments
```

Note that it was installed as a `pypi` package:

```
Package          Version       Build               Size       Kind   Source
pygments         2.17.2                            4.1 MiB    pypi   pygments-2.17.2-py3-none-any.http.whl
```

This is a dependency of the `rich` package. As you can see by running:

```
pixi tree --invert pygments
```

Let's explicitly add `pygments` to the `pyproject.toml` file.

```
pixi add pygments
```

This will add the following to the `pyproject.toml` file:

```toml
[tool.pixi.dependencies]
pygments = "=2.19.1,<3"
```

We can now see that the `pygments` package is now installed as a conda package.

```
pixi list pygments
```

Now results in:

```
Package   Version  Build         Size       Kind   Source
pygments  2.19.1   pyhd8ed1ab_0  867.8 KiB  conda  pygments
```

This way, PyPI dependencies and conda dependencies can be mixed and matched to seamlessly interoperate.

```
pixi run python -c 'import pixi_py; pixi_py.say_hello()'
```

And it still works!

>  pixi 允许 PyPI 包依赖于 conda 包
>  例如我们安装了 `rich` 之后，发现 `rich` 也在 PyPI 上安装了它的依赖 `pygments`
>  我们可以 `pixi add pygments` 将它添加为 conda 依赖，项目仍然可以工作

## Conclusion
In this tutorial, you've seen how easy it is to use a `pyproject.toml` to manage your Pixi dependencies and environments. We have also explored how to use PyPI and conda dependencies seamlessly together in the same project and install optional dependencies to manage Python packages.

Hopefully, this provides a flexible and powerful way to manage your Python projects and a fertile base for further exploration with Pixi.

Thanks for reading! Happy Coding 🚀

Any questions? Feel free to reach out or share this tutorial on [X](https://twitter.com/prefix_dev), [join our Discord](https://discord.gg/kKV8ZxyzY4), send us an [e-mail](mailto:hi@prefix.dev) or follow our [GitHub](https://github.com/prefix-dev).