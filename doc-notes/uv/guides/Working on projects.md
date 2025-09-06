---
completed: true
version: 0.8,14
---
# Working on projects
uv supports managing Python projects, which define their dependencies in a `pyproject.toml` file.
>  uv 管理 Python 项目，Python 项目的依赖定义在 `pyproject.toml` 文件中

## Creating a new project
You can create a new Python project using the `uv init` command:

```
uv init hello-world
cd hello-world
```

Alternatively, you can initialize a project in the working directory:

```
mkdir hello-world
cd hello-world
uv init
```

uv will create the following files:

```
├── .gitignore
├── .python-version
├── README.md
├── main.py
└── pyproject.toml
```

The `main.py` file contains a simple "Hello world" program. Try it out with `uv run`:

```
uv run main.py
```

## Project structure
A project consists of a few important parts that work together and allow uv to manage your project. In addition to the files created by `uv init`, uv will create a virtual environment and `uv.lock` file in the root of your project the first time you run a project command, i.e., `uv run`, `uv sync`, or `uv lock`.
>  `uv` 会在第一次运行项目命令的时候 (`uv run, uv sync, uv lock`) 创建一个虚拟环境，以及 ` uv.lock ` 文件

A complete listing would look like:

```
.
├── .venv
│   ├── bin
│   ├── lib
│   └── pyvenv.cfg
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

### `pyproject.toml`
The `pyproject.toml` contains metadata about your project:

pyproject.toml

```toml
[project]
name = "hello-world"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
dependencies = []
```

You'll use this file to specify dependencies, as well as details about the project such as its description or license. You can edit this file manually, or use commands like `uv add` and `uv remove` to manage your project from the terminal.

>  `pyproject.toml` 包含项目元数据，我们可以手动修改，也可以使用例如 `uv add, uv remove` 的命令进行管理

Tip
See the official [`pyproject.toml` guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) for more details on getting started with the `pyproject.toml` format.

You'll also use this file to specify uv [configuration options](https://docs.astral.sh/uv/concepts/configuration-files/) in a [`[tool.uv]`](https://docs.astral.sh/uv/reference/settings/) section.
>  在 `[tool.uv]` 部分可以指定 uv 的配置选项

### `.python-version`
The `.python-version` file contains the project's default Python version. This file tells uv which Python version to use when creating the project's virtual environment.
>  `.python-version` 文件包含项目的默认 Python 版本，uv 根据它确定哪个 Python 版本将用于创建项目的虚拟环境

### `.venv`
The `.venv` folder contains your project's virtual environment, a Python environment that is isolated from the rest of your system. This is where uv will install your project's dependencies.
>  `.venv` 文件夹包含了项目的虚拟环境，uv 会将项目依赖安装到这里

See the [project environment](https://docs.astral.sh/uv/concepts/projects/layout/#the-project-environment) documentation for more details.

### `uv.lock`
`uv.lock` is a cross-platform lockfile that contains exact information about your project's dependencies. Unlike the `pyproject.toml` which is used to specify the broad requirements of your project, the lockfile contains the exact resolved versions that are installed in the project environment. This file should be checked into version control, allowing for consistent and reproducible installations across machines.
>  `uv.lock` 是跨平台的锁文件，包含了关于项目依赖的精确信息
>  `pyproject.toml` 指定的需求比较宽泛，而 lockfile 则包含在项目环境中安装的精确解析的版本
>  该文件应该被版本控制，确保跨多个机器的可复现安装

`uv.lock` is a human-readable TOML file but is managed by uv and should not be edited manually.
>  `uv.lock` 为 toml 格式，但是不应该手动管理

See the [lockfile](https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile) documentation for more details.

## Managing dependencies
You can add dependencies to your `pyproject.toml` with the `uv add` command. This will also update the lockfile and project environment:

```
uv add requests
```

>  `uv add` 会更新 `pyproject.toml` 中的依赖，也会更新项目中的锁文件

You can also specify version constraints or alternative sources:

```
# Specify a version constraint
uv add 'requests==2.31.0'

# Add a git dependency
uv add git+https://github.com/psf/requests
```

>  可以指定精确的版本或者另外的 source

If you're migrating from a `requirements.txt` file, you can use `uv add` with the `-r` flag to add all dependencies from the file:

```
# Add all dependencies from `requirements.txt`.
uv add -r requirements.txt -c constraints.txt
```

>  `uv add -r` 用于从某个文件添加依赖

To remove a package, you can use `uv remove`:

```
uv remove requests
```

>  `uv remove` 移除包

To upgrade a package, run `uv lock` with the `--upgrade-package` flag:

```
uv lock --upgrade-package requests
```

The `--upgrade-package` flag will attempt to update the specified package to the latest compatible version, while keeping the rest of the lockfile intact.

>  要更新包，通过 `uv lock --upgrade-package`，它会把包更新到最新的兼容的版本，同时确保锁文件的其他内容不变

See the documentation on [managing dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/) for more details.

## Managing version
The `uv version` command can be used to read your package's version.

To get the version of your package, run `uv version`:

```
uv version
```

To get the version without the package name, use the `--short` option:

```
uv version --short
```

To get version information in a JSON format, use the `--output-format json` option:

```
uv version --output-format json
```

See the [publishing guide](https://docs.astral.sh/uv/guides/package/#updating-your-version) for details on updating your package version.

## Running commands
`uv run` can be used to run arbitrary scripts or commands in your project environment.
>  `uv run` 用于运行项目中任意的脚本和命令

Prior to every `uv run` invocation, uv will verify that the lockfile is up-to-date with the `pyproject.toml`, and that the environment is up-to-date with the lockfile, keeping your project in-sync without the need for manual intervention. `uv run` guarantees that your command is run in a consistent, locked environment.
>  在每次 `uv run` 调用前，uv 会验证 lockfile 和 `pyproject.toml` 是否一致，以及检查环境是否与 lockfrile 同步，进而无需手动干预也可以保持项目同步
>  `uv run` 确保命令在一个一致且锁定的环境中运行

For example, to use `flask`:

```
uv add flask
uv run -- flask run -p 3000
```

Or, to run a script:

example.py

```python
# Require a project dependency
import flask

print("hello world")
```

```
uv run example.py
```

Alternatively, you can use `uv sync` to manually update the environment then activate it before executing a command:
>  也可以先用 `uv sync` 手动更新环境，然后执行下面的命令激活环境，然后再直接运行脚本或命令

[macOS and Linux](https://docs.astral.sh/uv/guides/projects/#__tabbed_1_1)

```
uv sync
source .venv/bin/activate
flask run -p 3000
python example.py
```

[Windows](https://docs.astral.sh/uv/guides/projects/#__tabbed_1_2)

```
uv sync
.venv\Scripts\activate
flask run -p 3000
python example.py
```

Note
The virtual environment must be active to run scripts and commands in the project without `uv run`. Virtual environment activation differs per shell and platform.

>  要不使用 `uv run` 运行脚本和命令，必须先激活虚拟环境
>  每个平台和 shell 的虚拟环境激活都不同

See the documentation on [running commands and scripts](https://docs.astral.sh/uv/concepts/projects/run/) in projects for more details.

## Building distributions
`uv build` can be used to build source distributions and binary distributions (wheel) for your project.
>  `uv build` 用于构建项目的原发布和二进制发布

By default, `uv build` will build the project in the current directory, and place the built artifacts in a `dist/` subdirectory:
>  `uv build` 默认在当前目录构建项目，将构建产物放在 `dist/` 目录

```
uv build
ls dist/
```

See the documentation on [building projects](https://docs.astral.sh/uv/concepts/projects/build/) for more details.

## Next steps
To learn more about working on projects with uv, see the [projects concept](https://docs.astral.sh/uv/concepts/projects/) page and the [command reference](https://docs.astral.sh/uv/reference/cli/#uv).

Or, read on to learn how to [build and publish your project to a package index](https://docs.astral.sh/uv/guides/package/).