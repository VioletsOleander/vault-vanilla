---
version: 0.53.0
completed: true
---
# Making a Pixi project
Pixi's biggest strength is its ability to create reproducible powerful and flexible projects. Let's go over the common steps to create a Pixi project.

## Creating a Pixi project
To create a new Pixi project, you can use the `pixi init` command:

```
pixi init my_project
```

>  `pixi init` 初始化项目

This command creates a new directory called `my_project` with the following structure:

```
my_project
├── .gitattributes
├── .gitignore
└── pixi.toml
```

The `pixi.toml` file is the manifest of your Pixi project. It contains all the information about your project, such as its channels, platforms, dependencies, tasks, and more.

>  `pixi.toml` 文件包含了项目的 channels, 平台，依赖，任务等元信息

The one created by `pixi init` is a minimal manifest that looks like this:

pixi.toml

```toml
[workspace]
authors = ["Jane Doe <jane.doe@example.com>"]
channels = ["conda-forge"]
name = "my_project"
platforms = ["osx-arm64"]
version = "0.1.0"

[tasks]

[dependencies]
```

Do you want autocompletion of the manifest file?
As `pixi.toml` has a JSON schema, it is possible to use IDE’s like VSCode to edit the field with autocompletion. Install the Even [Better TOML VSCode](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml) extension to get the best experience. Or use the integrated schema support in PyCharm.

## Managing dependencies
After creating the project, you can start adding dependencies to the project. Pixi uses the `pixi add` command to add dependencies to the project. This command will , by default, add the **conda** dependency to the `pixi.toml`, solve the dependencies, write the lockfile and install the package in the environment. For example, lets add `numpy` and `pytest` to the project.

```
pixi add numpy pytest
```

>  `pixi add` 添加依赖，默认行为是将依赖作为 conda 依赖添加到 `pixi.toml`，解析依赖、写入 lockfile 然后安装依赖

This results in these lines being added:

pixi.toml

```toml
[dependencies]
numpy = ">=2.2.6,<3"
pytest = ">=8.3.5,<9"
```

You can also specify the version of the dependency you want to add.

```
pixi add numpy==2.2.6 pytest==8.3.5
```

### PyPI dependencies
Pixi normally uses `conda` packages for dependencies, but you can also add dependencies from PyPI. Pixi will make sure it doesn't try to install the same package from both sources, and avoid conflicts between them.

If you want to add them to your project you can do that with the `--pypi` flag:

```
pixi add --pypi requests
```

>  `pixi add --pypi` 添加 PyPI 依赖, pixi 确保不会从 conda, pypi 安装相同的包，并且会避免它们之间的冲突

This will add the `requests` package from PyPI to the project:

pixi.toml

```toml
[pypi-dependencies]
requests = ">=2.31.0,<3"
```

## Lockfile
Pixi will always create a lockfile when the dependencies are solved. This file will contain all the exact version of the packages and their dependencies. Resulting in a reproducible environment, that you can share with others or use for testing and deployment.
>  Pixi 会在求解了依赖之后创建 lockfile
>  lockfile 包含了所有包的确切版本，以及它们依赖的确切版本，确保可复现的环境

The lockfile is called `pixi.lock` and it is created in the root of the project. It contains all the information about the dependencies, such as their versions, channels, platforms, and more.
>  `pixi.lock` 会包含依赖的版本、channels、platforms 等各种详细信息

pixi.lock

```yaml
version: 6
environments:
  default:
    channels:
    - url: https://prefix.dev/conda-forge/
    indexes:
    - https://pypi.org/simple
    packages:
      osx-arm64:
      - conda: https://prefix.dev/conda-forge/osx-arm64/bzip2-1.0.8-h99b78c6_7.conda
      - pypi: ...
packages:
- conda: https://prefix.dev/conda-forge/osx-arm64/bzip2-1.0.8-h99b78c6_7.conda
  sha256: adfa71f158cbd872a36394c56c3568e6034aa55c623634b37a4836bd036e6b91
  md5: fc6948412dbbbe9a4c9ddbbcfe0a79ab
  depends:
  - __osx >=11.0
  license: bzip2-1.0.6
  license_family: BSD
  size: 122909
  timestamp: 1720974522888
- pypi: ...
```

## Managing tasks
Pixi has a built-in cross-platform task runner that allows you to define tasks in the manifest. This is a great way to share tasks with others and to ensure that the same tasks are run in the same environment. The tasks are defined in the `pixi.toml` file under the `[tasks]` section.
>  Pixi 有内建的跨平台任务运行器，任务可以定义在 `pixi.toml` 中的 `[tasks]`

You can add one to your project by running the `pixi task add` command.

```
pixi task add hello "echo Hello, World!"
```

>  `pixi task add` 添加任务

This will add the following lines to the `pixi.toml` file:

pixi.toml

```toml
[tasks]
hello = "echo Hello, World!"
```

You can then run the task using the `pixi run` command:

```
pixi run hello
```

>  `pixi run` 运行任务

This will execute the command `echo Hello, World!` in the environment.

Do you want use more powerful features?
Tasks can be much more powerful for example:

```toml
[tasks.name-of-powerful-task]
cmd = "echo This task can do much more! Like have {{ arguments }} and {{ "minijinja" | capitalize }} templates."

# List of tasks that must be run before this one.
depends-on = ["other-task"]

# Working directory relative to the root of the project
cwd = "current/working/directory"

# Define required arguments for the task
args = [{ arg = "arguments", default = "default arguments" }]

# Only run if the input files have changed
input = ["src"]
# Only run if the output files are missing
output = ["output.txt"]

# Set environment variables for the task
env = { MY_ENV_VAR = "value" }
```

More information about tasks can be found in the [Tasks](https://pixi.sh/v0.53.0/workspace/advanced_tasks/) section of the documentation.

## Environments
Pixi always creates a virtual environment for your project. These environments are [located](https://pixi.sh/v0.53.0/reference/pixi_configuration/#detached-environments "Find out how to move this location if required") in the `.pixi/envs` directory in the root of your project.
>  Pixi 会为项目创建虚拟环境，位于 `.pixi/envs` 目录下

Using these environments is as simple as running the `pixi run` or `pixi shell` command. These commands will automatically activate the environment and run the command in it.

```
pixi run python -VV
# or:
pixi shell
python -VV
exit
```

>  `pixi run python`, `pixi shell` 会自动激活虚拟环境

