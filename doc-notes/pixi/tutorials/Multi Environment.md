---
completed: true
version: 0.55.0
---
# Multi Environment
In this tutorial we will show you how to use multiple environments in one Pixi workspace.
>  我们讨论在一个 pixi workspace 中使用多个环境

## Why Is This Useful?
When developing a workspace you often need different tools, libraries or test environments. With Pixi you can define multiple environments in one workspace and switch between them easily. A developer often needs all the tools they can get, whereas your testing infrastructure might not require all those tools, and your production environment might require even less. Setting up different environments for these different use cases can be a hassle, but with Pixi it's easy.

## Glossary
This tutorial possibly uses some new terms, here is a quick overview:

#### **Feature**
A feature defines a part of an environment, but are not useful without being part of an environment. You can define multiple features in one workspace. A feature can contain `tasks`, `dependencies`, `platforms`, `channels` and [more](https://pixi.sh/v0.53.0/reference/pixi_manifest/#the-feature-table). You can mix multiple features to create an environment. Features are defined by adding `[feature.<name>.*]` to a table in the manifest file.
>  功能
>  一个功能定义了环境的一部分，但若没有包含在某个环境中，则无法发挥作用
>  可以在一个 workspace 定义多个功能
>  一个功能可以包含 `tasks, dependencies, platforms, channels` 等，可以混合多个功能来创造一个环境
>  在清单文件中添加 `[feature.<name>.*]` table 可以实现添加功能

#### **Environment**
An environment is a collection of features. Environments can actually be installed and activated to run tasks in. You can define multiple environments in one workspace. Defining environments is done by adding them to the `[environments]` table in the manifest file.
>  环境
>  一个环境就是一组功能，环境可以被安装并且被激活，以在其中运行任务
>  一个 workspace 内可以定义多个环境
>  在清单文件中添加环境到 `[environments]` table 可以实现定义环境

#### **Default**
Instead of specifying `[feature.<name>.dependencies]`, one can populate `[dependencies]` directly. These top level table, are added to the "default" feature, which is added to every environment, unless you specifically opt-out.
>  默认功能
>  我们在顶层 table `[dependencies]` 中添加的依赖都会被添加到默认功能 (也就是我们默认 `pixi add` 的依赖都会添加到默认功能中)，所有的环境都会包含默认功能

## Let's Get Started
We'll simply start with a new workspace, you can skip this step if you already have a Pixi workspace.

```
pixi init workspace
cd workspace
pixi add python
```

Now we have a new Pixi workspace with the following structure:

```
├── .pixi
│   └── envs
│       └── default
├── pixi.lock
└── pixi.toml
```

Note the `.pixi/envs/default` directory, this is where the default environment is stored. If no environment is specified, Pixi will create or use the `default` environment.
>  pixi 将默认环境存储在 `.pixi/envs/default` 目录，没有指定环境时，pixi 就会创建或使用 `default` 环境

### Adding a feature
Let's start adding a simple `test` feature to our workspace. We can do this through the command line, or by editing the `pixi.toml` file. Here we will use the command line, and add a `pytest` dependency to the `test` feature in our workspace.

```
pixi add --feature test pytest
```

>  上述命令为 workspace 添加了一个 `test` feature，并且该 feature 具有 `pytest` 依赖

This will add the following to our `pixi.toml` file:

```toml
[feature.test.dependencies]
pytest = "*"
```

This table acts exactly the same as a normal `dependencies` table, but it is only used when the `test` feature is part of an environment.

>  该命令实际上会修改 `pixi.toml` 中的 `[feature.test.dependencies]` ，这个 table 的格式和 `dependencies` table 一致，但这个 table 只会在 `test` feature 是当前环境的一部分时才会被使用

### Adding an environment
We will add the `test` environment to our workspace to add some testing tools. We can do this through the command line, or by editing the `pixi.toml` file. Here we will use the command line:

```
pixi workspace environment add test --feature test
```

>  上述命令为 workspace 添加了名为 `test` 的环境，并且该环境具有 `test` feature

This will add the following to our `pixi.toml` file:

```toml
[environments]
test = ["test"]
```

>  实际上它会修改 `pixi.toml` 中的 `[environments]` table

### Running a task
We can now run a task in our new environment.

```
pixi run --environment test pytest --version
```

>  创建环境后，就可以指定在该环境下运行任务

This has created the test environment, and run the `pytest --version` command in it. You can see the environment will be added to the `.pixi/envs` directory.

```
├── .pixi
│   └── envs
│       ├── default
│       └── test
```

If you want to see the environment, you can use the `pixi list` command.

```
pixi list --environment test
```

If you have special test commands that always fit with the test environment you can add them to the `test` feature.

```
# Adding the 'test' task to the 'test' feature and setting it to run `pytest`
pixi task add test --feature test pytest
```

>  上述命令为 `test` feature 添加了一个名为 `test` 的任务，它的内容是运行 `pytest`

This will add the following to our `pixi.toml` file:

```toml
[feature.test.tasks]
test = "pytest"
```

>  该命令实际上会修改 `feature.test.tasks` table 的内容

Now you don't have to specify the environment when running the test command.

```
pixi run test
```

In this example similar to running `pixi run --environment test pytest`

This works as long as there is only one of the environments that has the `test` task.

>  如果只有一个环境具有 `test` 任务，我们之后再运行 `test` 任务就不需要指定环境，直接 `pixi run test` 即可

## Using multiple environments to test multiple versions of a package
In this example we will use multiple environments to test a package against multiple versions of Python. This is a common use-case when developing a python library. This workflow can be translated to any setup where you want to have multiple environments to test against a different dependency setups.
>  利用 pixi 的功能，我们可以实现在 workspace 中使用多个环境，在多个版本的 Python 下测试一个包

For this example we assume you have run the commands in the previous example, and have a project with a `test` environment. To allow python being flexible in the new environments we need to set it to a more flexible version e.g. `*`.

```
pixi add "python=*"
```

We will start by setting up two features, `py311` and `py312`.

```
pixi add --feature py311 python=3.11
pixi add --feature py312 python=3.12
```

We'll add the `test` and Python features to the corresponding environments.

```
pixi workspace environment add test-py311 --feature py311 --feature test
pixi workspace environment add test-py312 --feature py312 --feature test
```

This should result in adding the following to the `pixi.toml`:

```toml
[feature.py311.dependencies]
python = "3.11.*"

[feature.py312.dependencies]
python = "3.12.*"

[environments]
test-py311 = ["py311", "test"]
test-py312 = ["py312", "test"]
```

Now we can run the test command in both environments.

```
pixi run --environment test-py311 test
pixi run --environment test-py312 test
# Or using the task directly, which will spawn a dialog to select the environment of choice
pixi run test
```

>  我们将某个 task 添加到某个 feature 之后，使用 `pixi run <task-name>`，pixi 会提示我们选择一个环境 (这个环境包含了 task 附加的 feature) 来运行该 task

These could now run in CI to test separate environments:

.github/workflows/test.yml

```yaml
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      environment: [test-py311, test-py312]
  steps:
  - uses: actions/checkout@v4
  - uses: prefix-dev/setup-pixi@v0
    with:
      environments: ${{ matrix.environment }}
  - run: pixi run -e ${{ matrix.environment }} test
```

More info on that in the GitHub actions [documentation](https://pixi.sh/v0.53.0/integration/ci/github_actions/).

## Development, Testing, Production environments
This assumes a clean project, so if you have been following along, you might want to start a new project.

```
pixi init production_project
cd production_project
```

Like before we'll start with creating multiple features.

```
pixi add numpy python # default feature
pixi add --feature dev jupyterlab
pixi add --feature test pytest
```

Now we'll add the environments. To accommodate the different use-cases we'll add a `production`, `test` and `default` environment.

- The `production` environment will only have the `default` feature, as that is the bare minimum for the project to run.
- The `test` environment will have the `test` and the `default` features, as we want to test the project and require the testing tools.
- The `default` environment will have the `dev` and `test` features.

>   我们先添加好三个 feature: `dev, test, default`
>   之后我们设置三个环境:
>   - `production`，只有 `default` feature，只需要运行项目的最小依赖
>   - `test`，有 `default, test` feature，具有需要运行测试工具的依赖
>   - `default`，有 `dev, test, default` feature，默认的开发环境

We make this the default environment as it will be the easiest to run locally, as it avoids the need to specify the environment when running tasks.

We'll also add the `solve-group` `prod` to the environments, this will make sure that the dependencies are solved as if they were in the same environment. This will result in the `production` environment having the exact same versions of the dependencies as the `default` and `test` environment. This way we can be sure that the project will run in the same way in all environments.

```
pixi workspace environment add production --solve-group prod
pixi workspace environment add test --feature test --solve-group prod
# --force is used to overwrite the default environment (--force 使得这个 default 环境覆盖掉 pixi 默认创建的 default 环境)
pixi workspace environment add default --feature dev --feature test --solve-group prod --force
```

>  我们还将 `solve-group` `prod` (名为 `prod` 的解析组) 添加到上述的三个环境中，这会使得三个环境的依赖一同解析，使得它们的共享依赖版本完全相同
>  这使得我们确保项目在三个环境中都运行得一样

> `solve-group` 是 pixi 用于控制依赖解析的一种高级特性，默认情况下，每个环境独立进行依赖解析，即使它们共享相同的依赖项 (`default` feature 中的依赖项)，也可能安装不同的包版本
> 使用 `solve-group` 之后，多个环境会被视为同一组，它们的依赖会被一起统一求解 (一起算出一个兼容的版本组合)

>  例如没有 `solve-group`，每个环境找自己合适的依赖版本，可能导致 `dev, prod` 使用不同版本的 `request` 包，有了 `solve-group`，各个环境使用的 `request` 包就是一个版本
>  这可以使得我们的开发、测试、生产环境行为一致

If we run `pixi list -x` for the environments we can see that the different environments have the exact same dependency versions.

```
# Default environment
Package     Version  Build               Size       Kind   Source
jupyterlab  4.3.4    pyhd8ed1ab_0        6.9 MiB    conda  jupyterlab
numpy       2.2.1    py313ha4a2180_0     6.2 MiB    conda  numpy
pytest      8.3.4    pyhd8ed1ab_1        253.1 KiB  conda  pytest
python      3.13.1   h4f43103_105_cp313  12.3 MiB   conda  python

Environment: test
Package  Version  Build               Size       Kind   Source
numpy    2.2.1    py313ha4a2180_0     6.2 MiB    conda  numpy
pytest   8.3.4    pyhd8ed1ab_1        253.1 KiB  conda  pytest
python   3.13.1   h4f43103_105_cp313  12.3 MiB   conda  python

Environment: production
Package  Version  Build               Size      Kind   Source
numpy    2.2.1    py313ha4a2180_0     6.2 MiB   conda  numpy
python   3.13.1   h4f43103_105_cp313  12.3 MiB  conda  python
```

### Non default environments
When you want to have an environment that doesn't have the `default` feature, you can use the `--no-default-feature` flag. This will result in the environment not having the `default` feature, and only the features you specify.
>  如果我们想要一个没有 `default` feature 的环境，我们可以使用 `--no-default-feature` 

A common use-case of this would be having an environment that can generate your documentation.
>  这通常用于创建一个只用于生成文档的环境

Let's add the `mkdocs` dependency to the `docs` feature.

```
pixi add --feature docs mkdocs
```

Now we can add the `docs` environment without the `default` feature.

```
pixi workspace environment add docs --feature docs --no-default-feature
```

If we run `pixi list -x -e docs` we can see that it only has the `mkdocs` dependency.

```
Environment: docs
Package  Version  Build         Size     Kind   Source
mkdocs   1.6.1    pyhd8ed1ab_1  3.4 MiB  conda  mkdocs
```

## Conclusion
The multiple environment feature is extremely powerful and can be used in many different ways. There is much more to explore in the [reference](https://pixi.sh/v0.53.0/reference/pixi_manifest/#the-feature-and-environments-tables) and [advanced](https://pixi.sh/v0.53.0/workspace/multi_environment/) sections. If there are any questions, or you know how to improve this tutorial, feel free to reach out to us on [GitHub](https://github.com/prefix-dev/pixi).
