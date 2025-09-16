---
completed: false
version: 0.53.0
---
We support the use of the `pyproject.toml` as our manifest file in pixi. This allows the user to keep one file with all configuration. The `pyproject.toml` file is a standard for Python projects. We don't advise to use the `pyproject.toml` file for anything else than python projects, the `pixi.toml` is better suited for other types of projects.
>  我们支持在 pixi 中使用 `pyproject.toml` 作为项目的配置文件，便于用户将所有配置集中在一个文件中
>  `pyproject.toml` 是 Python 项目的标准配置文件，在其他类型项目建议使用 `pixi.toml`

## Initial setup of the `pyproject.toml` file
When you already have a `pyproject.toml` file in your project, you can run `pixi init` in a that folder. Pixi will automatically

- Add a `[tool.pixi.workspace]` section to the file, with the platform and channel information required by pixi;
- Add the current project as an editable pypi dependency;
- Add some defaults to the `.gitignore` and `.gitattributes` files.

>  如果项目中已经有了 `pyproject.toml`，可以运行 `pixi init`，pixi 会自动:
>  - 为 `pyproject.toml` 添加 `[tool.pixi.workspace]` 部分，其中包含 pixi 要求的平台和 channel 信息
>  - 将当前项目添加为可编辑的 pypi 依赖
>  - 将一些默认信息添加到 `.gitignore, .gitattributes` 文件中

If you do not have an existing `pyproject.toml` file , you can run `pixi init --format pyproject` in your project folder. In that case, Pixi will create a `pyproject.toml` manifest from scratch with some sane defaults.
>  如果没有现存的 `pyproject.toml`，可以运行 `pixi init --format pyproject`

## Python dependency
The `pyproject.toml` file supports the `requires_python` field. Pixi understands that field and automatically adds the version to the dependencies.
>  pixi 会理解 `pyproject.toml` 文件中的 `requires-python` ，并自动将这个版本添加到依赖中

This is an example of a `pyproject.toml` file with the `requires_python` field, which will be used as the python dependency:

pyproject.toml

```toml
[project]
name = "my_project"
requires-python = ">=3.9"

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"]
```

Which is equivalent to:

equivalent pixi.toml

```toml
[workspace]
name = "my_project"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"]

[dependencies]
python = ">=3.9"
```

## Dependency section
The `pyproject.toml` file supports the `dependencies` field. Pixi understands that field and automatically adds the dependencies to the project as `[pypi-dependencies]`.
>  pixi 会理解 `pyproject.toml` 文件中的 `dependencies`，并自动将这些依赖添加到项目中

This is an example of a `pyproject.toml` file with the `dependencies` field:

pyproject.toml

```toml
[project]
name = "my_project"
requires-python = ">=3.9"
dependencies = [
    "numpy",
    "pandas",
    "matplotlib",
]

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"]
```

Which is equivalent to:

equivalent pixi.toml

```toml
[workspace]
name = "my_project"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"]

[pypi-dependencies]
numpy = "*"
pandas = "*"
matplotlib = "*"

[dependencies]
python = ">=3.9"
```

You can overwrite these with conda dependencies by adding them to the `dependencies` field:
>  我们可以将这些依赖添加到 `[tool.pixi.dependencies]` 中，将它们作为 conda 依赖

pyproject.toml

```toml
[project]
name = "my_project"
requires-python = ">=3.9"
dependencies = [
    "numpy",
    "pandas",
    "matplotlib",
]

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"]

[tool.pixi.dependencies]
numpy = "*"
pandas = "*"
matplotlib = "*"
```

This would result in the conda dependencies being installed and the pypi dependencies being ignored. As Pixi takes the conda dependencies over the pypi dependencies.
>  这会使得这些依赖以 conda 形式下载，PyPI 上的依赖会被忽略，因为 pixi 优先考虑 conda 依赖

## Optional dependencies
If your python project includes groups of optional dependencies, Pixi will automatically interpret them as [Pixi features](https://pixi.sh/v0.53.0/reference/pixi_manifest/#the-feature-table) of the same name with the associated `pypi-dependencies`.

You can add them to Pixi environments manually, or use `pixi init` to setup the project, which will create one environment per feature. Self-references to other groups of optional dependencies are also handled.

For instance, imagine you have a project folder with a `pyproject.toml` file similar to:

pyproject.toml

```toml
[project]
name = "my_project"
dependencies = ["package1"]

[project.optional-dependencies]
test = ["pytest"]
all = ["package2","my_project[test]"]
```

Running `pixi init` in that project folder will transform the `pyproject.toml` file into:

pyproject.toml

```toml
[project]
name = "my_project"
dependencies = ["package1"]

[project.optional-dependencies]
test = ["pytest"]
all = ["package2","my_project[test]"]

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64"] # if executed on linux

[tool.pixi.environments]
default = {features = [], solve-group = "default"}
test = {features = ["test"], solve-group = "default"}
all = {features = ["all", "test"], solve-group = "default"}
```

In this example, three environments will be created by pixi:

- **default** with 'package1' as pypi dependency
- **test** with 'package1' and 'pytest' as pypi dependencies
- **all** with 'package1', 'package2' and 'pytest' as pypi dependencies

All environments will be solved together, as indicated by the common `solve-group`, and added to the lock file. You can edit the `[tool.pixi.environments]` section manually to adapt it to your use case (e.g. if you do not need a particular environment).

## Dependency groups
If your python project includes dependency groups, Pixi will automatically interpret them as [Pixi features](https://pixi.sh/v0.53.0/reference/pixi_manifest/#the-feature-table) of the same name with the associated `pypi-dependencies`.

You can add them to Pixi environments manually, or use `pixi init` to setup the project, which will create one environment per dependency group.

For instance, imagine you have a project folder with a `pyproject.toml` file similar to:

`[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-7-1)[project] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-7-2)name = "my_project" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-7-3)dependencies = ["package1"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-7-4)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-7-5)[dependency-groups] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-7-6)test = ["pytest"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-7-7)docs = ["sphinx"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-7-8)dev = [{include-group = "test"}, {include-group = "docs"}]`

Running `pixi init` in that project folder will transform the `pyproject.toml` file into:

`[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-1)[project] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-2)name = "my_project" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-3)dependencies = ["package1"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-4)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-5)[dependency-groups] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-6)test = ["pytest"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-7)docs = ["sphinx"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-8)dev = [{include-group = "test"}, {include-group = "docs"}] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-9)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-10)[tool.pixi.workspace] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-11)channels = ["conda-forge"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-12)platforms = ["linux-64"] # if executed on linux [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-13)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-14)[tool.pixi.environments] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-15)default = {features = [], solve-group = "default"} [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-16)test = {features = ["test"], solve-group = "default"} [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-17)docs = {features = ["docs"], solve-group = "default"} [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-8-18)dev = {features = ["dev"], solve-group = "default"}`

In this example, four environments will be created by pixi:

- **default** with 'package1' as pypi dependency
- **test** with 'package1' and 'pytest' as pypi dependencies
- **docs** with 'package1', 'sphinx' as pypi dependencies
- **dev** with 'package1', 'sphinx' and 'pytest' as pypi dependencies

All environments will be solved together, as indicated by the common `solve-group`, and added to the lock file. You can edit the `[tool.pixi.environments]` section manually to adapt it to your use case (e.g. if you do not need a particular environment).

## Example[#](https://pixi.sh/v0.53.0/python/pyproject_toml/#example "Permanent link")

As the `pyproject.toml` file supports the full Pixi spec with `[tool.pixi]` prepended an example would look like this:

pyproject.toml

`[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-1)[project] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-2)name = "my_project" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-3)requires-python = ">=3.9" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-4)dependencies = [     [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-5)    "numpy",    [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-6)    "pandas",    [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-7)    "matplotlib",    [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-8)    "ruff", [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-9)] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-10)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-11)[tool.pixi.workspace] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-12)channels = ["conda-forge"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-13)platforms = ["linux-64", "osx-arm64", "osx-64", "win-64"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-14)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-15)[tool.pixi.dependencies] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-16)compilers = "*" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-17)cmake = "*" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-18)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-19)[tool.pixi.tasks] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-20)start = "python my_project/main.py" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-21)lint = "ruff lint" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-22)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-23)[tool.pixi.system-requirements] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-24)cuda = "11.0" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-25)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-26)[tool.pixi.feature.test.dependencies] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-27)pytest = "*" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-28)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-29)[tool.pixi.feature.test.tasks] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-30)test = "pytest" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-31)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-32)[tool.pixi.environments] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-9-33)test = ["test"]`

## Build-system section[#](https://pixi.sh/v0.53.0/python/pyproject_toml/#build-system-section "Permanent link")

The `pyproject.toml` file normally contains a `[build-system]` section. Pixi will use this section to build and install the project if it is added as a pypi path dependency.

If the `pyproject.toml` file does not contain any `[build-system]` section, Pixi will fall back to [uv](https://github.com/astral-sh/uv)'s default, which is equivalent to the below:

pyproject.toml

`[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-10-1)[build-system] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-10-2)requires = ["setuptools >= 40.8.0"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-10-3)build-backend = "setuptools.build_meta:__legacy__"`

Including a `[build-system]` section is **highly recommended**. If you are not sure of the [build-backend](https://packaging.python.org/en/latest/tutorials/packaging-projects/#choosing-build-backend) you want to use, including the `[build-system]` section below in your `pyproject.toml` is a good starting point. `pixi init --format pyproject` defaults to `hatchling`. The advantages of `hatchling` over `setuptools` are outlined on its [website](https://hatch.pypa.io/latest/why/#build-backend).

pyproject.toml

`[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-11-1)[build-system] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-11-2)build-backend = "hatchling.build" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-11-3)requires = ["hatchling"]`

## Development dependencies with `[tool.uv.sources]`[#](https://pixi.sh/v0.53.0/python/pyproject_toml/#development-dependencies-with-tooluvsources "Permanent link")

Because pixi is using `uv` for building its `pypi-dependencies`, one can use the `tool.uv.sources` section to specify sources for any pypi-dependencies referenced from the main pixi manifest.

### Why is this useful?[#](https://pixi.sh/v0.53.0/python/pyproject_toml/#why-is-this-useful "Permanent link")

When you are setting up a monorepo of some sort and you want to be able for source dependencies to reference each other, you need to use the `[tool.uv.sources]` section to specify the sources for those dependencies. This is because `uv` handles both the resolution of PyPI dependencies and the building of any source dependencies.

### Example[#](https://pixi.sh/v0.53.0/python/pyproject_toml/#example_1 "Permanent link")

Given a source tree:

`[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-12-1). [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-12-2)├── main_project [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-12-3)│   └── pyproject.toml (references a) [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-12-4)├── a [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-12-5)│   └── pyproject.toml (has a dependency on b) [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-12-6)└── b     [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-12-7)    └── pyproject.toml`

Concretly what this looks like in the `pyproject.toml` for `main_project`:

`[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-13-1)[tool.pixi.pypi-dependencies] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-13-2)a = { path = "../a" }`

Then the `pyproject.toml` for `a` should contain a `[tool.uv.sources]` section.

`[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-14-1)[project] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-14-2)name = "a" [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-14-3)# other fields [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-14-4)dependencies = ["flask", "b"] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-14-5)[](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-14-6)[tool.uv.sources] [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-14-7)# Override the default source for flask with main git branch [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-14-8)flask = { git = "github.com/pallets/flask", branch = "main" } [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-14-9)# Reference to b [](https://pixi.sh/v0.53.0/python/pyproject_toml/#__codelineno-14-10)b = { path = "../b" }`

More information about what is allowed in this sections is available in the [uv docs](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-sources)

Note

The main `pixi.toml` or `pyproject.toml` is parsed directly by pixi and not processed by `uv`. This means that you **cannot** use the `[tool.uv.sources]` section in the main `pixi.toml` or `pyproject.toml`. This is a limitation we are aware of, feel free to open an issue if you would like support for [this](https://github.com/prefix-dev/pixi/issues/new/choose).