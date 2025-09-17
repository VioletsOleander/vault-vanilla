---
completed: true
version: 0.8.17
---
# The uv build backend
A build backend transforms a source tree (i.e., a directory) into a source distribution or a wheel.
>  构建后端将源码树 (即一个目录) 转化为源发布包或 wheel 包

uv supports all build backends (as specified by [PEP 517](https://peps.python.org/pep-0517/)), but also provides a native build backend (`uv_build`) that integrates tightly with uv to improve performance and user experience.
>  uv 支持所有由 PEP 517 规定的构建后端，并提供了一个原生构建后端 `uv_build`

## Choosing a build backend
The uv build backend is a great choice for most Python projects. It has reasonable defaults, with the goal of requiring zero configuration for most users, but provides flexible configuration to accommodate most Python project structures. It integrates tightly with uv, to improve messaging and user experience. It validates project metadata and structures, preventing common mistakes. And, finally, it's very fast.
>  `uv_build` 后端是大多数 Python 项目的理想选择，它具备合理的默认配置，其目标是让大多数用户零配置就能使用
>  它也提供了灵活的配置选项，来适应多种 Python 项目结构
>  `uv_build` 和 `uv` 深度集成，可提升信息显示质量和用户体验
>  它还能验证项目元信息和结构，避免常见的错误
>  以及，它的构建速度非常快

The uv build backend currently **only supports pure Python code**. An alternative backend is required to build a [library with extension modules](https://docs.astral.sh/uv/concepts/projects/init/#projects-with-extension-modules).
>  目前，`uv_build` 后端仅支持纯 Python 代码，如果需要构建带有拓展模块的库，需要使用其他的构建后端

Tip
While the backend supports a number of options for configuring your project structure, when build scripts or a more flexible project layout are required, consider using the [hatchling](https://hatch.pypa.io/latest/config/build/#build-system) build backend instead.
>  虽然 `uv_build` 支持许多选项来配置项目结构，但如果要求构建脚本或更灵活的项目布局，建议使用 hatchling

## Using the uv build backend
To use uv as a build backend in an existing project, add `uv_build` to the [`[build-system]`](https://docs.astral.sh/uv/concepts/projects/config/#build-systems) section in your `pyproject.toml`:
>  要使用 `uv_build` 作为后端，在 `[build-system]` 中添加以下部分:

pyproject.toml

```toml
[build-system]
requires = ["uv_build>=0.8.17,<0.9.0"]
build-backend = "uv_build"
```

Note
The uv build backend follows the same [versioning policy](https://docs.astral.sh/uv/reference/policies/versioning/) as uv. Including an upper bound on the `uv_build` version ensures that your package continues to build correctly as new versions are released.
>  uv build backend 遵循和 uv 相同的版本策略
>  为 `uv_build` 版本添加上界可以确保在新版本发布时，包仍然可以正确构建

To create a new project that uses the uv build backend, use `uv init`:

```
uv init
```

When the project is built, e.g., with [`uv build`](https://docs.astral.sh/uv/guides/package/), the uv build backend will be used to create the source distribution and wheel.
>  使用 `uv init` 就会创建使用 uv build backend 的项目

## Bundled build backend
The build backend is published as a separate package (`uv_build`) that is optimized for portability and small binary size. However, the `uv` executable also includes a copy of the build backend, which will be used during builds performed by uv, e.g., during `uv build`, if its version is compatible with the `uv_build` requirement. If it's not compatible, a compatible version of the `uv_build` package will be used. Other build frontends, such as `python -m build`, will always use the `uv_build` package, typically choosing the latest compatible version.
>  uv build backend 以独立包的形式发布 (`uv_build`)，该包经过优化，可移植性好且二进制体积小
>  `uv` 可执行文件也包含了该构建后端的一份拷贝，当 `uv` 使用构建操作时 (`uv build`)，将优先使用内置版本，但如果内置版本不兼容，则使用一个兼容版本的 `uv_build`
>  其他的构建前端例如 `python -m build` 则通常使用独立包版本的 `uv_build`，通常会选择最新的兼容版本

## Modules
Python packages are expected to contain one or more Python modules, which are directories containing an `__init__.py`. By default, a single root module is expected at `src/<package_name>/__init__.py`.
>  Python 包通常包含一个或多个 Python 模块，Python 模块即包含了 `__init__.py` 的目录
>  默认情况下，期望在 `src/<package_name>/__init__.py` 的位置存在一个根模块

>  这里的概念似乎有误？

For example, the structure for a project named `foo` would be:

```
pyproject.toml
src
└── foo
    └── __init__.py
```

uv normalizes the package name to determine the default module name: the package name is lowercased and dots and dashes are replaced with underscores, e.g., `Foo-Bar` would be converted to `foo_bar`.
>  uv 会将包名规范化来确定默认的模块名: 包名会变为小写，点好和连字符会被替换为 `_`

The `src/` directory is the default directory for module discovery.
>  `src/` 目录是查找模块的默认目录

These defaults can be changed with the `module-name` and `module-root` settings. For example, to use a `FOO` module in the root directory, as in the project structure:
>  可以在 `module-name, module-root` 设定这些默认值

```
pyproject.toml
FOO
└── __init__.py
```

The correct build configuration would be:

pyproject.toml

```toml
[tool.uv.build-backend]
module-name = "FOO"
module-root = ""
```

