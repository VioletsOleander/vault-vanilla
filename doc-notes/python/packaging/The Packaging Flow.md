The document aims to outline the flow involved in publishing/distributing a [distribution package](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package), usually to the [Python Package Index (PyPI)](https://pypi.org/). It is written for package publishers, who are assumed to be the package author.

While the [tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) walks through the process of preparing a simple package for release, it does not fully enumerate what steps and files are required, and for what purpose.

Publishing a package requires a flow from the author’s source code to an end user’s Python environment. The steps to achieve this are:

- Have a source tree containing the package. This is typically a checkout from a version control system (VCS).
- Prepare a configuration file describing the package metadata (name, version and so forth) and how to create the build artifacts. For most packages, this will be a `pyproject.toml` file, maintained manually in the source tree.
- Create build artifacts to be sent to the package distribution service (usually PyPI); these will normally be a [source distribution (“sdist”)](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist) and one or more [built distributions (“wheels”)](https://packaging.python.org/en/latest/glossary/#term-Built-Distribution). These are made by a build tool using the configuration file from the previous step. Often there is just one generic wheel for a pure Python package.
- Upload the build artifacts to the package distribution service.

>  发布一个包是一个从作者的源代码到终端用户的 Python 环境的工作流，其中涉及的步骤有：
>  - 准备需要发布的源码版本
>  - 准备描述了包的元数据 (名称、版本等) 以及如何创建构建产物的配置文件，一般是在源码树中维护的 `pyproject.toml`
>  - 创建要发送到包分布服务 (PyPI) 的构建产物，通常包含了一个源发布 sdist 和多个构建发布 wheels，注意这些构建产物依赖于配置文件 `pyproject.toml` 创建。对于纯 Python 包，通常只有一个通用的 wheel
>  - 上传构建产物

At that point, the package is present on the package distribution service. To use the package, end users must:

- Download one of the package’s build artifacts from the package distribution service.
- Install it in their Python environment, usually in its `site-packages` directory. This step may involve a build/compile step which, if needed, must be described by the package metadata.

These last 2 steps are typically performed by [pip](https://packaging.python.org/en/latest/key_projects/#pip) when an end user runs `pip install`.

The steps above are described in more detail below.

>  要使用该软件包，用户必须：
> - 从包分发服务下载软件包的一个构建产物
> - 将其安装到他们的 Python 环境中，通常是在其 `site-packages` 目录中。此步骤可能涉及构建/编译步骤，如果需要，必须由包元数据进行描述
>  这最后两步通常在终端用于运行 `pip install` 时由 [pip](https://packaging.python.org/en/latest/key_projects/#pip) 执行

## The source tree
The source tree contains the package source code, usually a checkout from a VCS. The particular version of the code used to create the build artifacts will typically be a checkout based on a tag associated with the version.

## The configuration file
The configuration file depends on the tool used to create the build artifacts. The standard practice is to use a `pyproject.toml` file in the [TOML format](https://github.com/toml-lang/toml).

At a minimum, the `pyproject.toml` file needs a `[build-system]` table specifying your build tool. There are many build tools available, including but not limited to [flit](https://packaging.python.org/en/latest/key_projects/#flit), [hatch](https://packaging.python.org/en/latest/key_projects/#hatch), [pdm](https://packaging.python.org/en/latest/key_projects/#pdm), [poetry](https://packaging.python.org/en/latest/key_projects/#poetry), [Setuptools](https://packaging.python.org/en/latest/key_projects/#setuptools), [trampolim](https://pypi.org/project/trampolim/), and [whey](https://pypi.org/project/whey/). Each tool’s documentation will show what to put in the `[build-system]` table.

>  配置文件具体依赖于用于创建构建产物的工具，标准做法是使用 `pyproject.toml`
>  `pyproject.toml` 至少需要包含一个 `[build-system]` table, 该 table 用于指定构建工具

For example, here is a table for using [hatch](https://packaging.python.org/en/latest/key_projects/#hatch):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

With such a table in the `pyproject.toml` file, a “[frontend](https://packaging.python.org/en/latest/glossary/#term-Build-Frontend)” tool like [build](https://packaging.python.org/en/latest/key_projects/#build) can run your chosen build tool’s “[backend](https://packaging.python.org/en/latest/glossary/#term-Build-Backend)” to create the build artifacts. Your build tool may also provide its own frontend. An install tool like [pip](https://packaging.python.org/en/latest/key_projects/#pip) also acts as a frontend when it runs your build tool’s backend to install from a source distribution.
>  根据 `pyproject.toml` 文件中的该 table，前端工具例如 `build` 会运行指定构建工具的后端以创建构建产物
>  当然，我们指定的构建工具也可能会提供自己的前端
>  安装工具例如 `pip` 在根据源码分发进行安装时，也会充当前端，运行我们指定的构建工具的后端

The particular build tool you choose dictates what additional information is required in the `pyproject.toml` file. For example, you might specify:

- a `[project]` table containing project [Core Metadata](https://packaging.python.org/en/latest/specifications/core-metadata/) (name, version, author and so forth),
- a `[tool]` table containing tool-specific configuration options.

>  我们选择的特定构建工具还决定了 `pyproject.toml` 中会需要哪些额外信息，例如，我们可能需要指定
>  - `[project]` table，包含项目的核心元数据
>  - `[tool]` table，包含特定于工具的配置选项

Refer to the [pyproject.toml guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#writing-pyproject-toml) for a complete guide to `pyproject.toml` configuration.

## Build artifacts
### The source distribution (sdist)
A source distribution contains enough to install the package from source in an end user’s Python environment. As such, it needs the package source, and may also include tests and documentation. These are useful for end users wanting to develop your sources, and for end user systems where some local compilation step is required (such as a C extension).
>  源发布应该包含可以让用户根据源发布在 Python 环境中安装该包的信息，因此，它需要包含包的源代码，可能还需要测试和文档文件

The [build](https://packaging.python.org/en/latest/key_projects/#build) package knows how to invoke your build tool to create one of these:

```
python3 -m build --sdist source-tree-directory
```

Or, your build tool may provide its own interface for creating an sdist.

>  可以利用 `build` 包调用我们指定的构建工具，创建 sdist

### The built distributions (wheels)
A built distribution contains only the files needed for an end user’s Python environment. No compilation steps are required during the install, and the wheel file can simply be unpacked into the `site-packages` directory. This makes the install faster and more convenient for end users.
>  构建好的分发仅包含用户的 Python 环境中所需要的文件，在安装时不会需要编译步骤，wheel 文件会直接被解压缩到 `site-packages` 目录中

A pure Python package typically needs only one “generic” wheel. A package with compiled binary extensions needs a wheel for each supported combination of Python interpreter, operating system, and CPU architecture that it supports. If a suitable wheel file is not available, tools like [pip](https://packaging.python.org/en/latest/key_projects/#pip) will fall back to installing the source distribution.
>  一个纯 Python 包一般仅需要提供一个 “通用” 的 wheel，但包含了编译了的二进制拓展的 Python 包需要为它支持的每一种 Python 解释器、OS、CPU 架构的组合提供一个 wheel
>  `pip` 在找不到合适的 wheel 文件时，会安装 sdist

The [build](https://packaging.python.org/en/latest/key_projects/#build) package knows how to invoke your build tool to create one of these:

```
python3 -m build --wheel source-tree-directory
```

>  可以利用 `build` 包调用我们指定的构建工具，创建 wheel

Or, your build tool may provide its own interface for creating a wheel.

Note
The default behaviour of [build](https://packaging.python.org/en/latest/key_projects/#build) is to make both an sdist and a wheel from the source in the current directory; the above examples are deliberately specific.

>  `build` 默认会同时创建 sdist 和 wheel 

## Upload to the package distribution service
The [twine](https://packaging.python.org/en/latest/key_projects/#twine) tool can upload build artifacts to PyPI for distribution, using a command like:

```
twine upload dist/package-name-version.tar.gz dist/package-name-version-py3-none-any.whl
```

>  `twine` 可以用于上传构建产物到 PyPI

Or, your build tool may provide its own interface for uploading.

## Download and install
Now that the package is published, end users can download and install the package into their Python environment. Typically this is done with [pip](https://packaging.python.org/en/latest/key_projects/#pip), using a command like:

```
python3 -m pip install package-name
```

>  `pip` 用于安装已发布的构建产物/包

End users may also use other tools like [Pipenv](https://packaging.python.org/en/latest/key_projects/#pipenv), [poetry](https://packaging.python.org/en/latest/key_projects/#poetry), or [pdm](https://packaging.python.org/en/latest/key_projects/#pdm).

>  Last updated on Feb 26, 2025