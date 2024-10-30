# 1 User guide
## 1.1 Quickstart
### Installation
You can install the latest version of `setuptools` using [pip](https://pypi.org/project/pip):

```
pip install --upgrade setuptools[core]
```

Most of the times, however, you don’t have to…
> 大多数情况下，并不需要直接下载 `setuptools` 

Instead, when creating new Python packages, it is recommended to use a command line tool called [build](https://pypi.org/project/build). This tool will automatically download `setuptools` and any other build-time dependencies that your project might have. You just need to specify them in a `pyproject.toml` file at the root of your package, as indicated in the [following section](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#basic-use).
> 当创建新的 Python 包时，推荐使用命令行工具 `build` ，该工具会自动下载 `setuptools` 和其他构建时依赖
> 我们仅需要在我们包的根目录中的 `pyproject.toml` 中指定它们

You can also [install build](https://build.pypa.io/en/latest/installation.html "(in build v1.2.2)") using [pip](https://pypi.org/project/pip):

```
pip install --upgrade build
```

This will allow you to run the command: `python -m build`.

> [!Important]
> Please note that some operating systems might be equipped with the `python3` and `pip3` commands instead of `python` and `pip` (but they should be equivalent). If you don’t have `pip` or `pip3` available in your system, please check out [pip installation docs](https://pip.pypa.io/en/latest/installation/ "(in pip v24.3)").

Every python package must provide a `pyproject.toml` and specify the backend (build system) it wants to use. The distribution can then be generated with whatever tool that provides a `build sdist` -like functionality.
> 所有的 python 包都必须提供 `pyproject.toml` ，并指定它想要使用的后端（构建系统）
> 然后，python 包发布就可以通过任意提供了类似 `build sdist` 功能的工具生成

### Basic Use
When creating a Python package, you must provide a `pyproject.toml` file containing a `build-system` section similar to the example below:
> 创建 Python 包时，我们需要在 `pyproject.toml` 文件中指定一个 `build-system` section

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

This section declares what are your build system dependencies, and which library will be used to actually do the packaging.
> 该 section 声明了我们的构建系统依赖是什么，以及在实际打包时应该使用什么库

> [!Note]
> Package maintainers might be tempted to use `setuptools[core]` as the requirement, given the guidance above. Avoid doing so, as the extra is currently considered an internal implementation detail and is likely to go away in the future and the Setuptools team will not support compatibility for problems arising from packages published with this extra declared. Vendored packages will satisfy the dependencies in the most common isolated build scenarios.

> 避免使用 `setuptools[core]` 作为依赖项，因为 `core` 这个额外的依赖目前被认为是一个内部实现细节，并且将来可能会被移除，Setuptools 团队也不会支持因使用声明了这个额外依赖而产生的兼容性问题
> 在最常见的隔离构建场景中，捆绑的包（vendored packages）将满足依赖项
> (在许多情况下，Python 包管理系统（如 `pip`）已经包含了捆绑的依赖项，这些捆绑的包通常在构建时就已经包含了必要的依赖项，在隔离构建场景中，例如使用虚拟环境或容器，这些捆绑的包可以确保构建过程中所需的依赖项被正确安装和使用)

> [!Note]
> Historically this documentation has unnecessarily listed `wheel` in the `requires` list, and many projects still do that. This is not recommended, as the backend no longer requires the `wheel` package, and listing it explicitly causes it to be unnecessarily required for source distribution builds. You should only include `wheel` in `requires` if you need to explicitly access it during build time (e.g. if your project needs a `setup.py` script that imports `wheel`).

> 历史上，`build-system` 部分的 `requires` 中还需要列出 `wheel`，现在已不推荐
> 因为现在的构建后端不再需要 `wheel` 包来构建轮子文件，显式列出它会导致在构建源码分发时不必要的依赖
> 我们仅在需要在构建时显式访问 `wheel` 时再列出 `wheel` ，例如我们的项目需要一个 `setup.py` 脚本来导入 `wheel`

In addition to specifying a build system, you also will need to add some package information such as metadata, contents, dependencies, etc. This can be done in the same `pyproject.toml` file, or in a separated one: `setup.cfg` or `setup.py` [1](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#setup-py).
> 除了指定构建系统以外，我们还需要在 `pyproject.toml` 中添加一些包信息，例如元数据、内容、依赖等
> 这些信息也可以通过 `setup.cfg/py` 添加

The following example demonstrates a minimum configuration (which assumes the project depends on [requests](https://pypi.org/project/requests) and [importlib-metadata](https://pypi.org/project/importlib-metadata) to be able to run):
> 最小的配置示例如下

pyproject.toml
```toml
[project]
name = "mypackage"
version = "0.0.1"
dependencies = [
    "requests",
    'importlib-metadata; python_version<"3.10"',
]
```

See [Configuring setuptools using pyproject.toml files](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) for more information.

Finally, you will need to organize your Python code to make it ready for distributing into something that looks like the following (optional files marked with `#`):
> 我们最后需要组织好 Python 项目的结构，准备分发

```
mypackage
├── pyproject.toml  # and/or setup.cfg/setup.py (depending on the configuration method)
|   # README.rst or README.md (a nice description of your package)
|   # LICENCE (properly chosen license information, e.g. MIT, BSD-3, GPL-3, MPL-2, etc...)
└── mypackage
    ├── __init__.py
    └── ... (other Python files)
```

With [build installed in your system](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#install-build), you can then run:

```
python -m build
```

You now have your distribution ready (e.g. a `tar.gz` file and a `.whl` file in the `dist` directory), which you can [upload](https://twine.readthedocs.io/en/stable/index.html "(in twine v5.1)") to [PyPI](https://pypi.org/)!
> 配置好 `pyproject.toml` 和组织好项目结构之后，运行 `python -m build` ，我们的发布就会打包为 `dist` 目录中的 `tar.gz` 文件以及 `.whl` 文件，用于发布

Of course, before you release your project to [PyPI](https://pypi.org/), you’ll want to add a bit more information to help people find or learn about your project. And maybe your project will have grown by then to include a few dependencies, and perhaps some data files and scripts. In the next few sections, we will walk through the additional but essential information you need to specify to properly package your project.

> [!Info: Using `setup.py`]
> Setuptools offers first class support for `setup.py` files as a configuration mechanism.
>
>It is important to remember, however, that running this file as a script (e.g. `python setup.py sdist`) is strongly **discouraged**, and that the majority of the command line interfaces are (or will be) **deprecated** (e.g. `python setup.py install`, `python setup.py bdist_wininst`, …).
>
>We also recommend users to expose as much as possible configuration in a more _declarative_ way via the [pyproject.toml](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) or [setup.cfg](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html), and keep the `setup.py` minimal with only the dynamic parts (or even omit it completely if applicable).
>
>See [Why you shouldn’t invoke setup.py directly](https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html) for more background.

> setuptools 提供了使用 `setup.py` 文件作为配置机制的首要支持
> 但不建议将该文件作为脚本运行，其许多命令行界面将被弃用
> 我们建议将尽可能多的配置以声明的方式展现在 `pyproject.toml` 或 `setup.cfg` ，保持 `setup.py` 最小，仅用于动态部分，甚至完全不使用它
### Overview
#### Package discovery
For projects that follow a simple directory structure, `setuptools` should be able to automatically detect all [packages](https://docs.python.org/3.11/glossary.html#term-package "(in Python v3.11)") and [namespaces](https://docs.python.org/3.11/glossary.html#term-namespace "(in Python v3.11)"). However, complex projects might include additional folders and supporting files that not necessarily should be distributed (or that can confuse `setuptools` auto discovery algorithm).
> 对于目录结构简单的项目，`setuptools` 可以自动检测到所有的包和命名空间
> 但复杂的项目可能包含额外的不必要发布的文件夹和支持文件

Therefore, `setuptools` provides a convenient way to customize which packages should be distributed and in which directory they should be found, as shown in the example below:
> 此时，可以自定义 `setuptools` 应该在哪些目录中找哪些包用于发布

pyproject.toml
```toml
# ...
[tool.setuptools.packages]
find = {}  # Scan the project directory with the default parameters

# OR
[tool.setuptools.packages.find]
# All the following settings are optional:
where = ["src"]  # ["."] by default
include = ["mypackage*"]  # ["*"] by default
exclude = ["mypackage.tests*"]  # empty by default
namespaces = false  # true by default
```

When you pass the above information, alongside other necessary information, `setuptools` walks through the directory specified in `where` (defaults to `.`) and filters the packages it can find following the `include` patterns (defaults to `*`), then it removes those that match the `exclude` patterns (defaults to empty) and returns a list of Python packages.
> `setuptools` 会查找在 `where` 中指定的目录，并且根据 `include` 中指定的模式过滤包，同时根据 `exclude` 中的模式进一步排除过滤出的包中的部分包，最后返回一个 Python 包列表

For more details and advanced use, go to [Package Discovery and Namespace Packages](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#package-discovery).

> [!Tip]
> Starting with version 61.0.0, setuptools’ automatic discovery capabilities have been improved to detect popular project layouts (such as the [flat-layout](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#flat-layout) and [src-layout](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout)) without requiring any special configuration. Check out our [reference docs](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#package-discovery) for more information.

#### Entry points and automatic script creation
Setuptools supports automatic creation of scripts upon installation, that run code within your package if you specify them as [entry points](https://packaging.python.org/en/latest/specifications/entry-points/ "(in Python Packaging User Guide)"). An example of how this feature can be used in `pip`: it allows you to run commands like `pip install` instead of having to type `python -m pip install`.
> `setuptools` 支持在安装时自动创建脚本，并且如果将它们指定为 entry point，可以用这些脚本运行我们包内的代码
> 例如，`pip` 允许我们使用 `pip install` 而不是 `python -m pip install`

The following configuration examples show how to accomplish this:

```toml
pyproject.toml

[project.scripts]
cli-name = "mypkg.mymodule:some_func"
```

When this project is installed, a `cli-name` executable will be created. `cli-name` will invoke the function `some_func` in the `mypkg/mymodule.py` file when called by the user. Note that you can also use the `entry-points` mechanism to advertise components between installed packages and implement plugin systems. For detailed usage, go to [Entry Points](https://setuptools.pypa.io/en/latest/userguide/entry_point.html).
> 当该项目被安装时，一个 `cli-name` 可执行文件将会被创建，被运行时，它会调用 `mypkg/mymodule.py` 中的 `some_func` 
> 我们还可以使用 `entry-points` 机制在已安装的包之间宣传组件并实现插件系统

#### Dependency management
Packages built with `setuptools` can specify dependencies to be automatically installed when the package itself is installed. The example below shows how to configure this kind of dependencies:
> 使用 `setuptools` 构建的包可以指定在包安装时需要自动安装的依赖

pyproject.toml
```toml
[project]
# ...
dependencies = [
    "docutils",
    "requests <= 0.4",
]
# ...
```

Each dependency is represented by a string that can optionally contain version requirements (e.g. one of the operators <, >, <=, >=, == or !=, followed by a version identifier), and/or conditional environment markers, e.g. `sys_platform == "win32"` (see [Version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/ "(in Python Packaging User Guide)") for more information).
> 其中的每个依赖都通过字符串加上可选的版本要求表示，也可以包含条件环境标记

When your project is installed, all of the dependencies not already installed will be located (via PyPI), downloaded, built (if necessary), and installed. This, of course, is a simplified scenario. You can also specify groups of extra dependencies that are not strictly required by your package to work, but that will provide additional functionalities. For more advanced use, see [Dependencies Management in Setuptools](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html).
> 指定了依赖后，安装时所有未被安装的依赖会通过 PyPI 定位、安装、构建
> 也可以指定一些不是必须的，但可以为我们的项目提供额外功能的依赖

#### Including Data Files
Setuptools offers three ways to specify data files to be included in your packages. For the simplest use, you can simply use the `include_package_data` keyword:
> setuptools 提供了三种指定我们的包需要包括的数据文件的方式，最简单的是使用 `include_package_data` 关键字

pyproject.toml
```toml
[tool.setuptools]
include-package-data = true
# This is already the default behaviour if you are using
# pyproject.toml to configure your build.
# You can deactivate that with `include-package-data = false`
```

This tells setuptools to install any data files it finds in your packages. The data files must be specified via the [MANIFEST.in](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html#using-manifest-in) file or automatically added by a [Revision Control System plugin](https://setuptools.pypa.io/en/latest/userguide/extension.html#adding-support-for-revision-control-systems). For more details, see [Data Files Support](https://setuptools.pypa.io/en/latest/userguide/datafiles.html).
> 这让 setuptools 在安装时安装我们包中找到的任意数据文件
> 数据文件必须通过 `MANIFEST.in` 文件指定，或通过插件自动添加

#### Development mode
`setuptools` allows you to install a package without copying any files to your interpreter directory (e.g. the `site-packages` directory). This allows you to modify your source code and have the changes take effect without you having to rebuild and reinstall. Here’s how to do it:
> `setuptools` 允许我们在不拷贝任何文件到解释器目录（例如 `site-packages`）的情况下安装包，这允许我们修改源码后可以立即看到改变生效，而不需要重新构建和重新安装包（开发模式下，包会直接指向源代码目录，而不是 `site-packages` 目录）

```
pip install --editable .
```

See [Development Mode (a.k.a. “Editable Installs”)](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) for more information.

> [!Tip]
>Prior to [pip v21.1](https://pip.pypa.io/en/latest/news/#v21-1 "(in pip v24.3)"), a `setup.py` script was required to be compatible with development mode. With late versions of pip, projects without `setup.py` may be installed in this mode.
>
>If you have a version of `pip` older than v21.1 or is using a different packaging-related tool that does not support [**PEP 660**](https://peps.python.org/pep-0660/), you might need to keep a `setup.py` file in your repository if you want to use editable installs.
>
>A simple script will suffice, for example:
> ```python
>from setuptools import setup
>
>setup()
>```
>You can still keep all the configuration in [pyproject.toml](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) and/or [setup.cfg](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html)

> pip v21.1以前，要启动开发模式安装，需要 `setup.py` 脚本

> [!Note]
>
>When building from source code (for example, by `python -m build` or `pip install -e .`) some directories hosting build artefacts and cache files may be created, such as `build`, `dist`, `*.egg-info` [2](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#cache). You can configure your version control system to ignore them (see [GitHub’s .gitignore template](https://github.com/github/gitignore/blob/main/Python.gitignore) for an example).

> 从源码构建时，一些目录会被创建以存放构建文件和缓存文件，记得配置好 `.gitignore` 以避免提交它们

#### Uploading your package to PyPI
After generating the distribution files, the next step would be to upload your distribution so others can use it. This functionality is provided by [twine](https://pypi.org/project/twine) and is documented in the [Python packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/ "(in Python Packaging User Guide)").
> 生成了发布文件之后，就可以上传到 PyPI

#### Transitioning from `setup.py` to declarative config
To avoid executing arbitrary scripts and boilerplate code, we are transitioning from defining all your package information by running `setup()` to doing this declaratively - by using `pyproject.toml` (or older `setup.cfg`).

To ease the challenges of transitioning, we provide a quick [guide](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) to understanding how `pyproject.toml` is parsed by `setuptools`. (Alternatively, here is the [guide](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html) for `setup.cfg`).

>[!Note]
>
>The approach `setuptools` would like to take is to eventually use a single declarative format (`pyproject.toml`) instead of maintaining 2 (`pyproject.toml` / `setup.cfg`). Yet, chances are, `setup.cfg` will continue to be maintained for a long time.

### Resources on Python packaging
Packaging in Python can be hard and is constantly evolving. [Python Packaging User Guide](https://packaging.python.org/) has tutorials and up-to-date references that can help you when it is time to distribute your work.

---

Notes
 
 [1]([1](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#id2), [2](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#id3), [3](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#id4), [4](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#id5), [5](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#id6), [6](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#id8)) New projects are advised to avoid `setup.py` configurations (beyond the minimal stub) when custom scripting during the build is not necessary. Examples are kept in this document to help people interested in maintaining or contributing to existing packages that use `setup.py`. Note that you can still keep most of configuration declarative in [setup.cfg](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html) or [pyproject.toml](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) and use `setup.py` only for the parts not supported in those files (e.g. C extensions). See [note](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#setuppy-discouraged).

[2](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#id9) If you feel that caching is causing problems to your build, specially after changes in the configuration files, consider removing `build`, `dist`, `*.egg-info` before rebuilding or installing your project.

## 1.2 Package Discovery and Namespace Packages

> [!Note]
> a full specification for the keywords supplied to `setup.cfg` or `setup.py` can be found at [keywords reference](https://setuptools.pypa.io/en/latest/references/keywords.html)

> [!Important]
> The examples provided here are only to demonstrate the functionality introduced. More metadata and options arguments need to be supplied if you want to replicate them on your system. If you are completely new to setuptools, the [Quickstart](https://setuptools.pypa.io/en/latest/userguide/quickstart.html) section is a good place to start.

`Setuptools` provides powerful tools to handle package discovery, including support for namespace packages.
> Setuptools 提供了工具帮助处理包查找，包括了对于命名空间包的支持

Normally, you would specify the packages to be included manually in the following manner:
> 一般情况下，我们需要如下显式指定需要包含的包

```toml
# ...
[tool.setuptools]
packages = ["mypkg", "mypkg.subpkg1", "mypkg.subpkg2"]
# ...
```

If your packages are not in the root of the repository or do not correspond exactly to the directory structure, you also need to configure `package_dir`:
> 如果包不是直接在仓库的根目录下，或者没有直接和目录结构对应，我们需要配置 `package_dir` ，指定包所在的目录

```toml
[tool.setuptools]
# ...
package-dir = {"" = "src"}
    # directory containing all the packages (e.g.  src/mypkg1, src/mypkg2)

# OR

[tool.setuptools.package-dir]
mypkg = "lib"
# mypkg.module corresponds to lib/module.py
"mypkg.subpkg1" = "lib1"
# mypkg.subpkg1.module1 corresponds to lib1/module1.py
"mypkg.subpkg2" = "lib2"
# mypkg.subpkg2.module2 corresponds to lib2/module2.py
# ...
```

This can get tiresome really quickly. To speed things up, you can rely on setuptools automatic discovery, or use the provided tools, as explained in the following sections.
> 使用 setuptools 的自动寻找功能，可以帮助我们免除手动指定包位置的麻烦

> [!Important]
> Although `setuptools` allows developers to create a very complex mapping between directory names and package names, it is better to _keep it simple_ and reflect the desired package hierarchy in the directory structure, preserving the same names.

### Automatic discovery
By default `setuptools` will consider 2 popular project layouts, each one with its own set of advantages and disadvantages [1](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#layout1) [2](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#layout2) as discussed in the following sections.
> `setuptools` 默认考虑两种流行的项目布局

Setuptools will automatically scan your project directory looking for these layouts and try to guess the correct values for the [packages](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#declarative-config) and [py_modules](https://setuptools.pypa.io/en/latest/references/keywords.html) configuration.
> `setuptools` 会自动扫描我们的项目目录，查找布局，并且尝试猜测 packages 和 py_modules 配置的值

> [!Important]
>
>Automatic discovery will **only** be enabled if you **don’t** provide any configuration for `packages` and `py_modules`. If at least one of them is explicitly set, automatic discovery will not take place.
>
>**Note**: specifying `ext_modules` might also prevent auto-discover from taking place, unless your opt into [Configuring setuptools using pyproject.toml files](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) (which will disable the backward compatible behaviour).

> 只有在我们没有提供对 `packages/py_modules` 的任何配置时，自动搜索才会被启用
#### src-layout
The project should contain a `src` directory under the project root and all modules and packages meant for distribution are placed inside this directory:
> src 布局中，项目应该在根目录下包含 `src` 目录，并且所有需要发布的模块和包都应该放在该目录中

```
project_root_directory
├── pyproject.toml  # AND/OR setup.cfg, setup.py
├── ...
└── src/
    └── mypkg/
        ├── __init__.py
        ├── ...
        ├── module.py
        ├── subpkg1/
        │   ├── __init__.py
        │   ├── ...
        │   └── module1.py
        └── subpkg2/
            ├── __init__.py
            ├── ...
            └── module2.py
```

This layout is very handy when you wish to use automatic discovery, since you don’t have to worry about other Python files or folders in your project root being distributed by mistake. In some circumstances it can be also less error-prone for testing or when using [**PEP 420**](https://peps.python.org/pep-0420/) -style packages. On the other hand you cannot rely on the implicit `PYTHONPATH=.` to fire up the Python REPL and play with your package (you will need an [editable install](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs) to be able to do that).
> 该布局风格下，我们不必担心自动查找找到我们不想发布的包
> 同时，在进行测试或者使用 PEP 420风格的包（命名空间包）时，该风格下的自动查找也更不易错，但使用该风格时，直接运行 Python 交互式环境（REPL）并使用我们的包时，如果我们依赖于 `.`（当前目录）作为 `PYTHONPATH` 的一部分来导入我们的包，那么这种方式可能不会按预期工作，这是因为没有显式的初始化文件，Python 可能无法正确识别我们的包，为此，需要使用可编辑安装（editable install），以在不进行完整安装的情况下从源代码运行包

#### flat-layout
_(also known as “adhoc”)_

The package folder(s) are placed directly under the project root:
> flat 布局下，包的目录直接放在项目的根目录下（而不是 `src` 下）

```
project_root_directory
├── pyproject.toml  # AND/OR setup.cfg, setup.py
├── ...
└── mypkg/
    ├── __init__.py
    ├── ...
    ├── module.py
    ├── subpkg1/
    │   ├── __init__.py
    │   ├── ...
    │   └── module1.py
    └── subpkg2/
        ├── __init__.py
        ├── ...
        └── module2.py
```

This layout is very practical for using the REPL, but in some situations it can be more error-prone (e.g. during tests or if you have a bunch of folders or Python files hanging around your project root).
> 使用 REPL 时，该布局很方便，但测试时，容易出现错误

To avoid confusion, file and folder names that are used by popular tools (or that correspond to well-known conventions, such as distributing documentation alongside the project code) are automatically filtered out in the case of _flat-layout_:
> 在 flat 布局下，自动查找程序会排除一些广泛知道的目录名和文件名（包名和模块名）

Reserved package names

```
FlatLayoutPackageFinder.DEFAULT_EXCLUDE_: tuple[str, ...]_ _= ('ci', 'ci.*', 'bin', 'bin.*', 'debian', 'debian.*', 'doc', 'doc.*', 'docs', 'docs.*', 'documentation', 'documentation.*', 'manpages', 'manpages.*', 'news', 'news.*', 'newsfragments', 'newsfragments.*', 'changelog', 'changelog.*', 'test', 'test.*', 'tests', 'tests.*', 'unit_test', 'unit_test.*', 'unit_tests', 'unit_tests.*', 'example', 'example.*', 'examples', 'examples.*', 'scripts', 'scripts.*', 'tools', 'tools.*', 'util', 'util.*', 'utils', 'utils.*', 'python', 'python.*', 'build', 'build.*', 'dist', 'dist.*', 'venv', 'venv.*', 'env', 'env.*', 'requirements', 'requirements.*', 'tasks', 'tasks.*', 'fabfile', 'fabfile.*', 'site_scons', 'site_scons.*', 'benchmark', 'benchmark.*', 'benchmarks', 'benchmarks.*', 'exercise', 'exercise.*', 'exercises', 'exercises.*', 'htmlcov', 'htmlcov.*', '[._]*', '[._]*.*')
```

Reserved top-level module names

```
FlatLayoutModuleFinder.DEFAULT_EXCLUDE_: tuple[str, ...]_ _= ('setup', 'conftest', 'test', 'tests', 'example', 'examples', 'build', 'toxfile', 'noxfile', 'pavement', 'dodo', 'tasks', 'fabfile', '[Ss][Cc]onstruct', 'conanfile', 'manage', 'benchmark', 'benchmarks', 'exercise', 'exercises', '[._]*')
```


> [!Warning]
> If you are using auto-discovery with _flat-layout_, `setuptools` will refuse to create [distribution archives](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package "(in Python Packaging User Guide)") with multiple top-level packages or modules.
>
>This is done to prevent common errors such as accidentally publishing code not meant for distribution (e.g. maintenance-related scripts).
>
>Users that purposefully want to create multi-package distributions are advised to use [Custom discovery](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#custom-discovery) or the `src-layout`.

There is also a handy variation of the _flat-layout_ for utilities/libraries that can be implemented with a single Python file:

##### single-module distribution
A standalone module is placed directly under the project root, instead of inside a package folder:
> 对于可以由单个 Python 文件实现的实用程序/库，还有一种 flat 布局的变体，即模块直接放在根目录下，而不是在包目录中

```
project_root_directory
├── pyproject.toml  # AND/OR setup.cfg, setup.py
├── ...
└── single_file_lib.py
```

### Custom discovery
If the automatic discovery does not work for you (e.g., you want to _include_ in the distribution top-level packages with reserved names such as `tasks`, `example` or `docs`, or you want to _exclude_ nested packages that would be otherwise included), you can use the provided tools for package discovery:
> 我们还可以使用 setuptools 提供的工具进行自定义包查找

pyproject. toml
```toml
# ...
[tool.setuptools.packages]
find = {}  # Scanning implicit namespaces is active by default
# OR
find = {namespaces = false}  # Disable implicit namespaces
```

#### Finding simple packages
Let’s start with the first tool. `find:` (`find_packages()`) takes a source directory and two lists of package name patterns to exclude and include, and then returns a list of `str` representing the packages it could find. To use it, consider the following directory:
> 第一个工具是 `find/find_packages()` 
> 它接受一个源目录，以及两个包名列表 (exclude and include pattern)，然后返回一个 str 列表，表明它可以找到的包

```
mypkg
├── pyproject.toml  # AND/OR setup.cfg, setup.py
└── src
    ├── pkg1
    │   └── __init__.py
    ├── pkg2
    │   └── __init__.py
    ├── additional
    │   └── __init__.py
    └── pkg
        └── namespace
            └── __init__.py
```

To have setuptools to automatically include packages found in `src` that start with the name `pkg` and not `additional`:

pypyproject.toml
```
[tool.setuptools.packages.find]
where = ["src"]
include = ["pkg*"]  # alternatively: `exclude = ["additional*"]`
namespaces = false
```

> [!Note]
>
>When using `tool.setuptools.packages.find` in `pyproject.toml`, setuptools will consider [**implicit namespaces**](https://peps.python.org/pep-0420/) by default when scanning your project directory. To avoid `pkg.namespace` from being added to your package list you can set `namespaces = false`. This will prevent any folder without an `__init__.py` file from being scanned.

>[!Important]
>
> `include` and `exclude` accept strings representing [`glob`](https://docs.python.org/3.11/library/glob.html#module-glob "(in Python v3.11)") patterns. These patterns should match the **full** name of the Python module (as if it was written in an `import` statement).
>
>For example if you have `util` pattern, it will match `util/__init__.py` but not `util/files/__init__.py`.
>
>The fact that the parent package is matched by the pattern will not dictate if the submodule will be included or excluded from the distribution. You will need to explicitly add a wildcard (e.g. `util*`) if you want the pattern to also match submodules.

> 父包被 include/exclude pattern 识别并且 include/exclude 并不会影响子包，如果需要 pattern 同样匹配子模块，需要添加 `*`

#### Finding namespace packages
`setuptools` provides `find_namespace:` (`find_namespace_packages()`) which behaves similarly to `find:` but works with namespace packages.
> 第二个工具是 `find_namespace/find_namespace_packages()` ，针对命名空间包

Before diving in, it is important to have a good understanding of what [**namespace packages**](https://peps.python.org/pep-0420/) are. Here is a quick recap.

When you have two packages organized as follows:

```
/Users/Desktop/timmins/foo/__init__.py
/Library/timmins/bar/__init__.py
```

If both `Desktop` and `Library` are on your `PYTHONPATH`, then a namespace package called `timmins` will be created automatically for you when you invoke the import mechanism, allowing you to accomplish the following:

```
>>> import timmins.foo
>>> import timmins.bar
```

as if there is only one `timmins` on your system. The two packages can then be distributed separately and installed individually without affecting the other one.
> 对于路径不同的两个包，如果他们的直接父目录名称相同，且父目录的父目录处于 `PYTHONPATH` 中，则调用 `import` 时，Python 会自动创建名称为直接父目录名称的命名空间包

Now, suppose you decide to package the `foo` part for distribution and start by creating a project directory organized as follows:

```
foo
├── pyproject.toml  # AND/OR setup.cfg, setup.py
└── src
    └── timmins
        └── foo
            └── __init__.py
```

If you want the `timmins.foo` to be automatically included in the distribution, then you will need to specify:

pypyproject.toml
```toml
[tool.setuptools.packages.find]
where = ["src"]
```

When using `tool.setuptools.packages.find` in `pyproject.toml`, setuptools will consider [**implicit namespaces**](https://peps.python.org/pep-0420/) by default when scanning your project directory.

After installing the package distribution, `timmins.foo` would become available to your interpreter.

> [!Warning]
>
>Please have in mind that `find_namespace:` (setup.cfg), `find_namespace_packages()` (setup.py) and `find` (pyproject.toml) will scan **all** folders that you have in your project directory if you use a [flat-layout](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#flat-layout).
>
>If used naïvely, this might result in unwanted files being added to your final wheel. For example, with a project directory organized as follows:
>
>```
foo
├── docs
│   └── conf.py
├── timmins
│   └── foo
│       └── __init__.py
└── tests
>    └── tests_foo
>        └── __init__.py
>```
>
>final users will end up installing not only `timmins.foo`, but also `docs` and `tests.tests_foo`.
>
>A simple way to fix this is to adopt the aforementioned [src-layout](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#src-layout), or make sure to properly configure the `include` and/or `exclude` accordingly.

> [!Tip]
>
>After [building your package](https://setuptools.pypa.io/en/latest/build_meta.html#building), you can have a look if all the files are correct (nothing missing or extra), by running the following commands:
>
>```
tar tf dist/*.tar.gz
unzip -l dist/*.whl
>```
>
>This requires the `tar` and `unzip` to be installed in your OS. On Windows you can also use a GUI program such as [7zip](https://www.7-zip.org/).

### Legacy Namespace Packages
The fact you can create namespace packages so effortlessly above is credited to [**PEP 420**](https://peps.python.org/pep-0420/). It used to be more cumbersome to accomplish the same result. Historically, there were two methods to create namespace packages. One is the `pkg_resources` style supported by `setuptools` and the other one being `pkgutils` style offered by `pkgutils` module in Python. Both are now considered _deprecated_ despite the fact they still linger in many existing packages. These two differ in many subtle yet significant aspects and you can find out more on [Python packaging user guide](https://packaging.python.org/guides/packaging-namespace-packages/).

#### `pkg_resource` style namespace package
This is the method `setuptools` directly supports. Starting with the same layout, there are two pieces you need to add to it. First, an `__init__.py` file directly under your namespace package directory that contains the following:

```python
__import__("pkg_resources").declare_namespace(__name__)
```

And the `namespace_packages` keyword in your `setup.cfg` or `setup.py`:

setup.py
```python
setup(
    # ...
    namespace_packages=['timmins']
)
```

And your directory should look like this

```
foo
├── pyproject.toml  # AND/OR setup.cfg, setup.py
└── src
    └── timmins
        ├── __init__.py
        └── foo
            └── __init__.py
```

Repeat the same for other packages and you can achieve the same result as the previous section.

#### `pkgutil` style namespace package

This method is almost identical to the `pkg_resource` except that the `namespace_packages` declaration is omitted and the `__init__.py` file contains the following:

```python
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

The project layout remains the same and `pyproject.toml/setup.cfg` remains the same.

---

[1](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#id1) [https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure)

[2](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#id2) [https://blog.ionelmc.ro/2017/09/25/rehashing-the-src-layout/](https://blog.ionelmc.ro/2017/09/25/rehashing-the-src-layout/)

## 1.5 Entry Points
Entry points are a type of metadata that can be exposed by packages on installation. They are a very useful feature of the Python ecosystem, and come specially handy in two scenarios:

1. The package would like to provide commands to be run at the terminal. This functionality is known as _console_ scripts. The command may also open up a GUI, in which case it is known as a _GUI_ script. An example of a console script is the one provided by the [pip](https://pypi.org/project/pip) package, which allows you to run commands like `pip install` in the terminal.

2. A package would like to enable customization of its functionalities via _plugins_. For example, the test framework [pytest](https://pypi.org/project/pytest) allows customization via the `pytest11` entry point, and the syntax highlighting tool [pygments](https://pypi.org/project/pygments) allows specifying additional styles using the entry point `pygments.styles`.

> Entry points 即包在安装时暴露的元数据，它们可以用于
> - 包需要提供可以在终端运行的命令，即控制台脚本/GUI 脚本，例如 `pip`
> - 包希望通过插件自定义它的功能，例如 `pytest` 允许通过 `pytest1` 入口点自定义测试框架，以及语法高亮工具 `pygments` 允许通过入口点 `pygments.styles` 指定额外的风格

### Console Scripts
Let us start with console scripts. First consider an example without entry points. Imagine a package defined thus:

```
project_root_directory
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins
        ├── __init__.py
        └── ...
```

with `__init__.py` as:

```python
def hello_world():
    print("Hello world")
```

Now, suppose that we would like to provide some way of executing the function `hello_world()` from the command-line. One way to do this is to create a file `src/timmins/__main__.py` providing a hook as follows:

```python
from . import hello_world

if __name__ == '__main__':
    hello_world()
```

Then, after installing the package `timmins`, we may invoke the `hello_world()` function as follows, through the [runpy](https://docs.python.org/3/library/runpy.html) module:

```
$ python -m timmins
Hello world
```

> 为包定义 `__main__.py` 模块，可以让包被 `python -m <package-name>` 时直接运行 `__main__.py` 中的语句

Instead of this approach using `__main__.py`, you can also create a user-friendly CLI executable that can be called directly without `python -m`. In the above example, to create a command `hello-world` that invokes `timmins.hello_world`, add a console script entry point to your configuration:
> 使用 `__main__.py` 方便我们用 `python -m` 调用包
> 但 setuptools 支持创建可以直接调用的 CLI 可执行文件来调用我们的模块

pyproject.toml
```toml
[project.scripts]
hello-world = "timmins:hello_world"
```

After installing the package, a user may invoke that function by simply calling `hello-world` on the command line:

```
$ hello-world
Hello world
```

Note that any function used as a console script, i.e. `hello_world()` in this example, should not accept any arguments. If your function requires any input from the user, you can use regular command-line argument parsing utilities like [`argparse`](https://docs.python.org/3.11/library/argparse.html#module-argparse "(in Python v3.11)") within the body of the function to parse user input given via [`sys.argv`](https://docs.python.org/3.11/library/sys.html#sys.argv "(in Python v3.11)").
> 注意作为控制台脚本的入口函数应该不接受任何参数
> 如果需要用户的输入，则应该在函数体内通过命令行参数解析程序例如 `argparse` 来解析存储在 `sys.argv` 中的用户输入

You may have noticed that we have used a special syntax to specify the function that must be invoked by the console script, i.e. we have written `timmins:hello_world` with a colon `:` separating the package name and the function name. The full specification of this syntax is discussed in the [last section](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-syntax) of this document, and this can be used to specify a function located anywhere in your package, not just in `__init__.py`.
> 注意指定包中函数的语法是通过 `:` 隔开

### GUI Scripts
In addition to `console_scripts`, Setuptools supports `gui_scripts`, which will launch a GUI application without running in a terminal window.

For example, if we have a project with the same directory structure as before, with an `__init__.py` file containing the following:

import PySimpleGUI as sg

def hello_world():
    sg.Window(title="Hello world", layout=[[]], margins=(100, 50)).read()

Then, we can add a GUI script entry point:

pyproject.toml

[project.gui-scripts]
hello-world = "timmins:hello_world"

setup.cfgsetup.py

Note

To be able to import `PySimpleGUI`, you need to add `pysimplegui` to your package dependencies. See [Dependencies Management in Setuptools](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html) for more information.

Now, running:

$ hello-world

will open a small application window with the title ‘Hello world’.

Note that just as with console scripts, any function used as a GUI script should not accept any arguments, and any user input can be parsed within the body of the function. GUI scripts also use the same syntax (discussed in the [last section](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-syntax)) for specifying the function to be invoked.

Note

The difference between `console_scripts` and `gui_scripts` only affects Windows systems. [[1]](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#use-for-scripts) `console_scripts` are wrapped in a console executable, so they are attached to a console and can use `sys.stdin`, `sys.stdout` and `sys.stderr` for input and output. `gui_scripts` are wrapped in a GUI executable, so they can be started without a console, but cannot use standard streams unless application code redirects them. Other platforms do not have the same distinction.

Note

Console and GUI scripts work because behind the scenes, installers like [pip](https://pypi.org/project/pip) create wrapper scripts around the function(s) being invoked. For example, the `hello-world` entry point in the above two examples would create a command `hello-world` launching a script like this: [[1]](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#use-for-scripts)

import sys
from timmins import hello_world
sys.exit(hello_world())

### Advertising Behavior
Console/GUI scripts are one use of the more general concept of entry points. Entry points more generally allow a packager to advertise behavior for discovery by other libraries and applications. This feature enables “plug-in”-like functionality, where one library solicits entry points and any number of other libraries provide those entry points.

A good example of this plug-in behavior can be seen in [pytest plugins](https://docs.pytest.org/en/latest/writing_plugins.html), where pytest is a test framework that allows other libraries to extend or modify its functionality through the `pytest11` entry point.

The console/GUI scripts work similarly, where libraries advertise their commands and tools like `pip` create wrapper scripts that invoke those commands.

### Entry Points for Plugins
Let us consider a simple example to understand how we can implement entry points corresponding to plugins. Say we have a package `timmins` with the following directory structure:

timmins
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins
        └── __init__.py

and in `src/timmins/__init__.py` we have the following code:

def hello_world():
    print('Hello world')

Basically, we have defined a `hello_world()` function which will print the text ‘Hello world’. Now, let us say we want to print the text ‘Hello world’ in different ways. The current function just prints the text as it is - let us say we want another style in which the text is enclosed within exclamation marks:

!!! Hello world !!!

Let us see how this can be done using plugins. First, let us separate the style of printing the text from the text itself. In other words, we can change the code in `src/timmins/__init__.py` to something like this:

def display(text):
    print(text)

def hello_world():
    display('Hello world')

Here, the `display()` function controls the style of printing the text, and the `hello_world()` function calls the `display()` function to print the text ‘Hello world`.

Right now the `display()` function just prints the text as it is. In order to be able to customize it, we can do the following. Let us introduce a new _group_ of entry points named `timmins.display`, and expect plugin packages implementing this entry point to supply a `display()`-like function. Next, to be able to automatically discover plugin packages that implement this entry point, we can use the [`importlib.metadata`](https://docs.python.org/3.11/library/importlib.metadata.html#module-importlib.metadata "(in Python v3.11)") module, as follows:

from importlib.metadata import entry_points
display_eps = entry_points(group='timmins.display')

Note

Each `importlib.metadata.EntryPoint` object is an object containing a `name`, a `group`, and a `value`. For example, after setting up the plugin package as described below, `display_eps` in the above code will look like this: [[2]](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#package-metadata)

(
    EntryPoint(name='excl', value='timmins_plugin_fancy:excl_display', group='timmins.display'),
    ...,
)

`display_eps` will now be a list of `EntryPoint` objects, each referring to `display()`-like functions defined by one or more installed plugin packages. Then, to import a specific `display()`-like function - let us choose the one corresponding to the first discovered entry point - we can use the `load()` method as follows:

display = display_eps[0].load()

Finally, a sensible behaviour would be that if we cannot find any plugin packages customizing the `display()` function, we should fall back to our default implementation which prints the text as it is. With this behaviour included, the code in `src/timmins/__init__.py` finally becomes:

from importlib.metadata import entry_points
display_eps = entry_points(group='timmins.display')
try:
    display = display_eps[0].load()
except IndexError:
    def display(text):
        print(text)

def hello_world():
    display('Hello world')

That finishes the setup on `timmins`’s side. Next, we need to implement a plugin which implements the entry point `timmins.display`. Let us name this plugin `timmins-plugin-fancy`, and set it up with the following directory structure:

timmins-plugin-fancy
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins_plugin_fancy
        └── __init__.py

And then, inside `src/timmins_plugin_fancy/__init__.py`, we can put a function named `excl_display()` that prints the given text surrounded by exclamation marks:

def excl_display(text):
    print('!!!', text, '!!!')

This is the `display()`-like function that we are looking to supply to the `timmins` package. We can do that by adding the following in the configuration of `timmins-plugin-fancy`:

pyproject.toml
```
# Note the quotes around timmins.display in order to escape the dot .
[project.entry-points."timmins.display"]
excl = "timmins_plugin_fancy:excl_display"
```

setup.cfgsetup.py

Basically, this configuration states that we are a supplying an entry point under the group `timmins.display`. The entry point is named `excl` and it refers to the function `excl_display` defined by the package `timmins-plugin-fancy`.

Now, if we install both `timmins` and `timmins-plugin-fancy`, we should get the following:

```
>>> from timmins import hello_world
>>> hello_world()
!!! Hello world !!!
```

whereas if we only install `timmins` and not `timmins-plugin-fancy`, we should get the following:

```
>>> from timmins import hello_world
>>> hello_world()
Hello world
```

Therefore, our plugin works.

Our plugin could have also defined multiple entry points under the group `timmins.display`. For example, in `src/timmins_plugin_fancy/__init__.py` we could have two `display()`-like functions, as follows:

def excl_display(text):
    print('!!!', text, '!!!')

def lined_display(text):
    print(''.join(['-' for _ in text]))
    print(text)
    print(''.join(['-' for _ in text]))

The configuration of `timmins-plugin-fancy` would then change to:

pyproject.toml

```
[project.entry-points."timmins.display"]
excl = "timmins_plugin_fancy:excl_display"
lined = "timmins_plugin_fancy:lined_display"
```

setup.cfgsetup.py

On the `timmins` side, we can also use a different strategy of loading entry points. For example, we can search for a specific display style:

display_eps = entry_points(group='timmins.display')
try:
    display = display_eps['lined'].load()
except KeyError:
    # if the 'lined' display is not available, use something else
    ...

Or we can also load all plugins under the given group. Though this might not be of much use in our current example, there are several scenarios in which this is useful:

display_eps = entry_points(group='timmins.display')
for ep in display_eps:
    display = ep.load()
    # do something with display
    ...

Another point is that in this particular example, we have used plugins to customize the behaviour of a function (`display()`). In general, we can use entry points to enable plugins to not only customize the behaviour of functions, but also of entire classes and modules. This is unlike the case of console/GUI scripts, where entry points can only refer to functions. The syntax used for specifying the entry points remains the same as for console/GUI scripts, and is discussed in the [last section](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-syntax).

Tip

The recommended approach for loading and importing entry points is the [`importlib.metadata`](https://docs.python.org/3.11/library/importlib.metadata.html#module-importlib.metadata "(in Python v3.11)") module, which is a part of the standard library since Python 3.8 and is non-provisional since Python 3.10. For older versions of Python, its backport [importlib_metadata](https://pypi.org/project/importlib_metadata) should be used. While using the backport, the only change that has to be made is to replace `importlib.metadata` with `importlib_metadata`, i.e.

from importlib_metadata import entry_points
...

In summary, entry points allow a package to open its functionalities for customization via plugins. The package soliciting the entry points need not have any dependency or prior knowledge about the plugins implementing the entry points, and downstream users are able to compose functionality by pulling together plugins implementing the entry points.

### Entry Points Syntax
The syntax for entry points is specified as follows:
> 指定入口点的语法：

```
<name> = <package_or_module>[:<object>[.<attr>[.<nested-attr>]*]]
```

Here, the square brackets `[]` denote optionality and the asterisk `*` denotes repetition. `name` is the name of the script/entry point you want to create, the left hand side of `:` is the package or module that contains the object you want to invoke (think about it as something you would write in an import statement), and the right hand side is the object you want to invoke (e.g. a function).

To make this syntax more clear, consider the following examples:

**Package or module**

If you supply:

```
<name> = <package_or_module>
```

as the entry point, where `<package_or_module>` can contain `.` in the case of sub-modules or sub-packages, then, tools in the Python ecosystem will roughly interpret this value as:

```
import <package_or_module>
parsed_value = <package_or_module>
```

**Module-level object**

If you supply:

```
<name> = <package_or_module>:<object>
```

where `<object>` does not contain any `.`, this will be roughly interpreted as:

```
from <package_or_module> import <object>
parsed_value = <object>
```

**Nested object**

If you supply:

```
<name> = <package_or_module>:<object>.<attr>.<nested_attr>
```

this will be roughly interpreted as:

```
from <package_or_module> import <object>
parsed_value = <object>.<attr>.<nested_attr>
```

In the case of console/GUI scripts, this syntax can be used to specify a function, while in the general case of entry points as used for plugins, it can be used to specify a function, class or module.

---

\[1\]([1](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#id4), [2](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#id5)) Reference: [https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts](https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts)

[2](https://setuptools.pypa.io/en/latest/userguide/entry_point.html#id6) Reference: [https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata)