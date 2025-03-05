# Installing Packages
>  Last updated on Feb 26, 2025

This section covers the basics of how to install Python [packages](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package).

It’s important to note that the term “package” in this context is being used to describe a bundle of software to be installed (i.e. as a synonym for a [distribution](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package)). It does not refer to the kind of [package](https://packaging.python.org/en/latest/glossary/#term-Import-Package) that you import in your Python source code (i.e. a container of modules). It is common in the Python community to refer to a [distribution](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) using the term “package”. Using the term “distribution” is often not preferred, because it can easily be confused with a Linux distribution, or another larger software distribution like Python itself.
>  注意，在此上下文中，“包” 用于描述需要安装的软件集合，即作为 “发布” 的同义词，而并不特指一个 Python package
>  Python 社区常用 “包” 替代 “发布”，以避免和 Linux 发行或更大的软件发行例如 Python 本身混淆

## Requirements for Installing Packages
This section describes the steps to follow before installing other Python packages.

### Ensure you can run Python from the command line
Before you go any further, make sure you have Python and that the expected version is available from your command line. You can check this by running:

```
python3 --version
```

You should get some output like `Python 3.6.3`. If you do not have Python, please install the latest 3.x version from [python.org](https://www.python.org/) or refer to the [Installing Python](https://docs.python-guide.org/starting/installation/#installation "(in pythonguide v0.0.1)") section of the Hitchhiker’s Guide to Python.

Note
If you’re a newcomer and you get an error like this:

```
>>> python3 --version
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'python3' is not defined
```

It’s because this command and other suggested commands in this tutorial are intended to be run in a _shell_ (also called a _terminal_ or _console_). See the Python for Beginners [getting started tutorial](https://opentechschool.github.io/python-beginners/en/getting_started.html#what-is-python-exactly) for an introduction to using your operating system’s shell and interacting with Python.

Note
If you’re using an enhanced shell like IPython or the Jupyter notebook, you can run system commands like those in this tutorial by prefacing them with a `!` character:

```
In [1]: import sys
        !{sys.executable} --version
Python 3.6.3
```

It’s recommended to write `{sys.executable}` rather than plain `python` in order to ensure that commands are run in the Python installation matching the currently running notebook (which may not be the same Python installation that the `python` command refers to).

Note
Due to the way most Linux distributions are handling the Python 3 migration, Linux users using the system Python without creating a virtual environment first should replace the `python` command in this tutorial with `python3` and the `python -m pip` command with `python3 -m pip --user`. Do _not_ run any of the commands in this tutorial with `sudo`: if you get a permissions error, come back to the section on creating virtual environments, set one up, and then continue with the tutorial as written.

### Ensure you can run pip from the command line
Additionally, you’ll need to make sure you have [pip](https://packaging.python.org/en/latest/key_projects/#pip) available. You can check this by running:

```
python3 -m pip --version
```

If you installed Python from source, with an installer from [python.org](https://www.python.org/), or via [Homebrew](https://brew.sh/) you should already have pip. If you’re on Linux and installed using your OS package manager, you may have to install pip separately, see [Installing pip/setuptools/wheel with Linux Package Managers](https://packaging.python.org/en/latest/guides/installing-using-linux-tools/).

If `pip` isn’t already installed, then first try to bootstrap it from the standard library:

```
python3 -m ensurepip --default-pip
```

If that still doesn’t allow you to run `python -m pip`:

- Securely Download [get-pip.py](https://bootstrap.pypa.io/get-pip.py) [1](https://packaging.python.org/en/latest/tutorials/installing-packages/#id7)
- Run `python get-pip.py`. [2](https://packaging.python.org/en/latest/tutorials/installing-packages/#id8) This will install or upgrade pip. Additionally, it will install [Setuptools](https://packaging.python.org/en/latest/key_projects/#setuptools) and [wheel](https://packaging.python.org/en/latest/key_projects/#wheel) if they’re not installed already.
    Warning
    Be cautious if you’re using a Python install that’s managed by your operating system or another package manager. get-pip.py does not coordinate with those tools, and may leave your system in an inconsistent state. You can use `python get-pip.py --prefix=/usr/local/` to install in `/usr/local` which is designed for locally-installed software.

### Ensure pip, setuptools, and wheel are up to date
While `pip` alone is sufficient to install from pre-built binary archives, up to date copies of the `setuptools` and `wheel` projects are useful to ensure you can also install from source archives:

```
python3 -m pip install --upgrade pip setuptools wheel
```

### Optionally, create a virtual environment
See [section below](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments) for details, but here’s the basic [venv](https://docs.python.org/3/library/venv.html "(in Python v3.13)") [3](https://packaging.python.org/en/latest/tutorials/installing-packages/#id9) command to use on a typical Linux system:

```
python3 -m venv tutorial_env
source tutorial_env/bin/activate
```

This will create a new virtual environment in the `tutorial_env` subdirectory, and configure the current shell to use it as the default `python` environment.

## Creating Virtual Environments
Python “Virtual Environments” allow Python [packages](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) to be installed in an isolated location for a particular application, rather than being installed globally. If you are looking to safely install global command line tools, see [Installing stand alone command line tools](https://packaging.python.org/en/latest/guides/installing-stand-alone-command-line-tools/).

Imagine you have an application that needs version 1 of LibFoo, but another application requires version 2. How can you use both these applications? If you install everything into /usr/lib/python3.6/site-packages (or whatever your platform’s standard location is), it’s easy to end up in a situation where you unintentionally upgrade an application that shouldn’t be upgraded.

Or more generally, what if you want to install an application and leave it be? If an application works, any change in its libraries or the versions of those libraries can break the application.

Also, what if you can’t install [packages](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) into the global site-packages directory? For instance, on a shared host.

In all these cases, virtual environments can help you. They have their own installation directories and they don’t share libraries with other virtual environments.

Currently, there are two common tools for creating Python virtual environments:

- [venv](https://docs.python.org/3/library/venv.html "(in Python v3.13)") is available by default in Python 3.3 and later, and installs [pip](https://packaging.python.org/en/latest/key_projects/#pip) into created virtual environments in Python 3.4 and later (Python versions prior to 3.12 also installed [Setuptools](https://packaging.python.org/en/latest/key_projects/#setuptools)).
- [virtualenv](https://packaging.python.org/en/latest/key_projects/#virtualenv) needs to be installed separately, but supports Python 2.7+ and Python 3.3+, and [pip](https://packaging.python.org/en/latest/key_projects/#pip), [Setuptools](https://packaging.python.org/en/latest/key_projects/#setuptools) and [wheel](https://packaging.python.org/en/latest/key_projects/#wheel) are always installed into created virtual environments by default (regardless of Python version).

The basic usage is like so:

Using [venv](https://docs.python.org/3/library/venv.html "(in Python v3.13)"):

```
python3 -m venv <DIR>
source <DIR>/bin/activate
```

Using [virtualenv](https://packaging.python.org/en/latest/key_projects/#virtualenv):

```
python3 -m virtualenv <DIR>
source <DIR>/bin/activate
```

For more information, see the [venv](https://docs.python.org/3/library/venv.html "(in Python v3.13)") docs or the [virtualenv](https://virtualenv.pypa.io/en/stable/index.html "(in virtualenv v0.1)") docs.

The use of **source** under Unix shells ensures that the virtual environment’s variables are set within the current shell, and not in a subprocess (which then disappears, having no useful effect).

In both of the above cases, Windows users should _not_ use the **source** command, but should rather run the **activate** script directly from the command shell like so:

```
<DIR>\Scripts\activate
```

Managing multiple virtual environments directly can become tedious, so the [dependency management tutorial](https://packaging.python.org/en/latest/tutorials/managing-dependencies/#managing-dependencies) introduces a higher level tool, [Pipenv](https://packaging.python.org/en/latest/key_projects/#pipenv), that automatically manages a separate virtual environment for each project and application that you work on.

## Use pip for Installing
[pip](https://packaging.python.org/en/latest/key_projects/#pip) is the recommended installer. Below, we’ll cover the most common usage scenarios. For more detail, see the [pip docs](https://pip.pypa.io/en/latest/ "(in pip v25.1)"), which includes a complete [Reference Guide](https://pip.pypa.io/en/latest/cli/ "(in pip v25.1)").

## Installing from PyPI
The most common usage of [pip](https://packaging.python.org/en/latest/key_projects/#pip) is to install from the [Python Package Index](https://packaging.python.org/en/latest/glossary/#term-Python-Package-Index-PyPI) using a [requirement specifier](https://packaging.python.org/en/latest/glossary/#term-Requirement-Specifier). Generally speaking, a requirement specifier is composed of a project name followed by an optional [version specifier](https://packaging.python.org/en/latest/glossary/#term-Version-Specifier). A full description of the supported specifiers can be found in the [Version specifier specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers). Below are some examples.
>  `pip` 使用需求规范来确定需要安装的包，一般来说，一个需求规范由一个项目名和一个可选的版本规范组成

To install the latest version of “SomeProject”:

```
python3 -m pip install "SomeProject"
```

To install a specific version:

```
python3 -m pip install "SomeProject==1.4"
```

To install greater than or equal to one version and less than another:

```
python3 -m pip install "SomeProject>=1,<2"
```

To install a version that’s [compatible](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers-compatible-release) with a certain version: [4](https://packaging.python.org/en/latest/tutorials/installing-packages/#id10)

```
python3 -m pip install "SomeProject~=1.4.2"
```

In this case, this means to install any version “\==1.4.\*” version that’s also “>=1.4.2”.

## Source Distributions vs Wheels
[pip](https://packaging.python.org/en/latest/key_projects/#pip) can install from either [Source Distributions (sdist)](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist) or [Wheels](https://packaging.python.org/en/latest/glossary/#term-Wheel), but if both are present on PyPI, pip will prefer a compatible [wheel](https://packaging.python.org/en/latest/glossary/#term-Wheel). You can override pip's default behavior by e.g. using its [–no-binary](https://pip.pypa.io/en/latest/cli/pip_install/#install-no-binary "(in pip v25.1)") option.

[Wheels](https://packaging.python.org/en/latest/glossary/#term-Wheel) are a pre-built [distribution](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) format that provides faster installation compared to [Source Distributions (sdist)](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist), especially when a project contains compiled extensions.

If [pip](https://packaging.python.org/en/latest/key_projects/#pip) does not find a wheel to install, it will locally build a wheel and cache it for future installs, instead of rebuilding the source distribution in the future.
>  如果 `pip` 没有找到要安装的 wheel，它会本地构建一个 wheel，并将其缓存，以供将来安装

## Upgrading packages
Upgrade an already installed `SomeProject` to the latest from PyPI.

```
python3 -m pip install --upgrade SomeProject
```

## Installing to the User Site
To install [packages](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) that are isolated to the current user, use the `--user` flag:

```
python3 -m pip install --user SomeProject
```

For more information see the [User Installs](https://pip.pypa.io/en/latest/user_guide/#user-installs) section from the pip docs.

Note that the `--user` flag has no effect when inside a virtual environment - all installation commands will affect the virtual environment.

If `SomeProject` defines any command-line scripts or console entry points, `--user` will cause them to be installed inside the [user base](https://docs.python.org/3/library/site.html#site.USER_BASE)’s binary directory, which may or may not already be present in your shell’s `PATH`. (Starting in version 10, pip displays a warning when installing any scripts to a directory outside `PATH`.) If the scripts are not available in your shell after installation, you’ll need to add the directory to your `PATH`:

- On Linux and macOS you can find the user base binary directory by running `python -m site --user-base` and adding `bin` to the end. For example, this will typically print `~/.local` (with `~` expanded to the absolute path to your home directory) so you’ll need to add `~/.local/bin` to your `PATH`. You can set your `PATH` permanently by [modifying ~/.profile](https://stackoverflow.com/a/14638025).
- On Windows you can find the user base binary directory by running `py -m site --user-site` and replacing `site-packages` with `Scripts`. For example, this could return `C:\Users\Username\AppData\Roaming\Python36\site-packages` so you would need to set your `PATH` to include `C:\Users\Username\AppData\Roaming\Python36\Scripts`. You can set your user `PATH` permanently in the [Control Panel](https://docs.microsoft.com/en-us/windows/win32/shell/user-environment-variables?redirectedfrom=MSDN). You may need to log out for the `PATH` changes to take effect.

## Requirements files
Install a list of requirements specified in a [Requirements File](https://pip.pypa.io/en/latest/user_guide/#requirements-files "(in pip v25.1)").

```
python3 -m pip install -r requirements.txt
```

## Installing from VCS
Install a project from VCS in “editable” mode. For a full breakdown of the syntax, see pip’s section on [VCS Support](https://pip.pypa.io/en/latest/topics/vcs-support/#vcs-support "(in pip v25.1)").

```
python3 -m pip install -e SomeProject @ git+https://git.repo/some_pkg.git          # from git
python3 -m pip install -e SomeProject @ hg+https://hg.repo/some_pkg                # from mercurial
python3 -m pip install -e SomeProject @ svn+svn://svn.repo/some_pkg/trunk/         # from svn
python3 -m pip install -e SomeProject @ git+https://git.repo/some_pkg.git@feature  # from a branch
```

## Installing from other Indexes
Install from an alternate index

```
python3 -m pip install --index-url http://my.package.repo/simple/ SomeProject
```

Search an additional index during install, in addition to [PyPI](https://packaging.python.org/en/latest/glossary/#term-Python-Package-Index-PyPI)

```
python3 -m pip install --extra-index-url http://my.package.repo/simple SomeProject
```

## Installing from a local src tree
Installing from local src in [Development Mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html "(in setuptools v75.8.1.post20250225)"), i.e. in such a way that the project appears to be installed, but yet is still editable from the src tree.

```
python3 -m pip install -e <path>
```

You can also install normally from src

```
python3 -m pip install <path>
```

## Installing from local archives
Install a particular source archive file.

```
python3 -m pip install ./downloads/SomeProject-1.0.4.tar.gz
```

Install from a local directory containing archives (and don’t check [PyPI](https://packaging.python.org/en/latest/glossary/#term-Python-Package-Index-PyPI))

```
python3 -m pip install --no-index --find-links=file:///local/dir/ SomeProject
python3 -m pip install --no-index --find-links=/local/dir/ SomeProject
python3 -m pip install --no-index --find-links=relative/dir/ SomeProject
```

## Installing from other sources
To install from other data sources (for example Amazon S3 storage) you can create a helper application that presents the data in a format compliant with the [simple repository API](https://packaging.python.org/en/latest/specifications/simple-repository-api/#simple-repository-api):, and use the `--extra-index-url` flag to direct pip to use that index.

```
./s3helper --port=7777
python -m pip install --extra-index-url http://localhost:7777 SomeProject
```

## Installing Prereleases
Find pre-release and development versions, in addition to stable versions. By default, pip only finds stable versions.

```
python3 -m pip install --pre SomeProject
```

## Installing “Extras”
Extras are optional “variants” of a package, which may include additional dependencies, and thereby enable additional functionality from the package. If you wish to install an extra for a package which you know publishes one, you can include it in the pip installation command:
>  extras 是软件包的可选的 “变体”，它们可能包含额外的依赖项，从而启用额外的功能

```
python3 -m pip install 'SomePackage[PDF]'
python3 -m pip install 'SomePackage[PDF]==3.0'
python3 -m pip install -e '.[PDF]'  # editable project in current directory
```


---

[1](https://packaging.python.org/en/latest/tutorials/installing-packages/#id2) “Secure” in this context means using a modern browser or a tool like **curl** that verifies SSL certificates when downloading from https URLs.
[2](https://packaging.python.org/en/latest/tutorials/installing-packages/#id3) Depending on your platform, this may require root or Administrator access. [pip](https://packaging.python.org/en/latest/key_projects/#pip) is currently considering changing this by [making user installs the default behavior](https://github.com/pypa/pip/issues/1668).
[3](https://packaging.python.org/en/latest/tutorials/installing-packages/#id4) Beginning with Python 3.4, `venv` (a stdlib alternative to [virtualenv](https://packaging.python.org/en/latest/key_projects/#virtualenv)) will create virtualenv environments with `pip` pre-installed, thereby making it an equal alternative to [virtualenv](https://packaging.python.org/en/latest/key_projects/#virtualenv).
[4](https://packaging.python.org/en/latest/tutorials/installing-packages/#id5) The compatible release specifier was accepted in [**PEP 440**](https://peps.python.org/pep-0440/) and support was released in [Setuptools](https://packaging.python.org/en/latest/key_projects/#setuptools) v8.0 and [pip](https://packaging.python.org/en/latest/key_projects/#pip) v6.0

# Packaging Python Projects
This tutorial walks you through how to package a simple Python project. It will show you how to add the necessary files and structure to create the package, how to build the package, and how to upload it to the Python Package Index (PyPI).

Tip
If you have trouble running the commands in this tutorial, please copy the command and its output, then [open an issue](https://github.com/pypa/packaging-problems/issues/new?template=packaging_tutorial.yml&title=Trouble+with+the+packaging+tutorial&guide=https://packaging.python.org/tutorials/packaging-projects) on the [packaging-problems](https://github.com/pypa/packaging-problems) repository on GitHub. We’ll do our best to help you!

Some of the commands require a newer version of [pip](https://packaging.python.org/en/latest/key_projects/#pip), so start by making sure you have the latest version installed:

```
python3 -m pip install --upgrade pip
```

## A simple project
This tutorial uses a simple project named `example_package_YOUR_USERNAME_HERE`. If your username is `me`, then the package would be `example_package_me`; this ensures that you have a unique package name that doesn’t conflict with packages uploaded by other people following this tutorial. We recommend following this tutorial as-is using this project, before packaging your own project.

Create the following file structure locally:

```
packaging_tutorial/
└── src/
    └── example_package_YOUR_USERNAME_HERE/
        ├── __init__.py
        └── example.py
```

The directory containing the Python files should match the project name. This simplifies the configuration and is more obvious to users who install the package.

Creating the file `__init__.py` is recommended because the existence of an `__init__.py` file allows users to import the directory as a regular package, even if (as is the case in this tutorial) `__init__.py` is empty. [1](https://packaging.python.org/en/latest/tutorials/packaging-projects/#namespace-packages)

`example.py` is an example of a module within the package that could contain the logic (functions, classes, constants, etc.) of your package. Open that file and enter the following content:

```python
def add_one(number):
    return number + 1
```

If you are unfamiliar with Python’s [modules](https://packaging.python.org/en/latest/glossary/#term-Module) and [import packages](https://packaging.python.org/en/latest/glossary/#term-Import-Package), take a few minutes to read over the [Python documentation for packages and modules](https://docs.python.org/3/tutorial/modules.html#packages).

Once you create this structure, you’ll want to run all of the commands in this tutorial within the `packaging_tutorial` directory.

## Creating the package files
You will now add files that are used to prepare the project for distribution. When you’re done, the project structure will look like this:

```
packaging_tutorial/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   └── example_package_YOUR_USERNAME_HERE/
│       ├── __init__.py
│       └── example.py
└── tests/
```

## Creating a test directory
`tests/` is a placeholder for test files. Leave it empty for now.

## Choosing a build backend
Tools like [pip](https://packaging.python.org/en/latest/key_projects/#pip) and [build](https://packaging.python.org/en/latest/key_projects/#build) do not actually convert your sources into a [distribution package](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) (like a wheel); that job is performed by a [build backend](https://packaging.python.org/en/latest/glossary/#term-Build-Backend). The build backend determines how your project will specify its configuration, including metadata (information about the project, for example, the name and tags that are displayed on PyPI) and input files. Build backends have different levels of functionality, such as whether they support building [extension modules](https://packaging.python.org/en/latest/glossary/#term-Extension-Module), and you should choose one that suits your needs and preferences.

You can choose from a number of backends; this tutorial uses [Hatchling](https://packaging.python.org/en/latest/key_projects/#hatch) by default, but it will work identically with [Setuptools](https://packaging.python.org/en/latest/key_projects/#setuptools), [Flit](https://packaging.python.org/en/latest/key_projects/#flit), [PDM](https://packaging.python.org/en/latest/key_projects/#pdm), and others that support the `[project]` table for [metadata](https://packaging.python.org/en/latest/tutorials/packaging-projects/#configuring-metadata).

>  将源码转化为发布包的工作 (例如 wheel) 是由构建后端完成的，我们使用的构建后端决定了我们的项目应该如何指定其配置，包括了元数据和输入文件

Note
Some build backends are part of larger tools that provide a command-line interface with additional features like project initialization and version management, as well as building, uploading, and installing packages. This tutorial uses single-purpose tools that work independently.

The `pyproject.toml` tells [build frontend](https://packaging.python.org/en/latest/glossary/#term-Build-Frontend) tools like [pip](https://packaging.python.org/en/latest/key_projects/#pip) and [build](https://packaging.python.org/en/latest/key_projects/#build) which backend to use for your project. Below are some examples for common build backends, but check your backend’s own documentation for more details.
>  `pyproject.toml` 会被构建前端例如 `pip, bulid` 用于确定使用哪个构建后端

Hatchling
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

The `requires` key is a list of packages that are needed to build your package. The [frontend](https://packaging.python.org/en/latest/glossary/#term-Build-Frontend) should install them automatically when building your package. Frontends usually run builds in isolated environments, so omitting dependencies here may cause build-time errors. This should always include your backend’s package, and might have other build-time dependencies.
>  `requries` 指定了需要构建我们的发布包所需的包，构建前端会 (在构建时) 自动安装它们，前端一般在独立的环境中执行构建，因此 `requries` 不能忽略，`requries` 应该总是包括了我们的构建后端包，以及可选的其他构建时依赖

The `build-backend` key is the name of the Python object that frontends will use to perform the build.
>  `build-backend` 指定了构建前端会用来执行构建的 Python 对象

Both of these values will be provided by the documentation for your build backend, or generated by its command line interface. There should be no need for you to customize these settings.

Additional configuration of the build tool will either be in a `tool` section of the `pyproject.toml`, or in a special file defined by the build tool. For example, when using `setuptools` as your build backend, additional configuration may be added to a `setup.py` or `setup.cfg` file, and specifying `setuptools.build_meta` in your build allows the tools to locate and use these automatically.

### Configuring metadata
Open `pyproject.toml` and enter the following content. Change the `name` to include your username; this ensures that you have a unique package name that doesn’t conflict with packages uploaded by other people following this tutorial.

hatchling/pdm
```toml
[project]
name = "example_package_YOUR_USERNAME_HERE"
version = "0.0.1"
authors = [
  { name="Example Author", email="author@example.com" },
]
description = "A small example package"
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: MIT License",
]

[project.urls]
Homepage = "https://github.com/pypa/sampleproject"
Issues = "https://github.com/pypa/sampleproject/issues"
```

- `name` is the _distribution name_ of your package. This can be any name as long as it only contains letters, numbers, `.`, `_` , and `-`. It also must not already be taken on PyPI. **Be sure to update this with your username** for this tutorial, as this ensures you won’t try to upload a package with the same name as one which already exists.

- `version` is the package version. (Some build backends allow it to be specified another way, such as from a file or Git tag.)
- `authors` is used to identify the author of the package; you specify a name and an email for each author. You can also list `maintainers` in the same format.
- `description` is a short, one-sentence summary of the package.
- `readme` is a path to a file containing a detailed description of the package. This is shown on the package detail page on PyPI. In this case, the description is loaded from `README.md` (which is a common pattern). There also is a more advanced table form described in the [pyproject.toml guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#writing-pyproject-toml).
- `requires-python` gives the versions of Python supported by your project. An installer like [pip](https://packaging.python.org/en/latest/key_projects/#pip) will look back through older versions of packages until it finds one that has a matching Python version.
- `classifiers` gives the index and [pip](https://packaging.python.org/en/latest/key_projects/#pip) some additional metadata about your package. In this case, the package is only compatible with Python 3 and is OS-independent. You should always include at least which version(s) of Python your package works on and which operating systems your package will work on. For a complete list of classifiers, see [https://pypi.org/classifiers/](https://pypi.org/classifiers/).
>  `classifiers` 用于给 `pip` 和 PyPI 关于包的额外元信息，我们应该至少提供我们的包兼容哪个 Python 版本以及哪个 OS

- `license` is the [SPDX license expression](https://packaging.python.org/en/latest/glossary/#term-License-Expression) of your package. Not supported by all the build backends yet.
- `license-files` is the list of glob paths to the license files, relative to the directory where `pyproject.toml` is located. Not supported by all the build backends yet.
- `urls` lets you list any number of extra links to show on PyPI. Generally this could be to the source, documentation, issue trackers, etc.

See the [pyproject.toml guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#writing-pyproject-toml) for details on these and other fields that can be defined in the `[project]` table. Other common fields are `keywords` to improve discoverability and the `dependencies` that are required to install your package.

## Creating README.md
Open `README.md` and enter the following content. You can customize this if you’d like.

```
# Example Package

This is a simple example package. You can use
[GitHub-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.
```

## Creating a LICENSE
It’s important for every package uploaded to the Python Package Index to include a license. This tells users who install your package the terms under which they can use your package. For help picking a license, see [https://choosealicense.com/](https://choosealicense.com/). Once you have chosen a license, open `LICENSE` and enter the license text. For example, if you had chosen the MIT license:
>  上传到 PyPI 的包都需要包括一个 LICENSE

```
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Most build backends automatically include license files in packages. See your backend’s documentation for more details. If you include the path to license in the `license-files` key of `pyproject.toml`, and your build backend supports [**PEP 639**](https://peps.python.org/pep-0639/), the file will be automatically included in the package.

## Including other files
The files listed above will be included automatically in your [source distribution](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist). If you want to include additional files, see the documentation for your build backend.

## Generating distribution archives
The next step is to generate [distribution packages](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) for the package. These are archives that are uploaded to the Python Package Index and can be installed by [pip](https://packaging.python.org/en/latest/key_projects/#pip).

Make sure you have the latest version of PyPA’s [build](https://packaging.python.org/en/latest/key_projects/#build) installed:

```
python3 -m pip install --upgrade build
```

Tip
If you have trouble installing these, see the [Installing Packages](https://packaging.python.org/en/latest/tutorials/installing-packages/) tutorial.

Now run this command from the same directory where `pyproject.toml` is located:

```
python3 -m build
```

This command should output a lot of text and once completed should generate two files in the `dist` directory:

```
dist/
├── example_package_YOUR_USERNAME_HERE-0.0.1-py3-none-any.whl
└── example_package_YOUR_USERNAME_HERE-0.0.1.tar.gz
```

The `tar.gz` file is a [source distribution](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist) whereas the `.whl` file is a [built distribution](https://packaging.python.org/en/latest/glossary/#term-Built-Distribution). Newer [pip](https://packaging.python.org/en/latest/key_projects/#pip) versions preferentially install built distributions, but will fall back to source distributions if needed. You should always upload a source distribution and provide built distributions for the platforms your project is compatible with. In this case, our example package is compatible with Python on any platform so only one built distribution is needed.

## Uploading the distribution archives
Finally, it’s time to upload your package to the Python Package Index!

The first thing you’ll need to do is register an account on TestPyPI, which is a separate instance of the package index intended for testing and experimentation. It’s great for things like this tutorial where we don’t necessarily want to upload to the real index. To register an account, go to [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/) and complete the steps on that page. You will also need to verify your email address before you’re able to upload any packages. For more details, see [Using TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/).

To securely upload your project, you’ll need a PyPI [API token](https://test.pypi.org/help/#apitoken). Create one at [https://test.pypi.org/manage/account/#api-tokens](https://test.pypi.org/manage/account/#api-tokens), setting the “Scope” to “Entire account”. **Don’t close the page until you have copied and saved the token — you won’t see that token again.**

Now that you are registered, you can use [twine](https://packaging.python.org/en/latest/key_projects/#twine) to upload the distribution packages. You’ll need to install Twine:

```
python3 -m pip install --upgrade twine
```

Once installed, run Twine to upload all of the archives under `dist`:

```
python3 -m twine upload --repository testpypi dist/*
```

You will be prompted for an API token. Use the token value, including the `pypi-` prefix. Note that the input will be hidden, so be sure to paste correctly.

After the command completes, you should see output similar to this:

```
Uploading distributions to https://test.pypi.org/legacy/
Enter your API token:
Uploading example_package_YOUR_USERNAME_HERE-0.0.1-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.2/8.2 kB • 00:01 • ?
Uploading example_package_YOUR_USERNAME_HERE-0.0.1.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.8/6.8 kB • 00:00 • ?
```

Once uploaded, your package should be viewable on TestPyPI; for example: `https://test.pypi.org/project/example_package_YOUR_USERNAME_HERE`.

## Installing your newly uploaded package
You can use [pip](https://packaging.python.org/en/latest/key_projects/#pip) to install your package and verify that it works. Create a [virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments) and install your package from TestPyPI:

```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps example-package-YOUR-USERNAME-HERE
```

Make sure to specify your username in the package name!

pip should install the package from TestPyPI and the output should look something like this:

```
Collecting example-package-YOUR-USERNAME-HERE
  Downloading https://test-files.pythonhosted.org/packages/.../example_package_YOUR_USERNAME_HERE_0.0.1-py3-none-any.whl
Installing collected packages: example_package_YOUR_USERNAME_HERE
Successfully installed example_package_YOUR_USERNAME_HERE-0.0.1
```

Note
This example uses `--index-url` flag to specify TestPyPI instead of live PyPI. Additionally, it specifies `--no-deps`. Since TestPyPI doesn’t have the same packages as the live PyPI, it’s possible that attempting to install dependencies may fail or install something unexpected. While our example package doesn’t have any dependencies, it’s a good practice to avoid installing dependencies when using TestPyPI.

You can test that it was installed correctly by importing the package. Make sure you’re still in your virtual environment, then run Python:

```
python3
```

and import the package:

```
>>> from example_package_YOUR_USERNAME_HERE import example
>>> example.add_one(2)
3
```

## Next steps
**Congratulations, you’ve packaged and distributed a Python project!** ✨ 🍰 ✨

Keep in mind that this tutorial showed you how to upload your package to Test PyPI, which isn’t a permanent storage. The Test system occasionally deletes packages and accounts. It is best to use TestPyPI for testing and experiments like this tutorial.

When you are ready to upload a real package to the Python Package Index you can do much the same as you did in this tutorial, but with these important differences:

- Choose a memorable and unique name for your package. You don’t have to append your username as you did in the tutorial, but you can’t use an existing name.
- Register an account on [https://pypi.org](https://pypi.org/) - note that these are two separate servers and the login details from the test server are not shared with the main server.
- Use `twine upload dist/*` to upload your package and enter your credentials for the account you registered on the real PyPI. Now that you’re uploading the package in production, you don’t need to specify `--repository`; the package will upload to [https://pypi.org/](https://pypi.org/) by default.
- Install your package from the real PyPI using `python3 -m pip install [your-package]`.

At this point if you want to read more on packaging Python libraries here are some things you can do:

- Read about advanced configuration for your chosen build backend: [Hatchling](https://hatch.pypa.io/latest/config/metadata/), [setuptools](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html "(in setuptools v75.8.1.post20250225)"), [Flit](https://flit.pypa.io/en/stable/pyproject_toml.html "(in Flit v3.11.0)"), [PDM](https://pdm-project.org/latest/reference/pep621/).
- Look at the [guides](https://packaging.python.org/en/latest/guides/) on this site for more advanced practical information, or the [discussions](https://packaging.python.org/en/latest/discussions/) for explanations and background on specific topics.
- Consider packaging tools that provide a single command-line interface for project management and packaging, such as [hatch](https://packaging.python.org/en/latest/key_projects/#hatch), [flit](https://packaging.python.org/en/latest/key_projects/#flit), [pdm](https://packaging.python.org/en/latest/key_projects/#pdm), and [poetry](https://packaging.python.org/en/latest/key_projects/#poetry).

---

Notes
[1](https://packaging.python.org/en/latest/tutorials/packaging-projects/#id1) Technically, you can also create Python packages without an `__init__.py` file, but those are called [namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/) and considered an **advanced topic** (not covered in this tutorial). If you are only getting started with Python packaging, it is recommended to stick with _regular packages_ and `__init__.py` (even if the file is empty).
