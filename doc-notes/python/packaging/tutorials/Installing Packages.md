>  Last updated on Feb 26, 2025

This section covers the basics of how to install Python [packages](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package).

It’s important to note that the term “package” in this context is being used to describe a bundle of software to be installed (i.e. as a synonym for a [distribution](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package)). It does not refer to the kind of [package](https://packaging.python.org/en/latest/glossary/#term-Import-Package) that you import in your Python source code (i.e. a container of modules). It is common in the Python community to refer to a [distribution](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) using the term “package”. Using the term “distribution” is often not preferred, because it can easily be confused with a Linux distribution, or another larger software distribution like Python itself.
>  注意，在此上下文中，“包” 用于描述需要安装的软件集合，即作为 “发布” 的同义词，而并不特指一个 Python package
>  Python 社区常用 “包” 替代 “发布”，以避免和 Linux 发行或更大的软件发行例如 Python 本身混淆

# Requirements for Installing Packages
This section describes the steps to follow before installing other Python packages.

## Ensure you can run Python from the command line
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

## Ensure you can run pip from the command line
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

## Ensure pip, setuptools, and wheel are up to date
While `pip` alone is sufficient to install from pre-built binary archives, up to date copies of the `setuptools` and `wheel` projects are useful to ensure you can also install from source archives:

```
python3 -m pip install --upgrade pip setuptools wheel
```

## Optionally, create a virtual environment
See [section below](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments) for details, but here’s the basic [venv](https://docs.python.org/3/library/venv.html "(in Python v3.13)") [3](https://packaging.python.org/en/latest/tutorials/installing-packages/#id9) command to use on a typical Linux system:

```
python3 -m venv tutorial_env
source tutorial_env/bin/activate
```

This will create a new virtual environment in the `tutorial_env` subdirectory, and configure the current shell to use it as the default `python` environment.

# Creating Virtual Environments
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

# Use pip for Installing
[pip](https://packaging.python.org/en/latest/key_projects/#pip) is the recommended installer. Below, we’ll cover the most common usage scenarios. For more detail, see the [pip docs](https://pip.pypa.io/en/latest/ "(in pip v25.1)"), which includes a complete [Reference Guide](https://pip.pypa.io/en/latest/cli/ "(in pip v25.1)").

# Installing from PyPI
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

# Source Distributions vs Wheels
[pip](https://packaging.python.org/en/latest/key_projects/#pip) can install from either [Source Distributions (sdist)](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist) or [Wheels](https://packaging.python.org/en/latest/glossary/#term-Wheel), but if both are present on PyPI, pip will prefer a compatible [wheel](https://packaging.python.org/en/latest/glossary/#term-Wheel). You can override pip's default behavior by e.g. using its [–no-binary](https://pip.pypa.io/en/latest/cli/pip_install/#install-no-binary "(in pip v25.1)") option.

[Wheels](https://packaging.python.org/en/latest/glossary/#term-Wheel) are a pre-built [distribution](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) format that provides faster installation compared to [Source Distributions (sdist)](https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist), especially when a project contains compiled extensions.

If [pip](https://packaging.python.org/en/latest/key_projects/#pip) does not find a wheel to install, it will locally build a wheel and cache it for future installs, instead of rebuilding the source distribution in the future.
>  如果 `pip` 没有找到要安装的 wheel，它会本地构建一个 wheel，并将其缓存，以供将来安装

# Upgrading packages
Upgrade an already installed `SomeProject` to the latest from PyPI.

```
python3 -m pip install --upgrade SomeProject
```

# Installing to the User Site
To install [packages](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) that are isolated to the current user, use the `--user` flag:

```
python3 -m pip install --user SomeProject
```

For more information see the [User Installs](https://pip.pypa.io/en/latest/user_guide/#user-installs) section from the pip docs.

Note that the `--user` flag has no effect when inside a virtual environment - all installation commands will affect the virtual environment.

If `SomeProject` defines any command-line scripts or console entry points, `--user` will cause them to be installed inside the [user base](https://docs.python.org/3/library/site.html#site.USER_BASE)’s binary directory, which may or may not already be present in your shell’s `PATH`. (Starting in version 10, pip displays a warning when installing any scripts to a directory outside `PATH`.) If the scripts are not available in your shell after installation, you’ll need to add the directory to your `PATH`:

- On Linux and macOS you can find the user base binary directory by running `python -m site --user-base` and adding `bin` to the end. For example, this will typically print `~/.local` (with `~` expanded to the absolute path to your home directory) so you’ll need to add `~/.local/bin` to your `PATH`. You can set your `PATH` permanently by [modifying ~/.profile](https://stackoverflow.com/a/14638025).
- On Windows you can find the user base binary directory by running `py -m site --user-site` and replacing `site-packages` with `Scripts`. For example, this could return `C:\Users\Username\AppData\Roaming\Python36\site-packages` so you would need to set your `PATH` to include `C:\Users\Username\AppData\Roaming\Python36\Scripts`. You can set your user `PATH` permanently in the [Control Panel](https://docs.microsoft.com/en-us/windows/win32/shell/user-environment-variables?redirectedfrom=MSDN). You may need to log out for the `PATH` changes to take effect.

# Requirements files
Install a list of requirements specified in a [Requirements File](https://pip.pypa.io/en/latest/user_guide/#requirements-files "(in pip v25.1)").

```
python3 -m pip install -r requirements.txt
```

# Installing from VCS
Install a project from VCS in “editable” mode. For a full breakdown of the syntax, see pip’s section on [VCS Support](https://pip.pypa.io/en/latest/topics/vcs-support/#vcs-support "(in pip v25.1)").

```
python3 -m pip install -e SomeProject @ git+https://git.repo/some_pkg.git          # from git
python3 -m pip install -e SomeProject @ hg+https://hg.repo/some_pkg                # from mercurial
python3 -m pip install -e SomeProject @ svn+svn://svn.repo/some_pkg/trunk/         # from svn
python3 -m pip install -e SomeProject @ git+https://git.repo/some_pkg.git@feature  # from a branch
```

# Installing from other Indexes
Install from an alternate index

```
python3 -m pip install --index-url http://my.package.repo/simple/ SomeProject
```

Search an additional index during install, in addition to [PyPI](https://packaging.python.org/en/latest/glossary/#term-Python-Package-Index-PyPI)

```
python3 -m pip install --extra-index-url http://my.package.repo/simple SomeProject
```

# Installing from a local src tree
Installing from local src in [Development Mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html "(in setuptools v75.8.1.post20250225)"), i.e. in such a way that the project appears to be installed, but yet is still editable from the src tree.

```
python3 -m pip install -e <path>
```

You can also install normally from src

```
python3 -m pip install <path>
```

# Installing from local archives
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

# Installing from other sources
To install from other data sources (for example Amazon S3 storage) you can create a helper application that presents the data in a format compliant with the [simple repository API](https://packaging.python.org/en/latest/specifications/simple-repository-api/#simple-repository-api):, and use the `--extra-index-url` flag to direct pip to use that index.

```
./s3helper --port=7777
python -m pip install --extra-index-url http://localhost:7777 SomeProject
```

# Installing Prereleases
Find pre-release and development versions, in addition to stable versions. By default, pip only finds stable versions.

```
python3 -m pip install --pre SomeProject
```

# Installing “Extras”
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
