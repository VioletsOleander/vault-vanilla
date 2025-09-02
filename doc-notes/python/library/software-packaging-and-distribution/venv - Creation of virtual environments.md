---
completed:
---
# `venv` — Creation of virtual environments
Added in version 3.3.

**Source code:** [Lib/venv/](https://github.com/python/cpython/tree/3.13/Lib/venv/)

---

The `venv` module supports creating lightweight “virtual environments”, each with their own independent set of Python packages installed in their [`site`](https://docs.python.org/3/library/site.html#module-site "site: Module responsible for site-specific configuration.") directories. A virtual environment is created on top of an existing Python installation, known as the virtual environment’s “base” Python, and by default is isolated from the packages in the base environment, so that only those explicitly installed in the virtual environment are available.
>  `venv` 模块支持创建轻量级的虚拟环境，虚拟环境有自己独立的一组 Python 包，安装在它的 `site` 目录
>  虚拟环境基于已有的 Python 安装上创建 (虚拟环境的 base Python)，默认与 base 环境中的包隔离

When used from within a virtual environment, common installation tools such as [pip](https://pypi.org/project/pip/) will install Python packages into a virtual environment without needing to be told to do so explicitly.
>  在虚拟环境中，`pip` 自动会将包安装到虚拟环境中

A virtual environment is (amongst other things):

- Used to contain a specific Python interpreter and software libraries and binaries which are needed to support a project (library or application). These are by default isolated from software in other virtual environments and Python interpreters and libraries installed in the operating system.
- Contained in a directory, conventionally named `.venv` or `venv` in the project directory, or under a container directory for lots of virtual environments, such as `~/.virtualenvs`.
- Not checked into source control systems such as Git.
- Considered as disposable – it should be simple to delete and recreate it from scratch. You don’t place any project code in the environment.
- Not considered as movable or copyable – you just recreate the same environment in the target location.

>  虚拟环境:
>  - 用于包含特定的 Python 解释器和软件库、二进制文件，用于支持一个项目，他们默认和其他虚拟环境中，以及系统中安装的 Python 解释器、库是隔离的
>  - 包含一个目录，通常是项目目录下名为 `.venv or venv`，或者在一个存储了多个虚拟环境的一个容器目录中，例如 `~/vritualenvs`
>  - 不会被纳入版本控制系统中
>  - 被视为可丢弃的，应该可以轻松从头删除并重新创建，你不会在环境中放置项目代码
>  - 不被视为可移动或可复制的，只需要在目标地址重新创建相同的环境即可

See [**PEP 405**](https://peps.python.org/pep-0405/) for more background on Python virtual environments.

See also: [Python Packaging User Guide: Creating and using virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments)

[Availability](https://docs.python.org/3/library/intro.html#availability): not Android, not iOS, not WASI.

This module is not supported on [mobile platforms](https://docs.python.org/3/library/intro.html#mobile-availability) or [WebAssembly platforms](https://docs.python.org/3/library/intro.html#wasm-availability).

## Creating virtual environments
[Virtual environments](https://docs.python.org/3/library/venv.html#venv-def) are created by executing the `venv` module:

```
python -m venv /path/to/new/virtual/environment
```

This creates the target directory (including parent directories as needed) and places a `pyvenv.cfg` file in it with a `home` key pointing to the Python installation from which the command was run. It also creates a `bin` (or `Scripts` on Windows) subdirectory containing a copy or symlink of the Python executable (as appropriate for the platform or arguments used at environment creation time). It also creates a `lib/pythonX.Y/site-packages` subdirectory (on Windows, this is `Libsite-packages`). If an existing directory is specified, it will be re-used.
>  `python -m venv xxx` 创建目标目录，并创建 `pyvenv.cfg` 文件，该文件包含指向运行命令时的 Python 安装路径的 `home` key
>  它还创建了 `bin` 目录，包含 Python 可执行文件的副本或符号链接
>  它还创建 `lib/pythonX.Y/site-packages` 子目录

>  Changed in version 3.5: The use of `venv` is now recommended for creating virtual environments.

>  Deprecated since version 3.6, removed in version 3.8: **pyvenv** was the recommended tool for creating virtual environments for Python 3.3 and 3.4, and replaced in 3.5 by executing `venv` directly.

On Windows, invoke the `venv` command as follows:

```
python -m venv C:\path\to\new\virtual\environment
```

The command, if run with `-h`, will show the available options:

```
usage: venv [-h] [--system-site-packages] [--symlinks | --copies] [--clear] [--upgrade] [--without-pip] [--prompt PROMPT] [--upgrade-deps] [--without-scm-ignore-files] ENV_DIR [ENV_DIR ...]

Creates virtual Python environments in one or more target directories.

Once an environment has been created, you may wish to activate it, e.g. by sourcing an activate script in its bin directory.
```

```
ENV_DIR
A required argument specifying the directory to create the environment in.

--system-site-packages
Give the virtual environment access to the system site-packages directory.

--symlinks
Try to use symlinks rather than copies, when symlinks are not the default for the platform.

--copies
Try to use copies rather than symlinks, even when symlinks are the default for the platform.

--clear
Delete the contents of the environment directory if it already exists, before environment creation.

--upgrade
Upgrade the environment directory to use this version of Python, assuming Python has been upgraded in-place.

--without-pip
Skips installing or upgrading pip in the virtual environment (pip is bootstrapped by default).

--prompt <PROMPT>
Provides an alternative prompt prefix for this environment.

--upgrade-deps
Upgrade core dependencies (pip) to the latest version in PyPI.

--without-scm-ignore-files
Skips adding SCM ignore files to the environment directory (Git is supported by default).
```

>  Changed in version 3.4: Installs pip by default, added the `--without-pip` and `--copies` options.

>  Changed in version 3.4: In earlier versions, if the target directory already existed, an error was raised, unless the `--clear` or `--upgrade` option was provided.

>  Changed in version 3.9: Add `--upgrade-deps` option to upgrade pip + setuptools to the latest on PyPI.

>  Changed in version 3.12: `setuptools` is no longer a core venv dependency.

>  Changed in version 3.13: Added the `--without-scm-ignore-files` option.

>  Changed in version 3.13: `venv` now creates a `.gitignore` file for Git by default.

Note
While symlinks are supported on Windows, they are not recommended. Of particular note is that double-clicking `python.exe` in File Explorer will resolve the symlink eagerly and ignore the virtual environment.

Note
On Microsoft Windows, it may be required to enable the `Activate.ps1` script by setting the execution policy for the user. You can do this by issuing the following PowerShell command:

```
PS C:\> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

See [About Execution Policies](https://go.microsoft.com/fwlink/?LinkID=135170) for more information.

The created `pyvenv.cfg` file also includes the `include-system-site-packages` key, set to `true` if `venv` is run with the `--system-site-packages` option, `false` otherwise.

Unless the `--without-pip` option is given, [`ensurepip`](https://docs.python.org/3/library/ensurepip.html#module-ensurepip "ensurepip: Bootstrapping the \"pip\" installer into an existing Python installation or virtual environment.") will be invoked to bootstrap `pip` into the virtual environment.

Multiple paths can be given to `venv`, in which case an identical virtual environment will be created, according to the given options, at each provided path.

## How venvs work
When a Python interpreter is running from a virtual environment, [`sys.prefix`](https://docs.python.org/3/library/sys.html#sys.prefix "sys.prefix") and [`sys.exec_prefix`](https://docs.python.org/3/library/sys.html#sys.exec_prefix "sys.exec_prefix") point to the directories of the virtual environment, whereas [`sys.base_prefix`](https://docs.python.org/3/library/sys.html#sys.base_prefix "sys.base_prefix") and [`sys.base_exec_prefix`](https://docs.python.org/3/library/sys.html#sys.base_exec_prefix "sys.base_exec_prefix") point to those of the base Python used to create the environment. It is sufficient to check `sys.prefix != sys.base_prefix` to determine if the current interpreter is running from a virtual environment.
>  在虚拟环境运行 Python 解释器时，`sys.prefix, sys.exec_prefix` 指向虚拟环境的目录
>  `sys.base_prefix, sys.base_exec_prefix` 指向用于创建该环境的 base Python 的目录

A virtual environment may be “activated” using a script in its binary directory (`bin` on POSIX; `Scripts` on Windows). This will prepend that directory to your `PATH`, so that running **python** will invoke the environment’s Python interpreter and you can run installed scripts without having to use their full path. The invocation of the activation script is platform-specific (`_<venv>_` must be replaced by the path to the directory containing the virtual environment):
>  虚拟环境可以使用其二进制目录中的脚本激活，它会把它的目录加入 `PATH` 的前面，这使得运行 `python` 会使用虚拟环境中的解释器

|Platform|Shell|Command to activate virtual environment|
|---|---|---|
|POSIX|bash/zsh|`$ source _<venv>_/bin/activate`|
|fish|`$ source _<venv>_/bin/activate.fish`|
|csh/tcsh|`$ source _<venv>_/bin/activate.csh`|
|pwsh|`$ _<venv>_/bin/Activate.ps1`|
|Windows|cmd.exe|`C:\> _<venv>_\Scripts\activate.bat`|
|PowerShell|`PS C:\> _<venv>_\Scripts\Activate.ps1`|

>  Added in version 3.4: **fish** and **csh** activation scripts.

>  Added in version 3.8: PowerShell activation scripts installed under POSIX for PowerShell Core support.

You don’t specifically _need_ to activate a virtual environment, as you can just specify the full path to that environment’s Python interpreter when invoking Python. Furthermore, all scripts installed in the environment should be runnable without activating it.

In order to achieve this, scripts installed into virtual environments have a “shebang” line which points to the environment’s Python interpreter, `#!/_<path-to-venv>_/bin/python`. This means that the script will run with that interpreter regardless of the value of `PATH`. On Windows, “shebang” line processing is supported if you have the [Python Launcher for Windows](https://docs.python.org/3/using/windows.html#launcher) installed. Thus, double-clicking an installed script in a Windows Explorer window should run it with the correct interpreter without the environment needing to be activated or on the `PATH`.
>  安装到虚拟环境中的脚本都有 shebang line，指向该环境的 Python 解释器
>  这意味着无论 `PATH` 值如何，该脚本都会使用该解释器运行

When a virtual environment has been activated, the `VIRTUAL_ENV` environment variable is set to the path of the environment. Since explicitly activating a virtual environment is not required to use it, `VIRTUAL_ENV` cannot be relied upon to determine whether a virtual environment is being used.
>  虚拟环境被激活后，`VIRTUAL_ENV` 环境变量会被设定为该环境的路径

Warning
Because scripts installed in environments should not expect the environment to be activated, their shebang lines contain the absolute paths to their environment’s interpreters. Because of this, environments are inherently non-portable, in the general case. You should always have a simple means of recreating an environment (for example, if you have a requirements file `requirements.txt`, you can invoke `pip install -r requirements.txt` using the environment’s `pip` to install all of the packages needed by the environment). If for any reason you need to move the environment to a new location, you should recreate it at the desired location and delete the one at the old location. If you move an environment because you moved a parent directory of it, you should recreate the environment in its new location. Otherwise, software installed into the environment may not work as expected.

You can deactivate a virtual environment by typing `deactivate` in your shell. The exact mechanism is platform-specific and is an internal implementation detail (typically, a script or shell function will be used).
>  在 shell 中输入 `deactivate` 可以退出虚拟环境，其实现通常就是一个脚本或者一个 shell function
