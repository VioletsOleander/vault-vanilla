---
completed: true
version: 0.8,14
---
# Using tools
Many Python packages provide applications that can be used as tools. uv has specialized support for easily invoking and installing tools.
>  许多 Python 包提供了可以作为工具使用的应用程序，uv 对这些工具的调用和安装提供了特别的支持

## Running tools
The `uvx` command invokes a tool without installing it.
>  `uvx` 命令会在不安装工具的情况下调用它

For example, to run `ruff`:

```
uvx ruff
```

Note
This is exactly equivalent to:

```
uv tool run ruff
```

`uvx` is provided as an alias for convenience.

>  `uvx ruff` 等价于 `uv tool run ruff`

Arguments can be provided after the tool name:

```
uvx pycowsay hello from uv
```

>  `uvx` 可以在工具名之后提供参数

Tools are installed into temporary, isolated environments when using `uvx`.
>  工具会被安装在一个临时的隔离环境

Note
If you are running a tool in a [_project_](https://docs.astral.sh/uv/concepts/projects/) and the tool requires that your project is installed, e.g., when using `pytest` or `mypy`, you'll want to use [`uv run`](https://docs.astral.sh/uv/guides/projects/#running-commands) instead of `uvx`. Otherwise, the tool will be run in a virtual environment that is isolated from your project.
>  如果我们在一个项目中运行工具，并且工具要求我们的项目已经安装，例如使用 `pytest, mypy` 时，应该使用 `uv run` 而不是 `uvx`，否则，工具会在一个和项目隔离的虚拟环境中运行

If your project has a flat structure, e.g., instead of using a `src` directory for modules, the project itself does not need to be installed and `uvx` is fine. In this case, using `uv run` is only beneficial if you want to pin the version of the tool in the project's dependencies.
>  如果项目的结构为 flat，那么项目本身不需要被安装，可以使用 `uvx`

## Commands with different package names
When `uvx ruff` is invoked, uv installs the `ruff` package which provides the `ruff` command. However, sometimes the package and command names differ.
>  调用 `uvx ruff` 时，uv 会安装提供 `ruff` 命令的 `ruff` 包

The `--from` option can be used to invoke a command from a specific package, e.g., `http` which is provided by `httpie`:

```
uvx --from httpie http
```

>  如果包名和命令名不同，可以使用 `--from` 调用来自特定包的特定命令

## Requesting specific versions
To run a tool at a specific version, use `command@<version>`:

```
uvx ruff@0.3.0 check
```

To run a tool at the latest version, use `command@latest`:

```
uvx ruff@latest check
```

The `--from` option can also be used to specify package versions, as above:

```
uvx --from 'ruff==0.3.0' ruff check
```

Or, to constrain to a range of versions:

```
uvx --from 'ruff>0.2.0,<0.3.0' ruff check
```

Note the `@` syntax cannot be used for anything other than an exact version.

## Requesting extras
The `--from` option can be used to run a tool with extras:

```
uvx --from 'mypy[faster-cache,reports]' mypy --xml-report mypy_report
```

This can also be combined with version selection:

```
uvx --from 'mypy[faster-cache,reports]==1.13.0' mypy --xml-report mypy_report
```

## Requesting different sources
The `--from` option can also be used to install from alternative sources.
>  `--from` 也可以用于指定另外的安装源

For example, to pull from git:

```
uvx --from git+https://github.com/httpie/cli httpie
```

You can also pull the latest commit from a specific named branch:

```
uvx --from git+https://github.com/httpie/cli@master httpie
```

Or pull a specific tag:

```
uvx --from git+https://github.com/httpie/cli@3.2.4 httpie
```

Or even a specific commit:

```
uvx --from git+https://github.com/httpie/cli@2843b87 httpie
```

## Commands with plugins
Additional dependencies can be included, e.g., to include `mkdocs-material` when running `mkdocs`:

```
uvx --with mkdocs-material mkdocs --help
```

## Installing tool
If a tool is used often, it is useful to install it to a persistent environment and add it to the `PATH` instead of invoking `uvx` repeatedly.
>  如果一个工具要被频繁使用，可以将它安装到一个持续的环境，并且加入 `PATH`，而不是反复调用 `uvx`

Tip
`uvx` is a convenient alias for `uv tool run`. All of the other commands for interacting with tools require the full `uv tool` prefix.
>  注意，`uvx` 是 `uv tool run` 的别名，所有其他和工具交互的命令都需要以 `uv tool` 为前缀

To install `ruff`:

```
uv tool install ruff
```

When a tool is installed, its executables are placed in a `bin` directory in the `PATH` which allows the tool to be run without uv. If it's not on the `PATH`, a warning will be displayed and `uv tool update-shell` can be used to add it to the `PATH`.
>  工具被安装后，它的可执行文件会被放在 `PATH` 中的一个 `bin` 目录 (一般是 `/usr/.local/bin`)

After installing `ruff`, it should be available:

```
ruff --version
```

Unlike `uv pip install`, installing a tool does not make its modules available in the current environment. For example, the following command will fail:

```
python -c "import ruff"
```

This isolation is important for reducing interactions and conflicts between dependencies of tools, scripts, and projects.
>  安装工具并不会让它的 modules 在当前环境中可用
>  这种隔离是为了减少工具、脚本的依赖和项目依赖的交互和冲突

Unlike `uvx`, `uv tool install` operates on a _package_ and will install all executables provided by the tool.
>  和 `uvx` 不同，`uv tool install` 以 package 的粒度工作，会安装工具提供的所有可执行文件

For example, the following will install the `http`, `https`, and `httpie` executables:

```
uv tool install httpie
```

Additionally, package versions can be included without `--from`:

```
uv tool install 'httpie>0.1.0'
```

And, similarly, for package sources:

```
uv tool install git+https://github.com/httpie/cli
```

As with `uvx`, installations can include additional packages:

```
uv tool install mkdocs --with mkdocs-material
```

Multiple related executables can be installed together in the same tool environment, using the `--with-executables-from` flag. For example, the following will install the executables from `ansible`, plus those ones provided by `ansible-core` and `ansible-lint`:

```
uv tool install --with-executables-from ansible-core,ansible-lint ansible
```

## Upgrading tools
To upgrade a tool, use `uv tool upgrade`:

```
uv tool upgrade ruff
```

Tool upgrades will respect the version constraints provided when installing the tool. For example, `uv tool install ruff >=0.3,<0.4` followed by `uv tool upgrade ruff` will upgrade Ruff to the latest version in the range `>=0.3,<0.4`.

To instead replace the version constraints, re-install the tool with `uv tool install`:

```
uv tool install ruff>=0.4
```

To instead upgrade all tools:

```
uv tool upgrade --all
```

## Requesting Python versions
By default, uv will use your default Python interpreter (the first it finds) when running, installing, or upgrading tools. You can specify the Python interpreter to use with the `--python` option.

For example, to request a specific Python version when running a tool:

```
uvx --python 3.10 ruff
```

Or, when installing a tool:

```
uv tool install --python 3.10 ruff
```

Or, when upgrading a tool:

```
uv tool upgrade --python 3.10 ruff
```

For more details on requesting Python versions, see the [Python version](https://docs.astral.sh/uv/concepts/python-versions/#requesting-a-version) concept page.

## Legacy Windows Scripts
Tools also support running [legacy setuptools scripts](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#scripts). These scripts are available via `$(uv tool dir)\<tool-name>\Scripts` when installed.

Currently only legacy scripts with the `.ps1`, `.cmd`, and `.bat` extensions are supported.

For example, below is an example running a Command Prompt script.

```
uv tool run --from nuitka==2.6.7 nuitka.cmd --version
```

In addition, you don't need to specify the extension. `uvx` will automatically look for files ending in `.ps1`, `.cmd`, and `.bat` in that order of execution on your behalf.

```
uv tool run --from nuitka==2.6.7 nuitka --version
```

## Next steps
To learn more about managing tools with uv, see the [Tools concept](https://docs.astral.sh/uv/concepts/tools/) page and the [command reference](https://docs.astral.sh/uv/reference/cli/#uv-tool).

Or, read on to learn how to [work on projects](https://docs.astral.sh/uv/guides/projects/).