---
completed: true
version: 0.8,14
---
# Project structure and files
## The `pyproject.toml`
Python project metadata is defined in a [`pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) file. uv requires this file to identify the root directory of a project.

Tip
`uv init` can be used to create a new project. See [Creating projects](https://docs.astral.sh/uv/concepts/projects/init/) for details.

A minimal project definition includes a name and version:

pyproject.toml

```toml
[project]
name = "example"
version = "0.1.0"
```

>  最小的项目定义包含了名称和版本

Additional project metadata and configuration includes:

- [Python version requirement](https://docs.astral.sh/uv/concepts/projects/config/#python-version-requirement)
- [Dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [Build system](https://docs.astral.sh/uv/concepts/projects/config/#build-systems)
- [Entry points (commands)](https://docs.astral.sh/uv/concepts/projects/config/#entry-points)

## The project environment
When working on a project with uv, uv will create a virtual environment as needed. While some uv commands will create a temporary environment (e.g., `uv run --isolated`), uv also manages a persistent environment with the project and its dependencies in a `.venv` directory next to the `pyproject.toml`. It is stored inside the project to make it easy for editors to find — they need the environment to give code completions and type hints. It is not recommended to include the `.venv` directory in version control; it is automatically excluded from `git` with an internal `.gitignore` file.
>  一些 uv 命令会创建临时环境，例如 `uv run --isolated`，但 uv 总是会在 `.venv` 中维护一个持久的环境

To run a command in the project environment, use `uv run`. Alternatively the project environment can be activated as normal for a virtual environment.
>  要在项目环境中运行命令，使用 `uv run`
>  或者可以使用通常激活虚拟环境的方式来激活环境

When `uv run` is invoked, it will create the project environment if it does not exist yet or ensure it is up-to-date if it exists. The project environment can also be explicitly created with `uv sync`. See the [locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/) documentation for details.
>  `uv run` 被调用后，它会在环境不存在或者环境不够新的情况下创建项目环境
>  也可以使用 `uv sync` 显式创建项目环境

It is _not_ recommended to modify the project environment manually, e.g., with `uv pip install`. For project dependencies, use `uv add` to add a package to the environment. For one-off requirements, use [`uvx`](https://docs.astral.sh/uv/guides/tools/) or [`uv run --with`](https://docs.astral.sh/uv/concepts/projects/run/#requesting-additional-dependencies).
>  不推荐手动修改项目环境，例如使用 `uv pip install`
>  可以通过 `uv add` 来为环境添加项目依赖

Tip
If you don't want uv to manage the project environment, set [`managed = false`](https://docs.astral.sh/uv/reference/settings/#managed) to disable automatic locking and syncing of the project. For example:

pyproject.toml

```toml
[tool.uv]
managed = false
```

## The lockfile
uv creates a `uv.lock` file next to the `pyproject.toml`.

`uv.lock` is a _universal_ or _cross-platform_ lockfile that captures the packages that would be installed across all possible Python markers such as operating system, architecture, and Python version.
>  `uv.lock` 是一个通用的或跨平台的锁文件
>  它记录了在所有可能的 Python markers 例如 OS, 架构以及 Python 版本下要安装的包

Unlike the `pyproject.toml`, which is used to specify the broad requirements of your project, the lockfile contains the exact resolved versions that are installed in the project environment. This file should be checked into version control, allowing for consistent and reproducible installations across machines.
>  `uv.lock` 包含了安装在项目环境中精确的解析的版本

A lockfile ensures that developers working on the project are using a consistent set of package versions. Additionally, it ensures when deploying the project as an application that the exact set of used package versions is known.
>  lockfile 确保开发者使用一致的一组包版本，同时确保部署项目时使用精确的一组包版本

The lockfile is [automatically created and updated](https://docs.astral.sh/uv/concepts/projects/sync/#automatic-lock-and-sync) during uv invocations that use the project environment, i.e., `uv sync` and `uv run`. The lockfile may also be explicitly updated using `uv lock`.
>  lockfile 会在使用了项目环境中的 uv 调用中被自动创建和更新，例如 `uv sync, uv run`
>  lockfile 也可以使用 `uv lock` 显式更新

>  Rust 中，库通常只需要版本管理 `Cargo.toml`，应用则还需要额外版本管理 `Cargo.lock`
>  首先，在最外层，应用的每个版本 (例如 `v0.1.0`) 都是根据一组特定的、精确的依赖构建的，应用直接面向用户，因此必须确保应用能够传递出在 ChangLog 中描述的特定，即开发者在自己的机器上，用某一组精确的依赖构建出了版本 `v0.1.0`，知道了有特定的特性，写在了 ChangLog 上，则用户自己构建的 `v0.1.0` 也必须有相同的特性，为了确保这样的面向最终用户的完全可复现性，就必须要求 `Cargo.lock`
>  而在内层中，库是面向开发者的，如果开发者依赖库 A 的 `v0.1.0` 和库 B 的 `v0.1.0` ，并且库 A 和库 B 同时依赖库 C 的 `v0.1.0` 和 `v0.1.1`，那么如果要面向开发者确保库 A 和库 B 在各自版本的完全可复现性，就需要通过库 A 和库 B 的 `Cargo.lock` 安装依赖，此时就会在库 C 上发生冲突
>  为了避免这样的情况，实际安装依赖时还是会根据库的 `Cargo.toml` 进行依赖解析，找到一组可行的依赖组，这样的好处是大大减少了冲突的可能性，但坏处就是开发者可能会发现自己构架的库和库的 ChangLog 中说明的特性存在差异，因为自己构建的库不是完全按照库的 `Cargo.lock` 构建的
>  但这并不会过于影响，因为开发者是在开发的，开发的目标是为用户提供特性，虽然库的构建不完全可复现，但开发者终究是在一组精确的依赖上开发特性，不论库的可复现情况如何，只要开发者能够开发出想要的功能，就满足了向用户提供功能的目标，如果开发者在开发过程中发现这种 “不完全可复现性” 影响了开发，则可以自行调整，无论如何，要确保应用对于最终用户的完全可复现性，对于用户来说，库的 “不完全可复现性” 完全无所谓
>  如果大家都遵循 SemVer，这种库的 “不完全可复现性” 对于开发者的影响将大大减小，因为依赖解析是按照 SemVer 解析的，故这样解析出来的一组可行依赖对于开发者来说，至少在 API 级别上会完全符合各个库的 ChangeLog 描述

`uv.lock` is a human-readable TOML file but is managed by uv and should not be edited manually. The `uv.lock` format is specific to uv and not usable by other tools.

### Relationship to `pylock.toml`
In [PEP 751](https://peps.python.org/pep-0751/), Python standardized a new resolution file format, `pylock.toml`.

`pylock.toml` is a resolution output format intended to replace `requirements.txt` (e.g., in the context of `uv pip compile`, whereby a "locked" `requirements.txt` file is generated from a set of input requirements). `pylock.toml` is standardized and tool-agnostic, such that in the future, `pylock.toml` files generated by uv could be installed by other tools, and vice versa.

Some of uv's functionality cannot be expressed in the `pylock.toml` format; as such, uv will continue to use the `uv.lock` format within the project interface.

However, uv supports `pylock.toml` as an export target and in the `uv pip` CLI. For example:

- To export a `uv.lock` to the `pylock.toml` format, run: `uv export -o pylock.toml`
- To generate a `pylock.toml` file from a set of requirements, run: `uv pip compile -o pylock.toml -r requirements.in`
- To install from a `pylock.toml` file, run: `uv pip sync pylock.toml` or `uv pip install -r pylock.toml`