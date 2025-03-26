`pyproject.toml` is a configuration file used by packaging tools, as well as other tools such as linters, type checkers, etc. There are three possible TOML tables in this file.

- The `[build-system]` table is **strongly recommended**. It allows you to declare which [build backend](https://packaging.python.org/en/latest/glossary/#term-Build-Backend) you use and which other dependencies are needed to build your project.
- The `[project]` table is the format that most build backends use to specify your project’s basic metadata, such as the dependencies, your name, etc.
- The `[tool]` table has tool-specific subtables, e.g., `[tool.hatch]`, `[tool.black]`, `[tool.mypy]`. We only touch upon this table here because its contents are defined by each tool. Consult the particular tool’s documentation to know what it can contain.

>  `pyproject.toml` 可以包含三类 table
>  - `[build-system]` 声明使用哪个构建后端以及构建所需的依赖
>  - `[project]` 声明项目元数据
>  - `[tool]` 取决于特定的工具

Note
The `[build-system]` table should always be present, regardless of which build backend you use (`[build-system]` _defines_ the build tool you use).

On the other hand, the `[project]` table is understood by _most_ build backends, but some build backends use a different format.

A notable exception is [Poetry](https://python-poetry.org/), which before version 2.0 (released January 5, 2025) did not use the `[project]` table, it used the `[tool.poetry]` table instead. With version 2.0, it supports both. Also, the [setuptools](https://setuptools.pypa.io/) build backend supports both the `[project]` table, and the older format in `setup.cfg` or `setup.py`.

For new projects, use the `[project]` table, and keep `setup.py` only if some programmatic configuration is needed (such as building C extensions), but the `setup.cfg` and `setup.py` formats are still valid. See [Is setup.py deprecated?](https://packaging.python.org/en/latest/discussions/setup-py-deprecated/#setup-py-deprecated).

# Declaring the build backend
The `[build-system]` table contains a `build-backend` key, which specifies the build backend to be used. It also contains a `requires` key, which is a list of dependencies needed to build the project – this is typically just the build backend package, but it may also contain additional dependencies. You can also constrain the versions, e.g., `requires = ["setuptools >= 61.0"]`.
>  `[build-system]` 中，`build-backend` 指定使用的构建后端，`requries` 指定构建依赖，通常仅包含构建后端的包 (及其版本)

Usually, you’ll just copy what your build backend’s documentation suggests (after [choosing your build backend](https://packaging.python.org/en/latest/tutorials/packaging-projects/#choosing-build-backend)). Here are the values for some common build backends:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

# Static vs. dynamic metadata
The rest of this guide is devoted to the `[project]` table.

Most of the time, you will directly write the value of a `[project]` field. For example: `requires-python = ">= 3.8"`, or `version = "1.0"`.

However, in some cases, it is useful to let your build backend compute the metadata for you. For example: many build backends can read the version from a `__version__` attribute in your code, a Git tag, or similar. In such cases, you should mark the field as dynamic using, e.g.,

```toml
[project]
dynamic = ["version"]
```

When a field is dynamic, it is the build backend’s responsibility to fill it. Consult your build backend’s documentation to learn how it does it.

# Basic information
## `name`
Put the name of your project on PyPI. This field is required and is the only field that cannot be marked as dynamic.

```toml
[project]
name = "spam-eggs"
```

The project name must consist of ASCII letters, digits, underscores “`_`”, hyphens “`-`” and periods “`.`”. It must not start or end with an underscore, hyphen or period.

Comparison of project names is case insensitive and treats arbitrarily long runs of underscores, hyphens, and/or periods as equal. For example, if you register a project named `cool-stuff`, users will be able to download it or declare a dependency on it using any of the following spellings: `Cool-Stuff`, `cool.stuff`, `COOL_STUFF`, `CoOl__-.-__sTuFF`.

>  `name` : 项目在 PyPI 上的名字，这是唯一不能是 dynamic 的 field

## `version`
Put the version of your project.

```toml
[project]
version = "2020.0.0"
```

Some more complicated version specifiers like `2020.0.0a1` (for an alpha release) are possible; see the [specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers) for full details.

This field is required, although it is often marked as dynamic using

```toml
[project]
dynamic = ["version"]
```

This allows use cases such as filling the version from a `__version__` attribute or a Git tag. Consult the [Single-sourcing the Project Version](https://packaging.python.org/en/latest/discussions/single-source-version/#single-source-version) discussion for more details.

# Dependencies and requirements
## `dependencies` / `optional-dependencies`
If your project has dependencies, list them like this:

```toml
[project]
dependencies = [
  "httpx",
  "gidgethub[httpx]>4.0.0",
  "django>2.1; os_name != 'nt'",
  "django>2.0; os_name == 'nt'",
]
```

See [Dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#dependency-specifiers) for the full syntax you can use to constrain versions.

You may want to make some of your dependencies optional, if they are only needed for a specific feature of your package. In that case, put them in `optional-dependencies`.

```toml
[project.optional-dependencies]
gui = ["PyQt5"]
cli = [
  "rich",
  "click",
]
```

Each of the keys defines a “packaging extra”. In the example above, one could use, e.g., `pip install your-project-name[gui]` to install your project with GUI support, adding the PyQt5 dependency.

>  `[project.optional-dependencies]` 中每个 key 都定义一个额外包，在 `pip install` 时可以通过 `[key]` 以指定同时下载额外依赖

## `requires-python
This lets you declare the minimum version of Python that you support [1](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#requires-python-upper-bounds).

```toml
[project]
requires-python = ">= 3.8"
```

# Creating executable scripts
To install a command as part of your package, declare it in the `[project.scripts]` table.

```toml
[project.scripts]
spam-cli = "spam:main_cli"
```

In this example, after installing your project, a `spam-cli` command will be available. Executing this command will do the equivalent of `import sys; from spam import main_cli; sys.exit(main_cli())`.

>  `[project.scripts]` 中的 key 会在安装时成为一个命令，其执行等价于 `sys.exit(func())`

On Windows, scripts packaged this way need a terminal, so if you launch them from within a graphical application, they will make a terminal pop up. To prevent this from happening, use the `[project.gui-scripts]` table instead of `[project.scripts]`.

```toml
[project.gui-scripts]
spam-gui = "spam:main_gui"
```

In that case, launching your script from the command line will give back control immediately, leaving the script to run in the background.

The difference between `[project.scripts]` and `[project.gui-scripts]` is only relevant on Windows.

# About your project

## `authors` / `maintainers`
Both of these fields contain lists of people identified by a name and/or an email address.

```toml
[project]
authors = [
  {name = "Pradyun Gedam", email = "pradyun@example.com"},
  {name = "Tzu-Ping Chung", email = "tzu-ping@example.com"},
  {name = "Another person"},
  {email = "different.person@example.com"},
]
maintainers = [
  {name = "Brett Cannon", email = "brett@example.com"}
]
```

## `description`
This should be a one-line description of your project, to show as the “headline” of your project page on PyPI ([example](https://pypi.org/project/pip)), and other places such as lists of search results ([example](https://pypi.org/search?q=pip)).

```toml
[project]
description = "Lovely Spam! Wonderful Spam!"
```

## `readme`
This is a longer description of your project, to display on your project page on PyPI. Typically, your project will have a `README.md` or `README.rst` file and you just put its file name here.

```
[project]
readme = "README.md"
```

The README’s format is auto-detected from the extension:

- `README.md` → [GitHub-flavored Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax),
- `README.rst` → [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) (without Sphinx extensions).

You can also specify the format explicitly, like this:

```
[project]
readme = {file = "README.txt", content-type = "text/markdown"}
# or
readme = {file = "README.txt", content-type = "text/x-rst"}
```

## `license
[**PEP 639**](https://peps.python.org/pep-0639/) (accepted in August 2024) has changed the way the `license` field is declared. Make sure your preferred build backend supports [**PEP 639**](https://peps.python.org/pep-0639/) before trying to apply the newer guidelines. As of February 2025, [setuptools](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html "(in setuptools v75.8.1.post20250225)") and [flit](https://flit.pypa.io/en/stable/pyproject_toml.html#pyproject-toml-project "(in Flit v3.11.0)") don’t support [**PEP 639**](https://peps.python.org/pep-0639/) yet.

## [**PEP 639**](https://peps.python.org/pep-0639/) license declaration
This is a valid [SPDX license expression](https://packaging.python.org/en/latest/glossary/#term-License-Expression) consisting of one or more [license identifiers](https://packaging.python.org/en/latest/glossary/#term-License-Identifier). The full license list is available at the [SPDX license list page](https://spdx.org/licenses/). The supported list version is 3.17 or any later compatible one.

```
[project]
license = "GPL-3.0-or-later"
# or
license = "MIT AND (Apache-2.0 OR BSD-2-Clause)"
```

As a general rule, it is a good idea to use a standard, well-known license, both to avoid confusion and because some organizations avoid software whose license is unapproved.

If your project is licensed with a license that doesn’t have an existing SPDX identifier, you can create a custom one in format `LicenseRef-[idstring]`. The custom identifiers must follow the SPDX specification, [clause 10.1](https://spdx.github.io/spdx-spec/v2.2.2/other-licensing-information-detected/) of the version 2.2 or any later compatible one.

```
[project]
license = "LicenseRef-My-Custom-License"
```

## Legacy license declaration
This can take two forms. You can put your license in a file, typically `LICENSE` or `LICENSE.txt`, and link that file here:

```
[project]
license = {file = "LICENSE"}
```

or you can write the name of the license:

```
[project]
license = {text = "MIT License"}
```

If you are using a standard, well-known license, it is not necessary to use this field. Instead, you should use one of the [classifiers](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#classifiers) starting with `License ::`. (As a general rule, it is a good idea to use a standard, well-known license, both to avoid confusion and because some organizations avoid software whose license is unapproved.)

## `license-files`
[**PEP 639**](https://peps.python.org/pep-0639/) (accepted in August 2024) has introduced the `license-files` field. Make sure your preferred build backend supports [**PEP 639**](https://peps.python.org/pep-0639/) before declaring the field. As of February 2025, [setuptools](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html "(in setuptools v75.8.1.post20250225)") and [flit](https://flit.pypa.io/en/stable/pyproject_toml.html#pyproject-toml-project "(in Flit v3.11.0)") don’t support [**PEP 639**](https://peps.python.org/pep-0639/) yet.

This is a list of license files and files containing other legal information you want to distribute with your package.

```
[project]
license-files = ["LICEN[CS]E*", "vendored/licenses/*.txt", "AUTHORS.md"]
```

The glob patterns must follow the specification:

- Alphanumeric characters, underscores (`_`), hyphens (`-`) and dots (`.`) will be matched verbatim.
- Special characters: `*`, `?`, `**` and character ranges: [] are supported.
- Path delimiters must be the forward slash character (`/`).
- Patterns are relative to the directory containing `pyproject.toml`, and thus may not start with a slash character.
- Parent directory indicators (`..`) must not be used.
- Each glob must match at least one file.

Literal paths are valid globs. Any characters or character sequences not covered by this specification are invalid.

## `keywords
This will help PyPI’s search box to suggest your project when people search for these keywords.

```
[project]
keywords = ["egg", "bacon", "sausage", "tomatoes", "Lobster Thermidor"]
```

## `classifiers`
A list of PyPI classifiers that apply to your project. Check the [full list of possibilities](https://pypi.org/classifiers).

```toml
classifiers = [
  # How mature is this project? Common values are
  #   3 - Alpha
  #   4 - Beta
  #   5 - Production/Stable
  "Development Status :: 4 - Beta",

  # Indicate who your project is intended for
  "Intended Audience :: Developers",
  "Topic :: Software Development :: Build Tools",

  # Specify the Python versions you support here.
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.6",
  "Programming Language :: Python :: 3.7",
  "Programming Language :: Python :: 3.8",
  "Programming Language :: Python :: 3.9",
]
```

Although the list of classifiers is often used to declare what Python versions a project supports, this information is only used for searching and browsing projects on PyPI, not for installing projects. To actually restrict what Python versions a project can be installed on, use the [requires-python](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#requires-python) argument.

To prevent a package from being uploaded to PyPI, use the special `Private :: Do Not Upload` classifier. PyPI will always reject packages with classifiers beginning with `Private ::`.

## `urls
A list of URLs associated with your project, displayed on the left sidebar of your PyPI project page.

Note
See [Well-known labels](https://packaging.python.org/en/latest/specifications/well-known-project-urls/#well-known-labels) for a listing of labels that PyPI and other packaging tools are specifically aware of, and [PyPI’s project metadata docs](https://docs.pypi.org/project_metadata/#project-urls) for PyPI-specific URL processing.

```
[project.urls]
Homepage = "https://example.com"
Documentation = "https://readthedocs.org"
Repository = "https://github.com/me/spam.git"
Issues = "https://github.com/me/spam/issues"
Changelog = "https://github.com/me/spam/blob/master/CHANGELOG.md"
```

Note that if the label contains spaces, it needs to be quoted, e.g., `Website = "https://example.com"` but `"Official Website" = "https://example.com"`.

Users are advised to use [Well-known labels](https://packaging.python.org/en/latest/specifications/well-known-project-urls/#well-known-labels) for their project URLs where appropriate, since consumers of metadata (like package indices) can specialize their presentation.

For example in the following metadata, neither `MyHomepage` nor `"Download Link"` is a well-known label, so they will be rendered verbatim:

```
[project.urls]
MyHomepage = "https://example.com"
"Download Link" = "https://example.com/abc.tar.gz"
```

Whereas in this metadata `HomePage` and `DOWNLOAD` both have well-known equivalents (`homepage` and `download`), and can be presented with those semantics in mind (the project’s home page and its external download location, respectively).

```
[project.urls]
HomePage = "https://example.com"
DOWNLOAD = "https://example.com/abc.tar.gz"
```

# Advanced plugins
Some packages can be extended through plugins. Examples include [Pytest](https://pytest.org/) and [Pygments](https://pygments.org/). To create such a plugin, you need to declare it in a subtable of `[project.entry-points]` like this:

```
[project.entry-points."spam.magical"]
tomatoes = "spam:main_tomatoes"
```

See the [Plugin guide](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#plugin-entry-points) for more information.

# A full example

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "spam-eggs"
version = "2020.0.0"
dependencies = [
  "httpx",
  "gidgethub[httpx]>4.0.0",
  "django>2.1; os_name != 'nt'",
  "django>2.0; os_name == 'nt'",
]
requires-python = ">=3.8"
authors = [
  {name = "Pradyun Gedam", email = "pradyun@example.com"},
  {name = "Tzu-Ping Chung", email = "tzu-ping@example.com"},
  {name = "Another person"},
  {email = "different.person@example.com"},
]
maintainers = [
  {name = "Brett Cannon", email = "brett@example.com"}
]
description = "Lovely Spam! Wonderful Spam!"
readme = "README.rst"
license = "MIT"  # or license = {file = "LICENSE.txt"} for legacy declaration
license-files = ["LICEN[CS]E.*"]
keywords = ["egg", "bacon", "sausage", "tomatoes", "Lobster Thermidor"]
classifiers = [
  "Development Status :: 4 - Beta",
  "Programming Language :: Python"
]

[project.optional-dependencies]
gui = ["PyQt5"]
cli = [
  "rich",
  "click",
]

[project.urls]
Homepage = "https://example.com"
Documentation = "https://readthedocs.org"
Repository = "https://github.com/me/spam.git"
"Bug Tracker" = "https://github.com/me/spam/issues"
Changelog = "https://github.com/me/spam/blob/master/CHANGELOG.md"

[project.scripts]
spam-cli = "spam:main_cli"

[project.gui-scripts]
spam-gui = "spam:main_gui"

[project.entry-points."spam.magical"]
tomatoes = "spam:main_tomatoes"
```

---

[1](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#id2) Think twice before applying an upper bound like `requires-python = "<= 3.10"` here. [This blog post](https://iscinumpy.dev/post/bound-version-constraints/#pinning-the-python-version-is-special) contains some information regarding possible problems.

>  Last updated on Feb 26, 2025