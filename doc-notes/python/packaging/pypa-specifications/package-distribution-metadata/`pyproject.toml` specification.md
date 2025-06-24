> [!Warning]
> This is a **technical, formal specification**. For a gentle, user-friendly guide to `pyproject.toml`, see [Writing your pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#writing-pyproject-toml).

The `pyproject.toml` file acts as a configuration file for packaging-related tools (as well as other tools).
>  `pyproject.toml` 的作用是作为打包相关的工具所使用的配置文件

> [!Note]
> This specification was originally defined in [**PEP 518**](https://peps.python.org/pep-0518/) and [**PEP 621**](https://peps.python.org/pep-0621/).

The `pyproject.toml` file is written in [TOML](https://toml.io/). Three tables are currently specified, namely [build-system](https://packaging.python.org/en/latest/specifications/pyproject-toml/#pyproject-build-system-table), [project](https://packaging.python.org/en/latest/specifications/pyproject-toml/#pyproject-project-table) and [tool](https://packaging.python.org/en/latest/specifications/pyproject-toml/#pyproject-tool-table). Other tables are reserved for future use (tool-specific configuration should use the `[tool]` table).
>  `pyproject.toml` 中可以定义三个 table: `build-system`, `project`, `tool`
>  特定于工具的配置应该都写在 `tool` table 中

## Declaring build system dependencies: the `[build-system]` table
The `[build-system]` table declares any Python level dependencies that must be installed in order to run the project’s build system successfully.
>  `[build-system]` table 应该声明在成功运行项目的构建系统之前必须安装的任何 Python 级依赖项

The `[build-system]` table is used to store build-related data. Initially, only one key of the table is valid and is mandatory for the table: `requires`. This key must have a value of a list of strings representing dependencies required to execute the build system. The strings in this list follow the [version specifier specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers).
>  `[build-system]` table 应该用于存储构建相关的数据
>  该 table 中只有一个 key 是必须的: `requries` ，该 key 的值必须是一个字符串列表，表示执行构建系统所要求的依赖项
>  该列表中的字符串遵循版本说明符规范

An example `[build-system]` table for a project built with `setuptools` is:

```toml
[build-system]
# Minimum requirements for the build system to execute.
requires = ["setuptools"]
```

Build tools are expected to use the example configuration file above as their default semantics when a `pyproject.toml` file is not present.

Tools should not require the existence of the `[build-system]` table. A `pyproject.toml` file may be used to store configuration details other than build-related data and thus lack a `[build-system]` table legitimately. If the file exists but is lacking the `[build-system]` table then the default values as specified above should be used. If the table is specified but is missing required fields then the tool should consider it an error.

To provide a type-specific representation of the resulting data from the TOML file for illustrative purposes only, the following [JSON Schema](https://json-schema.org/) would match the data format:

```json
{
    "$schema": "http://json-schema.org/schema#",

    "type": "object",
    "additionalProperties": false,

    "properties": {
        "build-system": {
            "type": "object",
            "additionalProperties": false,

            "properties": {
                "requires": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["requires"]
        },

        "tool": {
            "type": "object"
        }
    }
}
```

## Declaring project metadata: the `[project]` table
The `[project]` table specifies the project’s [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata).

There are two kinds of metadata: _static_ and _dynamic_. Static metadata is specified in the `pyproject.toml` file directly and cannot be specified or changed by a tool (this includes data _referred_ to by the metadata, e.g. the contents of files referenced by the metadata). Dynamic metadata is listed via the `dynamic` key (defined later in this specification) and represents metadata that a tool will later provide.

The lack of a `[project]` table implicitly means the [build backend](https://packaging.python.org/en/latest/glossary/#term-Build-Backend) will dynamically provide all keys.

The only keys required to be statically defined are:

- `name`

The keys which are required but may be specified _either_ statically or listed as dynamic are:

- `version`

All other keys are considered optional and may be specified statically, listed as dynamic, or left unspecified.

The complete list of keys allowed in the `[project]` table are:

- `authors`
- `classifiers`
- `dependencies`
- `description`
- `dynamic`
- `entry-points`
- `gui-scripts`
- `keywords`
- `license`
- `license-files`
- `maintainers`
- `name`
- `optional-dependencies`
- `readme`
- `requires-python`
- `scripts`
- `urls`
- `version`

### `name`

- [TOML](https://toml.io/) type: string
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Name](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-name)
    

The name of the project.

Tools SHOULD [normalize](https://packaging.python.org/en/latest/specifications/name-normalization/#name-normalization) this name, as soon as it is read for internal consistency.

### `version`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#version "Link to this heading")

- [TOML](https://toml.io/) type: string
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Version](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-version)
    

The version of the project, as defined in the [Version specifier specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers).

Users SHOULD prefer to specify already-normalized versions.

### `description`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#description "Link to this heading")

- [TOML](https://toml.io/) type: string
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Summary](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-summary)
    

The summary description of the project in one line. Tools MAY error if this includes multiple lines.

### `readme`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#readme "Link to this heading")

- [TOML](https://toml.io/) type: string or table
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Description](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-description) and [Description-Content-Type](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-description-content-type)
    

The full description of the project (i.e. the README).

The key accepts either a string or a table. If it is a string then it is a path relative to `pyproject.toml` to a text file containing the full description. Tools MUST assume the file’s encoding is UTF-8. If the file path ends in a case-insensitive `.md` suffix, then tools MUST assume the content-type is `text/markdown`. If the file path ends in a case-insensitive `.rst`, then tools MUST assume the content-type is `text/x-rst`. If a tool recognizes more extensions than this PEP, they MAY infer the content-type for the user without specifying this key as `dynamic`. For all unrecognized suffixes when a content-type is not provided, tools MUST raise an error.

The `readme` key may also take a table. The `file` key has a string value representing a path relative to `pyproject.toml` to a file containing the full description. The `text` key has a string value which is the full description. These keys are mutually-exclusive, thus tools MUST raise an error if the metadata specifies both keys.

A table specified in the `readme` key also has a `content-type` key which takes a string specifying the content-type of the full description. A tool MUST raise an error if the metadata does not specify this key in the table. If the metadata does not specify the `charset` parameter, then it is assumed to be UTF-8. Tools MAY support other encodings if they choose to. Tools MAY support alternative content-types which they can transform to a content-type as supported by the [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata). Otherwise tools MUST raise an error for unsupported content-types.

### `requires-python`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#requires-python "Link to this heading")

- [TOML](https://toml.io/) type: string
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Requires-Python](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-requires-python)
    

The Python version requirements of the project.

### `license`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#license "Link to this heading")

- [TOML](https://toml.io/) type: string
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [License-Expression](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-license-expression)
    

Text string that is a valid SPDX [license expression](https://packaging.python.org/en/latest/glossary/#term-License-Expression), as specified in [License Expression](https://packaging.python.org/en/latest/specifications/license-expression/). Tools SHOULD validate and perform case normalization of the expression.

#### Legacy specification[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#legacy-specification "Link to this heading")

- [TOML](https://toml.io/) type: table
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [License](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-license)
    

The table may have one of two keys. The `file` key has a string value that is a file path relative to `pyproject.toml` to the file which contains the license for the project. Tools MUST assume the file’s encoding is UTF-8. The `text` key has a string value which is the license of the project. These keys are mutually exclusive, so a tool MUST raise an error if the metadata specifies both keys.

The table subkeys were deprecated by [**PEP 639**](https://peps.python.org/pep-0639/) in favor of the string value.

### `license-files`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#license-files "Link to this heading")

- [TOML](https://toml.io/) type: array of strings
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [License-File](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-license-file)
    

An array specifying paths in the project source tree relative to the project root directory (i.e. directory containing `pyproject.toml` or legacy project configuration files, e.g. `setup.py`, `setup.cfg`, etc.) to file(s) containing licenses and other legal notices to be distributed with the package.

The strings MUST contain valid glob patterns, as specified in [glob patterns](https://packaging.python.org/en/latest/specifications/glob-patterns/).

Patterns are relative to the directory containing `pyproject.toml`,

Tools MUST assume that license file content is valid UTF-8 encoded text, and SHOULD validate this and raise an error if it is not.

Build tools:

- MUST include all files matched by a listed pattern in all distribution archives.
    
- MUST list each matched file path under a License-File field in the Core Metadata.
    

If the `license-files` key is present and is set to a value of an empty array, then tools MUST NOT include any license files and MUST NOT raise an error. If the `license-files` key is not defined, tools can decide how to handle license files. For example they can choose not to include any files or use their own logic to discover the appropriate files in the distribution.

### `authors`/`maintainers`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#authors-maintainers "Link to this heading")

- [TOML](https://toml.io/) type: Array of inline tables with string keys and values
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Author](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-author), [Author-email](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-author-email), [Maintainer](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-maintainer), and [Maintainer-email](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-maintainer-email)
    

The people or organizations considered to be the “authors” of the project. The exact meaning is open to interpretation — it may list the original or primary authors, current maintainers, or owners of the package.

The “maintainers” key is similar to “authors” in that its exact meaning is open to interpretation.

These keys accept an array of tables with 2 keys: `name` and `email`. Both values must be strings. The `name` value MUST be a valid email name (i.e. whatever can be put as a name, before an email, in [**RFC 822**](https://datatracker.ietf.org/doc/html/rfc822.html)) and not contain commas. The `email` value MUST be a valid email address. Both keys are optional, but at least one of the keys must be specified in the table.

Using the data to fill in [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) is as follows:

1. If only `name` is provided, the value goes in [Author](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-author) or [Maintainer](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-maintainer) as appropriate.
    
2. If only `email` is provided, the value goes in [Author-email](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-author-email) or [Maintainer-email](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-maintainer-email) as appropriate.
    
3. If both `email` and `name` are provided, the value goes in [Author-email](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-author-email) or [Maintainer-email](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-maintainer-email) as appropriate, with the format `{name} <{email}>`.
    
4. Multiple values should be separated by commas.
    

### `keywords`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#keywords "Link to this heading")

- [TOML](https://toml.io/) type: array of strings
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Keywords](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-keywords)
    

The keywords for the project.

### `classifiers`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#classifiers "Link to this heading")

- [TOML](https://toml.io/) type: array of strings
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Classifier](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-classifier)
    

Trove classifiers which apply to the project.

The use of `License ::` classifiers is deprecated and tools MAY issue a warning informing users about that. Build tools MAY raise an error if both the `license` string value (translating to `License-Expression` metadata field) and the `License ::` classifiers are used.

### `urls`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#urls "Link to this heading")

- [TOML](https://toml.io/) type: table with keys and values of strings
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Project-URL](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-project-url)
    

A table of URLs where the key is the URL label and the value is the URL itself. See [Well-known Project URLs in Metadata](https://packaging.python.org/en/latest/specifications/well-known-project-urls/#well-known-project-urls) for normalization rules and well-known rules when processing metadata for presentation.

### Entry points[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#entry-points "Link to this heading")

- [TOML](https://toml.io/) type: table (`[project.scripts]`, `[project.gui-scripts]`, and `[project.entry-points]`)
    
- [Entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/#entry-points)
    

There are three tables related to entry points. The `[project.scripts]` table corresponds to the `console_scripts` group in the [entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/#entry-points). The key of the table is the name of the entry point and the value is the object reference.

The `[project.gui-scripts]` table corresponds to the `gui_scripts` group in the [entry points specification](https://packaging.python.org/en/latest/specifications/entry-points/#entry-points). Its format is the same as `[project.scripts]`.

The `[project.entry-points]` table is a collection of tables. Each sub-table’s name is an entry point group. The key and value semantics are the same as `[project.scripts]`. Users MUST NOT create nested sub-tables but instead keep the entry point groups to only one level deep.

Build back-ends MUST raise an error if the metadata defines a `[project.entry-points.console_scripts]` or `[project.entry-points.gui_scripts]` table, as they would be ambiguous in the face of `[project.scripts]` and `[project.gui-scripts]`, respectively.

### `dependencies`/`optional-dependencies`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#dependencies-optional-dependencies "Link to this heading")

- [TOML](https://toml.io/) type: Array of [**PEP 508**](https://peps.python.org/pep-0508/) strings (`dependencies`), and a table with values of arrays of [**PEP 508**](https://peps.python.org/pep-0508/) strings (`optional-dependencies`)
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Requires-Dist](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-requires-dist) and [Provides-Extra](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-provides-extra)
    

The (optional) dependencies of the project.

For `dependencies`, it is a key whose value is an array of strings. Each string represents a dependency of the project and MUST be formatted as a valid [**PEP 508**](https://peps.python.org/pep-0508/) string. Each string maps directly to a [Requires-Dist](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-requires-dist) entry.

For `optional-dependencies`, it is a table where each key specifies an extra and whose value is an array of strings. The strings of the arrays must be valid [**PEP 508**](https://peps.python.org/pep-0508/) strings. The keys MUST be valid values for [Provides-Extra](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-provides-extra). Each value in the array thus becomes a corresponding [Requires-Dist](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-requires-dist) entry for the matching [Provides-Extra](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-provides-extra) metadata.

### `dynamic`[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#dynamic "Link to this heading")

- [TOML](https://toml.io/) type: array of string
    
- Corresponding [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) field: [Dynamic](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata-dynamic)
    

Specifies which keys listed by this PEP were intentionally unspecified so another tool can/will provide such metadata dynamically. This clearly delineates which metadata is purposefully unspecified and expected to stay unspecified compared to being provided via tooling later on.

- A build back-end MUST honour statically-specified metadata (which means the metadata did not list the key in `dynamic`).
    
- A build back-end MUST raise an error if the metadata specifies `name` in `dynamic`.
    
- If the [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) specification lists a field as “Required”, then the metadata MUST specify the key statically or list it in `dynamic` (build back-ends MUST raise an error otherwise, i.e. it should not be possible for a required key to not be listed somehow in the `[project]` table).
    
- If the [core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#core-metadata) specification lists a field as “Optional”, the metadata MAY list it in `dynamic` if the expectation is a build back-end will provide the data for the key later.
    
- Build back-ends MUST raise an error if the metadata specifies a key statically as well as being listed in `dynamic`.
    
- If the metadata does not list a key in `dynamic`, then a build back-end CANNOT fill in the requisite metadata on behalf of the user (i.e. `dynamic` is the only way to allow a tool to fill in metadata and the user must opt into the filling in).
    
- Build back-ends MUST raise an error if the metadata specifies a key in `dynamic` but the build back-end was unable to determine the data for it (omitting the data, if determined to be the accurate value, is acceptable).
    

## Arbitrary tool configuration: the `[tool]` table[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#arbitrary-tool-configuration-the-tool-table "Link to this heading")

The `[tool]` table is where any tool related to your Python project, not just build tools, can have users specify configuration data as long as they use a sub-table within `[tool]`, e.g. the [flit](https://pypi.python.org/pypi/flit) tool would store its configuration in `[tool.flit]`.

A mechanism is needed to allocate names within the `tool.*` namespace, to make sure that different projects do not attempt to use the same sub-table and collide. Our rule is that a project can use the subtable `tool.$NAME` if, and only if, they own the entry for `$NAME` in the Cheeseshop/PyPI.

## History[](https://packaging.python.org/en/latest/specifications/pyproject-toml/#history "Link to this heading")

- May 2016: The initial specification of the `pyproject.toml` file, with just a `[build-system]` containing a `requires` key and a `[tool]` table, was approved through [**PEP 518**](https://peps.python.org/pep-0518/).
    
- November 2020: The specification of the `[project]` table was approved through [**PEP 621**](https://peps.python.org/pep-0621/).
    
- December 2024: The `license` key was redefined, the `license-files` key was added and `License::` classifiers were deprecated through [**PEP 639**](https://peps.python.org/pep-0639/).
    

[

  




](https://packaging.python.org/en/latest/specifications/dependency-groups/)