---
version: 4.0.1
---
## Synopsis

```
Generate a Project Buildsystem
 cmake [<options>] -B <path-to-build> [-S <path-to-source>]
 cmake [<options>] <path-to-source | path-to-existing-build>

Build a Project
 cmake --build <dir> [<options>] [-- <build-tool-options>]

Install a Project
 cmake --install <dir> [<options>]

Open a Project
 cmake --open <dir>

Run a Script
 cmake [-D <var>=<value>]... -P <cmake-script-file>

Run a Command-Line Tool
 cmake -E <command> [<options>]

Run the Find-Package Tool
 cmake --find-package [<options>]

Run a Workflow Preset
 cmake --workflow <options>

View Help
 cmake --help[-<topic>]
```

## Description
The **cmake** executable is the command-line interface of the cross-platform buildsystem generator CMake. The above [Synopsis](https://cmake.org/cmake/help/latest/manual/cmake.1.html#synopsis) lists various actions the tool can perform as described in sections below.
>  `cmake` 是跨平台构建系统生成器 CMake 的命令行接口

To build a software project with CMake, [Generate a Project Buildsystem](https://cmake.org/cmake/help/latest/manual/cmake.1.html#generate-a-project-buildsystem). Optionally use **cmake** to [Build a Project](https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project), [Install a Project](https://cmake.org/cmake/help/latest/manual/cmake.1.html#install-a-project) or just run the corresponding build tool (e.g. `make`) directly. **cmake** can also be used to [View Help](https://cmake.org/cmake/help/latest/manual/cmake.1.html#view-help).
>  要用 CMake 构建软件项目，需要用 `cmake` 生成项目构建系统 (Generate a Project Buildsystem)，并使用 `cmake` 构建项目、安装项目 (Build a Project, Install a Project)

The other actions are meant for use by software developers writing scripts in the [`CMake language`](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#manual:cmake-language\(7\) "cmake-language(7)") to support their builds.
>  `cmake` 的其他行为 (例如 Run a Script, Runa Command-Line Tool 等) 是供软件开发者编写基于 CMake 语言的脚本以支持其构建

For graphical user interfaces that may be used in place of **cmake**, see [`ccmake`](https://cmake.org/cmake/help/latest/manual/ccmake.1.html#manual:ccmake\(1\) "ccmake(1)") and [`cmake-gui`](https://cmake.org/cmake/help/latest/manual/cmake-gui.1.html#manual:cmake-gui\(1\) "cmake-gui(1)"). For command-line interfaces to the CMake testing and packaging facilities, see [`ctest`](https://cmake.org/cmake/help/latest/manual/ctest.1.html#manual:ctest\(1\) "ctest(1)") and [`cpack`](https://cmake.org/cmake/help/latest/manual/cpack.1.html#manual:cpack\(1\) "cpack(1)").
>  CMake 的测试和打包工具接口为 `ctest, cpack`

For more information on CMake at large, [see also](https://cmake.org/cmake/help/latest/manual/cmake.1.html#see-also) the links at the end of this manual.

## Introduction to CMake Buildsystems
A _buildsystem_ describes how to build a project's executables and libraries from its source code using a _build tool_ to automate the process. For example, a buildsystem may be a `Makefile` for use with a command-line `make` tool or a project file for an Integrated Development Environment (IDE). In order to avoid maintaining multiple such buildsystems, a project may specify its buildsystem abstractly using files written in the [`CMake language`](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#manual:cmake-language\(7\) "cmake-language(7)"). From these files CMake generates a preferred buildsystem locally for each user through a backend called a _generator_.
>  一个构建系统描述了如何使用构建工具自动化将项目源码构建为可执行文件和库
>  项目可以使用 CMake 语言抽象地描述其构建系统，以避免为不同的构建系统维护不同的构建文件
>  CMake 基于对构建系统的抽象描述，通过称为生成器的后端，为每个用户生成本地首选的构建系统

To generate a buildsystem with CMake, the following must be selected:
>  要使用 CMake 生成构建系统，需要指定以下三个部分

Source Tree
The top-level directory containing source files provided by the project. The project specifies its buildsystem using files as described in the [`cmake-language(7)`](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#manual:cmake-language\(7\) "cmake-language(7)") manual, starting with a top-level file named `CMakeLists.txt`. These files specify build targets and their dependencies as described in the [`cmake-buildsystem(7)`](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#manual:cmake-buildsystem\(7\) "cmake-buildsystem(7)") manual.
>  源树
>  源树指包含项目源码的顶层目录
>  项目通过 `CMakeLists.txt` 描述其构建系统，这些文件指定需要构建的目标，以及它们各自的依赖

Build Tree
The top-level directory in which buildsystem files and build output artifacts (e.g. executables and libraries) are to be stored. CMake will write a `CMakeCache.txt` file to identify the directory as a build tree and store persistent information such as buildsystem configuration options.
>  构建树
>  构建树指构建输出物 (可执行文件和库) 存放的顶层目录
>  CMake 会写入一个 `CMakeCache.txt` 文件来标识某个目录为构建树，并存储例如构建系统配置选项等持久化信息

To maintain a pristine source tree, perform an _out-of-source_ build by using a separate dedicated build tree. An _in-source_ build in which the build tree is placed in the same directory as the source tree is also supported, but discouraged.
>  建议构建树和源树分离，即进行外部构建

Generator
This chooses the kind of buildsystem to generate. See the [`cmake-generators(7)`](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#manual:cmake-generators\(7\) "cmake-generators(7)") manual for documentation of all generators. Run [`cmake --help`](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-h) to see a list of generators available locally. Optionally use the [`-G`](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-G) option below to specify a generator, or simply accept the default CMake chooses for the current platform.
>  生成器
>  生成器指定需要生成的构建系统的类别，可以通过 `-G` 指定生成器，或者让 CMake 根据当前平台默认选择

When using one of the [Command-Line Build Tool Generators](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#command-line-build-tool-generators) CMake expects that the environment needed by the compiler toolchain is already configured in the shell. When using one of the [IDE Build Tool Generators](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#ide-build-tool-generators), no particular environment is needed.
>  如果生成器属于 Command-Line Build Tool Generators，则 CMake 期望当前环境已经配置好了所需要的编译工具链，如果生成器属于 IDE Build Tool Generators, CMake 不要求特定环境

## Generate a Project Buildsystem
Run CMake with one of the following command signatures to specify the source and build trees and generate a buildsystem:

`cmake [<options>] -B <path-to-build> [-S <path-to-source>]`

*Added in version 3.13.*
 
Uses `<path-to-build>` as the build tree and `<path-to-source>` as the source tree. The specified paths may be absolute or relative to the current working directory. The source tree must contain a `CMakeLists.txt` file. The build tree will be created automatically if it does not already exist. For example:

```
$ cmake -S src -B build
```

`cmake [<options>] <path-to-source>`

Uses the current working directory as the build tree, and `<path-to-source>` as the source tree. The specified path may be absolute or relative to the current working directory. The source tree must contain a `CMakeLists.txt` file and must _not_ contain a `CMakeCache.txt` file because the latter identifies an existing build tree. For example:

```
$ mkdir build ; cd build
$ cmake ../src
```

`cmake [<options>] <path-to-existing-build>`

Uses `<path-to-existing-build>` as the build tree, and loads the path to the source tree from its `CMakeCache.txt` file, which must have already been generated by a previous run of CMake. The specified path may be absolute or relative to the current working directory. For example:

```
$ cd build
$ cmake .
```

>  仅指定一个路径时 (`cmake [<options>] <path>`)，CMake 根据 `<path>` 中是否存在 `CMakeCache.txt` 进行判断，如果有，则 `<path>` 是 `<path-to-existing-build>` ，当前目录为源树，如果没有，则 `<path>` 是 `<path-to-source>` ，当前目录为构建树

In all cases the `<options>` may be zero or more of the [Options](https://cmake.org/cmake/help/latest/manual/cmake.1.html#options) below.

The above styles for specifying the source and build trees may be mixed. Paths specified with [`-S`](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-S) or [`-B`](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-B) are always classified as source or build trees, respectively. Paths specified with plain arguments are classified based on their content and the types of paths given earlier. If only one type of path is given, the current working directory (cwd) is used for the other. For example:

|Command Line|Source Dir|Build Dir|
|---|---|---|
|`cmake -B build`|_cwd_|`build`|
|`cmake -B build src`|`src`|`build`|
|`cmake -B build -S src`|`src`|`build`|
|`cmake src`|`src`|_cwd_|
|`cmake build` (existing)|_loaded_|`build`|
|`cmake -S src`|`src`|_cwd_|
|`cmake -S src build`|`src`|`build`|
|`cmake -S src -B build`|`src`|`build`|

*Changed in version 3.23*: CMake warns when multiple source paths are specified. This has never been officially documented or supported, but older versions accidentally accepted multiple source paths and used the last path specified. Avoid passing multiple source path arguments.

After generating a buildsystem one may use the corresponding native build tool to build the project. For example, after using the [`Unix Makefiles`](https://cmake.org/cmake/help/latest/generator/Unix%20Makefiles.html#generator:Unix%20Makefiles "Unix Makefiles") generator one may run `make` directly:

```
$ make
$ make install
```

Alternatively, one may use **cmake** to [Build a Project](https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project) by automatically choosing and invoking the appropriate native build tool.

## Return Value (Exit Code)
Upon regular termination, the **cmake** executable returns the exit code `0`.

If termination is caused by the command [`message(FATAL_ERROR)`](https://cmake.org/cmake/help/latest/command/message.html#command:message "message(fatal_error)"), or another error condition, then a non-zero exit code is returned.

## See Also
The following resources are available to get help using CMake:

Home Page [https://cmake.org](https://cmake.org/)
The primary starting point for learning about CMake.

Online Documentation and Community Resources [https://cmake.org/documentation](https://cmake.org/documentation)
Links to available documentation and community resources may be found on this web page.

Discourse Forum [https://discourse.cmake.org](https://discourse.cmake.org/)
The Discourse Forum hosts discussion and questions about CMake.