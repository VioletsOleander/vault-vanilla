---
completed: 
version: 4.1.0
---
# target_link_libraries
Specify libraries or flags to use when linking a given target and/or its dependents. [Usage requirements](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#target-usage-requirements) from linked library targets will be propagated. Usage requirements of a target's dependencies affect compilation of its own sources.

>  `target_link_libraries` 用于指定一个目标需要链接哪些其他库，它告诉链接器在构建某个目标时需要包含哪些库，这些库可以是项目内部构建的库，也可以是系统库或者是 `find_package` 找到的第三方库
>  `target_link_libraries` 还会根据链接方式 (`PRIVATE, PUBLID, INTERFACE`) 自动处理头文件路径、编译定义等信息，并将这些信息传递给更高层的依赖者

## Overview
This command has several signatures as detailed in subsections below. All of them have the general form

```
target_link_libraries(<target> ... <item>... ...)
```

The named `<target>` must have been created by a command such as [`add_executable()`](https://cmake.org/cmake/help/latest/command/add_executable.html#command:add_executable "add_executable") or [`add_library()`](https://cmake.org/cmake/help/latest/command/add_library.html#command:add_library "add_library") and must not be an [ALIAS target](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#alias-targets). If policy [`CMP0079`](https://cmake.org/cmake/help/latest/policy/CMP0079.html#policy:CMP0079 "CMP0079") is not set to `NEW` then the target must have been created in the current directory. Repeated calls for the same `<target>` append items in the order called.

>  `<target>` 是需要配置的 CMake 目标名称，这个目标必须已经通过 `add_executable` 或 `add_library` 定义过
>  `INTREFACE` 表示 `<target>` 内部不直接依赖 `<item>`，但 `target` 的消费者 (链接到 `<target>` 的其他目标) 需要链接 `<item>`