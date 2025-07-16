---
completed: 
version: 4.1.0
---
# add_library
Add a library to the project using the specified source files.

>  `add_library` 用于定义和创建库，库是一组编译好的代码或数据，可以被其他程序或库链接重用
>  `add_library` 会告诉 CMake 一个逻辑上的 “库” 目标，同时指定库的类型，例如静态库、共享库、模块库，并告诉 CMake 哪些源文件应该被编译以生成这个库
>  在编译时，CMake 会协调编译器和链接器将指定的源文件编译并打包成相应的库文件

## Interface Libraries

```
add_library(<name> INTERFACE)
```

Add an [Interface Library](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#interface-libraries) target that may specify usage requirements for dependents but does not compile sources and does not produce a library artifact on disk.

>  `INTERFACE` 在 CMake 中用于管理传递性依赖和使用要求
>  当指定了 `INTERFACE` 关键字之后，我们告诉 CMake 要创建一个特殊的库目标，这个库目标不会实际编译成任何库文件，而仅仅作为一个使用要求的容器存在

>  这些 “使用要求” 包括: 
>  - include 目录: 库 `<name>` 的头文件在哪里
>  - 编译定义: 使用库 `<name>` 的代码在编译时需要定义的宏
>  - 链接库: 库 `<name>` 依赖的其他库 (任何链接 `<name>` 的目标也需要这些库)
>  - 编译选项: 特定的编译器标志
>  因此，接口库只是作为一组规则和属性的容器，它自己不产生任何编译后的代码

An interface library with no source files is not included as a target in the generated buildsystem. However, it may have properties set on it and it may be installed and exported. Typically, `INTERFACE_*` properties are populated on an interface target using the commands:

- [`set_property()`](https://cmake.org/cmake/help/latest/command/set_property.html#command:set_property "set_property"),
- [`target_link_libraries(INTERFACE)`](https://cmake.org/cmake/help/latest/command/target_link_libraries.html#command:target_link_libraries "target_link_libraries(interface)"),
- [`target_link_options(INTERFACE)`](https://cmake.org/cmake/help/latest/command/target_link_options.html#command:target_link_options "target_link_options(interface)"),
- [`target_include_directories(INTERFACE)`](https://cmake.org/cmake/help/latest/command/target_include_directories.html#command:target_include_directories "target_include_directories(interface)"),
- [`target_compile_options(INTERFACE)`](https://cmake.org/cmake/help/latest/command/target_compile_options.html#command:target_compile_options "target_compile_options(interface)"),
- [`target_compile_definitions(INTERFACE)`](https://cmake.org/cmake/help/latest/command/target_compile_definitions.html#command:target_compile_definitions "target_compile_definitions(interface)"), and
- [`target_sources(INTERFACE)`](https://cmake.org/cmake/help/latest/command/target_sources.html#command:target_sources "target_sources(interface)"),

and then it is used as an argument to [`target_link_libraries()`](https://cmake.org/cmake/help/latest/command/target_link_libraries.html#command:target_link_libraries "target_link_libraries") like any other target.

Added in version 3.15: An interface library can have [`PUBLIC_HEADER`](https://cmake.org/cmake/help/latest/prop_tgt/PUBLIC_HEADER.html#prop_tgt:PUBLIC_HEADER "PUBLIC_HEADER") and [`PRIVATE_HEADER`](https://cmake.org/cmake/help/latest/prop_tgt/PRIVATE_HEADER.html#prop_tgt:PRIVATE_HEADER "PRIVATE_HEADER") properties. The headers specified by those properties can be installed using the [`install(TARGETS)`](https://cmake.org/cmake/help/latest/command/install.html#targets "install(targets)") command.

```
add_library(<name> INTERFACE [EXCLUDE_FROM_ALL] <sources>...)
```

Added in version 3.19.

Add an [Interface Library](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#interface-libraries) target with source files (in addition to usage requirements and properties as documented by the [`above signature`](https://cmake.org/cmake/help/latest/command/add_library.html#interface "add_library(interface)")). Source files may be listed directly in the `add_library` call or added later by calls to [`target_sources()`](https://cmake.org/cmake/help/latest/command/target_sources.html#command:target_sources "target_sources") with the `PRIVATE` or `PUBLIC` keywords.

If an interface library has source files (i.e. the [`SOURCES`](https://cmake.org/cmake/help/latest/prop_tgt/SOURCES.html#prop_tgt:SOURCES "SOURCES") target property is set), or header sets (i.e. the [`HEADER_SETS`](https://cmake.org/cmake/help/latest/prop_tgt/HEADER_SETS.html#prop_tgt:HEADER_SETS "HEADER_SETS") target property is set), it will appear in the generated buildsystem as a build target much like a target defined by the [`add_custom_target()`](https://cmake.org/cmake/help/latest/command/add_custom_target.html#command:add_custom_target "add_custom_target") command. It does not compile any sources, but does contain build rules for custom commands created by the [`add_custom_command()`](https://cmake.org/cmake/help/latest/command/add_custom_command.html#command:add_custom_command "add_custom_command") command.

The options are:

`EXCLUDE_FROM_ALL`

Set the [`EXCLUDE_FROM_ALL`](https://cmake.org/cmake/help/latest/prop_tgt/EXCLUDE_FROM_ALL.html#prop_tgt:EXCLUDE_FROM_ALL "EXCLUDE_FROM_ALL") target property automatically. See documentation of that target property for details.

Note
In most command signatures where the `INTERFACE` keyword appears, the items listed after it only become part of that target's usage requirements and are not part of the target's own settings. However, in this signature of `add_library`, the `INTERFACE` keyword refers to the library type only. Sources listed after it in the `add_library` call are `PRIVATE` to the interface library and do not appear in its [`INTERFACE_SOURCES`](https://cmake.org/cmake/help/latest/prop_tgt/INTERFACE_SOURCES.html#prop_tgt:INTERFACE_SOURCES "INTERFACE_SOURCES") target property.