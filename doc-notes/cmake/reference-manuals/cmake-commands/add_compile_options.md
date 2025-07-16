---
completed: 
version: 4.1.0
---
# add_compile_options
Add options to the compilation of source files.

```
add_compile_options(<option> ...)
```

Adds options to the [`COMPILE_OPTIONS`](https://cmake.org/cmake/help/latest/prop_dir/COMPILE_OPTIONS.html#prop_dir:COMPILE_OPTIONS "COMPILE_OPTIONS") directory property. These options are used when compiling targets from the current directory and below.

>  `add_compile_options` 用于向 CMake 生成的构建系统中所有或部分的编译命令添加自定义的编译器选项，其中 `<options>` 表示编译器选项，通常以 `-` 开头，例如 `-Wall, -O2, -std=c++11` 等

Note
These options are not used when linking. See the [`add_link_options()`](https://cmake.org/cmake/help/latest/command/add_link_options.html#command:add_link_options "add_link_options") command for that.