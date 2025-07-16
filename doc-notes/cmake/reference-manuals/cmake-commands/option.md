---
completed: true
version: 4.1.0
---
# option
Provide a boolean option that the user can optionally select.

```
option(<variable> "<help_text>" [value])
```

If no initial `<value>` is provided, boolean `OFF` is the default value. If `<variable>` is already set as a normal or cache variable, then the command does nothing (see policy [`CMP0077`](https://cmake.org/cmake/help/latest/policy/CMP0077.html#policy:CMP0077 "CMP0077")).

In CMake project mode, a boolean cache variable is created with the option value. In CMake script mode, a boolean variable is set with the option value.

>  `option` 命令用于为用户提供一个布尔选项，如果没有设定默认值，默认值就是 `OFF`
>  之后在命令行中，可以用 `-D` 指定变量的值，例如 `-D<variable>=ON`

## See Also
- The [`CMakeDependentOption`](https://cmake.org/cmake/help/latest/module/CMakeDependentOption.html#module:CMakeDependentOption "CMakeDependentOption") module to specify boolean options that depend on the values of other options or a set of conditions.