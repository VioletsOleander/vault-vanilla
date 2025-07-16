---
version: 4.0.1
---
# cmake_parse_arguments
Parse function or macro arguments.

```cmake
cmake_parse_arguments(<prefix> <options> <one_value_keywords>
                      <multi_value_keywords> <args>...)

cmake_parse_arguments(PARSE_ARGV <N> <prefix> <options>
                      <one_value_keywords> <multi_value_keywords>)
```

*Added in version 3.5*: This command is implemented natively. Previously, it has been defined in the module [`CMakeParseArguments`](https://cmake.org/cmake/help/latest/module/CMakeParseArguments.html#module:CMakeParseArguments "CMakeParseArguments").
>  3.5 版本后，该命令成为内建命令

This command is for use in macros or functions. It processes the arguments given to that macro or function, and defines a set of variables which hold the values of the respective options.
>  该命令解析传递给宏或函数的参数，并定义一组变量存储相应选项的值

The first signature reads arguments passed in the `<args>...`. This may be used in either a [`macro()`](https://cmake.org/cmake/help/latest/command/macro.html#command:macro "macro") or a [`function()`](https://cmake.org/cmake/help/latest/command/function.html#command:function "function").
>  第一个 signature 读取通过 `<args> ...` 传递的参数，宏和函数中都可以调用第一个 signature

*Added in version 3.7*: The `PARSE_ARGV` signature is only for use in a [`function()`](https://cmake.org/cmake/help/latest/command/function.html#command:function "function") body. In this case, the arguments that are parsed come from the `ARGV#` variables of the calling function. The parsing starts with the `<N>`-th argument, where `<N>` is an unsigned integer. This allows for the values to have special characters like `;` in them.

The `<options>` argument contains all options for the respective function or macro. These are keywords that have no value following them, like the `OPTIONAL` keyword of the [`install()`](https://cmake.org/cmake/help/latest/command/install.html#command:install "install") command.
>  `<options>` 参数包含对应函数或宏的所有选项，这些选项是后面没有跟随值的关键字，例如 `install()` 命令中的 `OPTIONAL` 关键字

The `<one_value_keywords>` argument contains all keywords for this function or macro which are followed by one value, like the `DESTINATION` keyword of the [`install()`](https://cmake.org/cmake/help/latest/command/install.html#command:install "install") command.
>  `<one_value_keywords>` 参数包含对应函数或宏的所有后面跟随一个值的关键字，例如 `install()` 命令中的 `DESTINATION` 关键字

The `<multi_value_keywords>` argument contains all keywords for this function or macro which can be followed by more than one value, like the `TARGETS` or `FILES` keywords of the [`install()`](https://cmake.org/cmake/help/latest/command/install.html#command:install "install") command.
>  `<multi_value_keywords>` 参数包含对应函数或宏的所有后面跟随多个值的关键字，例如 `install()` 命令中的 `TARGETS, FILES` 关键字

*Changed in version 3.5*: All keywords must be unique. Each keyword can only be specified once in any of the `<options>`, `<one_value_keywords>`, or `<multi_value_keywords>`. A warning will be emitted if uniqueness is violated.

When done, `cmake_parse_arguments` will consider for each of the keywords listed in `<options>`, `<one_value_keywords>`, and `<multi_value_keywords>`, a variable composed of the given `<prefix>` followed by `"_"` and the name of the respective keyword. For `<one_value_keywords>` and `<multi_value_keywords>`, these variables will then hold the respective value(s) from the argument list, or be undefined if the associated keyword was not given (policy [`CMP0174`](https://cmake.org/cmake/help/latest/policy/CMP0174.html#policy:CMP0174 "CMP0174") can also affect the behavior for `<one_value_keywords>`). For the `<options>` keywords, these variables will always be defined, and they will be set to `TRUE` if the keyword is present, or `FALSE` if it is not.
>  `cmake_parse_arguments` 在解析完 `<args> ...` 后，会针对 `<options>, <one_value_keywords>, <multi_value_keywords>` 中所有的关键字，考虑一个名称为 `<prefix>_` + 相应关键字名的变量
>  对于 `<one_value_keywords>, <multi_value_keywords>` ，这些变量将保存来自参数列表相应的值，如果参数列表中没有给定对应的关键字，则变量未定义
>  对于 `<options>` 关键字，这些变量将永远有定义，如果参数列表中出现了对应关键字，则值为 `TRUE`，否则为 `FALSE`

All remaining arguments are collected in a variable `<prefix>_UNPARSED_ARGUMENTS` that will be undefined if all arguments were recognized. This can be checked afterwards to see whether your macro or function was called with unrecognized parameters.
>  没有对应上的参数会被收集为 `<prefix>_UNPARSED_ARGUMENTS`，如果所有传入的参数都被识别 (都对应上) ，则 `<prefix>_UNPRASED_ARGUMENTS` 将未定义

*Added in version 3.15*: `<one_value_keywords>` and `<multi_value_keywords>` that were given no values at all are collected in a variable `<prefix>_KEYWORDS_MISSING_VALUES` that will be undefined if all keywords received values. This can be checked to see if there were keywords without any values given.

*Changed in version 3.31*: If a `<one_value_keyword>` is followed by an empty string as its value, policy [`CMP0174`](https://cmake.org/cmake/help/latest/policy/CMP0174.html#policy:CMP0174 "CMP0174") controls whether a corresponding `<prefix>_<keyword>` variable is defined or not.

Choose a `<prefix>` carefully to avoid clashing with existing variable names. When used inside a function, it is usually suitable to use the prefix `arg`. There is a very strong convention that all keywords are fully uppercase, so this prefix results in variables of the form `arg_SOME_KEYWORD`. This makes the code more readable, and it minimizes the chance of clashing with cache variables, which also have a strong convention of being all uppercase.
>  在函数中，`<prefix>` 通常用 `arg`
>  所有关键字全部大写，故最后得到的变量一般为 `arg_SOME_KEYWORD` 的形式

```cmake
function(my_install)
    set(options OPTIONAL FAST)
    set(oneValueArgs DESTINATION RENAME)
    set(multiValueArgs TARGETS CONFIGURATIONS)
    cmake_parse_arguments(PARSE_ARGV 0 arg
        "${options}" "${oneValueArgs}" "${multiValueArgs}"
    )

    # The above will set or unset variables with the following names:
    #   arg_OPTIONAL
    #   arg_FAST
    #   arg_DESTINATION
    #   arg_RENAME
    #   arg_TARGETS
    #   arg_CONFIGURATIONS
    #
    # The following will also be set or unset:
    #   arg_UNPARSED_ARGUMENTS
    #   arg_KEYWORDS_MISSING_VALUES
```

When used inside a macro, `arg` might not be a suitable prefix because the code will affect the calling scope. If another macro also called in the same scope were to use `arg` in its own call to `cmake_parse_arguments()`, and if there are any common keywords between the two macros, the later call's variables can overwrite or remove those of the earlier macro's call. Therefore, it is advisable to incorporate something unique from the macro name in the `<prefix>`, such as `arg_lowercase_macro_name`.

```
macro(my_install)
    set(options OPTIONAL FAST)
    set(oneValueArgs DESTINATION RENAME)
    set(multiValueArgs TARGETS CONFIGURATIONS)
    cmake_parse_arguments(arg_my_install
        "${options}" "${oneValueArgs}" "${multiValueArgs}"
        ${ARGN}
    )
    # ...
endmacro()

macro(my_special_install)
    # NOTE: Has the same keywords as my_install()
    set(options OPTIONAL FAST)
    set(oneValueArgs DESTINATION RENAME)
    set(multiValueArgs TARGETS CONFIGURATIONS)
    cmake_parse_arguments(arg_my_special_install
        "${options}" "${oneValueArgs}" "${multiValueArgs}"
        ${ARGN}
    )
    # ...
endmacro()
```

Suppose the above macros are called one after the other, like so:

```
my_install(TARGETS foo bar DESTINATION bin OPTIONAL blub CONFIGURATIONS)
my_special_install(TARGETS barry DESTINATION sbin RENAME FAST)
```

After these two calls, the following describes the variables that will be set or unset:

```
arg_my_install_OPTIONAL = TRUE
arg_my_install_FAST = FALSE # was not present in call to my_install
arg_my_install_DESTINATION = "bin"
arg_my_install_RENAME <UNSET> # was not present
arg_my_install_TARGETS = "foo;bar"
arg_my_install_CONFIGURATIONS <UNSET> # was not present
arg_my_install_UNPARSED_ARGUMENTS = "blub" # nothing expected after "OPTIONAL"
arg_my_install_KEYWORDS_MISSING_VALUES = "CONFIGURATIONS" # value was missing

arg_my_special_install_OPTIONAL = FALSE # was not present
arg_my_special_install_FAST = TRUE
arg_my_special_install_DESTINATION = "sbin"
arg_my_special_install_RENAME <UNSET> # value was missing
arg_my_special_install_TARGETS = "barry"
arg_my_special_install_CONFIGURATIONS <UNSET> # was not present
arg_my_special_install_UNPARSED_ARGUMENTS <UNSET>
arg_my_special_install_KEYWORDS_MISSING_VALUES = "RENAME"
```

Keywords terminate lists of values. If a keyword is given directly after a `<one_value_keyword>`, that preceding `<one_value_keyword>` receives no value and the keyword is added to the `<prefix>_KEYWORDS_MISSING_VALUES` variable. In the above example, the call to `my_special_install()` contains the `RENAME` keyword immediately followed by the `FAST` keyword. In this case, `FAST` terminates processing of the `RENAME` keyword. `arg_my_special_install_FAST` is set to `TRUE`, `arg_my_special_install_RENAME` is unset, and `arg_my_special_install_KEYWORDS_MISSING_VALUES` contains the value `RENAME`.

## See Also
- [`function()`](https://cmake.org/cmake/help/latest/command/function.html#command:function "function")
- [`macro()`](https://cmake.org/cmake/help/latest/command/macro.html#command:macro "macro")
