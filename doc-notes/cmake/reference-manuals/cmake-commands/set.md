---
completed: 
version: 4.1.0
---
# set
Set a normal, cache, or environment variable to a given value. See the [cmake-language(7) variables](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cmake-language-variables) documentation for the scopes and interaction of normal variables and cache entries.

Signatures of this command that specify a `<value>...` placeholder expect zero or more arguments. Multiple arguments will be joined as a [semicolon-separated list](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cmake-language-lists) to form the actual variable value to be set.

## Set Normal Variable

```
set(<variable> <value>... [PARENT_SCOPE])
```

Set or unset `<variable>` in the current function or directory scope:

- If at least one `<value>...` is given, set the variable to that value.
- If no value is given, unset the variable. This is equivalent to [`unset(<variable>)`](https://cmake.org/cmake/help/latest/command/unset.html#command:unset "unset").

>  如果给定了值，将变量设定为该值，如果没有给定值，则等价于使用 `unset`

If the `PARENT_SCOPE` option is given the variable will be set in the scope above the current scope. Each new directory or [`function()`](https://cmake.org/cmake/help/latest/command/function.html#command:function "function") command creates a new scope. A scope can also be created with the [`block()`](https://cmake.org/cmake/help/latest/command/block.html#command:block "block") command. `set(PARENT_SCOPE)` will set the value of a variable into the parent directory, calling function, or encompassing scope (whichever is applicable to the case at hand). The previous state of the variable's value stays the same in the current scope (e.g., if it was undefined before, it is still undefined and if it had a value, it is still that value).

The [`block(PROPAGATE)`](https://cmake.org/cmake/help/latest/command/block.html#command:block "block(propagate)") and [`return(PROPAGATE)`](https://cmake.org/cmake/help/latest/command/return.html#command:return "return(propagate)") commands can be used as an alternate method to the [`set(PARENT_SCOPE)`](https://cmake.org/cmake/help/latest/command/set.html#command:set "set(parent_scope)") and [`unset(PARENT_SCOPE)`](https://cmake.org/cmake/help/latest/command/unset.html#command:unset "unset(parent_scope)") commands to update the parent scope.

Note
When evaluating [Variable References](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#variable-references) of the form `${VAR}`, CMake first searches for a normal variable with that name. If no such normal variable exists, CMake will then search for a cache entry with that name. Because of this, **unsetting a normal variable can expose a cache variable that was previously hidden**. To force a variable reference of the form `${VAR}` to return an empty string, use `set(<variable> "")`, which clears the normal variable but leaves it defined.