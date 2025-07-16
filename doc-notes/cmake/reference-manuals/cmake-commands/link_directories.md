---
completed: 
version: 4.1.0
---
# link_directories
Add directories in which the linker will look for libraries.

```
link_directories([AFTER|BEFORE] directory1 [directory2 ...])
```

Adds the paths in which the linker should search for libraries. Relative paths given to this command are interpreted as relative to the current source directory, see [`CMP0015`](https://cmake.org/cmake/help/latest/policy/CMP0015.html#policy:CMP0015 "CMP0015").

>  `link_directories` 用于将指定的目录添加到链接器的库文件搜索路径列表中，这样，当我们使用 `target_link_libraries` 命令链接一个库 (例如 `mylib`) 的时候，链接器就能在指定的目录中查找这些库文件 (例如 `libmylib.a, libmylib.so`)

The command will apply only to targets created after it is called.

## See Also
- [`target_link_directories()`](https://cmake.org/cmake/help/latest/command/target_link_directories.html#command:target_link_directories "target_link_directories")
- [`target_link_libraries()`](https://cmake.org/cmake/help/latest/command/target_link_libraries.html#command:target_link_libraries "target_link_libraries")