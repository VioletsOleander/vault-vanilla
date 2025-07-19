---
completed: 
version: 4.1.0
---
# aux_source_directory
Find all source files in a directory.

```
aux_source_directory(<dir> <variable>)
```

Collects the names of all the source files in the specified directory and stores the list in the `<variable>` provided. This command is intended to be used by projects that use explicit template instantiation. Template instantiation files can be stored in a `Templates` subdirectory and collected automatically using this command to avoid manually listing all instantiations.

It is tempting to use this command to avoid writing the list of source files for a library or executable target. While this seems to work, there is no way for CMake to generate a build system that knows when a new source file has been added. Normally the generated build system knows when it needs to rerun CMake because the `CMakeLists.txt` file is modified to add a new source. When the source is just added to the directory without modifying this file, one would have to manually rerun CMake to generate a build system incorporating the new file.

>  `aux_source_directory` 用于收集指定目录下的所有源文件，并将它们存储在一个变量中
>  这个命令的初衷是为显示模板实例化等场景提供便利，例如，如果我们有一个 `Templates` 子目录，里面存放了大量模板实例化文件，就可以使用 `aux_source_directory` 收集它们，而无需手动列出它们

>  CMake 推荐的做法是尽量显示列出源文件而不是使用 `aux_source_directory`