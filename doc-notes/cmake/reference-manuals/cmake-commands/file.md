---
completed: 
version: 4.1.0
---
# file
File manipulation command.

This command is dedicated to file and path manipulation requiring access to the filesystem.

For other path manipulation, handling only syntactic aspects, have a look at [`cmake_path()`](https://cmake.org/cmake/help/latest/command/cmake_path.html#command:cmake_path "cmake_path") command.

Note
The sub-commands [RELATIVE_PATH](https://cmake.org/cmake/help/latest/command/file.html#relative-path), [TO_CMAKE_PATH](https://cmake.org/cmake/help/latest/command/file.html#to-cmake-path) and [TO_NATIVE_PATH](https://cmake.org/cmake/help/latest/command/file.html#to-native-path) has been superseded, respectively, by sub-commands [RELATIVE_PATH](https://cmake.org/cmake/help/latest/command/cmake_path.html#cmake-path-relative-path), [CONVERT ... TO_CMAKE_PATH_LIST](https://cmake.org/cmake/help/latest/command/cmake_path.html#cmake-path-to-cmake-path-list) and [CONVERT ... TO_NATIVE_PATH_LIST](https://cmake.org/cmake/help/latest/command/cmake_path.html#cmake-path-to-native-path-list) of [`cmake_path()`](https://cmake.org/cmake/help/latest/command/cmake_path.html#command:cmake_path "cmake_path") command.

>  `file` 用于执行各种与文件系统相关的命令，例如识别文件类型、读写文件、更改文件属性等

>  `file(GLOB <variable> ...)` 用于查找与给定模式匹配的文件，并将它们的路径存储在 CMake 变量 `variable` 中