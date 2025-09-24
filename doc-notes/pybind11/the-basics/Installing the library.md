---
completed: true
---
# Installing the library
There are several ways to get the pybind11 source, which lives at [pybind/pybind11 on GitHub](https://github.com/pybind/pybind11). The pybind11 developers recommend one of the first three ways listed here, submodule, PyPI, or conda-forge, for obtaining pybind11.

## Include as a submodule
When you are working on a project in Git, you can use the pybind11 repository as a submodule. From your git repository, use:

```
git submodule add -b stable ../../pybind/pybind11 extern/pybind11
git submodule update --init
```

>  `git submodule add` 会将子模块的配置信息写入 `.gitsubmodule`，默认只会写入 `path, url` 信息，如果添加了 `-b`，还会写入 `branch`，也就是指定了后续检出的分支不是 HEAD 而是指定分支
>  `git submodule update --init` 会先读取 `.gitsubmodule` 文件内容，将它写入本地配置文件 `.git/config` 中，完成初始化，然后再根据 `.git/config` 的内容执行实际的拉取和检出操作
>  在 `.git/config` 初始化之后，我们要再更新就直接 `git submodule update` 即可
>  如果后续修改了 `.gitsubmodule`，需要执行 `git submodule sync` 同步更新 `.git/config` 中的内容

This assumes you are placing your dependencies in `extern/`, and that you are using GitHub; if you are not using GitHub, use the full https or ssh URL instead of the relative URL `../../pybind/pybind11` above. Some other servers also require the `.git` extension (GitHub does not).

From here, you can now include `extern/pybind11/include`, or you can use the various integration tools (see [Build systems](https://pybind11.readthedocs.io/en/stable/compiling.html#compiling)) pybind11 provides directly from the local folder.
>  同步之后，我们可以直接 `include` pybind11 的 `/include` 目录下的头文件，使用它的功能

## Include with PyPI
You can download the sources and CMake files as a Python package from PyPI using Pip. Just use:

```
pip install pybind11
```

This will provide pybind11 in a standard Python package format. 

>  pybind11 也将源码和 CMake files 发布到了 PyPI，可以通过 pip 下载
>  这将以标准 Python 包格式提供 pybind11

>  可以这样使用的原因是 pybind11 是一个纯头文件库，它的所有代码都写在 `xxx.h` 这样的头文件中
>  我们通过 `pip` 将这些头文件下载到 `site-packages/` 下的 `pybind11/` 目录中，之后我们项目中的 C++ 代码就直接 `include` 这些头文件即可，在编译时，我们通过 ` CMakeLists.txt ` 指定好这些头文件的路径，就可以成功编译，得到 `.so` 或 `.pyd` 文件

>  传统的 C++ 库通常包含头文件、源文件和预编译好的二进制文件
>  我们通常会先编译这些库获得编译好的库文件，再通过静态链接或动态链接使用它们

>  静态链接发生在链接时，会将所需的库代码复制到最终可执行文件中
>  动态链接仅在文件中保存一个引用，指向所需的库文件，实际的库文件内容在运行时加载到内存


>  只要我们在源码中调用了当前文件没有定义的函数，编译器就会标记它为外部符号引用，链接器会解决这些外部符号引用，它根据命令行指定的库文件、指定的库目录 (`-L`) 和默认系统目录 (`/usr/lib`)，找到目标文件，并读取每个目标文件 (`.o, .obj `) 的符号表 (记录了该文件中定义的所有函数和变量的地址，以及它所引用的外部函数和变量的列表)，将它们汇总
>  然后链接器从中提取出我们需要的函数的代码，将它复制到最终的可执行文件中

>  相较于纯头文件式编程，使用静态链接库的优势在于模块化编译，不需要每次都重新编写别人写的代码，同时是按需取用，不需要将别人写的代码的全部都编到可执行文件中
>  实际性能不存在太大差异，我们讨论性能差异一般还是在静态链接和动态链接之间比较 (实际上动态链接的一点额外开销对于现代系统可以忽略不计了)

If you want pybind11 available directly in your environment root, you can use:

```
pip install "pybind11[global]"
```

This is not recommended if you are installing with your system Python, as it will add files to `/usr/local/include/pybind11` and `/usr/local/share/cmake/pybind11`, so unless that is what you want, it is recommended only for use in virtual environments or your `pyproject.toml` file (see [Build systems](https://pybind11.readthedocs.io/en/stable/compiling.html#compiling)).

>  如果指定了 `global`，会将 pybind11 的头文件复制到系统的全局 include 目录 `/usr/local/share/cmake/pybind11`，并且把 CMake 文件放置到系统级别的 CMake 路径 `/usr/local/shard/cmake/pybind11`
>  推荐使用虚拟环境或者在 `pyproject.toml` 中使用 (让构建系统自己处理)

>  没有指定 global 则只会放在 python 包路径 `site-packages/pybind11`，头文件和 CMake 配置都不会自动注册，此时我们需要在配置文件例如 `setup.py` 中指定

```python
# setup.py
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

ext_modules = [
    Pybind11Extension(
        "example",
        ["example.cpp"],
        include_dirs=[pybind11.get_include()],  # 关键：自动获取头文件路径
        language='c++'
    ),
]

setup(
    name="example",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)```

>  然后运行

```bash
python setup.py build_ext --inplace
```

> 此时 Python 可以导入并调用我们的 C++ 拓展模块中的内容

## Include with conda-forge
You can use pybind11 with conda packaging via [conda-forge](https://github.com/conda-forge/pybind11-feedstock):

```
conda install -c conda-forge pybind11
```

## Include with vcpkg
You can download and install pybind11 using the Microsoft [vcpkg](https://github.com/Microsoft/vcpkg/) dependency manager:

```
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
vcpkg install pybind11
```

The pybind11 port in vcpkg is kept up to date by Microsoft team members and community contributors. If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg/) on the vcpkg repository.

## Global install with brew
The brew package manager (Homebrew on macOS, or Linuxbrew on Linux) has a [pybind11 package](https://github.com/Homebrew/homebrew-core/blob/master/Formula/p/pybind11.rb). To install:

```
brew install pybind11
```

## Other options
Other locations you can find pybind11 are [listed here](https://repology.org/project/python:pybind11/versions); these are maintained by various packagers and the community.
