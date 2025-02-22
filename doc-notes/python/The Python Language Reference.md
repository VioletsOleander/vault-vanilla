---
completed:
---
# 5. The import system
>  version: 3.13.2

Python code in one [module](https://docs.python.org/3/glossary.html#term-module) gains access to the code in another module by the process of [importing](https://docs.python.org/3/glossary.html#term-importing) it. The [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement is the most common way of invoking the import machinery, but it is not the only way. Functions such as [`importlib.import_module()`](https://docs.python.org/3/library/importlib.html#importlib.import_module "importlib.import_module") and built-in [`__import__()`](https://docs.python.org/3/library/functions.html#import__ "__import__") can also be used to invoke the import machinery.
>  在一个模块中的 Python 代码通过导入机制访问其他模块中的 Python 代码，`import` 语句是调用导入机制最直接的方式，但不是唯一的方式，函数 `importlib.import_module()` 和内建函数 `__import__()` 也可以用于调用导入机制

The [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement combines two operations; it searches for the named module, then it binds the results of that search to a name in the local scope. The search operation of the `import` statement is defined as a call to the [`__import__()`](https://docs.python.org/3/library/functions.html#import__ "__import__") function, with the appropriate arguments. The return value of [`__import__()`](https://docs.python.org/3/library/functions.html#import__ "__import__") is used to perform the name binding operation of the `import` statement. See the `import` statement for the exact details of that name binding operation.
>  `import` 语句包含了两个操作：搜索命名模块、将搜索结果绑定到本地作用域的一个名字
>  `import` 语句的搜索操作即以合适的参数调用 `__import__()` 函数，其返回值会用于执行名字绑定操作

A direct call to [`__import__()`](https://docs.python.org/3/library/functions.html#import__ "__import__") performs only the module search and, if found, the module creation operation. While certain side-effects may occur, such as the importing of parent packages, and the updating of various caches (including [`sys.modules`](https://docs.python.org/3/library/sys.html#sys.modules "sys.modules")), only the [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement performs a name binding operation.
>  直接调用 `__import__()` 仅执行模块搜索，如果找到了模块，会执行模块创建操作，同时存在一些副作用
>  只有 `import` 语句会执行名字绑定操作

When an [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement is executed, the standard builtin [`__import__()`](https://docs.python.org/3/library/functions.html#import__ "__import__") function is called. Other mechanisms for invoking the import system (such as [`importlib.import_module()`](https://docs.python.org/3/library/importlib.html#importlib.import_module "importlib.import_module")) may choose to bypass [`__import__()`](https://docs.python.org/3/library/functions.html#import__ "__import__") and use their own solutions to implement import semantics.

When a module is first imported, Python searches for the module and if found, it creates a module object [1](https://docs.python.org/3/reference/import.html?spm=5176.28103460.0.0.297c5d27Z86TXs#fnmo), initializing it. If the named module cannot be found, a [`ModuleNotFoundError`](https://docs.python.org/3/library/exceptions.html#ModuleNotFoundError "ModuleNotFoundError") is raised. Python implements various strategies to search for the named module when the import machinery is invoked. These strategies can be modified and extended by using various hooks described in the sections below.
>  导入一个模块时，Python 首先进行搜索，如果找到，会创建一个模块对象并初始化它，如果找不到，抛出 `ModuleNotFoundError`

Changed in version 3.3: The import system has been updated to fully implement the second phase of [**PEP 302**](https://peps.python.org/pep-0302/). There is no longer any implicit import machinery - the full import system is exposed through [`sys.meta_path`](https://docs.python.org/3/library/sys.html#sys.meta_path "sys.meta_path"). In addition, native namespace package support has been implemented (see [**PEP 420**](https://peps.python.org/pep-0420/)).

## 5.1. `importlib`
The [`importlib`](https://docs.python.org/3/library/importlib.html#module-importlib "importlib: The implementation of the import machinery.") module provides a rich API for interacting with the import system. For example [`importlib.import_module()`](https://docs.python.org/3/library/importlib.html#importlib.import_module "importlib.import_module") provides a recommended, simpler API than built-in [`__import__()`](https://docs.python.org/3/library/functions.html#import__ "__import__") for invoking the import machinery. Refer to the [`importlib`](https://docs.python.org/3/library/importlib.html#module-importlib "importlib: The implementation of the import machinery.") library documentation for additional detail.
>  `importlib` 模块提供了和导入系统交互的 API

## 5.2. Packages
Python has only one type of module object, and all modules are of this type, regardless of whether the module is implemented in Python, C, or something else. To help organize modules and provide a naming hierarchy, Python has a concept of [packages](https://docs.python.org/3/glossary.html#term-package).
>  Python 只有一种类型的模块对象，所有模块都是该类型
>  为了帮助组织模块层次，Python 引入了包的概念

You can think of packages as the directories on a file system and modules as files within directories, but don’t take this analogy too literally since packages and modules need not originate from the file system. For the purposes of this documentation, we’ll use this convenient analogy of directories and files. Like file system directories, packages are organized hierarchically, and packages may themselves contain subpackages, as well as regular modules.
>  包可以比作文件系统的目录，模块是目录下的文件
>  包也可以包含子包

It’s important to keep in mind that all packages are modules, but not all modules are packages. Or put another way, packages are just a special kind of module. Specifically, any module that contains a `__path__` attribute is considered a package.
>  所有的包都是模块，但不是所有的模块都是包
>  或者说，包是特殊的一类模块，任意包含了 `__path__` 属性的模块都认为是包

All modules have a name. Subpackage names are separated from their parent package name by a dot, akin to Python’s standard attribute access syntax. Thus you might have a package called [`email`](https://docs.python.org/3/library/email.html#module-email "email: Package supporting the parsing, manipulating, and generating email messages."), which in turn has a subpackage called [`email.mime`](https://docs.python.org/3/library/email.mime.html#module-email.mime "email.mime: Build MIME messages.") and a module within that subpackage called [`email.mime.text`](https://docs.python.org/3/library/email.mime.html#module-email.mime.text "email.mime.text").
>  所有模块都有名字，名字的定义实例如上

### 5.2.1. Regular packages
Python defines two types of packages, [regular packages](https://docs.python.org/3/glossary.html#term-regular-package) and [namespace packages](https://docs.python.org/3/glossary.html#term-namespace-package). Regular packages are traditional packages as they existed in Python 3.2 and earlier. A regular package is typically implemented as a directory containing an `__init__.py` file. When a regular package is imported, this `__init__.py` file is implicitly executed, and the objects it defines are bound to names in the package’s namespace. The `__init__.py` file can contain the same Python code that any other module can contain, and Python will add some additional attributes to the module when it is imported.
>  Python 定义了两类包：常规包和命名空间包
>  常规包通过包含 `__init__.py` 的目录实现，导入一个常规包时，其 `__init__.py` 会被隐式执行，`__init__.py` 中定义的对象也会绑定到包的命名空间中的名字
>  `__init__.py` 模块可以包含任意常规 Python 代码，且 Python 会在导入时为它添加一些额外属性

For example, the following file system layout defines a top level `parent` package with three subpackages:

```
parent/
    __init__.py
    one/
        __init__.py
    two/
        __init__.py
    three/
        __init__.py
```

Importing `parent.one` will implicitly execute `parent/__init__.py` and `parent/one/__init__.py`. Subsequent imports of `parent.two` or `parent.three` will execute `parent/two/__init__.py` and `parent/three/__init__.py` respectively.

### 5.2.2. Namespace packages
A namespace package is a composite of various [portions](https://docs.python.org/3/glossary.html#term-portion), where each portion contributes a subpackage to the parent package. Portions may reside in different locations on the file system. Portions may also be found in zip files, on the network, or anywhere else that Python searches during import. Namespace packages may or may not correspond directly to objects on the file system; they may be virtual modules that have no concrete representation.
>  命名空间包是由多个部分 (portion) 组成的，每个部分为父包贡献一个子包
>  部分可以位于文件系统的不同位置，部分可以在 zip 文件中、网络中、或者任意 Python 在 import 时搜索的位置找到
>  命名空间包可能或可能不直接对应于文件系统上的对象；它们可能是没有具体表示形式的虚拟模块

Namespace packages do not use an ordinary list for their `__path__` attribute. They instead use a custom iterable type which will automatically perform a new search for package portions on the next import attempt within that package if the path of their parent package (or [`sys.path`](https://docs.python.org/3/library/sys.html#sys.path "sys.path") for a top level package) changes.
>  命名空间包的 `__path__` 属性不使用普通列表，而是使用一个自定义的可迭代类型
>  如果命名空间包的父包的路径 (对于顶级包就是 `sys.path` ) 变化了，在该命名空间包内的下一次导入尝试时，该命名空间的自定义可迭代类型将自动对包的部分重新搜索

With namespace packages, there is no `parent/__init__.py` file. In fact, there may be multiple `parent` directories found during import search, where each one is provided by a different portion. Thus `parent/one` may not be physically located next to `parent/two`. In this case, Python will create a namespace package for the top-level `parent` package whenever it or one of its subpackages is imported.
>  命名空间包中没有 `parent/__init__.py` ，在导入过程中可能会搜索到多个 `parent` 目录，每个目录由不同的部分提供
>  因此，`parent/one` 可能不会物理上在 `parent/two` 旁边，此时，顶级 `parent` 包或者其任何子包被导入时，Python 都会为顶级 `parent` 包创建一个命名空间包

See also [**PEP 420**](https://peps.python.org/pep-0420/) for the namespace package specification.

## 5.9. References
The import machinery has evolved considerably since Python’s early days. The original [specification for packages](https://www.python.org/doc/essays/packages/) is still available to read, although some details have changed since the writing of that document.

The original specification for [`sys.meta_path`](https://docs.python.org/3/library/sys.html#sys.meta_path "sys.meta_path") was [**PEP 302**](https://peps.python.org/pep-0302/), with subsequent extension in [**PEP 420**](https://peps.python.org/pep-0420/).

[**PEP 420**](https://peps.python.org/pep-0420/) introduced [namespace packages](https://docs.python.org/3/glossary.html#term-namespace-package) for Python 3.3. [**PEP 420**](https://peps.python.org/pep-0420/) also introduced the `find_loader()` protocol as an alternative to `find_module()`.

[**PEP 366**](https://peps.python.org/pep-0366/) describes the addition of the `__package__` attribute for explicit relative imports in main modules.

[**PEP 328**](https://peps.python.org/pep-0328/) introduced absolute and explicit relative imports and initially proposed `__name__` for semantics [**PEP 366**](https://peps.python.org/pep-0366/) would eventually specify for `__package__`.

[**PEP 338**](https://peps.python.org/pep-0338/) defines executing modules as scripts.

[**PEP 451**](https://peps.python.org/pep-0451/) adds the encapsulation of per-module import state in spec objects. It also off-loads most of the boilerplate responsibilities of loaders back onto the import machinery. These changes allow the deprecation of several APIs in the import system and also addition of new methods to finders and loaders.

Footnotes

[1](https://docs.python.org/3/reference/import.html#id1) See [`types.ModuleType`](https://docs.python.org/3/library/types.html#types.ModuleType "types.ModuleType").

[2](https://docs.python.org/3/reference/import.html#id3) The importlib implementation avoids using the return value directly. Instead, it gets the module object by looking the module name up in [`sys.modules`](https://docs.python.org/3/library/sys.html#sys.modules "sys.modules"). The indirect effect of this is that an imported module may replace itself in [`sys.modules`](https://docs.python.org/3/library/sys.html#sys.modules "sys.modules"). This is implementation-specific behavior that is not guaranteed to work in other Python implementations.
