---
completed:
---
# `pathlib` — Object-oriented filesystem paths
Added in version 3.4.

**Source code:** [Lib/pathlib/](https://github.com/python/cpython/tree/3.13/Lib/pathlib/)

---

This module offers classes representing filesystem paths with semantics appropriate for different operating systems. Path classes are divided between [pure paths](https://docs.python.org/3.13/library/pathlib.html#pure-paths), which provide purely computational operations without I/O, and [concrete paths](https://docs.python.org/3.13/library/pathlib.html#concrete-paths), which inherit from pure paths but also provide I/O operations.
>  `pathlib` 模块提供了用于表示适合于不同 OS 下文件路径的类
>  路径类分为纯路径和具体路径，纯路径仅提供计算操作，没有 IO 操作，具体路径继承纯路径，同时提供 IO 操作

![Inheritance diagram showing the classes available in pathlib. The most basic class is PurePath, which has three direct subclasses: PurePosixPath, PureWindowsPath, and Path. Further to these four classes, there are two classes that use multiple inheritance: PosixPath subclasses PurePosixPath and Path, and WindowsPath subclasses PureWindowsPath and Path.](https://docs.python.org/3.13/_images/pathlib-inheritance.png)

If you’ve never used this module before or just aren’t sure which class is right for your task, [`Path`](https://docs.python.org/3.13/library/pathlib.html#pathlib.Path "pathlib.Path") is most likely what you need. It instantiates a [concrete path](https://docs.python.org/3.13/library/pathlib.html#concrete-paths) for the platform the code is running on.
>  通常使用 `Path` 类即可，它会根据代码运行的平台实例化一个具体路径

Pure paths are useful in some special cases; for example:

1. If you want to manipulate Windows paths on a Unix machine (or vice versa). You cannot instantiate a [`WindowsPath`](https://docs.python.org/3.13/library/pathlib.html#pathlib.WindowsPath "pathlib.WindowsPath") when running on Unix, but you can instantiate [`PureWindowsPath`](https://docs.python.org/3.13/library/pathlib.html#pathlib.PureWindowsPath "pathlib.PureWindowsPath").
2. You want to make sure that your code only manipulates paths without actually accessing the OS. In this case, instantiating one of the pure classes may be useful since those simply don’t have any OS-accessing operations.

See also [**PEP 428**](https://peps.python.org/pep-0428/): The pathlib module – object-oriented filesystem paths.

See also For low-level path manipulation on strings, you can also use the [`os.path`](https://docs.python.org/3.13/library/os.path.html#module-os.path "os.path: Operations on pathnames.") module.

## Basic use
Importing the main class:

```python
>>> from pathlib import Path
```

Listing subdirectories:

```python
>>>  p = Path('.')
>>>  [x for x in p.iterdir() if x.is_dir()]
[PosixPath('.hg'), PosixPath('docs'), PosixPath('dist'),
 PosixPath('__pycache__'), PosixPath('build')]
```

Listing Python source files in this directory tree:

```python
>>>  list(p.glob('**/*.py'))
[PosixPath('test_pathlib.py'), PosixPath('setup.py'),
 PosixPath('pathlib.py'), PosixPath('docs/conf.py'),
 PosixPath('build/lib/pathlib.py')]
```

Navigating inside a directory tree:

```python
>>>  p = Path('/etc')
>>>  q = p / 'init.d' / 'reboot'
>>>  q
PosixPath('/etc/init.d/reboot')
>>>  q.resolve()
PosixPath('/etc/rc.d/init.d/halt')
```

Querying path properties:

```python
>>>  q.exists()
True
>>>  q.is_dir()
False
```

Opening a file:

```python
>>>  with q.open() as f: f.readline()
...
'#!/bin/bash\n'
```

## Exceptions
_exception_ pathlib.UnsupportedOperation

An exception inheriting [`NotImplementedError`](https://docs.python.org/3.13/library/exceptions.html#NotImplementedError "NotImplementedError") that is raised when an unsupported operation is called on a path object.

>  Added in version 3.13.
