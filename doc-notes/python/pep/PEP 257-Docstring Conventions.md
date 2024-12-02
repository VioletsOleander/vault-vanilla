## Abstract
This PEP documents the semantics and conventions associated with Python docstrings.

## Rationale
The aim of this PEP is to standardize the high-level structure of docstrings: what they should contain, and how to say it (without touching on any markup syntax within docstrings). The PEP contains conventions, not laws or syntax.

> “A universal convention supplies all of maintainability, clarity, consistency, and a foundation for good programming habits too. What it doesn’t do is insist that you follow it against your will. That’s Python!”
> 
> —Tim Peters on comp.lang.python, 2001-06-16

If you violate these conventions, the worst you’ll get is some dirty looks. But some software (such as the [Docutils](https://docutils.sourceforge.io/) docstring processing system [PEP 256](https://peps.python.org/pep-0256/ "PEP 256 – Docstring Processing System Framework"), [PEP 258](https://peps.python.org/pep-0258/ "PEP 258 – Docutils Design Specification")) will be aware of the conventions, so following them will get you the best results.

## Specification
### What is a Docstring?
A docstring is a string literal that occurs as the first statement in a module, function, class, or method definition. Such a docstring becomes the `__doc__` special attribute of that object.
>  docstring 为作为模块、函数、类、方法定义的第一个语句的字符串字面值
>  该 docstring 会作为对象的 `__doc__` 属性的值

All modules should normally have docstrings, and all functions and classes exported by a module should also have docstrings. Public methods (including the `__init__` constructor) should also have docstrings. A package may be documented in the module docstring of the `__init__.py` file in the package directory.
>  所有的模块都要求有 docstring，模块导出的所有函数和类都要求有 docstring
>  类的公有方法要求有 docstring ( 包括 `__init__` )
>  包的 docstring 可以写在模块的 `__init__.py` 文件中

String literals occurring elsewhere in Python code may also act as documentation. They are not recognized by the Python bytecode compiler and are not accessible as runtime object attributes (i.e. not assigned to `__doc__`), but two types of extra docstrings may be extracted by software tools:

1. String literals occurring immediately after a simple assignment at the top level of a module, class, or `__init__` method are called “attribute docstrings”.
2. String literals occurring immediately after another docstring are called “additional docstrings”.

>  除了模块、函数、类、方法的第一行，出现在 Python 代码中其他地方的字符串字面值也可以作为 docstring，包括以下两类 ( 注意 Python bytecode 编译器不会识别它们，故无法通过 `__doc__` 属性访问它们，它们将由其他软件工具提取 )：
>  - 紧随着模块、类、`__init__` 方法中的顶级赋值语句之后的字符串字面值，它们称为 “属性 docstring”
>  - 紧随着另一个 docstring 的字符串字面值，它们称为 “额外 docstring”

Please see [PEP 258](https://peps.python.org/pep-0258/ "PEP 258 – Docutils Design Specification"), “Docutils Design Specification”, for a detailed description of attribute and additional docstrings.

For consistency, always use `"""triple double quotes"""` around docstrings. Use `r"""raw triple double quotes"""` if you use any backslashes in your docstrings.
>  建议 docstring 都使用 `"""` 包围，如果 docstring 中存在反斜杠，使用 `r"""..."""`

There are two forms of docstrings: one-liners and multi-line docstrings.
>  docstring 分为两类：单行、多行

### One-line Docstrings
One-liners are for really obvious cases. They should really fit on one line. For example:

```python
def kos_root():
    """Return the pathname of the KOS root directory."""
    global _kos_root
    if _kos_root: return _kos_root
    ...
```

Notes:

- Triple quotes are used even though the string fits on one line. This makes it easy to later expand it.
- The closing quotes are on the same line as the opening quotes. This looks better for one-liners.
- There’s no blank line either before or after the docstring.
- The docstring is a phrase ending in a period. It prescribes the function or method’s effect as a command (“Do this”, “Return that”), not as a description; e.g. don’t write “Returns the pathname …”.
- The one-line docstring should NOT be a “signature” reiterating the function/method parameters (which can be obtained by introspection). Don’t do:

    ```python
    def function(a, b):
        """function(a, b) -> list"""
    ```
    
    This type of docstring is only appropriate for C functions (such as built-ins), where introspection is not possible. However, the nature of the _return value_ cannot be determined by introspection, so it should be mentioned. The preferred form for such a docstring would be something like:
    ```python
    def function(a, b):
        """Do X and return a list."""
    ```     
    
    (Of course “Do X” should be replaced by a useful description!)

>  一行能解释的简单情况就使用单行 docstring
>  注意：
>  - 单行的 docstring 仍然使用 `"""` ，同时 opening `"""` 和 closing `"""` 保持在同一行
>  - docstring 上下没有空行
>  - docstring 为以 `.` 结尾的短语，以命令的语气规定函数或方法的效果，而不是描述的语气 (祈使句)
>  - dosctring 不应该是重新列举函数/方法的参数的签名，函数/方法的参数应该通过 introspection 获取，这类 docstring 应该仅用于 C 函数 ( 因为 C 函数无法通过 introspection 获取参数 )。注意 introspection 无法获取返回值的类型，因此 docstring 中应该提及这一点

### Multi-line Docstrings
Multi-line docstrings consist of a summary line just like a one-line docstring, followed by a blank line, followed by a more elaborate description. The summary line may be used by automatic indexing tools; it is important that it fits on one line and is separated from the rest of the docstring by a blank line. The summary line may be on the same line as the opening quotes or on the next line. The entire docstring is indented the same as the quotes at its first line (see example below).
>  多行 docstring 第一行为总结，之后跟一个空行，然后是详细描述
>  注意总结行的长度必须在一行以内
>  总结行可以和 opening `"""` 在同一行，也可以在下一行

Insert a blank line after all docstrings (one-line or multi-line) that document a class – generally speaking, the class’s methods are separated from each other by a single blank line, and the docstring needs to be offset from the first method by a blank line.
>  类的 docstring 后面应该有一个空行

The docstring of a script (a stand-alone program) should be usable as its “usage” message, printed when the script is invoked with incorrect or missing arguments (or perhaps with a “-h” option, for “help”). Such a docstring should document the script’s function and command line syntax, environment variables, and files. Usage messages can be fairly elaborate (several screens full) and should be sufficient for a new user to use the command properly, as well as a complete quick reference to all options and arguments for the sophisticated user.
>  脚本的 docstring 应该认为在脚本被 `-h` 选项调用或者被错误调用时被打印
>  因此其 docstring 应该记录脚本的功能、命令行语法、环境变量、文件
>  关于使用方法的信息应该足够详细，目标是新用户可以通过它正确使用该命令/脚本，同时也可以作为熟练用户对所有选项和参数的快速参考

The docstring for a module should generally list the classes, exceptions and functions (and any other objects) that are exported by the module, with a one-line summary of each. (These summaries generally give less detail than the summary line in the object’s docstring.) The docstring for a package (i.e., the docstring of the package’s `__init__.py` module) should also list the modules and subpackages exported by the package.
>  模块的 docstring 应该列举出该模块导出的类、异常、函数 (以及其他任意对象)，同时各自附带一行总结性描述 (该描述一般比对象自己的 docstring 中的描述更简洁)
>  包的 docstring (包的 `__init__.py` 模块的 docstring ) 应该列举出包导出的所有模块和子包

The docstring for a function or method should summarize its behavior and document its arguments, return value(s), side effects, exceptions raised, and restrictions on when it can be called (all if applicable). Optional arguments should be indicated. It should be documented whether keyword arguments are part of the interface.
>  函数/方法的 docstring 应该总结其行为，并记录其参数 (包括可选参数)、返回值、side effect、抛出的异常、函数被调用时的限制
>  关键字参数是否作为函数接口的一部分也应该记录

The docstring for a class should summarize its behavior and list the public methods and instance variables. If the class is intended to be subclassed, and has an additional interface for subclasses, this interface should be listed separately (in the docstring). The class constructor should be documented in the docstring for its `__init__` method. Individual methods should be documented by their own docstring.
>  类的 docstring 应该总结其行为，列出其公有方法和实例变量
>  如果类会被继承，并且其子类有额外的接口，该接口也应该在类的 docstring 中分别列出
>  类的构造函数记录在 `__init__` 方法的 docstring 中，其余方法有各自的 docstring

If a class subclasses another class and its behavior is mostly inherited from that class, its docstring should mention this and summarize the differences. Use the verb “override” to indicate that a subclass method replaces a superclass method and does not call the superclass method; use the verb “extend” to indicate that a subclass method calls the superclass method (in addition to its own behavior).
>  如果类为另一个类的子类并且继承了大部分行为，其 docstring 应该提及这一点，并且总结差异
>  如果子类的方法重载了父类的方法，用 "override" 说明
>  如果子类的方法调用了父类的方法并且基于此有额外的行为，用 “extend” 说明

_Do not_ use the Emacs convention of mentioning the arguments of functions or methods in upper case in running text. Python is case sensitive and the argument names can be used for keyword arguments, so the docstring should document the correct argument names. It is best to list each argument on a separate line. For example:
>  每个参数最好各自一行

```python
def complex(real=0.0, imag=0.0):
    """Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    if imag == 0.0 and real == 0.0:
        return complex_zero
    ...
```

Unless the entire docstring fits on a line, place the closing quotes on a line by themselves. This way, Emacs’ `fill-paragraph` command can be used on it.
>  多行 docstring 的 closing `"""` 自己一行

### Handling Docstring Indentation
Docstring processing tools will strip a uniform amount of indentation from the second and further lines of the docstring, equal to the minimum indentation of all non-blank lines after the first line. Any indentation in the first line of the docstring (i.e., up to the first newline) is insignificant and removed. Relative indentation of later lines in the docstring is retained. Blank lines should be removed from the beginning and end of the docstring.
>  docstring 处理工具将从 docstring 的第二行及后续行中删除统一数量的缩进，该数量等于第一行之后所有非空白行的最小缩进量
>  docstring 第一行中的任何缩进（即，直到第一个换行符）都是无关紧要的，并会被移除，docstring 后续行之间的相对缩进将被保留
>  docstring 开头和末尾的空白行应被移除

Since code is much more precise than words, here is an implementation of the algorithm:

```python
def trim(docstring):
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)
```

The docstring in this example contains two newline characters and is therefore 3 lines long. The first and last lines are blank:

```python
def foo():
    """
    This is the second line of the docstring.
    """
```

To illustrate:

```
>>> print repr(foo.__doc__)
'\n    This is the second line of the docstring.\n    '
>>> foo.__doc__.splitlines()
['', '    This is the second line of the docstring.', '    ']
>>> trim(foo.__doc__)
'This is the second line of the docstring.'
```

Once trimmed, these docstrings are equivalent:

```python
def foo():
    """A multi-line
    docstring.
    """

def bar():
    """
    A multi-line
    docstring.
    """
```

## Copyright
This document has been placed in the public domain.

## Acknowledgements
The “Specification” text comes mostly verbatim from [PEP 8](https://peps.python.org/pep-0008/ "PEP 8 – Style Guide for Python Code") by Guido van Rossum.

This document borrows ideas from the archives of the Python [Doc-SIG](https://www.python.org/community/sigs/current/doc-sig/). Thanks to all members past and present.

---

Source: [https://github.com/python/peps/blob/main/peps/pep-0257.rst](https://github.com/python/peps/blob/main/peps/pep-0257.rst)

Last modified: [2024-04-17 11:35:59 GMT](https://github.com/python/peps/commits/main/peps/pep-0257.rst)