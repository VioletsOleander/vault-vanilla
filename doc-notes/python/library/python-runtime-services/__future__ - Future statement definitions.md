---
completed: true
version: 3.13.7
---
# `__future__` — Future statement definitions

**Source code:** [Lib/__future__.py](https://github.com/python/cpython/tree/3.13/Lib/__future__.py)

---

Imports of the form `from __future__ import feature` are called [future statements](https://docs.python.org/3/reference/simple_stmts.html#future). These are special-cased by the Python compiler to allow the use of new Python features in modules containing the future statement before the release in which the feature becomes standard.
>  形式为 `from __future__ import feature` 的 `impoart` 被称为未来语句
>  未来语句会被 Python 编译器特殊处理，以允许在某个新功能成为标准之前，就在当前模块中使用它

>  实际上这不是导入一个真实的模块，而是像一个编译时语法开关，告诉编译器接下来的代码请按新规则解析

While these future statements are given additional special meaning by the Python compiler, they are still executed like any other import statement and the [`__future__`](https://docs.python.org/3/library/__future__.html#module-__future__ "__future__: Future statement definitions") exists and is handled by the import system the same way any other Python module would be. 
>  虽然 future 语句对于 Python 编译器具有特殊意义，但它仍然按照正常的 import 语句执行，并且 `__future__` 模块也是真实存在的，该模块被 import system 处理的方式也和其他 Python 模块被处理的方式一样

This design serves three purposes:

- To avoid confusing existing tools that analyze import statements and expect to find the modules they’re importing.
- To document when incompatible changes were introduced, and when they will be — or were — made mandatory. This is a form of executable documentation, and can be inspected programmatically via importing [`__future__`](https://docs.python.org/3/library/__future__.html#module-__future__ "__future__: Future statement definitions") and examining its contents.
- To ensure that [future statements](https://docs.python.org/3/reference/simple_stmts.html#future) run under releases prior to Python 2.1 at least yield runtime exceptions (the import of [`__future__`](https://docs.python.org/3/library/__future__.html#module-__future__ "__future__: Future statement definitions") will fail, because there was no module of that name prior to 2.1).

>  这个设计有三个目的:
>  - 避免混淆现存的静态分析工具，它们会扫描代码中的 import 语句，并视图找出所依赖的模块
>  - 作为 “可执行文档” 来记录变更，可以通过检查 `__future__` 中的信息知道这个项目依赖于哪些未来特性，以及这些未来特性何时会生效
>  - 确保在旧的 Python 版本下不会出错

## Module Contents
No feature description will ever be deleted from [`__future__`](https://docs.python.org/3/library/__future__.html#module-__future__ "__future__: Future statement definitions"). Since its introduction in Python 2.1 the following features have found their way into the language using this mechanism:

|feature|optional in|mandatory in|effect|
|---|---|---|---|
|__future__.nested_scopes[](https://docs.python.org/3/library/__future__.html#future__.nested_scopes "Link to this definition")|2.1.0b1|2.2|[**PEP 227**](https://peps.python.org/pep-0227/): _Statically Nested Scopes_|
|__future__.generators[](https://docs.python.org/3/library/__future__.html#future__.generators "Link to this definition")|2.2.0a1|2.3|[**PEP 255**](https://peps.python.org/pep-0255/): _Simple Generators_|
|__future__.division[](https://docs.python.org/3/library/__future__.html#future__.division "Link to this definition")|2.2.0a2|3.0|[**PEP 238**](https://peps.python.org/pep-0238/): _Changing the Division Operator_|
|__future__.absolute_import[](https://docs.python.org/3/library/__future__.html#future__.absolute_import "Link to this definition")|2.5.0a1|3.0|[**PEP 328**](https://peps.python.org/pep-0328/): _Imports: Multi-Line and Absolute/Relative_|
|__future__.with_statement[](https://docs.python.org/3/library/__future__.html#future__.with_statement "Link to this definition")|2.5.0a1|2.6|[**PEP 343**](https://peps.python.org/pep-0343/): _The “with” Statement_|
|__future__.print_function[](https://docs.python.org/3/library/__future__.html#future__.print_function "Link to this definition")|2.6.0a2|3.0|[**PEP 3105**](https://peps.python.org/pep-3105/): _Make print a function_|
|__future__.unicode_literals[](https://docs.python.org/3/library/__future__.html#future__.unicode_literals "Link to this definition")|2.6.0a2|3.0|[**PEP 3112**](https://peps.python.org/pep-3112/): _Bytes literals in Python 3000_|
|__future__.generator_stop[](https://docs.python.org/3/library/__future__.html#future__.generator_stop "Link to this definition")|3.5.0b1|3.7|[**PEP 479**](https://peps.python.org/pep-0479/): _StopIteration handling inside generators_|
|__future__.annotations[](https://docs.python.org/3/library/__future__.html#future__.annotations "Link to this definition")|3.7.0b1|Never [[1]](https://docs.python.org/3/library/__future__.html#id2)|[**PEP 563**](https://peps.python.org/pep-0563/): _Postponed evaluation of annotations_|

_class_ __future__._Feature

Each statement in `__future__.py` is of the form:

```python
FeatureName = _Feature(OptionalRelease, MandatoryRelease, CompilerFlag)
```

where, normally, _OptionalRelease_ is less than _MandatoryRelease_, and both are 5-tuples of the same form as [`sys.version_info`](https://docs.python.org/3/library/sys.html#sys.version_info "sys.version_info"):

```
(PY_MAJOR_VERSION, # the 2 in 2.1.0a3; an int
 PY_MINOR_VERSION, # the 1; an int
 PY_MICRO_VERSION, # the 0; an int
 PY_RELEASE_LEVEL, # "alpha", "beta", "candidate" or "final"; string
 PY_RELEASE_SERIAL # the 3; an int
)
```


>  `__future__.py` 中的每个语句的形式都如上所示
>  其中 `OptionalRelease` 通常小于 `MandatoryRelease`

_Feature.getOptionalRelease()

_OptionalRelease_ records the first release in which the feature was accepted.

_Feature.getMandatoryRelease()

In the case of a _MandatoryRelease_ that has not yet occurred, _MandatoryRelease_ predicts the release in which the feature will become part of the language.

Else _MandatoryRelease_ records when the feature became part of the language; in releases at or after that, modules no longer need a future statement to use the feature in question, but may continue to use such imports.

_MandatoryRelease_ may also be `None`, meaning that a planned feature got dropped or that it is not yet decided.

_Feature.compiler_flag

_CompilerFlag_ is the (bitfield) flag that should be passed in the fourth argument to the built-in function [`compile()`](https://docs.python.org/3/library/functions.html#compile "compile") to enable the feature in dynamically compiled code. This flag is stored in the [`_Feature.compiler_flag`](https://docs.python.org/3/library/__future__.html#future__._Feature.compiler_flag "__future__._Feature.compiler_flag") attribute on [`_Feature`](https://docs.python.org/3/library/__future__.html#future__._Feature "__future__._Feature") instances.

[[1](https://docs.python.org/3/library/__future__.html#id1)]
`from __future__ import annotations` was previously scheduled to become mandatory in Python 3.10, but the Python Steering Council twice decided to delay the change ([announcement for Python 3.10](https://mail.python.org/archives/list/python-dev@python.org/message/CLVXXPQ2T2LQ5MP2Y53VVQFCXYWQJHKZ/); [announcement for Python 3.11](https://mail.python.org/archives/list/python-dev@python.org/message/VIZEBX5EYMSYIJNDBF6DMUMZOCWHARSO/)). No final decision has been made yet. See also [**PEP 563**](https://peps.python.org/pep-0563/) and [**PEP 649**](https://peps.python.org/pep-0649/).

See also:

[Future statements](https://docs.python.org/3/reference/simple_stmts.html#future)
How the compiler treats future imports.

[**PEP 236**](https://peps.python.org/pep-0236/) - Back to the __future__
The original proposal for the __future__ mechanism.