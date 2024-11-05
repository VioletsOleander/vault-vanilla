author: Larry Hastings

# Abstract
This document is designed to encapsulate the best practices for working with annotations dicts. If you write Python code that examines `__annotations__` on Python objects, we encourage you to follow the guidelines described below.
>本文件旨在封装关于注解字典（`__annotations__`）操作的最佳实践
>如果你编写检查Python对象上的 `__annotations__` 的Python代码，我们鼓励你遵循下面描述的指南。

The document is organized into four sections: best practices for accessing the annotations of an object in Python versions 3.10 and newer, best practices for accessing the annotations of an object in Python versions 3.9 and older, other best practices for `__annotations__` that apply to any Python version, and quirks of `__annotations__`.
>本文档分为四个部分：适用于Python 3.10及更新版本的对象注解访问最佳实践；适用于Python 3.9及更老版本的对象注解访问最佳实践；适用于任何Python版本的其他关于 `__annotations__` 的最佳实践；以及 `__annotations__` 的特殊之处。

Note that this document is specifically about working with `__annotations__`, not uses _for_ annotations. If you’re looking for information on how to use “type hints” in your code, please see the [`typing`](https://docs.python.org/3/library/typing.html#module-typing "typing: Support for type hints (see :pep:`484`).") module.
>请注意，本文档仅涉及与 `__annotations__` 相关的操作，而非使用注解本身。如果你想了解如何在代码中使用“类型提示”，请参考[`typing`](https://docs.python.org/3/library/typing.html#module-typing "typing: Support for type hints (see :pep:`484`).")模块。

## Accessing The Annotations Dict Of An Object In Python 3.10 And Newer
Python 3.10 adds a new function to the standard library: [`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations "inspect.get_annotations"). In Python versions 3.10 and newer, calling this function is the best practice for accessing the annotations dict of any object that supports annotations. This function can also “un-stringize” stringized annotations for you.
>Python 3.10在标准库中增加了一个新的函数：[`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations "inspect.get_annotations")。在Python 3.10及更高版本中，调用此函数是访问支持注解的任何对象的注解字典的最佳实践
>此函数还可以帮助我们将字符串化的注解“取消字符串化”。

If for some reason [`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations "inspect.get_annotations") isn’t viable for your use case, you may access the `__annotations__` data member manually. Best practice for this changed in Python 3.10 as well: as of Python 3.10, `o.__annotations__` is guaranteed to _always_ work on Python functions, classes, and modules. If you’re certain the object you’re examining is one of these three _specific_ objects, you may simply use `o.__annotations__` to get at the object’s annotations dict.
>如果出于某种原因，[`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations "inspect.get_annotations") 不适合你的用例，你可以手动访问 `__annotations__` 数据成员
>在Python 3.10中，访问 `__annotations__` 的最佳实践也发生了变化：从Python 3.10开始，`o.__annotations__` 保证总是可以在Python函数、类和模块上调用。如果你确定你正在检查的对象是这三个特定对象之一（函数、类或模块），你可以直接使用 `o.__annotations__` 来获取对象的注解字典。

However, other types of callables–for example, callables created by [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial "functools.partial")–may not have an `__annotations__` attribute defined. When accessing the `__annotations__` of a possibly unknown object, best practice in Python versions 3.10 and newer is to call [`getattr()`](https://docs.python.org/3/library/functions.html#getattr "getattr") with three arguments, for example `getattr(o, '__annotations__', None)`.
>然而，其他类型的可调用对象——例如由[`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial "functools.partial") 创建的可调用对象——可能没有定义 `__annotations__` 属性
>在访问可能是未知对象的 `__annotations__` 时，在Python 3.10及更高版本中，最佳实践是使用三个参数调用[`getattr()`](https://docs.python.org/3/library/functions.html#getattr "getattr")，例如 `getattr(o, '__annotations__', None)`。

Before Python 3.10, accessing `__annotations__` on a class that defines no annotations but that has a parent class with annotations would return the parent’s `__annotations__`. In Python 3.10 and newer, the child class’s annotations will be an empty dict instead.
>在Python 3.10之前，如果一个类定义了注解但它的父类也有注解，则访问此类的 `__annotations__` 会返回父类的 `__annotations__`
>但在Python 3.10及更高版本中，子类的注解将是空字典。

## Accessing The Annotations Dict Of An Object In Python 3.9 And Older
In Python 3.9 and older, accessing the annotations dict of an object is much more complicated than in newer versions. The problem is a design flaw in these older versions of Python, specifically to do with class annotations.
>在 Python 3.9 及更早版本中，访问对象的注解字典要比新版本复杂得多
>这个问题源于这些较旧版本 Python 的设计缺陷，特别是与类注解有关的设计。

Best practice for accessing the annotations dict of other objects–functions, other callables, and modules–is the same as best practice for 3.10, assuming you aren’t calling [`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations "inspect.get_annotations"): you should use three-argument [`getattr()`](https://docs.python.org/3/library/functions.html#getattr "getattr") to access the object’s `__annotations__` attribute.
>对于访问其他对象（如函数、可调用对象和模块）的注解字典的最佳实践，与 3.10 版本相同（假设你没有调用 [`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations)）：
>你应该使用三参数的 [`getattr()`](https://docs.python.org/3/library/functions.html#getattr) 来访问对象的 `__annotations__` 属性。

Unfortunately, this isn’t best practice for classes. The problem is that, since `__annotations__` is optional on classes, and because classes can inherit attributes from their base classes, accessing the `__annotations__` attribute of a class may inadvertently return the annotations dict of a _base class._ As an example:
>不幸的是，这并不是处理类的最佳实践
>问题在于，由于类上的 `__annotations__` 是可选的，并且类可以从其基类继承属性，因此访问类的 `__annotations__` 属性可能会意外地返回基类的注解字典
>举个例子：

```python
class Base:
    a: int = 3
    b: str = 'abc'

class Derived(Base):
    pass

print(Derived.__annotations__)
```

This will print the annotations dict from `Base`, not `Derived`.
>这将打印出 `Base` 的注解字典，而不是 `Derived` 的。

Your code will have to have a separate code path if the object you’re examining is a class (`isinstance(o, type)`). In that case, best practice relies on an implementation detail of Python 3.9 and before: if a class has annotations defined, they are stored in the class’s [`__dict__`](https://docs.python.org/3/reference/datamodel.html#type.__dict__ "type.__dict__") dictionary. Since the class may or may not have annotations defined, best practice is to call the [`get()`](https://docs.python.org/3/library/stdtypes.html#dict.get "dict.get") method on the class dict.
>如果你正在检查的对象是一个类（`isinstance(o, type)`），则你的代码将需要一个单独的代码路径
>在这种情况下，最佳实践依赖于 Python 3.9 及之前的实现细节：如果类定义了注解，那么它们会被存储在类的 `__dict__` 字典中。由于类可能或可能没有定义注解，最佳实践是调用类字典上的 `get()` 方法。

To put it all together, here is some sample code that safely accesses the `__annotations__` attribute on an arbitrary object in Python 3.9 and before:
>为了总结一下，这里有一些示例代码，在 Python 3.9 及之前版本中可以安全地访问任意对象的 `__annotations__` 属性：

```python
if isinstance(o, type):
    ann = o.__dict__.get('__annotations__', None)
else:
    ann = getattr(o, '__annotations__', None)
```

After running this code, `ann` should be either a dictionary or `None`. You’re encouraged to double-check the type of `ann` using [`isinstance()`](https://docs.python.org/3/library/functions.html#isinstance "isinstance") before further examination.
>运行这段代码后，`ann` 应该要么是一个字典，要么是 `None`
>建议在进一步检查之前使用 [`isinstance()`](https://docs.python.org/3/library/functions.html#isinstance) 来验证 `ann` 的类型。

Note that some exotic or malformed type objects may not have a [`__dict__`](https://docs.python.org/3/reference/datamodel.html#type.__dict__ "type.__dict__") attribute, so for extra safety you may also wish to use [`getattr()`](https://docs.python.org/3/library/functions.html#getattr "getattr") to access `__dict__`.
>请注意，一些奇异或格式错误的类型对象可能没有 `__dict__` 属性，因此为了额外的安全性，你也可以使用 [`getattr()`](https://docs.python.org/3/library/functions.html#getattr) 来访问 `__dict__`。

## Manually Un-Stringizing Stringized Annotations
In situations where some annotations may be “stringized”, and you wish to evaluate those strings to produce the Python values they represent, it really is best to call [`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations "inspect.get_annotations") to do this work for you.
>在某些情况下，注解可能是“字符串化”的，即以字符串形式表示，而你想评估这些字符串以生成它们所代表的 Python 值，最好的做法是调用 [`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations) 来为你完成这项工作。

If you’re using Python 3.9 or older, or if for some reason you can’t use [`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations "inspect.get_annotations"), you’ll need to duplicate its logic. You’re encouraged to examine the implementation of [`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations "inspect.get_annotations") in the current Python version and follow a similar approach.
>如果你使用的是 Python 3.9 或更早版本，或者出于某种原因不能使用 [`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations)，你将需要复制它的逻辑
>建议你检查当前 Python 版本中 [`inspect.get_annotations()`](https://docs.python.org/3/library/inspect.html#inspect.get_annotations) 的实现，并采用类似的策略。

In a nutshell, if you wish to evaluate a stringized annotation on an arbitrary object `o`:

- If `o` is a module, use `o.__dict__` as the `globals` when calling [`eval()`](https://docs.python.org/3/library/functions.html#eval "eval").
- If `o` is a class, use `sys.modules[o.__module__].__dict__` as the `globals`, and `dict(vars(o))` as the `locals`, when calling [`eval()`](https://docs.python.org/3/library/functions.html#eval "eval").
- If `o` is a wrapped callable using [`functools.update_wrapper()`](https://docs.python.org/3/library/functools.html#functools.update_wrapper "functools.update_wrapper"), [`functools.wraps()`](https://docs.python.org/3/library/functools.html#functools.wraps "functools.wraps"), or [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial "functools.partial"), iteratively unwrap it by accessing either `o.__wrapped__` or `o.func` as appropriate, until you have found the root unwrapped function.
- If `o` is a callable (but not a class), use [`o.__globals__`](https://docs.python.org/3/reference/datamodel.html#function.__globals__ "function.__globals__") as the globals when calling [`eval()`](https://docs.python.org/3/library/functions.html#eval "eval").

>简单来说，如果你想评估任意对象 `o` 上的字符串化注解：
>- 如果 `o` 是一个模块，使用 `o.__dict__` 作为 `globals` 调用 [`eval()`](https://docs.python.org/3/library/functions.html#eval)。
>- 如果 `o` 是一个类，使用 `sys.modules[o.__module__].__dict__` 作为 `globals`，并使用 `dict(vars(o))` 作为 `locals` 调用 [`eval()`]。
>- 如果 `o` 是使用 [`functools.update_wrapper()`](https://docs.python.org/3/library/functools.html#functools.update_wrapper)、[`functools.wraps()`](https://docs.python.org/3/library/functools.html#functools.wraps) 或 [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial) 封装的可调用对象，则通过访问 `o.__wrapped__` 或 `o.func` 来逐步解封装，直到找到根未封装的函数。
>- 如果 `o` 是一个可调用对象（但不是类），使用 `o.__globals__` 作为 `globals` 调用 `eval()`。

However, not all string values used as annotations can be successfully turned into Python values by [`eval()`](https://docs.python.org/3/library/functions.html#eval "eval"). String values could theoretically contain any valid string, and in practice there are valid use cases for type hints that require annotating with string values that specifically _can’t_ be evaluated. For example:

- [**PEP 604**](https://peps.python.org/pep-0604/) union types using `|`, before support for this was added to Python 3.10.
- Definitions that aren’t needed at runtime, only imported when [`typing.TYPE_CHECKING`](https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING "typing.TYPE_CHECKING") is true.

>然而，并非所有用于注解的字符串值都可以通过 [`eval()`](https://docs.python.org/3/library/functions.html#eval) 成功转换为 Python 值。字符串值理论上可以包含任何有效字符串，实际上存在一些使用场景，需要使用某些特定的字符串值进行类型提示，这些字符串值**不能**被评估。例如：
>- [**PEP 604**](https://peps.python.org/pep-0604/) 中的联合类型使用 `|`，在 Python 3.10 加入对这一特性的支持之前。
>- 不需要在运行时使用的定义，仅在 `typing.TYPE_CHECKING` 为真时导入。

If [`eval()`](https://docs.python.org/3/library/functions.html#eval "eval") attempts to evaluate such values, it will fail and raise an exception. So, when designing a library API that works with annotations, it’s recommended to only attempt to evaluate string values when explicitly requested to by the caller.
>如果 [`eval()`](https://docs.python.org/3/library/functions.html#eval) 尝试评估此类值，它将失败并引发异常
>因此，在设计一个处理注解的库 API 时，建议仅在调用者明确请求时才尝试评估字符串值。

## Best Practices For `__annotations__` In Any Python Version

- You should avoid assigning to the `__annotations__` member of objects directly. Let Python manage setting `__annotations__`.
- If you do assign directly to the `__annotations__` member of an object, you should always set it to a `dict` object.
- If you directly access the `__annotations__` member of an object, you should ensure that it’s a dictionary before attempting to examine its contents.
- You should avoid modifying `__annotations__` dicts.
- You should avoid deleting the `__annotations__` attribute of an object.

>- 应避免直接对对象的 `__annotations__` 成员进行赋值。让 Python 来管理设置 `__annotations__`。
>- 如果你确实需要直接对对象的 `__annotations__` 成员进行赋值，应始终将其设置为一个 `dict` 对象。
>- 如果你需要直接访问对象的 `__annotations__` 成员，应在试图检查其内容之前确保它是一个字典。
>- 应避免修改 `__annotations__` 字典。
>- 应避免删除对象的 `__annotations__` 属性。

## `__annotations__` Quirks
In all versions of Python 3, function objects lazy-create an annotations dict if no annotations are defined on that object. You can delete the `__annotations__` attribute using `del fn.__annotations__`, but if you then access `fn.__annotations__` the object will create a new empty dict that it will store and return as its annotations. Deleting the annotations on a function before it has lazily created its annotations dict will throw an `AttributeError`; using `del fn.__annotations__` twice in a row is guaranteed to always throw an `AttributeError`.
>在所有版本的 Python 3 中，如果函数对象没有定义注解，则会延迟创建一个注解字典
>你可以使用 `del fn.__annotations__` 删除 `__annotations__` 属性，但如果你随后访问 `fn.__annotations__`，对象将创建一个新的空字典并存储和返回这个字典作为其注解
>在函数尚未延迟创建其注解字典之前删除其上的注解会导致一个 `AttributeError`；连续两次使用 `del fn.__annotations__` 总是会抛出一个 `AttributeError`。

Everything in the above paragraph also applies to class and module objects in Python 3.10 and newer.
>上述段落中的所有内容同样适用于 Python 3.10 及更高版本中的类和模块对象。

In all versions of Python 3, you can set `__annotations__` on a function object to `None`. However, subsequently accessing the annotations on that object using `fn.__annotations__` will lazy-create an empty dictionary as per the first paragraph of this section. This is _not_ true of modules and classes, in any Python version; those objects permit setting `__annotations__` to any Python value, and will retain whatever value is set.
>在所有版本的 Python 3 中，你可以将函数对象的 `__annotations__` 设置为 `None` 然而，随后使用 `fn.__annotations__` 访问该对象的注解将会按本节第一段所述的方式延迟创建一个空字典
>对于模块和类来说，无论 Python 版本如何，都可以将 `__annotations__` 设置为任何 Python 值，并保留任何设置的值。

If Python stringizes your annotations for you (using `from __future__ import annotations`), and you specify a string as an annotation, the string will itself be quoted. In effect the annotation is quoted _twice._ For example:
>如果 Python 自己为你将注解字符串化（通过 `from __future__ import annotations`），并且你在注解中指定一个字符串，该字符串本身会被引号包围。实际上，注解会被双重引用。例如：

```python
from __future__ import annotations
def foo(a: "str"): pass

print(foo.__annotations__)
```

This prints `{'a': "'str'"}`. This shouldn’t really be considered a “quirk”; it’s mentioned here simply because it might be surprising.
>这将打印 `{'a': "'str'"}`。这不应被视为一个“特例”；这里提到这一点只是因为它可能会让人感到意外。