# Built-in Functions

The Python interpreter has a number of functions and types built into it that are always available. They are listed here in alphabetical order.

## `class property`
_class_ property(_fget=None_, _fset=None_, _fdel=None_, _doc=None_)

Return a property attribute.
>  `property` 返回一个 property 属性 (这个属性属于某个类)

_fget_ is a function for getting an attribute value. _fset_ is a function for setting an attribute value. _fdel_ is a function for deleting an attribute value. And _doc_ creates a docstring for the attribute.
>  `fget` 是获取该 property 属性值的函数，`fset` 是设定该 property 属性值的函数，`fdel` 是删除该 property 属性值的函数，`doc` 是为该 property 属性创建 docstring 的函数

A typical use is to define a managed attribute `x`:

```python
class C:
    def __init__(self):
        self._x = None

    def getx(self):
        return self._x

    def setx(self, value):
        self._x = value

    def delx(self):
        del self._x

    x = property(getx, setx, delx, "I'm the 'x' property.")
```

If _c_ is an instance of _C_, `c.x` will invoke the getter, `c.x = value` will invoke the setter, and `del c.x` the deleter.
>  上述代码为 `class C` 创建了一个 `property`: `x`
>  对于 `C` 的一个实例 `c`，语句 `c.x` 会调用 getter，语句 `c.x = value` 会调用 setter，语句 `del c.x` 会调用 deleter

If given, _doc_ will be the docstring of the property attribute. Otherwise, the property will copy _fget_’s docstring (if it exists). This makes it possible to create read-only properties easily using [`property()`](https://docs.python.org/3/library/functions.html#property "property") as a [decorator](https://docs.python.org/3/glossary.html#term-decorator):
>  如果给定 `doc`，则使用它作为 property attribute 的 docstring，否则使用 `fget` 的 docstring
>  一种常见的方式是以装饰器的方式使用 `property()` 来构建只读属性

```python
class Parrot:
    def __init__(self):
        self._voltage = 100000

    @property
    def voltage(self):
        """Get the current voltage."""
        return self._voltage
```

The `@property` decorator turns the `voltage()` method into a “getter” for a read-only attribute with the same name, and it sets the docstring for _voltage_ to “Get the current voltage.”
>  `@property` 装饰器将它装饰的函数 `voltage()` 变为了一个名字为 `voltage` 的属性的 getter，并且将 `voltage` 的 docstring 设定为 `volatge()` 的 docstring

@getter

@setter

@deleter

A property object has `getter`, `setter`, and `deleter` methods usable as decorators that create a copy of the property with the corresponding accessor function set to the decorated function. This is best explained with an example:
>  property 对象的 `getter, setter, deleter` 方法都可以用作装饰器，用于创建一个新的 property 副本，其对应的访问函数设置为被装饰的函数

```python
class C:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x
```

This code is exactly equivalent to the first example. Be sure to give the additional functions the same name as the original property (`x` in this case.)
>  如果这些函数的名字都是 `x`，它们就分别成为 `x` 的 getter, setter, deleter

The returned property object also has the attributes `fget`, `fset`, and `fdel` corresponding to the constructor arguments.

>  Changed in version 3.5: The docstrings of property objects are now writeable.

```
__name__
```

Attribute holding the name of the property. The name of the property can be changed at runtime.

>  Added in version 3.13.