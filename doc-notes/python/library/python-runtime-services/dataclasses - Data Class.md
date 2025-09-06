---
completed:
version: 3.13.7
---
# `dataclasses` — Data Classes
**Source code:** [Lib/dataclasses.py](https://github.com/python/cpython/tree/3.13/Lib/dataclasses.py)

---

This module provides a decorator and functions for automatically adding generated [special methods](https://docs.python.org/3/glossary.html#term-special-method) such as [`__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__ "object.__init__") and [`__repr__()`](https://docs.python.org/3/reference/datamodel.html#object.__repr__ "object.__repr__") to user-defined classes. It was originally described in [**PEP 557**](https://peps.python.org/pep-0557/).
>  `dataclasses` 模块提供了一个装饰器，以及各种函数，用于自动为用户定义的类添加生成的特殊方法例如 `__init__(), __repr__()`

The member variables to use in these generated methods are defined using [**PEP 526**](https://peps.python.org/pep-0526/) type annotations. For example, this code:
>  这些生成的方法的成员变量会使用 PEP 526 类型注释来定义
>  例如，对下面这个类用 `@dataclass` 装饰之后

```python
from dataclasses import dataclass

@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
```

will add, among other things, a `__init__()` that looks like:
>  这个类就会自动获得下面这个形式的 `__init__()` 方法

```python
def __init__(self, name: str, unit_price: float, quantity_on_hand: int = 0):
    self.name = name
    self.unit_price = unit_price
    self.quantity_on_hand = quantity_on_hand
```

Note that this method is automatically added to the class: it is not directly specified in the `InventoryItem` definition shown above.

>  Added in version 3.7.

## Module contents
@dataclasses.dataclass(_*_, _init=True_, _repr=True_, _eq=True_, _order=False_, _unsafe_hash=False_, _frozen=False_, _match_args=True_, _kw_only=False_, _slots=False_, _weakref_slot=False_)

This function is a [decorator](https://docs.python.org/3/glossary.html#term-decorator) that is used to add generated [special methods](https://docs.python.org/3/glossary.html#term-special-method) to classes, as described below.
>  `@dataclasses.dataclass` 函数是一个装饰器，用于为类添加生成的特殊方法

The `@dataclass` decorator examines the class to find `field`s. A `field` is defined as a class variable that has a [type annotation](https://docs.python.org/3/glossary.html#term-variable-annotation). With two exceptions described below, nothing in `@dataclass` examines the type specified in the variable annotation.
>  `@dataclass` 装饰器会检查类，找到 `fields`
>  `field` 被定义为一个具有类型注释的类变量，但除了下面描述的两个例外以外，`@dataclass` 中没有其他会检查变量注释中的类型的东西

The order of the fields in all of the generated methods is the order in which they appear in the class definition.

The `@dataclass` decorator will add various “dunder” methods to the class, described below. If any of the added methods already exist in the class, the behavior depends on the parameter, as documented below. The decorator returns the same class that it is called on; no new class is created.
>  `@dataclass` 装饰器会为类添加各种 dunder (double underscore) 方法
>  如果任意要添加的方法已经在类中存在，行为就由参数决定，该装饰器会返回他被调用的同一个类，不会创建新的类

If `@dataclass` is used just as a simple decorator with no parameters, it acts as if it has the default values documented in this signature. That is, these three uses of `@dataclass` are equivalent:
>  如果使用 `@dataclass` 而没有提供参数时，其行为等价于为它提供了所有的默认参数

```python
@dataclass
class C:
    ...

@dataclass()
class C:
    ...

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)
class C:
    ...
```

The parameters to `@dataclass` are:

- _init_: If true (the default), a [`__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__ "object.__init__") method will be generated.
    
    If the class already defines `__init__()`, this parameter is ignored.
    
- _repr_: If true (the default), a [`__repr__()`](https://docs.python.org/3/reference/datamodel.html#object.__repr__ "object.__repr__") method will be generated. The generated repr string will have the class name and the name and repr of each field, in the order they are defined in the class. Fields that are marked as being excluded from the repr are not included. For example: `InventoryItem(name='widget', unit_price=3.0, quantity_on_hand=10)`.
    
    If the class already defines `__repr__()`, this parameter is ignored.
    
- _eq_: If true (the default), an [`__eq__()`](https://docs.python.org/3/reference/datamodel.html#object.__eq__ "object.__eq__") method will be generated. This method compares the class as if it were a tuple of its fields, in order. Both instances in the comparison must be of the identical type.
    
    If the class already defines `__eq__()`, this parameter is ignored.
    
- _order_: If true (the default is `False`), [`__lt__()`](https://docs.python.org/3/reference/datamodel.html#object.__lt__ "object.__lt__"), [`__le__()`](https://docs.python.org/3/reference/datamodel.html#object.__le__ "object.__le__"), [`__gt__()`](https://docs.python.org/3/reference/datamodel.html#object.__gt__ "object.__gt__"), and [`__ge__()`](https://docs.python.org/3/reference/datamodel.html#object.__ge__ "object.__ge__") methods will be generated. These compare the class as if it were a tuple of its fields, in order. Both instances in the comparison must be of the identical type. If _order_ is true and _eq_ is false, a [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError "ValueError") is raised.
    
    If the class already defines any of `__lt__()`, `__le__()`, `__gt__()`, or `__ge__()`, then [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "TypeError") is raised.
    
- _unsafe_hash_: If true, force `dataclasses` to create a [`__hash__()`](https://docs.python.org/3/reference/datamodel.html#object.__hash__ "object.__hash__") method, even though it may not be safe to do so. Otherwise, generate a [`__hash__()`](https://docs.python.org/3/reference/datamodel.html#object.__hash__ "object.__hash__") method according to how _eq_ and _frozen_ are set. The default value is `False`.
    
    `__hash__()` is used by built-in [`hash()`](https://docs.python.org/3/library/functions.html#hash "hash"), and when objects are added to hashed collections such as dictionaries and sets. Having a `__hash__()` implies that instances of the class are immutable. Mutability is a complicated property that depends on the programmer’s intent, the existence and behavior of `__eq__()`, and the values of the _eq_ and _frozen_ flags in the `@dataclass` decorator.
    
    By default, `@dataclass` will not implicitly add a [`__hash__()`](https://docs.python.org/3/reference/datamodel.html#object.__hash__ "object.__hash__") method unless it is safe to do so. Neither will it add or change an existing explicitly defined `__hash__()` method. Setting the class attribute `__hash__ = None` has a specific meaning to Python, as described in the `__hash__()` documentation.
    
    If `__hash__()` is not explicitly defined, or if it is set to `None`, then `@dataclass` _may_ add an implicit `__hash__()` method. Although not recommended, you can force `@dataclass` to create a `__hash__()` method with `unsafe_hash=True`. This might be the case if your class is logically immutable but can still be mutated. This is a specialized use case and should be considered carefully.
    
    Here are the rules governing implicit creation of a `__hash__()` method. Note that you cannot both have an explicit `__hash__()` method in your dataclass and set `unsafe_hash=True`; this will result in a [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "TypeError").
    
    If _eq_ and _frozen_ are both true, by default `@dataclass` will generate a `__hash__()` method for you. If _eq_ is true and _frozen_ is false, `__hash__()` will be set to `None`, marking it unhashable (which it is, since it is mutable). If _eq_ is false, `__hash__()` will be left untouched meaning the `__hash__()` method of the superclass will be used (if the superclass is [`object`](https://docs.python.org/3/library/functions.html#object "object"), this means it will fall back to id-based hashing).
    
- _frozen_: If true (the default is `False`), assigning to fields will generate an exception. This emulates read-only frozen instances. See the [discussion](https://docs.python.org/3/library/dataclasses.html#dataclasses-frozen) below.
    
    If [`__setattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__setattr__ "object.__setattr__") or [`__delattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__delattr__ "object.__delattr__") is defined in the class and _frozen_ is true, then [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "TypeError") is raised.
    
- _match_args_: If true (the default is `True`), the [`__match_args__`](https://docs.python.org/3/reference/datamodel.html#object.__match_args__ "object.__match_args__") tuple will be created from the list of non keyword-only parameters to the generated [`__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__ "object.__init__") method (even if `__init__()` is not generated, see above). If false, or if `__match_args__` is already defined in the class, then `__match_args__` will not be generated.
    

> Added in version 3.10.

- _kw_only_: If true (the default value is `False`), then all fields will be marked as keyword-only. If a field is marked as keyword-only, then the only effect is that the [`__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__ "object.__init__") parameter generated from a keyword-only field must be specified with a keyword when `__init__()` is called. See the [parameter](https://docs.python.org/3/glossary.html#term-parameter) glossary entry for details. Also see the [`KW_ONLY`](https://docs.python.org/3/library/dataclasses.html#dataclasses.KW_ONLY "dataclasses.KW_ONLY") section.
    
    Keyword-only fields are not included in `__match_args__`.
    

> Added in version 3.10.

- _slots_: If true (the default is `False`), [`__slots__`](https://docs.python.org/3/reference/datamodel.html#object.__slots__ "object.__slots__") attribute will be generated and new class will be returned instead of the original one. If `__slots__` is already defined in the class, then [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "TypeError") is raised.
    

> Warning Calling no-arg [`super()`](https://docs.python.org/3/library/functions.html#super "super") in dataclasses using `slots=True` will result in the following exception being raised: `TypeError: super(type, obj): obj must be an instance or subtype of type`. The two-arg [`super()`](https://docs.python.org/3/library/functions.html#super "super") is a valid workaround. See [gh-90562](https://github.com/python/cpython/issues/90562) for full details.
> 
> Warning Passing parameters to a base class [`__init_subclass__()`](https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__ "object.__init_subclass__") when using `slots=True` will result in a [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "TypeError"). Either use `__init_subclass__` with no parameters or use default values as a workaround. See [gh-91126](https://github.com/python/cpython/issues/91126) for full details.

> Added in version 3.10.

> Changed in version 3.11: If a field name is already included in the `__slots__` of a base class, it will not be included in the generated `__slots__` to prevent [overriding them](https://docs.python.org/3/reference/datamodel.html#datamodel-note-slots). Therefore, do not use `__slots__` to retrieve the field names of a dataclass. Use [`fields()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.fields "dataclasses.fields") instead. To be able to determine inherited slots, base class `__slots__` may be any iterable, but _not_ an iterator.

- _weakref_slot_: If true (the default is `False`), add a slot named “__weakref__”, which is required to make an instance [`weakref-able`](https://docs.python.org/3/library/weakref.html#weakref.ref "weakref.ref"). It is an error to specify `weakref_slot=True` without also specifying `slots=True`.

> Added in version 3.11.

`field`s may optionally specify a default value, using normal Python syntax:

```python
@dataclass
class C:
    a: int       # 'a' has no default value
    b: int = 0   # assign a default value for 'b'
```

In this example, both `a` and `b` will be included in the added [`__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__ "object.__init__") method, which will be defined as:

```python
def __init__(self, a: int, b: int = 0):
```

[`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "TypeError") will be raised if a field without a default value follows a field with a default value. This is true whether this occurs in a single class, or as a result of class inheritance.
