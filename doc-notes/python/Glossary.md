**decorator**
A function returning another function, usually applied as a function transformation using the `@wrapper` syntax. Common examples for decorators are [`classmethod()`](https://docs.python.org/3/library/functions.html#classmethod "classmethod") and [`staticmethod()`](https://docs.python.org/3/library/functions.html#staticmethod "staticmethod").
>  装饰器即返回另一个函数的函数，通常通过 `@wrapper` 语法用作函数转换
>  装饰器的常见示例有 `classmethod(), staticmethod()`

The decorator syntax is merely syntactic sugar, the following two function definitions are semantically equivalent:

```python
def f(arg):
    ...
f = staticmethod(f)

@staticmethod
def f(arg):
    ...
```

>  装饰器语法的作用仅仅是作为语法糖，上述两个函数定义在语义上是相同的
>  即用 `@staticmethod` 修饰函数 `f` 等价于调用了语句 `f = staticmethod(f)`

The same concept exists for classes, but is less commonly used there. See the documentation for [function definitions](https://docs.python.org/3/reference/compound_stmts.html#function) and [class definitions](https://docs.python.org/3/reference/compound_stmts.html#class) for more about decorators.