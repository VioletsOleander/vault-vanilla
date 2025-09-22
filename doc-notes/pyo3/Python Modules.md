---
completed: true
---
# Python Modules
As shown in the Getting Started chapter, you can create a module as follows:

```rust
use pyo3::prelude::*;

// add bindings to the generated Python module
// N.B: "rust2py" must be the name of the `.so` or `.pyd` file.

/// This module is implemented in Rust.
#[pymodule]
fn rust2py(py: Python, m: &PyModule) -> PyResult<()> {
    // PyO3 aware function. All of our Python interfaces could be declared in a separate module.
    // Note that the `#[pyfn()]` annotation automatically converts the arguments from
    // Python objects to Rust values, and the Rust return value back into a Python object.
    // The `_py` argument represents that we're holding the GIL.
    #[pyfn(m, "sum_as_string")]
    fn sum_as_string_py(_py: Python, a: i64, b: i64) -> PyResult<String> {
        let out = sum_as_string(a, b);
        Ok(out)
    }

    Ok(())
}

// logic implemented as a normal Rust function
fn sum_as_string(a: i64, b: i64) -> String {
    format!("{}", a + b)
}

fn main() {}
```

>  `use pyo3::prelude::*` 将 pyo3 中的常用项集合都导入到当前作用域中
>  `py: Python` 表示 Python 解释器的 GIL (全局解释器锁)，它是 CPython 用于线程安全的机制，同一时间只有一个线程能执行 Python 字节码，所有与 Python 交互的操作都必须持有 GIL，`Python` 类型就是 GIL 的 “持有者” 抽象
>  `m: &PyModule` 指向当前正在构建的 Python 模块对象，我们可以往这个模块里添加函数、类、常量等
>  返回值 `PyResult<()>` 表示操作成功与否，类似于 `Result<(), PyErr>`

>  `#[pyfn(m, "sum_as_string")]` 是另一个过程宏，它将下面的函数以 `sum_as_string` 的名字注册到 Python 模块 `m` 中，并且会自动处理类型转换: Python 对象 -> Rust 类型，Rust 返回值 -> Python 对象

>  我们在纯 Rust 函数中实现实际的逻辑，过程宏修饰的函数仅作为和 Python 的接口层

>  `main` 函数为空是 Rust 的要求，仅仅是占位符，我们编译出的模块相对于 Python 的入口点是由 `#[pymodule]` 自动生成的 `PyInit_rust2py` 函数

The `#[pymodule]` procedural macro attribute takes care of exporting the initialization function of your module to Python. It can take as an argument the name of your module, which must be the name of the `.so` or `.pyd` file; the default is the Rust function's name.`
>  `#[pymodule]` 过程宏属性会将我们模块的初始化函数导出给 Python 使用
>  这个过程宏属性接受任意参数，作为我们模块的名字，这个名字也会是 `.so, .pyd` 文件的名字，默认是 Rust 函数的名字

>  过程宏是编译时自动运行的代码生成器，上述过程宏会自动生成一个符合 Python C API 要求的初始化函数，让 Python 可以加载这个模块
>  Python 加载扩展模块时，要求拓展模块有一个特定签名的函数，例如：

```c
PyMODINIT_FUNC PyInit_xxx(void)
```

>  这个函数必须返回一个 `module` 对象
>  `#[pymodule]` 会帮我们自动生成这样的函数

To import the module, either copy the shared library as described in [Get Started](https://pyo3.rs/v0.10.1/get_started) or use a tool, e.g. `maturin develop` with [maturin](https://github.com/PyO3/maturin) or `python setup.py develop` with [setuptools-rust](https://github.com/PyO3/setuptools-rust).
>  要导入这个模块，可以将共享库拷贝，或者使用工具，例如 `maturin develop` 或 `python setup.py develop`

## Documentation
The [Rust doc comments](https://doc.rust-lang.org/stable/book/first-edition/comments.html) of the module initialization function will be applied automatically as the Python docstring of your module.
>  模块初始化函数的 Rust doc comments 会被作为我们模块的 Python docstring

```python
import rust2py

print(rust2py.__doc__)
```

Which means that the above Python code will print `This module is implemented in Rust.`.

## Modules as objects
In Python, modules are first class objects. This means that you can store them as values or add them to dicts or other modules:
>  Python 中，modules 是第一类对象，这意味着我们可以将 modules 像 values 一样存储，以及将它们加到其他 modules 的 dicts 中 (以实现在单个 Rust 拓展模块中实现嵌套的 Python 模块结构)

```rust
#![allow(unused_variables)]
fn main() {
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule};
use pyo3::types::IntoPyDict;

#[pyfunction]
fn subfunction() -> String {
    "Subfunction".to_string()
}

#[pymodule]
fn submodule(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pyfunction!(subfunction))?;
    Ok(())
}

#[pymodule]
fn supermodule(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(submodule))?;
    Ok(())
}

fn nested_call() {
    let gil = GILGuard::acquire();
    let py = gil.python();
    let supermodule = wrap_pymodule!(supermodule)(py);
    let ctx = [("supermodule", supermodule)].into_py_dict(py);

    py.run("assert supermodule.submodule.subfunction() == 'Subfunction'", None, Some(&ctx)).unwrap();
}
}
```

>  `#[pyfunction]` 过程宏将函数转化为可以在 Python 调用的函数，它需要被 `wrap_pyfunction!` 宏包装后添加入 Python 模块才可以调用
>  `wrap_pyfunction!` 会将函数包装为一个 `Py<PyAny>` 类型的对象，即 Python 可调用对象，`module.add_wrapped` 将这个对象添加到模块中，作为属性

>  `#[pymodule]` 过程宏将函数转化为 Python 模块，`wrap_pymodule!` 将子模块封装为 Python 模块对象，使得它可以像普通模块一样导入或嵌套，然后通过 `add_wrapped` 添加到 `supermodule` 中

This way, you can create a module hierarchy within a single extension module.
>  通过这种方式，我们可以在单个拓展模块中创建模块层次结构
