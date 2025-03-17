# Overview
## Instantiating objects with Hydra
One of the best ways to drive different behavior in an application is to instantiate different implementations of an interface. The code using the instantiated object only knows the interface which remains constant, but the behavior is determined by the actual object instance.
>  在应用中驱动不同行为的一种方式是实例化一个接口的不同实现
>  我们编写的使用实例化对象的代码仅根据接口规范进行调用，而调用下的实际行为则由实际对象本身的实现决定

Hydra provides `hydra.utils.instantiate()` (and its alias `hydra.utils.call()`) for instantiating objects and calling functions. Prefer `instantiate` for creating objects and `call` for invoking functions.
>  `hydra.utils.instantiate()` 用于实例化对象，`hydra.utils.call()` 用于调用函数，二者实际是同一过程的两个别名

Call/instantiate supports:

- Constructing an object by calling the `__init__` method
- Calling functions, static functions, class methods and other callable global objects

>  `hydra.utils.instantiate/call()` 支持：
>  - 调用 `__init__` 方法构造一个对象
>  - 调用可调用的全局对象，例如函数、静态函数、类方法

Instantiate API:

```python
def instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
    """pass"""
    
# Alias for instantiate
call = instantiate
```

The config passed to these functions must have a key called `_target_`, with the value of a fully qualified class name, class method, static method or callable. For convenience, `None` config results in a `None` object.

>  向 `instantiate()` 传递的 `config` 必须有一个称为 `_target_` 的 key，它的值可以是一个类名、类方法、静态方法或可调用对象
>  如果 `_target_` 是类名，`instantiate()` 会返回实例化的类对象，如果 `_target_` 是可调用对象的名称，`instantiate()` 会返回调用的返回值
>  `config` 为 `None` 时，`instantiate()` 返回 `None` 对象

**Named arguments** : Config fields (except reserved fields like `_target_`) are passed as named arguments to the target. Named arguments in the config can be overridden by passing named argument with the same name in the `instantiate()` call-site.
>  向 `instantiate()` 传递的 `config` 中的域中，除了向 `_target_, _args_` 这样的保留域，其他的域都会视作要传递给 `_target_` 指定的目标的命名参数 (关键字参数)
>  `config` 中传递给 `_target_` 的命名参数会被 `instantiate()` 签名中的 `**kwargs` 中重新传递的命名参数覆盖

**Positional arguments** : The config may contain a `_args_` field representing positional arguments to pass to the target. The positional arguments can be overridden together by passing positional arguments in the `instantiate()` call-site.
>  `config` 中的 `_args_` 域可以包含需要传递给 `_target_` 的位置参数
>  这些位置参数同样会被 `instantiate()` 签名中 `*args` 中重新传递的位置参数覆盖

### Simple usage
Your application might have an Optimizer class:

Example class

```python
class Optimizer:
    algo: str
    lr: float

    def __init__(self, algo: str, lr: float) -> None:
        self.algo = algo
        self.lr = lr
```

Config

```
optimizer:
  _target_: my_app.Optimizer
  algo: SGD
  lr: 0.01
```

Instantiation

```python
opt = instantiate(cfg.optimizer)
print(opt)
# Optimizer(algo=SGD,lr=0.01)

# override parameters on the call-site
opt = instantiate(cfg.optimizer, lr=0.2)
print(opt)
# Optimizer(algo=SGD,lr=0.2)
```

>  使用时，我们在配置文件中指定好需要实例化的类及其参数，在代码中调用 `instantiate()` ，传入该配置即可

### Recursive instantiation
Let's add a Dataset and a Trainer class. The trainer holds a Dataset and an Optimizer instances.

Additional classes

```python
class Dataset:
    name: str
    path: str

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path


class Trainer:
    def __init__(self, optimizer: Optimizer, dataset: Dataset) -> None:
        self.optimizer = optimizer
        self.dataset = dataset
```

With the following config, you can instantiate the whole thing with a single call:

Example config

```
trainer:
  _target_: my_app.Trainer
  optimizer:
    _target_: my_app.Optimizer
    algo: SGD
    lr: 0.01
  dataset:
    _target_: my_app.Dataset
    name: Imagenet
    path: /datasets/imagenet
```

>  对于本身包含了其他类对象的类，其 `__init__` 中的指向其他类对象的参数在配置文件中的指定方式可以和指定一个类的配置方式一样，即包括了 `_target_` 和其他关键字参数
>  如上例中的 `trainer` 中的 `optimizer` 参数的配置指定方式

Hydra will instantiate nested objects recursively by default.

```python
trainer = instantiate(cfg.trainer)
print(trainer)
# Trainer(
#  optimizer=Optimizer(algo=SGD,lr=0.01),
#  dataset=Dataset(name=Imagenet, path=/datasets/imagenet)
# )
```

>  Hydra 会自行递归地初始化类实例

You can override parameters for nested objects:

```python
trainer = instantiate(
    cfg.trainer,
    optimizer={"lr": 0.3},
    dataset={"name": "cifar10", "path": "/datasets/cifar10"},
)
print(trainer)
# Trainer(
#   optimizer=Optimizer(algo=SGD,lr=0.3),
#   dataset=Dataset(name=cifar10, path=/datasets/cifar10)
# )
```

>  并且，也支持在 `instantiate()` 中以类似 C++ 初始化列表的格式额外指定需要覆盖的对象的初始化参数

Similarly, positional arguments of nested objects can be overridden:

```python
obj = instantiate(
    cfg.object,
    # pass 1 and 2 as positional arguments to the target object
    1, 2,  
    # pass 3 and 4 as positional arguments to a nested child object
    child={"_args_": [3, 4]},
)
```

### Disable recursive instantiation
You can disable recursive instantiation by setting `_recursive_` to `False` in the config node or in the call-site.
In that case the Trainer object will receive an `OmegaConf DictConfig` for nested dataset and optimizer instead of the instantiated objects.

```python
optimizer = instantiate(cfg.trainer, _recursive_=False)
print(optimizer)
```

>  设定 `_recursive_=False` 后，Hydra 不会执行递归实例化
>  上例中，关键字参数 `optimizer, dataset` 不会收到实例化后的对象，而是收到一个配置字典

Output:

```
Trainer(
  optimizer={
    '_target_': 'my_app.Optimizer', 'algo': 'SGD', 'lr': 0.01
  },
  dataset={
    '_target_': 'my_app.Dataset', 'name': 'Imagenet', 'path': '/datasets/imagenet'
  }
)
```

### Parameter conversion strategies
By default, the parameters passed to the target are either primitives (int, float, bool etc) or OmegaConf containers (`DictConfig`, `ListConfig`). OmegaConf containers have many advantages over primitive dicts and lists, including convenient attribute access for keys, [duck-typing as instances of dataclasses or attrs classes](https://omegaconf.readthedocs.io/en/latest/structured_config.html), and support for [variable interpolation](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation) and [custom resolvers](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html). If the callable targeted by `instantiate` leverages OmegaConf's features, it will make sense to pass `DictConfig` and `ListConfig` instances directly to that callable.

That being said, in many cases it's desired to pass normal Python dicts and lists, rather than `DictConfig` or `ListConfig` instances, as arguments to your callable. You can change instantiate's argument conversion strategy using the `_convert_` parameter. Supported values are:

- `"none"` : Default behavior, Use OmegaConf containers
- `"partial"` : Convert OmegaConf containers to dict and list, except Structured Configs, which remain as DictConfig instances.
- `"object"` : Convert OmegaConf containers to dict and list, except Structured Configs, which are converted to instances of the backing dataclass / attr class using `OmegaConf.to_object`.
- `"all"` : Convert everything to primitive containers

The conversion strategy applies recursively to all subconfigs of the instantiation target. Here is an example demonstrating the various conversion strategies:

```
from dataclasses import dataclassfrom omegaconf import DictConfig, OmegaConffrom hydra.utils import instantiate@dataclassclass Foo:    a: int = 123class MyTarget:    def __init__(self, foo, bar):        self.foo = foo        self.bar = barcfg = OmegaConf.create(    {        "_target_": "__main__.MyTarget",        "foo": Foo(),        "bar": {"b": 456},    })obj_none = instantiate(cfg, _convert_="none")assert isinstance(obj_none, MyTarget)assert isinstance(obj_none.foo, DictConfig)assert isinstance(obj_none.bar, DictConfig)obj_partial = instantiate(cfg, _convert_="partial")assert isinstance(obj_partial, MyTarget)assert isinstance(obj_partial.foo, DictConfig)assert isinstance(obj_partial.bar, dict)obj_object = instantiate(cfg, _convert_="object")assert isinstance(obj_object, MyTarget)assert isinstance(obj_object.foo, Foo)assert isinstance(obj_object.bar, dict)obj_all = instantiate(cfg, _convert_="all")assert isinstance(obj_all, MyTarget)assert isinstance(obj_all.foo, dict)assert isinstance(obj_all.bar, dict)
```

Passing the `_convert_` keyword argument to `instantiate` has the same effect as defining a `_convert_` attribute on your config object. Here is an example creating instances of `MyTarget` that are equivalent to the above:

```
cfg_none = OmegaConf.create({..., "_convert_": "none"})obj_none = instantiate(cfg_none)cfg_partial = OmegaConf.create({..., "_convert_": "partial"})obj_partial = instantiate(cfg_partial)cfg_object = OmegaConf.create({..., "_convert_": "object"})obj_object = instantiate(cfg_object)cfg_all = OmegaConf.create({..., "_convert_": "all"})obj_all = instantiate(cfg_all)
```

### Partial Instantiation

Sometimes you may not set all parameters needed to instantiate an object from the configuration, in this case you can set `_partial_` to be `True` to get a `functools.partial` wrapped object or method, then complete initializing the object in the application code. Here is an example:

Example classes

```
class Optimizer:    algo: str    lr: float    def __init__(self, algo: str, lr: float) -> None:        self.algo = algo        self.lr = lr    def __repr__(self) -> str:        return f"Optimizer(algo={self.algo},lr={self.lr})"class Model:    def __init__(self, optim_partial: Any, lr: float):        super().__init__()        self.optim = optim_partial(lr=lr)        self.lr = lr    def __repr__(self) -> str:        return f"Model(Optimizer={self.optim},lr={self.lr})"
```

Config

```
model:  _target_: my_app.Model  optim_partial:    _partial_: true    _target_: my_app.Optimizer    algo: SGD  lr: 0.01
```

Instantiation

```
model = instantiate(cfg.model)print(model)# "Model(Optimizer=Optimizer(algo=SGD,lr=0.01),lr=0.01)
```

If you are repeatedly instantiating the same config, using `_partial_=True` may provide a significant speedup as compared with regular (non-partial) instantiation.

```
factory = instantiate(config, _partial_=True)obj = factory()
```

In the above example, repeatedly calling `factory` would be faster than repeatedly calling `instantiate(config)`. A caveat of this approach is that the same keyword arguments would be re-used in each call to `factory`.

```
class Foo:    ...class Bar:    def __init__(self, foo):        self.foo = foobar_conf = {    "_target_": "__main__.Bar",    "foo": {"_target_": "__main__.Foo"},}bar_factory = instantiate(bar_conf, _partial_=True)bar1 = bar_factory()bar2 = bar_factory()assert bar1 is not bar2assert bar1.foo is bar2.foo  # the `Foo` instance is re-used here
```

This does not apply if `_partial_=False`, in which case a new `Foo` instance would be created with each call to `instantiate`.

### Instantiation of builtins

The value of `_target_` passed to `instantiate` should be a "dotpath" pointing to some callable that can be looked up via a combination of `import` and `getattr`. If you want to target one of Python's [built-in functions](https://docs.python.org/3/library/functions.html) (such as `len` or `print` or `divmod`), you will need to provide a dotpath looking up that function in Python's [`builtins`](https://docs.python.org/3/library/builtins.html) module.

```
from hydra.utils import instantiate# instantiate({"_target_": "len"}, [1,2,3])  # this gives an InstantiationExceptioninstantiate({"_target_": "builtins.len"}, [1,2,3])  # this works, returns the number 3
```

### Dotpath lookup machinery

Hydra looks up a given `_target_` by attempting to find a module that corresponds to a prefix of the given dotpath and then looking for an object in that module corresponding to the dotpath's tail. For example, to look up a `_target_` given by the dotpath `"my_module.my_nested_module.my_object"`, hydra first locates the module `my_module.my_nested_module`, then find `my_object` inside that nested module.

Hydra exposes an API allowing direct use of this dotpath lookup machinery. The following three functions, which can be imported from the

hydra.utils

module, accept a string-typed dotpath as an argument and return the located class/callable/object:

```
def get_class(path: str) -> type:    """    Look up a class based on a dotpath.    Fails if the path does not point to a class.    >>> import my_module    >>> from hydra.utils import get_class    >>> assert get_class("my_module.MyClass") is my_module.MyClass    """    ...def get_method(path: str) -> Callable[..., Any]:    """    Look up a callable based on a dotpath.    Fails if the path does not point to a callable object.    >>> import my_module    >>> from hydra.utils import get_method    >>> assert get_method("my_module.my_function") is my_module.my_function    """    ...# Alias for get_methodget_static_method = get_methoddef get_object(path: str) -> Any:    """    Look up a callable based on a dotpath.    >>> import my_module    >>> from hydra.utils import get_object    >>> assert get_object("my_module.my_object") is my_module.my_object    """    ...
```

[Edit this page](https://github.com/facebookresearch/hydra/edit/main/website/docs/advanced/instantiate_objects/overview.md)

Last updated on **Mar 16, 2025** by **dependabot[bot]**

[  
](https://hydra.cc/docs/advanced/overriding_packages/)