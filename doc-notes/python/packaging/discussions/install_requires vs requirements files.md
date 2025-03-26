# install_requires
`install_requires` is a [Setuptools](https://packaging.python.org/en/latest/key_projects/#setuptools) `setup.py` keyword that should be used to specify what a project **minimally** needs to run correctly. When the project is installed by [pip](https://packaging.python.org/en/latest/key_projects/#pip), this is the specification that is used to install its dependencies.

>  `install_requires` 是 `setuptools` 中 `setup.py` 的关键字，用于指定项目确保正常运行需要的 “最小” 依赖
>  当项目通过 `pip` 安装时，该规范会被用于安装其依赖项

For example, if the project requires A and B, your `install_requires` would be like so:

```
install_requires=[
   'A',
   'B'
]
```

Additionally, it’s best practice to indicate any known lower or upper bounds.

For example, it may be known, that your project requires at least v1 of ‘A’, and v2 of ‘B’, so it would be like so:

```
install_requires=[
   'A>=1',
   'B>=2'
]
```

It may also be known that project ‘A’ introduced a change in its v2 that breaks the compatibility of your project with v2 of ‘A’ and later, so it makes sense to not allow v2:

```
install_requires=[
   'A>=1,<2',
   'B>=2'
]
```

>  `install_requires` 的示例如上，可以指定版本范围

It is not considered best practice to use `install_requires` to pin dependencies to specific versions, or to specify sub-dependencies (i.e. dependencies of your dependencies). This is overly-restrictive, and prevents the user from gaining the benefit of dependency upgrades.
>  不建议在 `install_requires` 中将依赖固定在特定版本，也不建议指定自依赖项 (即依赖的依赖项)，这样的限制太强，会防止用户从依赖项升级中受益

Lastly, it’s important to understand that `install_requires` is a listing of “Abstract” requirements, i.e just names and version restrictions that don’t determine where the dependencies will be fulfilled from (i.e. from what index or source). The where (i.e. how they are to be made “Concrete”) is to be determined at install time using [pip](https://packaging.python.org/en/latest/key_projects/#pip) options. [1](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/#id4)
>  需要理解 `install_requires` 是一个 "抽象" 需求的列表，也就是其中的需求即包含了名称和版本限制，但并不决定依赖将从哪里获取 (即来自哪个索引或者源)
>  具体在何处安装依赖项 (将这些需求变得 “具体”) 将在安装时通过 `pip` 的选项决定

# Requirements files
[Requirements Files](https://pip.pypa.io/en/latest/user_guide/#requirements-files "(in pip v25.1)") described most simply, are just a list of [pip install](https://pip.pypa.io/en/latest/cli/pip_install/#pip-install "(in pip v25.1)") arguments placed into a file.
>  requirements files 就是将 `pip install` 的参数列表放入了一个文件中

Whereas `install_requires` defines the dependencies for a single project, [Requirements Files](https://pip.pypa.io/en/latest/user_guide/#requirements-files "(in pip v25.1)") are often used to define the requirements for a complete Python environment.
>  `install_requires` 通常定义单个项目的依赖项
>  requirements files 通常用于定义完整 Python 环境的依赖项

Whereas `install_requires` requirements are minimal, requirements files often contain an exhaustive listing of pinned versions for the purpose of achieving [repeatable installations](https://pip.pypa.io/en/latest/topics/repeatable-installs/#repeatability "(in pip v25.1)") of a complete environment.
>  `install_requires` 通常是项目运行的最小依赖
>  requirements 通常包含详细的已锁定版本号，目的是实现完整环境的可重复安装

Whereas `install_requires` requirements are “Abstract”, i.e. not associated with any particular index, requirements files often contain pip options like `--index-url` or `--find-links` to make requirements “Concrete”, i.e. associated with a particular index or directory of packages. [1](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/#id4)
>  `install_requires` 中的需求是抽象的，即不关联特定的索引
>  requirements files 通常会包含 `pip` 选项，例如 `--index-url` 或 `--find-links` ，以使得需求项具体化 (即与特定索引或包目录相关联)

Whereas `install_requires` metadata is automatically analyzed by pip during an install, requirements files are not, and only are used when a user specifically installs them using `python -m pip install -r`.
>  `install_requires` 元数据在包的安装过程中会被 `pip` 自动分析
>  requirements files 则不会，它们仅在用户明确使用 `pip install -r` 时才会被使用

---

[1]([1](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/#id2),[2](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/#id3)) For more on “Abstract” vs “Concrete” requirements, see [https://caremad.io/posts/2013/07/setup-vs-requirement/](https://caremad.io/posts/2013/07/setup-vs-requirement/).

Last updated on Mar 21, 2025