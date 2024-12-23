---
completed: true
version: 8.1.7
---
# Quickstart
You can get the library directly from PyPI:

```
pip install click
```

The installation into a [virtualenv](https://click.palletsprojects.com/en/8.1.x/quickstart/#virtualenv) is heavily recommended.

## virtualenv
Virtualenv is probably what you want to use for developing Click applications.

What problem does virtualenv solve? Chances are that you want to use it for other projects besides your Click script. But the more projects you have, the more likely it is that you will be working with different versions of Python itself, or at least different versions of Python libraries. Let’s face it: quite often libraries break backwards compatibility, and it’s unlikely that any serious application will have zero dependencies. So what do you do if two or more of your projects have conflicting dependencies?

Virtualenv to the rescue! Virtualenv enables multiple side-by-side installations of Python, one for each project. It doesn’t actually install separate copies of Python, but it does provide a clever way to keep different project environments isolated.

Create your project folder, then a virtualenv within it:

```
$ mkdir myproject
$ cd myproject
$ python3 -m venv .venv
```

Now, whenever you want to work on a project, you only have to activate the corresponding environment. On OS X and Linux, do the following:

```
$ . .venv/bin/activate
(venv) $
```

If you are a Windows user, the following command is for you:

```
> .venv\scripts\activate
(venv) >
```

Either way, you should now be using your virtualenv (notice how the prompt of your shell has changed to show the active environment).

And if you want to stop using the virtualenv, use the following command:

```
$ deactivate
```

After doing this, the prompt of your shell should be as familiar as before.

Now, let’s move on. Enter the following command to get Click activated in your virtualenv:

```
$ pip install click
```

A few seconds later and you are good to go.

## Screencast and Examples
There is a screencast available which shows the basic API of Click and how to build simple applications with it. It also explores how to build commands with subcommands.

- [Building Command Line Applications with Click](https://www.youtube.com/watch?v=kNke39OZ2k0)

Examples of Click applications can be found in the documentation as well as in the GitHub repository together with readme files:

- `inout`: [File input and output](https://github.com/pallets/click/tree/main/examples/inout)
- `naval`: [Port of docopt naval example](https://github.com/pallets/click/tree/main/examples/naval)
- `aliases`: [Command alias example](https://github.com/pallets/click/tree/main/examples/aliases)
- `repo`: [Git-/Mercurial-like command line interface](https://github.com/pallets/click/tree/main/examples/repo)
- `complex`: [Complex example with plugin loading](https://github.com/pallets/click/tree/main/examples/complex)
- `validation`: [Custom parameter validation example](https://github.com/pallets/click/tree/main/examples/validation)
- `colors`: [Color support demo](https://github.com/pallets/click/tree/main/examples/colors)
- `termui`: [Terminal UI functions demo](https://github.com/pallets/click/tree/main/examples/termui)
- `imagepipe`: [Multi command chaining demo](https://github.com/pallets/click/tree/main/examples/imagepipe)

## Basic Concepts - Creating a Command
Click is based on declaring commands through decorators. Internally, there is a non-decorator interface for advanced use cases, but it’s discouraged for high-level usage.
> click 基于通过装饰器声明命令

A function becomes a Click command line tool by decorating it through [`click.command()`](https://click.palletsprojects.com/en/8.1.x/api/#click.command "click.command"). At its simplest, just decorating a function with this decorator will make it into a callable script:
> 使用 `click.command()` 装饰一个函数就可以让它成为 Click 命令行工具（即可调用的脚本）

```python
import click

@click.command()
def hello():
    click.echo('Hello World!')
```

What’s happening is that the decorator converts the function into a [`Command`](https://click.palletsprojects.com/en/8.1.x/api/#click.Command "click.Command") which then can be invoked:
> 该装饰器将函数转化为 `Command` ，函数的调用方式不变，但是具备了解析命令行参数的功能，例如解析命令行参数 `--help` ，打印帮助页面

```python
if __name__ == '__main__':
    hello()
```

And what it looks like:

```
$ python hello.py
Hello World!
```

And the corresponding help page:

```
$ python hello.py --help
Usage: hello.py [OPTIONS]

Options:
  --help  Show this message and exit.
```

## Echoing
Why does this example use [`echo()`](https://click.palletsprojects.com/en/8.1.x/api/#click.echo "click.echo") instead of the regular [`print()`](https://docs.python.org/3/library/functions.html#print "(in Python v3.12)") function? The answer to this question is that Click attempts to support different environments consistently and to be very robust even when the environment is misconfigured. Click wants to be functional at least on a basic level even if everything is completely broken.

What this means is that the [`echo()`](https://click.palletsprojects.com/en/8.1.x/api/#click.echo "click.echo") function applies some error correction in case the terminal is misconfigured instead of dying with a [`UnicodeError`](https://docs.python.org/3/library/exceptions.html#UnicodeError "(in Python v3.12)").
> 相较于 `print()` ，`click.echo()` 更具健壮性，可以自行做一些错误修正，例如避免 `UnicodeError`

The echo function also supports color and other styles in output. It will automatically remove styles if the output stream is a file. On Windows, colorama is automatically installed and used. See [ANSI Colors](https://click.palletsprojects.com/en/8.1.x/utils/#ansi-colors).
> `echo()` 支持颜色和风格输出，如果输出流是文件，风格会被自动移除

If you don’t need this, you can also use the `print()` construct / function.

## Nesting Commands
Commands can be attached to other commands of type [`Group`](https://click.palletsprojects.com/en/8.1.x/api/#click.Group "click.Group"). This allows arbitrary nesting of scripts. As an example here is a script that implements two commands for managing databases:
> 命令可以附加到类型为 `Group` 的其他命令上，方便我们进行脚本嵌套

```python
@click.group()
def cli():
    pass

@click.command()
def initdb():
    click.echo('Initialized the database')

@click.command()
def dropdb():
    click.echo('Dropped the database')

cli.add_command(initdb)
cli.add_command(dropdb)
```

As you can see, the [`group()`](https://click.palletsprojects.com/en/8.1.x/api/#click.group "click.group") decorator works like the [`command()`](https://click.palletsprojects.com/en/8.1.x/api/#click.command "click.command") decorator, but creates a [`Group`](https://click.palletsprojects.com/en/8.1.x/api/#click.Group "click.Group") object instead which can be given multiple subcommands that can be attached with [`Group.add_command()`](https://click.palletsprojects.com/en/8.1.x/api/#click.Group.add_command "click.Group.add_command").
> `group()` 装饰器会创建 `Group` 对象，`Group` 对象可以被附加多个子命令

For simple scripts, it’s also possible to automatically attach and create a command by using the [`Group.command()`](https://click.palletsprojects.com/en/8.1.x/api/#click.Group.command "click.Group.command") decorator instead. The above script can instead be written like this:
> 使用 `Gropu.command()` 装饰器装饰函数可以直接为其创建命令并将其附加为 `Group` 对象的子命令

```python
@click.group()
def cli():
    pass

@cli.command()
def initdb():
    click.echo('Initialized the database')

@cli.command()
def dropdb():
    click.echo('Dropped the database')
```

You would then invoke the [`Group`](https://click.palletsprojects.com/en/8.1.x/api/#click.Group "click.Group") in your setuptools entry points or other invocations:

```python
if __name__ == '__main__':
    cli()
```

## Registering Commands Later
Instead of using the `@group.command()` decorator, commands can be decorated with the plain `@click.command()` decorator and registered with a group later with `group.add_command()`. This could be used to split commands into multiple Python modules.

```python
@click.command()
def greet():
    click.echo("Hello, World!")

@click.group()
def group():
    pass

group.add_command(greet)
```

## Adding Parameters
To add parameters, use the [`option()`](https://click.palletsprojects.com/en/8.1.x/api/#click.option "click.option") and [`argument()`](https://click.palletsprojects.com/en/8.1.x/api/#click.argument "click.argument") decorators:
> 通过 `option()/argument()` 装饰器可以为 `Command` 添加参数

```python
@click.command()
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def hello(count, name):
    for x in range(count):
        click.echo(f"Hello {name}!")
```

What it looks like:
> `Command` 相应的帮助文档也会更新

```
$ python hello.py --help
Usage: hello.py [OPTIONS] NAME

Options:
  --count INTEGER  number of greetings
  --help           Show this message and exit.
```

## Switching to Setuptools
In the code you wrote so far there is a block at the end of the file which looks like this: `if __name__ == '__main__':`. This is traditionally how a standalone Python file looks like. With Click you can continue doing that, but there are better ways through setuptools.

There are two main (and many more) reasons for this:

The first one is that setuptools automatically generates executable wrappers for Windows so your command line utilities work on Windows too.

The second reason is that setuptools scripts work with virtualenv on Unix without the virtualenv having to be activated. This is a very useful concept which allows you to bundle your scripts with all requirements into a virtualenv.

Click is perfectly equipped to work with that and in fact the rest of the documentation will assume that you are writing applications through setuptools.

I strongly recommend to have a look at the [Setuptools Integration](https://click.palletsprojects.com/en/8.1.x/setuptools/#setuptools-integration) chapter before reading the rest as the examples assume that you will be using setuptools.

# Setuptools Integration
When writing command line utilities, it’s recommended to write them as modules that are distributed with setuptools instead of using Unix shebangs.
> 编写命令行实用程序时，推荐将它们写为模块，然后使用 setuptools 发布

Why would you want to do that? There are a bunch of reasons:

1. One of the problems with the traditional approach is that the first module the Python interpreter loads has an incorrect name. This might sound like a small issue but it has quite significant implications.
    
    The first module is not called by its actual name, but the interpreter renames it to `__main__`. While that is a perfectly valid name it means that if another piece of code wants to import from that module it will trigger the import a second time under its real name and all of a sudden your code is imported twice.
    
2. Not on all platforms are things that easy to execute. On Linux and OS X you can add a comment to the beginning of the file (`#!/usr/bin/env python`) and your script works like an executable (assuming it has the executable bit set). This however does not work on Windows. While on Windows you can associate interpreters with file extensions (like having everything ending in `.py` execute through the Python interpreter) you will then run into issues if you want to use the script in a virtualenv.
    
    In fact running a script in a virtualenv is an issue with OS X and Linux as well. With the traditional approach you need to have the whole virtualenv activated so that the correct Python interpreter is used. Not very user friendly.
    
3. The main trick only works if the script is a Python module. If your application grows too large and you want to start using a package you will run into issues.

## Introduction
To bundle your script with setuptools, all you need is the script in a Python package and a `setup.py` file.

Imagine this directory structure:

```
yourscript.py
setup.py
```

Contents of `yourscript.py`:

```python
import click

@click.command()
def cli():
    """Example script."""
    click.echo('Hello World!')
```

Contents of `setup.py`:

```python
from setuptools import setup

setup(
    name='yourscript',
    version='0.1.0',
    py_modules=['yourscript'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'yourscript = yourscript:cli',
        ],
    },
)
```

The magic is in the `entry_points` parameter. Read the full [entry_points](https://packaging.python.org/en/latest/specifications/entry-points/) specification for more details. Below `console_scripts`, each line identifies one console script. The first part before the equals sign (`=`) is the name of the script that should be generated, the second part is the import path followed by a colon (`:`) with the Click command.

That’s it.

## Testing The Script
To test the script, you can make a new virtualenv and then install your package:

```
$ python3 -m venv .venv
$ . .venv/bin/activate
$ pip install --editable .
```

Afterwards, your command should be available:

```
$ yourscript
Hello World!
```

## Scripts in Packages
If your script is growing and you want to switch over to your script being contained in a Python package, the changes necessary are minimal. Let’s assume your directory structure changed to this:

```
project/
    yourpackage/
        __init__.py
        main.py
        utils.py
        scripts/
            __init__.py
            yourscript.py
    setup.py
```

In this case instead of using `py_modules` in your `setup.py` file you can use `packages` and the automatic package finding support of setuptools. In addition to that it’s also recommended to include other package data.

These would be the modified contents of `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name='yourpackage',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'yourscript = yourpackage.scripts.yourscript:cli',
        ],
    },
)
```
