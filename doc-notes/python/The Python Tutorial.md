The Python interpreter and the extensive standard library are freely available in source or binary form for all major platforms from the Python web site, [https://www.python.org/](https://www.python.org/), and may be freely distributed. The same site also contains distributions of and pointers to many free third party Python modules, programs and tools, and additional documentation.
> Python 解释器和标准库可以任意分发

The Python interpreter is easily extended with new functions and data types implemented in C or C++ (or other languages callable from C). Python is also suitable as an extension language for customizable applications.
> Python 解释器可以被 C/C++ 实现的函数和数据类型拓展

For a description of standard objects and modules, see [The Python Standard Library](https://docs.python.org/3/library/index.html#library-index). [The Python Language Reference](https://docs.python.org/3/reference/index.html#reference-index) gives a more formal definition of the language. To write extensions in C or C++, read [Extending and Embedding the Python Interpreter](https://docs.python.org/3/extending/index.html#extending-index) and [Python/C API Reference Manual](https://docs.python.org/3/c-api/index.html#c-api-index). There are also several books covering Python in depth.

This tutorial does not attempt to be comprehensive and cover every single feature, or even every commonly used feature. Instead, it introduces many of Python’s most noteworthy features, and will give you a good idea of the language’s flavor and style. After reading it, you will be able to read and write Python modules and programs, and you will be ready to learn more about the various Python library modules described in [The Python Standard Library](https://docs.python.org/3/library/index.html#library-index).

The [Glossary](https://docs.python.org/3/glossary.html#glossary) is also worth going through.

# 1. Whetting Your Appetite
If you do much work on computers, eventually you find that there’s some task you’d like to automate. For example, you may wish to perform a search-and-replace over a large number of text files, or rename and rearrange a bunch of photo files in a complicated way. Perhaps you’d like to write a small custom database, or a specialized GUI application, or a simple game.

If you’re a professional software developer, you may have to work with several C/C++/Java libraries but find the usual write/compile/test/re-compile cycle is too slow. Perhaps you’re writing a test suite for such a library and find writing the testing code a tedious task. Or maybe you’ve written a program that could use an extension language, and you don’t want to design and implement a whole new language for your application.

Python is just the language for you.

You could write a Unix shell script or Windows batch files for some of these tasks, but shell scripts are best at moving around files and changing text data, not well-suited for GUI applications or games. You could write a C/C++/Java program, but it can take a lot of development time to get even a first-draft program. Python is simpler to use, available on Windows, macOS, and Unix operating systems, and will help you get the job done more quickly.

Python is simple to use, but it is a real programming language, offering much more structure and support for large programs than shell scripts or batch files can offer. On the other hand, Python also offers much more error checking than C, and, being a _very-high-level language_, it has high-level data types built in, such as flexible arrays and dictionaries. Because of its more general data types Python is applicable to a much larger problem domain than Awk or even Perl, yet many things are at least as easy in Python as in those languages.
> Python 提供了比 C 多的错误检查

Python allows you to split your program into modules that can be reused in other Python programs. It comes with a large collection of standard modules that you can use as the basis of your programs — or as examples to start learning to program in Python. Some of these modules provide things like file I/O, system calls, sockets, and even interfaces to graphical user interface toolkits like Tk.
> Python 允许将程序分解为多个可以复用的模块，并提供了大量标准模块，包括文件 IO、系统调用、套接字

Python is an interpreted language, which can save you considerable time during program development because no compilation and linking is necessary. The interpreter can be used interactively, which makes it easy to experiment with features of the language, to write throw-away programs, or to test functions during bottom-up program development. It is also a handy desk calculator.
> Python 为解释性语言，不需要编译和链接
> 可以交互式使用解释器

Python enables programs to be written compactly and readably. Programs written in Python are typically much shorter than equivalent C, C++, or Java programs, for several reasons:

- the high-level data types allow you to express complex operations in a single statement;
- statement grouping is done by indentation instead of beginning and ending brackets;
- no variable or argument declarations are necessary.

> Python 无需变量或参数声明

Python is _extensible_: if you know how to program in C it is easy to add a new built-in function or module to the interpreter, either to perform critical operations at maximum speed, or to link Python programs to libraries that may only be available in binary form (such as a vendor-specific graphics library). Once you are really hooked, you can link the Python interpreter into an application written in C and use it as an extension or command language for that application.
> Python 是可拓展的，可以用 C 程序为解释器添加新的内嵌函数或模块，用于执行特定的操作，也可以用 C 程序将 Python 链接到仅有二进制形式的库
> 还可以将 Python 解释器链接到 C 实现的程序，作为该程序的拓展或命令语言

By the way, the language is named after the BBC show “Monty Python’s Flying Circus” and has nothing to do with reptiles. Making references to Monty Python skits in documentation is not only allowed, it is encouraged!

Now that you are all excited about Python, you’ll want to examine it in some more detail. Since the best way to learn a language is to use it, the tutorial invites you to play with the Python interpreter as you read.

In the next chapter, the mechanics of using the interpreter are explained. This is rather mundane information, but essential for trying out the examples shown later.

The rest of the tutorial introduces various features of the Python language and system through examples, beginning with simple expressions, statements and data types, through functions and modules, and finally touching upon advanced concepts like exceptions and user-defined classes.

# 2. Using the Python Interpreter
## 2.1. Invoking the Interpreter
The Python interpreter is usually installed as `/usr/local/bin/python3.12` on those machines where it is available; putting `/usr/local/bin` in your Unix shell’s search path makes it possible to start it by typing the command:

```
python3.12
```

to the shell.  [1](https://docs.python.org/3/tutorial/interpreter.html#id2)  Since the choice of the directory where the interpreter lives is an installation option, other places are possible; check with your local Python guru or system administrator. (E.g., `/usr/local/python` is a popular alternative location.)

On Windows machines where you have installed Python from the [Microsoft Store](https://docs.python.org/3/using/windows.html#windows-store), the `python3.12` command will be available. If you have the [py.exe launcher](https://docs.python.org/3/using/windows.html#launcher) installed, you can use the `py` command. See [Excursus: Setting environment variables](https://docs.python.org/3/using/windows.html#setting-envvars) for other ways to launch Python.

Typing an end-of-file character (Control-D on Unix, Control-Z on Windows) at the primary prompt causes the interpreter to exit with a zero exit status. If that doesn’t work, you can exit the interpreter by typing the following command: `quit()`.
> end-of-file(ctrl+D on Unix, ctrl+Z on Windows) 会让解释器退出，返回值 0
> `quit()` 等效

The interpreter’s line-editing features include interactive editing, history substitution and code completion on systems that support the [GNU Readline](https://tiswww.case.edu/php/chet/readline/rltop.html) library. Perhaps the quickest check to see whether command line editing is supported is typing Control-P to the first Python prompt you get. If it beeps, you have command line editing; see Appendix [Interactive Input Editing and History Substitution](https://docs.python.org/3/tutorial/interactive.html#tut-interacting) for an introduction to the keys. If nothing appears to happen, or if `^P` is echoed, command line editing isn’t available; you’ll only be able to use backspace to remove characters from the current line.
> 解释器的命令行编辑特性包括支持历史替换，以及在支持 GNU Readline 的机器上支持代码补全

The interpreter operates somewhat like the Unix shell: when called with standard input connected to a tty device, it reads and executes commands interactively; when called with a file name argument or with a file as standard input, it reads and executes a _script_ from that file.
> 解释器和 Unix shell 的工作方式类似，当使用 standard input 调用时，解释器交互式执行命令；当使用文件名作为参数或者以文件作为 standard input 调用时，解释器执行该文件脚本

A second way of starting the interpreter is `python -c command [arg] ...`, which executes the statement(s) in _command_, analogous to the shell’s [`-c`](https://docs.python.org/3/using/cmdline.html#cmdoption-c) option. Since Python statements often contain spaces or other characters that are special to the shell, it is usually advised to quote _command_ in its entirety.
> `python -c command [arg] ...` 执行 `command` 中的语句
> 一般建议将 `command` 引号括起，防止 shell 将其中的空白字符等替换

Some Python modules are also useful as scripts. These can be invoked using `python -m module [arg] ...`, which executes the source file for _module_ as if you had spelled out its full name on the command line.
> Python 模块也可以作为脚本调用，`python -m module [arg] ...`  即执行将该模块的源文件作为脚本执行

When a script file is used, it is sometimes useful to be able to run the script and enter interactive mode afterwards. This can be done by passing [`-i`](https://docs.python.org/3/using/cmdline.html#cmdoption-i) before the script.
> 执行脚本时，传入 `-i` 可以进行交互式执行

All command line options are described in [Command line and environment](https://docs.python.org/3/using/cmdline.html#using-on-general).

### 2.1.1. Argument Passing
When known to the interpreter, the script name and additional arguments thereafter are turned into a list of strings and assigned to the `argv` variable in the `sys` module. You can access this list by executing `import sys`. The length of the list is at least one; when no script and no arguments are given, `sys.argv[0]` is an empty string. When the script name is given as `'-'` (meaning standard input), `sys.argv[0]` is set to `'-'`. When [`-c`](https://docs.python.org/3/using/cmdline.html#cmdoption-c) _command_ is used, `sys.argv[0]` is set to `'-c'`. When [`-m`](https://docs.python.org/3/using/cmdline.html#cmdoption-m) _module_ is used, `sys.argv[0]` is set to the full name of the located module. Options found after [`-c`](https://docs.python.org/3/using/cmdline.html#cmdoption-c) _command_ or [`-m`](https://docs.python.org/3/using/cmdline.html#cmdoption-m) _module_ are not consumed by the Python interpreter’s option processing but left in `sys.argv` for the command or module to handle.
> 解释器将脚本名称和参数 (都转化为字符串) 放入 `sys` 模块的列表变量 `argv` 中
> `argv` 的长度至少为 1，当没有脚本和参数给定，`sys.argv[0]` 是一个空字符串
> 若给定脚本名称为 `-` ( 标准输入 )，`sys.argv[0] == -` 
> 当使用 `-c` ，`sys.argv[0] == -c`
> 当使用 `-m` ，`sys.argv[0]` 为定位到的模块的全名
> `-c/-m` 之后的选项不会被 Python 解释器处理，但是会留在 `sys.argv` 中

### 2.1.2. Interactive Mode
When commands are read from a tty, the interpreter is said to be in _interactive mode_. In this mode it prompts for the next command with the _primary prompt_, usually three greater-than signs (`>>>`); for continuation lines it prompts with the _secondary prompt_, by default three dots (`...`). The interpreter prints a welcome message stating its version number and a copyright notice before printing the first prompt:
> 若命令从 tty 读取，则解释器处于交互模式

```
$ python3.12
Python 3.12 (default, April 4 2022, 09:25:04)
[GCC 10.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>

Continuation lines are needed when entering a multi-line construct. As an example, take a look at this [`if`](https://docs.python.org/3/reference/compound_stmts.html#if) statement:

>>>

>>> the_world_is_flat = True
>>> if the_world_is_flat:
...     print("Be careful not to fall off!")
...
Be careful not to fall off!
```

For more on interactive mode, see [Interactive Mode](https://docs.python.org/3/tutorial/appendix.html#tut-interac).

## 2.2. The Interpreter and Its Environment
### 2.2.1. Source Code Encoding
By default, Python source files are treated as encoded in UTF-8. In that encoding, characters of most languages in the world can be used simultaneously in string literals, identifiers and comments — although the standard library only uses ASCII characters for identifiers, a convention that any portable code should follow. To display all these characters properly, your editor must recognize that the file is UTF-8, and it must use a font that supports all the characters in the file.
> Python 源文件默认被视为 UTF-8编码，因此大多数语言的字符可以作为字符串字面值、标识符、注释
> Python 标准库仅使用 ASCII 字符作为标识符

To declare an encoding other than the default one, a special comment line should be added as the _first_ line of the file. The syntax is as follows:
> 在文件的第一行可以声明使用特定的编码

```
# -*- coding: encoding -*-
```

where _encoding_ is one of the valid [`codecs`](https://docs.python.org/3/library/codecs.html#module-codecs "codecs: Encode and decode data and streams.") supported by Python.

For example, to declare that Windows-1252 encoding is to be used, the first line of your source code file should be:

```
# -*- coding: cp1252 -*-
```

One exception to the _first line_ rule is when the source code starts with a [UNIX “shebang” line](https://docs.python.org/3/tutorial/appendix.html#tut-scripts). In this case, the encoding declaration should be added as the second line of the file. For example:
> 若第一行被 UNIX “shebang” 占用，则在第二行声明

```
#!/usr/bin/env python3
# -*- coding: cp1252 -*-
```

Footnotes
[1] On Unix, the Python 3.x interpreter is by default not installed with the executable named `python`, so that it does not conflict with a simultaneously installed Python 2.x executable.

# 3. An Informal Introduction to Python
Many of the examples in this manual, even those entered at the interactive prompt, include comments. Comments in Python start with the hash character, `#`, and extend to the end of the physical line. A comment may appear at the start of a line or following whitespace or code, but not within a string literal. A hash character within a string literal is just a hash character. Since comments are to clarify code and are not interpreted by Python, they may be omitted when typing in examples.

Some examples:
```
# this is the first comment
spam = 1  # and this is the second comment
          # ... and now a third!
text = "# This is not a comment because it's inside quotes."
```

## 3.1. Using Python as a Calculator
Let’s try some simple Python commands. Start the interpreter and wait for the primary prompt, `>>>`. (It shouldn’t take long.)

### 3.1.1. Numbers
The interpreter acts as a simple calculator: you can type an expression at it and it will write the value. Expression syntax is straightforward: the operators `+`, `-`, `*` and `/` can be used to perform arithmetic; parentheses (`()`) can be used for grouping. For example:

```
>>> 2 + 2
4
>>> 50 - 5*6
20
>>> (50 - 5*6) / 4
5.0
>>> 8 / 5  # division always returns a floating-point number
1.6
```

The integer numbers (e.g. `2`, `4`, `20`) have type [`int`](https://docs.python.org/3/library/functions.html#int "int"), the ones with a fractional part (e.g. `5.0`, `1.6`) have type [`float`](https://docs.python.org/3/library/functions.html#float "float"). We will see more about numeric types later in the tutorial.

Division (`/`) always returns a float. To do [floor division](https://docs.python.org/3/glossary.html#term-floor-division) and get an integer result you can use the `//` operator; to calculate the remainder you can use `%`

```
>>> 17 / 3  # classic division returns a float
5.666666666666667
>>>
>>> 17 // 3  # floor division discards the fractional part
5
>>> 17 % 3  # the % operator returns the remainder of the division
2
>>> 5 * 3 + 2  # floored quotient * divisor + remainder
17
```

With Python, it is possible to use the `**` operator to calculate powers [1](https://docs.python.org/3/tutorial/introduction.html#id3):

```
>>> 5 ** 2  # 5 squared
25
>>> 2 ** 7  # 2 to the power of 7
128
```

The equal sign (`=`) is used to assign a value to a variable. Afterwards, no result is displayed before the next interactive prompt:

```
>>> width = 20
>>> height = 5 * 9
>>> width * height
900
```

If a variable is not “defined” (assigned a value), trying to use it will give you an error:

```
>>> n  # try to access an undefined variable
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'n' is not defined
```

There is full support for floating point; operators with mixed type operands convert the integer operand to floating point:

```
>>> 4 * 3.75 - 1
14.0
```

In interactive mode, the last printed expression is assigned to the variable `_`. This means that when you are using Python as a desk calculator, it is somewhat easier to continue calculations, for example:
> 交互模式中，上一次打印出的表达式会赋值给变量 `_`

```
>>> tax = 12.5 / 100
>>> price = 100.50
>>> price * tax
12.5625
>>> price + _
113.0625
>>> round(_, 2)
113.06
```

This variable should be treated as read-only by the user. Don’t explicitly assign a value to it — you would create an independent local variable with the same name masking the built-in variable with its magic behavior.
> 用户应该视该变量为只读

In addition to [`int`](https://docs.python.org/3/library/functions.html#int "int") and [`float`](https://docs.python.org/3/library/functions.html#float "float"), Python supports other types of numbers, such as [`Decimal`](https://docs.python.org/3/library/decimal.html#decimal.Decimal "decimal.Decimal") and [`Fraction`](https://docs.python.org/3/library/fractions.html#fractions.Fraction "fractions.Fraction"). Python also has built-in support for [complex numbers](https://docs.python.org/3/library/stdtypes.html#typesnumeric), and uses the `j` or `J` suffix to indicate the imaginary part (e.g. `3+5j`).

### 3.1.2. Text
Python can manipulate text (represented by type [`str`](https://docs.python.org/3/library/stdtypes.html#str "str"), so-called “strings”) as well as numbers. This includes characters “`!`”, words “`rabbit`”, names “`Paris`”, sentences “`Got your back.`”, etc. “`Yay! :)`”. They can be enclosed in single quotes (`'...'`) or double quotes (`"..."`) with the same result [2](https://docs.python.org/3/tutorial/introduction.html#id4).

```
>>> 'spam eggs'  # single quotes
'spam eggs'
>>> "Paris rabbit got your back :)! Yay!"  # double quotes
'Paris rabbit got your back :)! Yay!'
>>> '1975'  # digits and numerals enclosed in quotes are also strings
'1975'
```

To quote a quote, we need to “escape” it, by preceding it with `\`. Alternatively, we can use the other type of quotation marks:

```
>>> 'doesn\'t'  # use \' to escape the single quote...
"doesn't"
>>> "doesn't"  # ...or use double quotes instead
"doesn't"
>>> '"Yes," they said.'
'"Yes," they said.'
>>> "\"Yes,\" they said."
'"Yes," they said.'
>>> '"Isn\'t," they said.'
'"Isn\'t," they said.'
```

In the Python shell, the string definition and output string can look different. The [`print()`](https://docs.python.org/3/library/functions.html#print "print") function produces a more readable output, by omitting the enclosing quotes and by printing escaped and special characters:

```
>>> s = 'First line.\nSecond line.'  # \n means newline
>>> s  # without print(), special characters are included in the string
'First line.\nSecond line.'
>>> print(s)  # with print(), special characters are interpreted, so \n produces new line
First line.
Second line.
```

If you don’t want characters prefaced by `\` to be interpreted as special characters, you can use _raw strings_ by adding an `r` before the first quote:

```
>>> print('C:\some\name')  # here \n means newline!
C:\some
ame
>>> print(r'C:\some\name')  # note the r before the quote
C:\some\name
```

There is one subtle aspect to raw strings: a raw string may not end in an odd number of `\` characters; see [the FAQ entry](https://docs.python.org/3/faq/programming.html#faq-programming-raw-string-backslash) for more information and workarounds.
> 原始字符串 raw strings 不能以奇数个 `\` 作为结尾

String literals can span multiple lines. One way is using triple-quotes: `"""..."""` or `'''...'''`. End of lines are automatically included in the string, but it’s possible to prevent this by adding a `\` at the end of the line. The following example:
> 可以创建多行的字符串字面值，其中可以用 `\` 防止出现换行符

```
print("""\
Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
""")
```

produces the following output (note that the initial newline is not included):

```
Usage: thingy [OPTIONS]
     -h                        Display this usage message
     -H hostname               Hostname to connect to
```

Strings can be concatenated (glued together) with the `+` operator, and repeated with `*`:

```
>>> # 3 times 'un', followed by 'ium'
>>> 3 * 'un' + 'ium'
'unununium'
```

Two or more _string literals_ (i.e. the ones enclosed between quotes) next to each other are automatically concatenated.
> 相邻的字符串字面值会被自动连接

```
>>> 'Py' 'thon'
'Python'
```

This feature is particularly useful when you want to break long strings:

```
>>> text = ('Put several strings within parentheses '
...         'to have them joined together.')
>>> text
'Put several strings within parentheses to have them joined together.'
```

This only works with two literals though, not with variables or expressions:
> 但变量不行

```
>>> prefix = 'Py'
>>> prefix 'thon'  # can't concatenate a variable and a string literal
  File "<stdin>", line 1
    prefix 'thon'
           ^^^^^^
SyntaxError: invalid syntax
>>> ('un' * 3) 'ium'
  File "<stdin>", line 1
    ('un' * 3) 'ium'
               ^^^^^
SyntaxError: invalid syntax
```

If you want to concatenate variables or a variable and a literal, use `+`:

```
>>> prefix + 'thon'
'Python'
```

Strings can be _indexed_ (subscripted), with the first character having index 0. There is no separate character type; a character is simply a string of size one:
> Python 没有 character 类型，字符就是长度为 1 的字符串

```
>>> word = 'Python'
>>> word[0]  # character in position 0
'P'
>>> word[5]  # character in position 5
'n'
```

Indices may also be negative numbers, to start counting from the right:

```
>>> word[-1]  # last character
'n'
>>> word[-2]  # second-last character
'o'
>>> word[-6]
'P'
```

Note that since -0 is the same as 0, negative indices start from -1.

In addition to indexing, _slicing_ is also supported. While indexing is used to obtain individual characters, _slicing_ allows you to obtain a substring:
> 切片获得子串

```
>>> word[0:2]  # characters from position 0 (included) to 2 (excluded)
'Py'
>>> word[2:5]  # characters from position 2 (included) to 5 (excluded)
'tho'
```

Slice indices have useful defaults; an omitted first index defaults to zero, an omitted second index defaults to the size of the string being sliced.
> 切片时忽略的第二个索引默认是字符串的长度

```
>>> word[:2]   # character from the beginning to position 2 (excluded)
'Py'
>>> word[4:]   # characters from position 4 (included) to the end
'on'
>>> word[-2:]  # characters from the second-last (included) to the end
'on'
```

Note how the start is always included, and the end always excluded. This makes sure that `s[:i] + s[i:]` is always equal to `s`:

```
>>> word[:2] + word[2:]
'Python'
>>> word[:4] + word[4:]
'Python'
```

One way to remember how slices work is to think of the indices as pointing _between_ characters, with the left edge of the first character numbered 0. Then the right edge of the last character of a string of _n_ characters has index _n_, for example:

```
 +---+---+---+---+---+---+
 | P | y | t | h | o | n |
 +---+---+---+---+---+---+
 0   1   2   3   4   5   6
-6  -5  -4  -3  -2  -1
```

The first row of numbers gives the position of the indices 0…6 in the string; the second row gives the corresponding negative indices. The slice from _i_ to _j_ consists of all characters between the edges labeled _i_ and _j_, respectively.

For non-negative indices, the length of a slice is the difference of the indices, if both are within bounds. For example, the length of `word[1:3]` is 2.

Attempting to use an index that is too large will result in an error:

```
>>> word[42]  # the word only has 6 characters
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: string index out of range
```

However, out of range slice indexes are handled gracefully when used for slicing:
> slice 会自动处理越界的索引

```
>>> word[4:42]
'on'
>>> word[42:]
''
```

Python strings cannot be changed — they are [immutable](https://docs.python.org/3/glossary.html#term-immutable). Therefore, assigning to an indexed position in the string results in an error:
> Python strings 是不可变的 ( 即便是变量 )，无法进行索引赋值

```
>>> word[0] = 'J'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'str' object does not support item assignment
>>> word[2:] = 'py'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'str' object does not support item assignment
```

If you need a different string, you should create a new one:
> 我们通过创建新的 string 得到不同的 string

```
>>> 'J' + word[1:]
'Jython'
>>> word[:2] + 'py'
'Pypy'
```

The built-in function [`len()`](https://docs.python.org/3/library/functions.html#len "len") returns the length of a string:

```
>>> s = 'supercalifragilisticexpialidocious'
>>> len(s)
34
```

See also
[Text Sequence Type — str](https://docs.python.org/3/library/stdtypes.html#textseq)
    Strings are examples of _sequence types_, and support the common operations supported by such types.
[String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
    Strings support a large number of methods for basic transformations and searching.
[f-strings](https://docs.python.org/3/reference/lexical_analysis.html#f-strings)
    String literals that have embedded expressions.
[Format String Syntax](https://docs.python.org/3/library/string.html#formatstrings)
    Information about string formatting with [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format "str.format").
[printf-style String Formatting](https://docs.python.org/3/library/stdtypes.html#old-string-formatting)
    The old formatting operations invoked when strings are the left operand of the `%` operator are described in more detail here.

### 3.1.3. Lists
Python knows a number of _compound_ data types, used to group together other values. The most versatile is the _list_, which can be written as a list of comma-separated values (items) between square brackets. Lists might contain items of different types, but usually the items all have the same type.

```
>>> squares = [1, 4, 9, 16, 25]
>>> squares
[1, 4, 9, 16, 25]
```

Like strings (and all other built-in [sequence](https://docs.python.org/3/glossary.html#term-sequence) types), lists can be indexed and sliced:

```
>>> squares[0]  # indexing returns the item
1
>>> squares[-1]
25
>>> squares[-3:]  # slicing returns a new list
[9, 16, 25]
```

Lists also support operations like concatenation:

```
>>> squares + [36, 49, 64, 81, 100]
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

Unlike strings, which are [immutable](https://docs.python.org/3/glossary.html#term-immutable), lists are a [mutable](https://docs.python.org/3/glossary.html#term-mutable) type, i.e. it is possible to change their content:
> `lists` 是可变的

```
>>> cubes = [1, 8, 27, 65, 125]  # something's wrong here
>>> 4 ** 3  # the cube of 4 is 64, not 65!
64
>>> cubes[3] = 64  # replace the wrong value
>>> cubes
[1, 8, 27, 64, 125]
```

You can also add new items at the end of the list, by using the `list.append()` _method_ (we will see more about methods later):

```
>>> cubes.append(216)  # add the cube of 6
>>> cubes.append(7 ** 3)  # and the cube of 7
>>> cubes
[1, 8, 27, 64, 125, 216, 343]
```

Simple assignment in Python never copies data. When you assign a list to a variable, the variable refers to the _existing list_. Any changes you make to the list through one variable will be seen through all other variables that refer to it.:
> Python 中的简单赋值永远不会拷贝数据，只会引用

```
>>> rgb = ["Red", "Green", "Blue"]
>>> rgba = rgb
>>> id(rgb) == id(rgba)  # they reference the same object
True
>>> rgba.append("Alph")
>>> rgb
["Red", "Green", "Blue", "Alph"]
```

All slice operations return a new list containing the requested elements. This means that the following slice returns a [shallow copy](https://docs.python.org/3/library/copy.html#shallow-vs-deep-copy) of the list:
> 所有的 slice 操作都会返回一个新的列表，包含所求的元素，即进行一次浅拷贝

```
>>> correct_rgba = rgba[:]
>>> correct_rgba[-1] = "Alpha"
>>> correct_rgba
["Red", "Green", "Blue", "Alpha"]
>>> rgba
["Red", "Green", "Blue", "Alph"]
```

Assignment to slices is also possible, and this can even change the size of the list or clear it entirely:
> 可以用 slice 语法改变 list 的内容、长度

```
>>> letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> letters
['a', 'b', 'c', 'd', 'e', 'f', 'g']
>>> # replace some values
>>> letters[2:5] = ['C', 'D', 'E']
>>> letters
['a', 'b', 'C', 'D', 'E', 'f', 'g']
>>> # now remove them
>>> letters[2:5] = []
>>> letters
['a', 'b', 'f', 'g']
>>> # clear the list by replacing all the elements with an empty list
>>> letters[:] = []
>>> letters
[]
```

The built-in function [`len()`](https://docs.python.org/3/library/functions.html#len "len") also applies to lists:

```
>>> letters = ['a', 'b', 'c', 'd']
>>> len(letters)
4
```

It is possible to nest lists (create lists containing other lists), for example:

```
>>> a = ['a', 'b', 'c']
>>> n = [1, 2, 3]
>>> x = [a, n]
>>> x
[['a', 'b', 'c'], [1, 2, 3]]
>>> x[0]
['a', 'b', 'c']
>>> x[0][1]
'b'
```

## 3.2. First Steps Towards Programming
Of course, we can use Python for more complicated tasks than adding two and two together. For instance, we can write an initial sub-sequence of the [Fibonacci series](https://en.wikipedia.org/wiki/Fibonacci_sequence) as follows:

```
>>> # Fibonacci series:
>>> # the sum of two elements defines the next
>>> a, b = 0, 1
>>> while a < 10:
...     print(a)
...     a, b = b, a+b
...
0
1
1
2
3
5
8
```

This example introduces several new features.
- The first line contains a _multiple assignment_: the variables `a` and `b` simultaneously get the new values 0 and 1. On the last line this is used again, demonstrating that the expressions on the right-hand side are all evaluated first before any of the assignments take place. The right-hand side expressions are evaluated from the left to the right.
> 赋值符号右边的表达式一定先于左边估值，估值顺序从左到右
- The [`while`](https://docs.python.org/3/reference/compound_stmts.html#while) loop executes as long as the condition (here: `a < 10`) remains true. In Python, like in C, any non-zero integer value is true; zero is false. The condition may also be a string or list value, in fact any sequence; anything with a non-zero length is true, empty sequences are false. The test used in the example is a simple comparison. The standard comparison operators are written the same as in C: `<` (less than), `>` (greater than), `==` (equal to), `<=` (less than or equal to), `>=` (greater than or equal to) and `!=` (not equal to).
> 空序列为 false，非空为 true
- The _body_ of the loop is _indented_: indentation is Python’s way of grouping statements. At the interactive prompt, you have to type a tab or space(s) for each indented line. In practice you will prepare more complicated input for Python with a text editor; all decent text editors have an auto-indent facility. When a compound statement is entered interactively, it must be followed by a blank line to indicate completion (since the parser cannot guess when you have typed the last line). Note that each line within a basic block must be indented by the same amount.
- The [`print()`](https://docs.python.org/3/library/functions.html#print "print") function writes the value of the argument(s) it is given. It differs from just writing the expression you want to write (as we did earlier in the calculator examples) in the way it handles multiple arguments, floating-point quantities, and strings. Strings are printed without quotes, and a space is inserted between items, so you can format things nicely, like this:

``` 
>>> i = 256*256
>>> print('The value of i is', i)
The value of i is 65536

The keyword argument _end_ can be used to avoid the newline after the output, or end the output with a different string:

>>>

>>> a, b = 0, 1
>>> while a < 1000:
...     print(a, end=',')
...     a, b = b, a+b
...
0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,
````

Footnotes
[1] Since `**` has higher precedence than `-`, `-3**2` will be interpreted as `-(3**2)` and thus result in `-9`. To avoid this and get `9`, you can use `(-3)**2`.

[2] Unlike other languages, special characters such as `\n` have the same meaning with both single (`'...'`) and double (`"..."`) quotes. The only difference between the two is that within single quotes you don’t need to escape `"` (but you have to escape `\'`) and vice versa.

# 4. More Control Flow Tools
As well as the [`while`](https://docs.python.org/3/reference/compound_stmts.html#while) statement just introduced, Python uses a few more that we will encounter in this chapter.

## 4.1. `if` Statements
Perhaps the most well-known statement type is the [`if`](https://docs.python.org/3/reference/compound_stmts.html#if) statement. For example:

```
>>> x = int(input("Please enter an integer: "))
Please enter an integer: 42
>>> if x < 0:
...     x = 0
...     print('Negative changed to zero')
... elif x == 0:
...     print('Zero')
... elif x == 1:
...     print('Single')
... else:
...     print('More')
...
More
```

There can be zero or more [`elif`](https://docs.python.org/3/reference/compound_stmts.html#elif) parts, and the [`else`](https://docs.python.org/3/reference/compound_stmts.html#else) part is optional. The keyword ‘`elif`’ is short for ‘else if’, and is useful to avoid excessive indentation. An `if` … `elif` … `elif` … sequence is a substitute for the `switch` or `case` statements found in other languages.

If you’re comparing the same value to several constants, or checking for specific types or attributes, you may also find the `match` statement useful. For more details see [match Statements](https://docs.python.org/3/tutorial/controlflow.html#tut-match).

## 4.2. `for` Statements
The [`for`](https://docs.python.org/3/reference/compound_stmts.html#for) statement in Python differs a bit from what you may be used to in C or Pascal. Rather than always iterating over an arithmetic progression of numbers (like in Pascal), or giving the user the ability to define both the iteration step and halting condition (as C), Python’s `for` statement iterates over the items of any sequence (a list or a string), in the order that they appear in the sequence. For example (no pun intended):
> Python `for` 迭代任意给定序列中的 item

```
>>> # Measure some strings:
>>> words = ['cat', 'window', 'defenestrate']
>>> for w in words:
...     print(w, len(w))
...
cat 3
window 6
defenestrate 12
```

Code that modifies a collection while iterating over that same collection can be tricky to get right. Instead, it is usually more straight-forward to loop over a copy of the collection or to create a new collection:
> 技巧：创建一个序列的 copy 用于迭代，在迭代中修改原序列

```python
# Create a sample collection
users = {'Hans': 'active', 'Éléonore': 'inactive', '景太郎': 'active'}

# Strategy:  Iterate over a copy
for user, status in users.copy().items():
    if status == 'inactive':
        del users[user]

# Strategy:  Create a new collection
active_users = {}
for user, status in users.items():
    if status == 'active':
        active_users[user] = status
```

## 4.3. The [`range()`](https://docs.python.org/3/library/stdtypes.html#range "range") Function
If you do need to iterate over a sequence of numbers, the built-in function [`range()`](https://docs.python.org/3/library/stdtypes.html#range "range") comes in handy. It generates arithmetic progressions:

```
>>> for i in range(5):
...     print(i)
...
0
1
2
3
4
```

The given end point is never part of the generated sequence; `range(10)` generates 10 values, the legal indices for items of a sequence of length 10. It is possible to let the range start at another number, or to specify a different increment (even negative; sometimes this is called the ‘step’):

```
>>> list(range(5, 10))
[5, 6, 7, 8, 9]

>>> list(range(0, 10, 3))
[0, 3, 6, 9]

>>> list(range(-10, -100, -30))
[-10, -40, -70]
```

To iterate over the indices of a sequence, you can combine [`range()`](https://docs.python.org/3/library/stdtypes.html#range "range") and [`len()`](https://docs.python.org/3/library/functions.html#len "len") as follows:

```
>>> a = ['Mary', 'had', 'a', 'little', 'lamb']
>>> for i in range(len(a)):
...     print(i, a[i])
...
0 Mary
1 had
2 a
3 little
4 lamb
```

In most such cases, however, it is convenient to use the [`enumerate()`](https://docs.python.org/3/library/functions.html#enumerate "enumerate") function, see [Looping Techniques](https://docs.python.org/3/tutorial/datastructures.html#tut-loopidioms).

A strange thing happens if you just print a range:

```
>>> range(10)
range(0, 10)
```

In many ways the object returned by [`range()`](https://docs.python.org/3/library/stdtypes.html#range "range") behaves as if it is a list, but in fact it isn’t. It is an object which returns the successive items of the desired sequence when you iterate over it, but it doesn’t really make the list, thus saving space.
> `range()` 实际上返回一个对象，即可迭代对象 iterable，在被迭代时，它会返回所期望的结果，但它不显式存储这些结果，以节省空间

We say such an object is [iterable](https://docs.python.org/3/glossary.html#term-iterable), that is, suitable as a target for functions and constructs that expect something from which they can obtain successive items until the supply is exhausted. We have seen that the [`for`](https://docs.python.org/3/reference/compound_stmts.html#for) statement is such a construct, while an example of a function that takes an iterable is [`sum()`](https://docs.python.org/3/library/functions.html#sum "sum"):
> 可迭代对象一般被函数或构造迭代式地调用，例如被 `for` 语句调用，以及被 `sum` 函数调用

```
>>> sum(range(4))  # 0 + 1 + 2 + 3
6
```

Later we will see more functions that return iterables and take iterables as arguments. In chapter [Data Structures](https://docs.python.org/3/tutorial/datastructures.html#tut-structures), we will discuss in more detail about [`list()`](https://docs.python.org/3/library/stdtypes.html#list "list").

## 4.4. `break` and `continue` Statements, and `else` Clauses on Loops
The [`break`](https://docs.python.org/3/reference/simple_stmts.html#break) statement breaks out of the innermost enclosing [`for`](https://docs.python.org/3/reference/compound_stmts.html#for) or [`while`](https://docs.python.org/3/reference/compound_stmts.html#while) loop.

A `for` or `while` loop can include an `else` clause.

In a [`for`](https://docs.python.org/3/reference/compound_stmts.html#for) loop, the `else` clause is executed after the loop reaches its final iteration.

In a [`while`](https://docs.python.org/3/reference/compound_stmts.html#while) loop, it’s executed after the loop’s condition becomes false.

In either kind of loop, the `else` clause is **not** executed if the loop was terminated by a [`break`](https://docs.python.org/3/reference/simple_stmts.html#break).
> `for` , `while` 的 `else` 语句只有在 `for/while` 没有被 `break` 中断时才会被最后执行

This is exemplified in the following `for` loop, which searches for prime numbers:

```
>>> for n in range(2, 10):
...     for x in range(2, n):
...         if n % x == 0:
...             print(n, 'equals', x, '*', n//x)
...             break
...     else:
...         # loop fell through without finding a factor
...         print(n, 'is a prime number')
...
2 is a prime number
3 is a prime number
4 equals 2 * 2
5 is a prime number
6 equals 2 * 3
7 is a prime number
8 equals 2 * 4
9 equals 3 * 3
```

(Yes, this is the correct code. Look closely: the `else` clause belongs to the [`for`](https://docs.python.org/3/reference/compound_stmts.html#for) loop, **not** the [`if`](https://docs.python.org/3/reference/compound_stmts.html#if) statement.)

When used with a loop, the `else` clause has more in common with the `else` clause of a [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) statement than it does with that of [`if`](https://docs.python.org/3/reference/compound_stmts.html#if) statements: a [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) statement’s `else` clause runs when no exception occurs, and a loop’s `else` clause runs when no `break` occurs. For more on the `try` statement and exceptions, see [Handling Exceptions](https://docs.python.org/3/tutorial/errors.html#tut-handling).
> `try` 语句的 `else` 在没有异常发生时被运行

The [`continue`](https://docs.python.org/3/reference/simple_stmts.html#continue) statement, also borrowed from C, continues with the next iteration of the loop:

```
>>> for num in range(2, 10):
...     if num % 2 == 0:
...         print("Found an even number", num)
...         continue
...     print("Found an odd number", num)
...
Found an even number 2
Found an odd number 3
Found an even number 4
Found an odd number 5
Found an even number 6
Found an odd number 7
Found an even number 8
Found an odd number 9
```

## 4.5. `pass` Statements
The [`pass`](https://docs.python.org/3/reference/simple_stmts.html#pass) statement does nothing. It can be used when a statement is required syntactically but the program requires no action. For example:

```
>>> while True:
...     pass  # Busy-wait for keyboard interrupt (Ctrl+C)
...
```

This is commonly used for creating minimal classes:

```
>>> class MyEmptyClass:
...     pass
...
```

Another place [`pass`](https://docs.python.org/3/reference/simple_stmts.html#pass) can be used is as a place-holder for a function or conditional body when you are working on new code, allowing you to keep thinking at a more abstract level. The `pass` is silently ignored:

```
>>> def initlog(*args):
...     pass   # Remember to implement this!
...
```

## 4.6. `match` Statements
A [`match`](https://docs.python.org/3/reference/compound_stmts.html#match) statement takes an expression and compares its value to successive patterns given as one or more case blocks. This is superficially similar to a switch statement in C, Java or JavaScript (and many other languages), but it’s more similar to pattern matching in languages like Rust or Haskell. Only the first pattern that matches gets executed and it can also extract components (sequence elements or object attributes) from the value into variables.
> `match` 的结构和 `switch` 相似，但做的是模式匹配

The simplest form compares a subject value against one or more literals:

```python
def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case _:
            return "Something's wrong with the internet"
```

Note the last block: the “variable name” `_` acts as a _wildcard_ and never fails to match. If no case matches, none of the branches is executed.
> 变量名 `_` 作为通配符，一定会被匹配

You can combine several literals in a single pattern using `|` (“or”):

```python
case 401 | 403 | 404:
    return "Not allowed"
```

Patterns can look like unpacking assignments, and can be used to bind variables:
> match 的模式还可以是解包表达式，且还可以用于绑定变量 (模式中的变量会被赋值)

```python
# point is an (x, y) tuple
match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Y={y}")
    case (x, 0):
        print(f"X={x}")
    case (x, y):
        print(f"X={x}, Y={y}")
    case _:
        raise ValueError("Not a point")
```

Study that one carefully! The first pattern has two literals, and can be thought of as an extension of the literal pattern shown above. But the next two patterns combine a literal and a variable, and the variable _binds_ a value from the subject (`point`). The fourth pattern captures two values, which makes it conceptually similar to the unpacking assignment `(x, y) = point`.

If you are using classes to structure your data you can use the class name followed by an argument list resembling a constructor, but with the ability to capture attributes into variables:
> 若参数是类，则模式可以使用构造函数的形式

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def where_is(point):
    match point:
        case Point(x=0, y=0):
            print("Origin")
        case Point(x=0, y=y):
            print(f"Y={y}")
        case Point(x=x, y=0):
            print(f"X={x}")
        case Point():
            print("Somewhere else")
        case _:
            print("Not a point")
```

You can use positional parameters with some builtin classes that provide an ordering for their attributes (e.g. dataclasses). You can also define a specific position for attributes in patterns by setting the `__match_args__` special attribute in your classes. 
> 可以使用位置参数
> 可以定义类作为模式时特定的位置的属性，通过设定 `__match_args__`

If it’s set to (“x”, “y”), the following patterns are all equivalent (and all bind the `y` attribute to the `var` variable):

```python
Point(1, var)
Point(1, y=var)
Point(x=1, y=var)
Point(y=var, x=1)
```

A recommended way to read patterns is to look at them as an extended form of what you would put on the left of an assignment, to understand which variables would be set to what. Only the standalone names (like `var` above) are assigned to by a match statement. Dotted names (like `foo.bar`), attribute names (the `x=` and `y=` above) or class names (recognized by the “(…)” next to them like `Point` above) are never assigned to.

Patterns can be arbitrarily nested. For example, if we have a short list of Points, with `__match_args__` added, we could match it like this:

```python
class Point:
    __match_args__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

match points:
    case []:
        print("No points")
    case [Point(0, 0)]:
        print("The origin")
    case [Point(x, y)]:
        print(f"Single point {x}, {y}")
    case [Point(0, y1), Point(0, y2)]:
        print(f"Two on the Y axis at {y1}, {y2}")
    case _:
        print("Something else")
```

We can add an `if` clause to a pattern, known as a “guard”. If the guard is false, `match` goes on to try the next case block. Note that value capture happens before the guard is evaluated:

```python
match point:
    case Point(x, y) if x == y:
        print(f"Y=X at {x}")
    case Point(x, y):
        print(f"Not on the diagonal")
```

Several other key features of this statement:

- Like unpacking assignments, tuple and list patterns have exactly the same meaning and actually match arbitrary sequences. An important exception is that they don’t match iterators or strings.
- Sequence patterns support extended unpacking: `[x, y, *rest]` and `(x, y, *rest)` work similar to unpacking assignments. The name after `*` may also be `_`, so `(x, y, *_)` matches a sequence of at least two items without binding the remaining items.
- Mapping patterns: `{"bandwidth": b, "latency": l}` captures the `"bandwidth"` and `"latency"` values from a dictionary. Unlike sequence patterns, extra keys are ignored. An unpacking like `**rest` is also supported. (But `**_` would be redundant, so it is not allowed.)
- Subpatterns may be captured using the `as` keyword:
    `case (Point(x1, y1), Point(x2, y2) as p2): ...`
    will capture the second element of the input as `p2` (as long as the input is a sequence of two points)
- Most literals are compared by equality, however the singletons `True`, `False` and `None` are compared by identity.
- Patterns may use named constants. These must be dotted names to prevent them from being interpreted as capture variable:

```python
from enum import Enum
class Color(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'

color = Color(input("Enter your choice of 'red', 'blue' or 'green': "))

match color:
    case Color.RED:
        print("I see red!")
    case Color.GREEN:
        print("Grass is green")
    case Color.BLUE:
        print("I'm feeling the blues :(")
```    

For a more detailed explanation and additional examples, you can look into [**PEP 636**](https://peps.python.org/pep-0636/) which is written in a tutorial format.

## 4.7. Defining Functions
We can create a function that writes the Fibonacci series to an arbitrary boundary:

```
>>> def fib(n):    # write Fibonacci series up to n
... """Print a Fibonacci series up to n."""
...     a, b = 0, 1
...     while a < n:
...         print(a, end=' ')
...         a, b = b, a+b
...     print()
...
>>> # Now call the function we just defined:
>>> fib(2000)
0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597
```

The keyword [`def`](https://docs.python.org/3/reference/compound_stmts.html#def) introduces a function _definition_. It must be followed by the function name and the parenthesized list of formal parameters. The statements that form the body of the function start at the next line, and must be indented.

The first statement of the function body can optionally be a string literal; this string literal is the function’s documentation string, or _docstring_. (More about docstrings can be found in the section [Documentation Strings](https://docs.python.org/3/tutorial/controlflow.html#tut-docstrings).) There are tools which use docstrings to automatically produce online or printed documentation, or to let the user interactively browse through code; it’s good practice to include docstrings in code that you write, so make a habit of it.
> 函数体的第一个语句可以是一个字符串字面值，它将成为函数的 documentation string，使用 docstring 是好习惯

The _execution_ of a function introduces a new symbol table used for the local variables of the function. More precisely, all variable assignments in a function store the value in the local symbol table; whereas variable references first look in the local symbol table, then in the local symbol tables of enclosing functions, then in the global symbol table, and finally in the table of built-in names. Thus, global variables and variables of enclosing functions cannot be directly assigned a value within a function (unless, for global variables, named in a [`global`](https://docs.python.org/3/reference/simple_stmts.html#global) statement, or, for variables of enclosing functions, named in a [`nonlocal`](https://docs.python.org/3/reference/simple_stmts.html#nonlocal) statement), although they may be referenced.
> 函数的执行会为函数的局部变量引入新的符号表
> 函数内的所有赋值将值存储于局部符号表，变量引用也优先查看局部符号表，然后逐级向上，最后到 built-in names
> 因此不能通过在函数内直接赋值改变 global variables 以及 variables of enclosing 函数的值，除非使用 `global` , `nonlocal` 语句，当然它们的值是可以被引用的

The actual parameters (arguments) to a function call are introduced in the local symbol table of the called function when it is called; thus, arguments are passed using _call by value_ (where the _value_ is always an object _reference_, not the value of the object). [[1]] (https://docs.python.org/3/tutorial/controlflow.html#id2) When a function calls another function, or calls itself recursively, a new local symbol table is created for that call.
> 实参也是存储在局部符号表里的，因此参数实际上是值传递 ( 但值永远是一个对象引用，而不是对象的值 )，递归调用也会创建新的符号表

A function definition associates the function name with the function object in the current symbol table. The interpreter recognizes the object pointed to by that name as a user-defined function. Other names can also point to that same function object and can also be used to access the function:
> 函数定义将函数名和函数对象在当前的符号表关联
> 解释器将由函数名指向的对象视作用户定义的函数，允许用其他指向该对象的名称引用该对象并访问该函数

```
>>> fib
<function fib at 10042ed0>
>>> f = fib
>>> f(100)
0 1 1 2 3 5 8 13 21 34 55 89
```

Coming from other languages, you might object that `fib` is not a function but a procedure since it doesn’t return a value. In fact, even functions without a [`return`](https://docs.python.org/3/reference/simple_stmts.html#return) statement do return a value, albeit a rather boring one. This value is called `None` (it’s a built-in name). Writing the value `None` is normally suppressed by the interpreter if it would be the only value written. You can see it if you really want to using [`print()`](https://docs.python.org/3/library/functions.html#print "print"):
> 没有 `return` 的函数也会返回值 `None` ，`return` 没有参数也返回 `None`

```
>>> fib(0)
>>> print(fib(0))
None
```

It is simple to write a function that returns a list of the numbers of the Fibonacci series, instead of printing it:

```
>>> def fib2(n):  # return Fibonacci series up to n
... """Return a list containing the Fibonacci series up to n."""
...     result = []
...     a, b = 0, 1
...     while a < n:
...         result.append(a)    # see below
...         a, b = b, a+b
...     return result
...
>>> f100 = fib2(100)    # call it
>>> f100                # write the result
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
```

This example, as usual, demonstrates some new Python features:
- The [`return`](https://docs.python.org/3/reference/simple_stmts.html#return) statement returns with a value from a function. `return` without an expression argument returns `None`. Falling off the end of a function also returns `None`.
- The statement `result.append(a)` calls a _method_ of the list object `result`. A method is a function that ‘belongs’ to an object and is named `obj.methodname`, where `obj` is some object (this may be an expression), and `methodname` is the name of a method that is defined by the object’s type. Different types define different methods. Methods of different types may have the same name without causing ambiguity. (It is possible to define your own object types and methods, using _classes_, see [Classes](https://docs.python.org/3/tutorial/classes.html#tut-classes)) The method `append()` shown in the example is defined for list objects; it adds a new element at the end of the list. In this example it is equivalent to `result = result + [a]`, but more efficient.

## 4.8. More on Defining Functions
It is also possible to define functions with a variable number of arguments. There are three forms, which can be combined.
> 函数支持变长参数列表

### 4.8.1. Default Argument Values
The most useful form is to specify a default value for one or more arguments. This creates a function that can be called with fewer arguments than it is defined to allow. For example:

```python
def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        reply = input(prompt)
        if reply in {'y', 'ye', 'yes'}:
            return True
        if reply in {'n', 'no', 'nop', 'nope'}:
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)
```

This function can be called in several ways:

- giving only the mandatory argument: `ask_ok('Do you really want to quit?')`
- giving one of the optional arguments: `ask_ok('OK to overwrite the file?', 2)`
- or even giving all arguments: `ask_ok('OK to overwrite the file?', 2, 'Come on, only yes or no!')`

This example also introduces the [`in`](https://docs.python.org/3/reference/expressions.html#in) keyword. This tests whether or not a sequence contains a certain value.

The default values are evaluated at the point of function definition in the _defining_ scope, so that
> 默认值在函数的定义作用域内中的定义时刻被评估

```python
i = 5

def f(arg=i):
    print(arg)

i = 6
f()
```

will print `5`.

**Important warning:** The default value is evaluated only once. This makes a difference when the default is a mutable object such as a list, dictionary, or instances of most classes. For example, the following function accumulates the arguments passed to it on subsequent calls:
> 默认值仅评估一次

```python
def f(a, L=[]):
    L.append(a)
    return L

print(f(1))
print(f(2))
print(f(3))
```

This will print

```
[1]
[1, 2]
[1, 2, 3]
```

If you don’t want the default to be shared between subsequent calls, you can write the function like this instead:

```python
def f(a, L=None):
    if L is None:
        L = []
    L.append(a)
    return L
```

### 4.8.2. Keyword Argument
Functions can also be called using [keyword arguments](https://docs.python.org/3/glossary.html#term-keyword-argument) of the form `kwarg=value`. For instance, the following function:

```python
def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
    print("-- This parrot wouldn't", action, end=' ')
    print("if you put", voltage, "volts through it.")
    print("-- Lovely plumage, the", type)
    print("-- It's", state, "!")
```

accepts one required argument (`voltage`) and three optional arguments (`state`, `action`, and `type`). This function can be called in any of the following ways:

```python
parrot(1000)                                          # 1 positional argument
parrot(voltage=1000)                                  # 1 keyword argument
parrot(voltage=1000000, action='VOOOOOM')             # 2 keyword arguments
parrot(action='VOOOOOM', voltage=1000000)             # 2 keyword arguments
parrot('a million', 'bereft of life', 'jump')         # 3 positional arguments
parrot('a thousand', state='pushing up the daisies')  # 1 positional, 1 keyword
```

but all the following calls would be invalid:

```python
parrot()                     # required argument missing
parrot(voltage=5.0, 'dead')  # non-keyword argument after a keyword argument
parrot(110, voltage=220)     # duplicate value for the same argument
parrot(actor='John Cleese')  # unknown keyword argument
```

In a function call, keyword arguments must follow positional arguments. All the keyword arguments passed must match one of the arguments accepted by the function (e.g. `actor` is not a valid argument for the `parrot` function), and their order is not important. This also includes non-optional arguments (e.g. `parrot(voltage=1000)` is valid too). No argument may receive a value more than once. Here’s an example that fails due to this restriction:
> 调用函数时，关键字参数需要在位置参数后面

```
>>> def function(a):
...     pass
...
>>> function(0, a=0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: function() got multiple values for argument 'a'
```

When a final formal parameter of the form `**name` is present, it receives a dictionary (see [Mapping Types — dict](https://docs.python.org/3/library/stdtypes.html#typesmapping)) containing all keyword arguments except for those corresponding to a formal parameter. This may be combined with a formal parameter of the form `*name` (described in the next subsection) which receives a [tuple](https://docs.python.org/3/tutorial/datastructures.html#tut-tuples) containing the positional arguments beyond the formal parameter list. (`*name` must occur before `**name`.) For example, if we define a function like this:
> 形式为 `**name` 的参数接受一个字典，包含了除了形式参数的所有的关键字参数
> 形式为 `*name` 的参数接受一个元组，包含了形式参数列表以外的所有位置参数

```python
def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    for kw in keywords:
        print(kw, ":", keywords[kw])
```

It could be called like this:

```python
cheeseshop("Limburger", "It's very runny, sir.",
           "It's really very, VERY runny, sir.",
           shopkeeper="Michael Palin",
           client="John Cleese",
           sketch="Cheese Shop Sketch")
```

and of course it would print:

```
-- Do you have any Limburger ?
-- I'm sorry, we're all out of Limburger
It's very runny, sir.
It's really very, VERY runny, sir.
----------------------------------------
shopkeeper : Michael Palin
client : John Cleese
sketch : Cheese Shop Sketch
```

Note that the order in which the keyword arguments are printed is guaranteed to match the order in which they were provided in the function call.

### 4.8.3. Special parameters
By default, arguments may be passed to a Python function either by position or explicitly by keyword. For readability and performance, it makes sense to restrict the way arguments can be passed so that a developer need only look at the function definition to determine if items are passed by position, by position or keyword, or by keyword.

A function definition may look like:

```
def f(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2):
      -----------    ----------     ----------
        |             |                  |
        |        Positional or keyword   |
        |                                - Keyword only
         -- Positional only
```

where `/` and `*` are optional. If used, these symbols indicate the kind of parameter by how the arguments may be passed to the function: positional-only, positional-or-keyword, and keyword-only. Keyword parameters are also referred to as named parameters.
> 在函数定义的参数列表中，可以使用符号 `/` `*` 显式划分参数类型，规定了参数应该如何传入

#### 4.8.3.1. Positional-or-Keyword Arguments
If `/` and `*` are not present in the function definition, arguments may be passed to a function by position or by keyword.
> 若没有，则所有参数既可以通过位置传入，也可以通过关键字传入

#### 4.8.3.2. Positional-Only Parameters
Looking at this in a bit more detail, it is possible to mark certain parameters as _positional-only_. If _positional-only_, the parameters’ order matters, and the parameters cannot be passed by keyword. Positional-only parameters are placed before a `/` (forward-slash). The `/` is used to logically separate the positional-only parameters from the rest of the parameters. If there is no `/` in the function definition, there are no positional-only parameters.

Parameters following the `/` may be _positional-or-keyword_ or _keyword-only_.
> `/` 之前的参数仅能通过位置传入，之后的参数可以通过位置也可以通过关键字传入

#### 4.8.3.3. Keyword-Only Arguments
To mark parameters as _keyword-only_, indicating the parameters must be passed by keyword argument, place an `*` in the arguments list just before the first _keyword-only_ parameter.
> `*` 之后的参数只能通过关键字传入

#### 4.8.3.4. Function Examples
Consider the following example function definitions paying close attention to the markers `/` and `*`:

```
>>> def standard_arg(arg):
...     print(arg)
...
>>> def pos_only_arg(arg, /):
...     print(arg)
...
>>> def kwd_only_arg(*, arg):
...     print(arg)
...
>>> def combined_example(pos_only, /, standard, *, kwd_only):
...     print(pos_only, standard, kwd_only)
```

The first function definition, `standard_arg`, the most familiar form, places no restrictions on the calling convention and arguments may be passed by position or keyword:

```
>>> standard_arg(2)
2

>>> standard_arg(arg=2)
2
```

The second function `pos_only_arg` is restricted to only use positional parameters as there is a `/` in the function definition:

```
>>> pos_only_arg(1)
1

>>> pos_only_arg(arg=1)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: pos_only_arg() got some positional-only arguments passed as keyword arguments: 'arg'
```

The third function `kwd_only_args` only allows keyword arguments as indicated by a `*` in the function definition:

```
>>> kwd_only_arg(3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: kwd_only_arg() takes 0 positional arguments but 1 was given

>>> kwd_only_arg(arg=3)
3
```

And the last uses all three calling conventions in the same function definition:

```
>>> combined_example(1, 2, 3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: combined_example() takes 2 positional arguments but 3 were given

>>> combined_example(1, 2, kwd_only=3)
1 2 3

>>> combined_example(1, standard=2, kwd_only=3)
1 2 3

>>> combined_example(pos_only=1, standard=2, kwd_only=3)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: combined_example() got some positional-only arguments passed as keyword arguments: 'pos_only'
```

Finally, consider this function definition which has a potential collision between the positional argument `name` and `**kwds` which has `name` as a key:

```
def foo(name, **kwds):
    return 'name' in kwds
```

There is no possible call that will make it return `True` as the keyword `'name'` will always bind to the first parameter. For example:

```
>>> foo(1, **{'name': 2})
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: foo() got multiple values for argument 'name'
>>>
```

But using `/` (positional only arguments), it is possible since it allows `name` as a positional argument and `'name'` as a key in the keyword arguments:

```
>>> def foo(name, /, **kwds):
...     return 'name' in kwds
...
>>> foo(1, **{'name': 2})
True
```

In other words, the names of positional-only parameters can be used in `**kwds` without ambiguity.
> 划分界限后，positional-only 的参数的名称可以用在 `**kwds` 中且不引起歧义

#### 4.8.3.5. Recap
The use case will determine which parameters to use in the function definition:

```python
def f(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2):
```

As guidance:
- Use positional-only if you want the name of the parameters to not be available to the user. This is useful when parameter names have no real meaning, if you want to enforce the order of the arguments when the function is called or if you need to take some positional parameters and arbitrary keywords.
- Use keyword-only when names have meaning and the function definition is more understandable by being explicit with names or you want to prevent users relying on the position of the argument being passed.
- For an API, use positional-only to prevent breaking API changes if the parameter’s name is modified in the future.
> 对于 API，可以用 positional-only 防止之后参数名称改变导致 API 改变

### 4.8.4. Arbitrary Argument Lists
Finally, the least frequently used option is to specify that a function can be called with an arbitrary number of arguments. These arguments will be wrapped up in a tuple (see [Tuples and Sequences](https://docs.python.org/3/tutorial/datastructures.html#tut-tuples)). Before the variable number of arguments, zero or more normal arguments may occur.
> 通过 `*args` 让函数可以接受任意数量的参数

```python
def write_multiple_items(file, separator, *args):
    file.write(separator.join(args))
```

Normally, these _variadic_ arguments will be last in the list of formal parameters, because they scoop up all remaining input arguments that are passed to the function. Any formal parameters which occur after the `*args` parameter are ‘keyword-only’ arguments, meaning that they can only be used as keywords rather than positional arguments.
> 在 `*args` 之后的参数只能是 keyword-only 参数，只能通过关键字传入

```
>>> def concat(*args, sep="/"):
...     return sep.join(args)
...
>>> concat("earth", "mars", "venus")
'earth/mars/venus'
>>> concat("earth", "mars", "venus", sep=".")
'earth.mars.venus'
```

### 4.8.5. Unpacking Argument Lists
The reverse situation occurs when the arguments are already in a list or tuple but need to be unpacked for a function call requiring separate positional arguments. For instance, the built-in [`range()`](https://docs.python.org/3/library/stdtypes.html#range "range") function expects separate _start_ and _stop_ arguments. If they are not available separately, write the function call with the `*` -operator to unpack the arguments out of a list or tuple:
> 对于向函数传入参数，如果要通过解包一个元组/列表传参，可以指定 `*` 运算符

```
>>> list(range(3, 6))            # normal call with separate arguments
[3, 4, 5]
>>> args = [3, 6]
>>> list(range(*args))            # call with arguments unpacked from a list
[3, 4, 5]
```

In the same fashion, dictionaries can deliver keyword arguments with the `**` -operator:
> `**` 运算符用于解包一个字典

```
>>> def parrot(voltage, state='a stiff', action='voom'):
...     print("-- This parrot wouldn't", action, end=' ')
...     print("if you put", voltage, "volts through it.", end=' ')
...     print("E's", state, "!")
...
>>> d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
>>> parrot(**d)
-- This parrot wouldn't VOOM if you put four million volts through it. E's bleedin' demised !
```

### 4.8.6. Lambda Expressions
Small anonymous functions can be created with the [`lambda`](https://docs.python.org/3/reference/expressions.html#lambda) keyword. This function returns the sum of its two arguments: `lambda a, b: a+b`. Lambda functions can be used wherever function objects are required. They are syntactically restricted to a single expression. Semantically, they are just syntactic sugar for a normal function definition. Like nested function definitions, lambda functions can reference variables from the containing scope:
> `lambda` 创建匿名函数，`lambda` 只是正常函数定义的语法糖
> 返回 `lambda` 函数的函数:

```
>>> def make_incrementor(n):
...     return lambda x: x + n
...
>>> f = make_incrementor(42)
>>> f(0)
42
>>> f(1)
43
```

The above example uses a lambda expression to return a function. Another use is to pass a small function as an argument:
> 传递函数作为参数:

```
>>> pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
>>> pairs.sort(key=lambda pair: pair[1])
>>> pairs
[(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]
```

### 4.8.7. Documentation Strings
Here are some conventions about the content and formatting of documentation strings.

The first line should always be a short, concise summary of the object’s purpose. For brevity, it should not explicitly state the object’s name or type, since these are available by other means (except if the name happens to be a verb describing a function’s operation). This line should begin with a capital letter and end with a period.
> docstring 的第一行是对象目的的 concise 总结，不应该显式说明对象的名称或类型，以大写字母开始，句号结尾

If there are more lines in the documentation string, the second line should be blank, visually separating the summary from the rest of the description. The following lines should be one or more paragraphs describing the object’s calling conventions, its side effects, etc.
> docstring 的第二行是空行
> 剩余的为描述对象的调用常规、副作用等的段落

The Python parser does not strip indentation from multi-line string literals in Python, so tools that process documentation have to strip indentation if desired. This is done using the following convention. The first non-blank line _after_ the first line of the string determines the amount of indentation for the entire documentation string. (We can’t use the first line since it is generally adjacent to the string’s opening quotes so its indentation is not apparent in the string literal.) Whitespace “equivalent” to this indentation is then stripped from the start of all lines of the string. Lines that are indented less should not occur, but if they occur all their leading whitespace should be stripped. Equivalence of whitespace should be tested after expansion of tabs (to 8 spaces, normally).
> Python parser 不会去除多行 string literals 的缩进

Here is an example of a multi-line docstring:

```
>>> def my_function():
... """Do nothing, but document it.
...
...     No, really, it doesn't do anything.
...     """
...     pass
...
>>> print(my_function.__doc__)
Do nothing, but document it.

    No, really, it doesn't do anything.
```

### 4.8.8. Function Annotations
[Function annotations](https://docs.python.org/3/reference/compound_stmts.html#function) are completely optional metadata information about the types used by user-defined functions (see [**PEP 3107**](https://peps.python.org/pep-3107/) and [**PEP 484**](https://peps.python.org/pep-0484/) for more information).
> Function annotations 是用户定义的函数的 optional 的 metadata information

[Annotations](https://docs.python.org/3/glossary.html#term-function-annotation) are stored in the `__annotations__` attribute of the function as a dictionary and have no effect on any other part of the function. Parameter annotations are defined by a colon after the parameter name, followed by an expression evaluating to the value of the annotation. Return annotations are defined by a literal `->`, followed by an expression, between the parameter list and the colon denoting the end of the [`def`](https://docs.python.org/3/reference/compound_stmts.html#def) statement. The following example has a required argument, an optional argument, and the return value annotated:
> annotations 存储于 `__annotations__` 属性
> 参数的 annotations 在参数名之后的 `:` 之后定义，是一个表达式，作为 annotations 的值
> Return annotations 通过 `->` 定义，是一个表达式，在参数列表和标志 `def` 语句结尾的 `:` 之间

```
>>> def f(ham: str, eggs: str = 'eggs') -> str:
...     print("Annotations:", f.__annotations__)
...     print("Arguments:", ham, eggs)
...     return ham + ' and ' + eggs
...
>>> f('spam')
Annotations: {'ham': <class 'str'>, 'return': <class 'str'>, 'eggs': <class 'str'>}
Arguments: spam eggs
'spam and eggs'
```

## 4.9. Intermezzo: Coding Style
Now that you are about to write longer, more complex pieces of Python, it is a good time to talk about _coding style_. Most languages can be written (or more concise, _formatted_) in different styles; some are more readable than others. Making it easy for others to read your code is always a good idea, and adopting a nice coding style helps tremendously for that.

For Python, [**PEP 8**](https://peps.python.org/pep-0008/) has emerged as the style guide that most projects adhere to; it promotes a very readable and eye-pleasing coding style. Every Python developer should read it at some point; here are the most important points extracted for you:
- Use 4-space indentation, and no tabs.
    4 spaces are a good compromise between small indentation (allows greater nesting depth) and large indentation (easier to read). Tabs introduce confusion, and are best left out.
- Wrap lines so that they don’t exceed 79 characters.
    This helps users with small displays and makes it possible to have several code files side-by-side on larger displays.
- Use blank lines to separate functions and classes, and larger blocks of code inside functions.
- When possible, put comments on a line of their own.
- Use docstrings.
- Use spaces around operators and after commas, but not directly inside bracketing constructs: `a = f(1, 2) + g(3, 4)`.
- Name your classes and functions consistently; the convention is to use `UpperCamelCase` for classes and `lowercase_with_underscores` for functions and methods. Always use `self` as the name for the first method argument (see [A First Look at Classes](https://docs.python.org/3/tutorial/classes.html#tut-firstclasses) for more on classes and methods).
> `UpperCamelCase` for classes, `lowercase_with_underscores` for functions and methods
- Don’t use fancy encodings if your code is meant to be used in international environments. Python’s default, UTF-8, or even plain ASCII work best in any case.
- Likewise, don’t use non-ASCII characters in identifiers if there is only the slightest chance people speaking a different language will read or maintain the code.

Footnotes
[1] Actually, _call by object reference_ would be a better description, since if a mutable object is passed, the caller will see any changes the callee makes to it (items inserted into a list).

# 5. Data Structures
This chapter describes some things you’ve learned about already in more detail, and adds some new things as well.

## 5.1. More on Lists
The list data type has some more methods. Here are all of the methods of list objects:

`list.append(_x_)`
    Add an item to the end of the list. Equivalent to `a[len(a):] = [x]`.

`list.extend(_iterable_)`
    Extend the list by appending all the items from the iterable. Equivalent to `a[len(a):] = iterable`.

`list.insert(_i_, _x_)`
    Insert an item at a given position. The first argument is the index of the element before which to insert, so `a.insert(0, x)` inserts at the front of the list, and `a.insert(len(a), x)` is equivalent to `a.append(x)`.

`list.remove(_x_)`
    Remove the first item from the list whose value is equal to _x_. It raises a [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError "ValueError") if there is no such item.

`list.pop([_i_])`
    Remove the item at the given position in the list, and return it. If no index is specified, `a.pop()` removes and returns the last item in the list. It raises an [`IndexError`](https://docs.python.org/3/library/exceptions.html#IndexError "IndexError") if the list is empty or the index is outside the list range.

`list.clear()`
    Remove all items from the list. Equivalent to `del a[:]`.

`list.index(_x_[, _start_[, _end_]])`
    Return zero-based index in the list of the first item whose value is equal to _x_. Raises a [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError "ValueError") if there is no such item.
    The optional arguments _start_ and _end_ are interpreted as in the slice notation and are used to limit the search to a particular subsequence of the list. The returned index is computed relative to the beginning of the full sequence rather than the _start_ argument.

`list.count(_x_)`
    Return the number of times _x_ appears in the list.

`list.sort(_*_, _key=None_, _reverse=False_)`
    Sort the items of the list in place (the arguments can be used for sort customization, see [`sorted()`](https://docs.python.org/3/library/functions.html#sorted "sorted") for their explanation).

`list.reverse()`
    Reverse the elements of the list in place.

`list.copy()`
    Return a shallow copy of the list. Equivalent to `a[:]`.

An example that uses most of the list methods:

```
>>> fruits = ['orange', 'apple', 'pear', 'banana', 'kiwi', 'apple', 'banana']
>>> fruits.count('apple')
2
>>> fruits.count('tangerine')
0
>>> fruits.index('banana')
3
>>> fruits.index('banana', 4)  # Find next banana starting at position 4
6
>>> fruits.reverse()
>>> fruits
['banana', 'apple', 'kiwi', 'banana', 'pear', 'apple', 'orange']
>>> fruits.append('grape')
>>> fruits
['banana', 'apple', 'kiwi', 'banana', 'pear', 'apple', 'orange', 'grape']
>>> fruits.sort()
>>> fruits
['apple', 'apple', 'banana', 'banana', 'grape', 'kiwi', 'orange', 'pear']
>>> fruits.pop()
'pear'
```

You might have noticed that methods like `insert`, `remove` or `sort` that only modify the list have no return value printed – they return the default `None`. [1](https://docs.python.org/3/tutorial/datastructures.html#id2) This is a design principle for all mutable data structures in Python.
> 仅修改 list 的方法例如 `insert/remofe/sort` 没有返回值，即返回默认的 `None`
> 这个设计原则对于 Python 所有的可变数据结构都适用

Another thing you might notice is that not all data can be sorted or compared. For instance, `[None, 'hello', 10]` doesn’t sort because integers can’t be compared to strings and `None` can’t be compared to other types. Also, there are some types that don’t have a defined ordering relation. For example, `3+4j < 5+7j` isn’t a valid comparison.

### 5.1.1. Using Lists as Stacks
The list methods make it very easy to use a list as a stack, where the last element added is the first element retrieved (“last-in, first-out”). To add an item to the top of the stack, use `append()`. To retrieve an item from the top of the stack, use `pop()` without an explicit index. For example:

```
>>> stack = [3, 4, 5]
>>> stack.append(6)
>>> stack.append(7)
>>> stack
[3, 4, 5, 6, 7]
>>> stack.pop()
7
>>> stack
[3, 4, 5, 6]
>>> stack.pop()
6
>>> stack.pop()
5
>>> stack
[3, 4]
```

### 5.1.2. Using Lists as Queues
It is also possible to use a list as a queue, where the first element added is the first element retrieved (“first-in, first-out”); however, lists are not efficient for this purpose. While appends and pops from the end of list are fast, doing inserts or pops from the beginning of a list is slow (because all of the other elements have to be shifted by one).

To implement a queue, use [`collections.deque`](https://docs.python.org/3/library/collections.html#collections.deque "collections.deque") which was designed to have fast appends and pops from both ends. For example:
> list 实现的队列对于头部的修改较慢
> `collections.deque` 实现了较快的双端队列

```
>>> from collections import deque
>>> queue = deque(["Eric", "John", "Michael"])
>>> queue.append("Terry")           # Terry arrives
>>> queue.append("Graham")          # Graham arrives
>>> queue.popleft()                 # The first to arrive now leaves
'Eric'
>>> queue.popleft()                 # The second to arrive now leaves
'John'
>>> queue                           # Remaining queue in order of arrival
deque(['Michael', 'Terry', 'Graham'])
```

### 5.1.3. List Comprehensions
List comprehensions provide a concise way to create lists. Common applications are to make new lists where each element is the result of some operations applied to each member of another sequence or iterable, or to create a subsequence of those elements that satisfy a certain condition.

For example, assume we want to create a list of squares, like:

```
>>> squares = []
>>> for x in range(10):
...     squares.append(x**2)
...
>>> squares
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

Note that this creates (or overwrites) a variable named `x` that still exists after the loop completes. We can calculate the list of squares without any side effects using:

```
squares = list(map(lambda x: x**2, range(10)))
```

or, equivalently:

```
squares = [x**2 for x in range(10)]
```

which is more concise and readable.

A list comprehension consists of brackets containing an expression followed by a `for` clause, then zero or more `for` or `if` clauses. The result will be a new list resulting from evaluating the expression in the context of the `for` and `if` clauses which follow it. For example, this listcomp combines the elements of two lists if they are not equal:
> 列表推导式: 方括号内一个后面跟着 `for` 语句的表达式，`for` 语句后可以跟着 0个或多个 `for` 或 `if` 语句

```
>>> [(x, y) for x in [1,2,3] for y in [3,1,4] if x != y]
[(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]
```

and it’s equivalent to:

```
>>> combs = []
>>> for x in [1,2,3]:
...     for y in [3,1,4]:
...         if x != y:
...             combs.append((x, y))
...
>>> combs
[(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]
```

Note how the order of the [`for`](https://docs.python.org/3/reference/compound_stmts.html#for) and [`if`](https://docs.python.org/3/reference/compound_stmts.html#if) statements is the same in both these snippets.

If the expression is a tuple (e.g. the `(x, y)` in the previous example), it must be parenthesized.

```
>>> vec = [-4, -2, 0, 2, 4]
>>> # create a new list with the values doubled
>>> [x*2 for x in vec]
[-8, -4, 0, 4, 8]
>>> # filter the list to exclude negative numbers
>>> [x for x in vec if x >= 0]
[0, 2, 4]
>>> # apply a function to all the elements
>>> [abs(x) for x in vec]
[4, 2, 0, 2, 4]
>>> # call a method on each element
>>> freshfruit = ['  banana', '  loganberry ', 'passion fruit  ']
>>> [weapon.strip() for weapon in freshfruit]
['banana', 'loganberry', 'passion fruit']
>>> # create a list of 2-tuples like (number, square)
>>> [(x, x**2) for x in range(6)]
[(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25)]
>>> # the tuple must be parenthesized, otherwise an error is raised
>>> [x, x**2 for x in range(6)]
  File "<stdin>", line 1
    [x, x**2 for x in range(6)]
     ^^^^^^^
SyntaxError: did you forget parentheses around the comprehension target?
>>> # flatten a list using a listcomp with two 'for'
>>> vec = [[1,2,3], [4,5,6], [7,8,9]]
>>> [num for elem in vec for num in elem]
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

List comprehensions can contain complex expressions and nested functions:

```
>>> from math import pi
>>> [str(round(pi, i)) for i in range(1, 6)]
['3.1', '3.14', '3.142', '3.1416', '3.14159']
```

### 5.1.4. Nested List Comprehensions
The initial expression in a list comprehension can be any arbitrary expression, including another list comprehension.
> 列表推导式内的 initial expression 可以是任意表达式，包括另一个列表推导式

Consider the following example of a 3x4 matrix implemented as a list of 3 lists of length 4:

```
>>> matrix = [
...     [1, 2, 3, 4],
...     [5, 6, 7, 8],
...     [9, 10, 11, 12],
... ]
```

The following list comprehension will transpose rows and columns:

```
>>> [[row[i] for row in matrix] for i in range(4)]
[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
```

As we saw in the previous section, the inner list comprehension is evaluated in the context of the [`for`](https://docs.python.org/3/reference/compound_stmts.html#for) that follows it, so this example is equivalent to:

```
>>> transposed = []
>>> for i in range(4):
...     transposed.append([row[i] for row in matrix])
...
>>> transposed
[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
```

which, in turn, is the same as:

```
>>> transposed = []
>>> for i in range(4):
...     # the following 3 lines implement the nested listcomp
...     transposed_row = []
...     for row in matrix:
...         transposed_row.append(row[i])
...     transposed.append(transposed_row)
...
>>> transposed
[[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
```

In the real world, you should prefer built-in functions to complex flow statements. The [`zip()`](https://docs.python.org/3/library/functions.html#zip "zip") function would do a great job for this use case:

```
>>> list(zip(*matrix))
[(1, 5, 9), (2, 6, 10), (3, 7, 11), (4, 8, 12)]
```

See [Unpacking Argument Lists](https://docs.python.org/3/tutorial/controlflow.html#tut-unpacking-arguments) for details on the asterisk in this line.

## 5.2. The `del` statement
There is a way to remove an item from a list given its index instead of its value: the [`del`](https://docs.python.org/3/reference/simple_stmts.html#del) statement. This differs from the `pop()` method which returns a value. The `del` statement can also be used to remove slices from a list or clear the entire list (which we did earlier by assignment of an empty list to the slice). For example:
> 给定 item 的索引，可以用 `del` 将其移除
> `del` 不像 `pop()` ，不会返回值
> `del` 也可以移除 list 的一个 slice，或者整个 list

```
>>> a = [-1, 1, 66.25, 333, 333, 1234.5]
>>> del a[0]
>>> a
[1, 66.25, 333, 333, 1234.5]
>>> del a[2:4]
>>> a
[1, 66.25, 1234.5]
>>> del a[:]
>>> a
[]
```

[`del`](https://docs.python.org/3/reference/simple_stmts.html#del) can also be used to delete entire variables:

```
>>> del a
```

Referencing the name `a` hereafter is an error (at least until another value is assigned to it). We’ll find other uses for [`del`](https://docs.python.org/3/reference/simple_stmts.html#del) later.
> `del a` 之后，就不能再引用 `a` 这个名字，直到对 `a` 这个名字有新的赋值

## 5.3. Tuples and Sequences
We saw that lists and strings have many common properties, such as indexing and slicing operations. They are two examples of _sequence_ data types (see [Sequence Types — list, tuple, range](https://docs.python.org/3/library/stdtypes.html#typesseq)). Since Python is an evolving language, other sequence data types may be added. There is also another standard sequence data type: the _tuple_.
> list, string, tuple 都是序列数据类型

A tuple consists of a number of values separated by commas, for instance:

```
>>> t = 12345, 54321, 'hello!'
>>> t[0]
12345
>>> t
(12345, 54321, 'hello!')
>>> # Tuples may be nested:
>>> u = t, (1, 2, 3, 4, 5)
>>> u
((12345, 54321, 'hello!'), (1, 2, 3, 4, 5))
>>> # Tuples are immutable:
>>> t[0] = 88888
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
>>> # but they can contain mutable objects:
>>> v = ([1, 2, 3], [3, 2, 1])
>>> v
([1, 2, 3], [3, 2, 1])
```

As you see, on output tuples are always enclosed in parentheses, so that nested tuples are interpreted correctly; they may be input with or without surrounding parentheses, although often parentheses are necessary anyway (if the tuple is part of a larger expression). It is not possible to assign to the individual items of a tuple, however it is possible to create tuples which contain mutable objects, such as lists.
> tuple 不可变，因此不能给 tuple 中的某个 item 赋值
> 但可以创建包含了可变对象的 tuple，例如 tuple 包含 list

Though tuples may seem similar to lists, they are often used in different situations and for different purposes. Tuples are [immutable](https://docs.python.org/3/glossary.html#term-immutable), and usually contain a heterogeneous sequence of elements that are accessed via unpacking (see later in this section) or indexing (or even by attribute in the case of [`namedtuples`](https://docs.python.org/3/library/collections.html#collections.namedtuple "collections.namedtuple")). Lists are [mutable](https://docs.python.org/3/glossary.html#term-mutable), and their elements are usually homogeneous and are accessed by iterating over the list.
> tuple 不可变，一般包含异质的数据
> list 可变，一般包含同质的数据

A special problem is the construction of tuples containing 0 or 1 items: the syntax has some extra quirks to accommodate these. Empty tuples are constructed by an empty pair of parentheses; a tuple with one item is constructed by following a value with a comma (it is not sufficient to enclose a single value in parentheses). Ugly, but effective. For example:
> `()` 创建空的 tuple

```
>>> empty = ()
>>> singleton = 'hello',    # <-- note trailing comma
>>> len(empty)
0
>>> len(singleton)
1
>>> singleton
('hello',)
```

The statement `t = 12345, 54321, 'hello!'` is an example of _tuple packing_: the values `12345`, `54321` and `'hello!'` are packed together in a tuple. The reverse operation is also possible:

```
>>> x, y, z = t
```

This is called, appropriately enough, _sequence unpacking_ and works for any sequence on the right-hand side. Sequence unpacking requires that there are as many variables on the left side of the equals sign as there are elements in the sequence. Note that multiple assignment is really just a combination of tuple packing and sequence unpacking.
> sequence unpacking 对于任意序列都可以使用
> multiple assignment 实质上就是 tuple packing 以及 sequence unpacking

## 5.4. Sets
Python also includes a data type for _sets_. A set is an unordered collection with no duplicate elements. Basic uses include membership testing and eliminating duplicate entries. Set objects also support mathematical operations like union, intersection, difference, and symmetric difference.
> set: 无序、无重复

Curly braces or the [`set()`](https://docs.python.org/3/library/stdtypes.html#set "set") function can be used to create sets. Note: to create an empty set you have to use `set()`, not `{}`; the latter creates an empty dictionary, a data structure that we discuss in the next section.
> 空集合用 `set()` 而不是 `{}` 创建

Here is a brief demonstration:

```
>>> basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
>>> print(basket)                      # show that duplicates have been removed
{'orange', 'banana', 'pear', 'apple'}
>>> 'orange' in basket                 # fast membership testing
True
>>> 'crabgrass' in basket
False

>>> # Demonstrate set operations on unique letters from two words
>>>
>>> a = set('abracadabra')
>>> b = set('alacazam')
>>> a                                  # unique letters in a
{'a', 'r', 'b', 'c', 'd'}
>>> a - b                              # letters in a but not in b
{'r', 'd', 'b'}
>>> a | b                              # letters in a or b or both
{'a', 'c', 'r', 'd', 'b', 'm', 'z', 'l'}
>>> a & b                              # letters in both a and b
{'a', 'c'}
>>> a ^ b                              # letters in a or b but not both
{'r', 'd', 'b', 'm', 'z', 'l'}
```

Similarly to [list comprehensions](https://docs.python.org/3/tutorial/datastructures.html#tut-listcomps), set comprehensions are also supported:
> set comprhensions 同样存在

```
>>> a = {x for x in 'abracadabra' if x not in 'abc'}
>>> a
{'r', 'd'}
```

## 5.5. Dictionaries
Another useful data type built into Python is the _dictionary_ (see [Mapping Types — dict](https://docs.python.org/3/library/stdtypes.html#typesmapping)). Dictionaries are sometimes found in other languages as “associative memories” or “associative arrays”. Unlike sequences, which are indexed by a range of numbers, dictionaries are indexed by _keys_, which can be any immutable type; strings and numbers can always be keys. Tuples can be used as keys if they contain only strings, numbers, or tuples; if a tuple contains any mutable object either directly or indirectly, it cannot be used as a key. You can’t use lists as keys, since lists can be modified in place using index assignments, slice assignments, or methods like `append()` and `extend()`.
> 字典用 key 索引，key 可以是任意不可变类型，例如 string 或 number
> tuple 仅包含不可变类型时，也可以用作 key，否则不行

It is best to think of a dictionary as a set of _key: value_ pairs, with the requirement that the keys are unique (within one dictionary). A pair of braces creates an empty dictionary: `{}`. Placing a comma-separated list of key:value pairs within the braces adds initial key:value pairs to the dictionary; this is also the way dictionaries are written on output.
> key is unique in the dictionary
> `{}` 创建空字典

The main operations on a dictionary are storing a value with some key and extracting the value given the key. It is also possible to delete a key:value pair with `del`. If you store using a key that is already in use, the old value associated with that key is forgotten. It is an error to extract a value using a non-existent key.
> `del` 可以用于删除键值对

Performing `list(d)` on a dictionary returns a list of all the keys used in the dictionary, in insertion order (if you want it sorted, just use `sorted(d)` instead). To check whether a single key is in the dictionary, use the [`in`](https://docs.python.org/3/reference/expressions.html#in) keyword.
> `list(d)` 返回字典的所有 keys，顺序为 insertion order
> `in` 用于 check 特定 key 是否再字典中

Here is a small example using a dictionary:

```
>>> tel = {'jack': 4098, 'sape': 4139}
>>> tel['guido'] = 4127
>>> tel
{'jack': 4098, 'sape': 4139, 'guido': 4127}
>>> tel['jack']
4098
>>> del tel['sape']
>>> tel['irv'] = 4127
>>> tel
{'jack': 4098, 'guido': 4127, 'irv': 4127}
>>> list(tel)
['jack', 'guido', 'irv']
>>> sorted(tel)
['guido', 'irv', 'jack']
>>> 'guido' in tel
True
>>> 'jack' not in tel
False
```

The [`dict()`](https://docs.python.org/3/library/stdtypes.html#dict "dict") constructor builds dictionaries directly from sequences of key-value pairs:
> `dict()` 从 key-value pair 的序列构造字典

```
>>> dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
{'sape': 4139, 'guido': 4127, 'jack': 4098}
```

In addition, dict comprehensions can be used to create dictionaries from arbitrary key and value expressions:
> 同样存在字典推导式

```
>>> {x: x**2 for x in (2, 4, 6)}
{2: 4, 4: 16, 6: 36}
```

When the keys are simple strings, it is sometimes easier to specify pairs using keyword arguments:
> `dict` 在 keys 仅仅是 simple strings 时也可以用关键字参数

```
>>> dict(sape=4139, guido=4127, jack=4098)
{'sape': 4139, 'guido': 4127, 'jack': 4098}
```

## 5.6. Looping Techniques
When looping through dictionaries, the key and corresponding value can be retrieved at the same time using the [`items()`](https://docs.python.org/3/library/stdtypes.html#dict.items "dict.items") method.

```
>>> knights = {'gallahad': 'the pure', 'robin': 'the brave'}
>>> for k, v in knights.items():
...     print(k, v)
...
gallahad the pure
robin the brave
```

When looping through a sequence, the position index and corresponding value can be retrieved at the same time using the [`enumerate()`](https://docs.python.org/3/library/functions.html#enumerate "enumerate") function.

```
>>> for i, v in enumerate(['tic', 'tac', 'toe']):
...     print(i, v)
...
0 tic
1 tac
2 toe
```

To loop over two or more sequences at the same time, the entries can be paired with the [`zip()`](https://docs.python.org/3/library/functions.html#zip "zip") function.
> 在需要迭代多个序列时，可以用 `zip` 产生 pair 各个序列的 entry 的序列用于迭代

```
>>> questions = ['name', 'quest', 'favorite color']
>>> answers = ['lancelot', 'the holy grail', 'blue']
>>> for q, a in zip(questions, answers):
...     print('What is your {0}?  It is {1}.'.format(q, a))
...
What is your name?  It is lancelot.
What is your quest?  It is the holy grail.
What is your favorite color?  It is blue.
```

To loop over a sequence in reverse, first specify the sequence in a forward direction and then call the [`reversed()`](https://docs.python.org/3/library/functions.html#reversed "reversed") function.

```
>>> for i in reversed(range(1, 10, 2)):
...     print(i)
...
9
7
5
3
1
```

To loop over a sequence in sorted order, use the [`sorted()`](https://docs.python.org/3/library/functions.html#sorted "sorted") function which returns a new sorted list while leaving the source unaltered.
> `sorted()` 返回新的排序好的 list，原 list 不变

```
>>> basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
>>> for i in sorted(basket):
...     print(i)
...
apple
apple
banana
orange
orange
pear
```

Using [`set()`](https://docs.python.org/3/library/stdtypes.html#set "set") on a sequence eliminates duplicate elements. The use of [`sorted()`](https://docs.python.org/3/library/functions.html#sorted "sorted") in combination with [`set()`](https://docs.python.org/3/library/stdtypes.html#set "set") over a sequence is an idiomatic way to loop over unique elements of the sequence in sorted order.

```
>>> basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
>>> for f in sorted(set(basket)):
...     print(f)
...
apple
banana
orange
pear
```

It is sometimes tempting to change a list while you are looping over it; however, it is often simpler and safer to create a new list instead.

```
>>> import math
>>> raw_data = [56.2, float('NaN'), 51.7, 55.3, 52.5, float('NaN'), 47.8]
>>> filtered_data = []
>>> for value in raw_data:
...     if not math.isnan(value):
...         filtered_data.append(value)
...
>>> filtered_data
[56.2, 51.7, 55.3, 52.5, 47.8]
```

## 5.7. More on Conditions
The conditions used in `while` and `if` statements can contain any operators, not just comparisons.

The comparison operators `in` and `not in` are membership tests that determine whether a value is in (or not in) a container. The operators `is` and `is not` compare whether two objects are really the same object. All comparison operators have the same priority, which is lower than that of all numerical operators.
> membership test: `in/not in`
> 是否是同一对象: `is/is not`
> 所有的比较运算符优先级相同，且低于所有的数字运算符

Comparisons can be chained. For example, `a < b == c` tests whether `a` is less than `b` and moreover `b` equals `c`.

Comparisons may be combined using the Boolean operators `and` and `or`, and the outcome of a comparison (or of any other Boolean expression) may be negated with `not`. These have lower priorities than comparison operators; between them, `not` has the highest priority and `or` the lowest, so that `A and not B or C` is equivalent to `(A and (not B)) or C`. As always, parentheses can be used to express the desired composition.
> 布尔运算符: `and/or/not` ，它们的优先级比比较运算符低
> 优先级: `not > and > or`

The Boolean operators `and` and `or` are so-called _short-circuit_ operators: their arguments are evaluated from left to right, and evaluation stops as soon as the outcome is determined. For example, if `A` and `C` are true but `B` is false, `A and B and C` does not evaluate the expression `C`. When used as a general value and not as a Boolean, the return value of a short-circuit operator is the last evaluated argument.
> 短路运算符的返回值总是它最后评估的参数 argument

It is possible to assign the result of a comparison or other Boolean expression to a variable. For example,

```
>>> string1, string2, string3 = '', 'Trondheim', 'Hammer Dance'
>>> non_null = string1 or string2 or string3
>>> non_null
'Trondheim'
```

Note that in Python, unlike C, assignment inside expressions must be done explicitly with the [walrus operator](https://docs.python.org/3/faq/design.html#why-can-t-i-use-an-assignment-in-an-expression) `:=`. This avoids a common class of problems encountered in C programs: typing `=` in an expression when `==` was intended.
> Python 中，表达式内的赋值必须显式地写为 `:=`

## 5.8. Comparing Sequences and Other Types
Sequence objects typically may be compared to other objects with the same sequence type. The comparison uses _lexicographical_ ordering: first the first two items are compared, and if they differ this determines the outcome of the comparison; if they are equal, the next two items are compared, and so on, until either sequence is exhausted. If two items to be compared are themselves sequences of the same type, the lexicographical comparison is carried out recursively. If all items of two sequences compare equal, the sequences are considered equal. If one sequence is an initial sub-sequence of the other, the shorter sequence is the smaller (lesser) one. Lexicographical ordering for strings uses the Unicode code point number to order individual characters. Some examples of comparisons between sequences of the same type:
> 序列对象可以和其他同类地序列对象比较，比较使用字典序，按照 item 顺序比较，如果 item 本身也是序列，则会递归比较
> 如果所有 item 都相同，则序列认为是相同

```
(1, 2, 3)              < (1, 2, 4)
[1, 2, 3]              < [1, 2, 4]
'ABC' < 'C' < 'Pascal' < 'Python'
(1, 2, 3, 4)           < (1, 2, 4)
(1, 2)                 < (1, 2, -1)
(1, 2, 3)             == (1.0, 2.0, 3.0)
(1, 2, ('aa', 'ab'))   < (1, 2, ('abc', 'a'), 4)
```

Note that comparing objects of different types with `<` or `>` is legal provided that the objects have appropriate comparison methods. For example, mixed numeric types are compared according to their numeric value, so 0 equals 0.0, etc. Otherwise, rather than providing an arbitrary ordering, the interpreter will raise a [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "TypeError") exception.
> 如果不同的对象定义好了恰当的比较方法，则不同的对象间可以比较，例如 `0 == 0.0` ，否则 `TypeError`

Footnotes
[1] Other languages may return the mutated object, which allows method chaining, such as `d->insert("a")->remove("b")->sort();`.

# 6. Modules
If you quit from the Python interpreter and enter it again, the definitions you have made (functions and variables) are lost. Therefore, if you want to write a somewhat longer program, you are better off using a text editor to prepare the input for the interpreter and running it with that file as input instead. This is known as creating a _script_. As your program gets longer, you may want to split it into several files for easier maintenance. You may also want to use a handy function that you’ve written in several programs without copying its definition into each program.

To support this, Python has a way to put definitions in a file and use them in a script or in an interactive instance of the interpreter. Such a file is called a _module_; definitions from a module can be _imported_ into other modules or into the _main_ module (the collection of variables that you have access to in a script executed at the top level and in calculator mode).
> Python 支持将定义放在一个文件中，然后在脚本中或者在解释器的交互示例中使用该文件，这种文件被称为模块
> 模块中的定义可以被导入到其他模块或者 main 模块，main 模块是在顶层执行的模块

A module is a file containing Python definitions and statements. The file name is the module name with the suffix `.py` appended. Within a module, the module’s name (as a string) is available as the value of the global variable `__name__`. For instance, use your favorite text editor to create a file called `fibo.py` in the current directory with the following contents:
> 模块即包含了 Python 定义和语句的文件，文件名即模块名 + `.py`
> 模块内通过全局变量 `__name__` 访问模块名

```python
# Fibonacci numbers module

def fib(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()

def fib2(n):   # return Fibonacci series up to n
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a+b
    return result
```

Now enter the Python interpreter and import this module with the following command:
> `import` 导入模块

```
>>> import fibo
```

This does not add the names of the functions defined in `fibo` directly to the current [namespace](https://docs.python.org/3/glossary.html#term-namespace) (see [Python Scopes and Namespaces](https://docs.python.org/3/tutorial/classes.html#tut-scopes) for more details); it only adds the module name `fibo` there. Using the module name you can access the functions:
> `import fibo` 不会将 `fibo` 中定义的函数名直接加入到当前命名空间，而是只将模块名加入到当前命名空间
> 可以通过模块名访问其函数

```
>>> fibo.fib(1000)
0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987
>>> fibo.fib2(100)
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
>>> fibo.__name__
'fibo'
```

If you intend to use a function often you can assign it to a local name:
> 可以为模块的函数创建局部名称

```
>>> fib = fibo.fib
>>> fib(500)
0 1 1 2 3 5 8 13 21 34 55 89 144 233 377
```

## 6.1. More on Modules
A module can contain executable statements as well as function definitions. These statements are intended to initialize the module. They are executed only the _first_ time the module name is encountered in an import statement. [1](https://docs.python.org/3/tutorial/modules.html#id3) (They are also run if the file is executed as a script.)
> 模块也可以包含可执行语句，这些语句用于初始化模块，它们仅在模块名称被 `import` 的第一次被执行

Each module has its own private namespace, which is used as the global namespace by all functions defined in the module. Thus, the author of a module can use global variables in the module without worrying about accidental clashes with a user’s global variables. On the other hand, if you know what you are doing you can touch a module’s global variables with the same notation used to refer to its functions, `modname.itemname`.
> 模块有自己的私有命名空间，作为模块内函数的全局命名空间，因此模块的全局变量不会和用户定义的全局变量冲突

Modules can import other modules. It is customary but not required to place all [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statements at the beginning of a module (or script, for that matter). The imported module names, if placed at the top level of a module (outside any functions or classes), are added to the module’s global namespace.
> 模块可以 import 其他模块，模块在 top level 上 import 的模块名会被加入模块的全局命名空间

There is a variant of the [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement that imports names from a module directly into the importing module’s namespace. For example:

```
>>> from fibo import fib, fib2
>>> fib(500)
0 1 1 2 3 5 8 13 21 34 55 89 144 233 377
```

This does not introduce the module name from which the imports are taken in the local namespace (so in the example, `fibo` is not defined).
> `from .. import ..` 不会将模块名引入当前命名空间

There is even a variant to import all names that a module defines:

```
>>> from fibo import *
>>> fib(500)
0 1 1 2 3 5 8 13 21 34 55 89 144 233 377
```

This imports all names except those beginning with an underscore (`_`). In most cases Python programmers do not use this facility since it introduces an unknown set of names into the interpreter, possibly hiding some things you have already defined.
> `from .. import *` 导入了模块内定义的所有名称，除了以 `_` 开头的

Note that in general the practice of importing `*` from a module or package is frowned upon, since it often causes poorly readable code. However, it is okay to use it to save typing in interactive sessions.

If the module name is followed by `as`, then the name following `as` is bound directly to the imported module.
> `import .. as ..` 为模块绑定别名

```
>>> import fibo as fib
>>> fib.fib(500)
0 1 1 2 3 5 8 13 21 34 55 89 144 233 377
```

This is effectively importing the module in the same way that `import fibo` will do, with the only difference of it being available as `fib`.

It can also be used when utilising [`from`](https://docs.python.org/3/reference/simple_stmts.html#from) with similar effects:

```
>>> from fibo import fib as fibonacci
>>> fibonacci(500)
0 1 1 2 3 5 8 13 21 34 55 89 144 233 377
```

Note
For efficiency reasons, each module is only imported once per interpreter session. Therefore, if you change your modules, you must restart the interpreter – or, if it’s just one module you want to test interactively, use [`importlib.reload()`](https://docs.python.org/3/library/importlib.html#importlib.reload "importlib.reload"), e.g. `import importlib; importlib.reload(modulename)`.
> 每个解释器 session 仅会导入各个模块一次，如果需要重新导入，使用 `importlib.reload()`

### 6.1.1. Executing modules as scripts
When you run a Python module with

```
python fibo.py <arguments>
```

the code in the module will be executed, just as if you imported it, but with the `__name__` set to `"__main__"`. 
> 使用 `python module-name.py <arguments>` 运行模块时，模块内的代码会被运行，但是模块的全局变量 `__name__` 会被设定为 `__main__` 

That means that by adding this code at the end of your module:

```
if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
```

you can make the file usable as a script as well as an importable module, because the code that parses the command line only runs if the module is executed as the “main” file:
> 因此通过 `if __name__ == "__main__"` 可以让文件即作为 importable 模块也可以作为脚本，因为 `if` 下的代码模块作为 “main” 文件执行时才会执行

```
$ python fibo.py 50
0 1 1 2 3 5 8 13 21 34
```

If the module is imported, the code is not run:
> 如果只是 import 该模块，这部分代码不会运行

```
>>> import fibo
>>>
```

This is often used either to provide a convenient user interface to a module, or for testing purposes (running the module as a script executes a test suite).

### 6.1.2. The Module Search Path
When a module named `spam` is imported, the interpreter first searches for a built-in module with that name. These module names are listed in [`sys.builtin_module_names`](https://docs.python.org/3/library/sys.html#sys.builtin_module_names "sys.builtin_module_names"). If not found, it then searches for a file named `spam.py` in a list of directories given by the variable [`sys.path`](https://docs.python.org/3/library/sys.html#sys.path "sys.path"). [`sys.path`](https://docs.python.org/3/library/sys.html#sys.path "sys.path") is initialized from these locations:
> Python 在 import 中优先搜索内建模块名，然后在变量 `sys.path` 指定的路径中搜索模块名，`sys.path` 初始包括

- The directory containing the input script (or the current directory when no file is specified).
> 当前目录 
- [`PYTHONPATH`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH) (a list of directory names, with the same syntax as the shell variable `PATH`).
> `PYTHONPATH` 
- The installation-dependent default (by convention including a `site-packages` directory, handled by the [`site`](https://docs.python.org/3/library/site.html#module-site "site: Module responsible for site-specific configuration.") module).
> 依赖于安装的默认路径，一般是 `site-packages`  

More details are at [The initialization of the sys.path module search path](https://docs.python.org/3/library/sys_path_init.html#sys-path-init).

Note
On file systems which support symlinks, the directory containing the input script is calculated after the symlink is followed. In other words the directory containing the symlink is **not** added to the module search path.
> 支持符号链接的文件系统中，解释器会在符号链接被解释后再计算输入脚本所在的目录，因此包含符号链接的目录不会被添加到模块搜索路径中，解释器只知道真实路径

After initialization, Python programs can modify [`sys.path`](https://docs.python.org/3/library/sys.html#sys.path "sys.path"). The directory containing the script being run is placed at the beginning of the search path, ahead of the standard library path. This means that scripts in that directory will be loaded instead of modules of the same name in the library directory. This is an error unless the replacement is intended. See section [Standard Modules](https://docs.python.org/3/tutorial/modules.html#tut-standardmodules) for more information.
> Python 可以在初始化后修改 `sys.path` ，当前运行脚本所在的目录会在 search path 的开始，其后是标准库路径，因此当前目录的模块优先级更高

### 6.1.3. “Compiled” Python files
To speed up loading modules, Python caches the compiled version of each module in the `__pycache__` directory under the name `module._version_.pyc`, where the version encodes the format of the compiled file; it generally contains the Python version number. For example, in CPython release 3.3 the compiled version of spam.py would be cached as `__pycache__/spam.cpython-33.pyc`. This naming convention allows compiled modules from different releases and different versions of Python to coexist.
> 为了加速模块加载，Python 会将各个模块编译的版本缓存在 `__pychache__` 文件夹下，文件名称为 `module._version_.pyc` ，其中的版本实际上决定了 compiled file 的格式，一般版本就是 Python 的版本号
> 这允许来自不同的 Python 版本的 compiled modules 共存

Python checks the modification date of the source against the compiled version to see if it’s out of date and needs to be recompiled. This is a completely automatic process. Also, the compiled modules are platform-independent, so the same library can be shared among systems with different architectures.
> Python 会检查 compiled 模块和源文件的修改日期，以决定是否需要重新编译
> compiled 模块是不依赖于平台的

Python does not check the cache in two circumstances. First, it always recompiles and does not store the result for the module that’s loaded directly from the command line. Second, it does not check the cache if there is no source module. To support a non-source (compiled only) distribution, the compiled module must be in the source directory, and there must not be a source module.
> 在两种情况下，Python 不会 check cache:
> 其一，对于从命令行直接装载的模块，Python 总是会重新编译，并且不会储存结果
> 其二，如果没有源模块，Python 不会 check cache
> 如果需要一个 non-source (compiled only) 的分发，compiled 模块必须存储在源目录中，并且源目录没有源模块

Some tips for experts:
- You can use the [`-O`](https://docs.python.org/3/using/cmdline.html#cmdoption-O) or [`-OO`](https://docs.python.org/3/using/cmdline.html#cmdoption-OO) switches on the Python command to reduce the size of a compiled module. The `-O` switch removes assert statements, the `-OO` switch removes both assert statements and __doc__ strings. Since some programs may rely on having these available, you should only use this option if you know what you’re doing. “Optimized” modules have an `opt-` tag and are usually smaller. Future releases may change the effects of optimization.
> `-O` 移除模块的 assert 语句，`-OO` 移除模块的 assert 语句和 docstring
- A program doesn’t run any faster when it is read from a `.pyc` file than when it is read from a `.py` file; the only thing that’s faster about `.pyc` files is the speed with which they are loaded.
> `.pyc` 文件仅在加载时比 `.py` 文件快
- The module [`compileall`](https://docs.python.org/3/library/compileall.html#module-compileall "compileall: Tools for byte-compiling all Python source files in a directory tree.") can create `.pyc` files for all modules in a directory.
- There is more detail on this process, including a flow chart of the decisions, in [**PEP 3147**](https://peps.python.org/pep-3147/).

## 6.2. Standard Modules
Python comes with a library of standard modules, described in a separate document, the Python Library Reference (“Library Reference” hereafter). Some modules are built into the interpreter; these provide access to operations that are not part of the core of the language but are nevertheless built in, either for efficiency or to provide access to operating system primitives such as system calls. The set of such modules is a configuration option which also depends on the underlying platform. For example, the [`winreg`](https://docs.python.org/3/library/winreg.html#module-winreg "winreg: Routines and objects for manipulating the Windows registry. (Windows)") module is only provided on Windows systems. One particular module deserves some attention: [`sys`](https://docs.python.org/3/library/sys.html#module-sys "sys: Access system-specific parameters and functions."), which is built into every Python interpreter. The variables `sys.ps1` and `sys.ps2` define the strings used as primary and secondary prompts:
> Python 提供了一个标准模块的库
> 其中一些模块是内建于解释器的，它们提供了关于不是语言核心的一些操作，主要关于操作效率以及对于 OS 原语例如系统调用的访问
> 标准模块的集合属于配置选项，也依赖于平台，例如 `winreg` 模块仅在 Windows 平台提供
> 我们需要注意一个特殊的模块 `sys` ，它内建于每一个 Python 解释器，变量 `sys.ps1/ps2` 定义了用于 primary/secondary prompt 的字符串

```
>>> import sys
>>> sys.ps1
'>>> '
>>> sys.ps2
'... '
>>> sys.ps1 = 'C> '
C> print('Yuck!')
Yuck!
C>
```

These two variables are only defined if the interpreter is in interactive mode.
> 这两个变量仅在解释器处于 interactive 模式下被定义

The variable `sys.path` is a list of strings that determines the interpreter’s search path for modules. It is initialized to a default path taken from the environment variable [`PYTHONPATH`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH), or from a built-in default if [`PYTHONPATH`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH) is not set. You can modify it using standard list operations:
> 变量 `sys.path` 是一个字符串列表，定义了解释器对于模块的搜索路径
> 它根据环境变量 `PYTHONPATH` 初始化，如果 `PYTHONPATH` 没有设定，则根据内建的默认变量设定
> 可以用标准的列表操作对它进行修改

```
>>> import sys
>>> sys.path.append('/ufs/guido/lib/python')
```

## 6.3. The [`dir()`](https://docs.python.org/3/library/functions.html#dir "dir") Function
The built-in function [`dir()`](https://docs.python.org/3/library/functions.html#dir "dir") is used to find out which names a module defines. It returns a sorted list of strings:
> 内建函数 `dir()` 用于找到模块定义了哪些名称，它返回一个有序的字符串列表，注意不是 `dict()`

```
>>> import fibo, sys
>>> dir(fibo)
['__name__', 'fib', 'fib2']
>>> dir(sys)  
['__breakpointhook__', '__displayhook__', '__doc__', '__excepthook__',
 '__interactivehook__', '__loader__', '__name__', '__package__', '__spec__',
 '__stderr__', '__stdin__', '__stdout__', '__unraisablehook__',
 '_clear_type_cache', '_current_frames', '_debugmallocstats', '_framework',
 '_getframe', '_git', '_home', '_xoptions', 'abiflags', 'addaudithook',
 'api_version', 'argv', 'audit', 'base_exec_prefix', 'base_prefix',
 'breakpointhook', 'builtin_module_names', 'byteorder', 'call_tracing',
 'callstats', 'copyright', 'displayhook', 'dont_write_bytecode', 'exc_info',
 'excepthook', 'exec_prefix', 'executable', 'exit', 'flags', 'float_info',
 'float_repr_style', 'get_asyncgen_hooks', 'get_coroutine_origin_tracking_depth',
 'getallocatedblocks', 'getdefaultencoding', 'getdlopenflags',
 'getfilesystemencodeerrors', 'getfilesystemencoding', 'getprofile',
 'getrecursionlimit', 'getrefcount', 'getsizeof', 'getswitchinterval',
 'gettrace', 'hash_info', 'hexversion', 'implementation', 'int_info',
 'intern', 'is_finalizing', 'last_traceback', 'last_type', 'last_value',
 'maxsize', 'maxunicode', 'meta_path', 'modules', 'path', 'path_hooks',
 'path_importer_cache', 'platform', 'prefix', 'ps1', 'ps2', 'pycache_prefix',
 'set_asyncgen_hooks', 'set_coroutine_origin_tracking_depth', 'setdlopenflags',
 'setprofile', 'setrecursionlimit', 'setswitchinterval', 'settrace', 'stderr',
 'stdin', 'stdout', 'thread_info', 'unraisablehook', 'version', 'version_info',
 'warnoptions']
```

Without arguments, [`dir()`](https://docs.python.org/3/library/functions.html#dir "dir") lists the names you have defined currently:
> 没有参数时，`dir()` 列出我们当前已经定义的名字

```
>>> a = [1, 2, 3, 4, 5]
>>> import fibo
>>> fib = fibo.fib
>>> dir()
['__builtins__', '__name__', 'a', 'fib', 'fibo', 'sys']
```

Note that it lists all types of names: variables, modules, functions, etc.
> 注意 `dir()` 会列出所有类型的名字：变量、模块、函数等等

[`dir()`](https://docs.python.org/3/library/functions.html#dir "dir") does not list the names of built-in functions and variables. If you want a list of those, they are defined in the standard module [`builtins`](https://docs.python.org/3/library/builtins.html#module-builtins "builtins: The module that provides the built-in namespace."):
> 但 `dir()` 不会列出内建函数和变量的名称
> 这些内建变量和函数都定义于标准模块 `builtins`

```
>>> import builtins
>>> dir(builtins)  
['ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException',
 'BlockingIOError', 'BrokenPipeError', 'BufferError', 'BytesWarning',
 'ChildProcessError', 'ConnectionAbortedError', 'ConnectionError',
 'ConnectionRefusedError', 'ConnectionResetError', 'DeprecationWarning',
 'EOFError', 'Ellipsis', 'EnvironmentError', 'Exception', 'False',
 'FileExistsError', 'FileNotFoundError', 'FloatingPointError',
 'FutureWarning', 'GeneratorExit', 'IOError', 'ImportError',
 'ImportWarning', 'IndentationError', 'IndexError', 'InterruptedError',
 'IsADirectoryError', 'KeyError', 'KeyboardInterrupt', 'LookupError',
 'MemoryError', 'NameError', 'None', 'NotADirectoryError', 'NotImplemented',
 'NotImplementedError', 'OSError', 'OverflowError',
 'PendingDeprecationWarning', 'PermissionError', 'ProcessLookupError',
 'ReferenceError', 'ResourceWarning', 'RuntimeError', 'RuntimeWarning',
 'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError',
 'SystemExit', 'TabError', 'TimeoutError', 'True', 'TypeError',
 'UnboundLocalError', 'UnicodeDecodeError', 'UnicodeEncodeError',
 'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 'UserWarning',
 'ValueError', 'Warning', 'ZeroDivisionError', '_', '__build_class__',
 '__debug__', '__doc__', '__import__', '__name__', '__package__', 'abs',
 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes', 'callable',
 'chr', 'classmethod', 'compile', 'complex', 'copyright', 'credits',
 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'exit',
 'filter', 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr',
 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass',
 'iter', 'len', 'license', 'list', 'locals', 'map', 'max', 'memoryview',
 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 'property',
 'quit', 'range', 'repr', 'reversed', 'round', 'set', 'setattr', 'slice',
 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars',
 'zip']
```

## 6.4. Packages
Packages are a way of structuring Python’s module namespace by using “dotted module names”. For example, the module name `A.B` designates a submodule named `B` in a package named `A`. Just like the use of modules saves the authors of different modules from having to worry about each other’s global variable names, the use of dotted module names saves the authors of multi-module packages like NumPy or Pillow from having to worry about each other’s module names.
> Python 通过包来组织 Python 的模块命名空间，包就是一组模块的集合，包内可以包含多个模块和子包
> 例如，模块名称 `A.B` 表示包 `A` 中的子模块 `B` 
> 就像模块让模块作者不需要担心不同模块之间的全局变量名称冲突一样，点分模块名称使得向 Numpy 或 Pillow 这样的多模块包的作者不需要担心和其他包内的模块名称发生冲突

Suppose you want to design a collection of modules (a “package”) for the uniform handling of sound files and sound data. There are many different sound file formats (usually recognized by their extension, for example: `.wav`, `.aiff`, `.au`), so you may need to create and maintain a growing collection of modules for the conversion between the various file formats. There are also many different operations you might want to perform on sound data (such as mixing, adding echo, applying an equalizer function, creating an artificial stereo effect), so in addition you will be writing a never-ending stream of modules to perform these operations. Here’s a possible structure for your package (expressed in terms of a hierarchical filesystem):
> 假设我们要设计一组模块 (一个包) 用于处理声音文件和声音数据
> 声音文件有许多格式，例如 `.wav/.aiff/.au` ，因此我们需要创建并维护 a growing collection of modules 用于不同文件格式之间的转化，以及我们对于声音文件也会有不同的处理操作，因此我们也需要不断写出新的模块以执行这些操作
> 包的结构如下

```
sound/                          Top-level package
      __init__.py               Initialize the sound package
      formats/                  Subpackage for file format conversions
              __init__.py
              wavread.py
              wavwrite.py
              aiffread.py
              aiffwrite.py
              auread.py
              auwrite.py
              ...
      effects/                  Subpackage for sound effects
              __init__.py
              echo.py
              surround.py
              reverse.py
              ...
      filters/                  Subpackage for filters
              __init__.py
              equalizer.py
              vocoder.py
              karaoke.py
              ...
```

When importing the package, Python searches through the directories on `sys.path` looking for the package subdirectory.
> import 包时，Python 会在 `sys.path` 中的目录中搜索 package 子目录

The `__init__.py` files are required to make Python treat directories containing the file as packages (unless using a [namespace package](https://docs.python.org/3/glossary.html#term-namespace-package), a relatively advanced feature). This prevents directories with a common name, such as `string`, from unintentionally hiding valid modules that occur later on the module search path. In the simplest case, `__init__.py` can just be an empty file, but it can also execute initialization code for the package or set the `__all__` variable, described later.
> `__init__.py` 文件用于让 Python 将包含了文件的目录视作一个包，否则只是简单目录
> `__init__.py` 可以是空文件，也可以执行包的初始化代码，或者设定 `__all__` 变量

Users of the package can import individual modules from the package, for example:
> import 包中的模块

```
import sound.effects.echo
```

This loads the submodule `sound.effects.echo`. It must be referenced with its full name.
> import 的模块需要以全名引用 `xxx.xxx.<module-name>`

```
sound.effects.echo.echofilter(input, output, delay=0.7, atten=4)
```

An alternative way of importing the submodule is:

```
from sound.effects import echo
```

This also loads the submodule `echo`, and makes it available without its package prefix, so it can be used as follows:
> `from xxx.xxx import <module-name>` 直接注册了模块名称 `<module-name>` ，因此可以直接通过模块名称引用

```
echo.echofilter(input, output, delay=0.7, atten=4)
```

Yet another variation is to import the desired function or variable directly:
> 另外，可以直接 import 包中的函数或变量

```
from sound.effects.echo import echofilter
```

Again, this loads the submodule `echo`, but this makes its function `echofilter()` directly available:
> import 包中的函数或变量实际上会装载包，但会注册函数或变量名，因此可以直接引用它们

```
echofilter(input, output, delay=0.7, atten=4)
```

Note that when using `from package import item`, the item can be either a submodule (or subpackage) of the package, or some other name defined in the package, like a function, class or variable. The `import` statement first tests whether the item is defined in the package; if not, it assumes it is a module and attempts to load it. If it fails to find it, an [`ImportError`](https://docs.python.org/3/library/exceptions.html#ImportError "ImportError") exception is raised.
> import 可以导入包、子包、模块、函数、类、变量，import 首先检查导入的名称是否是包内定义的名称，然后再假定这是模块名称并进行装载
> 若找不到，则 `ImportError`

Contrarily, when using syntax like `import item.subitem.subsubitem`, each item except for the last must be a package; the last item can be a module or a package but can’t be a class or function or variable defined in the previous item.
> 语法 `import xx.xx.xx` 要求除了最后一个名称，其他名称都是包，最后一个名称可以是模块或包，但不能是前一个 item 内定义的函数或类或变量

### 6.4.1. Importing * From a Package
Now what happens when the user writes `from sound.effects import *`? Ideally, one would hope that this somehow goes out to the filesystem, finds which submodules are present in the package, and imports them all. This could take a long time and importing sub-modules might have unwanted side-effects that should only happen when the sub-module is explicitly imported.
> `from xxx.xxx import *` 默认会导入包中的所有子模块

The only solution is for the package author to provide an explicit index of the package. The [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement uses the following convention: if a package’s `__init__.py` code defines a list named `__all__`, it is taken to be the list of module names that should be imported when `from package import *` is encountered. It is up to the package author to keep this list up-to-date when a new version of the package is released. Package authors may also decide not to support it, if they don’t see a use for importing * from their package. For example, the file `sound/effects/__init__.py` could contain the following code:
> `import` 遵守：若包的 `__init__.py` 定义了名为 `__all__` 的列表，则在 `from package import *` 时，仅导入名称在该列表中的子模块

```
__all__ = ["echo", "surround", "reverse"]
```

This would mean that `from sound.effects import *` would import the three named submodules of the `sound.effects` package.

Be aware that submodules might become shadowed by locally defined names. For example, if you added a `reverse` function to the `sound/effects/__init__.py` file, the `from sound.effects import *` would only import the two submodules `echo` and `surround`, but _not_ the `reverse` submodule, because it is shadowed by the locally defined `reverse` function:
>  `__init__.py` 内定义的函数或变量名称会 shadow ` from xxx import *` 中导入的模块名，此时 `__all__` 中的对应名称指向的是该函数或变量

```
__all__ = [
    "echo",      # refers to the 'echo.py' file
    "surround",  # refers to the 'surround.py' file
    "reverse",   # !!! refers to the 'reverse' function now !!!
]

def reverse(msg: str):  # <-- this name shadows the 'reverse.py' submodule
    return msg[::-1]    #     in the case of a 'from sound.effects import *'
```

If `__all__` is not defined, the statement `from sound.effects import *` does _not_ import all submodules from the package `sound.effects` into the current namespace; it only ensures that the package `sound.effects` has been imported (possibly running any initialization code in `__init__.py`) and then imports whatever names are defined in the package. This includes any names defined (and submodules explicitly loaded) by `__init__.py`. It also includes any submodules of the package that were explicitly loaded by previous [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statements. Consider this code:
> 若 `__all__` 未定义，则 `from xxx import *` 不会将该包的子模块导入到当前的命名空间，仅会保证包的名称会被导入，同时会运行 `__init__.py` 

```
import sound.effects.echo
import sound.effects.surround
from sound.effects import *
```

In this example, the `echo` and `surround` modules are imported in the current namespace because they are defined in the `sound.effects` package when the `from...import` statement is executed. (This also works when `__all__` is defined.)

Although certain modules are designed to export only names that follow certain patterns when you use `import *`, it is still considered bad practice in production code.
> 不推荐 `import *`

Remember, there is nothing wrong with using `from package import specific_submodule`! In fact, this is the recommended notation unless the importing module needs to use submodules with the same name from different packages.
> 推荐使用 `from package import submodule`

### 6.4.2. Intra-package References
When packages are structured into subpackages (as with the `sound` package in the example), you can use absolute imports to refer to submodules of siblings packages. For example, if the module `sound.filters.vocoder` needs to use the `echo` module in the `sound.effects` package, it can use `from sound.effects import echo`.

You can also write relative imports, with the `from module import name` form of import statement. These imports use leading dots to indicate the current and parent packages involved in the relative import. From the `surround` module for example, you might use:
> `import` 中 `.` 表示当前模块所处的包，`..` 表示父包

```
from . import echo
from .. import formats
from ..filters import equalizer
```

Note that relative imports are based on the name of the current module. Since the name of the main module is always `"__main__"`, modules intended for use as the main module of a Python application must always use absolute imports.
> 注意作为主模块执行的模块的名称会被改为 `__main__` ，而相对 import 则是基于当前模块的名称来寻路的，因此如果需要使用相对 import 的模块不能作为主模块，主模块只能使用绝对 import

### 6.4.3. Packages in Multiple Directories
Packages support one more special attribute, [`__path__`](https://docs.python.org/3/reference/import.html#path__ "__path__"). This is initialized to be a list containing the name of the directory holding the package’s `__init__.py` before the code in that file is executed. This variable can be modified; doing so affects future searches for modules and subpackages contained in the package.
> 包还支持一个特殊属性：`__path__` ，该变量被初始化为包含了包的 `__init__.py` 的目录的名称，初始化在 `__init__.py` 被执行之前发生

While this feature is not often needed, it can be used to extend the set of modules found in a package.

Footnotes
\[1\] In fact function definitions are also ‘statements’ that are ‘executed’; the execution of a module-level function definition adds the function name to the module’s global namespace.

# 7. Input and Output
There are several ways to present the output of a program; data can be printed in a human-readable form, or written to a file for future use. This chapter will discuss some of the possibilities.

## 7.1. Fancier Output Formatting
So far we’ve encountered two ways of writing values: _expression statements_ and the [`print()`](https://docs.python.org/3/library/functions.html#print "print") function. (A third way is using the [`write()`](https://docs.python.org/3/library/io.html#io.TextIOBase.write "io.TextIOBase.write") method of file objects; the standard output file can be referenced as `sys.stdout`. See the Library Reference for more information on this.)

Often you’ll want more control over the formatting of your output than simply printing space-separated values. There are several ways to format output.

- To use [formatted string literals](https://docs.python.org/3/tutorial/inputoutput.html#tut-f-strings), begin a string with `f` or `F` before the opening quotation mark or triple quotation mark. Inside this string, you can write a Python expression between `{` and `}` characters that can refer to variables or literal values.
    
```
>>> year = 2016
>>> event = 'Referendum'
>>> f'Results of the {year} {event}'
'Results of the 2016 Referendum'
``` 
    
- The [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format "str.format") method of strings requires more manual effort. You’ll still use `{` and `}` to mark where a variable will be substituted and can provide detailed formatting directives, but you’ll also need to provide the information to be formatted. In the following code block there are two examples of how to format variables:
    
```
>>> yes_votes = 42_572_654
>>> total_votes = 85_705_149
>>> percentage = yes_votes / total_votes
>>> '{:-9} YES votes  {:2.2%}'.format(yes_votes, percentage)
' 42572654 YES votes  49.67%'
``` 
    
Notice how the `yes_votes` are padded with spaces and a negative sign only for negative numbers. The example also prints `percentage` multiplied by 100, with 2 decimal places and followed by a percent sign (see [Format Specification Mini-Language](https://docs.python.org/3/library/string.html#formatspec) for details).
    
- Finally, you can do all the string handling yourself by using string slicing and concatenation operations to create any layout you can imagine. The string type has some methods that perform useful operations for padding strings to a given column width.

When you don’t need fancy output but just want a quick display of some variables for debugging purposes, you can convert any value to a string with the [`repr()`](https://docs.python.org/3/library/functions.html#repr "repr") or [`str()`](https://docs.python.org/3/library/stdtypes.html#str "str") functions.
> `str()` 和 `repr()` 用于将任何值转换为字符串

The [`str()`](https://docs.python.org/3/library/stdtypes.html#str "str") function is meant to return representations of values which are fairly human-readable, while [`repr()`](https://docs.python.org/3/library/functions.html#repr "repr") is meant to generate representations which can be read by the interpreter (or will force a [`SyntaxError`](https://docs.python.org/3/library/exceptions.html#SyntaxError "SyntaxError") if there is no equivalent syntax). For objects which don’t have a particular representation for human consumption, [`str()`](https://docs.python.org/3/library/stdtypes.html#str "str") will return the same value as [`repr()`](https://docs.python.org/3/library/functions.html#repr "repr"). Many values, such as numbers or structures like lists and dictionaries, have the same representation using either function. Strings, in particular, have two distinct representations.
> `str()` 意在返回人类可读的表示，`repr()` 意在生成由解释器阅读的表示
> 对于不存在人类可读表示的对象，`str()` 返回的值和 `repr()` 相同
> 像数字和列表/字典这样的结构，事实上仅存在一种表示，而字符串则有两种不同的表示

Some examples:

```
>>> s = 'Hello, world.'
>>> str(s)
'Hello, world.'
>>> repr(s)
"'Hello, world.'"
>>> str(1/7)
'0.14285714285714285'
>>> x = 10 * 3.25
>>> y = 200 * 200
>>> s = 'The value of x is ' + repr(x) + ', and y is ' + repr(y) + '...'
>>> print(s)
The value of x is 32.5, and y is 40000...
>>> # The repr() of a string adds string quotes and backslashes:
>>> hello = 'hello, world\n'
>>> hellos = repr(hello)
>>> print(hellos)
'hello, world\n'
>>> # The argument to repr() may be any Python object:
>>> repr((x, y, ('spam', 'eggs')))
"(32.5, 40000, ('spam', 'eggs'))"
```

The [`string`](https://docs.python.org/3/library/string.html#module-string "string: Common string operations.") module contains a [`Template`](https://docs.python.org/3/library/string.html#string.Template "string.Template") class that offers yet another way to substitute values into strings, using placeholders like `$x` and replacing them with values from a dictionary, but offers much less control of the formatting.
> `string` 模块包含了 `Template` 类，也提供了一种格式化字符串的方式，它使用像 `$x` 这样的占位符，然后用字典中的值替换它们

### 7.1.1. Formatted String Literals
[Formatted string literals](https://docs.python.org/3/reference/lexical_analysis.html#f-strings) (also called f-strings for short) let you include the value of Python expressions inside a string by prefixing the string with `f` or `F` and writing expressions as `{expression}`.

An optional format specifier can follow the expression. This allows greater control over how the value is formatted. The following example rounds pi to three places after the decimal:
> 格式字符串中的 `{expression}` 的 expression 之后还可以跟随额外的格式指示符

```
>>> import math
>>> print(f'The value of pi is approximately {math.pi:.3f}.')
The value of pi is approximately 3.142.
```

Passing an integer after the `':'` will cause that field to be a minimum number of characters wide. This is useful for making columns line up.
> `:<int>` 指定宽度

```
>>> table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
>>> for name, phone in table.items():
...     print(f'{name:10} ==> {phone:10d}')
...
Sjoerd     ==>       4127
Jack       ==>       4098
Dcab       ==>       7678
```

Other modifiers can be used to convert the value before it is formatted. `'!a'` applies [`ascii()`](https://docs.python.org/3/library/functions.html#ascii "ascii"), `'!s'` applies [`str()`](https://docs.python.org/3/library/stdtypes.html#str "str"), and `'!r'` applies [`repr()`](https://docs.python.org/3/library/functions.html#repr "repr"):
> 在格式化之前将值转化：`!a/s/r`

```
>>> animals = 'eels'
>>> print(f'My hovercraft is full of {animals}.')
My hovercraft is full of eels.
>>> print(f'My hovercraft is full of {animals!r}.')
My hovercraft is full of 'eels'.
```

The `=` specifier can be used to expand an expression to the text of the expression, an equal sign, then the representation of the evaluated expression:
> `{expression-name=}` 格式化为 `expression-name=<expression-value>`

```
>>> bugs = 'roaches'
>>> count = 13
>>> area = 'living room'
>>> print(f'Debugging {bugs=} {count=} {area=}')
Debugging bugs='roaches' count=13 area='living room'
```

See [self-documenting expressions](https://docs.python.org/3/whatsnew/3.8.html#bpo-36817-whatsnew) for more information on the `=` specifier. For a reference on these format specifications, see the reference guide for the [Format Specification Mini-Language](https://docs.python.org/3/library/string.html#formatspec).

### 7.1.2. The String format() Method
Basic usage of the [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format "str.format") method looks like this:

```
>>> print('We are the {} who say "{}!"'.format('knights', 'Ni'))
We are the knights who say "Ni!"
```

The brackets and characters within them (called format fields) are replaced with the objects passed into the [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format "str.format") method. A number in the brackets can be used to refer to the position of the object passed into the [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format "str.format") method.

```
>>> print('{0} and {1}'.format('spam', 'eggs'))
spam and eggs
>>> print('{1} and {0}'.format('spam', 'eggs'))
eggs and spam
```

If keyword arguments are used in the [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format "str.format") method, their values are referred to by using the name of the argument.
> `str.format()` 还支持关键字参数，参数名称在 `{}` 中指定

```
>>> print('This {food} is {adjective}.'.format(
...       food='spam', adjective='absolutely horrible'))
This spam is absolutely horrible.
```

Positional and keyword arguments can be arbitrarily combined:

```
>>> print('The story of {0}, {1}, and {other}.'.format('Bill', 'Manfred', other='Georg'))
The story of Bill, Manfred, and Georg.
```

If you have a really long format string that you don’t want to split up, it would be nice if you could reference the variables to be formatted by name instead of by position. This can be done by simply passing the dict and using square brackets `'[]'` to access the keys.
> 参数为一个字典的巧妙用法，其中使用了 `[]` 来访问 keys，注意 keys 没有引号包围

```
>>> table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 8637678}
>>> print('Jack: {0[Jack]:d}; Sjoerd: {0[Sjoerd]:d}; '
...       'Dcab: {0[Dcab]:d}'.format(table))
Jack: 4098; Sjoerd: 4127; Dcab: 8637678
```

This could also be done by passing the `table` dictionary as keyword arguments with the `**` notation.
> 也可以用 `**` 来解包字典，得到一系列形式为 `key=val` 的关键字参数

```
>>> table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 8637678}
>>> print('Jack: {Jack:d}; Sjoerd: {Sjoerd:d}; Dcab: {Dcab:d}'.format(**table))
Jack: 4098; Sjoerd: 4127; Dcab: 8637678
```

This is particularly useful in combination with the built-in function [`vars()`](https://docs.python.org/3/library/functions.html#vars "vars"), which returns a dictionary containing all local variables:
> 和内建函数 `vars()` 结合使用
> `vars()` 返回包含了所有局部变量的字典

```
>>> table = {k: str(v) for k, v in vars().items()}
>>> message = " ".join([f'{k}: ' + '{' + k +'};' for k in table.keys()])
>>> print(message.format(**table))
__name__: __main__; __doc__: None; __package__: None; __loader__: ...
```

As an example, the following lines produce a tidily aligned set of columns giving integers and their squares and cubes:

```
>>> for x in range(1, 11):
...     print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))
...
 1   1    1
 2   4    8
 3   9   27
 4  16   64
 5  25  125
 6  36  216
 7  49  343
 8  64  512
 9  81  729
10 100 1000
```

For a complete overview of string formatting with [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format "str.format"), see [Format String Syntax](https://docs.python.org/3/library/string.html#formatstrings).

### 7.1.3. Manual String Formatting
Here’s the same table of squares and cubes, formatted manually:

```
>>> for x in range(1, 11):
...     print(repr(x).rjust(2), repr(x*x).rjust(3), end=' ')
...     # Note use of 'end' on previous line
...     print(repr(x*x*x).rjust(4))
...
 1   1    1
 2   4    8
 3   9   27
 4  16   64
 5  25  125
 6  36  216
 7  49  343
 8  64  512
 9  81  729
10 100 1000
```

(Note that the one space between each column was added by the way [`print()`](https://docs.python.org/3/library/functions.html#print "print") works: it always adds spaces between its arguments.)

The [`str.rjust()`](https://docs.python.org/3/library/stdtypes.html#str.rjust "str.rjust") method of string objects right-justifies a string in a field of a given width by padding it with spaces on the left. There are similar methods [`str.ljust()`](https://docs.python.org/3/library/stdtypes.html#str.ljust "str.ljust") and [`str.center()`](https://docs.python.org/3/library/stdtypes.html#str.center "str.center"). These methods do not write anything, they just return a new string. If the input string is too long, they don’t truncate it, but return it unchanged; this will mess up your column lay-out but that’s usually better than the alternative, which would be lying about a value. (If you really want truncation you can always add a slice operation, as in `x.ljust(n)[:n]`.)
> `str.rjust/ljust/center()` 不改变原 string，而是返回新的 string

There is another method, [`str.zfill()`](https://docs.python.org/3/library/stdtypes.html#str.zfill "str.zfill"), which pads a numeric string on the left with zeros. It understands about plus and minus signs:

```
>>> '12'.zfill(5)
'00012'
>>> '-3.14'.zfill(7)
'-003.14'
>>> '3.14159265359'.zfill(5)
'3.14159265359'
```

### 7.1.4. Old string formatting
The % operator (modulo) can also be used for string formatting. Given `format % values` (where _format_ is a string), `%` conversion specifications in _format_ are replaced with zero or more elements of _values_. This operation is commonly known as string interpolation. For example:

```
>>> import math
>>> print('The value of pi is approximately %5.3f.' % math.pi)
The value of pi is approximately 3.142.
```

More information can be found in the [printf-style String Formatting](https://docs.python.org/3/library/stdtypes.html#old-string-formatting) section.

## 7.2. Reading and Writing Files
[`open()`](https://docs.python.org/3/library/functions.html#open "open") returns a [file object](https://docs.python.org/3/glossary.html#term-file-object), and is most commonly used with two positional arguments and one keyword argument: `open(filename, mode, encoding=None)`
> `open()` 返回一个文件对象，一般会传入两个位置参数和一个关键字参数

```
>>> f = open('workfile', 'w', encoding="utf-8")
```

The first argument is a string containing the filename. The second argument is another string containing a few characters describing the way in which the file will be used. _mode_ can be `'r'` when the file will only be read, `'w'` for only writing (an existing file with the same name will be erased), and `'a'` opens the file for appending; any data written to the file is automatically added to the end. `'r+'` opens the file for both reading and writing. The _mode_ argument is optional; `'r'` will be assumed if it’s omitted.
> 默认为 `r`
> `w` : 重新写，`a` : append，`r+` ：读写

Normally, files are opened in _text mode_, that means, you read and write strings from and to the file, which are encoded in a specific _encoding_. If _encoding_ is not specified, the default is platform dependent (see [`open()`](https://docs.python.org/3/library/functions.html#open "open")). Because UTF-8 is the modern de-facto standard, `encoding="utf-8"` is recommended unless you know that you need to use a different encoding. Appending a `'b'` to the mode opens the file in _binary mode_. Binary mode data is read and written as [`bytes`](https://docs.python.org/3/library/stdtypes.html#bytes "bytes") objects. You can not specify _encoding_ when opening file in binary mode.
> 默认以 text mode 打开文件，即对于文件的读写以 string 的形式进行，string 通过特定的 encoding 编码，默认编码取决于平台
> `b` 为 binary mode，即对于文件的读写以 byte 的形式进行，无法指定 encoding

In text mode, the default when reading is to convert platform-specific line endings (`\n` on Unix, `\r\n` on Windows) to just `\n`. When writing in text mode, the default is to convert occurrences of `\n` back to platform-specific line endings. This behind-the-scenes modification to file data is fine for text files, but will corrupt binary data like that in `JPEG` or `EXE` files. Be very careful to use binary mode when reading and writing such files.
> text mode 下，读取时会默认将 `\r\n` 转化为 `\n` ，写入时会默认将 `\n` 转化为 `\r\n` (Windows 平台)
> 注意不要对于 binary 数据文件例如 JPEG/EXE 使用 text mode 打开，否则该操作会污染文件内容

It is good practice to use the [`with`](https://docs.python.org/3/reference/compound_stmts.html#with) keyword when dealing with file objects. The advantage is that the file is properly closed after its suite finishes, even if an exception is raised at some point. Using `with` is also much shorter than writing equivalent [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) - [`finally`](https://docs.python.org/3/reference/compound_stmts.html#finally) blocks:
> 处理文件对象时，推荐使用 `with` 关键字，以方便文件在完成操作后自动关闭，即使处理中会出现异常
> 而对于语法上等价的 `try-finally` 块则更为冗长

```
>>> with open('workfile', encoding="utf-8") as f:
...     read_data = f.read()

>>> # We can check that the file has been automatically closed.
>>> f.closed
True
```

If you’re not using the [`with`](https://docs.python.org/3/reference/compound_stmts.html#with) keyword, then you should call `f.close()` to close the file and immediately free up any system resources used by it.

Warning
Calling `f.write()` without using the `with` keyword or calling `f.close()` **might** result in the arguments of `f.write()` not being completely written to the disk, even if the program exits successfully.
> 如果没有正常关闭文件，即便程序正常退出，也可能导致 `f.write()` 不会完全将参数中的内容写入磁盘

After a file object is closed, either by a [`with`](https://docs.python.org/3/reference/compound_stmts.html#with) statement or by calling `f.close()`, attempts to use the file object will automatically fail.

```
>>> f.close()
>>> f.read()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: I/O operation on closed file.
```

### 7.2.1. Methods of File Objects
The rest of the examples in this section will assume that a file object called `f` has already been created.

To read a file’s contents, call `f.read(size)`, which reads some quantity of data and returns it as a string (in text mode) or bytes object (in binary mode). _size_ is an optional numeric argument. When _size_ is omitted or negative, the entire contents of the file will be read and returned; it’s your problem if the file is twice as large as your machine’s memory. Otherwise, at most _size_ characters (in text mode) or _size_ bytes (in binary mode) are read and returned. If the end of the file has been reached, `f.read()` will return an empty string (`''`).
> `f.read(size)` 读取一定数量的数据，在文本模式下，将其以字符串形式返回，在二进制模式下，将其以字节对象形式返回
> 若 size 忽略或为负数，则整个文件内容被读取并返回
> 达到文件末尾时，`f.read()` 返回空字符串

```
>>> f.read()
'This is the entire file.\n'
>>> f.read()
''
```

`f.readline()` reads a single line from the file; a newline character (`\n`) is left at the end of the string, and is only omitted on the last line of the file if the file doesn’t end in a newline. This makes the return value unambiguous; if `f.readline()` returns an empty string, the end of the file has been reached, while a blank line is represented by `'\n'`, a string containing only a single newline.
> `f.readline()` 读取单独一行，包括 `\n` ，仅有在文件的最后一行没有以换行符结束的情况下，才不会包括 `\n`
> 同样，到达文件末尾时，返回空字符串
> 读取空行时，返回 `\n`

```
>>> f.readline()
'This is the first line of the file.\n'
>>> f.readline()
'Second line of the file\n'
>>> f.readline()
''
```

For reading lines from a file, you can loop over the file object. This is memory efficient, fast, and leads to simple code:
> 直接遍历文件对象，可以得到文件的每一行

```
>>> for line in f:
...     print(line, end='')
...
This is the first line of the file.
Second line of the file
```

If you want to read all the lines of a file in a list you can also use `list(f)` or `f.readlines()`.
>  `list(f)/f.readlines()` 得到包含了文件所有行的列表

`f.write(string)` writes the contents of _string_ to the file, returning the number of characters written.
> `f.write(string)` 写入 string，返回写入的字符数

```
>>> f.write('This is a test\n')
15
```

Other types of objects need to be converted – either to a string (in text mode) or a bytes object (in binary mode) – before writing them:
> 要将其他类型的对象写入文件，需要先进行转化，转化为 string 或者 bytes 对象，对应两种模式

```
>>> value = ('the answer', 42)
>>> s = str(value)  # convert the tuple to string
>>> f.write(s)
18
```

`f.tell()` returns an integer giving the file object’s current position in the file represented as number of bytes from the beginning of the file when in binary mode and an opaque number when in text mode.
> `f.tell()` 返回一个整数，表示文件对象当前的位置，在 binary 模式下，即从文件开始到当前位置的字节数，text 模式下，则返回一个 opaque number

To change the file object’s position, use `f.seek(offset, whence)`. The position is computed from adding _offset_ to a reference point; the reference point is selected by the _whence_ argument. A _whence_ value of 0 measures from the beginning of the file, 1 uses the current file position, and 2 uses the end of the file as the reference point. _whence_ can be omitted and defaults to 0, using the beginning of the file as the reference point.
> `f.seek(offset, whence)` 用于改变文件对象的位置，通过从当前引用的点加上 `offset` 得到，当前引用的点由 `whence` 决定
> whence 为 0 表示文件开始，2 表示文件结尾，默认为 0

```
>>> f = open('workfile', 'rb+')
>>> f.write(b'0123456789abcdef')
16
>>> f.seek(5)      # Go to the 6th byte in the file
5
>>> f.read(1)
b'5'
>>> f.seek(-3, 2)  # Go to the 3rd byte before the end
13
>>> f.read(1)
b'd'
```

In text files (those opened without a `b` in the mode string), only seeks relative to the beginning of the file are allowed (the exception being seeking to the very file end with `seek(0, 2)`) and the only valid _offset_ values are those returned from the `f.tell()`, or zero. Any other _offset_ value produces undefined behaviour.
> text 模式下，仅允许从文件头部开始 seek，或者 `seek(0,2)`
> 同时仅允许用 `f.tell()` 的返回值或者 0 作为 offset，否则行为未定义

File objects have some additional methods, such as [`isatty()`](https://docs.python.org/3/library/io.html#io.IOBase.isatty "io.IOBase.isatty") and [`truncate()`](https://docs.python.org/3/library/io.html#io.IOBase.truncate "io.IOBase.truncate") which are less frequently used; consult the Library Reference for a complete guide to file objects.
j
### 7.2.2. Saving structured data with `json`
Strings can easily be written to and read from a file. Numbers take a bit more effort, since the [`read()`](https://docs.python.org/3/library/io.html#io.TextIOBase.read "io.TextIOBase.read") method only returns strings, which will have to be passed to a function like [`int()`](https://docs.python.org/3/library/functions.html#int "int"), which takes a string like `'123'` and returns its numeric value 123. When you want to save more complex data types like nested lists and dictionaries, parsing and serializing by hand becomes complicated.
> `read()` 方法仅返回 string

Rather than having users constantly writing and debugging code to save complicated data types to files, Python allows you to use the popular data interchange format called [JSON (JavaScript Object Notation)](https://json.org/). The standard module called [`json`](https://docs.python.org/3/library/json.html#module-json "json: Encode and decode the JSON format.") can take Python data hierarchies, and convert them to string representations; this process is called _serializing_. Reconstructing the data from the string representation is called _deserializing_. Between serializing and deserializing, the string representing the object may have been stored in a file or data, or sent over a network connection to some distant machine.
> Python 支持 JSON 数据交换格式，标准模块 `json` 接受 Python 数据结构，将其转化为字符串表示，该过程为 serializing；同时可以从字符串表示中重构 Python 数据结构，该过程为 deserializing

Note
The JSON format is commonly used by modern applications to allow for data exchange. Many programmers are already familiar with it, which makes it a good choice for interoperability.

If you have an object `x`, you can view its JSON string representation with a simple line of code:
> `json.dumps()` 将 Python 对象转化为 JSON 字符串表示

```
>>> import json
>>> x = [1, 'simple', 'list']
>>> json.dumps(x)
'[1, "simple", "list"]'
```

Another variant of the [`dumps()`](https://docs.python.org/3/library/json.html#json.dumps "json.dumps") function, called [`dump()`](https://docs.python.org/3/library/json.html#json.dump "json.dump"), simply serializes the object to a [text file](https://docs.python.org/3/glossary.html#term-text-file). So if `f` is a [text file](https://docs.python.org/3/glossary.html#term-text-file) object opened for writing, we can do this:
> `json.dump()` 将 Python 对象序列化到一个文本文件中

```
json.dump(x, f)
```

To decode the object again, if `f` is a [binary file](https://docs.python.org/3/glossary.html#term-binary-file) or [text file](https://docs.python.org/3/glossary.html#term-text-file) object which has been opened for reading:
> `json.load(f)` 将文件中的对象表示逆序列化到 Python 对象，文件可以是二进制也可以是文本

```
x = json.load(f)
```

Note
JSON files must be encoded in UTF-8. Use `encoding="utf-8"` when opening JSON file as a [text file](https://docs.python.org/3/glossary.html#term-text-file) for both of reading and writing.
> JSON 文件必须以 UTF-8 编码打开

This simple serialization technique can handle lists and dictionaries, but serializing arbitrary class instances in JSON requires a bit of extra effort. The reference for the [`json`](https://docs.python.org/3/library/json.html#module-json "json: Encode and decode the JSON format.") module contains an explanation of this.
> 该序列化技巧对于列表和字典较为有效，对于对象需要其他处理方法

See also
[`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle "pickle: Convert Python objects to streams of bytes and back.") - the pickle module

Contrary to [JSON](https://docs.python.org/3/tutorial/inputoutput.html#tut-json), _pickle_ is a protocol which allows the serialization of arbitrarily complex Python objects. As such, it is specific to Python and cannot be used to communicate with applications written in other languages. It is also insecure by default: deserializing pickle data coming from an untrusted source can execute arbitrary code, if the data was crafted by a skilled attacker.
> pickle 允许序列化任意复杂的 Python 对象，但仅能由 Python 读取
> pickle 默认不安全，逆序列化 pickle 数据可能会导致机器执行任意的代码

# 8. Errors and Exceptions
Until now error messages haven’t been more than mentioned, but if you have tried out the examples you have probably seen some. There are (at least) two distinguishable kinds of errors: _syntax errors_ and _exceptions_.
> 主要有两类错误：语法错误和异常

## 8.1. Syntax Errors
Syntax errors, also known as parsing errors, are perhaps the most common kind of complaint you get while you are still learning Python:
> 语法错误也就是解析错误

```
>>> while True print('Hello world')
  File "<stdin>", line 1
    while True print('Hello world')
               ^^^^^
SyntaxError: invalid syntax
```

The parser repeats the offending line and displays little ‘arrow’s pointing at the token in the line where the error was detected. The error may be caused by the absence of a token _before_ the indicated token. In the example, the error is detected at the function [`print()`](https://docs.python.org/3/library/functions.html#print "print"), since a colon (`':'`) is missing before it. File name and line number are printed so you know where to look in case the input came from a script.

## 8.2. Exceptions
Even if a statement or expression is syntactically correct, it may cause an error when an attempt is made to execute it. Errors detected during execution are called _exceptions_ and are not unconditionally fatal: you will soon learn how to handle them in Python programs. Most exceptions are not handled by programs, however, and result in error messages as shown here:
> 在执行时检查到的错误称为异常
> 大多数异常不会被处理，而是会被抛出

```
>>> 10 * (1/0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ZeroDivisionError: division by zero
>>> 4 + spam*3
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'spam' is not defined
>>> '2' + 2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can only concatenate str (not "int") to str
```

The last line of the error message indicates what happened. Exceptions come in different types, and the type is printed as part of the message: the types in the example are [`ZeroDivisionError`](https://docs.python.org/3/library/exceptions.html#ZeroDivisionError "ZeroDivisionError"), [`NameError`](https://docs.python.org/3/library/exceptions.html#NameError "NameError") and [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "TypeError"). The string printed as the exception type is the name of the built-in exception that occurred. This is true for all built-in exceptions, but need not be true for user-defined exceptions (although it is a useful convention). Standard exception names are built-in identifiers (not reserved keywords).
> 异常有多种类型，打印出的异常类型就是发生的内建异常的名称，标准异常的名称都是内建的 identifier

The rest of the line provides detail based on the type of exception and what caused it.

The preceding part of the error message shows the context where the exception occurred, in the form of a stack traceback. In general it contains a stack traceback listing source lines; however, it will not display lines read from standard input.

[Built-in Exceptions](https://docs.python.org/3/library/exceptions.html#bltin-exceptions) lists the built-in exceptions and their meanings.

## 8.3. Handling Exceptions
It is possible to write programs that handle selected exceptions. Look at the following example, which asks the user for input until a valid integer has been entered, but allows the user to interrupt the program (using Control-C or whatever the operating system supports); note that a user-generated interruption is signalled by raising the [`KeyboardInterrupt`](https://docs.python.org/3/library/exceptions.html#KeyboardInterrupt "KeyboardInterrupt") exception.
> Python 允许编写处理特定异常的程序
> 例如以下程序要求用户输入有效的整数，每次异常都会被捕获并且被处理
> 注意到用户生成的中断会通过发起 `KeyboardInterrupt` 异常来提出

```
>>> while True:
...     try:
...         x = int(input("Please enter a number: "))
...         break
...     except ValueError:
...         print("Oops!  That was no valid number.  Try again...")
...
```

The [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) statement works as follows.
- First, the _try clause_ (the statement(s) between the [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) and [`except`](https://docs.python.org/3/reference/compound_stmts.html#except) keywords) is executed.
- If no exception occurs, the _except clause_ is skipped and execution of the [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) statement is finished.
- If an exception occurs during execution of the [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) clause, the rest of the clause is skipped. Then, if its type matches the exception named after the [`except`](https://docs.python.org/3/reference/compound_stmts.html#except) keyword, the _except clause_ is executed, and then execution continues after the try/except block.
- If an exception occurs which does not match the exception named in the _except clause_, it is passed on to outer [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) statements; if no handler is found, it is an _unhandled exception_ and execution stops with an error message.
> `try` 语句的工作方式：
> 首先执行 try clause
> 若没有异常发生，跳过 except clause，try 语句的执行完成
> 若异常发生，跳过剩余的 try clause，如果异常类型匹配 except 之后指定的异常，执行 except clause，如果没有匹配，跳往 try 语句之外，如果没有找到异常处理语句，则作为未处理的异常被抛出

A [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) statement may have more than one _except clause_, to specify handlers for different exceptions. At most one handler will be executed. Handlers only handle exceptions that occur in the corresponding _try clause_, not in other handlers of the same `try` statement. An _except clause_ may name multiple exceptions as a parenthesized tuple, for example:
> try 语句可以有多个 except clause，但最多只有一个被执行
> except 子句中的异常处理语句仅处理 try clause 中出现的异常，其他 except clause 中的异常与它无关
> 一个 except clause 可以捕获多个异常，异常放在一个元组中

```
... except (RuntimeError, TypeError, NameError):
...     pass
```

A class in an [`except`](https://docs.python.org/3/reference/compound_stmts.html#except) clause matches exceptions which are instances of the class itself or one of its derived classes (but not the other way around — an _except clause_ listing a derived class does not match instances of its base classes). For example, the following code will print B, C, D in that order:
> except clause 中的类会匹配它的实例或者它的衍生类的实例，但反过来不行，衍生类不会匹配基类的示例

```
class B(Exception):
    pass

class C(B):
    pass

class D(C):
    pass

for cls in [B, C, D]:
    try:
        raise cls()
    except D:
        print("D")
    except C:
        print("C")
    except B:
        print("B")
```

Note that if the _except clauses_ were reversed (with `except B` first), it would have printed B, B, B — the first matching _except clause_ is triggered.

When an exception occurs, it may have associated values, also known as the exception’s _arguments_. The presence and types of the arguments depend on the exception type.
> 异常发生时，它会有关联的值，也被称为异常的参数
> 异常参数的类型取决于异常的类型

The _except clause_ may specify a variable after the exception name. The variable is bound to the exception instance which typically has an `args` attribute that stores the arguments. For convenience, builtin exception types define [`__str__()`](https://docs.python.org/3/reference/datamodel.html#object.__str__ "object.__str__") to print all the arguments without explicitly accessing `.args`.
> except 子句可以在异常名称之后指定一个变量，该变量会和异常实例绑定，并且会有 `args` 属性，存储了异常的参数
> 内建的异常类型定义了 `__str__()` 方法，用于打印出所有的参数

```
>>> try:
...     raise Exception('spam', 'eggs')
... except Exception as inst:
...     print(type(inst))    # the exception type
...     print(inst.args)     # arguments stored in .args
...     print(inst)          # __str__ allows args to be printed directly,
...                          # but may be overridden in exception subclasses
...     x, y = inst.args     # unpack args
...     print('x =', x)
...     print('y =', y)
...
<class 'Exception'>
('spam', 'eggs')
('spam', 'eggs')
x = spam
y = eggs
```

The exception’s [`__str__()`](https://docs.python.org/3/reference/datamodel.html#object.__str__ "object.__str__") output is printed as the last part (‘detail’) of the message for unhandled exceptions.
> 异常的 `__str__()` 输出会作为未处理异常的 detail 信息被打印出

[`BaseException`](https://docs.python.org/3/library/exceptions.html#BaseException "BaseException") is the common base class of all exceptions. One of its subclasses, [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception "Exception"), is the base class of all the non-fatal exceptions. Exceptions which are not subclasses of [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception "Exception") are not typically handled, because they are used to indicate that the program should terminate. They include [`SystemExit`](https://docs.python.org/3/library/exceptions.html#SystemExit "SystemExit") which is raised by [`sys.exit()`](https://docs.python.org/3/library/sys.html#sys.exit "sys.exit") and [`KeyboardInterrupt`](https://docs.python.org/3/library/exceptions.html#KeyboardInterrupt "KeyboardInterrupt") which is raised when a user wishes to interrupt the program.
> `BaseException` 是所有异常类的共同基类，它的一个子类 `Exception` 是所有 non-fatal 异常的共同基类
> 不是 `Exception` 的子类的异常类 (fatal) 一般不会被处理，而是用于表明程序应该终止，它们包括了 `SystemExit` (由 `sys.exit()` 抛出)，以及 `KeyboardInterrupt` (由用户中断抛出)

[`Exception`](https://docs.python.org/3/library/exceptions.html#Exception "Exception") can be used as a wildcard that catches (almost) everything. However, it is good practice to be as specific as possible with the types of exceptions that we intend to handle, and to allow any unexpected exceptions to propagate on.
> `Exception` 可以用于通配符，捕获几乎任意异常

The most common pattern for handling [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception "Exception") is to print or log the exception and then re-raise it (allowing a caller to handle the exception as well):
> 最常见的处理异常的模式是首先打印或者记录异常，然后重新抛出它，以允许调用者处理该异常

```python
import sys

try:
    f = open('myfile.txt')
    s = f.readline()
    i = int(s.strip())
except OSError as err:
    print("OS error:", err)
except ValueError:
    print("Could not convert data to an integer.")
except Exception as err:
    print(f"Unexpected {err=}, {type(err)=}")
    raise
```

The [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) … [`except`](https://docs.python.org/3/reference/compound_stmts.html#except) statement has an optional _else clause_, which, when present, must follow all _except clauses_. It is useful for code that must be executed if the _try clause_ does not raise an exception. For example:
> try...except 语句有一个可选的 else 子句，它必须在所有 except 子句之后
> 它在 try 子句没有发出异常时被执行

```python
for arg in sys.argv[1:]:
    try:
        f = open(arg, 'r')
    except OSError:
        print('cannot open', arg)
    else:
        print(arg, 'has', len(f.readlines()), 'lines')
        f.close()
```

The use of the `else` clause is better than adding additional code to the [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) clause because it avoids accidentally catching an exception that wasn’t raised by the code being protected by the `try` … `except` statement.

Exception handlers do not handle only exceptions that occur immediately in the _try clause_, but also those that occur inside functions that are called (even indirectly) in the _try clause_. For example:
> try 子句中调用的函数发生了异常，也会被捕获

```
>>> def this_fails():
...     x = 1/0
...
>>> try:
...     this_fails()
... except ZeroDivisionError as err:
...     print('Handling run-time error:', err)
...
Handling run-time error: division by zero
```

## 8.4. Raising Exceptions
The [`raise`](https://docs.python.org/3/reference/simple_stmts.html#raise) statement allows the programmer to force a specified exception to occur. For example:
> `raise` 语句用于提出异常

```
>>> raise NameError('HiThere')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: HiThere
```

The sole argument to [`raise`](https://docs.python.org/3/reference/simple_stmts.html#raise) indicates the exception to be raised. This must be either an exception instance or an exception class (a class that derives from [`BaseException`](https://docs.python.org/3/library/exceptions.html#BaseException "BaseException"), such as [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception "Exception") or one of its subclasses). If an exception class is passed, it will be implicitly instantiated by calling its constructor with no arguments:
> `raise` 语句的参数需要是一个异常类或者一个异常实例，若是一个异常类，它也会被调用无参数的构造函数被实例化

```python
raise ValueError  # shorthand for 'raise ValueError()'
```

If you need to determine whether an exception was raised but don’t intend to handle it, a simpler form of the [`raise`](https://docs.python.org/3/reference/simple_stmts.html#raise) statement allows you to re-raise the exception:
> 在 except 子句中的 `raise` 会再次抛出被捕获的异常

```
>>> try:
...     raise NameError('HiThere')
... except NameError:
...     print('An exception flew by!')
...     raise
...
An exception flew by!
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
NameError: HiThere
```

## 8.5. Exception Chaining
If an unhandled exception occurs inside an [`except`](https://docs.python.org/3/reference/compound_stmts.html#except) section, it will have the exception being handled attached to it and included in the error message:
> 若在 except 子句中出现了其他异常，则 except 捕获的未被处理的异常和新的异常会被一起抛出

```
>>> try:
...     open("database.sqlite")
... except OSError:
...     raise RuntimeError("unable to handle error")
...
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
FileNotFoundError: [Errno 2] No such file or directory: 'database.sqlite'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
RuntimeError: unable to handle error
```

To indicate that an exception is a direct consequence of another, the [`raise`](https://docs.python.org/3/reference/simple_stmts.html#raise) statement allows an optional [`from`](https://docs.python.org/3/reference/simple_stmts.html#raise) clause:
> `raise` 语句允许一个可选的 `from` 子句用于表明一个异常是另一个异常的直接结果，`from` 后面接一个异常实例或者 `None`

```
# exc must be exception instance or None.
raise RuntimeError from exc
```

This can be useful when you are transforming exceptions. For example:

```
>>> def func():
...     raise ConnectionError
...
>>> try:
...     func()
... except ConnectionError as exc:
...     raise RuntimeError('Failed to open database') from exc
...
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "<stdin>", line 2, in func
ConnectionError

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
RuntimeError: Failed to open database
```

It also allows disabling automatic exception chaining using the `from None` idiom:
> 如果使用 `from None` 可以取消自动的异常链接，仅抛出最新的异常

```
>>> try:
...     open('database.sqlite')
... except OSError:
...     raise RuntimeError from None
...
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
RuntimeError
```

For more information about chaining mechanics, see [Built-in Exceptions](https://docs.python.org/3/library/exceptions.html#bltin-exceptions).

## 8.6. User-defined Exceptions
Programs may name their own exceptions by creating a new exception class (see [Classes](https://docs.python.org/3/tutorial/classes.html#tut-classes) for more about Python classes). Exceptions should typically be derived from the [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception "Exception") class, either directly or indirectly.
> 用户可以通过直接从 `Exception` 类直接或间接地继承来定义自己的异常类

Exception classes can be defined which do anything any other class can do, but are usually kept simple, often only offering a number of attributes that allow information about the error to be extracted by handlers for the exception.
> 异常类一般保持简单，仅提供一定数量的属性，方便异常处理语句提取错误信息

Most exceptions are defined with names that end in “Error”, similar to the naming of the standard exceptions.
> 大多数异常的名称以 `Error` 结尾

Many standard modules define their own exceptions to report errors that may occur in functions they define.
> 许多标准模块定义了它们自己的异常，用于报告它们自己定义的函数中会发生的异常

## 8.7. Defining Clean-up Actions
The [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) statement has another optional clause which is intended to define clean-up actions that must be executed under all circumstances. For example:
> try 语句的另一个可选子句是 finally，它用于定义必须在任意情况下都需要执行的清洁动作

```
>>> try:
...     raise KeyboardInterrupt
... finally:
...     print('Goodbye, world!')
...
Goodbye, world!
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
KeyboardInterrupt
```

If a [`finally`](https://docs.python.org/3/reference/compound_stmts.html#finally) clause is present, the `finally` clause will execute as the last task before the [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) statement completes. The `finally` clause runs whether or not the `try` statement produces an exception. The following points discuss more complex cases when an exception occurs:
> `finally` 子句会作为 try 语句执行的最后一个子句，无论是否发生异常

- If an exception occurs during execution of the `try` clause, the exception may be handled by an [`except`](https://docs.python.org/3/reference/compound_stmts.html#except) clause. If the exception is not handled by an `except` clause, the exception is re-raised after the `finally` clause has been executed.
> 若异常在 try 子句内发生，没有被 except 子句处理，则会在 finally 子句执行后再被抛出
- An exception could occur during execution of an `except` or `else` clause. Again, the exception is re-raised after the `finally` clause has been executed.
> 若异常在 except 或 else 中发生，则该异常在 finally 子句执行后被抛出   
- If the `finally` clause executes a [`break`](https://docs.python.org/3/reference/simple_stmts.html#break), [`continue`](https://docs.python.org/3/reference/simple_stmts.html#continue) or [`return`](https://docs.python.org/3/reference/simple_stmts.html#return) statement, exceptions are not re-raised.
> 若 finally 子句执行了 break/continue/return，则异常不会再次被抛出 
- If the `try` statement reaches a [`break`](https://docs.python.org/3/reference/simple_stmts.html#break), [`continue`](https://docs.python.org/3/reference/simple_stmts.html#continue) or [`return`](https://docs.python.org/3/reference/simple_stmts.html#return) statement, the `finally` clause will execute just prior to the `break`, `continue` or `return` statement’s execution.
> 若 try 语句包含了 break/continue/return，则 finally 子句会在它们的执行之前被执行
- If a `finally` clause includes a `return` statement, the returned value will be the one from the `finally` clause’s `return` statement, not the value from the `try` clause’s `return` statement.
> 若 finally 子句包含了 return 语句，程序的返回值将是 finally 中的 return 返回的，而不是 try 子句中的 return 返回的 

For example:

```
>>> def bool_return():
...     try:
...         return True
...     finally:
...         return False
...
>>> bool_return()
False
```

A more complicated example:

```
>>> def divide(x, y):
...     try:
...         result = x / y
...     except ZeroDivisionError:
...         print("division by zero!")
...     else:
...         print("result is", result)
...     finally:
...         print("executing finally clause")
...
>>> divide(2, 1)
result is 2.0
executing finally clause
>>> divide(2, 0)
division by zero!
executing finally clause
>>> divide("2", "1")
executing finally clause
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 3, in divide
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

As you can see, the [`finally`](https://docs.python.org/3/reference/compound_stmts.html#finally) clause is executed in any event. The [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "TypeError") raised by dividing two strings is not handled by the [`except`](https://docs.python.org/3/reference/compound_stmts.html#except) clause and therefore re-raised after the `finally` clause has been executed.

In real world applications, the [`finally`](https://docs.python.org/3/reference/compound_stmts.html#finally) clause is useful for releasing external resources (such as files or network connections), regardless of whether the use of the resource was successful.
> 实际应用中，finally 非常适合用于在无论资源的使用是否成功的情况下释放外部资源，例如文件和网络链接

## 8.8. Predefined Clean-up Actions
Some objects define standard clean-up actions to be undertaken when the object is no longer needed, regardless of whether or not the operation using the object succeeded or failed. Look at the following example, which tries to open a file and print its contents to the screen.
> 一些对象定义了在对象再不被需要的时候的标准的 clean-up 动作，这些 clean-up 与使用该对象的操作是否成功或失败无关

```python
for line in open("myfile.txt"):
    print(line, end="")
```

The problem with this code is that it leaves the file open for an indeterminate amount of time after this part of the code has finished executing. This is not an issue in simple scripts, but can be a problem for larger applications. The [`with`](https://docs.python.org/3/reference/compound_stmts.html#with) statement allows objects like files to be used in a way that ensures they are always cleaned up promptly and correctly.
> `with` 语句允许像文件这样的对象总是会被正确地并且及时地被清理

```python
with open("myfile.txt") as f:
    for line in f:
        print(line, end="")
```

After the statement is executed, the file _f_ is always closed, even if a problem was encountered while processing the lines. Objects which, like files, provide predefined clean-up actions will indicate this in their documentation.
> `f` 总是会被预定义的 clean-up 动作关闭，即便在处理时出现了问题

## 8.9. Raising and Handling Multiple Unrelated Exceptions
There are situations where it is necessary to report several exceptions that have occurred. This is often the case in concurrency frameworks, when several tasks may have failed in parallel, but there are also other use cases where it is desirable to continue execution and collect multiple errors rather than raise the first exception.
> 有时我们需要报告多个出现的异常，例如在并发性框架内，我们希望继续执行，并且收集更多的错误信息，而不是直接抛出第一个异常

The builtin [`ExceptionGroup`](https://docs.python.org/3/library/exceptions.html#ExceptionGroup "ExceptionGroup") wraps a list of exception instances so that they can be raised together. It is an exception itself, so it can be caught like any other exception.
> 内建的 `ExceptionGroup` 包含了一系列的异常实例，因此它们可以被同时抛出，同时 `ExceptionGroup` 本身也是异常类，因此可以被捕获

```
>>> def f():
...     excs = [OSError('error 1'), SystemError('error 2')]
...     raise ExceptionGroup('there were problems', excs)
...
>>> f()
  + Exception Group Traceback (most recent call last):
  |   File "<stdin>", line 1, in <module>
  |   File "<stdin>", line 3, in f
  | ExceptionGroup: there were problems
  +-+---------------- 1 ----------------
    | OSError: error 1
    +---------------- 2 ----------------
    | SystemError: error 2
    +------------------------------------
>>> try:
...     f()
... except Exception as e:
...     print(f'caught {type(e)}: e')
...
caught <class 'ExceptionGroup'>: e
>>>
```

By using `except*` instead of `except`, we can selectively handle only the exceptions in the group that match a certain type. In the following example, which shows a nested exception group, each `except*` clause extracts from the group exceptions of a certain type while letting all other exceptions propagate to other clauses and eventually to be reraised.
> 使用 `except*` ，我们可以选择性处理 group 中匹配特定类型的异常，让其他的异常继续传播

```
>>> def f():
...     raise ExceptionGroup(
...         "group1",
...         [
...             OSError(1),
...             SystemError(2),
...             ExceptionGroup(
...                 "group2",
...                 [
...                     OSError(3),
...                     RecursionError(4)
...                 ]
...             )
...         ]
...     )
...
>>> try:
...     f()
... except* OSError as e:
...     print("There were OSErrors")
... except* SystemError as e:
...     print("There were SystemErrors")
...
There were OSErrors
There were SystemErrors
  + Exception Group Traceback (most recent call last):
  |   File "<stdin>", line 2, in <module>
  |   File "<stdin>", line 2, in f
  | ExceptionGroup: group1
  +-+---------------- 1 ----------------
    | ExceptionGroup: group2
    +-+---------------- 1 ----------------
      | RecursionError: 4
      +------------------------------------
>>>
```

Note that the exceptions nested in an exception group must be instances, not types. This is because in practice the exceptions would typically be ones that have already been raised and caught by the program, along the following pattern:
> 注意嵌套在 exception group 内的异常必须是实例，而不是类型，因为实践中用于构造 ExceptionGroup 的异常一般是已经被抛出并且被程序捕获的异常实例，一般的模式如下

```
>>> excs = []
... for test in tests:
...     try:
...         test.run()
...     except Exception as e:
...         excs.append(e)
...
>>> if excs:
...    raise ExceptionGroup("Test Failures", excs)
...
```

## 8.10. Enriching Exceptions with Notes
When an exception is created in order to be raised, it is usually initialized with information that describes the error that has occurred. There are cases where it is useful to add information after the exception was caught. For this purpose, exceptions have a method `add_note(note)` that accepts a string and adds it to the exception’s notes list. The standard traceback rendering includes all notes, in the order they were added, after the exception.
> 异常在要被抛出时被创建时，它一般会使用描述了发生了错误的信息初始化
> 我们可以在异常被捕获后添加信息，使用 `add_note()` 方法
> 该方法接受字符串，并将它加入异常的 notes list
> 标准的 traceback 会渲染全部的 notes，顺序按照它们加入的顺序

```
>>> try:
...     raise TypeError('bad type')
... except Exception as e:
...     e.add_note('Add some information')
...     e.add_note('Add some more information')
...     raise
...
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
TypeError: bad type
Add some information
Add some more information
>>>
```

For example, when collecting exceptions into an exception group, we may want to add context information for the individual errors. In the following each exception in the group has a note indicating when this error has occurred.
> 例如收集异常到一个 exception group 时，我们可能希望为各个独立的 error 添加上下文信息

```
>>> def f():
...     raise OSError('operation failed')
...
>>> excs = []
>>> for i in range(3):
...     try:
...         f()
...     except Exception as e:
...         e.add_note(f'Happened in Iteration {i+1}')
...         excs.append(e)
...
>>> raise ExceptionGroup('We have some problems', excs)
  + Exception Group Traceback (most recent call last):
  |   File "<stdin>", line 1, in <module>
  | ExceptionGroup: We have some problems (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "<stdin>", line 3, in <module>
    |   File "<stdin>", line 2, in f
    | OSError: operation failed
    | Happened in Iteration 1
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "<stdin>", line 3, in <module>
    |   File "<stdin>", line 2, in f
    | OSError: operation failed
    | Happened in Iteration 2
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "<stdin>", line 3, in <module>
    |   File "<stdin>", line 2, in f
    | OSError: operation failed
    | Happened in Iteration 3
    +------------------------------------
>>>
```

# 9. Classes
Classes provide a means of bundling data and functionality together. Creating a new class creates a new _type_ of object, allowing new _instances_ of that type to be made. Each class instance can have attributes attached to it for maintaining its state. Class instances can also have methods (defined by its class) for modifying its state.
> 类将数据和函数捆绑在一起，创造一个新类就是创造一个新的对象类型，并且允许该对象实例化
> 每个类实例都有对应的属性用于管理它的状态，类实例拥有类定义的方法用于修改它的状态

Compared with other programming languages, Python’s class mechanism adds classes with a minimum of new syntax and semantics. It is a mixture of the class mechanisms found in C++ and Modula-3. Python classes provide all the standard features of Object Oriented Programming: the class inheritance mechanism allows multiple base classes, a derived class can override any methods of its base class or classes, and a method can call the method of a base class with the same name. Objects can contain arbitrary amounts and kinds of data. As is true for modules, classes partake of the dynamic nature of Python: they are created at runtime, and can be modified further after creation.
> Python 提供了面向对象编程的所有标准特性：允许继承多个基类，允许衍生类覆盖基类的方法，同时衍生类的方法可以调用基类中和它同名的方法，对象可以包含任意数量和类型的数据
> 模块和类都遵循 Python 的动态性质：它们都在运行时创建，并且可以在创建后被修改

In C++ terminology, normally class members (including the data members) are _public_ (except see below [Private Variables](https://docs.python.org/3/tutorial/classes.html#tut-private)), and all member functions are _virtual_. As in Modula-3, there are no shorthands for referencing the object’s members from its methods: the method function is declared with an explicit first argument representing the object, which is provided implicitly by the call. As in Smalltalk, classes themselves are objects. This provides semantics for importing and renaming. Unlike C++ and Modula-3, built-in types can be used as base classes for extension by the user. Also, like in C++, most built-in operators with special syntax (arithmetic operators, subscripting etc.) can be redefined for class instances.
> 所有的类成员都是公有，所有的成员函数都是虚函数
> 成员函数的声明需要以 self 参数作为第一个参数，以显式地代表对象，在函数被调用时，它会隐式地被提供
> 类本身也是对象，因此可以被导入和重命名
> 内建类型可以被用作基类，方便用户拓展
> 大多数有特殊语法的内建运算符可以为类成员重定义

(Lacking universally accepted terminology to talk about classes, I will make occasional use of Smalltalk and C++ terms. I would use Modula-3 terms, since its object-oriented semantics are closer to those of Python than C++, but I expect that few readers have heard of it.)

## 9.1. A Word About Names and Objects
Objects have individuality, and multiple names (in multiple scopes) can be bound to the same object. This is known as aliasing in other languages. This is usually not appreciated on a first glance at Python, and can be safely ignored when dealing with immutable basic types (numbers, strings, tuples). However, aliasing has a possibly surprising effect on the semantics of Python code involving mutable objects such as lists, dictionaries, and most other types. This is usually used to the benefit of the program, since aliases behave like pointers in some respects. For example, passing an object is cheap since only a pointer is passed by the implementation; and if a function modifies an object passed as an argument, the caller will see the change — this eliminates the need for two different argument passing mechanisms as in Pascal.
> 对象可以有多个名字 (在不同的作用域 )绑定到同一个对象，在其他语言中，这被称为别名
> 别名在某种程度上起到指针的作用，因此，传递一个对象实际上是开销不大的，因为实际上只会传递一个指针，并且如果函数修改了作为参数传递给它的对象，函数的调用者也会发现对象的改变

## 9.2. Python Scopes and Namespaces
Before introducing classes, I first have to tell you something about Python’s scope rules. Class definitions play some neat tricks with namespaces, and you need to know how scopes and namespaces work to fully understand what’s going on. Incidentally, knowledge about this subject is useful for any advanced Python programmer.
> 我们首要介绍 Python 的作用域规则

Let’s begin with some definitions.

A _namespace_ is a mapping from names to objects. Most namespaces are currently implemented as Python dictionaries, but that’s normally not noticeable in any way (except for performance), and it may change in the future. Examples of namespaces are: the set of built-in names (containing functions such as [`abs()`](https://docs.python.org/3/library/functions.html#abs "abs"), and built-in exception names); the global names in a module; and the local names in a function invocation. In a sense the set of attributes of an object also form a namespace. The important thing to know about namespaces is that there is absolutely no relation between names in different namespaces; for instance, two different modules may both define a function `maximize` without confusion — users of the modules must prefix it with the module name.
> 定义：命名空间是一个将名字映射到对象的映射
> Python 中，目前大多数命名空间都以字典的形式实现
> 命名空间的例子有：内建的名称集合 (包含了内建函数的名字和内建异常的名字)；模块的全局命名空间；函数调用时的局部命名空间
> 一个对象的属性集合也构成了命名空间
> 注意不同的命名空间内的名字是没有联系的和冲突的

By the way, I use the word _attribute_ for any name following a dot — for example, in the expression `z.real`, `real` is an attribute of the object `z`. Strictly speaking, references to names in modules are attribute references: in the expression `modname.funcname`, `modname` is a module object and `funcname` is an attribute of it. In this case there happens to be a straightforward mapping between the module’s attributes and the global names defined in the module: they share the same namespace! [1](https://docs.python.org/3/tutorial/classes.html#id2)
> 严格地说，对于模块中名称的应用都是属性引用

Attributes may be read-only or writable. In the latter case, assignment to attributes is possible. Module attributes are writable: you can write `modname.the_answer = 42`. Writable attributes may also be deleted with the [`del`](https://docs.python.org/3/reference/simple_stmts.html#del) statement. For example, `del modname.the_answer` will remove the attribute `the_answer` from the object named by `modname`.
> 属性可以只读可以可写，如果可写，就可以为其赋值，也可以用 `del` 语句删除
> 例如 `del modname.the_answer` 就会移除名为 `modname` 的对象中的属性 `the_answer`

Namespaces are created at different moments and have different lifetimes. The namespace containing the built-in names is created when the Python interpreter starts up, and is never deleted. The global namespace for a module is created when the module definition is read in; normally, module namespaces also last until the interpreter quits. The statements executed by the top-level invocation of the interpreter, either read from a script file or interactively, are considered part of a module called [`__main__`](https://docs.python.org/3/library/__main__.html#module-__main__ "__main__: The environment where top-level code is run. Covers command-line interfaces, import-time behavior, and ``__name__ == '__main__'``."), so they have their own global namespace. (The built-in names actually also live in a module; this is called [`builtins`](https://docs.python.org/3/library/builtins.html#module-builtins "builtins: The module that provides the built-in namespace.").)
> 命名空间在不同的时机被创建，有不同的生命周期
> 包含了内建名称的命名空间在 Python 解释器启动时被创建，并且永远不会被删除
> 一个模块的全局命名空间在模块定义被读取时创建，一般模块的命名空间也会持续到解释器退出
> 解释器在 top-level 调用执行的语句，无论是读取脚本文件还是交互式执行，都被视作 `__main__` 模块的一部分，因此也拥有自己的全局命名空间，实际上，内建名字也属于 `__main__` 模块

The local namespace for a function is created when the function is called, and deleted when the function returns or raises an exception that is not handled within the function. (Actually, forgetting would be a better way to describe what actually happens.) Of course, recursive invocations each have their own local namespace.
> 函数的局部命名空间在函数被调用时被创建，在函数返回或者抛出一个未被自己处理的异常时被删除
> 递归调用的函数每个都有自己的命名空间

A _scope_ is a textual region of a Python program where a namespace is directly accessible. “Directly accessible” here means that an unqualified reference to a name attempts to find the name in the namespace.
> 作用域是 Python 程序的一个正文区域，其中的命名空间可以直接被访问
> 直接访问即对于一个名称的未被修饰的访问会直接尝试访问命名空间内的名字

Although scopes are determined statically, they are used dynamically. At any time during execution, there are 3 or 4 nested scopes whose namespaces are directly accessible:

- the innermost scope, which is searched first, contains the local names
- the scopes of any enclosing functions, which are searched starting with the nearest enclosing scope, contain non-local, but also non-global names
- the next-to-last scope contains the current module’s global names
- the outermost scope (searched last) is the namespace containing built-in names

> 作用域是被静态决定的，但被动态使用
> 在执行的任意时刻，会有3-4个其命名空间可以直接被访问的嵌套的作用域
> 最内部的作用域：优先被搜索，它包含了局部名字
> 任意 enclosing 函数的作用域：其次被搜索，包含了非局部非全局的名字
> next-to-last 的作用域，包含了当前模块的全局名字
> 最外部的名字：最后被搜索，该命名空间包含了内建名字

If a name is declared global, then all references and assignments go directly to the next-to-last scope containing the module’s global names. To rebind variables found outside of the innermost scope, the [`nonlocal`](https://docs.python.org/3/reference/simple_stmts.html#nonlocal) statement can be used; if not declared nonlocal, those variables are read-only (an attempt to write to such a variable will simply create a _new_ local variable in the innermost scope, leaving the identically named outer variable unchanged).
> 若名称被声明为全局，则所有的对该名字的引用和赋值都会直接到包含了模块的全局名称的 next-to-last 作用域
> 要将该名字重新绑定到 innermost 的作用域，可以使用 `nonlocal` 语句
> 如果没有使用 `nonlocal` 语句，则该全局名称对于局部来说实质上是只读的，任意尝试对该变量的写入仅仅会在 innermost 作用域创建一个新的局部变量，而外部的变量的值不会被影响

Usually, the local scope references the local names of the (textually) current function. Outside functions, the local scope references the same namespace as the global scope: the module’s namespace. Class definitions place yet another namespace in the local scope.
> 通常，局部作用域引用的是 (文本上) 当前函数的局部名字，在函数外，局部作用域引用的命名空间就和全局作用域引用的相同，即模块的命名空间
> 另外，类别定义会在局部添加另一个命名空间

It is important to realize that scopes are determined textually: the global scope of a function defined in a module is that module’s namespace, no matter from where or by what alias the function is called. On the other hand, the actual search for names is done dynamically, at run time — however, the language definition is evolving towards static name resolution, at “compile” time, so don’t rely on dynamic name resolution! (In fact, local variables are already determined statically.)
> 要意识到作用域是文本上定义的
> 因此，定义于模块内的函数的全局作用域指的就是该模块的命名空间，而不是该函数被调用时所处的模块的命名空间
> 换句话说，名称的搜索是在运行时动态进行的，但目前局部变量的搜索已经是静态进行的了

A special quirk of Python is that – if no [`global`](https://docs.python.org/3/reference/simple_stmts.html#global) or [`nonlocal`](https://docs.python.org/3/reference/simple_stmts.html#nonlocal) statement is in effect – assignments to names always go into the innermost scope. Assignments do not copy data — they just bind names to objects. The same is true for deletions: the statement `del x` removes the binding of `x` from the namespace referenced by the local scope. In fact, all operations that introduce new names use the local scope: in particular, [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statements and function definitions bind the module or function name in the local scope.
> 如果没有使用 `global` 或 `nonlocal` 语句，则对于名称的赋值总是指向 innermost 的作用域
> 注意赋值不会拷贝数据，仅仅是将名称绑定到对象，同样，`del x` 语句仅仅是将名称 `x` 的绑定从局部作用域引用的命名空间中移除
> 实际上，所有引入新的名字的操作都使用的是局部作用域，特别地，`import` 语句和函数定义都是将模块和函数的名称绑定到局部作用域

The [`global`](https://docs.python.org/3/reference/simple_stmts.html#global) statement can be used to indicate that particular variables live in the global scope and should be rebound there; the [`nonlocal`](https://docs.python.org/3/reference/simple_stmts.html#nonlocal) statement indicates that particular variables live in an enclosing scope and should be rebound there.
> `global` 语句用于表示特定的变量属于全局作用域，`nonlocal` 语句用于表明特定的变量属于包围的作用域

### 9.2.1. Scopes and Namespaces Example
This is an example demonstrating how to reference the different scopes and namespaces, and how [`global`](https://docs.python.org/3/reference/simple_stmts.html#global) and [`nonlocal`](https://docs.python.org/3/reference/simple_stmts.html#nonlocal) affect variable binding:

```python
def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)
```

The output of the example code is:

```
After local assignment: test spam
After nonlocal assignment: nonlocal spam
After global assignment: nonlocal spam
In global scope: global spam
```

Note how the _local_ assignment (which is default) didn’t change _scope_test_'s binding of _spam_. The [`nonlocal`](https://docs.python.org/3/reference/simple_stmts.html#nonlocal) assignment changed _scope_test_'s binding of _spam_, and the [`global`](https://docs.python.org/3/reference/simple_stmts.html#global) assignment changed the module-level binding.
> 默认赋值就是局部赋值，它不会改变函数 `scope_test` 对于名字 `spam` 的绑定 (即该名字的指向对象不变)，而 `nonlocal` 赋值改变了函数 `scpoe_test` 对于名字 `spam` 的绑定，`global` 赋值改变了模块级别对于名字 `spam` 的绑定

You can also see that there was no previous binding for _spam_ before the [`global`](https://docs.python.org/3/reference/simple_stmts.html#global) assignment.

## 9.3. A First Look at Classes
Classes introduce a little bit of new syntax, three new object types, and some new semantics.

### 9.3.1. Class Definition Syntax
The simplest form of class definition looks like this:

```
class ClassName:
    <statement-1>
    .
    .
    .
    <statement-N>
```

Class definitions, like function definitions ([`def`](https://docs.python.org/3/reference/compound_stmts.html#def) statements) must be executed before they have any effect. (You could conceivably place a class definition in a branch of an [`if`](https://docs.python.org/3/reference/compound_stmts.html#if) statement, or inside a function.)
> 类定义和函数定义一样，必须在使用之前定义，类的定义可以放在 `if` 分支内，或者在函数内

In practice, the statements inside a class definition will usually be function definitions, but other statements are allowed, and sometimes useful — we’ll come back to this later. The function definitions inside a class normally have a peculiar form of argument list, dictated by the calling conventions for methods — again, this is explained later.
> 实践中，类定义内的语句一般是函数定义语句，但也允许其他语句

When a class definition is entered, a new namespace is created, and used as the local scope — thus, all assignments to local variables go into this new namespace. In particular, function definitions bind the name of the new function here.
> 进入一个类定义时，一个新的命名空间会被创建，并作为局部作用域使用，因此类定义内所有对于局部变量的赋值都在这个新的命名空间内，以及函数定义也会将函数名绑定到该命名空间内的函数

When a class definition is left normally (via the end), a _class object_ is created. This is basically a wrapper around the contents of the namespace created by the class definition; we’ll learn more about class objects in the next section. The original local scope (the one in effect just before the class definition was entered) is reinstated, and the class object is bound here to the class name given in the class definition header (`ClassName` in the example).
> 当类定义正常退出后，一个类对象就会被创建，类对象基本就是类定义创建的命名空间内的内容的 warpper
> 此时，原来的局部作用域被恢复，并且该类对象会被绑定到类定义声明的类名称

### 9.3.2. Class Objects
Class objects support two kinds of operations: attribute references and instantiation.
> 类对象支持两类操作：属性引用、实例化

_Attribute references_ use the standard syntax used for all attribute references in Python: `obj.name`. Valid attribute names are all the names that were in the class’s namespace when the class object was created. So, if the class definition looked like this:
> 属性引用的语法就是 `obj.name` ，有效的属性名称是类对象被创建时，类的命名空间内所有的名字

```python
class MyClass:
    """A simple example class"""
    i = 12345

    def f(self):
        return 'hello world'
```

then `MyClass.i` and `MyClass.f` are valid attribute references, returning an integer and a function object, respectively. Class attributes can also be assigned to, so you can change the value of `MyClass.i` by assignment. `__doc__` is also a valid attribute, returning the docstring belonging to the class: `"A simple example class"`.
> 类的属性可以被赋值，例如我们可以通过赋值改变 `MyClass.i` 

Class _instantiation_ uses function notation. Just pretend that the class object is a parameterless function that returns a new instance of the class. For example (assuming the above class):

```python
x = MyClass()
```

creates a new _instance_ of the class and assigns this object to the local variable `x`.

The instantiation operation (“calling” a class object) creates an empty object. Many classes like to create objects with instances customized to a specific initial state. Therefore a class may define a special method named [`__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__ "object.__init__"), like this:

```python
def __init__(self):
    self.data = []
```

When a class defines an [`__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__ "object.__init__") method, class instantiation automatically invokes `__init__()` for the newly created class instance. So in this example, a new, initialized instance can be obtained by:
> 类实例化会自动调用 `__init__()` 

```
x = MyClass()
```

Of course, the [`__init__()`](https://docs.python.org/3/reference/datamodel.html#object.__init__ "object.__init__") method may have arguments for greater flexibility. In that case, arguments given to the class instantiation operator are passed on to `__init__()`. For example,

```
>>> class Complex:
...     def __init__(self, realpart, imagpart):
...         self.r = realpart
...         self.i = imagpart
...
>>> x = Complex(3.0, -4.5)
>>> x.r, x.i
(3.0, -4.5)
```

### 9.3.3. Instance Objects
Now what can we do with instance objects? The only operations understood by instance objects are attribute references. There are two kinds of valid attribute names: data attributes and methods.
> 实例对象唯一理解的操作就是属性引用
> 有两类有效的属性名称：数据属性、方法

_data attributes_ correspond to “instance variables” in Smalltalk, and to “data members” in C++. Data attributes need not be declared; like local variables, they spring into existence when they are first assigned to. For example, if `x` is the instance of `MyClass` created above, the following piece of code will print the value `16`, without leaving a trace:
> 数据属性对应于实例变量/数据成员
> 数据属性不需要被声明，它们和局部变量一样，当被赋值的时候就开始存在

```python
x.counter = 1
while x.counter < 10:
    x.counter = x.counter * 2
print(x.counter)
del x.counter
```

The other kind of instance attribute reference is a _method_. A method is a function that “belongs to” an object.

Valid method names of an instance object depend on its class. By definition, all attributes of a class that are function objects define corresponding methods of its instances. So in our example, `x.f` is a valid method reference, since `MyClass.f` is a function, but `x.i` is not, since `MyClass.i` is not. But `x.f` is not the same thing as `MyClass.f` — it is a _method object_, not a function object.
> 一个实例对象的有效方法名取决于它的类
> 定义上，一个类的所有属于函数对象的属性都定义了它的实例的方法
> 因此 `x.f` 是一个有效的方法引用，因为 `MyClass.f` 是一个函数
> 但注意 `MyClass.f` 和 `x.f` 不同，`MyClass.f` 是一个函数对象，而不是方法对象

### 9.3.4. Method Objects
Usually, a method is called right after it is bound:

```python
x.f()
```

In the `MyClass` example, this will return the string `'hello world'`. However, it is not necessary to call a method right away: `x.f` is a method object, and can be stored away and called at a later time. For example:
> 可以给方法对象添加别名，之后用别名调用

```python
xf = x.f
while True:
    print(xf())
```

will continue to print `hello world` until the end of time.

What exactly happens when a method is called? You may have noticed that `x.f()` was called without an argument above, even though the function definition for `f()` specified an argument. What happened to the argument? Surely Python raises an exception when a function that requires an argument is called without any — even if the argument isn’t actually used…

Actually, you may have guessed the answer: the special thing about methods is that the instance object is passed as the first argument of the function. In our example, the call `x.f()` is exactly equivalent to `MyClass.f(x)`. In general, calling a method with a list of _n_ arguments is equivalent to calling the corresponding function with an argument list that is created by inserting the method’s instance object before the first argument.
> 方法的第一个参数就是实例对象
> 方法调用 `x.f()` 实际上等价于 `MyClass.f(x)`
> 调用一个方法实际上就等价于调用相应的函数，并且将调用的实例对象作为第一个参数

In general, methods work as follows. When a non-data attribute of an instance is referenced, the instance’s class is searched. If the name denotes a valid class attribute that is a function object, references to both the instance object and the function object are packed into a method object. When the method object is called with an argument list, a new argument list is constructed from the instance object and the argument list, and the function object is called with this new argument list.
> 方法的工作流程：当一个实例的非数据属性被引用，实例的类会被搜索，如果该名称是一个有效的类别属性 (函数对象)，则对于实例对象的引用和对于该函数对象的引用会被打包到一个方法对象 (`xf = x.f`)
> 当方法对象被调用时，Python 会根据调用参数列表和实例对象构造一个新的参数列表，然后使用新的参数列表调用该函数对象

### 9.3.5. Class and Instance Variables
Generally speaking, instance variables are for data unique to each instance and class variables are for attributes and methods shared by all instances of the class:
> 总的来说，实例变量是每个实例独立的数据，类别变量是由所有实例共享的属性和方法

```
class Dog:

    kind = 'canine'         # class variable shared by all instances

    def __init__(self, name):
        self.name = name    # instance variable unique to each instance

>>> d = Dog('Fido')
>>> e = Dog('Buddy')
>>> d.kind                  # shared by all dogs
'canine'
>>> e.kind                  # shared by all dogs
'canine'
>>> d.name                  # unique to d
'Fido'
>>> e.name                  # unique to e
'Buddy'
```

As discussed in [A Word About Names and Objects](https://docs.python.org/3/tutorial/classes.html#tut-object), shared data can have possibly surprising effects with involving [mutable](https://docs.python.org/3/glossary.html#term-mutable) objects such as lists and dictionaries. For example, the _tricks_ list in the following code should not be used as a class variable because just a single list would be shared by all _Dog_ instances:
> 共享的类别变量是可变类型时，所有的实例都可以修改它

```
class Dog:

    tricks = []             # mistaken use of a class variable

    def __init__(self, name):
        self.name = name

    def add_trick(self, trick):
        self.tricks.append(trick)

>>> d = Dog('Fido')
>>> e = Dog('Buddy')
>>> d.add_trick('roll over')
>>> e.add_trick('play dead')
>>> d.tricks                # unexpectedly shared by all dogs
['roll over', 'play dead']
```

Correct design of the class should use an instance variable instead:

```
class Dog:

    def __init__(self, name):
        self.name = name
        self.tricks = []    # creates a new empty list for each dog

    def add_trick(self, trick):
        self.tricks.append(trick)

>>> d = Dog('Fido')
>>> e = Dog('Buddy')
>>> d.add_trick('roll over')
>>> e.add_trick('play dead')
>>> d.tricks
['roll over']
>>> e.tricks
['play dead']
```

## 9.4. Random Remarks
If the same attribute name occurs in both an instance and in a class, then attribute lookup prioritizes the instance:
> 如果实例和类别具有相同名字的属性，则实例的属性被优先引用

```
>>> class Warehouse:
...    purpose = 'storage'
...    region = 'west'
...
>>> w1 = Warehouse()
>>> print(w1.purpose, w1.region)
storage west
>>> w2 = Warehouse()
>>> w2.region = 'east'
>>> print(w2.purpose, w2.region)
storage east
```

Data attributes may be referenced by methods as well as by ordinary users (“clients”) of an object. In other words, classes are not usable to implement pure abstract data types. In fact, nothing in Python makes it possible to enforce data hiding — it is all based upon convention. (On the other hand, the Python implementation, written in C, can completely hide implementation details and control access to an object if necessary; this can be used by extensions to Python written in C.)
> 数据属性可以被方法引用，也可以被对象的使用者引用，因此类不能用于实现纯抽象的数据类型
> Python 中不能实现数据隐藏，但这只是基于惯例，实际上可以通过 C 实现的拓展来隐藏实现细节以及控制对于一个对象的访问

Clients should use data attributes with care — clients may mess up invariants maintained by the methods by stamping on their data attributes. Note that clients may add data attributes of their own to an instance object without affecting the validity of the methods, as long as name conflicts are avoided — again, a naming convention can save a lot of headaches here.
> 因此，即便类的方法维护了一个不可变的数据属性，用户也可以破坏它

There is no shorthand for referencing data attributes (or other methods!) from within methods. I find that this actually increases the readability of methods: there is no chance of confusing local variables and instance variables when glancing through a method.
> 在方法内引用数据属性需要使用 `self.` 前缀，Python 中没有对于数据属性引用的 shorthand

Often, the first argument of a method is called `self`. This is nothing more than a convention: the name `self` has absolutely no special meaning to Python. Note, however, that by not following the convention your code may be less readable to other Python programmers, and it is also conceivable that a _class browser_ program might be written that relies upon such a convention.
> 按照惯例，方法的第一个参数名称为 `self` ，该名称并没有特殊意义

Any function object that is a class attribute defines a method for instances of that class. It is not necessary that the function definition is textually enclosed in the class definition: assigning a function object to a local variable in the class is also ok. For example:
> 任意是类别属性的函数对象都为该类别的实例定义了方法
> 因此，函数定义并不一定要在文本上由类别的定义包围，我们可以通过将一个函数对象赋值给类别中的局部变量实现为实例定义方法

```python
# Function defined outside the class
def f1(self, x, y):
    return min(x, x+y)

class C:
    f = f1

    def g(self):
        return 'hello world'

    h = g
```

Now `f`, `g` and `h` are all attributes of class `C` that refer to function objects, and consequently they are all methods of instances of `C` — `h` being exactly equivalent to `g`. Note that this practice usually only serves to confuse the reader of a program.
> 此时 `f, g, h` 都作为类别 `C` 的属性，指向函数对象，并且都是 `C` 的实例的方法

Methods may call other methods by using method attributes of the `self` argument:
> 可以通过 `self` 参数在方法使用方法属性调用其他方法

```python
class Bag:
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

    def addtwice(self, x):
        self.add(x)
        self.add(x)
```

Methods may reference global names in the same way as ordinary functions. The global scope associated with a method is the module containing its definition. (A class is never used as a global scope.) While one rarely encounters a good reason for using global data in a method, there are many legitimate uses of the global scope: for one thing, functions and modules imported into the global scope can be used by methods, as well as functions and classes defined in it. Usually, the class containing the method is itself defined in this global scope, and in the next section we’ll find some good reasons why a method would want to reference its own class.
> 和普通函数一样，方法可以引用全局名字，和方法相关的全局作用域就是包含了它的定义的模块 (注意类一定不会被作为全局作用域)
> 例如，在全局作用域导入的方法和模块 (以及其中定义的函数和类) 可以由我们的方法调用
> 一般来说，类本身就会定义在全局作用域内

Each value is an object, and therefore has a _class_ (also called its _type_). It is stored as `object.__class__`.
> 事实上在 Python 中每个值都是对象，因此都有对应的类 (也称为它的类型)，可以通过 `object.__class__` 引用

## 9.5. Inheritance
Of course, a language feature would not be worthy of the name “class” without supporting inheritance. The syntax for a derived class definition looks like this:

```python
class DerivedClassName(BaseClassName):
    <statement-1>
    .
    .
    .
    <statement-N>
```

The name `BaseClassName` must be defined in a namespace accessible from the scope containing the derived class definition. In place of a base class name, other arbitrary expressions are also allowed. This can be useful, for example, when the base class is defined in another module:
> 继承时，名称 `BaseClassName` 必须在衍生类可以访问的作用域中的命名空间中有定义
> 允许用表达式访问基类名称

```python
class DerivedClassName(modname.BaseClassName):
```

Execution of a derived class definition proceeds the same as for a base class. When the class object is constructed, the base class is remembered. This is used for resolving attribute references: if a requested attribute is not found in the class, the search proceeds to look in the base class. This rule is applied recursively if the base class itself is derived from some other class.
> 当衍生类的定义被执行，衍生类对象会被构建，同时基类会被记住，用于解析属性引用: 若一个请求的属性没有在类中找到，则会继续到基类中寻找
> 如果基类也有基类，则递归进行搜寻

There’s nothing special about instantiation of derived classes: `DerivedClassName()` creates a new instance of the class. Method references are resolved as follows: the corresponding class attribute is searched, descending down the chain of base classes if necessary, and the method reference is valid if this yields a function object.
> 方法引用的解析如下：首先搜索对应类的属性，如果必要，沿着继承链往下搜索基类的属性，方法能引用到一个函数对象时，方法的引用就是有效的

Derived classes may override methods of their base classes. Because methods have no special privileges when calling other methods of the same object, a method of a base class that calls another method defined in the same base class may end up calling a method of a derived class that overrides it. (For C++ programmers: all methods in Python are effectively `virtual`.)
> 衍生类可以重载它们基类的方法
> 基类中的一个方法调用调用类内的其他方法时，因为方法之间并没有定义特殊的优先级，因此可能实际调用的时衍生类中重载了该方法的方法 (Python 中所有的方法都是虚函数)

An overriding method in a derived class may in fact want to extend rather than simply replace the base class method of the same name. There is a simple way to call the base class method directly: just call `BaseClassName.methodname(self, arguments)`. This is occasionally useful to clients as well. (Note that this only works if the base class is accessible as `BaseClassName` in the global scope.)
> 注意衍生类中的重载方法实际上应该延伸而不是覆盖基类中相同名字的方法
> 调用基类方法的一个直接方式是 `BaseClassName.methodname(self, arguments)`

Python has two built-in functions that work with inheritance:

- Use [`isinstance()`](https://docs.python.org/3/library/functions.html#isinstance "isinstance") to check an instance’s type: `isinstance(obj, int)` will be `True` only if `obj.__class__` is [`int`](https://docs.python.org/3/library/functions.html#int "int") or some class derived from [`int`](https://docs.python.org/3/library/functions.html#int "int").
- Use [`issubclass()`](https://docs.python.org/3/library/functions.html#issubclass "issubclass") to check class inheritance: `issubclass(bool, int)` is `True` since [`bool`](https://docs.python.org/3/library/functions.html#bool "bool") is a subclass of [`int`](https://docs.python.org/3/library/functions.html#int "int"). However, `issubclass(float, int)` is `False` since [`float`](https://docs.python.org/3/library/functions.html#float "float") is not a subclass of [`int`](https://docs.python.org/3/library/functions.html#int "int")
> Python 和继承相关的内建函数
> `isinstance()` 用于检查对象是否是特定类 (或者其衍生类) 的实例
> `issubclass()` 用于检查类是否是特定类的衍生类

### 9.5.1. Multiple Inheritance
Python supports a form of multiple inheritance as well. A class definition with multiple base classes looks like this:

```python
class DerivedClassName(Base1, Base2, Base3):
    <statement-1>
    .
    .
    .
    <statement-N>
```

For most purposes, in the simplest cases, you can think of the search for attributes inherited from a parent class as depth-first, left-to-right, not searching twice in the same class where there is an overlap in the hierarchy. Thus, if an attribute is not found in `DerivedClassName`, it is searched for in `Base1`, then (recursively) in the base classes of `Base1`, and if it was not found there, it was searched for in `Base2`, and so on.
> 对于多继承，我们可以对于基类中属性的搜索行为理解为深度优先，从左向右，如果在层次结构中存在重叠，则不会在同一个类中搜索两次
> 因此，如果在 `DerivedClassName` 中没有搜索到某一属性，则 Python 会首先搜索 `Base1` 然后递归式向上搜索，如果没有找到，则从 `Base2` 开始搜索，以此类推

In fact, it is slightly more complex than that; the method resolution order changes dynamically to support cooperative calls to [`super()`](https://docs.python.org/3/library/functions.html#super "super"). This approach is known in some other multiple-inheritance languages as call-next-method and is more powerful than the super call found in single-inheritance languages.
> 实际中的情况会略微复杂，方法解析的顺序是会动态变化的，以支持对于 `super()` 的协同调用
> 例如以下代码：
```python
class Grandparent:
    def say_hello(self): 
        print("Hello from Grandparent") 

class ParentA(Grandparent): 
    def say_hello(self):
        print("Hello from ParentA")
        super().say_hello() # 调用超类Grandparent的方法

class ParentB(Grandparent):
    def say_hello(self):
        print("Hello from ParentB")
        super().say_hello() # 调用超类Grandparent的方法

class Child(ParentA, ParentB):
    def say_hello(self):
        print("Hello from Child")
        super().say_hello() # 调用超类ParentA或ParentB的方法

child = Child()
child.say_hello()
```
> 执行结果是
```
Hello from Child
Hello from ParentA
Hello from ParentB
Hello from Grandparent
```

Dynamic ordering is necessary because all cases of multiple inheritance exhibit one or more diamond relationships (where at least one of the parent classes can be accessed through multiple paths from the bottommost class). For example, all classes inherit from [`object`](https://docs.python.org/3/library/functions.html#object "object"), so any case of multiple inheritance provides more than one path to reach [`object`](https://docs.python.org/3/library/functions.html#object "object"). To keep the base classes from being accessed more than once, the dynamic algorithm linearizes the search order in a way that preserves the left-to-right ordering specified in each class, that calls each parent only once, and that is monotonic (meaning that a class can be subclassed without affecting the precedence order of its parents). Taken together, these properties make it possible to design reliable and extensible classes with multiple inheritance. For more detail, see [The Python 2.3 Method Resolution Order](https://docs.python.org/3/howto/mro.html#python-2-3-mro).
> 动态排序是必要的，因为所有的多继承情况都会有一个或多个的钻石关系 (一个基类被底层的衍生类通过多条路径访问)
> 例如，所有的类都继承自 `object` ，因此任意情况的多继承都会为衍生类提供访问 `object` 的多条路径
> 而动态算法线性化了搜索顺序，保持了在每个类中指定的从左到右的顺序，对于每个基类的调用只会有一次，并且是单调的

## 9.6. Private Variables
“Private” instance variables that cannot be accessed except from inside an object don’t exist in Python. However, there is a convention that is followed by most Python code: a name prefixed with an underscore (e.g. `_spam`) should be treated as a non-public part of the API (whether it is a function, a method or a data member). It should be considered an implementation detail and subject to change without notice.
> Python 中并不存在只允许在对象内部访问的实例变量
> 但大多数 Python 代码遵循一个约定：以下划线为前缀的名字应该被视作 API 的非公用的部分

Since there is a valid use-case for class-private members (namely to avoid name clashes of names with names defined by subclasses), there is limited support for such a mechanism, called _name mangling_. Any identifier of the form `__spam` (at least two leading underscores, at most one trailing underscore) is textually replaced with `_classname__spam`, where `classname` is the current class name with leading underscore(s) stripped. This mangling is done without regard to the syntactic position of the identifier, as long as it occurs within the definition of a class.
> Python 提供了名称改编机制来支持类私有成员的使用，帮助避免基类和衍生类之间的名称发生冲突
> Python 中，任意形式为 `__spam` (至少两个先导下划线，最多一个后缀下划线) 的名字会在文本上替换为 `_classname__spam` ，其中 `classname` 是当前类的名字 (它的先导下划线会被去除)
> 名称改变和标识符的语义位置无关，只要标识符出现在类的定义中，就会被改编 

See also
The [private name mangling specifications](https://docs.python.org/3/reference/expressions.html#private-name-mangling) for details and special cases.

Name mangling is helpful for letting subclasses override methods without breaking intraclass method calls. For example:
> 名称改编有助于让子类在不破坏类内方法调用的情况下重写方法

```python
class Mapping:
    def __init__(self, iterable):
        self.items_list = []
        self.__update(iterable)

    def update(self, iterable):
        for item in iterable:
            self.items_list.append(item)

    __update = update   # private copy of original update() method

class MappingSubclass(Mapping):

    def update(self, keys, values):
        # provides new signature for update()
        # but does not break __init__()
        for item in zip(keys, values):
            self.items_list.append(item)
```

The above example would work even if `MappingSubclass` were to introduce a `__update` identifier since it is replaced with `_Mapping__update` in the `Mapping` class and `_MappingSubclass__update` in the `MappingSubclass` class respectively.
> 该例中，基类中的 `__init__` 对于 `__update` 的调用就会指向基类的 `update` 函数，而不会调用衍生类的 `update` ，即便在衍生类中也定义了 `__update` 也是同样的，因为名称会被替换

Note that the mangling rules are designed mostly to avoid accidents; it still is possible to access or modify a variable that is considered private. This can even be useful in special circumstances, such as in the debugger.

Notice that code passed to `exec()` or `eval()` does not consider the classname of the invoking class to be the current class; this is similar to the effect of the `global` statement, the effect of which is likewise restricted to code that is byte-compiled together. The same restriction applies to `getattr()`, `setattr()` and `delattr()`, as well as when referencing `__dict__` directly.
> 注意传递给 `exec()/eval()` 执行的代码不会考虑到调用类的类名 (和 `global` 语句的效果类似，且效果也限制于一起字节编译的代码)，以对名称进行改编，这些限制对于 ` getattr()/setattr()/delattr() ` 也成立，以及对于访问 ` __dict__ ` 也成立，因此它们访问 ` __spam ` 时会直接访问 ` __spam ` ，而不是访问改编的名字 ` _classname__spam ` ，故常常会抛出错误

## 9.7. Odds and Ends
Sometimes it is useful to have a data type similar to the Pascal “record” or C “struct”, bundling together a few named data items. The idiomatic approach is to use [`dataclasses`](https://docs.python.org/3/library/dataclasses.html#module-dataclasses "dataclasses: Generate special methods on user-defined classes.") for this purpose:
> 有时要创建类似于 C 的结构体的数据结构时 (将几个数据项的名字绑定到一起)，一个惯用的方法是使用 `dataclasses` 

```
from dataclasses import dataclass

@dataclass
class Employee:
    name: str
    dept: str
    salary: int

>>>

>>> john = Employee('john', 'computer lab', 1000)
>>> john.dept
'computer lab'
>>> john.salary
1000
```

A piece of Python code that expects a particular abstract data type can often be passed a class that emulates the methods of that data type instead. For instance, if you have a function that formats some data from a file object, you can define a class with methods [`read()`](https://docs.python.org/3/library/io.html#io.TextIOBase.read "io.TextIOBase.read") and [`readline()`](https://docs.python.org/3/library/io.html#io.TextIOBase.readline "io.TextIOBase.readline") that get the data from a string buffer instead, and pass it as an argument.
> 上述的代码实际上是使用类来模仿结构体的行为，在 Python 中，如果一段 Python 代码期待特定的抽象数据类型，我们可以实际上为它传递一个模仿该数据类型行为的类
> Python 的哲学支持 duck typing，也就是说 "if it looks like a duck and quacks like a duck, it's a duck"，即如果一个对象和另一个对象支持相同的接口，则二者可以互相交替地使用

[Instance method objects](https://docs.python.org/3/reference/datamodel.html#instance-methods) have attributes, too: [`m.__self__`](https://docs.python.org/3/reference/datamodel.html#method.__self__ "method.__self__") is the instance object with the method `m()`, and [`m.__func__`](https://docs.python.org/3/reference/datamodel.html#method.__func__ "method.__func__") is the [function object](https://docs.python.org/3/reference/datamodel.html#user-defined-funcs) corresponding to the method.
> 实例方法也是对象，实例方法对象也同样有属性，例如 `m.__self__` 是具有方法 `m()` 的实例对象，`m.__func__` 是该方法对应的函数对象

## 9.8. Iterators
By now you have probably noticed that most container objects can be looped over using a [`for`](https://docs.python.org/3/reference/compound_stmts.html#for) statement:
> Python 为容器对象提供了统一的迭代语法：使用 `for` 语句

```python
for element in [1, 2, 3]:
    print(element)
for element in (1, 2, 3):
    print(element)
for key in {'one':1, 'two':2}:
    print(key)
for char in "123":
    print(char)
for line in open("myfile.txt"):
    print(line, end='')
```

This style of access is clear, concise, and convenient. The use of iterators pervades and unifies Python. Behind the scenes, the [`for`](https://docs.python.org/3/reference/compound_stmts.html#for) statement calls [`iter()`](https://docs.python.org/3/library/functions.html#iter "iter") on the container object. The function returns an iterator object that defines the method [`__next__()`](https://docs.python.org/3/library/stdtypes.html#iterator.__next__ "iterator.__next__") which accesses elements in the container one at a time. When there are no more elements, [`__next__()`](https://docs.python.org/3/library/stdtypes.html#iterator.__next__ "iterator.__next__") raises a [`StopIteration`](https://docs.python.org/3/library/exceptions.html#StopIteration "StopIteration") exception which tells the `for` loop to terminate. You can call the [`__next__()`](https://docs.python.org/3/library/stdtypes.html#iterator.__next__ "iterator.__next__") method using the [`next()`](https://docs.python.org/3/library/functions.html#next "next") built-in function; this example shows how it all works:
> `for` 语句实际上对容器对象调用了 `iter()` ，该函数返回一个迭代器对象，该对象定义了方法 `__next__()` ，该方法每次访问容器中的一个元素
> 当没有更多元素时，会抛出 `StopIteration` 异常，告诉 `for` 循环终止
> 我们可以使用 `next()` 内建函数来调用 `__next__()` 方法

```
>>> s = 'abc'
>>> it = iter(s)
>>> it
<str_iterator object at 0x10c90e650>
>>> next(it)
'a'
>>> next(it)
'b'
>>> next(it)
'c'
>>> next(it)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    next(it)
StopIteration
```

Having seen the mechanics behind the iterator protocol, it is easy to add iterator behavior to your classes. Define an [`__iter__()`](https://docs.python.org/3/library/stdtypes.html#container.__iter__ "container.__iter__") method which returns an object with a [`__next__()`](https://docs.python.org/3/library/stdtypes.html#iterator.__next__ "iterator.__next__") method. If the class defines `__next__()`, then `__iter__()` can just return `self`:
> 我们可以为自己定义的类添加迭代器行为：
> 定义 `__iter__()` 方法，`__iter__()` 方法返回一个迭代器对象；定义 `__next__()` 方法，`__next__()` 方法接受一个迭代器对象，迭代式返回元素
> 如果类中定义了 `__next__()` 方法，我们可以直接定义 `__iter__()` 方法返回 `self` ，即返回对象本身作为迭代器对象

```
class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

>>>

>>> rev = Reverse('spam')
>>> iter(rev)
<__main__.Reverse object at 0x00A1DB50>
>>> for char in rev:
...     print(char)
...
m
a
p
s
```

## 9.9. Generators
[Generators](https://docs.python.org/3/glossary.html#term-generator) are a simple and powerful tool for creating iterators. They are written like regular functions but use the [`yield`](https://docs.python.org/3/reference/simple_stmts.html#yield) statement whenever they want to return data. Each time [`next()`](https://docs.python.org/3/library/functions.html#next "next") is called on it, the generator resumes where it left off (it remembers all the data values and which statement was last executed). An example shows that generators can be trivially easy to create:
> 生成器是用于创建迭代器的工具
> 生成器的写法和常规的函数类似，唯一的差别是在需要返回数据时使用 `yield` 语句
> 每一次调用 `next()` 时，生成器会从上次停止的地方继续 (生成器会记忆所有的数据值和上一次执行的语句)

```
def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]

>>>

>>> for char in reverse('golf'):
...     print(char)
...
f
l
o
g
```

Anything that can be done with generators can also be done with class-based iterators as described in the previous section. What makes generators so compact is that the [`__iter__()`](https://docs.python.org/3/library/stdtypes.html#iterator.__iter__ "iterator.__iter__") and [`__next__()`](https://docs.python.org/3/reference/expressions.html#generator.__next__ "generator.__next__") methods are created automatically.
> 当然，生成器可以做的也可以用之前描述的基于类的迭代器实现
> 生成器利用 `yield` 语句自动定义了 `__iter__()` 和 `__next__()` 方法

Another key feature is that the local variables and execution state are automatically saved between calls. This made the function easier to write and much more clear than an approach using instance variables like `self.index` and `self.data`.
> 生成器的另一个关键特性是在它的调用之间，局部变量和执行状态会被自动存储
> 因此不需要像之前的例子一样，写一个类，分别实现 `__iter__()` 和 `__next__()` 以及数据属性 `self.index/data` 然后在 `__next__()` 中显式地对这些变量进行维护

In addition to automatic method creation and saving program state, when generators terminate, they automatically raise [`StopIteration`](https://docs.python.org/3/library/exceptions.html#StopIteration "StopIteration"). In combination, these features make it easy to create iterators with no more effort than writing a regular function.
> 除了自动创建 `__next__()` 和 `__iter__()` 方法、自动保存程序状态以外，当生成器结束，它们会自动抛出 `StopIteration` 异常
> 因此生成器非常便于创建迭代器

## 9.10. Generator Expressions
Some simple generators can be coded succinctly as expressions using a syntax similar to list comprehensions but with parentheses instead of square brackets. These expressions are designed for situations where the generator is used right away by an enclosing function. Generator expressions are more compact but less versatile than full generator definitions and tend to be more memory friendly than equivalent list comprehensions.
> 生成器表达式使用类似列表推导式的语法（使用圆括号而不是方括号）调用简单的生成器
> 生成器表达式更紧凑，但功能相较于完整生成器定义略有限制
> 生成器表达式相较于等价的列表推导式占用内存更少

Examples:

```
>>> sum(i*i for i in range(10))                 # sum of squares
285

>>> xvec = [10, 20, 30]
>>> yvec = [7, 5, 3]
>>> sum(x*y for x,y in zip(xvec, yvec))         # dot product
260

>>> unique_words = set(word for line in page  for word in line.split())

>>> valedictorian = max((student.gpa, student.name) for student in graduates)

>>> data = 'golf'
>>> list(data[i] for i in range(len(data)-1, -1, -1))
['f', 'l', 'o', 'g']
```

Footnotes
[1](https://docs.python.org/3/tutorial/classes.html#id1) Except for one thing. Module objects have a secret read-only attribute called [`__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__ "object.__dict__") which returns the dictionary used to implement the module’s namespace; the name [`__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__ "object.__dict__") is an attribute but not a global name. Obviously, using this violates the abstraction of namespace implementation, and should be restricted to things like post-mortem debuggers.
> 模块对象有一个只读的属性 `__dict__` ，该属性返回用于实现该模块命名空间的字典
> 名字 `__dict__` 是模块的一个属性，但它不是全局名字，也就是说在模块内无法直接访问名字 `__dict__` ，这样的实现是为了保持命名空间的抽象性

# 10. Brief Tour of the Standard Library
## 10.1. Operating System Interface
The [`os`](https://docs.python.org/3/library/os.html#module-os "os: Miscellaneous operating system interfaces.") module provides dozens of functions for interacting with the operating system:
> `os` 模块提供和 OS 交互的函数

```
>>> import os
>>> os.getcwd()      # Return the current working directory
'C:\\Python312'
>>> os.chdir('/server/accesslogs')   # Change current working directory
>>> os.system('mkdir today')   # Run the command mkdir in the system shell
0
```

Be sure to use the `import os` style instead of `from os import *`. This will keep [`os.open()`](https://docs.python.org/3/library/os.html#os.open "os.open") from shadowing the built-in [`open()`](https://docs.python.org/3/library/functions.html#open "open") function which operates much differently.

The built-in [`dir()`](https://docs.python.org/3/library/functions.html#dir "dir") and [`help()`](https://docs.python.org/3/library/functions.html#help "help") functions are useful as interactive aids for working with large modules like [`os`](https://docs.python.org/3/library/os.html#module-os "os: Miscellaneous operating system interfaces."):
> 对于大型的模块例如 `os` ，可以采用 `dir()/help()` 内建函数了解其信息

```
>>> import os
>>> dir(os)
<returns a list of all module functions>
>>> help(os)
<returns an extensive manual page created from the module's docstrings>
```

For daily file and directory management tasks, the [`shutil`](https://docs.python.org/3/library/shutil.html#module-shutil "shutil: High-level file operations, including copying.") module provides a higher level interface that is easier to use:
> 对于文件和目录管理，`shutil` 模块提供了更高层次的接口

```
>>> import shutil
>>> shutil.copyfile('data.db', 'archive.db')
'archive.db'
>>> shutil.move('/build/executables', 'installdir')
'installdir'
```

## 10.2. File Wildcards
The [`glob`](https://docs.python.org/3/library/glob.html#module-glob "glob: Unix shell style pathname pattern expansion.") module provides a function for making file lists from directory wildcard searches:
> `glob` 模块提供了在目录中根据通配符收集文件列表的函数

```
>>> import glob
>>> glob.glob('*.py')
['primes.py', 'random.py', 'quote.py']
```

## 10.3. Command Line Arguments
Common utility scripts often need to process command line arguments. These arguments are stored in the [`sys`](https://docs.python.org/3/library/sys.html#module-sys "sys: Access system-specific parameters and functions.") module’s _argv_ attribute as a list. For instance, let’s take the following `demo.py` file:
> 命令行参数以列表形式存储于 `sys` 模块的 argv 属性

```
# File demo.py
import sys
print(sys.argv)
```

Here is the output from running `python demo.py one two three` at the command line:

```
['demo.py', 'one', 'two', 'three']
```

The [`argparse`](https://docs.python.org/3/library/argparse.html#module-argparse "argparse: Command-line option and argument parsing library.") module provides a more sophisticated mechanism to process command line arguments. The following script extracts one or more filenames and an optional number of lines to be displayed:
> `argparse` 模块提供了处理命令行参数的更复杂的机制

```python
import argparse

parser = argparse.ArgumentParser(
    prog='top',
    description='Show top lines from each file')
parser.add_argument('filenames', nargs='+')
parser.add_argument('-l', '--lines', type=int, default=10)
args = parser.parse_args()
print(args)
```

When run at the command line with `python top.py --lines=5 alpha.txt beta.txt`, the script sets `args.lines` to `5` and `args.filenames` to `['alpha.txt', 'beta.txt']`.

## 10.4. Error Output Redirection and Program Termination
The [`sys`](https://docs.python.org/3/library/sys.html#module-sys "sys: Access system-specific parameters and functions.") module also has attributes for _stdin_, _stdout_, and _stderr_. The latter is useful for emitting warnings and error messages to make them visible even when _stdout_ has been redirected:
> `sys` 模块除了 `argv` 外，还有属性 `stdin/stdout/stderr` 
> stderr 可以用于在 stdout 被重定向的情况下也发出错误和警告信息

```
>>> sys.stderr.write('Warning, log file not found starting a new one\n')
Warning, log file not found starting a new one
```

The most direct way to terminate a script is to use `sys.exit()`.
> 结束脚本执行可以用 `sys.exit()`

## 10.5. String Pattern Matching
The [`re`](https://docs.python.org/3/library/re.html#module-re "re: Regular expression operations.") module provides regular expression tools for advanced string processing. For complex matching and manipulation, regular expressions offer succinct, optimized solutions:
> `re` 模块提供了用于字符串处理的正则表达式工具

```
>>> import re
>>> re.findall(r'\bf[a-z]*', 'which foot or hand fell fastest')
['foot', 'fell', 'fastest']
>>> re.sub(r'(\b[a-z]+) \1', r'\1', 'cat in the the hat')
'cat in the hat'
```

When only simple capabilities are needed, string methods are preferred because they are easier to read and debug:
> 字符串对象的方法也可以做简单的替换

```
>>> 'tea for too'.replace('too', 'two')
'tea for two'
```

## 10.6. Mathematics
The [`math`](https://docs.python.org/3/library/math.html#module-math "math: Mathematical functions (sin() etc.). module gives access to the underlying C library functions for floating-point math:
> `math` 模块提供了对于处理浮点数运算的底层 C 库函数的访问接口

```
>>> import math
>>> math.cos(math.pi / 4)
0.70710678118654757
>>> math.log(1024, 2)
10.0
```

The [`random`](https://docs.python.org/3/library/random.html#module-random "random: Generate pseudo-random numbers with various common distributions.") module provides tools for making random selections:
> `random` 模块提供进行随机抽样的函数

```
>>> import random
>>> random.choice(['apple', 'pear', 'banana'])
'apple'
>>> random.sample(range(100), 10)   # sampling without replacement
[30, 83, 16, 4, 8, 81, 41, 50, 18, 33]
>>> random.random()    # random float
0.17970987693706186
>>> random.randrange(6)    # random integer chosen from range(6)
4
```

The [`statistics`](https://docs.python.org/3/library/statistics.html#module-statistics "statistics: Mathematical statistics functions") module calculates basic statistical properties (the mean, median, variance, etc.) of numeric data:
> `statistics` 模块提供计算基本统计量的函数

```
>>> import statistics
>>> data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
>>> statistics.mean(data)
1.6071428571428572
>>> statistics.median(data)
1.25
>>> statistics.variance(data)
1.3720238095238095
```

The SciPy project < [https://scipy.org](https://scipy.org/) > has many other modules for numerical computations.
> 更多的数学计算模块见 SciPy 项目

## 10.7. Internet Access
There are a number of modules for accessing the internet and processing internet protocols. Two of the simplest are [`urllib.request`](https://docs.python.org/3/library/urllib.request.html#module-urllib.request "urllib.request: Extensible library for opening URLs.") for retrieving data from URLs and [`smtplib`](https://docs.python.org/3/library/smtplib.html#module-smtplib "smtplib: SMTP protocol client (requires sockets).") for sending mail:
> 关于处理网络协议和访问的最简单的两个模块是 `urllib.request` 和 `smtplib` ，其中 `urllib.request` 用于从 URL 中获取数据，`smptlib` 用于发送邮件

```
>>> from urllib.request import urlopen
>>> with urlopen('http://worldtimeapi.org/api/timezone/etc/UTC.txt') as response:
...     for line in response:
...         line = line.decode()             # Convert bytes to a str
...         if line.startswith('datetime'):
...             print(line.rstrip())         # Remove trailing newline
...
datetime: 2022-01-01T01:36:47.689215+00:00
```

```
>>> import smtplib
>>> server = smtplib.SMTP('localhost')
>>> server.sendmail('soothsayer@example.org', 'jcaesar@example.org',
... """To: jcaesar@example.org
... From: soothsayer@example.org
...
... Beware the Ides of March.
... """)
>>> server.quit()
```

(Note that the second example needs a mailserver running on localhost.)

## 10.8. Dates and Times
The [`datetime`](https://docs.python.org/3/library/datetime.html#module-datetime "datetime: Basic date and time types.") module supplies classes for manipulating dates and times in both simple and complex ways. While date and time arithmetic is supported, the focus of the implementation is on efficient member extraction for output formatting and manipulation. The module also supports objects that are timezone aware.
> `datetime` 模块提供处理时间和日期的函数，主要聚焦于提取日期和时间成员用于输出格式化，该模块提供针对时区的函数

```
>>> # dates are easily constructed and formatted
>>> from datetime import date
>>> now = date.today()
>>> now
datetime.date(2003, 12, 2)
>>> now.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B.")
'12-02-03. 02 Dec 2003 is a Tuesday on the 02 day of December.'

>>> # dates support calendar arithmetic
>>> birthday = date(1964, 7, 31)
>>> age = now - birthday
>>> age.days
14368
```

## 10.9. Data Compression
Common data archiving and compression formats are directly supported by modules including: [`zlib`](https://docs.python.org/3/library/zlib.html#module-zlib "zlib: Low-level interface to compression and decompression routines compatible with gzip."), [`gzip`](https://docs.python.org/3/library/gzip.html#module-gzip "gzip: Interfaces for gzip compression and decompression using file objects."), [`bz2`](https://docs.python.org/3/library/bz2.html#module-bz2 "bz2: Interfaces for bzip2 compression and decompression."), [`lzma`](https://docs.python.org/3/library/lzma.html#module-lzma "lzma: A Python wrapper for the liblzma compression library."), [`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile "zipfile: Read and write ZIP-format archive files.") and [`tarfile`](https://docs.python.org/3/library/tarfile.html#module-tarfile "tarfile: Read and write tar-format archive files.").
> `zlib/gzip/bz2/lzma/zipfile/tarfile` 模块支持常用的数据归档和压缩操作

```
>>> import zlib
>>> s = b'witch which has which witches wrist watch'
>>> len(s)
41
>>> t = zlib.compress(s)
>>> len(t)
37
>>> zlib.decompress(t)
b'witch which has which witches wrist watch'
>>> zlib.crc32(s)
226805979
```

## 10.10. Performance Measurement
Some Python users develop a deep interest in knowing the relative performance of different approaches to the same problem. Python provides a measurement tool that answers those questions immediately.
> Python 提供了性能度量工具

For example, it may be tempting to use the tuple packing and unpacking feature instead of the traditional approach to swapping arguments. The [`timeit`](https://docs.python.org/3/library/timeit.html#module-timeit "timeit: Measure the execution time of small code snippets.") module quickly demonstrates a modest performance advantage:
> `timeit` 模块可以用于计时语句执行的时间

```
>>> from timeit import Timer
>>> Timer('t=a; a=b; b=t', 'a=1; b=2').timeit()
0.57535828626024577
>>> Timer('a,b = b,a', 'a=1; b=2').timeit()
0.54962537085770791
```

In contrast to [`timeit`](https://docs.python.org/3/library/timeit.html#module-timeit "timeit: Measure the execution time of small code snippets.")’s fine level of granularity, the [`profile`](https://docs.python.org/3/library/profile.html#module-profile "profile: Python source profiler.") and [`pstats`](https://docs.python.org/3/library/profile.html#module-pstats "pstats: Statistics object for use with the profiler.") modules provide tools for identifying time critical sections in larger blocks of code.
> `profile/pstats` 模块提供对于关键代码区域的计时工具

## 10.11. Quality Control
One approach for developing high quality software is to write tests for each function as it is developed and to run those tests frequently during the development process.

The [`doctest`](https://docs.python.org/3/library/doctest.html#module-doctest "doctest: Test pieces of code within docstrings.") module provides a tool for scanning a module and validating tests embedded in a program’s docstrings. Test construction is as simple as cutting-and-pasting a typical call along with its results into the docstring. This improves the documentation by providing the user with an example and it allows the doctest module to make sure the code remains true to the documentation:
> `doctest` 模块提供了扫描一个模块并且使用嵌入在程序的 docstring 中的测试对程序进行验证的工具
> 使用 `doctest` 模块可以使得测试构建就是将调用和它对应的结果放入 docstring 中

```python
def average(values):
    """Computes the arithmetic mean of a list of numbers.

    >>> print(average([20, 30, 70]))
    40.0
    """
    return sum(values) / len(values)

import doctest
doctest.testmod()   # automatically validate the embedded tests
```

The [`unittest`](https://docs.python.org/3/library/unittest.html#module-unittest "unittest: Unit testing framework for Python.") module is not as effortless as the [`doctest`](https://docs.python.org/3/library/doctest.html#module-doctest "doctest: Test pieces of code within docstrings.") module, but it allows a more comprehensive set of tests to be maintained in a separate file:
> `unittest` 模块比 `doctest` 模块更加难以使用，但该模块允许在一个分离的文件中维护一个综合的测试集合

```python
import unittest

class TestStatisticalFunctions(unittest.TestCase):

    def test_average(self):
        self.assertEqual(average([20, 30, 70]), 40.0)
        self.assertEqual(round(average([1, 5, 7]), 1), 4.3)
        with self.assertRaises(ZeroDivisionError):
            average([])
        with self.assertRaises(TypeError):
            average(20, 30, 70)

unittest.main()  # Calling from the command line invokes all tests
```

## 10.12. Batteries Included
Python has a “batteries included” philosophy. This is best seen through the sophisticated and robust capabilities of its larger packages. For example:
> Python 具有“内置电池”哲学，即 Python 标准库提供了大量的功能，使得开发者能够无需额外依赖第三方库即可完成许多常见的任务，包括但不限于网络编程、文本处理、日期/时间处理、数据持久化、安全性和加密等

- The [`xmlrpc.client`](https://docs.python.org/3/library/xmlrpc.client.html#module-xmlrpc.client "xmlrpc.client: XML-RPC client access.") and [`xmlrpc.server`](https://docs.python.org/3/library/xmlrpc.server.html#module-xmlrpc.server "xmlrpc.server: Basic XML-RPC server implementations.") modules make implementing remote procedure calls into an almost trivial task. Despite the modules’ names, no direct knowledge or handling of XML is needed.
- The [`email`](https://docs.python.org/3/library/email.html#module-email "email: Package supporting the parsing, manipulating, and generating email messages.") package is a library for managing email messages, including MIME and other [**RFC 2822**](https://datatracker.ietf.org/doc/html/rfc2822.html) -based message documents. Unlike [`smtplib`](https://docs.python.org/3/library/smtplib.html#module-smtplib "smtplib: SMTP protocol client (requires sockets).") and [`poplib`](https://docs.python.org/3/library/poplib.html#module-poplib "poplib: POP3 protocol client (requires sockets).") which actually send and receive messages, the email package has a complete toolset for building or decoding complex message structures (including attachments) and for implementing internet encoding and header protocols.
- The [`json`](https://docs.python.org/3/library/json.html#module-json "json: Encode and decode the JSON format.") package provides robust support for parsing this popular data interchange format. The [`csv`](https://docs.python.org/3/library/csv.html#module-csv "csv: Write and read tabular data to and from delimited files.") module supports direct reading and writing of files in Comma-Separated Value format, commonly supported by databases and spreadsheets. XML processing is supported by the [`xml.etree.ElementTree`](https://docs.python.org/3/library/xml.etree.elementtree.html#module-xml.etree.ElementTree "xml.etree.ElementTree: Implementation of the ElementTree API."), [`xml.dom`](https://docs.python.org/3/library/xml.dom.html#module-xml.dom "xml.dom: Document Object Model API for Python.") and [`xml.sax`](https://docs.python.org/3/library/xml.sax.html#module-xml.sax "xml.sax: Package containing SAX2 base classes and convenience functions.") packages. Together, these modules and packages greatly simplify data interchange between Python applications and other tools.
- The [`sqlite3`](https://docs.python.org/3/library/sqlite3.html#module-sqlite3 "sqlite3: A DB-API 2.0 implementation using SQLite 3.x.") module is a wrapper for the SQLite database library, providing a persistent database that can be updated and accessed using slightly nonstandard SQL syntax.
- Internationalization is supported by a number of modules including [`gettext`](https://docs.python.org/3/library/gettext.html#module-gettext "gettext: Multilingual internationalization services."), [`locale`](https://docs.python.org/3/library/locale.html#module-locale "locale: Internationalization services."), and the [`codecs`](https://docs.python.org/3/library/codecs.html#module-codecs "codecs: Encode and decode data and streams.") package.

> `xmlrpc.client/server` 模块用于处理远程过程调用
> `email` 用于管理电子邮件信息，`email` 不像 `smtplib` 和 `poplib` 一样直接发送和接受信息，而是提供了完整的用于构建或解码复杂信息结构（包括附件）以及用于执行互联网编码和协议头的工具包
> `json` 提供解析 json 文件的支持，`csv` 提供直接读写 csv 文件的支持，`xml.etree.ElementTree/xml.dom/xlm.sax` 提供了 XML 处理支持
> `sqlite3` 是 SQLite 数据库库的包装器，它提供了一个持续的数据库，可以被更新和使用 SQL 语法访问
> `gettext/local/codecs` 包支持了中间交互

# 11. Brief Tour of the Standard Library — Part II
This second tour covers more advanced modules that support professional programming needs. These modules rarely occur in small scripts.

## 11.1. Output Formatting
The [`reprlib`](https://docs.python.org/3/library/reprlib.html#module-reprlib "reprlib: Alternate repr() implementation with size limits.") module provides a version of [`repr()`](https://docs.python.org/3/library/functions.html#repr "repr") customized for abbreviated displays of large or deeply nested containers:
> `reprlib` 模块提供了一个可以展现大型或深度嵌入的容器的简要表述的 `repr()` 版本

```
>>> import reprlib
>>> reprlib.repr(set('supercalifragilisticexpialidocious'))
"{'a', 'c', 'd', 'e', 'f', 'g', ...}"
```

The [`pprint`](https://docs.python.org/3/library/pprint.html#module-pprint "pprint: Data pretty printer.") module offers more sophisticated control over printing both built-in and user defined objects in a way that is readable by the interpreter. When the result is longer than one line, the “pretty printer” adds line breaks and indentation to more clearly reveal data structure:
> `pprint` 模块提供了用解释器可读的方式打印内建和用户定义对象的工具，当结果超出一行时，它会添加换行和缩进，以清晰展示数据结构

```
>>> import pprint
>>> t = [[[['black', 'cyan'], 'white', ['green', 'red']], [['magenta',
...     'yellow'], 'blue']]]
...
>>> pprint.pprint(t, width=30)
[[[['black', 'cyan'],
   'white',
   ['green', 'red']],
  [['magenta', 'yellow'],
   'blue']]]
```

The [`textwrap`](https://docs.python.org/3/library/textwrap.html#module-textwrap "textwrap: Text wrapping and filling") module formats paragraphs of text to fit a given screen width:
> `textwarp` 模块将段落文本 fit in 一个给定的屏幕宽度

```
>>> import textwrap
>>> doc = """The wrap() method is just like fill() except that it returns
... a list of strings instead of one big string with newlines to separate
... the wrapped lines."""
...
>>> print(textwrap.fill(doc, width=40))
The wrap() method is just like fill()
except that it returns a list of strings
instead of one big string with newlines
to separate the wrapped lines.
```

The [`locale`](https://docs.python.org/3/library/locale.html#module-locale "locale: Internationalization services.") module accesses a database of culture specific data formats. The grouping attribute of locale’s format function provides a direct way of formatting numbers with group separators:
> `locale` 模块访问一个具有针对文化的数据格式的数据库

```
>>> import locale
>>> locale.setlocale(locale.LC_ALL, 'English_United States.1252')
'English_United States.1252'
>>> conv = locale.localeconv()          # get a mapping of conventions
>>> x = 1234567.8
>>> locale.format_string("%d", x, grouping=True)
'1,234,567'
>>> locale.format_string("%s%.*f", (conv['currency_symbol'],
...                      conv['frac_digits'], x), grouping=True)
'$1,234,567.80'
```

## 11.2. Templating
The [`string`](https://docs.python.org/3/library/string.html#module-string "string: Common string operations.") module includes a versatile [`Template`](https://docs.python.org/3/library/string.html#string.Template "string.Template") class with a simplified syntax suitable for editing by end-users. This allows users to customize their applications without having to alter the application.
> `string` 模块中包含了 `Template` 类
> 该类允许用户在不修改应用的前提下自定义应用

The format uses placeholder names formed by `$` with valid Python identifiers (alphanumeric characters and underscores). Surrounding the placeholder with braces allows it to be followed by more alphanumeric letters with no intervening spaces. Writing `$$` creates a single escaped `$`:
> Template 的格式使用的占位符名称由 `$` 加上有效的 Python 标识符构成，标识符旁边可以由括号围起（可选）
> `$$` 用于转义 `$`

```
>>> from string import Template
>>> t = Template('${village}folk send $$10 to $cause.')
>>> t.substitute(village='Nottingham', cause='the ditch fund')
'Nottinghamfolk send $10 to the ditch fund.'
```

The [`substitute()`](https://docs.python.org/3/library/string.html#string.Template.substitute "string.Template.substitute") method raises a [`KeyError`](https://docs.python.org/3/library/exceptions.html#KeyError "KeyError") when a placeholder is not supplied in a dictionary or a keyword argument. For mail-merge style applications, user supplied data may be incomplete and the [`safe_substitute()`](https://docs.python.org/3/library/string.html#string.Template.safe_substitute "string.Template.safe_substitute") method may be more appropriate — it will leave placeholders unchanged if data is missing:
> 当占位符没有在提供的字典中的 keys 出现时，`substitute()` 会抛出 `KeyError` 
> `safe_substitute()` 则会让这些占位符保持原样

```
>>> t = Template('Return the $item to $owner.')
>>> d = dict(item='unladen swallow')
>>> t.substitute(d)
Traceback (most recent call last):
  ...
KeyError: 'owner'
>>> t.safe_substitute(d)
'Return the unladen swallow to $owner.'
```

Template subclasses can specify a custom delimiter. For example, a batch renaming utility for a photo browser may elect to use percent signs for placeholders such as the current date, image sequence number, or file format:
> 继承 Template 的类可以指定自定义的分隔符

```
>>> import time, os.path
>>> photofiles = ['img_1074.jpg', 'img_1076.jpg', 'img_1077.jpg']
>>> class BatchRename(Template):
...     delimiter = '%'
...
>>> fmt = input('Enter rename style (%d-date %n-seqnum %f-format):  ')
Enter rename style (%d-date %n-seqnum %f-format):  Ashley_%n%f

>>> t = BatchRename(fmt)
>>> date = time.strftime('%d%b%y')
>>> for i, filename in enumerate(photofiles):
...     base, ext = os.path.splitext(filename)
...     newname = t.substitute(d=date, n=i, f=ext)
...     print('{0} --> {1}'.format(filename, newname))

img_1074.jpg --> Ashley_0.jpg
img_1076.jpg --> Ashley_1.jpg
img_1077.jpg --> Ashley_2.jpg
```

Another application for templating is separating program logic from the details of multiple output formats. This makes it possible to substitute custom templates for XML files, plain text reports, and HTML web reports.

## 11.3. Working with Binary Data Record Layouts
The [`struct`](https://docs.python.org/3/library/struct.html#module-struct "struct: Interpret bytes as packed binary data.") module provides [`pack()`](https://docs.python.org/3/library/struct.html#struct.pack "struct.pack") and [`unpack()`](https://docs.python.org/3/library/struct.html#struct.unpack "struct.unpack") functions for working with variable length binary record formats. The following example shows how to loop through header information in a ZIP file without using the [`zipfile`](https://docs.python.org/3/library/zipfile.html#module-zipfile "zipfile: Read and write ZIP-format archive files.") module. Pack codes `"H"` and `"I"` represent two and four byte unsigned numbers respectively. The `"<"` indicates that they are standard size and in little-endian byte order:
> `struct` 模块提供 `pack()/unpack()` 函数用于处理变长的二进制记录格式
>  Pack 码 H 和 I 分别表示2和4字节无符号数，<表示它们是标准大小，小端序

```python
import struct

with open('myfile.zip', 'rb') as f:
    data = f.read()

start = 0
for i in range(3):                      # show the first 3 file headers
    start += 14
    fields = struct.unpack('<IIIHH', data[start:start+16])
    crc32, comp_size, uncomp_size, filenamesize, extra_size = fields

    start += 16
    filename = data[start:start+filenamesize]
    start += filenamesize
    extra = data[start:start+extra_size]
    print(filename, hex(crc32), comp_size, uncomp_size)

    start += extra_size + comp_size     # skip to the next header
```

## 11.4. Multi-threading
Threading is a technique for decoupling tasks which are not sequentially dependent. Threads can be used to improve the responsiveness of applications that accept user input while other tasks run in the background. A related use case is running I/O in parallel with computations in another thread.

The following code shows how the high level [`threading`](https://docs.python.org/3/library/threading.html#module-threading "threading: Thread-based parallelism.") module can run tasks in background while the main program continues to run:
> `threading` 模块用于多线程

```python
import threading, zipfile

class AsyncZip(threading.Thread):
    def __init__(self, infile, outfile):
        threading.Thread.__init__(self)
        self.infile = infile
        self.outfile = outfile

    def run(self):
        f = zipfile.ZipFile(self.outfile, 'w', zipfile.ZIP_DEFLATED)
        f.write(self.infile)
        f.close()
        print('Finished background zip of:', self.infile)

background = AsyncZip('mydata.txt', 'myarchive.zip')
background.start()
print('The main program continues to run in foreground.')

background.join()    # Wait for the background task to finish
print('Main program waited until background was done.')
```

The principal challenge of multi-threaded applications is coordinating threads that share data or other resources. To that end, the threading module provides a number of synchronization primitives including locks, events, condition variables, and semaphores.
> `thraeding` 模块提供了许多同步原语，包括锁、事件、条件变量、旗语

While those tools are powerful, minor design errors can result in problems that are difficult to reproduce. So, the preferred approach to task coordination is to concentrate all access to a resource in a single thread and then use the [`queue`](https://docs.python.org/3/library/queue.html#module-queue "queue: A synchronized queue class.") module to feed that thread with requests from other threads. Applications using [`Queue`](https://docs.python.org/3/library/queue.html#queue.Queue "queue.Queue") objects for inter-thread communication and coordination are easier to design, more readable, and more reliable.
> 对于任务协同更偏好的方式是将所有对资源的访问集中在单个线程，然后用 `queue` 模块来将来自其他线程的请求 feed 该线程
> 使用 `Queue` 对象来管理线程间通讯和协调的应用往往更加易于设计和可靠

## 11.5. Logging
The [`logging`](https://docs.python.org/3/library/logging.html#module-logging "logging: Flexible event logging system for applications.") module offers a full featured and flexible logging system. At its simplest, log messages are sent to a file or to `sys.stderr`:
> `logging` 模块提供了日志系统
> 最简单的情况下，日志信息会发送到一个文件或者到 `sys.stderr`

```python
import logging
logging.debug('Debugging information')
logging.info('Informational message')
logging.warning('Warning:config file %s not found', 'server.conf')
logging.error('Error occurred')
logging.critical('Critical error -- shutting down')
```

This produces the following output:

```
WARNING:root:Warning:config file server.conf not found
ERROR:root:Error occurred
CRITICAL:root:Critical error -- shutting down
```

By default, informational and debugging messages are suppressed and the output is sent to standard error. Other output options include routing messages through email, datagrams, sockets, or to an HTTP Server. New filters can select different routing based on message priority: [`DEBUG`](https://docs.python.org/3/library/logging.html#logging.DEBUG "logging.DEBUG"), [`INFO`](https://docs.python.org/3/library/logging.html#logging.INFO "logging.INFO"), [`WARNING`](https://docs.python.org/3/library/logging.html#logging.WARNING "logging.WARNING"), [`ERROR`](https://docs.python.org/3/library/logging.html#logging.ERROR "logging.ERROR"), and [`CRITICAL`](https://docs.python.org/3/library/logging.html#logging.CRITICAL "logging.CRITICAL").
> 默认情况下 information 和 debugging 信息不会出现，其他信息会被送到 stderr
> 其他的输出选项包括 routing 信息等
> 可以基于信息优先级：debug、info、warning、error、critiral 来筛选 routing 信息

The logging system can be configured directly from Python or can be loaded from a user editable configuration file for customized logging without altering the application.

## 11.6. Weak References
Python does automatic memory management (reference counting for most objects and [garbage collection](https://docs.python.org/3/glossary.html#term-garbage-collection) to eliminate cycles). The memory is freed shortly after the last reference to it has been eliminated.
> Python 进行的是自动的内存管理（对于多数对象，进行引用计数，使用垃圾收集来消除循环引用）
> 内存在最后一个引用被消除时候就会被释放

This approach works fine for most applications but occasionally there is a need to track objects only as long as they are being used by something else. Unfortunately, just tracking them creates a reference that makes them permanent. The [`weakref`](https://docs.python.org/3/library/weakref.html#module-weakref "weakref: Support for weak references and weak dictionaries.") module provides tools for tracking objects without creating a reference. When the object is no longer needed, it is automatically removed from a weakref table and a callback is triggered for weakref objects. Typical applications include caching objects that are expensive to create:
> 当一个对象被其他对象使用时，有时需要追踪一个对象
> 但仅仅追踪该对象会导致创建一个永久的引用
> `weakref` 模块提供不创建引用追踪对象的函数，当对象不再需要，会被自动从 weakref 表中移除，同时会为 weakref 对象调用一个回调函数
> 该模块常见的应用是对于创建很昂贵的对象进行缓存

```
>>> import weakref, gc
>>> class A:
...     def __init__(self, value):
...         self.value = value
...     def __repr__(self):
...         return str(self.value)
...
>>> a = A(10)                   # create a reference
>>> d = weakref.WeakValueDictionary()
>>> d['primary'] = a            # does not create a reference
>>> d['primary']                # fetch the object if it is still alive
10
>>> del a                       # remove the one reference
>>> gc.collect()                # run garbage collection right away
0
>>> d['primary']                # entry was automatically removed
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    d['primary']                # entry was automatically removed
  File "C:/python312/lib/weakref.py", line 46, in __getitem__
    o = self.data[key]()
KeyError: 'primary'
```

## 11.7. Tools for Working with Lists
Many data structure needs can be met with the built-in list type. However, sometimes there is a need for alternative implementations with different performance trade-offs.
> 大多数数据结构可以用内建的 list 类型实现
> 但也存在考虑性能的其他实现

The [`array`](https://docs.python.org/3/library/array.html#module-array "array: Space efficient arrays of uniformly typed numeric values.") module provides an [`array`](https://docs.python.org/3/library/array.html#array.array "array.array") object that is like a list that stores only homogeneous data and stores it more compactly. The following example shows an array of numbers stored as two byte unsigned binary numbers (typecode `"H"`) rather than the usual 16 bytes per entry for regular lists of Python int objects:
> `array` 模块提供了 `array` 对象，仅存储同质的数据，并且紧凑存储
> 可以用类别码 H 表示存储数据为2字节的无符号数
> list 默认使用16个字节存储每一个 Python int 对象

```
>>> from array import array
>>> a = array('H', [4000, 10, 700, 22222])
>>> sum(a)
26932
>>> a[1:3]
array('H', [10, 700])
```

The [`collections`](https://docs.python.org/3/library/collections.html#module-collections "collections: Container datatypes") module provides a [`deque`](https://docs.python.org/3/library/collections.html#collections.deque "collections.deque") object that is like a list with faster appends and pops from the left side but slower lookups in the middle. These objects are well suited for implementing queues and breadth first tree searches:
> `collections` 模块提供 `deque` 对象，提供比 list 更快的 append 和 pop 操作，但是在中间查找更慢

```
>>> from collections import deque
>>> d = deque(["task1", "task2", "task3"])
>>> d.append("task4")
>>> print("Handling", d.popleft())
Handling task1
```

```
unsearched = deque([starting_node])
def breadth_first_search(unsearched):
    node = unsearched.popleft()
    for m in gen_moves(node):
        if is_goal(m):
            return m
        unsearched.append(m)
```

In addition to alternative list implementations, the library also offers other tools such as the [`bisect`](https://docs.python.org/3/library/bisect.html#module-bisect "bisect: Array bisection algorithms for binary searching.") module with functions for manipulating sorted lists:
> `bisect` 模块提供对于有序的 list 进行操作的函数

```
>>> import bisect
>>> scores = [(100, 'perl'), (200, 'tcl'), (400, 'lua'), (500, 'python')]
>>> bisect.insort(scores, (300, 'ruby'))
>>> scores
[(100, 'perl'), (200, 'tcl'), (300, 'ruby'), (400, 'lua'), (500, 'python')]
```

The [`heapq`](https://docs.python.org/3/library/heapq.html#module-heapq "heapq: Heap queue algorithm (a.k.a. priority queue).") module provides functions for implementing heaps based on regular lists. The lowest valued entry is always kept at position zero. This is useful for applications which repeatedly access the smallest element but do not want to run a full list sort:
> `headq` 提供了基于常规的 lsit 实现最小堆的函数 
> 方便需要不断访问最小值但不需要完全排序的应用 `

```
>>> from heapq import heapify, heappop, heappush
>>> data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
>>> heapify(data)                      # rearrange the list into heap order
>>> heappush(data, -5)                 # add a new entry
>>> [heappop(data) for i in range(3)]  # fetch the three smallest entries
[-5, 0, 1]
```

## 11.8. Decimal Floating-Point Arithmetic
The [`decimal`](https://docs.python.org/3/library/decimal.html#module-decimal "decimal: Implementation of the General Decimal Arithmetic  Specification.") module offers a [`Decimal`](https://docs.python.org/3/library/decimal.html#decimal.Decimal "decimal.Decimal") datatype for decimal floating-point arithmetic. Compared to the built-in [`float`](https://docs.python.org/3/library/functions.html#float "float") implementation of binary floating point, the class is especially helpful for
> `decimal` 模块提供了 `Decimal` 数据类型用于十进制浮点运算，可以用于需要精确的十进制浮点数表示的应用中（运算结果和手算匹配）
> 内建的 `float` 执行的是二进制浮点运算

- financial applications and other uses which require exact decimal representation,
- control over precision,
- control over rounding to meet legal or regulatory requirements,
- tracking of significant decimal places, or
- applications where the user expects the results to match calculations done by hand.

For example, calculating a 5% tax on a 70 cent phone charge gives different results in decimal floating point and binary floating point. The difference becomes significant if the results are rounded to the nearest cent:

```
>>> from decimal import *
>>> round(Decimal('0.70') * Decimal('1.05'), 2)
Decimal('0.74')
>>> round(.70 * 1.05, 2)
0.73
```

The [`Decimal`](https://docs.python.org/3/library/decimal.html#decimal.Decimal "decimal.Decimal") result keeps a trailing zero, automatically inferring four place significance from multiplicands with two place significance. Decimal reproduces mathematics as done by hand and avoids issues that can arise when binary floating point cannot exactly represent decimal quantities.
> `Decimal` 结果会保持一个尾部零，并且会自动从两位有效性的乘数推断出四位的有效性
> `Decimal` 会匹配手算的结果，避免了二进制浮点数无法精确表示十进制数值时会出现的问题

Exact representation enables the [`Decimal`](https://docs.python.org/3/library/decimal.html#decimal.Decimal "decimal.Decimal") class to perform modulo calculations and equality tests that are unsuitable for binary floating point:

```
>>> Decimal('1.00') % Decimal('.10')
Decimal('0.00')
>>> 1.00 % 0.10
0.09999999999999995

>>> sum([Decimal('0.1')]*10) == Decimal('1.0')
True
>>> 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 == 1.0
False
```

The [`decimal`](https://docs.python.org/3/library/decimal.html#module-decimal "decimal: Implementation of the General Decimal Arithmetic  Specification.") module provides arithmetic with as much precision as needed:
> `decimal` 模块也提供了所需要的更高精度的算术

```
>>> getcontext().prec = 36
>>> Decimal(1) / Decimal(7)
Decimal('0.142857142857142857142857142857142857')
```

# 12. Virtual Environments and Packages
## 12.1. Introduction
Python applications will often use packages and modules that don’t come as part of the standard library. Applications will sometimes need a specific version of a library, because the application may require that a particular bug has been fixed or the application may be written using an obsolete version of the library’s interface.

This means it may not be possible for one Python installation to meet the requirements of every application. If application A needs version 1.0 of a particular module but application B needs version 2.0, then the requirements are in conflict and installing either version 1.0 or 2.0 will leave one application unable to run.

The solution for this problem is to create a [virtual environment](https://docs.python.org/3/glossary.html#term-virtual-environment), a self-contained directory tree that contains a Python installation for a particular version of Python, plus a number of additional packages.
> 虚拟环境是一个自洽的目录树，它包含了一个特定版本的 Python 安装，以及一些额外的包

Different applications can then use different virtual environments. To resolve the earlier example of conflicting requirements, application A can have its own virtual environment with version 1.0 installed while application B has another virtual environment with version 2.0. If application B requires a library be upgraded to version 3.0, this will not affect application A’s environment.

## 12.2. Creating Virtual Environments
The module used to create and manage virtual environments is called [`venv`](https://docs.python.org/3/library/venv.html#module-venv "venv: Creation of virtual environments."). [`venv`](https://docs.python.org/3/library/venv.html#module-venv "venv: Creation of virtual environments.") will install the Python version from which the command was run (as reported by the [`--version`](https://docs.python.org/3/using/cmdline.html#cmdoption-version) option). For instance, executing the command with `python3.12` will install version 3.12.
> 用于创建虚拟环境的模块称为 `venv` 
> `venv` 根据命令行中运行的 Python 版本安装对应的 Python 版本，例如，执行 `python3.12 ...` 会安装 Python 3.12

To create a virtual environment, decide upon a directory where you want to place it, and run the [`venv`](https://docs.python.org/3/library/venv.html#module-venv "venv: Creation of virtual environments.") module as a script with the directory path:
> 要创建虚拟环境，首先决定一个我们需要放置该环境的目录，然后将 `venv` 模块作为一个脚本运行，同时参数是目录路径

```
python -m venv tutorial-env
```

This will create the `tutorial-env` directory if it doesn’t exist, and also create directories inside it containing a copy of the Python interpreter and various supporting files.
> 该命令会在 `tutorial-env` 不存在时创建该目录，同时在该目录下创建包含了一个 Python 解释器的拷贝以及一些支持文件的目录

A common directory location for a virtual environment is `.venv`. This name keeps the directory typically hidden in your shell and thus out of the way while giving it a name that explains why the directory exists. It also prevents clashing with `.env` environment variable definition files that some tooling supports.
> `venv` 用的隐藏目录是 `.venv` 

Once you’ve created a virtual environment, you may activate it.
> 运行激活脚本以激活环境
> 激活脚本是 shell 脚本

On Windows, run:

```
tutorial-env\Scripts\activate
```

On Unix or MacOS, run:

```
source tutorial-env/bin/activate
```

(This script is written for the bash shell. If you use the **csh** or **fish** shells, there are alternate `activate.csh` and `activate.fish` scripts you should use instead.)

Activating the virtual environment will change your shell’s prompt to show what virtual environment you’re using, and modify the environment so that running `python` will get you that particular version and installation of Python. For example:
> 激活之后，环境会改为虚拟环境，此时运行 `python` 得到的是特定安装版本的 Python

```
$ source ~/envs/tutorial-env/bin/activate
(tutorial-env) $ python
Python 3.5.1 (default, May  6 2016, 10:59:36)
  ...
>>> import sys
>>> sys.path
['', '/usr/local/lib/python35.zip', ...,
'~/envs/tutorial-env/lib/python3.5/site-packages']
>>>
```

To deactivate a virtual environment, type:

```
deactivate
```

into the terminal.

## 12.3. Managing Packages with pip
You can install, upgrade, and remove packages using a program called **pip**. By default `pip` will install packages from the [Python Package Index](https://pypi.org/). You can browse the Python Package Index by going to it in your web browser.
> `pip` 默认从 Python Package Index 中安装包

`pip` has a number of subcommands: “install”, “uninstall”, “freeze”, etc. (Consult the [Installing Python Modules](https://docs.python.org/3/installing/index.html#installing-index) guide for complete documentation for `pip`.)
> `pip` 有一系列子命令：install、uninstall、freeze

You can install the latest version of a package by specifying a package’s name:
> 默认安装最新版本

```
(tutorial-env) $ python -m pip install novas
Collecting novas
  Downloading novas-3.1.1.3.tar.gz (136kB)
Installing collected packages: novas
  Running setup.py install for novas
Successfully installed novas-3.1.1.3
```

You can also install a specific version of a package by giving the package name followed by `==` and the version number:
> 可以用过 `==` 指定需要安装的版本

```
(tutorial-env) $ python -m pip install requests==2.6.0
Collecting requests==2.6.0
  Using cached requests-2.6.0-py2.py3-none-any.whl
Installing collected packages: requests
Successfully installed requests-2.6.0
```

If you re-run this command, `pip` will notice that the requested version is already installed and do nothing. You can supply a different version number to get that version, or you can run `python -m pip install --upgrade` to upgrade the package to the latest version:
> `--upgrade` 用于更新到最新版本

```
(tutorial-env) $ python -m pip install --upgrade requests
Collecting requests
Installing collected packages: requests
  Found existing installation: requests 2.6.0
    Uninstalling requests-2.6.0:
      Successfully uninstalled requests-2.6.0
Successfully installed requests-2.7.0
```

`python -m pip uninstall` followed by one or more package names will remove the packages from the virtual environment.
> uninstall 用于从虚拟环境中移除包

`python -m pip show` will display information about a particular package:
> show 用于展示特定包的信息

```
(tutorial-env) $ python -m pip show requests
---
Metadata-Version: 2.0
Name: requests
Version: 2.7.0
Summary: Python HTTP for Humans.
Home-page: http://python-requests.org
Author: Kenneth Reitz
Author-email: me@kennethreitz.com
License: Apache 2.0
Location: /Users/akuchling/envs/tutorial-env/lib/python3.4/site-packages
Requires:
```

`python -m pip list` will display all of the packages installed in the virtual environment:
> list 会列出所有安装的包

```
(tutorial-env) $ python -m pip list
novas (3.1.1.3)
numpy (1.9.2)
pip (7.0.3)
requests (2.7.0)
setuptools (16.0)
```

`python -m pip freeze` will produce a similar list of the installed packages, but the output uses the format that `python -m pip install` expects. A common convention is to put this list in a `requirements.txt` file:
> freeze 同样列出所有安装的包的信息，但是格式如何 pip install 需要的格式
> 一般可以将该列表放在 `requirements.txt` 中

```
(tutorial-env) $ python -m pip freeze > requirements.txt
(tutorial-env) $ cat requirements.txt
novas==3.1.1.3
numpy==1.9.2
requests==2.7.0
```

The `requirements.txt` can then be committed to version control and shipped as part of an application. Users can then install all the necessary packages with `install -r`:
> `requirements.txt` 可以被 commit，作为应用程序的一部分
> 用户可以通过 `install -r requirements.txt` 安装其中需要的包 `

```
(tutorial-env) $ python -m pip install -r requirements.txt
Collecting novas==3.1.1.3 (from -r requirements.txt (line 1))
  ...
Collecting numpy==1.9.2 (from -r requirements.txt (line 2))
  ...
Collecting requests==2.7.0 (from -r requirements.txt (line 3))
  ...
Installing collected packages: novas, numpy, requests
  Running setup.py install for novas
Successfully installed novas-3.1.1.3 numpy-1.9.2 requests-2.7.0
```

`pip` has many more options. Consult the [Installing Python Modules](https://docs.python.org/3/installing/index.html#installing-index) guide for complete documentation for `pip`. When you’ve written a package and want to make it available on the Python Package Index, consult the [Python packaging user guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

# 13. What Now?
Reading this tutorial has probably reinforced your interest in using Python — you should be eager to apply Python to solving your real-world problems. Where should you go to learn more?

This tutorial is part of Python’s documentation set. Some other documents in the set are:

- [The Python Standard Library](https://docs.python.org/3/library/index.html#library-index):
    You should browse through this manual, which gives complete (though terse) reference material about types, functions, and the modules in the standard library. The standard Python distribution includes a _lot_ of additional code. There are modules to read Unix mailboxes, retrieve documents via HTTP, generate random numbers, parse command-line options, compress data, and many other tasks. Skimming through the Library Reference will give you an idea of what’s available.
- [Installing Python Modules](https://docs.python.org/3/installing/index.html#installing-index) explains how to install additional modules written by other Python users.
- [The Python Language Reference](https://docs.python.org/3/reference/index.html#reference-index): A detailed explanation of Python’s syntax and semantics. It’s heavy reading, but is useful as a complete guide to the language itself.

More Python resources:
- [https://www.python.org](https://www.python.org/): The major Python web site. It contains code, documentation, and pointers to Python-related pages around the web.
- [https://docs.python.org](https://docs.python.org/): Fast access to Python’s documentation.
- [https://pypi.org](https://pypi.org/): The Python Package Index, previously also nicknamed the Cheese Shop [1](https://docs.python.org/3/tutorial/whatnow.html#id2), is an index of user-created Python modules that are available for download. Once you begin releasing code, you can register it here so that others can find it.
- [https://code.activestate.com/recipes/langs/python/](https://code.activestate.com/recipes/langs/python/): The Python Cookbook is a sizable collection of code examples, larger modules, and useful scripts. Particularly notable contributions are collected in a book also titled Python Cookbook (O’Reilly & Associates, ISBN 0-596-00797-3.)
- [https://pyvideo.org](https://pyvideo.org/) collects links to Python-related videos from conferences and user-group meetings.
- [https://scipy.org](https://scipy.org/): The Scientific Python project includes modules for fast array computations and manipulations plus a host of packages for such things as linear algebra, Fourier transforms, non-linear solvers, random number distributions, statistical analysis and the like.

For Python-related questions and problem reports, you can post to the newsgroup _comp.lang.python_, or send them to the mailing list at [python-list@python.org](mailto:python-list%40python.org). The newsgroup and mailing list are gatewayed, so messages posted to one will automatically be forwarded to the other. There are hundreds of postings a day, asking (and answering) questions, suggesting new features, and announcing new modules. Mailing list archives are available at [https://mail.python.org/pipermail/](https://mail.python.org/pipermail/).

Before posting, be sure to check the list of [Frequently Asked Questions](https://docs.python.org/3/faq/index.html#faq-index) (also called the FAQ). The FAQ answers many of the questions that come up again and again, and may already contain the solution for your problem.

Footnotes
[1](https://docs.python.org/3/tutorial/whatnow.html#id1) “Cheese Shop” is a Monty Python’s sketch: a customer enters a cheese shop, but whatever cheese he asks for, the clerk says it’s missing.

# 14. Interactive Input Editing and History Substitution
Some versions of the Python interpreter support editing of the current input line and history substitution, similar to facilities found in the Korn shell and the GNU Bash shell. This is implemented using the [GNU Readline](https://tiswww.case.edu/php/chet/readline/rltop.html) library, which supports various styles of editing. This library has its own documentation which we won’t duplicate here.
> 一些 Python 解释器版本基于 GNU Readling 库实现了编辑辅助和历史替换功能

## 14.1. Tab Completion and History Editing
Completion of variable and module names is [automatically enabled](https://docs.python.org/3/library/site.html#rlcompleter-config) at interpreter startup so that the Tab key invokes the completion function; it looks at Python statement names, the current local variables, and the available module names. For dotted expressions such as `string.a`, it will evaluate the expression up to the final `'.'` and then suggest completions from the attributes of the resulting object. Note that this may execute application-defined code if an object with a [`__getattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattr__ "object.__getattr__") method is part of the expression. The default configuration also saves your history into a file named `.python_history` in your user directory. The history will be available again during the next interactive interpreter session.
> Tab 会自动调用补全函数，它观察 Python 语句名称、当前的局部变量、可用的模块名称、模块的属性以进行补全
> 如果定义了 `__getattr__` 方法的对象是表达式的一部分，则Tab 补全可能会执行一部分应用定义的代码
> 补全的默认配置会将命令历史存储在用户目录中的 `.python_history` ，这使得我们在下一次的解释器会话中也可以使用之前的历史

## 14.2. Alternatives to the Interactive Interpreter
This facility is an enormous step forward compared to earlier versions of the interpreter; however, some wishes are left: It would be nice if the proper indentation were suggested on continuation lines (the parser knows if an indent token is required next). The completion mechanism might use the interpreter’s symbol table. A command to check (or even suggest) matching parentheses, quotes, etc., would also be useful.

One alternative enhanced interactive interpreter that has been around for quite some time is [IPython](https://ipython.org/), which features tab completion, object exploration and advanced history management. It can also be thoroughly customized and embedded into other applications. Another similar enhanced interactive environment is [bpython](https://bpython-interpreter.org/).
> Python 原生解释器的替代有 IPython，它包含了 tab 补全，对象探索和历史管理等，以及 `bpython`

# 15. Floating-Point Arithmetic: Issues and Limitations
Floating-point numbers are represented in computer hardware as base 2 (binary) fractions. For example, the **decimal** fraction `0.625` has value 6/10 + 2/100 + 5/1000, and in the same way the **binary** fraction `0.101` has value 1/2 + 0/4 + 1/8. These two fractions have identical values, the only real difference being that the first is written in base 10 fractional notation, and the second in base 2.

Unfortunately, most decimal fractions cannot be represented exactly as binary fractions. A consequence is that, in general, the decimal floating-point numbers you enter are only approximated by the binary floating-point numbers actually stored in the machine.
> 大多数十进制小数实际上不能被二进制准确表示

The problem is easier to understand at first in base 10. Consider the fraction 1/3. You can approximate that as a base 10 fraction:

```
0.3
```

or, better,

```
0.33
```

or, better,

```
0.333
```

and so on. No matter how many digits you’re willing to write down, the result will never be exactly 1/3, but will be an increasingly better approximation of 1/3.

In the same way, no matter how many base 2 digits you’re willing to use, the decimal value 0.1 cannot be represented exactly as a base 2 fraction. In base 2, 1/10 is the infinitely repeating fraction

```
0.0001100110011001100110011001100110011001100110011...
```

Stop at any finite number of bits, and you get an approximation. On most machines today, floats are approximated using a binary fraction with the numerator using the first 53 bits starting with the most significant bit and with the denominator as a power of two. In the case of 1/10, the binary fraction is `3602879701896397 / 2 ** 55` which is close to but not exactly equal to the true value of 1/10.

Many users are not aware of the approximation because of the way values are displayed. Python only prints a decimal approximation to the true decimal value of the binary approximation stored by the machine. On most machines, if Python were to print the true decimal value of the binary approximation stored for 0.1, it would have to display:

```
>>> 0.1
0.1000000000000000055511151231257827021181583404541015625
```

That is more digits than most people find useful, so Python keeps the number of digits manageable by displaying a rounded value instead:

```
>>> 1 / 10
0.1
```

Just remember, even though the printed result looks like the exact value of 1/10, the actual stored value is the nearest representable binary fraction.
> Python 会展示归约之后的浮点值，当然实际存储的浮点值是原始浮点值的最近的二进制近似

Interestingly, there are many different decimal numbers that share the same nearest approximate binary fraction. For example, the numbers `0.1` and `0.10000000000000001` and `0.1000000000000000055511151231257827021181583404541015625` are all approximated by `3602879701896397 / 2 ** 55`. Since all of these decimal values share the same approximation, any one of them could be displayed while still preserving the invariant `eval(repr(x)) == x`.

Historically, the Python prompt and built-in [`repr()`](https://docs.python.org/3/library/functions.html#repr "repr") function would choose the one with 17 significant digits, `0.10000000000000001`. Starting with Python 3.1, Python (on most systems) is now able to choose the shortest of these and simply display `0.1`.

Note that this is in the very nature of binary floating point: this is not a bug in Python, and it is not a bug in your code either. You’ll see the same kind of thing in all languages that support your hardware’s floating-point arithmetic (although some languages may not _display_ the difference by default, or in all output modes).

For more pleasant output, you may wish to use string formatting to produce a limited number of significant digits:

```
>>> format(math.pi, '.12g')  # give 12 significant digits
'3.14159265359'

>>> format(math.pi, '.2f')   # give 2 digits after the point
'3.14'

>>> repr(math.pi)
'3.141592653589793'
```

It’s important to realize that this is, in a real sense, an illusion: you’re simply rounding the _display_ of the true machine value.

One illusion may beget another. For example, since 0.1 is not exactly 1/10, summing three values of 0.1 may not yield exactly 0.3, either:

```
>>> 0.1 + 0.1 + 0.1 == 0.3
False
```

Also, since the 0.1 cannot get any closer to the exact value of 1/10 and 0.3 cannot get any closer to the exact value of 3/10, then pre-rounding with [`round()`](https://docs.python.org/3/library/functions.html#round "round") function cannot help:

```
>>> round(0.1, 1) + round(0.1, 1) + round(0.1, 1) == round(0.3, 1)
False
```

Though the numbers cannot be made closer to their intended exact values, the [`math.isclose()`](https://docs.python.org/3/library/math.html#math.isclose "math.isclose") function can be useful for comparing inexact values:

```
>>> math.isclose(0.1 + 0.1 + 0.1, 0.3)
True
```

Alternatively, the [`round()`](https://docs.python.org/3/library/functions.html#round "round") function can be used to compare rough approximations:

```
>>> round(math.pi, ndigits=2) == round(22 / 7, ndigits=2)
True
```

Binary floating-point arithmetic holds many surprises like this. The problem with “0.1” is explained in precise detail below, in the “Representation Error” section. See [Examples of Floating Point Problems](https://jvns.ca/blog/2023/01/13/examples-of-floating-point-problems/) for a pleasant summary of how binary floating point works and the kinds of problems commonly encountered in practice. Also see [The Perils of Floating Point](http://www.indowsway.com/floatingpoint.htm) for a more complete account of other common surprises.

As that says near the end, “there are no easy answers.” Still, don’t be unduly wary of floating point! The errors in Python float operations are inherited from the floating-point hardware, and on most machines are on the order of no more than 1 part in `2**53` per operation. That’s more than adequate for most tasks, but you do need to keep in mind that it’s not decimal arithmetic and that every float operation can suffer a new rounding error.

While pathological cases do exist, for most casual use of floating-point arithmetic you’ll see the result you expect in the end if you simply round the display of your final results to the number of decimal digits you expect. [`str()`](https://docs.python.org/3/library/stdtypes.html#str "str") usually suffices, and for finer control see the [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format "str.format") method’s format specifiers in [Format String Syntax](https://docs.python.org/3/library/string.html#formatstrings).

For use cases which require exact decimal representation, try using the [`decimal`](https://docs.python.org/3/library/decimal.html#module-decimal "decimal: Implementation of the General Decimal Arithmetic  Specification.") module which implements decimal arithmetic suitable for accounting applications and high-precision applications.

Another form of exact arithmetic is supported by the [`fractions`](https://docs.python.org/3/library/fractions.html#module-fractions "fractions: Rational numbers.") module which implements arithmetic based on rational numbers (so the numbers like 1/3 can be represented exactly).

If you are a heavy user of floating-point operations you should take a look at the NumPy package and many other packages for mathematical and statistical operations supplied by the SciPy project. See <[https://scipy.org](https://scipy.org/)>.

Python provides tools that may help on those rare occasions when you really _do_ want to know the exact value of a float. The [`float.as_integer_ratio()`](https://docs.python.org/3/library/stdtypes.html#float.as_integer_ratio "float.as_integer_ratio") method expresses the value of a float as a fraction:

```
>>> x = 3.14159
>>> x.as_integer_ratio()
(3537115888337719, 1125899906842624)
```

Since the ratio is exact, it can be used to losslessly recreate the original value:

```
>>> x == 3537115888337719 / 1125899906842624
True
```

The [`float.hex()`](https://docs.python.org/3/library/stdtypes.html#float.hex "float.hex") method expresses a float in hexadecimal (base 16), again giving the exact value stored by your computer:

```
>>> x.hex()
'0x1.921f9f01b866ep+1'
```

This precise hexadecimal representation can be used to reconstruct the float value exactly:

```
>>> x == float.fromhex('0x1.921f9f01b866ep+1')
True
```

Since the representation is exact, it is useful for reliably porting values across different versions of Python (platform independence) and exchanging data with other languages that support the same format (such as Java and C99).

Another helpful tool is the [`sum()`](https://docs.python.org/3/library/functions.html#sum "sum") function which helps mitigate loss-of-precision during summation. It uses extended precision for intermediate rounding steps as values are added onto a running total. That can make a difference in overall accuracy so that the errors do not accumulate to the point where they affect the final total:
> `sum()` 函数可以用于缓解求和时的精度损失，该函数延伸了中间结果的精度

```
>>> 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 == 1.0
False
>>> sum([0.1] * 10) == 1.0
True
```

The [`math.fsum()`](https://docs.python.org/3/library/math.html#math.fsum "math.fsum") goes further and tracks all of the “lost digits” as values are added onto a running total so that the result has only a single rounding. This is slower than [`sum()`](https://docs.python.org/3/library/functions.html#sum "sum") but will be more accurate in uncommon cases where large magnitude inputs mostly cancel each other out leaving a final sum near zero:
> `math.fsum()` 追踪在相加时所有遗失的数位，然后加到最后结果，因此整个求和结果只会经过一次归约
> 它比 `sum()` 慢，但是更精确

```
>>> arr = [-0.10430216751806065, -266310978.67179024, 143401161448607.16,
...        -143401161400469.7, 266262841.31058735, -0.003244936839808227]
>>> float(sum(map(Fraction, arr)))   # Exact summation with single rounding
8.042173697819788e-13
>>> math.fsum(arr)                   # Single rounding
8.042173697819788e-13
>>> sum(arr)                         # Multiple roundings in extended precision
8.042178034628478e-13
>>> total = 0.0
>>> for x in arr:
...     total += x                   # Multiple roundings in standard precision
...
>>> total                            # Straight addition has no correct digits!
-0.0051575902860057365
```

## 15.1. Representation Error
This section explains the “0.1” example in detail, and shows how you can perform an exact analysis of cases like this yourself. Basic familiarity with binary floating-point representation is assumed.

_Representation error_ refers to the fact that some (most, actually) decimal fractions cannot be represented exactly as binary (base 2) fractions. This is the chief reason why Python (or Perl, C, C++, Java, Fortran, and many others) often won’t display the exact decimal number you expect.
> 表示误差

Why is that? 1/10 is not exactly representable as a binary fraction. Since at least 2000, almost all machines use IEEE 754 binary floating-point arithmetic, and almost all platforms map Python floats to IEEE 754 binary64 “double precision” values. IEEE 754 binary64 values contain 53 bits of precision, so on input the computer strives to convert 0.1 to the closest fraction it can of the form _J_/2**_N_ where _J_ is an integer containing exactly 53 bits. Rewriting

```
1 / 10 ~= J / (2**N)
```

as

```
J ~= 2**N / 10
```

and recalling that _J_ has exactly 53 bits (is `>= 2**52` but `< 2**53`), the best value for _N_ is 56:

```
>>> 2**52 <=  2**56 // 10  < 2**53
True
```

That is, 56 is the only value for _N_ that leaves _J_ with exactly 53 bits. The best possible value for _J_ is then that quotient rounded:

```
>>> q, r = divmod(2**56, 10)
>>> r
6
```

Since the remainder is more than half of 10, the best approximation is obtained by rounding up:

```
>>> q+1
7205759403792794
```

Therefore the best possible approximation to 1/10 in IEEE 754 double precision is:

```
7205759403792794 / 2 ** 56
```

Dividing both the numerator and denominator by two reduces the fraction to:

```
3602879701896397 / 2 ** 55
```

Note that since we rounded up, this is actually a little bit larger than 1/10; if we had not rounded up, the quotient would have been a little bit smaller than 1/10. But in no case can it be _exactly_ 1/10!

So the computer never “sees” 1/10: what it sees is the exact fraction given above, the best IEEE 754 double approximation it can get:

```
>>> 0.1 * 2 ** 55
3602879701896397.0
```

If we multiply that fraction by `10**55`, we can see the value out to 55 decimal digits:

```
>>> 3602879701896397 * 10 ** 55 // 2 ** 55
1000000000000000055511151231257827021181583404541015625
```

meaning that the exact number stored in the computer is equal to the decimal value 0.1000000000000000055511151231257827021181583404541015625. Instead of displaying the full decimal value, many languages (including older versions of Python), round the result to 17 significant digits:

```
>>> format(0.1, '.17f')
'0.10000000000000001'
```

The [`fractions`](https://docs.python.org/3/library/fractions.html#module-fractions "fractions: Rational numbers.") and [`decimal`](https://docs.python.org/3/library/decimal.html#module-decimal "decimal: Implementation of the General Decimal Arithmetic  Specification.") modules make these calculations easy:

```
>>> from decimal import Decimal
>>> from fractions import Fraction

>>> Fraction.from_float(0.1)
Fraction(3602879701896397, 36028797018963968)

>>> (0.1).as_integer_ratio()
(3602879701896397, 36028797018963968)

>>> Decimal.from_float(0.1)
Decimal('0.1000000000000000055511151231257827021181583404541015625')

>>> format(Decimal.from_float(0.1), '.17')
'0.10000000000000001'
```

# 16. Appendix
## 16.1. Interactive Mode
### 16.1.1. Error Handling
When an error occurs, the interpreter prints an error message and a stack trace. In interactive mode, it then returns to the primary prompt; when input came from a file, it exits with a nonzero exit status after printing the stack trace. (Exceptions handled by an [`except`](https://docs.python.org/3/reference/compound_stmts.html#except) clause in a [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) statement are not errors in this context.) Some errors are unconditionally fatal and cause an exit with a nonzero exit status; this applies to internal inconsistencies and some cases of running out of memory. All error messages are written to the standard error stream; normal output from executed commands is written to standard output.
> 一些 error 是无条件 fatal，导致直接以非零退出条件退出，一般这些 error 会导致内部不一致或者内存越界
> 所有的错误信息都会被写入 stderr 流
> 被执行命令的正常输出一般写入 stdout 流

Typing the interrupt character (usually Control-C or Delete) to the primary or secondary prompt cancels the input and returns to the primary prompt. [1](https://docs.python.org/3/tutorial/appendix.html#id2) Typing an interrupt while a command is executing raises the [`KeyboardInterrupt`](https://docs.python.org/3/library/exceptions.html#KeyboardInterrupt "KeyboardInterrupt") exception, which may be handled by a [`try`](https://docs.python.org/3/reference/compound_stmts.html#try) statement.
> 命令行中断会抛出一个 `KeyboardInterrupt` 异常，该异常其实可以用 `try` 语句处理

### 16.1.2. Executable Python Scripts
On BSD’ish Unix systems, Python scripts can be made directly executable, like shell scripts, by putting the line
> 在 BSD 的 Unix 系统中，可以让 Python 脚本之间像 shell 脚本一样可执行

```
#!/usr/bin/env python3
```

(assuming that the interpreter is on the user’s `PATH`) at the beginning of the script and giving the file an executable mode. The `#!` must be the first two characters of the file. On some platforms, this first line must end with a Unix-style line ending (`'\n'`), not a Windows (`'\r\n'`) line ending. Note that the hash, or pound, character, `'#'`, is used to start a comment in Python.
> 将该行放在脚本顶端，且让文件权限可执行即可
> `#!` 必须是前两个字符

The script can be given an executable mode, or permission, using the **chmod** command.

```
$ chmod +x myscript.py
```

On Windows systems, there is no notion of an “executable mode”. The Python installer automatically associates `.py` files with `python.exe` so that a double-click on a Python file will run it as a script. The extension can also be `.pyw`, in that case, the console window that normally appears is suppressed.
> Windows 中，Python 安装程序会自动将 `.py` 文件和 `python.exe` 关联，使得双击可以直接运行该脚本
> 文件拓展也可以是 `.pyw` ，此时双击运行不会出现控制台窗口

### 16.1.3. The Interactive Startup File
When you use Python interactively, it is frequently handy to have some standard commands executed every time the interpreter is started. You can do this by setting an environment variable named [`PYTHONSTARTUP`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONSTARTUP) to the name of a file containing your start-up commands. This is similar to the `.profile` feature of the Unix shells.
> 可以设置 `PYTHONSTARTUP` 指向一个脚本，指定解释器每次启动需要执行的命令
> 这和 Unix shell 的 `.profile` 文件类似

This file is only read in interactive sessions, not when Python reads commands from a script, and not when `/dev/tty` is given as the explicit source of commands (which otherwise behaves like an interactive session). It is executed in the same namespace where interactive commands are executed, so that objects that it defines or imports can be used without qualification in the interactive session. You can also change the prompts `sys.ps1` and `sys.ps2` in this file.
> 该文件只会在交互式解释会话被阅读，Python 执行脚本时不会，同时给定 `dev/tty` 作为显式的命令来源时，也不会阅读

If you want to read an additional start-up file from the current directory, you can program this in the global start-up file using code like `if os.path.isfile('.pythonrc.py'): exec(open('.pythonrc.py').read())`. If you want to use the startup file in a script, you must do this explicitly in the script:

```python
import os
filename = os.environ.get('PYTHONSTARTUP')
if filename and os.path.isfile(filename):
    with open(filename) as fobj:
        startup_file = fobj.read()
    exec(startup_file)
```

### 16.1.4. The Customization Modules
Python provides two hooks to let you customize it: sitecustomize and usercustomize. To see how it works, you need first to find the location of your user site-packages directory. Start Python and run this code:
> Python 提供了两个可以自定义的钩子: sitecustomize, usercustomize
> 我们首先找到用户的 site-packages 目录

```
>>> import site
>>> site.getusersitepackages()
'/home/user/.local/lib/python3.x/site-packages'
```

Now you can create a file named `usercustomize.py` in that directory and put anything you want in it. It will affect every invocation of Python, unless it is started with the [`-s`](https://docs.python.org/3/using/cmdline.html#cmdoption-s) option to disable the automatic import.
> 在该目录中创建名为 `usercustomize.py` 的文件，然后在其中写脚本，这会影响 Python 的调用

sitecustomize works in the same way, but is typically created by an administrator of the computer in the global site-packages directory, and is imported before usercustomize. See the documentation of the [`site`](https://docs.python.org/3/library/site.html#module-site "site: Module responsible for site-specific configuration.") module for more details.
> sitecustomize 一般由管理员创建在全局的 site-packages 目录下，它在 usercustomize 之前被 import

Footnotes
[1](https://docs.python.org/3/tutorial/appendix.html#id1) A problem with the GNU Readline package may prevent this.