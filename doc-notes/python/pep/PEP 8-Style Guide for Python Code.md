---
completed: true
---
# Introduction
This document gives coding conventions for the Python code comprising the standard library in the main Python distribution. Please see the companion informational PEP describing [style guidelines for the C code in the C implementation of Python](https://peps.python.org/pep-0007/ "PEP 7 – Style Guide for C Code").

This document and [PEP 257](https://peps.python.org/pep-0257/ "PEP 257 – Docstring Conventions") (Docstring Conventions) were adapted from Guido’s original Python Style Guide essay, with some additions from Barry’s style guide [[2]](https://peps.python.org/pep-0008/#id6).

This style guide evolves over time as additional conventions are identified and past conventions are rendered obsolete by changes in the language itself.

Many projects have their own coding style guidelines. In the event of any conflicts, such project-specific guides take precedence for that project.

# A Foolish Consistency is the Hobgoblin of Little Minds
One of Guido’s key insights is that code is read much more often than it is written. The guidelines provided here are intended to improve the readability of code and make it consistent across the wide spectrum of Python code. As [PEP 20](https://peps.python.org/pep-0020/ "PEP 20 – The Zen of Python") says, “Readability counts”.
> PEP8 旨在提高 Python 代码的可读性

A style guide is about consistency. Consistency with this style guide is important. Consistency within a project is more important. Consistency within one module or function is the most important.

However, know when to be inconsistent – sometimes style guide recommendations just aren’t applicable. When in doubt, use your best judgment. Look at other examples and decide what looks best. And don’t hesitate to ask!

In particular: do not break backwards compatibility just to comply with this PEP!
> 不要为了遵循 PEP 而破坏向后兼容性

Some other good reasons to ignore a particular guideline:

1. When applying the guideline would make the code less readable, even for someone who is used to reading code that follows this PEP.
2. To be consistent with surrounding code that also breaks it (maybe for historic reasons) – although this is also an opportunity to clean up someone else’s mess (in true XP style).
3. Because the code in question predates the introduction of the guideline and there is no other reason to be modifying that code.
4. When the code needs to remain compatible with older versions of Python that don’t support the feature recommended by the style guide.

# Code Lay-out
## Indentation
Use 4 spaces per indentation level.
> 4空格缩进

Continuation lines should align wrapped elements either vertically using Python’s implicit line joining inside parentheses, brackets and braces, or using a _hanging indent_ [[1]] (https://peps.python.org/pep-0008/#fn-hi). When using a hanging indent the following should be considered; there should be no arguments on the first line and further indentation should be used to clearly distinguish itself as a continuation line:
>续行应该通过以下方式对齐换行元素：
>要么使用Python的括号、方括号和花括号内的隐式行连接，使它们垂直对齐；要么使用“悬挂缩进”
>当使用悬挂缩进时，应考虑：第一行不应有任何参数，进一步的缩进应被用来明确区分续行

```python
# Correct:

# Aligned with opening delimiter.
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# Add 4 spaces (an extra level of indentation) to distinguish arguments from the rest.
def long_function_name(
        var_one, var_two, var_three,
        var_four):
    print(var_one)

# Hanging indents should add a level.
foo = long_function_name(
    var_one, var_two,
    var_three, var_four)
```

```python
# Wrong:

# Arguments on first line forbidden when not using vertical alignment.
foo = long_function_name(var_one, var_two,
    var_three, var_four)

# Further indentation required as indentation is not distinguishable.
def long_function_name(
    var_one, var_two, var_three,
    var_four):
    print(var_one)
```

The 4-space rule is optional for continuation lines.
> 续行的悬挂缩进不一定要 4 空格缩进

Optional:

```python
# Hanging indents *may* be indented to other than 4 spaces.
foo = long_function_name(
  var_one, var_two,
  var_three, var_four)
```

When the conditional part of an `if` -statement is long enough to require that it be written across multiple lines, it’s worth noting that the combination of a two character keyword (i.e. `if`), plus a single space, plus an opening parenthesis creates a natural 4-space indent for the subsequent lines of the multiline conditional. This can produce a visual conflict with the indented suite of code nested inside the `if` -statement, which would also naturally be indented to 4 spaces. This PEP takes no explicit position on how (or whether) to further visually distinguish such conditional lines from the nested suite inside the `if` -statement. Acceptable options in this situation include, but are not limited to:
>当 `if` 语句的条件部分足够长，需要多行书写时，值得注意的是，两个字符的关键字（即 `if`），加上一个空格，再加上一个左括号，自然形成了后续多行条件的4个空格缩进，这可能会与嵌套在 `if` 语句内部的代码块的缩进产生视觉冲突，因为后者也会自然缩进4个空格
>本PEP并未明确说明如何（或是否）进一步在视觉上区分这些条件行和 `if` 语句内部嵌套的代码块
>在这种情况下，可接受的选项包括但不限于：

```python
# No extra indentation.
if (this_is_one_thing and
    that_is_another_thing):
    do_something()

# Add a comment, which will provide some distinction in editors
# supporting syntax highlighting.
if (this_is_one_thing and
    that_is_another_thing):
    # Since both conditions are true, we can frobnicate.
    do_something()

# Add some extra indentation on the conditional continuation line.
if (this_is_one_thing
        and that_is_another_thing):
    do_something()
```

(Also see the discussion of whether to break before or after binary operators below.)

The closing brace/bracket/parenthesis on multiline constructs may either line up under the first non-whitespace character of the last line of list, as in:
>多行结构的闭合大括号/方括号/圆括号可以选择与列表的最后一行的第一个非空白字符对齐，如下所示：

```python
my_list = [
    1, 2, 3,
    4, 5, 6,
    ]
result = some_function_that_takes_arguments(
    'a', 'b', 'c',
    'd', 'e', 'f',
    )
```

or it may be lined up under the first character of the line that starts the multiline construct, as in:
>或者它可以与多行结构开始行的第一个字符对齐，如下所示：

```python
my_list = [
    1, 2, 3,
    4, 5, 6,
]
result = some_function_that_takes_arguments(
    'a', 'b', 'c',
    'd', 'e', 'f',
)
```

## Tabs or Spaces?
Spaces are the preferred indentation method.

Tabs should be used solely to remain consistent with code that is already indented with tabs.

Python disallows mixing tabs and spaces for indentation.
> Space > Tab，禁止混合使用

## Maximum Line Length
Limit all lines to a maximum of 79 characters.
>将所有行的长度限制为最多79个字符

For flowing long blocks of text with fewer structural restrictions (docstrings or comments), the line length should be limited to 72 characters.
>对于具有较少结构限制的长文本块（如文档字符串或注释），行长度应限制为72个字符

Limiting the required editor window width makes it possible to have several files open side by side, and works well when using code review tools that present the two versions in adjacent columns.

The default wrapping in most tools disrupts the visual structure of the code, making it more difficult to understand. The limits are chosen to avoid wrapping in editors with the window width set to 80, even if the tool places a marker glyph in the final column when wrapping lines. Some web based tools may not offer dynamic line wrapping at all.

Some teams strongly prefer a longer line length. For code maintained exclusively or primarily by a team that can reach agreement on this issue, it is okay to increase the line length limit up to 99 characters, provided that comments and docstrings are still wrapped at 72 characters.
>有些团队强烈偏好较长的行长度。对于仅由能够在这个问题上达成一致的团队维护的代码，可以将行长度限制增加到99个字符，前提是注释和文档字符串仍然限制在72个字符内

The Python standard library is conservative and requires limiting lines to 79 characters (and docstrings/comments to 72).
>Python标准库则较为保守，要求限制行长度为79个字符（文档字符串/注释为72个字符）

The preferred way of wrapping long lines is by using Python’s implied line continuation inside parentheses, brackets and braces. Long lines can be broken over multiple lines by wrapping expressions in parentheses. These should be used in preference to using a backslash for line continuation.
> 偏好使用括号进行隐式行连接而不是使用反斜杠进行行连接

Backslashes may still be appropriate at times. For example, long, multiple `with`-statements could not use implicit continuation before Python 3.10, so backslashes were acceptable for that case:

```python
with open('/path/to/some/file/you/want/to/read') as file_1, \
     open('/path/to/some/file/being/written', 'w') as file_2:
    file_2.write(file_1.read())
```

(See the previous discussion on [multiline if-statements](https://peps.python.org/pep-0008/#multiline-if-statements) for further thoughts on the indentation of such multiline `with`-statements.)

Another such case is with `assert` statements.

Make sure to indent the continued line appropriately.

## Should a Line Break Before or After a Binary Operator?
For decades the recommended style was to break after binary operators. But this can hurt readability in two ways: the operators tend to get scattered across different columns on the screen, and each operator is moved away from its operand and onto the previous line. Here, the eye has to do extra work to tell which items are added and which are subtracted:

```python
# Wrong:
# operators sit far away from their operands
income = (gross_wages +
          taxable_interest +
          (dividends - qualified_dividends) -
          ira_deduction -
          student_loan_interest)
```

To solve this readability problem, mathematicians and their publishers follow the opposite convention. Donald Knuth explains the traditional rule in his _Computers and Typesetting_ series: “Although formulas within a paragraph always break after binary operations and relations, displayed formulas always break before binary operations” [[3]](https://peps.python.org/pep-0008/#id7).

Following the tradition from mathematics usually results in more readable code:

```python
# Correct:
# easy to match operators with operands
income = (gross_wages
          + taxable_interest
          + (dividends - qualified_dividends)
          - ira_deduction
          - student_loan_interest)
```

In Python code, it is permissible to break before or after a binary operator, as long as the convention is consistent locally. For new code Knuth’s style is suggested.
> 推荐在运算符之前换行

## Blank Lines
Surround top-level function and class definitions with two blank lines.
> 顶级函数和类定义由两个空行包围

Method definitions inside a class are surrounded by a single blank line.
> 类内方法定义由一个空行包围

Extra blank lines may be used (sparingly) to separate groups of related functions. Blank lines may be omitted between a bunch of related one-liners (e.g. a set of dummy implementations).
>可以在适当的情况下使用额外的空行来分隔相关的函数组。在一组相关的单行代码（例如一组 dummy 实现）之间可以省略空行

Use blank lines in functions, sparingly, to indicate logical sections.
>在函数内部，适当地使用空行来表示逻辑上的段落

Python accepts the control-L (i.e. ^L) form feed character as whitespace; many tools treat these characters as page separators, so you may use them to separate pages of related sections of your file. Note, some editors and web-based code viewers may not recognize control-L as a form feed and will show another glyph in its place.

## Source File Encoding
Code in the core Python distribution should always use UTF-8, and should not have an encoding declaration.

In the standard library, non-UTF-8 encodings should be used only for test purposes. Use non-ASCII characters sparingly, preferably only to denote places and human names. If using non-ASCII characters as data, avoid noisy Unicode characters like z̯̯͡a̧͎̺l̡͓̫g̹̲o̡̼̘ and byte order marks.

All identifiers in the Python standard library MUST use ASCII-only identifiers, and SHOULD use English words wherever feasible (in many cases, abbreviations and technical terms are used which aren’t English).

Open source projects with a global audience are encouraged to adopt a similar policy.

## Imports
- Imports should usually be on separate lines:

    ```python
    # Correct:
    import os
    import sys
    
    # Wrong:
    import sys, os
    ```
    
    It’s okay to say this though:

    ```python
    # Correct:
    from subprocess import Popen, PIPE
    ```
    
- Imports are always put at the top of the file, just after any module comments and docstrings, and before module globals and constants.
    import 在模块 docstring 和注释后面，在模块全局变量和常量前面
    
    Imports should be grouped in the following order:
    Import 顺序按照组来组织：
    
    1. Standard library imports. 标准库
    2. Related third party imports. 相关第三方库
    3. Local application/library specific imports. 本地应用、库
    
    You should put a blank line between each group of imports.
    组间用空行隔开
    
- Absolute imports are recommended, as they are usually more readable and tend to be better behaved (or at least give better error messages) if the import system is incorrectly configured (such as when a directory inside a package ends up on `sys.path`):
    推荐绝对 import
    
    ```python
    import mypkg.sibling
    from mypkg import sibling
    from mypkg.sibling import example
    ```
    
    However, explicit relative imports are an acceptable alternative to absolute imports, especially when dealing with complex package layouts where using absolute imports would be unnecessarily verbose:
    包组织较复杂时可以接受相对 import
    
    ```python
    from . import sibling
    from .sibling import example
    ```
    
    Standard library code should avoid complex package layouts and always use absolute imports.
    标准库应避免复杂组织，使用绝对 import
    
- When importing a class from a class-containing module, it’s usually okay to spell this:
    
    ```python
    from myclass import MyClass
    from foo.bar.yourclass import YourClass
    ```
    
    If this spelling causes local name clashes, then spell them explicitly:
    
    ```python
    import myclass
    import foo.bar.yourclass
    ```
    
    and use `myclass.MyClass` and `foo.bar.yourclass.YourClass`.
    
- Wildcard imports (`from <module> import *`) should be avoided, as they make it unclear which names are present in the namespace, confusing both readers and many automated tools. There is one defensible use case for a wildcard import, which is to republish an internal interface as part of a public API (for example, overwriting a pure Python implementation of an interface with the definitions from an optional accelerator module and exactly which definitions will be overwritten isn’t known in advance).
    
    When republishing names this way, the guidelines below regarding public and internal interfaces still apply.
    

## Module Level Dunder Names
Module level “dunders” (i.e. names with two leading and two trailing underscores) such as `__all__`, `__author__`, `__version__`, etc. should be placed after the module docstring but before any import statements _except_ `from __future__` imports. Python mandates that future-imports must appear in the module before any other code except docstrings:
> 模块级别的 dunder 应该在模块 docstring 之后，在 import 语句之前，除了 `from __future__` 的 import，`from __future__` 的 import 必须出现在 docstring 之后的第一个位置

```python
"""This is the example module.

This module does stuff.
"""

from __future__ import barry_as_FLUFL

__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Cardinal Biggles'

import os
import sys
```

# String Quotes
In Python, single-quoted strings and double-quoted strings are the same. This PEP does not make a recommendation for this. Pick a rule and stick to it. When a string contains single or double quote characters, however, use the other one to avoid backslashes in the string. It improves readability.

For triple-quoted strings, always use double quote characters to be consistent with the docstring convention in [PEP 257](https://peps.python.org/pep-0257/ "PEP 257 – Docstring Conventions").
> 三引号时使用 `"` ，保持和 PEP 257 docstring 规范一致

# Whitespace in Expressions and Statements
## Pet Peeves
Avoid extraneous whitespace in the following situations:
> 以下情况下避免额外的空格：

- Immediately inside parentheses, brackets or braces:
    
    ```python
    # Correct:
    spam(ham[1], {eggs: 2})
    
    # Wrong:
    spam( ham[ 1 ], { eggs: 2 } )
    ```
    
- Between a trailing comma and a following close parenthesis:
    
    ```python
    # Correct:
    foo = (0,)
    
    # Wrong:
    bar = (0, )
    ```
    
- Immediately before a comma, semicolon, or colon:
    冒号、分号、逗号之前不要有空格
    ```python
    # Correct:
    if x == 4: print(x, y); x, y = y, x
    
    # Wrong:
    if x == 4 : print(x , y) ; x , y = y , x
    ```
    
- However, in a slice the colon acts like a binary operator, and should have equal amounts on either side (treating it as the operator with the lowest priority). In an extended slice, both colons must have the same amount of spacing applied. Exception: when a slice parameter is omitted, the space is omitted:
    然而，在切片中，冒号的作用像一个二元运算符，应该在两边有相等的数量的空格（将其视为优先级最低的运算符）
    在扩展切片中，两个冒号必须应用相同数量的间距
    例外情况：当省略切片参数时，空格也会被省略
    
    ```python
    # Correct:
    ham[1:9], ham[1:9:3], ham[:9:3], ham[1::3], ham[1:9:]
    ham[lower:upper], ham[lower:upper:], ham[lower::step]
    ham[lower+offset : upper+offset]
    ham[: upper_fn(x) : step_fn(x)], ham[:: step_fn(x)]
    ham[lower + offset : upper + offset]
    
    # Wrong:
    ham[lower + offset:upper + offset]
    ham[1: 9], ham[1 :9], ham[1:9 :3]
    ham[lower : : step]
    ham[ : upper]
    ```
    
- Immediately before the open parenthesis that starts the argument list of a function call:
    
    ```python
    # Correct:
    spam(1)
    
    # Wrong:
    spam (1)
    ```
    
- Immediately before the open parenthesis that starts an indexing or slicing:
    
    ```python
    # Correct:
    dct['key'] = lst[index]
    
    # Wrong:
    dct ['key'] = lst [index]
    ```
    
- More than one space around an assignment (or other) operator to align it with another:
    
    ```python
    # Correct:
    x = 1
    y = 2
    long_variable = 3
    
    # Wrong:
    x             = 1
    y             = 2
    long_variable = 3
    ```

## Other Recommendations
- Avoid trailing whitespace anywhere. Because it’s usually invisible, it can be confusing: e.g. a backslash followed by a space and a newline does not count as a line continuation marker. Some editors don’t preserve it and many projects (like CPython itself) have pre-commit hooks that reject it.
>避免在任何地方使用尾随空白字符。因为这些字符通常是看不见的，所以可能会引起混淆：例如，反斜杠后面跟着空格和换行符并不被视为行续行标记。一些编辑器不会保留这些字符，而且许多项目（如CPython本身）都有预提交钩子来拒绝这些字符
- Always surround these binary operators with a single space on either side: assignment (`=`), augmented assignment (`+=`, `-=` etc.), comparisons (`==`, `<`, `>`, `!=`, `<>`, `<=`, `>=`, `in`, `not in`, `is`, `is not`), Booleans (`and`, `or`, `not`).
>总是在这些二元运算符的两侧各添加一个空格：赋值运算符（`=`）、增强赋值运算符（`+=`、`-=` 等）、比较运算符（`==`、`<`、`>`、`!=`、`<>`、`<=`、`>=`、`in`、`not in`、`is`、`is not`）、布尔运算符（`and`、`or`、`not`）
- If operators with different priorities are used, consider adding whitespace around the operators with the lowest priority(ies). Use your own judgment; however, never use more than one space, and always have the same amount of whitespace on both sides of a binary operator:
    为最低优先级的运算符两边各添加一个空格
    
    ```python
    # Correct:
    i = i + 1
    submitted += 1
    x = x*2 - 1
    hypot2 = x*x + y*y
    c = (a+b) * (a-b)
    
    # Wrong:
    i=i+1
    submitted +=1
    x = x * 2 - 1
    hypot2 = x * x + y * y
    c = (a + b) * (a - b)
    ```
    
- Function annotations should use the normal rules for colons and always have spaces around the `->` arrow if present. (See [Function Annotations](https://peps.python.org/pep-0008/#function-annotations) below for more about function annotations.):
    函数注释中，`->` 两边各一个空格
    ```python
    # Correct:
    def munge(input: AnyStr): ...
    def munge() -> PosInt: ...
    
    # Wrong:
    def munge(input:AnyStr): ...
    def munge()->PosInt: ...
    ```
    
- Don’t use spaces around the `=` sign when used to indicate a keyword argument, or when used to indicate a default value for an _unannotated_ function parameter:
    指定关键字参数/默认参数值时，`=` 两边不加空格  
    ```python
    # Correct:
    def complex(real, imag=0.0):
        return magic(r=real, i=imag)
    
    # Wrong:
    def complex(real, imag = 0.0):
        return magic(r = real, i = imag)
    ```
    
    When combining an argument annotation with a default value, however, do use spaces around the `=` sign:
    如果有参数注释时，则在默认值的 `=` 两边加空格 
    ```python
    # Correct:
    def munge(sep: AnyStr = None): ...
    def munge(input: AnyStr, sep: AnyStr = None, limit=1000): ...
    
    # Wrong:
    def munge(input: AnyStr=None): ...
    def munge(input: AnyStr, limit = 1000): ...
    ```
    
- Compound statements (multiple statements on the same line) are generally discouraged:
    
    ```python
    # Correct:
    if foo == 'blah':
        do_blah_thing()
    do_one()
    do_two()
    do_three()
    ```
    
    Rather not:
    
    ```python
    # Wrong:
    if foo == 'blah': do_blah_thing()
    do_one(); do_two(); do_three()
    ```
    
- While sometimes it’s okay to put an if/for/while with a small body on the same line, never do this for multi-clause statements. Also avoid folding such long lines!
    
    Rather not:
    
    ```python
    # Wrong:
    if foo == 'blah': do_blah_thing()
    for x in lst: total += x
    while t < 10: t = delay()
    ```
    
    Definitely not:
    
    ```python
    # Wrong:
    if foo == 'blah': do_blah_thing()
    else: do_non_blah_thing()
    
    try: something()
    finally: cleanup()
    
    do_one(); do_two(); do_three(long, argument,
                                 list, like, this)
    
    if foo == 'blah': one(); two(); three()
    ```

# When to Use Trailing Commas
Trailing commas are usually optional, except they are mandatory when making a tuple of one element. For clarity, it is recommended to surround the latter in (technically redundant) parentheses:
>尾随逗号通常是可选的，除了在创建只有一个元素的元组时，这时它们是强制性的
>为了清晰起见，建议用括号（从技术上讲是多余的）将这种情况括起来：

```python
# Correct:
FILES = ('setup.cfg',)

# Wrong:
FILES = 'setup.cfg',
```

When trailing commas are redundant, they are often helpful when a version control system is used, when a list of values, arguments or imported items is expected to be extended over time. The pattern is to put each value (etc.) on a line by itself, always adding a trailing comma, and add the close parenthesis/bracket/brace on the next line. However it does not make sense to have a trailing comma on the same line as the closing delimiter (except in the above case of singleton tuples):
>当尾随逗号是冗余的时候，如果使用了版本控制系统，并且预计一个值列表、参数列表或导入的项目在未来会被扩展，那么它们通常会有帮助
>这种做法是将每个值（等）单独放在一行，并始终添加尾随逗号，然后在下一行添加关闭括号/方括号/花括号
>然而，在关闭定界符的同一行上添加尾随逗号是没有意义的（除非在单例元组的上述情况下）

```python
# Correct:
FILES = [
    'setup.cfg',
    'tox.ini',
    ]
initialize(FILES,
           error=True,
           )

# Wrong:
FILES = ['setup.cfg', 'tox.ini',]
initialize(FILES, error=True,)
```

# Comments
Comments that contradict the code are worse than no comments. Always make a priority of keeping the comments up-to-date when the code changes!
>与代码相矛盾的注释比没有注释还要糟糕
>当代码更改时，始终优先保持注释是最新的

Comments should be complete sentences. The first word should be capitalized, unless it is an identifier that begins with a lower case letter (never alter the case of identifiers!).
> 注释需要是完整的句子，第一个字母大写，除非是用以小写字母开头的 identifier 作为句子第一个单词

Block comments generally consist of one or more paragraphs built out of complete sentences, with each sentence ending in a period.
> block comment  一般是成段的

You should use one or two spaces after a sentence-ending period in multi-sentence comments, except after the final sentence.
>在多句注释中，句末的句点后应使用一到两个空格，但最终句子之后不应使用

Ensure that your comments are clear and easily understandable to other speakers of the language you are writing in.

Python coders from non-English speaking countries: please write your comments in English, unless you are 120% sure that the code will never be read by people who don’t speak your language.

## Block Comments
Block comments generally apply to some (or all) code that follows them, and are indented to the same level as that code. Each line of a block comment starts with a `#` and a single space (unless it is indented text inside the comment).
>块注释通常适用于其后的某些（或全部）代码，并且缩进到与该代码相同的级别
>块注释的每一行以 `#` 和一个空格开始（除非注释中的文本已缩进）

Paragraphs inside a block comment are separated by a line containing a single `#`.
> block comment 以一个 `# ` 的空行分离段落

## Inline Comments
Use inline comments sparingly.
> 不要密集使用 inline comment

An inline comment is a comment on the same line as a statement. Inline comments should be separated by at least two spaces from the statement. They should start with a # and a single space.
> inline comment 和语句至少离两个空格

Inline comments are unnecessary and in fact distracting if they state the obvious. Don’t do this:

```python
x = x + 1                 # Increment x
```

But sometimes, this is useful:

```python
x = x + 1                 # Compensate for border
```

## Documentation Strings
Conventions for writing good documentation strings (a.k.a. “docstrings”) are immortalized in [PEP 257](https://peps.python.org/pep-0257/ "PEP 257 – Docstring Conventions").

- Write docstrings for all public modules, functions, classes, and methods. Docstrings are not necessary for non-public methods, but you should have a comment that describes what the method does. This comment should appear after the `def` line.
> 为所有的公有模块、函数、类、方法写 docstring
> 非公有方法则不是必须，但需要有注释描述其目的，注释在 `def` 下一行

- [PEP 257](https://peps.python.org/pep-0257/ "PEP 257 – Docstring Conventions") describes good docstring conventions. Note that most importantly, the `"""` that ends a multiline docstring should be on a line by itself:
    结束 docstring 的 `"""` 应该自成一行
    
    ```python
    """Return a foobang
    
    Optional plotz says to frobnicate the bizbaz first.
    """
    ```
    
- For one liner docstrings, please keep the closing `"""` on the same line:
    单行的 docstring 则保持 `"""` 在同一行
    ```python
    """Return an ex-parrot."""
    ```

# Naming Conventions
The naming conventions of Python’s library are a bit of a mess, so we’ll never get this completely consistent – nevertheless, here are the currently recommended naming standards. New modules and packages (including third party frameworks) should be written to these standards, but where an existing library has a different style, internal consistency is preferred.

## Overriding Principle
Names that are visible to the user as public parts of the API should follow conventions that reflect usage rather than implementation.
> 作为 API 的公有部分展现给用户的名字应该遵循反映其用法的命名，而不是反映其实现

## Descriptive: Naming Styles
There are a lot of different naming styles. It helps to be able to recognize what naming style is being used, independently from what they are used for.

The following naming styles are commonly distinguished:
>以下是一些常见的命名风格区分：

- `b` (single lowercase letter)
- `B` (single uppercase letter)
- `lowercase`
- `lower_case_with_underscores`
- `UPPERCASE`
- `UPPER_CASE_WITH_UNDERSCORES`
- `CapitalizedWords` (or CapWords, or CamelCase – so named because of the bumpy look of its letters [4](https://peps.python.org/pep-0008/#id8)). This is also sometimes known as StudlyCaps.
    
    Note: When using acronyms in CapWords, capitalize all the letters of the acronym. Thus HTTPServerError is better than HttpServerError.
    CamelCase 中的缩略词需要全大写，例如 `HTTPServerError` 而不是 `HttpServerError`
    
- `mixedCase` (differs from CapitalizedWords by initial lowercase character!)
- `Capitalized_Words_With_Underscores` (ugly!)

There’s also the style of using a short unique prefix to group related names together. This is not used much in Python, but it is mentioned for completeness. For example, the `os.stat()` function returns a tuple whose items traditionally have names like `st_mode`, `st_size`, `st_mtime` and so on. (This is done to emphasize the correspondence with the fields of the POSIX system call struct, which helps programmers familiar with that.)
>还有一种命名风格是使用短且唯一的前缀来将相关名称分组，这在 Python 中不常用，但为了完整性而提到它
>例如，`os.stat()` 函数返回一个元组，该元组的项传统上命名为 `st_mode`、`st_size`、`st_mtime` 等（这样做是为了强调与 POSIX 系统调用结构字段的对应关系，这有助于熟悉该结构的程序员）

The X11 library uses a leading X for all its public functions. In Python, this style is generally deemed unnecessary because attribute and method names are prefixed with an object, and function names are prefixed with a module name.
>X11 库在其所有公共函数名前都使用 `X` 前缀
>在 Python 中，由于属性和方法名已由对象前缀，函数名由模块名前缀，因此认为这种风格通常是不必要的

In addition, the following special forms using leading or trailing underscores are recognized (these can generally be combined with any case convention):
>此外，以下几种使用前导或后置下划线的特殊形式也被认可（这些通常可以与任何命名约定结合使用）

- `_single_leading_underscore`: weak “internal use” indicator. E.g. `from M import *` does not import objects whose names start with an underscore.
> `_single_leading_underscore` : 弱的“内部使用”指示符
> 例如，` from M import *` 不会导入名称以下划线开头的对象

- `single_trailing_underscore_`: used by convention to avoid conflicts with Python keyword, e.g. :
> `single_trailing_underscore_` 通常用来避免与 Python 关键字冲突，例如：

```python
tkinter.Toplevel(master, class_='ClassName')
```

- `__double_leading_underscore`: when naming a class attribute, invokes name mangling (inside class FooBar, `__boo` becomes `_FooBar__boo`; see below).
> `__double_leading_underscore` : 以这种方式命名类属性时，会触发名称改写 (例如在类 `FooBar` 中，`__boo` 变为 `_FooBar__boo`；参见下方说明)

- `__double_leading_and_trailing_underscore__`: “magic” objects or attributes that live in user-controlled namespaces. E.g. `__init__`, `__import__` or `__file__`. Never invent such names; only use them as documented.
> `__double_leading_and_trailing_underscore__`: 用于命名存在于用户控制的命名空间中的“魔法”对象或属性
> 例如，`__init__`、`__import__` 或 `__file__`
> 不要发明这样的名字，只按文档要求使用它们

## Prescriptive: Naming Conventions
### Names to Avoid
Never use the characters ‘l’ (lowercase letter el), ‘O’ (uppercase letter oh), or ‘I’ (uppercase letter eye) as single character variable names.

In some fonts, these characters are indistinguishable from the numerals one and zero. When tempted to use ‘l’, use ‘L’ instead.
> 永远不要使用 `l` ，`O` ，`I` 作为单字符变量名，因为在一些字体中 `l/I` 和 `O/0` 会难以区分

### ASCII Compatibility
Identifiers used in the standard library must be ASCII compatible as described in the [policy section](https://peps.python.org/pep-3131/#policy-specification "PEP 3131 – Supporting Non-ASCII Identifiers § Policy Specification") of [PEP 3131](https://peps.python.org/pep-3131/ "PEP 3131 – Supporting Non-ASCII Identifiers").
> 标准库 identifier 必须使用 ASCII 字符

### Package and Module Names
Modules should have short, all-lowercase names. Underscores can be used in the module name if it improves readability. Python packages should also have short, all-lowercase names, although the use of underscores is discouraged.
>模块名应该有简短且全小写，如果下划线能提高可读性，可以在模块名中使用下划线
>尽管不鼓励使用下划线，但 Python 包也应该有简短且全小写的名字

When an extension module written in C or C++ has an accompanying Python module that provides a higher level (e.g. more object oriented) interface, the C/C++ module has a leading underscore (e.g. `_socket`).
>当用 C 或 C++ 编写的扩展模块有一个对应的 Python 模块提供更高级别接口 (例如，更面向对象的接口)，C/C++ 模块的名称前应带一个下划线 (例如，`_socket` )

### Class Names
Class names should normally use the CapWords convention.
> 类名使用 `CapsWords` 规范

The naming convention for functions may be used instead in cases where the interface is documented and used primarily as a callable.
>当该类主要作为可调用对象被使用，并且已经在文档中记录了该接口时，类名可以使用函数的命名约定

Note that there is a separate convention for builtin names: most builtin names are single words (or two words run together), with the CapWords convention used only for exception names and builtin constants.
>需要注意的是，内建名称遵循一套单独的约定：大多数内建名称是单个单词 (或者两个单词连在一起)
>只有异常名称和内建常量使用 CapWords 命名约定

### Type Variable Names
Names of type variables introduced in [PEP 484](https://peps.python.org/pep-0484/ "PEP 484 – Type Hints") should normally use CapWords preferring short names: `T`, `AnyStr`, `Num`. It is recommended to add suffixes `_co` or `_contra` to the variables used to declare covariant or contravariant behavior correspondingly:
> PEP 484 引入的类型变量的名称应该使用 CapWords，并且偏好短的名称，例如 `T/AnyStr/Num`
> 建议在用于声明协变或逆变行为的变量后添加后缀 `_co` 或 `_contra`

```python
from typing import TypeVar

VT_co = TypeVar('VT_co', covariant=True)
KT_contra = TypeVar('KT_contra', contravariant=True)
```

### Exception Names
Because exceptions should be classes, the class naming convention applies here. However, you should use the suffix “Error” on your exception names (if the exception actually is an error).
> 异常应该是类，故异常命名遵循类名规范
> 以及异常名应该有 `Error` 后缀

### Global Variable Names
(Let’s hope that these variables are meant for use inside one module only.) The conventions are about the same as those for functions.
> 全局变量应该仅在单个模块内使用
> 全局变量的命名规则和函数命名规则相同

Modules that are designed for use via `from M import *` should use the `__all__` mechanism to prevent exporting globals, or use the older convention of prefixing such globals with an underscore (which you might want to do to indicate these globals are “module non-public”).
> 设计为会被通过 `from M import *` 使用的模块应该使用 `__all__` 来避免导出全局变量，或者使用更老的规范，即为不希望被导出的全局变量名称添加前缀下划线

### Function and Variable Names
Function names should be lowercase, with words separated by underscores as necessary to improve readability.
> 函数名全小写，必要时用下划线分离单词以提高可读性

Variable names follow the same convention as function names.
> 变量名规范的函数名一致

mixedCase is allowed only in contexts where that’s already the prevailing style (e.g. threading.py), to retain backwards compatibility.

### Function and Method Arguments
Always use `self` for the first argument to instance methods.
> 实例方法的第一个参数一定是 `self`

Always use `cls` for the first argument to class methods.
> 类方法的第一个参数一定是 `cls`

If a function argument’s name clashes with a reserved keyword, it is generally better to append a single trailing underscore rather than use an abbreviation or spelling corruption. Thus `class_` is better than `clss`. (Perhaps better is to avoid such clashes by using a synonym.)
> 如果函数参数的名称和某个保留的关键字相同，最好在后面添加 `_` 
> 例如 `class_` ，这好于随意的缩写例如 `clss` ，更好的方式是用另一个近义词

### Method Names and Instance Variables
Use the function naming rules: lowercase with words separated by underscores as necessary to improve readability.
> 方法名和函数名规范一致

Use one leading underscore only for non-public methods and instance variables.
> 私有的方法和实例变量要有前导下划线

To avoid name clashes with subclasses, use two leading underscores to invoke Python’s name mangling rules.

Python mangles these names with the class name: if class Foo has an attribute named `__a`, it cannot be accessed by `Foo.__a`. (An insistent user could still gain access by calling `Foo._Foo__a`.) Generally, double leading underscores should be used only to avoid name conflicts with attributes in classes designed to be subclassed.

Note: there is some controversy about the use of `__names` (see below).

> 要避免和子类的命名冲突时，在名称前添加两个前导下划线以触发 Python 的名称改编机制
> 名称改编机制会将类 `Foo` 中名为 `__a` 的属性改编为 `_Foo__a` ，注意 `__a` 不能通过 `Foo.__a` 访问，但可以被 `Foo._Foo__a` 访问
> 双前导下划线应该仅在避免与子类命名冲突时使用
> 对于 `__names` 的用法也存在一些争议

### Constants
Constants are usually defined on a module level and written in all capital letters with underscores separating words. Examples include `MAX_OVERFLOW` and `TOTAL`.
> 常量定义于模块级别，名称全大写，使用下划线分隔单词

### Designing for Inheritance
Always decide whether a class’s methods and instance variables (collectively: “attributes”) should be public or non-public. If in doubt, choose non-public; it’s easier to make it public later than to make a public attribute non-public.
> 在设计时，确保决定好一个类的方法和实例变量 (统称为属性) 应该是公有还是私有
> 如果不确定，就定性为私有，将私有属性变为公有比将公有属性变为私有容易

Public attributes are those that you expect unrelated clients of your class to use, with your commitment to avoid backwards incompatible changes. Non-public attributes are those that are not intended to be used by third parties; you make no guarantees that non-public attributes won’t change or even be removed.
> 公有属性在设计上应该期待被第三方使用，公有属性的设计应该避免向后不兼容的更改
> 私有属性则相反，我们不需要保证私有属性不会被改变或被移除

We don’t use the term “private” here, since no attribute is really private in Python (without a generally unnecessary amount of work).

Another category of attributes are those that are part of the “subclass API” (often called “protected” in other languages). Some classes are designed to be inherited from, either to extend or modify aspects of the class’s behavior. When designing such a class, take care to make explicit decisions about which attributes are public, which are part of the subclass API, and which are truly only to be used by your base class.
> 另一类属性即属于“子类 API”的属性 (保护的)，这针对于设计上就是需要被继承和拓展的类
> 设计这些类时，注意明确决定好哪些属性为公有，哪些是子类 API 的一部分，哪些仅用于基类

With this in mind, here are the Pythonic guidelines:

- Public attributes should have no leading underscores.
> 公有属性没有前导下划线

- If your public attribute name collides with a reserved keyword, append a single trailing underscore to your attribute name. This is preferable to an abbreviation or corrupted spelling. (However, notwithstanding this rule, ‘cls’ is the preferred spelling for any variable or argument which is known to be a class, especially the first argument to a class method.)
    
    Note 1: See the argument name recommendation above for class methods.

> 如果公有属性名和保留关键字冲突，添加后缀下划线 (然而，尽管有这条规则，“cls” 是任何已知为类的变量或参数的首选拼写，尤其是作为类方法的第一个参数)

- For simple public data attributes, it is best to expose just the attribute name, without complicated accessor/mutator methods. Keep in mind that Python provides an easy path to future enhancement, should you find that a simple data attribute needs to grow functional behavior. In that case, use properties to hide functional implementation behind simple data attribute access syntax.
    
    Note 1: Try to keep the functional behavior side-effect free, although side-effects such as caching are generally fine.
    
    Note 2: Avoid using properties for computationally expensive operations; the attribute notation makes the caller believe that access is (relatively) cheap.

> 对于简单的公有数据属性，最好只暴露属性名称，而不使用复杂的访问器/修改器方法
> 请记住，Python 提供了一条通往未来增强功能的简单路径，以防你需要将一个简单的数据属性扩展为具有功能行为，在这种情况下，使用属性可以在简单的数据属性访问语法背后隐藏功能实现 (但注意避免为计算昂贵的操作使用属性，因为属性标识会让调用者相信访问它是相对经济的)

- If your class is intended to be subclassed, and you have attributes that you do not want subclasses to use, consider naming them with double leading underscores and no trailing underscores. This invokes Python’s name mangling algorithm, where the name of the class is mangled into the attribute name. This helps avoid attribute name collisions should subclasses inadvertently contain attributes with the same name.
    
    Note 1: Note that only the simple class name is used in the mangled name, so if a subclass chooses both the same class name and attribute name, you can still get name collisions.
    
    Note 2: Name mangling can make certain uses, such as debugging and `__getattr__()`, less convenient. However the name mangling algorithm is well documented and easy to perform manually.
    
    Note 3: Not everyone likes name mangling. Try to balance the need to avoid accidental name clashes with potential use by advanced callers.

> 如果类设计上需要被继承，并且有不想子类使用的属性，则使用两个前导下划线，且不添加后缀下划线，以调用 Python 的名称改编机制，避免和子类的名称冲突

## Public and Internal Interfaces
Any backwards compatibility guarantees apply only to public interfaces. Accordingly, it is important that users be able to clearly distinguish between public and internal interfaces.
> 所有的向后兼容性仅对于公共接口做出保证
> 用户也需要区分公共和内部接口

Documented interfaces are considered public, unless the documentation explicitly declares them to be provisional or internal interfaces exempt from the usual backwards compatibility guarantees. All undocumented interfaces should be assumed to be internal.
> 记录在文档中的接口都认为是公共的，除非文档明确声明它们是临时的或不受通常向后兼容性保证约束的内部接口
> 没有记录在文档中的接口都认为是内部的

To better support introspection, modules should explicitly declare the names in their public API using the `__all__` attribute. Setting `__all__` to an empty list indicates that the module has no public API.

Even with `__all__` set appropriately, internal interfaces (packages, modules, classes, functions, attributes or other names) should still be prefixed with a single leading underscore.

> 模块应该使用 `__all__` 属性显式声明它的公共 API 名字，将 `__all__` 设定为空列表意味着模块没有公共 API
> 即便 `__all__` 设定好了，内部接口 (包、模块、类、函数、属性以及其他名字) 仍应该有前导下划线

An interface is also considered internal if any containing namespace (package, module or class) is considered internal.
> 如果任意包含它的命名空间 (包、模块、类) 是内部的，则接口就认为是内部的

Imported names should always be considered an implementation detail. Other modules must not rely on indirect access to such imported names unless they are an explicitly documented part of the containing module’s API, such as `os.path` or a package’s `__init__` module that exposes functionality from submodules.
> 导入的名称始终应被视为实现细节
> 除非这些导入的名称是包含它的模块的 API 中明确记录的部分，例如 `os.path` 或包的 `__init__` 模块 (用于从子模块中暴露功能)，否则其他模块不应该依赖于间接访问这些导入的名称 (意思是没有在文档中记录的 API 不应该用 `<module-name>.<attribute-name>` 访问)

# Programming Recommendations

- Code should be written in a way that does not disadvantage other implementations of Python (PyPy, Jython, IronPython, Cython, Psyco, and such).
    
    For example, do not rely on CPython’s efficient implementation of in-place string concatenation for statements in the form `a += b` or `a = a + b`. This optimization is fragile even in CPython (it only works for some types) and isn’t present at all in implementations that don’t use refcounting. In performance sensitive parts of the library, the `''.join()` form should be used instead. This will ensure that concatenation occurs in linear time across various implementations.
> 代码应以不会对 Python 其他实现 (如 PyPy、Jython、IronPython、Cython、Psyco 等) 造成不利影响的方式编写
> 例如，不要依赖 CPython 对形式为 `a += b` 或 `a = a + b` 的语句中 in-place 字符串连接的高效实现。即使在 CPython 中，这种优化也很脆弱 (仅适用于某些类型)，并且在不使用引用计数的实现中根本不存在
> 在库的性能敏感部分，应使用 `''.join()` 形式代替，这将确保在各种实现中连接操作都以线性时间完成。

- Comparisons to singletons like None should always be done with `is` or `is not`, never the equality operators.
    
    Also, beware of writing `if x` when you really mean `if x is not None` – e.g. when testing whether a variable or argument that defaults to None was set to some other value. The other value might have a type (such as a container) that could be false in a boolean context!

> 与单例 (如 `None`)的比较应始终使用 `is` 或 `is not`，而不要使用等于运算符
> 另外，要注意 `if x` 和 `if x is not None` 之间的区别
> 例如，在测试一个默认值为 `None` 的变量或参数 `x` 是否被设置为其他值时，不要使用前者，`x` 被设定为的值可能一种类型 (如容器)，在这种布尔上下文中可能是假的！( `x` 即便被设定为了 `[]` ，`if x` 也会返回 false )

- Use `is not` operator rather than `not ... is`. While both expressions are functionally identical, the former is more readable and preferred:

```python
# Correct:
if foo is not None:

# Wrong:
if not foo is None:
```

> 使用 `is not` 而不是 `not ... is`
> 二者的功能相同，但是前者可读性更高

- When implementing ordering operations with rich comparisons, it is best to implement all six operations (`__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`) rather than relying on other code to only exercise a particular comparison.
    
    To minimize the effort involved, the `functools.total_ordering()` decorator provides a tool to generate missing comparison methods.
    
    [PEP 207](https://peps.python.org/pep-0207/ "PEP 207 – Rich Comparisons") indicates that reflexivity rules _are_ assumed by Python. Thus, the interpreter may swap `y > x` with `x < y`, `y >= x` with `x <= y`, and may swap the arguments of `x == y` and `x != y`. The `sort()` and `min()` operations are guaranteed to use the `<` operator and the `max()` function uses the `>` operator. However, it is best to implement all six operations so that confusion doesn’t arise in other contexts.

> 在使用丰富的比较操作实现排序操作时，最好实现所有六种操作（`__eq__`、`__ne__`、`__lt__`、`__le__`、`__gt__`、`__ge__`），而不是写一些只执行特定的比较的代码
> 为了尽量减少工作量，`functools.total_ordering()` 装饰器提供了一个工具来生成缺失的比较方法
> [PEP 207](https://peps.python.org/pep-0207/ "PEP 207 – Rich Comparisons") 指出，Python 假设了反射规则，因此，解释器可能会交换 `y > x` 和 `x < y`，`y >= x` 和 `x <= y`，以及 `x == y` 和 `x != y` 的参数，`sort()` 和 `min()` 操作保证会使用 `<` 运算符，而 `max()` 函数则使用 `>` 运算符，但最好还是实现所有六种操作，以免在其他上下文中引起混淆

- Always use a def statement instead of an assignment statement that binds a lambda expression directly to an identifier:

```python
# Correct:
def f(x): return 2*x

# Wrong:
f = lambda x: 2*x
```
    
The first form means that the name of the resulting function object is specifically ‘f’ instead of the generic ‘ `<lambda>` ’. This is more useful for tracebacks and string representations in general. The use of the assignment statement eliminates the sole benefit a lambda expression can offer over an explicit def statement (i.e. that it can be embedded inside a larger expression)

> 始终使用 `def` 语句而不是直接将 lambda 表达式绑定到标识符的赋值语句，换句话说，需要定义有名字的函数对象时，使用 `def` 语句而不是 lambda 表达式
> 第一种形式意味着生成的函数对象的名称是特定的 “f”，而不是通用的“`<lambda>`”
> 这在回溯和字符串表示方面更有用
> 使用赋值语句消除了 lambda 表达式相对于显式的 `def` 语句所能提供的唯一好处(即它可以嵌入更大的表达式中)

- Derive exceptions from `Exception` rather than `BaseException`. Direct inheritance from `BaseException` is reserved for exceptions where catching them is almost always the wrong thing to do.
    
    Design exception hierarchies based on the distinctions that code _catching_ the exceptions is likely to need, rather than the locations where the exceptions are raised. Aim to answer the question “What went wrong?” programmatically, rather than only stating that “A problem occurred” (see [PEP 3151](https://peps.python.org/pep-3151/ "PEP 3151 – Reworking the OS and IO exception hierarchy") for an example of this lesson being learned for the builtin exception hierarchy)
    
    Class naming conventions apply here, although you should add the suffix “Error” to your exception classes if the exception is an error. Non-error exceptions that are used for non-local flow control or other forms of signaling need no special suffix.

> 从 `Exception` 类中继承异常类，而不是从 `BaseException` 中继承，直接继承 `BaseException` 的异常类在设计上是不允许捕获的
> 基于用于捕获异常的代码之间的差异来设计异常层次结构，而不是基于异常被抛出的位置，我们设计的异常需要在程序语义上回答“发生了什么错误”，而不是仅表明“出现了一个问题”
> 异常类的命名遵循类的命名规范，如果异常是错误，应该添加后缀 `Error` ，用于非局部的流程控制或者其他形式的信号的非错误异常则没有特殊后缀要求

- Use exception chaining appropriately. `raise X from Y` should be used to indicate explicit replacement without losing the original traceback.
    
    When deliberately replacing an inner exception (using `raise X from None`), ensure that relevant details are transferred to the new exception (such as preserving the attribute name when converting KeyError to AttributeError, or embedding the text of the original exception in the new exception message).

> 恰当使用异常链，`raise X from Y` 应该用于表示明确的替换，并且不会丢失原来的回溯信息
> 当有意替换一个内部异常 (使用 `raise X from None` ) 时，确保将相关的细节都转移到新的异常中 (例如在将 KeyError 向 AttributeError 转换时保留属性名称，或在新异常的消息中嵌入原始异常的文本)

- When catching exceptions, mention specific exceptions whenever possible instead of using a bare `except:` clause:

```python
try:
    import platform_specific_module
except ImportError:
    platform_specific_module = None
```
    
A bare `except:` clause will catch SystemExit and KeyboardInterrupt exceptions, making it harder to interrupt a program with Control-C, and can disguise other problems. If you want to catch all exceptions that signal program errors, use `except Exception:` (bare except is equivalent to `except BaseException:`).

A good rule of thumb is to limit use of bare ‘except’ clauses to two cases:
    
1. If the exception handler will be printing out or logging the traceback; at least the user will be aware that an error has occurred.
2. If the code needs to do some cleanup work, but then lets the exception propagate upwards with `raise`. `try...finally` can be a better way to handle this case.

> 捕获异常时，尽可能提到具体的异常，而不是仅使用空的 `except:` 子句
> 空的 `except:` 子句实际上等价于 `except BaseException:` ，它会捕获所有异常，包括 SystemExit 和 KeyboradInterrupt 异常，使得我们不容易直接通过 Ctrl-C 中断程序 (会被捕获)，并且可能会掩盖其他问题
> 如果我们确实希望捕获所有表示程序错误的异常，使用 `except Exception:`
> 一个很好的经验法则是，仅在以下两种情况有限地使用空的 `except:` 子句
> 1. 异常处理程序会打印或记录回溯，至少用户会知道错误已发生
> 2. 代码想要先做一些清理工作，然后再使用 `raise` 将异常向上抛出 (这种情况下使用 `try ... finally` 结构可能是更好的选择)

- When catching operating system errors, prefer the explicit exception hierarchy introduced in Python 3.3 over introspection of `errno` values.
> 在捕获操作系统错误时，优先使用 Python 3.3 引入的显式异常层次结构，而不是通过检查 `errno` 值来进行推测

- Additionally, for all try/except clauses, limit the `try` clause to the absolute minimum amount of code necessary. Again, this avoids masking bugs:

```python
# Correct:
try:
    value = collection[key]
except KeyError:
    return key_not_found(key)
else:
    return handle_value(value)

# Wrong:
try:
    # Too broad!
    return handle_value(collection[key])
except KeyError:
    # Will also catch KeyError raised by handle_value()
    return key_not_found(key)
```

> 此外，对于所有的 `try/except` 子句，`try` 子句中的代码量应尽量保持在绝对最小，这样做可以避免隐藏 bugs，详见上面的例子

- When a resource is local to a particular section of code, use a `with` statement to ensure it is cleaned up promptly and reliably after use. A try/finally statement is also acceptable.
>当资源仅限于代码的某个特定部分时，使用 `with` 语句以确保在使用后能够及时且可靠地进行清理，也可以使用 `try/finally` 语句

- Context managers should be invoked through separate functions or methods whenever they do something other than acquire and release resources:

```python
# Correct:
with conn.begin_transaction():
    do_stuff_in_transaction(conn)

# Wrong:
with conn:
    do_stuff_in_transaction(conn)
```
    
The latter example doesn’t provide any information to indicate that the `__enter__` and `__exit__` methods are doing something other than closing the connection after a transaction. Being explicit is important in this case.

>  当上下文管理器所做的操作不仅仅是获取和释放资源时，应通过单独的函数或方法调用它们
>  上例中，后一个示例没有提供任何信息来表明 `__enter__` 和 `__exit__` 方法在事务结束后除了关闭连接之外还做了其他事情，在这种情况下，明确表达是很重要的

- Be consistent in return statements. Either all return statements in a function should return an expression, or none of them should. If any return statement returns an expression, any return statements where no value is returned should explicitly state this as `return None`, and an explicit return statement should be present at the end of the function (if reachable):

```python
#  Correct:

def foo(x):
    if x >= 0:
        return math.sqrt(x)
    else:
        return None

def bar(x):
    if x < 0:
        return None
    return math.sqrt(x)

# Wrong:

def foo(x):
    if x >= 0:
        return math.sqrt(x)

def bar(x):
    if x < 0:
        return
    return math.sqrt(x)
```

> 在返回语句上保持一致性
> 在一个函数中，要么所有的返回语句都返回一个表达式，要么都不返回表达式
> 如果其中的任意返回语句返回了表达式，则任意其他没有值需要返回的返回语句需要显式地写为 `return None` ，并且在函数的末尾 (如果可达的话) 也要有显式的返回语句

- Use `''.startswith()` and `''.endswith()` instead of string slicing to check for prefixes or suffixes.
    
    startswith() and endswith() are cleaner and less error prone:

```python
# Correct:
if foo.startswith('bar'):

# Wrong:
if foo[:3] == 'bar':
```

> 使用 `''.startswith()/endswith()` 而不是字符串切片来检查前缀或后缀，这样更加整洁且不易错

- Object type comparisons should always use isinstance() instead of comparing types directly:
    
```python
# Correct:
if isinstance(obj, int):

# Wrong:
if type(obj) is type(1):
```

> 对象类型的比较应该总是使用 `isinstance()` 而不是直接比较

- For sequences, (strings, lists, tuples), use the fact that empty sequences are false:
    
```python
# Correct:
if not seq:
if seq:

# Wrong:
if len(seq):
if not len(seq):
```

> 多利用空序列表示 false 的性质

- Don’t write string literals that rely on significant trailing whitespace. Such trailing whitespace is visually indistinguishable and some editors (or more recently, reindent.py) will trim them.
> 不要编写依赖显著尾随空格的字符串字面量，这样的尾随空格在视觉上无法区分，并且一些编辑器 (或最近的 reindent.py) 会去除它们

- Don’t compare boolean values to True or False using `==`:
    
```python
# Correct:
if greeting:

# Wrong:
if greeting == True:
```

Worse:

```python
# Wrong:
if greeting is True:
```

> 不要将布尔值和 `True/False` 用 `==` 比较，或者用 `is` 比较

- Use of the flow control statements `return`/`break`/`continue` within the finally suite of a `try...finally`, where the flow control statement would jump outside the finally suite, is discouraged. This is because such statements will implicitly cancel any active exception that is propagating through the finally suite:
    
```python
# Wrong:
def foo():
    try:
        1 / 0
    finally:
        return 42
```

>  不推荐在 `try ... finally` 结构中使用流程控制语句 `return/break/continue`，它们会使流程跳到 `finally` 块外面，因此这些语句会隐式地取消任何需要通过 `finally` 块传播的活跃异常

## Function Annotations
With the acceptance of [PEP 484](https://peps.python.org/pep-0484/ "PEP 484 – Type Hints"), the style rules for function annotations have changed.

- Function annotations should use [PEP 484](https://peps.python.org/pep-0484/ "PEP 484 – Type Hints") syntax (there are some formatting recommendations for annotations in the previous section).
- The experimentation with annotation styles that was recommended previously in this PEP is no longer encouraged.
- However, outside the stdlib, experiments within the rules of [PEP 484](https://peps.python.org/pep-0484/ "PEP 484 – Type Hints") are now encouraged. For example, marking up a large third party library or application with [PEP 484](https://peps.python.org/pep-0484/ "PEP 484 – Type Hints") style type annotations, reviewing how easy it was to add those annotations, and observing whether their presence increases code understandability.
- The Python standard library should be conservative in adopting such annotations, but their use is allowed for new code and for big refactorings.

> 函数注释遵循 PEP 484

- For code that wants to make a different use of function annotations it is recommended to put a comment of the form:
    
```python
# type: ignore
```
    
near the top of the file; this tells type checkers to ignore all annotations. (More fine-grained ways of disabling complaints from type checkers can be found in [PEP 484](https://peps.python.org/pep-0484/ "PEP 484 – Type Hints").)

> 如果代码需要使用不同风格的函数注释，在文件顶部添加注释 `# type: ignore`，以告诉类型检查器忽略所有的注释

- Like linters, type checkers are optional, separate tools. Python interpreters by default should not issue any messages due to type checking and should not alter their behavior based on annotations.

> linter 和 type checker 都被认为是可选的额外工具，Python 解释器默认不会对有类型注释导致的类型检查问题抛出问题，也不会根据注释改变其行为

- Users who don’t want to use type checkers are free to ignore them. However, it is expected that users of third party library packages may want to run type checkers over those packages. For this purpose [PEP 484](https://peps.python.org/pep-0484/ "PEP 484 – Type Hints") recommends the use of stub files: .pyi files that are read by the type checker in preference of the corresponding .py files. Stub files can be distributed with a library, or separately (with the library author’s permission) through the typeshed repo [[5]] (https://peps.python.org/pep-0008/#id9).
>  不想使用类型检查器的用户可以自由忽略它们。然而，第三方库包的用户可能希望对这些包运行类型检查器。为此，[PEP 484](https://peps.python.org/pep-0484/ "PEP 484 – Type Hints") 建议使用存根文件：优先于对应的 `.py` 文件被类型检查器读取的 `.pyi` 文件。存根文件可以随库一起分发，或者在库作者的许可下通过 `typeshed` 仓库分别分发

## Variable Annotations
[PEP 526](https://peps.python.org/pep-0526/ "PEP 526 – Syntax for Variable Annotations") introduced variable annotations. The style recommendations for them are similar to those on function annotations described above:

- Annotations for module level variables, class and instance variables, and local variables should have a single space after the colon.
- There should be no space before the colon.
- If an assignment has a right hand side, then the equality sign should have exactly one space on both sides:
    
```python
# Correct:

code: int

class Point:
    coords: Tuple[int, int]
    label: str = '<unknown>'

# Wrong:

code:int  # No space after colon
code : int  # Space before colon

class Test:
    result: int=0  # No spaces around equality sign
```

> 模块级别变量、类和实例变量、局部变量的注释在 `:` 有一个空格
> `:` 前没有空格
> 注释后的赋值符号 `=` 左右各一个空格 

- Although the [PEP 526](https://peps.python.org/pep-0526/ "PEP 526 – Syntax for Variable Annotations") is accepted for Python 3.6, the variable annotation syntax is the preferred syntax for stub files on all versions of Python (see [PEP 484](https://peps.python.org/pep-0484/ "PEP 484 – Type Hints") for details).

Footnotes
[1](https://peps.python.org/pep-0008/#id2) _Hanging indentation_ is a type-setting style where all the lines in a paragraph are indented except the first line. In the context of Python, the term is used to describe a style where the opening parenthesis of a parenthesized statement is the last non-whitespace character of the line, with subsequent lines being indented until the closing parenthesis.

# References
[2](https://peps.python.org/pep-0008/#id1) Barry’s GNU Mailman style guide [http://barry.warsaw.us/software/STYLEGUIDE.txt](http://barry.warsaw.us/software/STYLEGUIDE.txt)

[3](https://peps.python.org/pep-0008/#id3) Donald Knuth’s _The TeXBook_, pages 195 and 196.

[4](https://peps.python.org/pep-0008/#id4) [http://www.wikipedia.com/wiki/Camel_case](http://www.wikipedia.com/wiki/Camel_case)

[5](https://peps.python.org/pep-0008/#id5) Typeshed repo [https://github.com/python/typeshed](https://github.com/python/typeshed)

# Copyright
This document has been placed in the public domain.

---

Source: [https://github.com/python/peps/blob/main/peps/pep-0008.rst](https://github.com/python/peps/blob/main/peps/pep-0008.rst)

Last modified: [2024-09-09 14:02:27 GMT](https://github.com/python/peps/commits/main/peps/pep-0008.rst)


