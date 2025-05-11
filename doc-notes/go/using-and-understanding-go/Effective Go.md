## Introduction
Go is a new language. Although it borrows ideas from existing languages, it has unusual properties that make effective Go programs different in character from programs written in its relatives. A straightforward translation of a C++ or Java program into Go is unlikely to produce a satisfactory result—Java programs are written in Java, not Go. On the other hand, thinking about the problem from a Go perspective could produce a successful but quite different program. 

In other words, to write Go well, it's important to understand its properties and idioms. It's also important to know the established conventions for programming in Go, such as naming, formatting, program construction, and so on, so that programs you write will be easy for other Go programmers to understand.

This document gives tips for writing clear, idiomatic Go code. It augments the [language specification](https://go.dev/ref/spec), the [Tour of Go](https://go.dev/tour/), and [How to Write Go Code](https://go.dev/doc/code.html), all of which you should read first.

Note added January, 2022: This document was written for Go's release in 2009, and has not been updated significantly since. Although it is a good guide to understand how to use the language itself, thanks to the stability of the language, it says little about the libraries and nothing about significant changes to the Go ecosystem since it was written, such as the build system, testing, modules, and polymorphism. There are no plans to update it, as so much has happened and a large and growing set of documents, blogs, and books do a fine job of describing modern Go usage. Effective Go continues to be useful, but the reader should understand it is far from a complete guide. See [issue 28782](https://go.dev/issue/28782) for context.

#### Examples
The [Go package sources](https://go.dev/src/) are intended to serve not only as the core library but also as examples of how to use the language. Moreover, many of the packages contain working, self-contained executable examples you can run directly from the [go.dev](https://go.dev/) web site, such as [this one](https://go.dev/pkg/strings/##example-Map) (if necessary, click on the word "Example" to open it up). If you have a question about how to approach a problem or how something might be implemented, the documentation, code and examples in the library can provide answers, ideas and background.

## Formatting 
Formatting issues are the most contentious but the least consequential. People can adapt to different formatting styles but it's better if they don't have to, and less time is devoted to the topic if everyone adheres to the same style. The problem is how to approach this Utopia without a long prescriptive style guide.

With Go we take an unusual approach and let the machine take care of most formatting issues. The `gofmt` program (also available as `go fmt`, which operates at the package level rather than source file level) reads a Go program and emits the source in a standard style of indentation and vertical alignment, retaining and if necessary reformatting comments. If you want to know how to handle some new layout situation, run `gofmt`; if the answer doesn't seem right, rearrange your program (or file a bug about `gofmt`), don't work around it.
>  `gofmt, go fmt` 会读取 Go 程序，将其格式化

As an example, there's no need to spend time lining up the comments on the fields of a structure. `Gofmt` will do that for you. Given the declaration

```go
type T struct {
    name string // name of the object
    value int // its value
}
```

`gofmt` will line up the columns:

```go
type T struct {
    name    string // name of the object
    value   int    // its value
}
```

All Go code in the standard packages has been formatted with `gofmt`.

Some formatting details remain. Very briefly:

Indentation: We use tabs for indentation and `gofmt` emits them by default. Use spaces only if you must.

>  缩进使用 Tab

Line length: Go has no line length limit. Don't worry about overflowing a punched card. If a line feels too long, wrap it and indent with an extra tab.

>  Go 没有行长度限制，如果觉得太长，可以换行并在下一行前添加 Tab

Parentheses: Go needs fewer parentheses than C and Java: control structures (`if`, `for`, `switch`) do not have parentheses in their syntax. Also, the operator precedence hierarchy is shorter and clearer, so

```go
x<<8 + y<<16
```

means what the spacing implies, unlike in the other languages.

>  Go 中的控制结构 `if, for, switch` 都不需要括号

## Commentary
Go provides C-style `/* */` block comments and C++-style `//` line comments. Line comments are the norm; block comments appear mostly as package comments, but are useful within an expression or to disable large swaths of code.
>  `//` 为常用注释，`/* */` 通常用于包注释

Comments that appear before top-level declarations, with no intervening newlines, are considered to document the declaration itself. These “doc comments” are the primary documentation for a given Go package or command. For more about doc comments, see “[Go Doc Comments](https://go.dev/doc/comment)”.
 >  Doc Comments 见详细文档

## Names 
Names are as important in Go as in any other language. They even have semantic effect: the visibility of a name outside a package is determined by whether its first character is upper case. It's therefore worth spending a little time talking about naming conventions in Go programs.

#### Package names 
When a package is imported, the package name becomes an accessor for the contents. After

```go
import "bytes"
```

the importing package can talk about `bytes.Buffer`. It's helpful if everyone using the package can use the same name to refer to its contents, which implies that the package name should be good: short, concise, evocative. 
>  包名应该简短、简洁且具有表现力

By convention, packages are given lower case, single-word names; there should be no need for underscores or mixedCaps. Err on the side of brevity, since everyone using your package will be typing that name. 
>  包名应该仅有一个单词，小写

And don't worry about collisions _a priori_. The package name is only the default name for imports; it need not be unique across all source code, and in the rare case of a collision the importing package can choose a different name to use locally. In any case, confusion is rare because the file name in the import determines just which package is being used.

Another convention is that the package name is the base name of its source directory; the package in `src/encoding/base64` is imported as `"encoding/base64"` but has name `base64`, not `encoding_base64` and not `encodingBase64`.
>  包名应该是其源代码目录的 base name，例如在 `src/encoding/base64` 的包通过 `encoding/base64` 导入，其名称是 `base64`

The importer of a package will use the name to refer to its contents, so exported names in the package can use that fact to avoid repetition. (Don't use the `import .` notation, which can simplify tests that must run outside the package they are testing, but should otherwise be avoided.) 

For instance, the buffered reader type in the `bufio` package is called `Reader`, not `BufReader`, because users see it as `bufio.Reader`, which is a clear, concise name. 

Moreover, because imported entities are always addressed with their package name, `bufio.Reader` does not conflict with `io.Reader`. 

Similarly, the function to make new instances of `ring.Ring`—which is the definition of a _constructor_ in Go—would normally be called `NewRing`, but since `Ring` is the only type exported by the package, and since the package is called `ring`, it's called just `New`, which clients of the package see as `ring.New`. Use the package structure to help you choose good names.

Another short example is `once.Do`; `once.Do(setup)` reads well and would not be improved by writing `once.DoOrWaitUntilDone(setup)`. Long names don't automatically make things more readable. A helpful doc comment can often be more valuable than an extra long name.

#### Getters
Go doesn't provide automatic support for getters and setters. There's nothing wrong with providing getters and setters yourself, and it's often appropriate to do so, but it's neither idiomatic nor necessary to put `Get` into the getter's name. 

If you have a field called `owner` (lower case, unexported), the getter method should be called `Owner` (upper case, exported), not `GetOwner`. 
>  如果有一个名为 `owner` 的字段 (小写，未导出)，其 getter 方法应该是 `Owner` (大写，导出) 而不是 `GetOwner`

The use of upper-case names for export provides the hook to discriminate the field from the method. A setter function, if needed, will likely be called `SetOwner`. Both names read well in practice:
>  setter 函数可以被命名为 `SetOwner`

```go
owner := obj.Owner()
if owner != user {
    obj.SetOwner(user)
}
```

#### Interface names
By convention, one-method interfaces are named by the method name plus an -er suffix or similar modification to construct an agent noun: `Reader`, `Writer`, `Formatter`, `CloseNotifier` etc.
>  单方法接口的名称是方法名 + `er` 后缀，例如 `Reader, Writer, Formatter, CloseNotifier`

There are a number of such names and it's productive to honor them and the function names they capture. `Read`, `Write`, `Close`, `Flush`, `String` and so on have canonical signatures and meanings. 

To avoid confusion, don't give your method one of those names unless it has the same signature and meaning. 

Conversely, if your type implements a method with the same meaning as a method on a well-known type, give it the same name and signature; call your string-converter method `String` not `ToString`.
>  如果我们的自定义类型实现了一个和知名类型上的方法具有相同含义的方法，就为其赋予相同的名称和签名
>  例如，将我们自定义的字符串转换方法命名为 `String` 而不是 `ToString`

#### MixedCaps
Finally, the convention in Go is to use `MixedCaps` or `mixedCaps` rather than underscores to write multiword names.
>  Go 使用 `MixedCaps` 或 `mixedCaps` ，不使用下划线连接多个词

## Semicolons 
Like C, Go's formal grammar uses semicolons to terminate statements, but unlike in C, those semicolons do not appear in the source. Instead the lexer uses a simple rule to insert semicolons automatically as it scans, so the input text is mostly free of them.
>  Go 的形式语法使用 `;` 终止语句，但 `;` 会由 Go 的 lexer 自动插入，不需要在源码中表示

The rule is this. If the last token before a newline is an identifier (which includes words like `int` and `float64`), a basic literal such as a number or string constant, or one of the tokens

```
break continue fallthrough return ++ -- ) }
```

the lexer always inserts a semicolon after the token. This could be summarized as, “if the newline comes after a token that could end a statement, insert a semicolon”.

>  lexer 插入 `;` 的规则可以概括为，如果 newline 出现在可能结束语句的 token 后，就插入 `;`

A semicolon can also be omitted immediately before a closing brace, so a statement such as

```
    go func() { for { dst <- <-src } }()
```

needs no semicolons. Idiomatic Go programs have semicolons only in places such as `for` loop clauses, to separate the initializer, condition, and continuation elements. 

>  Go 程序中，`;` 仅出现在例如 `for` 的子句中，用于分割 initializer, condition, continuation elements

They are also necessary to separate multiple statements on a line, should you write code that way.

One consequence of the semicolon insertion rules is that you cannot put the opening brace of a control structure (`if`, `for`, `switch`, or `select`) on the next line. If you do, a semicolon will be inserted before the brace, which could cause unwanted effects. Write them like this

```go
if i < f() {
    g()
}
```

>  Go 的 lexer 的 `;` 插入规则导致我们需要将 `{` 写在对应的控制结构的同一行
  
not like this

```go
if i < f()  // wrong!
{           // wrong!
    g()
}
```

## Control structures
The control structures of Go are related to those of C but differ in important ways. There is no `do` or `while` loop, only a slightly generalized `for`; `switch` is more flexible; `if` and `switch` accept an optional initialization statement like that of `for`; `break` and `continue` statements take an optional label to identify what to break or continue; and there are new control structures including a type switch and a multiway communications multiplexer, `select`. The syntax is also slightly different: there are no parentheses and the bodies must always be brace-delimited.
>  Go 的循环结构只有 `for`，接收可选的初始化语句
>  Go 的 `break, continue` 语句接收可选的标签，以确定在哪里 break 或 continue
>  Go 的条件结构 `switch, if` 接收可选的初始化语句
>  Go 有自己独有的控制结构，包括了 type switch 和多路通信选择器 `select`
>  Go 的控制结构没有括号，且代码块必须用大括号包围

### If
In Go a simple `if` looks like this:

```go
if x > 0 {
    return y
}
```

Mandatory braces encourage writing simple `if` statements on multiple lines. It's good style to do so anyway, especially when the body contains a control statement such as a `return` or `break`.

Since `if` and `switch` accept an initialization statement, it's common to see one used to set up a local variable.

```go
if err := file.Chmod(0664); err != nil {
    log.Print(err)
    return err
}
```

In the Go libraries, you'll find that when an `if` statement doesn't flow into the next statement—that is, the body ends in `break`, `continue`, `goto`, or `return`—the unnecessary `else` is omitted.

```go
f, err := os.Open(name)
if err != nil {
    return err
}
codeUsing(f)
```

This is an example of a common situation where code must guard against a sequence of error conditions. The code reads well if the successful flow of control runs down the page, eliminating error cases as they arise. Since error cases tend to end in `return` statements, the resulting code needs no `else` statements.

```go
f, err := os.Open(name)
if err != nil {
    return err
}
d, err := f.Stat()
if err != nil {
    f.Close()
    return err
}
codeUsing(f, d)
```

### Redeclaration and reassignment
An aside: The last example in the previous section demonstrates a detail of how the `:=` short declaration form works. The declaration that calls `os.Open` reads,

```go
f, err := os.Open(name)
```

This statement declares two variables, `f` and `err`. A few lines later, the call to `f.Stat` reads,

```go
d, err := f.Stat()
```

which looks as if it declares `d` and `err`. Notice, though, that `err` appears in both statements. This duplication is legal: `err` is declared by the first statement, but only _re-assigned_ in the second. This means that the call to `f.Stat` uses the existing `err` variable declared above, and just gives it a new value.
>  Go 中的 `:=` 不会重声明变量，只会重新赋值

In a `:=` declaration a variable `v` may appear even if it has already been declared, provided:

- this declaration is in the same scope as the existing declaration of `v` (if `v` is already declared in an outer scope, the declaration will create a new variable §),
- the corresponding value in the initialization is assignable to `v`, and
- there is at least one other variable that is created by the declaration.

>  `:=` 中可以出现已经被声明过的变量 `v` ，只要
>  - `:=` 的声明和 `v` 之前的声明处于同一作用域 (如果 `v` 已经在外部作用域被声明，则 `:=` 声明会创建新变量)
>  - `:=` 给出的对应值可以被赋予 `v`
>  - `:=` 中至少还有一个被声明创建的其他变量

This unusual property is pure pragmatism, making it easy to use a single `err` value, for example, in a long `if-else` chain. You'll see it used often.
>  这一性质完全出于实用的角度考虑，使得我们可以复用 `err` 变量

§ It's worth noting here that in Go the scope of function parameters and return values is the same as the function body, even though they appear lexically outside the braces that enclose the body.
>  注意在 Go 中，函数参数的作用域和返回值的作用域与函数体相同，尽管它们语法上出现在函数体的大括号外

### For
The Go `for` loop is similar to—but not the same as—C's. It unifies `for` and `while` and there is no `do-while`. There are three forms, only one of which has semicolons.

```go
// Like a C for
for init; condition; post { }

// Like a C while
for condition { }

// Like a C for(;;)
for { }
```

Short declarations make it easy to declare the index variable right in the loop.

```go
sum := 0
for i := 0; i < 10; i++ {
    sum += i
}
```

If you're looping over an array, slice, string, or map, or reading from a channel, a `range` clause can manage the loop.

```go
for key, value := range oldMap {
    newMap[key] = value
}
```

If you only need the first item in the range (the key or index), drop the second:

```go
for key := range m {
    if key.expired() {
        delete(m, key)
    }
}
```

If you only need the second item in the range (the value), use the _blank identifier_, an underscore, to discard the first:

```go
sum := 0
for _, value := range array {
    sum += value
}
```

The blank identifier has many uses, as described in [a later section](https://go.dev/doc/effective_go#blank).

For strings, the `range` does more work for you, breaking out individual Unicode code points by parsing the UTF-8. Erroneous encodings consume one byte and produce the replacement rune U+FFFD. (The name (with associated builtin type) `rune` is Go terminology for a single Unicode code point. See [the language specification](https://go.dev/ref/spec#Rune_literals) for details.) The loop
>  `range` 遍历 string 时，会解析 UTF-8，分离出单个 Unicode 码点
>  错误的编码会消耗一个字节，并产生替换符 `U+FFFD`
>  `rune` 是 Go 中表示单个 Unicode 码点的术语

```go
for pos, char := range "日本\x80語" { // \x80 is an illegal UTF-8 encoding
    fmt.Printf("character %#U starts at byte position %d\n", char, pos)
}
```

prints

```
character U+65E5 '日' starts at byte position 0
character U+672C '本' starts at byte position 3
character U+FFFD '�' starts at byte position 6
character U+8A9E '語' starts at byte position 7
```

Finally, Go has no comma operator and `++` and `--` are statements not expressions. Thus if you want to run multiple variables in a `for` you should use parallel assignment (although that precludes `++` and `--`).

```go
// Reverse a
for i, j := 0, len(a)-1; i < j; i, j = i+1, j-1 {
    a[i], a[j] = a[j], a[i]
}
```

### Switch
Go's `switch` is more general than C's. The expressions need not be constants or even integers, the cases are evaluated top to bottom until a match is found, and if the `switch` has no expression it switches on `true`. It's therefore possible—and idiomatic—to write an `if` - `else` - `if` - `else` chain as a `switch`.
>  Go 的 `switch` 不要求表达式是常数甚至整数
>  如果 `swtich` 没有表达式，则默认表达式为 `true`

```go
func unhex(c byte) byte {
    switch {
    case '0' <= c && c <= '9':
        return c - '0'
    case 'a' <= c && c <= 'f':
        return c - 'a' + 10
    case 'A' <= c && c <= 'F':
        return c - 'A' + 10
    }
    return 0
}
```

There is no automatic fall through, but cases can be presented in comma-separated lists.
>  cases 可以用逗号分隔

```go
func shouldEscape(c byte) bool {
    switch c {
    case ' ', '?', '&', '=', '#', '+', '%':
        return true
    }
    return false
}
```

Although they are not nearly as common in Go as some other C-like languages, `break` statements can be used to terminate a `switch` early. Sometimes, though, it's necessary to break out of a surrounding loop, not the switch, and in Go that can be accomplished by putting a label on the loop and "breaking" to that label. This example shows both uses.
>  `break` 可以用于提前终止 `switch`
>  `break` 可以通过指定标签，提前终止外部循环而不只是 `switch`

```go
Loop:
    for n := 0; n < len(src); n += size {
        switch {
        case src[n] < sizeOne:
            if validateOnly {
                break
            }
            size = 1
            update(src[n])

        case src[n] < sizeTwo:
            if n+1 >= len(src) {
                err = errShortInput
                break Loop
            }
            if validateOnly {
                break
            }
            size = 2
            update(src[n] + src[n+1]<<shift)
        }
    }
```

Of course, the `continue` statement also accepts an optional label but it applies only to loops.
>  `continue` 语句也接收一个可选的标签，但该语句仅在循环中使用

To close this section, here's a comparison routine for byte slices that uses two `switch` statements:

```go
// Compare returns an integer comparing the two byte slices,
// lexicographically.
// The result will be 0 if a == b, -1 if a < b, and +1 if a > b
func Compare(a, b []byte) int {
    for i := 0; i < len(a) && i < len(b); i++ {
        switch {
        case a[i] > b[i]:
            return 1
        case a[i] < b[i]:
            return -1
        }
    }
    switch {
    case len(a) > len(b):
        return 1
    case len(a) < len(b):
        return -1
    }
    return 0
}
```

### Type switch
A switch can also be used to discover the dynamic type of an interface variable. Such a _type switch_ uses the syntax of a type assertion with the keyword `type` inside the parentheses. If the switch declares a variable in the expression, the variable will have the corresponding type in each clause. It's also idiomatic to reuse the name in such cases, in effect declaring a new variable with the same name but a different type in each case.
>  `switch` 可以用于确定接口变量的动态类型
>  type switch 使用 type assertion 的语法，在括号内使用关键字 `type`，如果 `switch` 在表达式内声明了一个变量，该变量将在每个子句中有对应的类型
>  同样惯用的做法是在每个 cases 复用名称，这实际上是在每个 case 中声明了一个新的变量，具有相同的名称和不同的类型

```go
var t interface{}
t = functionOfSomeType()
switch t := t.(type) {
default:
    fmt.Printf("unexpected type %T\n", t)     // %T prints whatever type t has
case bool:
    fmt.Printf("boolean %t\n", t)             // t has type bool
case int:
    fmt.Printf("integer %d\n", t)             // t has type int
case *bool:
    fmt.Printf("pointer to boolean %t\n", *t) // t has type *bool
case *int:
    fmt.Printf("pointer to integer %d\n", *t) // t has type *int
}
```

## Functions
### Multiple return values 
One of Go's unusual features is that functions and methods can return multiple values. This form can be used to improve on a couple of clumsy idioms in C programs: in-band error returns such as `-1` for `EOF` and modifying an argument passed by address.
>  Go 中的函数和方法可以返回多个值

In C, a write error is signaled by a negative count with the error code secreted away in a volatile location. In Go, `Write` can return a count _and_ an error: “Yes, you wrote some bytes but not all of them because you filled the device”. The signature of the `Write` method on files from package `os` is:
>  在 C 中，写入错误通过一个负的计数表示，且错误码写在一个易变的位置
>  在 Go 中，`Write` 可以返回一个计数和一个错误
>  `os` 包中，`Write` 方法的签名如下

```go
func (file *File) Write(b []byte) (n int, err error)
```

and as the documentation says, it returns the number of bytes written and a non-nil `error` when `n` `!=` `len(b)`. This is a common style; see the section on error handling for more examples.
>  该方法返回写入的字节数，若 `n! = len(b)` (没有全部写入) 还返回一个非空的 ` error `

A similar approach obviates the need to pass a pointer to a return value to simulate a reference parameter. Here's a simple-minded function to grab a number from a position in a byte slice, returning the number and the next position.
>  类似地，利用多返回值，可以避免传递一个指向返回值的指针来模拟引用参数
>  例如该函数从字节切片中的一个位置获取一个数，返回该数和下一个位置

```go
func nextInt(b []byte, i int) (int, int) {
    for ; i < len(b) && !isDigit(b[i]); i++ {
    }
    x := 0
    for ; i < len(b) && isDigit(b[i]); i++ {
        x = x*10 + int(b[i]) - '0'
    }
    return x, i
}
```

You could use it to scan the numbers in an input slice `b` like this:
>  该函数可以用于遍历切片

```go
    for i := 0; i < len(b); {
        x, i = nextInt(b, i)
        fmt.Println(x)
    }
```

### Named result parameters
The return or result "parameters" of a Go function can be given names and used as regular variables, just like the incoming parameters. When named, they are initialized to the zero values for their types when the function begins; if the function executes a `return` statement with no arguments, the current values of the result parameters are used as the returned values.
>  Go 函数的返回或结果参数可以具有名称，并像常规变量一样使用 (例如输入参数)
>  给定名称时，它们会在函数开始时被初始化为其类型的零值，如果函数执行了没有参数的 `return` 语句，结果参数的当前值就会被返回

The names are not mandatory but they can make code shorter and clearer: they're documentation. If we name the results of `nextInt` it becomes obvious which returned `int` is which.
>  对返回参数给予名称可以让函数的目的更加清晰

```go
func nextInt(b []byte, pos int) (value, nextPos int) {
```

Because named results are initialized and tied to an unadorned return, they can simplify as well as clarify. Here's a version of `io.ReadFull` that uses them well:

```go
func ReadFull(r Reader, buf []byte) (n int, err error) {
    for len(buf) > 0 && err == nil {
        var nr int
        nr, err = r.Read(buf)
        n += nr
        buf = buf[nr:]
    }
    return
}
```

### Defer
Go's `defer` statement schedules a function call (the _deferred_ function) to be run immediately before the function executing the `defer` returns. It's an unusual but effective way to deal with situations such as resources that must be released regardless of which path a function takes to return. The canonical examples are unlocking a mutex or closing a file.
>  Go 的 `defer` 语句将一个函数调用的执行推迟到在执行 `defer` 语句的函数返回之前执行
>  `defer` 用于处理函数无论通过哪条路径返回，都需要释放资源的情况，典型的例子是释放互斥锁和关闭文件

```go
// Contents returns the file's contents as a string.
func Contents(filename string) (string, error) {
    f, err := os.Open(filename)
    if err != nil {
        return "", err
    }
    defer f.Close()  // f.Close will run when we're finished.

    var result []byte
    buf := make([]byte, 100)
    for {
        n, err := f.Read(buf[0:])
        result = append(result, buf[0:n]...) // append is discussed later.
        if err != nil {
            if err == io.EOF {
                break
            }
            return "", err  // f will be closed if we return here.
        }
    }
    return string(result), nil // f will be closed if we return here.
}
```

Deferring a call to a function such as `Close` has two advantages. First, it guarantees that you will never forget to close the file, a mistake that's easy to make if you later edit the function to add a new return path. Second, it means that the close sits near the open, which is much clearer than placing it at the end of the function.
>  将例如 `Close` 这样的调用推迟有两点好处，其一，它确保不会忘记资源释放，即便我们之后会添加新的返回路径；其二，它意味着在代码中 close 和 open 接近，这比将 close 放在函数末端更加清晰

The arguments to the deferred function (which include the receiver if the function is a method) are evaluated when the _defer_ executes, not when the _call_ executes. Besides avoiding worries about variables changing values as the function executes, this means that a single deferred call site can defer multiple function executions. Here's a silly example.
>  被推迟的函数的参数 (如果是方法，也包括了 receiver) 会在 `defer` 执行时被评估，而不是在实际调用时
>  这避免了函数实际执行时，参数改变值的可能性，也意味着单个 `defer` 调用点可以推迟多个函数的执行，示例如下

```go
for i := 0; i < 5; i++ {
    defer fmt.Printf("%d ", i)
}
```

Deferred functions are executed in LIFO order, so this code will cause `4 3 2 1 0` to be printed when the function returns. A more plausible example is a simple way to trace function execution through the program. We could write a couple of simple tracing routines like this:

```go
func trace(s string)   { fmt.Println("entering:", s) }
func untrace(s string) { fmt.Println("leaving:", s) }

// Use them like this:
func a() {
    trace("a")
    defer untrace("a")
    // do something....
}
```

>  被推迟的函数以后进先出的方式执行 (栈)

We can do better by exploiting the fact that arguments to deferred functions are evaluated when the `defer` executes. The tracing routine can set up the argument to the untracing routine. This example:

```go
func trace(s string) string {
    fmt.Println("entering:", s)
    return s
}

func un(s string) {
    fmt.Println("leaving:", s)
}

func a() {
    defer un(trace("a"))
    fmt.Println("in a")
}

func b() {
    defer un(trace("b"))
    fmt.Println("in b")
    a()
}

func main() {
    b()
}
```

prints

```
entering: b
in b
entering: a
in a
leaving: a
leaving: b
```

>  这个例子挺有意思，利用了 `defer` 直接评估被推迟函数参数的性质，简化了代码

For programmers accustomed to block-level resource management from other languages, `defer` may seem peculiar, but its most interesting and powerful applications come precisely from the fact that it's not block-based but function-based. In the section on `panic` and `recover` we'll see another example of its possibilities.
>  对于习惯于从其他语言中使用块级资源管理的程序员来说，`defer` 可能显得有些奇特，但它的最有趣和强大的应用恰恰来自于它不是基于块而是基于函数的这一特点
>  在关于 `panic` 和 `recover` 的部分，我们将看到它可能性的另一个例子

## Data
### Allocation with `new` 
Go has two allocation primitives, the built-in functions `new` and `make`. They do different things and apply to different types, which can be confusing, but the rules are simple. 
>  Go 有两个分配原语，分别是内建函数 `new` 和 `make`

Let's talk about `new` first. It's a built-in function that allocates memory, but unlike its namesakes in some other languages it does not _initialize_ the memory, it only _zeros_ it. That is, `new(T)` allocates zeroed storage for a new item of type `T` and returns its address, a value of type `*T`. In Go terminology, it returns a pointer to a newly allocated zero value of type `T`.
>  `new` 分配内存，它不负责初始化内存，只会清零
>  `new(T)` 会为类型 `T` 的新对象分配一块清零的存储空间，并返回其地址，即类型为 `*T` 的值
>  也就是说，它返回了一个指向类型 `T` 的一个新分配零值的指针

Since the memory returned by `new` is zeroed, it's helpful to arrange when designing your data structures that the zero value of each type can be used without further initialization. This means a user of the data structure can create one with `new` and get right to work. For example, the documentation for `bytes.Buffer` states that "the zero value for `Buffer` is an empty buffer ready to use." Similarly, `sync.Mutex` does not have an explicit constructor or `Init` method. Instead, the zero value for a `sync.Mutex` is defined to be an unlocked mutex.
>  因为 `new` 返回清零的内存，故设计数据结构时，可以考虑让零值具有相对意义，这样就可以直接使用 `new` 的数据
>  例如，`bytes.Buffer` 的文档中说明 `Buffer` 的零值是一个准备就绪的空缓存空间；`sync.Mutex` 没有显式的构造函数或 `Init` 方法，其零值直接定义为一个解锁状态的互斥锁

The zero-value-is-useful property works transitively. Consider this type declaration.
>  zero-value-is-useful 性质是可传递的

```go
type SyncedBuffer struct {
    lock    sync.Mutex
    buffer  bytes.Buffer
}
```

Values of type `SyncedBuffer` are also ready to use immediately upon allocation or just declaration. In the next snippet, both `p` and `v` will work correctly without further arrangement.

```go
p := new(SyncedBuffer)  // type *SyncedBuffer
var v SyncedBuffer      // type  SyncedBuffer
```

### Constructors and composite literals
Sometimes the zero value isn't good enough and an initializing constructor is necessary, as in this example derived from package `os`.

```go
func NewFile(fd int, name string) *File {
    if fd < 0 {
        return nil
    }
    f := new(File)
    f.fd = fd
    f.name = name
    f.dirinfo = nil
    f.nepipe = 0
    return f
}
```

>  有时，使用构造函数也是必要的

There's a lot of boilerplate in there. We can simplify it using a _composite literal_, which is an expression that creates a new instance each time it is evaluated.

```go
func NewFile(fd int, name string) *File {
    if fd < 0 {
        return nil
    }
    f := File{fd, name, nil, 0}
    return &f
}
```

>  可以通过组合字面量简化上述代码，组合字面量是在每次评估时创建一个新实例的表达式
>  上述代码直接用 `f := File{fd, name, nil, 0}` 简化了 `f := new(File), f.fd = fd, ...` 

Note that, unlike in C, it's perfectly OK to return the address of a local variable; the storage associated with the variable survives after the function returns. In fact, taking the address of a composite literal allocates a fresh instance each time it is evaluated, so we can combine these last two lines.
>  和 C 不同，Go 中可以在函数中返回局部变量的地址，该变量对应的存储空间在函数返回后也存在 (因为组合字面量创建的新实例实际也是 `new` 出来的)
>  对组合字面量取地址也会在每次评估时创建新实例，故合并最后两行也是可以的

```go
    return &File{fd, name, nil, 0}
```

The fields of a composite literal are laid out in order and must all be present. However, by labeling the elements explicitly as _field_ `:` _value_ pairs, the initializers can appear in any order, with the missing ones left as their respective zero values. Thus we could say
>  组合字面量的字段需要按序，并且都需要出现
>  可以显式以 `field: value` 的形式指定字段，就不需要按序，且缺少的字段默认为零值

```go
    return &File{fd: fd, name: name}
```

As a limiting case, if a composite literal contains no fields at all, it creates a zero value for the type. The expressions `new(File)` and `&File{}` are equivalent.
>  如果组合字面量没有字段，则为该类型创建零值，故 `new(File)` 和 `&File{}` 等价

Composite literals can also be created for arrays, slices, and maps, with the field labels being indices or map keys as appropriate. In these examples, the initializations work regardless of the values of `Enone`, `Eio`, and `Einval`, as long as they are distinct.

```go
a := [...]string   {Enone: "no error", Eio: "Eio", Einval: "invalid argument"}
s := []string      {Enone: "no error", Eio: "Eio", Einval: "invalid argument"}
m := map[int]string{Enone: "no error", Eio: "Eio", Einval: "invalid argument"}
```

>  数组、切片、映射的创建也支持组合自变量，字段标签为相应的索引或键
>  上述示例中，只要字段标签各不相同，初始化就可以成功，与字段标签的值无关

### Allocation with `make`
Back to allocation. The built-in function `make(T,` _args_ `)` serves a purpose different from `new(T)`. It creates slices, maps, and channels only, and it returns an _initialized_ (not _zeroed_) value of type `T` (not `*T`). The reason for the distinction is that these three types represent, under the covers, references to data structures that must be initialized before use. A slice, for example, is a three-item descriptor containing a pointer to the data (inside an array), the length, and the capacity, and until those items are initialized, the slice is `nil`. For slices, maps, and channels, `make` initializes the internal data structure and prepares the value for use. 
>  内建函数 `make(T, args)` 和 `new(T)` 的目的不同，`make` 仅用于创建切片、映射、通道，并且它返回的是类型 `T` 初始化好的值，不是指向零值的指针 (`*T`)
>  这样设计的原因在于切片、映射、通道本质上引用的数据结构就是需要在使用前初始化好的
>  例如，切片是一个包含三元素的描述符，包含了指向数据的指针、长度、容量，而这三者如果没有初始化，切片就是 `nil`
>  因此 `make` 会为切片、映射、通道初始化好其内部数据结构，并准备好要使用的值

For instance,

```go
make([]int, 10, 100)
```

allocates an array of 100 ints and then creates a slice structure with length 10 and a capacity of 100 pointing at the first 10 elements of the array. (When making a slice, the capacity can be omitted; see the section on slices for more information.) In contrast, `new([]int)` returns a pointer to a newly allocated, zeroed slice structure, that is, a pointer to a `nil` slice value.

>  例如，`make([]int, 10, 100)` 会先分配包含 100 个 `int` 的数组，然后创建一个长度 10，容量 100，指向数组前 10 个元素的切片
>  而 `new([]int)` 仅返回一个指向新分配的，清零的切片结构，即一个指向 `nil` 切片值的指针

These examples illustrate the difference between `new` and `make`.

```go
var p *[]int = new([]int)       // allocates slice structure; *p == nil; rarely useful
var v  []int = make([]int, 100) // the slice v now refers to a new array of 100 ints

// Unnecessarily complex:
var p *[]int = new([]int)
*p = make([]int, 100, 100)

// Idiomatic:
v := make([]int, 100)
```

Remember that `make` applies only to maps, slices and channels and does not return a pointer. To obtain an explicit pointer allocate with `new` or take the address of a variable explicitly.
>  注意 `make` 仅用于映射、切片和通道，且返回值而非指针
>  要获取指针，需要显式取地址

### Arrays
Arrays are useful when planning the detailed layout of memory and sometimes can help avoid allocation, but primarily they are a building block for slices, the subject of the next section. To lay the foundation for that topic, here are a few words about arrays.
>  Arrays 在规划内存的详细布局时非常有用，有时还可以避免分配，但 Arrays 的主要作用是支持 Slices

There are major differences between the ways arrays work in Go and C. In Go,

- Arrays are values. Assigning one array to another copies all the elements.
- In particular, if you pass an array to a function, it will receive a _copy_ of the array, not a pointer to it.
- The size of an array is part of its type. The types `[10]int` and `[20]int` are distinct.

>  C 和 Go 中数组的方式有所不同，在 Go 中
>  - 数组是值，将一个数组赋值给另一个会将所有的元素拷贝
>  - 特别地，如果将一个数组传递给函数，函数将接收该数组的一个值拷贝，而不是指针
>  - 数组的大小是数组类型的一部分，类型 `[10]int` 和 `[20]int` 是不同的

The value property can be useful but also expensive; if you want C-like behavior and efficiency, you can pass a pointer to the array.

```go
func Sum(a *[3]float64) (sum float64) {
    for _, v := range *a {
        sum += v
    }
    return
}

array := [...]float64{7.0, 8.5, 9.1}
x := Sum(&array)  // Note the explicit address-of operator
```

But even this style isn't idiomatic Go. Use slices instead.

>  要让数组像在 C 中一样工作，我们需要传递指向数组的指针
>  但使用数组指针也不是 Go 中的惯例做法，应该使用切片

### Slices
Slices wrap arrays to give a more general, powerful, and convenient interface to sequences of data. Except for items with explicit dimension such as transformation matrices, most array programming in Go is done with slices rather than simple arrays.
>  切片对数组进行了包装，为序列型数据的操作提供了更方便和通用的接口
>  除了具有明确维度的元素 (例如变换矩阵) 以外，Go 中的大多数数组编程都使用切片完成，而不是数组本身

Slices hold references to an underlying array, and if you assign one slice to another, both refer to the same array. If a function takes a slice argument, changes it makes to the elements of the slice will be visible to the caller, analogous to passing a pointer to the underlying array. 
>  切片持有对一个底层数组的引用，如果我们将一个切片赋值给另一个，两个切片都会指向相同的数组
>  如果函数接收切片类型参数，则函数对切片元素的修改对于函数调用者也是可见的，就类似于传递了数组指针

A `Read` function can therefore accept a slice argument rather than a pointer and a count; the length within the slice sets an upper limit of how much data to read. Here is the signature of the `Read` method of the `File` type in package `os`:

```go
func (f *File) Read(buf []byte) (n int, err error)
```

>  因此，`Read` 函数可以接收一个切片类型参数，而不是接收一个数组指针和一个计数器，切片内的长度限制了通过切片能够读取的最大数据长度
>  上述是 `os` 包中 `File` 类型的 `Read` 方法声明

The method returns the number of bytes read and an error value, if any. To read into the first 32 bytes of a larger buffer `buf`, _slice_ (here used as a verb) the buffer.

```go
    n, err := f.Read(buf[0:32])
```

>  该方法返回读取的字节数量和一个错误值 (如果有)
>  如果我们要读取一个大缓冲区 `buf` 的前 32 个字节，就对其进行切片，如上所示

Such slicing is common and efficient. In fact, leaving efficiency aside for the moment, the following snippet would also read the first 32 bytes of the buffer.

```go
    var n int
    var err error
    for i := 0; i < 32; i++ {
        nbytes, e := f.Read(buf[i:i+1])  // Read one byte.
        n += nbytes
        if nbytes == 0 || e != nil {
            err = e
            break
        }
    }
```

The length of a slice may be changed as long as it still fits within the limits of the underlying array; just assign it to a slice of itself. The _capacity_ of a slice, accessible by the built-in function `cap`, reports the maximum length the slice may assume.
>  只要切片仍然处于底层数组的范围内，其长度就可以任意改变 (通过对自己赋值进行改变)
>  切片的容量可以通过内建函数 `cap` 访问，它表示了切片长度的最大值

 Here is a function to append data to a slice. If the data exceeds the capacity, the slice is reallocated. The resulting slice is returned. The function uses the fact that `len` and `cap` are legal when applied to the `nil` slice, and return 0.

```go
func Append(slice, data []byte) []byte {
    l := len(slice)
    if l + len(data) > cap(slice) {  // reallocate
        // Allocate double what's needed, for future growth.
        newSlice := make([]byte, (l+len(data))*2)
        // The copy function is predeclared and works for any slice type.
        copy(newSlice, slice)
        slice = newSlice
    }
    slice = slice[0:l+len(data)]
    copy(slice[l:], data)
    return slice
}
```

>  上例展示了一个向切片追加数据的函数
>  如果加入数据后，`len` 超过了 `cap` ，则我们需要重新分配一个 slice，并返回新分配的 slice

We must return the slice afterwards because, although `Append` can modify the elements of `slice`, the slice itself (the run-time data structure holding the pointer, length, and capacity) is passed by value.
>  注意，函数 `Append` 可能会创建新的 slice，故我们必须返回 slice
>  尽管 `Append` 会原地修改 `slice` 中的元组，但切片本身 (保存了指针、长度、容量的运行时结构) 是按值传递的

The idea of appending to a slice is so useful it's captured by the `append` built-in function. To understand that function's design, though, we need a little more information, so we'll return to it later.

### Two-dimensional slices
Go's arrays and slices are one-dimensional. To create the equivalent of a 2D array or slice, it is necessary to define an array-of-arrays or slice-of-slices, like this:

```go
type Transform [3][3]float64  // A 3x3 array, really an array of arrays.
type LinesOfText [][]byte     // A slice of byte slices.
```

>  要构造二维数组或切片，我们需要定义数组的数组或切片的切片，如上所示

Because slices are variable-length, it is possible to have each inner slice be a different length. That can be a common situation, as in our `LinesOfText` example: each line has an independent length.
>  因为切片是变长的，故切片的切片中，每个内部切片可以是不同的长度
>  例如在下例中 `LinesofText` 是存储 `[]byte` 的切片，而每个 `[]byte` 都有各自的长度

```go
text := LinesOfText{
    []byte("Now is the time"),
    []byte("for all good gophers"),
    []byte("to bring some fun to the party."),
}
```

Sometimes it's necessary to allocate a 2D slice, a situation that can arise when processing scan lines of pixels, for instance. There are two ways to achieve this. One is to allocate each slice independently; the other is to allocate a single array and point the individual slices into it. Which to use depends on your application. If the slices might grow or shrink, they should be allocated independently to avoid overwriting the next line; if not, it can be more efficient to construct the object with a single allocation.
>  要直接分配二维切片有两种方法
>  一种是独立地为每个切片分配内存
>  另一种是先分配完整的数组，再将各个切片指向该数组
>  如果切片可能增长或缩小，则各个切片应该独立分配；否则，可以使用单次分配构建数组更方便

For reference, here are sketches of the two methods. First, a line at a time:
>  两种方法的简要示例如下，第一个例子是独立为每个切片执行分配

```go
// Allocate the top-level slice.
picture := make([][]uint8, YSize) // One row per unit of y.
// Loop over the rows, allocating the slice for each row.
for i := range picture {
    picture[i] = make([]uint8, XSize)
}
```

And now as one allocation, sliced into lines:
>  第二个例子是仅使用一次分配，将其切片为各个行

```go
// Allocate the top-level slice, the same as before.
picture := make([][]uint8, YSize) // One row per unit of y.
// Allocate one large slice to hold all the pixels.
pixels := make([]uint8, XSize*YSize) // Has type []uint8 even though picture is [][]uint8.
// Loop over the rows, slicing each row from the front of the remaining pixels slice.
for i := range picture {
    picture[i], pixels = pixels[:XSize], pixels[XSize:]
}
```

### Maps
Maps are a convenient and powerful built-in data structure that associate values of one type (the _key_) with values of another type (the _element_ or _value_). The key can be of any type for which the equality operator is defined, such as integers, floating point and complex numbers, strings, pointers, interfaces (as long as the dynamic type supports equality), structs and arrays. Slices cannot be used as map keys, because equality is not defined on them. 
>  映射将一种类型的值关联到另一种类型的值
>  键可以是任意定义了 `=` 算子的类型 1，例如整形、浮点、复数、字符串、指针、接口、结构体、数组
>  切片不能用作键，因为切片类型没有定义 `=` 

Like slices, maps hold references to an underlying data structure. If you pass a map to a function that changes the contents of the map, the changes will be visible in the caller.
>  和切片类似，映射本质上保存了指向底层数据接口的引用，向函数传递映射时，函数中对映射内容的改变对于函数调用者也是可见的

Maps can be constructed using the usual composite literal syntax with colon-separated key-value pairs, so it's easy to build them during initialization.
>  可以使用常用的复合字面量语法 (元素是 `:` 分隔的键值对) 来构造映射

```go
var timeZone = map[string]int{
    "UTC":  0*60*60,
    "EST": -5*60*60,
    "CST": -6*60*60,
    "MST": -7*60*60,
    "PST": -8*60*60,
}
```

Assigning and fetching map values looks syntactically just like doing the same for arrays and slices except that the index doesn't need to be an integer.
>  对映射值的获取和赋值在语法上对数组/切片元素的获取和赋值一样，差异仅在于映射的索引不必是整数

```go
offset := timeZone["EST"]
```

An attempt to fetch a map value with a key that is not present in the map will return the zero value for the type of the entries in the map. For instance, if the map contains integers, looking up a non-existent key will return `0`. A set can be implemented as a map with value type `bool`. Set the map entry to `true` to put the value in the set, and then test it by simple indexing.
>  如果映射中没有传入的键，将返回映射值类型的零值

```go
attended := map[string]bool{
    "Ann": true,
    "Joe": true,
    ...
}

if attended[person] { // will be false if person is not in the map
    fmt.Println(person, "was at the meeting")
}
```

Sometimes you need to distinguish a missing entry from a zero value. Is there an entry for `"UTC"` or is that 0 because it's not in the map at all? You can discriminate with a form of multiple assignment.
>  有时我们需要区分缺失条目的情况的条目的值本身就是零值的情况
>  此时我们可以使用多重赋值的形式来区分，形式如下

```go
var seconds int
var ok bool
seconds, ok = timeZone[tz]
```

For obvious reasons this is called the “comma ok” idiom. In this example, if `tz` is present, `seconds` will be set appropriately and `ok` will be true; if not, `seconds` will be set to zero and `ok` will be false. 
>  该形式被称为 "comma ok" 惯例
>  在上例中，如果 `tz` 存在，则 `ok` 将是 `true` ，否则是 `false`

Here's a function that puts it together with a nice error report:

```go
func offset(tz string) int {
    if seconds, ok := timeZone[tz]; ok {
        return seconds
    }
    log.Println("unknown time zone:", tz)
    return 0
}
```

To test for presence in the map without worrying about the actual value, you can use the [blank identifier](https://go.dev/doc/effective_go#blank) (`_`) in place of the usual variable for the value.
>  要测试某个键是否存在于映射中，而不关心具体的值，可以使用空白标识符 `_` 

```go
_, present := timeZone[tz]
```

To delete a map entry, use the `delete` built-in function, whose arguments are the map and the key to be deleted. It's safe to do this even if the key is already absent from the map.
>  要删除映射条目，使用 `delete` 内建函数，即便映射中没有要删除的条目，也是安全的

```go
delete(timeZone, "PDT")  // Now on Standard Time
```

### Printing
Formatted printing in Go uses a style similar to C's `printf` family but is richer and more general. The functions live in the `fmt` package and have capitalized names: `fmt.Printf`, `fmt.Fprintf`, `fmt.Sprintf` and so on. The string functions (`Sprintf` etc.) return a string rather than filling in a provided buffer.
>  Go 中的格式化打印函数定义在 `fmt` 包中
>  和 C 中不同，字符串函数 (`Sprintf` 等) 返回一个字符串，而不是填充提供的缓冲区

You don't need to provide a format string. For each of `Printf`, `Fprintf` and `Sprintf` there is another pair of functions, for instance `Print` and `Println`. These functions do not take a format string but instead generate a default format for each argument. The `Println` versions also insert a blank between arguments and append a newline to the output while the `Print` versions add blanks only if the operand on neither side is a string. 
> `Printf, Fprintf, Sprintf` 都有各自对应的一对类似 `Print, Println` 的函数 (例如 `Fprint`)，这些函数不接受格式字符串，而是为每个参数生成默认格式，` Println ` 会在参数之间插入空格，并且在末尾插入换行符，` Print ` 则在两边操作数都不是字符串时才添加空格

In this example each line produces the same output.

```go
fmt.Printf("Hello %d\n", 23)
fmt.Fprint(os.Stdout, "Hello ", 23, "\n")
fmt.Println("Hello", 23) 
fmt.Println(fmt.Sprint("Hello ", 23))
```

The formatted print functions `fmt.Fprint` and friends take as a first argument any object that implements the `io.Writer` interface; the variables `os.Stdout` and `os.Stderr` are familiar instances.
>  格式化打印函数 `fmt.Fprint` 和其相关函数的第一个参数是实现了 `io.Writer` 接口的对象，例如 `os.Stdout, os.Stderr`

Here things start to diverge from C. First, the numeric formats such as `%d` do not take flags for signedness or size; instead, the printing routines use the type of the argument to decide these properties.

```go
var x uint64 = 1<<64 - 1
fmt.Printf("%d %x; %d %x\n", x, x, int64(x), int64(x))
```

prints

```go
18446744073709551615 ffffffffffffffff; -1 -1
```

>  Go 中的格式化形式和 C 中有所不同
>  首先，数值格式例如 `%d` 不接收用于表示符号或大小的标志，打印函数会根据参数的类型自行决定这些性质

If you just want the default conversion, such as decimal for integers, you can use the catchall format `%v` (for “value”); the result is exactly what `Print` and `Println` would produce. Moreover, that format can print _any_ value, even arrays, slices, structs, and maps. 
>  如果只想用默认的转换，例如整数的默认转换是十进制，可以使用通用格式符 `%v` ，其结果将和 `Print` , `Println` 输出的结果完全一致
>  此外，该格式可以打印任意值，包括了数组、切片、结构体、映射

Here is a print statement for the time zone map defined in the previous section.

```go
fmt.Printf("%v\n", timeZone)  // or just fmt.Println(timeZone)
```

which gives output:

```go
map[CST:-21600 EST:-18000 MST:-25200 PST:-28800 UTC:0]
```

For maps, `Printf` and friends sort the output lexicographically by key.
>  `Printf` 和其相关函数打印 `map` 时，会按词表序排序 key

When printing a struct, the modified format `%+v` annotates the fields of the structure with their names, and for any value the alternate format `%#v` prints the value in full Go syntax.
>  打印结构体时，格式符 `%+v` 会为结构体的字段注释字段名，`%#v` 会以完成 Go 语法打印值

```go
type T struct {
    a int
    b float64
    c string
}
t := &T{ 7, -2.35, "abc\tdef" }
fmt.Printf("%v\n", t)
fmt.Printf("%+v\n", t)
fmt.Printf("%#v\n", t)
fmt.Printf("%#v\n", timeZone)
```

prints

```go
&{7 -2.35 abc   def}
&{a:7 b:-2.35 c:abc     def}
&main.T{a:7, b:-2.35, c:"abc\tdef"}
map[string]int{"CST":-21600, "EST":-18000, "MST":-25200, "PST":-28800, "UTC":0}
```

(Note the ampersands.) That quoted string format is also available through `%q` when applied to a value of type `string` or `[]byte`. The alternate format `%#q` will use backquotes instead if possible. (The `%q` format also applies to integers and runes, producing a single-quoted rune constant.) Also, `%x` works on strings, byte arrays and byte slices as well as on integers, generating a long hexadecimal string, and with a space in the format (`% x`) it puts spaces between the bytes.
>  当应用于 `string, []byte` 类型的值时，带引号的字符串格式可以通过 ` %q ` 获得，`%#q` 会在可能的情况下使用反引号代替 (`%q` 也适用于整数和 runes，生成单引号括起的 rune 常量)
>  `%x` 适用于字符串、字节数组、字节切片和整数，它生成一个长的十六进制字符串，`% x` 会在字节之间插入空格

Another handy format is `%T`, which prints the _type_ of a value.

```go
fmt.Printf("%T\n", timeZone)
```

prints

```go
map[string]int
```

>  `%T` 会打印值的类型

If you want to control the default format for a custom type, all that's required is to define a method with the signature `String() string` on the type. For our simple type `T`, that might look like this.

```go
func (t *T) String() string {
    return fmt.Sprintf("%d/%g/%q", t.a, t.b, t.c)
}
fmt.Printf("%v\n", t)
```

to print in the format

```go
7/-2.35/"abc\tdef"
```

>  如果要控制自定义类型的默认格式，我们需要在该类型上定义一个签名为 `String() string` 的方法

(If you need to print _values_ of type `T` as well as pointers to `T`, the receiver for `String` must be of value type; this example used a pointer because that's more efficient and idiomatic for struct types. See the section below on [pointers vs. value receivers](https://go.dev/doc/effective_go#pointers_vs_values) for more information.)
>  如果需要打印类型为 `T` 的值以及指向 `T` 的指针，则 `String() string` 方法的接收者必须是值类型，该例使用了指针类型，因为对于结构体类型来说，这更加高效且符合惯例

Our `String` method is able to call `Sprintf` because the print routines are fully reentrant and can be wrapped this way. There is one important detail to understand about this approach, however: don't construct a `String` method by calling `Sprintf` in a way that will recur into your `String` method indefinitely. This can happen if the `Sprintf` call attempts to print the receiver directly as a string, which in turn will invoke the method again. It's a common and easy mistake to make, as this example shows.
>  可以在该 `String` 方法中调用 `Sprintf` ，因为打印函数是完全可重复的，并且可以通过这种方式进行封装
>  但注意不要导致 `String` 无限递归，例如 `Sprintf` 调用尝试直接将 receiver 打印为字符串，这进而会导致 `String` 又被调用，例如下面的错误示范

```go
type MyString string

func (m MyString) String() string {
    return fmt.Sprintf("MyString=%s", m) // Error: will recur forever.
}
```

It's also easy to fix: convert the argument to the basic string type, which does not have the method.
>  处理这个问题也很简单，我们将参数转化为基本的 `string` 类型，基本的 `string` 没有定义 `String` 方法

```go
type MyString string
func (m MyString) String() string {
    return fmt.Sprintf("MyString=%s", string(m)) // OK: note conversion.
}
```

In the [initialization section](https://go.dev/doc/effective_go#initialization) we'll see another technique that avoids this recursion.

Another printing technique is to pass a print routine's arguments directly to another such routine. 
>  另一个打印技术是将一个打印函数的参数直接传递给另一个函数

The signature of `Printf` uses the type `...interface{}` for its final argument to specify that an arbitrary number of parameters (of arbitrary type) can appear after the format.
>  `Printf` 的签名中，其最后一个参数 `v` 的类型是 `...interface{}`，这指定了在 ` format ` 参数之后，可以有任意数量，任意类型的参数

```go
func Printf(format string, v ...interface{}) (n int, err error) {
```

Within the function `Printf`, `v` acts like a variable of type `[]interface{}` but if it is passed to another variadic function, it acts like a regular list of arguments. 
>  在函数 `Printf` 中，`v` 的行为类似于类型为 `[]interface{}` 的变量，但当他被传递给另一个可变参数函数时，它的行为就像一个普通的参数列表

Here is the implementation of the function `log.Println` we used above. It passes its arguments directly to `fmt.Sprintln` for the actual formatting.
>  以下是上面使用的函数 `log.Println` 的实现，它直接将其参数传递给 `fmt.Sprintln` 来执行格式化

```go
// Println prints to the standard logger in the manner of fmt.Println.
func Println(v ...interface{}) {
    std.Output(2, fmt.Sprintln(v...))  // Output takes parameters (int, string)
}
```

We write `...` after `v` in the nested call to `Sprintln` to tell the compiler to treat `v` as a list of arguments; otherwise it would just pass `v` as a single slice argument.
>  我们在调用 `Sprintln` 时在参数 ` v ` 后写上 `...` 来告诉编译器将 `v` 视为一组参数，否则编译器会将 `v` 视作单个切片参数传递

There's even more to printing than we've covered here. See the `godoc` documentation for package `fmt` for the details.

By the way, a `...` parameter can be of a specific type, for instance `...int` for a min function that chooses the least of a list of integers:
>  另外，`...` 参数可以是任意类型，例如 `...int` 等

```go
func Min(a ...int) int {
    min := int(^uint(0) >> 1)  // largest int
    for _, i := range a {
        if i < min {
            min = i
        }
    }
    return min
}
```

### Append
Now we have the missing piece we needed to explain the design of the `append` built-in function. The signature of `append` is different from our custom `Append` function above. Schematically, it's like this:
>  我们现在可以解释内建 `append` 函数的设计
>  `append` 的签名和之前自定义的 `Append` 函数不同，其签名类似：

```go
func append(slice []T, elements ...T) []T
```

where _T_ is a placeholder for any given type. You can't actually write a function in Go where the type `T` is determined by the caller. That's why `append` is built in: it needs support from the compiler.
>  其中 `T` 是**任意**给定类型的占位符
>  在 Go 中，我们实际上无法编写一个函数，让调用者决定该函数的参数类型 `T` ，故 `append` 函数需要是内建函数：它需要编译器的支持

What `append` does is append the elements to the end of the slice and return the result. The result needs to be returned because, as with our hand-written `Append`, the underlying array may change. 
>  `append` 将元素追加到切片的末尾，并返回结果 (切片)，结果必须返回，因为切片的底层数组可能会改变

This simple example

```go
x := []int{1,2,3}
x = append(x, 4, 5, 6)
fmt.Println(x)
```

prints `[1 2 3 4 5 6]`. So `append` works a little like `Printf`, collecting an arbitrary number of arguments.
>  `append` 可以接收任意数量的参数

But what if we wanted to do what our `Append` does and append a slice to a slice? Easy: use `...` at the call site, just as we did in the call to `Output` above. This snippet produces identical output to the one above.
>  如果我们需要为一个切片附加切片，我们在调用时需要传入 `...` 
>  例如下例中，这使得编译器将 `y` 视作一组参数，而不是单个切片参数传递

```go
x := []int{1,2,3}
y := []int{4,5,6}
x = append(x, y...)
fmt.Println(x)
```

Without that `...`, it wouldn't compile because the types would be wrong; `y` is not of type `int`.
>  如果没有 `...` ，编译会失败，因为类型不匹配 (`y` 不是 `int` 类型)

## Initialization
Although it doesn't look superficially very different from initialization in C or C++, initialization in Go is more powerful. Complex structures can be built during initialization and the ordering issues among initialized objects, even among different packages, are handled correctly.
>  Go 中的初始化和 C 中的初始化表面上看没有太大区别，但 Go 中的初始化功能更加强大
>  在 Go 中，复杂的结构可以在初始化过程中构建，并且初始化对象间的顺序问题 (即便是在不同包之间的对象) 也能正确处理

### Constants
Constants in Go are just that—constant. They are created at compile time, even when defined as locals in functions, and can only be numbers, characters (runes), strings or booleans. Because of the compile-time restriction, the expressions that define them must be constant expressions, evaluatable by the compiler. For instance, `1<<3` is a constant expression, while `math.Sin(math.Pi/4)` is not because the function call to `math.Sin` needs to happen at run time.
>  Go 中的常量都在编译时创建，即便是在函数中作为局部变量定义也是如此，并且常量只能是数字、字符 (rune)、字符串或布尔值
>  因为是在编译时创建，定义了常量的表达式只能是可以被编译器评估的常量表达式
>  例如 `1<<3` 是一个常量表达式，而 `math.Sin(math.Pi/4)` 则不是，因为对 `math.Sin` 的函数调用需要在运行时执行

>  编译的任务是将源码转化为机器码，故只能处理简单、确定的表达式
>  像 `math.Sin` 这样的数学计算函数，需要依赖运行时的资源，如 CPU 浮点运算单元、内存等进行执行，编译时实现会增加编译器的复杂性和编译时间，收益有限

In Go, enumerated constants are created using the `iota` enumerator. Since `iota` can be part of an expression and expressions can be implicitly repeated, it is easy to build intricate sets of values.
>  Go 中的枚举常量通过 `iota` 枚举器创建
>  因为 `iota` 可以作为表达式的一部分，并且表达式可以隐式重复，故可以很容易构造复杂的值集合

```go
type ByteSize float64

const (
    _           = iota // ignore first value by assigning to blank identifier
    KB ByteSize = 1 << (10 * iota)
    MB
    GB
    TB
    PB
    EB
    ZB
    YB
)
```

The ability to attach a method such as `String` to any user-defined type makes it possible for arbitrary values to format themselves automatically for printing. Although you'll see it most often applied to structs, this technique is also useful for scalar types such as floating-point types like `ByteSize`.
>  我们可以为任意用户定义的类型实现例如 `String` 这样的方法，使得任意值都可以被自动格式化以供打印
>  不仅仅是对于自定义的结构体类型，对于标量类型，例如 `ByteSize` 这样的浮点数类型，我们也可以定义它们的 `String` 方法

```go
func (b ByteSize) String() string {
    switch {
    case b >= YB:
        return fmt.Sprintf("%.2fYB", b/YB)
    case b >= ZB:
        return fmt.Sprintf("%.2fZB", b/ZB)
    case b >= EB:
        return fmt.Sprintf("%.2fEB", b/EB)
    case b >= PB:
        return fmt.Sprintf("%.2fPB", b/PB)
    case b >= TB:
        return fmt.Sprintf("%.2fTB", b/TB)
    case b >= GB:
        return fmt.Sprintf("%.2fGB", b/GB)
    case b >= MB:
        return fmt.Sprintf("%.2fMB", b/MB)
    case b >= KB:
        return fmt.Sprintf("%.2fKB", b/KB)
    }
    return fmt.Sprintf("%.2fB", b)
}
```

The expression `YB` prints as `1.00YB`, while `ByteSize(1e13)` prints as `9.09TB`.
>  实现了上述方法后，表达式 `YB` 将被打印为 `1.00YB`，表达式 `ByteSize(1e13)` 将被打印为 `9.09TB`

The use here of `Sprintf` to implement `ByteSize` 's `String` method is safe (avoids recurring indefinitely) not because of a conversion but because it calls `Sprintf` with `%f`, which is not a string format: `Sprintf` will only call the `String` method when it wants a string, and `%f` wants a floating-point value.
>  该李忠，我们使用 `Sprintf` 实现 `ByteSize` 的 `String` 方法是安全的 (不会导致无限递归)，这并不是因为类型转换，而是我们使用 `%f` 调用 `Sprintf` (`%f` 不是字符串格式，即不是 `%s`): `Sprintf` 只有在需要字符串时才会调用对象的 `String` 方法，而 `%f` 需要的是一个浮点数值

### Variables
Variables can be initialized just like constants but the initializer can be a general expression computed at run time.
>  变量可以像常量一样初始化，并且其初始化值可以是一个在运行时计算的通用表达式

```go
var (
    home   = os.Getenv("HOME")
    user   = os.Getenv("USER")
    gopath = os.Getenv("GOPATH")
)
```

### The init function
Finally, each source file can define its own niladic `init` function to set up whatever state is required. (Actually each file can have multiple `init` functions.) And finally means finally: `init` is called after all the variable declarations in the package have evaluated their initializers, and those are evaluated only after all the imported packages have been initialized.
>  每个源文件都可以定义自己的无参数 `init` 函数，用于设置所需的任何状态 (实际上每个文件可以有多个 `init` 函数)
>  `init` 会在包中的所有变量声明评估完它们的初始化值后才被调用，而包中的所有变量初始化值的评估会在所有被导入的包完成初始化后才执行

Besides initializations that cannot be expressed as declarations, a common use of `init` functions is to verify or repair correctness of the program state before real execution begins.
>  除了用于处理无法以声明形式表达的初始化以外，`init` 函数的另一个常见用途时在程序的实际执行开始之前验证并修复程序状态的正确性

```go
func init() {
    if user == "" {
        log.Fatal("$USER not set")
    }
    if home == "" {
        home = "/home/" + user
    }
    if gopath == "" {
        gopath = home + "/go"
    }
    // gopath may be overridden by --gopath flag on command line.
    flag.StringVar(&gopath, "gopath", gopath, "override default GOPATH")
}
```

## Methods
### Pointers vs. Values 
As we saw with `ByteSize`, methods can be defined for any named type (except a pointer or an interface); the receiver does not have to be a struct.
>  可以为任意命名类型 (指针和接口除外) 定义方法，接收者不必一定是结构体

In the discussion of slices above, we wrote an `Append` function. We can define it as a method on slices instead. To do this, we first declare a named type to which we can bind the method, and then make the receiver for the method a value of that type.

```go
type ByteSlice []byte

func (slice ByteSlice) Append(data []byte) []byte {
    // Body exactly the same as the Append function defined above.
}
```

>  要定义方法，我们首先需要声明一个绑定该方法的类型，然后将该方法的接收者设定为该类型的值

This still requires the method to return the updated slice. We can eliminate that clumsiness by redefining the method to take a _pointer_ to a `ByteSlice` as its receiver, so the method can overwrite the caller's slice.

```go
func (p *ByteSlice) Append(data []byte) {
    slice := *p
    // Body as above, without the return.
    *p = slice
}
```

> 方法的接收者可以为一个指向该类型的指针值，例如上述例子，这样方法就可以覆盖写调用者的切片

In fact, we can do even better. If we modify our function so it looks like a standard `Write` method, like this,

```go
func (p *ByteSlice) Write(data []byte) (n int, err error) {
    slice := *p
    // Again as above.
    *p = slice
    return len(data), nil
}
```

then the type `*ByteSlice` satisfies the standard interface `io.Writer`, which is handy. For instance, we can print into one.

>  我们为自定义类型定义满足特定签名的方法，可以使其实现特定的接口
>  例如上例中的类型 `*ByteSlice` 通过实现 `Write` 满足了标准接口 ` io.Writer ` 

```go
    var b ByteSlice
    fmt.Fprintf(&b, "This hour has %d days\n", 7)
```

We pass the address of a `ByteSlice` because only `*ByteSlice` satisfies `io.Writer`. The rule about pointers vs. values for receivers is that value methods can be invoked on pointers and values, but pointer methods can only be invoked on pointers.
>  因为 `*ByteSlice` 实现了 `io.Writer` ，故在上例中，`fmt.Fprintf` 的第一个参数我们可以传入 `&b` ，注意只有指针类型实现了接口，故需要传递地址
>  关于指针接收者和值接收者的规则是：接收者是值类型的方法可以被指针和值类型调用，接收者是指针类型的方法只能被指针类型调用

This rule arises because pointer methods can modify the receiver; invoking them on a value would cause the method to receive a copy of the value, so any modifications would be discarded. The language therefore disallows this mistake. There is a handy exception, though. When the value is addressable, the language takes care of the common case of invoking a pointer method on a value by inserting the address operator automatically. In our example, the variable `b` is addressable, so we can call its `Write` method with just `b.Write`. The compiler will rewrite that to `(&b).Write` for us.
>  这一规则产生的原因在于指针方法可以修改接收者，如果值可以调用这些方法，那么方法实际上接收到的是值的副本，故任意的修改都会被丢弃，这不符合方法在定义时所期待的功能语义，因此在语言本身上禁止了这样的调用
>  但是有一个例外情况，当值是可寻址的时候，Go 语言会自动插入取地址符，从而处理通过值调用指针方法的情况
>  在上述例子中，变量 `b` 是可寻址的，故我们其实可以直接调用 `b.Write` ，编译器会将其重写为 `(&b).Write`

By the way, the idea of using `Write` on a slice of bytes is central to the implementation of `bytes.Buffer`.
>  为字节切片类型实现 `Write` 方法的思想是 `bytes.Buffer` 实现的核心

## Interfaces and other types
### Interfaces 
Interfaces in Go provide a way to specify the behavior of an object: if something can do _this_, then it can be used _here_. We've seen a couple of simple examples already; custom printers can be implemented by a `String` method while `Fprintf` can generate output to anything with a `Write` method. Interfaces with only one or two methods are common in Go code, and are usually given a name derived from the method, such as `io.Writer` for something that implements `Write`.
>  Go 中的接口提供了一种指定对象行为的方式：如果某个对象可以实现这个，则它可以在这里被使用
>  我们已经见过了许多实例：通过实现 `String` ，可以进行打印；通过实现 `Write` ，可以进行 `Fprintf`
>  Go 代码中，仅包含一个或两个的接口很常见，并且接口通常根据方法命名，例如实现 `Writer` 方法的对象被称为 ` io.Writer `

A type can implement multiple interfaces. For instance, a collection can be sorted by the routines in package `sort` if it implements `sort.Interface`, which contains `Len()`, `Less(i, j int) bool`, and `Swap(i, j int)`, and it could also have a custom formatter. 
>  一个类型可以实现多个接口
>  例如，如果一个集合类型实现了 `sort.Interface` 接口 (包含了 `Len(), Less(i,j int) bool, Swap(i, j int)` 方法)，则该集合就可以借助 `sort` 包中的例程进行排序，而该集合继续可以实现其他接口以拥有自定义的格式化器

In this contrived example `Sequence` satisfies both.

```go
type Sequence []int

// Methods required by sort.Interface.
func (s Sequence) Len() int {
    return len(s)
}
func (s Sequence) Less(i, j int) bool {
    return s[i] < s[j]
}
func (s Sequence) Swap(i, j int) {
    s[i], s[j] = s[j], s[i]
}

// Copy returns a copy of the Sequence.
func (s Sequence) Copy() Sequence {
    copy := make(Sequence, 0, len(s))
    return append(copy, s...)
}

// Method for printing - sorts the elements before printing.
func (s Sequence) String() string {
    s = s.Copy() // Make a copy; don't overwrite argument.
    sort.Sort(s)
    str := "["
    for i, elem := range s { // Loop is O(N²); will fix that in next example.
        if i > 0 {
            str += " "
        }
        str += fmt.Sprint(elem)
    }
    return str + "]"
}
```

### Conversions
The `String` method of `Sequence` is recreating the work that `Sprint` already does for slices. (It also has complexity O(N²), which is poor.) We can share the effort (and also speed it up) if we convert the `Sequence` to a plain `[]int` before calling `Sprint`.

```go
func (s Sequence) String() string {
    s = s.Copy()
    sort.Sort(s)
    return fmt.Sprint([]int(s))
}
```

>  上例中，`Sequence` 类型实现的 `String` 方法有点冗余，因为它在重复实现已经为切片类型实现的 `Sprint` 工作
>  如果我们可以在调用 `Sprint` 方法前，将 ` Sequence ` 转换为 `[]int` 类型，就可以复用已经为 `[]int` 实现的 `Sprint`

This method is another example of the conversion technique for calling `Sprintf` safely from a `String` method. Because the two types (`Sequence` and `[]int`) are the same if we ignore the type name, it's legal to convert between them. The conversion doesn't create a new value, it just temporarily acts as though the existing value has a new type. (There are other legal conversions, such as from integer to floating point, that do create a new value.)
>  通过类型转换也是安全地在 `String` 方法中调用 `Sprintf` 的一个技巧
>  因为 `Sequence` 和 `[]int` 类型本质上是相同的，故二者之间的转换完全合法，这种转换不会创建新值，只是暂时地使现有的值表现为具有新类型
>  (有其他合法的转换，例如从整数到浮点数的转换，这些转换确实会创建新值)

It's an idiom in Go programs to convert the type of an expression to access a different set of methods. As an example, we could use the existing type `sort.IntSlice` to reduce the entire example to this:
>  将表达式的类型转换为另一个类型，以利用一些方法，这样的技巧在 Go 中是惯用的
>  例如，我们可以进一步利用现有的类型 `sort.IntSlice` 简化之前的示例

```go
type Sequence []int

// Method for printing - sorts the elements before printing
func (s Sequence) String() string {
    s = s.Copy()
    sort.IntSlice(s).Sort()
    return fmt.Sprint([]int(s))
}
```

Now, instead of having `Sequence` implement multiple interfaces (sorting and printing), we're using the ability of a data item to be converted to multiple types (`Sequence`, `sort.IntSlice` and `[]int`), each of which does some part of the job. That's more unusual in practice but can be effective.
>  现在，我们没有为 `Sequence` 实现接口，直接利用了数据类型之间的转换 (`Sequence, sort.IntSlice, []int` ) 实现了功能，每个类型都负责完成一部分工作
>  这种方法在实践中不太常见，但可以非常有效

### Interface conversions and type assertions
[Type switches](https://go.dev/doc/effective_go#type_switch) are a form of conversion: they take an interface and, for each case in the switch, in a sense convert it to the type of that case.
>  `type switch` 是一种转换形式：它们接收一个接口，并在 `switch` 中的每个 `case` 中，以某种方式将其转化为该 `case` 的类型

Here's a simplified version of how the code under `fmt.Printf` turns a value into a string using a type switch. If it's already a string, we want the actual string value held by the interface, while if it has a `String` method we want the result of calling the method.
>  以下是 `fmt.Printf` 中使用 type switch 将值转化为字符串的简化代码
>  如果值已经是 `string` 类型，就直接返回值，如果它有一个 `String` 方法，则调用该方法并返回其结果

```go
type Stringer interface {
    String() string
}

var value interface{} // Value provided by caller.
switch str := value.(type) {
case string:
    return str
case Stringer:
    return str.String()
}
```

The first case finds a concrete value; the second converts the interface into another interface. It's perfectly fine to mix types this way.

What if there's only one type we care about? If we know the value holds a `string` and we just want to extract it? A one-case type switch would do, but so would a _type assertion_. A type assertion takes an interface value and extracts from it a value of the specified explicit type. The syntax borrows from the clause opening a type switch, but with an explicit type rather than the `type` keyword:
>  考虑如果只有一种类型是我们关心的，例如我们知道值保存了一个 `string` ，并且只是想要提取它
>  一个 one-case switch 可以实现这一点，类型断言也可以实现这一点
>  类型断言接收一个接口值，然后从接口值中提取指定的显式类型的值，其语法借鉴了 type switch 的 opening clause，但使用的是显式类型而不是 `type` 关键字：

```go
value.(typeName)
```

and the result is a new value with the static type `typeName`. That type must either be the concrete type held by the interface, or a second interface type that the value can be converted to. 
>  类型断言的结果就是一个具有静态类型 `typeName` 的值
>  类型 `typeName` 必须要么是接口存储的具体类型，要么是值可以转换为的第二个接口类型

To extract the string we know is in the value, we could write:
>  我们知道 `value` 中存储了 `string`，要提取这个 `string` 值，可以这样写：

```go
str := value.(string)
```

But if it turns out that the value does not contain a string, the program will crash with a run-time error. To guard against that, use the "comma, ok" idiom to test, safely, whether the value is a string:
>  但如果 `value` 并不包含一个 `string` 值，程序将出现运行时错误并崩溃
>  为了避免这种情况，可以使用 "comma, ok" 惯例来测试 `value` 是否是 `string`

```go
str, ok := value.(string)
if ok {
    fmt.Printf("string value is: %q\n", str)
} else {
    fmt.Printf("value is not a string\n")
}
```

If the type assertion fails, `str` will still exist and be of type string, but it will have the zero value, an empty string.
>  上例中，如果类型断言失败，`str` 将是类型 `string` 的零值，即空字符串

As an illustration of the capability, here's an `if`-`else` statement that's equivalent to the type switch that opened this section.

```go
if str, ok := value.(string); ok {
    return str
} else if str, ok := value.(Stringer); ok {
    return str.String()
}
```

### Generality
If a type exists only to implement an interface and will never have exported methods beyond that interface, there is no need to export the type itself. Exporting just the interface makes it clear the value has no interesting behavior beyond what is described in the interface. It also avoids the need to repeat the documentation on every instance of a common method.
>  如果某个类型仅用于实现一个接口，并且永远不会导出该接口之外的方法，则无需导出该类型本身
>  仅导出接口就可以明确表明该值除了具有接口中描述的行为之外

In such cases, the constructor should return an interface value rather than the implementing type. As an example, in the hash libraries both `crc32.NewIEEE` and `adler32.New` return the interface type `hash.Hash32`. Substituting the CRC-32 algorithm for Adler-32 in a Go program requires only changing the constructor call; the rest of the code is unaffected by the change of algorithm.
>  在这种情况下，该类型的构造函数应该返回接口类型值而不是实现类型本身的值
>  例如，在哈希库中，`crc32.NewIEEE` 和 `adler32.New` 都返回接口类型 `hash.Hash32` 的值 (因为这两个类型都没有实现了 `hash.Hash32` 接口以外的方法)
>  在 Go 程序中，用 CRC-32 算法替换 Adler-32 算法仅需要将构造函数调用修改就行，`crc32.NewIEEE` , `adler32.New` 类型值的行为都是一样的，故其余的代码不会受到算法更改的影响 

>  这应该就是 Go 中对多态概念的实现

A similar approach allows the streaming cipher algorithms in the various `crypto` packages to be separated from the block ciphers they chain together. The `Block` interface in the `crypto/cipher` package specifies the behavior of a block cipher, which provides encryption of a single block of data. Then, by analogy with the `bufio` package, cipher packages that implement this interface can be used to construct streaming ciphers, represented by the `Stream` interface, without knowing the details of the block encryption.
>  类似的方法用于在各种 `crypto` 包中从它们组合在一起的块加密算法中分离出各个流式加密算法
>  `crypto/cipher` 包中的 `Block` 接口指定了一个块加密器的行为，该加密器提供了对单个数据块的加密功能
>  类似于 `bufio` 包，实现该接口的加密包可以被用于构建流式加密器 (由 `Stream` 接口表示)，而不需要知道块加密的具体细节

The `crypto/cipher` interfaces look like this:
>  `crypto/cipher` 中两个接口的定义如下所示

```go
type Block interface {
    BlockSize() int
    Encrypt(dst, src []byte)
    Decrypt(dst, src []byte)
}

type Stream interface {
    XORKeyStream(dst, src []byte)
}
```

Here's the definition of the counter mode (CTR) stream, which turns a block cipher into a streaming cipher; notice that the block cipher's details are abstracted away:
>  CTR 流的定义如下所示，它将一个块加密器转化为流加密器，注意块加密器的细节被抽象掉了

```go
// NewCTR returns a Stream that encrypts/decrypts using the given Block in
// counter mode. The length of iv must be the same as the Block's block size.
func NewCTR(block Block, iv []byte) Stream
```

`NewCTR` applies not just to one specific encryption algorithm and data source but to any implementation of the `Block` interface and any `Stream`. 
>  `NewCTR` 函数并不仅适用于某种特定的加密算法或数据源，而是适用于任何实现了 `Block` 接口的类型和任何实现 `Stream` 接口的类型

Because they return interface values, replacing CTR encryption with other encryption modes is a localized change. The constructor calls must be edited, but because the surrounding code must treat the result only as a `Stream`, it won't notice the difference.

### Interfaces and methods
Since almost anything can have methods attached, almost anything can satisfy an interface. One illustrative example is in the `http` package, which defines the `Handler` interface. Any object that implements `Handler` can serve HTTP requests.
>  由于几乎任何东西都可以附加方法，因此几乎任何东西都可以满足一个接口
>  一个具有代表性的例子可以在 `http` 包中找到，它定义了 `Handler` 接口。任何实现 `Handler` 的对象都可以处理 HTTP 请求

```go
type Handler interface {
    ServeHTTP(ResponseWriter, *Request)
}
```

`ResponseWriter` is itself an interface that provides access to the methods needed to return the response to the client. Those methods include the standard `Write` method, so an `http.ResponseWriter` can be used wherever an `io.Writer` can be used. `Request` is a struct containing a parsed representation of the request from the client.
>  上例中，`ResponseWriter` 本身是一个接口，提供了返回相应给客户端所需的方法，这些方法包括标准的 `Writer` 方法，故可以在任何使用 `io.Writer` 的地方使用 `http.ResponseWriter`
>  上例中，`Request` 是一个结构体，包含了从客户端解析后的请求表示

For brevity, let's ignore POSTs and assume HTTP requests are always GETs; that simplification does not affect the way the handlers are set up. Here's a trivial implementation of a handler to count the number of times the page is visited
>  为了简洁起见，让我们忽略 POST 请求，并假设 HTTP 请求始终为 GET；这种简化不会影响处理器的设置方式
>  以下是一个简单实现的处理器示例，用于统计页面被访问的次数:

```go
// Simple counter server.
type Counter struct {
    n int
}

func (ctr *Counter) ServeHTTP(w http.ResponseWriter, req *http.Request) {
    ctr.n++
    fmt.Fprintf(w, "counter = %d\n", ctr.n)
}
```

(Keeping with our theme, note how `Fprintf` can print to an `http.ResponseWriter`.) In a real server, access to `ctr.n` would need protection from concurrent access. See the `sync` and `atomic` packages for suggestions.
>  在实际的服务器中，对 `ctr.n` 的访问处理需要考虑防止并发访问，详情参阅 `sync, atmoic` 包

For reference, here's how to attach such a server to a node on the URL tree.
>  仅供参考，以下是将此类服务器附加到 URL 树中节点的方法。

```go
import "net/http"
...
ctr := new(Counter)
http.Handle("/counter", ctr)
```

But why make `Counter` a struct? An integer is all that's needed. (The receiver needs to be a pointer so the increment is visible to the caller.)
>  我们进一步简化，将 `Counter` 简化为 `int` 类型

```go
// Simpler counter server.
type Counter int

func (ctr *Counter) ServeHTTP(w http.ResponseWriter, req *http.Request) {
    *ctr++
    fmt.Fprintf(w, "counter = %d\n", *ctr)
}
```

What if your program has some internal state that needs to be notified that a page has been visited? Tie a channel to the web page.
>  如果程序中有一些内部状态，需要在某个页面被访问时被告知，可以将一个通道绑定到网页上

```go
// A channel that sends a notification on each visit.
// (Probably want the channel to be buffered.)
type Chan chan *http.Request

func (ch Chan) ServeHTTP(w http.ResponseWriter, req *http.Request) {
    ch <- req
    fmt.Fprint(w, "notification sent")
}
```

Finally, let's say we wanted to present on `/args` the arguments used when invoking the server binary. It's easy to write a function to print the arguments.
>  最后，假设我们要在 `/args` 上显示启动服务器二进制文件时使用的参数，编写一个打印这些参数的函数非常简单：

```go
func ArgServer() {
    fmt.Println(os.Args)
}
```

How do we turn that into an HTTP server? We could make `ArgServer` a method of some type whose value we ignore, but there's a cleaner way. Since we can define a method for any type except pointers and interfaces, we can write a method for a function. The `http` package contains this code:
>  我们如何将其变成一个 HTTP 服务器？我们可以让 `ArgServer` 成为某个类型的某个方法，而忽略该类型的值，但有一个更简洁的方法
>  由于我们可以为除指针和接口之外的任何类型定义方法，因此我们可以**为函数编写方法**，`http` 包包含以下代码：

```go
// The HandlerFunc type is an adapter to allow the use of
// ordinary functions as HTTP handlers.  If f is a function
// with the appropriate signature, HandlerFunc(f) is a
// Handler object that calls f.
type HandlerFunc func(ResponseWriter, *Request)

// ServeHTTP calls f(w, req).
func (f HandlerFunc) ServeHTTP(w ResponseWriter, req *Request) {
    f(w, req)
}
```

`HandlerFunc` is a type with a method, `ServeHTTP`, so values of that type can serve HTTP requests. Look at the implementation of the method: the receiver is a function, `f`, and the method calls `f`. That may seem odd but it's not that different from, say, the receiver being a channel and the method sending on the channel.
>  `HandlerFunc` 是一个带有方法 `ServeHTTP` 的类型，故该类型的值都可以处理 HTTP 请求
>  该方法的接收者是一个函数 `f`，并且该方法会调用 `f`，这看起来有点奇怪，但其实于接收者是一个通道并且方法在通道上发送数据并没有太大区别

To make `ArgServer` into an HTTP server, we first modify it to have the right signature.

```go
// Argument server.
func ArgServer(w http.ResponseWriter, req *http.Request) {
    fmt.Fprintln(w, os.Args)
}
```

`ArgServer` now has the same signature as `HandlerFunc`, so it can be converted to that type to access its methods, just as we converted `Sequence` to `IntSlice` to access `IntSlice.Sort`. The code to set it up is concise:

```go
http.Handle("/args", http.HandlerFunc(ArgServer))
```

>  我们修改 `ArgServer` 函数的签名，使其符合类型 `HandlerFunc`，此时它可以被转换为 `HandlerFunc` 类型，进而使用其方法
>  就类似于我们将 `Sequence` 转换为 `IntSlice` 以使用 `IntSlice.Sort`

When someone visits the page `/args`, the handler installed at that page has value `ArgServer` and type `HandlerFunc`. The HTTP server will invoke the method `ServeHTTP` of that type, with `ArgServer` as the receiver, which will in turn call `ArgServer` (via the invocation `f(w, req)` inside `HandlerFunc.ServeHTTP`). The arguments will then be displayed.
>  当有人访问页面 `/args` 时，安装在该页面的处理器具有值 `ArgServer` 和类型 `HandlerFunc`，HTTP 服务器将调用该类型的 `ServeHTTP` 方法，`ArgServer` 是该方法的接收者
>  这又会通过调用 ` HandlerFunc.ServeHTTP ` 中的 ` f(w, req) ` 来间接调用 ` ArgServer ` ，随后，参数将会被显示出来

In this section we have made an HTTP server from a struct, an integer, a channel, and a function, all because interfaces are just sets of methods, which can be defined for (almost) any type.
>  在本节中，我们通过一个结构体、一个整数、一个通道和一个函数构建了一个HTTP服务器，这一切都是因为接口只是方法的集合，可以为几乎任何类型定义这些方法

## The blank identifier
We've mentioned the blank identifier a couple of times now, in the context of [`for` `range` loops](https://go.dev/doc/effective_go#for) and [maps](https://go.dev/doc/effective_go#maps). The blank identifier can be assigned or declared with any value of any type, with the value discarded harmlessly. It's a bit like writing to the Unix `/dev/null` file: it represents a write-only value to be used as a place-holder where a variable is needed but the actual value is irrelevant. 
>  我们已经多次提到了空白标识符
>  空白标识符可以被赋值或声明为任何类型的任意值，其值会被无害地丢弃，它有点像写入 Unix 的 `/dev/null` 文件：表示一种只写的值，用于占位，在需要变量但实际值无关紧要的情况下使用

It has uses beyond those we've seen already
>  它还有其他用途

### The blank identifier in multiple assignment
The use of a blank identifier in a `for` `range` loop is a special case of a general situation: multiple assignment.
>  在 `for, range` 循环中使用空白标识符实际上更普遍情况的一个特例：多重赋值

If an assignment requires multiple values on the left side, but one of the values will not be used by the program, a blank identifier on the left-hand-side of the assignment avoids the need to create a dummy variable and makes it clear that the value is to be discarded. For instance, when calling a function that returns a value and an error, but only the error is important, use the blank identifier to discard the irrelevant value.
>  如果赋值语句需要左侧有多个值，但程序不会使用其中一个值时，赋值左侧的空白标识符可以避免创建一个dummy变量（占位变量），并明确表明该值将被丢弃
>  例如，当调用一个返回值和错误的函数时，如果只关心错误而不需要返回值，可以使用空白标识符来丢弃无关的值：

```go
if _, err := os.Stat(path); os.IsNotExist(err) {
    fmt.Printf("%s does not exist\n", path)
}
```

Occasionally you'll see code that discards the error value in order to ignore the error; this is terrible practice. Always check error returns; they're provided for a reason
>  有时你会看到代码中会丢弃错误值以忽略错误，这是一种糟糕的做法，始终要检查错误返回值；它们的存在是有原因的

```go
// Bad! This code will crash if path does not exist.
fi, _ := os.Stat(path)
if fi.IsDir() {
    fmt.Printf("%s is a directory\n", path)
}
```

### Unused imports and variables
It is an error to import a package or to declare a variable without using it. Unused imports bloat the program and slow compilation, while a variable that is initialized but not used is at least a wasted computation and perhaps indicative of a larger bug. When a program is under active development, however, unused imports and variables often arise and it can be annoying to delete them just to have the compilation proceed, only to have them be needed again later. The blank identifier provides a workaround.
>  在程序中导入一个未使用的包或声明一个未使用的变量是一种错误
>  未使用的导入会使程序变得臃肿，减缓编译速度，而被初始化但未被使用的变量至少是多余的计算，甚至可能表明存在更大的问题
>  然而，在程序处于积极开发阶段时，未使用的导入和变量往往会涌现，删除它们以使编译能够继续进行可能会让人感到厌烦，但之后又可能需要它们，空白标识符提供了一种解决方法

This half-written program has two unused imports (`fmt` and `io`) and an unused variable (`fd`), so it will not compile, but it would be nice to see if the code so far is correct.
>  这个半完成的程序有两个未使用的导入 `fmt` 和 `io` 以及一个未使用的变量 `fd`，因此它无法编译

```go
package main

import (
    "fmt"
    "io"
    "log"
    "os"
)

func main() {
    fd, err := os.Open("test.go")
    if err != nil {
        log.Fatal(err)
    }
    // TODO: use fd.
}
```

To silence complaints about the unused imports, use a blank identifier to refer to a symbol from the imported package. Similarly, assigning the unused variable `fd` to the blank identifier will silence the unused variable error. This version of the program does compile.
>  要消除关于未使用导入的错误，可以使用空白标识符来引用被导入包中的符号
>  同样地，将未使用的变量 `fd` 赋值给空白标识符将消除未使用变量的错误
>  此版本的程序是可以编译的：

```go
package main

import (
    "fmt"
    "io"
    "log"
    "os"
)

var _ = fmt.Printf // For debugging; delete when done.
var _ io.Reader    // For debugging; delete when done.

func main() {
    fd, err := os.Open("test.go")
    if err != nil {
        log.Fatal(err)
    }
    // TODO: use fd.
    _ = fd
}
```

By convention, the global declarations to silence import errors should come right after the imports and be commented, both to make them easy to find and as a reminder to clean things up later.
>  按照惯例，用于消除导入错误的全局声明应在导入之后立即出现，并且应添加注释，以便于找到它们，同时也提醒日后清理这些内容

### Import for side effect
An unused import like `fmt` or `io` in the previous example should eventually be used or removed: blank assignments identify code as a work in progress. But sometimes it is useful to import a package only for its side effects, without any explicit use. For example, during its `init` function, the `[net/http/pprof](https://go.dev/pkg/net/http/pprof/)` package registers HTTP handlers that provide debugging information. It has an exported API, but most clients need only the handler registration and access the data through a web page. To import the package only for its side effects, rename the package to the blank identifier:
>  未使用的导入 (如前例中的 `fmt` 或 `io`) 最终应被使用或移除：空白赋值表示代码尚处于开发中
>  但有时，导入一个包仅是为了其副作用，而无需任何显式的使用，例如，`net/http/pprof` 包在其 `init` 函数中会注册提供调试信息的 HTTP 处理程序，它有一个导出的 API，但大多数客户端只需要处理程序的注册，并通过网页访问数据
>  要仅为了其副作用导入该包，可以将包重命名为空白标识符：

```go
import _ "net/http/pprof"
```

This form of import makes clear that the package is being imported for its side effects, because there is no other possible use of the package: in this file, it doesn't have a name. (If it did, and we didn't use that name, the compiler would reject the program.)
>  这种形式的导入明确表明该包被导入是为了其副作用，因为没有其他可能使用该包的方式：在这个文件中，它没有名称 (如果有名称，但我们没有使用该名称，编译器会报错)

### Interface checks
As we saw in the discussion of [interfaces](https://go.dev/doc/effective_go#interfaces_and_types) above, a type need not declare explicitly that it implements an interface. Instead, a type implements the interface just by implementing the interface's methods. In practice, most interface conversions are static and therefore checked at compile time. For example, passing an `*os.File` to a function expecting an `io.Reader` will not compile unless `*os.File` implements the `io.Reader` interface.
>  正如我们在上面关于[接口](https://go.dev/doc/effective_go#interfaces_and_types)的讨论中看到的，类型无需显式声明它实现了某个接口，相反，只要类型实现了接口的方法，它就自动实现了该接口
>  在实践中，大多数接口转换是静态的，因此会在编译时进行检查，例如，除非 `*os.File` 实现了 `io.Reader` 接口，否则将一个 `*os.File` 传递给期望接收 `io.Reader` 的函数将无法通过编译

Some interface checks do happen at run-time, though. One instance is in the `encoding/json` package, which defines a `Marshaler` interface. When the JSON encoder receives a value that implements that interface, the encoder invokes the value's marshaling method to convert it to JSON instead of doing the standard conversion. The encoder checks this property at run time with a [type assertion](https://go.dev/doc/effective_go#interface_conversions) like:
>  尽管如此，确实有一些接口检查是在运行时进行的
>  一个例子出现在 `encoding/json` 包中，该包定义了一个 `Marshaler` 接口，当 JSON 编码器接收到实现了该接口的值时，编码器会调用该值的序列化方法将其转换为 JSON，而不是执行标准的转换过程
>  编码器通过类似 type assertion 的方式在运行时检查这一特性：

```go
m, ok := val.(json.Marshaler)
```

If it's necessary only to ask whether a type implements an interface, without actually using the interface itself, perhaps as part of an error check, use the blank identifier to ignore the type-asserted value:
>  如果只需要判断某种类型是否实现了某个接口，而不需要实际使用该接口本身，例如作为错误检查的一部分，则可以使用空白标识符来忽略类型断言的值：

```go
if _, ok := val.(json.Marshaler); ok {
    fmt.Printf("value %v of type %T implements json.Marshaler\n", val, val)
}
```

One place this situation arises is when it is necessary to guarantee within the package implementing the type that it actually satisfies the interface. If a type—for example, `json.RawMessage` —needs a custom JSON representation, it should implement `json.Marshaler`, but there are no static conversions that would cause the compiler to verify this automatically. If the type inadvertently fails to satisfy the interface, the JSON encoder will still work, but will not use the custom implementation. To guarantee that the implementation is correct, a global declaration using the blank identifier can be used in the package:
>  这种情况的一个典型场景是，在实现某个类型的具体包中，需要确保该类型确实满足指定的接口
>  例如，当一个类型 (如 `json.RawMessage`) 需要自定义的 JSON 表示时，它应该实现 `json.Marshaler` 接口，但编译器不会自动通过静态转换来验证这一点，如果类型无意中未能正确实现该接口，JSON 编码器仍然可以正常工作，但不会使用自定义的实现方式
>  为了确保实现是正确的，可以在包中使用空白标识符进行全局声明：

```go
var _ json.Marshaler = (*RawMessage)(nil)
```

In this declaration, the assignment involving a conversion of a `*RawMessage` to a `Marshaler` requires that `*RawMessage` implements `Marshaler`, and that property will be checked at compile time. Should the `json.Marshaler` interface change, this package will no longer compile and we will be on notice that it needs to be updated.
>  在该声明中，涉及将一个 `*RawMessage` 转换为 `Marshaler` 的赋值操作，它要求 `*RawMessage` 实现了 `Marshaler`，并且该属性将在编译时进行检查
>  如果 `json.Marshaler` 接口发生变化，此包将无法继续编译，我们将意识到需要对其进行更新

The appearance of the blank identifier in this construct indicates that the declaration exists only for the type checking, not to create a variable. Don't do this for every type that satisfies an interface, though. By convention, such declarations are only used when there are no static conversions already present in the code, which is a rare event.
>  在这个结构中出现空白标识符的是为了表明该声明仅用于类型检查，而不是为了创建变量
>  不过，不要为满足接口的每种类型都这样做，按照惯例，只有在代码中不存在静态转换的情况下才会使用这样的声明，而这是一种很少见的情况

## Embedding
Go does not provide the typical, type-driven notion of subclassing, but it does have the ability to “borrow” pieces of an implementation by _embedding_ types within a struct or interface.
>  Go 并不提供典型的、基于类型的继承概念，但它可以通过在结构体或接口中嵌入类型来“借用”部分实现。

Interface embedding is very simple. We've mentioned the `io.Reader` and `io.Writer` interfaces before; here are their definitions.
>  接口嵌入非常简单
>  我们之前提到过 `io.Reader` 和 `io.Writer` 接口；以下是它们的定义:

```go
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}
```

The `io` package also exports several other interfaces that specify objects that can implement several such methods. For instance, there is `io.ReadWriter`, an interface containing both `Read` and `Write`. We could specify `io.ReadWriter` by listing the two methods explicitly, but it's easier and more evocative to embed the two interfaces to form the new one, like this:
>  `io` 包还导出了几个其他接口，这些接口指定了可以实现多个此类方法的对象，例如，有一个 `io.ReadWriter` 接口，它同时包含 `Read` 和 `Write` 方法
>  我们可以显式列出这两种方法来定义 `io.ReadWriter`，但嵌入这两个接口以形成新的接口会更简单且更具描述性，如下所示：

```go
// ReadWriter is the interface that combines the Reader and Writer interfaces.
type ReadWriter interface {
    Reader
    Writer
}
```

This says just what it looks like: A `ReadWriter` can do what a `Reader` does _and_ what a `Writer` does; it is a union of the embedded interfaces. Only interfaces can be embedded within interfaces.
>  上例中的代码的意思是：一个 `ReadWriter` 可以做 `Reader` 能做的 **以及** `Writer` 能做的事情；即它是嵌入接口 (`Reader, Writer`)的联合
>  只有接口可以被嵌入到其他接口中

The same basic idea applies to structs, but with more far-reaching implications. The `bufio` package has two struct types, `bufio.Reader` and `bufio.Writer`, each of which of course implements the analogous interfaces from package `io`. And `bufio` also implements a buffered reader/writer, which it does by combining a reader and a writer into one struct using embedding: it lists the types within the struct but does not give them field names.
>  相同的基本思想也适用于结构体，但其影响更为深远
>  `bufio` 包中有两种结构体类型，分别是 `bufio.Reader` 和 `bufio.Writer`，它们当然都实现了来自 `io` 包的相应接口
>  此外，`bufio` 还实现了一个带缓冲的读写器，它通过嵌入的方式将一个 `Reader` 和一个 `Writer` 组合成一个结构体来完成这一功能：它在结构体中列出这些类型但并未为其指定字段名:

```go
// ReadWriter stores pointers to a Reader and a Writer.
// It implements io.ReadWriter.
type ReadWriter struct {
    *Reader  // *bufio.Reader
    *Writer  // *bufio.Writer
}
```

The embedded elements are pointers to structs and of course must be initialized to point to valid structs before they can be used. 
>  上例中，嵌入的元素是指向结构体的指针，当然在可以使用之前必须初始化为指向有效的结构体

The `ReadWriter` struct could be written as
>   `ReadWriter` 结构体也可以写成如下形式：

```go
type ReadWriter struct {
    reader *Reader
    writer *Writer
}
```

but then to promote the methods of the fields and to satisfy the `io` interfaces, we would also need to provide forwarding methods, like this
>  在这种形式下，为了推广这些字段的方法并满足 `io` 接口，我们还需要提供转发方法，就像这样：
>  (为 `ReadWriter` 实现 `Read` 方法，本质是调用其中的 `reader` 的 `Read` 方法)

```go
func (rw *ReadWriter) Read(p []byte) (n int, err error) {
    return rw.reader.Read(p)
}
```

By embedding the structs directly, we avoid this bookkeeping. The methods of embedded types come along for free, which means that `bufio.ReadWriter` not only has the methods of `bufio.Reader` and `bufio.Writer`, it also satisfies all three interfaces: `io.Reader`, `io.Writer`, and `io.ReadWriter`.
>  通过直接嵌入结构体，我们可以避免这种额外的管理工作
>  嵌入类型的方法会自动继承，这意味着 `bufio.ReadWriter` 不仅拥有 `bufio.Reader` 和 `bufio.Writer` 的方法，还满足所有三个接口：`io.Reader`、`io.Writer` 和 `io.ReadWriter`

There's an important way in which embedding differs from subclassing. When we embed a type, the methods of that type become methods of the outer type, but when they are invoked the receiver of the method is the inner type, not the outer one. In our example, when the `Read` method of a `bufio.ReadWriter` is invoked, it has exactly the same effect as the forwarding method written out above; the receiver is the `reader` field of the `ReadWriter`, not the `ReadWriter` itself.
>  嵌入与继承有一个重要的差异: 当我们嵌入一个类型时，该类型的方法会成为外部类型的方法，但当这些方法被调用时，方法的接收者是内部类型，而不是外部类型
>  在我们的示例中，当 `bufio.ReadWriter` 的 `Read` 方法被调用时，其效果与上面写的转发方法完全相同；接收者是 `ReadWriter` 的 `reader` 字段，而不是 `ReadWriter` 本身

Embedding can also be a simple convenience. This example shows an embedded field alongside a regular, named field.
>  嵌入也可以是一个简单的便利，这个示例展示了嵌入字段与常规命名字段并排显示

```go
type Job struct {
    Command string
    *log.Logger
}
```

The `Job` type now has the `Print`, `Printf`, `Println` and other methods of `*log.Logger`. We could have given the `Logger` a field name, of course, but it's not necessary to do so. And now, once initialized, we can log to the `Job`:
>  `Job` 类型现在具有 `Print`、`Printf`、`Println` 以及其他与 `*log.Logger` 相同的方法，当然，我们可以为 `Logger` 提供一个字段名称，但这是没有必要的
>  现在，一旦初始化完成，我们就可以向 `Job` 记录日志。

```go
job.Println("starting now...")
```

The `Logger` is a regular field of the `Job` struct, so we can initialize it in the usual way inside the constructor for `Job`, like this,
>  `Logger` 是 `Job` 结构体的一个普通字段，因此我们可以在 `Job` 的构造函数中以通常的方式初始化它，如下所示：

```go
func NewJob(command string, logger *log.Logger) *Job {
    return &Job{command, logger}
}
```

or with a composite literal,
>  或者使用复合字面值

```go
job := &Job{command, log.New(os.Stderr, "Job: ", log.Ldate)}
```

If we need to refer to an embedded field directly, the type name of the field, ignoring the package qualifier, serves as a field name, as it did in the `Read` method of our `ReadWriter` struct. Here, if we needed to access the `*log.Logger` of a `Job` variable `job`, we would write `job.Logger`, which would be useful if we wanted to refine the methods of `Logger`.
>  如果我们需要直接引用嵌入的字段，那么忽略包限定符的字段类型名称将作为字段名使用，就像在 `ReadWriter` 结构体的 `Read` 方法中一样
>  在这里，如果我们需要访问 `Job` 类型变量 `job` 的 `*log.Logger`，我们会写成 `job.Logger`，这在我们想要细化 `Logger` 的方法时会很有用。

```go
func (job *Job) Printf(format string, args ...interface{}) {
    job.Logger.Printf("%q: %s", job.Command, fmt.Sprintf(format, args...))
}
```

Embedding types introduces the problem of name conflicts but the rules to resolve them are simple. First, a field or method `X` hides any other item `X` in a more deeply nested part of the type. If `log.Logger` contained a field or method called `Command`, the `Command` field of `Job` would dominate it.
>  嵌入类型引入了名称冲突的问题，但解决这些冲突的规则很简单
>  首先，一个字段或方法 `X` 会隐藏其类型中更深层嵌套部分中的任何其他名为 `X` 的项: 如果 `log.Logger` 包含一个名为 `Command` 的字段或方法，则 `Job` 的 `Command` 字段将优先于它

Second, if the same name appears at the same nesting level, it is usually an error; it would be erroneous to embed `log.Logger` if the `Job` struct contained another field or method called `Logger`. However, if the duplicate name is never mentioned in the program outside the type definition, it is OK. This qualification provides some protection against changes made to types embedded from outside; there is no problem if a field is added that conflicts with another field in another subtype if neither field is ever used.
>  其次，如果相同的名字出现在相同的嵌套层级中，通常是一个错误: 如果 `Job` 结构体中包含另一个名为 `Logger` 的字段或方法，则嵌入 `log.Logger` 将会导致问题
>  然而，如果该重复名称在类型定义之外的程序中从未被提及，则是可以接受的，这一限定条件提供了一些保护，防止来自外部对类型的更改带来的影响；如果两个字段虽然存在冲突但都从未被使用过，则不会出现问题

## Concurrency
### Share by communicating 
Concurrent programming is a large topic and there is space only for some Go-specific highlights here.

Concurrent programming in many environments is made difficult by the subtleties required to implement correct access to shared variables. Go encourages a different approach in which shared values are passed around on channels and, in fact, never actively shared by separate threads of execution. Only one goroutine has access to the value at any given time. Data races cannot occur, by design. 
>  在许多环境中，并发编程因为要实现对共享变量的正确访问而变得困难
>  Go 鼓励一种不同的方法，即通过通道传递共享值，并且单独的执行线程不会主动共享这些值，那么，在任何时候，只有一个 goroutine 可以访问共享值，数据竞争不可能发生

To encourage this way of thinking we have reduced it to a slogan:

> Do not communicate by sharing memory; instead, share memory by communicating.

>  为了鼓励这种思维方式，我们将其简化为一句口号：
>  不要通过共享内存进行通信，而是通过通信来共享内存

This approach can be taken too far. Reference counts may be best done by putting a mutex around an integer variable, for instance. But as a high-level approach, using channels to control access makes it easier to write clear, correct programs.
>  该方法可能会被推得太远，例如，通过在整数变量周围加上互斥锁来实现引用计数可能是最佳选择，但从高层设计的角度来看，使用通道来控制访问可以使得编写更加清晰的程序变得更加容易

One way to think about this model is to consider a typical single-threaded program running on one CPU. It has no need for synchronization primitives. Now run another such instance; it too needs no synchronization. Now let those two communicate; if the communication is the synchronizer, there's still no need for other synchronization. Unix pipelines, for example, fit this model perfectly. Although Go's approach to concurrency originates in Hoare's Communicating Sequential Processes (CSP), it can also be seen as a type-safe generalization of Unix pipes.
>  关于这个模型的一种思考方式是考虑一个在单个 CPU 上运行的典型单线程程序，它不需要同步原语，现在再运行一个这样的实例，它同样不需要同步
>  然后让这两个实例通信，如果通信本身就是同步器，则仍然不需要其他同步机制
>  例如，Unix 管道完全符合这种模式，尽管 Go 的并发方法起源于 Hoare 的通信顺序进程，它也可以被视为 Unix 管道的一种类型安全的泛化

### Goroutines
They're called _goroutines_ because the existing terms—threads, coroutines, processes, and so on—convey inaccurate connotations. A goroutine has a simple model: it is a function executing concurrently with other goroutines in the same address space. It is lightweight, costing little more than the allocation of stack space. And the stacks start small, so they are cheap, and grow by allocating (and freeing) heap storage as required.
>  goroutine 之所以称为 goroutine，是因为现有的术语——线程、协程、进程等——会带来不准确的联想
>  一个 goroutine 的模型很简单：它是一个与其他 goroutines 在相同的地址空间并发执行的函数
>  它非常轻量，除了栈空间的分配外，几乎不需要额外开销，并且栈空间初始很小，因此成本很低，并且会根据需要通过分配 (和释放) 堆空间来动态增长

Goroutines are multiplexed onto multiple OS threads so if one should block, such as while waiting for I/O, others continue to run. Their design hides many of the complexities of thread creation and management.
>  goroutine 会被多路复用到多个操作系统线程上
>  因此，如果一个 goroutine 需要阻塞，例如等待 IO，其他的 goroutine 可以继续运行
>  goroutine 的设计隐藏了许多线程创建和管理的复杂性

Prefix a function or method call with the `go` keyword to run the call in a new goroutine. When the call completes, the goroutine exits, silently. (The effect is similar to the Unix shell's `&` notation for running a command in the background.)
>  在函数或方法调用之前加上 `go` 关键字，就会在一个新的 goroutine 中执行该调用
>  当调用完成时，goroutine 会自动退出，且不会产生任何提示 (其效果类似于在 Unix Shell 中使用 `&` 符号在后台运行命令)

```go
go list.Sort()  // run list.Sort concurrently; don't wait for it.
```

A function literal can be handy in a goroutine invocation.
>  goroutine 可以调用函数字面量 (类似匿名函数)

```go
func Announce(message string, delay time.Duration) {
    go func() {
        time.Sleep(delay)
        fmt.Println(message)
    }()  // Note the parentheses - must call the function.
}
```

In Go, function literals are closures: the implementation makes sure the variables referred to by the function survive as long as they are active.
>  在 Go 中，函数字面量是闭包：函数字面量引用的变量只要处于活动状态，就确保会一直存在

These examples aren't too practical because the functions have no way of signaling completion. For that, we need channels.
>  这些例子不太实用，因为这些函数没有办法表明它们已经完成，为此，我们需要使用通道

### Channels
Like maps, channels are allocated with `make`, and the resulting value acts as a reference to an underlying data structure. If an optional integer parameter is provided, it sets the buffer size for the channel. The default is zero, for an unbuffered or synchronous channel.
>  和 maps 类似，通道也通过 `make` 分配，并且生成的值是对底层数据结构的引用
>  如果提供了可选的整数参数，该参数将设置通道的缓冲区大小，默认为零，表示一个无缓冲或同步的通道

```go
ci := make(chan int)            // unbuffered channel of integers
cj := make(chan int, 0)         // unbuffered channel of integers
cs := make(chan *os.File, 100)  // buffered channel of pointers to Files
```

Unbuffered channels combine communication—the exchange of a value—with synchronization—guaranteeing that two calculations (goroutines) are in a known state.
>  无缓冲通道将通信 (值的交换) 和同步 (确保两个计算 (goroutines) 处于一个已知状态) 结合在一起

There are lots of nice idioms using channels. 
>  有许多使用通道的惯用法

Here's one to get us started. In the previous section we launched a sort in the background. A channel can allow the launching goroutine to wait for the sort to complete.
>  在上一节中，我们在后台启动了一个排序操作，通道可以让启动该 goroutine 的主程序等待排序完成:

```go
c := make(chan int)  // Allocate a channel.
// Start the sort in a goroutine; when it completes, signal on the channel.
go func() {
    list.Sort()
    c <- 1  // Send a signal; value does not matter.
}()
doSomethingForAWhile()
<-c   // Wait for sort to finish; discard sent value.
```

Receivers always block until there is data to receive. If the channel is unbuffered, the sender blocks until the receiver has received the value. If the channel has a buffer, the sender blocks only until the value has been copied to the buffer; if the buffer is full, this means waiting until some receiver has retrieved a value.
>  通道的接收者总是会阻塞，知道有东西可以接收
>  如果通道是无缓冲的，发送者也会阻塞，直到接收者接收到值
>  如果通道有缓冲，发送者只会阻塞到值被复制到缓冲区为止，如果缓冲区慢，则意味着需要等待某个接收者取走一个值为止

A buffered channel can be used like a semaphore, for instance to limit throughput. In this example, incoming requests are passed to `handle`, which sends a value into the channel, processes the request, and then receives a value from the channel to ready the “semaphore” for the next consumer. The capacity of the channel buffer limits the number of simultaneous calls to `process`.
>  有缓冲的通道可以像信号量一样使用，例如用来限制吞吐量
>  在下面的例子中，传入的请求被传递给 `handle` ，`handle` 向通道发送一个值，处理请求，然后从通道接收一个值，以便为下一个消费者准备好 “信号量”
>  通道缓冲区的容量限制了对 `process` 的同时调用次数

```go
var sem = make(chan int, MaxOutstanding)

func handle(r *Request) {
    sem <- 1    // Wait for active queue to drain.
    process(r)  // May take a long time.
    <-sem       // Done; enable next request to run.
}

func Serve(queue chan *Request) {
    for {
        req := <-queue
        go handle(req)  // Don't wait for handle to finish.
    }
}
```

Once `MaxOutstanding` handlers are executing `process`, any more will block trying to send into the filled channel buffer, until one of the existing handlers finishes and receives from the buffer.
>  当有 `MaxOutstanding` 个 `handler` 在处理 `process` ，任意更多的请求都会尝试向已经满的通道缓冲区发送数据，直到现有 `handler` 中的某一个完成了处理，并从缓冲区接收数据，这些请求才可以进入

This design has a problem, though: `Serve` creates a new goroutine for every incoming request, even though only `MaxOutstanding` of them can run at any moment. As a result, the program can consume unlimited resources if the requests come in too fast. We can address that deficiency by changing `Serve` to gate the creation of the goroutines:
>  该设计存在一个问题: `Serve` 为每个传入的请求创建一个新的 goroutine，即便任何时候只能运行其中的 `MaxOutstanding` 个
>  因此，如果请求来得太快，该程序会消耗无限的资源
>  我们可以通过修改 `Server` ，限制 goroutine 的创建来解决这个问题

```go
func Serve(queue chan *Request) {
    for req := range queue {
        sem <- 1
        go func() {
            process(req)
            <-sem
        }()
    }
}
```

(Note that in Go versions before 1.22 this code has a bug: the loop variable is shared across all goroutines. See the [Go wiki](https://go.dev/wiki/LoopvarExperiment) for details.)

Another approach that manages resources well is to start a fixed number of `handle` goroutines all reading from the request channel. The number of goroutines limits the number of simultaneous calls to `process`. This `Serve` function also accepts a channel on which it will be told to exit; after launching the goroutines it blocks receiving from that channel.
>  另一个管理资源的有效方法是启动固定数量的 `handle` goroutine，所有 goroutine 都从请求通道中读取数据
>  goroutine 的数量限制了同时调用 `process` 的次数
>  此 `Serve` 函数还接收一个通道，用于接收退出信号，在启动 goroutines 之后，它会阻塞，等待从该通道中接收信号

```go
func handle(queue chan *Request) {
    for r := range queue {
        process(r)
    }
}

func Serve(clientRequests chan *Request, quit chan bool) {
    // Start handlers
    for i := 0; i < MaxOutstanding; i++ {
        go handle(clientRequests)
    }
    <-quit  // Wait to be told to exit.
}
```

### Channels of channels
One of the most important properties of Go is that a channel is a first-class value that can be allocated and passed around like any other. A common use of this property is to implement safe, parallel demultiplexing.
>  Go 的最重要特性之一就是通道是一种一级值，可以像其他值一样被传递和分配
>  该特性的常见用法之一是实现安全的多路并行分解

In the example in the previous section, `handle` was an idealized handler for a request but we didn't define the type it was handling. If that type includes a channel on which to reply, each client can provide its own path for the answer. Here's a schematic definition of type `Request`.
>  在上一节的例子中，`handle` 是一个理想化的请求处理程序，但我们并没有定义它所处理的类型
>  如果该类型包含一个用于回复的通道，则每个客户端都可以为其回复提供路径，以下是 `Request` 类型的一个示例定义：

```go
type Request struct {
    args        []int
    f           func([]int) int
    resultChan  chan int
}
```

The client provides a function and its arguments, as well as a channel inside the request object on which to receive the answer.
>  客户端在 `Request` 中提供一个函数和其参数，以及接收回复的通道

```go
func sum(a []int) (s int) {
    for _, v := range a {
        s += v
    }
    return
}

request := &Request{[]int{3, 4, 5}, sum, make(chan int)}
// Send request
clientRequests <- request
// Wait for response.
fmt.Printf("answer: %d\n", <-request.resultChan)
```

On the server side, the handler function is the only thing that changes.
>  在服务器端，`handler` 函数可以利用该通道发送回复

```go
func handle(queue chan *Request) {
    for req := range queue {
        req.resultChan <- req.f(req.args)
    }
}
```

There's clearly a lot more to do to make it realistic, but this code is a framework for a rate-limited, parallel, non-blocking RPC system, and there's not a mutex in sight.
>  显然，还有许多工作要做才能使其更贴近现实，但这段代码是一个限速、并行、非阻塞的远程过程调用系统的框架，而且看不到任何一个互斥锁

### Parallelization
Another application of these ideas is to parallelize a calculation across multiple CPU cores. If the calculation can be broken into separate pieces that can execute independently, it can be parallelized, with a channel to signal when each piece completes.
>  这些概念的另一个应用是在多个 CPU 核上并行计算，如果计算可以被分解为可以独立执行的各个部分，计算就可以被并行化，并使用通道标记每个部分何时完成

Let's say we have an expensive operation to perform on a vector of items, and that the value of the operation on each item is independent, as in this idealized example.
>  假设我们需要对一个向量的各个成分执行相互独立的操作，我们先定义针对每个成分的处理函数如下：

```go
type Vector []float64

// Apply the operation to v[i], v[i+1] ... up to v[n-1].
func (v Vector) DoSome(i, n int, u Vector, c chan int) {
    for ; i < n; i++ {
        v[i] += u.Op(v[i])
    }
    c <- 1    // signal that this piece is done
}
```

We launch the pieces independently in a loop, one per CPU. They can complete in any order but it doesn't matter; we just count the completion signals by draining the channel after launching all the goroutines.
>  我们通过循环独立启动这些任务，每个 CPU 一个
>  这些任务可以以任意顺序完成，我们不关心其完成顺序，只需在启动所有 goroutines 之后通过清空通道来计数完成信号即可

```go
const numCPU = 4 // number of CPU cores

func (v Vector) DoAll(u Vector) {
    c := make(chan int, numCPU)  // Buffering optional but sensible.
    for i := 0; i < numCPU; i++ {
        go v.DoSome(i*len(v)/numCPU, (i+1)*len(v)/numCPU, u, c)
    }
    // Drain the channel.
    for i := 0; i < numCPU; i++ {
        <-c    // wait for one task to complete
    }
    // All done.
}
```

Rather than create a constant value for `numCPU`, we can ask the runtime what value is appropriate. The function ` runtime.NumCPU ` returns the number of hardware CPU cores in the machine, so we could write
>  我们可以询问运行时 CPU 的核心数量，函数 `runtime.NumCPU` 会返回机器的硬件 CPU 核心数量

```go
var numCPU = runtime.NumCPU()
```

There is also a function `runtime.GOMAXPROCS`, which reports (or sets) the user-specified number of cores that a Go program can have running simultaneously. It defaults to the value of `runtime.NumCPU` but can be overridden by setting the similarly named shell environment variable or by calling the function with a positive number. Calling it with zero just queries the value. Therefore if we want to honor the user's resource request, we should write
>  函数 `runtime.GOMAXPROCS` 可以报告 (或设置) Go 程序中可以同时运行的用户指定的核心数量，其默认值是 `runtime.NumCPU` ，但可以通过设置同名的环境变量或调用该函数传入正数以覆盖
>  调用该函数传入 0 表示仅查询该值，因此，如果我们像尊重用户的资源请求，我们应该写：

```go
var numCPU = runtime.GOMAXPROCS(0)
```

Be sure not to confuse the ideas of concurrency—structuring a program as independently executing components—and parallelism—executing calculations in parallel for efficiency on multiple CPUs. Although the concurrency features of Go can make some problems easy to structure as parallel computations, Go is a concurrent language, not a parallel one, and not all parallelization problems fit Go's model. For a discussion of the distinction, see the talk cited in [this blog post](https://go.dev/blog/concurrency-is-not-parallelism).
>  注意区分并发——将程序结构化为独立执行的组件，和并行——在多个 CPU 上执行并行计算以提高效率
>  虽然 Go 的一些并发特性使得问题容易被构造为并行计算，但 Go 是一个并发语言，而不是并行语言，并非所有并行化问题都适合 Go

### A leaky buffer
The tools of concurrent programming can even make non-concurrent ideas easier to express. Here's an example abstracted from an RPC package. The client goroutine loops receiving data from some source, perhaps a network. To avoid allocating and freeing buffers, it keeps a free list, and uses a buffered channel to represent it. If the channel is empty, a new buffer gets allocated. Once the message buffer is ready, it's sent to the server on `serverChan`.
>  并发编程的工具甚至可以让非并发的概念更易于表达
>  以下是一个从 RPC 包中抽象出来的示例，客户端 goroutine 循环从某个来源接收数据，为了避免频繁分配和释放缓存，它维护了一个空闲列表，并使用带缓存的通道来表示它
>  如果通道为空 (没有空闲)，则会分配一个新的缓冲区，一旦消息缓冲区就绪，它就会通过 `serverChan` 发送到服务器

```go
var freeList = make(chan *Buffer, 100)
var serverChan = make(chan *Buffer)

func client() {
    for {
        var b *Buffer
        // Grab a buffer if available; allocate if not.
        select {
        case b = <-freeList:
            // Got one; nothing more to do.
        default:
            // None free, so allocate a new one.
            b = new(Buffer)
        }
        load(b)              // Read next message from the net.
        serverChan <- b      // Send to server.
    }
}
```

The server loop receives each message from the client, processes it, and returns the buffer to the free list.
>  服务器循环从客户端接收消息，处理消息，然后将缓冲区返回给空闲列表

```go
func server() {
    for {
        b := <-serverChan    // Wait for work.
        process(b)
        // Reuse buffer if there's room.
        select {
        case freeList <- b:
            // Buffer on free list; nothing more to do.
        default:
            // Free list full, just carry on.
        }
    }
}
```

The client attempts to retrieve a buffer from `freeList`; if none is available, it allocates a fresh one. The server's send to `freeList` puts `b` back on the free list unless the list is full, in which case the buffer is dropped on the floor to be reclaimed by the garbage collector. (The `default` clauses in the `select` statements execute when no other case is ready, meaning that the `selects` never block.) This implementation builds a leaky bucket free list in just a few lines, relying on the buffered channel and the garbage collector for bookkeeping.
>  客户端尝试从 `freeList` 获取一个缓冲区，如果没有可用的，则分配一个新的
>  服务器向 `freeList` 发送数据时，会将 `b` 放回空闲列表，除非列表已满，在这种情况下，缓冲区会被丢弃，由垃圾回收器回收 (`select` 语句中的 `default` 子句在没有其他 case 就绪时执行，这意味着 `select` 永远不会阻塞)
>  这个实现构造了一个泄露式的空闲列表，依赖带缓冲的通道和垃圾回收器运行

## Errors
Library routines must often return some sort of error indication to the caller. As mentioned earlier, Go's multivalue return makes it easy to return a detailed error description alongside the normal return value. It is good style to use this feature to provide detailed error information. For example, as we'll see, `os.Open` doesn't just return a `nil` pointer on failure, it also returns an error value that describes what went wrong.
>  库函数通常需要向调用者返回某种形式的错误指示
>  如之前所述，Go 的多值返回机制使得在返回正常返回值的同时返回详细的错误描述非常容易，使用这一机制提供详细的错误信息是良好的风格
>  例如，`os.Open` 不仅在失败时返回一个 `nil` 指针，还会返回一个描述错误原因的错误值

By convention, errors have type `error`, a simple built-in interface.
>  错误的类型一般都是 `error` ，一个简单的内建接口，如下所示：

```go
type error interface {
    Error() string
}
```

A library writer is free to implement this interface with a richer model under the covers, making it possible not only to see the error but also to provide some context. As mentioned, alongside the usual `*os.File` return value, `os.Open` also returns an error value. If the file is opened successfully, the error will be `nil`, but when there is a problem, it will hold an `os.PathError`:
>  一个库的作者可以自由地实现该接口
>  如前所述，`os.Open` 除了通常的 `*os.File` 返回值以外，还会返回错误码，如果文件正常打开，错误是 `nil` ，如果打开出现问题，它将包含一个 `os.PathError` (`os.PathError` 实现了 `error` 接口):

```go
// PathError records an error and the operation and
// file path that caused it.
type PathError struct {
    Op string    // "open", "unlink", etc.
    Path string  // The associated file.
    Err error    // Returned by the system call.
}

func (e *PathError) Error() string {
    return e.Op + " " + e.Path + ": " + e.Err.Error()
}
```

`PathError`'s `Error` generates a string like this:

```go
open /etc/passwx: no such file or directory
```

Such an error, which includes the problematic file name, the operation, and the operating system error it triggered, is useful even if printed far from the call that caused it; it is much more informative than the plain "no such file or directory".

When feasible, error strings should identify their origin, such as by having a prefix naming the operation or package that generated the error. For example, in package `image`, the string representation for a decoding error due to an unknown format is "image: unknown format".
>  在可行的情况下，错误字符串应该标识其来源，例如具有一个表明生成错误的操作或包的前缀名
>  例如，包 `image` 中，由于未知格式导致的解码错误的字符串表示为 `"image: unknown format"`

Callers that care about the precise error details can use a type switch or a type assertion to look for specific errors and extract details. For `PathErrors` this might include examining the internal `Err` field for recoverable failures.
>  可以通过 type switch 或 type assertion 获取详细的错误类型，提取错误细节

```go
for try := 0; try < 2; try++ {
    file, err = os.Create(filename)
    if err == nil {
        return
    }
    if e, ok := err.(*os.PathError); ok && e.Err == syscall.ENOSPC {
        deleteTempFiles()  // Recover some space.
        continue
    }
    return
}
```

The second `if` statement here is another [type assertion](https://go.dev/doc/effective_go#interface_conversions). If it fails, `ok` will be false, and `e` will be `nil`. If it succeeds, `ok` will be true, which means the error was of type `*os.PathError`, and then so is `e`, which we can examine for more information about the error.

### Panic
The usual way to report an error to a caller is to return an `error` as an extra return value. The canonical `Read` method is a well-known instance; it returns a byte count and an `error`. But what if the error is unrecoverable? Sometimes the program simply cannot continue.

For this purpose, there is a built-in function `panic` that in effect creates a run-time error that will stop the program (but see the next section). The function takes a single argument of arbitrary type—often a string—to be printed as the program dies. It's also a way to indicate that something impossible has happened, such as exiting an infinite loop.
>  遇到无法恢复的错误时，内建函数 `panic` 会创建一个运行时错误并终止程序
>  该函数接收一个任意类型参数 (通常是字符串)，在程序终止时打印它

```go
// A toy implementation of cube root using Newton's method.
func CubeRoot(x float64) float64 {
    z := x/3   // Arbitrary initial value
    for i := 0; i < 1e6; i++ {
        prevz := z
        z -= (z*z*z-x) / (3*z*z)
        if veryClose(z, prevz) {
            return z
        }
    }
    // A million iterations has not converged; something is wrong.
    panic(fmt.Sprintf("CubeRoot(%g) did not converge", x))
}
```

This is only an example but real library functions should avoid `panic`. If the problem can be masked or worked around, it's always better to let things continue to run rather than taking down the whole program. 
>  实际中的库函数应该避免调用 `panic` ，如果问题可以被掩盖或者解决，最好让程序继续，而不是让整个程序崩溃

One possible counterexample is during initialization: if the library truly cannot set itself up, it might be reasonable to panic, so to speak.
>  一个可能的反例是在初始化期间：如果库无法完成自身设置，则触发 `panic` 是合理的

```go
var user = os.Getenv("USER")

func init() {
    if user == "" {
        panic("no value for $USER")
    }
}
```

### Recover
When `panic` is called, including implicitly for run-time errors such as indexing a slice out of bounds or failing a type assertion, it immediately stops execution of the current function and begins unwinding the stack of the goroutine, running any deferred functions along the way. If that unwinding reaches the top of the goroutine's stack, the program dies. However, it is possible to use the built-in function `recover` to regain control of the goroutine and resume normal execution.
>  `panic` 被调用时，它会终止当前函数执行，并回溯 goroutine 的栈，沿途执行任何延迟函数
>  如果此回溯到达 goroutine 的栈顶部，程序将崩溃
>  然而，可以使用内建函数 `recover` 来重新控制 goroutine 并恢复正常的执行流程

A call to `recover` stops the unwinding and returns the argument passed to `panic`. Because the only code that runs while unwinding is inside deferred functions, `recover` is only useful inside deferred functions.
>  `recover` 调用会停止回溯，并返回传递给 `panic` 的参数
>  因为在回溯过程中，唯一会运行的代码是在推迟函数中的代码，故 `recover` 只有在推迟函数中才有用

One application of `recover` is to shut down a failing goroutine inside a server without killing the other executing goroutines.
>  `recover` 的一个应用是关闭服务器中的故障 goroutine 而不杀死其他执行中的 goroutine (不调用 `recover` 的话，整个程序都会终止)

```go
func server(workChan <-chan *Work) {
    for work := range workChan {
        go safelyDo(work)
    }
}

func safelyDo(work *Work) {
    defer func() {
        if err := recover(); err != nil {
            log.Println("work failed:", err)
        }
    }()
    do(work)
}
```

In this example, if `do(work)` panics, the result will be logged and the goroutine will exit cleanly without disturbing the others. There's no need to do anything else in the deferred closure; calling `recover` handles the condition completely.
>  在上例中，如果 `do(work)` panic，其结果将被记录，goroutine 将退出而不影响其他 goroutine，我们需要做的仅仅是在推迟函数中调用 `recover` 处理程序故障即可

Because `recover` always returns `nil` unless called directly from a deferred function, deferred code can call library routines that themselves use `panic` and `recover` without failing. As an example, the deferred function in `safelyDo` might call a logging function before calling `recover`, and that logging code would run unaffected by the panicking state.
>  在非延迟函数中调用 `recover` 总是返回 `nil` (因为上下文中没有需要恢复的 `panic`) ，故我们在延迟函数中可以调用本身使用了 `panic` 和 `recover` 的代码，当前的 `panic` 上下文不会影响它们
>  例如，在 `safeDo` 中的延迟函数，可以先调用日志函数，再调用 `recover` ，当前的 panic 状态不会影响日志函数

With our recovery pattern in place, the `do` function (and anything it calls) can get out of any bad situation cleanly by calling `panic`. We can use that idea to simplify error handling in complex software. Let's look at an idealized version of a `regexp` package, which reports parsing errors by calling `panic` with a local error type. Here's the definition of `Error`, an `error` method, and the `Compile` function.
>  使用我们的恢复机制后，`do` 函数 (及其调用的任何内容) 可以通过调用 `panic` 干净地拜托任何糟糕的情况
>  考虑一个理想化的 `regexp` 包，它通过使用本地错误类型调用 `panic` 来报告解析错误

```go
// Error is the type of a parse error; it satisfies the error interface.
type Error string
func (e Error) Error() string {
    return string(e)
}

// error is a method of *Regexp that reports parsing errors by
// panicking with an Error.
func (regexp *Regexp) error(err string) {
    panic(Error(err))
}

// Compile returns a parsed representation of the regular expression.
func Compile(str string) (regexp *Regexp, err error) {
    regexp = new(Regexp)
    // doParse will panic if there is a parse error.
    defer func() {
        if e := recover(); e != nil {
            regexp = nil    // Clear return value.
            err = e.(Error) // Will re-panic if not a parse error.
        }
    }()
    return regexp.doParse(str), nil
}
```

If `doParse` panics, the recovery block will set the return value to `nil` —deferred functions can modify named return values. It will then check, in the assignment to `err`, that the problem was a parse error by asserting that it has the local type `Error`. If it does not, the type assertion will fail, causing a run-time error that continues the stack unwinding as though nothing had interrupted it. This check means that if something unexpected happens, such as an index out of bounds, the code will fail even though we are using `panic` and `recover` to handle parse errors.
>  如果 `doParse` panic，恢复部分代码会将返回值 `regexp` 设置为 `nil` (延迟函数可以修改命名的返回值)
>  然后，它会在将错误赋给 `err` 时检查问题是否是解析错误 (通过 type assertion `e` 是 `Error` 类型)，如果不是，type assertion 将失败，导致运行时错误，调用栈将被继续展开，就好像没有发生任何中断一样
>  这一 type assertion 检查意味着，如果发生了意外情况，例如索引越界，即便我们使用了 `panic` 和 `recover` 来处理解析错误，代码仍会失败 (因为我们没有针对处理除了解析故障意外的其他错误)

With error handling in place, the `error` method (because it's a method bound to a type, it's fine, even natural, for it to have the same name as the builtin `error` type) makes it easy to report parse errors without worrying about unwinding the parse stack by hand:
>  带有错误处理的情况下，`error` 方法（因为它绑定到某个类型，所以与内置的 `error` 类型同名是完全可以接受的，甚至是自然的）使得报告解析错误变得简单，而无需手动清理解析堆栈

```go
if pos == 0 {
    re.error("'*' illegal at start of expression")
}
```

Useful though this pattern is, it should be used only within a package. `Parse` turns its internal `panic` calls into `error` values; it does not expose `panics` to its client. That is a good rule to follow.
>  这个错误处理模式应该在包内实现，即 `Parse` 将其内部的 `panic` 调用转化为 `error` 值，不会将 panics 暴露给客户端

By the way, this re-panic idiom changes the panic value if an actual error occurs. However, both the original and new failures will be presented in the crash report, so the root cause of the problem will still be visible. Thus this simple re-panic approach is usually sufficient—it's a crash after all—but if you want to display only the original value, you can write a little more code to filter unexpected problems and re-panic with the original error. That's left as an exercise for the reader.
>  顺便说一下，这个重新引发 panic 的惯用法会在实际错误发生时更改 panic 值
>  然而，在崩溃报告中仍然会显示原始的和新的错误，因此问题的根本原因仍然可见
>  因此，这种简单的重新 panic 方法通常是足够的——毕竟这是一次崩溃——但如果你只想显示原始值，可以编写一些额外的代码来过滤意外问题，并使用原始错误重新 panic

## A web server
Let's finish with a complete Go program, a web server. This one is actually a kind of web re-server. Google provides a service at `chart.apis.google.com` that does automatic formatting of data into charts and graphs. It's hard to use interactively, though, because you need to put the data into the URL as a query. The program here provides a nicer interface to one form of data: given a short piece of text, it calls on the chart server to produce a QR code, a matrix of boxes that encode the text. That image can be grabbed with your cell phone's camera and interpreted as, for instance, a URL, saving you typing the URL into the phone's tiny keyboard.

Here's the complete program. An explanation follows.

```go
package main

import (
    "flag"
    "html/template"
    "log"
    "net/http"
)

var addr = flag.String("addr", ":1718", "http service address") // Q=17, R=18

var templ = template.Must(template.New("qr").Parse(templateStr))

func main() {
    flag.Parse()
    http.Handle("/", http.HandlerFunc(QR))
    err := http.ListenAndServe(*addr, nil)
    if err != nil {
        log.Fatal("ListenAndServe:", err)
    }
}

func QR(w http.ResponseWriter, req *http.Request) {
    templ.Execute(w, req.FormValue("s"))
}

const templateStr = `
<html>
<head>
<title>QR Link Generator</title>
</head>
<body>
{{if .}}
<img src="http://chart.apis.google.com/chart?chs=300x300&cht=qr&choe=UTF-8&chl={{.}}" />
<br>
{{.}}
<br>
<br>
{{end}}
<form action="/" name=f method="GET">
    <input maxLength=1024 size=70 name=s value="" title="Text to QR Encode">
    <input type=submit value="Show QR" name=qr>
</form>
</body>
</html>
`
```

The pieces up to `main` should be easy to follow. The one flag sets a default HTTP port for our server. The template variable `templ` is where the fun happens. It builds an HTML template that will be executed by the server to display the page; more about that in a moment.

The `main` function parses the flags and, using the mechanism we talked about above, binds the function `QR` to the root path for the server. Then `http.ListenAndServe` is called to start the server; it blocks while the server runs.

`QR` just receives the request, which contains form data, and executes the template on the data in the form value named `s`.

The template package `html/template` is powerful; this program just touches on its capabilities. In essence, it rewrites a piece of HTML text on the fly by substituting elements derived from data items passed to `templ.Execute`, in this case the form value. Within the template text (`templateStr`), double-brace-delimited pieces denote template actions. The piece from `{{if .}}` to `{{end}}` executes only if the value of the current data item, called `.` (dot), is non-empty. That is, when the string is empty, this piece of the template is suppressed.

The two snippets `{{.}}` say to show the data presented to the template—the query string—on the web page. The HTML template package automatically provides appropriate escaping so the text is safe to display.

The rest of the template string is just the HTML to show when the page loads. If this is too quick an explanation, see the [documentation](https://go.dev/pkg/html/template/) for the template package for a more thorough discussion.

And there you have it: a useful web server in a few lines of code plus some data-driven HTML text. Go is powerful enough to make a lot happen in a few lines.