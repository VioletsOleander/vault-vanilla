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
>  可以显式以 `field: value` 的形式指定字段，就不需要按需，且缺少的字段默认为零值

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

There are major differences between the ways arrays work in Go and C. In Go,

- Arrays are values. Assigning one array to another copies all the elements.
- In particular, if you pass an array to a function, it will receive a _copy_ of the array, not a pointer to it.
- The size of an array is part of its type. The types `[10]int` and `[20]int` are distinct.

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

### Slices[¶](https://go.dev/doc/effective_go#slices)

Slices wrap arrays to give a more general, powerful, and convenient interface to sequences of data. Except for items with explicit dimension such as transformation matrices, most array programming in Go is done with slices rather than simple arrays.

Slices hold references to an underlying array, and if you assign one slice to another, both refer to the same array. If a function takes a slice argument, changes it makes to the elements of the slice will be visible to the caller, analogous to passing a pointer to the underlying array. A `Read` function can therefore accept a slice argument rather than a pointer and a count; the length within the slice sets an upper limit of how much data to read. Here is the signature of the `Read` method of the `File` type in package `os`:

func (f *File) Read(buf []byte) (n int, err error)

The method returns the number of bytes read and an error value, if any. To read into the first 32 bytes of a larger buffer `buf`, _slice_ (here used as a verb) the buffer.

    n, err := f.Read(buf[0:32])

Such slicing is common and efficient. In fact, leaving efficiency aside for the moment, the following snippet would also read the first 32 bytes of the buffer.

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

The length of a slice may be changed as long as it still fits within the limits of the underlying array; just assign it to a slice of itself. The _capacity_ of a slice, accessible by the built-in function `cap`, reports the maximum length the slice may assume. Here is a function to append data to a slice. If the data exceeds the capacity, the slice is reallocated. The resulting slice is returned. The function uses the fact that `len` and `cap` are legal when applied to the `nil` slice, and return 0.

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

We must return the slice afterwards because, although `Append` can modify the elements of `slice`, the slice itself (the run-time data structure holding the pointer, length, and capacity) is passed by value.

The idea of appending to a slice is so useful it's captured by the `append` built-in function. To understand that function's design, though, we need a little more information, so we'll return to it later.

### Two-dimensional slices[¶](https://go.dev/doc/effective_go#two_dimensional_slices)

Go's arrays and slices are one-dimensional. To create the equivalent of a 2D array or slice, it is necessary to define an array-of-arrays or slice-of-slices, like this:

type Transform [3][3]float64  // A 3x3 array, really an array of arrays.
type LinesOfText [][]byte     // A slice of byte slices.

Because slices are variable-length, it is possible to have each inner slice be a different length. That can be a common situation, as in our `LinesOfText` example: each line has an independent length.

text := LinesOfText{
    []byte("Now is the time"),
    []byte("for all good gophers"),
    []byte("to bring some fun to the party."),
}

Sometimes it's necessary to allocate a 2D slice, a situation that can arise when processing scan lines of pixels, for instance. There are two ways to achieve this. One is to allocate each slice independently; the other is to allocate a single array and point the individual slices into it. Which to use depends on your application. If the slices might grow or shrink, they should be allocated independently to avoid overwriting the next line; if not, it can be more efficient to construct the object with a single allocation. For reference, here are sketches of the two methods. First, a line at a time:

// Allocate the top-level slice.
picture := make([][]uint8, YSize) // One row per unit of y.
// Loop over the rows, allocating the slice for each row.
for i := range picture {
    picture[i] = make([]uint8, XSize)
}

And now as one allocation, sliced into lines:

// Allocate the top-level slice, the same as before.
picture := make([][]uint8, YSize) // One row per unit of y.
// Allocate one large slice to hold all the pixels.
pixels := make([]uint8, XSize*YSize) // Has type []uint8 even though picture is [][]uint8.
// Loop over the rows, slicing each row from the front of the remaining pixels slice.
for i := range picture {
    picture[i], pixels = pixels[:XSize], pixels[XSize:]
}

### Maps[¶](https://go.dev/doc/effective_go#maps)

Maps are a convenient and powerful built-in data structure that associate values of one type (the _key_) with values of another type (the _element_ or _value_). The key can be of any type for which the equality operator is defined, such as integers, floating point and complex numbers, strings, pointers, interfaces (as long as the dynamic type supports equality), structs and arrays. Slices cannot be used as map keys, because equality is not defined on them. Like slices, maps hold references to an underlying data structure. If you pass a map to a function that changes the contents of the map, the changes will be visible in the caller.

Maps can be constructed using the usual composite literal syntax with colon-separated key-value pairs, so it's easy to build them during initialization.

var timeZone = map[string]int{
    "UTC":  0*60*60,
    "EST": -5*60*60,
    "CST": -6*60*60,
    "MST": -7*60*60,
    "PST": -8*60*60,
}

Assigning and fetching map values looks syntactically just like doing the same for arrays and slices except that the index doesn't need to be an integer.

offset := timeZone["EST"]

An attempt to fetch a map value with a key that is not present in the map will return the zero value for the type of the entries in the map. For instance, if the map contains integers, looking up a non-existent key will return `0`. A set can be implemented as a map with value type `bool`. Set the map entry to `true` to put the value in the set, and then test it by simple indexing.

attended := map[string]bool{
    "Ann": true,
    "Joe": true,
    ...
}

if attended[person] { // will be false if person is not in the map
    fmt.Println(person, "was at the meeting")
}

Sometimes you need to distinguish a missing entry from a zero value. Is there an entry for `"UTC"` or is that 0 because it's not in the map at all? You can discriminate with a form of multiple assignment.

var seconds int
var ok bool
seconds, ok = timeZone[tz]

For obvious reasons this is called the “comma ok” idiom. In this example, if `tz` is present, `seconds` will be set appropriately and `ok` will be true; if not, `seconds` will be set to zero and `ok` will be false. Here's a function that puts it together with a nice error report:

func offset(tz string) int {
    if seconds, ok := timeZone[tz]; ok {
        return seconds
    }
    log.Println("unknown time zone:", tz)
    return 0
}

To test for presence in the map without worrying about the actual value, you can use the [blank identifier](https://go.dev/doc/effective_go#blank) (`_`) in place of the usual variable for the value.

_, present := timeZone[tz]

To delete a map entry, use the `delete` built-in function, whose arguments are the map and the key to be deleted. It's safe to do this even if the key is already absent from the map.

delete(timeZone, "PDT")  // Now on Standard Time

### Printing[¶](https://go.dev/doc/effective_go#printing)

Formatted printing in Go uses a style similar to C's `printf` family but is richer and more general. The functions live in the `fmt` package and have capitalized names: `fmt.Printf`, `fmt.Fprintf`, `fmt.Sprintf` and so on. The string functions (`Sprintf` etc.) return a string rather than filling in a provided buffer.

You don't need to provide a format string. For each of `Printf`, `Fprintf` and `Sprintf` there is another pair of functions, for instance `Print` and `Println`. These functions do not take a format string but instead generate a default format for each argument. The `Println` versions also insert a blank between arguments and append a newline to the output while the `Print` versions add blanks only if the operand on neither side is a string. In this example each line produces the same output.

fmt.Printf("Hello %d\n", 23)
fmt.Fprint(os.Stdout, "Hello ", 23, "\n")
fmt.Println("Hello", 23)
fmt.Println(fmt.Sprint("Hello ", 23))

The formatted print functions `fmt.Fprint` and friends take as a first argument any object that implements the `io.Writer` interface; the variables `os.Stdout` and `os.Stderr` are familiar instances.

Here things start to diverge from C. First, the numeric formats such as `%d` do not take flags for signedness or size; instead, the printing routines use the type of the argument to decide these properties.

var x uint64 = 1<<64 - 1
fmt.Printf("%d %x; %d %x\n", x, x, int64(x), int64(x))

prints

18446744073709551615 ffffffffffffffff; -1 -1

If you just want the default conversion, such as decimal for integers, you can use the catchall format `%v` (for “value”); the result is exactly what `Print` and `Println` would produce. Moreover, that format can print _any_ value, even arrays, slices, structs, and maps. Here is a print statement for the time zone map defined in the previous section.

fmt.Printf("%v\n", timeZone)  // or just fmt.Println(timeZone)

which gives output:

map[CST:-21600 EST:-18000 MST:-25200 PST:-28800 UTC:0]

For maps, `Printf` and friends sort the output lexicographically by key.

When printing a struct, the modified format `%+v` annotates the fields of the structure with their names, and for any value the alternate format `%#v` prints the value in full Go syntax.

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

prints

&{7 -2.35 abc   def}
&{a:7 b:-2.35 c:abc     def}
&main.T{a:7, b:-2.35, c:"abc\tdef"}
map[string]int{"CST":-21600, "EST":-18000, "MST":-25200, "PST":-28800, "UTC":0}

(Note the ampersands.) That quoted string format is also available through `%q` when applied to a value of type `string` or `[]byte`. The alternate format `%#q` will use backquotes instead if possible. (The `%q` format also applies to integers and runes, producing a single-quoted rune constant.) Also, `%x` works on strings, byte arrays and byte slices as well as on integers, generating a long hexadecimal string, and with a space in the format (`% x`) it puts spaces between the bytes.

Another handy format is `%T`, which prints the _type_ of a value.

fmt.Printf("%T\n", timeZone)

prints

map[string]int

If you want to control the default format for a custom type, all that's required is to define a method with the signature `String() string` on the type. For our simple type `T`, that might look like this.

func (t *T) String() string {
    return fmt.Sprintf("%d/%g/%q", t.a, t.b, t.c)
}
fmt.Printf("%v\n", t)

to print in the format

7/-2.35/"abc\tdef"

(If you need to print _values_ of type `T` as well as pointers to `T`, the receiver for `String` must be of value type; this example used a pointer because that's more efficient and idiomatic for struct types. See the section below on [pointers vs. value receivers](https://go.dev/doc/effective_go#pointers_vs_values) for more information.)

Our `String` method is able to call `Sprintf` because the print routines are fully reentrant and can be wrapped this way. There is one important detail to understand about this approach, however: don't construct a `String` method by calling `Sprintf` in a way that will recur into your `String` method indefinitely. This can happen if the `Sprintf` call attempts to print the receiver directly as a string, which in turn will invoke the method again. It's a common and easy mistake to make, as this example shows.

type MyString string

func (m MyString) String() string {
    return fmt.Sprintf("MyString=%s", m) // Error: will recur forever.
}

It's also easy to fix: convert the argument to the basic string type, which does not have the method.

type MyString string
func (m MyString) String() string {
    return fmt.Sprintf("MyString=%s", string(m)) // OK: note conversion.
}

In the [initialization section](https://go.dev/doc/effective_go#initialization) we'll see another technique that avoids this recursion.

Another printing technique is to pass a print routine's arguments directly to another such routine. The signature of `Printf` uses the type `...interface{}` for its final argument to specify that an arbitrary number of parameters (of arbitrary type) can appear after the format.

func Printf(format string, v ...interface{}) (n int, err error) {

Within the function `Printf`, `v` acts like a variable of type `[]interface{}` but if it is passed to another variadic function, it acts like a regular list of arguments. Here is the implementation of the function `log.Println` we used above. It passes its arguments directly to `fmt.Sprintln` for the actual formatting.

// Println prints to the standard logger in the manner of fmt.Println.
func Println(v ...interface{}) {
    std.Output(2, fmt.Sprintln(v...))  // Output takes parameters (int, string)
}

We write `...` after `v` in the nested call to `Sprintln` to tell the compiler to treat `v` as a list of arguments; otherwise it would just pass `v` as a single slice argument.

There's even more to printing than we've covered here. See the `godoc` documentation for package `fmt` for the details.

By the way, a `...` parameter can be of a specific type, for instance `...int` for a min function that chooses the least of a list of integers:

func Min(a ...int) int {
    min := int(^uint(0) >> 1)  // largest int
    for _, i := range a {
        if i < min {
            min = i
        }
    }
    return min
}

### Append[¶](https://go.dev/doc/effective_go#append)

Now we have the missing piece we needed to explain the design of the `append` built-in function. The signature of `append` is different from our custom `Append` function above. Schematically, it's like this:

func append(slice []_T_, elements ..._T_) []_T_

where _T_ is a placeholder for any given type. You can't actually write a function in Go where the type `T` is determined by the caller. That's why `append` is built in: it needs support from the compiler.

What `append` does is append the elements to the end of the slice and return the result. The result needs to be returned because, as with our hand-written `Append`, the underlying array may change. This simple example

x := []int{1,2,3}
x = append(x, 4, 5, 6)
fmt.Println(x)

prints `[1 2 3 4 5 6]`. So `append` works a little like `Printf`, collecting an arbitrary number of arguments.

But what if we wanted to do what our `Append` does and append a slice to a slice? Easy: use `...` at the call site, just as we did in the call to `Output` above. This snippet produces identical output to the one above.

x := []int{1,2,3}
y := []int{4,5,6}
x = append(x, y...)
fmt.Println(x)

Without that `...`, it wouldn't compile because the types would be wrong; `y` is not of type `int`.