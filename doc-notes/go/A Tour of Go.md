# Basics
## Packages, variables, and functions
**Packages**
Every Go program is made up of packages.

Programs start running in package `main`.

>  所有的 Go 程序都由包组成，程序从 `main` 包开始运行

This program is using the packages with import paths `"fmt"` and `"math/rand"`.

By convention, the package name is the same as the last element of the import path. For instance, the `"math/rand"` package comprises files that begin with the statement `package rand`.
>  按照惯例，包的名字和导入路径的最后一部分相同，例如，导入路径是 `math/rand` 的包中包含的文件中都会以该语句为开头 `package rand`

```go
package main

import (
    "fmt"
    "math/rand"
)

func main() {
    fmt.Println("My favorite number is", rand.Intn(10))
}
```

**Imports**
This code groups the imports into a parenthesized, "factored" import statement.

You can also write multiple import statements, like:

```go
import "fmt"
import "math"
```

But it is good style to use the factored import statement.

```go
package main

import (
    "fmt"
    "math"
)

func main() {
    fmt.Printf("Now you have %g problems.\n", math.Sqrt(7))
}
```

**Exported names**
In Go, a name is exported if it begins with a capital letter. For example, `Pizza` is an exported name, as is `Pi`, which is exported from the `math` package.

`pizza` and `pi` do not start with a capital letter, so they are not exported.

When importing a package, you can refer only to its exported names. Any "unexported" names are not accessible from outside the package.

>  Go 中，如果名字以大写字母开头，则它是一个导出的名称
>  导入一个包时，只能引用其导出的名称

Run the code. Notice the error message.

To fix the error, rename `math.pi` to `math.Pi` and try it again.

```go
package main

import (
    "fmt"
    "math"
)

func main() {
    fmt.Println(math.Pi)
}
```

**Functions**
A function can take zero or more arguments.

In this example, `add` takes two parameters of type `int`.

Notice that the type comes _after_ the variable name.

>  函数接受零个或多个参数
>  参数的类型在参数的名字后面

(For more about why types look the way they do, see the [article on Go's declaration syntax](https://go.dev/blog/gos-declaration-syntax).)

```go
package main

import "fmt"

func add(x int, y int) int {
    return x + y
}

func main() {
    fmt.Println(add(42, 13))
}
```

**Functions continued**
When two or more consecutive named function parameters share a type, you can omit the type from all but the last.
>  连续的函数参数具有相同类型时，可以进行简写

In this example, we shortened

```go
x int, y int
```

to

```go
x, y int
```

```go
package main

import "fmt"

func add(x, y int) int {
    return x + y
}

func main() {
    fmt.Println(add(42, 13))
}
```

**Multiple results**
A function can return any number of results.
>  函数可以返回任意数量的结果

The `swap` function returns two strings.

```go
package main

import "fmt"

func swap(x, y string) (string, string) {
    return y, x
}

func main() {
    a, b := swap("hello", "world")
    fmt.Println(a, b)
}
```

**Named return values**
Go's return values may be named. If so, they are treated as variables defined at the top of the function.

These names should be used to document the meaning of the return values.

A `return` statement without arguments returns the named return values. This is known as a "naked" return.

Naked return statements should be used only in short functions, as with the example shown here. They can harm readability in longer functions.

>  Go 中的返回值可以命名，它们会被视作在函数顶部定义的变量，返回值的名字可以用于说明返回值的含义
>  不带有参数的 `return` 语句会返回命名的返回值，这称为 “裸” 返回

```go
package main

import "fmt"

func split(sum int) (x, y int) {
    x = sum * 4 / 9
    y = sum - x
    return
}

func main() {
    fmt.Println(split(17))
}
```

**Variables**
The `var` statement declares a list of variables; as in function argument lists, the type is last.

A `var` statement can be at package or function level. We see both in this example.

>  `var` 语句用于声明一系列变量，变量的类型在后面
>  `var` 语句可以是包级别或者函数级别

```go
package main

import "fmt"

var c, python, java bool

func main() {
    var i int
    fmt.Println(i, c, python, java)
}
```

**Variables with initializers**
A var declaration can include initializers, one per variable.

If an initializer is present, the type can be omitted; the variable will take the type of the initializer.

>  `var` 声明支持初始值，如果给定初始值，可以忽略类型

```go
package main

import "fmt"

var i, j int = 1, 2

func main () {
	var c, python, java = true, false, "no!"
	fmt.Println (i, j, c, python, java)
}
```

**Short variable declarations**
Inside a function, the `:=` short assignment statement can be used in place of a `var` declaration with implicit type.

Outside a function, every statement begins with a keyword (`var`, `func`, and so on) and so the `:=` construct is not available.

```go
package main

import "fmt"

func main() {
    var i, j int = 1, 2
    k := 3
    c, python, java := true, false, "no!"

    fmt.Println(i, j, k, c, python, java)
}
```