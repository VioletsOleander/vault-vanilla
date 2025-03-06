# Basics
## Packages, variables, and functions
### Packages
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

### Functions
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

### Variables
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

>  在函数中，`xxx := value` 等价于 `var xxx = value`
>  在函数外，要求每个语句都以一个关键字开头，因此 `:=` 不可用

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

**Basic types**
Go's basic types are

```
bool

string

int  int8  int16  int32  int64
uint uint8 uint16 uint32 uint64 uintptr

byte // alias for uint8

rune // alias for int32
     // represents a Unicode code point

float32 float64

complex64 complex128
```

The example shows variables of several types, and also that variable declarations may be "factored" into blocks, as with import statements.

The `int`, `uint`, and `uintptr` types are usually 32 bits wide on 32-bit systems and 64 bits wide on 64-bit systems. When you need an integer value you should use `int` unless you have a specific reason to use a sized or unsigned integer type.
>  `int, uint, uintptr` 在 32 位机上是 32 位，在 64 位机上是 64 位

```go
package main

import (
    "fmt"
    "math/cmplx"
)

var (
    ToBe   bool       = false
    MaxInt uint64     = 1<<64 - 1
    z      complex128 = cmplx.Sqrt(-5 + 12i)
)

func main() {
    fmt.Printf("Type: %T Value: %v\n", ToBe, ToBe)
    fmt.Printf("Type: %T Value: %v\n", MaxInt, MaxInt)
    fmt.Printf("Type: %T Value: %v\n", z, z)
}
```

**Zero values**
Variables declared without an explicit initial value are given their _zero value_.
>  声明的变量没有给定显式初始值时，都会被赋予零值

The zero value is:

- `0` for numeric types,
- `false` for the boolean type, and
- `""` (the empty string) for strings.

```go
package main

import "fmt"

func main() {
    var i int
    var f float64
    var b bool
    var s string
    fmt.Printf("%v %v %v %q\n", i, f, b, s)
}
```

**Type conversions**
The expression `T(v)` converts the value `v` to the type `T`.
>  表达式 `T(v)` 用于类型转换

Some numeric conversions:

```go
var i int = 42
var f float64 = float64(i)
var u uint = uint(f)
```

Or, put more simply:

```go
i := 42
f := float64(i)
u := uint(f)
```

Unlike in C, in Go assignment between items of different type requires an explicit conversion. Try removing the `float64` or `uint` conversions in the example and see what happens.
>  Go 中，不同类型间的赋值必须有显式类型转换

```go
package main

import (
    "fmt"
    "math"
)

func main() {
    var x, y int = 3, 4
    var f float64 = math.Sqrt(float64(x*x + y*y))
    var z uint = uint(f)
    fmt.Println(x, y, z)
}
```

**Type inference**
When declaring a variable without specifying an explicit type (either by using the `:=` syntax or `var =` expression syntax), the variable's type is inferred from the value on the right hand side.

When the right hand side of the declaration is typed, the new variable is of that same type:

```go
var i int
j := i // j is an int
```

But when the right hand side contains an untyped numeric constant, the new variable may be an `int`, `float64`, or `complex128` depending on the precision of the constant:

```go
i := 42           // int
f := 3.142        // float64
g := 0.867 + 0.5i // complex128
```

Try changing the initial value of `v` in the example code and observe how its type is affected.

```go
package main

import "fmt"

func main() {
    v := 'll' // change me!
    fmt.Printf("v is of type %T\n", v)
}
```

**Constants**
Constants are declared like variables, but with the `const` keyword.

Constants can be character, string, boolean, or numeric values.

Constants cannot be declared using the `:=` syntax.

>  常量的声明和变量类似，但带有 `const` 关键字
>  常量不能用 `:=` 声明

```go
package main

import "fmt"

const Pi = 3.14

func main() {
    const World = "世界"
    fmt.Println("Hello", World)
    fmt.Println("Happy", Pi, "Day")

    const Truth = true
    fmt.Println("Go rules?", Truth)
}
```

**Numeric Constants**
Numeric constants are high-precision _values_.

An untyped constant takes the type needed by its context.

Try printing `needInt(Big)` too.

(An `int` can store at maximum a 64-bit integer, and sometimes less.)

```go
package main

import "fmt"

const (
    // Create a huge number by shifting a 1 bit left 100 places.
    // In other words, the binary number that is 1 followed by 100 zeroes.
    Big = 1 << 100
    // Shift it right again 99 places, so we end up with 1<<1, or 2.
    Small = Big >> 99
)

func needInt(x int) int { return x*10 + 1 }
func needFloat(x float64) float64 {
    return x * 0.1
}

func main() {
    fmt.Println(needInt(Small))
    fmt.Println(needFloat(Small))
    fmt.Println(needFloat(Big))
}
```

## Flow control statements: for, if, else, switch and defer
### looping construct
**For**
Go has only one looping construct, the `for` loop.

The basic `for` loop has three components separated by semicolons:

- the init statement: executed before the first iteration
- the condition expression: evaluated before every iteration
- the post statement: executed at the end of every iteration

The init statement will often be a short variable declaration, and the variables declared there are visible only in the scope of the `for` statement.

The loop will stop iterating once the boolean condition evaluates to `false`.

**Note:** Unlike other languages like C, Java, or JavaScript there are no parentheses surrounding the three components of the `for` statement and the braces `{ }` are always required.

>  Go 只有 `for` 用于循环构造
>  `for` 构造中，`init` 语句通常是一个简短的变量声明，在这里声明的变量仅在 `for` 语句的作用域内可见

```go
package main

import "fmt"

func main() {
    sum := 0
    for i := 0; i < 10; i++ {
        sum += i
    }
    fmt.Println(sum)
}
```

**For continued**
The init and post statements are optional.
>  `for` 构造可以只给出条件

```go
package main

import "fmt"

func main() {
    sum := 1
    for ; sum < 1000; {
        sum += sum
    }
    fmt.Println(sum)
}
```

**For is Go's "while"**
At that point you can drop the semicolons: C's `while` is spelled `for` in Go.
>  仅给出条件是 `for` 就等价于一个 while

```go
package main

import "fmt"

func main() {
    sum := 1
    for sum < 1000 {
        sum += sum
    }
    fmt.Println(sum)
}
```

**Forever**
If you omit the loop condition it loops forever, so an infinite loop is compactly expressed.

```go
package main

func main() {
    for {
    }
}
```

### Condition construct
**If**
Go's `if` statements are like its `for` loops; the expression need not be surrounded by parentheses `( )` but the braces `{ }` are required.

```go
package main

import (
    "fmt"
    "math"
)

func sqrt(x float64) string {
    if x < 0 {
        return sqrt(-x) + "i"
    }
    return fmt.Sprint(math.Sqrt(x))
}

func main() {
    fmt.Println(sqrt(2), sqrt(-4))
}
```

**If with a short statement**
Like `for`, the `if` statement can start with a short statement to execute before the condition.

Variables declared by the statement are only in scope until the end of the `if`.

>  `if` 也可以在条件前声明临时变量，该变量仅对 `if` 结构内的作用域可见

(Try using `v` in the last `return` statement.)

```go
package main

import (
    "fmt",
    "math"
)

func pow(x, n, lim float64) float64 {
    if v := math.Pow(x, n); v < lim {
        return v
    }
    return lim
}

func main() {
    fmt.Println(
        pow(3, 2, 10),
        pow(3, 3, 20),
    )
}

```

**If and else**
Variables declared inside an `if` short statement are also available inside any of the `else` blocks.

(Both calls to `pow` return their results before the call to `fmt.Println` in `main` begins.)

```go
package main

import (
    "fmt"
    "math"
)

func pow(x, n, lim float64) float64 {
    if v := math.Pow(x, n); v < lim {
        return v
    } else {
        fmt.Printf("%g >= %g\n", v, lim)
    }
    // can't use v here, though
    return lim
}

func main() {
    fmt.Println(
        pow(3, 2, 10),
        pow(3, 3, 20),
    )
}
```

**Switch**
A `switch` statement is a shorter way to write a sequence of `if - else` statements. It runs the first case whose value is equal to the condition expression.

Go's switch is like the one in C, C++, Java, JavaScript, and PHP, except that Go only runs the selected case, not all the cases that follow. In effect, the `break` statement that is needed at the end of each case in those languages is provided automatically in Go. Another important difference is that Go's switch cases need not be constants, and the values involved need not be integers.
>  Go 的 `switch` 只会运行匹配的部分，之后的部分不会随之运行，Go 的 `switch` case 也不必要是常数和整形

```go
package main

import (
    "fmt"
    "runtime"
)

func main() {
    fmt.Print("Go runs on ")
    switch os := runtime.GOOS; os {
    case "darwin":
        fmt.Println("OS X.")
    case "linux":
        fmt.Println("Linux.")
    default:
        // freebsd, openbsd,
        // plan9, windows...
        fmt.Printf("%s.\n", os)
    }
}
```

**Switch evaluation order**
Switch cases evaluate cases from top to bottom, stopping when a case succeeds.

(For example,

```go
switch i {
case 0:
case f():
}
```

does not call `f` if `i==0`.)

**Note:** Time in the Go playground always appears to start at 2009-11-10 23:00:00 UTC, a value whose significance is left as an exercise for the reader.

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    fmt.Println("When's Saturday?")
    today := time.Now().Weekday()
    switch time.Saturday {
    case today + 0:
        fmt.Println("Today.")
    case today + 1:
        fmt.Println("Tomorrow.")
    case today + 2:
        fmt.Println("In two days.")
    default:
        fmt.Println("Too far away.")
    }
}
```

**Switch with no condition**
Switch without a condition is the same as `switch true`.

This construct can be a clean way to write long if-then-else chains.

>  没有条件的 `switch` 等价于 `switch true`

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    t := time.Now()
    switch {
    case t.Hour() < 12:
        fmt.Println("Good morning!")
    case t.Hour() < 17:
        fmt.Println("Good afternoon.")
    default:
        fmt.Println("Good evening.")
    }
}
```

### Defer
**Defer**
A defer statement defers the execution of a function until the surrounding function returns.

The deferred call's arguments are evaluated immediately, but the function call is not executed until the surrounding function returns.

>  `defer` 语句将函数的执行推迟到其周围函数执行完成后
>  被 defer 的调用的参数会被立即评估，但函数不会被立即执行

```go
package main

import "fmt"

func main() {
    defer fmt.Println("world")

    fmt.Println("hello")
}
```

**Stacking defers**
Deferred function calls are pushed onto a stack. When a function returns, its deferred calls are executed in last-in-first-out order.
>  被 defer 的函数调用以栈形式存储，当一个函数返回时，Go 从栈中调出最晚被 defer 的函数调用，即后进先出

To learn more about defer statements read this [blog post](https://go.dev/blog/defer-panic-and-recover).

```go
package main

import "fmt"

func main() {
    fmt.Println("counting")

    for i := 0; i < 10; i++ {
        defer fmt.Println(i)
    }

    fmt.Println("done")
}
```

## More types: structs, slices, and maps
### Pointers
**Pointers**
Go has pointers. A pointer holds the memory address of a value.

The type `*T` is a pointer to a `T` value. Its zero value is `nil`.

```go
var p *int
```

>  类型 `*T` 表示指向 `T` 类型的指针，指针的默认值为 `nil` 

The `&` operator generates a pointer to its operand.

```go
i := 42
p = &i
```

>  `&` 用于取地址

The `*` operator denotes the pointer's underlying value.

```go
fmt.Println(*p) // read i through the pointer p
*p = 21         // set i through the pointer p
```

This is known as "dereferencing" or "indirecting".

>  `*` 用于解引用，即获取指针所指向地址的值

Unlike C, Go has no pointer arithmetic.
>  Go 不支持指针算数

```go
package main

import "fmt"

func main() {
    i, j := 42, 2701

    p := &i         // point to i
    fmt.Println(*p) // read i through the pointer
    *p = 21         // set i through the pointer
    fmt.Println(i)  // see the new value of i

    p = &j         // point to j
    *p = *p / 37   // divide j through the pointer
    fmt.Println(j) // see the new value of j
}
```

### Structs
**Structs**
A `struct` is a collection of fields.

```go
package main

import "fmt"

type Vertex struct {
    X int
    Y int
}

func main() {
    fmt.Println(Vertex{1, 2})
}
```

**Struct Fields**
Struct fields are accessed using a dot.

```go
package main

import "fmt"

type Vertex struct {
    X int
    Y int
}

func main() {
    v := Vertex{1, 2}
    v.X = 4
    fmt.Println(v.X)
}
```

**Pointers to structs**
Struct fields can be accessed through a struct pointer.

To access the field `X` of a struct when we have the struct pointer `p` we could write `(*p).X`. However, that notation is cumbersome, so the language permits us instead to write just `p.X`, without the explicit dereference.

```go
package main

import "fmt"

type Vertex struct {
    X int
    Y int
}

func main() {
    v := Vertex{1, 2}
    p := &v
    p.X = 1e9
    fmt.Println(v)
}
```

**Struct Literals**
A struct literal denotes a newly allocated struct value by listing the values of its fields.

You can list just a subset of fields by using the `Name:` syntax. (And the order of named fields is irrelevant.)
>  结构体初始化时可以用 `field_name: value` 的格式

The special prefix `&` returns a pointer to the struct value.

```go
package main

import "fmt"

type Vertex struct {
    X, Y int
}

var (
    v1 = Vertex{1, 2}  // has type Vertex
    v2 = Vertex{X: 1}  // Y:0 is implicit
    v3 = Vertex{}      // X:0 and Y:0
    p  = &Vertex{1, 2} // has type *Vertex
)

func main() {
    fmt.Println(v1, p, v2, v3)
}
```

### Slices
**Arrays**
The type `[n]T` is an array of `n` values of type `T`.

The expression

```
var a [10]int
```

declares a variable `a` as an array of ten integers.

>  类型 `[n]T` 表示 `n` 维类型为 `T` 的数组

An array's length is part of its type, so arrays cannot be resized. This seems limiting, but don't worry; Go provides a convenient way of working with arrays.
>  数组的长度是其类型的一部分，故数组不可变长

```go
package main

import "fmt"

func main() {
    var a [2]string
    a[0] = "Hello"
    a[1] = "World"
    fmt.Println(a[0], a[1])
    fmt.Println(a)

    primes := [6]int{2, 3, 5, 7, 11, 13}
    fmt.Println(primes)
}
```

**Slices**
An array has a fixed size. A slice, on the other hand, is a dynamically-sized, flexible view into the elements of an array. In practice, slices are much more common than arrays.

The type `[]T` is a slice with elements of type `T`.

A slice is formed by specifying two indices, a low and high bound, separated by a colon:

```go
a[low : high]
```

This selects a half-open range which includes the first element, but excludes the last one.

The following expression creates a slice which includes elements 1 through 3 of `a`:

```go
a[1:4]
```

>  切片是可变长的，切片的类型为 `[]T`
>  切片通过指定特定数组范围初始化

```go
package main

import "fmt"

func main() {
    primes := [6]int{2, 3, 5, 7, 11, 13}

    var s []int = primes[1:4]
    fmt.Println(s)
}
```

**Slices are like references to arrays**
A slice does not store any data, it just describes a section of an underlying array.

Changing the elements of a slice modifies the corresponding elements of its underlying array.

Other slices that share the same underlying array will see those changes.

>  切片不存储数据，仅引用

```go
package main

import "fmt"

func main() {
    names := [4]string{
        "John",
        "Paul",
        "George",
        "Ringo",
    }
    fmt.Println(names)

    a := names[0:2]
    b := names[1:3]
    fmt.Println(a, b)

    b[0] = "XXX"
    fmt.Println(a, b)
    fmt.Println(names)
}
```

**Slice literals**
A slice literal is like an array literal without the length.

This is an array literal:

```
[3]bool{true, true, false}
```

And this creates the same array as above, then builds a slice that references it:

```
[]bool{true, true, false}
```

>  切片字面值即先创建一个数组，然后引用它

```go
package main

import "fmt"

func main() {
    q := []int{2, 3, 5, 7, 11, 13}
    fmt.Println(q)

    r := []bool{true, false, true, true, false, true}
    fmt.Println(r)

    s := []struct {
        i int
        b bool
    }{
        {2, true},
        {3, false},
        {5, true},
        {7, true},
        {11, false},
        {13, true},
    }
    fmt.Println(s)
}
```

**Slice defaults**
When slicing, you may omit the high or low bounds to use their defaults instead.

The default is zero for the low bound and the length of the slice for the high bound.

For the array

```go
var a [10]int
```

these slice expressions are equivalent:

```go
a[0:10]
a[:10]
a[0:]
a[:]
```

**Slice length and capacity**
A slice has both a _length_ and a _capacity_.

The length of a slice is the number of elements it contains.

The capacity of a slice is the number of elements in the underlying array, counting from the first element in the slice.

The length and capacity of a slice `s` can be obtained using the expressions `len(s)` and `cap(s)`.

You can extend a slice's length by re-slicing it, provided it has sufficient capacity. Try changing one of the slice operations in the example program to extend it beyond its capacity and see what happens.

>  切片的长度指它包含的元素数量，切片的容量指在它引用的数组中，从它引用的第一个元素到数组尾部的元素数量
>  切片的长度可以在容量的范围下增大，方法就是对自己重新切片

```go
package main

import "fmt"

func main() {
    s := []int{2, 3, 5, 7, 11, 13}
    printSlice(s)

    // Slice the slice to give it zero length.
    s = s[:0]
    printSlice(s)

    // Extend its length.
    s = s[:4]
    printSlice(s)

    // Drop its first two values.
    s = s[2:]
    printSlice(s)
}

func printSlice(s []int) {
    fmt.Printf("len=%d cap=%d %v\n", len(s), cap(s), s)
}
```

**Nil slices**
The zero value of a slice is `nil`.

A nil slice has a length and capacity of 0 and has no underlying array.

>  切片的零值也是 `nil` ，`nil` 长度容量都为 0，没有底层数组

**Creating a slice with make**
Slices can be created with the built-in `make` function; this is how you create dynamically-sized arrays.

The `make` function allocates a zeroed array and returns a slice that refers to that array:

```go
a := make([]int, 5)  // len(a)=5
```

To specify a capacity, pass a third argument to `make`:

```go
b := make([]int, 0, 5) // len(b)=0, cap(b)=5

b = b[:cap(b)] // len(b)=5, cap(b)=5
b = b[1:]      // len(b)=4, cap(b)=4
```

>  `make` 函数用于构建切片，它会分配全零数组，然后返回其切片

**Slices of slices**
Slices can contain any type, including other slices.
>  切片的底层数组可以存储任意类型，包括切片

```go
package main

import (
	"fmt"
	"strings"
)

func main () {
	// Create a tic-tac-toe board.
	board := [][]string{
		[]string{"_", "_", "_"},
		[]string{"_", "_", "_"},
		[]string{"_", "_", "_"},
	}

	// The players take turns.
	board[0][0] = "X"
	board[2][2] = "O"
	board[1][2] = "X"
	board[1][0] = "O"
	board[0][2] = "X"

	for i := 0; i < len(board); i++ {
		fmt.Printf("%s\n", strings.Join(board[i], " "))
	}
}
```

**Appending to a slice**
It is common to append new elements to a slice, and so Go provides a built-in `append` function. The [documentation](https://go.dev/pkg/builtin/#append) of the built-in package describes `append`.

```go
func append(s []T, vs ...T) []T
```

The first parameter `s` of `append` is a slice of type `T`, and the rest are `T` values to append to the slice.

The resulting value of `append` is a slice containing all the elements of the original slice plus the provided values.

If the backing array of `s` is too small to fit all the given values a bigger array will be allocated. The returned slice will point to the newly allocated array.

>  `append` 函数用于向切片添加新元素，添加新元素可能会增大底层数组 (创建一个新数组)

(To learn more about slices, read the [Slices: usage and internals](https://go.dev/blog/go-slices-usage-and-internals) article.)

**Range**
The `range` form of the `for` loop iterates over a slice or map.

When ranging over a slice, two values are returned for each iteration. The first is the index, and the second is a copy of the element at that index.

>  `for i, v := range slice` 用于遍历切片，类似 python `enumerate`

```go
package main

import "fmt"

var pow = []int{1, 2, 4, 8, 16, 32, 64, 128}

func main() {
    for i, v := range pow {
        fmt.Printf("2**%d = %d\n", i, v)
    }
}
```

**Range continued**
You can skip the index or value by assigning to `_`.

```go
for i, _ := range pow
for _, value := range pow
```

If you only want the index, you can omit the second variable.

```go
for i := range pow
```

### Maps
**Maps**
A map maps keys to values.

The zero value of a map is `nil`. A `nil` map has no keys, nor can keys be added.

The `make` function returns a map of the given type, initialized and ready for use.

>  `map` 的零值也是 `nil` ，`make` 也用于构造 map

```go
package main

import "fmt"

type Vertex struct {
    Lat, Long float64
}

var m map[string]Vertex

func main() {
    m = make(map[string]Vertex)
    m["Bell Labs"] = Vertex{
        40.68433, -74.39967,
    }
    fmt.Println(m["Bell Labs"])
}
```

**Map literals**
Map literals are like struct literals, but the keys are required.

```go
package main

import "fmt"

type Vertex struct {
    Lat, Long float64
}

var m = map[string]Vertex{
    "Bell Labs": Vertex{
        40.68433, -74.39967,
    },
    "Google": Vertex{
        37.42202, -122.08408,
    },
}

func main() {
    fmt.Println(m)
}
```

**Map literals continued**
If the top-level type is just a type name, you can omit it from the elements of the literal.

```go
package main

import "fmt"

type Vertex struct {
    Lat, Long float64
}

var m = map[string]Vertex{
    "Bell Labs": {40.68433, -74.39967},
    "Google":    {37.42202, -122.08408},
}

func main() {
    fmt.Println(m)
}
```

**Mutating Maps**
Insert or update an element in map `m`:

```
m[key] = elem
```

Retrieve an element:

```
elem = m[key]
```

Delete an element:

```
delete(m, key)
```

Test that a key is present with a two-value assignment:

```go
elem, ok = m[key]
```

If `key` is in `m`, `ok` is `true`. If not, `ok` is `false`.

If `key` is not in the map, then `elem` is the zero value for the map's element type.

**Note:** If `elem` or `ok` have not yet been declared you could use a short declaration form:

```go
elem, ok := m[key]
```

**Function values**
Functions are values too. They can be passed around just like other values.

Function values may be used as function arguments and return values.

>  函数本身也是值，可以用于函数参数和返回值

```go
package main

import (
    "fmt"
    "math"
)

func compute(fn func(float64, float64) float64) float64 {
    return fn(3, 4)
}

func main() {
    hypot := func(x, y float64) float64 {
        return math.Sqrt(x*x + y*y)
    }
    fmt.Println(hypot(5, 12))

    fmt.Println(compute(hypot))
    fmt.Println(compute(math.Pow))
}
```

**Function closures**
Go functions may be closures. A closure is a function value that references variables from outside its body. The function may access and assign to the referenced variables; in this sense the function is "bound" to the variables.
>  函数可以是闭包，闭包即函数引用了函数体外的变量，该函数会访问并对这个变量赋值，某种意义上，该函数绑定到了该变量上 (绑定的变量不限于一个)
>  闭包一般是匿名的

For example, the `adder` function returns a closure. Each closure is bound to its own `sum` variable.
>  下例中，`addr` 函数返回一个闭包，每个闭包都绑定到其自己的 `sum` 变量

```go
package main

import "fmt"

func adder() func(int) int {
    sum := 0
    return func(x int) int {
        sum += x
        return sum
    }
}

func main() {
    pos, neg := adder(), adder()
    for i := 0; i < 10; i++ {
        fmt.Println(
            pos(i),
            neg(-2*i),
        )
    }
}
```

# Methods and interface
## Methods
Go does not have classes. However, you can define methods on types.

A method is a function with a special _receiver_ argument.

The receiver appears in its own argument list between the `func` keyword and the method name.

>  Go 没有类，但可以为类型定义方法
>  方法即带有一个特殊 receiver 参数的函数，receiver 应该在 `func` 关键字和函数名之间指定

In this example, the `Abs` method has a receiver of type `Vertex` named `v`.

```go
package main

import (
    "fmt"
    "math"
)

type Vertex struct {
    X, Y float64
}

func (v Vertex) Abs() float64 {
    return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func main() {
    v := Vertex{3, 4}
    fmt.Println(v.Abs())
}
```

**Methods are functions**
Remember: a method is just a function with a receiver argument.

Here's `Abs` written as a regular function with no change in functionality.

```go
package main

import (
    "fmt"
    "math"
)

type Vertex struct {
    X, Y float64
}

func Abs(v Vertex) float64 {
    return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func main() {
    v := Vertex{3, 4}
    fmt.Println(v.Abs())
}
```

**Methods continued**
You can declare a method on non-struct types, too.

In this example we see a numeric type `MyFloat` with an `Abs` method.

You can only declare a method with a receiver whose type is defined in the same package as the method. You cannot declare a method with a receiver whose type is defined in another package (which includes the built-in types such as `int`).

>  不能为定义在其他包中的类型定义方法，即不能用它们作为 receiver，这也包括了内建类型例如 `int` ，因为它们都定义在其他包中

```go
package main

import (
    "fmt"
    "math"
)

type MyFloat float64

func (f MyFloat) Abs() float64 {
    if f < 0 {
        return float64(-f)
    }
    return float64(f)
}

func main() {
    f := MyFloat(-math.Sqrt2)
    fmt.Println(f.Abs())
}
```

**Pointer receivers**  
You can declare methods with pointer receivers.

This means the receiver type has the literal syntax `*T` for some type `T`. (Also, `T` cannot itself be a pointer such as `*int`.)

>  receiver 还可以是指针类型，即 `*T` ，注意其中的 `T` 需要是某个自定义类型，不能是内建类型以及其他包中的类型

For example, the `Scale` method here is defined on `*Vertex`.

Methods with pointer receivers can modify the value to which the receiver points (as `Scale` does here). Since methods often need to modify their receiver, pointer receivers are more common than value receivers.
>  指针 receiver 比值形式的 receiver 更加常用，因为指针 receiver 传递指针，故方法可以修改 receiver 自身的值，而值形式的 receiver 传递拷贝的值，方法中的修改仅在方法内有效

Try removing the `*` from the declaration of the `Scale` function on line 16 and observe how the program's behavior changes.

With a value receiver, the `Scale` method operates on a copy of the original `Vertex` value. (This is the same behavior as for any other function argument.) The `Scale` method must have a pointer receiver to change the `Vertex` value declared in the `main` function.
>  Go 中任意的函数参数传递形式都是值传递

```go
package main

import (
    "fmt"
    "math"
)

type Vertex struct {
    X, Y float64
}

func (v Vertex) Abs() float64 {
    return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func (v *Vertex) Scale(f float64) {
    v.X = v.X * f
    v.Y = v.Y * f
}

func main() {
    v := Vertex{3, 4}
    v.Scale(10)
    fmt.Println(v.Abs())
}
```

**Pointers and functions**
Here we see the `Abs` and `Scale` methods rewritten as functions.

Again, try removing the `*` from line 16. Can you see why the behavior changes? What else did you need to change for the example to compile?

(If you're not sure, continue to the next page.)

```go
package main

import (
    "fmt"
    "math"
)

type Vertex struct {
    X, Y float64
}

func Abs(v Vertex) float64 {
    return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func Scale(v *Vertex) (f float64) {
    v.X = v.X * f
    v.Y = v.Y * f
}

func main() {
    v := Vertex{3, 4}
    v.Scale(10)
    fmt.Println(v.Abs())
}
```

**Methods and pointer indirection**
Comparing the previous two programs, you might notice that functions with a pointer argument must take a pointer:

```go
var v Vertex
ScaleFunc(v, 5)  // Compile error!
ScaleFunc(&v, 5) // OK
```

while methods with pointer receivers take either a value or a pointer as the receiver when they are called:

```go
var v Vertex
v.Scale(5)  // OK
p := &v
p.Scale(10) // OK
```

For the statement `v.Scale(5)`, even though `v` is a value and not a pointer, the method with the pointer receiver is called automatically. That is, as a convenience, Go interprets the statement `v.Scale(5)` as `(&v).Scale(5)` since the `Scale` method has a pointer receiver.

>  receiver 为 `*T` 的方法，不仅可以由类型为 `*T` 的变量调用，类型为 `T` 的变量也可以调用，Go 会自动将其转化为 `(&v).method`

```go
package main

import "fmt"

type Vertex struct {
    X, Y float64
}

func (v *Vertex) Scale(f float64) {
    v.X = v.X * f
    v.Y = v.Y * f
}

func ScaleFunc(v *Vertex, f float64) {
    v.X = v.X * f
    v.Y = v.Y * f
}

func main() {
    v := Vertex{3, 4}
    v.Scale(2)
    ScaleFunc(&v, 10)

    p := &Vertex{4, 3}
    p.Scale(3)
    ScaleFunc(p, 8)

    fmt.Println(v, p)
}
```

**Methods and pointer indirection (2)**
The equivalent thing happens in the reverse direction.

Functions that take a value argument must take a value of that specific type:

```go
var v Vertex
fmt.Println(AbsFunc(v))  // OK
fmt.Println(AbsFunc(&v)) // Compile error!
```

while methods with value receivers take either a value or a pointer as the receiver when they are called:

```go
var v Vertex
fmt.Println(v.Abs()) // OK
p := &v
fmt.Println(p.Abs()) // OK
```

>  同样，receiver 类型为 `T` 的方法也可以由类型为 `*T` 的变量调用

In this case, the method call `p.Abs()` is interpreted as `(*p).Abs()`.

```go
package main

import (
    "fmt"
    "math"
)

type Vertex struct {
    X, Y float64
}

func (v Vertex) Abs() float64 {
    return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func AbsFunc(v Vertex) float64 {
    return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func main() {
    v := Vertex{3, 4}
    fmt.Println(v.Abs())
    fmt.Println(AbsFunc(v))

    p := &Vertex{4, 3}
    fmt.Println(p.Abs())
    fmt.Println(AbsFunc(*p))
}
```

**Choosing a value or pointer receiver**
There are two reasons to use a pointer receiver.

The first is so that the method can modify the value that its receiver points to.

The second is to avoid copying the value on each method call. This can be more efficient if the receiver is a large struct, for example.

In this example, both `Scale` and `Abs` are methods with receiver type `*Vertex`, even though the `Abs` method needn't modify its receiver.

In general, all methods on a given type should have either value or pointer receivers, but not a mixture of both. (We'll see why over the next few pages.)

```go
package main

import (
    "fmt"
    "math"
)

type Vertex struct {
    X, Y float64
}

func (v *Vertex) Scale(f float64) {
    v.X = v.X * f
    v.Y = v.Y * f
}

func (v *Vertex) Abs() float64 {
    return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func main() {
    v := &Vertex{3, 4}
    fmt.Printf("Before scaling: %+v, Abs: %v\n", v, v.Abs())
    v.Scale(5)
    fmt.Printf("After scaling: %+v, Abs: %v\n", v, v.Abs())
}
```

## Interface
**Interfaces**
An _interface type_ is defined as a set of method signatures.

A value of interface type can hold any value that implements those methods.

>  一个接口类型定义为一个方法签名的集合
>  一个接口类型的值可以保存任意实现了这些方法的值，也就是实现了对应的方法的自定义类型的值

**Note:** There is an error in the example code on line 22. `Vertex` (the value type) doesn't implement `Abser` because the `Abs` method is defined only on `*Vertex` (the pointer type).

```go
package main

import (
    "fmt"
    "math"
)

type Abser interface {
    Abs() float64
}

func main() {
    var a Abser
    f := MyFloat(-math.Sqrt2)
    v := Vertex{3, 4}

    a = f  // a MyFloat implements Abser
    a = &v // a *Vertex implements Abser

    // In the following line, v is a Vertex (not *Vertex)
    // and does NOT implement Abser.
    a = v

    fmt.Println(a.Abs())
}

type MyFloat float64

func (f MyFloat) Abs() float64 {
    if f < 0 {
        return float64(-f)
    }
    return float64(f)
}

type Vertex struct {
    X, Y float64
}

func (v *Vertex) Abs() float64 {
    return math.Sqrt(v.X*v.X + v.Y*v.Y)
}
```

>  该例中，接口类型 `Abser` 中包含了方法签名 `Abs() float64` ，而自定义类型 `MyFloat` 和 `Vertex` 都实现了各自的 `Abs(float 64`) 方法，因此类型为接口类型 `Abser` `a` 可以存储类型为 `MyFloat` 的值，也可以存储类型为 `Vertex` 的值

**Interfaces are implemented implicitly**
A type implements an interface by implementing its methods. There is no explicit declaration of intent, no "implements" keyword.

Implicit interfaces decouple the definition of an interface from its implementation, which could then appear in any package without prearrangement.

>  因此，一个自定义类型通过实现其方法实现了一个接口，而不是显式声明该类型实现某某接口

```go
package main

import "fmt"

type I interface {
    M()
}

type T struct {
    S string
}

// This method means type T implements the interface I,
// but we don't need to explicitly declare that it does so.
func (t T) M() {
    fmt.Println(t.S)
}

func main() {
    var i I = T{"hello"}
    i.M()
}
```

**Interface values**
Under the hood, interface values can be thought of as a tuple of a value and a concrete type:

```
(value, type)
```

An interface value holds a value of a specific underlying concrete type.

Calling a method on an interface value executes the method of the same name on its underlying type.

>  可以认为接口值实际上是一个元组 `(value, type)`，它存储了值和具体的类型
>  调用一个接口值得方法会执行其实际类型的对应方法

```go
package main

import (
    "fmt"
    "math"
)

type I interface {
    M()
}

type T struct {
    S string
}

func (t *T) M() {
    fmt.Println(t.S)
}

type F float64

func (f F) M() {
    fmt.Println(f)
}

func main() {
    var i I

    i = &T{"Hello"}
    describe(i)
    i.M()

    i = F(math.Pi)
    describe(i)
    i.M()
}

func describe(i I) {
    fmt.Printf("(%v, %T)\n", i, i)
}
```

**Interface values with nil underlying values**
If the concrete value inside the interface itself is nil, the method will be called with a nil receiver.

In some languages this would trigger a null pointer exception, but in Go it is common to write methods that gracefully handle being called with a nil receiver (as with the method `M` in this example.)

Note that an interface value that holds a nil concrete value is itself non-nil.

>  如果接口存储的具体值是 `nil` ，仍然可以调用其方法，该方法会收到的 receiver 值就是 `nil` 
>  也就是 Go 中不存在空指针不能调用方法的情况
>  注意存储的具体值是 `nil` 的接口本身不是零值

```go
package main

import "fmt"

type I interface {
    M()
}

type T struct {
    S string
}

func (t *T) M() {
    if t == nil {
        fmt.Println("<nil>")
        return
    }
    fmt.Println(t.S)
}

func main() {
    var i I

    var t *T
    i = t
    describe(i)
    i.M()

    i = &T{"hello"}
    describe(i)
    i.M()
}

func describe(i I) {
    fmt.Printf("(%v, %T)\n", i, i)
}
```

**Nil interface values**
A nil interface value holds neither value nor concrete type.

Calling a method on a nil interface is a run-time error because there is no type inside the interface tuple to indicate which _concrete_ method to call.

>  零的接口值即没有值，也没有具体类型
>  零的接口值调用方法会出现 runtime error，因为没有具体类型，无法确定调用哪个具体方法

```go
package main

import "fmt"

type I interface {
    M()
}

func main() {
    var i I
    describe(i)
    i.M()
}

func describe(i I) {
    fmt.Printf("(%v, %T)\n", i, i)
}
```

**The empty interface**
The interface type that specifies zero methods is known as the _empty interface_:

```go
interface{}
```

An empty interface may hold values of any type. (Every type implements at least zero methods.)

Empty interfaces are used by code that handles values of unknown type. For example, `fmt.Print` takes any number of arguments of type `interface{}`.

>  不指定任何方法签名的接口成为空接口
>  空接口可以存储任意类型的值，因为所有类型都至少实现了一个 “空方法”
>  因为空接口可以存储任意类型的值，它可以用在处理未知类型的值的代码中，例如，`fmt.Print` 接受任意数量类型为 `interface{}` 的参数

```go
package main

import "fmt"

func main() {
    var i interface{}
    describe(i)

    i = 42
    describe(i)

    i = "hello"
    describe(i)
}

func describe(i interface{}) {
    fmt.Printf("(%v, %T)\n", i, i)
}
```

**Type assertions**
A _type assertion_ provides access to an interface value's underlying concrete value.

```go
t := i.(T)
```

This statement asserts that the interface value `i` holds the concrete type `T` and assigns the underlying `T` value to the variable `t`.

If `i` does not hold a `T`, the statement will trigger a panic.

>  类型断言用于断定一个接口值的具体类型
>  `t := i.(T)` 语句断言接口值 `i` 存储了具体类型 `T` ，并且将 `T` 的值赋予变量 `t`
>  如果 `i` 没有存储具体类型 `T` ，该语句会触发 panic

To _test_ whether an interface value holds a specific type, a type assertion can return two values: the underlying value and a boolean value that reports whether the assertion succeeded.

```go
t, ok := i.(T)
```

If `i` holds a `T`, then `t` will be the underlying value and `ok` will be true.

If not, `ok` will be false and `t` will be the zero value of type `T`, and no panic occurs.

>  `t, ok := i.(T)` 会在 `ok` 中存储断言的成功与否，如果失败，不会 panic，`t` 中会存储零值

Note the similarity between this syntax and that of reading from a map.

```go
package main

import "fmt"

func main() {
    var i interface{} = "hello"

    s := i.(string)
    fmt.Println(s)

    s, ok := i.(string)
    fmt.Println(s, ok)

    f, ok := i.(float64)
    fmt.Println(f, ok)

    f = i.(float64) // panic
    fmt.Println(f)
}
```

**Type switches**
A _type switch_ is a construct that permits several type assertions in series.

A type switch is like a regular switch statement, but the cases in a type switch specify types (not values), and those values are compared against the type of the value held by the given interface value.

```go
switch v := i.(type) {
case T:
    // here v has type T
case S:
    // here v has type S
default:
    // no match; here v has the same type as i
}
```

>  type switch 构造允许执行连续的类型断言
>  type switch 和常规的 switch 语句类似，但其中的 case 是类型而不是值，这些 case 会和接口值的类型进行比较

The declaration in a type switch has the same syntax as a type assertion `i.(T)`, but the specific type `T` is replaced with the keyword `type`.
>  type switch 中的声明的语法和类型断言一样，即 `i.(T)` ，但其中具体的类型 `T` 要替换为关键字 `type`

This switch statement tests whether the interface value `i` holds a value of type `T` or `S`. In each of the `T` and `S` cases, the variable `v` will be of type `T` or `S` respectively and hold the value held by `i`. In the default case (where there is no match), the variable `v` is of the same interface type and value as `i`.
>  该 type switch 中，如果接口值的类型是 `T` 或 `S` ，则变量 `v` 的类型就会是 `T` 或 `S` ，默认情况下 (没有匹配)，`v` 的类型就会是接口类型

```go
package main

import "fmt"

func do(i interface{}) {
    switch v := i.(type) {
    case int:
        fmt.Printf("Twice %v is %v\n", v, v*2)
    case string:
        fmt.Printf("%q is %v bytes long\n", v, len(v))
    default:
        fmt.Printf("I don't know about type %T!\n", v)
    }
}

func main() {
    do(21)
    do("hello")
    do(true)
}
```

**Stringers**
One of the most ubiquitous interfaces is [`Stringer`](https://go.dev/pkg/fmt/#Stringer) defined by the [`fmt`](https://go.dev/pkg/fmt/) package.

```go
type Stringer interface {
    String() string
}
```

A `Stringer` is a type that can describe itself as a string. The `fmt` package (and many others) look for this interface to print values.

>  最常见的一个接口类型是 `Stringer` ，在 `fmt` 包中定义
>  `Stringer` 类型即可以将自己描述为字符串的类型，`fmt` 包通过该类型打印值

```go
package main

import "fmt"

type Person struct {
    Name string
    Age  int
}

func (p Person) String() string {
    return fmt.Sprintf("%v (%v years)", p.Name, p.Age)
}

func main() {
    a := Person{"Arthur Dent", 42}
    z := Person{"Zaphod Beeblebrox", 9001}
    fmt.Println(a, z)
}
```

### Errors
**Errors**
Go programs express error state with `error` values.

The `error` type is a built-in interface similar to `fmt.Stringer`:

```go
type error interface {
    Error() string
}
```

(As with `fmt.Stringer`, the `fmt` package looks for the `error` interface when printing values.)

>  Go 程序使用 `error` 值表达错误状态
>  `error` 类型是一个内建的接口类型，它和 `fmt.Stringer` 类似，当 `fmt` 打印值时，如果遇到 `error` 接口类型的值，它会调用其 `Error` 方法获取字符串

Functions often return an `error` value, and calling code should handle errors by testing whether the error equals `nil`.

```go
i, err := strconv.Atoi("42")
if err != nil {
    fmt.Printf("couldn't convert number: %v\n", err)
    return
}
fmt.Println("Converted integer:", i)
```

A nil `error` denotes success; a non-nil `error` denotes failure.

>  函数经常会返回一个 `error` 类型的值，我们可以测试该值是否等于 `nil` 以判断是否出现了错误
>  非零的 `error` 值标识出现错误

```go
package main

import (
    "fmt"
    "time"
)

type MyError struct {
    When time.Time
    What string
}

func (e *MyError) Error() string {
    return fmt.Sprintf("at %v, %s",
        e.When, e.What)
}

func run() error {
    return &MyError{
        time.Now(),
        "it didn't work",
    }
}

func main() {
    if err := run(); err != nil {
        fmt.Println(err)
    }
}
```

### Readers
**Readers**
The `io` package specifies the `io.Reader` interface, which represents the read end of a stream of data.

The Go standard library contains [many implementations](https://cs.opensource.google/search?q=Read%5C\(%5Cw%2B%5Cs%5C%5B%5C%5Dbyte%5C\)&ss=go%2Fgo) of this interface, including files, network connections, compressors, ciphers, and others.

>  `io` 包定义了 `io.Reader` 接口，该接口表示了对数据流的读取端
>  Go 的标准库包含了许多对该接口的实现，包括了文件、网络连接、压缩器、加密器等

The `io.Reader` interface has a `Read` method:

```go
func (T) Read(b []byte) (n int, err error)
```

`Read` populates the given byte slice with data and returns the number of bytes populated and an error value. It returns an `io.EOF` error when the stream ends.

>  `io.Reader` 接口有一个 `Read` 方法，该方法用数据填充给定的字节切片，然后返回填充的字节数和一个错误值
>  数据流结束时，它返回一个 `io.EOF` 错误

The example code creates a [`strings.Reader`](https://go.dev/pkg/strings/#Reader) and consumes its output 8 bytes at a time.

```go
package main

import (
    "fmt"
    "io"
    "strings"
)

func main() {
    r := strings.NewReader("Hello, Reader!")

    b := make([]byte, 8)
    for {
        n, err := r.Read(b)
        fmt.Printf("n = %v err = %v b = %v\n", n, err, b)
        fmt.Printf("b[:n] = %q\n", b[:n])
        if err == io.EOF {
            break
        }
    }
}
```

### Images
**Images**
[Package image](https://go.dev/pkg/image/#Image) defines the `Image` interface:

```go
package image

type Image interface {
    ColorModel() color.Model
    Bounds() Rectangle
    At(x, y int) color.Color
}
```

**Note**: the `Rectangle` return value of the `Bounds` method is actually an [`image.Rectangle`](https://go.dev/pkg/image/#Rectangle), as the declaration is inside package `image`.

>  `image` 包定义了 `Image` 接口

(See [the documentation](https://go.dev/pkg/image/#Image) for all the details.)

The `color.Color` and `color.Model` types are also interfaces, but we'll ignore that by using the predefined implementations `color.RGBA` and `color.RGBAModel`. These interfaces and types are specified by the [image/color package](https://go.dev/pkg/image/color/)

```go
package main

import (
    "fmt"
    "image"
)

func main() {
    m := image.NewRGBA(image.Rect(0, 0, 100, 100))
    fmt.Println(m.Bounds())
    fmt.Println(m.At(0, 0).RGBA())
}
```

# Generics
**Type parameters**
Go functions can be written to work on multiple types using type parameters. The type parameters of a function appear between brackets, before the function's arguments.
>  可以通过类型参数定义泛型函数
>  函数的类型参数在函数名之后，在函数参数之前

```go
func Index[T comparable](s []T, x T) int
```

This declaration means that `s` is a slice of any type `T` that fulfills the built-in constraint `comparable`. `x` is also a value of the same type.
>  以上的声明指参数 `s` 是类型 `T` 的切片，`x` 是类型 `T` 的值，而类型 `T` 是任意满足内建约束 `comparable` 的类型
>  满足 `comparable` 的类型的值可以用 `==` 和 `!=` 互相比较

`comparable` is a useful constraint that makes it possible to use the `==` and `!=` operators on values of the type. In this example, we use it to compare a value to all slice elements until a match is found. This `Index` function works for any type that supports comparison.

```go
package main

import "fmt"

// Index returns the index of x in s, or -1 if not found.
func Index[T comparable](s []T, x T) int {
    for i, v := range s {
        // v and x are type T, which has the comparable
        // constraint, so we can use == here.
        if v == x {
            return i
        }
    }
    return -1
}

func main() {
    // Index works on a slice of ints
    si := []int{10, 20, 15, -10}
    fmt.Println(Index(si, 15))

    // Index also works on a slice of strings
    ss := []string{"foo", "bar", "baz"}
    fmt.Println(Index(ss, "hello"))
}
```

**Generic types**
In addition to generic functions, Go also supports generic types. A type can be parameterized with a type parameter, which could be useful for implementing generic data structures.
>  Go 也支持泛型类型，用于实现泛型数据结构，定义类型时，提供类型参数即可定义泛型类型

This example demonstrates a simple type declaration for a singly-linked list holding any type of value.

As an exercise, add some functionality to this list implementation.

```go
package main

// List represents a singly-linked list that holds
// values of any type.
type List[T any] struct {
    next *List[T]
    val  T
}

func main() {
}
```

# Concurrency
**Goroutines**
A _goroutine_ is a lightweight thread managed by the Go runtime.

```go
go f(x, y, z)
```

starts a new goroutine running

```go
f(x, y, z)
```

The evaluation of `f`, `x`, `y`, and `z` happens in the current goroutine and the execution of `f` happens in the new goroutine.

>  go 例程指有 Go 运行时管理的一个轻量线程
>  `go f(x,y,z)` 会启动一个新的 go 例程，该例程运行 `f(x, y, z)` ，对 `f, x, y, z` 的评估在当前 go 例程中发生，而对 `f` 的执行则在新的 go 例程中发生

Goroutines run in the same address space, so access to shared memory must be synchronized. The [`sync`](https://go.dev/pkg/sync/) package provides useful primitives, although you won't need them much in Go as there are other primitives. (See the next slide.)
>  go 例程都在相同的地址空间中运行，因此对共享内存的访问需要进行同步
>  `sync` 包提供了相关的原语


**Channels**
Channels are a typed conduit through which you can send and receive values with the channel operator, `<-`.

```go
ch <- v    // Send v to channel ch.
v := <-ch  // Receive from ch, and
           // assign value to v.
```

(The data flows in the direction of the arrow.)

>  channels 是有类型的通道，我们可以通过 channels 发送和接受值

Like maps and slices, channels must be created before use:

```go
ch := make(chan int)
```

By default, sends and receives block until the other side is ready. This allows goroutines to synchronize without explicit locks or condition variables.

>  和 map 和 slices 一样，channels 在使用前也必须要被创建
>  发送和接受在默认情况下是阻塞式的，直到另一端准备好，这使得 go 例程可以在没有显式锁和条件变量的条件下同步

The example code sums the numbers in a slice, distributing the work between two goroutines. Once both goroutines have completed their computation, it calculates the final result.

```go
package main

import "fmt"

func sum(s []int, c chan int) {
    sum := 0
    for _, v := range s {
        sum += v
    }
    c <- sum // send sum to c
}

func main() {
    s := []int{7, 2, 8, -9, 4, 0}

    c := make(chan int)
    go sum(s[:len(s)/2], c)
    go sum(s[len(s)/2:], c)
    x, y := <-c, <-c // receive from c

    fmt.Println(x, y, x+y)
}
```

**Buffered Channels**
Channels can be _buffered_. Provide the buffer length as the second argument to `make` to initialize a buffered channel:

```go
ch := make(chan int, 100)
```

Sends to a buffered channel block only when the buffer is full. Receives block when the buffer is empty.

>  channels 可以被缓存
>  向缓存的 channel 发送消息只会在缓存已经满时才阻塞，接受则在缓存已经空时才阻塞

Modify the example to overfill the buffer and see what happens.

```go
package main

import "fmt"

func main() {
    ch := make(chan int, 2)
    ch <- 1
    ch <- 2
    fmt.Println(<-ch)
    fmt.Println(<-ch)
}
```

**Range and Close**
A sender can `close` a channel to indicate that no more values will be sent. Receivers can test whether a channel has been closed by assigning a second parameter to the receive expression: after

```go
v, ok := <-ch
```

`ok` is `false` if there are no more values to receive and the channel is closed.

>  发送端可以 `close` 一个 channel，表示没有更多的值需要发送
>  接收端可以为接受表达式添加第二个接受参数，以测试 channel 是否被关闭

The loop `for i := range c` receives values from the channel repeatedly until it is closed.
>  循环 `for i:= range c` 会从 channel `c` 反复接受值，直到该 channel 被关闭

**Note:** Only the sender should close a channel, never the receiver. Sending on a closed channel will cause a panic.
>  只有发送端可以关闭 channel，向关闭的 channel 发送值会 panic

**Another note:** Channels aren't like files; you don't usually need to close them. Closing is only necessary when the receiver must be told there are no more values coming, such as to terminate a `range` loop.
>  channels 和文件不一样，我们不需要经常关闭它们，关闭 channels 的操作仅在接收端必须直到没有更多值的时候才是必要的，例如为了终止一个 `range` 循环

```go
package main

import (
    "fmt"
)

func fibonacci(n int, c chan int) {
    x, y := 0, 1
    for i := 0; i < n; i++ {
        c <- x
        x, y = y, x+y
    }
    close(c)
}

func main() {
    c := make(chan int, 10)
    go fibonacci(cap(c), c)
    for i := range c {
        fmt.Println(i)
    }
}
```

**Select**
The `select` statement lets a goroutine wait on multiple communication operations.

A `select` blocks until one of its cases can run, then it executes that case. It chooses one at random if multiple are ready.

>  `select` 语句让一个 go 例程等待多个通信操作 (例如通道操作)
>  `select` 会阻塞，直到其中某个 case 可以执行，然后就执行该 case，如果多个 case 都准备好了，则随机执行一个

```go
package main

import "fmt"

func fibonacci(c, quit chan int) {
    x, y := 0, 1
    for {
        select {
        case c <- x:
            x, y = y, x+y
        case <-quit:
            fmt.Println("quit")
            return
        }
    }
}

func main() {
    c := make(chan int)
    quit := make(chan int)
    go func() {
        for i := 0; i < 10; i++ {
            fmt.Println(<-c)
        }
        quit <- 0
    }()
    fibonacci(c, quit)
}
```

**Default Selection**
The `default` case in a `select` is run if no other case is ready.
>  `select` 中，如果没有其他的 case 是准备好的，就运行 `default` case

Use a `default` case to try a send or receive without blocking:
>  `default` case 可以用于实现非阻塞的发送或接受操作，即在等待时执行 `default`

```go
select {
case i := <-c:
    // use i
default:
    // receiving from c would block
}
```

**sync.Mutex**
We've seen how channels are great for communication among goroutines.
>  根据之前的例子，可以看到 channels 可以很方便地用于 go 例程之间的通信

But what if we don't need communication? What if we just want to make sure only one goroutine can access a variable at a time to avoid conflicts?

This concept is called _mutual exclusion_, and the conventional name for the data structure that provides it is _mutex_.

>  如果我们不需要通信，只需要确定每次只有一个 go 例程允许访问一个变量，以避免冲突
>  也就是说，对变量的访问是互斥的，Go 为此提供了互斥锁

Go's standard library provides mutual exclusion with [`sync.Mutex`](https://go.dev/pkg/sync/#Mutex) and its two methods:

- `Lock`
- `Unlock`

We can define a block of code to be executed in mutual exclusion by surrounding it with a call to `Lock` and `Unlock` as shown on the `Inc` method.

We can also use `defer` to ensure the mutex will be unlocked as in the `Value` method.

>  Go 的标准库提供了 `sync.Mutex` 类型，它带有两个方法 `Lock, Unlock`
>  要将某个代码块在互斥模式下执行，则使用对 `Lock` 和 `Unlock` 的调用将其围绕
>  还可以使用 `defer` 来确保互斥锁会被解锁

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// SafeCounter is safe to use concurrently.
type SafeCounter struct {
    mu sync.Mutex
    v  map[string]int
}

// Inc increments the counter for the given key.
func (c *SafeCounter) Inc(key string) {
    c.mu.Lock()
    // Lock so only one goroutine at a time can access the map c.v.
    c.v[key]++
    c.mu.Unlock()
}

// Value returns the current value of the counter for the given key.
func (c *SafeCounter) Value(key string) int {
    c.mu.Lock()
    // Lock so only one goroutine at a time can access the map c.v.
    defer c.mu.Unlock()
    return c.v[key]
}

func main() {
    c := SafeCounter{v: make(map[string]int)}
    for i := 0; i < 1000; i++ {
        go c.Inc("somekey")
    }

    time.Sleep(time.Second)
    fmt.Println(c.Value("somekey"))
}
```

# Where to Go from here...
You can get started by [installing Go](https://go.dev/doc/install/).

Once you have Go installed, the [Go Documentation](https://go.dev/doc/) is a great place to continue. It contains references, tutorials, videos, and more.

To learn how to organize and work with Go code, read [How to Write Go Code](https://go.dev/doc/code).

If you need help with the standard library, see the [package reference](https://go.dev/pkg/). For help with the language itself, you might be surprised to find the [Language Spec](https://go.dev/ref/spec) is quite readable.

To further explore Go's concurrency model, watch [Go Concurrency Patterns](https://www.youtube.com/watch?v=f6kdp27TYZs) ([slides](https://go.dev/talks/2012/concurrency.slide)) and [Advanced Go Concurrency Patterns](https://www.youtube.com/watch?v=QDDwwePbDtw) ([slides](https://go.dev/talks/2013/advconc.slide)) and read the [Share Memory by Communicating](https://go.dev/doc/codewalk/sharemem/) codewalk.

To get started writing web applications, watch [A simple programming environment](https://vimeo.com/53221558) ([slides](https://go.dev/talks/2012/simple.slide)) and read the [Writing Web Applications](https://go.dev/doc/articles/wiki/) tutorial.

The [First Class Functions in Go](https://go.dev/doc/codewalk/functions/) codewalk gives an interesting perspective on Go's function types.

The [Go Blog](https://go.dev/blog/) has a large archive of informative Go articles.

Visit [the Go home page](https://go.dev/) for more.