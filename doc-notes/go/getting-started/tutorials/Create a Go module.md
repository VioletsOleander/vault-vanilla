---
completed: true
---
# Create a module
This is the first part of a tutorial that introduces a few fundamental features of the Go language. If you're just getting started with Go, be sure to take a look at [Tutorial: Get started with Go](https://go.dev/doc/tutorial/getting-started.html), which introduces the `go` command, Go modules, and very simple Go code.

In this tutorial you'll create two modules. The first is a library which is intended to be imported by other libraries or applications. The second is a caller application which will use the first.

This tutorial's sequence includes seven brief topics that each illustrate a different part of the language.

1. Create a module -- Write a small module with functions you can call from another module.
2. [Call your code from another module](https://go.dev/doc/tutorial/call-module-code.html) -- Import and use your new module.
3. [Return and handle an error](https://go.dev/doc/tutorial/handle-errors.html) -- Add simple error handling.
4. [Return a random greeting](https://go.dev/doc/tutorial/random-greeting.html) -- Handle data in slices (Go's dynamically-sized arrays).
5. [Return greetings for multiple people](https://go.dev/doc/tutorial/greetings-multiple-people.html) -- Store key/value pairs in a map.
6. [Add a test](https://go.dev/doc/tutorial/add-a-test.html) -- Use Go's built-in unit testing features to test your code.
7. [Compile and install the application](https://go.dev/doc/tutorial/compile-install.html) -- Compile and install your code locally.

**Note:** For other tutorials, see [Tutorials](https://go.dev/doc/tutorial/index.html).

## Prerequisites
- **Some programming experience.** The code here is pretty simple, but it helps to know something about functions, loops, and arrays.
- **A tool to edit your code.** Any text editor you have will work fine. Most text editors have good support for Go. The most popular are VSCode (free), GoLand (paid), and Vim (free).
- **A command terminal.** Go works well using any terminal on Linux and Mac, and on PowerShell or cmd in Windows.

## Start a module that others can use
Start by creating a Go module. In a module, you collect one or more related packages for a discrete and useful set of functions. For example, you might create a module with packages that have functions for doing financial analysis so that others writing financial applications can use your work. For more about developing modules, see [Developing and publishing modules](https://go.dev/doc/modules/developing).
>  一个模块可以包含一个或者多个互相关联的包

Go code is grouped into packages, and packages are grouped into modules. Your module specifies dependencies needed to run your code, including the Go version and the set of other modules it requires.
>  Go 代码以包进行划分，多个包可以组成一个模块
>  模块需要指定用于运行代码所需要的依赖，包括了 Go 版本和它所需要的其他模块

As you add or improve functionality in your module, you publish new versions of the module. Developers writing code that calls functions in your module can import the module's updated packages and test with the new version before putting it into production use.
>  随着我们向模块中添加和改进功能时，我们可以发布该模块的新版本

(1) Open a command prompt and `cd` to your home directory.

(2) Create a `greetings` directory for your Go module source code.

(3) Start your module using the [`go mod init` command](https://go.dev/ref/mod#go-mod-init).

Run the `go mod init` command, giving it your module path -- here, use `example.com/greetings`. If you publish a module, this _must_ be a path from which your module can be downloaded by Go tools. That would be your code's repository.

For more on naming your module with a module path, see [Managing dependencies](https://go.dev/doc/modules/managing-dependencies#naming_module).

```
$ go mod init example.com/greetings
go: creating new go.mod: module example.com/greetings
```

The `go mod init` command creates a go.mod file to track your code's dependencies. So far, the file includes only the name of your module and the Go version your code supports. But as you add dependencies, the go.mod file will list the versions your code depends on. This keeps builds reproducible and gives you direct control over which module versions to use.
>  `go mod init` 命令创建 `go.mod` 文件以追踪依赖，起始时，该文件仅包含我们的模块名和 Go 版本，随着我们添加依赖，`go.mod` 文件中将列出我们的代码所依赖的模块版本，以保证可复现性

(4) In your text editor, create a file in which to write your code and call it greetings.go.

(5) Paste the following code into your greetings.go file and save the file.

```go
package greetings

import "fmt"

// Hello returns a greeting for the named person.
func Hello(name string) string {
    // Return a greeting that embeds the name in a message.
    message := fmt.Sprintf("Hi, %v. Welcome!", name)
    return message
}
```

This is the first code for your module. It returns a greeting to any caller that asks for one. You'll write code that calls this function in the next step.

In this code, you:

- Declare a `greetings` package to collect related functions.
- Implement a `Hello` function to return the greeting.
    
    This function takes a `name` parameter whose type is `string`. The function also returns a `string`. In Go, a function whose name starts with a capital letter can be called by a function not in the same package. This is known in Go as an exported name. For more about exported names, see [Exported names](https://go.dev/tour/basics/3) in the Go tour.
    
    ![](https://go.dev/doc/tutorial/images/function-syntax.png)
- Declare a `message` variable to hold your greeting.
    
    In Go, the `:=` operator is a shortcut for declaring and initializing a variable in one line (Go uses the value on the right to determine the variable's type). Taking the long way, you might have written this as:
        
        var message string
        message = fmt.Sprintf("Hi, %v. Welcome!", name)
        
    - Use the `fmt` package's [`Sprintf` function](https://pkg.go.dev/fmt/#Sprintf) to create a greeting message. The first argument is a format string, and `Sprintf` substitutes the `name` parameter's value for the `%v` format verb. Inserting the value of the `name` parameter completes the greeting text.
    - Return the formatted greeting text to the caller.

In the next step, you'll call this function from another module.

>  在这段代码中，我们声明了 `greetings` 包，在包中实现了 `Hello` 函数
>  在 Go 中，函数名以大写字母开头的函数可以被不在同一包内的其他函数调用，这在 Go 中称为导出名字
>  在 Go 中，`:=` 运算符是声明并且初始化变量的简化写法，它等价于 `var message string` 和 `message = fmt.Springf("...")` 
>  `fmt` 包的 `Sprintf` 函数的第一个参数为格式字符串，其中的 `%v` 会被参数 `name` 替换

# Add a test
Now that you've gotten your code to a stable place (nicely done, by the way), add a test. Testing your code during development can expose bugs that find their way in as you make changes. In this topic, you add a test for the `Hello` function.

**Note:** This topic is part of a multi-part tutorial that begins with [Create a Go module](https://go.dev/doc/tutorial/create-module.html).

Go's built-in support for unit testing makes it easier to test as you go. Specifically, using naming conventions, Go's `testing` package, and the `go test` command, you can quickly write and execute tests.
>  Go 内建了对单元测试的支持，我们只需要遵循命名规范，Go 的 `testing` 包和 `go test` 命令就可以帮助我们编写和执行测试

(1) In the greetings directory, create a file called `greetings_test.go`.

Ending a file's name with `_test.go` tells the ` go test ` command that this file contains test functions.

>  包含了测试函数的文件应该以 `_test.go` 为后缀，`go test` 将自动搜索这些文件

(2) In `greetings_test.go`, paste the following code and save the file.

```go
package greetings

import (
    "testing"
    "regexp"
)

// TestHelloName calls greetings.Hello with a name, checking
// for a valid return value.
func TestHelloName(t *testing.T) {
    name := "Gladys"
    want := regexp.MustCompile(`\b`+name+`\b`)
    msg, err := Hello("Gladys")
    if !want.MatchString(msg) || err != nil {
        t.Errorf(`Hello("Gladys") = %q, %v, want match for %#q, nil`, msg, err, want)
    }
}

// TestHelloEmpty calls greetings.Hello with an empty string,
// checking for an error.
func TestHelloEmpty(t *testing.T) {
    msg, err := Hello("")
    if msg != "" || err == nil {
        t.Errorf(`Hello("") = %q, %v, want "", error`, msg, err)
    }
}
```

In this code, you:

- Implement test functions in the same package as the code you're testing.
- Create two test functions to test the `greetings.Hello` function. Test function names have the form `TestName`, where `Name` says something about the specific test. Also, test functions take a pointer to the ` testing ` package's [` testing.T ` type](https://go.dev/pkg/testing/#T) as a parameter. You use this parameter's methods for reporting and logging from your test.
- Implement two tests:
    - `TestHelloName` calls the `Hello` function, passing a `name` value with which the function should be able to return a valid response message. If the call returns an error or an unexpected response message (one that doesn't include the name you passed in), you use the `t` parameter's [`Errorf` method](https://go.dev/pkg/testing/#T.Errorf) to print a message to the console.
    - `TestHelloEmpty` calls the `Hello` function with an empty string. This test is designed to confirm that your error handling works. If the call returns a non-empty string or no error, you use the `t` parameter's [`Errorf` method](https://go.dev/pkg/testing/#T.Errorf) to print a message to the console.

>  在 `greetings_test.go` 中
>  - 包仍然是 `greetings` ，即测试函数和被测试的代码在同一个包
>  - 有两个函数 `TestHelloEmpty, TestHelloName` ，用来测试 `greetings.Hello` 函数。测试函数的命名形式应该是 `Test<Name>` ，其中 `Name` 用于表示测试的具体信息。另外，测试函数应该接收一个指向 `testing.T` 类型的指针作为参数。我们可以使用该参数的方法进行报告和日志记录
>  - `TestHelloName` 调用 `Hello` 函数，传入 `name` ，对 `Hello` 的返回值进行判断，如果返回了错误或不是期望的回应消息，就使用 `t` 参数的 `Errorf` 方法，向控制台打印错误信息
>  - `TestHelloEmpty` 调用 `Hello` 函数，传入空字符串，以测试 `Hello` 函数的错误处理能力，如果返回了非空的字符串而没有报错，就使用 `t` 参数的 `Errorf` 方法向控制台打印错误信息

(3) At the command line in the greetings directory, run the [`go test` command](https://go.dev/cmd/go/#hdr-Test_packages) to execute the test.

The `go test` command executes test functions (whose names begin with `Test`) in test files (whose names end with `_test.go`). You can add the ` -v ` flag to get verbose output that lists all of the tests and their results.

>  在目录中，运行 `go test` 即可运行测试
>  `go test` 会执行 `<filename>_test.go` 文件中的测试函数 (以 `Test` 开头的函数)，`go test -v` 可以获取详细输出，包括了各个测试及其结果

The tests should pass.

```
$ go test
PASS
ok      example.com/greetings   0.364s

$ go test -v
=== RUN   TestHelloName
--- PASS: TestHelloName (0.00s)
=== RUN   TestHelloEmpty
--- PASS: TestHelloEmpty (0.00s)
PASS
ok      example.com/greetings   0.372s
```

(4) Break the `greetings.Hello` function to view a failing test.

The `TestHelloName` test function checks the return value for the name you specified as a `Hello` function parameter. To view a failing test result, change the `greetings.Hello` function so that it no longer includes the name.

In `greetings/greetings.go`, paste the following code in place of the ` Hello ` function. Note that the highlighted lines change the value that the function returns, as if the ` name ` argument had been accidentally removed.

```go
// Hello returns a greeting for the named person.
func Hello(name string) (string, error) {
    // If no name was given, return an error with a message.
    if name == "" {
        return name, errors.New("empty name")
    }
    // Create a message using a random format.
    // message := fmt.Sprintf(randomFormat(), name) // < highlighted
    message := fmt.Sprint(randomFormat()) // < highlighted
    return message, nil
}
```

(5) At the command line in the greetings directory, run `go test` to execute the test.

This time, run `go test` without the `-v` flag. The output will include results for only the tests that failed, which can be useful when you have a lot of tests. The `TestHelloName` test should fail -- `TestHelloEmpty` still passes.

>  没有添加 `-v` ，`go test` 只会显示没有通过的测试信息

```
$ go test
--- FAIL: TestHelloName (0.00s)
    greetings_test.go:15: Hello("Gladys") = "Hail, %v! Well met!", <nil>, want match for `\bGladys\b`, nil
FAIL
exit status 1
FAIL    example.com/greetings   0.182s
```

In the next (and last) topic, you'll see how to compile and install your code to run it locally.