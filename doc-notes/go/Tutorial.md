# Get started with Go
In this tutorial, you'll get a brief introduction to Go programming. Along the way, you will:

- Install Go (if you haven't already).
- Write some simple "Hello, world" code.
- Use the `go` command to run your code.
- Use the Go package discovery tool to find packages you can use in your own code.
- Call functions of an external module.

**Note:** For other tutorials, see [Tutorials](https://go.dev/doc/tutorial/index.html).

## Prerequisites
- **Some programming experience.** The code here is pretty simple, but it helps to know something about functions.
- **A tool to edit your code.** Any text editor you have will work fine. Most text editors have good support for Go. The most popular are VSCode (free), GoLand (paid), and Vim (free).
- **A command terminal.** Go works well using any terminal on Linux and Mac, and on PowerShell or cmd in Windows.

## Install Go
Just use the [Download and install](https://go.dev/doc/install) steps.

## Write some code
Get started with Hello, World.

(1) Open a command prompt and cd to your home directory.

(2) Create a hello directory for your first Go source code.

(3) Enable dependency tracking for your code.

When your code imports packages contained in other modules, you manage those dependencies through your code's own module. That module is defined by a go.mod file that tracks the modules that provide those packages. That go.mod file stays with your code, including in your source code repository.
>  需要导入其他模块包含的包时，我们需要定义一个 `go.mod` 模块管理它们，该模块追踪提供包的模块

To enable dependency tracking for your code by creating a go.mod file, run the [`go mod init` command](https://go.dev/ref/mod#go-mod-init), giving it the name of the module your code will be in. The name is the module's module path.
>  `go mod init` 命令会帮助创建 `go.mod` 文件，我们为该命令提供我们代码所在的模块名称/路径

In actual development, the module path will typically be the repository location where your source code will be kept. For example, the module path might be `github.com/mymodule`. If you plan to publish your module for others to use, the module path _must_ be a location from which Go tools can download your module. For more about naming a module with a module path, see [Managing dependencies](https://go.dev/doc/modules/managing-dependencies#naming_module).
>  实际开发中，模块路径一般是保存源代码的仓库位置

For the purposes of this tutorial, just use `example/hello`.

```
$ go mod init example/hello
go: creating new go.mod: module example/hello
```

(4)In your text editor, create a file hello.go in which to write your code.

(5) Paste the following code into your hello.go file and save the file.

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
```

This is your Go code. In this code, you:

- Declare a `main` package (a package is a way to group functions, and it's made up of all the files in the same directory).
- Import the popular [`fmt` package](https://pkg.go.dev/fmt/), which contains functions for formatting text, including printing to the console. This package is one of the [standard library](https://pkg.go.dev/std) packages you got when you installed Go.
- Implement a `main` function to print a message to the console. A `main` function executes by default when you run the `main` package.

>  上述代码中，我们：
>  1. 声明了 `main` 包，包用于组织函数，一个包由相同目录中的所有文件组成
>  2. 导入 `fmt` 包，它包含了用于格式化文本的函数，其中就包括了将文本打印到控制台的函数。`fmt` 包是标准库的一部分
>  3. 实现了 `main` 函数，该函数将消息打印到控制台，运行 `main` 包时，`main` 函数是被默认执行的函数

(6) Run your code to see the greeting.

```
$ go run .
Hello, World!
```

The [`go run` command](https://go.dev/cmd/go/#hdr-Compile_and_run_Go_program) is one of many `go` commands you'll use to get things done with Go. Use the following command to get a list of the others:

```
$ go help
```

## Call code in an external package
When you need your code to do something that might have been implemented by someone else, you can look for a package that has functions you can use in your code.

(1) Make your printed message a little more interesting with a function from an external module.

1. Visit pkg.go.dev and [search for a "quote" package](https://pkg.go.dev/search?q=quote).
2. Locate and click the `rsc.io/quote` package in search results (if you see `rsc.io/quote/v3`, ignore it for now).
3. In the **Documentation** section, under **Index**, note the list of functions you can call from your code. You'll use the `Go` function.
4. At the top of this page, note that package `quote` is included in the `rsc.io/quote` module.

You can use the pkg.go.dev site to find published modules whose packages have functions you can use in your own code. Packages are published in modules -- like `rsc.io/quote` -- where others can use them. Modules are improved with new versions over time, and you can upgrade your code to use the improved versions.

>  `pkg.go.dev` 中包含了许多模块，包发布在模块中，例如 `rsc.io/quote`，包中有可以直接由我们使用的函数，我们可以导入包然后使用它们

(2) In your Go code, import the `rsc.io/quote` package and add a call to its `Go` function.

After adding the highlighted lines, your code should include the following:

```go
package main

import "fmt"

import "rsc.io/quote"

func main() {
    fmt.Println(quote.Go())
}
```

(3) Add new module requirements and sums.

Go will add the `quote` module as a requirement, as well as a go.sum file for use in authenticating the module. For more, see [Authenticating modules](https://go.dev/ref/mod#authenticating) in the Go Modules Reference.

```
$ go mod tidy
go: finding module for package rsc.io/quote
go: found rsc.io/quote in rsc.io/quote v1.5.2
```

>  `go mod tidy` 会帮助我们安装所需模块，同时会添加一个 `go.sum` 文件用于验证模块

(4) Run your code to see the message generated by the function you're calling.

```
$ go run .
Don't communicate by sharing memory, share memory by communicating.
```

Notice that your code calls the `Go` function, printing a clever message about communication.

When you ran `go mod tidy`, it located and downloaded the `rsc.io/quote` module that contains the package you imported. By default, it downloaded the latest version -- v1.5.2.

## Write more code
With this quick introduction, you got Go installed and learned some of the basics. To write some more code with another tutorial, take a look at [Create a Go module](https://go.dev/doc/tutorial/create-module.html).

# Create a Go module
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

# Getting started with multi-module workspaces
This tutorial introduces the basics of multi-module workspaces in Go. With multi-module workspaces, you can tell the Go command that you’re writing code in multiple modules at the same time and easily build and run code in those modules.

In this tutorial, you’ll create two modules in a shared multi-module workspace, make changes across those modules, and see the results of those changes in a build.

**Note:** For other tutorials, see [Tutorials](https://go.dev/doc/tutorial/index.html).

## Prerequisites

- **An installation of Go 1.18 or later.**
- **A tool to edit your code.** Any text editor you have will work fine.
- **A command terminal.** Go works well using any terminal on Linux and Mac, and on PowerShell or cmd in Windows.

This tutorial requires go1.18 or later. Make sure you’ve installed Go at Go 1.18 or later using the links at [go.dev/dl](https://go.dev/dl).

## Create a module for your code
To begin, create a module for the code you’ll write.

(1) Open a command prompt and change to your home directory.

The rest of the tutorial will show a $ as the prompt. The commands you use will work on Windows too.

(2) From the command prompt, create a directory for your code called workspace.

(3) Initialize the module

Our example will create a new module `hello` that will depend on the golang.org/x/example module.
   
Create the hello module:
   
```
$ mkdir hello
$ cd hello
$ go mod init example.com/hello
go: creating new go.mod: module example.com/hello
```

Add a dependency on the golang.org/x/example/hello/reverse package by using `go get`.
   
```
$ go get golang.org/x/example/hello/reverse
```

Create hello.go in the hello directory with the following contents:

```go
package main

import (
    "fmt"

    "golang.org/x/example/hello/reverse"
)

func main() {
    fmt.Println(reverse.String("Hello"))
}
```

Now, run the hello program:

```
$ go run .
olleH
```

## Create the workspace
In this step, we’ll create a `go.work` file to specify a workspace with the module.

#### Initialize the workspace
In the `workspace` directory, run:

```
$ go work init ./hello
```

The `go work init` command tells `go` to create a `go.work` file for a workspace containing the modules in the `./hello` directory.

>  `go work init` 命令用于让 Go 为包含了 `./hello` 目录中的模块的工作空间创建一个 `go.work` 文件

The `go` command produces a `go.work` file that looks like this:

```go
go 1.18

use ./hello
```

The `go.work` file has similar syntax to `go.mod`.

The `go` directive tells Go which version of Go the file should be interpreted with. It’s similar to the `go` directive in the `go.mod` file.

The `use` directive tells Go that the module in the `hello` directory should be main modules when doing a build.

So in any subdirectory of `workspace` the module will be active.

>  `go.work` 文件的语法和 `go.mod` 类似，其中 `go` 指令告诉 Go 该文件应该使用哪个版本的 Go 来解释，这和 `go.mod` 中的 `go` 指令一致
>  其中 `use` 指令告诉 Go `hello` 目录中的模块是在构建时的主模块，因此在 `workspace` 的任何子目录中，`hello` 目录中的模块都将是活跃状态

#### Run the program in the workspace directory
In the `workspace` directory, run:

```
$ go run ./hello
olleH
```

The Go command includes all the modules in the workspace as main modules. This allows us to refer to a package in the module, even outside the module. Running the `go run` command outside the module or the workspace would result in an error because the `go` command wouldn’t know which modules to use.
>  Go 命令会将同一个工作区内的所有模块都包含为主模块，也就是说，在工作区内的模块可以引用其他模块的包

Next, we’ll add a local copy of the `golang.org/x/example/hello` module to the workspace. That module is stored in a subdirectory of the `go.googlesource.com/example` Git repository. We’ll then add a new function to the `reverse` package that we can use instead of `String`.

## Download and modify the `golang.org/x/example/hello` module
In this step, we’ll download a copy of the Git repo containing the `golang.org/x/example/hello` module, add it to the workspace, and then add a new function to it that we will use from the hello program.

(1) Clone the repository

From the workspace directory, run the `git` command to clone the repository:

```
$ git clone https://go.googlesource.com/example
Cloning into 'example'...
remote: Total 165 (delta 27), reused 165 (delta 27)
Receiving objects: 100% (165/165), 434.18 KiB | 1022.00 KiB/s, done.
Resolving deltas: 100% (27/27), done.
```

(2) Add the module to the workspace

The Git repo was just checked out into `./example`. The source code for the `golang.org/x/example/hello` module is in `./example/hello`. Add it to the workspace:

```
$ go work use ./example/hello
```

The `go work use` command adds a new module to the go.work file. It will now look like this:

```go
go 1.18

use (
    ./hello
    ./example/hello
)
```

>  `go work use` 命令用于将新的模块加入到 `go.work` 文件中

The workspace now includes both the `example.com/hello` module and the `golang.org/x/example/hello` module, which provides the `golang.org/x/example/hello/reverse` package.

This will allow us to use the new code we will write in our copy of the `reverse` package instead of the version of the package in the module cache that we downloaded with the `go get` command.

>  现在，工作区包含了 `hello` 模块和 `example/hello` 模块，其中 `example/hello` 模块中包含了 `example/hello/reverse` 包

(3) Add the new function.

We’ll add a new function to reverse a number to the `golang.org/x/example/hello/reverse` package.

Create a new file named `int.go` in the `workspace/example/hello/reverse` directory containing the following contents:

```go
package reverse

import "strconv"

// Int returns the decimal reversal of the integer i.
func Int(i int) int {
    i, _ = strconv.Atoi(String(strconv.Itoa(i)))
    return i
}
```

>  一个包可以包含多个 `.go` 文件，我们为 `reverse` 包再创建一个 `int.go` 文件，并为它定义 `Int` 函数

(4) Modify the hello program to use the function.

Modify the contents of `workspace/hello/hello.go` to contain the following contents:

```go
package main

import (
    "fmt"

    "golang.org/x/example/hello/reverse"
)

func main() {
    fmt.Println(reverse.String("Hello"), reverse.Int(24601))
}
```

>  定义好 `Int` 函数后，我们可以直接在导入了 `reverse` 包的其他 `.go` 文件中使用该函数

#### Run the code in the workspace
From the workspace directory, run

```
$ go run ./hello
olleH 10642
```

The Go command finds the `example.com/hello` module specified in the command line in the `hello` directory specified by the `go.work` file, and similarly resolves the `golang.org/x/example/hello/reverse` import using the `go.work` file.

>  运行 `go run ./hello` 后，Go 会找到 `./hello` 目录中的 `hello` 模块，根据该模块的 `go.mod` 文件，发现该模块需要 `golang.org/x/example/hello/reverse` 包 (即模块中有的包导入了这个包)，Go 进而会根据工作区的 `go.work` 文件在工作区查找对应的模块和包

`go.work` can be used instead of adding [`replace`](https://go.dev/ref/mod#go-mod-file-replace) directives to work across multiple modules.

Since the two modules are in the same workspace it’s easy to make a change in one module and use it in another.

#### Future step
Now, to properly release these modules we’d need to make a release of the `golang.org/x/example/hello` module, for example at `v0.1.0`. This is usually done by tagging a commit on the module’s version control repository. See the [module release workflow documentation](https://go.dev/doc/modules/release-workflow) for more details. Once the release is done, we can increase the requirement on the `golang.org/x/example/hello` module in `hello/go.mod`:

```
cd hello
go get golang.org/x/example/hello@v0.1.0
```

That way, the `go` command can properly resolve the modules outside the workspace.

>  `go get` 还可以获取特定模块的特定版本发布

## Learn more about workspaces
The `go` command has a couple of subcommands for working with workspaces in addition to `go work init` which we saw earlier in the tutorial:

- `go work use [-r] [dir]` adds a `use` directive to the `go.work` file for `dir`, if it exists, and removes the `use` directory if the argument directory doesn’t exist. The `-r` flag examines subdirectories of `dir` recursively.
- `go work edit` edits the `go.work` file similarly to `go mod edit`
- `go work sync` syncs dependencies from the workspace’s build list into each of the workspace modules.

>  Go 用于处理工作空间的其他常用命令如上，其中：
>  - `go work use [-r] [dir]` 如果 `dir` 存在，该命令会在 `go.work` 文件中为 `dir` 添加 `use` 指令，如果 `dir` 不存在，该命令会在 `go.work` 文件中移除对 `dir` 的 `use` 指令
>  - `go work edit` 和 `go mod edit` 类似
>  - `go work sync` 将工作区的构建列表中的依赖同步到工作区的每个模块

See [Workspaces](https://go.dev/ref/mod#workspaces) in the Go Modules Reference for more detail on workspaces and `go.work` files.

