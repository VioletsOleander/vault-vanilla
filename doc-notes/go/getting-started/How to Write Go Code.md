# Introduction
This document demonstrates the development of a simple Go package inside a module and introduces the [go tool](https://go.dev/cmd/go/), the standard way to fetch, build, and install Go modules, packages, and commands.

# Code organization
Go programs are organized into packages. A package is a collection of source files in the same directory that are compiled together. Functions, types, variables, and constants defined in one source file are visible to all other source files within the same package.
>  Go 程序按照包组织，一个包是相同目录中的一组一同编译的源文件，在同一个包内，其他源文件中定义的函数、类型、变量、常数对于其他源文件都是可见的

A repository contains one or more modules. A module is a collection of related Go packages that are released together. A Go repository typically contains only one module, located at the root of the repository. A file named `go.mod` there declares the module path: the import path prefix for all packages within the module. The module contains the packages in the directory containing its `go.mod` file as well as subdirectories of that directory, up to the next subdirectory containing another `go.mod` file (if any).
>  一个仓库包含一个或多个模块，一个模块是一组一起发布的 Go 包，一个 Go 仓库通常仅包含一个模块，位于在仓库的根目录
>  仓库根目录下的名为 `go.mod` 的文件声明了模块路径，该路径是导入该模块中所有包所需要的前缀
>  模块包含了其 `go.mod` 文件所在目录 (一般是仓库根目录) 下的包及其子目录中的包，直到下一个包含了另一个 `go.mod` 文件的子目录 (如果有)

Note that you don't need to publish your code to a remote repository before you can build it. A module can be defined locally without belonging to a repository. However, it's a good habit to organize your code as if you will publish it someday.
>   我们无需将代码发布到远程仓库即可构建我们的代码。模块可以在本地定义，而不必属于某个仓库
>   但养成组织代码的习惯，仿佛总有一天会发布它，是个好的做法

Each module's path not only serves as an import path prefix for its packages, but also indicates where the `go` command should look to download it. For example, in order to download the module `golang.org/x/tools`, the `go` command would consult the repository indicated by `https://golang.org/x/tools` (described more [here](https://go.dev/cmd/go/#hdr-Remote_import_paths)).
>  每个模块的路径不仅仅作为其包的导入路径前缀，还表示了 `go` 命令应该去哪里下载它
>  例如，为了下载模块 `golang.org/x/tools` ，`go` 命令会查询 URL `https://golang.org/x/tools` 中的仓库

An import path is a string used to import a package. A package's import path is its module path joined with its subdirectory within the module. For example, the module `github.com/google/go-cmp` contains a package in the directory `cmp/`. That package's import path is `github.com/google/go-cmp/cmp`. Packages in the standard library do not have a module path prefix.
>  导入路径是用于导入一个包的字符串，一个包的导入路径是其所属的模块路径 + 它在该模块中的子目录
>  例如，模块 `github.com/google/go-cmp` 在目录 `cmp/` 包含了一个包，则该包的导入路径就是 `github.com/google/go-cmp/cmp`
>  标准库中的包导入没有模块名作为前缀

# Your first program 
To compile and run a simple program, first choose a module path (we'll use `example/user/hello`) and create a `go.mod` file that declares it:

```
$ mkdir hello # Alternatively, clone it if it already exists in version control.
$ cd hello
$ go mod init example/user/hello
go: creating new go.mod: module example/user/hello
$ cat go.mod
module example/user/hello

go 1.16
$
```

>  要编译并运行一个简单程序，我们首先选择一个模块路径，例如 `example/user/hello` ，然后创建一个 `go.mod` 文件，在 `go.mod` 文件中声明该模块路径 (即 `module example/user/hello`，实际 Go 在创建时会为我们自动声明)

The first statement in a Go source file must be `package name`. Executable commands must always use `package main`.
>  Go 源文件中的第一个语句一定是 `package name` ，即包声明
>  可执行命令必须总是使用 `package main`，即如果文件是一个可执行命令，必须在 `main` 包中，`main` 包是 Go 的程序入口点

Next, create a file named `hello.go` inside that directory containing the following Go code:

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, world.")
}
```

Now you can build and install that program with the `go` tool:
>  使用 `go` 工具构建且安装该程序

```
$ go install example/user/hello
$
```

This command builds the `hello` command, producing an executable binary. It then installs that binary as `$HOME/go/bin/hello` (or, under Windows, `%USERPROFILE%\go\bin\hello.exe`).
>  `go install example/user/hello` 命令构建了 `hello` 命令，生成了可执行二进制文件，然后将其安装在了 `$HOME/go/bin/hello`

The install directory is controlled by the `GOPATH` and `GOBIN` [environment variables](https://go.dev/cmd/go/#hdr-Environment_variables). If `GOBIN` is set, binaries are installed to that directory. If `GOPATH` is set, binaries are installed to the `bin` subdirectory of the first directory in the `GOPATH` list. Otherwise, binaries are installed to the `bin` subdirectory of the default `GOPATH` (`$HOME/go` or `%USERPROFILE%\go`).
>  二进制文件的安装路径由环境变量 `GOPATH` 和 `GOBIN` 控制，如果 `GOBIN` 被设置，二进制文件就会安装到 `GOBIN`，如果 `GOPATH` 被设置，二进制文件会被安装在 `GOPATH` 的第一个目录中的 `bin` 子目录下
>  否则，二进制文件会被安装到默认 `GOPATH` (`$HOME/go`) 下的 `bin` 子目录下

You can use the `go env` command to portably set the default value for an environment variable for future `go` commands:

```
$ go env -w GOBIN=/somewhere/else/bin
$
```

To unset a variable previously set by `go env -w`, use `go env -u`:

```
$ go env -u GOBIN
$
```

>  `go env` 命令可以用于设置环境变量
>  `go env -u` 用于取消设置环境变量

Commands like `go install` apply within the context of the module containing the current working directory. If the working directory is not within the `example/user/hello` module, `go install` may fail.
>  类似 `go install` 的命令是在当前工作目录所在的模块的上下文中应用的，如果当前工作目录不在 `example/user/hello` 模块中，`go install` 可能会失败

For convenience, `go` commands accept paths relative to the working directory, and default to the package in the current working directory if no other path is given. So in our working directory, the following commands are all equivalent:

```
$ go install example/user/hello

$ go install .

$ go install
```

>  `go` 命令接受相对于当前工作目录的相对路径，如果没有给定路径，则默认 (根据当前工作目录的 `go.mod` 文件) 使用当前工作目录中的包

Next, let's run the program to ensure it works. For added convenience, we'll add the install directory to our `PATH` to make running binaries easy:

```
# Windows users should consult /wiki/SettingGOPATH
# for setting %PATH%.
$ export PATH=$PATH:$(dirname $(go list -f '{{.Target}}' .))
$ hello
Hello, world.
$
```

If you're using a source control system, now would be a good time to initialize a repository, add the files, and commit your first change. Again, this step is optional: you do not need to use source control to write Go code.

```
$ git init
Initialized empty Git repository in /home/user/hello/.git/
$ git add go.mod hello.go
$ git commit -m "initial commit"
[master (root-commit) 0b4507d] initial commit
 1 file changed, 7 insertion(+)
 create mode 100644 go.mod hello.go
$
```

The `go` command locates the repository containing a given module path by requesting a corresponding HTTPS URL and reading metadata embedded in the HTML response (see `[go help importpath](https://go.dev/cmd/go/#hdr-Remote_import_paths)`). Many hosting services already provide that metadata for repositories containing Go code, so the easiest way to make your module available for others to use is usually to make its module path match the URL for the repository.
>  `go` 命令通过请求与模块路径对应的 HTTPS URL (例如，模块路径是 `golang.org/x/tools`，Go 就会请求 `https://golang.org/x/tools`) 并且读取返回的 HTML 回应中的元数据来定位包含了该模块的仓库 (即这些元数据告诉了 Go 对应模块的代码存储在哪个仓库中)
>  许多托管服务 (例如 GitHub)  已经为包含 Go 代码的仓库提供了元数据，因此，为了让我们的模块可以被其他人使用，最简单的方法就是让模块路径与仓库 URL 匹配 (例如模块路径是 `github.com/username/repo`，则 Go 请求 `https://github.com/username/repo` 就会得到所想要的元数据，并且 `https://github.com/username/repo` 中本身也包含了我们的 Go 代码)

## Importing packages from your module 
Let's write a `morestrings` package and use it from the `hello` program. First, create a directory for the package named `$HOME/hello/morestrings`, and then a file named `reverse.go` in that directory with the following contents:

```go
// Package morestrings implements additional functions to manipulate UTF-8
// encoded strings, beyond what is provided in the standard "strings" package.
package morestrings

// ReverseRunes returns its argument string reversed rune-wise left to right.
func ReverseRunes(s string) string {
    r := []rune(s)
    for i, j := 0, len(r)-1; i < len(r)/2; i, j = i+1, j-1 {
        r[i], r[j] = r[j], r[i]
    }
    return string(r)
}
```

Because our `ReverseRunes` function begins with an upper-case letter, it is [exported](https://go.dev/ref/spec#Exported_identifiers), and can be used in other packages that import our `morestrings` package.
>  因为 `ReverseRunes` 函数以大写字母开头，故它是被导出的，可以被用在导入了 `morestrings` 包的其他包中

Let's test that the package compiles with `go build`:

```
$ cd $HOME/hello/morestrings
$ go build
$
```

This won't produce an output file. Instead it saves the compiled package in the local build cache.

>  我们可以用 `go build` 测试我们的包是否可以编译，这不会生成输出文件，但会将编译好的包存储在本地构建缓存中

After confirming that the `morestrings` package builds, let's use it from the `hello` program. To do so, modify your original `$HOME/hello/hello.go` to use the `morestrings` package:

```go
package main

import (
    "fmt"

    "example/user/hello/morestrings"
)

func main() {
    fmt.Println(morestrings.ReverseRunes("!oG ,olleH"))
}
```

>  成功 `build` 的包就可以被导入了

Install the `hello` program:

```
$ go install example/user/hello
```

>  使用 `install` 编译并安装程序

Running the new version of the program, you should see a new, reversed message:

```
$ hello
Hello, Go!
```

## Importing packages from remote modules 
An import path can describe how to obtain the package source code using a revision control system such as Git or Mercurial. The `go` tool uses this property to automatically fetch packages from remote repositories. For instance, to use `github.com/google/go-cmp/cmp` in your program:

```go
package main

import (
    "fmt"

    "example/user/hello/morestrings"
    "github.com/google/go-cmp/cmp"
)

func main() {
    fmt.Println(morestrings.ReverseRunes("!oG ,olleH"))
    fmt.Println(cmp.Diff("Hello World", "Hello Go"))
}
```

>  `go` 工具也可以从远程仓库自动获取包

Now that you have a dependency on an external module, you need to download that module and record its version in your `go.mod` file. The `go mod tidy` command adds missing module requirements for imported packages and removes requirements on modules that aren't used anymore.

>  当我们对外部模块有依赖时，我们需要在 `go.mod` 文件中记录其版本
>  `go mod tidy` 命令会根据导入的包整理 `go.mod` 中的模块信息

```
$ go mod tidy
go: finding module for package github.com/google/go-cmp/cmp
go: found github.com/google/go-cmp/cmp in github.com/google/go-cmp v0.5.4
$ go install example/user/hello
$ hello
Hello, Go!
  string(
-     "Hello World",
+     "Hello Go",
  )
$ cat go.mod
module example/user/hello

go 1.16

require github.com/google/go-cmp v0.5.4
$
```

Module dependencies are automatically downloaded to the `pkg/mod` subdirectory of the directory indicated by the `GOPATH` environment variable. The downloaded contents for a given version of a module are shared among all other modules that `require` that version, so the `go` command marks those files and directories as read-only. To remove all downloaded modules, you can pass the `-modcache` flag to `go clean`:

>  模块依赖会被自动下载到由 `GOPATH` 变量指示的目录中的 `pkg/mod` 子目录中
>  这里的模块会被所有 `require` 它的模块共享，`go` 命令会将这些文件和目录为只读
>  要移除所有下载的模块，可以向 `go clean` 传递 `-modcache`

```
$ go clean -modcache
$
```

# Testing 
Go has a lightweight test framework composed of the `go test` command and the `testing` package.

You write a test by creating a file with a name ending in `_test.go` that contains functions named `TestXXX` with signature `func (t *testing.T)`. The test framework runs each such function; if the function calls a failure function such as `t.Error` or `t.Fail`, the test is considered to have failed.

>  Go 的测试框架由 `go test` 命令和 `testing` 包组成
>  要测试，我们创建一个后缀为 `_test.go` 的文件，该文件包含了名为 `TestXXX` 的函数，其签名为 `func (t *testing.T)` ，测试框架会运行每个这样的函数，如果函数调用了像 `t.Error, t.Fail` 这样的失败函数，就认为测试失败

Add a test to the `morestrings` package by creating the file `$HOME/hello/morestrings/reverse_test.go` containing the following Go code.

```go
package morestrings

import "testing"

func TestReverseRunes(t *testing.T) {
    cases := []struct {
        in, want string
    }{
        {"Hello, world", "dlrow ,olleH"},
        {"Hello, 世界", "界世 ,olleH"},
        {"", ""},
    }
    for _, c := range cases {
        got := ReverseRunes(c.in)
        if got != c.want {
            t.Errorf("ReverseRunes(%q) == %q, want %q", c.in, got, c.want)
        }
    }
}
```

Then run the test with `go test`:

```
$ cd $HOME/hello/morestrings
$ go test
PASS
ok  	example/user/hello/morestrings 0.165s
$
```

Run `[go help test](https://go.dev/cmd/go/#hdr-Test_packages)` and see the [testing package documentation](https://go.dev/pkg/testing/) for more detail.

# What's next 
Subscribe to the [golang-announce](https://groups.google.com/group/golang-announce) mailing list to be notified when a new stable version of Go is released.

See [Effective Go](https://go.dev/doc/effective_go.html) for tips on writing clear, idiomatic Go code.

Take [A Tour of Go](https://go.dev/tour/) to learn the language proper.

Visit the [documentation page](https://go.dev/doc/#articles) for a set of in-depth articles about the Go language and its libraries and tools.

# Getting help
For real-time help, ask the helpful gophers in the community-run [gophers Slack server](https://gophers.slack.com/messages/general/) (grab an invite [here](https://invite.slack.golangbridge.org/)).

The official mailing list for discussion of the Go language is [Go Nuts](https://groups.google.com/group/golang-nuts).

Report bugs using the [Go issue tracker](https://go.dev/issue).