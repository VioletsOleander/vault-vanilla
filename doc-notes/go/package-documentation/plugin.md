---
completed: true
version: 1.24.1
---
Package plugin implements loading and symbol resolution of Go plugins.
>  `plugin` 包实现了 Go plugin 的加载和符号解析

A plugin is a Go main package with exported functions and variables that has been built with:

```
go build -buildmode=plugin
```

>  plugin 是一个包含了导出函数和变量的 Go main package，它通过 `go build -buildmode=plugin` 构建

When a plugin is first opened, the init functions of all packages not already part of the program are called. The main function is not run. A plugin is only initialized once, and cannot be closed.
>  当 plugin 被打开后，插件中所有尚未被主程序包含的包的 `init` 函数会被调用，以初始化它们，但插件的 `main` 函数不会运行
>  一个 plugin 只能被初始化一次，并且一旦加载后不再关闭

>  插件的功能类似于动态链接库，使得程序在运行时可以直接加载和使用编译好的动态链接库

**Warnings** 
The ability to dynamically load parts of an application during execution, perhaps based on user-defined configuration, may be a useful building block in some designs. In particular, because applications and dynamically loaded functions can share data structures directly, plugins may enable very high-performance integration of separate parts.

However, the plugin mechanism has many significant drawbacks that should be considered carefully during the design. For example:

- Plugins are currently supported only on Linux, FreeBSD, and macOS, making them unsuitable for applications intended to be portable.
- Plugins are poorly supported by the Go race detector. Even simple race conditions may not be automatically detected. See [https://go.dev/issue/24245](https://go.dev/issue/24245) for more information.
- Applications that use plugins may require careful configuration to ensure that the various parts of the program be made available in the correct location in the file system (or container image). By contrast, deploying an application consisting of a single static executable is straightforward.
- Reasoning about program initialization is more difficult when some packages may not be initialized until long after the application has started running.
- Bugs in applications that load plugins could be exploited by an attacker to load dangerous or untrusted libraries.
- Runtime crashes are likely to occur unless all parts of the program (the application and all its plugins) are compiled using exactly the same version of the toolchain, the same build tags, and the same values of certain flags and environment variables.
- Similar crashing problems are likely to arise unless all common dependencies of the application and its plugins are built from exactly the same source code.
- Together, these restrictions mean that, in practice, the application and its plugins must all be built together by a single person or component of a system. In that case, it may be simpler for that person or component to generate Go source files that blank-import the desired set of plugins and then compile a static executable in the usual way.

>  plugin 机制存在的缺点有
>  - 支持的平台有限
>  - 对于 Go 的 race detector 支持不足，即便是简单的竞争条件也无法自动检查到
>  - 使用插件的应用程序需要仔细配置，确保能够在文件系统的正确位置找到插件，相比之下部署单一静态可执行文件要更加容易
>  - 某些包可能在应用启动很长时间后才初始化时，不便于理解程序的初始化
>  - 安全性
>  - 除非应用的所有部分 (程序本身和其 plugins) 都使用完全相同的工具链、版本、构建标签、环境变量值进行编译，否则很可能会发生运行时崩溃
>  - 除非应用和其插件的公共依赖都从完全相同的源代码构建，否则容易发生崩溃
>  - 综上所述，这些限制意味着实践中，应用和其插件应该由同一个人或者系统的某个组件一起构建，但在该情况下，该构建者简单地生成一个导入所需插件机和的源文件，然后静态编译反而是更简单的方式

For these reasons, many users decide that traditional inter-process communication (IPC) mechanisms such as sockets, pipes, remote procedure call (RPC), shared memory mappings, or file system operations may be more suitable despite the performance overheads.
>  因此，许多用户认为，尽管存在性能开销，但传统的 IPC 机制，例如 sockets, pipes, RPC, shared memory mappings, file system operations 可能更为合适

### Types 
#### type `Plugin`

```go
type Plugin struct {
	// contains filtered or unexported fields
}
```

Plugin is a loaded Go plugin.

#### func `Open`

```go
func Open(path string) (*Plugin, error)
```

Open opens a Go plugin. If a path has already been opened, then the existing *[Plugin](https://pkg.go.dev/plugin#Plugin) is returned. It is safe for concurrent use by multiple goroutines.

#### func `(*Plugin) Lookup`

```go
func (*Plugin) Lookup(symName string) (Symbol, error)
```

Lookup searches for a symbol named `symName` in plugin p. A symbol is any exported variable or function. It reports an error if the symbol is not found. It is safe for concurrent use by multiple goroutines.

#### type `Symbol `

```go
type Symbol any
```

A Symbol is a pointer to a variable or function.

For example, a plugin defined as

```go
package main

import "fmt"

var V int

func F() { fmt.Printf("Hello, number %d\n", V) }
```

may be loaded with the [Open](https://pkg.go.dev/plugin#Open) function and then the exported package symbols V and F can be accessed

```go
p, err := plugin.Open("plugin_name.so")
if err != nil {
	panic(err)
}
v, err := p.Lookup("V")
if err != nil {
	panic(err)
}
f, err := p.Lookup("F")
if err != nil {
	panic(err)
}
*v.(*int) = 7
f.(func())() // prints "Hello, number 7"
```