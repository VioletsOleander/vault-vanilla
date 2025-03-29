Each Go module is defined by a go.mod file that describes the module’s properties, including its dependencies on other modules and on versions of Go.
>  每个 Go module 都由 `go.mod` 文件定义，它描述了模块属性，包括了对其他模块的依赖和对 Go 版本的依赖

These properties include:

- The current module’s **module path**. This should be a location from which the module can be downloaded by Go tools, such as the module code’s repository location. This serves as a unique identifier, when combined with the module’s version number. It is also the prefix of the package path for all packages in the module. For more about how Go locates the module, see the [Go Modules Reference](https://go.dev/ref/mod#vcs-find).
- The minimum **version of Go** required by the current module.
- A list of minimum versions of other **modules required** by the current module.
- Instructions, optionally, to **replace** a required module with another module version or a local directory, or to **exclude** a specific version of a required module.

>  `go.mod` 描述的模块属性包括:
>  - 当前模块的模块路径。Go tools 应该可以用这一路径下载该模块，故模块路径一般是模块代码的 repo location。模块路径+模块版本是模块的唯一标识符，模块路径也是模块中包路径的前缀
>  - 当前模块要求的 Go 的最低版本。
>  - 当前模块依赖的其他模块要求的 Go 的最低版本。
>  - 指令 (可选)。用于将某个要求的模块用其他模块版本替代，或者排除要求的模块的某个特定版本

Go generates a go.mod file when you run the [`go mod init` command](https://go.dev/ref/mod#go-mod-init). The following example creates a go.mod file, setting the module’s module path to `example/mymodule`:

```
$ go mod init example/mymodule
```

>  `go mod init` 会创建 `go.mod` 文件

Use `go` commands to manage dependencies. The commands ensure that the requirements described in your go.mod file remain consistent and the content of your go.mod file is valid. These commands include the [`go get`](https://go.dev/ref/mod#go-get) and [`go mod tidy`](https://go.dev/ref/mod#go-mod-tidy) and [`go mod edit`](https://go.dev/ref/mod#go-mod-edit) commands.

For reference on `go` commands, see [Command go](https://go.dev/cmd/go/). You can get help from the command line by typing `go help` _command-name_, as with `go help mod tidy`.

**See also**

- Go tools make changes to your go.mod file as you use them to manage dependencies. For more, see [Managing dependencies](https://go.dev/doc/modules/managing-dependencies).
- For more details and constraints related to go.mod files, see the [Go modules reference](https://go.dev/ref/mod#go-mod-file).

# Example 
A go.mod file includes directives as shown in the following example. These are described elsewhere in this topic.

```
module example.com/mymodule

go 1.14

require (
    example.com/othermodule v1.2.3
    example.com/thismodule v1.2.3
    example.com/thatmodule v1.2.3
)

replace example.com/thatmodule => ../thatmodule
exclude example.com/thismodule v1.3.0
```

# module 
Declares the module’s module path, which is the module’s unique identifier (when combined with the module version number). The module path becomes the import prefix for all packages the module contains.
>  `module` 指令用于声明模块路径，作为模块的标识符和模块中包的导入前缀

For more, see [`module` directive](https://go.dev/ref/mod#go-mod-file-module) in the Go Modules Reference.

## Syntax 

```
module module-path
```

module-path: The module's module path, usually the repository location from which the module can be downloaded by Go tools. For module versions v2 and later, this value must end with the major version number, such as `/v2`.

>  模块路径可以以 `/v2` 这样的 major version number 作为后缀

## Examples 
The following examples substitute `example.com` for a repository domain from which the module could be downloaded.

- Module declaration for a v0 or v1 module:
    
    ```
    module example.com/mymodule
    ```
    
- Module path for a v2 module:
    
    ```
    module example.com/mymodule/v2
    ```

## Notes 
The module path must uniquely identify your module. For most modules, the path is a URL where the `go` command can find the code (or a redirect to the code). For modules that won’t ever be downloaded directly, the module path can be just some name you control that will ensure uniqueness. The prefix `example/` is also reserved for use in examples like these.
>  大多数模块的模块路径都是 URL
>  如果不是可以被直接下载的模块，其路径可以仅是名称

For more details, see [Managing dependencies](https://go.dev/doc/modules/managing-dependencies#naming_module).

In practice, the module path is typically the module source’s repository domain and path to the module code within the repository. The `go` command relies on this form when downloading module versions to resolve dependencies on the module user’s behalf.
>  实践中，模块路径一般是形式为 source repo domain + path to the module within the repo 的 URL
>  `go` 命令会用它来下载模块

Even if you’re not at first intending to make your module available for use from other code, using its repository path is a best practice that will help you avoid having to rename the module if you publish it later.
>  使用 repo path 是 best practice

If at first you don’t know the module’s eventual repository location, consider temporarily using a safe substitute, such as the name of a domain you own or a name you control (such as your company name), along with a path following from the module’s name or source directory. For more, see [Managing dependencies](https://go.dev/doc/modules/managing-dependencies#naming_module).

For example, if you’re developing in a `stringtools` directory, your temporary module path might be `<company-name>/stringtools`, as in the following example, where _company-name_ is your company’s name:

```
go mod init <company-name>/stringtools
```

# go 
Indicates that the module was written assuming the semantics of the Go version specified by the directive.
>  `go` 指示了该模块是基于某个最小的 Go 版本写成的

For more, see [`go` directive](https://go.dev/ref/mod#go-mod-file-go) in the Go Modules Reference.

## Syntax

```
go minimum-go-version
```

minimum-go-version: The minimum version of Go required to compile packages in this module.
>  `go` 指示了编译本模块中的包，需要的最小 Go 版本

## Examples

- Module must run on Go version 1.14 or later:
    
    ```
    go 1.14
    ```

## Notes 
The `go` directive sets the minimum version of Go required to use this module. Before Go 1.21, the directive was advisory only; now it is a mandatory requirement: Go toolchains refuse to use modules declaring newer Go versions.

The `go` directive is an input into selecting which Go toolchain to run. See “[Go toolchains](https://go.dev/doc/toolchain)” for details.

>  `go` 指令指示的 Go 版本会被用于选择 Go toolchain

The `go` directive affects use of new language features:

- For packages within the module, the compiler rejects use of language features introduced after the version specified by the `go` directive. For example, if a module has the directive `go 1.12`, its packages may not use numeric literals like `1_000_000`, which were introduced in Go 1.13.
- If an older Go version builds one of the module’s packages and encounters a compile error, the error notes that the module was written for a newer Go version. For example, suppose a module has `go 1.13` and a package uses the numeric literal `1_000_000`. If that package is built with Go 1.12, the compiler notes that the code is written for Go 1.13.

The `go` directive also affects the behavior of the `go` command:

- At `go 1.14` or higher, automatic [vendoring](https://go.dev/ref/mod#vendoring) may be enabled. If the file `vendor/modules.txt` is present and consistent with `go.mod`, there is no need to explicitly use the `-mod=vendor` flag.
- At `go 1.16` or higher, the `all` package pattern matches only packages transitively imported by packages and tests in the [main module](https://go.dev/ref/mod#glos-main-module). This is the same set of packages retained by [`go mod vendor`](https://go.dev/ref/mod#go-mod-vendor) since modules were introduced. In lower versions, `all` also includes tests of packages imported by packages in the main module, tests of those packages, and so on.
- At `go 1.17` or higher:
    - The `go.mod` file includes an explicit [`require` directive](https://go.dev/ref/mod#go-mod-file-require) for each module that provides any package transitively imported by a package or test in the main module. (At `go 1.16` and lower, an indirect dependency is included only if [minimal version selection](https://go.dev/ref/mod#minimal-version-selection) would otherwise select a different version.) This extra information enables [module graph pruning](https://go.dev/ref/mod#graph-pruning) and [lazy module loading](https://go.dev/ref/mod#lazy-loading).
    - Because there may be many more `// indirect` dependencies than in previous `go` versions, indirect dependencies are recorded in a separate block within the `go.mod` file.
    - `go mod vendor` omits `go.mod` and `go.sum` files for vendored dependencies. (That allows invocations of the `go` command within subdirectories of `vendor` to identify the correct main module.)
    - `go mod vendor` records the `go` version from each dependency’s `go.mod` file in `vendor/modules.txt`.
- At `go 1.21` or higher:
    - The `go` line declares a required minimum version of Go to use with this module.
    - The `go` line must be greater than or equal to the `go` line of all dependencies.
    - The `go` command no longer attempts to maintain compatibility with the previous older version of Go.
    - The `go` command is more careful about keeping checksums of `go.mod` files in the `go.sum` file.

A `go.mod` file may contain at most one `go` directive. Most commands will add a `go` directive with the current Go version if one is not present.