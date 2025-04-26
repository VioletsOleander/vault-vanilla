---
version: 1.24.1
---
Go is a tool for managing Go source code.

Usage:

```
go <command> [arguments]
```

The commands are:

```
bug         start a bug report
build       compile packages and dependencies
clean       remove object files and cached files
doc         show documentation for package or symbol
env         print Go environment information
fix         update packages to use new APIs
fmt         gofmt (reformat) package sources
generate    generate Go files by processing source
get         add dependencies to current module and install them
install     compile and install packages and dependencies
list        list packages or modules
mod         module maintenance
work        workspace maintenance
run         compile and run Go program
telemetry   manage telemetry data and settings
test        test packages
tool        run specified go tool
version     print Go version
vet         report likely mistakes in packages
```

Use "go help <command>" for more information about a command.

Additional help topics:

```
buildconstraint build constraints
buildjson       build -json encoding
buildmode       build modes
c               calling between Go and C
cache           build and test caching
environment     environment variables
filetype        file types
goauth          GOAUTH environment variable
go.mod          the go.mod file
gopath          GOPATH environment variable
goproxy         module proxy protocol
importpath      import path syntax
modules         modules, module versions, and more
module-auth     module authentication using go.sum
packages        package lists and patterns
private         configuration for downloading non-public code
testflag        testing flags
testfunc        testing functions
vcs             controlling version control with GOVCS
```

Use "go help \<topic\>" for more information about that topic.

#### Start a bug report
Usage:

```
go bug
```

Bug opens the default browser and starts a new bug report. The report includes useful system information.

#### Compile packages and dependencies 
Usage:

```
go build [-o output] [build flags] [packages]
```

Build compiles the packages named by the import paths, along with their dependencies, but it does not install the results.
>  `go build` 用于编译给定包导入路径的指向的包，同时会编译它的依赖，但不会安装编译结果

If the arguments to build are a list of .go files from a single directory, build treats them as a list of source files specifying a single package.
>  如果 `build` 的参数是同一目录下的多个 `.go` 文件，`build` 将这些文件视作单个包的源文件列表

When compiling packages, build ignores files that end in '\_test.go'.
>  编译包时， `build` 会忽略以 `_test.go` 为后缀的文件

When compiling a single main package, build writes the resulting executable to an output file named after the last non-major-version component of the package import path. The '.exe' suffix is added when writing a Windows executable. So 'go build example/sam' writes 'sam' or 'sam.exe'. 'go build example.com/foo/v2' writes 'foo' or 'foo.exe', not 'v2.exe'.
>  编译单个 main 包时，`build` 会输出可执行文件，文件名称是包导入路径的最后一个 non-major-version component
>  例如 `go build example/sam` 会输出 `sam` 可执行文件，`go build example.com/foo/v2` 会输出 `foo` 可执行文件

When compiling a package from a list of .go files, the executable is named after the first source file. 'go build ed.go rx.go' writes 'ed' or 'ed.exe'.
>  从一个 `.go` 文件列表中编译包时，可执行文件以第一个 `.go` 文件命名
>  例如 `go build ed.go rx.go` 生成 `ed`

When compiling multiple packages or a single non-main package, build compiles the packages but discards the resulting object, serving only as a check that the packages can be built.
>  编译多个包或者单个 non-main 包时，`build` 执行编译，但会丢弃编译得到的对象，故仅用于检查是否成功编译

The -o flag forces build to write the resulting executable or object to the named output file or directory, instead of the default behavior described in the last two paragraphs. If the named output is an existing directory or ends with a slash or backslash, then any resulting executables will be written to that directory.
>  `-o` 强制让 `build` 将编译结果写出到指定文件或目录下

The build flags are shared by the build, clean, get, install, list, run, and test commands:

```
-C dir
	Change to dir before running the command.
	Any files named on the command line are interpreted after
	changing directories.
	If used, this flag must be the first one in the command line.
-a
	force rebuilding of packages that are already up-to-date.
-n
	print the commands but do not run them.
-p n
	the number of programs, such as build commands or
	test binaries, that can be run in parallel.
	The default is GOMAXPROCS, normally the number of CPUs available.
-race
	enable data race detection.
	Supported only on linux/amd64, freebsd/amd64, darwin/amd64, darwin/arm64, windows/amd64,
	linux/ppc64le and linux/arm64 (only for 48-bit VMA).
-msan
	enable interoperation with memory sanitizer.
	Supported only on linux/amd64, linux/arm64, linux/loong64, freebsd/amd64
	and only with Clang/LLVM as the host C compiler.
	PIE build mode will be used on all platforms except linux/amd64.
-asan
	enable interoperation with address sanitizer.
	Supported only on linux/arm64, linux/amd64, linux/loong64.
	Supported on linux/amd64 or linux/arm64 and only with GCC 7 and higher
	or Clang/LLVM 9 and higher.
	And supported on linux/loong64 only with Clang/LLVM 16 and higher.
-cover
	enable code coverage instrumentation.
-covermode set,count,atomic
	set the mode for coverage analysis.
	The default is "set" unless -race is enabled,
	in which case it is "atomic".
	The values:
	set: bool: does this statement run?
	count: int: how many times does this statement run?
	atomic: int: count, but correct in multithreaded tests;
		significantly more expensive.
	Sets -cover.
-coverpkg pattern1,pattern2,pattern3
	For a build that targets package 'main' (e.g. building a Go
	executable), apply coverage analysis to each package whose
	import path matches the patterns. The default is to apply
	coverage analysis to packages in the main Go module. See
	'go help packages' for a description of package patterns.
	Sets -cover.
-v
	print the names of packages as they are compiled.
-work
	print the name of the temporary work directory and
	do not delete it when exiting.
-x
	print the commands.
-asmflags '[pattern=]arg list'
	arguments to pass on each go tool asm invocation.
-buildmode mode
	build mode to use. See 'go help buildmode' for more.
-buildvcs
	Whether to stamp binaries with version control information
	("true", "false", or "auto"). By default ("auto"), version control
	information is stamped into a binary if the main package, the main module
	containing it, and the current directory are all in the same repository.
	Use -buildvcs=false to always omit version control information, or
	-buildvcs=true to error out if version control information is available but
	cannot be included due to a missing tool or ambiguous directory structure.
-compiler name
	name of compiler to use, as in runtime.Compiler (gccgo or gc).
-gccgoflags '[pattern=]arg list'
	arguments to pass on each gccgo compiler/linker invocation.
-gcflags '[pattern=]arg list'
	arguments to pass on each go tool compile invocation.
-installsuffix suffix
	a suffix to use in the name of the package installation directory,
	in order to keep output separate from default builds.
	If using the -race flag, the install suffix is automatically set to race
	or, if set explicitly, has _race appended to it. Likewise for the -msan
	and -asan flags. Using a -buildmode option that requires non-default compile
	flags has a similar effect.
-json
	Emit build output in JSON suitable for automated processing.
	See 'go help buildjson' for the encoding details.
-ldflags '[pattern=]arg list'
	arguments to pass on each go tool link invocation.
-linkshared
	build code that will be linked against shared libraries previously
	created with -buildmode=shared.
-mod mode
	module download mode to use: readonly, vendor, or mod.
	By default, if a vendor directory is present and the go version in go.mod
	is 1.14 or higher, the go command acts as if -mod=vendor were set.
	Otherwise, the go command acts as if -mod=readonly were set.
	See https://golang.org/ref/mod#build-commands for details.
-modcacherw
	leave newly-created directories in the module cache read-write
	instead of making them read-only.
-modfile file
	in module aware mode, read (and possibly write) an alternate go.mod
	file instead of the one in the module root directory. A file named
	"go.mod" must still be present in order to determine the module root
	directory, but it is not accessed. When -modfile is specified, an
	alternate go.sum file is also used: its path is derived from the
	-modfile flag by trimming the ".mod" extension and appending ".sum".
-overlay file
	read a JSON config file that provides an overlay for build operations.
	The file is a JSON struct with a single field, named 'Replace', that
	maps each disk file path (a string) to its backing file path, so that
	a build will run as if the disk file path exists with the contents
	given by the backing file paths, or as if the disk file path does not
	exist if its backing file path is empty. Support for the -overlay flag
	has some limitations: importantly, cgo files included from outside the
	include path must be in the same directory as the Go package they are
	included from, and overlays will not appear when binaries and tests are
	run through go run and go test respectively.
-pgo file
	specify the file path of a profile for profile-guided optimization (PGO).
	When the special name "auto" is specified, for each main package in the
	build, the go command selects a file named "default.pgo" in the package's
	directory if that file exists, and applies it to the (transitive)
	dependencies of the main package (other packages are not affected).
	Special name "off" turns off PGO. The default is "auto".
-pkgdir dir
	install and load all packages from dir instead of the usual locations.
	For example, when building with a non-standard configuration,
	use -pkgdir to keep generated packages in a separate location.
-tags tag,list
	a comma-separated list of additional build tags to consider satisfied
	during the build. For more information about build tags, see
	'go help buildconstraint'. (Earlier versions of Go used a
	space-separated list, and that form is deprecated but still recognized.)
-trimpath
	remove all file system paths from the resulting executable.
	Instead of absolute file system paths, the recorded file names
	will begin either a module path@version (when using modules),
	or a plain import path (when using the standard library, or GOPATH).
-toolexec 'cmd args'
	a program to use to invoke toolchain programs like vet and asm.
	For example, instead of running asm, the go command will run
	'cmd args /path/to/asm <arguments for asm>'.
	The TOOLEXEC_IMPORTPATH environment variable will be set,
	matching 'go list -f {{.ImportPath}}' for the package being built.
```

The -asmflags, -gccgoflags, -gcflags, and -ldflags flags accept a space-separated list of arguments to pass to an underlying tool during the build. To embed spaces in an element in the list, surround it with either single or double quotes. The argument list may be preceded by a package pattern and an equal sign, which restricts the use of that argument list to the building of packages matching that pattern (see 'go help packages' for a description of package patterns). Without a pattern, the argument list applies only to the packages named on the command line. The flags may be repeated with different patterns in order to specify different arguments for different sets of packages. If a package matches patterns given in multiple flags, the latest match on the command line wins. For example, 'go build -gcflags=-S fmt' prints the disassembly only for package fmt, while 'go build -gcflags=all=-S fmt' prints the disassembly for fmt and all its dependencies.
>  skip

For more about specifying packages, see 'go help packages'. For more about where packages and binaries are installed, run 'go help gopath'. For more about calling between Go and C/C++, run 'go help c'.

Note: Build adheres to certain conventions such as those described by 'go help gopath'. Not all projects can follow these conventions, however. Installations that have their own conventions or that use a separate software build system may choose to use lower-level invocations such as 'go tool compile' and 'go tool link' to avoid some of the overheads and design decisions of the build tool.

See also: go install, go get, go clean.

#### Compile and run Go program 
Usage:

```
go run [build flags] [-exec xprog] package [arguments...]
```

Run compiles and runs the named main Go package. Typically the package is specified as a list of .go source files from a single directory, but it may also be an import path, file system path, or pattern matching a single known package, as in 'go run .' or 'go run my/cmd'.
>  `go run` 编译并运行指定的 main 包
>  其中 `package` 可以是单个目录中的 `.go` 源文件列表，或者一个 import path, file system path, pattern matching a single known package

If the package argument has a version suffix (like @latest or @v1.0.0), "go run" builds the program in module-aware mode, ignoring the go.mod file in the current directory or any parent directory, if there is one. This is useful for running programs without affecting the dependencies of the main module.

If the package argument doesn't have a version suffix, "go run" may run in module-aware mode or GOPATH mode, depending on the GO111MODULE environment variable and the presence of a go.mod file. See 'go help modules' for details. If module-aware mode is enabled, "go run" runs in the context of the main module.

>  skip both

By default, 'go run' runs the compiled binary directly: 'a.out arguments...'. If the -exec flag is given, 'go run' invokes the binary using xprog:

```
'xprog a.out arguments...'.
```

>  `go run` 默认会运行编译后的文件，形式为 `a.out arguments...`，如果给定了 `-exec` 则使用 `xprog` ，形式为 ` xprog a.out arguments... `

If the -exec flag is not given, GOOS or GOARCH is different from the system default, and a program named `go_GOOS_GOARCH_exec` can be found on the current search path, 'go run' invokes the binary using that program, for example ' `go_js_wasm_exec a.out arguments...` '. This allows execution of cross-compiled programs when a simulator or other execution method is available.
>  如果未提供 `-exec`，并且 `GOOS, GOARCH` 和系统默认值不同，同时在当前搜索路径可以找到名为 `go_GOOS_GOARCH_exec` 的程序，`go run` 会用该程序调用二进制文件
>  这用于借助模拟器或其他执行方法执行交叉编译的程序

By default, 'go run' compiles the binary without generating the information used by debuggers, to reduce build time. To include debugger information in the binary, use 'go build'.
>  `go run` 默认不会在 binary 中生成 debug 信息，以减少编译时间

The exit status of Run is not the exit status of the compiled binary.
>  `go run` 的退出状态不是 compiled binary 的退出状态

For more about build flags, see 'go help build'. For more about specifying packages, see 'go help packages'.

See also: go build.

#### Build modes 
The 'go build' and 'go install' commands take a `-buildmode` argument which indicates which kind of object file is to be built. Currently supported values are:
>  `go build, go install` 命令接收 `-buildmode` 参数，用于指定需要构建的目标文件类型

```
-buildmode=archive
	Build the listed non-main packages into .a files. Packages named
	main are ignored.

-buildmode=c-archive
	Build the listed main package, plus all packages it imports,
	into a C archive file. The only callable symbols will be those
	functions exported using a cgo //export comment. Requires
	exactly one main package to be listed.

-buildmode=c-shared
	Build the listed main package, plus all packages it imports,
	into a C shared library. The only callable symbols will
	be those functions exported using a cgo //export comment.
	On wasip1, this mode builds it to a WASI reactor/library,
	of which the callable symbols are those functions exported
	using a //go:wasmexport directive. Requires exactly one
	main package to be listed.

-buildmode=default
	Listed main packages are built into executables and listed
	non-main packages are built into .a files (the default
	behavior).

-buildmode=shared
	Combine all the listed non-main packages into a single shared
	library that will be used when building with the -linkshared
	option. Packages named main are ignored.

-buildmode=exe
	Build the listed main packages and everything they import into
	executables. Packages not named main are ignored.

-buildmode=pie
	Build the listed main packages and everything they import into
	position independent executables (PIE). Packages not named
	main are ignored.

-buildmode=plugin
	Build the listed main packages, plus all packages that they
	import, into a Go plugin. Packages not named main are ignored.
```

On AIX, when linking a C program that uses a Go archive built with `-buildmode=c-archive`, you must pass `-Wl`, `-bnoobjreorder` to the C compiler.