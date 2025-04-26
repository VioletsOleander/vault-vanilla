---
version: 1.24.2
completed:
---
### Variables 

```go
var (
	// ErrInvalid indicates an invalid argument.
	// Methods on File will return this error when the receiver is nil.
	ErrInvalid = fs.ErrInvalid // "invalid argument"

	ErrPermission = fs.ErrPermission // "permission denied"
	ErrExist      = fs.ErrExist      // "file already exists"
	ErrNotExist   = fs.ErrNotExist   // "file does not exist"
	ErrClosed     = fs.ErrClosed     // "file already closed"
	ErrNoDeadline       = errNoDeadline()       // "file type does not support deadline"
	ErrDeadlineExceeded = errDeadlineExceeded() // "i/o timeout"
)
```

Portable analogs of some common system call errors.

Errors returned from this package may be tested against these errors with [errors.Is](https://pkg.go.dev/errors#Is).

>  各个 `Err...` 变量表示了常见的系统调用错误

```go
var (
	Stdin  = NewFile(uintptr(syscall.Stdin), "/dev/stdin")
    Stdout = NewFile(uintptr(syscall.Stdout), "/dev/stdout")
    Stderr = NewFile(uintptr(syscall.Stderr), "/dev/stderr")
)
```

Stdin, Stdout, and Stderr are open Files pointing to the standard input, standard output, and standard error file descriptors.

Note that the Go runtime writes to standard error for panics and crashes; closing Stderr may cause those messages to go elsewhere, perhaps to a file opened later.

>  `Stdin/out/err` 为打开的文件，指向了标准输入、输出、错误文件描述符

```go
var Args []string
```

`Args` hold the command-line arguments, starting with the program name.

>  `Args` 保存了命令行参数，其中 `Args[0]` 是程序名称

```go
var ErrProcessDone = errors.New("os: process already finished")
```

`ErrProcessDone` indicates a [Process](https://pkg.go.dev/os@go1.24.2#Process) has finished.

### Functions
#### `func Exit`

```go
func Exit(code int)
```

Exit causes the current program to exit with the given status code. Conventionally, code zero indicates success, non-zero an error. The program terminates immediately; deferred functions are not run.
>  `Exit` 直接中止程序，不会运行推迟的函数

For portability, the status code should be in the range \[0, 125\].

### Types 

#### `type File`
##### `func Open`

```go
func Open(name string) (*File, error)
```

`Open` opens the named file for reading. If successful, methods on the returned file can be used for reading; the associated file descriptor has mode `O_RDONLY`. If there is an error, it will be of type `*PathError `.

>  `Open` 成功后，返回的 `*File` 关联的文件描述符的模式为 `O_RDONLY`

##### `func (*File) Close`

```go
func (f *File) Close() error
```

Close closes the [File](https://pkg.go.dev/os#File), rendering it unusable for I/O. On files that support [File.SetDeadline](https://pkg.go.dev/os#File.SetDeadline), any pending I/O operations will be canceled and return immediately with an [ErrClosed](https://pkg.go.dev/os#ErrClosed) error. Close will return an error if it has already been called.
