### Functions 
#### func Fprintf

```go
func Fprintf(w io.Writer, format string, a ...any) (n int, err error)
```

`Fprintf` formats according to a format specifier and writes to w. It returns the number of bytes written and any write error encountered.

>  `Fprintf` 根据格式指示符 `format` 进行格式化，将格式化结果写入 `w`，返回写入的字节数量以及遇到的任意写入错误

#### func Printf

```go
func Printf(format string, a ...any) (n int, err error)
```

`Printf` formats according to a format specifier and writes to standard output. It returns the number of bytes written and any write error encountered.

>  `Printf` 根据 `format` 格式化，将格式化结果写入标准输出


