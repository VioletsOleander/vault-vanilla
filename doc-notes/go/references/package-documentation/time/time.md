---
version: 1.24.3
---
### Overview 
- [Monotonic Clocks](https://pkg.go.dev/time@go1.24.3#hdr-Monotonic_Clocks)
- [Timer Resolution](https://pkg.go.dev/time@go1.24.3#hdr-Timer_Resolution)

Package time provides functionality for measuring and displaying time.

The calendrical calculations always assume a Gregorian calendar, with no leap seconds.

>  `time` 提供衡量和显示时间的功能
>  日历计算始终假设使用格力高日历，不考虑闰秒

#### `type Time`

```go
type Time struct {
	// contains filtered or unexported fields
}
```

A Time represents an instant in time with nanosecond precision.

>  `Time` 表示纳秒精度的一个时刻

Programs using times should typically store and pass them as values, not pointers. That is, time variables and struct fields should be of type [time.Time](https://pkg.go.dev/time@go1.24.3#Time), not `*time.Time`.
>  时间应该都通过值传递，而不是指针
>  即时间变量都应该是类型 `time.Time` ，而不是 `*time.Time`

The zero value of type Time is January 1, year 1, 00:00:00.000000000 UTC. As this time is unlikely to come up in practice, the [Time.IsZero](https://pkg.go.dev/time@go1.24.3#Time.IsZero) method gives a simple way of detecting a time that has not been initialized explicitly.
>  `Time` 的零值为 January 1, year 1, 00:00:00.000000000 UTC
>  `Time.IsZero()` 方法用于判断是否为零值

##### `func Now`

```go
func Now() Time
```

`Now` returns the current local time.

##### `func Unix()`

```go
func Unix(sec int64, nsec int64) Time
```

Unix returns the local Time corresponding to the given Unix time, sec seconds and nsec nanoseconds since January 1, 1970 UTC. It is valid to pass nsec outside the range [0, 999999999]. Not all sec values have a corresponding time value. One such value is 1<<63-1 (the largest int64 value).
