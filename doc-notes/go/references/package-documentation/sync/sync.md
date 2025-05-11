---
version: 1.24.3
---
### Overview 
Package sync provides basic synchronization primitives such as mutual exclusion locks. 
>  `sync` 提供了基本的同步原语，例如互斥锁

Other than the [Once](https://pkg.go.dev/sync@go1.24.3#Once) and [WaitGroup](https://pkg.go.dev/sync@go1.24.3#WaitGroup) types, most are intended for use by low-level library routines. Higher-level synchronization is better done via channels and communication.
>  除了 `Once` 和 `WaitGroup` 类型外，`sync` 中的大多数类型旨在用于低级库例程，高级的同步语义最好通过通道和通讯实现

Values containing the types defined in this package should not be copied.
>  包含了 `sync` 中定义的类型的值不应该被复制 (应该通过指针传递)

### Types 
#### `type Mutex`

```go
type Mutex struct {
	// contains filtered or unexported fields
}
```

A Mutex is a mutual exclusion lock. The zero value for a Mutex is an unlocked mutex.
>  `Mutex` 类型表示互斥锁，`Mutex` 类型的零值表示解开的互斥锁

A Mutex must not be copied after first use.
>  一旦 `Mutex` 被使用了，就不应该被复制
>  因为互斥锁的状态是动态的，依赖于当前线程的操作，如果执行复制，互斥锁的副本之间的状态不一定会是一致的，因此，复制得到的锁也不应该被使用

In the terminology of [the Go memory model](https://go.dev/ref/mem), the n'th call to [Mutex.Unlock](https://pkg.go.dev/sync@go1.24.3#Mutex.Unlock) “synchronizes before” the m'th call to [Mutex.Lock](https://pkg.go.dev/sync@go1.24.3#Mutex.Lock) for any n < m. A successful call to [Mutex.TryLock](https://pkg.go.dev/sync@go1.24.3#Mutex.TryLock) is equivalent to a call to Lock. A failed call to `TryLock` does not establish any “synchronizes before” relation at all.
>  以 Go 的内存模型的术语来说，对于任意的 $n < m$，第 $n$ 次对 `Mutex.Unlock` 的调用将 synchronize before 第 $m$ 次对 `Mutex.Lock` 的调用
>  如果一个操作 A synchronize before 另一个操作 B，意味着 A 的执行结果对于 B 是可见的
>  对 `Mutex.TryLock` 的一次成功调用等价于调用 `Mutex.Lock` ，即成功获取锁并返回，(因此会建立 synchronize before 关系)；对 `Mutex.TryLock` 的一次失败调用不会建立 synchronize before 关系

##### `func (*Mutex) Lock`

```go
func (m *Mutex) Lock()
```

`Lock` locks m. If the lock is already in use, the calling goroutine blocks until the mutex is available.

>  `Lock` 方法锁住 `Mutex`，如果 `Mutex` 已经被锁住，则调用 `Lock` 方法的 goroutine 将阻塞，直到 `Mutex` 被释放并且可以被它上锁 

##### `func (*Mutex) TryLock`
added in go1.18

```go
func (m *Mutex) TeyLock()
```

`TryLock` tries to lock m and reports whether it succeeded.

Note that while correct uses of `TryLock` do exist, they are rare, and use of `TryLock` is often a sign of a deeper problem in a particular use of mutexes.

>  `TryLock` 尝试锁住 `Mutex` ，并报告是否成功

##### `func (*Mutex) UnLock`

```go
func (m *Mutex) UnLock
```

Unlock unlocks m. It is a run-time error if m is not locked on entry to Unlock.

A locked [Mutex](https://pkg.go.dev/sync@go1.24.3#Mutex) is not associated with a particular goroutine. It is allowed for one goroutine to lock a Mutex and then arrange for another goroutine to unlock it.

>  `Unlock` 解锁 `Mutex`，如果 `Mutex` 已经解锁，则触发 runtime error
>  锁住的 `Mutex` 并不关联特定的 goroutine，可以由一个 goroutine 上锁，另一个 goroutine 解锁

