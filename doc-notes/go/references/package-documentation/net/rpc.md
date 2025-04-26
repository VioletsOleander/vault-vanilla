---
version: 1.24.2
---
### Overview 
Package `rpc` provides access to the exported methods of an object across a network or other I/O connection. A server registers an object, making it visible as a service with the name of the type of the object. After registration, exported methods of the object will be accessible remotely. A server may register multiple objects (services) of different types but it is an error to register multiple objects of the same type.
>  `rpc` 提供了通过网络或其他 IO 连接访问一个对象的导出方法的功能
>  server 注册一个对象，**将该对象作为服务可见**，其类型就是服务名称，注册后，该对象的导出方法可以通过远程访问
>  server 可以注册多个不同类型的对象/服务，但注册多个相同类型的对象是错误的

Only methods that satisfy these criteria will be made available for remote access; other methods will be ignored:

- the method's type is exported.
- the method is exported.
- the method has two arguments, both exported (or builtin) types.
- the method's second argument is a pointer.
- the method has return type error.

>  可以被远程访问的方法需要满足以下条件
>  - 方法的类型被导出
>  - 方法被导出
>  - 方法有两个参数，参数类型都是导出的 (或是内建的)
>  - 方法的第二个参数是指针
>  - 方法的返回类型是 `error`

In effect, the method must look schematically like

```go
func (t *T) MethodName(argType T1, replyType *T2) error
```

where T1 and T2 can be marshaled by encoding/gob. These requirements apply even if a different codec is used. (In the future, these requirements may soften for custom codecs.)

>  方法的定义模板如上
>  其中类型 `T1, T2` 必须是可以被 encoding/gob 的类型，即使使用了不同的编码器，这些要求仍然适用

The method's first argument represents the arguments provided by the caller; the second argument represents the result parameters to be returned to the caller. The method's return value, if non-nil, is passed back as a string that the client sees as if created by [errors.New](https://pkg.go.dev/errors#New). If an error is returned, the reply parameter will not be sent back to the client.
>  方法的第一个参数代表 caller 提供的参数，第二个参数表示应该返回给 caller 的参数
>  方法的返回值 (如果非 `nil`) 会以字符串形式传递，client 回将其视为由 ` errors.New ` 创建的错误，如果方法返回了错误，` reply ` 参数不会发送回客户端

The server may handle requests on a single connection by calling [ServeConn](https://pkg.go.dev/net/rpc@go1.24.2#ServeConn). More typically it will create a network listener and call [Accept](https://pkg.go.dev/net/rpc@go1.24.2#Accept) or, for an HTTP listener, [HandleHTTP](https://pkg.go.dev/net/rpc@go1.24.2#HandleHTTP) and [http.Serve](https://pkg.go.dev/net/http#Serve).
>  server 可以调用 `ServeConn` 处理单个链接上的请求
>  更一般地，server 会创建一个网络监听器，调用 `Accept` ，对于 HTTP 监听器，就是调用 `HandleHTTP` 和 `http.Serve`

A client wishing to use the service establishes a connection and then invokes [NewClient](https://pkg.go.dev/net/rpc@go1.24.2#NewClient) on the connection. The convenience function [Dial](https://pkg.go.dev/net/rpc@go1.24.2#Dial) ([DialHTTP](https://pkg.go.dev/net/rpc@go1.24.2#DialHTTP)) performs both steps for a raw network connection (an HTTP connection). The resulting [Client](https://pkg.go.dev/net/rpc@go1.24.2#Client) object has two methods, [Call](https://pkg.go.dev/net/rpc@go1.24.2#Call) and Go, that specify the service and method to call, a pointer containing the arguments, and a pointer to receive the result parameters.
>  client 要使用服务，需要建立链接，然后调用链接上的 `NewClient`
>  函数 `Dial` (`DialHTTP`) 会为原始网络链接 (HTTP 链接) 执行这两步
>  `NewClient` 返回的 `Client` 对象有两个方法: `Call, Go`，调用这些方法时，需要提供要调用的服务和方法、包含参数的指针、接收结果参数的指针

The Call method waits for the remote call to complete while the Go method launches the call asynchronously and signals completion using the Call structure's Done channel.
>  `Call` 方法阻塞等待 RPC 完成，`Go` 方法异步启动，在 `Call` 结构中的 `Done` 通道中发送完成信号

Unless an explicit codec is set up, package [encoding/gob](https://pkg.go.dev/encoding/gob) is used to transport the data.
>  默认使用包 `encoding/gob` 进行数据传输

Here is a simple example. A server wishes to export an object of type Arith:

```go
package server

import "errors"

type Args struct {
	A, B int
}

type Quotient struct {
	Quo, Rem int
}

type Arith int

func (t *Arith) Multiply(args *Args, reply *int) error {
	*reply = args.A * args.B
	return nil
}

func (t *Arith) Divide(args *Args, quo *Quotient) error {
	if args.B == 0 {
		return errors.New("divide by zero")
	}
	quo.Quo = args.A / args.B
	quo.Rem = args.A % args.B
	return nil
}
```

The server calls (for HTTP service):

```go
arith := new(Arith)
rpc.Register(arith)
rpc.HandleHTTP()
l, err := net.Listen("tcp", ":1234")
if err != nil {
	log.Fatal("listen error:", err)
}
go http.Serve(l, nil)
```

At this point, clients can see a service "`Arith`" with methods "`Arith.Multiply`" and "`Arith.Divide`". To invoke one, a client first dials the server:

```go
client, err := rpc.DialHTTP("tcp", serverAddress + ":1234")
if err != nil {
	log.Fatal("dialing:", err)
}
```

Then it can make a remote call:

```go
// Synchronous call
args := &server.Args{7,8}
var reply int
err = client.Call("Arith.Multiply", args, &reply)
if err != nil {
	log.Fatal("arith error:", err)
}
fmt.Printf("Arith: %d*%d=%d", args.A, args.B, reply)
```

or

```go
// Asynchronous call
quotient := new(Quotient)
divCall := client.Go("Arith.Divide", args, quotient, nil)
replyCall := <-divCall.Done	// will be equal to divCall
// check errors, print, etc.
```

A server implementation will often provide a simple, type-safe wrapper for the client.

The `net/rpc` package is frozen and is not accepting new features.