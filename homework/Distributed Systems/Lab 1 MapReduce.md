# Introduction
In this lab you'll build a MapReduce system. You'll implement a worker process that calls application Map and Reduce functions and handles reading and writing files, and a coordinator process that hands out tasks to workers and copes with failed workers. You'll be building something similar to the [MapReduce paper](http://research.google.com/archive/mapreduce-osdi04.pdf). (Note: this lab uses "coordinator" instead of the paper's "master".)
>  要求实现 worker 进程和 coordinator 进程
>  worker 处理 Map, Reduce 调用和文件读写
>  coordinator 处理任务分发和故障处理

# Getting started
You need to [setup Go](http://nil.csail.mit.edu/6.5840/2023/labs/go.html) to do the labs.

Fetch the initial lab software with [git](https://git-scm.com/) (a version control system). To learn more about git, look at the [Pro Git book](https://git-scm.com/book/en/v2) or the [git user's manual](http://www.kernel.org/pub/software/scm/git/docs/user-manual.html).

```
$ git clone git://g.csail.mit.edu/6.5840-golabs-2023 6.5840
$ cd 6.5840
$ ls
Makefile src
$
```

We supply you with a simple sequential mapreduce implementation in `src/main/mrsequential.go`. It runs the maps and reduces one at a time, in a single process. 

We also provide you with a couple of MapReduce applications: word-count in `mrapps/wc.go`, and a text indexer in `mrapps/indexer.go`. You can run word count sequentially as follows:

```
$ cd ~/6.5840
$ cd src/main
$ go build -buildmode=plugin ../mrapps/wc.go
$ rm mr-out*
$ go run mrsequential.go wc.so pg*.txt
$ more mr-out-0
A 509
ABOUT 2
ACT 8
...
```

`mrsequential.go` leaves its output in the file `mr-out-0`. The input is from the text files named `pg-xxx.txt`.

Feel free to borrow code from `mrsequential.go`. You should also have a look at `mrapps/wc.go` to see what MapReduce application code looks like.

For this lab and all the others, we might issue updates to the code we provide you. To ensure that you can fetch those updates and easily merge them using git pull, it's best to leave the code we provide in the original files. You can add to the code we provide as directed in the lab write-ups; just don't move it. It's OK to put your own new functions in new files.

# Your Job (moderate/hard)
Your job is to implement a distributed MapReduce, consisting of two programs, the coordinator and the worker. There will be just one coordinator process, and one or more worker processes executing in parallel. 

>  one coordinator/master, multiple parallel workers

In a real system the workers would run on a bunch of different machines, but for this lab you'll run them all on a single machine. The workers will talk to the coordinator via RPC. Each worker process will ask the coordinator for a task, read the task's input from one or more files, execute the task, and write the task's output to one or more files. The coordinator should notice if a worker hasn't completed its task in a reasonable amount of time (for this lab, use ten seconds), and give the same task to a different worker.

>  workers talk with coordinators with RPC
>  worker ask for a task/split, read the file, and write the outputs to one or more files
>  if a worker hasn't complete the task in 10s, the coordinator should give the same task to a different worker

We have given you a little code to start you off. The "main" routines for the coordinator and worker are in `main/mrcoordinator.go` and `main/mrworker.go`; don't change these files. You should put your implementation in `mr/coordinator.go`, `mr/worker.go`, and `mr/rpc.go`.

Here's how to run your code on the word-count MapReduce application. First, make sure the word-count plugin is freshly built:

```
$ go build -buildmode=plugin ../mrapps/wc.go
```

In the main directory, run the coordinator.

```
$ rm mr-out*
$ go run mrcoordinator.go pg-*.txt
```

The `pg-*.txt` arguments to `mrcoordinator.go` are the input files; each file corresponds to one "split", and is the input to one Map task.

>  each `pg-*.txt` file corresponds to one 'split', therefore is the input to one `map` task

In one or more other windows, run some workers:

```
$ go run mrworker.go wc.so
```

When the workers and coordinator have finished, look at the output in `mr-out-*`. When you've completed the lab, the sorted union of the output files should match the sequential output, like this:

```
$ cat mr-out-* | sort | more
A 509
ABOUT 2
ACT 8
...
```

We supply you with a test script in `main/test-mr.sh`. The tests check that the wc and indexer MapReduce applications produce the correct output when given the pg-xxx.txt files as input. The tests also check that your implementation runs the Map and Reduce tasks in parallel, and that your implementation recovers from workers that crash while running tasks.

>  In the implementation, the Map and Reduce tasks should run in parallel, and can recovers from worker crash

If you run the test script now, it will hang because the coordinator never finishes:

```
$ cd ~/6.5840/src/main
$ bash test-mr.sh
*** Starting wc test.
```

You can change `ret := false` to true in the `Done` function in `mr/coordinator.go` so that the coordinator exits immediately. Then:

```
$ bash test-mr.sh
*** Starting wc test.
sort: No such file or directory
cmp: EOF on mr-wc-all
--- wc output is not the same as mr-correct-wc.txt
--- wc test: FAIL
$
```

The test script expects to see output in files named `mr-out-X`, one for each reduce task. The empty implementations of `mr/coordinator.go` and `mr/worker.go` don't produce those files (or do much of anything else), so the test fails.

>  test script expects to see output files named `mr-out-X`, one for each reduce task

When you've finished, the test script output should look like this:

```
$ bash test-mr.sh
*** Starting wc test.
--- wc test: PASS
*** Starting indexer test.
--- indexer test: PASS
*** Starting map parallelism test.
--- map parallelism test: PASS
*** Starting reduce parallelism test.
--- reduce parallelism test: PASS
*** Starting job count test.
--- job count test: PASS
*** Starting early exit test.
--- early exit test: PASS
*** Starting crash test.
--- crash test: PASS
*** PASSED ALL TESTS
$
```

You may see some errors from the Go RPC package that look like

```
2019/12/16 13:27:09 rpc.Register: method "Done" has 1 input parameters; needs exactly three
```

Ignore these messages; registering the coordinator as an [RPC server](https://golang.org/src/net/rpc/server.go) checks if **all its methods** are suitable for RPCs (have 3 inputs); we know that `Done` is **not called via RPC**.

### A few rules:
- The map phase should divide the intermediate keys into buckets for `nReduce` reduce tasks, where `nReduce` is the number of reduce tasks -- the argument that `main/mrcoordinator.go` passes to `MakeCoordinator()`. Each mapper should create `nReduce` intermediate files for consumption by the reduce tasks.
>  map 阶段应该将 intermediate keys 划分到 `nReduce` 个 buckets
>  `nReduce` 是 reduce task 的数量，该参数会由 `main/mrcoordinator.go` 传递给 `MakeCoordinator()`
>  故每个 mapper 都应该创建 `nReduce` 个中间文件

- The worker implementation should put the output of the X'th reduce task in the file `mr-out-X`.
>  第 X 个 reduce task 的输出应该写入 `mr-out-X` 文件

- A `mr-out-X` file should contain one line per Reduce function output. The line should be generated with the Go `"%v %v"` format, called with the key and value. Have a look in `main/mrsequential.go` for the line commented "this is the correct format". The test script will fail if your implementation deviates too much from this format.
>  每次 `Reduce` 函数的输出都占据 `mr-out-X` 的一行，格式为 `%v %v`

- You can modify `mr/worker.go`, `mr/coordinator.go`, and `mr/rpc.go`. You can temporarily modify other files for testing, but make sure your code works with the original versions; we'll test with the original versions.
>  将实现写在 `mr/worker.go, mr/coordinator.go, mr/rpc.go`

- The worker should put intermediate Map output in files in the current directory, where your worker can later read them as input to Reduce tasks.
>  intermediate Map 输出应该放在当前目录，便于 Reduce 任务读取

- `main/mrcoordinator.go` expects `mr/coordinator.go` to implement a `Done()` method that returns true when the MapReduce job is completely finished; at that point, `mrcoordinator.go` will exit.
>  `mr/coordinator.go` 应该实现 `Done()` 函数，它在任务完成时返回 `true`

- When the job is completely finished, the worker processes should exit. A simple way to implement this is to use the return value from `call()`: if the worker fails to contact the coordinator, it can assume that the coordinator has exited because the job is done, so the worker can terminate too. Depending on your design, you might also find it helpful to have a "please exit" pseudo-task that the coordinator can give to workers.
>  如果 worker 无法联系 coordinator，就认为 coordinator 已经完成了任务并退出，故 worker 也可以结束

### Hints
- The [Guidance page](http://nil.csail.mit.edu/6.5840/2023/labs/guidance.html) has some tips on developing and debugging.

- One way to get started is to modify `mr/worker.go` 's `Worker()` to send an RPC to the coordinator asking for a task. Then modify the coordinator to respond with the file name of an as-yet-unstarted map task. Then modify the worker to read that file and call the application Map function, as in ` mrsequential.go `.
>  先修改 `mr/worker.go` 中的 `Worker()` ，向 coordinator 发送 RPC，请求任务
>  再修改 `mr/coordinator.go` ，回应尚未启动的 map 任务的文件名
>  再修改 `mr/worker.go` ，读取文件，调用 Map 函数

- The application Map and Reduce functions are loaded at run-time using the Go plugin package, from files whose names end in .so.
>  应用端的 Map, Reduce 函数会在运行时使用 `plugin` 包加载，故会编译为 `.so`

- If you change anything in the `mr/directory`, you will probably have to re-build any MapReduce plugins you use, with something like `go build -buildmode=plugin ../mrapps/wc.go`
>  如果修改了 `mr/` 下的文件，一般需要重新编译 `mrapps/` 下的文件

- This lab relies on the workers sharing a file system. That's straightforward when all workers run on the same machine, but would require a global filesystem like GFS if the workers ran on different machines.

- A reasonable naming convention for intermediate files is `mr-X-Y`, where X is the Map task number, and Y is the reduce task number.
>  intermediate 文件可以命名为 `mr-X-Y`，其中 X 为 Map task number, Y 为 Reduce task number

- The worker's map task code will need a way to store intermediate key/value pairs in files in a way that can be correctly read back during reduce tasks. One possibility is to use Go's `encoding/json` package. To write key/value pairs in JSON format to an open file:

    ```go
      enc := json.NewEncoder(file)
      for _, kv := ... {
        err := enc.Encode(&kv)
    ```
    
    and to read such a file back:
    
    ```go
      dec := json.NewDecoder(file)
      for {
        var kv KeyValue
        if err := dec.Decode(&kv); err != nil {
          break
        }
        kva = append(kva, kv)
      }
    ```

>  Intermediate file 可以是 JSON 格式

- The map part of your worker can use the `ihash(key)` function (in `worker.go`) to pick the reduce task for a given key.
>  可以使用 `ihash(key)` 函数 (在 `worker.go` 中) 计算 key 的 bucket

- You can steal some code from `mrsequential.go` for reading Map input files, for sorting intermediate key/value pairs between the Map and Reduce, and for storing Reduce output in files.
>  读取 Map 输入文件，排序中间键值对，排序 Reduce 输出文件的代码可以参考 `mrsequential.go`

- The coordinator, as an RPC server, will be concurrent; don't forget to lock shared data.
>  coordinator 应该是并发的，注意对共享数据上锁

- Use Go's race detector, with `go run -race`. `test-mr.sh` has a comment at the start that tells you how to run it with -race. When we grade your labs, we will **not** use the race detector. Nevertheless, if your code has races, there's a good chance it will fail when we test it even without the race detector.

- Workers will sometimes need to wait, e.g. reduces can't start until the last map has finished. One possibility is for workers to periodically ask the coordinator for work, sleeping with `time.Sleep()` between each request. Another possibility is for the relevant RPC handler in the coordinator to have a loop that waits, either with `time.Sleep()` or `sync.Cond`. Go runs the handler for each RPC in its own thread, so the fact that one handler is waiting needn't prevent the coordinator from processing other RPCs.
>  worker 可以周期性 (`time.Sleep()`) 向 coordinator 请求任务，或者让 coordinator 的 RPC handler 循环等待
>  Go 在不同的 thread 运行不同的 RPC handler，故当前 handler 在等待某个 RPC 不会阻碍其他 RPC 的处理

- The coordinator can't reliably distinguish between crashed workers, workers that are alive but have stalled for some reason, and workers that are executing but too slowly to be useful. The best you can do is have the coordinator wait for some amount of time, and then give up and re-issue the task to a different worker. For this lab, have the coordinator wait for ten seconds; after that the coordinator should assume the worker has died (of course, it might not have).
>  coordinator 无法正确区分 worker 是崩溃还是停机还是执行太慢，故最好让 coordinator 等待一定时间 (e.g. 10s)，然后向新 worker 重新分发任务

- If you choose to implement Backup Tasks (Section 3.6), note that we test that your code doesn't schedule extraneous tasks when workers execute tasks without crashing. Backup tasks should only be scheduled after some relatively long period of time (e.g., 10s).

- To test crash recovery, you can use the `mrapps/crash.go` application plugin. It randomly exits in the Map and Reduce functions.
>  `mrapps/crash.go` 用于进行崩溃测试

- To ensure that nobody observes partially written files in the presence of crashes, the MapReduce paper mentions the trick of using a temporary file and atomically renaming it once it is completely written. You can use `ioutil.TempFile` to create a temporary file and `os.Rename` to atomically rename it.
>  为了避免崩溃时，不会由其他进程看到部分写入的文件，一个技巧是先向一个暂时文件写入，然后在完成写入后，原子化地将其重命名
>  `ioutil.TempFile` 用于创建暂时文件，`os.Rename` 用于原子化重命名

- `test-mr.sh` runs all its processes in the sub-directory ` mr-tmp `, so if something goes wrong and you want to look at intermediate or output files, look there. Feel free to temporarily modify `test-mr.sh` to exit after the failing test, so the script does not continue testing (and overwrite the output files).
>  `test-mr.sh` 在 `mr-tmp` 运行所有进程，中间文件都会在该目录

- `test-mr-many.sh` runs `test-mr.sh` many times in a row, which you may want to do in order to spot low-probability bugs. It takes as an argument the number of times to run the tests. You should not run several ` test-mr.sh ` instances in parallel because the coordinator will reuse the same socket, causing conflicts.
>  `test-mr-manay.sh` 用于多次运行 `test-mr.sh`

- Go RPC sends only struct fields whose names start with capital letters. Sub-structures must also have capitalized field names.
>  Go RPC 仅会发送名称为大写开头的 struct fields, sub-structures 也必须有大写开头的名称

- When calling the RPC `call()` function, the reply struct should contain all default values. RPC calls should look like this:
    
      reply := SomeType{}
      call(..., &reply)
    
    without setting any fields of reply before the call. If you pass reply structures that have non-default fields, the RPC system may silently return incorrect values.

### No-credit challenge exercises
Implement your own MapReduce application (see examples in `mrapps/*`), e.g., Distributed Grep (Section 2.3 of the MapReduce paper).

Get your MapReduce coordinator and workers to run on separate machines, as they would in practice. You will need to set up your RPCs to communicate over TCP/IP instead of Unix sockets (see the commented out line in `Coordinator.server()`), and read/write files using a shared file system. For example, you can ssh into multiple [Athena cluster](http://kb.mit.edu/confluence/display/istcontrib/Getting+Started+with+Athena) machines at MIT, which use [AFS](http://kb.mit.edu/confluence/display/istcontrib/AFS+at+MIT+-+An+Introduction) to share files; or you could rent a couple AWS instances and use [S3](https://aws.amazon.com/s3/) for storage.

# Code Analysis
The  `main` folder:

`main/mrcoordinator.go` 
接收输入文件，调用 `mr/coordinator.go` 中定义的 `MakeCoordinator()` 函数创建 `Coordinator` 实例 (其类型也定义在 `mr/coordinator.go` 中)，之后循环等待 `Coordinator` 实例的 `Done()` 方法返回 `true` (表示 MapReduce 任务完成)，之后退出
因此其作用是启动一个 Coordinator 进程，并等待 MapReduce 完成

`main/mrworker.go` 
接收 `mrapps/` 中的源文件 (定义了 `Map, Reduce` 函数) 编译出的 `.so ` 文件，从 `.so` 文件中查找到 `Map`, `Reduce` 函数，然后将其作为参数，传递给 `mr/worker.go` 中定义的 `Worker()` 函数，就退出
因此其作用是启动一个 Worker 进程
