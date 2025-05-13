# Lab guidance
## Hardness of assignments
Each lab task is tagged to indicate roughly how long we expect the task to take:

- Easy: A few hours.
- Moderate: ~ 6 hours (per week).
- Hard: More than 6 hours (per week). If you start late, your solution is unlikely to pass all tests.

Most of the labs require only a modest amount of code (perhaps a few hundred lines per lab part), but can be conceptually difficult and may require a good deal of thought and debugging. Some of the tests are difficult to pass.

Don't start a lab the night before it is due; it's more efficient to do the labs in several sessions spread over multiple days. Tracking down bugs in distributed systems is difficult, because of concurrency, crashes, and an unreliable network.

## Tips

- Do the [Online Go tutorial](http://tour.golang.org/) and consult [Effective Go](https://golang.org/doc/effective_go.html). See [Editors](https://golang.org/doc/editors.html) to set up your editor for Go.
- Use Go's [race detector](https://blog.golang.org/race-detector), with `go test -race`. Fix any races it reports.
- Read this [guide](https://thesquareplanet.com/blog/students-guide-to-raft/) for Raft specific advice.
- Advice on [locking](http://nil.csail.mit.edu/6.5840/2023/labs/raft-locking.txt) in labs.
- Advice on [structuring](http://nil.csail.mit.edu/6.5840/2023/labs/raft-structure.txt) your Raft lab.
- This [Diagram of Raft interactions](http://nil.csail.mit.edu/6.5840/2023/notes/raft_diagram.pdf) may help you understand code flow between different parts of the system.
- It may be helpful when debugging to insert print statements when a peer sends or receives a message, and collect the output in a file with `go test > out`. Then, by studying the trace of messages in the out file, you can identify where your implementation deviates from the desired behavior.
- Structure your debug messages in a consistent format so that you can use grep to search for specific lines in out.
- You might find `DPrintf` useful instead of calling `log.Printf` directly to turn printing on and off as you debug different problems.
- Learn about Go's `Printf` format strings: [Go format strings](https://golang.org/pkg/fmt/).
- You can use colors or columns to help you parse log output. [This post](https://blog.josejg.com/debugging-pretty/) explains one strategy.
- To learn more about git, look at the [Pro Git book](https://git-scm.com/book/en/v2) or the [git user's manual](http://www.kernel.org/pub/software/scm/git/docs/user-manual.html).

### Debugging
Efficient debugging takes experience. It helps to be systematic: form a hypothesis about a possible cause of the problem; collect evidence that might be relevant; think about the information you've gathered; repeat as needed. For extended debugging sessions it helps to keep notes, both to accumulate evidence and to remind yourself why you've discarded specific earlier hypotheses.
>  系统性的 debug: 对问题的可能原因形成假设; 收集相关的证据; 思考其中的信息; 如有必要，重复上述步骤

One approach is to progressively narrow down the specific point in time at which things start to go wrong. You could add code at various points in the execution that tests whether the system has reached the bad state. 
>  一种方法是逐渐缩小到开始出现问题的具体时间点，可以在执行过程中的不同位置添加代码，以测试系统是否进入不良状态

Or your code could print messages with relevant state at various points; collect the output in a file, and look through the file for the first point where things look wrong.

The Raft labs involve events, such as RPCs arriving or timeouts expiring or peers failing, that may occur at times you don't expect, or may be interleaved in unexpected orders. For example, one peer may decide to become a candidate while another peer thinks it is already the leader. 
>  Raft lab 涉及了一些事件，例如 RPC 到达、timeout 到期、节点故障，这些事件可能以意想不到的方式交错，以及在意想不到的时间点出现
>  例如，一个节点决定成为 candidate，同时另一个节点认为它已经是 leader

It's worth thinking through the "what can happen next" possibilities. For example, when your Raft code releases a mutex, the very next thing that happens (before the next line of code is executed!) might be the delivery (and processing) of an RPC request, or a timeout going off. Add Print statements to find out the actual order of events during execution.
>  可以思考某个动作后，“接下来会发生什么” 的可能性
>  例如，释放了一个互斥锁后，紧接着发生的事情 (在下一行代码执行之前) 可能是一个 RPC 请求的到达 (和处理)，或者某个 timeout 触发
>  可以通过 `Print` 找到执行中事件的实际执行顺序

The Raft paper's Figure 2 must be followed fairly exactly. It is easy to miss a condition that Figure 2 says must be checked, or a state change that it says must be made. If you have a bug, re-check that all of your code adheres closely to Figure 2.

As you're writing code (i.e., before you have a bug), it may be worth adding explicit checks for conditions that the code assumes to be true, perhaps using Go's [panic](https://gobyexample.com/panic). Such checks may help detect situations where later code unwittingly violates the assumptions.

If code used to work, but now it doesn't, maybe a change you've recently made is at fault.

The bug is often in the very last place you think to look, so be sure to look even at code you feel certain is correct.

The TAs are happy to help you think about your code during office hours, but you're likely to get the most mileage out of limited office hour time if you've already dug as deep as you can into the situation.