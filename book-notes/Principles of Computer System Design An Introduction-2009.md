# 9 Atomicity: All-or-Nothing and Before-or-After  
## 9.1 Atomicity
### 9.1.5 Before-or-After Atomicity: Coordinating Concurrent Threads  
In Chapter 5 we learned how to express opportunities for concurrency by creating threads, the goal of concurrency being to improve performance by running several things at the same time. Moreover, Section 9.1.2 above pointed out that interrupts can also cre­ate concurrency. 

Concurrent threads do not represent any special problem until their paths cross. The way that paths cross can always be described in terms of shared, writable data: concurrent threads happen to take an interest in the same piece of writable data at about the same time. It is not even necessary that the concurrent threads be running simultaneously; if one is stalled (perhaps because of an interrupt) in the middle of an action, a different, running thread can take an interest in the data that the stalled thread was, and will sometime again be, working with.  
>  并发线程在出现路径交叉时，就会导致问题
>  导致路径交叉的方式总是可以用共享、可写的数据来描述：并发线程在同一时间对相同的可写数据需要执行操作，甚至不一定要同时
>  如果其中一个执行线程在执行操作的时候被阻塞，例如因为中断，则另一个执行线程可能会对该执行线程正在处理或者将来会继续处理的数据进行修改

From the point of view of the programmer of an application, Chapter 5 introduced two quite different kinds of concurrency coordination requirements: sequence coordina­tion and before-or-after atomicity. Sequence coordination is a constraint of the type “Action W must happen before action X . For correctness, the first action must complete before the second action begins. For example, reading of typed characters from a key­ board must happen before running the program that presents those characters on a display. As a general rule, when writing a program one can anticipate the sequence coor­dination constraints, and the programmer knows the identity of the concurrent actions. Sequence coordination thus is usually explicitly programmed, using either special lan­guage constructs or shared variables such as the event counts of Chapter 5.  
>  从程序开发者的角度，有两种不同的并发协调要求：顺序协调、前后原子性
>  顺序协调即类似 Action W 必须在 Action X 之后发生这样的约束，为了确保正确性，上一 Action 必须在下一个 Action 开始之前完成
>  例如，从键盘上读取输入字符的操作必须在将这些字符显示在屏幕上之前完成
>  一般来说，程序开发者在编写程序时可以预见顺序协调约束，并且知道并发 Actions 是什么，故程序开发者会将顺序协调显示编程，要么使用特殊的语言结构，要么使用共享变量，例如事件计数器等

In contrast, before-or-after atomicity is a more general constraint that several actions that concurrently operate on the same data should not interfere with one another. We define before-or-after atomicity as follows:  
>  相较之下，前后原子性则是更加通用的约束条件，它约束多个并发对同个数据操作的动作不能相互干扰
>  我们将前后原子性定义如下：

**Before-or-after atomicity**  
Concurrent actions have the before-or-after property if their effect from the point of view of their invokers is the same as if the actions occurred either completely before or completely after one another.  
>  Before-or-after atomicity
>  如果并发动作的效果，从其调用者的视角来看，和它们完全发生在彼此之前或者彼此之后的效果相同，则这些并发动作具有 before-or-after 原子性

In Chapter 5 we saw how before-or-after actions can be created with explicit locks and a thread manager that implements the procedures ACQUIRE and RELEASE. Chapter 5 showed some examples of before-or-after actions using locks, and emphasized that programming correct before-or-after actions, for example coordinating a bounded buffer with several producers or several consumers, can be a tricky proposition. To be confident of correct­ness, one needs to establish a compelling argument that every action that touches a shared variable follows the locking protocol.  
>  可以通过显示锁或者一个实现了过程 ACQUIRE 和 RELEASE 的线程管理器来创建满足 before-or-after 原子性的并发动作
>  编写正确的 before-or-after actions (例如协调带有多个消费者和生产者的有界缓冲区) 是一个棘手的问题，为了确保正确性，需要论证每个触及了共享变量的动作都遵循了锁协议

One thing that makes before-or-after atomicity different from sequence coordination is that the programmer of an action that must have the before-or-after property does not necessarily know the identities of all the other actions that might touch the shared variable. This lack of knowledge can make it problematic to coordinate actions by explicit program steps. Instead, what the programmer needs is an automatic, implicit mechanism that ensures proper handling of every shared variable. This chapter will describe several such mechanisms. 
>  before-or-after 原子性和 sequence 协调性的差异在于：编写 before-or-after 操作的程序员并不知道所有其他的操作是否可能会触及共享变量，故难以使用显式的程序步来协调操作
>  因此，程序员需要的是一个自动的且隐式的机制，能够确保每个共享变量都能被正确处理，本章将描述几个这样的机制

Put another way, correct coordination requires discipline in the way concurrent threads read and write shared data.  
>  换句话说，正确的协调能够使得并发线程在读写共享数据时保持纪律性

Applications for before-or-after atomicity in a computer system abound. In an operating system, several concurrent threads may decide to use a shared printer at about the same time. It would not be useful for printed lines of different threads to be interleaved in the printed output. Moreover, it doesn’t really matter which thread gets to use the printer first; the primary consideration is that one use of the printer be complete before the next begins, so the requirement is to give each print job the before-or-after atomicity property.  
>  计算机系统中，before-or-after 原子性的应用有许多
>  例如 OS 中，多个并发线程可能在同一时间决定使用一个共享打印机，此时不能让打印机的占用情况是由多个线程交织的，且我们并不在意哪个线程先使用打印机，主要的要求时在上一个使用打印机的线程完成之后，下一个线程才能开始，因此我们需要给每个打印任务 before-or-after 原子性

For a more detailed example, let us return to the banking application and the TRANSFER procedure. This time the account balances are held in shared memory variables (recall that the declaration keyword `reference` means that the argument is call-by-reference, so that TRANSFER can change the values of those arguments):  

```
procedure TRANSFER (reference debit_account, reference credit_account, amount) 
    debit_account <- debit_account - amount 
    credit_account <- credit_account + amount  
```

>  我们考虑一个更加详细的例子：银行程序的转账过程 (`TRANSFER`)
>  账户存款使用一个共享变量存储

Despite their unitary appearance, a program statement such as $^{\mathfrak{c}}X\leftarrow X+Y^{\mathfrak{s}}$ is actually composite: it involves reading the values of $X$ and ${Y}$ performing an addition, and then writing the result back into $X$ . If a concurrent thread reads and changes the value of $\boldsymbol{x}$ between the read and the write done by this statement, that other thread may be surprised when this statement overwrites its change.  
>  注意，一个程序语句例如 `X <- X + Y` 虽然表面上看是单个语句，但实际上由多个过程复合，它包括了：读取 `X, Y` 的值；执行加法；将值写回 `X`
>  如果在读取和写回过程之间，另一个并发线程读取并改变了 `X` 的值，则该线程就会发现其写入被覆盖了

Suppose this procedure is applied to accounts A (initially containing $\$300$ ) and $B$ (ini­tially containing $\$100$ ) as in  

```
TRANSFER (A,B,$10)  
```

We expect account $A$ , the debit account, to end up with $\$290$ , and account $B$ , the credit account, to end up with $\$110$ . Suppose, however, a second, concurrent thread is executing the statement  

```
TRANSFER (B,C,$25)
```

where account $C$ starts with $\$175$ . When both threads complete their transfers, we expect $B$ to end up with $\$85$ and $C$ with $\$200$ . Further, this expectation should be fulfilled no matter which of the two transfers happens first. 

>  假设有以上两个转账行为发生，我们期望最后的结果是 `B=$85, C=$200, A=$290`
>  无论这两个行为哪个先后发生，结果都应该是一样的

But the variable `credit_account` in the first thread is bound to the same object (account $B$ ) as the variable `debit_account` in the second thread. The risk to correctness occurs if the two transfers happen at about the same time. To understand this risk, consider Figure 9.2, which illustrates several possible time sequences of the READ and WRITE steps of the two threads with respect to variable $B$ .  
>  但是这两个 `TRANSFER` 都涉及对 `B` 绑定的变量 `credit_account` 的修改，故如果这两个并发线程同时发生，就可能出现风险

![[pics/Principles of Computer System Design-Fig9.2.png]]

With each time sequence the figure shows the history of values of the cell containing the balance of account B. If both steps 1–1 and 1–2 precede both steps 2–1 and 2–2, (or vice versa) the two transfers will work as anticipated, and $B$ ends up with $\$85$ . If, however, step 2–1 occurs after step 1–1, but before step 1–2, a mistake will occur: one of the two transfers will not affect account $B$ , even though it should have. The first two cases illustrate histories of shared variable $B$ in which the answers are the correct result; the remaining four cases illustrate four different sequences that lead to two incorrect values for $B$ .  

Six possible histories of variable $B$ if two threads that share $B$ do not coordinate their concurrent activities.  
>  参照 Fig9.2，如果并发线程不协调它们的活动，`B` 将出现多种可能值

Thus our goal is to ensure that one of the first two time sequences actually occurs. One way to achieve this goal is that the two steps 1–1 and 1–2 should be atomic, and the two steps 2–1 and 2–2 should similarly be atomic. In the original program, the steps  

```
debit_account <- debit_account - amount and 
credit_account <- credit_account + amount  
```

should each be atomic. There should be no possibility that a concurrent thread that intends to change the value of the shared variable `debit_account` read its value between the READ and WRITE steps of this statement.  

>  为了确保协调，一种方式就是使得整个 `TRANSFRE` 过程是原子性的

### 9.1.6 Correctness and Serialization  
The notion that the first two sequences of Figure 9.2 are correct and the other four are wrong is based on our understanding of the banking application. 
>  在上一个例子中，我们基于常识 (转账的合理结果) 为并发的操作定义了正确性

It would be better to have a more general concept of correctness that is independent of the application. Application independence is a modularity goal: we want to be able to make an argument for correctness of the mechanism that provides before-or-after atomicity without getting into the question of whether or not the application using the mechanism is correct.  
>  我们希望有一个更通用的正确性概念，它独立于应用
>  应用无关性是一个模块化目标，我们不需要为使用该机制的应用程序论证正确性，而是为提供了 before-or-after 原子性的机制论证正确性

There is such a correctness concept: coordination among concurrent actions can be considered to be correct if every result is guaranteed to be one that could have been obtained by some purely serial application of those same actions.  
>  有这样的正确性概念：如果并发操作在协调后得到的结果保证可以通过对这些操作的某个顺序应用得到，则协调就是正确的

The reasoning behind this concept of correctness involves several steps. Consider Figure 9.3,which shows, abstractly, the effect of applying some action, whether atomic or not, to a system: the action changes the state of the system. Now, if we are sure that:  

![[pics/Principles of Computer System Design-Fig9.3.png]]

1.  the old state of the system was correct from the point of view of the application, and   
2.  the action, performing all by itself, correctly transforms any correct old state to a correct new state,  

then we can reason that the new state must also be correct. This line of reasoning holds for any application-dependent definition of “correct” and “correctly transform”, so our reasoning method is independent of those definitions and thus of the application.  

>  对这一正确性概念的分析涉及多个步骤
>  Fig9.3 抽象地展示了对一个系统应用某个动作的效果，无论动作是不是原子的，它都改变了系统状态，因此，如果我们有
>  1. 从应用的视角看，系统的旧状态是正确的
>  2. 该动作只要完整执行，就正确地将任意正确的旧状态转化为一个正确的新状态
>  故我们可以推断新的状态一定是正确的
>  这种推理方式适用于任何应用程序相关的 “正确” 和 “正确转换” 的定义，故我们的推理方法独立于这些定义的具体内容，故独立于应用

![[pics/Principles of Computer System Design-Fig9.4.png]]

The corresponding requirement when several actions act concurrently, as in Figure 9.4, is that the resulting new state ought to be one of those that would have resulted from some serialization of the several actions, as in Figure 9.5. This correctness criterion means that concurrent actions are correctly coordinated if their result is guaranteed to be one that would have been obtained by some purely serial application of those same actions.  
>  当多个操作并行执行时，对应的要求，如 Fig9.4 所示，是结果状态应该是这些动作的某次串行执行可以得到的
>  这一正确性准则意味着如果并发动作的结果能够保证以纯串行的方式应用这些动作得到的结果相同，则可以认为对这些并发动作的协调是正确的

When several actions act con­currently, they together produce a new state. If the actions are before-or-after and the old state was correct, the new state will be correct.  
>  显然，如果并发的动作都是满足 before-or-after 原子性的，则得到的状态将满足正确性

So long as the only coordination requirement is before-or-after atomicity, any serializa­tion will do.  
>  显然，如果协调要求是 before-or-after 原子性，则任意的串行执行都满足该协调要求

![[pics/Principles of Computer System Design-Fig9.5.png]]

Moreover, we do not even need to insist that the system actually traverse the interme­diate states along any particular path of Figure 9.5—it may instead follow the dotted trajectory through intermediate states that are not by themselves correct, according to the application’s definition. As long as the intermediate states are not visible above the implementing layer, and the system is guaranteed to end up in one of the acceptable final states, we can declare the coordination to be correct because there exists a trajectory that leads to that state for which a correctness argument could have been applied to every step.  

Since our definition of before-or-after atomicity is that each before-or-after action act as though it ran either completely before or completely after each other before-or-after action, before-or-after atomicity leads directly to this concept of correctness. Put another way, before-or-after atomicity has the effect of serializing the actions, so it follows that before-or-after atomicity guarantees correctness of coordination. 
>  显然，before-or-after 原子性将使得该正确性被满足
>  换句话说，before-or-after 原子性的效果等价于序列化并发动作

A different way of expressing this idea is to say that when concurrent actions have the before-or-after prop­erty, they are serializable: there exists some serial order of those concurrent transactions that would, if followed, lead to the same ending state.\* Thus in Figure 9.2, the sequences of case 1 and case 2 could result from a serialized order, but the actions of cases 3 through 6 could not.  
>  或者说，满足 before-or-after 性质的并发动作是可序列化的：存在某个对这些动作的序列化顺序使得可以获得当前结果

We insist that the final state be one that could have been reached by some serialization of the atomic actions, but we don't care which serialization. In addition, we do not need to insist that the intermediate states ever actually exist. The actual state trajectory could be that shown by the dotted lines, but only if there is no way of observing the intermediate states from the outside.  

In the example of Figure 9.2, there were only two concurrent actions and each of the concurrent actions had only two steps. As the number of concurrent actions and the number of steps in each action grows there will be a rapidly growing number of possible orders in which the individual steps can occur, but only some of those orders will ensure a correct result. Since the purpose of concurrency is to gain performance, one would like to have a way of choosing from the set of correct orders the one correct order that has the highest performance. As one might guess, making that choice can in general be quite difficult. In Sections 9.4 and 9.5 of this chapter we will encounter several programming disciplines that ensure choice from a subset of the possible orders, all members of which are guaranteed to be correct but, unfortunately, may not include the correct order that has the highest performance.  

In some applications it is appropriate to use a correctness requirement that is stronger than serializability. For example, the designer of a banking system may want to avoid anachronisms by requiring what might be called external time consistency: if there is any external evidence (such as a printed receipt) that before-or-after action $T_{1}$ ended before before-or-after action $T_{2}$ began, the serialization order of $T_{1}$ and $T_{2}$ inside the system should be that $T_{1}$ precedes $T_{2}$ . For another example of a stronger correctness require­ment, a processor architect may require sequential consistency: when the processor concurrently performs multiple instructions from the same instruction stream, the result should be as if the instructions were executed in the original order specified by the programmer.  
>  某些应用场景下，可能会使用比可串行化更强的正确性要求
>  例如，银行系统的设计者可能通过要求外部时间一致性来避免年代错误：如果存在任何外部证据，例如打印的收据，证明 before-or-after action $T_1$ 在 before-or-after action $T_2$ 开始之前结束，则系统内部对 $T_1$ 和 $T_2$ 的串行化顺序应为 $T_1$ 先于 $T_2$
>  另一个例子是，处理器架构可能要求顺序一致性：当处理器并发处理来自同一指令流的多个指令时，结果应该等同于与程序员指定的原始顺序执行这些指令得到的结果相同

Returning to our example, a real funds-transfer application typically has several distinct before-or-after atomicity requirements. Consider the following auditing procedure; its purpose is to verify that the sum of the balances of all accounts is zero (in double-entry bookkeeping, accounts belonging to the bank, such as the amount of cash in the vault, have negative balances):  

```
procedure AUDIT() 
    sum <- 0 
    for each W <- in bank.accounts 
        sum <- sum + W.balance 
    if (sum!=0) call for investigation  
```

Suppose that AUDIT is running in one thread at the same time that another thread is transferring money from account A to account $B$ . If AUDIT examines account A before the transfer and account $B$ after the transfer, it will count the transferred amount twice and thus will compute an incorrect answer. So the entire auditing procedure should occur either before or after any individual transfer: we want it to be a before-or-after action.  

There is yet another before-or-after atomicity requirement: if AUDIT should run after the statement in TRANSFER  

```
debit_account <- debit_account - amount 
```

but before the statement 

```
credit_account <- credit_account + amount  
```

it will calculate a sum that does not include amount; we therefore conclude that the two balance updates should occur either completely before or completely after any AUDIT action; put another way, TRANSFER should be a before-or-after action.  

## 9.5 Before-or-After Atomicity II: Pragmatics  
The previous section showed that a version history system that provides all-or-nothing atomicity can be extended to also provide before-or-after atomicity. When the all-or­-nothing atomicity design uses a log and installs data updates in cell storage, other, con­current actions can again immediately see those updates, so we again need a scheme to provide before-or-after atomicity. When a system uses logs for all-or-nothing atomicity, it usually adopts the mechanism introduced in Chapter 5—locks—for before-or-after atomicity. However, as Chapter 5 pointed out, programming with locks is hazardous, and the traditional programming technique of debugging until the answers seem to be correct is unlikely to catch all locking errors. We now revisit locks, this time with the goal of using them in stylized ways that allow us to develop arguments that the locks correctly implement before-or-after atomicity.  
>  上一节展示了提供 all-or-nothing 原子性的版本历史系统可以被扩展以同时提供 before-or-after 的原子性
>  当 all-or-nothing 原子性设计使用日志并将数据更新安装到单元存储中时，其他并发操作可以再次立即看到这些更新，因此我们仍然需要一种方案来提供 before-or-after 原子性
>  当系统使用日志来实现 all-or-nothing 原子性时，它通常采用第5章介绍的机制——锁——来实现 before-or-after 原子性，然而，正如第5章所指出的，使用锁进行编程存在风险，并且传统的调试技术（直到结果看起来正确为止）不太可能捕捉到所有锁错误
>  我们现在重新审视锁，这次的目标是以一种规范的方式使用它们，从而能够开发出论证锁正确实现 all-or-nothing 原子性的方法。

### 9.5.2 Simple Locking  
The second locking discipline, known as simple locking, is similar in spirit to, though not quite identical with, the mark-point discipline. 
>  第二种锁机制，称为 simple locking，在思想上与 mark-point 机制相似，但并不完全相同

The simple locking discipline has two rules. First, each transaction must acquire a lock for every shared data object it intends to read or write before doing any actual reading and writing. Second, it may release its locks only after the transaction installs its last update and commits or completely restores the data and aborts. 
>  simple locking 机制包含两条规则: 首先，每个事务在实际读取或写入任何共享数据对象之前，必须为它打算读取或写入的**每一个**共享数据对象获取锁；其次，只有在事务安装了其最后一次更新并提交，或者完全恢复数据并中止之后 (all-or-nothing)，才能释放其锁

Analogous to the mark point, the transaction has what is called a lock point: the first instant at which it has acquired all of its locks. The collection of locks it has acquired when it reaches its lock point is called its lock set.  
>  类似于 mark point 的概念，事务有一个所谓的 lock point：即它首次获得所有所需锁的时间点
>  当事务达到 lock point 时，它所获取的锁集合被称为 lock set

A lock manager can enforce simple locking by requiring that each transaction supply its intended lock set as an argument to the `begin_transaction` operation, which acquires all of the locks of the lock set, if necessary waiting for them to become available. The lock manager can also interpose itself on all calls to read data and to log changes, to verify that they refer to variables that are in the lock set. The lock manager also intercepts the call to commit or abort (or, if the application uses roll-forward recovery, to log an END record) at which time it automatically releases all of the locks of the lock set. 
>  锁管理器可以通过要求每个事务在调用 `begin_transaction` 操作时提供其预期的 lock set 作为参数来执行 simple locking，如果必要，锁管理器会等待这些锁可用，再获取 lock set 中的所有锁
>  锁管理器还可以拦截对读取数据和记录更改的所有调用，以验证它们是否引用了 lock set 中包含的变量 (是否持有锁)
>  此外，在事务调用提交或中止时（如果应用程序使用向前恢复机制，则是记录 END 记录），锁管理器会自动释放 lock set 中所有的锁

The simple locking discipline correctly coordinates concurrent transactions. We can make that claim using a line of argument analogous to the one used for correctness of the mark-point discipline. Imagine that an all-seeing outside observer maintains an ordered list to which it adds each transaction identifier as soon as the transaction reaches its lock point and removes it from the list when it begins to release its locks. 
>  simple locking 机制正确地协调了并发事务
>  我们可以用与 mark-point discipline 正确性论证类似的方式来进行论证，想象一下，一个无所不知的外部观察者维护着一个有序列表，它在每个事务到达其 lock point 时立即在列表中添加该事务的标识符，并在其开始释放锁时从列表中移除它

Under the simple locking discipline each transaction has agreed not to read or write anything until that transaction has been added to the observer’s list. We also know that all transactions that precede this one in the list must have already passed their lock point. Since no data object can appear in the lock sets of two transactions, no data object in any transaction’s lock set appears in the lock set of the transaction preceding it in the list, and by induction to any transaction earlier in the list. Thus all of this transaction’s input values are the same as they will be when the preceding transaction in the list commits or aborts. 
>  在 simple locking 机制下，每个事务在被添加到观察者的列表之前不读取或写入任何内容
>  我们还知道，在列表中的所有先前事务必须已经通过了它们的 lock point，由于没有数据对象同时可以出现在两个事务的 lock set 中，因此任何事务的 lock set 中的数据对象不会出现在列表中其前面的事务的 lock set 中，并且通过归纳法，也不会出现在列表中更早的任何事务的 lock set 中
>  因此，这个事务的所有输入值与当列表中前面的事务提交或终止时的输入值相同 (因为之前的事务和当前事务的输入值不相关，它们不对当前事务需要的对象进行读写)

The same argument applies to the transaction before the preceding one, so all inputs to any transaction are identical to the inputs that would be available if all the transactions ahead of it in the list ran serially, in the order of the list. Thus the simple locking discipline ensures that this transaction runs completely after the preceding one and completely before the next one. Concurrent transactions will produce results as if they had been serialized in the order that they reached their lock points.  
>  同样的论点也适用于前面的事务，所以任何事务的所有输入都与其在列表中顺序运行时可用的输入相同
>  因此，simple locking 机制确保此事务完全在前一个事务之后运行，并完全在下一个事务之前运行
>  并发事务产生的结果，就好像它们按照达到 lock point 的顺序进行了串行化一样

As with the mark-point discipline, simple locking can miss some opportunities for concurrency. In addition, the simple locking discipline creates a problem that can be significant in some applications. Because it requires the transaction to acquire a lock on every shared object that it will either read or write (recall that the mark-point discipline requires marking only of shared objects that the transaction will write), applications that discover which objects need to be read by reading other shared data objects have no alternative but to lock every object that they might need to read. To the extent that the set of objects that an application might need to read is larger than the set for which it eventually does read, the simple locking discipline can interfere with opportunities for concurrency. On the other hand, when the transaction is straightforward (such as the TRANSFER trans­ action of Figure 9.16, which needs to lock only two records, both of which are known at the outset) simple locking can be effective.  
>  与 mark-point 约束一样，simple locking 机制可能会错失一些并发的机会，此外，simple locking 机制在某些应用中会引发一个相当重要的问题
>  由于它要求事务对所有它将读取或写入的共享对象都获取锁（回想一下，mark-point 约束只要求标记事务将要写入的共享对象），那些通过读取其他共享数据对象来发现需要读取哪些对象的应用程序，除了锁定它们可能需要读取的所有对象外别无选择
>  在应用程序可能需要读取的对象集合大于其最终实际读取的对象集合的程度上，simple locking 机制可能会妨碍并发的机会。然而，当事务较为简单（例如图 9.16 中的 TRANSFER 事务，只需要锁定两个记录，且这两个记录在开始时就已知）时，simple locking 机制可以是有效的。

### 9.5.3 Two-Phase Locking  
The third locking discipline, called two-phase locking, like the read-capture discipline, avoids the requirement that a transaction must know in advance which locks to acquire. Two-phase locking is widely used, but it is harder to argue that it is correct. The two-phase locking discipline allows a transaction to acquire locks as it proceeds, and the transaction may read or write a data object as soon as it acquires a lock on that object. 
>  第三种锁定机制称为 two-phase locking，与 read-capture 机制类似，它避免了事务必须提前知道需要获取哪些锁的要求
>  two-phase locking 被广泛使用，但很难证明它是正确的，两阶段锁定机制允许事务在执行过程中逐步获取锁，并且一旦对某个对象获取了锁，事务就可以读取或写入该对象

The primary constraint is that the transaction may not release any locks until it passes its lock point. Further, the transaction can release a lock on an object that it only reads any time after it reaches its lock point if it will never need to read that object again, even to abort. The name of the discipline comes about because the number of locks acquired by a transaction monotonically increases up to the lock point (the first phase), after which it monotonically decreases (the second phase). Just as with simple locking, two-phase locking orders concurrent transactions so that they produce results as if they had been serialized in the order they reach their lock points. 
>  两阶段锁主要的约束条件是: 事务在通过其 lock point 之前不能释放任何锁；并且，事务只有在确定自己以后 (到事务终止前) 不会再需要读取某个它读取的对象时，它才可以在达到 lock point 之后释放对该对象的锁
>  这种锁定机制之所以被称为“两阶段”，是因为事务获取的锁数量在达到 lock point 之前(第一阶段) 单调递增，而在达到锁点之后 (第二阶段) 单调递减
>  与 simple locking 一样，two-phase locking 会按照事务到达 lock point 的顺序对并发事务进行排序，使得它们产生的结果如同这些事务按此顺序被串行化了一样

A lock manager can implement two-phase locking by intercepting all calls to read and write data; it acquires a lock (perhaps having to wait) on the first use of each shared variable. As with simple locking, it then holds the locks until it intercepts the call to commit, abort, or log the END record of the transaction, at which time it releases them all at once.  
>  锁管理器可以通过拦截所有对数据的读取和写入调用来实现 two-phase locking
>  它会在每次使用共享变量时获取锁 (可能需要等待)，与 simple locking 相同，它会一直持有这些锁，直到拦截到提交、中止或记录事务结束的调用，此时它会一次性释放所有锁

The extra flexibility of two-phase locking makes it harder to argue that it guarantees before-or-after atomicity. Informally, once a transaction has acquired a lock on a data object, the value of that object is the same as it will be when the transaction reaches its lock point, so reading that value now must yield the same result as waiting till then to read it. Furthermore, releasing a lock on an object that it hasn’t modified must be harmless if this transaction will never look at the object again, even to abort.
>  two-phase locking 的额外灵活性使得论证其保证了 before-or-after 原子性更加困难
>  非正式地说，当一个事务对某个数据对象获取了锁，该对象的值将在该事务到达其 lock point 之前都保持不变，因此现在读取它的值与之后读取它的值应该获得相同的结果
>  此外，如果事务永远不会再查看该对象，那么释放一个它还未修改的对象的锁一定是无害的

 A formal argument that two-phase locking leads to correct before-or-after atomicity can be found in most advanced texts on concurrency control and transactions. See, for example, Trans­ action Processing, by Gray and Reuter [Suggestions for Further Reading 1.1.5].  
 >  关于 two-phase locking 是如何构成 before-of-after 原子性的正式论证在此处略过

The two-phase locking discipline can potentially allow more concurrency than the simple locking discipline, but it still unnecessarily blocks certain serializable, and therefore correct, action orderings. 
>  相较于 simple locking 机制，two-phase locking 机制理论上可以允许更多的并发性，但它仍然会不必要地阻塞某些可串行化 (因此是正确的) 操作顺序

For example, suppose transaction $\mathrm{T}_{1}$ reads $X$ and writes $Y,$ while transaction $\mathrm{T}_{2}$ just does a (blind) write to Y. Because the lock sets of $\mathrm{T}_{1}$ and $\mathrm{T}_{2}$ intersect at variable Y, the two-phase locking discipline will force transaction $\mathrm{T}_{2}$ to run either completely before or completely after $\mathrm{T}_{1}$ . 
>  例如，假设事务 $T_1$ 读取 $X$，写入 $Y$，事务 $T_2$ 对 $Y$ 进行一次盲写
>  因为 $T_1, T_2$ 的锁集在变量 $Y$ 处相交，two-phase locking 机制将迫使事务 $T_2$ 完全在 $T_1$ 之后运行

But the sequence  

```
T_1: READ X
T_2: WRITE Y
T_1: WRITE Y
```

in which the write of $\mathrm{T}_{2}$ occurs between the two steps of $\mathrm{T}_{1}$ , yields the same result as running $\mathrm{T}_{2}$ completely before $\mathrm{T}_{1}$ , so the result is always correct, even though this sequence would be prevented by two-phase locking. 

>  但是在上例中的事务序列中，$T_2$ 的写入在 $T_1$ 的两步之间，这与先完全执行 $T_1$，再执行 $T_2$ 得到的结果是相同的
>  因此，即便这样的顺序会被 two-phase locking 阻止，它也是完全正确的

Disciplines that allow all possible concurrency while at the same time ensuring before-or-after atomicity are quite difficult to devise. (Theorists identify the problem as NP-complete.)  
>  能够确保 before-or-after 原子性并且还允许所有可能的并发的锁机制非常难以设计 (理论学家将这个问题归类为 NP 完全问题)

There are two interactions between locks and logs that require some thought: (1) individual transactions that abort, and (2) system recovery. 
>  在锁和日志之间有两种交互需要考虑
>  (1) 单独的事务终止 (2) 系统恢复

Aborts are the easiest to deal with. Since we require that an aborting transaction restore its changed data objects to their original values before releasing any locks, no special account need be taken of aborted transactions. For purposes of before-or-after atomicity they look just like committed transactions that didn’t change anything. The rule about not releasing any locks on modified data before the end of the transaction is essential to accomplishing an abort. If a lock on some modified object were released, and then the transaction decided to abort, it might find that some other transaction has now acquired that lock and changed the object again. Backing out an aborted change is likely to be impossible unless the locks on modified objects have been held.  
>  中止容易处理，因为我们要求中止的事务在释放任何锁之前，将它改变的数据对象恢复为其原来的值，因此无需特别考虑中止的事务
>  从 before-or-after 原子性的目的来看，这些中止的事务看起来就像没有改变任何值的事务
>  "在事务结束之前不得释放对已修改数据的任何锁" 这一规则是实现事务正确中止的关键规则，如果一个在某个已修改对象上的锁被释放，然后事务再决定中止，就可能会有其他事务获取了这个锁，并且又修改了该对象
>  因此，除非对已修改的对象的锁被保持，否则事务在中止时要撤回其更改是不可能的

The interaction between log-based recovery and locks is less obvious. The question is whether locks themselves are data objects for which changes should be logged. To analyze this question, suppose there is a system crash. At the completion of crash recovery there should be no pending transactions because any transactions that were pending at the time of the crash should have been rolled back by the recovery procedure, and recovery does not allow any new transactions to begin until it completes. 
>  基于日志恢复和锁之间的交互并不显而易见
>  问题在于，锁本身是否是需要记录其更改的数据对象
>  为了分析该问题，我们假设存在系统崩溃，在崩溃恢复完成时，不应该存在任何未完成的事务，因为任何在崩溃时尚未完成的事务应该会被崩溃恢复程序回滚，并且崩溃恢复程序完成回滚之前，不会允许任何新的事务开始

Since locks exist only to coordinate pending transactions, it would clearly be an error if there were locks still set when crash recovery is complete. That observation suggests that locks belong in volatile storage, where they will automatically disappear on a crash, rather than in non­ volatile storage, where the recovery procedure would have to hunt them down to release them. The bigger question, however, is whether or not the log-based recovery algorithm will construct a correct system state—correct in the sense that it could have arisen from some serial ordering of those transactions that committed before the crash.  
>  因为锁只是用于协调尚未完成的事务, 因此如果在崩溃恢复完成时还存在未释放的锁，这显然是一个错误
>  这一观察表明，锁应该存在于易失性存储中，它们会在崩溃时自动消失，而不是在非易失性存储中，让恢复程序去寻找并释放它们
>  更大的问题是，基于日志的恢复算法是否可以构建出一个正确的系统状态——“正确”的含义是该状态可以从崩溃前提交的事务的某种串行应用顺序的得到

Continue to assume that the locks are in volatile memory, and at the instant of a crash all record of the locks is lost. Some set of transactions—the ones that logged a BEGIN record but have not yet logged an END record—may not have been completed. But we know that the transactions that were not complete at the instant of the crash had nonoverlapping lock sets at the moment that the lock values vanished. 
>  我们继续假设锁存放在易失性存储中，在崩溃时，所有关于锁的记录都会消失
>  一些事务——那些记录了 BEGIN 但尚未记录 END 的事务——可能尚未完成，但是我们可以确认的是：在崩溃发生时，这些事务的 lock set 都是不相交的

The recovery algorithm of Figure 9.23 will systematically UNDO or REDO installs for the incomplete transactions, but every such UNDO or REDO must modify a variable whose lock was in some transaction’s lock set at the time of the crash. Because those lock sets must have been non-overlapping, those particular actions can safely be redone or undone without concern for before-or-after atomicity during recovery. Put another way, the locks created a particular serialization of the transactions and the log has captured that serialization. 
>  Fig 9.23 中的算法将系统地为未完成的事务执行 UNDO 或 REDO 安装操作，但每次 UNDO 或 REDO 都必须修改某个变量，在崩溃时，该变量的锁是处于某个事务的 lock set 中的
>  因为这些 lock set 不相交，因此可以在恢复时安全地执行 UNDO 或 REDO，而不必担心影响 before-or-after 原子性
>  换句话说，锁创建了事务的一种特定序列化方式，而日志捕获了这种序列化

Since RECOVER performs UNDO actions in reverse order as specified in the log, and it per­forms REDO actions in forward order, again as specified in the log, RECOVER reconstructs exactly that same serialization. Thus even a recovery algorithm that reconstructs the entire database from the log is guaranteed to produce the same serialization as when the transactions were originally performed. So long as no new transactions begin until recovery is complete, there is no danger of miscoordination, despite the absence of locks during recovery.  
>  因为恢复过程按照日志中的顺序反向执行 UNDO 操作，并且正向执行 REDO 操作，因此恢复过程可以完全重建相同的序列化
>  因此，从日志中重构整个数据库的恢复算法是保证产生和事务最初执行时相同的序列化结果的
>  只要在恢复完成之前没有新的事务开始，就不会存在协调错误的风险，即便在恢复过程中并没有锁

## 9.6 Atomicity across Layers and Multiple Sites
### 9.6.3 Multiple-Site Atomicity: Distributed Two-Phase Commit
If a transaction requires executing component transactions at several sites that are sepa­ rated by a best-effort network, obtaining atomicity is more difficult because any of the messages used to coordinate the transactions of the various sites can be lost, delayed, or duplicated. In Chapter 4 we learned of a method, known as Remote Procedure Call (RPC) for performing an action at another site. In Chapter 7[on-line] we learned how to design protocols such as RPC with a persistent sender to ensure at-least-once execu­ tion and duplicate suppression to ensure at-most-once execution. Unfortunately, neither of these two assurances is exactly what is needed to ensure atomicity of a multiple-site transaction. However, by properly combining a two-phase commit protocol with persis­ tent senders, duplicate suppression, and single-site transactions, we can create a correct multiple-site transaction. We assume that each site, on its own, is capable of implement­ ing local transactions, using techniques such as version histories or logs and locks for allor-nothing atomicity and before-or-after atomicity. Correctness of the multiple-site ato­ micity protocol will be achieved if all the sites commit or if all the sites abort; we will have failed if some sites commit their part of a multiple-site transaction while others abort their part of that same transaction.

Suppose the multiple-site transaction consists of a coordinator Alice requesting com­ ponent transactions X, Y, and Z of worker sites Bob, Charles, and Dawn, respectively. The simple expedient of issuing three remote procedure calls certainly does not produce a transaction for Alice because Bob may do X while Charles may report that he cannot do Y. Conceptually, the coordinator would like to send three messages, to the three workers, like this one to Bob:

From: Alice To: Bob Re: my transaction 91 if (Charles does Y and Dawn does Z) then do $\mathsf{X},$ please.

and let the three workers handle the details. We need some clue how Bob could accom­ plish this strange request.

The clue comes from recognizing that the coordinator has created a higher-layer transaction and each of the workers is to perform a transaction that is nested in the higher-layer transaction. Thus, what we need is a distributed version of the two-phase commit protocol. The complication is that the coordinator and workers cannot reliably communicate. The problem thus reduces to constructing a reliable distributed version of the two-phase commit protocol. We can do that by applying persistent senders and duplicate suppression.

Phase one of the protocol starts with coordinator Alice creating a top-layer outcome record for the overall transaction. Then Alice begins persistently sending to Bob an RPClike message:

From: Alice To: Bob Re: my transaction 271 Please do X as part of my transaction.

Similar messages go from Alice to Charles and Dawn, also referring to transaction 271, and requesting that they do Y and Z, respectively. As with an ordinary remote procedure call, if Alice doesn’t receive a response from one or more of the workers in a reasonable time she resends the message to the non-responding workers as many times as necessary to elicit a response.

A worker site, upon receiving a request of this form, checks for duplicates and then creates a transaction of its own, but it makes the transaction a nested one, with its superior being Alice’s original transaction. It then goes about doing the pre-commit part of the requested action, reporting back to Alice that this much has gone well:

From: Bob To: Alice Re: your transaction 271 My part X is ready to commit.

Alice, upon collecting a complete set of such responses then moves to the two-phase commit part of the transaction, by sending messages to each of Bob, Charles, and Dawn saying, e.g.:
Two-phase-commit message #1 :
From: Alice To: Bob Re: my transaction 271 PREPARE to commit X.

Bob, upon receiving this message, commits—but only tentatively—or aborts. Having created durable tentative versions (or logged to journal storage its planned updates) and having recorded an outcome record saying that it is PREPARED either to commit or abort, Bob then persistently sends a response to Alice reporting his state:

![](https://cdn-mineru.openxlab.org.cn/extract/0ad2aa4a-dde4-4e4b-82df-c25c550c440b/365944f06d4e3d6f621257cff800c056f69fe31ab38228c7b570a0f6d34c9da0.jpg)

## FIGURE 9.37

Timing diagram for distributed two-phase commit, using 3Nmessages. (The initial RPC request and response messages are not shown.) Each of the four participants maintains its own version history or recovery log. The diagram shows log entries made by the coordinator and by one of the workers.

sending the PREPARE message but before sending the COMMIT or ABORT message the worker sites are in left in the PREPARED state with no way to proceed. Even without that concern, Alice and her co-workers are standing uncomfortably close to a multiple-site atomicity problem that, at least in principle, can not be solved. The only thing that rescues them is our observation that the several workers will do their parts eventually, not necessarily simultaneously. If she had required simultaneous action, Alice would have been in trouble.

The unsolvable problem is known as the dilemma of the two generals.