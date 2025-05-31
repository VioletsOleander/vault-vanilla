### Introduction
You can either do a [final project](http://nil.csail.mit.edu/6.5840/2023/project.html) based on your own ideas, or this lab.

In this lab you'll build a key/value storage system that "shards," or partitions, the keys over a set of replica groups. A shard is a subset of the key/value pairs; for example, all the keys starting with "a" might be one shard, all the keys starting with "b" another, etc. The reason for sharding is performance. Each replica group handles puts and gets for just a few of the shards, and the groups operate in parallel; thus total system throughput (puts and gets per unit time) increases in proportion to the number of groups.
>  构建在副本组上划分 keys 的 key-value 存储系统，一个 shard 是一个 key-value 对子集
>  例如，所有以 'a' 开头的 keys 属于一个 shard，所有以 'b' 开头的 keys s 属于另一个 shard
>  做 sharding 的理由在于性能，每个副本组 (Raft group)仅处理一部分 shards，这些组并行运作，故总的系统吞吐 (单位时间的 puts/gets) 依照组的数量成比例增长

Your sharded key/value store will have two main components. First, a set of replica groups. Each replica group is responsible for a subset of the shards. A replica consists of a handful of servers that use Raft to replicate the group's shards. The second component is the "shard controller". The shard controller decides which replica group should serve each shard; this information is called the configuration. The configuration changes over time. Clients consult the shard controller in order to find the replica group for a key, and replica groups consult the controller in order to find out what shards to serve. There is a single shard controller for the whole system, implemented as a fault-tolerant service using Raft.
>  sharded key/value store 包含两个主要成分
>  其一，一个副本组的集合，每个副本组负责一些 shards
>  其二，shard controller，它决定哪个副本组应该服务哪个 shard，即决定配置，配置会随着时间改变
>  客户端咨询 shard controller 以确认特定的 key 应该去找哪个副本组，副本组咨询 shard controller 以确定它们应该负责哪些 shards 的复制
>  整个系统只有一个 shard controller，用 Raft 实现为容错服务

A sharded storage system must be able to shift shards among replica groups. One reason is that some groups may become more loaded than others, so that shards need to be moved to balance the load. Another reason is that replica groups may join and leave the system: new replica groups may be added to increase capacity, or existing replica groups may be taken offline for repair or retirement.
>  sharded storage system 需要能在副本组之间迁移 shards，原因包括: 负载均衡、副本组会加入以及离开系统

The main challenge in this lab will be handling reconfiguration -- changes in the assignment of shards to groups. Within a single replica group, all group members must agree on when a reconfiguration occurs relative to client Put/Append/Get requests. For example, a Put may arrive at about the same time as a reconfiguration that causes the replica group to stop being responsible for the shard holding the Put's key. All replicas in the group must agree on whether the Put occurred before or after the reconfiguration. If before, the Put should take effect and the new owner of the shard will see its effect; if after, the Put won't take effect and client must re-try at the new owner. The recommended approach is to have each replica group use Raft to log not just the sequence of Puts, Appends, and Gets but also the sequence of reconfigurations. You will need to ensure that at most one replica group is serving requests for each shard at any one time.
>  难度在于处理重配置——修改对 groups 的 shards 分配
>  在单个副本组内，当重配置相对于客户端 Put/Append/Get 请求发生时，需要确保所有的组成员都达成一致
>  例如，在副本组正好要执行重配置，以移除对某个 shard 的责任时，一个涉及了该 shard 内的 key 的 Put 到达，此时组内的所有副本需要就 Put 是在重配置之前发生还是重配置之后发生达成一致
>  如果在之前，该组应该负责处理 Put，如果在之后，客户端需要重试，找到新的组
>  推荐让 Raft 记录重配置操作
>  需要确保对于每个 shard，每次只有一个复制组对其负责

Reconfiguration also requires interaction among the replica groups. For example, in configuration 10 group G1 may be responsible for shard S1. In configuration 11, group G2 may be responsible for shard S1. During the reconfiguration from 10 to 11, G1 and G2 must use RPC to move the contents of shard S1 (the key/value pairs) from G1 to G2.
>  重配置进行时，复制组需要通过 RPC 来迁移所有者改变的 shards

Only RPC may be used for interaction among clients and servers. For example, different instances of your server are not allowed to share Go variables or files.

This lab uses "configuration" to refer to the assignment of shards to replica groups. This is not the same as Raft cluster membership changes. You don't have to implement Raft cluster membership changes.

This lab's general architecture (a configuration service and a set of replica groups) follows the same general pattern as Flat Datacenter Storage, BigTable, Spanner, FAWN, Apache HBase, Rosebud, Spinnaker, and many others. These systems differ in many details from this lab, though, and are also typically more sophisticated and capable. For example, the lab doesn't evolve the sets of peers in each Raft group; its data and query models are very simple; and handoff of shards is slow and doesn't allow concurrent client access.

Your Lab 4 sharded server, Lab 4 shard controller, and Lab 3 kvraft must all use the same Raft implementation.

### Getting Started
Do a git pull to get the latest lab software.

We supply you with skeleton code and tests in `src/shardctrler` and `src/shardkv`.

To get up and running, execute the following commands:

```
$ cd ~/6.5840
$ git pull
...
$ cd src/shardctrler
$ go test
--- FAIL: TestBasic (0.00s)
        test_test.go:11: wanted 1 groups, got 0
FAIL
exit status 1
FAIL    shardctrler     0.008s
$
```

When you're done, your implementation should pass all the tests in the `src/shardctrler` directory, and all the ones in `src/shardkv`.

### Part A: The Shard controller (30 points)(easy)
First you'll implement the shard controller, in `shardctrler/server.go` and `client.go`. When you're done, you should pass all the tests in the `shardctrler` directory:

```
$ cd ~/6.5840/src/shardctrler
$ go test
Test: Basic leave/join ...
  ... Passed
Test: Historical queries ...
  ... Passed
Test: Move ...
  ... Passed
Test: Concurrent leave/join ...
  ... Passed
Test: Minimal transfers after joins ...
  ... Passed
Test: Minimal transfers after leaves ...
  ... Passed
Test: Multi-group join/leave ...
  ... Passed
Test: Concurrent multi leave/join ...
  ... Passed
Test: Minimal transfers after multijoins ...
  ... Passed
Test: Minimal transfers after multileaves ...
  ... Passed
Test: Check Same config on servers ...
  ... Passed
PASS
ok  	6.5840/shardctrler	5.863s
$
```

The shardctrler manages a sequence of numbered configurations. Each configuration describes a set of replica groups and an assignment of shards to replica groups. Whenever this assignment needs to change, the shard controller creates a new configuration with the new assignment. Key/value clients and servers contact the shardctrler when they want to know the current (or a past) configuration.
>  shard controller 管理一个有编号的配置序列，每个配置描述一组副本组及其负责的 shards
>  当需要改变配置时，shard controller 创建一个新的配置
>  客户端和 servers 在需要知道某个配置时，询问 shard controller

Your implementation must support the RPC interface described in `shardctrler/common.go`, which consists of Join, Leave, Move, and Query RPCs. These RPCs are intended to allow an administrator (and the tests) to control the `shardctrler`: to add new replica groups, to eliminate replica groups, and to move shards between replica groups.

The Join RPC is used by an administrator to add new replica groups. Its argument is a set of mappings from unique, non-zero replica group identifiers (GIDs) to lists of server names. The shardctrler should react by creating a new configuration that includes the new replica groups. The new configuration should divide the shards as evenly as possible among the full set of groups, and should move as few shards as possible to achieve that goal. The shardctrler should allow re-use of a GID if it's not part of the current configuration (i.e. a GID should be allowed to Join, then Leave, then Join again).
>  Join RPC 用于添加新的副本组，其参数是 GIDs 到 server name list 的映射
>  shard controller 应该创建一个包含了新副本组的配置作为回应
>  新配置应该尽可能平均分配 shards，并且最好移动的 shards 数量不多

The Leave RPC's argument is a list of GIDs of previously joined groups. The shardctrler should create a new configuration that does not include those groups, and that assigns those groups' shards to the remaining groups. The new configuration should divide the shards as evenly as possible among the groups, and should move as few shards as possible to achieve that goal.
>  Leave RPC 同理

The Move RPC's arguments are a shard number and a GID. The shardctrler should create a new configuration in which the shard is assigned to the group. The purpose of Move is to allow us to test your software. A Join or Leave following a Move will likely un-do the Move, since Join and Leave re-balance.
>  Move RPC 的参数是一个 shard number 和一个 GID

The Query RPC's argument is a configuration number. The shardctrler replies with the configuration that has that number. If the number is -1 or bigger than the biggest known configuration number, the shardctrler should reply with the latest configuration. The result of Query(-1) should reflect every Join, Leave, or Move RPC that the shardctrler finished handling before it received the Query(-1) RPC.
>  Query RPC 的参数是一个 configuration number

The very first configuration should be numbered zero. It should contain no groups, and all shards should be assigned to GID zero (an invalid GID). The next configuration (created in response to a Join RPC) should be numbered 1, &c. There will usually be significantly more shards than groups (i.e., each group will serve more than one shard), in order that load can be shifted at a fairly fine granularity.
>  第一个配置从 number 0 开始，不包含 groups，所有 shards 都分配给 GID 0

Your task is to implement the interface specified above in `client.go` and `server.go` in the `shardctrler/` directory. Your `shardctrler` must be fault-tolerant, using your Raft library from Lab 2/3. You have completed this task when you pass all the tests in `shardctrler/.`
>  实现 `client.go, server.go` 中指定的接口

- Start with a stripped-down copy of your kvraft server.
- You should implement duplicate client request detection for RPCs to the shard controller. The `shardctrler` tests don't test this, but the `shardkv` tests will later use your `shardctrler` on an unreliable network; you may have trouble passing the `shardkv` tests if your `shardctrler` doesn't filter out duplicate RPCs.
>  shard controller 应该检测重复 RPC

- The code in your state machine that performs the shard rebalancing needs to be deterministic. In Go, map iteration order is [not deterministic](https://blog.golang.org/maps#TOC_7.).
>  执行 rebalancing 的代码应该是确定性的

- Go maps are references. If you assign one variable of type map to another, both variables refer to the same map. Thus if you want to create a new Config based on a previous one, you need to create a new map object (with `make()`) and copy the keys and values individually.
- The Go race detector (`go test -race`) may help you find bugs.

### Part B: Sharded Key/Value Server (60 points)(hard)
Do a git pull to get the latest lab software.

Now you'll build shardkv, a sharded fault-tolerant key/value storage system. You'll modify `shardkv/client.go`, `shardkv/common.go`, and `shardkv/server.go`.

Each shardkv server operates as part of a replica group. Each replica group serves Get, Put, and Append operations for some of the key-space shards. Use `key2shard()` in `client.go` to find which shard a key belongs to. Multiple replica groups cooperate to serve the complete set of shards. A single instance of the `shardctrler` service assigns shards to replica groups; when this assignment changes, replica groups have to **hand off shards to each other, while ensuring that clients do not see inconsistent responses.**
>  每个 shardkv server 属于一个 replica group
>  每个 replica group 提供对一些 key-space shards 的 Get, Put, Append 操作服务
>  `client.go` 使用 `key2shard()` 找到 key 对应的 shard
>  多个 replica groups 一起提供对完成一组 shards 的服务
>  `shardctrler` 服务为 replica groups 分配 shards，分配情况变化时，replica groups 之间需要互相交换 shards，并确保 client 不会看到不一致状态

Your storage system must provide a linearizable interface to applications that use its client interface. That is, completed application calls to the `Clerk.Get()`, `Clerk.Put()`, and `Clerk.Append()` methods in `shardkv/client.go` must appear to have affected all replicas in the same order. A `Clerk.Get()` should see the value written by the most recent Put/Append to the same key. This must be true even when Gets and Puts arrive at about the same time as configuration changes.

Each of your shards is only required to make progress when a majority of servers in the shard's Raft replica group is alive and can talk to each other, and can talk to a majority of the `shardctrler` servers. Your implementation must operate (serve requests and be able to re-configure as needed) even if a minority of servers in some replica group(s) are dead, temporarily unavailable, or slow.

A shardkv server is a member of only a single replica group. The set of servers in a given replica group will never change.
>  一个 shard server 只能是一个 replica group 的成员
>  replica group 的 shard servers 不会变化

We supply you with `client.go` code that sends each RPC to the replica group responsible for the RPC's key. It re-tries if the replica group says it is not responsible for the key; in that case, the client code **asks the shard controller for the latest configuration and tries again**. You'll have to modify `client.go` as part of your support for dealing with duplicate client RPCs, much as in the kvraft lab.
>  `client.go` 向负责对应 shard 的 replica group 发送请求
>  如果 replica group 回应自己不负责该 shard，`client.go` 需要向 shard controller 请求最新配置，然后重试

When you're done your code should pass all the shardkv tests other than the challenge tests:

```
$ cd ~/6.5840/src/shardkv
$ go test
Test: static shards ...
  ... Passed
Test: join then leave ...
  ... Passed
Test: snapshots, join, and leave ...
  ... Passed
Test: servers miss configuration changes...
  ... Passed
Test: concurrent puts and configuration changes...
  ... Passed
Test: more concurrent puts and configuration changes...
  ... Passed
Test: concurrent configuration change and restart...
  ... Passed
Test: unreliable 1...
  ... Passed
Test: unreliable 2...
  ... Passed
Test: unreliable 3...
  ... Passed
Test: shard deletion (challenge 1) ...
  ... Passed
Test: unaffected shard access (challenge 2) ...
  ... Passed
Test: partial migration shard access (challenge 2) ...
  ... Passed
PASS
ok  	6.5840/shardkv	101.503s
$
```

Your server should not call the shard controller's Join() handler. The tester will call Join() when appropriate.

Your first task is to pass the very first shardkv test. In this test, there is only a single assignment of shards, so your code should be very similar to that of your Lab 3 server. The biggest modification will be to have your server detect when a configuration happens and start accepting requests whose keys match shards that it now owns.

Now that you solution works for the static sharding case, it's time to tackle the problem of configuration changes. You will need to make your servers watch for configuration changes, and when one is detected, to start the shard migration process. If a replica group loses a shard, it must stop serving requests to keys in that shard immediately, and start migrating the data for that shard to the replica group that is taking over ownership. If a replica group gains a shard, it needs to wait for the previous owner to send over the old shard data before accepting requests for that shard.
>  shardkv servers 需要观察配置变化，如果发现了配置变化，需要进行 shard 迁移
>  如果一个 replica group 失去了 shard，它需要立即停止对该 shard 的服务，然后将该 shard 迁移
>  如果一个 replica group 获得了 shard，它需要等待迁移完成，再接收请求

Implement shard migration during configuration changes. Make sure that all servers in a replica group do the migration at the same point in the sequence of operations they execute, so that they all either accept or reject concurrent client requests. You should focus on passing the second test ("join then leave") before working on the later tests. You are done with this task when you pass all tests up to, but not including, TestDelete.

Your server will need to periodically poll the shardctrler to learn about new configurations. The tests expect that your code polls roughly every 100 milliseconds; more often is OK, but much less often may cause problems.
>  server 需要定期轮询 `shardctrler` 以获取新的配置
>  轮询间隔大约是 100ms

Servers will need to send RPCs to each other in order to transfer shards during configuration changes. The shardctrler's Config struct contains server names, but you need a `labrpc.ClientEnd` in order to send an RPC. You should use the `make_end()` function passed to `StartServer()` to turn a server name into a `ClientEnd`. `shardkv/client.go` contains code that does this.

- Add code to `server.go` to periodically fetch the latest configuration from the `shardctrler`, and add code to reject client requests if the receiving group isn't responsible for the client's key's shard. You should still pass the first test.
>  在 `server.go` 中添加周期性从 `shardctrler` 获取最新配置的代码

- Your server should respond with an `ErrWrongGroup` error to a client RPC with a key that the server isn't responsible for (i.e. for a key whose shard is not assigned to the server's group). Make sure your Get, Put, and Append handlers make this decision correctly in the face of a concurrent re-configuration.

- Process re-configurations one at a time, in order.

- If a test fails, check for gob errors (e.g. "gob: type not registered for interface ..."). Go doesn't consider gob errors to be fatal, although they are fatal for the lab.

- You'll need to provide at-most-once semantics (duplicate detection) for client requests across shard movement.

- Think about how the `shardkv` client and server should deal with `ErrWrongGroup`. Should the client change the sequence number if it receives `ErrWrongGroup`? Should the server update the client state if it returns `ErrWrongGroup` when executing a Get/Put request?

- After a server has moved to a new configuration, it is acceptable for it to continue to store shards that it no longer owns (though this would be regrettable in a real system). This may help simplify your server implementation.

- When group G1 needs a shard from G2 during a configuration change, does it matter at what point during its processing of log entries G2 sends the shard to G1?

- You can send an entire map in an RPC request or reply, which may help keep the code for shard transfer simple.

- If one of your RPC handlers includes in its reply a map (e.g. a key/value map) that's part of your server's state, you may get bugs due to races. The RPC system has to read the map in order to send it to the caller, but it isn't holding a lock that covers the map. Your server, however, may proceed to modify the same map while the RPC system is reading it. The solution is for the RPC handler to include a copy of the map in the reply.

- If you put a map or a slice in a Raft log entry, and your key/value server subsequently sees the entry on the applyCh and saves a reference to the map/slice in your key/value server's state, you may have a race. Make a copy of the map/slice, and store the copy in your key/value server's state. The race is between your key/value server modifying the map/slice and Raft reading it while persisting its log.

- During a configuration change, a pair of groups may need to move shards in both directions between them. If you see deadlock, this is a possible source.

### No-credit challenge exercises
These two features would be essential if you were to build a system like this for production use.

#### Garbage collection of state
When a replica group loses ownership of a shard, that replica group should eliminate the keys that it lost from its database. It is wasteful for it to keep values that it no longer owns, and no longer serves requests for. However, this poses some issues for migration. Say we have two groups, G1 and G2, and there is a new configuration C that moves shard S from G1 to G2. If G1 erases all keys in S from its database when it transitions to C, how does G2 get the data for S when it tries to move to C?

Cause each replica group to keep old shards no longer than absolutely necessary. Your solution must work even if all the servers in a replica group like G1 above crash and are then brought back up. You have completed this challenge if you pass TestChallenge1Delete.

#### Client requests during configuration changes
The simplest way to handle configuration changes is to disallow all client operations until the transition has completed. While conceptually simple, this approach is not feasible in production-level systems; it results in long pauses for all clients whenever machines are brought in or taken out. It would be better to continue serving shards that are not affected by the ongoing configuration change.

Modify your solution so that client operations for keys in unaffected shards continue to execute during a configuration change. You have completed this challenge when you pass TestChallenge2Unaffected.

While the optimization above is good, we can still do better. Say that some replica group G3, when transitioning to C, needs shard S1 from G1, and shard S2 from G2. We really want G3 to immediately start serving a shard once it has received the necessary state, even if it is still waiting for some other shards. For example, if G1 is down, G3 should still start serving requests for S2 once it receives the appropriate data from G2, despite the transition to C not yet having completed.

Modify your solution so that replica groups start serving shards the moment they are able to, even if a configuration is still ongoing. You have completed this challenge when you pass TestChallenge2Partial.

### Handin procedure
Before submitting, please run _all_ the tests one final time.

Also, note that your Lab 4 sharded server, Lab 4 shard controller, and Lab 3 kvraft must all use the same Raft implementation.

Before submitting, double check that your solution works with:

```
$ go test ./raft
$ go test ./kvraft
$ go test ./shardctrler
$ go test ./shardkv
```