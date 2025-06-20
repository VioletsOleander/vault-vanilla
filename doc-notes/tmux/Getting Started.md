>  Nicholas Marriott edited this page on Oct 13, 2022
### About tmux
tmux is a program which runs in a terminal and allows multiple other terminal programs to be run inside it. Each program inside tmux gets its own terminal managed by tmux, which can be accessed from the single terminal where tmux is running - this called multiplexing and tmux is a terminal multiplexer.
>  tmux 是一个在单个 terminal 运行的程序，它允许在其中运行多个其他的 terminal 程序，tmux 中的每个程序都有自己的 terminal，这些 terminals 由 tmux 管理
>  这些 terminals 可以从运行了 tmux 的单个 terminal 中访问 - 这被称为复用，而 tmux 就是一个终端复用器

tmux - and any programs running inside it - may be detached from the terminal where it is running (the outside terminal) and later reattached to the same or another terminal.
>  tmux 本身以及在 tmux 中运行的任何程序可以从运行 tmux 的终端 (the outside terminal) 分离，并且稍后重新连接到同一个终端或另一个终端

Programs run inside tmux may be full screen interactive programs like _vi(1)_ or _top(1)_, shells like _bash(1)_ or _ksh(1)_, or any other program that can be run in a Unix terminal.
>  在 tmux 中运行的程序可以是全屏交互式的程序，例如 vi, top，也可以是 shell 程序，例如 shell, ksh，或者任意可以在 Unix 终端中运行的其他程序

There is a powerful feature set to access, manage and organize programs inside tmux, both interactively and from scripts.
>  tmux 提供了许多功能用于访问、管理和组织 tmux 中的程序

The main uses of tmux are to:

- Protect running programs on a remote server from connection drops by running them inside tmux.
- Allow programs running on a remote server to be accessed from multiple different local computers.
- Work with multiple programs and shells together in one terminal, a bit like a window manager.

>  tmux 的主要作用有
>  - 通过在 tmux 中运行程序，使得远程服务器上运行的程序免受连接中断的影响
>  - 允许在远程服务器上运行的程序从多个不同的本地计算机访问
>  - 在单个终端中同时处理多个程序和 shell，类似于窗口管理器

For example:

- A user connects to a remote server using _ssh(1)_ from an _xterm(1)_ on their work computer and run several programs. perhaps an editor, a compiler and a few shells.
- They work with these programs interactively, perhaps start compiling, then close the _xterm(1)_ with tmux and go home for the day.
- They are then able to connect to the same remote server from home, attach to tmux, and continue from where they were previously.

>  例如
> - 用户使用工作电脑上的 `xterm` 中的 `ssh` 连接到远程服务器，并运行了多个程序，比如一个编辑器、一个编译器和几个 shell。
> - 他们以交互方式使用这些程序，可能开始编译，然后通过 tmux 关闭 `xterm(1)` 并回家。
> - 然后，他们可以从家中再次连接到同一台远程服务器，**附加到 tmux 会话**，并从之前中断的地方继续工作。

Here is a screenshot of tmux in an _xterm(1)_ showing the shell:

![](https://github.com/tmux/tmux/wiki/images/tmux_default.png)

### About this document
This document gives an overview of some of tmux's key concepts, a description of how to use the main features interactively and some information on basic customization and configuration.

Note that this document may mention features only available in the latest tmux release. Only the latest tmux release is supported. Releases are made approximately every six months.

tmux may be installed from package management systems on most major platforms. See [this document](https://github.com/tmux/tmux/wiki/Installing) for instructions on how to install tmux or how to build it from source.

### Other documentation and help
Here are several places to find documentation and help about tmux:

- ![](https://github.com/tmux/tmux/wiki/images/man_tmux.png)[The manual page](https://man.openbsd.org/tmux) has detailed reference documentation on tmux and a description of every command, flag and option. Once tmux is installed it is also available in section 1:
    
    ```
    $ man 1 tmux
    ```
    
- [The FAQ](https://github.com/tmux/tmux/wiki/FAQ) has solutions to commonly asked questions, mostly about specific configuration issues.
- The [tmux-users@googlegroups.com mailing list](mailto:tmux-users@googlegroups.com).

### Basic concepts
tmux has a set of basic concepts and terms it is important to be familiar with. This section gives a description of how the terminals inside tmux are grouped together and the various terms tmux uses.

#### The tmux server and clients
tmux keeps all its state in a single main process, called the tmux server. This runs in the background and manages all the programs running inside tmux and keeps track of their output. The tmux server is started automatically when the user runs a tmux command and by default exits when there are no running programs.
>  tmux 将其所有状态保存在单个主进程中，该进程称为 tmux server
>  tmux server 在后台运行，管理所有在 tmux 中运行的程序，并跟踪它们的输出
>  当用户运行 `tmux` 命令时，tmux server 会自动启动，并且默认会在没有正在运行的程序时退出

Users attach to the tmux server by starting a client. This takes over the terminal where it is run and talks to the server using a socket file in `/tmp`. Each client runs in one terminal, which may be an _X(7)_ terminal such as _xterm(1)_, the system console, or a terminal inside another program (such as tmux itself). Each client is identified by the name of the outside terminal where it is started, for example `/dev/ttypf`.
>  用户通过启动一个 client 来连接到 tmux server，该 client 会接管运行它的终端，然后通过 `/tmp` 目录中的套接字文件和 server 交流
>  每个 client 都在一个终端运行，该终端可以是像 xterm 这样的 X 终端、系统控制台，或者另一个程序内的终端 (例如 tmux 内的终端)
>  每个 client 通过它启动时所在的外部终端名称来标识，例如 `/dev/ttypf`

#### Sessions, windows and panes
![](https://github.com/tmux/tmux/wiki/images/tmux_with_panes.png)Every terminal inside tmux belongs to one pane, this is a rectangular area which shows the content of the terminal inside tmux. Because each terminal inside tmux is shown in only one pane, the term pane can be used to mean all of the pane, the terminal and the program running inside it. The screenshot to the right shows tmux with panes.
>  tmux 内的每个终端都属于一个面板，面板是一个矩形区域，用于显示 tmux 内部终端的内容
>  因为 tmux 中的每个终端仅显示在一个面板中，故术语 'pane/面板' 可以也用于指代该面板显示的终端以及终端内运行的程序

Each pane appears in one window. A window is made up of one or more panes which together cover its entire area - so multiple panes may be visible at the same time. A window normally takes up the whole of the terminal where tmux is attached, but it can be bigger or smaller. The sizes and positions of all the panes in a window is called the window layout.
>  每个面板出现在一个窗口中，一个窗口由一个或多个面板组成，组成了窗口的面板共同覆盖整个窗口区域，故可以同时看到多个面板
>  一个窗口通常占用了 tmux 本身附加到的终端的整个屏幕，但也可以更大或者更小
>  窗口中所有面板的大小和位置称为窗口布局

Every window has a name - by default tmux will choose one but it can be changed by the user. Window names do not have to be unique, windows are usually identified by the session and the window index rather than their name.
>  每个窗口都有一个名称，tmux 会默认为其选择一个，用户也可以更改
>  窗口名称不必唯一，因为窗口通常通过会话和窗口索引来标识

![](https://github.com/tmux/tmux/wiki/images/tmux_pane_diagram.png)Each pane is separated from the panes around it by a line, this is called the pane border. There is one pane in each window called the active pane, this is where any text typed is sent and is the default pane used for commands that target the window. The pane border of the active pane is marked in green, or if there are only two panes then the top, bottom, left or right half of the border is green.
>  每个面板之间用一条线分离，这条线称为面板边界
>  每个窗口都有一个面板，称为活动面板，任何输入的文本都会被发送到活动面板，并且针对窗口的命令的默认执行对象也是活动面板
>  活动面板的边界是绿色

Multiple windows are grouped together into sessions. If a window is part of a session, it is said to be linked to that session. Windows may be linked to multiple sessions at the same time, although mostly they are only in one. Each window in a session has a number, called the window index - the same window may be linked at different indexes in different sessions. A session's window list is all the windows linked to that session in order of their indexes.
>  多个窗口组合在一起形成会话
>  如果一个窗口属于某个会话，称它与该会话相关联
>  窗口可以同时关联到多个会话，虽然大部分情况下仅关联一个
>  会话中的每个窗口都有一个编号，称为窗口索引 - 同一个窗口在不同的会话中可能具有不同的索引
>  会话的窗口列表即按窗口索引顺序排列所有关联到它的窗口

Each session has one current window, this is the window displayed when the session is attached and is the default window for any commands that target the session. If the current window is changed, the previous current window becomes known as the last window.
>  每个会话都有一个当前窗口，它是当会话被连接时展示的窗口，同时也是针对会话执行的任何命令的默认目标窗口
>  如果当前窗口变化，上一个 “当前窗口” 称为 "上一个窗口"

A session may be attached to one or more clients, which means it is shown on the outside terminal where that client is running. Any text typed into that outside terminal is sent to the active pane in the current window of the attached session. Sessions do not have an index but they do have a name, which must be unique.
>  一个会话可以连接到一个或多个客户端，会话连接到客户端意味着会话会在运行该客户端的外部终端中显示
>  输入到外部终端的任何文本会被发送给它运行的客户端的连接的会话的当前窗口的活动面板 (tmux 内的一个 terminal)
>  会话没有索引，由唯一的名称标识

In summary:

- Programs run in terminals in panes, which each belong to one window.
- Each window has a name and one active pane.
- Windows are linked to one or more sessions.
- Each session has a list of windows, each with an index.
- One of the windows in a session is the current window.
- Sessions are attached to one or more clients, or are detached (attached to no clients).
- Each client is attached to one session.

>  总结
>  - 程序在 terminals 中运行, terminals 在 pane 内, pane 属于一个 window
>  - 每个 window 都有一个名称和一个 active pane
>  - windows 关联到一个或多个 sessions
>  - 每个 session 有一个 windows 列表，其中每个 window 有一个索引
>  - session 中的一个 windows 是 current window
>  - sessions attach 到一个或多个 clients, 也可以是 detached 状态 (没有 attach 到任何 clients)
>  - 每个 clients attach 到一个 session

>  因此，运行 tmux 时，我们是在当前的 (外部) 终端上运行 client，和 tmux server 交流，当前的 (外部) 终端退出后，仅会导致 client 退出，sessions 回到 detached 状态，但是不会退出

#### Summary of terms

| Term           | Description                                                                          |
| -------------- | ------------------------------------------------------------------------------------ |
| Client         | Attaches a tmux session from an outside terminal such as _xterm(1)_                  |
| Session        | Groups one or more windows together                                                  |
| Window         | Groups one or more panes together, linked to one or more sessions                    |
| Pane           | Contains a terminal and running program, appears in one window                       |
| Active pane    | The pane in the current window where typing is sent; one per window                  |
| Current window | The window in the attached session where typing is sent; one per session             |
| Last window    | The previous current window                                                          |
| Session name   | The name of a session, defaults to a number starting from zero                       |
| Window list    | The list of windows in a session in order by number                                  |
| Window name    | The name of a window, defaults to the name of the running program in the active pane |
| Window index   | The number of a window in a session's window list                                    |
| Window layout  | The size and position of the panes in a window                                       |

### Using tmux interactively
#### Creating sessions
To create the first tmux session, tmux is run from the shell. A new session is created using the `new-session` command - `new` for short:

```
$ tmux new
```

Without arguments, `new-session` creates a new session and attaches it. Because this is the first session, the tmux server is started and the tmux run from the shell becomes the first client and attaches to it.

>  在 shell 中运行 `tmux new/new-session` 用于创建新 session
>  如果是第一个 session, tmux server 会启动，然后从当前 shell 启动的 tmux 会成为第一个 client，然后连接到新创建的 session

The new session will have one window (at index 0) with a single pane containing a shell. The shell prompt should appear at the top of the terminal and the green status line at the bottom (more on the status line is below).
>  新的 session 将只有一个 window 和一个 pane, pane 中将运行 shell 程序
>  故 pane 中 shell 程序的 prompt 将出现在当前 terminal 中，同时底端出现状态栏

By default, the first session will be called `0`, the second `1` and so on. `new-session` allows a name to be specified for the session with the `-s` flag:

```
$ tmux new -smysession
```

This creates a new session called `mysession`. 

>  `-s` 可以用于指定 session 名称，默认名称是 session 的编号

A command may be given instead of running a shell by passing additional arguments. If one argument is given, tmux will pass it to the shell, if more than one it runs the command directly. For example these run _emacs(1)_:

```
$ tmux new 'emacs ~/.tmux.conf'
```

Or:

```
$ tmux new -- emacs ~/.tmux.conf
```

>  新创建的 session 中的第一个 windows 中的第一个 pane 可以运行其他命令而不是 shell, 我们通过在 `tmux new` 中传递额外参数即可
>  如果仅传递一个参数，`tmux` 默认运行 shell，并将该参数传递给 shell
>  如果传递多个参数，则 `tmux` 直接运行该命令

By default, tmux calls the first window in the session after whatever is running in it. The `-n` flag gives a name to use instead, in this case a window `mytopwindow` running _top(1)_:

```
$ tmux new -nmytopwindow top
```

>  `tmux new` 默认让新 session 连接到它的第一个 window, `-n` 可以用于指定该 window 名称

`new-session` has other flags - some are covered below. A full list is [in the tmux manual](https://man.openbsd.org/tmux#new-session).

#### The status line
When a tmux client is attached, it shows a status line on the bottom line of the screen. By default this is green and shows:

- On the left, the name of the attached session: `[0]`.
- In the middle, a list of the windows in the session, with their index, for example with one window called `ksh` at index 0: `0:ksh`.
- On the right, the pane title in quotes (this defaults to the name of the host running tmux) and the time and the date.

>  session 被一个 tmux client 连接后，会在屏幕底部展示状态栏
>  状态栏左边是 session 名称 `[0]`
>  中间是 session 中的 windows 列表，包含了 windows 的索引和其名称
>  右边是 pane 标题，默认是运行 tmux 的 host 的名字，以及时间和日期

![](https://github.com/tmux/tmux/wiki/images/tmux_status_line_diagram.png)
As new windows are opened, the window list grows - if there are too many windows to fit on the width of the terminal, a `<` or `>` will be added at the left or right or both to show there are hidden windows.

In the window list, the current window is marked with a `*` after the name, and the last window with a `-`.
>  window 列表中，当前 window 以 `*` 标记，上一个 window 以 `-` 标记
 
#### The prefix key
Once a tmux client is attached, any keys entered are forwarded to the program running in the active pane of the current window. For keys that control tmux itself, a special key must be pressed first - this is called the prefix key.
>  tmux client 连接到一个 session 后，任意键入的按键都会被发送给其当前 window 的 active pane 中的当前运行的程序
>  如果要控制 tmux 本身，需要键入一个前缀按键

The default prefix key is `C-b`, which means the `Ctrl` key and `b`. In tmux, modifier keys are shown by prefixing a key with `C-` for the control key, `M-` for the meta key (normally `Alt` on modern computers) and `S-` for the shift key. These may be combined together, so `C-M-x` means pressing the control key, meta key and `x` together.
>  默认的前缀按键是 `C-b` ，即 `Ctrl`  + `b`
>  tmux 中，修饰按键通过在键名前添加前缀表示，`C-` 表示控制键 `Ctrl` , `M-` 表示元键 `Alt` ，`S-` 表示 `Shift` ，这些符号可以组合使用，例如 `C-M-x` 

When the prefix key is pressed, tmux waits for another key press and that determines what tmux command is executed. Keys like this are shown here with a space between them: `C-b c` means first the prefix key `C-b` is pressed, then it is released and then the `c` key is pressed. Care must be taken to release the `Ctrl` key after pressing `C-b` if necessary - `C-b c` is different from `C-b C-c`.
>  按下前缀键后，tmux 等待另一个键的按下，它会决定 tmux 将执行什么命令
>  `C-b c` 表示 `Ctrl+b` + `c` ，注意 `C-b C-c` 是另一组按键

Pressing `C-b` twice sends the `C-b` key to the program running in the active pane.
>  `C-b C-b` 会将 `C-b` 发送给 active pane 中当前运行的程序

#### Help keys
Every default tmux key binding has a short description to help remember what the key does. A list of all the keys can be seen by pressing `C-b ?`.
>  每个默认的 tmux 快捷键都有简短的功能描述
>  通过 `C-b ?` 查看

![](https://github.com/tmux/tmux/wiki/images/tmux_list_keys.png)

`C-b ?` enters view mode to show text. A pane in view mode has its own key bindings which do not need the prefix key. These broadly follow _emacs(1)_. The most important are `Up`, `Down`, `C-Up`, `C-Down` to scroll up and down, and `q` to exit the mode. The line number of the top visible line together with the total number of lines is shown in the top right.
>  `C-b ?` 将进入 view mode 以显示文本
>  位于 view mode 的 pane 有自己的快捷键，不需要使用前缀键，这些快捷键大致遵循 emacs
>  最重要的是 `Up, Down, C-Up, C-Down` 来上下划动，以及 `q` 来退出
>  右上角会显示当前可见区域顶部的行号和总行数

Alternatively, the same list can be seen from the shell by running:

```
$ tmux lsk -N|more
```

>  `tmux lsk -N` 也会输出 tmux 的快捷键列表

`C-b /` shows the description of a single key - a prompt at the bottom of the terminal appears. Pressing a key will show its description in the same place. For example, pressing `C-b /` then `?` shows:

```
C-b ? List key bindings
```

>  `C-b /` 用于显示单个键的描述，例如 `C-b /` + `?` 会显示 `C-b ?` 的描述

#### Commands and flags
tmux has a large set of commands. These all have a name like `new-window` or `new-session` or `list-keys` and many also have a shorter alias like `neww` or `new` or `lsk`.
>  tmux 有许多命令，其名称都形如 `new-window, new-session, list-keys` ，它们也有短的别名，例如 `neww, new, lsk`

Any time a key binding is used, it runs one or more tmux commands. For example, `C-b c` runs the `new-window` command.
>  任意的 tmux 快捷键本质都是运行一个或者多个 tmux 命令，例如 `C-b c` 会运行 `new-window` 命令

Commands can also be used from the shell, as with `new-session` and `list-keys` above.
>  也可以直接在 shell 中运行 tmux 命令，例如 `tmux new-session, tmux list-keys`

Each command has zero or more flags, in the same way as standard Unix commands. Flags may or may not take a single argument themselves. In addition, commands may take additional arguments after the flags. Flags are passed after the command, for example to run the `new-session` command (alias `new`) with flags `-d` and `-n`:

```
$ tmux new-session -d -nmysession
```

>  每个 tmux 命令可以有零个或多个 flags，与标准的 Unix 命令形式相同
>  flags 本身可以接收零个或单个参数
>  另外，tmux 命令可以在 flags 之后接收额外的参数
>  flags 在命令之后传递，示例如上

All commands and their flags are documented in the tmux manual page.
>  tmux 手册中描述了所有的 tmux 命令和其 flags

This document focuses on the available key bindings, but commands are mentioned for information or where there is a useful flag. They can be entered from the shell or from the command prompt, described in the next section.
>  本文档关注于可用的快捷键，但也会提到一些命令和有用的 flags

#### The command prompt

![](https://github.com/tmux/tmux/wiki/images/tmux_command_prompt.png)

tmux has an interactive command prompt. This can be opened by pressing `C-b :` and appears instead of the status line, as shown in this screenshot.
>  tmux 本身提供了一个交互式命令提示符，通过快捷键 `C-b :` 打开

At the prompt, commands can be entered similarly to how they are at the shell. Output will either be shown for a short period in the status line, or switch the active pane into view mode.
>  在该 prompt 中，可以和在 shell 中一样键入 tmux 命令 (此时不需要 `tmux` 前缀)，输出可能会在 status line 之间显示，或者将 active pane 切换到 view mode 显示

By default, the command prompt uses keys similar to _emacs(1)_; however, if the `VISUAL` or `EDITOR` environment variables are set to something containing `vi` (such as `vi` or `vim` or `nvi`), then _vi(1)_-style keys are used instead.
>  command prompt 默认使用和 emacs 类似的键
>  如果 `VISUAL` 或 `EDITOR` 环境变量被设置为包含了 `vi` ，则使用 `vi` 风格的键

Multiple commands may be entered together at the command prompt by separating them with a semicolon (`;`). This is called a command sequence.
>  command prompt 可以接收多个命令输入，通过 `;` 分开

#### Attaching and detaching
Detaching from tmux means that the client exits and detaches from the outside terminal, returning to the shell and leaving the tmux session and any programs inside it running in the background. To detach tmux, the `C-b d` key binding is used. When tmux detaches, it will print a message with the session name:
>  从 tmux 分离意味着 client 进程断开和 session 的连接，并且 outside terminal 的 client 进程会退出
>  outside terminal 会回到 shell 程序，而 tmux session 和其中运行的任意程序会保留在后台运行
>  `C-b d` (等价于 `tmux detach`) 用于分离 tmux，分离后，tmux 会打印带有 session 名称的消息

```
[detached (from session mysession)]
```

The `attach-session` command attaches to an existing session. Without arguments, it will attach to the most recently used session that is not already attached:
>  `attach-session` 命令用于连接到一个现存的 session
>  没有参数时，它会连接到最近连接过的 session

```
$ tmux attach
```

Or `-t` gives the name of a session to attach to:

```
$ tmux attach -tmysession
```

>  `-t` 指定要连接的目标 session

By default, attaching to a session does not detach any other clients attached to the same session. The `-d` flag does this:

```
$ tmux attach -dtmysession
```

>  默认情况下，连接到某个 session 不会断开该 session 和其他 client 的连接
>  如果指定了 `-d` ，则会断开其他 client 的连接

The `new-session` command has a `-A` flag to attach to an existing session if it exists, or create a new one if it does not. For a session named `mysession`:

```
$ tmux new -Asmysession
```

>  `new-session` 的 `-A` 会在 `-s` 指定的 session 存在时直接附加到现存 session，不存在则创建一个

The `-D` flag may be added to make `new-session` also behave like `attach-session` with `-d` and detach any other clients attached to the session.
>  `-D` 类似，会让 session 和其他 client 断开连接

#### Listing sessions
The `list-session` command (alias `ls`) shows a list of available sessions that can be attached. This shows four sessions called `1`, `2`, `myothersession` and `mysession`:

```
$ tmux ls
1: 3 windows (created Sat Feb 22 11:44:51 2020)
2: 1 windows (created Sat Feb 22 11:44:51 2020)
myothersession: 2 windows (created Sat Feb 22 11:44:51 2020)
mysession: 1 windows (created Sat Feb 22 11:44:51 2020)
```

#### Killing tmux entirely
If there are no sessions, windows or panes inside tmux, the server will exit. It can also be entirely killed using the `kill-server` command. For example, at the command prompt:
>  tmux server 会在没有 session, windows, panes 时退出
>  可以通过 `kill-server` 命令手动结束 server

```
:kill-server
```

#### Creating new windows

![](https://github.com/tmux/tmux/wiki/images/tmux_new_windows.png)

A new window can be created in an attached session with the `C-b c` key binding which runs the `new-window` command. The new window is created at the first available index - so the second window will have index 1. The new window becomes the current window of the session.

If there are any gaps in the window list, they are filled by new windows. So if there are windows with indexes 0 and 2, the next new window will be created as index 1.

The `new-window` command has some useful flags which can be used with the command prompt:

- The `-d` flag creates the window, but does not make it the current window.
- `-n` allows a name for the new window to be given. For example using the command prompt to create a window called `mynewwindow` without making it the current window:
    
    ```
    :neww -dnmynewwindow
    ```
    
- The `-t` flag specifies a target for the window. Command targets have a special syntax, but for simple use with `new-window` it is enough just to give a window index. This creates a window at index 999:
    
    ```
    :neww -t999
    ```
    

A command to be run in the new window may be given to `new-window` in the same way as `new-session`. For example to create a new window running _top(1)_:

```
:neww top
```

#### Splitting the window

![](https://github.com/tmux/tmux/wiki/images/tmux_split_h.png)

A pane is created by splitting a window. This is done with the `split-window` command which is bound to two keys by default:

- `C-b %` splits the current pane into two horizontally, producing two panes next to each other, one on the left and one on the right.
    
- `C-b "` splits the current pane into two vertically, producing two panes one above the other.
    

Each time a pane is split into two, each of those panes may be split again using the same key bindings, until the pane becomes too small.

![](https://github.com/tmux/tmux/wiki/images/tmux_split_v.png)

`split-window` has several useful flags:

- `-h` does a horizontal split and `-v` a vertical split.
    
- `-d` does not change the active pane to the newly created pane.
    
- `-f` makes a new pane spanning the whole width or height of the window instead of being constrained to the size of the pane being split.
    
- `-b` puts the new pane to the left or above of the pane being split instead of to the right or below.
    

A command to be run in the new pane may be given to `split-window` in the same way as `new-session` and `new-window`.

#### Changing the current window

[](https://github.com/tmux/tmux/wiki/Getting-Started#changing-the-current-window)

There are several key bindings to change the current window of a session:

- `C-b 0` changes to window 0, `C-b 1` to window 1, up to window `C-b 9` for window 9.
    
- `C-b '` prompts for a window index and changes to that window.
    
- `C-b n` changes to the next window in the window list by number. So pressing `C-b n` when in window 1 will change to window 2 if it exists.
    
- `C-b p` changes to the previous window in the window list by number.
    
- `C-b l` changes to the last window, which is the window that was last the current window before the window that is now.
    

These are all variations of the `select-window` command.

#### Changing the active pane

[](https://github.com/tmux/tmux/wiki/Getting-Started#changing-the-active-pane)

The active pane can be changed between the panes in a window with these key bindings:

- `C-b Up`, `C-b Down`, `C-b Left` and `C-b Right` change to the pane above, below, left or right of the active pane. These keys wrap around the window, so pressing `C-b Down` on a pane at the bottom will change to a pane at the top.

![](https://github.com/tmux/tmux/wiki/images/tmux_display_panes.png)

- `C-b q` prints the pane numbers and their sizes on top of the panes for a short time. Pressing one of the number keys before they disappear changes the active pane to the chosen pane, so `C-b q 1` will change to pane number 1.
    
- `C-b o` moves to the next pane by pane number and `C-b C-o` swaps that pane with the active pane, so they exchange positions and sizes in the window.
    

These use the `select-pane` and `display-panes` commands.

Pane numbers are not fixed, instead panes are numbered by their position in the window, so if the pane with number 0 is swapped with the pane with number 1, the numbers are swapped as well as the panes themselves.

#### Choosing sessions, windows and panes

[](https://github.com/tmux/tmux/wiki/Getting-Started#choosing-sessions-windows-and-panes)

![](https://github.com/tmux/tmux/wiki/images/tmux_choose_tree1.png)tmux includes a mode where sessions, windows or panes can be chosen from a tree, this is called tree mode. It can be used to browse sessions, windows and panes; to change the attached session, the current window or active pane; to kill sessions, windows and panes; or apply a command to several at once by tagging them.

There are two key bindings to enter tree mode: `C-b s` starts showing only sessions and with the attached session selected; `C-b w` starts with sessions expanded so windows are shown and with the current window in the attached session selected.

Tree mode splits the window into two sections: the top half has a tree of sessions, windows and panes and the bottom half has a preview of the area around the cursor in each pane. For sessions the preview shows the active panes in as many windows will fit; for windows as many panes as will fit; and for panes only the selected pane.

![](https://github.com/tmux/tmux/wiki/images/tmux_choose_tree2.png)

Keys to control tree mode do not require the prefix. The list may be navigated with the `Up` and `Down` keys. `Enter` changes to the selected item (it becomes the attached session, current window or active pane) and exits the mode. `Right` expands the item if possible - sessions expand to show their windows and windows to show their panes. `Left` collapses the item to hide any windows or panes. `O` changes the order of the items and `q` exits tree mode.

Items in the tree are tagged by pressing `t` and untagged by pressing `t` again. Tagged items are shown in bold and with `*` after their name. All tagged items may be untagged by pressing `T`. Tagged items may be killed together by pressing `X`, or a command applied to them all by pressing `:` for a prompt.

Each item in the tree has as shortcut key in brackets at the start of the line. Pressing this key will immediately choose that item (as if it had been selected and `Enter` pressed). The first ten items are keys `0` to `9` and after that keys `M-a` to `M-z` are used.

This is a list of the keys available in tree mode without pressing the prefix key:

|Key|Function|
|---|---|
|`Enter`|Change the attached session, current window or active pane|
|`Up`|Select previous item|
|`Down`|Select next item|
|`Right`|Expand item|
|`Left`|Collapse item|
|`x`|Kill selected item|
|`X`|Kill tagged items|
|`<`|Scroll preview left|
|`>`|Scroll preview right|
|`C-s`|Search by name|
|`n`|Repeat last search|
|`t`|Toggle if item is tagged|
|`T`|Tag no items|
|`C-t`|Tag all items|
|`:`|Prompt for a command to run for the selected item or each tagged item|
|`O`|Change sort field|
|`r`|Reverse sort order|
|`v`|Toggle preview|
|`q`|Exit tree mode|

Tree mode is activated with the `choose-tree` command.

#### Detaching other clients

[](https://github.com/tmux/tmux/wiki/Getting-Started#detaching-other-clients)

![](https://github.com/tmux/tmux/wiki/images/tmux_choose_client.png)

A list of clients is available by pressing `C-b D` (that is, `C-b S-d`). This is similar to tree mode and is called client mode.

Each client is shown in the list in the top half with its name, attached session, size and the time and date when it was last used; the bottom half has a preview of the selected client with as much of its status line as will fit.

The movement and tag keys are the same as tree mode, but others are different, for example the `Enter` key detaches the selected client.

This is a list of the keys in client mode without the movement and tagging keys that are the same as tree mode:

|Key|Function|
|---|---|
|`Enter`|Detach selected client|
|`d`|Detach selected client, same as `Enter`|
|`D`|Detach tagged clients|
|`x`|Detach selected client and try to kill the shell it was started from|
|`X`|Detach tagged clients and try to kill the shells they were started from|

Other than using client mode, the `detach-client` command has a `-a` flag to detach all clients other than the attached client.

#### Killing a session, window or pane

[](https://github.com/tmux/tmux/wiki/Getting-Started#killing-a-session-window-or-pane)

Pressing `C-b &` prompts for confirmation then kills (closes) the current window. All panes in the window are killed at the same time. `C-b x` kills only the active pane. These are bound to the `kill-window` and `kill-pane` commands.

The `kill-session` command kills the attached session and all its windows and detaches the client. There is no key binding for `kill-session` but it can be used from the command prompt or the `:` prompt in tree mode.

#### Renaming sessions and windows

[](https://github.com/tmux/tmux/wiki/Getting-Started#renaming-sessions-and-windows)

![](https://github.com/tmux/tmux/wiki/images/tmux_rename_session.png)

`C-b $` will prompt for a new name for the attached session. This uses the `rename-session` command. Likewise, `C-b ,` prompts for a new name for the current window, using the `rename-window` command.

#### Swapping and moving

[](https://github.com/tmux/tmux/wiki/Getting-Started#swapping-and-moving)

tmux allows panes and windows to be swapped with the `swap-pane` and `swap-window` commands.

To make swapping easy, a single pane can be marked. There is one marked pane across all sessions. The `C-b m` key binding toggles whether the active pane in the current window in the attached session is the marked pane. `C-b M` clears the marked pane entirely so that no pane is marked. The marked pane is shown by a green background to its border and the window containing the marked pane has an `M` flag in the status line.

![](https://github.com/tmux/tmux/wiki/images/tmux_marked_pane.png)

Once a pane is marked, it can be swapped with the active pane in the current window with the `swap-pane` command, or the window containing the marked pane can be swapped with the current window using the `swap-window` command. For example, using the command prompt:

```
:swap-pane
```

Panes can additionally be swapped with the pane above or below using the `C-b {` and `C-b }` key bindings.

Moving windows uses the `move-window` command or the `C-b .` key binding. Pressing `C-b .` will prompt for a new index for the current window. If a window already exists at the given index, an error will be shown. An existing window can be replaced by using the `-k` flag - to move a window to index 999:

```
:move-window -kt999
```

If there are gaps in the window list, the indexes can be renumbered with the `-r` flag to `move-window`. For example, this will change a window list of 0, 1, 3, 999 into 0, 1, 2, 3:

```
:movew -r
```

#### Resizing and zooming panes

[](https://github.com/tmux/tmux/wiki/Getting-Started#resizing-and-zooming-panes)

Panes may be resized in small steps with `C-b C-Left`, `C-b C-Right`, `C-b C-Up` and `C-b C-Down` and in larger steps with `C-b M-Left`, `C-b M-Right`, `C-b M-Up` and `C-b M-Down`. These use the `resize-pane` command.

A single pane may be temporarily made to take up the whole window with `C-b z`, hiding any other panes. Pressing `C-b z` again puts the pane and window layout back to how it was. This is called zooming and unzooming. A window where a pane has been zoomed is marked with a `Z` in the status line. Commands that change the size or position of panes in a window automatically unzoom the window.

#### Window layouts

[](https://github.com/tmux/tmux/wiki/Getting-Started#window-layouts)

![](https://github.com/tmux/tmux/wiki/images/tmux_tiled.png)

The panes in a window may be automatically arranged into one of several named layouts, these may be rotated between with the `C-b Space` key binding or chosen directly with `C-b M-1`, `C-b M-2` and so on.

The available layouts are:

|Name|Key|Description|
|---|---|---|
|even-horizontal|`C-b M-1`|Spread out evenly across|
|even-vertical|`C-b M-2`|Spread out evenly up and down|
|main-horizontal|`C-b M-3`|One large pane at the top, the rest spread out evenly across|
|main-vertical|`C-b M-4`|One large pane on the left, the rest spread out evenly up and down|
|tiled|`C-b M-5`|Tiled in the same number of rows as columns|

#### Copy and paste

[](https://github.com/tmux/tmux/wiki/Getting-Started#copy-and-paste)

tmux has its own copy and paste system. A piece of copied text is called a paste buffer. Text is copied using copy mode, entered with `C-b [`, and the most recently copied text is pasted into the active pane with `C-b ]`.

![](https://github.com/tmux/tmux/wiki/images/tmux_copy_mode.png)

Paste buffers can be given names but by default they are assigned a name by tmux, such as `buffer0` or `buffer1`. Buffers like this are called automatic buffers and at most 50 are kept - once there are 50 buffers, the oldest is removed when another is added. If a buffer is given a name, it is called a named buffer; named buffers are not deleted no matter how many there are.

It is possible to configure tmux to send any copied text to the system clipboard: [this document](https://github.com/tmux/tmux/wiki/Clipboard) explains the different ways to configure this.

Copy mode freezes any output in a pane and allows text to be copied. View mode (described earlier) is a read-only form of copy mode.

Like the command prompt, copy mode uses keys similar to _emacs(1)_; however, if the `VISUAL` or `EDITOR` environment variables are set to something containing `vi`, then _vi(1)_-style keys are used instead. The following keys are some of those available in copy mode with _emacs(1)_ keys:

|Key|Action|
|---|---|
|`Up`, `Down`, `Left`, `Right`|Move the cursor|
|`C-Space`|Start a selection|
|`C-w`|Copy the selection and exit copy mode|
|`q`|Exit copy mode|
|`C-g`|Stop selecting without copying, or stop searching|
|`C-a`|Move the cursor to the start of the line|
|`C-e`|Move the cursor to the end of the line|
|`C-r`|Search interactively backwards|
|`M-f`|Move the cursor to the next word|
|`M-b`|Move the cursor to the previous word|

A full list of keys for both _vi(1)_ and _emacs(1)_ is [available in the manual page](https://man.openbsd.org/tmux#WINDOWS_AND_PANES).

![](https://github.com/tmux/tmux/wiki/images/tmux_buffer_mode.png)

Once some text is copied, the most recent may be pasted with `C-b ]` or an older buffer pasted by using buffer mode, entered with `C-b =`. Buffer mode is similar to client mode and tree mode and offers a list of buffers together with a preview of their contents. As well as the navigation and tagging keys used in tree mode and client mode, buffer mode supports the following keys:

|Key|Function|
|---|---|
|`Enter`|Paste selected buffer|
|`p`|Paste selected buffer, same as `Enter`|
|`P`|Paste tagged buffers|
|`d`|Delete selected buffer|
|`D`|Delete tagged buffers|

A buffer may be renamed using the `set-buffer` command. The `-b` flag gives the existing buffer name and `-n` the new name. This converts it into a named buffer. For example, to rename `buffer0` to `mybuffer` from the command prompt:

```
:setb -bbuffer0 -nmybuffer
```

`set-buffer` can also be used to create buffers. To create a buffer called `foo` with text `bar`:

```
:setb -bfoo bar
```

`load-buffer` will load a buffer from a file:

```
:loadb -bbuffername ~/a/file
```

`set-buffer` or `load-buffer` without `-b` creates an automatic buffer.

An existing buffer can be saved to a file with `save-buffer`:

```
:saveb -bbuffer0 ~/saved_buffer
```

#### Finding windows and panes

[](https://github.com/tmux/tmux/wiki/Getting-Started#finding-windows-and-panes)

![](https://github.com/tmux/tmux/wiki/images/tmux_find_window.png)

`C-b f` prompts for some text and then enters tree mode with a filter to show only panes where that text appears in the visible content or title of the pane or in the window name. If panes are found, only those panes appear in the tree, and the text `filter: active` is shown above the preview. If no panes are found, all panes are shown in the tree and the text `filter: no matches` appears above the preview.

#### Using the mouse

[](https://github.com/tmux/tmux/wiki/Getting-Started#using-the-mouse)

tmux has rich support for the mouse. It can be used to change the active pane or window, to resize panes, to copy text, or to choose items from menus.

Support for the mouse is enabled with the `mouse` option; options and the configuration file are described in detail in the next section. To turn the mouse on from the command prompt, use the `set-option` command:

```
:set -g mouse on
```

Once the mouse is enabled:

![](https://github.com/tmux/tmux/wiki/images/tmux_pane_menu.png)

- Pressing the left button on a pane will make that pane the active pane.
    
- Pressing the left button on a window name on the status line will make that the current window.
    
- Dragging with the left button on a pane border resizes the pane.
    
- Dragging with the left button inside a pane selects text; the selected text is copied when the mouse is released.
    
- Pressing the right button on a pane opens a menu with various commands. When the mouse button is released, the selected command is run with the pane as target. Each menu item also has a key shortcut shown in brackets.
    
- Pressing the right button on a window or on the session name on the status line opens a similar menu for the window or session.
    

### Configuring tmux
#### The configuration file
When the tmux server is started, tmux runs a file called `.tmux.conf` in the user's home directory. This file contains a list of tmux commands which are executed in order. It is important to note that `.tmux.conf` is _only_ run when the server is started, not when a new session is created.
>  tmux server 被启动时，tmux 会运行用户家目录下的 `.tmux.conf` 文件
>  该文件包含了一系列 tmux 命令，它们会被顺序执行
>  注意 `.tmux.conf` 只会在 tmux server 启动时被运行，在新的会话被创建时不会运行

A different configuration file may be run from `.tmux.conf` or from a running tmux server using the `source-file` command, for example to run `.tmux.conf` again from a running server using the command prompt:

```
:source ~/.tmux.conf
```

Commands in a configuration file appear one per line. Any lines starting with `#` are comments and are ignored:

```
# This is a comment - the command below turns the status line off
set -g status off
```

Lines in the configuration file are processed similar to the shell, for example:

- Arguments may be enclosed in `'` or `"` to include spaces, or spaces may be escaped. These four lines do the same thing:
    
    ```
    set -g status-left "hello word"
    set -g status-left "hello\ word"
    set -g status-left 'hello word'
    set -g status-left hello\ word
    ```
    
- But escaping doesn't happen inside `'`s. The string here is `hello\ world` not `hello world`:
    
    ```
    set -g status-left 'hello\ word'
    ```
    
- `~` is expanded to the home directory (except inside `'`s):
    
    ```
    source ~/myfile
    ```
    
- Environment variables can be set and are also expanded (but not inside `'`s):
    
    ```
    MYFILE=myfile
    source "~/$MYFILE"
    ```
    
    Any variables set in the configuration file will be passed on to new panes created inside tmux.
    
- A few special characters like `\n` (newline) and `\t` (tab) are replaced. A literal `\` must be given as `\\`.
    

Although tmux configuration files have some features similar to the shell, they are not shell scripts and cannot use shell constructs like `$()`.

#### Key bindings

[](https://github.com/tmux/tmux/wiki/Getting-Started#key-bindings)

tmux key bindings are changed using the `bind-key` and `unbind-key` commands. Each key binding in tmux belongs to a named key table. There are four default key tables:

- The `root` table contains key bindings for keys pressed without the prefix key.
    
- The `prefix` table contains key bindings for keys pressed after the prefix key, like those mentioned so far in this document.
    
- The `copy-mode` table contains key bindings for keys used in copy mode with _emacs(1)_-style keys.
    
- The `copy-mode-vi` table contains key bindings for keys used in copy mode with _vi(1)_-style keys.
    

All the key bindings or those for a single table can be listed with the `list-keys` command. By default, this shows the keys as a series of `bind-key` commands. The `-T` flag gives the key table to show and the `-N` flag shows the key help, like the `C-b ?` key binding.

For example to list only keys in the `prefix` table:

```
$ tmux lsk -Tprefix
bind-key    -T prefix C-b     send-prefix
bind-key    -T prefix C-o     rotate-window
...
```

Or:

```
$ tmux lsk -Tprefix -N
C-b     Send the prefix key
C-o     Rotate through the panes
...
```

`bind-key` commands can be used to set a key binding, either interactively or most commonly from the configuration file. Like `list-keys`, `bind-key` has a `-T` flag for the key table to use. If `-T` is not given, the key is put in the `prefix` table; the `-n` flag is a shorthand for `-Troot` to use the `root` table.

For example, the `list-keys` command shows that `C-b 9` changes to window 9 using the `select-window` command:

```
$ tmux lsk -Tprefix 9
bind-key -T prefix 9 select-window -t :=9
```

A similar key binding to make `C-b M-0` change to window 10 can be added like this:

```
bind M-0 selectw -t:=10
```

The `-t` flag to `select-window` specifies the target window. In this example, the `:` means the target is a window and `=` means the name must match `10` exactly. Targets are documented further in the [COMMANDS section of the manual page](https://man.openbsd.org/tmux#COMMANDS).

The `unbind-key` command removes a key binding. Like `bind-key` it has `-T` and `-n` flags for the key table. It is not necessary to remove a key binding before binding it again, `bind-key` will replace any existing key binding. `unbind-key` is necessary only to completely remove a key binding:

```
unbind M-0
```

#### Copy mode key bindings

[](https://github.com/tmux/tmux/wiki/Getting-Started#copy-mode-key-bindings)

Copy mode key bindings are set in the `copy-mode` and `copy-mode-vi` key tables. Copy mode has a separate set of commands which are passed using the `-X` flag to the `send-keys` command, for example the copy mode `start-of-line` command moves the cursor to the start of the line and is bound to `C-a` in the `copy-mode` key table:

```
$ tmux lsk -Tcopy-mode C-a
bind-key -T copy-mode C-a send-keys -X start-of-line
```

A full list of copy mode commands is [available in the manual page](https://man.openbsd.org/tmux#WINDOWS_AND_PANES). Here is a selection:

|Command|_emacs(1)_|_vi(1)_|Description|
|---|---|---|---|
|begin-selection|C-Space|Space|Start selection|
|cancel|q|q|Exit copy mode|
|clear-selection|C-g|Escape|Clear selection|
|copy-pipe|||Copy and pipe to the command in the first argument|
|copy-selection-and-cancel|M-w|Enter|Copy the selection and exit copy mode|
|cursor-down|Down|j|Move the cursor down|
|cursor-left|Left|h|Move the cursot left|
|cursor-right|Right|l|Move the cursor right|
|cursor-up|Up|k|Move the cursor up|
|end-of-line|C-e|$|Move the cursor to the end of the line|
|history-bottom|M->|G|Move to the bottom of the history|
|history-top|M-<|g|Move to the top of the history|
|middle-line|M-r|M|Move to middle line|
|next-word-end|M-f|e|Move to the end of the next word|
|page-down|PageDown|C-f|Page down|
|page-up|PageUp|C-b|Page up|
|previous-word|M-b|b|Move to the previous word|
|rectangle-toggle|R|v|Toggle rectangle selection|
|search-again|n|n|Repeat the last search|
|search-backward||?|Search backwards, the first argument is the search term|
|search-backward-incremental|C-r||Search backwards incrementally, usually used with the `-i` flag to `command-prompt`|
|search-forward||/|Search forwards, the first argument is the search term|
|search-forward-incremental|C-s||Search forwards incrementally|
|search-reverse|N|N|Repeat the last search but reverse the direction|
|start-of-line|C-a|0|Move to the start of the line|

#### Types of option

[](https://github.com/tmux/tmux/wiki/Getting-Started#types-of-option)

tmux is configured by setting options. There are several types of options:

- Server options which affect the entire server.
    
- Session options which affect one or all sessions.
    
- Window options which affect one or all windows.
    
- Pane options which affect one or all panes.
    
- User options which are not used by tmux but are reserved for the user.
    

Session and window options have both a global set of options and a set for each session or window. If the option is not present in the session or window set, the global option is used. Pane options are similar except the window options are also checked.

When configuring tmux, it is most common to set server options and global session or window options. This document only covers these.

#### Showing options

[](https://github.com/tmux/tmux/wiki/Getting-Started#showing-options)

Options are displayed using the `show-options` command. The `-g` flag shows global options. It can show server, session or window options:

- `-s` shows server options:

```
$ tmux show -s
backspace C-?
buffer-limit 50
...
```

- `-g` with no other flags shows global session options:

```
$ tmux show -g
activity-action other
assume-paste-time 1
...
```

- `-g` and `-w` together show global window options:

```
$ tmux show -wg
aggressive-resize off
allow-rename off
...
```

An individual option value may be shown by giving its name to `show-option`. When an option name is given, it is not necessary to give `-s` or `-w` because tmux can work it out from the option name. For example, to show the `status` option:

```
$ tmux show -g status
status on
```

#### Changing options

[](https://github.com/tmux/tmux/wiki/Getting-Started#changing-options)

Options are set or unset using the `set-option` command. Like `show-option`, it is not necessary to give `-s` or `-w` because tmux can work out it out from the option name. `-g` is necessary to set global session or window options; for server options it does nothing.

To set the `status` option:

```
set -g status off
```

Or the `default-terminal` option:

```
set -s default-terminal 'tmux-256color'
```

The `-u` flag unsets an option. Unsetting a global option restores it to its default value, for example:

```
set -gu status
```

#### Formats

[](https://github.com/tmux/tmux/wiki/Getting-Started#formats)

Many options make use of formats. Formats provide a powerful syntax to configure how text appears, based on various attributes of the tmux server, a session, window or pane. Formats are enclosed in `#{}` in string options or as a single uppercase letter like `#F`. This is the default `status-right` with several formats:

```
$ tmux show -s status-right
status-right "#{?window_bigger,[#{window_offset_x}#,#{window_offset_y}] ,}\"#{=21:pane_title}\" %H:%M %d-%b-%y"
```

Formats are described [in this document](https://github.com/tmux/tmux/wiki/Formats) and [in the manual page](https://man.openbsd.org/tmux#FORMATS).

#### Embedded commands

[](https://github.com/tmux/tmux/wiki/Getting-Started#embedded-commands)

Some options may contain embedded shell commands. This is limited to the status line options such as `status-left`. Embedded shell commands are enclosed in `#()`. They can either:

1. Print a line and exit, in which case the line will be shown in the status line and the command run at intervals to update it. For example:
    
    ```
    set -g status-left '#(uptime)'
    ```
    
    The maximum interval is set by the `status-interval` option but commands may also be run sooner if tmux needs. Commands will not be run more than once a second.
    
2. Stay running and print a line whenever needed, for example:
    
    ```
    set -g status-left '#(while :; do uptime; sleep 1; done)'
    ```
    

Note that is it not usually necessary to use an embedded command for the date and time since tmux will expand the date formats like `%H` and `%S` itself in the status line options. If a command like _date(1)_ is used, any `%`s must be doubled as `%%`.

#### Colours and styles

[](https://github.com/tmux/tmux/wiki/Getting-Started#colours-and-styles)

tmux allows the colour and attribute of text to be configured with a simple syntax, this is known as the style. There are two places styles appear:

- In options, such as `status-style`.
    
- Enclosed in `#[]` in an option value, this is called an embedded style (see the next section).
    

A style has a number of terms separated by spaces or commas, the most useful are:

- `default` uses the default colour; this must appear on its own. The default colour is often set by another option, for example for embedded styles in the `status-left` option, it is `status-style`.
    
- `bg` sets the background colour. The colour is also given, for example `bg=red`.
    
- `fg` sets the foreground colour. Like `bg`, the colour is given: `fg=green`.
    
- `bright` or `bold`, `underscore`, `reverse`, `italics` set the attributes. These appear alone, such as: `bright,reverse`.
    

Colours may be one of `black`, `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `white` for the standard terminal colours; `brightred`, `brightyellow` and so on for the bright variants; `colour0` to `colour255` for the colours from the 256-colour palette; `default` for the default colour; or a hexadecimal RGB colour such as `#882244`.

The remaining style terms are described [in the manual page](https://man.openbsd.org/tmux#STYLES).

For example, to set the status line background to blue using the `status-style` option:

```
set -g status-style 'bg=blue'
```

#### Embedded styles

[](https://github.com/tmux/tmux/wiki/Getting-Started#embedded-styles)

Embedded styles are included inside another option in between `#[` and `]`. Each changes the style of following text until the next embedded style or the end of the text.

For example, to put some text in red and blue in `status-left`:

```
set -g status-left 'default #[fg=red] red #[fg=blue] blue'
```

Because this is long it is also necessary to also increase the `status-left-length` option:

```
set -g status-left-length 100
```

Or embedded styles can be used conditionally, for example to show `P` in red if the prefix has been pressed or in the default style if not:

```
set -g status-left '#{?client_prefix,#[bg=red],}P#[default] [#{session_name}] '
```

#### List of useful options

[](https://github.com/tmux/tmux/wiki/Getting-Started#list-of-useful-options)

This is a short list of the most commonly used tmux options, apart from style options:

|Option|Type|Description|
|---|---|---|
|`base-index`|session|If set, then windows indexes start from this instead of from 0|
|`buffer-limit`|server|The maximum number of automatic buffers to keep, the default is 50|
|`default-terminal`|server|The default value of the `TERM` environment variable inside tmux|
|`display-panes-time`|window|The time in milliseconds the pane numbers are shown for `C-b q`|
|`display-time`|session|The time in milliseconds for which messages on the status line are shown|
|`escape-time`|server|The time tmux waits after receiving an `Escape` key to see if it is part of a longer key sequence|
|`focus-events`|server|Whether focus key sequences are sent by tmux when the active pane changes and when received from the outside terminal if it supports them|
|`history-limit`|session|The maximum number of lines kept in the history for each pane|
|`mode-keys`|window|Whether _emacs(1)_ or _vi(1)_ key bindings are used in copy mode|
|`mouse`|session|If the mouse is enabled|
|`pane-border-status`|window|Whether a status line appears in every pane border: `top` or `bottom`|
|`prefix`|session|The prefix key, the default is `C-b`|
|`remain-on-exit`|window|Whether panes are automatically killed when the program running in the exits|
|`renumber-windows`|session|If `on`, windows are automatically renumbered to close any gaps in the window list|
|`set-clipboard`|server|Whether tmux should attempt to set the external _X(7)_ clipboard when text is copied and if the outside terminal supports it|
|`set-titles`|session|If `on`, tmux will set the title of the outside terminal|
|`status`|session|Whether the status line if visible|
|`status-keys`|session|Whether _emacs(1)_ or _vi(1)_ key bindings are used at the command prompt|
|`status-interval`|session|The maximum time in seconds before the status line is redrawn|
|`status-position`|session|The position of the status line: `top` or `bottom`|
|`synchronize-panes`|window|If `on`, typing in any pane in the window is sent to all panes in the window - care should be taken with this option!|
|`terminal-overrides`|server|Any capabilities tmux should override from the `TERM` given for the outside terminal|

#### List of style and format options

[](https://github.com/tmux/tmux/wiki/Getting-Started#list-of-style-and-format-options)

This is a list of the most commonly used tmux style and format options:

|Option|Type|Description|
|---|---|---|
|`display-panes-active-colour`|session|The style of the active pane number for `C-b q`|
|`display-panes-colour`|session|The style of the pane numbers, apart from the active pane for`C-b q`|
|`message-style`|session|The style of messages shown on the status line and of the command prompt|
|`mode-style`|window|The style of the selection in copy mode|
|`pane-active-border-style`|window|The style of the active pane border|
|`pane-border-format`|window|The format of text that appears in the pane border status line if `pane-border-status` is set|
|`pane-border-style`|window|The style of the pane borders, apart from the active pane|
|`status-left-length`|session|The maximum length of the status line left|
|`status-left-style`|session|The style of the status line left|
|`status-left`|session|The format of the text in the status line left|
|`status-right-length`|session|The maximum length of the status line right|
|`status-right-style`|session|The style of the status line right|
|`status-right`|session|The format of the text in the status line right|
|`status-style`|session|The style of the status line as a whole, parts may be overridden by more specific options like `status-left-style`|
|`window-active-style`|window|The style of the default colour in the active pane in the window|
|`window-status-current-format`|window|The format of the current window in the window list|
|`window-status-current-style`|window|The style of the current window in the window list|
|`window-status-format`|window|The format of windows in the window list, apart from the current window|
|`window-status-separator`|window|The separator between windows in the window list|
|`window-status-style`|window|The style of windows in the window list, apart from the current window|
|`window-style`|window|The style of the default colour of panes in the window, apart from the active pane|

### Common configuration changes

[](https://github.com/tmux/tmux/wiki/Getting-Started#common-configuration-changes)

This section shows examples of some common configuration changes for `.tmux.conf`.

#### Changing the prefix key

[](https://github.com/tmux/tmux/wiki/Getting-Started#changing-the-prefix-key)

The prefix key is set by the `prefix` option. The `C-b` key is also bound to the `send-prefix` command in the prefix key table so pressing `C-b` twice sends it through to the active pane. To change to `C-a`:

```
set -g prefix C-a
unbind C-b
bind C-a send-prefix
```

#### Customizing the status line

[](https://github.com/tmux/tmux/wiki/Getting-Started#customizing-the-status-line)

There are many options for customizing the status line. The simplest options are:

- Turn the status line off: `set -g status off`
    
- Move it to the top: `set -g status-position top`
    
- Set the background colour to red: `set -g status-style bg=red`
    
- Change the text on the right to the time only: `set -g status-right '%H:%M'`
    
- Underline the current window: `set -g window-status-current-style 'underscore'`
    

#### Configuring the pane border

[](https://github.com/tmux/tmux/wiki/Getting-Started#configuring-the-pane-border)

The pane border colours may be set:

```
set -g pane-border-style fg=red
set -g pane-active-border-style 'fg=red,bg=yellow'
```

Each pane may be given a status line with the `pane-border-status` option, for example to show the pane title in bold:

```
set -g pane-border-status top
set -g pane-border-format '#[bold]#{pane_title}#[default]'
```

#### _vi(1)_ key bindings

[](https://github.com/tmux/tmux/wiki/Getting-Started#vi1-key-bindings)

tmux supports key bindings based on _vi(1)_ for copy mode and the command prompt. There are two options that set the key bindings:

1. `mode-keys` sets the key bindings for copy mode. If this is set to `vi`, then the `copy-mode-vi` key table is used in copy mode; otherwise the `copy-mode` key table is used.
    
2. `status-keys` sets the key bindings for the command prompt.
    

If either of the `VISUAL` or `EDITOR` environment variables are set to something containing `vi` (such as `vi`, `vim`, `nvi`) when the tmux server is first started, both of these options are set to `vi`.

To set both to use _vi(1)_ keys:

```
set -g mode-keys vi
set -g status-keys vi
```

#### Mouse copying behaviour

[](https://github.com/tmux/tmux/wiki/Getting-Started#mouse-copying-behaviour)

When dragging the mouse to copy text, tmux copies and exits copy mode when the mouse button is released. Alternative behaviours are configured by changing the `MouseDragEnd1Pane` key binding. The three most useful are:

1. Do not copy or clear the selection or exit copy mode when the mouse is released. The keyboard must be used to copy the selection:

```
unbind -Tcopy-mode MouseDragEnd1Pane
```

2. Copy and clear the selection but do not exit copy mode:

```
bind -Tcopy-mode MouseDragEnd1Pane send -X copy-selection
```

3. Copy but do not clear the selection:

```
bind -Tcopy-mode MouseDragEnd1Pane send -X copy-selection-no-clear
```

### Other features

[](https://github.com/tmux/tmux/wiki/Getting-Started#other-features)

tmux has a large set of features and commands not mentioned in this document, many allowing powerful scripting. Here is a list of some that may be worth further reading:

- Alerts: `monitor-activity`, `monitor-bell`, `monitor-silence`, `activity-action`, `bell-action` and other options.
    
- Options for individual session, windows and panes.
    
- Moving panes with `join-pane` and `break-pane`.
    
- Sending keys to panes with `send-keys`.
    
- The command prompt `history-file` option.
    
- Saved layout strings with `select-layout`.
    
- Command sequences (separated by `;`): `select-window; kill-window`.
    
- Configuration file syntax: `{}`, `%if` and so on.
    
- Mouse key bindings: `MouseDown1Pane` and so on.
    
- Locking: `lock-command`, `lock-after-time` and other options.
    
- Capturing pane content with `capture-pane` and piping with `pipe-pane`.
    
- Linking windows: the `link-window` command.
    
- Session groups: the `-t` flag to `new-session`.
    
- Respawing window and panes with `respawn-window` and `respawn-pane`.
    
- Custom menus with the `display-menu` command and custom prompts with `command-prompt` and `confirm-before`.
    
- Different key tables: `bind-key` and the `-T` flag to `switch-client`.
    
- Empty panes: the `split-window` with an empty command and `-I` to `display-message`.
    
- Hooks: `set-hook` and `show-hooks`.
    
- Synchronization for scripts with `wait-for`.