---
edition: "10"
---
## Summary of GDB
The purpose of a debugger such as GDB is to allow you to see what is going on “inside” another program while it executes—or what another program was doing at the moment it crashed.

GDB can do four main kinds of things (plus other things in support of these) to help you catch bugs in the act:

- Start your program, specifying anything that might affect its behavior.
- Make your program stop on specified conditions.
- Examine what has happened, when your program has stopped.
- Change things in your program, so you can experiment with correcting the effects of one bug and go on to learn about another.

You can use GDB to debug programs written in C and C++. For more information, see [Supported Languages](https://sourceware.org/gdb/current/onlinedocs/gdb#Supported-Languages). For more information, see [C and C++](https://sourceware.org/gdb/current/onlinedocs/gdb#C).

Support for D is partial. For information on D, see [D](https://sourceware.org/gdb/current/onlinedocs/gdb#D).

Support for Modula-2 is partial. For information on Modula-2, see [Modula-2](https://sourceware.org/gdb/current/onlinedocs/gdb#Modula_002d2).

Support for OpenCL C is partial. For information on OpenCL C, see [OpenCL C](https://sourceware.org/gdb/current/onlinedocs/gdb#OpenCL-C).

Debugging Pascal programs which use sets, subranges, file variables, or nested functions does not currently work. GDB does not support entering expressions, printing values, or similar features using Pascal syntax.

GDB can be used to debug programs written in Fortran, although it may be necessary to refer to some variables with a trailing underscore.

GDB can be used to debug programs written in Objective-C, using either the Apple/NeXT or the GNU Objective-C runtime.

### Free Software
GDB is _free software_, protected by the GNU General Public License (GPL). The GPL gives you the freedom to copy or adapt a licensed program—but every person getting a copy also gets with it the freedom to modify that copy (which means that they must get access to the source code), and the freedom to distribute further copies. Typical software companies use copyrights to limit your freedoms; the Free Software Foundation uses the GPL to preserve these freedoms.

Fundamentally, the General Public License is a license which says that you have these freedoms and that you cannot take these freedoms away from anyone else.

### Free Software Needs Free Documentation
The biggest deficiency in the free software community today is not in the software—it is the lack of good free documentation that we can include with the free software. Many of our most important programs do not come with free reference manuals and free introductory texts. Documentation is an essential part of any software package; when an important free software package does not come with a free manual and a free tutorial, that is a major gap. We have many such gaps today.

Consider Perl, for instance. The tutorial manuals that people normally use are non-free. How did this come about? Because the authors of those manuals published them with restrictive terms—no copying, no modification, source files not available—which exclude them from the free software world.

That wasn’t the first time this sort of thing happened, and it was far from the last. Many times we have heard a GNU user eagerly describe a manual that he is writing, his intended contribution to the community, only to learn that he had ruined everything by signing a publication contract to make it non-free.

Free documentation, like free software, is a matter of freedom, not price. The problem with the non-free manual is not that publishers charge a price for printed copies—that in itself is fine. (The Free Software Foundation sells printed copies of manuals, too.) The problem is the restrictions on the use of the manual. Free manuals are available in source code form, and give you permission to copy and modify. Non-free manuals do not allow this.

The criteria of freedom for a free manual are roughly the same as for free software. Redistribution (including the normal kinds of commercial redistribution) must be permitted, so that the manual can accompany every copy of the program, both on-line and on paper.

Permission for modification of the technical content is crucial too. When people modify the software, adding or changing features, if they are conscientious they will change the manual too—so they can provide accurate and clear documentation for the modified program. A manual that leaves you no choice but to write a new manual to document a changed version of the program is not really available to our community.

Some kinds of limits on the way modification is handled are acceptable. For example, requirements to preserve the original author’s copyright notice, the distribution terms, or the list of authors, are ok. It is also no problem to require modified versions to include notice that they were modified. Even entire sections that may not be deleted or changed are acceptable, as long as they deal with nontechnical topics (like this one). These kinds of restrictions are acceptable because they don’t obstruct the community’s normal use of the manual.

However, it must be possible to modify all the _technical_ content of the manual, and then distribute the result in all the usual media, through all the usual channels. Otherwise, the restrictions obstruct the use of the manual, it is not free, and we need another manual to replace it.

Please spread the word about this issue. Our community continues to lose manuals to proprietary publishing. If we spread the word that free software needs free reference manuals and free tutorials, perhaps the next person who wants to contribute by writing documentation will realize, before it is too late, that only free manuals contribute to the free software community.

If you are writing documentation, please insist on publishing it under the GNU Free Documentation License or another free documentation license. Remember that this decision requires your approval—you don’t have to let the publisher decide. Some commercial publishers will use a free license if you insist, but they will not propose the option; it is up to you to raise the issue and say firmly that this is what you want. If the publisher you are dealing with refuses, please try other publishers. If you’re not sure whether a proposed license is free, write to [licensing@gnu.org](mailto:licensing@gnu.org).

You can encourage commercial publishers to sell more free, copylefted manuals and tutorials by buying them, and particularly by buying copies from the publishers that paid for their writing or for major improvements. Meanwhile, try to avoid buying non-free documentation at all. Check the distribution terms of a manual before you buy it, and insist that whoever seeks your business must respect your freedom. Check the history of the book, and try to reward the publishers that have paid or pay the authors to work on it.

The Free Software Foundation maintains a list of free documentation published by other publishers, at [http://www.fsf.org/doc/other-free-books.html](http://www.fsf.org/doc/other-free-books.html).

### Contributors to GDB
Richard Stallman was the original author of GDB, and of many other GNU programs. Many others have contributed to its development. This section attempts to credit major contributors. One of the virtues of free software is that everyone is free to contribute to it; with regret, we cannot actually acknowledge everyone here. The file ChangeLog in the GDB distribution approximates a blow-by-blow account.

Changes much prior to version 2.0 are lost in the mists of time.

> _Plea:_ Additions to this section are particularly welcome. If you or your friends (or enemies, to be evenhanded) have been unfairly omitted from this list, we would like to add your names!

So that they may not regard their many labors as thankless, we particularly thank those who shepherded GDB through major releases: Andrew Cagney (releases 6.3, 6.2, 6.1, 6.0, 5.3, 5.2, 5.1 and 5.0); Jim Blandy (release 4.18); Jason Molenda (release 4.17); Stan Shebs (release 4.14); Fred Fish (releases 4.16, 4.15, 4.13, 4.12, 4.11, 4.10, and 4.9); Stu Grossman and John Gilmore (releases 4.8, 4.7, 4.6, 4.5, and 4.4); John Gilmore (releases 4.3, 4.2, 4.1, 4.0, and 3.9); Jim Kingdon (releases 3.5, 3.4, and 3.3); and Randy Smith (releases 3.2, 3.1, and 3.0).

Richard Stallman, assisted at various times by Peter TerMaat, Chris Hanson, and Richard Mlynarik, handled releases through 2.8.

Michael Tiemann is the author of most of the GNU C++ support in GDB, with significant additional contributions from Per Bothner and Daniel Berlin. James Clark wrote the GNU C++ demangler. Early work on C++ was by Peter TerMaat (who also did much general update work leading to release 3.0).

GDB uses the BFD subroutine library to examine multiple object-file formats; BFD was a joint project of David V. Henkel-Wallace, Rich Pixley, Steve Chamberlain, and John Gilmore.

David Johnson wrote the original COFF support; Pace Willison did the original support for encapsulated COFF.

Brent Benson of Harris Computer Systems contributed DWARF 2 support.

Adam de Boor and Bradley Davis contributed the ISI Optimum V support. Per Bothner, Noboyuki Hikichi, and Alessandro Forin contributed MIPS support. Jean-Daniel Fekete contributed Sun 386i support. Chris Hanson improved the HP9000 support. Noboyuki Hikichi and Tomoyuki Hasei contributed Sony/News OS 3 support. David Johnson contributed Encore Umax support. Jyrki Kuoppala contributed Altos 3068 support. Jeff Law contributed HP PA and SOM support. Keith Packard contributed NS32K support. Doug Rabson contributed Acorn Risc Machine support. Bob Rusk contributed Harris Nighthawk CX-UX support. Chris Smith contributed Convex support (and Fortran debugging). Jonathan Stone contributed Pyramid support. Michael Tiemann contributed SPARC support. Tim Tucker contributed support for the Gould NP1 and Gould Powernode. Pace Willison contributed Intel 386 support. Jay Vosburgh contributed Symmetry support. Marko Mlinar contributed OpenRISC 1000 support.

Andreas Schwab contributed M68K GNU/Linux support.

Rich Schaefer and Peter Schauer helped with support of SunOS shared libraries.

Jay Fenlason and Roland McGrath ensured that GDB and GAS agree about several machine instruction sets.

Patrick Duval, Ted Goldstein, Vikram Koka and Glenn Engel helped develop remote debugging. Intel Corporation, Wind River Systems, AMD, and ARM contributed remote debugging modules for the i960, VxWorks, A29K UDI, and RDI targets, respectively.

Brian Fox is the author of the readline libraries providing command-line editing and command history.

Andrew Beers of SUNY Buffalo wrote the language-switching code, the Modula-2 support, and contributed the Languages chapter of this manual.

Fred Fish wrote most of the support for Unix System Vr4. He also enhanced the command-completion support to cover C++ overloaded symbols.

Hitachi America (now Renesas America), Ltd. sponsored the support for H8/300, H8/500, and Super-H processors.

NEC sponsored the support for the v850, Vr4xxx, and Vr5xxx processors.

Mitsubishi (now Renesas) sponsored the support for D10V, D30V, and M32R/D processors.

Toshiba sponsored the support for the TX39 Mips processor.

Matsushita sponsored the support for the MN10200 and MN10300 processors.

Fujitsu sponsored the support for SPARClite and FR30 processors.

Kung Hsu, Jeff Law, and Rick Sladkey added support for hardware watchpoints.

Michael Snyder added support for tracepoints.

Stu Grossman wrote gdbserver.

Jim Kingdon, Peter Schauer, Ian Taylor, and Stu Grossman made nearly innumerable bug fixes and cleanups throughout GDB.

The following people at the Hewlett-Packard Company contributed support for the PA-RISC 2.0 architecture, HP-UX 10.20, 10.30, and 11.0 (narrow mode), HP’s implementation of kernel threads, HP’s aC++ compiler, and the Text User Interface (nee Terminal User Interface): Ben Krepp, Richard Title, John Bishop, Susan Macchia, Kathy Mann, Satish Pai, India Paul, Steve Rehrauer, and Elena Zannoni. Kim Haase provided HP-specific information in this manual.

DJ Delorie ported GDB to MS-DOS, for the DJGPP project. Robert Hoehne made significant contributions to the DJGPP port.

Cygnus Solutions has sponsored GDB maintenance and much of its development since 1991. Cygnus engineers who have worked on GDB fulltime include Mark Alexander, Jim Blandy, Per Bothner, Kevin Buettner, Edith Epstein, Chris Faylor, Fred Fish, Martin Hunt, Jim Ingham, John Gilmore, Stu Grossman, Kung Hsu, Jim Kingdon, John Metzler, Fernando Nasser, Geoffrey Noer, Dawn Perchik, Rich Pixley, Zdenek Radouch, Keith Seitz, Stan Shebs, David Taylor, and Elena Zannoni. In addition, Dave Brolley, Ian Carmichael, Steve Chamberlain, Nick Clifton, JT Conklin, Stan Cox, DJ Delorie, Ulrich Drepper, Frank Eigler, Doug Evans, Sean Fagan, David Henkel-Wallace, Richard Henderson, Jeff Holcomb, Jeff Law, Jim Lemke, Tom Lord, Bob Manson, Michael Meissner, Jason Merrill, Catherine Moore, Drew Moseley, Ken Raeburn, Gavin Romig-Koch, Rob Savoye, Jamie Smith, Mike Stump, Ian Taylor, Angela Thomas, Michael Tiemann, Tom Tromey, Ron Unrau, Jim Wilson, and David Zuhn have made contributions both large and small.

Andrew Cagney, Fernando Nasser, and Elena Zannoni, while working for Cygnus Solutions, implemented the original GDB/MI interface.

Jim Blandy added support for preprocessor macros, while working for Red Hat.

Andrew Cagney designed GDB’s architecture vector. Many people including Andrew Cagney, Stephane Carrez, Randolph Chung, Nick Duffek, Richard Henderson, Mark Kettenis, Grace Sainsbury, Kei Sakamoto, Yoshinori Sato, Michael Snyder, Andreas Schwab, Jason Thorpe, Corinna Vinschen, Ulrich Weigand, and Elena Zannoni, helped with the migration of old architectures to this new framework.

Andrew Cagney completely re-designed and re-implemented GDB’s unwinder framework, this consisting of a fresh new design featuring frame IDs, independent frame sniffers, and the sentinel frame. Mark Kettenis implemented the DWARF 2 unwinder, Jeff Johnston the libunwind unwinder, and Andrew Cagney the dummy, sentinel, tramp, and trad unwinders. The architecture-specific changes, each involving a complete rewrite of the architecture’s frame code, were carried out by Jim Blandy, Joel Brobecker, Kevin Buettner, Andrew Cagney, Stephane Carrez, Randolph Chung, Orjan Friberg, Richard Henderson, Daniel Jacobowitz, Jeff Johnston, Mark Kettenis, Theodore A. Roth, Kei Sakamoto, Yoshinori Sato, Michael Snyder, Corinna Vinschen, and Ulrich Weigand.

Christian Zankel, Ross Morley, Bob Wilson, and Maxim Grigoriev from Tensilica, Inc. contributed support for Xtensa processors. Others who have worked on the Xtensa port of GDB in the past include Steve Tjiang, John Newlin, and Scott Foehner.

Michael Eager and staff of Xilinx, Inc., contributed support for the Xilinx MicroBlaze architecture.

Initial support for the FreeBSD/mips target and native configuration was developed by SRI International and the University of Cambridge Computer Laboratory under DARPA/AFRL contract FA8750-10-C-0237 ("CTSRD"), as part of the DARPA CRASH research programme.

Initial support for the FreeBSD/riscv target and native configuration was developed by SRI International and the University of Cambridge Computer Laboratory (Department of Computer Science and Technology) under DARPA contract HR0011-18-C-0016 ("ECATS"), as part of the DARPA SSITH research programme.

The original port to the OpenRISC 1000 is believed to be due to Alessandro Forin and Per Bothner. More recent ports have been the work of Jeremy Bennett, Franck Jullien, Stefan Wallentowitz and Stafford Horne.

Weimin Pan, David Faust and Jose E. Marchesi contributed support for the Linux kernel BPF virtual architecture. This work was sponsored by Oracle.

## 1 A Sample GDB Session
You can use this manual at your leisure to read all about GDB. However, a handful of commands are enough to get started using the debugger. This chapter illustrates those commands.

One of the preliminary versions of GNU `m4` (a generic macro processor) exhibits the following bug: sometimes, when we change its quote strings from the default, the commands used to capture one macro definition within another stop working. In the following short `m4` session, we define a macro `foo` which expands to `0000`; we then use the `m4` built-in `defn` to define `bar` as the same thing. However, when we change the open quote string to `<QUOTE>` and the close quote string to `<UNQUOTE>`, the same procedure fails to define a new synonym `baz`:
>  GNU `m4` (一个通用的宏处理器) 的一个早期版本存在以下 bug: 有时，当我们将其默认 quote 字符串改为其他内容时，捕获一个宏定义到另一个宏定义的命令将停止工作
>  在以下简短的 `m4` 会话中，我们定义了一个宏 `foo`，它展开为 `0000`，然后我们用 `m4` 内建函数 `defn` 将 `bar` 定义为相同的内容
>  当我们将 open quote 字符串改为 `<QUOTE>` ，将 close quote 字符串改为 `<UNQUOTE>` 后，相同的步骤未能成功定义新宏 `baz`

```
$ cd gnu/m4
$ ./m4
define(foo,0000)

foo
> 0000
define(bar,defn(‘foo’))

bar
> 0000
changequote(<QUOTE>,<UNQUOTE>)

define(baz,defn(<QUOTE>foo<UNQUOTE>))
baz
Ctrl-d
> m4: End of input: 0: fatal error: EOF in string
```

Let us use GDB to try to see what is going on.
>  我们用 GDB debug `m4` (`$ gdb m4`) ，看一下问题在哪里

```
$ gdb m4
> GDB is free software and you are welcome to distribute copies
 of it under certain conditions; type "show copying" to see
 the conditions.
> There is absolutely no warranty for GDB; type "show warranty"
 for details.

> GDB 17.0.50.20250416-git, Copyright 1999 Free Software Foundation, Inc...
(gdb)
```

GDB reads only enough symbol data to know where to find the rest when needed; as a result, the first prompt comes up very quickly. We now tell GDB to use a narrower display width than usual, so that examples fit in this manual.
>  GDB 在读取时，仅读取足够的符号数据，在需要时再根据这些符号数据去找到区域的数据，因此，GDB 的第一个提示符会很快出现
>  我们先设定 GDB 以更窄的宽度显示:

```
(gdb) set width 70
```

We need to see how the `m4` built-in `changequote` works. Having looked at the source, we know the relevant subroutine is `m4_changequote`, so we set a breakpoint there with the GDB `break` command.
>  我们需要知道 `m4` 内建函数 `changequote` 的工作原理，**查看源代码后**，我们知道相关的子例程是 `m4_changequote`，故我们使用 GDB `break` 命令在该例程上设置断点:

```
(gdb) break m4_changequote
> Breakpoint 1 at 0x62f4: file builtin.c, line 879.
```

Using the `run` command, we start `m4` running under GDB control; as long as control does not reach the `m4_changequote` subroutine, the program runs as usual:
>  使用 `run` 命令，我们可以在 GDB 控制下启动 `m4` ，只要没有到达 `m4_changequote` 子例程，程序就像往常一样运行:

```
(gdb) run
> Starting program: /work/Editorial/gdb/gnu/m4/m4
define(foo,0000)

foo
> 0000
```

To trigger the breakpoint, we call `changequote`. GDB suspends execution of `m4`, displaying information about the context where it stops.
>  为了触发断点，我们调用 `changequote` ，GDB 暂停了 `m4` 的执行，并展示了停止处的上下文:

```
changequote(<QUOTE>,<UNQUOTE>)

> Breakpoint 1, m4_changequote (argc=3, argv=0x33c70)
    at builtin.c:879
879         if (bad_argc(TOKEN_DATA_TEXT(argv[0]),argc,1,3))
```

Now we use the command `n` (`next`) to advance execution to the next line of the current function.
>  现在，我们使用命令 `n` (`next`) 来执行到**当前函数**的下一行:

```
(gdb) n
> 882         set_quotes((argc >= 2) ? TOKEN_DATA_TEXT(argv[1])\
 : nil,
```

`set_quotes` looks like a promising subroutine. We can go into it by using the command `s` (`step`) instead of `next`. `step` goes to the next line to be executed in _any_ subroutine, so it steps into `set_quotes`.
>  `set_quote` 看起来可能是存在问题的子例程，我们可以使用命令 `s` (`step`) 来进入它 (而不是 `next`)
>  `step` 会进入任意子例程即将执行的下一行，因此它会进入 `set_quotes`

```
(gdb) s
> set_quotes (lq=0x34c78 "<QUOTE>", rq=0x34c88 "<UNQUOTE>")
    at input.c:530
530         if (lquote != def_lquote)
```

The display that shows the subroutine where `m4` is now suspended (and its arguments) is called a stack frame display. It shows a summary of the stack. 
>  显示当前 `m4` 中挂起的子例程 (和其参数) 的页面称为栈帧显示 (当前栈帧在 `set_quotes`)，它提供了栈的摘要信息

We can use the `backtrace` command (which can also be spelled `bt`), to see where we are in the stack as a whole: the `backtrace` command displays a stack frame for each active subroutine.
>  我们可以使用 `backtrace` 命令 (`bt`) 查看我们在整个栈中的位置: `backtrace` 为每个活动的子例程显示一个栈帧 (沿着调用顺序，各个例程的位置从栈顶部到栈底部)

```
(gdb) bt
> #0  set_quotes (lq=0x34c78 "<QUOTE>", rq=0x34c88 "<UNQUOTE>")
    at input.c:530
> #1  0x6344 in m4_changequote (argc=3, argv=0x33c70)
    at builtin.c:882
> #2  0x8174 in expand_macro (sym=0x33320) at macro.c:242
> #3  0x7a88 in expand_token (obs=0x0, t=209696, td=0xf7fffa30)
    at macro.c:71
> #4  0x79dc in expand_input () at macro.c:40
> #5  0x2930 in main (argc=0, argv=0xf7fffb20) at m4.c:195
```

We step through a few more lines to see what happens. The first two times, we can use ‘s’; the next two times we use `n` to avoid falling into the `xstrdup` subroutine.
>  我们再走几行，看看会发生什么
>  前两次我们使用 `s`，后两次我们使用 `n` 以避免进入 `xstrdup` 子例程

```
(gdb) s
> 0x3b5c  532         if (rquote != def_rquote)
(gdb) s
> 0x3b80  535         lquote = (lq == nil || *lq == '\0') ?  \
> def_lquote : xstrdup(lq);
(gdb) n
> 536         rquote = (rq == nil || *rq == '\0') ? def_rquote\
 : xstrdup(rq);
(gdb) n
> 538         len_lquote = strlen(rquote);
```

The last line displayed looks a little odd; we can examine the variables `lquote` and `rquote` to see if they are in fact the new left and right quotes we specified. We use the command `p` (`print`) to see their values.
>  最后一行显示的内容有点奇怪，我们可以检查变量 `lquote` 和 `rquote` ，来查看它们是否确实是我们指定的新 left/right quote
>  我们使用命令 `p` (`print`) 查看变量值:

```
(gdb) p lquote
> $1 = 0x35d40 "<QUOTE>"
(gdb) p rquote
> $2 = 0x35d50 "<UNQUOTE>"
```

`lquote` and `rquote` are indeed the new left and right quotes. To look at some context, we can display ten lines of source surrounding the current line with the `l` (`list`) command.
>  `lquote` 和 `rquote` 确实是我们指定的新 left/right quote
>  为了查看一些上下文，我们可以使用 `l` (`list`) 命令显示当前行周围 10 行的源代码:

```
(gdb) l
> 533             xfree(rquote);
> 534
> 535         lquote = (lq == nil || *lq == '\0') ? def_lquote\
 : xstrdup (lq);
> 536         rquote = (rq == nil || *rq == '\0') ? def_rquote\
 : xstrdup (rq);
> 537
> 538         len_lquote = strlen(rquote);
> 539         len_rquote = strlen(lquote);
> 540     }
> 541
> 542     void
```

Let us step past the two lines that set `len_lquote` and `len_rquote`, and then examine the values of those variables.
>  让我们步过设置 `len_lquote, len_rquote` 的两行代码，然后检查这两个变量的值

```
(gdb) n
> 539         len_rquote = strlen(lquote);
(gdb) n
> 540     }
(gdb) p len_lquote
> $3 = 9
(gdb) p len_rquote
> $4 = 7
```

That certainly looks wrong, assuming `len_lquote` and `len_rquote` are meant to be the lengths of `lquote` and `rquote` respectively. We can set them to better values using the `p` command, since it can print the value of any expression—and that expression can include subroutine calls and assignments.
>  假设 `len_lquote, len_rquote` 是要表示 `lquote, rquote` 的长度，则这两个变量显然不对
>  我们可以使用 `p` 命令将它们设置为更好的值，因为 `p` 命令可以打印任意表达式的值，且这个表达式**可以包含子例程调用和赋值**

```
(gdb) p len_lquote=strlen(lquote)
> $5 = 7
(gdb) p len_rquote=strlen(rquote)
> $6 = 9
```

Is that enough to fix the problem of using the new quotes with the `m4` built-in `defn`? We can allow `m4` to continue executing with the `c` (`continue`) command, and then try the example that caused trouble initially:
>  我们用 `c` (`continue`) 命令让 `m4` 继续执行，然后尝试最初引起问题的示例，看看是否解决了问题:

```
(gdb) c
> Continuing.

define(baz,defn(<QUOTE>foo<UNQUOTE>))

> baz
> 0000
```

Success! The new quotes now work just as well as the default ones. The problem seems to have been just the two typos defining the wrong lengths. We allow `m4` exit by giving it an EOF as input:
>  发现成功了，新的 quote 定义不再引起问题
>  问题来自于两个拼写错误，颠倒了 `len_lquote, len_rquote` 的赋值
>  我们向 `m4` 提供 EOF 作为输入使其退出

```
Ctrl-d
> Program exited normally.
```

The message ‘Program exited normally.’ is from GDB; it indicates `m4` has finished executing. We can end our GDB session with the GDB `quit` command.
>  Program exited normally 的消息来自于 GDB，它表示 `m4` 完成了执行
>  我们可以使用 `quit` 命令结束 GDB 会话:

```
(gdb) quit
```

## 2 Getting In and Out of GDB
This chapter discusses how to start GDB, and how to get out of it. The essentials are:

- type ‘ `gdb` ’ to start GDB.
- type `quit`, `exit` or `Ctrl-d` to exit.

### 2.1 Invoking GDB
Invoke GDB by running the program `gdb`. Once started, GDB reads commands from the terminal until you tell it to exit.

You can also run `gdb` with a variety of arguments and options, to specify more of your debugging environment at the outset.

The command-line options described here are designed to cover a variety of situations; in some environments, some of these options may effectively be unavailable.

The most usual way to start GDB is with one argument, specifying an executable program:
>  最常见的启动 GDB 的方式是给定一个可执行文件作为参数

```
gdb program
```

You can also start with both an executable program and a core file specified:
>  也可以给定一个可执行文件和 core 文件

```
gdb program core
```

You can, instead, specify a process ID as a second argument or use option `-p`, if you want to debug a running process:

```
gdb program 1234
gdb -p 1234
```

would attach GDB to process `1234`. With option `-p` you can omit the program filename.
>  如果要 debug 一个正在运行的进程，需要给定可执行文件的同时指定其 pid
>  或者使用 `-p` 就不需要指定可执行文件
>  上述例子中，GDB 将连接到进程 `1234`

Taking advantage of the second command-line argument requires a fairly complete operating system; when you use GDB as a remote debugger attached to a bare board, there may not be any notion of “process”, and there is often no way to get a core dump. GDB will warn you if it is unable to attach or to read core dumps.
>  通过两个命令行参数启动 GDB 一般需要一个完整的操作系统
>  当我们使用 GDB 作为连接到裸板的远程 debugger 时，板上可能没有 “进程” 的概念，并且通常无法获取核心转储文件
>  如果 GDB 无法连接到进程或读取核心转储文件，它会发出警告

You can optionally have `gdb` pass any arguments after the executable file to the inferior using `--args`. This option stops option processing.

```
gdb --args gcc -O2 -c foo.c
```

This will cause `gdb` to debug `gcc`, and to set `gcc` ’s command-line arguments (see [Arguments](https://sourceware.org/gdb/current/onlinedocs/gdb#Arguments)) to ‘ `-O2 -c foo.c` ’.

>  可以使用 `--args` 将可执行文件后的所有参数传递给 `gdb` 调试的程序，`--args` 选项将停止选项处理
>  上例中 `gdb` 将 debug `gcc` ，并且将 `gcc` 的命令行参数设置为 `-O2 -c foo.c`

You can run `gdb` without printing the front material, which describes GDB’s non-warranty, by specifying `--silent` (or `-q`/`--quiet`):

```
gdb --silent
```

>  指定 `--slient/-q/--quiet` 可以让 `gdb` 不打印其免责声明

You can further control how GDB starts up by using command-line options. GDB itself can remind you of the options available.

Type

```
gdb -help
```

to display all available options and briefly describe their use (‘ `gdb -h` ’ is a shorter equivalent).

All options and command line arguments you give are processed in sequential order. The order makes a difference when the ‘-x’ option is used.

-  [File Options](https://sourceware.org/gdb/current/onlinedocs/gdb#File-Options): Choosing files
-  [Mode Options](https://sourceware.org/gdb/current/onlinedocs/gdb#Mode-Options): Choosing modes
-  [Startup](https://sourceware.org/gdb/current/onlinedocs/gdb#Startup): What GDB does during startup
-  [Initialization Files](https://sourceware.org/gdb/current/onlinedocs/gdb#Initialization-Files): Initialization Files

>  temporarily skip 2.1.1 - 2.1.4

### 2.2 Quitting GDB

`quit [expression]`
`exit [expression]`
`q`

To exit GDB, use the `quit` command (abbreviated `q`), the `exit` command, or type an end-of-file character (usually Ctrl-d). If you do not supply expression, GDB will terminate normally; otherwise it will terminate using the result of expression as the error code.

An interrupt (often Ctrl-c) does not exit from GDB, but rather terminates the action of any GDB command that is in progress and returns to GDB command level. It is safe to type the interrupt character at any time because GDB does not allow it to take effect until a time when it is safe.
>  中断 (Ctrl-c) 通常不会退出 GDB，而是终止任何正在执行的 GDB 命令，返回到 GDB 命令级别
>  在 GDB 中，任何时候中断都是安全的，因为 GDB 只有在安全的时候才会让中断信号生效

If you have been using GDB to control an attached process or device, you can release it with the `detach` command (see [Debugging an Already-running Process](https://sourceware.org/gdb/current/onlinedocs/gdb#Attach)).
>  如果 GDB 正在控制一个其连接的进程或设备，`detach` 可以断开连接

### 2.3 Shell Commands
If you need to execute occasional shell commands during your debugging session, there is no need to leave or suspend GDB; you can just use the `shell` command.

`shell command-string`
`!command-string`

Invoke a shell to execute command-string. Note that no space is needed between `!` and command-string. On GNU and Unix systems, the environment variable `SHELL`, if it exists, determines which shell to run. Otherwise GDB uses the default shell (`/bin/sh` on GNU and Unix systems, `cmd.exe` on MS-Windows, `COMMAND.COM` on MS-DOS, etc.).

>  在 debugging 会话中需要执行 shell 命令时，可以使用 `shell command-string` 或 `!command-string`
>  环境便来给你 `SHELL` 决定了执行命令的 shell

You may also invoke shell commands from expressions, using the `$_shell` convenience function. See [$_shell convenience function](https://sourceware.org/gdb/current/onlinedocs/gdb#g_t_0024_005fshell-convenience-function).

The utility `make` is often needed in development environments. You do not have to use the `shell` command for this purpose in GDB:

`make make-args`

Execute the `make` program with the specified arguments. This is equivalent to ‘ `shell make make-args` ’.

>  `make` 命令可以直接在 GDB session 中执行，等价于 `shell make make-args`

>  the rest is temporarily skipped

`pipe [command] | shell_command`
`| [command] | shell_command`
`pipe -d delim command delim shell_command`
`| -d delim command delim shell_command`

Executes command and sends its output to `shell_command`. Note that no space is needed around ` | `. If no command is provided, the last command executed is repeated.

In case the command contains a `|`, the option `-d delim` can be used to specify an alternate delimiter string `delim` that separates the command from the `shell_command`.

Example:

```
(gdb) p var
$1 = {
  black = 144,
  red = 233,
  green = 377,
  blue = 610,
  white = 987
}

(gdb) pipe p var|wc
      7      19      80
(gdb) |p var|wc -l
7

(gdb) p /x var
$4 = {
  black = 0x90,
  red = 0xe9,
  green = 0x179,
  blue = 0x262,
  white = 0x3db
}
(gdb) ||grep red
  red => 0xe9,

(gdb) | -d ! echo this contains a | char\n ! sed -e 's/|/PIPE/'
this contains a PIPE char
(gdb) | -d xxx echo this contains a | char!\n xxx sed -e 's/|/PIPE/'
this contains a PIPE char!
(gdb)
```

The convenience variables `$_shell_exitcode` and `$_shell_exitsignal` can be used to examine the exit status of the last shell command launched by `shell`, `make`, `pipe` and `|`. See [Convenience Variables](https://sourceware.org/gdb/current/onlinedocs/gdb#Convenience-Vars).

## 3 GDB Commands
You can abbreviate a GDB command to the first few letters of the command name, if that abbreviation is unambiguous; and you can repeat certain GDB commands by typing just RET. You can also use the TAB key to get GDB to fill out the rest of a word in a command (or to show you the alternatives available, if there is more than one possibility).
>  可以将 GDB 命令缩写为命令名称的前几个字母，只要这种缩写是唯一的
>  可以仅键入 RET 来重复某些 GDB 命令
>  可以使用 TAB 自动补全

### 3.1 Command Syntax
A GDB command is a single line of input. There is no limit on how long it can be. It starts with a command name, which is followed by arguments whose meaning depends on the command name. For example, the command `step` accepts an argument which is the number of times to step, as in ‘step 5’. You can also use the `step` command with no arguments. Some commands do not allow any arguments.
>  一条 GDB 命令是一行输入，它没有长度限制
>  它以命令名称开始，后面跟着根据命令名称而定的参数
>  例如，`step` 命令接收表示步进次数的参数，例如 `step 5`
>   一些命名不允许任何参数

## 5 Stopping and Continuing
The principal purposes of using a debugger are so that you can stop your program before it terminates; or so that, if your program runs into trouble, you can investigate and find out why.

Inside GDB, your program may stop for any of several reasons, such as a signal, a breakpoint, or reaching a new line after a GDB command such as `step`. You may then examine and change variables, set new breakpoints or remove old ones, and then continue execution. 
>  在 GDB 中，我们的程序可能因为多种原因停止运行，例如接收到信号、触发断点，或执行了诸如 `step` 等 GDB 命令后到达新行
>  此时，我们可以检查和修改变量，设置新的断点或移除旧的断点，然后继续执行程序

Usually, the messages shown by GDB provide ample explanation of the status of your program—but you can also explicitly request this information at any time.

`info program`

Display information about the status of your program: whether it is running or not, what process it is, and why it stopped.

>  `info program` 会显示关于程序状态的信息: 程序是否在运行，处于哪个进程，以及为什么停止

### 5.1 Breakpoints, Watchpoints, and Catchpoints
A _breakpoint_ makes your program stop whenever a certain point in the program is reached. For each breakpoint, you can add conditions to control in finer detail whether your program stops. You can set breakpoints with the `break` command and its variants (see [Setting Breakpoints](https://sourceware.org/gdb/current/onlinedocs/gdb#Set-Breaks)), to specify the place where your program should stop by line number, function name or exact address in the program.
>  breakpoint 会让程序执行到特定位置时停止，对于每个断点，可以添加条件以更精细地控制程序是否停止
>  可以通过 `break` 命令及其变体设置断点，指定程序在哪一个行号、函数名或程序中的确切地址停止

On some systems, you can set breakpoints in shared libraries before the executable is run.
>  在一些系统上，可以在程序执行之前在共享库中设置断点

A _watchpoint_ is a special breakpoint that stops your program when the value of an expression changes. The expression may be a value of a variable, or it could involve values of one or more variables combined by operators, such as ‘a + b’. This is sometimes called _data breakpoints_. You must use a different command to set watchpoints (see [Setting Watchpoints](https://sourceware.org/gdb/current/onlinedocs/gdb#Set-Watchpoints)), but aside from that, you can manage a watchpoint like any other breakpoint: you enable, disable, and delete both breakpoints and watchpoints using the same commands.
>  watchpoint 是一类特殊的 breakpoint，它当表达式发生变化时停止程序，表达式可以是某个变量的值，或者是通过运算符组合的一个或多个变量的值，例如 `a + b`
>  watchpoint 有时也被称为 datapoint，我们需要使用一个不同的命令来设置 watchpoint，然后可以像管理其他断点一样管理观察点: 使用相同的命令启用、禁用、删除断点和观察点

You can arrange to have values from your program displayed automatically whenever GDB stops at a breakpoint. See [Automatic Display](https://sourceware.org/gdb/current/onlinedocs/gdb#Auto-Display).
>  我们可以安排程序在每次断点处停止时自动显示某些值

A _catchpoint_ is another special breakpoint that stops your program when a certain kind of event occurs, such as the throwing of a C++ exception or the loading of a library. As with watchpoints, you use a different command to set a catchpoint (see [Setting Catchpoints](https://sourceware.org/gdb/current/onlinedocs/gdb#Set-Catchpoints)), but aside from that, you can manage a catchpoint like any other breakpoint. (To stop when your program receives a signal, use the `handle` command; see [Signals](https://sourceware.org/gdb/current/onlinedocs/gdb#Signals).)
>  catchpoint 是另一类特殊的断点，当特定类型的事件发生时，它会停止我们的程序，例如抛出 C++ 异常或加载库
>  我们需要使用一个不同的命令设置捕获点，然后就像管理其他断点一样管理捕获点 (要在我们的程序接收到信号是停止，使用 `handle` 命令)

GDB assigns a number to each breakpoint, watchpoint, or catchpoint when you create it; these numbers are successive integers starting with one. In many of the commands for controlling various features of breakpoints you use the breakpoint number to say which breakpoint you want to change. Each breakpoint may be _enabled_ or _disabled_; if disabled, it has no effect on your program until you enable it again.
>  当我们创建 breakpoint, watchpoint, catchpoint 时，GDB 会为其分配一个编号，这些编号是从 1 开始逐渐递增的整数
>  许多用于控制断点各种功能的命令都要求使用编号来指定断点
>  每个断点可以是 enabled 或 disabled 状态，如果 disabled，断点将不会对程序产生影响

Some GDB commands accept a space-separated list of breakpoints on which to operate. A list element can be either a single breakpoint number, like ‘5’, or a range of such numbers, like ‘5-7’. When a breakpoint list is given to a command, all breakpoints in that list are operated on.
>  一些 GDB 命令接收空格分离的断点列表，列表中的元素可以是单独的断点编号，例如 `5`，也可以是一个编号范围，例如 `5-7`

#### 5.1.1 Setting Breakpoints
Breakpoints are set with the `break` command (abbreviated `b`). The debugger convenience variable ‘ `$bpnum` ’ records the number of the breakpoint you’ve set most recently:

```
(gdb) b main
Breakpoint 1 at 0x11c6: file zeoes.c, line 24.
(gdb) p $bpnum
$1 = 1
```

>  断点通过 `break` (缩写为 `b`) 命令设置，GDB 内建变量 `$bpnum` 记录了最近设置的断点编号

A breakpoint may be mapped to multiple code locations for example with inlined functions, Ada generics, C++ templates or overloaded function names. GDB then indicates the number of code locations in the breakpoint command output:

```
(gdb) b some_func
Breakpoint 2 at 0x1179: some_func. (3 locations)
(gdb) p $bpnum
$2 = 2
(gdb)
```

>  一个断点可以被映射到多个代码位置，例如内联函数、Ada 通用程序、C++、模板或重载函数名
>  GDB 会在 `break` 命令输出中显示断点关联的代码位置的数量

When your program stops on a breakpoint, the convenience variables ‘ `$_hit_bpnum` ’ and ‘ `$_hit_locno` ’ are respectively set to the number of the encountered breakpoint and the number of the breakpoint’s code location:

```
Thread 1 "zeoes" hit Breakpoint 2.1, some_func () at zeoes.c:8
8	  printf("some func\n");
(gdb) p $_hit_bpnum
$5 = 2
(gdb) p $_hit_locno
$6 = 1
(gdb)
```

>  程序在断点处停止时，GDB 内建变量 `$_hit_bpnum` 和 `$_hit_locno` 会分别被设置为遇到的断点编号和停止处代码位置的编号

Note that ‘ `$_hit_bpnum` ’ and ‘ `$bpnum` ’ are not equivalent: ‘ `$_hit_bpnum` ’ is set to the breakpoint number **last hit**, while ‘ `$bpnum` ’ is set to the breakpoint number **last set**.
>  注意，`$_hit_bpnum` 和 `$bpnum` 不等价: `$_hit_bpnum` 是最后一次触发的断点编号，`$bpnum` 是最后一次设置的断点编号

If the encountered breakpoint has only one code location, ‘ `$_hit_locno` ’ is set to 1:

```
Breakpoint 1, main (argc=1, argv=0x7fffffffe018) at zeoes.c:24
24	  if (argc > 1)
(gdb) p $_hit_bpnum
$3 = 1
(gdb) p $_hit_locno
$4 = 1
(gdb)
```

>  如果触发的断点是唯一的代码位置，则 `$_hit_locno` 就是 1

The ‘ `$_hit_bpnum` ’ and ‘ `$_hit_locno` ’ variables can typically be used in a breakpoint command list. (see [Breakpoint Command Lists](https://sourceware.org/gdb/current/onlinedocs/gdb#Break-Commands)). For example, as part of the breakpoint command list, you can disable completely the encountered breakpoint using disable `$_hit_bpnum` or disable the specific encountered breakpoint location using disable `$_hit_bpnum`. `$ _hit_locno`. If a breakpoint has only one location, ‘ `$_hit_locno` ’ is set to 1 and the commands disable `$_hit_bpnum` and disable `$_hit_bpnum`. `$ _hit_locno` both disable the breakpoint.
>  skip

You can also define aliases to easily disable the last hit location or last hit breakpoint:

```
(gdb) alias lld = disable $_hit_bpnum.$_hit_locno
(gdb) alias lbd = disable $_hit_bpnum
```


`break locspec`

Set a breakpoint at all the code locations in your program that result from resolving the given `locspec`. `locspec` can specify a function name, a line number, an address of an instruction, and more. See [Location Specifications](https://sourceware.org/gdb/current/onlinedocs/gdb#Location-Specifications), for the various forms of locspec. The breakpoint will stop your program just before it executes the instruction at the address of any of the breakpoint’s code locations.
>  `break locspec` 会在程序中所有通过解析给定的 `locspec` 得到的代码位置设定断点，`locspec` 可以是函数名、行号、指令地址等
>  程序会在执行任意断点代码位置处的指令之前停止

When using source languages that permit overloading of symbols, such as C++, a function name may refer to more than one symbol, and thus more than one place to break. See [Ambiguous Expressions](https://sourceware.org/gdb/current/onlinedocs/gdb#Ambiguous-Expressions), for a discussion of that situation.

It is also possible to insert a breakpoint that will stop the program only if a specific thread (see [Thread-Specific Breakpoints](https://sourceware.org/gdb/current/onlinedocs/gdb#Thread_002dSpecific-Breakpoints)), specific inferior (see [Inferior-Specific Breakpoints](https://sourceware.org/gdb/current/onlinedocs/gdb#Inferior_002dSpecific-Breakpoints)), or a specific task (see [Ada Tasks](https://sourceware.org/gdb/current/onlinedocs/gdb#Ada-Tasks)) hits that breakpoint.

`info breakpoints [list…]`
`info break [list…]`

Print a table of all breakpoints, watchpoints, tracepoints, and catchpoints set and not deleted. Optional argument n means print information only about the specified breakpoint(s) (or watchpoint(s) or tracepoint(s) or catchpoint(s)). For each breakpoint, following columns are printed:

_Breakpoint Numbers_

_Type_ : Breakpoint, watchpoint, tracepoint, or catchpoint.

_Disposition_ : Whether the breakpoint is marked to be disabled or deleted when hit.

_Enabled or Disabled_ : Enabled breakpoints are marked with ‘y’. ‘n’ marks breakpoints that are not enabled.

_Address_ : Where the breakpoint is in your program, as a memory address. For a pending breakpoint whose address is not yet known, this field will contain ‘ `<PENDING>` ’. Such breakpoint won’t fire until a shared library that has the symbol or line referred by breakpoint is loaded. See below for details. A breakpoint with several locations will have ‘ `<MULTIPLE>` ’ in this field—see below for details.

_What_ : Where the breakpoint is in the source for your program, as a file and line number. For a pending breakpoint, the original string passed to the breakpoint command will be listed as it cannot be resolved until the appropriate shared library is loaded in the future.

#### 5.1.6 Break Conditions
The simplest sort of breakpoint breaks every time your program reaches a specified place. You can also specify a _condition_ for a breakpoint. A condition is just a Boolean expression in your programming language (see [Expressions](https://sourceware.org/gdb/current/onlinedocs/gdb#Expressions)). A breakpoint with a condition evaluates the expression each time your program reaches it, and your program stops only if the condition is _true_.
>  我们可以为 breakpoint 设定指定一个 condition, condition 即我们在编程语言中编写的一个布尔表达式
>  带有 condition 的 breakpoint 会在每次程序到达断点时评估该表达式，只有当条件为真时，程序才会停止运行

This is the converse of using assertions for program validation; in that situation, you want to stop when the assertion is violated—that is, when the condition is false. In C, if you want to test an assertion expressed by the condition assert, you should set the condition ‘! assert’ on the appropriate breakpoint.

Conditions are also accepted for watchpoints; you may not need them, since a watchpoint is inspecting the value of an expression anyhow—but it might be simpler, say, to just set a watchpoint on a variable name, and specify a condition that tests whether the new value is an interesting one.

Break conditions can have side effects, and may even call functions in your program. This can be useful, for example, to activate functions that log program progress, or to use your own print functions to format special data structures. The effects are completely predictable unless there is another enabled breakpoint at the same address. (In that case, GDB might see the other breakpoint first and stop your program without checking the condition of this one.) Note that breakpoint commands are usually more convenient and flexible than break conditions for the purpose of performing side effects when a breakpoint is reached (see [Breakpoint Command Lists](https://sourceware.org/gdb/current/onlinedocs/gdb#Break-Commands)).

Breakpoint conditions can also be evaluated on the target’s side if the target supports it. Instead of evaluating the conditions locally, GDB encodes the expression into an agent expression (see [Agent Expressions](https://sourceware.org/gdb/current/onlinedocs/gdb#Agent-Expressions)) suitable for execution on the target, independently of GDB. Global variables become raw memory locations, locals become stack accesses, and so forth.

In this case, GDB will only be notified of a breakpoint trigger when its condition evaluates to true. This mechanism may provide faster response times depending on the performance characteristics of the target since it does not need to keep GDB informed about every breakpoint trigger, even those with false conditions.

Break conditions can be specified when a breakpoint is set, by using ‘if’ in the arguments to the `break` command. See [Setting Breakpoints](https://sourceware.org/gdb/current/onlinedocs/gdb#Set-Breaks). They can also be changed at any time with the `condition` command.

You can also use the `if` keyword with the `watch` command. The `catch` command does not recognize the `if` keyword; `condition` is the only way to impose a further condition on a catchpoint.

`condition bnum expression`

Specify expression as the break condition for breakpoint, watchpoint, or catchpoint number bnum. After you set a condition, breakpoint bnum stops your program only if the value of expression is true (nonzero, in C). When you use `condition`, GDB checks expression immediately for syntactic correctness, and to determine whether symbols in it have referents in the context of your breakpoint. If expression uses symbols not referenced in the context of the breakpoint, GDB prints an error message:

No symbol "foo" in current context.

GDB does not actually evaluate expression at the time the `condition` command (or a command that sets a breakpoint with a condition, like `break if …`) is given, however. See [Expressions](https://sourceware.org/gdb/current/onlinedocs/gdb#Expressions).

`condition -force bnum expression`

When the `-force` flag is used, define the condition even if expression is invalid at all the current locations of breakpoint bnum. This is similar to the `-force-condition` option of the `break` command.

`condition bnum`

Remove the condition from breakpoint number bnum. It becomes an ordinary unconditional breakpoint.

A special case of a breakpoint condition is to stop only when the breakpoint has been reached a certain number of times. This is so useful that there is a special way to do it, using the _ignore count_ of the breakpoint. Every breakpoint has an ignore count, which is an integer. Most of the time, the ignore count is zero, and therefore has no effect. But if your program reaches a breakpoint whose ignore count is positive, then instead of stopping, it just decrements the ignore count by one and continues. As a result, if the ignore count value is n, the breakpoint does not stop the next n times your program reaches it.

`ignore bnum count`

Set the ignore count of breakpoint number bnum to count. The next count times the breakpoint is reached, your program’s execution does not stop; other than to decrement the ignore count, GDB takes no action.

To make the breakpoint stop the next time it is reached, specify a count of zero.

When you use `continue` to resume execution of your program from a breakpoint, you can specify an ignore count directly as an argument to `continue`, rather than using `ignore`. See [Continuing and Stepping](https://sourceware.org/gdb/current/onlinedocs/gdb#Continuing-and-Stepping).

If a breakpoint has a positive ignore count and a condition, the condition is not checked. Once the ignore count reaches zero, GDB resumes checking the condition.

You could achieve the effect of the ignore count with a condition such as ‘$foo-- <= 0’ using a debugger convenience variable that is decremented each time. See [Convenience Variables](https://sourceware.org/gdb/current/onlinedocs/gdb#Convenience-Vars).

Ignore counts apply to breakpoints, watchpoints, tracepoints, and catchpoints.

### 5.2 Continuing and Stepping
_Continuing_ means resuming program execution until your program completes normally. In contrast, _stepping_ means executing just one more “step” of your program, where “step” may mean either one line of source code, or one machine instruction (depending on what particular command you use). Either when continuing or when stepping, your program may stop even sooner, due to a breakpoint or a signal. (If it stops due to a signal, you may want to use `handle`, or use ‘ `signal 0` ’ to resume execution (see [Signals](https://sourceware.org/gdb/current/onlinedocs/gdb#Signals)), or you may step into the signal’s handler (see [stepping and signal handlers](https://sourceware.org/gdb/current/onlinedocs/gdb#stepping-and-signal-handlers)).)
>  Continuing 意味着恢复程序的执行，直到程序正常完成，而 Stepping 则意味着至执行程序中的 “一步”，“一步” 可以指源代码中的一行，或一条机器指令
>  无论是继续执行还是单步执行，程序仍然会因为断点或信号再次停止 (如果是因为信号而停止，可能需要使用 `handle` 或 `signal 0` 还恢复执行)，或者可以步入信号处理程序

`continue [ignore-count]`
`c [ignore-count]`
`fg [ignore-count]`

Resume program execution, at the address where your program last stopped; any breakpoints set at that address are bypassed. The optional argument ignore-count allows you to specify a further number of times to ignore a breakpoint at this location; its effect is like that of `ignore` (see [Break Conditions](https://sourceware.org/gdb/current/onlinedocs/gdb#Conditions)).
>  `continue/c/fg` 会从程序上一次停止的地方恢复程序执行，该地址上的任何断点会被略过，`[ignore-count]` 用以设置之后忽略这个地方断点的次数

The argument ignore-count is meaningful only when your program stopped due to a breakpoint. At other times, the argument to `continue` is ignored.

The synonyms `c` and `fg` (for _foreground_, as the debugged program is deemed to be the foreground program) are provided purely for convenience, and have exactly the same behavior as `continue`.

To resume execution at a different place, you can use `return` (see [Returning from a Function](https://sourceware.org/gdb/current/onlinedocs/gdb#Returning)) to go back to the calling function; or `jump` (see [Continuing at a Different Address](https://sourceware.org/gdb/current/onlinedocs/gdb#Jumping)) to go to an arbitrary location in your program.

A typical technique for using stepping is to set a breakpoint (see [Breakpoints; Watchpoints; and Catchpoints](https://sourceware.org/gdb/current/onlinedocs/gdb#Breakpoints)) at the beginning of the function or the section of your program where a problem is believed to lie, run your program until it stops at that breakpoint, and then step through the suspect area, examining the variables that are interesting, until you see the problem happen.

`step`

Continue running your program until control reaches a different source line, then stop it and return control to GDB. This command is abbreviated `s`.
>  `step/s` 将继续运行程序，直到控制权达到不同的源代码行时停止，然后返回控制权给 GDB

> _Warning:_ If you use the `step` command while control is within a function that was compiled without debugging information, execution proceeds until control reaches a function that does have debugging information. Likewise, it will not step into a function which is compiled without debugging information. To step through functions without debugging information, use the `stepi` command, described below.

The `step` command only stops at the first instruction of a source line. This prevents the multiple stops that could otherwise occur in `switch` statements, `for` loops, etc. `step` continues to stop if a function that has debugging information is called within the line. In other words, `step` _steps inside_ any functions called within the line.

Also, the `step` command only enters a function if there is line number information for the function. Otherwise it acts like the `next` command. This avoids problems when using `cc -gl` on MIPS machines. Previously, `step` entered subroutines if there was any debugging information about the routine.

`step count`

Continue running as in `step`, but do so count times. If a breakpoint is reached, or a signal not related to stepping occurs before count steps, stepping stops right away.

`next [count]`

Continue to the next source line in the current (innermost) stack frame. This is similar to `step`, but function calls that appear within the line of code are executed without stopping. Execution stops when control reaches a different line of code at the original stack level that was executing when you gave the `next` command. This command is abbreviated `n`.
>  `next` 将继续执行到当前 (最内层) 栈帧的下一个源代码行处停止

An argument count is a repeat count, as for `step`.

The `next` command only stops at the first instruction of a source line. This prevents multiple stops that could otherwise occur in `switch` statements, `for` loops, etc.

### 5.4 Signals
A signal is an asynchronous event that can happen in a program. The operating system defines the possible kinds of signals, and gives each kind a name and a number. For example, in Unix `SIGINT` is the signal a program gets when you type an interrupt character (often Ctrl-c); `SIGSEGV` is the signal a program gets from referencing a place in memory far away from all the areas in use; `SIGALRM` occurs when the alarm clock timer goes off (which happens only if your program has requested an alarm).
>  signal 是会在程序中发生的一个异步事件，操作系统定义了 signal 的可能类型，并且为每种 signal 赋予了一个名称和编号
>  例如，Unix 系统中，`SIGINT` 是用户键入 Ctrl-c 时，程序接收到的中断信号; `SIGSEGV` 是程序尝试访问远离它没有权限的内存区域时收到的信号; `SIGALRM` 则是在程序请求了闹钟时，闹钟定时器触发后收到的信号

Some signals, including `SIGALRM`, are a normal part of the functioning of your program. Others, such as `SIGSEGV`, indicate errors; these signals are _fatal_ (they kill your program immediately) if the program has not specified in advance some other way to handle the signal. `SIGINT` does not indicate an error in your program, but it is normally fatal so it can carry out the purpose of the interrupt: to kill the program.
>  一些 signal，包括 `SIGALRM` ，是程序正常运行的一部分
>  一些 signal，例如 `SIGSEGV`，则表示出现了错误，如果程序没有提前指定其他方式来处理这些信号，这些信号就是致命的 (它们会立即中止程序)
>  `SIGINT` 不表示程序本身存在错误，但通常是致命的，以便实现中断的目的: 中止程序

GDB has the ability to detect any occurrence of a signal in your program. You can tell GDB in advance what to do for each kind of signal, apart from SIGKILL, which has its usual effect regardless.
>  GDB 可以检测程序中任何信号的发生，我们可以预先告诉 GDB 对每种信号 (除了 `SIGKILL` ，它始终具有其常规效果) 应该如何处理

When specifying a signal by number, GDB translates the number to the target platform according to the corresponding signal name. For example, GDB always treats signal 1 as `SIGHUP`. So, when specifying ‘1’ as a signal, GDB will translate this to the target’s `SIGHUP`, whatever that might be.
>  当通过数字指定信号时，GDB 根据对应的信号名称，将数字转换到目标平台上的信号
>  例如，GDB 总是将信号 1 视作 `SIGHUP` ，因此，当指定 1 作为信号时，GDB 会将其转换为目标平台的 `SIGHUP` 

Numbers may only be used for signals 1 through 15. GDB uses this mapping:

|Number|Name|
|---|---|
|1|SIGHUP|
|2|SIGINT|
|3|SIGQUIT|
|4|SIGILL|
|5|SIGTRAP|
|6|SIGABRT|
|7|SIGEMT|
|8|SIGFPE|
|9|SIGKILL|
|10|SIGBUS|
|11|SIGSEGV|
|12|SIGSYS|
|13|SIGPIPE|
|14|SIGALRM|
|15|SIGTERM|

Normally, GDB is set up to let the non-erroneous signals like `SIGALRM` be silently passed to your program (so as not to interfere with their role in the program’s functioning) but to stop your program immediately whenever an error signal happens. You can change these settings with the `handle` command.
>  通常，GDB 会让非错误信号 (例如 `SIGALRM`) 静默地传递给我们的程序 (以便不影响它们在程序功能中的作用)，但会在错误信号发生时中止程序

`info signals`
`info handle`

Print a table of all the kinds of signals and how GDB has been told to handle each one. You can use this to see the signal numbers of all the defined types of signals.

`info signals sig`

Similar, but print information only about the specified signal number.

`info handle` is an alias for `info signals`.

`catch signal [signal… | ‘all’]`

Set a catchpoint for the indicated signals. See [Set Catchpoints](https://sourceware.org/gdb/current/onlinedocs/gdb#Set-Catchpoints), for details about this command.

`handle signal [ signal … ] [keywords…]`

Change the way GDB handles each signal. Each signal can be the number of a signal or its name (with or without the ‘SIG’ at the beginning); a list of signal numbers of the form ‘low-high’; or the word ‘ `all` ’, meaning all the known signals, except ` SIGINT ` and ` SIGTRAP `, which are used by GDB. Optional argument keywords, described below, say what changes to make to all of the specified signals.

The keywords allowed by the `handle` command can be abbreviated. Their full names are:

- `nostop` : GDB should not stop your program when this signal happens. It may still print a message telling you that the signal has come in.
- `stop` : GDB should stop your program when this signal happens. This implies the `print` keyword as well.
- `print` : GDB should print a message when this signal happens.
- `noprint` : GDB should not mention the occurrence of the signal at all. This implies the `nostop` keyword as well.
- `pass`, `noignore` : GDB should allow your program to see this signal; your program can handle the signal, or else it may terminate if the signal is fatal and not handled. `pass` and `noignore` are synonyms.
- `nopass`, `ignore` : GDB should not allow your program to see this signal. `nopass` and `ignore` are synonyms.

When a signal stops your program, the signal is not visible to the program until you continue. Your program sees the signal then, if `pass` is in effect for the signal in question _at that time_. In other words, after GDB reports a signal, you can use the `handle` command with `pass` or `nopass` to control whether your program sees that signal when you continue.

The default is set to `nostop`, `noprint`, `pass` for non-erroneous signals such as `SIGALRM`, `SIGWINCH` and `SIGCHLD`, and to `stop`, `print`, `pass` for the erroneous signals.

You can also use the `signal` command to prevent your program from seeing a signal, or cause it to see a signal it normally would not see, or to give it any signal at any time. For example, if your program stopped due to some sort of memory reference error, you might store correct values into the erroneous variables and continue, hoping to see more execution; but your program would probably terminate immediately as a result of the fatal signal once it saw the signal. To prevent this, you can continue with ‘signal 0’. See [Giving your Program a Signal](https://sourceware.org/gdb/current/onlinedocs/gdb#Signaling).

GDB optimizes for stepping the mainline code. If a signal that has `handle nostop` and `handle pass` set arrives while a stepping command (e.g., `stepi`, `step`, `next`) is in progress, GDB lets the signal handler run and then resumes stepping the mainline code once the signal handler returns. In other words, GDB steps over the signal handler. This prevents signals that you’ve specified as not interesting (with `handle nostop`) from changing the focus of debugging unexpectedly. Note that the signal handler itself may still hit a breakpoint, stop for another signal that has `handle stop` in effect, or for any other event that normally results in stopping the stepping command sooner. Also note that GDB still informs you that the program received a signal if `handle print` is set.

If you set `handle pass` for a signal, and your program sets up a handler for it, then issuing a stepping command, such as `step` or `stepi`, when your program is stopped due to the signal will step _into_ the signal handler (if the target supports that).

Likewise, if you use the `queue-signal` command to queue a signal to be delivered to the current thread when execution of the thread resumes (see [Giving your Program a Signal](https://sourceware.org/gdb/current/onlinedocs/gdb#Signaling)), then a stepping command will step into the signal handler.

Here’s an example, using `stepi` to step to the first instruction of `SIGUSR1`’s handler:

(gdb) handle SIGUSR1
Signal        Stop      Print   Pass to program Description
SIGUSR1       Yes       Yes     Yes             User defined signal 1
(gdb) c
Continuing.

Program received signal SIGUSR1, User defined signal 1.
main () sigusr1.c:28
28        p = 0;
(gdb) si
sigusr1_handler () at sigusr1.c:9
9       {

The same, but using `queue-signal` instead of waiting for the program to receive the signal first:

(gdb) n
28        p = 0;
(gdb) queue-signal SIGUSR1
(gdb) si
sigusr1_handler () at sigusr1.c:9
9       {
(gdb)

On some targets, GDB can inspect extra signal information associated with the intercepted signal, before it is actually delivered to the program being debugged. This information is exported by the convenience variable `$_siginfo`, and consists of data that is passed by the kernel to the signal handler at the time of the receipt of a signal. The data type of the information itself is target dependent. You can see the data type using the `ptype $_siginfo` command. On Unix systems, it typically corresponds to the standard `siginfo_t` type, as defined in the signal.h system header.

Here’s an example, on a GNU/Linux system, printing the stray referenced address that raised a segmentation fault.

(gdb) continue
Program received signal SIGSEGV, Segmentation fault.
0x0000000000400766 in main ()
69        *(int *)p = 0;
(gdb) ptype $_siginfo
type = struct {
    int si_signo;
    int si_errno;
    int si_code;
    union {
        int _pad[28];
        struct {...} _kill;
        struct {...} _timer;
        struct {...} _rt;
        struct {...} _sigchld;
        struct {...} _sigfault;
        struct {...} _sigpoll;
    } _sifields;
}
(gdb) ptype $_siginfo._sifields._sigfault
type = struct {
    void *si_addr;
}
(gdb) p $_siginfo._sifields._sigfault.si_addr
$1 = (void *) 0x7ffff7ff7000

Depending on target support, `$_siginfo` may also be writable.