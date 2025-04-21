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

## 8 Examining the Stack
When your program has stopped, the first thing you need to know is where it stopped and how it got there.
>  当我们的程序停止时，我们需要知道它在哪里停止，并且是如何到达那里的

Each time your program performs a function call, information about the call is generated. That information includes the location of the call in your program, the arguments of the call, and the local variables of the function being called. The information is saved in a block of data called a _stack frame_. The stack frames are allocated in a region of memory called the _call stack_.
>  我们的程序每次执行一次函数调用，都会生成关于该调用的信息
>  信息包括了调用的位置、调用的参数、被调用函数的局部变量
>  这些信息存在一个称为栈帧的数据块中，栈帧会在内存中一块称为调用栈的区域中进行分配

When your program stops, the GDB commands for examining the stack allow you to see all of this information.

One of the stack frames is _selected_ by GDB and many GDB commands refer implicitly to the selected frame. In particular, whenever you ask GDB for the value of a variable in your program, the value is found in the selected frame. There are special GDB commands to select whichever frame you are interested in. See [Selecting a Frame](https://sourceware.org/gdb/current/onlinedocs/gdb#Selection).
>  其中一个栈帧被 GDB 选中，并且许多 GDB 命令会隐式引用所选的帧

When your program stops, GDB automatically selects the currently executing frame and describes it briefly, similar to the `frame` command (see [Information about a Frame](https://sourceware.org/gdb/current/onlinedocs/gdb#Frame-Info)).
>  程序停止时，GDB 自动选择当前执行帧，并简要描述它

### 8.1 Stack Frames
The call stack is divided up into contiguous pieces called _stack frames_, or _frames_ for short; each frame is the data associated with one call to one function. The frame contains the arguments given to the function, the function’s local variables, and the address at which the function is executing.
>  调用栈被划分为连续的片段，这些片段被称为栈帧，每个栈帧与某个函数的一次调用相关联，包括了该函数的参数、局部变量、执行的地址

When your program is started, the stack has only one frame, that of the function `main`. This is called the _initial_ frame or the _outermost_ frame. Each time a function is called, a new frame is made. Each time a function returns, the frame for that function invocation is eliminated. If a function is recursive, there can be many frames for the same function. The frame for the function in which execution is actually occurring is called the _innermost_ frame. This is the most recently created of all the stack frames that still exist.
>  程序启动时，只有一个 `main` 函数的栈帧，称为初始帧或最外层帧
>  函数调用会创建新帧，函数返回会消除对应的帧，当前正在执行的函数对应最内层帧

Inside your program, stack frames are identified by their addresses. A stack frame consists of many bytes, each of which has its own address; each kind of computer has a convention for choosing one byte whose address serves as the address of the frame. Usually this address is kept in a register called the _frame pointer register_ (see [$fp](https://sourceware.org/gdb/current/onlinedocs/gdb#Registers)) while execution is going on in that frame.
>  栈帧在程序中通过地址标识，栈帧本身包括多个字节的数据，其中一个字节的地址会用于表示栈帧的地址
>  通常当前栈帧的地址会保存在帧指针寄存器中 `$fp`

GDB labels each existing stack frame with a _level_, a number that is zero for the innermost frame, one for the frame that called it, and so on upward. These level numbers give you a way of designating stack frames in GDB commands. The terms _frame number_ and _frame level_ can be used interchangeably to describe this number.
>  GDB 为每个现存栈帧标记一个级别，最内层帧是 0
>  “帧级别” 和 "帧编号" 可以交换使用，本质是在 GDB 命令中标识帧的数字

Some compilers provide a way to compile functions so that they operate without stack frames. (For example, the GCC option

```
‘-fomit-frame-pointer’
```

generates functions without a frame.) This is occasionally done with heavily used library functions to save the frame setup time. GDB has limited facilities for dealing with these function invocations. If the innermost function invocation has no stack frame, GDB nevertheless regards it as though it had a separate frame, which is numbered zero as usual, allowing correct tracing of the function call chain. However, GDB has no provision for frameless functions elsewhere in the stack.

### 8.2 Backtraces
A backtrace is a summary of how your program got where it is. It shows one line per frame, for many frames, starting with the currently executing frame (frame zero), followed by its caller (frame one), and on up the stack.

To print a backtrace of the entire stack, use the `backtrace` command, or its alias `bt`. This command will print one line per frame for frames in the stack. By default, all stack frames are printed. You can stop the backtrace at any time by typing the system interrupt character, normally Ctrl-c.

The names `where` and `info stack` (abbreviated `info s`) are additional aliases for `backtrace`.

In a multi-threaded program, GDB by default shows the backtrace only for the current thread. To display the backtrace for several or all of the threads, use the command `thread apply` (see [thread apply](https://sourceware.org/gdb/current/onlinedocs/gdb#Threads)). For example, if you type thread apply all backtrace, GDB will display the backtrace for all the threads; this is handy when you debug a core dump of a multi-threaded program.

Each line in the backtrace shows the frame number and the function name. The program counter value is also shown—unless you use `set print address off`. The backtrace also shows the source file name and line number, as well as the arguments to the function. The program counter value is omitted if it is at the beginning of the code for that line number.
>  backtrace 的每一行都显示了帧号和函数名称，以及程序计数器的值 (执行指令的地址)
>  backtrace 还显示源文件名、行号以及传递给函数的参数，如果程序计数器位于该行号代码的起始位置，则会省略其值

Here is an example of a backtrace. It was made with the command ‘ `bt 3` ’, so it shows the innermost three frames.

```
#0  m4_traceon (obs=0x24eb0, argc=1, argv=0x2b8c8)
    at builtin.c:993
#1  0x6e38 in expand_macro (sym=0x2b600, data=...) at macro.c:242
#2  0x6840 in expand_token (obs=0x0, t=177664, td=0xf7fffb08)
    at macro.c:71
(More stack frames follow...)
```

The display for frame zero does not begin with a program counter value, indicating that your program has stopped at the beginning of the code for line `993` of `builtin.c`.
>  上例中，第零帧的显示不以程序计数器的值开头，这表明的程序在 `builtin.c` 文件的第 993 行代码开始处停止运行

The value of parameter `data` in frame 1 has been replaced by `…`. By default, GDB prints the value of a parameter only if it is a scalar (integer, pointer, enumeration, etc). See command set print frame-arguments in [Print Settings](https://sourceware.org/gdb/current/onlinedocs/gdb#Print-Settings) for more details on how to configure the way function parameter values are printed. The command `set print frame-info` (see [Print Settings](https://sourceware.org/gdb/current/onlinedocs/gdb#Print-Settings)) controls what frame information is printed.
>  在第 1 帧中，参数 `data` 的值已被替换为 `…`
>  默认情况下，GDB 只在参数是标量 (例如整数、指针、枚举等) 打印参数的值