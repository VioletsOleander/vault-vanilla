# Course overview + the shell

## Motivation
As computer scientists, we know that computers are great at aiding in repetitive tasks. However, far too often, we forget that this applies just as much to our _use_ of the computer as it does to the computations we want our programs to perform. We have a vast range of tools available at our fingertips that enable us to be more productive and solve more complex problems when working on any computer-related problem. Yet many of us utilize only a small fraction of those tools; we only know enough magical incantations by rote to get by, and blindly copy-paste commands from the internet when we get stuck.

This class is an attempt to address this.

We want to teach you how to make the most of the tools you know, show you new tools to add to your toolbox, and hopefully instill in you some excitement for exploring (and perhaps building) more tools on your own. This is what we believe to be the missing semester from most Computer Science curricula.

## Class structure
The class consists of 11 1-hour lectures, each one centering on a [particular topic](https://missing.csail.mit.edu/2020/). The lectures are largely independent, though as the semester goes on we will presume that you are familiar with the content from the earlier lectures. We have lecture notes online, but there will be a lot of content covered in class (e.g. in the form of demos) that may not be in the notes. We will be recording lectures and posting the recordings online.

We are trying to cover a lot of ground over the course of just 11 1-hour lectures, so the lectures are fairly dense. To allow you some time to get familiar with the content at your own pace, each lecture includes a set of exercises that guide you through the lecture’s key points. After each lecture, we are hosting office hours where we will be present to help answer any questions you might have. If you are attending the class online, you can send us questions at [missing-semester@mit.edu](mailto:missing-semester@mit.edu).

Due to the limited time we have, we won’t be able to cover all the tools in the same level of detail a full-scale class might. Where possible, we will try to point you towards resources for digging further into a tool or topic, but if something particularly strikes your fancy, don’t hesitate to reach out to us and ask for pointers!

## Topic 1: The Shell
### What is the shell?
Computers these days have a variety of interfaces for giving them commands; fanciful graphical user interfaces, voice interfaces, and even AR/VR are everywhere. These are great for 80% of use-cases, but they are often fundamentally restricted in what they allow you to do — you cannot press a button that isn’t there or give a voice command that hasn’t been programmed. To take full advantage of the tools your computer provides, we have to go old-school and drop down to a textual interface: The Shell.

Nearly all platforms you can get your hands on have a shell in one form or another, and many of them have several shells for you to choose from. While they may vary in the details, at their core they are all roughly the same: they allow you to run programs, give them input, and inspect their output in a semi-structured way.

In this lecture, we will focus on the Bourne Again SHell, or “bash” for short. This is one of the most widely used shells, and its syntax is similar to what you will see in many other shells. To open a shell _prompt_ (where you can type commands), you first need a _terminal_. Your device probably shipped with one installed, or you can install one fairly easily.

### Using the shell
When you launch your terminal, you will see a _prompt_ that often looks a little like this:

```
missing:~$ 
```

This is the main textual interface to the shell. It tells you that you are on the machine `missing` and that your “current working directory”, or where you currently are, is `~` (short for “home”). The `$` tells you that you are not the root user (more on that later). At this prompt you can type a _command_, which will then be interpreted by the shell. The most basic command is to execute a program:

```
missing:~$ date
Fri 10 Jan 2020 11:49:31 AM EST
missing:~$ 
```

Here, we executed the `date` program, which (perhaps unsurprisingly) prints the current date and time. The shell then asks us for another command to execute. We can also execute a command with _arguments_:

```
missing:~$ echo hello
hello
```

In this case, we told the shell to execute the program `echo` with the argument `hello`. The `echo` program simply prints out its arguments. The shell parses the command by splitting it by whitespace, and then runs the program indicated by the first word, supplying each subsequent word as an argument that the program can access. If you want to provide an argument that contains spaces or other special characters (e.g., a directory named “My Photos”), you can either quote the argument with `'` or `"` (`"My Photos"`), or escape just the relevant characters with `\` (`My\ Photos`).

But how does the shell know how to find the `date` or `echo` programs? Well, the shell is a programming environment, just like Python or Ruby, and so it has variables, conditionals, loops, and functions (next lecture!). When you run commands in your shell, you are really writing a small bit of code that your shell interprets. If the shell is asked to execute a command that doesn’t match one of its programming keywords, it consults an _environment variable_ called `$PATH` that lists which directories the shell should search for programs when it is given a command:

```
missing:~$ echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
missing:~$ which echo
/bin/echo
missing:~$ /bin/echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

When we run the `echo` command, the shell sees that it should execute the program `echo`, and then searches through the `:`-separated list of directories in `$PATH` for a file by that name. When it finds it, it runs it (assuming the file is _executable_; more on that later). We can find out which file is executed for a given program name using the `which` program. We can also bypass `$PATH` entirely by giving the _path_ to the file we want to execute.

### Navigating in the shell
A path on the shell is a delimited list of directories; separated by `/` on Linux and macOS and `\` on Windows. On Linux and macOS, the path `/` is the “root” of the file system, under which all directories and files lie, whereas **on Windows there is one root for each disk partition** (e.g., `C:\`). We will generally assume that you are using a Linux filesystem in this class. A path that starts with `/` is called an _absolute_ path. Any other path is a _relative_ path. Relative paths are relative to the current working directory, which we can see with the `pwd` command and change with the `cd` command. In a path, `.` refers to the current directory, and `..` to its parent directory:

```
missing:~$ pwd
/home/missing
missing:~$ cd /home
missing:/home$ pwd
/home
missing:/home$ cd ..
missing:/$ pwd
/
missing:/$ cd ./home
missing:/home$ pwd
/home
missing:/home$ cd missing
missing:~$ pwd
/home/missing
missing:~$ ../../bin/echo hello
hello
```

Notice that our shell prompt kept us informed about what our current working directory was. You can configure your prompt to show you all sorts of useful information, which we will cover in a later lecture.

In general, when we run a program, it will operate in the current directory unless we tell it otherwise. For example, it will usually search for files there, and create new files there if it needs to.

To see what lives in a given directory, we use the `ls` command:

```
missing:~$ ls
missing:~$ cd ..
missing:/home$ ls
missing
missing:/home$ cd ..
missing:/$ ls
bin
boot
dev
etc
home
...
```

Unless a directory is given as its first argument, `ls` will print the contents of the current directory. Most commands accept flags and options (flags with values) that start with `-` to modify their behavior. Usually, running a program with the `-h` or `--help` flag will print some help text that tells you what flags and options are available. For example, `ls --help` tells us:

```
  -l                         use a long listing format
```

```
missing:~$ ls -l /home
drwxr-xr-x 1 missing  users  4096 Jun 15  2019 missing
```

This gives us a bunch more information about each file or directory present. First, the `d` at the beginning of the line tells us that `missing` is a directory. Then follow three groups of three characters (`rwx`). These indicate what permissions the owner of the file (`missing`), the owning group (`users`), and everyone else respectively have on the relevant item. A `-` indicates that the given principal does not have the given permission. Above, only the owner is allowed to modify (`w`) the `missing` directory (i.e., add/remove files in it). To enter a directory, a user must have “search” (represented by “execute”: `x`) permissions on that directory (and its parents). To list its contents, a user must have read (`r`) permissions on that directory. For files, the permissions are as you would expect. Notice that nearly all the files in `/bin` have the `x` permission set for the last group, “everyone else”, so that anyone can execute those programs.

Some other handy programs to know about at this point are `mv` (to rename/move a file), `cp` (to copy a file), and `mkdir` (to make a new directory).

If you ever want _more_ information about a program’s arguments, inputs, outputs, or how it works in general, give the `man` program a try. It takes as an argument the name of a program, and shows you its _manual page_. Press `q` to exit.

```
missing:~$ man ls
```

### Connecting programs
In the shell, programs have two primary “streams” associated with them: their input stream and their output stream. When the program tries to read input, it reads from the input stream, and when it prints something, it prints to its output stream. Normally, a program’s input and output are both your terminal. That is, your keyboard as input and your screen as output. However, we can also rewire those streams!
>  shell 中运行的程序都有 “输入流” 和 “输出流” 的概念

The simplest form of redirection is `< file` and `> file`. These let you rewire the input and output streams of a program to a file respectively:

```
missing:~$ echo hello > hello.txt
missing:~$ cat hello.txt
hello
missing:~$ cat < hello.txt
hello
missing:~$ cat < hello.txt > hello2.txt
missing:~$ cat hello2.txt
hello
```

Demonstrated in the example above, `cat` is a program that con `cat` enates files. When given file names as arguments, it prints the contents of each of the files in sequence to its output stream. But when `cat` is not given any arguments, it prints contents from its input stream to its output stream (like in the third example above).
>  `cat` 的默认输出流是终端

You can also use `>>` to append to a file. Where this kind of input/output redirection really shines is in the use of _pipes_. The `|` operator lets you “chain” programs such that the output of one is the input of another:

```
missing:~$ ls -l / | tail -n1
drwxr-xr-x 1 root  root  4096 Jun 20  2019 var
missing:~$ curl --head --silent google.com | grep --ignore-case content-length | cut --delimiter=' ' -f2
219
```

We will go into a lot more detail about how to take advantage of pipes in the lecture on data wrangling.

### A versatile and powerful tool
On most Unix-like systems, one user is special: the “root” user. You may have seen it in the file listings above. The root user is above (almost) all access restrictions, and can create, read, update, and delete any file in the system. You will not usually log into your system as the root user though, since it’s too easy to accidentally break something. Instead, you will be using the `sudo` command. As its name implies, it lets you “do” something “as su” (short for “super user”, or “root”). When you get permission denied errors, it is usually because you need to do something as root. Though make sure you first double-check that you really wanted to do it that way!

One thing you need to be root in order to do is writing to the `sysfs` file system mounted under `/sys`. `sysfs` exposes a number of kernel parameters as files, so that you can easily reconfigure the kernel on the fly without specialized tools. **Note that sysfs does not exist on Windows or macOS.**
>  `/sys` 挂载了 `sysfs` 文件系统，它将 kernel 参数暴露为文件，故可以通过修改文件配置 kernel 参数

For example, the brightness of your laptop’s screen is exposed through a file called `brightness` under

```
/sys/class/backlight
```

By writing a value into that file, we can change the screen brightness. Your first instinct might be to do something like:

```
$ sudo find -L /sys/class/backlight -maxdepth 2 -name '*brightness*'
/sys/class/backlight/thinkpad_screen/brightness
$ cd /sys/class/backlight/thinkpad_screen
$ sudo echo 3 > brightness
An error occurred while redirecting file 'brightness'
open: Permission denied
```

This error may come as a surprise. After all, we ran the command with `sudo`! This is an important thing to know about the shell. Operations like `|`, `>`, and `<` are done _by the shell_, not by the individual program. `echo` and friends do not “know” about `|`. They just read from their input and write to their output, whatever it may be. In the case above, the _shell_ (which is authenticated just as your user) tries to open the brightness file for writing, before setting that as `sudo echo` ’s output, but is prevented from doing so since the shell does not run as root. Using this knowledge, we can work around this:
>  注意，shell 中，`|, >, <` 等操作是由 shell 完成的，而不是独立的程序完成的，故 `|, >, <` 等操作对于 `echo` 等命令都是不可知的

```
$ echo 3 | sudo tee brightness
```

Since the `tee` program is the one to open the `/sys` file for writing, and _it_ is running as `root`, the permissions all work out. You can control all sorts of fun and useful things through `/sys`, such as the state of various system LEDs (your path might be different):

```
$ echo 1 | sudo tee /sys/class/leds/input6::scrolllock/brightness
```

## Next steps
At this point you know your way around a shell enough to accomplish basic tasks. You should be able to navigate around to find files of interest and use the basic functionality of most programs. In the next lecture, we will talk about how to perform and automate more complex tasks using the shell and the many handy command-line programs out there.

## Exercises
All classes in this course are accompanied by a series of exercises. Some give you a specific task to do, while others are open-ended, like “try using X and Y programs”. We highly encourage you to try them out.

We have not written solutions for the exercises. If you are stuck on anything in particular, feel free to send us an email describing what you’ve tried so far, and we will try to help you out.

1. For this course, you need to be using a Unix shell like Bash or ZSH. If you are on Linux or macOS, you don’t have to do anything special. If you are on Windows, you need to make sure you are not running cmd.exe or PowerShell; you can use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/) or a Linux virtual machine to use Unix-style command-line tools. To make sure you’re running an appropriate shell, you can try the command `echo $SHELL`. If it says something like `/bin/bash` or `/usr/bin/zsh`, that means you’re running the right program.
2. Create a new directory called `missing` under `/tmp`.
3. Look up the `touch` program. The `man` program is your friend.
4. Use `touch` to create a new file called `semester` in `missing`.
5. Write the following into that file, one line at a time:
    
    ```
    #!/bin/sh
    curl --head --silent https://missing.csail.mit.edu
    ```
    
    The first line might be tricky to get working. It’s helpful to know that `#` starts a comment in Bash, and `!` has a special meaning even within double-quoted (`"`) strings. Bash treats single-quoted strings (`'`) differently: they will do the trick in this case. See the Bash [quoting](https://www.gnu.org/software/bash/manual/html_node/Quoting.html) manual page for more information.
    
6. Try to execute the file, i.e. type the path to the script (`./semester`) into your shell and press enter. Understand why it doesn’t work by consulting the output of `ls` (hint: look at the permission bits of the file).
7. Run the command by explicitly starting the `sh` interpreter, and giving it the file `semester` as the first argument, i.e. `sh semester`. Why does this work, while `./semester` didn’t?
8. Look up the `chmod` program (e.g. use `man chmod`).
9. Use `chmod` to make it possible to run the command `./semester` rather than having to type `sh semester`. How does your shell know that the file is supposed to be interpreted using `sh`? See this page on the [shebang](https://en.wikipedia.org/wiki/Shebang_\(Unix\)) line for more information.
10. Use `|` and `>` to write the “last modified” date output by `semester` into a file called `last-modified.txt` in your home directory.
11. Write a command that reads out your laptop battery’s power level or your desktop machine’s CPU temperature from `/sys`. Note: if you’re a macOS user, your OS doesn’t have sysfs, so you can skip this exercise.

# Shell Tools and Scripting
In this lecture, we will present some of the basics of using bash as a scripting language along with a number of shell tools that cover several of the most common tasks that you will be constantly performing in the command line.

## Shell Scripting
So far we have seen how to execute commands in the shell and pipe them together. However, in many scenarios you will want to perform a series of commands and make use of control flow expressions like conditionals or loops.

Shell scripts are the next step in complexity. Most shells have their own scripting language with variables, control flow and its own syntax. What makes shell scripting different from other scripting programming languages is that it is optimized for performing shell-related tasks. Thus, creating command pipelines, saving results into files, and reading from standard input are primitives in shell scripting, which makes it easier to use than general purpose scripting languages. For this section we will focus on **bash scripting** since it is the most common.
>  shell 脚本语言和其他的脚本语言的区别就在于它被优化用于执行与 shell 相关的任务，它的基础操作是: 创建命令流水线、将结果保存到文件中、从标准输入中读取数据
>  因此，在执行这些任务时，shell 脚本语言比通用脚本语言更易于使用

To assign variables in bash, use the syntax `foo=bar` and access the value of the variable with `$foo`. Note that `foo = bar` will not work since it is interpreted as calling the `foo` program with arguments `=` and `bar`. In general, in shell scripts the space character will perform argument splitting. This behavior can be confusing to use at first, so always check for that.
>  bash 脚本中，为变量赋值的语法是 `foo=bar` ，访问变量的值的语法是 `$foo`
>  注意变量赋值之间没有空格，`foo = bar` 将会被解释为调用 `foo` 程序，传入参数 `=` 和 `bar`

Strings in bash can be defined with `'` and `"` delimiters, but they are not equivalent. Strings delimited with `'` are literal strings and will not substitute variable values whereas `"` delimited strings will.

```bash
foo=bar
echo "$foo"
# prints bar
echo '$foo'
# prints $foo
```

>  bash 脚本中，`'`, `"` 可以用于定义字符串
>  以 `'` 包围的字符串是字面字符串，不会将其中的变量替换为值
>  以 `"` 包围的字符串则会将其中的变量替换为值

As with most programming languages, bash supports control flow techniques including `if`, `case`, `while` and `for`. Similarly, `bash` has functions that take arguments and can operate with them. Here is an example of a function that creates a directory and `cd`s into it.

```bash
mcd () {
    mkdir -p "$1"
    cd "$1"
}
```

>  bash 脚本支持定义函数，如上所示

Here `$1` is the first argument to the script/function. Unlike other scripting languages, bash uses a variety of special variables to refer to arguments, error codes, and other relevant variables. 
>  `$1` 表示脚本/脚本的第一个参数

Below is a list of some of them. A more comprehensive list can be found [here](https://tldp.org/LDP/abs/html/special-chars.html).

- `$0` - Name of the script
- `$1` to `$9` - Arguments to the script. `$1` is the first argument and so on.
- `$@` - All the arguments
- `$#` - Number of arguments
- `$?` - Return code of the previous command
- `$$` - Process identification number (PID) for the current script
- `!!` - Entire last command, including arguments. A common pattern is to execute a command only for it to fail due to missing permissions; you can quickly re-execute the command with sudo by doing `sudo !!`
- `$_` - Last argument from the last command. If you are in an interactive shell, you can also quickly get this value by typing `Esc` followed by `.` or `Alt+.`

>  bash 使用一系列特殊变量来代表参数、错误码和其他相关变量
>  - `$0` 脚本名
>  - `$1` to `$9` 脚本的参数
>  - `$@` 脚本的所有参数
>  - `$#` 参数数量
>  - `$?` 上一个命令的返回码
>  - `$$` 当前脚本的 PID
>  - `!!` 上一个命令 (包括其参数)，例如之前的命令因为权限失败后，我们可以 `sudo !!` 重新执行
>  - `$_` 上一个命令的最后一个参数，在交互式 shell 中，可以通过 `Esc + .` 或 `Alt+.` 获得

Commands will often return output using `STDOUT`, errors through `STDERR`, and a Return Code to report errors in a more script-friendly manner. The return code or exit status is the way scripts/commands have to communicate how execution went. A value of 0 usually means everything went OK; anything different from 0 means an error occurred.
>  命令通常使用 `STDOUT` 返回输出，使用 `STDERR` 返回错误，以及会返回一个返回码
>  返回码为 0 通常意味着 OK，非零则意味着错误

Exit codes can be used to conditionally execute commands using `&&` (and operator) and `||` (or operator), both of which are [short-circuiting](https://en.wikipedia.org/wiki/Short-circuit_evaluation) operators. Commands can also be separated within the same line using a semicolon `;`. The `true` program will always have a 0 return code and the `false` command will always have a 1 return code. Let’s see some examples
>  我们可以利用返回码，以及与或算符 `&&` `||` 来条件执行命令 (利用其短路性质)
>  同一行的命令可以用 `;` 隔开

```bash
false || echo "Oops, fail"
# Oops, fail

true || echo "Will not be printed"
#

true && echo "Things went well"
# Things went well

false && echo "Will not be printed"
#

true ; echo "This will always run"
# This will always run

false ; echo "This will always run"
# This will always run
```

Another common pattern is wanting to get the output of a command as a variable. This can be done with _command substitution_. Whenever you place `$( CMD )` it will execute `CMD`, get the output of the command and substitute it in place. For example, if you do `for file in $(ls)`, the shell will first call `ls` and then iterate over those values. 
>  如果我们希望将一个命令的输出作为变量，可以使用 command substitution 机制
>  `$( CMD )` 将执行 `CMD`，然后将其输出替换 `$( CMD )`
>  例如 `for file in $(ls)` 中，shell 会首先调用 `ls` ，然后在其返回值中迭代

A lesser known similar feature is _process substitution_, `<( CMD )` will execute `CMD` and place the output in a temporary file and substitute the `<()` with that file’s name. This is useful when commands expect values to be passed by file instead of by STDIN. For example, `diff <(ls foo) <(ls bar)` will show differences between files in dirs `foo` and `bar`.
>  一个类似的机制是 process substitution
>  `<( CMD )` 将执行 `CMD` ，将其输出放在一个临时文件中，然后用该文件名替换 `<()`
>  这在命令期待接收文件名时较为有用，例如 `diff <(ls foo) <(ls bar)` 将展示 `foo, bar` 目录中的文件不同

Since that was a huge information dump, let’s see an example that showcases some of these features. It will iterate through the arguments we provide, `grep` for the string `foobar`, and append it to the file as a comment if it’s not found.

```bash
#!/bin/bash

echo "Starting program at $(date)" # Date will be substituted

echo "Running program $0 with $# arguments with pid $$"

for file in "$@"; do
    grep foobar "$file" > /dev/null 2> /dev/null
    # When pattern is not found, grep has exit status 1
    # We redirect STDOUT and STDERR to a null register since we do not care about them
    if [[ $? -ne 0 ]]; then
        echo "File $file does not have any foobar, adding one"
        echo "# foobar" >> "$file"
    fi
done
```

In the comparison we tested whether `$?` was not equal to 0. Bash implements many comparisons of this sort - you can find a detailed list in the manpage for [`test`](https://www.man7.org/linux/man-pages/man1/test.1.html). When performing comparisons in bash, try to use double brackets `[[ ]]` in favor of simple brackets `[ ]`. Chances of making mistakes are lower although it won’t be portable to `sh`. A more detailed explanation can be found [here](http://mywiki.wooledge.org/BashFAQ/031).
>  在比较中，我们测试了 `$?` (上一个命令的返回码) 是否不等于 0
>  在进行比较时，尽量使用 `[[ ]]` 而不是 `[ ]`

When launching scripts, you will often want to provide arguments that are similar. Bash has ways of making this easier, expanding expressions by carrying out filename expansion. These techniques are often referred to as shell _globbing_.
>  bash 提供了 shell globbing (通配) 技术来通过表达式拓展出文件名

- Wildcards - Whenever you want to perform some sort of wildcard matching, you can use `?` and `*` to match one or any amount of characters respectively. For instance, given files `foo`, `foo1`, `foo2`, `foo10` and `bar`, the command `rm foo?` will delete `foo1` and `foo2` whereas `rm foo*` will delete all but `bar`.
- Curly braces `{}` - Whenever you have a common substring in a series of commands, you can use curly braces for bash to expand this automatically. This comes in very handy when moving or converting files.

>  - 通配符: `*, ?`
>  - 花括号: `{}` 

```bash
convert image.{png,jpg}
# Will expand to
convert image.png image.jpg

cp /path/to/project/{foo,bar,baz}.sh /newpath
# Will expand to
cp /path/to/project/foo.sh /path/to/project/bar.sh /path/to/project/baz.sh /newpath

# Globbing techniques can also be combined
mv *{.py,.sh} folder
# Will move all *.py and *.sh files


mkdir foo bar
# This creates files foo/a, foo/b, ... foo/h, bar/a, bar/b, ... bar/h
touch {foo,bar}/{a..h}
touch foo/x bar/y
# Show differences between files in foo and bar
diff <(ls foo) <(ls bar)
# Outputs
# < x
# ---
# > y
```

Writing `bash` scripts can be tricky and unintuitive. There are tools like [shellcheck](https://github.com/koalaman/shellcheck) that will help you find errors in your sh/bash scripts.

Note that scripts need not necessarily be written in bash to be called from the terminal. For instance, here’s a simple Python script that outputs its arguments in reversed order:

```python
#!/usr/local/bin/python
import sys
for arg in reversed(sys.argv[1:]):
    print(arg)
```

The kernel knows to execute this script with a python interpreter instead of a shell command because we included a [shebang](https://en.wikipedia.org/wiki/Shebang_\(Unix\)) line at the top of the script. 
>  脚本不一定需要使用 Bash 编写才能从终端调用
>  因为我们在这段脚本的顶部添加了 shebang 行，故 kernel 知道使用 python 解释器执行该脚本，而不是 bash

It is good practice to write shebang lines using the [`env`](https://www.man7.org/linux/man-pages/man1/env.1.html) command that will resolve to wherever the command lives in the system, increasing the portability of your scripts. To resolve the location, `env` will make use of the `PATH` environment variable we introduced in the first lecture. For this example the shebang line would look like `#!/usr/bin/env python`.
>  写 shebang 行时，建议使用 [`env`](https://www.man7.org/linux/man-pages/man1/env.1.html) 来解析命令在系统中的位置，从而提高脚本的可移植性
>  `env` 将利用我们在第一讲中介绍的 `PATH` 环境变量来解析路径
>  对于这个例子，shebang 行应该写为 `#!/usr/bin/env python`

Some differences between shell functions and scripts that you should keep in mind are:

- Functions have to be in the same language as the shell, while scripts can be written in any language. This is why including a shebang for scripts is important.
- Functions are loaded once when their definition is read. Scripts are loaded every time they are executed. This makes functions slightly faster to load, but whenever you change them you will have to reload their definition.
- Functions are executed in the current shell environment whereas scripts execute in their own process. Thus, functions can modify environment variables, e.g. change your current directory, whereas scripts can’t. Scripts will be passed by value environment variables that have been exported using [`export`](https://www.man7.org/linux/man-pages/man1/export.1p.html)
- As with any programming language, functions are a powerful construct to achieve modularity, code reuse, and clarity of shell code. Often shell scripts will include their own function definitions.

>  shell 函数和脚本的一些差异
>  - 函数必须与 shell 使用相同的语言编写，而脚本可以使用任何语言。这就是为什么为脚本添加 shebang (解释器声明) 很重要
>  - 函数在其定义被读取后仅加载一次，脚本在每次被执行时都会重新加载，故函数的加载会稍快，但函数修改后，必须重新加载
>  - 函数在当前 shell 函数执行，脚本在独立进程执行，故函数可以修改环境变量，脚本不行，脚本是通过值传递获取那些被 `export` 导出的环境变量的

## Shell Tools
### Finding how to use commands
At this point, you might be wondering how to find the flags for the commands in the aliasing section such as `ls -l`, `mv -i` and `mkdir -p`. More generally, given a command, how do you go about finding out what it does and its different options? You could always start googling, but since UNIX predates StackOverflow, there are built-in ways of getting this information.

As we saw in the shell lecture, the first-order approach is to call said command with the `-h` or `--help` flags. A more detailed approach is to use the `man` command. Short for manual, [`man`](https://www.man7.org/linux/man-pages/man1/man.1.html) provides a manual page (called manpage) for a command you specify. For example, `man rm` will output the behavior of the `rm` command along with the flags that it takes, including the `-i` flag we showed earlier. In fact, what I have been linking so far for every command is the online version of the Linux manpages for the commands. 

Even non-native commands that you install will have manpage entries if the developer wrote them and included them as part of the installation process. 
>  即便是 non-native 的命令，只要开发者在安装包中提供了其文档，通常也可以用 `man` 查看

For interactive tools such as the ones based on ncurses, help for the commands can often be accessed within the program using the `:help` command or typing `?`.
>  对于交互式的工具，通常 `:help` 或 `?` 可以获取帮助


Sometimes manpages can provide overly detailed descriptions of the commands, making it hard to decipher what flags/syntax to use for common use cases. [TLDR pages](https://tldr.sh/) are a nifty complementary solution that focuses on giving example use cases of a command so you can quickly figure out which options to use. For instance, I find myself referring back to the tldr pages for [`tar`](https://tldr.inbrowser.app/pages/common/tar) and [`ffmpeg`](https://tldr.inbrowser.app/pages/common/ffmpeg) way more often than the manpages.
>   TLDR pages 专注于提供命令的示例用法，而不像 man pages 那样详细

### Finding files
One of the most common repetitive tasks that every programmer faces is finding files or directories. All UNIX-like systems come packaged with [`find`](https://www.man7.org/linux/man-pages/man1/find.1.html), a great shell tool to find files. `find` will recursively search for files matching some criteria. Some examples:
>  `find` 会递归地搜索匹配特定条件的文件

```bash
# Find all directories named src
find . -name src -type d
# Find all python files that have a folder named test in their path
find . -path '*/test/*.py' -type f
# Find all files modified in the last day
find . -mtime -1
# Find all zip files with size in range 500k to 10M
find . -size +500k -size -10M -name '*.tar.gz'
```

>  例如: 
>  - `find . -name src -type d` 
>  - `find . -path '*/test/*.py -type f`
>  - `find . -mtime -1`
>  - `find . -size +500k -size -10M -name '*.tar.gz'`

Beyond listing files, find can also perform actions over files that match your query. This property can be incredibly helpful to simplify what could be fairly monotonous tasks.
>  `find` 还可以对匹配的文件执行特定的操作

```bash
# Delete all files with .tmp extension
find . -name '*.tmp' -exec rm {} \;
# Find all PNG files and convert them to JPG
find . -name '*.png' -exec convert {} {}.jpg \;
```

>  例如:
>  -  `find . -name '*.tmp' -exec rm {} \;`
>  -  `find . -name '*.png' -exec convert {} {}.jpg \`

Despite `find` ’s ubiquitousness, its syntax can sometimes be tricky to remember. For instance, to simply find files that match some pattern `PATTERN` you have to execute `find -name '*PATTERN*'` (or `-iname` if you want the pattern matching to be case insensitive). You could start building aliases for those scenarios, but part of the shell philosophy is that it is good to explore alternatives. Remember, one of the best properties of the shell is that you are just calling programs, so you can find (or even write yourself) replacements for some. For instance, [`fd`](https://github.com/sharkdp/fd) is a simple, fast, and user-friendly alternative to `find`. It offers some nice defaults like colorized output, default regex matching, and Unicode support. It also has, in my opinion, a more intuitive syntax. For example, the syntax to find a pattern `PATTERN` is `fd PATTERN`.
>  `find` 的语法有时可能难以记住
>  例如，要找到匹配 `PATTERN` 的文件，需要执行 `find -name '*PATTERN*'`
>  `fd` 是 `find` 的另一个用户友好的替代，具有更好的功能和更直观的语法

Most would agree that `find` and `fd` are good, but some of you might be wondering about the efficiency of looking for files every time versus compiling some sort of index or database for quickly searching. That is what [`locate`](https://www.man7.org/linux/man-pages/man1/locate.1.html) is for. `locate` uses a database that is updated using [`updatedb`](https://www.man7.org/linux/man-pages/man1/updatedb.1.html). In most systems, `updatedb` is updated daily via [`cron`](https://www.man7.org/linux/man-pages/man8/cron.8.html). Therefore one trade-off between the two is speed vs freshness. Moreover `find` and similar tools can also find files using attributes such as file size, modification time, or file permissions, while `locate` just uses the file name. A more in-depth comparison can be found [here](https://unix.stackexchange.com/questions/60205/locate-vs-find-usage-pros-and-cons-of-each-other).
>  相较于 `find, fd` 每次查找文件，`locate` 会构建数据库和索引以快速搜索
>  `locate` 使用 `updatedb` 更新其数据库，在大多数系统中 `updatedb` 本身会被 `cron` 每天更新一次
>  不过 `find, fd` 可以根据文件属性来搜索文件，`locate` 则仅使用文件名

### Finding code
Finding files by name is useful, but quite often you want to search based on file _content_. A common scenario is wanting to search for all files that contain some pattern, along with where in those files said pattern occurs. To achieve this, most UNIX-like systems provide [`grep`](https://www.man7.org/linux/man-pages/man1/grep.1.html), a generic tool for matching patterns from the input text. `grep` is an incredibly valuable shell tool that we will cover in greater detail during the data wrangling lecture.
>  `grep` 用于从任意的输入文本中搜索 pattern，我们可以利用它来搜索内容匹配特定 patter 的文件

For now, know that `grep` has many flags that make it a very versatile tool. Some I frequently use are `-C` for getting **C**ontext around the matching line and `-v` for in**v**erting the match, i.e. print all lines that do **not** match the pattern. For example, `grep -C 5` will print 5 lines before and after the match. When it comes to quickly searching through many files, you want to use `-R` since it will **R**ecursively go into directories and look for files for the matching string.
>  `grep` 常用的 flag 有: 
>  -  `-C` 获取匹配行周围的 Context (例如 `grep -C 5`)
>  - `-v` 进行 invert 匹配 (打印所有不匹配的行)
>  - `-R` 会 Recursively 进入目录并查找匹配的文件

But `grep -R` can be improved in many ways, such as ignoring `.git` folders, using multi CPU support, &c. Many `grep` alternatives have been developed, including [ack](https://github.com/beyondgrep/ack3), [ag](https://github.com/ggreer/the_silver_searcher) and [rg](https://github.com/BurntSushi/ripgrep). All of them are fantastic and pretty much provide the same functionality. For now I am sticking with ripgrep (`rg`), given how fast and intuitive it is. Some examples:

```bash
# Find all python files where I used the requests library
rg -t py 'import requests'
# Find all files (including hidden files) without a shebang line
rg -u --files-without-match "^#\!"
# Find all matches of foo and print the following 5 lines
rg foo -A 5
# Print statistics of matches (# of matched lines and files )
rg --stats PATTERN
```

>  `grep` 也有很多替代，例如 `rg`
>  - `rg -t py 'improt requests'`
>  - `rg -u --files-without-math "^#\!"`
>  - `rg foo -A 5`
>  - `rg -stas PATTERN`

Note that as with `find`/`fd`, it is important that you know that these problems can be quickly solved using one of these tools, while the specific tools you use are not as important.

### Finding shell commands
So far we have seen how to find files and code, but as you start spending more time in the shell, you may want to find specific commands you typed at some point. The first thing to know is that typing the up arrow will give you back your last command, and if you keep pressing it you will slowly go through your shell history.

The `history` command will let you access your shell history programmatically. It will print your shell history to the standard output. If we want to search there we can pipe that output to `grep` and search for patterns. `history | grep find` will print commands that contain the substring “find”.
>  `history` 将 shell history 打印到标准输出

In most shells, you can make use of `Ctrl+R` to perform backwards search through your history. After pressing `Ctrl+R`, you can type a substring you want to match for commands in your history. As you keep pressing it, you will cycle through the matches in your history. This can also be enabled with the UP/DOWN arrows in [zsh](https://github.com/zsh-users/zsh-history-substring-search). A nice addition on top of `Ctrl+R` comes with using [fzf](https://github.com/junegunn/fzf/wiki/Configuring-shell-key-bindings#ctrl-r) bindings. `fzf` is a general-purpose fuzzy finder that can be used with many commands. Here it is used to fuzzily match through your history and present results in a convenient and visually pleasing manner.
>  `Ctrl+R` 可以执行 history 上的向后搜索
>  `fzf` 则是一个通用的模糊查找工具，可以和许多命令一起使用

Another cool history-related trick I really enjoy is **history-based autosuggestions**. First introduced by the [fish](https://fishshell.com/) shell, this feature dynamically autocompletes your current shell command with the most recent command that you typed that shares a common prefix with it. It can be enabled in [zsh](https://github.com/zsh-users/zsh-autosuggestions) and it is a great quality of life trick for your shell.
>  基于历史的自动建议功能会根据最近输入的，和当前命令有共同前缀的命令、动态地补全当前 shell 命令
>  在 `zsh` 中可以启动该功能

You can modify your shell’s history behavior, like preventing commands with a leading space from being included. This comes in handy when you are typing commands with passwords or other bits of sensitive information. To do this, add `HISTCONTROL=ignorespace` to your `.bashrc` or `setopt HIST_IGNORE_SPACE` to your `.zshrc`. If you make the mistake of not adding the leading space, you can always manually remove the entry by editing your `.bash_history` or `.zsh_history`.
>  shell 的历史记录行为也可以修改，例如防止以空格开头的命令被包含在内: `HISTCONTROL=ignorespace`
>  历史记录也可以人为通过 `.bash_history` 修改

### Directory Navigation
So far, we have assumed that you are already where you need to be to perform these actions. But how do you go about quickly navigating directories? There are many simple ways that you could do this, such as writing shell aliases or creating symlinks with [ln -s](https://www.man7.org/linux/man-pages/man1/ln.1.html), but the truth is that developers have figured out quite clever and sophisticated solutions by now.

As with the theme of this course, you often want to optimize for the common case. Finding frequent and/or recent files and directories can be done through tools like [`fasd`](https://github.com/clvv/fasd) and [`autojump`](https://github.com/wting/autojump). Fasd ranks files and directories by [_frecency_](https://web.archive.org/web/20210421120120/https://developer.mozilla.org/en-US/docs/Mozilla/Tech/Places/Frecency_algorithm), that is, by both _frequency_ and _recency_. By default, `fasd` adds a `z` command that you can use to quickly `cd` using a substring of a _frecent_ directory. For example, if you often go to `/home/user/files/cool_project` you can simply use `z cool` to jump there. Using autojump, this same change of directory could be accomplished using `j cool`.
>  `fasd, autojump` 可以用于找到频繁使用的和最近访问的文件和目录
>  `fasd` 提供了 `z` 命令，可以用于通过部分匹配来跳转到经常/最近访问的目录
>  `autojump` 则提供了 `j` 命令

More complex tools exist to quickly get an overview of a directory structure: [`tree`](https://linux.die.net/man/1/tree), [`broot`](https://github.com/Canop/broot) or even full fledged file managers like [`nnn`](https://github.com/jarun/nnn) or [`ranger`](https://github.com/ranger/ranger).

## Exercises
1. Read [`man ls`](https://www.man7.org/linux/man-pages/man1/ls.1.html) and write an `ls` command that lists files in the following manner
    
    - Includes all files, including hidden files
    - Sizes are listed in human readable format (e.g. 454M instead of 454279954)
    - Files are ordered by recency
    - Output is colorized
    
    A sample output would look like this
    
    ```
     -rw-r--r--   1 user group 1.1M Jan 14 09:53 baz
     drwxr-xr-x   5 user group  160 Jan 14 09:53 .
     -rw-r--r--   1 user group  514 Jan 14 06:42 bar
     -rw-r--r--   1 user group 106M Jan 13 12:12 foo
     drwx------+ 47 user group 1.5K Jan 12 18:08 ..
    ```
    
2. Write bash functions `marco` and `polo` that do the following. Whenever you execute `marco` the current working directory should be saved in some manner, then when you execute `polo`, no matter what directory you are in, `polo` should `cd` you back to the directory where you executed `marco`. For ease of debugging you can write the code in a file `marco.sh` and (re)load the definitions to your shell by executing `source marco.sh`.
3. Say you have a command that fails rarely. In order to debug it you need to capture its output but it can be time consuming to get a failure run. Write a bash script that runs the following script until it fails and captures its standard output and error streams to files and prints everything at the end. Bonus points if you can also report how many runs it took for the script to fail.
    
    ```
     #!/usr/bin/env bash
    
     n=$(( RANDOM % 100 ))
    
     if [[ n -eq 42 ]]; then
        echo "Something went wrong"
        >&2 echo "The error was using magic numbers"
        exit 1
     fi
    
     echo "Everything went according to plan"
    ```
    
4. As we covered in the lecture `find`’s `-exec` can be very powerful for performing operations over the files we are searching for. However, what if we want to do something with **all** the files, like creating a zip file? As you have seen so far commands will take input from both arguments and STDIN. When piping commands, we are connecting STDOUT to STDIN, but some commands like `tar` take inputs from arguments. To bridge this disconnect there’s the [`xargs`](https://www.man7.org/linux/man-pages/man1/xargs.1.html) command which will execute a command using STDIN as arguments. For example `ls | xargs rm` will delete the files in the current directory.
    
    Your task is to write a command that recursively finds all HTML files in the folder and makes a zip with them. Note that your command should work even if the files have spaces (hint: check `-d` flag for `xargs`).
    
    If you’re on macOS, note that the default BSD `find` is different from the one included in [GNU coreutils](https://en.wikipedia.org/wiki/List_of_GNU_Core_Utilities_commands). You can use `-print0` on `find` and the `-0` flag on `xargs`. As a macOS user, you should be aware that command-line utilities shipped with macOS may differ from the GNU counterparts; you can install the GNU versions if you like by [using brew](https://formulae.brew.sh/formula/coreutils).
5. (Advanced) Write a command or script to recursively find the most recently modified file in a directory. More generally, can you list all files by recency?

# Editors (Vim)
Writing English words and writing code are very different activities. When programming, you spend more time switching files, reading, navigating, and editing code compared to writing a long stream. It makes sense that there are different types of programs for writing English words versus code (e.g. Microsoft Word versus Visual Studio Code).

As programmers, we spend most of our time editing code, so it’s worth investing time mastering an editor that fits your needs. Here’s how you learn a new editor:

- Start with a tutorial (i.e. this lecture, plus resources that we point out)
- Stick with using the editor for all your text editing needs (even if it slows you down initially)
- Look things up as you go: if it seems like there should be a better way to do something, there probably is

If you follow the above method, fully committing to using the new program for all text editing purposes, the timeline for learning a sophisticated text editor looks like this. In an hour or two, you’ll learn basic editor functions such as opening and editing files, save/quit, and navigating buffers. Once you’re 20 hours in, you should be as fast as you were with your old editor. After that, the benefits start: you will have enough knowledge and muscle memory that using the new editor saves you time. Modern text editors are fancy and powerful tools, so the learning never stops: you’ll get even faster as you learn more.

## Which editor to learn?
Programmers have [strong opinions](https://en.wikipedia.org/wiki/Editor_war) about their text editors.

Which editors are popular today? See this [Stack Overflow survey](https://insights.stackoverflow.com/survey/2019/#development-environments-and-tools) (there may be some bias because Stack Overflow users may not be representative of programmers as a whole). [Visual Studio Code](https://code.visualstudio.com/) is the most popular editor. [Vim](https://www.vim.org/) is the most popular command-line-based editor.

### Vim
All the instructors of this class use Vim as their editor. Vim has a rich history; it originated from the Vi editor (1976), and it’s still being developed today. Vim has some really neat ideas behind it, and for this reason, lots of tools support a Vim emulation mode (for example, 1.4 million people have installed [Vim emulation for VS code](https://github.com/VSCodeVim/Vim)). Vim is probably worth learning even if you finally end up switching to some other text editor.

It’s not possible to teach all of Vim’s functionality in 50 minutes, so we’re going to focus on explaining the philosophy of Vim, teaching you the basics, showing you some of the more advanced functionality, and giving you the resources to master the tool.

## Philosophy of Vim
When programming, you spend most of your time reading/editing, not writing. For this reason, Vim is a _modal_ editor: it has different modes for inserting text vs manipulating text. Vim is programmable (with Vimscript and also other languages like Python), and Vim’s interface itself is a programming language: keystrokes (with mnemonic names) are commands, and these commands are composable. Vim avoids the use of the mouse, because it’s too slow; Vim even avoids using the arrow keys because it requires too much movement.

The end result is an editor that can match the speed at which you think.

## Modal editing
Vim’s design is based on the idea that a lot of programmer time is spent reading, navigating, and making small edits, as opposed to writing long streams of text. For this reason, Vim has multiple operating modes.

- **Normal**: for moving around a file and making edits
- **Insert**: for inserting text
- **Replace**: for replacing text
- **Visual** (plain, line, or block): for selecting blocks of text
- **Command-line**: for running a command

Keystrokes have different meanings in different operating modes. For example, the letter `x` in Insert mode will just insert a literal character ‘x’, but in Normal mode, it will delete the character under the cursor, and in Visual mode, it will delete the selection.

In its default configuration, Vim shows the current mode in the bottom left. The initial/default mode is Normal mode. You’ll generally spend most of your time between Normal mode and Insert mode.

You change modes by pressing `<ESC>` (the escape key) to switch from any mode back to Normal mode. From Normal mode, enter Insert mode with `i`, Replace mode with `R`, Visual mode with `v`, Visual Line mode with `V`, Visual Block mode with `<C-v>` (Ctrl-V, sometimes also written `^V`), and Command-line mode with `:`.

You use the `<ESC>` key a lot when using Vim: consider remapping Caps Lock to Escape ([macOS instructions](https://vim.fandom.com/wiki/Map_caps_lock_to_escape_in_macOS)) or create an [alternative mapping](https://vim.fandom.com/wiki/Avoid_the_escape_key#Mappings) for `<ESC>` with a simple key sequence.

## Basics
### Inserting text
From Normal mode, press `i` to enter Insert mode. Now, Vim behaves like any other text editor, until you press `<ESC>` to return to Normal mode. This, along with the basics explained above, are all you need to start editing files using Vim (though not particularly efficiently, if you’re spending all your time editing from Insert mode).

### Buffers, tabs, and windows
Vim maintains a set of open files, called “buffers”. A Vim session has a number of tabs, each of which has a number of windows (split panes). Each window shows a single buffer. Unlike other programs you are familiar with, like web browsers, there is not a 1-to-1 correspondence between buffers and windows; windows are merely views. A given buffer may be open in _multiple_ windows, even within the same tab. This can be quite handy, for example, to view two different parts of a file at the same time.

By default, Vim opens with a single tab, which contains a single window.

### Command-line
Command mode can be entered by typing `:` in Normal mode. Your cursor will jump to the command line at the bottom of the screen upon pressing `:`. This mode has many functionalities, including opening, saving, and closing files, and [quitting Vim](https://twitter.com/iamdevloper/status/435555976687923200).

- `:q` quit (close window)
- `:w` save (“write”)
- `:wq` save and quit
- `:e {name of file}` open file for editing
- `:ls` show open buffers
- `:help {topic}` open help
    - `:help :w` opens help for the `:w` command
    - `:help w` opens help for the `w` movement

## Vim’s interface is a programming language
The most important idea in Vim is that Vim’s interface itself is a programming language. Keystrokes (with mnemonic names) are commands, and these commands _compose_. This enables efficient movement and edits, especially once the commands become muscle memory.

### Movement
You should spend most of your time in Normal mode, using movement commands to navigate the buffer. Movements in Vim are also called “nouns”, because they refer to chunks of text.

- Basic movement: `hjkl` (left, down, up, right)
- Words: `w` (next word), `b` (beginning of word), `e` (end of word)
- Lines: `0` (beginning of line), `^` (first non-blank character), `$` (end of line)
- Screen: `H` (top of screen), `M` (middle of screen), `L` (bottom of screen)
- Scroll: `Ctrl-u` (up), `Ctrl-d` (down)
- File: `gg` (beginning of file), `G` (end of file)
- Line numbers: `:{number}<CR>` or `{number}G` (line {number})
- Misc: `%` (corresponding item)
- Find: `f{character}`, `t{character}`, `F{character}`, `T{character}`
    - find/to forward/backward {character} on the current line
    - `,` / `;` for navigating matches
- Search: `/{regex}`, `n` / `N` for navigating matches

## Selection

Visual modes:

- Visual: `v`
- Visual Line: `V`
- Visual Block: `Ctrl-v`

Can use movement keys to make selection.

## Edits

Everything that you used to do with the mouse, you now do with the keyboard using editing commands that compose with movement commands. Here’s where Vim’s interface starts to look like a programming language. Vim’s editing commands are also called “verbs”, because verbs act on nouns.

- `i` enter Insert mode
    - but for manipulating/deleting text, want to use something more than backspace
- `o` / `O` insert line below / above
- `d{motion}` delete {motion}
    - e.g. `dw` is delete word, `d$` is delete to end of line, `d0` is delete to beginning of line
- `c{motion}` change {motion}
    - e.g. `cw` is change word
    - like `d{motion}` followed by `i`
- `x` delete character (equal to `dl`)
- `s` substitute character (equal to `cl`)
- Visual mode + manipulation
    - select text, `d` to delete it or `c` to change it
- `u` to undo, `<C-r>` to redo
- `y` to copy / “yank” (some other commands like `d` also copy)
- `p` to paste
- Lots more to learn: e.g. `~` flips the case of a character

## Counts

You can combine nouns and verbs with a count, which will perform a given action a number of times.

- `3w` move 3 words forward
- `5j` move 5 lines down
- `7dw` delete 7 words

## Modifiers

You can use modifiers to change the meaning of a noun. Some modifiers are `i`, which means “inner” or “inside”, and `a`, which means “around”.

- `ci(` change the contents inside the current pair of parentheses
- `ci[` change the contents inside the current pair of square brackets
- `da'` delete a single-quoted string, including the surrounding single quotes

# Demo

Here is a broken [fizz buzz](https://en.wikipedia.org/wiki/Fizz_buzz) implementation:

```
def fizz_buzz(limit):
    for i in range(limit):
        if i % 3 == 0:
            print('fizz')
        if i % 5 == 0:
            print('fizz')
        if i % 3 and i % 5:
            print(i)

def main():
    fizz_buzz(10)
```

We will fix the following issues:

- Main is never called
- Starts at 0 instead of 1
- Prints “fizz” and “buzz” on separate lines for multiples of 15
- Prints “fizz” for multiples of 5
- Uses a hard-coded argument of 10 instead of taking a command-line argument

See the lecture video for the demonstration. Compare how the above changes are made using Vim to how you might make the same edits using another program. Notice how very few keystrokes are required in Vim, allowing you to edit at the speed you think.

# Customizing Vim

Vim is customized through a plain-text configuration file in `~/.vimrc` (containing Vimscript commands). There are probably lots of basic settings that you want to turn on.

We are providing a well-documented basic config that you can use as a starting point. We recommend using this because it fixes some of Vim’s quirky default behavior. **Download our config [here](https://missing.csail.mit.edu/2020/files/vimrc) and save it to `~/.vimrc`.**

Vim is heavily customizable, and it’s worth spending time exploring customization options. You can look at people’s dotfiles on GitHub for inspiration, for example, your instructors’ Vim configs ([Anish](https://github.com/anishathalye/dotfiles/blob/master/vimrc), [Jon](https://github.com/jonhoo/configs/blob/master/editor/.config/nvim/init.lua) (uses [neovim](https://neovim.io/)), [Jose](https://github.com/JJGO/dotfiles/blob/master/vim/.vimrc)). There are lots of good blog posts on this topic too. Try not to copy-and-paste people’s full configuration, but read it, understand it, and take what you need.

# Extending Vim

There are tons of plugins for extending Vim. Contrary to outdated advice that you might find on the internet, you do _not_ need to use a plugin manager for Vim (since Vim 8.0). Instead, you can use the built-in package management system. Simply create the directory `~/.vim/pack/vendor/start/`, and put plugins in there (e.g. via `git clone`).

Here are some of our favorite plugins:

- [ctrlp.vim](https://github.com/ctrlpvim/ctrlp.vim): fuzzy file finder
- [ack.vim](https://github.com/mileszs/ack.vim): code search
- [nerdtree](https://github.com/scrooloose/nerdtree): file explorer
- [vim-easymotion](https://github.com/easymotion/vim-easymotion): magic motions

We’re trying to avoid giving an overwhelmingly long list of plugins here. You can check out the instructors’ dotfiles ([Anish](https://github.com/anishathalye/dotfiles), [Jon](https://github.com/jonhoo/configs), [Jose](https://github.com/JJGO/dotfiles)) to see what other plugins we use. Check out [Vim Awesome](https://vimawesome.com/) for more awesome Vim plugins. There are also tons of blog posts on this topic: just search for “best Vim plugins”.

# Vim-mode in other programs

Many tools support Vim emulation. The quality varies from good to great; depending on the tool, it may not support the fancier Vim features, but most cover the basics pretty well.

## Shell

If you’re a Bash user, use `set -o vi`. If you use Zsh, `bindkey -v`. For Fish, `fish_vi_key_bindings`. Additionally, no matter what shell you use, you can `export EDITOR=vim`. This is the environment variable used to decide which editor is launched when a program wants to start an editor. For example, `git` will use this editor for commit messages.

## Readline

Many programs use the [GNU Readline](https://tiswww.case.edu/php/chet/readline/rltop.html) library for their command-line interface. Readline supports (basic) Vim emulation too, which can be enabled by adding the following line to the `~/.inputrc` file:

```
set editing-mode vi
```

With this setting, for example, the Python REPL will support Vim bindings.

## Others

There are even vim keybinding extensions for web [browsers](http://vim.wikia.com/wiki/Vim_key_bindings_for_web_browsers) - some popular ones are [Vimium](https://chrome.google.com/webstore/detail/vimium/dbepggeogbaibhgnhhndojpepiihcmeb?hl=en) for Google Chrome and [Tridactyl](https://github.com/tridactyl/tridactyl) for Firefox. You can even get Vim bindings in [Jupyter notebooks](https://github.com/jupyterlab-contrib/jupyterlab-vim). Here is a [long list](https://reversed.top/2016-08-13/big-list-of-vim-like-software) of software with vim-like keybindings.

# Advanced Vim

Here are a few examples to show you the power of the editor. We can’t teach you all of these kinds of things, but you’ll learn them as you go. A good heuristic: whenever you’re using your editor and you think “there must be a better way of doing this”, there probably is: look it up online.

## Search and replace

`:s` (substitute) command ([documentation](http://vim.wikia.com/wiki/Search_and_replace)).

- `%s/foo/bar/g`
    - replace foo with bar globally in file
- `%s/\[.*\](\(.*\))/\1/g`
    - replace named Markdown links with plain URLs

## Multiple windows

- `:sp` / `:vsp` to split windows
- Can have multiple views of the same buffer.

## Macros

- `q{character}` to start recording a macro in register `{character}`
- `q` to stop recording
- `@{character}` replays the macro
- Macro execution stops on error
- `{number}@{character}` executes a macro {number} times
- Macros can be recursive
    - first clear the macro with `q{character}q`
    - record the macro, with `@{character}` to invoke the macro recursively (will be a no-op until recording is complete)
- Example: convert xml to json ([file](https://missing.csail.mit.edu/2020/files/example-data.xml))
    - Array of objects with keys “name” / “email”
    - Use a Python program?
    - Use sed / regexes
        - `g/people/d`
        - `%s/<person>/{/g`
        - `%s/<name>\(.*\)<\/name>/"name": "\1",/g`
        - …
    - Vim commands / macros
        - `Gdd`, `ggdd` delete first and last lines
        - Macro to format a single element (register `e`)
            - Go to line with `<name>`
            - `qe^r"f>s": "<ESC>f<C"<ESC>q`
        - Macro to format a person
            - Go to line with `<person>`
            - `qpS{<ESC>j@eA,<ESC>j@ejS},<ESC>q`
        - Macro to format a person and go to the next person
            - Go to line with `<person>`
            - `qq@pjq`
        - Execute macro until end of file
            - `999@q`
        - Manually remove last `,` and add `[` and `]` delimiters

# Resources

- `vimtutor` is a tutorial that comes installed with Vim - if Vim is installed, you should be able to run `vimtutor` from your shell
- [Vim Adventures](https://vim-adventures.com/) is a game to learn Vim
- [Vim Tips Wiki](http://vim.wikia.com/wiki/Vim_Tips_Wiki)
- [Vim Advent Calendar](https://vimways.org/2019/) has various Vim tips
- [Vim Golf](http://www.vimgolf.com/) is [code golf](https://en.wikipedia.org/wiki/Code_golf), but where the programming language is Vim’s UI
- [Vi/Vim Stack Exchange](https://vi.stackexchange.com/)
- [Vim Screencasts](http://vimcasts.org/)
- [Practical Vim](https://pragprog.com/titles/dnvim2/) (book)

# Exercises

1. Complete `vimtutor`. Note: it looks best in a [80x24](https://en.wikipedia.org/wiki/VT100) (80 columns by 24 lines) terminal window.
2. Download our [basic vimrc](https://missing.csail.mit.edu/2020/files/vimrc) and save it to `~/.vimrc`. Read through the well-commented file (using Vim!), and observe how Vim looks and behaves slightly differently with the new config.
3. Install and configure a plugin: [ctrlp.vim](https://github.com/ctrlpvim/ctrlp.vim).
    1. Create the plugins directory with `mkdir -p ~/.vim/pack/vendor/start`
    2. Download the plugin: `cd ~/.vim/pack/vendor/start; git clone https://github.com/ctrlpvim/ctrlp.vim`
    3. Read the [documentation](https://github.com/ctrlpvim/ctrlp.vim/blob/master/readme.md) for the plugin. Try using CtrlP to locate a file by navigating to a project directory, opening Vim, and using the Vim command-line to start `:CtrlP`.
    4. Customize CtrlP by adding [configuration](https://github.com/ctrlpvim/ctrlp.vim/blob/master/readme.md#basic-options) to your `~/.vimrc` to open CtrlP by pressing Ctrl-P.
4. To practice using Vim, re-do the [Demo](https://missing.csail.mit.edu/2020/editors/#demo) from lecture on your own machine.
5. Use Vim for _all_ your text editing for the next month. Whenever something seems inefficient, or when you think “there must be a better way”, try Googling it, there probably is. If you get stuck, come to office hours or send us an email.
6. Configure your other tools to use Vim bindings (see instructions above).
7. Further customize your `~/.vimrc` and install more plugins.
8. (Advanced) Convert XML to JSON ([example file](https://missing.csail.mit.edu/2020/files/example-data.xml)) using Vim macros. Try to do this on your own, but you can look at the [macros](https://missing.csail.mit.edu/2020/editors/#macros) section above if you get stuck.