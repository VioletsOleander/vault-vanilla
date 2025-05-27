---
edition: "6"
---
# Part 1  Learning the Shell 
# 1 What Is the Shell? 
When we speak of the command line, we are really referring to the shell. The shell is a program that takes keyboard commands and passes them to the operating system to carry out. 
>  shell 是一个程序，接收键盘命令，传递给 OS 执行

Almost all Linux distributions supply a shell program from the GNU Project called bash. The name “bash” is an acronym for “Bourne Again SHell”, a reference to the fact bash is an enhanced replacement for sh, the original Unix shell program written by Steve Bourne. 
>  几乎所有 Linux 发行版提供 GNU Project 的一个 shell 程序: bash
>  bash 是原始 Unix shell 程序的增强替代版本

## Terminal Emulators 
When using a graphical user interface (GUI), we need another program called a terminal emulator to interact with the shell. 
>  在使用 GUI 时，为了与 shell 交互，我们还需要另一个程序，称为终端模拟器

If we look through our desktop menus, we will probably find one. KDE uses konsole and GNOME uses gnome-terminal, though it's likely called simply “terminal” on our menu. A number of other terminal emulators are available for Linux, but they all basically do the same thing; give us access to the shell. You will probably develop a preference for one or another terminal emulator based on the number of bells and whistles it has. 
>  终端模拟器 ("terminal") 基本上做的都是同一件事: 为我们提供对 shell 的访问

## Making Your First Keystrokes 
So let's get started. Launch the terminal emulator! Once it comes up, we should see something like this: 

```
[me@linuxbox ~]$
```

This is called a shell prompt and it will appear whenever the shell is ready to accept input. While it may vary in appearance somewhat depending on the distribution, it will typically include your `username@machinename` , followed by the current working directory (more about that in a little bit) and a dollar sign.

>  shell prompt 通常会包含 username, machine name, current working directory, 以及一个 `$`

Note: If the last character of the prompt is a pound sign (“#”) rather than a dollar sign, the terminal session has superuser privileges. This means either we are logged in as the root user or we selected a terminal emulator that provides superuser (administrative) privileges. 

Assuming things are good so far, let's try some typing. Enter some gibberish at the prompt like so: 

```
[me@linuxbox ~]$ kaekfjaeifj 
```
Because this command makes no sense, the shell tells us so and give us another chance. 

```
bash: kaekfjaeifj: command not found 
[me@linuxbox ~]$ 
```

### Command History 
If we press the up-arrow key, we will see that the previous command `kaekfjaeifj` reappears after the prompt. This is called command history. Most Linux distributions remember the last 1000 commands by default. Press the down-arrow key and the previous command disappears. 
>  大多数 Linux 发行版保存最近的 1000 条命令历史

### Cursor Movement 
Recall the previous command by pressing the up-arrow key again. If we try the left and right-arrow keys, we'll see how we can position the cursor anywhere on the command line. This makes editing commands easy. 

**A Few Words About Mice and Focus** 
While the shell is all about the keyboard, you can also use a mouse with your terminal emulator. A mechanism built into the X Window System (the underlying engine that makes the GUI go) supports a quick copy and paste technique. If you highlight some text by holding down the left mouse button and dragging the mouse over it (or double clicking on a word), it is copied into a buffer maintained by X. Pressing the middle mouse button will cause the text to be pasted at the cursor location. Try it. 
>  终端模拟程序一般会提供鼠标支持
>  X Window 系统 (负责 GUI 的底层引擎) 内置了一种机制，支持快速的复制粘贴: 按住鼠标左键并拖动或双击某个单词，可以将内容复制到 X 维护的缓冲区
>  按下中键，可以粘贴

Note: Don't be tempted to use Ctrl-c and Ctrl-v to perform copy and paste inside a terminal window. They don't work. These control codes have different meanings to the shell and were assigned many years before the release of Microsoft Windows. 

Your graphical desktop environment (most likely KDE or GNOME), in an effort to behave like Windows, probably has its focus policy set to “click to focus.” This means for a window to get focus (become active) you need to click on it. 
>  GUI 环境遵循的焦点策略通常是 “点击以聚焦” (使得窗口活跃)

This is contrary to the traditional X behavior of “focus follows mouse” which means that a window gets focus just by passing the mouse over it. The window will not come to the foreground until you click on it but it will be able to receive input. Setting the focus policy to “focus follows mouse” will make the copy and paste technique even more useful. Give it a try if you can (though some desktop environments no longer support it). I think if you give it a chance you will prefer it. You will find this setting in the configuration program for your window manager. 

## Try Some Simple Commands 
Now that we have learned to enter text in our terminal emulator, let's try a few simple commands. Let's begin with the date command, which displays the current time and date. 

```
[me@linuxbox ~]$ date 
Thu Mar 8 15:09:41 EST 2025 
```

>  `date` : 当前时间

Another handy command is uptime which displays how long the system has been running and the average number of processes running over various periods of time. 

```
[me@linuxbox ~]$ uptime 
15:12:22 up 3 days, 23:40, 7 users, load average: 0.37, 0.37, 0.64 
```

>  `uptime`: 系统运行时间，以及不同时间段内的平均进程数

To see the current amount of free space on our disk drives, enter df. 

<html><body><table><tr><td colspan="6">[ me@linuxbox ~]$ df</td></tr><tr><td>Filesystem</td><td>1K-blocks</td><td></td><td></td><td></td><td>Used Available Use% Mounted on</td></tr><tr><td>/dev/sda2</td><td>15115452</td><td>5012392</td><td>9949716</td><td>34%／</td><td></td></tr><tr><td>/dev/sda5</td><td>59631908</td><td>26545424</td><td>30008432</td><td>47% /home</td><td></td></tr></table></body></html> 
<html><body><table><tr><td>/dev/sda1</td><td>147764</td><td>17370</td><td>122765</td><td>13% /boot</td><td></td></tr><tr><td>tmpfs</td><td>256856</td><td>0</td><td>256856</td><td></td><td>0% /dev/shm</td></tr></table></body></html> 

>  `df`: 磁盘容量

Likewise, to display the amount of free memory, enter the free command. 

<html><body><table><tr><td rowspan="2">[me@linuxbox ~]$ free</td><td colspan="5"></td></tr><tr><td>used</td><td>free</td><td>shared</td><td>buffers</td><td>cached</td></tr><tr><td>total Mem: 513712</td><td>503976</td><td>9736</td><td>0</td><td>5312</td><td>122916</td></tr><tr><td>-/+ buffers/cache: 375748</td><td></td><td>137964</td><td></td><td></td><td></td></tr><tr><td>Swap: 1052248</td><td>104712</td><td>947536</td><td></td><td></td><td></td></tr></table></body></html> 

>  `free` : 内存容量

## Ending a Terminal Session 
We can end a terminal session by either closing the terminal emulator window, by entering the exit command at the shell prompt, or pressing Ctrl-d. 

![](https://cdn-mineru.openxlab.org.cn/extract/e52c4087-794e-43f3-821f-0f8eb1fdf822/644bd0034d249d2d9a4f66eb98a0879c62a0b81884893ac67fb31c1a116a0de5.jpg) 

>  `exit` = Ctrl-d

**The Console Behind the Curtain** 
Even if we have no terminal emulator running, several terminal sessions continue to run behind the graphical desktop. We can access these sessions, called virtual terminals or virtual consoles, by pressing Ctrl-Alt-F1 through Ctrl-Alt-F6 on most Linux distributions. When a session is accessed, it presents a login prompt into which we can enter our username and password. To switch from one virtual console to another, press Alt-F1 through Alt-F6. On most system we can return to the graphical desktop by pressing Alt-F7. 

## Summing Up 
This chapter marks the beginning of our journey into the Linux command line with an introduction to the shell and a brief glimpse at the command line and a lesson on how to start and end a terminal session. We also saw how to issue some simple commands and perform a little light command line editing. That wasn't so scary was it? 

In the next chapter, we'll learn a few more commands and wander around the Linux file system. 

## Further Reading
- To learn more about Steve Bourne, father of the Bourne Shell, see this Wikipedia article: http://en.wikipedia.org/wiki/Steve_Bourne
- This Wikipedia article is about Brian Fox, the original author of bash: https://en.wikipedia.org/wiki/Brian_Fox_(computer_programmer)
- Here is an article about the concept of shells in computing: http://en.wikipedia.org/wiki/Shell_(computing)

# 2 Navigation 
The first thing we need to learn (besides how to type) is how to navigate the file system on our Linux system. In this chapter we will introduce the following commands: 

- `pwd` – Print name of current working directory 
- `cd` – Change directory 
- `ls` – List directory contents 

## Understanding the File System Tree 
Like Windows, a Unix-like operating system such as Linux organizes its files in what is called a hierarchical directory structure. This means they are organized in a **tree-like** pattern of directories (sometimes called folders in other systems), which may contain files and other directories. The first directory in the file system is called the root directory. The root directory contains files and subdirectories, which contain more files and subdirectories and so on. 

Note that unlike Windows, which has a separate file system tree for each storage device, Unix-like systems such as Linux always have a single file system tree, regardless of how many drives or storage devices are attached to the computer. Storage devices are attached (or more correctly, mounted) at various points on the tree according to the whims of the system administrator, the person (or people) responsible for the maintenance of the system. 
>  Windows 为每个存储设备都赋予独立的文件系统树
>  Unix-like 系统则只有单个文件系统树，存储设备根据系统管理员的意愿，挂载到文件系统树中的不同点上

## The Current Working Directory 
Most of us are probably familiar with a graphical file manager which represents the file system tree as in Figure 1. Notice that the tree is usually shown upended, that is, with the root at the top and the various branches descending below. 

![](https://cdn-mineru.openxlab.org.cn/extract/e52c4087-794e-43f3-821f-0f8eb1fdf822/258db881dd4047b64960a16439896187fa687423d6237749bae72a3e61b45b55.jpg) 

Figure 1: file system tree as shown by a graphical file manager 

However, the command line has no pictures, so to navigate the file system tree we need to think of it in a different way. 

Imagine that the file system is a maze shaped like an upside-down tree and we are able to stand in the middle of it. At any given time, we are inside a single directory and we can see the files contained in the directory and the pathway to the directory above us (called the parent directory) and any subdirectories below us. The directory we are standing in is called the current working directory. To display the current working directory, we use the `pwd` (print working directory) command. 

```
[ me@linuxbox ~]$ pwd 
/home/me 
```

>  `pwd`: print current directory

When we first log in to our system (or start a terminal emulator session) our current working directory is set to our home directory. Each user account is given its own home directory and it is the only place a regular user is allowed to write files. 

## Listing the Contents of a Directory 
To list the files and directories in the current working directory, we use the ls command. 

```
[me@linuxbox ~]$ ls 
Desktop Documents Music Pictures Public Templates Videos 
```

>  `ls` : list

Actually, we can use the ls command to list the contents of any directory, not just the current working directory, and there are many other fun things it can do as well. We'll spend more time with ls in the next chapter. 

## Changing the Current Working Directory 
To change our working directory (where we are standing in our tree-shaped maze) we use the cd command. To do this, type cd followed by the pathname of the desired working directory. A pathname is the route we take along the branches of the tree to get to the directory we want. We can specify pathnames in one of two different ways; as absolute pathnames or as relative pathnames. Let's look at absolute pathnames first. 

### Absolute Pathnames 
An absolute pathname begins with the root directory and follows the tree branch by branch until the path to the desired directory or file is completed. 

For example, there is a directory on our system in which most of our system's programs are installed. The directory’s pathname is /usr/bin. This means from the root directory (represented by the leading slash in the pathname) there is a directory called "usr" which contains a directory called "bin". 

>  大多数系统程序通常安装在 `/usr/bin`

```
[me@linuxbox ~]$ cd /usr/bin 
[me@linuxbox bin]$ pwd 
/usr/bin 
[me@linuxbox bin]$ ls 
..Listing of many, many files...
```

Now we can see that we have changed the current working directory to `/usr/bin` and that it is full of files. **Notice how the shell prompt has changed?** As a convenience, it is usually set up to automatically display **the name of the working directory.** 

### Relative Pathnames 
Where an absolute pathname starts from the root directory and leads to its destination, a relative pathname starts from the working directory. To do this, it uses a couple of special notations to represent relative positions in the file system tree. These special notations are "." (dot) and ".." (dot dot). 

The "." notation refers to the working directory and the ".." notation refers to the working directory's parent directory. Here is how it works. Let's change the working directory to /usr/bin again. 

```
[me@linuxbox ~]$ cd /usr/bin 
[me@linuxbox bin]$ pwd /usr/bin 
```

Now let's say that we wanted to change the working directory to the parent of /usr/bin which is /usr. We could do that two different ways, either using an absolute pathname: 

```
[me@linuxbox bin] cd /usr 
[me@linuxbox usr]$ pwd 
/usr 
```

or, using a relative pathname. 

```
[me@linuxbox bin]$ cd ..
[me@linuxbox usr]$ pwd 
/usr 
```

Two different methods with identical results. Which one should we use? The one that requires the least typing! 

Likewise, we can change the working directory from /usr to /usr/bin in two different ways, either using an absolute pathname: 

```
[me@linuxbox usr]$ cd /usr/bin 
[me@linuxbox bin]$ pwd 
/usr/bin 
```

or, using a relative pathname. 

```
[me@linuxbox usr]$ cd ./bin 
[me@linuxbox bin]$ pwd 
/usr/bin 
```

Now, there is something important to point out here. In almost all cases, we can omit the  "./". It is implied. Typing: 

<html><body><table><tr><td>[me@linuxbox usr]$ cd bin</td></tr></table></body></html> 

does the same thing. **In general, if we do not specify a pathname to something, the working directory will be assumed.** 

## Some Helpful Shortcuts 
In Table 2-1 we see some useful ways the current working directory can be quickly changed. 

Table 2-1: cd Shortcuts 
<html><body><table><tr><td>Shortcut</td><td>Result</td></tr><tr><td>cd</td><td>Changes the working directory to your home directory.</td></tr><tr><td>cd-</td><td>Changes the working directory to the previous working directory.</td></tr><tr><td>cd ~user_name</td><td>Changes the working directory to the home directory of user_name. For example, cd ~bob will change the directory to the home directory of user “bob."</td></tr></table></body></html> 

>  `cd -` : 切换到前一个工作目录 (挺好用的)
>  `cd ~user_name`: 切换到 `user_name` 的家目录

**Important Facts About Filenames** 
On Linux systems, files are named in a manner similar to other systems such as Windows, but there are some important differences. 
1. Filenames that begin with a period character are hidden. This only means that ls will not list them unless you say ls -a. When your account was created, several hidden files were placed in your home directory to configure things for your account. In Chapter 11 we will take a closer look at some of these files to see how you can customize your environment. In addition, some applications place their configuration and settings files in your home directory as hidden files. 
>  `.` 开头的文件是隐藏文件，需要使用 `ls -a` 才能列出

2. Filenames and commands in Linux, like Unix, are case sensitive. The filenames “File1” and “file1” refer to different files. 
3. Linux has no concept of a “file extension” like some other operating systems. You may name files any way you like. The contents and/or purpose of a file is determined by other means. Although Unix-like operating systems don’t use file extensions to determine the contents/purpose of files, many application programs do. 
4. Though Linux supports long filenames that may contain embedded spaces and punctuation characters, limit the punctuation characters in the names of files you create to period, dash, and underscore. Most importantly, do not embed spaces in filenames. If you want to represent spaces between words in a filename, use underscore characters. You will thank yourself later. 

## Summing Up 
This chapter explained how the shell treats the directory structure of the system. We learned about absolute and relative pathnames and the basic commands that we use to move about that structure. In the next chapter we will use this knowledge to go on a tour of a modern Linux system. 

# 3 Exploring the System 
Now that we know how to move around the file system, it's time for a guided tour of our Linux system. Before we start however, we’re going to learn some more commands that will be useful along the way. 

- `ls` – List directory contents 
- `file` – Determine file type 
- `less` – View file contents 

## Having More Fun with ls 
The ls command is probably the most used Linux command, and for good reason. With it, we can see directory contents and determine a variety of important file and directory attributes. As we have seen, we can simply enter ls to get a list of files and subdirectories contained in the current working directory. 

```
[me@linuxbox ~]$ ls 
Desktop Documents Music Pictures Public Templates Videos 
```

Besides the current working directory, we can specify a directory to list, like so: 

```
me@linuxbox ~]$ ls /usr 
bin games include lib local sbin share src 
```

We can even specify multiple directories. In the following example, we list both the user's home directory (symbolized by the “~” character) and the /usr directory. 

```
[me@linuxbox ~]$ ls ~ /usr 
/home/me: 
Desktop Documents Music Pictures Public Templates Videos 
/usr: 
bin games include lib local sbin share src 
```

>  `ls` 可以接收多个参数

We can also change the format of the output to reveal more detail. 

<html><body><table><tr><td>[me@linuxbox ~]$ ls -l</td></tr><tr><td>total 56</td></tr><tr><td>drwxrwxr-x 2 me me 4096 2017-10-26 17:20 Desktop drwxrwxr-x 2 4096 2017-10-26 17:20 D0cuments</td></tr><tr><td>me me drwxrwxr-x 2 4096 2017-10-26 17:20 Music drwxrwxr-x 4096 2017-10-26 17:20 Pictures</td></tr><tr><td>me me</td></tr><tr><td>2 me me drwxrwxr-x 2 me me 4096 2017-10-26 17:20 Public</td></tr><tr><td>drwxrwxr-x 2 me me 4096 2017-10-26 17:20 Templates</td></tr><tr><td>drwxrwxr-x 2 me me 4096 2017-10-26 17:20 Vide0s</td></tr><tr><td></td></tr></table></body></html> 

By adding “-l” to the command, we changed the output to the long format. 

### Options and Arguments 
This brings us to a very important point about how most commands work. Commands are often followed by one or more options that modify their behavior, and further, by one or more arguments, the items upon which the command acts. **So most commands look kind of like this:** 

<html><body><table><tr><td>command -options arguments</td></tr></table></body></html> 

Most commands use options which consist of **a single character preceded by a dash**, for example, “-l”. Many commands, however, including those from the GNU Project, also support long options, consisting of **a word preceded by two dashes**. 

Also, many commands allow **multiple short options to be strung together**. In the following example, the ls command is given two options, which are the l option to produce long format output, and the t option to sort the result by the file's modification time. 

```
[me@linuxbox ~]$ ls -lt 
```

We'll add the long option “--reverse” to reverse the order of the sort. 

```
[me@linuxbox ~]$ ls -lt --reverse 
```

Note that command options, like filenames in Linux, are case-sensitive. 

The ls command has a large number of possible options. The most common are listed in Table 3-1. 

Table 3-1: Common ls Options 
<html><body><table><tr><td> Option</td><td>Long Option</td><td>Description</td></tr><tr><td>-a</td><td>--all</td><td>List all files, even those with names that begin with a period, which are normally not listed (that is, hidden).</td></tr><tr><td>-A</td><td>- -almost-all</td><td>Like the -a option above except it does not list . (current directory) and .. (parent directory).</td></tr><tr><td>-d</td><td>- -directory</td><td>Ordinarily, if a directory is specified, ls will list the contents of the directory, not the directory itself. Use this option in conjunction with the - l option to see details about the directory rather than its contents.</td></tr><tr><td>-F</td><td>--classify</td><td>This option will append an indicator character to the end of each listed name. For example, a forward slash (/) if the name is a directory.</td></tr><tr><td>-h</td><td>- - human-readable</td><td>In long format listings, display file sizes in human readable format rather than in bytes.</td></tr><tr><td>-l</td><td></td><td>Display results in long format.</td></tr><tr><td>-r</td><td>- -reverse</td><td>Display the results in reverse order. Normally, ls displays its results in ascending alphabetical order.</td></tr><tr><td>-S</td><td></td><td>Sort results by file size.</td></tr><tr><td>-t</td><td></td><td>Sort by modification time.</td></tr></table></body></html> 

>  `-a, --all`: 列出所有文件，包括 `.` 开头的隐藏文件
> `-A, --almost-all` : 类似 `-a`，但不列出 `.` 和 `..`
>  `-d, --directory`: 通常情况下，如果给定目录，`ls` 会列出其内容，而不是目录本身，使用 `-d -l`，可以查看目录本身的细节
>  `-F, --classify`: 为每个列出的名字后附加一个指示字符，例如目录会添加 `/`
>  `-h, --human-readable`: 和 `-l` 一起使用时，文件大小以人类可读方式显示
>  `-l`
>  `-r, --reverse`: 逆序展示，正常情况下是按字典序升序展示
>  `-S`: 按文件大小排序
>  `-t`: 按修改时间排序

### A Longer Look at Long Format 
As we saw earlier, the -l option causes ls to display its results in long format. This format contains a great deal of useful information. Here is the Examples directory from an early Ubuntu system: 

```
-rw-r--r-- 1 root root 3576296 2017-04-03 11:05 Experience ubuntu.ogg -rw-r--r-- 1 root root 1186219 2017-04-03 11:05 kubuntu-leaflet.png -rw-r--r-- 1 root root 47584 2017-04-03 11:05 logo-Edubuntu.png -rw-r--r-- 1 root root 44355 2017-04-03 11:05 logo-Kubuntu.png -rw-r--r-- 1 root root 34391 2017-04-03 11:05 logo-Ubuntu.png -rw-r--r-- 1 root root 32059 2017-04-03 11:05 oo-cd-cover.odf -rw-r--r-- 1 root root 159744 2017-04-03 11:05 oo-derivatives.doc -rw-r--r-- 1 root root 27837 2017-04-03 11:05 oo-maxwell.odt -rw-r--r-- 1 root root 98816 2017-04-03 11:05 oo-trig.xls -rw-r--r-- 1 root root 453764 2017-04-03 11:05 oo-welcome.odt -rw-r--r-- 1 root root 358374 2017-04-03 11:05 ubuntu Sax.ogg 
```

Table 3-2 provides us with a look at the different fields from one of the files and their meanings. 

Table 3-2: ls Long Listing Fields 
<html><body><table><tr><td>Field</td><td>Meaning</td></tr><tr><td>-rw-r--r-</td><td>Access rights to the file. The first character indicates the type of file. Among the different types, a leading dash means a regular file, while a “d" indicates a directory. The next three characters are the access rights for the file's owner, the next three are for members of the file's group, and the final three are for everyone else. Chapter 9 "Permissions" discusses the full meaning of this in more detail.</td></tr><tr><td>1</td><td>File's number of hard links. See the sections "Symbolic Links" and "Hard Links" later in this chapter.</td></tr><tr><td>root</td><td>The username of the file's owner.</td></tr><tr><td>root 32059</td><td>The name of the group that owns the file.</td></tr><tr><td>2017-04-03 11:05</td><td>Size of the file in bytes.</td></tr><tr><td></td><td>Date and time of the file's last modification.</td></tr><tr><td>oo-cd-cover.odf</td><td>Name of the file.</td></tr></table></body></html> 

>  `-rw-r--r--`: 文件的访问权限，第一个字符表示文件的类型: `-` 表示常规文件; `d` 表示目录，之后三个字符表示文件所有者的访问权限，之后三个字符表示文件组成员的访问权限，之后三个字符表示所有其他成员的访问权限
>  `1`: 文件的硬链接数量
>  `root`: 文件所有者的用户名
>  `root`: 文件所有组的组名
>  `32059`: 文件的大小 (bytes)
>  `2017-04-03 11:05`: 文件上次修改的日期和时间
>  `oo-cd-cover.pdf`: 文件名
 
## Determining a File's Type with file 
As we explore the system it will be useful to know what kind of data files contain. To do this we will use the file command to determine a file's type. As we discussed earlier, filenames in Linux are not required to reflect a file's contents. While a file named “picture.jpg” would normally be expected to contain a JPEG compressed image, it is not required to in Linux. We invoke the file command this way: 

```
file filename 
```

>  `file`: 确定文件类型

When invoked, the file command will print a brief description of the file's contents. For example: 

```
[me@linuxbox ~]$ file picture.jpg 
picture.jpg: JPEG image data, JFIF standard 1.01 
```

There are many kinds of files. In fact, one of the basic ideas in Unix-like operating systems such as Linux is that “everything is a file.” As we proceed with our lessons, we will see just how true that statement is. 

While many of the files on our system are familiar, for example MP3 and JPEG, there are many kinds that are a little less obvious and a few that are quite strange. 

## Viewing File Contents with less 
The less command is a program to view text files. Throughout our Linux system, there are many files that contain human-readable text. The less program provides a convenient way to examine them. 

**What Is “Text”?** 
There are many ways to represent information on a computer. All methods involve defining a relationship between the information and some numbers that will be used to represent it. Computers, after all, only understand numbers and all data is converted to numeric representation. 

Some of these representation systems are very complex (such as compressed video files), while others are rather simple. One of the earliest and simplest is called ASCII text. ASCII (pronounced "As-Key") is short for American Standard Code for Information Interchange. This is a simple encoding scheme that was first used on Teletype machines to map keyboard characters to numbers. 

**Text is a simple one-to-one mapping of characters to numbers.** It is very compact. **Fifty characters of text translates to fifty bytes of data.** It is important to understand that text only contains a simple mapping of characters to numbers. It is not the same as a word processor document such as one created by Microsoft Word or LibreOffice Writer. Those files, in contrast to simple ASCII text, contain many non-text elements that are used to describe its structure and formatting. **Plain ASCII text files contain only the characters themselves and a few rudimentary control codes such as tabs, carriage returns and line feeds.** 
>  纯文本文件仅包含文字字符本身，以及基础的控制字符

Throughout a Linux system, many files are stored in text format and there are many Linux tools that work with text files. Even Windows recognizes the importance of this format. The well-known NOTEPAD.EXE program is an editor for plain ASCII text files. 

Why would we want to examine text files? Because many of the files that contain system settings (called configuration files) are stored in this format, and being able to read them gives us insight about how the system works. In addition, some of the actual programs that the system uses (called scripts) are stored in this format. In later chapters, we will learn how to edit text files in order to modify systems settings and write our own scripts, but for now we will just look at their contents. 

The less command is used like this: 

```
less filename 
```

Once started, the less program allows us to scroll forward and backward through a text file. For example, to examine the file that defines all the system's user accounts, enter the following command: 

```
[me@linuxbox ~]$ less /etc/passwd 
```

Once the less program starts, we can view the contents of the file. If the file is longer than one page, we can scroll up and down. To exit less, press the q key. 

The table below lists the most common keyboard commands used by less. 

Table 3-3: less Commands 
<html><body><table><tr><td>Command</td><td>Action</td></tr><tr><td>Page Up or b</td><td>Scroll back one page</td></tr><tr><td>Page Down or space</td><td>Scroll forward one page</td></tr><tr><td>Up arrow</td><td>Scroll up one line</td></tr><tr><td>Down arrow</td><td>Scroll down one line</td></tr><tr><td>G</td><td>Move to the end of the text file</td></tr><tr><td>1G or g</td><td>Move to the beginning of the text file</td></tr><tr><td>/characters</td><td>Search forward to the next occurrence of characters</td></tr><tr><td>n</td><td>Search for the next occurrence of the previous search</td></tr><tr><td>h </td><td>Display help screen</td></tr><tr><td>q</td><td>Quit less</td></tr></table></body></html> 

**Less Is More** 
The less program was designed as an **improved replacement** of an earlier Unix program called more. The name “less” is a play on the phrase “less is more” — a motto of modernist architects and designers. 

less falls into the class of programs called “**pagers**,” programs that allow the easy viewing of long text documents in a **page by page** manner. Whereas the more program could only page forward, the less program allows paging both forward and backward and has many other features as well. 

## Taking a Guided Tour 
The file system layout on a Linux system is much like that found on other Unix-like systems. The design is actually specified in a published standard called the **Linux Filesystem Hierarchy Standard**. Not all Linux distributions conform to the standard exactly but most come pretty close. 
>  Linux 文件系统的布局设计遵循 the Linux Filesystem Hierarchy Standard

Next, we are going to wander around the file system ourselves to see what makes our Linux system tick. This will give us a chance to practice our navigation skills. One of the things we will discover is that many of the interesting files are in plain human-readable text. As we go about our tour, try the following: 

1. cd into a given directory 
2. List the directory contents with ls -l 
3. If you see an interesting file, determine its contents with `file` 
4. If it looks like it might be text, try viewing it with less 
5. If we accidentally attempt to view a non-text file and it scrambles the terminal window, we can recover by entering the `reset` command. 

Remember the copy and paste trick! If you are using a mouse, you can double click on a filename to copy it and middle click to paste it into commands. 

As we wander around, don't be afraid to look at stuff. Regular users are largely prohibited from messing things up. That's the system administrator's job! If a command complains about something, just move on to something else. Spend some time looking around. The system is ours to explore. Remember, in Linux, there are no secrets! 

Table 3-4 lists just a few of the directories we can explore. There may be some slight differences depending on our Linux distribution. Don't be afraid to look around and try more! 

Table 3-4: Directories Found on Linux Systems 
<html><body><table><tr><td>Directory</td><td>Comments</td></tr><tr><td>/</td><td>The root directory. Where everything begins.</td></tr><tr><td>/bin</td><td>Contains binaries (programs) that must be present for the system to boot and run. Note that modern Linux distributions have deprecated /bin in favor of /usr/bin</td></tr></table></body></html> 

>  `/bin`: 包含用于系统启动和运行的程序，现代 Linux 发行版改为使用 `/usr/bin`

<html><body><table><tr><td>Directory Comments</td><td colspan="2"></td></tr><tr><td rowspan="3">/boot</td><td colspan="2">Contains the Linux kernel, initial RAM disk image (for drivers needed at boot time), and the boot loader.</td></tr><tr><td colspan="2">Interesting files:</td></tr><tr><td colspan="2">/boot/grub/grub.cfg or menu.lst,which is used to configure the boot loader. /boot/vmlinuz (or something similar), the Linux</td></tr><tr><td>/dev</td><td colspan="2">kernel This is a special directory that contains device nodes. “Everything is a file" also applies to devices. Here is where</td></tr><tr><td rowspan="3">/etc</td><td colspan="2">the kernel maintains a list of all the devices it understands. The /etc directory contains all of the system-wide configuration files. It also contains a collection of shell scripts that start each of the system services at boot time. Everything in this directory should be readable text.</td></tr><tr><td colspan="2">Interesting files: While everything in /etc is interesting, here are some all-time favorites: /etc/crontab, on systems that use the cron</td></tr><tr><td colspan="2">program, this file defines when automated jobs will run. /etc/fstab, a table of storage devices and their associated mount points. /etc/passwd, a list of the user accounts. In normal configurations, each user is given a directory in</td></tr><tr><td>/home /lib</td><td colspan="2">/home. Ordinary users can only write files in their home directories. This limitation protects the system from errant user activity.</td></tr><tr><td>/lost+found</td><td colspan="2">Contains shared library files used by the core system programs. These are similar to dynamic link libraries (DLLs) in Windows. This directory has been deprecated in modern distributions in favor of /usr / lib.</td></tr><tr><td></td><td colspan="2">Each formatted partition or device using a Linux file system, such as ext4, will have this directory. It is used in the case of a partial recovery from a file system corruption event. Unless something really bad has happened to our system, this directory will remain empty.</td></tr></table></body></html> 

> -  `/boot`: 包含 kernel, 初始 RAM 磁盘镜像 (用于 boot/引导时所需驱动程序), 以及 boot loader (引导加载程序)。其中 `/boot/grub/grub.cfg or menu.lst` 用于配置 boot loader；`/boot/vmlinuz` 为 kernel
> -  `/dev`: 包含了设备节点, kernel 在这里维护它所理解的设备列表
> -  `/etc`: 包含了所有系统范围的配置文件，也包含了在 boot 时启动系统服务的一系列 shell 脚本，该目录下的所有文件应该都是人类可读的。其中 `/etc/crontab` 定义了自动任务的运行时间；`/etc/fstab` 记录了存储设备及其关联的挂载点；`/etc/passwd` 记录了用户账户列表
> - `/home`: 包含了各个用户目录 
> - `/lib`: 包含了核心系统程序使用的共享库，现代 Linux 系统中改为使用 `/usr/lib`
> - `/lost+found`: 每个使用 Linux 文件系统格式化的分区或设备都会有这个目录，用于在文件系统损坏时部分恢复，该目录通常为空 


<html><body><table><tr><td>Directory</td><td>Comments</td></tr><tr><td>/media</td><td>On modern Linux systems the /media directory will contain the mount points for removable media such as USB drives, CD-ROMs, etc. that are mounted automatically at insertion.</td></tr><tr><td>/mnt</td><td>On older Linux systems, the /mnt directory contains mount points for devices that have been mounted manually.</td></tr><tr><td>/opt</td><td>The /opt directory is used to install “optional" software. This is mainly used to hold commercial software products that might be installed on the system.</td></tr><tr><td>/proc</td><td>The /proc directory is special. It's not a real file system in the sense of files stored on the hard drive. Rather, it is a virtual file system maintained by the Linux kernel. The “files" it contains are peepholes into the kernel itself. The files are readable and will give us a picture of how the kernel sees the computer. Browsing this directory can reveal</td></tr><tr><td>/root</td><td>many details about the computer's hardware. This is the home directory for the root account.</td></tr><tr><td>/run</td><td>This is a modern replacement for the traditional /tmp directory (see below). Unlike /tmp, the /run directory is mounted using the tempfs file system type which stores its contents in memory rather than on a physical disk.</td></tr><tr><td>/sbin</td><td>This directory contains“system" binaries. These are programs that perform vital system tasks that are generally reserved for the superuser. Note that modern Linux distributions have deprecated /sbin in favor of /usr/sbin (see below).</td></tr><tr><td>/sys</td><td>The /sys directory contains information about devices that have been detected by the kernel. This is much like the contents of the /dev directory but is more detailed including such things actual hardware addresses.</td></tr><tr><td>/tmp</td><td>The /tmp directory is intended for the storage of temporary, transient files created by various programs. Some distributions empty this directory each time the system is rebooted.</td></tr></table></body></html> 

>  - `/media`: 包含可移动媒体例如 USB 设备, CD-ROMs 等的挂载点，这些设备会在插入时自动挂载
>  - `/mnt`: 包含手动挂载设备的挂载点
>  - `/opt`: 用于安装 “optional” 的软件，主要用于安装于系统中的商业软件
>  - `/proc`: 该目录不是硬盘上存储文件的真实文件系统，而是由 kernel 维护的虚拟文件系统，其中的 “文件” 包含的是窥视 kernel 本身的窗口，这些文件都是人类可读的，会为我们提供 kernel 是如何看待计算机的视角
>  - `/root`: root 账户的主目录
>  - `/run`: 传统 `/tmp` 目录的现代替代品，`/run` 使用 tempfs 文件系统类型挂载，其内容存储在内存中而不是物理磁盘上
>  - `/sbin`: 包含了 “系统” 二进制文件，这些文件执行重要的系统任务，现代 Linux 使用 `/usr/sbin`
>  - `/sys`: 包含 kernel 检测到的设备信息，其内容类似 `/dev` ，但更加详细，包括实际的硬件地址等信息
>  - `/tmp`: 存储各种程序创建的临时文件，一些发行版会在系统 reboot 时清空此目录


<html><body><table><tr><td>Directory</td><td>Comments</td></tr><tr><td>/usr</td><td>The /usr directory tree is likely the largest one on a Linux system. It contains all the programs and support files used by regular users.</td></tr><tr><td>/usr/bin</td><td>/usr/bin contains the executable programs installed by the Linux distribution. It is not uncommon for this directory to hold thousands of programs.</td></tr><tr><td>/usr/lib /usr/local</td><td>The shared libraries for the programs in /usr/bin. The /usr/ local tree is where programs that are not</td></tr><tr><td></td><td>included with the distribution but are intended for system- wide use are installed. Programs compiled from source code are normally installed in /usr/ local/bin. On a newly installed Linux system, this tree exists, but it will be empty until the system administrator puts something in it.</td></tr><tr><td>/usr/sbin /usr/share</td><td>Contains more system administration programs.</td></tr><tr><td></td><td>/usr/share contains all the shared data used by programs in /usr/bin. This includes things such as default configuration files, icons, screen backgrounds, sound files, etc.</td></tr><tr><td>/usr/share/doc</td><td>Most packages installed on the system will include some kind of documentation. In /usr/share/doc, we will find documentation files organized by package.</td></tr><tr><td>/var</td><td>With the exception of /tmp and /home, the directories we have looked at so far remain relatively static, that is, their contents don't change. The /var directory tree is where data that is likely to change is stored. Various databases, spool files, user mail, etc. are located here.</td></tr><tr><td>/var/log</td><td>/var/log contains log files, records of various system activity. These are important and should be monitored from time to time. The most useful ones are /var/log/messages and/or /var/log/syslog though these are not available on all systems. Note that for security reasons, some systems only allow the superuser to view log files.</td></tr></table></body></html> 

>  - `/usr`: 是 Linux 系统上最大的目录，包含了所有普通用户使用的程序和支持文件
>  - `/usr/bin`: 包含了由 Linux 发行版安装的可执行程序，这个目录通常会包含上千个程序
>  - `/usr/lib`: 包含了 `/usr/bin` 中程序的共享库
>  - `/usr/local`: 包含了不在发行版中会包含的，但是为系统范围使用的程序，从源码编译的程序通常会被安装在 `/usr/local/bin`，在新安装的 Linux 系统长，这个树通常是空的，直到管理员向里面添加内容
> - `/usr/sbin`: 包含更多的系统管理程序
> - `/usr/share`: 包含了 `/usr/bin` 中程序使用的所有共享数据，例如默认配置文件、图标、屏幕背景、声音文件等
> - `/usr/share/doc`: 包含了系统上安装的包的文档
> - `/var`: 除了 `/tmp, /home` 目录以外，我们目前讨论的目录都相对静态，即它们的内容不会改变，而 `/var` 目录用于存储可能发生变化的数据，各种数据库、假脱机文件、用户邮件等都位于这里
> - `/var/log`: 包含了日志文件，记录了各种系统活动，这些日志文件非常重要，会被定期监控，其中最有用的是 `/var/log/messages, /var/log/syslog`，处于安全原因，一般仅有超级用户可以查看这些日志文件

<html><body><table><tr><td>Directory</td><td>Comments</td></tr><tr><td>~/.config and ~/.local</td><td>These two directories are located in the home directory of each desktop user. They are used to store user-specific configuration data for desktop applications.</td></tr></table></body></html> 

>  `~/.config, ~/.local`: 这两个目录位于每个桌面用户的家目录，它们用于存储特定于用户针对桌面应用的的配置数据

## Symbolic Links 
As we look around, we are likely to see a directory listing (for example in /usr/lib) with an entry like this: 

<html><body><table><tr><td>lrwxrwxrwx 1 root root 11 2007-08-11 07:34 libc.s0.6 -> libc-2.6.s0</td></tr></table></body></html> 

Notice how the first letter of the listing is “l” and the entry seems to have two filenames? This is a special kind of a file called a symbolic link (also known as a soft link or symlink). In most Unix-like systems it is possible to have a file referenced by multiple names. While the value of this might not be obvious, it is really a useful feature. 
>  第一个字母 `l` 表示了该文件是一个特殊的文件类型，称为符号链接/软连接 (symbolic/soft/sym link)
>  在多数类 Unix 系统中，可以用多个名称引用相同文件，这通过符号链接实现

Picture this scenario: A program requires the use of a shared resource of some kind contained in a file named “foo,” but “foo” has frequent version changes. It would be good to include the version number in the filename so the administrator or other interested party could see what version of “foo” is installed. This presents a problem. If we change the name of the shared resource, we have to track down every program that might use it and change it to look for a new resource name every time a new version of the resource is installed. That doesn't sound like fun at all. 

Here is where symbolic links save the day. Suppose we install version 2.6 of “foo,” which has the filename “foo- $2.6^{\prime\prime}$ and then create a symbolic link simply called “foo” that points to “foo-2.6.” This means that when a program opens the file “foo”, it is actually opening the file “foo- $2.6^{\prime\prime}$ . Now everybody is happy. The programs that rely on “foo” can find it and we can still see what actual version is installed. When it is time to upgrade to “foo2.7,” we just add the file to our system, delete the symbolic link “foo” and create a new one that points to the new version. Not only does this solve the problem of the version upgrade, but it also allows us to keep both versions on our machine. Imagine that “foo $2.7^{\mathfrak{n}}$ has a bug (damn those developers!) and we need to revert to the old version. Again, we just delete the symbolic link pointing to the new version and create a new symbolic link pointing to the old version. 

>  符号链接的一个使用场景：假设程序需要使用名为 `foo` 的资源，而 `foo` 会经常更新版本，其实际名称带有版本号，例如 `foo-2.6`
>  我们可以创建名为 `foo` 的符号链接，连接到实际文件，这使得程序代码不需要频繁修改对 `foo` 的引用形式

The directory listing at the beginning of this section (from the /usr/lib directory of a Fedora system) shows a symbolic link called libc.so.6 that points to a shared library file called libc-2.6.so. This means that programs looking for libc.so.6 will actually get the file libc-2.6.so. We will learn how to create symbolic links in the next chapter. 

## Hard Links 
While we are on the subject of links, we need to mention that there is a second type of link called a hard link. Hard links also allow files to have multiple names, but they do it in a different way. We’ll talk more about the differences between symbolic and hard links in the next chapter. 

## Summing Up 
With our tour behind us, we have learned a lot about our system. We've seen various files and directories and their contents. One thing we should take away from this is how open the system is. In Linux there are many important files that are plain human-readable text. Unlike many proprietary systems, Linux makes everything available for examination and study. 

## Further Reading 
- The full version of the Linux Filesystem Hierarchy Standard can be found here:  https://refspecs.linuxfoundation.org/fhs.shtml 
- An article about the directory structure of Unix and Unix-like systems:  http://en.wikipedia.org/wiki/Unix_directory_structure 
- A detailed description of the ASCII text format: http://en.wikipedia.org/wiki/ASCII 

# 4 Manipulating Files and Directories 
Now we’re are ready for some real work! This chapter will introduce the following commands: 

- `cp` – Copy files and directories 
- `mv` – Move/rename files and directories 
- `mkdir` – Create directories 
- `rm` – Remove files and directories 
- `ln` – Create hard and symbolic links 

These five commands are among the most frequently used Linux commands. They are used for manipulating both files and directories. 

Now, to be frank, some of the tasks performed by these commands are more easily done with a graphical file manager. With a file manager, we can drag and drop a file from one directory to another, cut and paste files, delete files, and so on. So why use these old command line programs? 

The answer is power and flexibility. While it is easy to perform simple file manipulations with a graphical file manager, complicated tasks can be easier with the command line programs. For example, how could we copy all the HTML files from one directory to another but only copy files that do not exist in the destination directory or are newer than the versions in the destination directory? It's pretty hard with a file manager but pretty easy with the command line. 

```
cp -u *.html destination
```

## Wildcards 
Before we begin using our commands, we need to talk about a shell feature that makes these commands so powerful. Since the shell uses filenames so much, it provides special characters to help us rapidly specify groups of filenames. These special characters are called wildcards. 

Using wildcards (which is also known as globbing) allows us to select filenames based on patterns of characters. Table 4-1 lists the wildcards and what they select. 

>  shell 提供了通配符解析语义，使用通配符可以选择出匹配模式的文件名

Table 4-1: Wildcards 
<html><body><table><tr><td>Wildcard</td><td>Meaning</td></tr><tr><td>*</td><td>Matches any characters</td></tr><tr><td>？</td><td>Matches any single character</td></tr><tr><td>[characters]</td><td>Matches any character that is a member of the set characters</td></tr><tr><td>[!characters] or [^characters]</td><td>Matches any character that is not a member of the set characters</td></tr><tr><td>[[:class:]]</td><td>Matches any character that is a member of the specified class</td></tr></table></body></html> 

>  通配符包括
>  `*`: 匹配任意 (单个/多个) 字符
>  `?`: 匹配任意单个字符
>  `[character]`: 匹配 `[]` 中指定的字符
>  `[!chacacter] or [^character]`: 匹配 `[]` 中未指定的字符
>  `[[:class:]]`: 匹配对应 `class` 中的字符

Table 4-2 lists the most commonly used character classes. 

Table 4-2: Commonly Used Character Classes 
<html><body><table><tr><td>Character Class</td><td>Meaning</td></tr><tr><td>[:alnum:]</td><td>Matches any alphanumeric character</td></tr><tr><td>[:alpha:]</td><td>Matches any alphabetic character</td></tr><tr><td>[:digit:]</td><td> Matches any numeral</td></tr><tr><td>[:lower:]</td><td>Matches any lowercase letter</td></tr><tr><td>[:upper:]</td><td>Matches any uppercase letter</td></tr></table></body></html> 

>  字符类包括
>  `[:alnum:]`: 字母和数字
>  `[:alpha:]`: 字母
>  `[:digit:]`: 数字
>  `[:lower:]`: 小写字母
>  `[:upper:]`: 大写字母

Using wildcards makes it possible to construct sophisticated selection criteria for filenames. Table 4-3 provides some examples of patterns and what they match. 

Table 4-3: Wildcard Examples 
<html><body><table><tr><td>Pattern</td><td>Matches</td></tr><tr><td>*</td><td>All files</td></tr><tr><td>g*</td><td>Any file beginning with “g"</td></tr><tr><td>b*.txt</td><td>Any file beginning with “b" followed by any characters and ending with ". txt"</td></tr> <tr><td>Data???</td><td>Any file beginning with “Data" followed by exactly three characters</td></tr><tr><td>[abc]*</td><td>Any file beginning with either an “a", a “b",or a “c”</td></tr><tr><td>BACKUP .[0-9][0-9][0-9]</td><td>Any file beginning with “BACKUP." followed by exactly three numerals</td></tr><tr><td>[[:upper:]]*</td><td>Any file beginning with an uppercase letter</td></tr><tr><td>[![:digit:] ]*</td><td>Any file not beginning with a numeral</td></tr><tr><td>*[[:lower:]123]</td><td>Any file ending with a lowercase letter or the numerals “1",“2", or “3"</td></tr></table></body></html> 

Wildcards can be used with any command that accepts filenames as arguments, but we’ll talk more about that in Chapter 7, "Seeing the World As the Shell Sees It. 

**Character Ranges** 
If you are coming from another Unix-like environment or have been reading some other books on this subject, you may have encountered the \[A-Z\] and \[a-z\] character range notations. These are traditional Unix notations and worked in older versions of Linux as well. They can still work, but you have to be careful with them because they will not produce the expected results unless properly configured. For now, you should avoid using them and use character classes instead. 
>  字符范围通配符例如 `[A-Z], [a-z]` 应该避免使用，因为它们可能不会产生预期的结果

**Dot Files** 
If we look at our home directory with ls using the -a option we will notice that there are a number of files and directories whose name begin with a dot. As we have discussed, these files are hidden. **It’s not a special attribute of the file; it only means that the file will not appear in the output of ls unless the -a or -A options are included.** This hidden characteristic also applies to wildcards. Hidden files will not appear unless we use a wildcard pattern such as `.*`. However, when we do this we will also see both . (the current directory) and .. (the current directory’s parent) in the results. To exclude them we can use patterns such as . `.[!.]*` or `.??*` . 
>  通配符 (例如 `*`) 通常不会匹配 `. ` 开头的文件，为此，需要显示指定为 `.*`

**Wildcards Work in the GUI Too** 
Wildcards are especially valuable not only because they are used so frequently on the command line, but because they are also supported by some graphical file managers. 

In Nautilus (the file manager for GNOME), you can select files by pressing Ctrl-s and entering a file selection pattern with wildcards and the files in the currently displayed directory will be selected. In some versions of Dolphin and Konqueror (the file managers for KDE), you can enter wildcards directly on the location bar. For example, if you want to see all the files starting with a lowercase “u” in the /usr/bin directory, enter “/ usr/bin/u\*” in the location bar and it will display the result. 

Many ideas originally found in the command line interface make their way into the graphical interface, too. It is one of the many things that make the Linux desktop so powerful. 

## `mkdir` – Create Directories 
The `mkdir` command is used to create directories. It works like this: 

```
mkdir directory...
```

**A note on notation:** When three periods follow an argument in the description of a command (as above), it means that the argument can be repeated, thus the following command:
>  命令描述中，如果参数后跟随 `...` ，表示该参数可以重复

```
mkdir dir1
```

would create a single directory named dir1, while the following: 

```
mkdir dir1 dir2 dir3 
```

would create three directories named dir1, dir2, and dir3. 

## `cp` – Copy Files and Directories 
The cp command copies files or directories. It can be used two different ways. The following: 

```
cp item1 item2 
```

copies the single file or directory item1 to the file or directory item2 and the following: 

```
cp item... directory 
```

copies multiple items (either files or directories) into a directory. 

>  `cp item1 item2` 将单个文件或目录 `item1` 拷贝到文件或目录 `item2` 中
>  `cp item... directory` 将多个文件拷贝到目录中

### Useful Options and Examples 
Table 4-4 lists some of the commonly used options for cp. 

Table 4-4: cp Options 
<html><body><table><tr><td>Option</td><td>Long Option</td><td>Meaning</td></tr><tr><td>-a</td><td>- -archive</td><td>Copy the files and directories and all of their attributes, including ownerships and permissions. Normally, copies take on the default attributes of the user performing the copy. We'll take a look at file permissions in Chapter 9 "Permissions."</td></tr><tr><td>-i</td><td>--interactive</td><td>Before overwriting an existing file, prompt the user for confirmation. If this option is not specified, cp will silently (meaning there will be no warning) overwrite files.</td></tr><tr><td>-r</td><td>--recursive</td><td>Recursively copy directories and their contents. This option (or the - a option) is required when copying directories.</td></tr><tr><td>-u</td><td>- -update</td><td>When copying files from one directory to another, only copy files that either don't exist or are newer than the existing corresponding files, in the destination directory. This is useful when copying large numbers of files as it skips files that don't need to be copied.</td></tr><tr><td>-V</td><td>- -verbose</td><td>Display informative messages as the copy is performed.</td></tr></table></body></html> 

>  `-a, --archive`: 拷贝文件和目录以及它们的所有属性，包括所有权和权限，注意，通常 (没有选项 `-a` 时)，拷贝得到的文件的属性采用的是执行 ` cp ` 的用户的属性
>  `-i, --interactive`: 在覆盖现存的文件之前，提示用户确认，如果没有指定 `-i`，则 `cp` 会静默地覆盖现存文件 (不会发出警告)
>  `-r, --recursive`: 递归复制目录及其内容，在拷贝目录时，需要使用该选项 (或者使用 `-a` 选项)
>  `-u, --update`: 将文件从一个目录拷贝到另一个目录时，仅拷贝另一个目录中不存在的文件或者比现存文件更新的文件，这在拷贝大量文件时非常有用，因为它跳过了不需要拷贝的文件
>  `-V, --verbose`

Table 4-5: cp Examples 
<html><body><table><tr><td>Command</td><td>Results</td></tr><tr><td>cp file1 file2</td><td>Copy file1 to file2. If file2 exists, it is overwritten with the contents of file1. If file2 does not exist, it is created.</td></tr><tr><td>cp -i file1 file2</td><td>Same as previous command, except that if file2 exists, the user is prompted before it is overwritten.</td></tr><tr><td>cp file1 file2 dir1</td><td>Copy file1 and file2 into directory dir1. The directory dir1 must already exist.</td></tr><tr><td>cp dir1/* dir2</td><td>Using a wildcard, copy all the files in dir1 into dir2. The directory dir2 must already exist.</td></tr><tr><td>cp -r dir1 dir2</td><td>Copy the contents of directory dir1 to directory dir2. If directory dir2 does not exist, it is created and, after the copy, will contain the same contents as directory dir1. If directory dir2 does exist, then directory dir1 (and its contents) will be copied into dir2.</td></tr></table></body></html> 

>  `cp file1 file2`: 将 `file1` 拷贝至 `file2`，如果 `file2` 存在，则会被 `file1` 的内容覆盖写，如果 `file2` 不存在，则会被创建
>  `cp -i file1 file2`: 和上一个命令效果一样，差异在于会在覆盖 `file2` 之前提示用户确认
>  `cp file1 file2 dir1`: 将 `file1, file2` 拷贝到目录 `dir1`，目录 `dir1` 必须已经存在
>  `cp dir1/* dir2`: 使用通配符，将目录 `dir1` 中的所有文件拷贝到 `dir2`，`dir2` 必须已经存在
>  `cp -r dir1 dir2`: 将 `dir1` 中的内容拷贝到 `dir2`，如果 `dir2` 不存在，则创建它，并且 `dir2` 的内容将和 `dir1` 相同，如果 ` dir2 ` 已经存在，则目录 ` dir1 ` 本身会被拷贝到 ` dir2 ` 中

## `mv` – Move and Rename Files 
The mv command performs both file moving and file renaming, depending on how it is used. In either case, the original filename no longer exists after the operation. mv is used in much the same way as cp, as shown here: 

```
mv item1 item2 
```

to move or rename the file or directory item1 to item2 or: 

```
mv item... directory 
```

to move one or more items from one directory to another. 

>  `mv item1 item2` 将目录/文件 `item1` 重命名为 `item2`
>  `mv item... directory` 将多个 `item` 移动到 `directory` 中

### Useful Options and Examples 
mv shares many of the same options as cp as described in Table 4-6. 

Table 4-6: mv Options 
<html><body><table><tr><td>Option</td><td>Long Option</td><td>Meaning</td></tr><tr><td>-i</td><td>--interactive</td><td>Before overwriting an existing file, prompt the user for confirmation. If this option is not specified, mv will silently overwrite files.</td></tr><tr><td>-u</td><td>- -update</td><td>When moving files from one directory to another, only move files that either don't exist, or are newer than the existing corresponding files in the destination directory.</td></tr><tr><td>-V</td><td>--verbose</td><td>Display informative messages as the move is performed.</td></tr></table></body></html> 

>  `-i, --interactive`: 在覆盖写现存的文件之前，提醒用户确认，如果没有指定 `-i`，则静默覆盖写
>  `-u, --update`: 当把文件从一个目录移动到另一个目录时，仅移动目标目录中不存在的或者更新的文件
>  `-V, --verbose`

Table 4-7 provides some examples of mv usage. 

Table 4-7: mv Examples 
<html><body><table><tr><td>Command</td><td>Results</td></tr><tr><td>mv file1 file2</td><td>Move file1 to file2. If file2 exists, it is overwritten with the contents of file1. If file2 does not exist, it is created. In either case, file1 ceases to exist.</td></tr><tr><td>mv -i file1 file2</td><td>Same as the previous command, except that if file2 exists, the user is prompted before it is overwritten.</td></tr><tr><td>mv file1 file2 dir1</td><td>Move file1 and file2 into directory dir1. The directory dir1 must already exist.</td></tr></table></body></html> 
<html><body><table><tr><td>mv dir1 dir2</td><td>If directory dir2 does not exist, create directory dir2 and move the contents of directory dir1 into dir2 and delete directory dir1. If directory dir2 does exist, move directory dir1 (and its contents) into directory dir2.</td></tr></table></body></html> 

>  `mv file1 file2`: 将 `file1` 移动到 `file2`，如果 `file2` 存在，则内容会被覆盖写，如果 `file2` 不存在，则会被创建
>  `mv -i file1 file2`: 和上一个命令一样，差异在于会在覆盖写之前提示用户
>  `mv file1 file2 dir1`: 将 `file1, file2` 移动到目录 `dir1`，`dir1` 必须已经存在
>  `mv dir1 dir2`: 如果 `dir2` 不存在，则创建 `dir2` ，将 `dir1` 的内容移动到 `dir2`，如果 `dir2` 已经存在，则将 `dir1` 移动到 `dir2` 中

## `rm` – Remove Files and Directories 
The rm command is used to remove (delete) files and directories, as shown here: 

```
rm item...
```

where item is one or more files or directories. 

### Useful Options and Examples 
Table 4-8 describes some of the common options for rm. 

Table 4-8: rm Options 
<html><body><table><tr><td>Option</td><td>Long Option</td><td>Meaning</td></tr><tr><td>-i</td><td>--interactive</td><td>Before deleting an existing file, prompt the user for confirmation. If this option is not specified, rm will silently delete files.</td></tr><tr><td>-r</td><td>- -recursive</td><td>Recursively delete directories. This means that if a directory being deleted has subdirectories, delete them too. To delete a directory, this option must be specified.</td></tr><tr><td>-f</td><td>--force</td><td>Ignore nonexistent files and do not prompt. This overrides the - - interactive option.</td></tr><tr><td>-V</td><td>- -verbose</td><td>Display informative messages as the deletion is performed.</td></tr></table></body></html> 

>  `-i, --interactive`: 删除现存文件之前，请求确认
>  `-r, --recursive`: 递归删除目录，要删除一个目录，必须给定 `-r`
>  `-f, --force`: 忽略不存在的文件并且不提示用户，该选项会覆盖 `--interactive`
>  `-V, --verbose`

Table 4-9 provides some examples of using the rm command. 

Table 4-9: rm Examples 
<html><body><table><tr><td>Command</td><td>Results</td></tr>
<tr><td>rm file1</td><td>Delete file1 silently.</td></tr><tr><td>rm -i file1</td><td>Same as the previous command, except that the user is prompted for confirmation before the deletion is performed.</td></tr><tr><td>rm -r file1 dir1</td><td>Delete file1 and dir1 and its contents.</td></tr><tr><td>rm -rf file1 dir1</td><td>Same as the previous command, except that if either file1 or dir1 do not exist, rm will continue silently.</td></tr></table></body></html> 

>  `rm file1`: 静默删除 `file1`
>  `rm -i file1`
>  `rm -r file1 dir1`: 删除 `file1, dir1`
>  `rm -rf file1 dir1`

**Be Careful with rm!** 
Unix-like operating systems such as Linux do not have an undelete command. Once you delete something with rm, it's gone. Linux assumes you're smart and you know what you're doing. 

Be particularly careful with wildcards. Consider this classic example. Let's say you want to delete just the HTML files in a directory. To do this, you type the following: 

```
rm *.html 
```

This is correct, but if you accidentally place a space between the \* and the .html like so: 

```
rm * .html 
```

the rm command will delete all the files in the directory and then complain that there is no file called `.html`. 

Here is a useful tip: whenever you use wildcards with rm (besides carefully checking your typing!), test the wildcard first with ls. This will let you see the files that will be deleted. Then press the up arrow key to recall the command and replace ls with rm. 
>  无论何时在使用 `rm` 时使用通配符，都先用 `ls` 测试通配符

## ln – Create Links 
The ln command is used to create either hard or symbolic links. It is used in one of two ways. The following creates a hard link: 

```
ln file link 
```

>  `ln file link` 创建硬链接

The following creates a symbolic link: 

```
ln -s item link 
```

to create a symbolic link where item is either a file or a directory. 

>  `ln -s item link`: `item` 可以是文件或目录

### Hard Links 
Hard links are the original Unix way of creating links, compared to symbolic links, which are more modern. By default, every file has a single hard link that gives the file its name. When we create a hard link, we create an additional directory entry for a file. Hard links have two important limitations: 

1. A hard link cannot reference a file outside its own file system. This means a link cannot reference a file that is not on the same disk partition as the link itself. 
2. A hard link may not reference a directory. 

>  默认情况下，每个文件只有一个硬链接，这个硬链接赋予了文件其名称
>  当我们创建一个额外的硬链接时，我们为该文件创建了一个额外的目录条目
>  硬链接的两个限制是
>  1. 硬链接不能引用其所在文件系统以外的文件，也就是一个链接不能引用不在同一磁盘分区上的文件
>  2. 硬链接不能引用目录

A hard link is indistinguishable from the file itself. Unlike a symbolic link, when we list a directory containing a hard link we will see no special indication of the link. When a hard link is deleted, the link is removed but the contents of the file itself continue to exist (that is, its space is not deallocated) until all links to the file are deleted. 
>  硬链接和文件本身无法区分
>  和符号链接不同，当我们列出包含硬链接的目录时，不会看到任何特殊的链接指示，当硬链接被删除时，链接会被移除，但文件本身的内容将继续存在 (即空间不会被释放)，直到所有指向该文件的硬链接都被删除

It is important to be aware of hard links because you might encounter them from time to time, but modern practice prefers symbolic links, which we will cover next. 

### Symbolic Links 
Symbolic links were created to overcome the limitations of hard links. Symbolic links work by creating a special type of file that contains a text pointer to the referenced file or directory. In this regard, they operate in much the same way as a Windows shortcut, though of course they predate the Windows feature by many years. 
>  符号链接通过创建一种特殊的文件来实现，该文件包含了指向所引用文件或目录的文本指针
>  它们的工作方式和 Windows 的快捷方式很相似

A file pointed to by a symbolic link, and the symbolic link itself are largely indistinguishable from one another. For example, if we write something to the symbolic link, the referenced file is written to. However when we delete a symbolic link, only the link is deleted, not the file itself. If the file is deleted before the symbolic link, the link will continue to exist but will point to nothing. In this case, the link is said to be broken. In many implementations, the ls command will display broken links in a distinguishing color, such as red, to reveal their presence. 
>  由符号链接所指向的文件和符号链接本身在很大程度上是无法区分的
>  例如，当我们向符号链接写入内容时，实际写入的是被引用的文件
>  但当我们删除符号链接时，仅会删除链接而不是文件本身
>  如果文件在符号链接被删除之前被删除，则链接会保持存在，但会指向空无一物的内容，这种情况下，我们称链接是 “损坏” 的
>  在许多实现中，`ls` 会以特殊的颜色 (例如红色) 显示损坏的链接

The concept of links can seem confusing, but hang in there. We're going to try all this stuff and it will, hopefully, become clear. 

## Let's Build a Playground 
Since we are going to do some real file manipulation, let's build a safe place to “play” with our file manipulation commands. First we need a directory to work in. We'll create one in our home directory and call it playground. 

### Creating Directories 
The `mkdir` command is used to create a directory. To create our playground directory we will first make sure we are in our home directory and will then create the new directory. 

```
[me@linuxbox ~]$ cd 
[me@linuxbox ~]$ mkdir playground 
```

To make our playground a little more interesting, let's create a couple of directories inside it called dir1 and dir2. To do this, we will change our current working directory to playground and execute another mkdir. 

```
[me@linuxbox ~]$ cd playground 
[me@linuxbox playground]$ mkdir dir1 dir2 
```

Notice that the mkdir command will accept multiple arguments allowing us to create both directories with a single command. 

### Copying Files 
Next, let's get some data into our playground. We'll do this by copying a file. Using the cp command, we'll copy the passwd file from the /etc directory to the current working directory. 

```
[me@linuxbox playground]$ cp /etc/passwd . 
```

Notice how we used shorthand for the current working directory, the single trailing period. So now if we perform an ls, we will see our file. 

```
[me@linuxbox playground]$ ls -l 

total 12 
drwxrwxr-x 2 me me 4096 2025-01-10 16:40 dir1 drwxrwxr-x 2 me me 4096 2025-01-10 16:40 dir2 -rw-r--r-- 1 me me 1650 2025-01-10 16:07 passwd 
```

Now, just for fun, let's repeat the copy using the “-v” option (verbose) to see what it does. 

```
[me@linuxbox playground]$ cp -v /etc/passwd .
`/etc/passwd' -> `./passwd' 
```

The cp command performed the copy again, but this time displayed a concise message indicating what operation it was performing. Notice that cp overwrote the first copy without any warning. Again this is a case of cp assuming that we know what we're doing. To get a warning, we'll include the “-i” (interactive) option. 

```
[me@linuxbox playground]$ cp -i /etc/passwd 
cp: overwrite \`./passwd'? 
```

Responding to the prompt by entering a y will cause the file to be overwritten, any other character (for example, n) will cause cp to leave the file alone. 

### Moving and Renaming Files 
Now, the name passwd doesn't seem very playful and this is a playground, so let's change it to something else. 

<html><body><table><tr><td>[ me@linuxbox playground]$ mv passwd fun</td></tr></table></body></html> 

Let's pass the fun around a little by moving our renamed file to each of the directories and back again. The following moves it first to the directory dir1: 

<html><body><table><tr><td>[ me@linuxbox playground]$ mv fun dir1</td></tr></table></body></html> 

The following then moves it from dir1 to dir2: 

```
[me@linuxbox playground]$ mv dir1/fun dir2 
```

Finally, the following brings it back to the current working directory: 

```
[me@linuxbox playground]$ mv dir2/fun 
```

Next, let's see the effect of mv on directories. First we will move our data file into dir1 again, like this: 

<html><body><table><tr><td>[ me@linuxbox playground]$ mv fun dir1</td></tr></table></body></html> 

Then we move dir1 into dir2 and confirm it with ls. 

```
[me@linuxbox playground]$ mv dir1 dir2 
[me@linuxbox playground]$ ls -l dir2 
total 4 
drwxrwxr-x 2 me me 4096 2025-01-11 06:06 dir1 
[me@linuxbox playground]$ ls -l dir2/dir1 
total 4 
-rw-r--r-- 1 me me 1650 2025-01-10 16:33 fun 
```

Note that since dir2 already existed, mv moved dir1 into dir2. If dir2 had not existed, mv would have renamed dir1 to dir2. Lastly, let's put everything back. 

```
[me@linuxbox playground]$ mv dir2/dir1 . 
[me@linuxbox playground]$ mv dir1/fun . 
```

### Creating Hard Links 
Now we'll try some links. We’ll first create some hard links to our data file like so: 

```
[me@linuxbox playground]$ ln fun fun-hard 
[me@linuxbox playground]$ ln fun dir1/fun-hard 
[me@linuxbox playground]$ ln fun dir2/fun-hard 
```

So now we have four instances of the file fun. Let's take a look at our playground directory. 

<html><body><table><tr><td colspan="6">[me@linuxbox playground]$ ls -l</td></tr><tr><td>total 16</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>drwxrwxr-x 2 me</td><td> me</td><td>4096</td><td>2025-01-14 16:17 dir1 4096 2025-01-14 16:17 dir2</td><td></td><td></td></tr><tr><td>drwxrwxr-x 2</td><td>me me</td><td></td><td>1650 2025-01-10 16:33 fun</td><td></td><td></td></tr><tr><td>-rw-r--r-- 4 -rw-r--r-- </td><td>me me 4 me me</td><td></td><td>1650 2025-01-10 16:33 fun-hard</td><td></td><td></td></tr></table></body></html> 

One thing we notice is that both the second fields in the listings for fun and fun-hard contain a 4 which is the number of hard links that now exist for the file. Remember that a file will always have at least one link because the file's name is created by a link. So, how do we know that fun and fun-hard are, in fact, the same file? In this case, ls is not very helpful. While we can see that fun and fun-hard are both the same size (field 5), our listing provides no way to be sure. To solve this problem, we're going to have to dig a little deeper. 

When thinking about hard links, it is helpful to imagine that files are made up of two parts. 

1. The data part containing the file's contents. 
2. The name part that holds the file's name. 

When we create hard links, we are actually creating additional name parts that all refer to the same data part. The system assigns a chain of disk blocks to what is called an inode, which is then associated with the name part. Each hard link therefore refers to a specific inode containing the file's contents. 

The ls command has a way to reveal this information. It is invoked with the -i option. 

<html><body><table><tr><td colspan="6">[me@linuxbox playground]$ ls -li</td></tr><tr><td>total 16</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>12353539 drwxrwxr-x 2 me</td><td></td><td> me</td><td></td><td>4096 2025-01-14 16:17 dir1</td><td></td></tr><tr><td>12353540 drwxrwxr-x 2</td><td>me</td><td> me</td><td></td><td>4096 2025-01-14 16:17 dir2</td><td></td></tr><tr><td>12353538 -rw-r--r-- 4 me</td><td></td><td> me</td><td></td><td>1650 2025-01-10 16:33 fun</td><td></td></tr><tr><td>12353538 -rw-r--r-- 4 me</td><td></td><td>me</td><td></td><td></td><td>1650 2025-01-10 16:33 fun-hard</td></tr></table></body></html> 

In this version of the listing, the first field is the inode number and, as we can see, both fun and fun-hard share the same inode number, which confirms they are the same file. 

### Creating Symbolic Links 
Symbolic links were created to overcome the two disadvantages of hard links. 
1. Hard links cannot span physical devices. 
2. Hard links cannot reference directories, only files. 
Symbolic links are a special type of file that contains a text pointer to the target file or directory. 
Creating symbolic links is similar to creating hard links. 

```
[me@linuxbox playground]$ ln -s fun fun-sym 
[me@linuxbox playground]$ ln -s ../fun dir1/fun-sym 
[me@linuxbox playground]$ ln -s ../fun dir2/fun-sym 
```

The first example is pretty straightforward; we simply add the $^{66}-5^{33}$ option to create a symbolic link rather than a hard link. But what about the next two? Remember, when we create a symbolic link, we are creating a text description of where the target file is rela - tive to the symbolic link. It's easier to see if we look at the ls output shown here: 

<html><body><table><tr><td colspan="6">[ me@linuxbox playground]$ ls -l dir1</td></tr><tr><td>total 4</td><td></td><td></td><td>1650 2025-01-10 16:33 fun-hard</td><td></td><td></td></tr><tr><td>-rw-r--r-- 4 me lrwxrwxrwx 1 me</td><td>me me</td><td></td><td></td><td></td><td>6 2025-01-15 15:17 fun-sym -> ../fun</td></tr></table></body></html> 
The listing for fun-sym in dir1 shows that it is a symbolic link by the leading l in the first field and that it points to ../fun, which is correct. Relative to the location of funsym, fun is in the directory above it. Notice too, that the length of the symbolic link file is 6, the number of characters in the string ../fun rather than the length of the file to which it is pointing. 

When creating symbolic links, we can either use absolute pathnames, as shown here: 

```
[me@linuxbox playground]$ ln -s /home/me/playground/fun dir1/fun-sym 
```

or relative pathnames, as we did in our earlier example. In most cases, using relative pathnames is more desirable because it allows a directory tree containing symbolic links and their referenced files to be renamed and/or moved without breaking the links. 

In addition to regular files, symbolic links can also reference directories. 

```
[me@linuxbox playground]$ ln -s dir1 dir1-sym 
[me@linuxbox playground]$ ls -l 
```

<html><body><table><tr><td colspan="8">total 16</td></tr><tr><td>drwxrwxr-x</td><td>2 me</td><td>me</td><td></td><td>4096 2025-01-15 15:17 dir1</td><td></td><td></td></tr><tr><td>lrwxrwxrwx</td><td>1 me</td><td>me</td><td></td><td></td><td></td><td>4 2025-01-16 14:45 dir1-sym -> dir1</td></tr><tr><td>drwxrwxr-x</td><td>2 me</td><td>me</td><td></td><td>4096 2025-01-15 15:17</td><td></td><td>dir2</td></tr><tr><td>-rw-r--r--</td><td>4 me</td><td>me</td><td>1650</td><td>2025-01-10 16:33</td><td></td><td>fun</td></tr><tr><td>-rw-r--r--</td><td>4 me</td><td>me</td><td>1650</td><td>2025-01-10 16:33</td><td></td><td> fun-hard</td></tr><tr><td>lrwxrwxrwx</td><td>1 me</td><td>me</td><td></td><td>3 2025-01-15</td><td>15:15</td><td>fun-sym -> fun</td></tr></table></body></html> 

### Removing Files and Directories 
As we covered earlier, the rm command is used to delete files and directories. We are going to use it to clean up our playground a little bit. First, let's delete one of our hard links. 

<html><body><table><tr><td>[ me@linuxbox playground]$ rm fun-hard [ me@linuxbox playground]$ ls -l</td></tr></table></body></html> 

That worked as expected. The file fun-hard is gone and the link count shown for fun is reduced from four to three, as indicated in the second field of the directory listing. Next, we'll delete the file fun, and just for enjoyment, we'll include the -i option to show what that does. 

<html><body><table><tr><td>[ me@linuxbox playground]$ rm -i fun rm: remove regular file “fun'?</td></tr></table></body></html> 

Enter y at the prompt and the file is deleted. But let's look at the output of ls now. Notice what happened to fun-sym? Since it's a symbolic link pointing to a now-nonexistent file, the link is broken. 

<html><body><table><tr><td colspan="5">[me@linuxbox playground]$ ls -l</td></tr><tr><td>total 8 drwxrwxr-x 2 me me</td><td></td><td></td><td>4096 2025-01-15 15:17 dir1</td><td></td></tr><tr><td>lrwxrwxrwx 1 me me</td><td></td><td>4 2025-01-16 14:45</td><td></td><td>dir1-sym -> dir1</td></tr></table></body></html> 
<html><body><table><tr><td>drwxrwxr-x 2 me</td><td></td><td></td><td> me</td><td></td><td>4096 2025-01-15 15:17 dir2</td><td></td><td></td><td></td></tr><tr><td>lrwxrwxrwx 1 me</td><td></td><td></td><td> me </td><td></td><td>3 2025-01-15 15:15</td><td></td><td>fun-sym -> fun</td><td></td></tr></table></body></html> 

Most Linux distributions configure ls to display broken links. The presence of a broken link is not in and of itself dangerous, but it is rather messy. If we try to use a broken link we will see this: 

<html><body><table><tr><td>[me@linuxbox playground]$ less fun-sym</td></tr><tr><td>fun-sym: No such file or directory</td></tr></table></body></html> 

Let's clean up a little. We'll delete the symbolic links here: 

<html><body><table><tr><td>[me@linuxbox playground]$ ls -l</td><td>[me@linuxbox playground]$ rm fun-sym dir1-sym</td><td></td><td></td></tr><tr><td>total 8</td><td></td><td></td><td>4096 2025-01-15 15:17 dir1</td></tr><tr><td>drwxrwxr-x 2 me</td><td> me</td><td></td><td></td></tr><tr><td>drwxrwxr-x 2 me</td><td> me</td><td>4096 2025-01-15 15:17 dir2</td><td></td></tr></table></body></html> 

One thing to remember about symbolic links is that most file operations are carried out on the link's target, not the link itself. rm is an exception. When we delete a link, it is the link that is deleted, not the target. 

Finally, we will remove our playground. To do this, we will return to our home directory and use rm with the recursive option (-r) to delete playground and all of its contents, including its subdirectories. 
`
```
[me@linuxbox playground]$ cd [me@linuxbox ~]$ rm -r playground 
```

### Creating Symlinks With The GUI 
The file managers in both GNOME and KDE provide an easy and automatic method of creating symbolic links. With GNOME, holding the Ctrl+Shift keys while dragging a file will create a link rather than copying (or moving) the file. In KDE, a small menu appears whenever a file is dropped, offering a choice of copying, moving, or linking the file. 

## Summing Up 
We've covered a lot of ground here and it will take a while for it all to fully sink in. Per - form the playground exercise over and over until it makes sense. It is important to get a good understanding of basic file manipulation commands and wildcards. Feel free to expand on the playground exercise by adding more files and directories, using wildcards to specify files for various operations. The concept of links is a little confusing at first, but take the time to learn how they work. They can be a real lifesaver. 

## Further Reading 
A discussion of symbolic links: http://en.wikipedia.org/wiki/Symbolic_link 

# 5 Working with Commands 
Up to this point, we have seen a series of mysterious commands, each with its own mysterious options and arguments. In this chapter, we will attempt to remove some of that mystery and even create our own commands. The commands introduced in this chapter are: 

- `type` – Indicate how a command name is interpreted 
- `which` – Display which executable program will be executed 
- `help` – Get help for shell builtins 
- `man` – Display a command's manual page 
- `apropos` – Display a list of appropriate commands 
- `info` – Display a command's info entry 
- `whatis` – Display one-line manual page descriptions alias – Create an alias for a command 

## What Exactly Are Commands? 
A command can be one of four different things: 
1. An executable program like all those files we saw in /usr/bin. Within this category, programs can be compiled binaries such as programs written in C and C++ , or programs written in scripting languages such as the shell, Perl, Python, Ruby, and so on. 
2. A command built into the shell itself. bash supports a number of commands internally called shell builtins. The cd command, for example, is a shell builtin. 
3. A shell function. Shell functions are miniature shell scripts incorporated into the environment. We will cover configuring the environment and writing shell functions in later chapters, but for now, just be aware that they exist. 
4. An alias. Aliases are commands that we can define ourselves, built from other commands. 

>  一个命令可以是以下四种类型之一
>  1. 可执行程序，就像我们在 `/usr/bin` 中看到的那些文件，“程序“ 可以是编译后得到的二进制文件，也可以是脚本语言编写的程序
>  2. 内建于 shell 自身的命令，shell 内建的命令称为 shell builtins，例如 bash 就内建了许多命令，包括 `cd` 等
>  3. shell 函数，shell 函数是嵌入到环境中的微型 shell 脚本
>  4. 别名，别名是我们自己定义的命令，它由其他命令组合而成

## Identifying Commands 
It is often useful to know exactly which of the four kinds of commands is being used and Linux provides a couple of ways to find out. 

### type – Display a Command's Type 
The type command is a shell builtin that displays the kind of command the shell will execute, given a particular command name. It works like this: 

```
type command 
```

where “command” is the name of the command we want to examine. 

>  `type` 命令是 shell 内建命令，它用于确定命令到底属于上述四种类型的哪一种

Here are some examples: 

```
[me@linuxbox ~]$ type type
type is a shell builtin 
[me@linuxbox ~]$ type ls
ls is aliased to `ls --color=auto' 
[me@linuxbox ~]$ type cp
cp is /usr/bin/cp
```

Here we see the results for three different commands. Notice the one for ls (taken from a Fedora system) and how the ls command is actually an alias for the ls command with the “--color=tty” option added. Now we know why the output from ls is displayed in color! 
>  注意到 `ls` 本身是 `ls --color=tty` 的别名

### which – Display an Executable's Location 
Sometimes there is more than one version of an executable program installed on a system. While this is not common on desktop systems, it's not unusual on large servers. To determine the exact location of a given executable, the which command is used. 

```
[me@linuxbox ~]$ which ls 
/usr/bin/ls 
```

which only works for executable programs, not builtins nor aliases that are substitutes for actual executable programs. 

>  如果我们确定命令是可执行文件，则 `which` 可以用于判断具体的可执行文件的位置，这可以用于判断我们执行的具体是哪一个版本的可执行文件 (因此叫 which)
>  `which` 仅适用于可执行文件类型的命令

When we try to use which on a shell builtin for example, cd, we either get no response or get an error message: 

```
[me@linuxbox ~]$ which cd 
/usr/bin/which: no cd in (/usr/local/bin:/usr/bin:/bin:/usr/local 
/games:/usr/games) 
```

This response is a fancy way of saying “command not found.” 

>  如果对 `which` 传递非可执行文件类型的命令，它将找不到对应的可执行文件

## Getting a Command's Documentation 
With this knowledge of what a command is, we can now search for the documentation available for each kind of command. 

### help – Get Help for Shell Builtins 
bash has a built-in help facility available for each of the shell builtins. To use it, type “help” followed by the name of the shell builtin. Here is an example: 
>  `help` 是 bash 内建命令，它可以用于获取关于各个 bash 内建命令的帮助信息

```
[me@linuxbox ~]$ help cd 
cd: cd [-L|[-P [-e]] [-@]] [dir] 
    Change the shell working directory. 
    
    Change the current directory to DIR. The default DIR is the value of the HOME shell variable. 
    
    The variable CDPATH defines the search path for the directory containing DIR. Alternative directory names in CDPATH are separated by a colon (:). A null directory name is the same as the current directory. If DIR begins with a slash (/), then CDPATH is not used.
    
    If the directory is not found, and the shell option \`cdable_vars' is set, the word is assumed to be a variable name. If that variable has a value, its value is used for DIR. 
    
    Options: 
        -L force symbolic links to be followed: resolve symbolic links in DIR after processing instances of \`. 
        -P use the physical directory structure without following symbolic links: resolve symbolic links in DIR before processing instances of \`. 
        -e if the -P option is supplied, and the current working directory cannot be determined successfully, exit with a non-zero status 
        -@ on systems that support it, present a file with extended attributes as a directory containing the file attributes 
        
    The default is to follow symbolic links, as if \`-L' were specified. \`..' is processed by removing the immediately previous pathname component back to a slash or the beginning of DIR. 
    
    Exit Status: 
    Returns 0 if the directory is changed, and if $PWD is set successfully when -P is used; non-zero otherwise. 
```

A note on notation: When square brackets appear in the description of a command's syntax, they indicate optional items. A vertical bar character indicates mutually exclusive items. 
>  在命令语法的描述中， `[]` 表示其中的内容是可选的，`|` 表示互斥的选项

In the case of the cd command above: 

```
cd [-L|[-P[-e]]] [dir]
```

This notation says that the command cd may be followed optionally by either a “-L” or a “-P” and further, if the “-P” option is specified the “-e” option may also be included followed by the optional argument “dir”. 

While the output of help for the cd commands is concise and accurate, it is by no means tutorial and as we can see, it also seems to mention a lot of things we haven't talked about yet! Don't worry. We'll get there. 

Helpful hint: By using the help command with the -m option, help will display its output in an alternate format. 
>  `help -m ...` 可以以另一种形式展示帮助信息

### --help – Display Usage Information 
Many executable programs support a “--help” option that displays a description of the command's supported syntax and options. For example: 

```
[me@linuxbox ~]$ mkdir --help
Usage: mkdir [OPTION] DIRECTORY...
    Create the DIRECTORY(ies), if they do not already exist.

    -Z, --context=CONTEXT (SELinux) set security context to CONTEXT Mandatory arguments to long options are mandatory for short options too.
    -m, --mode=MODE set file mode (as in chmod), not a=rwx – umask 
    -p, --parents no error if existing, make parent directories as needed
    -v, --verbose print a message for each created directory --help display this help and exit
    --version output version information and exit 
Report bugs to <bug-coreutils@gnu.org>.
```

Some programs don't support the “--help” option, but try it anyway. Often it results in an error message that will reveal the same usage information. 

>  大多数程序都会有 `--help` 选项

### man – Display a Program's Manual Page 
Most executable programs intended for command line use provide a formal piece of documentation called a manual or man page. A special paging program called man is used to view them. It is used like this: 

```
man program 
```

where “program” is the name of the command to view. 

>  许多可执行程序类型的命令都会提供一份正式的文档，称为手册
>  `man` 命令可以用于查看各个可执行程序类命令的文档

Man pages vary somewhat in format but generally contain the following: 
- A title (the page’s name) 
- A synopsis of the command's syntax 
- A description of the command's purpose 
- A listing and description of each of the command's options 

>  命令手册通常包含以下的信息
>  - 标题
>  - 命令语法的概述
>  - 命令用途的描述
>  - 命令各个选项的描述

Man pages, however, do not usually include examples, and are intended as a reference, not a tutorial. As an example, let's try viewing the man page for the ls command: 

```
[me@linuxbox ~]$ man ls 
```

>  手册通常不包含例子，主要用作参考，而不是教程

On most Linux systems, man uses less to display the manual page, so all of the familiar less commands work while displaying the page. 
>  大多数 Linux 系统中，`man` 使用 `less` 来展示手册

The “manual” that man displays is divided into sections and covers not only user commands but also system administration commands, programming interfaces, file formats and more. Table 5-1 describes the layout of the manual. 
>  `man` 展示的手册被划分为了多个章节，它实际上不仅涵盖了用户命令方面的手册，也涵盖了系统管理命令、编程接口、文件格式等的手册

Table 5-1: Man Page Organization 

<html><body><center><table><tr><td>Section</td><td>Contents</td></tr><tr><td>1</td><td>User commands</td></tr><tr><td>2</td><td> Programming interfaces for kernel system calls</td></tr><tr><td>3</td><td>Programming interfaces to the C library</td></tr><tr><td>4</td><td>Special files such as device nodes and drivers</td></tr><tr><td>5</td><td>File formats</td></tr><tr><td>6</td><td>Games and amusements such as screen savers</td></tr><tr><td>7</td><td>Miscellaneous</td></tr><tr><td>8</td><td>System administration commands</td></tr></center></table></body></html> 

>  整个手册的布局如上所示
>  - 第一部分: 用户命令
>  - 第二部分: kernel 系统调用的编程接口
>  - 第三部分: C 库的编程接口
>  - 第四部分: 特殊为念，例如设备节点和驱动
>  - 第五部分: 文件格式
>  - 第六部分: 游戏和娱乐，例如屏幕保护程序
>  - 第七部分: 杂项
>  - 第八部分: 系统管理命令

Sometimes we need to refer to a specific section of the manual to find what we are looking for. This is particularly true if we are looking for a file format that is also the name of a command. Without specifying a section number, we will always get the first instance of a match, probably in section 1. To specify a section number, we use man like this: 

<html><body><table><tr><td>man section search_term</td></tr></table></body></html> 

Here's an example: 

<html><body><table><tr><td>[ me@linuxbox ~]$ man 5 passwd</td></tr></table></body></html> 

This will display the man page describing the file format of the /etc/passwd file. 

>  有时，我们需要到特定的部分才能找到具体的手册
>  例如某个命令和文件格式共享一个名称，如果不具体指定哪个部分，我们一般看到的是第一个匹配的部分 (通常是 section 1)
>  例如 `man 5 passwd` 会展示描述 `/etc/passwd` 文件格式的手册

### `apropos` – Display Appropriate Commands 
It is also possible to search the list of man pages for possible matches based on a search term. It's crude but sometimes helpful. Here is an example of a search for man pages using the search term partition: 
>  `apropos` 用于在手册列表中搜索其内容能够匹配给定关键词的手册

```
[me@linuxbox ~]$ apropos partition

addpart (8) - simple wrapper around the "add partition"... 
all-swaps (7) - event signalling that all swap partitions...
cfdisk (8) - display or manipulate disk partition table
cgdisk (8) - Curses-based GUID partition table (GPT)...
delpart (8) - simple wrapper around the "del partition"...
fdisk (8) - manipulate disk partition table
fixparts (8) - MBR partition table repair utility
gdisk (8) - Interactive GUID partition table (GPT)...
mpartition (1) - partition an MSDOS hard disk
partprobe (8) - inform the OS of partition table changes
partx (8) - tell the Linux kernel about the presence...
resizepart (8) - simple wrapper around the "resize partition...
sfdisk (8) - partition table manipulator for Linux
sgdisk (8) - Command-line GUID partition table (GPT)...
```

The first field in each line of output is the name of the man page, and the second field shows the section. 
>  `apropos` 的输出中，每行的第一个字段是手册名称，第二个字段显示手册属于哪个章节

Note that the man command with the `-k` option performs the same function as `apropos`. 
>  `man -k` 的功能和 `apropos` 是一样的

### `whatis` – Display One-line Manual Page Descriptions 
The `whatis` program displays the name and a one-line description of a man page matching a specified keyword: 
>  `whatis` 显示与指定关键字匹配的手册页的名称和一行描述

```
[me@linuxbox ~]$ whatis ls
ls (1) - list directory
```

>  `apropos` 是根据关键词搜索命令或程序的简短描述，它会搜索 `man` 手册页面的名称和描述部分，返回所有包含指定关键词的命令或程序的描述
>  `apropos` 应该用于不确定命令或程序的具体名称，但知道它的功能关键词的场景下
>  `whatis` 用于显示指定命令或程序的简短描述，它会搜索 `man` 手册页面的名称和描述部分，并返回匹配的结果，如果找到多个匹配项，它会显示所有匹配项的描述
>  `whatis` 应该用于知道命令或程序的名称，但想快速确认它的功能时的场景

**The Most Brutal Man Page Of Them All** 
As we have seen, the manual pages supplied with Linux and other Unix-like systems are intended as reference documentation and not as tutorials. Many man pages are hard to read, but I think that the grand prize for difficulty has got to go to the man page for bash. As I was doing research for this book, I gave the bash man page careful review to ensure that I was covering most of its topics. When printed, it's more than 80 pages long and extremely dense, and its structure makes absolutely no sense to a new user. 

On the other hand, it is very accurate and concise, as well as being extremely complete. So check it out if you dare and look forward to the day when you can read it and it all makes sense. 

### `info` – Display a Program's Info Entry 
The GNU Project provides an alternative to man pages for their programs, called “info.” Info manuals are displayed with a reader program named, appropriately enough, info. Info pages are hyperlinked much like web pages. Here is a sample: 
>  `info` 命令由于显示程序的 Info 文档 (大多数命令会有 manual 文档，但不一定有 Info 文档，通常一些 GNU 工具和程序额外有 Info 文档)
>  manual 文档通常提供关于命令的简明参考信息，适合快速查找命令的用法和选项；Info 文档通常提供更详细的文档，包括背景信息、示例和更深入的解释，适合更全面的学习
>  `info` 命令会用 `info` 阅读程序展示 Info 文档，Info 文档以信息节点的形式组织，并且其中允许有超链接

```
File: coreutils.info, Node: ls invocation, Next: dir invocation, Up: Directory listing
10.1 `ls': List directory contents 
==================================
The `ls' program lists information about files (of any type, including directories). Options and file arguments can be intermixed arbitrarily, as usual.
    For non-option command-line arguments that are directories, by default `ls' lists the contents of directories, not recursively, and omitting files with names beginning with `.'. For other non-option arguments, by default `ls' lists just the filename. If no non-option argument is specified, `ls' operates on the current directory, acting as if it had been invoked with a single argument of `.'.
    By default, the output is sorted alphabetically, according to the
--zz-Info: (coreutils.info.gz)ls invocation, 63 lines --Top----------
```

The `info` program reads info files, which are tree structured into individual nodes, each containing a single topic. Info files contain hyperlinks that can move the reader from node to node. A hyperlink can be identified by its leading asterisk and is activated by placing the cursor upon it and pressing the Enter key. 
>  `info` 命令读取 Info 文件，这些文件是一个个节点，组织为一个树状结构，每个节点包含一个主体
>  Info 文件包含超链接，可以从一个节点跳转到另一个节点，超链接会有一个前导的 `*` ，将光标放在超链接上，按下 `Enter` 就可以跳转

To invoke info, type info followed optionally by the name of a program. Table 5-2 describes the commands used to control the reader while displaying an info page. 

Table 5-2: info Commands 
<html><body><center><table><tr><td>Command</td><td>Action</td></tr><tr><td>？</td><td>Display command help</td></tr><tr><td>PgUp or Backspace</td><td>Display previous page</td></tr><tr><td>PgDn or Space</td><td>Display next page</td></tr><tr><td>n</td><td>Next - Display the next node</td></tr><tr><td>p</td><td>Previous - Display the previous node</td></tr><tr><td>u</td><td>Up - Display the parent node of the currently displayed node, usually a menu</td></tr>
<tr><td>Enter</td><td>Follow the hyperlink at the cursor location</td></tr><tr><td>q</td><td>Quit</td></tr></table></center></body></html> 

>  `info` 阅读器中的常见快捷命令
>  `?` : 显示 help
>  `PgUp/Backspace`: 显示上一个 page
>  `PgDn/Space` : 显示下一个 page
>  `n`: 显示下一个节点
>  `p`: 显示上一个节点
>  `u` : 显示当前节点的父节点，通常是菜单
>  `Enter`: 跳转到光标处的超链接
>  `q` : 退出

Most of the command line programs we have discussed so far are part of the GNU Project's `coreutils` package, so typing the following: 

<html><body><table><tr><td>[ me@linuxbox ~]$ info coreutils</td></tr></table></body></html> 

will display a menu page with hyperlinks to each program contained in the `coreutils` package. 

>  目前为止讨论的大多数命令都是 GNU 的 `coreutils` 包的一部分
>  故 `info coreutils` 会显示一个菜单页面，它包含了许多超链接，指向了 `coreutils` 中的各个程序的 Info 节点

### README and Other Program Documentation Files 
Many software packages installed on our system have documentation files residing in the `/usr/share/doc` directory. Most of these are stored in plain text format and can be viewed with less. Some of the files are in HTML format and can be viewed with a web browser. We may encounter some files ending with a “`.gz`” extension. This indicates that they have been compressed with the `gzip` compression program. The `gzip` package includes a special version of less called `zless` that will display the contents of `gzip` compressed text files. 
>  大多数安装在系统中的软件在 `/usr/share/doc` 中有文档文件，大多数文档文件都以纯文本形式存储，部分以 HTML 格式存储
>  一些以 `.gz` 后缀的文件表示它们是通过 `gzip` 程序压缩的，`gzip` 包包含了 `zless` ，它可以展示 `gzip` 压缩的文本文件

## Creating Our Own Commands with alias 
Now for our first experience with programming! We will create a command of our own using the alias command. But before we start, we need to reveal a small command line trick. It's possible to put more than one command on a line by separating each command with a semicolon. It works like this: 

```
command1; command2; command3... 
```

>  需要在同一行输入多个命令时，可以用 `;` 号隔开

Here's the example we will use: 

```
[me@linuxbox ~]$ cd /usr; ls; cd - 
bin games include lib local sbin share src 
/home/me 
[me@linuxbox ~]$ 
```

As we can see, we have combined three commands on one line. First we change directory to `/usr` then list the directory and finally return to the original directory (by using 'cd -') so we end up where we started. Now let's turn this sequence into a new command using alias. The first thing we have to do is dream up a name for our new command. Let's try “test”. Before we do that, it would be a good idea to find out if the name “test” is already being used. To find out, we can use the type command again: 

```
[me@linuxbox ~]$ type test 
test is a shell builtin
```

Oops! The name test is already taken. Let's try foo: 

```
[me@linuxbox ~]$ type foo 
bash: type: foo: not found 
```

Great! “foo” is not taken. So let's create our alias: 

```bash
[me@linuxbox ~]$ alias foo='cd /usr; ls; cd -'
```

Notice the structure of this command shown here: 

```bash
alias name='string'
```

>  `alias` 命令可以用于创建命令别名
>  通过 `alias foo='cd /usr; ls; cd-'`，我们创建了别名 `foo` 以指代对应的命令序列
>  `alias` 的语法为 `alias name='string'`，注意 `=` 两边没有空格

After the command alias, we give alias a name followed immediately (no whitespace allowed) by an equal sign, followed immediately by a quoted string containing the meaning to be assigned to the name. After we define our alias, we can use it anywhere the shell would expect a command. Let's try it: 

```
[me@linuxbox ~]$ foo 
bin games include lib local sbin share src 
/home/me 
[me@linuxbox ~]$ 
```

We can also use the type command again to see our alias: 

```
[me@linuxbox ~]$ type foo 
foo is aliased to `cd /usr; ls; cd -'
```

To remove an alias, the unalias command is used, like so: 

```
[me@linuxbox ~]$ unalias foo 
[me@linuxbox ~]$ type foo 
bash: type: foo: not found
```

>  `unalias` 用于移除别名

While we purposefully avoided naming our alias with an existing command name, it is not uncommon to do so. This is often done to apply a commonly desired option to each invocation of a common command. For instance, we saw earlier how the ls command is often aliased to add color support: 

```
[me@linuxbox ~]$ type ls
ls is aliased to `ls --color=tty'
```

To see all the aliases defined in the environment, use the alias command without arguments. Here are some of the aliases defined by default on a Fedora system. Try to figure out what they all do: 

```
[me@linuxbox ~]$ alias
alias l.='ls -d .* --color=tty' 
alias ll='ls -l --color=tty' 
alias ls='ls --color=tty'
```

>  `alias` 命令不带参数可以列出环境中所有的表明

There is one tiny problem with defining aliases on the command line. They vanish when our shell session ends. In Chapter 11, "The Environment", we will see how to add our own aliases to the files that establish the environment each time we log on, but for now, enjoy the fact that we have taken our first, albeit tiny, step into the world of shell programming! 

## Summing Up 
Now that we have learned how to find the documentation for commands, go and look up the documentation for all the commands we have encountered so far. Study what additional options are available and try them! 

## Further Reading 
There are many online sources of documentation for Linux and the command line. Here are some of the best: 

- The Bash Reference Manual is a reference guide to the bash shell. It’s still a reference work but contains examples and is easier to read than the bash man page. http://www.gnu.org/software/bash/manual/bashref.html 
- The Bash FAQ contains answers to frequently asked questions regarding bash. This list is aimed at intermediate to advanced users, but contains a lot of good information.  http://mywiki.wooledge.org/BashFAQ 
- The GNU Project provides extensive documentation for its programs, which form the core of the Linux command line experience. You can see a complete list here: http://www.gnu.org/manual/manual.html 
- Wikipedia has an interesting article on man pages:  http://en.wikipedia.org/wiki/Man_page 

# Part 3 – Common Tasks and Essential Tools 
# 14 Package Management 
If we spend any time in the Linux community, we hear many opinions as to which of the many Linux distributions is “best.” Often, these discussions get really silly, focusing on such things as the prettiness of the desktop background (some people won't use Ubuntu because of its default color scheme!) and other trivial matters. 

The most important determinant of distribution quality is the packaging system and the vitality of the distribution's support community. As we spend more time with Linux, we see that its software landscape is extremely dynamic. Things are constantly changing. Most of the top-tier Linux distributions release new versions every six months and many individual program updates every day. To keep up with this blizzard of software, we need good tools for package management. 
>  Linux 发行版质量的决定因素是其包管理系统和发行版的支持社区的活力
>  Linux 的软件环境非常动态化，大多数顶级 Linux 发行版每 6 个月发布一个新版本，许多独立程序每天都会更新
> 为此，我们需要一个良好的包管理工具

Package management is a method of installing and maintaining software on the system. Today, most people can satisfy all of their software needs by installing packages from their Linux distributor. This contrasts with the early days of Linux, when one had to download and compile source code to install software. There isn’t anything wrong with compiling source code; in fact, having access to source code is the great wonder of Linux. It gives us (and everybody else) the ability to examine and improve the system. It's just that having a precompiled package is faster and easier to deal with. 
>  包管理是在系统上安装和维护软件的方法
>  如今，大多数软件可以直接从 Linux 发行商处直接安装预编译好的软件包，在早期，则需要下载并编译源代码

In this chapter, we will look at some of the command line tools used for package management. While all the major distributions provide powerful and sophisticated graphical programs for maintaining the system, it is important to learn about the command line programs, too. They can perform many tasks that are difficult (or impossible) to do with their graphical counterparts. 

## Packaging Systems 
Different distributions use different packaging systems, and as a general rule, a package intended for one distribution is not compatible with another distribution. Most distributions fall into one of two camps of packaging technologies: the Debian .deb camp and the Red Hat .rpm camp. 
>  不同的发行版使用不同的包管理系统
>  一般来说，为一个发行版设计的包与其他发行版是不兼容的
>  大多数发行版的打包技术都属于以下两类之一: 使用Debian 风格的 `.deb` 包，或 Red Hat 风格的 `.rpm` 包

There are some important exceptions such as Gentoo, Slackware, and Arch, but most others use one of these two basic systems as shown in Table 14-1. 
>  也有一些例外情况，例如 Gentoo, Slackware, Arch

Table 14-1: Major Packaging System Families 

| Packaging System       | Distributions (Partial Listing)                    |
| ---------------------- | -------------------------------------------------- |
| Debian Style (`.deb`)  | Debian, Ubuntu, Linux Mint, Raspberry Pi OS        |
| Red Hat Style (`.rpm`) | Fedora, CentOS, Red Hat Enterprise Linux, OpenSUSE |

>  使用 `.deb` 的发行版主要为 Debian, Ubuntu
>  使用 `.rpm` 的发行版主要为 Fedora, CentOS, OpenSUSE

## How a Package System Works 
The method of software distribution found in the proprietary software industry usually entails buying a piece of installation media such as an “install disk” or visiting a vendor's web site and downloading a product and then running an “installation wizard” to install a new application on the system. 
>  专有软件的常见分发方法是购买安装介质，例如安装光盘，或者访问供应商的网站下载产品，然后运行 “安装向导” 以在系统上安装应用

Linux doesn't work that way. Virtually all software for a Linux system will be found on the Internet. Most of it will be provided by the distribution vendor in the form of package files, and the rest will be available in source code form that can be installed manually. We'll talk about how to install software by compiling source code in chapter 23, “Compiling Programs.” 
>  几乎所有 Linux 系统的软件都可以在网络上找到，大多数会以软件包文件的形式由发行版供应商提供，其余的则是源代码形式，需要手动安装

### Package Files 
The basic unit of software in a packaging system is the package file. A package file is a compressed collection of files that comprise the software package. A package may consist of numerous programs and data files that support the programs. In addition to the files to be installed, the package file also includes metadata about the package, such as a text description of the package and its contents. Additionally, many packages contain pre- and post-installation scripts that perform configuration tasks before and after the package installation. 
>  packaging system 中的软件的基本单元是包文件，包文件是一个压缩的文件集合，包含了构成软件本身的文件
>  一个包可以包含许多程序和支持这些程序的数据文件，除了要安装的文件外，包文件还包含了关于包的元数据，例如包的文本描述
>  此外，许多包还包含预安装和后安装脚本，用于在软件包安装之前和之后执行配置任务

Package files are created by a person known as a package maintainer, often (but not always) an employee of the distribution vendor. The package maintainer gets the software in source code form from the upstream provider (the author of the program), compiles it, and creates the package metadata and any necessary installation scripts. Often, the package maintainer will apply modifications to the original source code to improve the program's integration with the other parts of the Linux distribution. 
>  包文件由包维护者创建，通常是发行版供应商的员工
>  包维护者获取源代码形式的软件，编译它，并创建包元数据和任何必要的安装脚本
>  包维护者通常会对源代码进行修改，以优化该程序和 Linux 发行版的其他部分的集成

### Repositories 
While some software projects choose to perform their own packaging and distribution, most packages today are created by the distribution vendors and interested third parties. Packages are made available to the users of a distribution in central repositories that may contain many thousands of packages, each specially built and maintained for the distribution. 
>  大多数软件包都是由发行版供应商创建的，一个发行版可用的包会被放置在中心存储库中，通常该存储库专门为发行版构建和维护

A distribution may maintain several different repositories for different stages of the software development life cycle. For example, there will usually be a “testing” repository that contains packages that have just been built and are intended for use by brave souls who are looking for bugs before the packages are released for general distribution. A distribution will often have a “development” repository where work-in-progress packages destined for inclusion in the distribution's next major release are kept. 
>  一个发行版可能会为软件开发周期的不同阶段维护不同的仓库
>  例如，通常会有一个 “testing” 仓库，存放刚刚构建好的，未稳定的软件包；通常会有一个 “development” 仓库，存放即将包含发行版的下一个主要版本中的正在开发中的软件包

A distribution may also have related third-party repositories. These are often needed to supply software that, for legal reasons such as patents or DRM anti-circumvention issues, cannot be included with the distribution. Perhaps the best known case is that of encrypted DVD support, which is not legal in the United States. The third-party repositories operate in countries where software patents and anti-circumvention laws do not apply. These repositories are usually wholly independent of the distribution they support, and to use them, one must know about them and manually include them in the configuration files for the package management system. 
>  一个发行版可能还会有相关的第三方仓库，这些仓库通常用于提供由于法律原因，例如专利或数字版权反规避问题无法在发行版中包含的软件
>  最著名的例子是加密 DVD 支持，这在 US 是违法的
>  第三方仓库通常在软件专利和反规避法都不适用的国家运营，这些仓库通常完全独立于它们所支持的发行版，要使用第三方仓库，用户需要手动将它们添加到包管理系统的配置文件中

### Dependencies 
Programs are seldom “standalone”; rather they rely on the presence of other software components to get their work done. Common activities, such as input/output for example, are handled by routines shared by many programs. These routines are stored in what are called shared libraries, which provide essential services to more than one program. 
>  程序很少是 “独立" 的，它们依赖于其他软件组件来完成工作
>  例如，常见的功能如输入输出会由许多程序共享的例程处理，这些例程存储在共享库中，共享库的作用就是为多个程序提供基本服务

If a package requires a shared resource such as a shared library, it is said to have a dependency. Modern package management systems all provide some method of dependency resolution to ensure that when a package is installed, all of its dependencies are installed, too. 
>  如果一个软件包需要类似共享库这样的共享资源，我们称它存在依赖
>  现代的包管理系统都提供了某种形式的依赖解析方法，以确保当安装某个软件包时，其所有的依赖也会被一并安装

### High and Low-level Package Tools 
Package management systems usually consist of two types of tools. 

- Low-level tools which handle tasks such as installing and removing package files 
- High-level tools that perform metadata searching and dependency resolution 

>  包管理系统通常包含两类工具
>  - 低级工具: 处理例如安装和移除包文件的任务
>  - 高级工具: 执行元数据搜索和依赖解析

In this chapter, we will look at the tools supplied with Debian-style systems (such as Ubuntu and many others) and those used by Red Hat products. While all Red Hat-style distributions rely on the same low-level program (`rpm`), they use different high-level tools. For our discussion, we will cover the high-level program ` dnf `, used by Red Hat Enterprise Linux, CentOS, and Fedora. Other Red Hat-style distributions provide high-level tools with comparable features (see Table 14-2). 
>  Red Hat 风格的发行版依赖于相同的低级工具 (`rpm`)，而使用不同的高级工具，包括了 `dnf` 和 `yum`

Table 14- 2: Packaging System Tools 

|              Distributions               | Low-Level Tools |     High-Level Tools     |
| :--------------------------------------: | :-------------: | :----------------------: |
|               Debian style               |     `dpkg`      | `apt, apt-get, aptitude` |
| Fedora, Red Hat Enterprise Linux, CentOS |      `rpm`      |        `dnf, yum`        |

>  Debian 风格的发行版的低级工具是 `dpkg` ，高级工具是 `apt, apt-get, aptitude`
>  Red Hat 风格的发行版的低级工具是 `rpm` ，高级工具是 `dnf, yum`

## Common Package Management Tasks 
Many operations can be performed with the command line package management tools. We will look at the most common. Be aware that the low-level tools also support the creation of package files, an activity outside the scope of this book. 
>  虽然包文件的管理任务主要通过高级工具完成，但低级工具实际上也支持了包的创建

In the discussion below, the term `package_name` refers to the actual name of a package rather than the term `package_file`, which is the name of the file that contains the package. Also, before any package operations can be performed, the package repository needs to be queried so that the local copy of its database can be synchronized. Red Hat’s `dnf` program does this automatically and updates the local database if too much time has elapsed since the last update. On the other hand, Debian’s apt program must be run with the update command to explicitly update the local database. This needs to be done every so often. In the examples below, the apt update command is done before any operations, but in real life this only needs to be done every few hours to stay safe. 
>  我们用 `package_name` 表示软件包的实际名称，`package_file` 表示包含该软件包的文件的名称
>  在执行任何包操作之前，都需要查询包存储库，以同步本地的数据库副本，Red Hat 的 `dnf` 程序会自动在距离上次更新时间较长的情况下执行此操作
>  Debian 的 `apt` 程序则必须通过运行 `update` 命令显式更新本地数据库 (`apt update`)，这需要经常执行

Since operations that involve installing or removing software on a system-wise basis is an administrative task, superuser privileges are required regardless of the package management tool. 
>  因为在系统级别上涉及了安装或移除软件的操作属于管理任务，故无论使用哪个包管理器，此时都需要管理员权限

### Finding a Package in a Repository 
Using the high-level tools to search repository metadata, a package can be located based on its name or description (see Table 14-3). 

Table 14-3: Package Search Commands 

|  Style   |               Command(s)               |
| :------: | :------------------------------------: |
|  Debian  | `apt update; apt search search_string` |
| Red  Hat |       `dnf search search_string`       |

For example, to search a `dnf` repository for the emacs text editor, we can use this command: 

```
dnf search emacs 
```

>  高级工具可以用于搜索仓库元数据，我们可以通过包的名称搜索包
>  相关命令为 `apt search; dnf search`

### Installing a Package from a Repository 
High-level tools permit a package to be downloaded from a repository and installed with full dependency resolution (see Table 14-4). 

Table 14-4: Package Installation Commands 

|  Style   |               Command(s)               |
| :------: | :------------------------------------: |
|  Debian  | `apt update; apt install package_name` |
| Red  Hat |       `dnf intsall package_name`       |

For example, to install the emacs text editor from an apt repository on a Debian system, we can use this command: 

```
apt update; apt install emacs
```

>  高级工具可以用于安装包，同时自动解析其依赖并安装其依赖
>  相关命令为 `apt install; dnf install`

### Installing a Package from a Package File 
If a package file has been downloaded from a source other than a repository, it can be installed directly (though without dependency resolution) using a low-level tool (see Table 14-5). 

Table 14-5: Low-Level Package Installation Commands 

|  Style   |       Command(s)       |
| :------: | :--------------------: |
|  Debian  | `dpkg -i package_file` |
| Red  Hat | `rpm -i package_file`  |

>  如果我们从其他来源下载了包的文件，则可以用低级工具直接安装它 (但是没有依赖解析)
>  相关的命令为 `dpkg -i/rpm -i package_file`

For example, if the `emacs-22.1-7.fc7-i386.rpm` package file had been downloaded from a non-repository site, it would be installed this way: 

```
rpm -i emacs-22.1-7.fc7-i386.rpm 
```

Note: Because this technique uses the low-level rpm program to perform the installation, no dependency resolution is performed. If rpm discovers a missing dependency, rpm will exit with an error. 

>  注意，直接使用低级工具安装包时是没有依赖解析的，如果低级工具发现依赖缺失，会直接退出并报错

### Removing a Package 
Packages can be uninstalled using either the high-level or low-level tools. The high-level tools are shown in Table 14-6. 

Table 14-6: Package Removal Commands 

|  Style   |        Command(s)         |
| :------: | :-----------------------: |
|  Debian  | `apt remove package_name` |
| Red  Hat | `dnf erase package_name`  |

For example, to uninstall the emacs package from a Debian-style system, we can use this command: 

```
apt remove emacs
```

>  可以使用高级工具卸载包
>  相关命令为 `apt remove; dnf erase`

### Updating Packages from a Repository 
The most common package management task is keeping the system up-to-date with the latest versions of packages. The high-level tools can perform this vital task in a single step (see Table 14-7). 

Table 14-7: Package Update Commands 

|  Style   |        Command(s)         |
| :------: | :-----------------------: |
|  Debian  | `apt update; apt upgrade` |
| Red  Hat |       `dnf update`        |

For example, to apply all available updates to the installed packages on a Debian-style system, we can use this command: 

```
apt update; apt upgrade 
```

>  高级工具可以用于更新已经安装的包
>  相关命令为 `apt upgrade; dnf update`

### Upgrading a Package from a Package File 
If an updated version of a package has been downloaded from a non-repository source, it can be installed, replacing the previous version (see Table 14-8). 

Table 14-8: Low-Level Package Upgrade Commands 

|  Style   |       Command(s)       |
| :------: | :--------------------: |
|  Debian  | `dpkg -i package_name` |
| Red  Hat | `rpm -U package_name`  |

For example, to update an existing installation of emacs to the version contained in the package file `emacs-22.1-7.fc7-i386.rpm` on a Red Hat system, we can use this command: 

```
rpm -U emacs-22.1-7.fc7-i386.rpm 
```

Note: `dpkg` does not have a specific option for upgrading a package versus installing one as rpm does. 

>  如果是从其他渠道下载了更新的软件包，则可以使用低级工具进行升级
>  相关命令为 `dpkg -i; rpm -U` (其中 `dpkg` 的 `-i` 选项是和安装包时用的 `-i` 一样的，而 `rpm` 则安装包时用 `-i` ，更新包时用 `-U`)

### Listing Installed Packages 
Table 14-9 lists the commands we can use to display a list of all the packages installed on the system. 

Table 14-9: Package Listing Commands 

|  Style   | Command(s) |
| :------: | :--------: |
|  Debian  | `dpkg -l`  |
| Red  Hat | `rpm -qa`  |

>  低级工具可以用于列出系统中安装的所有包
>  相关命令为 `dpkg -l; rpm -qa`

### Determining Whether a Package is Installed 
Table 14-10 list the low-level tools we can use to display whether a specified package is installed. 

Table 14-10: Package Status Commands 

|  Style   |      Command (s)       |
| :------: | :--------------------: |
|  Debian  | `dpkg -s package_name` |
| Red  Hat | `rpm -q package_name`  |

For example, to determine whether the emacs package is installed on a Debian style system, we can use this command: 

```
dpkg --status emacs
```

>  低级工具可以用于查询系统是否安装了特定包
>  相关命令为 `dpkg -s; rpm -q`

### Displaying Information About an Installed Package 
If the name of an installed package is known, we can use the commands in Table 14-11 to display a description of the package. 

Table 14-11: Package Information Commands 

|  Style   |       Command (s)       |
| :------: | :---------------------: |
|  Debian  | `apt show package_name` |
| Red  Hat | `dnf info package_name` |

For example, to see a description of the emacs package on a Debian-style system, we can use this command: 

```
apt-cache show emacs
```

>  高级工具可以用于展示包的详细信息
>  相关命令为 `apt show; dnf info`

### Finding Which Package Installed a File 
To determine what package is responsible for the installation of a particular file, we can use the commands in Table 14-12. 

Table 14-12: Package File Identification Commands 

|  Style   |     Command (s)     |
| :------: | :-----------------: |
|  Debian  | `dpkg -S file_name` |
| Red  Hat | `rpm -qf file_name` |

For example, to see what package installed the `/usr/bin/vim` file on a Red Hat system, we can use the following: 

```
rpm -qf /usr/bin/vim 
```

>  低级工具可以用于确定文件具体是由哪个包安装的
>  相关命令为 `dpkg -S; rpm -qf`


**Distribution-Independent Package Formats** 
Over the last several years distribution vendors have come out with universal package formats that are not tied to a particular Linux distribution. These include Snaps (developed and promoted by Canonical), Flatpaks (pioneered by Red Hat, but now widely available) and AppImages. Though they each work a little differently, their goal is to have an application and all of its dependencies bundled together and installed in a single piece. This is not an entirely new idea. In the early days of Linux (think the late 1990s) the was a technique called static linking which combined an application and its required libraries into a single large binary. 
>  发行版厂商提出了许多和特定发行版无关的统一包格式，包括了 Snaps, Flatpaks, Applmages 
>  包虽然格式不同，目的都是绑定应用和其依赖，作为一个整体进行安装
>  这并不是一个新的思想，类似的思想是静态链接，将应用和其所需的库组合成单个大的二进制文件

There are some benefits to this packaging approach. First among them is reducing the effort needed to distribute an application. Rather than tailoring the application to work with the libraries and other support files included a distribution's base system, the application is built once and can be installed on any system. Some of these formats also run the application in a containerized sandbox to provide additional security. 
>  这样的打包方法存在好处，首要的好处是减少了分发应用的麻烦，与其调整应用以适应不同发行版的基础系统中所包含的库和支持文件，不如仅构建一次应用，然后可以在任意系统上安装
>  一些打包格式可以在容器化的沙盒环境中运行应用，带来额外的安全性

But there are some serious downsides too. Applications packaged this way are large. Sometimes really large. This has two effects. First, they require a lot of disk space to store. Second, their large size can make them very slow to load. This may not be much of an issue on modern ultra-fast hardware, but on older machines it’s a real problem. The next technical problem has to do with distribution integration. Since these applications bring all of their stuff with them, they don’t take advantage of the underlying distribution's facilities. Sometimes the containerized application cannot access system resources needed of optimal performance. 
>  这样的打包方法也存在缺点，这样打包的应用会非常庞大，故需要占用更大的磁盘空间来存储，且其加载速度可能很慢
>  在现代高性能硬件上，这可能不是问题，但在较旧的设备上就不一定
>  此外，这样的应用由于自带了所有必要的组件，难以和发行版集成，无法利用发行版的底层实用库
>  有时，容器化的应用无法访问达到最佳性能所需的系统资源

Then there are the philosophical issues. Perhaps the biggest beneficiary of these all-in-one application packages are proprietary software vendors. They can build a Linux version once and every distribution can use it. No need to custom tailor their application for different distros. 
>  这样的打包方式的最大受益者可能是软件供应商，它们仅需要构建一次 Linux 版本，每个发行版就都可以使用软件，无需为不同的发行版定制应用

Users were not crying out for these packaging formats and they do little to enhance the open source community, thus until such time the various performance issues are resolved use of these formats is not recommended. 

## Summing Up 
In the chapters that follow, we will explore many different programs covering a wide range of application areas. While most of these programs are commonly installed by default, we may need to install additional packages if the necessary programs are not provided. With our newfound knowledge (and appreciation) of package management, we should have no problem installing and managing the programs we need. 

**The Linux Software Installation Myth** 
People migrating from other platforms sometimes fall victim to the myth that software is somehow difficult to install under Linux and that the variety of packaging schemes used by different distributions is a hindrance. Well, it is a hindrance, but only to proprietary software vendors that want to distribute binary-only versions of their secret software. 

The Linux software ecosystem is based on the idea of open source code. If a program developer releases source code for a program, it is likely that a person associated with a distribution will package the program and include it in their repository. This method ensures that the program is well integrated into the distribution, and the user is given the convenience of “one-stop shopping” for software, rather than having to search for each program's website. Recently, major proprietary platform vendors have begun building application stores that mimic this idea. 
>  通常，如果软件供应者愿意开源，则他不需要担心应用的分发问题，发行版供应商会自行调节软件以和发行版系统集成

Device drivers are handled in much the same way, except that instead of being separate items in a distribution's repository, they become part of the Linux kernel. Generally speaking, there is no such thing as a “driver disk” in Linux. Either the kernel supports a device or it doesn't, and the Linux kernel supports a lot of devices. Many more, in fact, than Windows does. 
>  设备驱动程序的处理方式和软件打包是大致相同的，只不过它们不是作为发行版软件仓库中的独立项目存在，而是成为 Linux 内核的一部分 (也就是安装 Linux 系统时，大部分设备驱动程序都已经作为内核的一部分被安装了)
>  一般来说，Linux 中没有 “驱动盘” 的概念 (即依赖于物理的驱动磁盘来安装驱动)，要么 Linux kernel 支持某个设备，要么不支持
>  事实上，Linux kernel 支持许多设备，比 Windows 支持的设备还要多得多

Of course, this is of no consolation if the particular device you need is not supported. When that happens, you need to look at the cause. A lack of driver support is usually caused by one of three things: 
>  当然，有时也会出现特定的设备不受支持
>  缺乏驱动支持通常由以下三种原因造成:

1. The device is too new. Since many hardware vendors don't actively support Linux development, it falls upon a member of the Linux community to write the kernel driver code. This takes time. 
2. The device is too exotic. Not all distributions include every possible device driver. Each distribution builds its own kernels, and since kernels are very configurable (which is what makes it possible to run Linux on everything from wristwatches to mainframes) they may have overlooked a particular device. By locating and downloading the source code for the driver, it is possible for you (yes, you) to compile and install the driver yourself. This process is not overly difficult, but it is rather involved. We'll talk about compiling software in a later chapter. 
3. The hardware vendor is hiding something. It has neither released source code for a Linux driver, nor has it released the technical documentation for somebody to create one for them. This means the hardware vendor is trying to keep the programming interfaces to the device a secret. Since we don't want secret devices in our computers, it is best that you avoid such products. 

>  1. 设备太新，因为许多硬件厂商并不积极支持 Linux 开发，故为设备编写 Linux 内核驱动代码的任务通常是 Linux 社区成员，这需要时间
>  2. 设备太特殊，不是所有的发行版都会包含所有可能的设备驱动，每个发行版都会构建自己的内核，而由于 Linux 内核的可配置性很高 (这使得 Linux 可以在从手表到大型机的各种设备上运行)，发行版的内核可能会忽略特定的设备。我们可以通过找到并下载对应驱动的源代码，并自己编译和安装驱动程序
>  3. 硬件厂商有所隐藏，它们既没有发布 Linux 驱动程序的源代码，也没有发布技术文档供他人为其创建驱动程序，这意味着硬件厂商试图将设备的编程接口保密，由于我们不希望在计算机中使用秘密设备，最好避免购买此类产品

## Further Reading 
Spend some time getting to know the package management system for your distribution. Each distribution provides documentation for its package management tools. In addition, here are some more generic sources: 

- The Debian GNU/Linux FAQ chapter on package management provides an overview of package management on Debian systems :  https://www.debian.org/doc/manuals/debian-faq/pkg-basics.en.html 
- The home page for the RPM project:  http://www.rpm.org 
- For a little background, the Wikipedia has an article on metadata:  http://en.wikipedia.org/wiki/Metadata 
- A good article comparing Snap, Flatpak, and AppImage formats: https://www.baeldung.com/linux/snaps-flatpak-appimage 

# Part 4 – Writing Shell Scripts 
# 24 – Writing Your First Script 
In the preceding chapters, we have assembled an arsenal of command line tools. While these tools can solve many kinds of computing problems, we are still limited to manually using them one by one on the command line. Wouldn’t it be great if we could get the shell to do more of the work? We can. By joining our tools together into programs of our own design, the shell can carry out complex sequences of tasks all by itself. We can enable it to do this by writing shell scripts. 

## What are Shell Scripts? 
In the simplest terms, a shell script is a file containing a series of commands. The shell reads this file and carries out the commands as though they have been entered directly on the command line. 

The shell is somewhat unique, in that it is both a powerful command line interface to the system and a scripting language interpreter. As we will see, most of the things that can be done on the command line can be done in scripts, and most of the things that can be done in scripts can be done on the command line. 

We have covered many shell features, but we have focused on those features most often used directly on the command line. The shell also provides a set of features usually (but not always) used when writing programs. 

## How to Write a Shell Script 
To successfully create and run a shell script, we need to do three things. 
1. Write a script. Shell scripts are ordinary text files. So, we need a text editor to write them. The best text editors will provide syntax highlighting, allowing us to see a color-coded view of the elements of the script. Syntax highlighting will help us spot certain kinds of common errors. vim, gedit, kate, and many other editors are good candidates for writing scripts. 
2. Make the script executable. The system is rather fussy about not letting any old text file be treated as a program, and for good reason! We need to set the script file’s permissions to allow execution. 
3. Put the script somewhere the shell can find it. The shell automatically searches certain directories for executable files when no explicit pathname is specified. For maximum convenience, we will place our scripts in these directories. 

## Script File Format 
In keeping with programming tradition, we’ll create a “Hello World” program to demonstrate an extremely simple script. Let’s fire up our text editors and enter the following script: 
#!/bin/bash # This is our first script. echo 'Hello World!' 
The last line of our script is pretty familiar; it’s just an echo command with a string argument. The second line is also familiar. It looks like a comment that we have seen used in many of the configuration files we have examined and edited. One thing about comments in shell scripts is that they may also appear at the ends of lines, provided they are preceded with at least one whitespace character, like so: 
echo 'Hello World!' # This is a comment too 
Everything from the # symbol onward on the line is ignored. 
Like many things, this works on the command line, too: 
[me@linuxbox \~]\$ echo 'Hello World!' # This is a comment too Hello World! 
Though comments are of little use on the command line, they will work. 
The first line of our script is a little mysterious. It looks as if it should be a comment since it starts with #, but it looks too purposeful to be just that. The #! character sequence is, in fact, a special construct called a shebang. The shebang is used to tell the kernel the name of the interpreter that should be used to execute the script that follows. Every shell script should include this as its first line. 
Let’s save our script file as hello_world. 
# Executable Permissions 
The next thing we have to do is make our script executable. This is easily done using chmod. 
[me@linuxbox $-]\$1$ ls -l hello_world 
$-r w-r--r-1$ me me 63 2009-03-07 10:10 hello_world 
[me@linuxbox \~]\$ chmod 755 hello_world 
$\mathsf{\Lambda}[\mathsf{m e}\ @\mathsf{l i n u}\times\mathsf{b o}\times\mathsf{\Lambda}\sim]\Phi$ ls -l hello_world 
-rwxr-xr-x 1 me me 63 2009-03-07 10:10 hello_world 
There are two common permission settings for scripts: 755 for scripts that everyone can execute, and 700 for scripts that only the owner can execute. Note that scripts must be readable to be executed. 

# Script File Location 
With the permissions set, we can now execute our script: 
[me@linuxbox \~]\$ ./hello_world Hello World! 
For the script to run, we must precede the script name with an explicit path. If we don’t, we get this: 
[me@linuxbox \~]\$ hello_world bash: hello_world: command not found 
Why is this? What makes our script different from other programs? As it turns out, nothing. Our script is fine. Its location is the problem. In Chapter 11, we discussed the PATH environment variable and its effect on how the system searches for executable programs. To recap, the system searches a list of directories each time it needs to find an executable program, if no explicit path is specified. This is how the system knows to execute / bin/ls when we type ls at the command line. The /bin directory is one of the directories that the system automatically searches. The list of directories is held within an environment variable named PATH. The PATH variable contains a colon-separated list of directories to be searched. We can view the contents of PATH. 
[me@linuxbox \~]\$ echo \$PATH 
/home/me/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin: 
/bin:/usr/games 
Here we see our list of directories. If our script iswere located in any of the directories in the list, our problem would be solved. Notice the first directory in the list, /home/me/ bin. Most Linux distributions configure the PATH variable to contain a bin directory in the user’s home directory to allow users to execute their own programs. So, if we create the bin directory and place our script within it, it should start to work like other programs. 
[me@linuxbox \~]\$ mkdir bin [me@linuxbox \~]\$ mv hello_world bin [me@linuxbox \~]\$ hello_world Hello World! 
And so it does. 
If the PATH variable does not contain the directory, we can easily add it by including this line in our .bashrc file: 
export PATH=\~/bin:"\$PATH" 
After this change is made, it will take effect in each new terminal session. To apply the change to the current terminal session, we must have the shell re-read the .bashrc file. This can be done by “sourcing” it. 
[me@linuxbox \~]\$ . .bashrc 
The dot (.) command is a synonym for the source command, a shell builtin that reads a specified file of shell commands and treats it like input from the keyboard. 
Note: Ubuntu (and most other Debian-based distributions) automatically adds the \~/bin directory to the PATH variable if the \~/bin directory exists when the user’s .bashrc file is executed. So, on Ubuntu systems, if we create the \~/bin directory and then log out and log in again, everything works. 
# Good Locations for Scripts 
The \~/bin directory is a good place to put scripts intended for personal use. If we write a script that everyone on a system is allowed to use, the traditional location is /usr/ local/bin. Scripts intended for use by the system administrator are often located in /usr/local/sbin. In most cases, locally supplied software, whether scripts or compiled programs, should be placed in the /usr/local hierarchy and not in /bin or / usr/bin. These directories are specified by the Linux Filesystem Hierarchy Standard to contain only files supplied and maintained by the Linux distributor. 
# More Formatting Tricks 
One of the key goals of serious script writing is ease of maintenance, that is, the ease with which a script may be modified by its author or others to adapt it to changing needs. Making a script easy to read and understand is one way to facilitate easy maintenance. 
# Long Option Names 
Many of the commands we have studied feature both short and long option names. For instance, the ls command has many options that can be expressed in either short or long form. For example, the following: 
[me@linuxbox \~]\$ ls -ad 
is equivalent to this: 
<html><body><table><tr><td>[me@linuxbox ~]$ ls --all --directory</td></tr></table></body></html> 
In the interests of reduced typing, short options are preferred when entering options on the command line, but when writing scripts, long options can provide improved readability. 
# Indentation and Line-Continuation 
When employing long commands, readability can be enhanced by spreading the command over several lines. In Chapter 17, we looked at a particularly long example of the find command. 
[me@linuxbox \~]\$ find playground \( -type f -not -perm 0600 -exec chmod 0600 ‘{}’ ‘;’ \) -or \( -type d -not -perm 0700 -exec chmod 0700 ‘{}’ ‘;’ \) 
Obviously, this command is a little hard to figure out at first glance. In a script, this com - mand might be easier to understand if written this way: 
find playground \ \( \ -type f \ -not -perm 0600 \ -exec chmod 0600 ‘{}’ ‘;’ \ \) \ -or \ \( \ -type d \ -not -perm 0700 \ -exec chmod 0700 ‘{}’ ‘;’ \ \) 
By using line continuations (backslash-linefeed sequences) and indentation, the logic of this complex command is more clearly described to the reader. This technique works on the command line, too, though it is seldom used, as it is awkward to type and edit. One difference between a script and a command line is that the script may employ tab characters to achieve indentation, whereas the command line cannot since tabs are used to activate completion. 
# Configuring vim For Script Writing 
The vim text editor has many, many configuration settings. There are several common options that can facilitate script writing. 
The following turns on syntax highlighting: 
# :syntax on 
With this setting, different elements of shell syntax will be displayed in different colors when viewing a script. This is helpful for identifying certain kinds of programming errors. It looks cool, too. Note that for this feature to work, you must have a complete version of vim installed, and the file you are editing must have a shebang indicating the file is a shell script. If you have difficulty with the previous command, try :set syntax=sh instead. 
The following turns on the option to highlight search results. 
# :set hlsearch 
Say we search for the word echo. With this option on, each instance of the word will be highlighted. 
The following sets the number of columns occupied by a tab character.: 
# :set tabstop=4 
The default is eight columns. Setting the value to 4 (which is a common practice) allows long lines to fit more easily on the screen. 
The following turns on the “auto indent” feature: 
# :set autoindent 
This causes vim to indent a new line the same amount as the line just typed. This speeds up typing on many kinds of programming constructs. To stop indentation, press Ctrl-d. 
These changes can be made permanent by adding these commands (without the leading colon characters) in your ${\sim}/$ .vimrc file. 
# Summing Up 
In this first chapter of scripting, we looked at how scripts are written and made to easily execute on our system. We also saw how we can use various formatting techniques to improve the readability (and thus the maintainability) of our scripts. In future chapters, ease of maintenance will come up again and again as a central principle in good script writing. 
# Further Reading 
For “Hello World” programs and examples in various programming languages, 
see: 
http://en.wikipedia.org/wiki/Hello_world 
This Wikipedia article talks more about the shebang mechanism: 
http://en.wikipedia.org/wiki/Shebang_(Unix) 