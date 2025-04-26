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

>  `cd -` : 切换到前一个工作目录
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

### A Longer Look at Long Format 
As we saw earlier, the -l option causes ls to display its results in long format. This format contains a great deal of useful information. Here is the Examples directory from an early Ubuntu system: 

```
-rw-r--r-- 1 root root 3576296 2017-04-03 11:05 Experience ubuntu.ogg -rw-r--r-- 1 root root 1186219 2017-04-03 11:05 kubuntu-leaflet.png -rw-r--r-- 1 root root 47584 2017-04-03 11:05 logo-Edubuntu.png -rw-r--r-- 1 root root 44355 2017-04-03 11:05 logo-Kubuntu.png -rw-r--r-- 1 root root 34391 2017-04-03 11:05 logo-Ubuntu.png -rw-r--r-- 1 root root 32059 2017-04-03 11:05 oo-cd-cover.odf -rw-r--r-- 1 root root 159744 2017-04-03 11:05 oo-derivatives.doc -rw-r--r-- 1 root root 27837 2017-04-03 11:05 oo-maxwell.odt -rw-r--r-- 1 root root 98816 2017-04-03 11:05 oo-trig.xls -rw-r--r-- 1 root root 453764 2017-04-03 11:05 oo-welcome.odt -rw-r--r-- 1 root root 358374 2017-04-03 11:05 ubuntu Sax.ogg 
```

Table 3-2 provides us with a look at the different fields from one of the files and their meanings. 

Table 3-2: ls Long Listing Fields 
<html><body><table><tr><td>Field</td><td>Meaning</td></tr><tr><td>-rw-r--r-</td><td>Access rights to the file. The first character indicates the type of file. Among the different types, a leading dash means a regular file, while a “d" indicates a directory. The next three characters are the access rights for the file's owner, the next three are for members of the file's group, and the final three are for everyone else. Chapter 9 "Permissions" discusses the full meaning of this in more detail.</td></tr><tr><td>1</td><td>File's number of hard links. See the sections "Symbolic Links" and "Hard Links" later in this chapter.</td></tr><tr><td>root</td><td>The username of the file's owner.</td></tr><tr><td>root 32059</td><td>The name of the group that owns the file.</td></tr><tr><td>2017-04-03 11:05</td><td>Size of the file in bytes.</td></tr><tr><td></td><td>Date and time of the file's last modification.</td></tr><tr><td>oo-cd-cover.odf</td><td>Name of the file.</td></tr></table></body></html> 
************

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

<html><body><table><tr><td>Directory</td><td>Comments</td></tr><tr><td>~/.config and ~/.local</td><td>These two directories are located in the home directory of each desktop user. They are used to store user-specific configuration data for desktop applications.</td></tr></table></body></html> 

## Symbolic Links 
As we look around, we are likely to see a directory listing (for example in /usr/lib) with an entry like this: 

<html><body><table><tr><td>lrwxrwxrwx 1 root root 11 2007-08-11 07:34 libc.s0.6 -> libc-2.6.s0</td></tr></table></body></html> 

Notice how the first letter of the listing is “l” and the entry seems to have two filenames? This is a special kind of a file called a symbolic link (also known as a soft link or symlink). In most Unix-like systems it is possible to have a file referenced by multiple names. While the value of this might not be obvious, it is really a useful feature. 

Picture this scenario: A program requires the use of a shared resource of some kind contained in a file named “foo,” but “foo” has frequent version changes. It would be good to include the version number in the filename so the administrator or other interested party could see what version of “foo” is installed. This presents a problem. If we change the name of the shared resource, we have to track down every program that might use it and change it to look for a new resource name every time a new version of the resource is installed. That doesn't sound like fun at all. 

Here is where symbolic links save the day. Suppose we install version 2.6 of “foo,” which has the filename “foo- $2.6^{\prime\prime}$ and then create a symbolic link simply called “foo” that points to “foo-2.6.” This means that when a program opens the file “foo”, it is actually opening the file “foo- $2.6^{\prime\prime}$ . Now everybody is happy. The programs that rely on “foo” can find it and we can still see what actual version is installed. When it is time to upgrade to “foo2.7,” we just add the file to our system, delete the symbolic link “foo” and create a new one that points to the new version. Not only does this solve the problem of the version upgrade, but it also allows us to keep both versions on our machine. Imagine that “foo $2.7^{\mathfrak{n}}$ has a bug (damn those developers!) and we need to revert to the old version. Again, we just delete the symbolic link pointing to the new version and create a new symbolic link pointing to the old version. 

The directory listing at the beginning of this section (from the /usr/lib directory of a Fedora system) shows a symbolic link called libc.so.6 that points to a shared library file called libc-2.6.so. This means that programs looking for libc.so.6 will actually get the file libc-2.6.so. We will learn how to create symbolic links in the next chapter. 
# Hard Links 
While we are on the subject of links, we need to mention that there is a second type of link called a hard link. Hard links also allow files to have multiple names, but they do it in a different way. We’ll talk more about the differences between symbolic and hard links in the next chapter. 
# Summing Up 
With our tour behind us, we have learned a lot about our system. We've seen various files and directories and their contents. One thing we should take away from this is how open the system is. In Linux there are many important files that are plain human-readable text. Unlike many proprietary systems, Linux makes everything available for examination and study. 
# Further Reading 
The full version of the Linux Filesystem Hierarchy Standard can be found here: 
https://refspecs.linuxfoundation.org/fhs.shtml 
An article about the directory structure of Unix and Unix-like systems: http:// 
en.wikipedia.org/wiki/Unix_directory_structure 
A detailed description of the ASCII text format: http://en.wikipedia.org/wiki/ 
ASCII 
# 4 – Manipulating Files and Directories 
Now we’re are ready for some real work! This chapter will introduce the following commands: 
cp – Copy files and directories mv – Move/rename files and directories mkdir – Create directories rm – Remove files and directories ln – Create hard and symbolic links 
These five commands are among the most frequently used Linux commands. They are used for manipulating both files and directories. 
Now, to be frank, some of the tasks performed by these commands are more easily done with a graphical file manager. With a file manager, we can drag and drop a file from one directory to another, cut and paste files, delete files, and so on. So why use these old command line programs? 
The answer is power and flexibility. While it is easy to perform simple file manipulations with a graphical file manager, complicated tasks can be easier with the command line programs. For example, how could we copy all the HTML files from one directory to another but only copy files that do not exist in the destination directory or are newer than the versions in the destination directory? It's pretty hard with a file manager but pretty easy with the command line. 
<html><body><table><tr><td>cp -u *.html destination</td></tr></table></body></html> 
# Wildcards 
Before we begin using our commands, we need to talk about a shell feature that makes these commands so powerful. Since the shell uses filenames so much, it provides special characters to help us rapidly specify groups of filenames. These special characters are 
called wildcards. Using wildcards (which is also known as globbing) allows us to select filenames based on patterns of characters. Table 4-1 lists the wildcards and what they select. 
Table 4-1: Wildcards 
<html><body><table><tr><td>Wildcard</td><td>Meaning</td></tr><tr><td>*</td><td>Matches any characters</td></tr><tr><td>？</td><td>Matches any single character</td></tr><tr><td>[characters]</td><td>Matches any character that is a member of the set characters</td></tr><tr><td>[!characters] or [^characters]</td><td>Matches any character that is not a member of the set characters</td></tr><tr><td>[[:class:]]</td><td>Matches any character that is a member of the specified class</td></tr></table></body></html> 
Table 4-2 lists the most commonly used character classes. 
Table 4-2: Commonly Used Character Classes 
<html><body><table><tr><td>Character Class</td><td>Meaning</td></tr><tr><td>[:alnum:]</td><td>Matches any alphanumeric character</td></tr><tr><td>[:alpha:]</td><td>Matches any alphabetic character</td></tr><tr><td>[:digit:]</td><td> Matches any numeral</td></tr><tr><td>[:lower:]</td><td>Matches any lowercase letter</td></tr><tr><td>[:upper:]</td><td>Matches any uppercase letter</td></tr></table></body></html> 
Using wildcards makes it possible to construct sophisticated selection criteria for filenames. Table 4-3 provides some examples of patterns and what they match. 
Table 4-3: Wildcard Examples 
<html><body><table><tr><td>Pattern</td><td>Matches</td></tr><tr><td>*</td><td>All files</td></tr><tr><td>g*</td><td>Any file beginning with “g"</td></tr><tr><td>b*.txt</td><td>Any file beginning with “b" followed by</td></tr></table></body></html> 
<html><body><table><tr><td></td><td>any characters and ending with “.txt"</td></tr><tr><td>Data???</td><td>Any file beginning with “Data" followed by exactly three characters</td></tr><tr><td>[abc]*</td><td>Any file beginning with either an “a", a “b",or a “c”</td></tr><tr><td>BACKUP .[0-9][0-9][0-9]</td><td>Any file beginning with “BACKUP." followed by exactly three numerals</td></tr><tr><td>[[:upper:]]*</td><td>Any file beginning with an uppercase letter</td></tr><tr><td>[![:digit:]]*</td><td>Any file not beginning with a numeral</td></tr><tr><td>*[[:lower:]123]</td><td>Any file ending with a lowercase letter or the numerals “1",“2", or “3"</td></tr></table></body></html> 
Wildcards can be used with any command that accepts filenames as arguments, but we’ll talk more about that in Chapter 7, "Seeing the World As the Shell Sees It. 
# Character Ranges 
If you are coming from another Unix-like environment or have been reading some other books on this subject, you may have encountered the [A-Z] and [az] character range notations. These are traditional Unix notations and worked in older versions of Linux as well. They can still work, but you have to be careful with them because they will not produce the expected results unless properly configured. For now, you should avoid using them and use character classes instead. 
# Dot Files 
If we look at our home directory with ls using the -a option we will notice that there are a number of files and directories whose name begin with a dot. As we have discussed, these files are hidden. It’s not a special attribute of the file; it only means that the file will not appear in the output of ls unless the -a or -A options are included. This hidden characteristic also applies to wildcards. Hidden files will not appear unless we use a wildcard pattern such as .\*. However, when we do this we will also see both . (the current directory) and .. (the current directory’s parent) in the results. To exclude them we can use patterns such as . $[!.]^{\star}$ or . $??^{\star}$ . 
# Wildcards Work in the GUI Too 
Wildcards are especially valuable not only because they are used so frequently on the command line, but because they are also supported by some graphical file managers. 
In Nautilus (the file manager for GNOME), you can select files by pressing Ctrl-s and entering a file selection pattern with wildcards and the files in the currently displayed directory will be selected. In some versions of Dolphin and Konqueror (the file managers for KDE), you can enter wildcards directly on the location bar. For example, if you want to see all the files starting with a lowercase “u” in the /usr/bin directory, enter “/ usr/bin/u\*” in the location bar and it will display the result. 
Many ideas originally found in the command line interface make their way into the graphical interface, too. It is one of the many things that make the Linux desktop so powerful. 
# mkdir – Create Directories 
The mkdir command is used to create directories. It works like this: 
would create a single directory named dir1, while the following: 
# mkdir dir1 dir2 dir3 
would create three directories named dir1, dir2, and dir3. 
# cp – Copy Files and Directories 
The cp command copies files or directories. It can be used two different ways. The following: 
# cp item1 item2 
copies the single file or directory item1 to the file or directory item2 and the following: 
# cp item... directory 
copies multiple items (either files or directories) into a directory. 
# Useful Options and Examples 
Table 4-4 lists some of the commonly used options for cp. 
Table 4-4: cp Options 
<html><body><table><tr><td>Option</td><td>Long Option</td><td>Meaning</td></tr><tr><td>-a</td><td>- -archive</td><td>Copy the files and directories and all of their attributes, including ownerships and permissions. Normally, copies take on the default attributes of the user performing the copy. We'll take a look at file permissions in Chapter 9 "Permissions."</td></tr><tr><td>-i</td><td>--interactive</td><td>Before overwriting an existing file, prompt the user for confirmation. If this option is not specified, cp will silently (meaning there will be no warning) overwrite files.</td></tr></table></body></html> 
<html><body><table><tr><td>-r</td><td>--recursive</td><td>Recursively copy directories and their contents. This option (or the - a option) is required when copying directories.</td></tr><tr><td>-u</td><td>- -update</td><td>When copying files from one directory to another, only copy files that either don't exist or are newer than the existing corresponding files, in the destination directory. This is useful when copying large numbers of files as it skips files that don't need to be copied.</td></tr><tr><td>-V</td><td>- -verbose</td><td>Display informative messages as the copy is performed.</td></tr></table></body></html> 
Table 4-5: cp Examples 
<html><body><table><tr><td>Command</td><td>Results</td></tr><tr><td>cp file1 file2</td><td>Copy file1 to file2. If file2 exists, it is overwritten with the contents of file1. If file2 does not exist, it is created.</td></tr><tr><td>cp -i file1 file2</td><td>Same as previous command, except that if file2 exists, the user is prompted before it is overwritten.</td></tr><tr><td>cp file1 file2 dir1</td><td>Copy file1 and file2 into directory dir1. The directory dir1 must already exist.</td></tr><tr><td>cp dir1/* dir2</td><td>Using a wildcard, copy all the files in dir1 into dir2. The directory dir2 must already exist.</td></tr><tr><td>cp -r dir1 dir2</td><td>Copy the contents of directory dir1 to directory dir2. If directory dir2 does not exist, it is created and, after the copy, will contain the same contents as directory dir1. If directory dir2 does exist, then directory dir1 (and its contents) will be copied into dir2.</td></tr></table></body></html> 
# mv – Move and Rename Files 
The mv command performs both file moving and file renaming, depending on how it is used. In either case, the original filename no longer exists after the operation. mv is used in much the same way as cp, as shown here: 
# mv item1 item2 
to move or rename the file or directory item1 to item2 or: 
# mv item... directory 
to move one or more items from one directory to another. 
# Useful Options and Examples 
mv shares many of the same options as cp as described in Table 4-6. 
Table 4-6: mv Options 
<html><body><table><tr><td>Option</td><td>Long Option</td><td>Meaning</td></tr><tr><td>-i</td><td>--interactive</td><td>Before overwriting an existing file, prompt the user for confirmation. If this option is not specified, mv will silently overwrite files.</td></tr><tr><td>-u</td><td>- -update</td><td>When moving files from one directory to another, only move files that either don't exist, or are newer than the existing corresponding files in the destination directory.</td></tr><tr><td>-V</td><td>--verbose</td><td>Display informative messages as the move is performed.</td></tr></table></body></html> 
Table 4-7 provides some examples of mv usage. 
Table 4-7: mv Examples 
<html><body><table><tr><td>Command</td><td>Results</td></tr><tr><td>mv file1 file2</td><td>Move file1 to file2. If file2 exists, it is overwritten with the contents of file1. If file2 does not exist, it is created. In either case, file1 ceases to exist.</td></tr><tr><td>mv -i file1 file2</td><td>Same as the previous command, except that if file2 exists, the user is prompted before it is overwritten.</td></tr><tr><td>mv file1 file2 dir1</td><td>Move file1 and file2 into directory dir1. The directory dir1 must already exist.</td></tr></table></body></html> 
<html><body><table><tr><td>mv dir1 dir2</td><td>If directory dir2 does not exist, create directory dir2 and move the contents of directory dir1 into dir2 and delete directory dir1. If directory dir2 does exist, move directory dir1 (and its contents) into directory dir2.</td></tr></table></body></html> 
# rm – Remove Files and Directories 
The rm command is used to remove (delete) files and directories, as shown here: 
rm item.. 
where item is one or more files or directories. 
# Useful Options and Examples 
Table 4-8 describes some of the common options for rm. 
Table 4-8: rm Options 
<html><body><table><tr><td>Option</td><td>Long Option</td><td>Meaning</td></tr><tr><td>-i</td><td>--interactive</td><td>Before deleting an existing file, prompt the user for confirmation. If this option is not specified, rm will silently delete files.</td></tr><tr><td>-r</td><td>- -recursive</td><td>Recursively delete directories. This means that if a directory being deleted has subdirectories, delete them too. To delete a directory, this option must be specified.</td></tr><tr><td>-f</td><td>--force</td><td>Ignore nonexistent files and do not prompt. This overrides the - - interactive option.</td></tr><tr><td>-V</td><td>- -verbose</td><td>Display informative messages as the deletion is performed.</td></tr></table></body></html> 
Table 4-9 provides some examples of using the rm command. 
Table 4-9: rm Examples 
<html><body><table><tr><td>Command</td><td>Results</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table></body></html> 
<html><body><table><tr><td>rm file1</td><td>Delete file1 silently.</td></tr><tr><td>rm -i file1</td><td>Same as the previous command, except that the user is prompted for confirmation before the deletion is performed.</td></tr><tr><td>rm -r file1 dir1</td><td>Delete file1 and dir1 and its contents.</td></tr><tr><td>rm -rf file1 dir1</td><td>Same as the previous command, except that if either file1 or dir1 do not exist, rm will continue silently.</td></tr></table></body></html> 
# Be Careful with rm! 
Unix-like operating systems such as Linux do not have an undelete command. Once you delete something with rm, it's gone. Linux assumes you're smart and you know what you're doing. 
Be particularly careful with wildcards. Consider this classic example. Let's say you want to delete just the HTML files in a directory. To do this, you type the fol - lowing: 
rm \*.html 
This is correct, but if you accidentally place a space between the \* and the .html like so: 
rm \* .html 
the rm command will delete all the files in the directory and then complain that there is no file called .html. 
Here is a useful tip: whenever you use wildcards with rm (besides carefully checking your typing!), test the wildcard first with ls. This will let you see the files that will be deleted. Then press the up arrow key to recall the command and replace ls with rm. 
# ln – Create Links 
The ln command is used to create either hard or symbolic links. It is used in one of two ways. The following creates a hard link: 
# ln file link 
The following creates a symbolic link: 
# ln -s item link 
to create a symbolic link where item is either a file or a directory. 
# Hard Links 
Hard links are the original Unix way of creating links, compared to symbolic links, which are more modern. By default, every file has a single hard link that gives the file its name. When we create a hard link, we create an additional directory entry for a file. Hard links have two important limitations: 
1. A hard link cannot reference a file outside its own file system. This means a link cannot reference a file that is not on the same disk partition as the link itself. 2. A hard link may not reference a directory. 
A hard link is indistinguishable from the file itself. Unlike a symbolic link, when we list a directory containing a hard link we will see no special indication of the link. When a hard link is deleted, the link is removed but the contents of the file itself continue to exist (that is, its space is not deallocated) until all links to the file are deleted. 
It is important to be aware of hard links because you might encounter them from time to time, but modern practice prefers symbolic links, which we will cover next. 
# Symbolic Links 
Symbolic links were created to overcome the limitations of hard links. Symbolic links work by creating a special type of file that contains a text pointer to the referenced file or directory. In this regard, they operate in much the same way as a Windows shortcut, though of course they predate the Windows feature by many years. 
A file pointed to by a symbolic link, and the symbolic link itself are largely indistinguishable from one another. For example, if we write something to the symbolic link, the referenced file is written to. However when we delete a symbolic link, only the link is deleted, not the file itself. If the file is deleted before the symbolic link, the link will continue to exist but will point to nothing. In this case, the link is said to be broken. In many implementations, the ls command will display broken links in a distinguishing color, such as red, to reveal their presence. 
The concept of links can seem confusing, but hang in there. We're going to try all this 
stuff and it will, hopefully, become clear. 
# Let's Build a Playground 
Since we are going to do some real file manipulation, let's build a safe place to “play” with our file manipulation commands. First we need a directory to work in. We'll create one in our home directory and call it playground. 
# Creating Directories 
The mkdir command is used to create a directory. To create our playground directory we will first make sure we are in our home directory and will then create the new directory. 
[me@linuxbox ~]$ cd [me@linuxbox ~]$ mkdir playground 
To make our playground a little more interesting, let's create a couple of directories inside it called dir1 and dir2. To do this, we will change our current working directory to playground and execute another mkdir. 
[me@linuxbox ~]$ cd playground [me@linuxbox playground] $|$1$ mkdir dir1 dir2 
Notice that the mkdir command will accept multiple arguments allowing us to create both directories with a single command. 
# Copying Files 
Next, let's get some data into our playground. We'll do this by copying a file. Using the cp command, we'll copy the passwd file from the /etc directory to the current working directory. 
[me@linuxbox playground] $$1$ cp /etc/passwd . 
Notice how we used shorthand for the current working directory, the single trailing period. So now if we perform an ls, we will see our file. 
[me@linuxbox playground]$ ls -l 
total 12 
drwxrwxr-x 2 me me 4096 2025-01-10 16:40 dir1 drwxrwxr-x 2 me me 4096 2025-01-10 16:40 dir2 -rw-r--r-- 1 me me 1650 2025-01-10 16:07 passwd 
Now, just for fun, let's repeat the copy using the “-v” option (verbose) to see what it does. 
[me@linuxbox playground] $$1$ cp -v /etc/passwd \`/etc/passwd' -> \`./passwd' 
The cp command performed the copy again, but this time displayed a concise message indicating what operation it was performing. Notice that cp overwrote the first copy without any warning. Again this is a case of cp assuming that we know what we're doing. To get a warning, we'll include the “-i” (interactive) option. 
[me@linuxbox playground] $$1$ cp -i /etc/passwd cp: overwrite \`./passwd'? 
Responding to the prompt by entering a y will cause the file to be overwritten, any other character (for example, n) will cause cp to leave the file alone. 
# Moving and Renaming Files 
Now, the name passwd doesn't seem very playful and this is a playground, so let's change it to something else. 
<html><body><table><tr><td>[me@linuxbox playground]$ mv passwd fun</td></tr></table></body></html> 
Let's pass the fun around a little by moving our renamed file to each of the directories and back again. The following moves it first to the directory dir1: 
<html><body><table><tr><td>[me@linuxbox playground]$ mv fun dir1</td></tr></table></body></html> 
The following then moves it from dir1 to dir2: 
[me@linuxbox playground] $$1$ mv dir1/fun dir2 
Finally, the following brings it back to the current working directory: 
[me@linuxbox playground] $$1$ mv dir2/fun 
Next, let's see the effect of mv on directories. First we will move our data file into dir1 again, like this: 
<html><body><table><tr><td>[me@linuxbox playground]$ mv fun dir1</td></tr></table></body></html> 
Then we move dir1 into dir2 and confirm it with ls. 
[me@linuxbox playground] $$1$ mv dir1 dir2 
[me@linuxbox playground] $$1$ ls -l dir2 
total 4 
drwxrwxr-x 2 me me 4096 2025-01-11 06:06 dir1 
[me@linuxbox playground] $$1$ ls -l dir2/dir1 
total 4 
-rw-r--r-- 1 me me 1650 2025-01-10 16:33 fun 
Note that since dir2 already existed, mv moved dir1 into dir2. If dir2 had not existed, mv would have renamed dir1 to dir2. Lastly, let's put everything back. 
[me@linuxbox playground] $$1$ mv dir2/dir1 . 
[me@linuxbox playground] $$1$ mv dir1/fun . 
# Creating Hard Links 
Now we'll try some links. We’ll first create some hard links to our data file like so: 
[me@linuxbox playground] $$1$ ln fun fun-hard [me@linuxbox playground] $$1$ ln fun dir1/fun-hard [me@linuxbox playground] $$1$ ln fun dir2/fun-hard 
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
# Creating Symbolic Links 
Symbolic links were created to overcome the two disadvantages of hard links. 
1. Hard links cannot span physical devices. 
2. Hard links cannot reference directories, only files. 
Symbolic links are a special type of file that contains a text pointer to the target file or directory. 
Creating symbolic links is similar to creating hard links. 
[me@linuxbox playground] $$1$ ln -s fun fun-sym [me@linuxbox playground] $$1$ ln -s ../fun dir1/fun-sym [me@linuxbox playground] $$1$ ln -s ../fun dir2/fun-sym 
The first example is pretty straightforward; we simply add the $^{66}-5^{33}$ option to create a symbolic link rather than a hard link. But what about the next two? Remember, when we create a symbolic link, we are creating a text description of where the target file is rela - tive to the symbolic link. It's easier to see if we look at the ls output shown here: 
<html><body><table><tr><td colspan="6">[me@linuxbox playground]$ ls -l dir1</td></tr><tr><td>total 4</td><td></td><td></td><td>1650 2025-01-10 16:33 fun-hard</td><td></td><td></td></tr><tr><td>-rw-r--r-- 4 me lrwxrwxrwx 1 me</td><td>me me</td><td></td><td></td><td></td><td>6 2025-01-15 15:17 fun-sym -> ../fun</td></tr></table></body></html> 
The listing for fun-sym in dir1 shows that it is a symbolic link by the leading l in the first field and that it points to ../fun, which is correct. Relative to the location of funsym, fun is in the directory above it. Notice too, that the length of the symbolic link file is 6, the number of characters in the string ../fun rather than the length of the file to which it is pointing. 
When creating symbolic links, we can either use absolute pathnames, as shown here: 
# [me@linuxbox playground] $$1$ ln -s /home/me/playground/fun dir1/fun-sym 
or relative pathnames, as we did in our earlier example. In most cases, using relative pathnames is more desirable because it allows a directory tree containing symbolic links and their referenced files to be renamed and/or moved without breaking the links. 
In addition to regular files, symbolic links can also reference directories. 
[me@linuxbox playground] $$1$ ln -s dir1 dir1-sym [me@linuxbox playground] $$1$ ls -l 
<html><body><table><tr><td colspan="8">total 16</td></tr><tr><td>drwxrwxr-x</td><td>2 me</td><td>me</td><td></td><td>4096 2025-01-15 15:17 dir1</td><td></td><td></td></tr><tr><td>lrwxrwxrwx</td><td>1 me</td><td>me</td><td></td><td></td><td></td><td>4 2025-01-16 14:45 dir1-sym -> dir1</td></tr><tr><td>drwxrwxr-x</td><td>2 me</td><td>me</td><td></td><td>4096 2025-01-15 15:17</td><td></td><td>dir2</td></tr><tr><td>-rw-r--r--</td><td>4 me</td><td>me</td><td>1650</td><td>2025-01-10 16:33</td><td></td><td>fun</td></tr><tr><td>-rw-r--r--</td><td>4 me</td><td>me</td><td>1650</td><td>2025-01-10 16:33</td><td></td><td> fun-hard</td></tr><tr><td>lrwxrwxrwx</td><td>1 me</td><td>me</td><td></td><td>3 2025-01-15</td><td>15:15</td><td>fun-sym -> fun</td></tr></table></body></html> 
# Removing Files and Directories 
As we covered earlier, the rm command is used to delete files and directories. We are going to use it to clean up our playground a little bit. First, let's delete one of our hard links. 
<html><body><table><tr><td>[me@linuxbox playground]$ rm fun-hard [me@linuxbox playground]$ ls -l</td></tr></table></body></html> 
That worked as expected. The file fun-hard is gone and the link count shown for fun is reduced from four to three, as indicated in the second field of the directory listing. Next, we'll delete the file fun, and just for enjoyment, we'll include the -i option to show what that does. 
<html><body><table><tr><td>[me@linuxbox playground]$ rm -i fun rm: remove regular file “fun'?</td></tr></table></body></html> 
Enter y at the prompt and the file is deleted. But let's look at the output of ls now. Notice what happened to fun-sym? Since it's a symbolic link pointing to a now-nonexistent file, the link is broken. 
<html><body><table><tr><td colspan="5">[me@linuxbox playground]$ ls -l</td></tr><tr><td>total 8 drwxrwxr-x 2 me me</td><td></td><td></td><td>4096 2025-01-15 15:17 dir1</td><td></td></tr><tr><td>lrwxrwxrwx 1 me me</td><td></td><td>4 2025-01-16 14:45</td><td></td><td>dir1-sym -> dir1</td></tr></table></body></html> 
<html><body><table><tr><td>drwxrwxr-x 2 me</td><td></td><td></td><td> me</td><td></td><td>4096 2025-01-15 15:17 dir2</td><td></td><td></td><td></td></tr><tr><td>lrwxrwxrwx 1 me</td><td></td><td></td><td> me </td><td></td><td>3 2025-01-15 15:15</td><td></td><td>fun-sym -> fun</td><td></td></tr></table></body></html> 
Most Linux distributions configure ls to display broken links. The presence of a broken link is not in and of itself dangerous, but it is rather messy. If we try to use a broken link we will see this: 
<html><body><table><tr><td>[me@linuxbox playground]$ less fun-sym</td></tr><tr><td>fun-sym: No such file or directory</td></tr></table></body></html> 
Let's clean up a little. We'll delete the symbolic links here: 
<html><body><table><tr><td>[me@linuxbox playground]$ ls -l</td><td>[me@linuxbox playground]$ rm fun-sym dir1-sym</td><td></td><td></td></tr><tr><td>total 8</td><td></td><td></td><td>4096 2025-01-15 15:17 dir1</td></tr><tr><td>drwxrwxr-x 2 me</td><td> me</td><td></td><td></td></tr><tr><td>drwxrwxr-x 2 me</td><td> me</td><td>4096 2025-01-15 15:17 dir2</td><td></td></tr></table></body></html> 
One thing to remember about symbolic links is that most file operations are carried out on the link's target, not the link itself. rm is an exception. When we delete a link, it is the link that is deleted, not the target. 
Finally, we will remove our playground. To do this, we will return to our home directory and use rm with the recursive option (-r) to delete playground and all of its contents, including its subdirectories. 
[me@linuxbox playground] $$1$ cd [me@linuxbox ~]$ rm -r playground 
# Creating Symlinks With The GUI 
The file managers in both GNOME and KDE provide an easy and automatic method of creating symbolic links. With GNOME, holding the Ctrl+Shift keys while dragging a file will create a link rather than copying (or moving) the file. In KDE, a small menu appears whenever a file is dropped, offering a choice of copying, moving, or linking the file. 
# Summing Up 
We've covered a lot of ground here and it will take a while for it all to fully sink in. Per - form the playground exercise over and over until it makes sense. It is important to get a good understanding of basic file manipulation commands and wildcards. Feel free to expand on the playground exercise by adding more files and directories, using wildcards to specify files for various operations. The concept of links is a little confusing at first, but take the time to learn how they work. They can be a real lifesaver. 
# Further Reading 
A discussion of symbolic links: http://en.wikipedia.org/wiki/Symbolic_link 
# 5 – Working with Commands 
Up to this point, we have seen a series of mysterious commands, each with its own mysterious options and arguments. In this chapter, we will attempt to remove some of that mystery and even create our own commands. The commands introduced in this chapter are: 
type – Indicate how a command name is interpreted which – Display which executable program will be executed help – Get help for shell builtins man – Display a command's manual page apropos – Display a list of appropriate commands info – Display a command's info entry whatis – Display one-line manual page descriptions alias – Create an alias for a command 
# What Exactly Are Commands? 
A command can be one of four different things: 
1. An executable program like all those files we saw in /usr/bin. Within this category, programs can be compiled binaries such as programs written in C and $\scriptstyle\mathbf{C}++$ , or programs written in scripting languages such as the shell, Perl, Python, Ruby, and so on. 
2. A command built into the shell itself. bash supports a number of commands internally called shell builtins. The cd command, for example, is a shell builtin. 
3. A shell function. Shell functions are miniature shell scripts incorporated into the environment. We will cover configuring the environment and writing shell functions in later chapters, but for now, just be aware that they exist. 
4. An alias. Aliases are commands that we can define ourselves, built from other commands. 
# Identifying Commands 
It is often useful to know exactly which of the four kinds of commands is being used and Linux provides a couple of ways to find out. 
# type – Display a Command's Type 
The type command is a shell builtin that displays the kind of command the shell will execute, given a particular command name. It works like this: 
# type command 
where “command” is the name of the command we want to examine. Here are some examples: 
[me@linuxbox $-]$1$ type type type is a shell builtin [me@linuxbox $-]$1$ type ls ls is aliased to \`ls --color $\c=$ auto' [me@linuxbox $-]$1$ type cp cp is /usr/bin/cp 
Here we see the results for three different commands. Notice the one for ls (taken from a Fedora system) and how the ls command is actually an alias for the ls command with the “--color $\mathbf{\sigma}=$ tty” option added. Now we know why the output from ls is displayed in color! 
# which – Display an Executable's Location 
Sometimes there is more than one version of an executable program installed on a system. While this is not common on desktop systems, it's not unusual on large servers. To determine the exact location of a given executable, the which command is used. 
[me@linuxbox ~]$ which ls /usr/bin/ls 
which only works for executable programs, not builtins nor aliases that are substitutes for actual executable programs. When we try to use which on a shell builtin for example, cd, we either get no response or get an error message: 
[me@linuxbox ~]$ which cd 
/usr/bin/which: no cd in (/usr/local/bin:/usr/bin:/bin:/usr/local 
/games:/usr/games) 
This response is a fancy way of saying “command not found.” 
# Getting a Command's Documentation 
With this knowledge of what a command is, we can now search for the documentation available for each kind of command. 
# help – Get Help for Shell Builtins 
bash has a built-in help facility available for each of the shell builtins. To use it, type “help” followed by the name of the shell builtin. Here is an example: 
[me@linuxbox ~]$ help cd 
cd: cd [-L|[-P [-e]] [-@]] [dir] Change the shell working directory. Change the current directory to DIR. The default DIR is the value of the HOME shell variable. The variable CDPATH defines the search path for the directory containing DIR. Alternative directory names in CDPATH are separated by a colon (:). A null directory name is the same as the current directory. If DIR begins with a slash (/), then CDPATH is not used. If the directory is not found, and the shell option \`cdable_vars' is set, the word is assumed to be a variable name. If that variable has a value, its value is used for DIR. 
# Options: 
-L force symbolic links to be followed: resolve symbolic links in DIR after processing instances of \`. 
-P use the physical directory structure without following symbolic links: resolve symbolic links in DIR before processing instances of \`. 
-e if the -P option is supplied, and the current working directory cannot be determined successfully, exit with a non-zero status 
- $\boldsymbol{\@parallel}$ on systems that support it, present a file with extended attributes as a directory containing the file attributes 
The default is to follow symbolic links, as if \`-L' were specified. \`..' is processed by removing the immediately previous pathname component back to a slash or the beginning of DIR. 
Exit Status: 
Returns 0 if the directory is changed, and if $PWD is set successfully when -P is used; non-zero otherwise. 
A note on notation: When square brackets appear in the description of a command's syntax, they indicate optional items. A vertical bar character indicates mutually exclusive items. In the case of the cd command above: 
This notation says that the command cd may be followed optionally by either a “-L” or a “-P” and further, if the “-P” option is specified the “-e” option may also be included followed by the optional argument “dir”. 
While the output of help for the cd commands is concise and accurate, it is by no means tutorial and as we can see, it also seems to mention a lot of things we haven't talked about yet! Don't worry. We'll get there. 
Helpful hint: By using the help command with the -m option, help will display its output in an alternate format. 
# --help – Display Usage Information 
Many executable programs support a “--help” option that displays a description of the command's supported syntax and options. For example: 
[me@linuxbox $-]$1$ mkdir --help 
Usage: mkdir [OPTION] DIRECTORY.. 
Create the DIRECTORY(ies), if they do not already exist. -Z, --context=CONTEXT (SELinux) set security context to CONTEXT 
Mandatory arguments to long options are mandatory for short options 
too. -m, --mode=MODE set file mode (as in chmod), not $\mathsf{a}\mathsf{=r w}\mathsf{\times}$ – umask -p, --parents no error if existing, make parent directories as 
needed -v, --verbose print a message for each created directory --help display this help and exit --version output version information and exit Report bugs to <bug-coreutils@gnu.org>. 
Some programs don't support the “--help” option, but try it anyway. Often it results in an error message that will reveal the same usage information. 
# man – Display a Program's Manual Page 
Most executable programs intended for command line use provide a formal piece of documentation called a manual or man page. A special paging program called man is used to view them. It is used like this: 
# man program 
where “program” is the name of the command to view. 
Man pages vary somewhat in format but generally contain the following: 
A title (the page’s name) 
A synopsis of the command's syntax 
A description of the command's purpose 
A listing and description of each of the command's options 
Man pages, however, do not usually include examples, and are intended as a reference, not a tutorial. As an example, let's try viewing the man page for the ls command: 
[me@linuxbox ~]$ man ls 
On most Linux systems, man uses less to display the manual page, so all of the familiar less commands work while displaying the page. 
The “manual” that man displays is divided into sections and covers not only user commands but also system administration commands, programming interfaces, file formats and more. Table 5-1 describes the layout of the manual. 
Table 5-1: Man Page Organization 
<html><body><table><tr><td>Section</td><td>Contents</td></tr><tr><td>1</td><td>User commands</td></tr><tr><td>２</td><td> Programming interfaces for kernel system calls</td></tr><tr><td>3</td><td>Programming interfaces to the C library</td></tr><tr><td>4</td><td>Special files such as device nodes and drivers</td></tr><tr><td>5</td><td>File formats</td></tr><tr><td>6</td><td>Games and amusements such as screen savers</td></tr><tr><td>7</td><td>Miscellaneous</td></tr><tr><td>8</td><td>System administration commands</td></tr></table></body></html> 
Sometimes we need to refer to a specific section of the manual to find what we are looking for. This is particularly true if we are looking for a file format that is also the name of a command. Without specifying a section number, we will always get the first instance of a match, probably in section 1. To specify a section number, we use man like this: 
<html><body><table><tr><td>man section search_term</td></tr></table></body></html> 
Here's an example: 
<html><body><table><tr><td>[me@linuxbox ~]$ man 5 passwd</td></tr></table></body></html> 
This will display the man page describing the file format of the /etc/passwd file. 
# apropos – Display Appropriate Commands 
It is also possible to search the list of man pages for possible matches based on a search term. It's crude but sometimes helpful. Here is an example of a search for man pages using the search term partition: 
<html><body><table><tr><td colspan="5">[me@linuxbox ~]$ apropos partition</td></tr><tr><td>addpart (8)</td><td> - simple wrapper around the "add partition".</td><td></td><td></td><td></td></tr><tr><td>all-swaps (7)</td><td>- event signalling that all swap partitions.</td><td></td><td></td><td></td></tr><tr><td>cfdisk (8)</td><td>- display or manipulate disk partition table</td><td></td><td></td><td></td></tr></table></body></html> 
<html><body><table><tr><td>cgdisk (8)</td><td>- Curses-based GUID partition table (GPT)...</td></tr><tr><td>delpart (8)</td><td>- simple wrapper around the "del partition".</td></tr><tr><td>fdisk (8)</td><td>- manipulate disk partition table</td></tr><tr><td>fixparts (8)</td><td>- MBR partition table repair utility</td></tr><tr><td></td><td>- Interactive GUID partition table (GPT).</td></tr><tr><td>gdisk (8)</td><td></td></tr><tr><td>mpartition (1)</td><td>- partition an MsDos hard disk</td></tr><tr><td>partprobe (8)</td><td>- inform the Os of partition table changes</td></tr><tr><td>partx (8) resizepart (8)</td><td>- tell the Linux kernel about the presence.. - simple wrapper around the "resize partition..</td></tr><tr><td></td><td></td></tr><tr><td>sfdisk (8)</td><td>- partition table manipulator for Linux</td></tr><tr><td>sgdisk (8)</td><td>- Command-line GUID partition table (GPT)..</td></tr></table></body></html> 
The first field in each line of output is the name of the man page, and the second field shows the section. Note that the man command with the $\bf\Pi^{\epsilon_{\mathrm{<}}}\bf k^{\prime\prime}$ option performs the same function as apropos. 
# whatis – Display One-line Manual Page Descriptions 
The whatis program displays the name and a one-line description of a man page matching a specified keyword: 
<html><body><table><tr><td>[me@linuxbox ~]$ whatis ls ls</td><td>(1) - list directory contents</td></tr></table></body></html> 
# The Most Brutal Man Page Of Them All 
As we have seen, the manual pages supplied with Linux and other Unix-like systems are intended as reference documentation and not as tutorials. Many man pages are hard to read, but I think that the grand prize for difficulty has got to go to the man page for bash. As I was doing research for this book, I gave the bash man page careful review to ensure that I was covering most of its topics. When printed, it's more than 80 pages long and extremely dense, and its structure makes absolutely no sense to a new user. 
On the other hand, it is very accurate and concise, as well as being extremely complete. So check it out if you dare and look forward to the day when you can read it and it all makes sense. 
# info – Display a Program's Info Entry 
The GNU Project provides an alternative to man pages for their programs, called “info.” Info manuals are displayed with a reader program named, appropriately enough, info. Info pages are hyperlinked much like web pages. Here is a sample: 
<html><body><table><tr><td>File: coreutils.info, Node: ls invocation, Next: dir invocation, Up: Directory listing 10.1 ^ls': List directory contents</td></tr><tr><td>The *ls' program lists information about files (of any type, including directories). Options and file arguments can be intermixed arbitrarily, as usual. For non-option command-line arguments that are directories, by default *ls' lists the contents of directories, not recursively, and omitting files with names beginning with ^.'. For other non-option arguments, by default *ls' lists just the filename. If no non-option</td></tr></table></body></html> 
The info program reads info files, which are tree structured into individual nodes, each containing a single topic. Info files contain hyperlinks that can move the reader from node to node. A hyperlink can be identified by its leading asterisk and is activated by placing the cursor upon it and pressing the Enter key. 
To invoke info, type info followed optionally by the name of a program. Table 5-2 describes the commands used to control the reader while displaying an info page. 
Table 5-2: info Commands 
<html><body><table><tr><td>Command</td><td>Action</td></tr><tr><td>？</td><td>Display command help</td></tr><tr><td>PgUp or Backspace</td><td>Display previous page</td></tr><tr><td>PgDn or Space</td><td>Display next page</td></tr><tr><td>n</td><td>Next - Display the next node</td></tr><tr><td>p</td><td>Previous - Display the previous node</td></tr><tr><td>u</td><td>Up - Display the parent node of the currently displayed node, usually a menu</td></tr></table></body></html> 
<html><body><table><tr><td>Enter</td><td>Follow the hyperlink at the cursor location</td></tr><tr><td>q</td><td>Quit</td></tr></table></body></html> 
Most of the command line programs we have discussed so far are part of the GNU Project's coreutils package, so typing the following: 
<html><body><table><tr><td>[me@linuxbox ~]$ info coreutils</td></tr></table></body></html> 
will display a menu page with hyperlinks to each program contained in the coreutils package. 
# README and Other Program Documentation Files 
Many software packages installed on our system have documentation files residing in the /usr/share/doc directory. Most of these are stored in plain text format and can be viewed with less. Some of the files are in HTML format and can be viewed with a web browser. We may encounter some files ending with a “.gz” extension. This indicates that they have been compressed with the gzip compression program. The gzip package includes a special version of less called zless that will display the contents of gzipcompressed text files. 
# Creating Our Own Commands with alias 
Now for our first experience with programming! We will create a command of our own using the alias command. But before we start, we need to reveal a small command line trick. It's possible to put more than one command on a line by separating each command with a semicolon. It works like this: 
command1; command2; command3... 
Here's the example we will use: 
[me@linuxbox ~]$ cd /usr; ls; cd - 
bin games include lib local sbin share src 
/home/me 
[me@linuxbox ~]$ 
As we can see, we have combined three commands on one line. First we change directory to /usr then list the directory and finally return to the original directory (by using 'cd -') so we end up where we started. Now let's turn this sequence into a new command using alias. The first thing we have to do is dream up a name for our new command. Let's try “test”. Before we do that, it would be a good idea to find out if the name “test” is already being used. To find out, we can use the type command again: 
<html><body><table><tr><td>[me@linuxbox ~]$ type test test is a shell builtin</td></tr></table></body></html> 
Oops! The name test is already taken. Let's try foo: 
[me@linuxbox $-]$1$ type foo bash: type: foo: not found 
Great! “foo” is not taken. So let's create our alias: 
[me@linuxbox ~]$ alias foo $\mathbf{\lambda}=\mathbf{\lambda}$ 'cd /usr; ls; cd -' 
Notice the structure of this command shown here: 
alias name $\underline{{\underline{{\mathbf{\Pi}}}}}=$ 'string' 
After the command alias, we give alias a name followed immediately (no whitespace allowed) by an equal sign, followed immediately by a quoted string containing the meaning to be assigned to the name. After we define our alias, we can use it anywhere the shell would expect a command. Let's try it: 
[me@linuxbox ~]$ foo 
bin games include lib local sbin share src 
/home/me 
[me@linuxbox ~]$ 
We can also use the type command again to see our alias: 
[me@linuxbox ~]$ type foo foo is aliased to \`cd /usr; ls; cd - 
To remove an alias, the unalias command is used, like so: 
[me@linuxbox ~]$ unalias foo $[\mathfrak{m}\in{\varnothing}{\mathrm{{linu}}}\times\mathfrak{b}\circ\times\ \mathfrak{-}]\Phi$ type foo bash: type: foo: not found 
While we purposefully avoided naming our alias with an existing command name, it is not uncommon to do so. This is often done to apply a commonly desired option to each invocation of a common command. For instance, we saw earlier how the ls command is often aliased to add color support: 
[me@linuxbox ~]$ type ls ls is aliased to \`ls --color=tty' 
To see all the aliases defined in the environment, use the alias command without arguments. Here are some of the aliases defined by default on a Fedora system. Try to figure out what they all do: 
[me@linuxbox ~]$ alias alias l. $\mathbf{\epsilon}=\mathbf{\epsilon}^{\prime}$ ls -d .\* --color=tty' alias ll='ls -l --color=tty' alias ls='ls --color=tty' 
There is one tiny problem with defining aliases on the command line. They vanish when our shell session ends. In Chapter 11, "The Environment", we will see how to add our own aliases to the files that establish the environment each time we log on, but for now, enjoy the fact that we have taken our first, albeit tiny, step into the world of shell pro - gramming! 
# Summing Up 
Now that we have learned how to find the documentation for commands, go and look up the documentation for all the commands we have encountered so far. Study what additional options are available and try them! 
# Further Reading 
There are many online sources of documentation for Linux and the command line. Here are some of the best: 
The Bash Reference Manual is a reference guide to the bash shell. It’s still a reference work but contains examples and is easier to read than the bash man page. http://www.gnu.org/software/bash/manual/bashref.html 
The Bash FAQ contains answers to frequently asked questions regarding bash. This list is aimed at intermediate to advanced users, but contains a lot of good information. 
http://mywiki.wooledge.org/BashFAQ 
The GNU Project provides extensive documentation for its programs, which form the core of the Linux command line experience. You can see a complete list here: http://www.gnu.org/manual/manual.html 
Wikipedia has an interesting article on man pages: 
http://en.wikipedia.org/wiki/Man_page 
# 6 – Redirection 
In this lesson we are going to unleash what may be the coolest feature of the command line. It's called I/O redirection. The “I/O” stands for input/output and with this facility we can redirect the input and output of commands to and from files, as well as connect multiple commands together into powerful command pipelines. To show off this facility, we will introduce the following commands: 
cat – Concatenate files 
sort – Sort lines of text 
uniq – Report or omit repeated lines 
grep – Print lines matching a pattern 
wc – Print newline, word, and byte counts for each file 
head – Output the first part of a file 
tail – Output the last part of a file 
tee – Read from standard input and write to standard output and files 
# Standard Input, Output, and Error 
Many of the programs that we have used so far produce output of some kind. This output often consists of two types: 
The program's results, that is, the data the program is designed to produce Status and error messages that tell us how the program is getting along 
If we look at a command like ls, we can see that it displays its results and its error messages on the screen. 
Keeping with the Unix theme of “everything is a file,” programs such as ls actually send their results to a special file called standard output (often expressed as stdout) and their status messages to another file called standard error (stderr). By default, both standard output and standard error are linked to the screen and not saved into a disk file. 
In addition, many programs take input from a facility called standard input (stdin), which is, by default, attached to the keyboard. 
I/O redirection allows us to change where output goes and where input comes from. Normally, output goes to the screen and input comes from the keyboard, but with I/O redirection, we can change that. 
# Redirecting Standard Output 
I/O redirection allows us to redefine where standard output goes. To redirect standard output to another file instead of the screen, we use the $>$ redirection operator followed by the name of the file. Why would we want to do this? It's often useful to store the output of a command in a file. For example, we could tell the shell to send the output of the ls command to the file ls-output.txt instead of the screen: 
[me@linuxbox ~]$ ls -l /usr/bin $>$ ls-output.txt 
Here, we created a long listing of the /usr/bin directory and sent the results to the file ls-output.txt. Let's examine the redirected output of the command, shown here: 
[me@linuxbox ~]$ ls -l ls-output.txt -rw-rw-r-- 1 me me 167878 2025-02-01 15:07 ls-output.txt 
Good — a nice, large, text file. If we look at the file with less, we will see that the file ls-output.txt does indeed contain the results from our ls command. 
[me@linuxbox ~]$ less ls-output.txt 
Now, let's repeat our redirection test, but this time with a twist. We'll change the name of the directory to one that does not exist: 
[me@linuxbox ~]$ ls -l /bin/usr $>$ ls-output.txt ls: cannot access /bin/usr: No such file or director 
We received an error message. This makes sense since we specified the nonexistent directory /bin/usr, but why was the error message displayed on the screen rather than being redirected to the file ls-output.txt? The answer is that the ls program does not send its error messages to standard output. Instead, like most well-written Unix programs, it sends its error messages to standard error (stderr). Since we only redirected standard output and not standard error, the error message was still sent to the screen. We'll see how to redirect standard error in just a minute, but first let's look at what happened to our output file: 
<html><body><table><tr><td>[me@linuxbox ~]$ ls -l ls-output.txt -rw-rw-r-- 1 me me 0 2025-02-01 15:08 ls-0utput.txt</td></tr></table></body></html> 
The file now has zero length! This is because when we redirect output with the $">"$ redirection operator, the destination file is always rewritten from the beginning. Since our ls command generated no results and only an error message, the redirection operation started to rewrite the file and then stopped because of the error, resulting in its truncation. In fact, if we ever need to actually truncate a file (or create a new, empty file), we can use a trick like this: 
<html><body><table><tr><td>[me@linuxbox ~]$ > ls-output.txt</td></tr></table></body></html> 
Simply using the redirection operator with no command preceding it will truncate an existing file or create a new, empty file. 
So, how can we append redirected output to a file instead of overwriting the file from the beginning? For that, we use the $\gg$ redirection operator, like so: 
# [me@linuxbox ~]$ ls -l /usr/bin >> ls-output.txt 
Using the $\gg$ operator will result in the output being appended to the file. If the file does not already exist, it is created just as though the $>$ operator had been used. Let's put it to the test by repeating a command and appending its output to a file: 
[me@linuxbox ~]$ ls -l /usr/bin $\gg$ ls-output.txt [me@linuxbox ~]$ ls -l /usr/bin >> ls-output.txt [me@linuxbox ~]$ ls -l /usr/bin >> ls-output.txt [me@linuxbox ~]$ ls -l ls-output.txt -rw-rw-r-- 1 me me 503634 2025-02-01 15:45 ls-output.txt 
We repeated the ls command three times resulting in an output file three times as large. 
# Group Commands 
Let’s imagine a situation where we want to execute a series of commands and send the results to a log file. With we know already, we could do this: 
[me@linuxbox ~]$ command1 $>$ logfile.txt [me@linuxbox ~]$ command2 $\gg$ logfile.txt [me@linuxbox ~]$ command3 $\gg$ logfile.txt 
The first command in this sequence creates/truncates a file named logfile.txt and each subsequent command appends its output to that file. This technique will work but there is a lot of redundant typing. There must be a better way. 
As we saw in the previous chapter, we can put multiple commands on a single line like this: 
[me@linuxbox ~]$ command1; command2; command3 
So we could place all of our commands and redirections on a single line: 
[me@linuxbox ~]$ command1 $>$ logfile.txt; command2 $\gg$ logfile.txt; command3 $\gg$ logfile.txt 
But what if we could treat the sequence as a single entity with a single output stream? We can do this by creating a group command. To do this, we surround our sequence with brace characters: 
[me@linuxbox ~]$ { command1; command2; command3; $\}>$ logfile.txt 
With our sequence surrounded by braces, the shell will consider it a single command in terms of redirection. Note that the shell requires whitespace around the braces and the final command in the sequence must be terminated with either a semicolon or a newline. 
# Redirecting Standard Error 
Redirecting standard error lacks the ease of a dedicated redirection operator. To redirect standard error we must refer to its file descriptor. A program can produce output on any of several numbered file streams. While we have referred to the first three of these file streams as standard input, output and error, the shell references them internally as file descriptors 0, 1, and 2, respectively. The shell provides a notation for redirecting files using the file descriptor number. Since standard error is the same as file descriptor number 2, we can redirect standard error with this notation: 
[me@linuxbox ~]$ ls -l /bin/usr $2>$ ls-error.txt 
The file descriptor $^{\infty}2^{\dag}$ is placed immediately before the redirection operator to perform the redirection of standard error to the file ls-error.txt. 
# Redirecting Standard Output and Standard Error to One File 
There are cases in which we may want to capture all of the output of a command to a single file. To do this, we must redirect both standard output and standard error at the same time. There are two ways to do this. Shown here is the traditional way, which works with old versions of the shell: 
# [me@linuxbox ~]$ ls -l /bin/usr $>$ ls-output.txt $2>21$ 
Using this method, we perform two redirections. First we redirect standard output to the file ls-output.txt and then we redirect file descriptor 2 (standard error) to file descriptor 1 (standard output) using the notation $2>\&1$ . 
Notice that the order of the redirections is significant. The redirection of standard error must always occur after redirecting standard output or it doesn't work. The following example redirects standard error to the file ls-output.txt: 
>ls-output.txt 2>&1 
However, if the order is changed to the following, standard error is directed to the screen. 
Recent versions of bash provide a second, more streamlined method for performing this combined redirection shown here: 
[me@linuxbox ~]$ ls -l /bin/usr &> ls-output.txt 
In this example, we use the single notation $\&>$ to redirect both standard output and standard error to the file ls-output.txt. We can also append the standard output and standard error streams to a single file like so: 
[me@linuxbox ~]$ ls -l /bin/usr &>> ls-output.txt 
# Disposing of Unwanted Output 
Sometimes “silence is golden,” and we don't want output from a command, we just want to throw it away. This applies particularly to error and status messages. The system provides a way to do this by redirecting output to a special file called “/dev/null”. This file is a system device often referred to as a bit bucket, which accepts input and does nothing with it. To suppress error messages from a command, we do this: 
[me@linuxbox ~]$ ls -l /bin/usr 2> /dev/null 
# /dev/null In Unix Culture 
The bit bucket is an ancient Unix concept and because of its universality, it has appeared in many parts of Unix culture. When someone says he/she is sending your comments to /dev/null, now you know what it means. For more examples, see the Wikipedia article on /dev/null. 
# Redirecting Standard Input 
Up to now, we haven't encountered any commands that make use of standard input (actually we have, but we’ll reveal that surprise a little bit later), so we need to introduce one. 
# cat – Concatenate Files 
The cat command reads one or more files and copies them to standard output like so: 
We can use it to display files without paging. For example, the following will display the contents of the file ls-output.txt: 
[me@linuxbox ~]$ cat ls-output.txt 
cat is often used to display short text files. Since cat can accept more than one file as an argument, it can also be used to join files together. Say we have downloaded a large file that has been split into multiple parts (multimedia files are often split this way on Usenet), and we want to join them back together. If the files were named: 
movie.mpeg.001 movie.mpeg.002 ... movie.mpeg.099 we could join them back together with this command as follows: 
cat movie.mpeg. ${\mathfrak{o}}^{\star}$ $>$ movie.mpeg 
Since wildcards always expand in sorted order, the arguments will be arranged in the correct order. 
This is all well and good, but what does this have to do with standard input? Nothing yet, but let's try something else. What happens if we enter cat with no arguments? 
[me@linuxbox ~]$ cat 
Nothing happens, it just sits there like it's hung. It might seem that way, but it's really doing exactly what it's supposed to do. 
If cat is not given any arguments, it reads from standard input and since standard input is, by default, attached to the keyboard, it's waiting for us to type something! Try adding the following text and pressing Enter: 
[me@linuxbox ~]$ cat The quick brown fox jumps over the lazy dog. 
Next, type a Ctrl-d (i.e., hold down the Ctrl key and press “d”) to tell cat that it has reached end of file (EOF) on standard input: 
[me@linuxbox $-]$1$ cat The quick brown fox jumps over the lazy dog. 
The quick brown fox jumps over the lazy dog. 
In the absence of filename arguments, cat copies standard input to standard output, so we see our line of text repeated. We can use this behavior to create short text files. Let's say we wanted to create a file called lazy_dog.txt containing the text in our example. We would do this: 
[me@linuxbox $-]$1$ cat $>$ lazy_dog.txt The quick brown fox jumps over the lazy dog. 
Type the command followed by the text we want to place in the file. Remember to type Ctrl-d at the end. Using the command line, we have implemented the world's dumbest word processor! To see our results, we can use cat to copy the file to stdout again. 
[me@linuxbox $-]$1$ cat lazy_dog.txt The quick brown fox jumps over the lazy dog. 
Now that we know how cat accepts standard input, in addition to filename arguments, let's try redirecting standard input. 
[me@linuxbox $-]$1$ cat $\mathbf{\boldsymbol{\mathsf{\Sigma}}}<\mathbf{\boldsymbol{\mathsf{\Sigma}}}$ lazy_dog.txt The quick brown fox jumps over the lazy dog. 
Using the $\rvert<$ redirection operator, we change the source of standard input from the keyboard to the file lazy_dog.txt. We see that the result is the same as passing a single filename argument. This is not particularly useful compared to passing a filename argument, but it serves to demonstrate using a file as a source of standard input. Other commands make better use of standard input, as we will soon see. 
Before we move on, check out the man page for cat, because it has several interesting options. 
# Pipelines 
The capability of commands to read data from standard input and send to standard output is utilized by a shell feature called pipelines. Using the pipe operator | (vertical bar), the standard output of one command can be piped into the standard input of another. 
To fully demonstrate this, we are going to need some commands. Remember how we said there was one we already knew that accepts standard input? It's less. We can use less to display, page by page, the output of any command that sends its results to standard output: 
# [me@linuxbox ~]$ ls -l /usr/bin | less 
This is extremely handy! Using this technique, we can conveniently examine the output of any command that produces standard output. 
# The Difference Between $>$ and | 
At first glance, it may be hard to understand the redirection performed by the pipeline operator | versus the redirection operator $>$ . Simply put, the redirection operator connects a command with a file, while the pipeline operator connects the output of one command with the input of a second command. 
command1 $>$ file1 command1 | command2 
A lot of people will try the following when they are learning about pipelines, “just to see what happens”: 
command1 $>$ command2 
Answer: sometimes something really bad. 
Here is an actual example submitted by a reader who was administering a Linuxbased server appliance. As the superuser, he did this: 
# cd /usr/bin # ls $>$ less 
The first command put him in the directory where most programs are stored and the second command told the shell to overwrite the file less with the output of the ls command. Since the /usr/bin directory already contained a file named less (the less program), the second command overwrote the less program file with the text from ls, thus destroying the less program on his system. 
The lesson here is that the $>$ redirection operator silently creates or overwrites files, so you need to treat it with a lot of respect. 
# Filters 
Pipelines are often used to perform complex operations on data. It is possible to put several commands together into a pipeline. Frequently, the commands used this way are referred to as filters. Filters take input, change it somehow, and then output it. The first one we will try is sort. Imagine we wanted to make a combined list of all the executable programs in /bin and /usr/bin, put them in sorted order and view the resulting list: 
# [me@linuxbox ~]$ ls /bin /usr/bin | sort | less 
Since we specified two directories (/bin and /usr/bin), the output of ls would have consisted of two sorted lists, one for each directory. By including sort in our pipeline, we changed the data to produce a single, sorted list. 
sort is a powerful command with many features and options. We’ll cover them in detail in Chapter 20. 
# uniq - Report or Omit Repeated Lines 
The uniq command is often used in conjunction with sort. uniq accepts a sorted list of data from either standard input or a single filename argument (see the uniq man page for details) and, by default, removes any duplicates from the list. So, to make sure our list has no duplicates (that is, any programs of the same name that appear in both the /bin and /usr/bin directories), we will add uniq to our pipeline. 
# [me@linuxbox ~]$ ls /bin /usr/bin | sort | uniq | less 
In this example, we use uniq to remove any duplicates from the output of the sort command. If we want to see the list of duplicates instead, we add the “-d” option to uniq like so: 
[me@linuxbox ~]$ ls /bin /usr/bin | sort | uniq -d | less 
# wc – Print Line, Word, and Byte Counts 
The wc (word count) command is used to display the number of lines, words, and bytes contained in files. Here's an example: 
[me@linuxbox ~]$ wc ls-output.txt 7902 64566 503634 ls-output.txt 
In this case, it prints out three numbers: lines, words, and bytes contained in ls-output.txt. Like our previous commands, if executed without command line arguments, wc accepts standard input. The “-l” option limits its output to only report lines. Adding it to a pipeline is a handy way to count things. To see the number of items we have in our sorted list, we can do this: 
[me@linuxbox ~]$ ls /bin /usr/bin | sort | uniq | wc -l 2728 
# grep – Print Lines Matching a Pattern 
grep is a powerful program used to find text patterns within files. It's used like this: 
grep pattern [file...] 
When grep encounters a “pattern” in the file, it prints out the lines containing it. The patterns that grep can match can be very complex, but for now we will concentrate on simple text matches. We'll cover the advanced patterns, called regular expressions in Chapter 19. 
Let's say we wanted to find all the files in our list of programs that had the word zip embedded in the name. Such a search might give us an idea of some of the programs on our system that had something to do with file compression. We would do this: 
[me@linuxbox ~]$ ls /bin /usr/bin | sort | uniq | grep zip 
bunzip2 
bzip2 
gunzip 
gzip 
unzip 
<html><body><table><tr><td>zip</td></tr><tr><td>zipcloak</td></tr><tr><td>zipgrep</td></tr><tr><td>zipinfo</td></tr><tr><td>zipnote</td></tr><tr><td>zipsplit</td></tr></table></body></html> 
Here are a few handy options for grep: 
-i, causes grep to ignore case when performing the search (normally searches are case sensitive) 
-l, causes grep to only output the names of the files containing text that matches the pattern. 
-v, causes grep to print only those lines that do not match the pattern. 
-w, causes grep to only match whole words. 
# head / tail – Print First / Last Part of Files 
Sometimes we don't want all the output from a command. We may only want the first few lines or the last few lines. The head command prints the first ten lines of a file, and the tail command prints the last ten lines. While both commands print ten lines of text by default, this can be adjusted with the -n option. 
<html><body><table><tr><td>[me@linuxbox ~]$ head -n 5 ls-output.txt</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="4"></td></tr><tr><td>total 343496</td><td></td><td></td><td></td><td></td></tr><tr><td>-rwxr-xr-x 1 root root</td><td></td><td></td><td></td><td>31316 2007-12-05 08:58[</td><td></td><td></td></tr><tr><td>-rwxr-xr-x 1 root root</td><td></td><td></td><td></td><td>8240 2007-12-09 13:39 411toppm</td><td></td><td></td></tr><tr><td>-rwxr-xr-x 1 root root</td><td></td><td></td><td></td><td>111276 2007-11-26 14:27 a2p</td><td></td><td></td></tr><tr><td>-rwxr-xr-x 1 root root</td><td></td><td></td><td></td><td>25368 2006-10-06 20:16 a52dec</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>[me@linuxbox ~]$ tail -n 5 ls-output.txt</td><td></td><td></td><td></td></tr><tr><td>-rwxr-xr-x 1 root root</td><td></td><td></td><td></td><td>5234 2007-06-27 10:56 znew</td><td></td><td></td></tr><tr><td>-rwxr-xr-x 1 root root</td><td></td><td></td><td></td><td></td><td></td><td>691 2005-09-10 04:21 zonetab2pot.py</td></tr><tr><td>-rw-r--r-- 1 root root</td><td></td><td></td><td></td><td>930 2007-11-01 12:23 z0netab2p0t.pyc</td><td></td><td></td></tr><tr><td>-rw-r--r-- 1 root root</td><td></td><td></td><td></td><td>930 2007-11-01 12:23 z0netab2p0t.pyo</td><td></td><td></td></tr><tr><td>lrwxrwxrwx 1 root root</td><td></td><td></td><td></td><td></td><td></td><td>6 2016-01-31 05:22 zs0elim -> s0elim</td></tr></table></body></html> 
These commands can be used in pipelines as well: 
<html><body><table><tr><td>[me@linuxbox ~]$ ls /usr/bin | tail -n 5</td></tr></table></body></html> 
<html><body><table><tr><td>znew</td></tr><tr><td>zonetab2pot .py</td></tr><tr><td>zonetab2pot .pyc</td></tr><tr><td>zonetab2pot .pyo</td></tr><tr><td>zsoelim</td></tr></table></body></html> 
Using the -n option with head and tail together allows us to cut an excerpt from the middle of a file. Let’s imagine we have a text file with a 5 line header and a 5 line footer that we want to remove leaving only the “good” part in the middle containing the data. We could do a trick like this: 
[me@linuxbox ~]$ head -n -5 text_header_footer.txt | tail -n +5 > text.txt 
The $-n$ option when used with head allows a negative value which causes all but the last $n$ lines to be output. Similarly, the -n option with tail allows a plus sign causing all but the first $n$ lines to be output. 
tail also has an option which allows us to follow the contents of a file in real time. This is useful for watching the progress of log files as they are being written. In the following example, we will look at the messages file in /var/log (or the /var/log/syslog file if messages is missing). Superuser privileges may be required to do this on some Linux distributions because log files may contain security information: 
[me@linuxbox ~]$ tail -f /var/log/messages 
Feb 8 13:40:05 twin4 dhclient: DHCPACK from 192.168.1.1 
Feb 8 13:40:05 twin4 dhclient: bound to 192.168.1.4 -- renewal in 1652 seconds. 
Feb 8 13:55:32 twin4 mountd[3953]: /var/NFSv4/musicbox exported to both 192.168.1.0/24 and twin7.localdomain in 
192.168.1.0/24,twin7.localdomain 
Feb 8 14:07:37 twin4 dhclient: DHCPREQUEST on eth0 to 192.168.1.1 port 67 
Feb 8 14:07:37 twin4 dhclient: DHCPACK from 192.168.1.1 
Feb 8 14:07:37 twin4 dhclient: bound to 192.168.1.4 -- renewal in 1771 seconds. 
Feb 8 14:09:56 twin4 smartd[3468]: Device: /dev/hda, SMART 
Prefailure Attribute: 8 Seek_Time_Performance changed from 237 to 236 Feb 8 14:10:37 twin4 mountd[3953]: /var/NFSv4/musicbox exported to both 192.168.1.0/24 and twin7.localdomain in 
192.168.1.0/24,twin7.localdomain 
Feb 8 14:25:07 twin4 sshd(pam_unix)[29234]: session opened for user 
me by ( $\mathbf{uid}{=}\Theta$ ) 
Feb 8 14:25:36 twin4 su(pam_unix)[29279]: session opened for user 
root by me(uid $=500$ ) 
Using the -f option, tail continues to monitor the file, and when new lines are appended, they immediately appear on the display. This continues until we press Ctrl-c. 
# tee – Read from Stdin and Output to Stdout and Files 
In keeping with our plumbing metaphor, Linux provides a command called tee which creates a “tee” fitting on our pipe. The tee program reads standard input and copies it to both standard output (allowing the data to continue down the pipeline) and to one or more files. This is useful for capturing a pipeline's contents at an intermediate stage of processing. Here we repeat one of our earlier examples, this time including tee to capture the entire directory listing to the file ls.txt before grep filters the pipeline's contents: 
<html><body><table><tr><td>[me@linuxbox ~]$ ls /usr/bin I tee ls.txt I grep zip</td></tr><tr><td>bunzip2 bzip2</td></tr><tr><td></td></tr><tr><td>gunzip</td></tr><tr><td>gzip unzip</td></tr><tr><td>zip</td></tr><tr><td>zipcloak</td></tr><tr><td>zipgrep</td></tr><tr><td>zipinfo</td></tr><tr><td>zipnote</td></tr><tr><td>zipsplit</td></tr></table></body></html> 
# Summing Up 
As always, check out the documentation of each of the commands we have covered in this chapter. We have seen only their most basic usage. They all have a number of interesting options. As we gain Linux experience, we will see that the redirection feature of the command line is extremely useful for solving specialized problems. There are many commands that make use of standard input and output, and almost all command line programs use standard error to display their informative messages. 
# Linux Is About Imagination 
When I am asked to explain the difference between Windows and Linux, I often use a toy analogy. 
Windows is like a Game Boy. You go to the store and buy one all shiny new in the box. You take it home, turn it on, and play with it. Pretty graphics, cute sounds. After a while, though, you get tired of the game that came with it, so you go back to the store and buy another one. This cycle repeats over and over. Finally, you go back to the store and say to the person behind the counter, “I want a game that does this!” only to be told that no such game exists because there is no “market demand” for it. Then you say, “But I only need to change this one thing!” The person behind the counter says you can't change it. The games are all sealed up in their cartridges. You discover that your toy is limited to the games others have decided that you need. 
Linux, on the other hand, is like the world's largest Erector Set. You open it, and it's just a huge collection of parts. There's a lot of steel struts, screws, nuts, gears, pulleys, motors, and a few suggestions on what to build. So, you start to play with it. You build one of the suggestions and then another. After a while you discover that you have your own ideas of what to make. You don't ever have to go back to the store, as you already have everything you need. The Erector Set takes on the shape of your imagination. It does what you want. 
Your choice of toys is, of course, a personal thing, so which toy would you find more satisfying? 
# 7 – Seeing the World as the Shell Sees It 
In this chapter we are going to look at some of the “magic” that occurs on the command line when we press the Enter key. While we will examine several interesting and complex features of the shell, we will do it with just one new command. 
echo – Display a line of text 
# Expansion 
Each time we type a command and press the Enter key, bash performs several substitutions upon the text before it carries out our command. We have seen a couple of cases of how a simple character sequence, for example \*, can have a lot of meaning to the shell. The process that makes this happen is called expansion. With expansion, we enter something and it is expanded into something else before the shell acts upon it. To demonstrate what we mean by this, let's take a look at the echo command. echo is a shell builtin that performs a very simple task. It prints its text arguments on standard output. 
[me@linuxbox $-]$1$ echo this is a test this is a test 
That's pretty straightforward. Any argument passed to echo gets displayed. Let's try another example. 
[me@linuxbox $-]$1$ echo \* 
Desktop Documents ls-output.txt Music Pictures Public Templates 
Videos 
So what just happened? Why didn't echo print $^{\star}?$ As we recall from our work with wildcards, the \* character means match any characters in a filename, but what we didn't see in our original discussion was how the shell does that. The simple answer is that the shell expands the \* into something else (in this instance, the names of the files in the current working directory) before the echo command is executed. When the Enter key is pressed, the shell automatically expands any qualifying characters on the command line before the command is carried out, so the echo command never saw the \*, only its expanded result. Knowing this, we can see that echo behaved as expected. 
# Pathname Expansion 
The mechanism by which wildcards work is called pathname expansion. If we try some of the techniques that we employed in earlier chapters, we will see that they are really expansions. Given a home directory that looks like this: 
[me@linuxbox ~]$ ls Desktop ls-output.txt Pictures Templates Documents Music Public Videos 
we could carry out the following expansions: 
[me@linuxbox ~]$ echo $D^{\star}$ Desktop Documents 
and this: 
[me@linuxbox ~]$ echo $\pmb{\star}_{\pmb{\mathsf{S}}}$ Documents Pictures Templates Videos 
or even this: 
[me@linuxbox ~]$ echo [[:upper:]]\* Desktop Documents Music Pictures Public Templates Videos 
and looking beyond our home directory, we could do this: 
[me@linuxbox ~]$ echo /usr/\*/share /usr/kerberos/share /usr/local/share 
# Pathname Expansion of Hidden Files 
As we know, filenames that begin with a period character are hidden. Pathname expansion also respects this behavior. An expansion such as the following does not reveal hidden files. 
echo \* 
It might appear at first glance that we could include hidden files in an expansion 
by starting the pattern with a leading period, like this: 
echo .\* 
It almost works. However, if we examine the results closely, we will see that the names . and .. will also appear in the results. Because these names refer to the current working directory and its parent directory, using this pattern will likely produce an incorrect result. We can see this if we try the following command: 
ls -d .\* | less 
To better perform pathname expansion in this situation, we have to employ a more specific pattern. 
echo .[!.]\* 
This pattern expands into every filename that begins with only one period followed by any other characters. This will work correctly with most hidden files (though it still won't include filenames with multiple leading periods). The ls command with the -A option (“almost all”) will provide a correct listing of hidden files. 
ls -A 
# Tilde Expansion 
As we may recall from our introduction to the cd command, the tilde character $(\sim)$ has a special meaning. When used at the beginning of a word, it expands into the name of the home directory of the named user or, if no user is named, the home directory of the current user. 
[me@linuxbox ~]$ echo ~ /home/me 
If user “bob” has an account, then it expands into this: 
[me@linuxbox ~]$ echo ~bob /home/bob 
# Arithmetic Expansion 
The shell allows arithmetic to be performed by expansion. This allows us to use the shell prompt as a calculator. 
<html><body><table><tr><td>[me@linuxbox ~]$ echo $((2 + 2)) 4</td></tr></table></body></html> 
Arithmetic expansion uses the following form: 
$((expression)) 
where expression is an arithmetic expression consisting of values and arithmetic operators. 
Arithmetic expansion supports only integers (whole numbers, no decimals) but can perform quite a number of different operations. Table 7-1 describes a few of the supported operators. 
Table 7-1: Arithmetic Operators 
<html><body><table><tr><td>Operator</td><td>Description</td></tr><tr><td>+</td><td>Addition</td></tr><tr><td></td><td>Subtraction</td></tr><tr><td>*</td><td>Multiplication</td></tr><tr><td>/</td><td>Division (but remember, since expansion supports only integer arithmetic, results are integers)</td></tr><tr><td>%</td><td>Modulo, which simply means “remainder"</td></tr><tr><td>**</td><td>Exponentiation</td></tr></table></body></html> 
Spaces are not significant in arithmetic expressions and expressions may be nested. For example, to multiply 5 squared by 3, we can use this: 
Single parentheses may be used to group multiple subexpressions. With this technique, we can rewrite the previous example and get the same result using a single expansion instead of two. 
[me@linuxbox ~]$ echo $(((5\*\*2) \* 3)) 75 
Here is an example using the division and remainder operators. Notice the effect of integer division. 
[me@linuxbox $-]$1$ echo Five divided by two equals $((5/2)) Five divided by two equals 2 
[me@linuxbox $-]$1$ echo with $$1(5\%2)$ ) left over. 
with 1 left over. 
Arithmetic expansion is covered in greater detail in Chapter 34. 
# Brace Expansion 
Perhaps the strangest expansion is called brace expansion. With it, we can create multiple text strings from a pattern containing braces. Here's an example: 
[me@linuxbox ~]$ echo Front-{A,B,C}-Back Front-A-Back Front-B-Back Front-C-Back 
Patterns to be brace expanded may contain a leading portion called a preamble and a trailing portion called a postscript. The brace expression itself may contain either a comma-separated list of strings or a range of integers or single characters. The pattern may not contain unquoted whitespace. Here is an example using a range of integers: 
[me@linuxbox $-]$1$ echo Number_{1..5} Number_1 Number_2 Number_3 Number_4 Number_5 
In bash version 4.0 and newer, integers may also be zero-padded like so: 
[me@linuxbox ~]$ echo {01..15} 
01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 
[me@linuxbox ~]$ echo {001..15} 
001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 
Here is a range of letters in reverse order: 
![](https://cdn-mineru.openxlab.org.cn/extract/e52c4087-794e-43f3-821f-0f8eb1fdf822/94cae717366543e491cdcc09a7387138afb43ca787c54fdef636113eed0a9c0d.jpg) 
Brace expansions may be nested. 
<html><body><table><tr><td>[me@linuxbox ~]$ echo a{A{1,2},B{3,4}}b aA1b aA2b aB3b aB4b</td></tr></table></body></html> 
So, what is this good for? The most common application is making lists of files or directories to be created. For example, if we were photographers and had a large collection of images that we wanted to organize into years and months, the first thing we might do is create a series of directories named in numeric “Year-Month” format. This way, the directory names would sort in chronological order. We could type out a complete list of directories, but that's a lot of work and it's error-prone. Instead, we could do this: 
<html><body><table><tr><td colspan="6">[me@linuxbox ~]$ mkdir Photos</td></tr><tr><td>[me@linuxbox ~]$ cd Photos</td><td colspan="5">[me@linuxbox Photos]$ mkdir {2007..2009}-{01..12}</td></tr><tr><td>[me@linuxbox Photos]$ ls</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2007-07</td><td></td><td></td><td>2008-07</td><td>2009-01</td><td></td></tr><tr><td>2007-01 2007-02</td><td>2008-01</td><td>2008-02</td><td>2008-08</td><td>2009-02</td><td>2009-07</td></tr><tr><td>2007-03</td><td>2007-08</td><td>2008-03</td><td>2008-09</td><td>2009-03</td><td>2009-08</td></tr><tr><td>2007-04</td><td>2007-09 2007-10</td><td>2008-04</td><td>2008-10</td><td>2009-04</td><td>2009-09</td></tr><tr><td>2007-05</td><td>2007-11</td><td>2008-05</td><td>2008-11</td><td>2009-05</td><td>2009-10 2009-11</td></tr><tr><td>2007-06</td><td>2007-12</td><td>2008-06</td><td>2008-12</td><td>2009-06</td><td>2009-12</td></tr></table></body></html> 
Pretty slick! 
# Parameter Expansion 
We're going to touch only briefly on parameter expansion in this chapter, but we'll be covering it extensively later. It's a feature that is more useful in shell scripts than directly on the command line. Many of its capabilities have to do with the system's ability to store small chunks of data and to give each chunk a name. Many such chunks, more properly called variables, are available for our examination. For example, the variable named USER contains our username. To invoke parameter expansion and reveal the contents of USER we would do this: 
[me@linuxbox ~]$ echo $USER me 
To see a list of available variables, try this: 
[me@linuxbox ~]$ printenv | less 
You may have noticed that with other types of expansion, if we mistype a pattern, the expansion will not take place, and the echo command will simply display the mistyped pattern. With parameter expansion, if we misspell the name of a variable, the expansion will still take place but will result in an empty string: 
[me@linuxbox ~]$ echo $SUER [me@linuxbox ~]$ 
# Command Substitution 
Command substitution allows us to use the output of a command as an expansion. 
[me@linuxbox $-]$1$ echo $$5$ (ls) 
Desktop Documents ls-output.txt Music Pictures Public Templates 
Videos 
One of my favorites goes something like this: 
[me@linuxbox ~]$ ls -l $(which cp) -rwxr-xr-x 1 root root 71516 2007-12-05 08:58 /bin/cp 
Here we passed the results of which cp as an argument to the ls command, thereby getting the listing of the cp program without having to know its full pathname. We are not limited to just simple commands. Entire pipelines can be used (only partial output is shown here): 
[me@linuxbox ~]$ file $(ls -d /usr/bin/\* | grep zip) 
/usr/bin/bunzip2: symbolic link to \`bzip2' 
/usr/bin/bzip2: ELF 32-bit LSB executable, Intel 80386, 
version 1 (SYSV), dynamically linked (uses shared libs), for 
GNU/Linux 2.6.9, stripped 
/usr/bin/bzip2recover: ELF 32-bit LSB executable, Intel 80386, 
version 1 (SYSV), dynamically linked (uses shared libs), for 
GNU/Linux 2.6.9, stripped 
/usr/bin/funzip: ELF 32-bit LSB executable, Intel 80386, 
version 1 (SYSV), dynamically linked (uses shared libs), for 
GNU/Linux 2.6.9, stripped 
/usr/bin/gpg-zip: Bourne shell script text executable 
/usr/bin/gunzip: symbolic link to \`../../bin/gunzip' 
/usr/bin/gzip: symbolic link to \`../../bin/gzip' 
/usr/bin/mzip: symbolic link to \`mtools' 
In this example, the results of the pipeline became the argument list of the file command. 
There is an alternate syntax for command substitution used by older shell programs that is also supported in bash. It uses backquotes instead of the dollar sign and parentheses. 
<html><body><table><tr><td>[me@linuxbox ~]$ ls -l “which cp -rwxr-xr-x 1 r00t r00t 71516 2007-12-05 08:58 /bin/cp</td></tr></table></body></html> 
# Quoting 
Now that we've seen how many ways the shell can perform expansions, it's time to learn how we can control it. Take for example the following: 
<html><body><table><tr><td>[me@linuxbox ~]$ echo this is a test this is a test</td></tr></table></body></html> 
or this one: 
[me@linuxbox $-]$1$ echo The total is $$100.00$ The total is 00.00 
In the first example, word-splitting by the shell removed extra whitespace from the echo command's list of arguments. In the second example, parameter expansion substituted an empty string for the value of $$1$ because it was an undefined variable. The shell provides a mechanism called quoting to selectively suppress unwanted expansions. 
# Double Quotes 
The first type of quoting we will look at is double quotes. If we place text inside double quotes, all the special characters used by the shell lose their special meaning and are treated as ordinary characters. The exceptions are $$1$ , \ (backslash), and \` (back-quote). This means that word-splitting, pathname expansion, tilde expansion, and brace expansion are suppressed, but parameter expansion, arithmetic expansion, and command substitution are still carried out. Using double quotes, we can cope with filenames containing embedded spaces. Say we were the unfortunate victim of a file called two words.txt. If we tried to use this on the command line, word-splitting would cause this to be treated as two separate arguments rather than the desired single argument. 
[me@linuxbox ~]$ ls -l two words.txt ls: cannot access two: No such file or directory ls: cannot access words.txt: No such file or directory 
By using double quotes, we stop the word-splitting and get the desired result; further, we can even repair the damage. 
[me@linuxbox $-]$1$ ls -l "two words.txt" $-r w-r w-r\textrm{-}1$ me me 18 2016-02-20 13:03 two words.txt [me@linuxbox $-]$1$ mv "two words.txt" two_words.txt 
There! Now we don't have to keep typing those pesky double quotes. 
Remember, parameter expansion, arithmetic expansion, and command substitution still take place within double quotes. 
[me@linuxbox ~]$ echo "$USER $((2+2)) $(df -h)" me 4 Filesystem Size Used Avail Use% Mounted on tmpfs 1.6G 2.0M 1.6G $1\%$ /run 
<html><body><table><tr><td>/dev/sda2</td><td>94G</td><td>19G</td><td>71G</td><td>21%／</td><td></td></tr><tr><td>tmpfs</td><td>7.8G</td><td>0</td><td>7.8G</td><td>0% /dev/shm</td><td></td></tr><tr><td>tmpfs</td><td>5.0M</td><td>4.0K</td><td>5.0M</td><td></td><td>1% /run/lock</td></tr><tr><td>/dev/sda1</td><td>975M</td><td>6.1M</td><td>969M</td><td></td><td>1% /boot/efi</td></tr><tr><td>/dev/sdb1</td><td>907G</td><td>574G</td><td>287G</td><td>67% /home</td><td></td></tr><tr><td>tmpfs</td><td>1.6G</td><td>1.8M</td><td>1.6G</td><td></td><td>1% /run/user/1000</td></tr></table></body></html> 
We should take a moment to look at the effect of double quotes on command substitution. First let's look a little deeper at how word splitting works. In our earlier example, we saw how word-splitting appears to remove extra spaces in our text. 
<html><body><table><tr><td>[me@linuxbox ~]$ echo this is a this is a test</td><td>test</td></tr></table></body></html> 
By default, word-splitting looks for the presence of spaces, tabs, and newlines (linefeed characters) and treats them as delimiters between words. This means unquoted spaces, tabs, and newlines are not considered to be part of the text. They serve only as separators. Since they separate the words into different arguments, our example command line contains a command followed by four distinct arguments. If we add double quotes: 
<html><body><table><tr><td>[me@linuxbox ~]$ echo "this is a this is a test</td><td></td><td>test"</td></tr></table></body></html> 
word-splitting is suppressed and the embedded spaces are not treated as delimiters; rather they become part of the argument. Once the double quotes are added, our command line contains a command followed by a single argument. 
The fact that newlines are considered delimiters by the word-splitting mechanism causes an interesting, albeit subtle, effect on command substitution. Consider the following: 
<html><body><table><tr><td>[me@linuxbox ~]$ echo $(df -h) Filesystem Size Used Avail Use% Mounted on tmpfs 1.6G 2.0M 1.6G 1% /run /dev/sda2 94G 19G 71G 21% / tmpfs 7.8G 0 7.8G 0% /dev/shm tmpfs</td></tr><tr><td>5.0M 4.0K 5.0M 1% /run/lock /dev/sda1 975M 6.1M 969M 1% /b0ot/efi</td></tr><tr><td>/dev/sdb1 907G 574G 287G 67% /h0me tmpfs 1.6G 1.8M 1.6G 1% /run/user/1000</td></tr><tr><td>[me@linuxbox ~]$ echo "$(df -h)"</td></tr><tr><td>Filesystem Size Used Avail Use% Mounted on tmpfs 1.6G 2.0M 1.6G 1% /run</td></tr></table></body></html> 
<html><body><table><tr><td>/dev/sda2</td><td>94G</td><td>19G</td><td>71G</td><td>21% ／</td><td></td></tr><tr><td>tmpfs</td><td>7.8G</td><td>0</td><td>7.8G</td><td>0% /dev/shm</td><td></td></tr><tr><td>tmpfs</td><td>5.0M</td><td>4.0K</td><td>5.0M</td><td></td><td>1% /run/lock</td></tr><tr><td>/dev/sda1</td><td>975M</td><td>6.1M</td><td>969M</td><td></td><td>1% /boot/efi</td></tr><tr><td>/dev/sdb1</td><td>907G</td><td>574G</td><td>287G</td><td>67% /home</td><td></td></tr><tr><td>tmpfs</td><td>1.6G</td><td>1.8M</td><td>1.6G</td><td></td><td>1% /run/user/1000</td></tr></table></body></html> 
In the first instance, the unquoted command substitution resulted in a command line containing 49 arguments. In the second, it resulted in a command line with one argument that includes the embedded spaces and newlines. 
# Single Quotes 
If we need to suppress all expansions, we use single quotes. Here is a comparison of unquoted, double quotes, and single quotes: 
$\mathsf{\Lambda}[\mathsf{m e}\oplus\mathsf{l i n u}\times\mathsf{b o}\times\mathsf{\Lambda}\sim]\Phi$ echo text $-/\star$ .txt {a,b} $$5$ (echo foo) $$1(2+2)$ ) $USER text /home/me/ls-output.txt a b foo 4 me 
[me@linuxbox $-]$1$ echo "text $-/\star$ .txt {a,b} $$5$ (echo foo) $$(2+2)$ ) $USER" text $-/\star$ .txt $\{\mathsf{a},\mathsf{b}\}$ foo 4 me 
[me@linuxbox $-]$1$ echo 'text $-/\star$ .txt {a,b} $$5$ (echo foo) $$(2+2)$ ) $USER' text $-/\star$ .txt {a,b} $$1$ (echo foo) $(( $2+2)$ ) $USER 
As we can see, with each succeeding level of quoting, more and more of the expansions are suppressed. 
# Escaping Characters 
Sometimes we want to quote only a single character. To do this, we can precede a character with a backslash, which in this context is called the escape character. Often this is done inside double quotes to selectively prevent an expansion. 
[me@linuxbox $-]$1$ echo "The balance for user $USER is: \$5.00" The balance for user me is: $$5.00$ 
It is also common to use escaping to eliminate the special meaning of a character in a filename. For example, it is possible to use characters in filenames that normally have special meaning to the shell. These would include $$1,2$ , spaces, and others. To include a special character in a filename we can do this: 
# [me@linuxbox ~]$ mv bad\&filename good_filename 
To allow a backslash character to appear, escape it by typing \\. Note that within single quotes, the backslash loses its special meaning and is treated as an ordinary character. 
Another use of the backslash escape is suppressing aliases. For example, assuming the ls command is aliased to $\begin{array}{r}{\mathsf{l s}={}^{\prime}\mathsf{l s}\mathsf{\Pi}-\mathsf{c o l o r}=\mathsf{a u t o}^{\prime}}\end{array}$ , the default on many Linux distributions, we can precede the command with a backslash and the alias will be ignored and the ls command will be executed without the color option. 
# Backslash Escape Sequences 
In addition to its role as the escape character, the backslash is also used as part of a notation to represent certain special characters called control codes. The first 32 characters in the ASCII coding scheme are used to transmit commands to teletype-like devices. Some of these codes are familiar (tab, backspace, linefeed, and carriage return), while others are not (null, end-of-transmission, and acknowledge). 
<html><body><table><tr><td>Escape Sequence</td><td>Meaning</td></tr><tr><td>\a</td><td>Bell (an alert that causes the computer to beep)</td></tr><tr><td>\b \n</td><td>Backspace Newline. On Unix-like systems, this</td></tr><tr><td></td><td>produces a linefeed.</td></tr><tr><td>\r \t</td><td>Carriage return</td></tr><tr><td></td><td>Tab</td></tr></table></body></html> 
The table above lists some of the common backslash escape sequences. The idea behind this representation using the backslash originated in the C programming language and has been adopted by many others, including the shell. 
Adding the -e option to echo will enable interpretation of escape sequences. You may also place them inside $$1$ '. Here, using the sleep command, a simple program that just waits for the specified number of seconds and then exits, we can create a primitive countdown timer: 
sleep 10; echo -e "Time's up\a" 
We could also do this: sleep 10; echo "Time's up" $'\a' 
# Summing Up 
As we move forward with using the shell, we will find that expansions and quoting will be used with increasing frequency, so it makes sense to get a good understanding of the way they work. In fact, it could be argued that they are the most important subjects to learn about the shell. Without a proper understanding of expansion, the shell will always be a source of mystery and confusion, with much of its potential power wasted. 
# Further Reading 
The bash man page has major sections on both expansion and quoting which cover these topics in a more formal manner. 
The Bash Reference Manual also contains chapters on expansion and quoting: http://www.gnu.org/software/bash/manual/bashref.html 
# 8 – Advanced Keyboard Tricks 
I often kiddingly describe Unix as “the operating system for people who like to type.” Of course, the fact that it even has a command line is a testament to that. But command line users don't like to type that much. Why else would so many commands have such short names like cp, ls, mv, and rm? In fact, one of the most cherished goals of the command line is laziness; doing the most work with the fewest number of keystrokes. Another goal is never having to lift our fingers from the keyboard and reach for the mouse. In this chapter, we will look at bash features that make keyboard use faster and more efficient. 
The following commands will make an appearance: 
clear – Clear the screen history – Display the contents of the history list 
# Command Line Editing 
bash uses a library (a shared collection of routines that different programs can use) called Readline to implement command line editing. We have already seen some of this. We know, for example, that the arrow keys move the cursor, but there are many more features. Think of these as additional tools that we can employ in our work. It’s not important to learn all of them, but many of them are very useful. Pick and choose as desired. 
Note: Some of the key sequences below (particularly those that use the Alt key) may be intercepted by the GUI for other functions. All of the key sequences should work properly when using a virtual console. 
# Cursor Movement 
The following table lists the keys used to move the cursor: 
Table 8-1: Cursor Movement Commands 
<html><body><table><tr><td> Key</td><td>Action</td></tr></table></body></html> 
<html><body><table><tr><td>Ctrl-a</td><td>Move cursor to the beginning of the line.</td></tr><tr><td>Ctrl-e</td><td>Move cursor to the end of the line.</td></tr><tr><td>ctrl-f</td><td> Move cursor forward one character; same as the right arrow key.</td></tr><tr><td>Ctrl-b</td><td> Move cursor backward one character; same as the left arrow key.</td></tr><tr><td>Alt-f</td><td>Move cursor forward one word.</td></tr><tr><td>Alt-b</td><td>Move cursor backward one word.</td></tr><tr><td>Ctrl-l</td><td>Clear the screen and move the cursor to the top-left corner. The clear command does the same thing.</td></tr></table></body></html> 
# Modifying Text 
Since we might make a mistake when composing a command, we need a way to correct them efficiently. Table 8-2 describes keyboard commands that are used to edit characters on the command line. 
Table 8-2: Text Editing Commands 
<html><body><table><tr><td>Key</td><td>Action</td></tr><tr><td>Ctrl-d</td><td>Delete the character at the cursor location.</td></tr><tr><td>Ctrl-t</td><td>Transpose (exchange) the character at the cursor location with the one preceding it.</td></tr><tr><td>Alt-t</td><td>Transpose the word at the cursor location with the one preceding it</td></tr><tr><td>Alt-l</td><td>Convert the characters from the cursor location to the end of the word to lowercase.</td></tr><tr><td>Alt-u</td><td>Convert the characters from the cursor location to the end of the word to uppercase.</td></tr></table></body></html> 
# Cutting and Pasting (Killing and Yanking) Text 
The Readline documentation uses the terms killing and yanking to refer to what we would commonly call cutting and pasting. Items that are cut are stored in a buffer (a temporary storage area in memory) called the kill-ring. 
Table 8-3: Cut and Paste Commands 
<html><body><table><tr><td>Key</td><td>Action</td></tr><tr><td>Ctrl-k</td><td>Kill text from the cursor location to the end of line.</td></tr><tr><td>Ctrl-u</td><td>Kill text from the cursor location to the beginning of the line.</td></tr><tr><td>Alt-d</td><td>Kill text from the cursor location to the end of the current word.</td></tr><tr><td>Alt- Backspace</td><td>Kill text from the cursor location to the beginning of the current word. If the cursor is at the beginning of a word, kill the previous word.</td></tr><tr><td>ctrl-y</td><td>Yank text from the kill-ring and insert it at the cursor location.</td></tr></table></body></html> 
# The Meta Key 
If you venture into the Readline documentation, which can be found in the “READLINE” section of the bash man page, you will encounter the term meta key. On modern keyboards this maps to the Alt key but it wasn't always so. 
Back in the dim times (before PCs but after Unix), not everybody had their own computer. What they might have had was a device called a terminal. A terminal was a communication device that featured a text display screen and a keyboard and just enough electronics inside to display text characters and move the cursor around. It was attached (usually by serial cable) to a larger computer or the communication network of a larger computer. There were many different brands of terminals, and they all had different keyboards and display feature sets. Since they all tended to at least understand ASCII, software developers wanting portable applications wrote to the lowest common denominator. Unix systems have an elaborate way of dealing with terminals and their different display features. Since the developers of Readline could not be sure of the presence of a dedicated extra control key, they invented one and called it meta. While the Alt key serves as the meta key on modern keyboards, you can also press and release the Esc key to get the same effect as holding down the Alt key if you're using a terminal (which you can still do in Linux!). 
# Completion 
Another way that the shell can help us is through a mechanism called completion. Completion occurs when we press the tab key while typing a command. Let's see how this works. Given a home directory that looks like this: 
[me@linuxbox ~]$ ls 
Desktop ls-output.txt Pictures Templates Videos 
Documents Music Public 
Try typing the following but don't press the Enter key: 
<html><body><table><tr><td>[me@linuxbox ~]$ ls l</td></tr></table></body></html> 
Now press the Tab key. 
[me@linuxbox ~]$ ls ls-output.txt 
See how the shell completed the line for us? Let's try another one. Again, don't press Enter. 
<html><body><table><tr><td>[me@linuxbox ~]$ ls D</td></tr></table></body></html> 
Press Tab. 
While this example shows completion of pathnames, which is its most common use, completion will also work on variables (if the beginning of the word is a $$1$ ), user names (if the word begins with ~), commands (if the word is the first word on the line) and hostnames (if the beginning of the word is $@$ ). Hostname completion works only for hostnames listed in /etc/hosts. 
There are a number of control and meta key sequences that are associated with completion, as listed in Table 8-4. 
Table 8-4: Completion Commands 
<html><body><table><tr><td>Key</td><td>Action</td></tr><tr><td>Alt-?</td><td>Display a list of possible completions. On most systems you can also do this by pressing the Tab key a second time, which is much easier.</td></tr><tr><td>Alt-*</td><td>Insert all possible completions. This is useful when you want to use more than one possible match.</td></tr></table></body></html> 
There are quite a few more that are rather obscure. A list appears in the bash man page under “READLINE”. 
# Programmable Completion 
Recent versions of bash have a facility called programmable completion. Programmable completion allows you (or more likely, your distribution provider) to add additional completion rules. Usually this is done to add support for specific applications. For example, it is possible to add completions for the option list of a command or match particular file types that an application supports. Ubuntu has a fairly large set defined by default. Programmable completion is implemented by shell functions, a kind of mini shell script that we will cover in later chapters. If you are curious, try the following: 
# set | less 
and see if you can find them. Not all distributions include them by default. 
# Using History 
As we discovered in Chapter 1, bash maintains a history of commands that have been entered. This list of commands is kept in our home directory in a file called 
.bash_history. The history facility is a useful resource for reducing the amount of typing we have to do, especially when combined with command line editing. 
# Searching History 
At any time, we can view the contents of the history list by doing the following: 
[me@linuxbox $-]$1$ history | less 
By default, most modern Linux distributions configure bash to store the last 1000 commands we have entered. We will see how to adjust this value in Chapter 11. Let's say we want to find the commands we used to list /usr/bin. This is one way we could do this: 
# [me@linuxbox ~]$ history | grep /usr/bin 
And let's say that among our results we got a line containing an interesting command like this: 
The 88 is the line number of the command in the history list. We could use this immediately using another type of expansion called history expansion. To use our discovered line, we could do this: 
[me@linuxbox ~]$ !88 
bash will expand !88 into the contents of the 88th line in the history list. There are other forms of history expansion that we will cover in the next section. 
bash also provides the ability to search the history list incrementally. This means we can tell bash to search the history list as we enter characters, with each additional character further refining our search. To start incremental search press Ctrl-r followed by the text we are looking for. When we find it, we can either press Enter to execute the command or press Ctrl-j to copy the line from the history list to the current command line. To find the next occurrence of the text (moving “up” the history list), press Ctrl-r again. To quit searching, press either Ctrl-g or Ctrl-c. Here we see it in action: 
First press Ctrl-r. 
The prompt changes to indicate that we are performing a reverse incremental search. It is “reverse” because we are searching from “now” to some time in the past. Next, we start typing our search text. In this example, /usr/bin: 
(reverse-i-search)\`/usr/bin': ls -l /usr/bin $>$ ls-output.txt 
Immediately, the search returns our result. With our result, we can execute the command by pressing Enter, or we can copy the command to our current command line for further editing by pressing Ctrl-j. Let's copy it. Press Ctrl-j. 
# [me@linuxbox ~]$ ls -l /usr/bin $>$ ls-output.txt 
Our shell prompt returns, and our command line is loaded and ready for action! The Table 8-5 lists some of the keystrokes used to manipulate the history list. 
Table 8-5: History Commands 
<html><body><table><tr><td>Key</td><td>Action</td></tr><tr><td>Ctrl-p</td><td>Move to the previous history entry. This is the same action as the up arrow.</td></tr><tr><td>Ctrl-n</td><td>Move to the next history entry. This is the same action as the down arrow.</td></tr><tr><td>Alt-<</td><td> Move to the beginning (top) of the history list.</td></tr><tr><td>Alt-></td><td>Move to the end (bottom) of the history list, i.e., the current command line.</td></tr><tr><td>ctrl-r</td><td>Reverse incremental search. This searches incrementally from the current command line up the history list.</td></tr><tr><td>Alt-p</td><td>Reverse search, nonincremental. With this key, type in the search string and press enter before the search is performed.</td></tr><tr><td>Alt-n</td><td>Forward search, nonincremental.</td></tr><tr><td>Ctrl-o</td><td>Execute the current item in the history list and advance to the next</td></tr></table></body></html> 
one. This is handy if we are trying to re-execute a sequence of commands in the history list. 
# History Expansion 
The shell offers a specialized type of expansion for items in the history list by using the ! character. We have already seen how the exclamation point can be followed by a number to insert an entry from the history list. There are a number of other expansion features, as described in Table 8-6. 
Table 8-6: History Expansion Commands 
<html><body><table><tr><td>Sequence</td><td>Action</td></tr><tr><td>!!</td><td>Repeat the last command. It is probably easier to press up arrow and enter.</td></tr><tr><td>!number</td><td>Repeat history list item number.</td></tr><tr><td>!string</td><td>Repeat last history list item starting with string.</td></tr><tr><td>!?string</td><td>Repeat last history list item containing string.</td></tr></table></body></html> 
Use caution with the !string and !?string forms unless youyou are absolutely sure of the contents of the history list items. We can mitigate this problem somewhat by appending “:p” to our expansion. This tells the shell to print the result of the expansion and place it into the command history. Here’s an example: 
<html><body><table><tr><td>[me@linuxbox ~]$ !ls:p</td></tr><tr><td></td></tr><tr><td>ls -l /usr/bin > ls-output.txt</td></tr></table></body></html> 
Now that the command has been recalled and placed as the most recent item on the history list, we can execute it with Up-Arrow and Return or !! and Return. 
By the way, history expansions such as !! are not recorded in the history list but their results are. 
Many more features are available in the history expansion mechanism, but this subject is already too arcane and our heads may explode if we continue. The HISTORY EXPANSION section of the bash man page goes into all the gory details. Feel free to explore! 
# script 
In addition to the command history feature in bash, most Linux distributions include a program called script that can be used to record an entire shell session and store it in a file. The basic syntax of the command is as follows: 
script $[\mathop{f i l e}]$ 
where file is the name of the file used for storing the recording. If no file is specified, the file typescript is used. See the script man page for a complete list of the program’s options and features. 
# Summing Up 
In this chapter we covered some of the keyboard tricks that the shell provides to help hardcore typists reduce their workloads. As time goes by and we become more involved with the command line, we can refer back to this chapter to pick up more of these tricks. For now, consider them optional and potentially helpful. 
# Further Reading 
The Wikipedia has a good article on computer terminals: http://en.wikipedia.org/wiki/Computer_terminal 
# 9 – Permissions 
Operating systems in the Unix tradition differ from those in the MS-DOS tradition in that they are not only multitasking systems, but also multi-user systems. 
What exactly does this mean? It means that more than one person can be using the computer at the same time. While a typical computer will likely have only one keyboard and monitor, it can still be used by more than one user. For example, if a computer is attached to a network or the Internet, remote users can log in via ssh (secure shell) and operate the computer. In fact, remote users can execute graphical applications and have the graphical output appear on a remote display. 
The multi-user capability of Linux is not a recent "innovation," but rather a feature that is deeply embedded into the design of the operating system. Considering the environment in which Unix was created, this makes perfect sense. Years ago, before computers were "personal," they were large, expensive, and centralized. A typical university computer system, for example, consisted of a large central computer located in one building and terminals that were located throughout the campus, each connected to the large central computer. The computer would support many users at the same time. 
To make this practical, a method had to be devised to protect the users from each other. After all, the actions of one user could not be allowed to crash the computer, nor could one user interfere with the files belonging to another user. 
In this chapter we will look at this essential part of system security and introduce the following commands: 
id – Display user identity chmod – Change a file's mode umask – Set the default file permissions su – Run a shell as another user sudo – Execute a command as another user chown – Change a file's owner chgrp – Change a file's group ownership addgroup – Add a user or a group to the system usermod – Modify a user account passwd – Change a user's password 
# Users, Group Members, and Everybody Else 
When we were exploring the system in Chapter 3, we may have encountered a problem when trying to examine a file such as /etc/shadow: 
$\mathsf{\Lambda}[\mathsf{m e}\oplus\mathsf{l i n u}\times\mathsf{b o}\times\mathsf{\Lambda}\sim]\Phi$ file /etc/shadow /etc/shadow: regular file, no read permission [me@linuxbox ~]$ less /etc/shadow /etc/shadow: Permission denied 
The reason for this error message is that, as regular users, we do not have permission to read this file. 
In the Unix security model, a user may own files and directories. When a user owns a file or directory, the user has control over its access. Users can, in turn, belong to a group consisting of one or more users who are given access to files and directories by their owners. In addition to granting access to a group, an owner may also grant some set of access rights to everybody, which are called others (sometimes referred to as the world). To find information about our identity, we use the id command. 
[me@linuxbox ~]$ id uid $\mathtt{\Gamma}=5\Theta\Theta$ (me) gid $=500$ (me) groups $\scriptstyle\mathbf{\lambda}=5\Theta\Theta$ (me) 
Let's look at the output. When user accounts are created, users are assigned a number called a user ID (uid) which is then, for the sake of the humans, mapped to a username. The user is assigned a primary group ID (gid) and may belong to additional groups. The above example is from a Fedora system. On other systems, such as Ubuntu, the output may look a little different: 
[me@linuxbox ~]$ id 
uid $\mathtt{\Gamma}=\mathtt{1000}$ (me) $9\dot{1}0=1000$ (me) 
group $\mathord{\mathsf{s}}=4$ (adm),20(dialout),24(cdrom),25(floppy),29(audio),30(dip),44(v 
ideo),46(plugdev),108(lpadmin),114(admin),1000(me) 
As we can see, the uid and gid numbers are different. This is simply because Fedora starts its numbering of regular user accounts at 500, while Ubuntu starts at 1000. We can also see that the Ubuntu user belongs to a lot more groups. This has to do with the way Ubuntu manages privileges for system devices and services. 
So where does this information come from? Like so many things in Linux, it comes from a couple of text files. User accounts are defined in the /etc/passwd file and groups are defined in the /etc/group file. When user accounts and groups are created, these files are modified along with /etc/shadow which holds information about the user's password. For each user account, the /etc/passwd file defines the user (login) name, uid, gid, user’s real name, home directory, and login shell. If we examine the contents of /etc/passwd and /etc/group, we notice that besides the regular user accounts, there are accounts for the superuser (always uid 0) and various other system users. 
In the next chapter, when we cover processes, we will see that some of these other “users” are, in fact, quite busy. 
While many Unix-like systems assign regular users to a common group such as “users”, modern Linux practice is to create a unique, single-member group with the same name as the user. This makes certain types of permission assignment easier. 
# Reading, Writing, and Executing 
Access rights to files and directories are defined in terms of read access, write access, and execution access. If we look at the output of the ls command, we can get some clue as to how this is implemented: 
<html><body><table><tr><td>[me@linuxbox ~]$ > foo.txt</td></tr><tr><td></td></tr><tr><td>[me@linuxbox ~]$ ls -l foo.txt -rw-rw-r-- 1 me me 0 2016-03-06 14:52 f00.txt</td></tr></table></body></html> 
The first 10 characters of the listing are the file attributes. The first of these characters is the file type. Table 9-1 describes the file types we are most likely to see (there are other, less common types too): 
Table 9-1: File Types 
<html><body><table><tr><td>Attribute</td><td>File Type</td></tr><tr><td>-</td><td>A regular file.</td></tr><tr><td>d</td><td>A directory.</td></tr><tr><td>l</td><td>A symbolic link. Notice that with symbolic links, the remaining file attributes are always “rwxrwxrwx" and are dummy values. The real</td></tr></table></body></html> 
<html><body><table><tr><td>file attributes are those of the file the symbolic link points to.</td></tr><tr><td>C A character special file. This file type refers to a device that handles data as a stream of bytes, such as a terminal or /dev/null.</td></tr><tr><td>b A block special file. This file type refers to a device that handles data in blocks, such as a hard disk or DVD drive.</td></tr></table></body></html> 
The remaining nine characters of the file attributes, called the file mode, represent the read, write, and execute permissions for the file's owner, the file's group owner, and everybody else. 
<html><body><table><tr><td>User</td><td>Group</td><td>Other</td></tr><tr><td>rwx</td><td>rwx</td><td>rwx</td></tr></table></body></html> 
Table 9-2 describes the effect the ${\mathsf{r}},{\mathsf{w}},$ and $\mathsf{x}$ mode attributes have on files and directories: 
Table 9-2: Permission Attributes 
<html><body><table><tr><td>Attribute</td><td>Files</td><td>Directories</td></tr><tr><td>r</td><td>Allows a file to be opened and read.</td><td>Allows a directory's contents to be listed, but no file information is available unless the execute attribute is also set.</td></tr><tr><td>W</td><td>Allows a file to be written to or truncated, however this attribute does not allow files to be renamed or deleted. The ability to delete or rename files is determined by directory attributes.</td><td>Allows files within a directory to be created, deleted, and renamed if the execute attribute is also set.</td></tr><tr><td>×</td><td>Allows a file to be treated as a program and executed. Program files written in scripting languages must also be set as readable to be executed.</td><td>Allows a directory to be entered (i.e., cd directory) and directory metadata (i.e, ls - l directory) to be accessed. File operations such cp, rm, and mv require this access to the</td></tr></table></body></html> 
<html><body><table><tr><td>directory.</td></tr></table></body></html> 
Table 9-3 provides some examples of file attribute settings: 
Table 9-3: Permission Attribute Examples 
<html><body><table><tr><td>File Attributes</td><td>Meaning</td></tr><tr><td>-rwx-</td><td>A regular file that is readable, writable, and executable by the file's owner. No one else has any access.</td></tr><tr><td>-rw-</td><td>A regular file that is readable and writable by the file's owner. No one else has any access.</td></tr><tr><td>-rw-r--r--</td><td>A regular file that is readable and writable by the file's owner. Members of the file's owner group may read the file. The file is readable by others.</td></tr><tr><td>- rwxr-xr-x</td><td>A regular file that is readable, writable, and executable by the file's owner. The file may be read and executed by everybody else.</td></tr><tr><td>- rw-rw--</td><td>A regular file that is readable and writable by the file's owner and members of the file's group owner only.</td></tr><tr><td>lrwxrwxrwx</td><td>A symbolic link. All symbolic links have “dummy" permissions. The real permissions are kept with the actual file pointed to by the symbolic link.</td></tr><tr><td>drwxrwx--</td><td>A directory. The owner and the members of the owner group may enter the directory and create, rename and remove files within the directory.</td></tr><tr><td>drwxr-x---</td><td>A directory. The owner may enter the directory and create, rename, and delete files within the directory. Members of the owner group may enter the directory but cannot create, delete, or rename files.</td></tr></table></body></html> 
# chmod – Change File Mode 
To change the mode (permissions) of a file or directory, the chmod command is used. Be aware that only the file’s owner or the superuser can change the mode of a file or directory. chmod supports two distinct ways of specifying mode changes: octal number representation, or symbolic representation. We will cover octal number representation first. 
# What the Heck is Octal? 
Octal (base 8), and its cousin, hexadecimal (base 16) are number systems often used to express numbers on computers. We humans, owing to the fact that we (or at least most of us) were born with 10 fingers, count using a base 10 number system. Computers, on the other hand, were born with only one finger and thus do all their counting in binary (base 2). Their number system has only two numerals, 0 and 1. So, in binary, counting looks like this: 
0, 1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010, 1011... 
In octal, counting is done with the numerals zero through seven, like so: 
0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21... 
Hexadecimal counting uses the numerals zero through nine plus the letters “A” through “F”: 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F, 10, 11, 12, 13... 
While we can see the sense in binary (since computers have only one finger), what are octal and hexadecimal good for? The answer has to do with human convenience. Many times, small portions of data are represented on computers as bit patterns. Take for example an RGB color. On most computer displays, each pixel is composed of three color components: eight bits of red, eight bits of green, and eight bits of blue. A lovely medium blue would be a 24 digit number: 
# 010000110110111111001101 
How would you like to read and write those kinds of numbers all day? I didn't think so. Here's where another number system would help. Each digit in a hexadecimal number represents four digits in binary. In octal, each digit represents three binary digits. So our 24 digit medium blue could be condensed to a six-digit hexadecimal number: 
# 436FCD 
Since the digits in the hexadecimal number “line up” with the bits in the binary number, we can see that the red component of our color is 43, the green 6F, and the blue CD. 
These days, hexadecimal notation (often referred to as “hex”) is more common than octal, but as we will soon see, octal's ability to express three bits of binary will be very useful... 
With octal notation, we use octal numbers to set the pattern of desired permissions. Since each digit in an octal number represents three binary digits, this maps nicely to the scheme used to store the file mode. Table 9-4 shows what we mean. 
Table 9-4: File Modes in Binary and Octal 
<html><body><table><tr><td>Octal</td><td>Binary</td><td>File Mode</td></tr><tr><td>0</td><td>000</td></tr><tr><td>1</td><td>001 --X</td></tr><tr><td>2 010</td><td>-W-</td></tr><tr><td>3 011</td><td>-WX</td></tr><tr><td>4 100</td><td>r--</td></tr><tr><td>5 101</td><td>r-x</td></tr><tr><td>6 110</td><td>rw-</td></tr><tr><td>7 111</td><td>rwx</td></tr></table></body></html> 
By using three octal digits, we can set the file mode for the owner, group owner, and world. 
[me@linuxbox ~]$ > foo.txt 
$\mathsf{\Lambda}[\mathsf{m e}\ @\mathsf{l i n u}\times\mathsf{b o}\times\mathsf{\Lambda}\sim]\Phi$ ls -l foo.txt 
$-r w-r w-r\textrm{-}1$ me me 0 2016-03-06 14:52 foo.txt 
[me@linuxbox ~]$ chmod 600 foo.txt 
[me@linuxbox $-]$1$ ls -l foo.txt 
-rw-- -- 1 me me 0 2016-03-06 14:52 foo.txt 
By passing the argument $"600"$ , we were able to set the permissions of the owner to read and write while removing all permissions from the group owner and others. Though remembering the octal to binary mapping may seem inconvenient, we will usually have only to use a few common ones: 7 $\mathsf{^{\prime}}(\mathsf{r w}\mathsf{\times}),6(\mathsf{r w}-),5(\mathsf{r}-\mathsf{\times}),4(\mathsf{r}--),$ , and 0 (---). 
chmod also supports a symbolic notation for specifying file modes. Symbolic notation is divided into three parts. 
Who the change will affect Which operation will be performed What permission will be set. 
To specify who is affected, a combination of the characters “u”, “g”, “o”, and “a” is used as shown in Table 9-5. 
Table 9-5: chmod Symbolic Notation 
<html><body><table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td>u</td><td>Short for “user" i.e. the file or directory's owner.</td></tr><tr><td>ｇ</td><td>Group owner.</td></tr><tr><td>0</td><td>Short for others.</td></tr><tr><td>a</td><td>Short for “all." This is the combination of “u", “g", and “o".</td></tr></table></body></html> 
If no character is specified, “all” will be assumed. The operation may be a $^{66}+^{99}$ indicating that a permission is to be added, a “-” indicating that a permission is to be taken away, or a $"="$ indicating that only the specified permissions are to be applied and that all others are to be removed. 
Permissions are specified with the “r”, “w”, and “x” characters. Table 9-6 provides some examples of symbolic notation: 
Table 9-6: chmod Symbolic Notation Examples 
<html><body><table><tr><td>Notation</td><td>Meaning</td></tr><tr><td>u+x</td><td>Add execute permission for the owner.</td></tr><tr><td>u-x</td><td>Remove execute permission from the owner.</td></tr><tr><td>+X</td><td>Add execute permission for the user, group, and others. This is equivalent to a+x.</td></tr><tr><td>o-rw</td><td>Remove the read and write permissions from anyone besides the user and group owner.</td></tr><tr><td>go=rw</td><td>Set the group owner and anyone besides the user to have read and write permission. If either the group owner or others previously had execute permission, it is removed.</td></tr><tr><td>u+x,go=rx</td><td>Add execute permission for the user and set the permissions for the group and others to read and execute. Multiple specifications may be separated by commas.</td></tr></table></body></html> 
Some people prefer to use octal notation, and some folks really like the symbolic. Symbolic notation does offer the advantage of allowing us to set a single attribute without disturbing any of the others. 
Take a look at the chmod man page for more details and a list of options. A word of caution regarding the “--recursive” option: it acts on both files and directories, so it's not as useful as we would hope since we rarely want files and directories to have the same permissions. 
# Setting File Mode with the GUI 
Now that we have seen how the permissions on files and directories are set, we can better understand the permission dialogs in the GUI. In both Files (GNOME) and Dolphin (KDE), right-clicking a file or directory icon will expose a properties dialog. Here is an example from GNOME: 
![](https://cdn-mineru.openxlab.org.cn/extract/e52c4087-794e-43f3-821f-0f8eb1fdf822/7ec463fe32c278ccef035e08e5d6a50c0807b13cdb7d5894ce3aa057151f3b8a.jpg) 
Figure 2: GNOME file permissions dialog 
Here we can see the settings for the owner, group, and others. 
# umask – Set Default Permissions 
The umask command controls the default permissions given to a file when it is created. It uses octal notation to express a mask of bits to be removed from a file's mode attributes. Let's take a look. 
[me@linuxbox ~]$ rm -f foo.txt 
[me@linuxbox ~]$ umask 
0002 
[me@linuxbox ~]$ > foo.txt 
[me@linuxbox ~]$ ls -l foo.txt 
-rw-rw-r-- 1 me me 0 2025-03-06 14:53 foo.txt 
We first removed any old copy of foo.txt to make sure we were starting fresh. Next, we ran the umask command without an argument to see the current value. It responded with the value 0002 (the value 0022 is another common default value), which is the octal representation of our mask. We next create a new instance of the file foo.txt and observe its permissions. 
We can see that both the user and group get read and write permission, while everyone else only gets read permission. The reason that world does not have write permission is because of the value of the mask. Let's repeat our example, this time setting the mask ourselves. 
<html><body><table><tr><td>[me@linuxbox ~]$ rm foo.txt</td><td></td></tr><tr><td>[me@linuxbox ~]$ umask 0000</td><td></td></tr><tr><td>[me@linuxbox ~]$ > foo.txt</td><td></td></tr><tr><td>[me@linuxbox ~]$ ls -l foo.txt</td><td></td></tr><tr><td>-rw-rw-rw- 1 me</td><td>me 0 2025-03-06 14:58 f00.txt</td></tr></table></body></html> 
When we set the mask to 0000 (effectively turning it off), we see that the file is now world writable. To understand how this works, we have to look at octal numbers again. If we change the mask to 0002, expand it into binary, and then compare it to the attributes we can see what happens. 
<html><body><table><tr><td>Original file mode</td><td>rw- rw- rw-</td></tr><tr><td>Mask</td><td>000 000 000 010</td></tr><tr><td>Result</td><td>rw- rw- r--</td></tr></table></body></html> 
Ignore for the moment the leading zeros (we'll get to those in a minute) and observe that where the 1 appears in our mask, an attribute was removed — in this case, the world write permission. That's what the mask does. Everywhere a 1 appears in the binary value of the mask, an attribute is unset. If we look at a mask value of 0022, we can see what it does. 
<html><body><table><tr><td>Original file mode</td><td>rw- rw- rw-</td></tr><tr><td>Mask</td><td>000 000 010 010</td></tr><tr><td>Result</td><td>rw- r-- r--</td></tr></table></body></html> 
Again, where a 1 appears in the binary value, the corresponding attribute is unset. Play with some values (try some sevens) to get used to how this works. When you're done, remember to clean up. 
[me@linuxbox $-]$1$ rm foo.txt; umask 0002 
Most of the time we won't have to change the mask; the default provided by the distribution will be fine. In some high-security situations, however, we will want to control it. 
# Some Special Permissions 
Though we usually see an octal permission mask expressed as a three-digit number, it is more technically correct to express it in four digits. Why? Because, in addition to read, write, and execute permission, there are some other, less used, permission settings. 
The first of these is the setuid bit (octal 4000). When applied to an executable file, it sets the effective user ID from that of the real user (the user actually running the program) to that of the program's owner. Most often this is given to a few programs owned by the superuser. When an ordinary user runs a program that is “setuid root” , the program runs with the effective privileges of the superuser. This allows the program to access files and directories that an ordinary user would normally be prohibited from accessing. Clearly, because this raises security concerns, the number of setuid programs must be held to an absolute minimum. 
The second less-used setting is the setgid bit (octal 2000), which, like the setuid bit, changes the effective group ID from the real group ID of the real user to that of the file owner. If the setgid bit is set on a directory, newly created files in the directory will be given the group ownership of the directory rather the group ownership of the file's creator. This is useful in a shared directory when members of a common group need access to all the files in the directory, regardless of the file owner's primary group. 
The third is called the sticky bit (octal 1000). This is a holdover from ancient Unix, where it was possible to mark an executable file as “not swappable.” On files, Linux ignores the sticky bit, but if applied to a directory, it prevents users from deleting or renaming files unless the user is either the owner of the directory, the owner of the file, or the superuser. This is often used to control access to a shared directory, such as /tmp. 
Here are some examples of using chmod with symbolic notation to set these special permissions. Here’s an example of assigning setuid to a program: 
chmod $\mathsf{u}{+}\mathsf{s}$ program 
Next, here’s and example of assigning setgid to a directory: 
chmod $9^{+}{\mathsf{s}}$ dir 
Finally, here’s an example of assigning the sticky bit to a directory: 
chmod +t dir 
When viewing the output from ls, you can determine the special permissions. Here are some examples. First, an example of a program that is setuid: 
-rwsr-xr-x 
Here’s an example of a directory that has the setgid attribute: 
drwxrwsr-x 
Here’s an example of a directory with the sticky bit set: 
drwxrwxrwt 
# Changing Identities 
Sometimes we may find it necessary to take on the identity of another user. Often we want to gain superuser privileges to carry out some administrative task, but it is also possible to “become” another regular user for such things as testing an account. There are three ways to take on an alternate identity. 
1. Log out and log back in as the alternate user. 
2. Use the su command. 
3. Use the sudo command. 
We will skip the first technique since we know how to do it and it lacks the convenience of the other two. From within our own shell session, the su command allows us to assume the identity of another user and either start a new shell session with that user's ID, or to issue a single command as that user. The sudo command allows an administrator to set up a configuration file called /etc/sudoers and define specific commands that particular users are permitted to execute under an assumed identity. The choice of which command to use is largely determined by which Linux distribution you use. Your distribution probably includes both commands, but its configuration will favor either one or the other. We'll start with su. Though be aware that the use of su is falling out of favor in modern Linux distributions. 
# su – Run a Shell with Substitute User and Group IDs 
The su command is used to start a shell as another user. The command syntax looks like this: 
![](https://cdn-mineru.openxlab.org.cn/extract/e52c4087-794e-43f3-821f-0f8eb1fdf822/c03dacc85930d1f439d406ad281600ef3b3b91c3be1349810efda93f98df8d81.jpg) 
If the “-l” option is included, the resulting shell session is a login shell for the specified user. This means the user's environment is loaded and the working directory is changed to the user's home directory. This is usually what we want. If the user is not specified, the superuser is assumed. Notice that (strangely) the -l may be abbreviated as -, which is how it is most often used. Assuming that the root account has a password set (which is not the custom in modern distributions) we can start a shell for the superuser this way: 
[me@linuxbox ~]$ su - Password: [root@linuxbox ~]# 
After entering the command, we are prompted for the superuser's password. If it is successfully entered, a new shell prompt appears indicating that this shell has superuser privileges (the trailing # rather than a $$1$ ), and the current working directory is now the home directory for the superuser (normally /root). Once in the new shell, we can carry out commands as the superuser. When finished, enter exit to return to the previous shell. 
[root@linuxbox ~]# exit [me@linuxbox ~]$ 
It is also possible to execute a single command rather than starting a new interactive command by using su this way. 
Using this form, a single command line is passed to the new shell for execution. It is important to enclose the command in quotes, as we do not want expansion to occur in our shell, but rather in the new shell. 
[me@linuxbox ~]$ su -c 'ls -l /root/\*' 
Password: 
-rw- - 1 root root 754 2007-08-11 03:19 /root/anaconda-ks.cfg 
/root/Mail: 
total 0 
[me@linuxbox ~]$ 
# sudo – Execute a Command as Another User 
The sudo command is like su in many ways but has some important additional capabilities. The administrator can configure sudo to allow an ordinary user to execute commands as a different user (usually the superuser) in a controlled way. In particular, a user may be restricted to one or more specific commands and no others. Another important difference is that the use of sudo does not require access to the superuser's password. Authenticating using sudo, requires the user’s own password. Let's say, for example, that sudo has been configured to allow us to run a fictitious backup program called “backup_script”, which requires superuser privileges. With sudo it would be done like this: 
[me@linuxbox ~]$ sudo backup_script Password: System Backup Starting... 
After entering the command, we are prompted for our password (not the superuser's) and once the authentication is complete, the specified command is carried out. One important difference between su and sudo is that sudo does not start a new shell, nor does it load another user's environment. This means that commands do not need to be quoted any differently than they would be without using sudo. Note that this behavior can be overridden by specifying various options. Note, too, that sudo can be used to start an interactive superuser session (much like su -) by using the -i option. See the sudo man page for details. 
To see what privileges are granted by sudo, use the -l option to list them: 
[me@linuxbox ~]$ sudo -l 
User me may run the following commands on this host: (ALL) ALL 
# Modern Linux Distributions and sudo 
One of the recurrent problems for regular users is how to perform certain tasks that require superuser privileges. These tasks include installing and updating software, editing system configuration files, and accessing devices. In the Windows world, this is often done by giving users administrative privileges. This allows users to perform these tasks. However, it also enables programs executed by the user to have the same abilities. This is desirable in most cases, but it also permits malware (malicious software) such as viruses to have free rein of the computer. 
In the Unix world, there has always been a larger division between regular users and administrators, owing to the multiuser heritage of Unix. The approach taken in Unix is to grant superuser privileges only when needed. To do this, the su and sudo commands are commonly used. 
Years ago, most Linux distributions relied on su for this purpose. su didn't require the configuration that sudo required, and having a root account is traditional in Unix. This, however introduced a problem. Users were tempted to operate as root unnecessarily. In fact, some users operated their systems as the root user exclusively, since it does away with all those annoying “permission denied” messages. This is how you reduce the security of a Linux system to that of a Windows system. Not a good idea. 
When Ubuntu was introduced, its creators took a different tack. By default, Ubuntu disables logins to the root account (by failing to set a password for the account) and instead uses sudo to grant superuser privileges. The initial user account is granted full access to superuser privileges via sudo and may grant similar powers to subsequent user accounts. This method of granting privileges is now the accepted standard is most modern distributions. 
# chown – Change File Owner and Group 
The chown command is used to change the owner and group owner of a file or directory. Superuser privileges are required to use this command. The syntax of chown looks like this: 
# chown [owner][:[group]] file.. 
chown can change the file owner and/or the file group owner depending on the first argument of the command. Table 9-7 provides some examples. 
Table 9-7: chown Argument Examples 
<html><body><table><tr><td> Argument</td><td>Results</td></tr><tr><td>bob</td><td>Changes the ownership of the file from its current owner to user bob.</td></tr><tr><td>bob:users</td><td>Changes the ownership of the file from its current owner to user bob and changes the file group owner to group users.</td></tr><tr><td>:admins</td><td>Changes the group owner to the group admins. The file owner is unchanged.</td></tr><tr><td>bob:</td><td>Changes the file owner from the current owner to user bob and changes the group owner to the login group of user bob.</td></tr></table></body></html> 
Let's say we have two users; janet, who has access to superuser privileges and tony, who does not. User janet wants to copy a file from her home directory to the home directory of user tony. Since user janet wants tony to be able to edit the file, janet changes the ownership of the copied file from janet to tony. 
[janet@linuxbox ~]$ sudo cp myfile.txt ~tony 
Password: 
[janet@linuxbox ~]$ sudo ls -l ~tony/myfile.txt 
-rw-r--r-- 1 root root root 2025-03-20 14:30 /home/tony/myfile.txt 
[janet@linuxbox ~]$ sudo chown tony: ~tony/myfile.txt 
[janet@linuxbox $-]$1$ sudo ls -l ~tony/myfile.txt 
-rw-r--r-- 1 tony tony tony 2025-03-20 14:30 /home/tony/myfile.txt 
Here we see user janet copy the file from her directory to the home directory of user tony. Next, janet changes the ownership of the file from root (a result of using sudo) to tony. Using the trailing colon in the first argument, janet also changed the group ownership of the file to the login group of tony, which happens to be group tony. 
Notice that after the first use of sudo, janet was not prompted for her password. This is because sudo, in most configurations, “trusts” us for several minutes until its timer 
runs out. 
# chgrp – Change Group Ownership 
In older versions of Unix, the chown command only changed file ownership, not group ownership. For that purpose, a separate command, chgrp was used. It works much the same way as chown, except for being more limited. 
# Exercising Our Privileges 
Now that we have learned how this permissions thing works, it's time to show it off. We are going to demonstrate the solution to a common problem — setting up a shared directory. Let's revisit our friends janet and tony. They both have music collections and want to set up a shared directory, where they will each store their music files as Ogg Vorbis or MP3. As before, user janet has access to superuser privileges via sudo. 
A group needs to be created that will have both janet and tony as members. This is done in two steps. First, using the groupadd command, we create the group followed with the usermod command to add users to the group: 
[janet@linuxbox ~]$ sudo groupadd music [janet@linuxbox ~]$ sudo usermod -a -G music janet [janet@linuxbox $-]$1$ sudo usermod -a -G music tony 
The options used with the usermod command are short for --append and --group and they add the specified user to the corresponding group in the /etc/group file. 
Next, janet creates the directory for the music files. 
[janet@linuxbox ~]$ sudo mkdir /usr/local/share/Music Password: 
Since janet is manipulating files outside of her home directory, superuser privileges are required. After the directory is created, it has the following ownerships and permissions: 
[janet@linuxbox ~]$ ls -ld /usr/local/share/Music drwxr-xr-x 2 root root 4096 2025-03-21 18:05 /usr/local/share/Music 
As we can see, the directory is owned by root and has permission mode 755. To make this directory shareable, janet needs to change the group ownership and the group permissions to allow writing. 
[janet@linuxbox ~]$ sudo chown :music /usr/local/share/Music [janet@linuxbox $-]$1$ sudo chmod 2775 /usr/local/share/Music [janet@linuxbox ~]$ ls -ld /usr/local/share/Music drwxrwsr-x 2 root music 4096 2025-03-21 18:05 /usr/local/share/Music 
Using the chown command, janet sets the group owner of the directory to music then uses chmod to set the directory permissions to 2755. This sets the setguid to cause all files in the directory to inherit the same group ownership as the directory. We did this by executing chmod 2755 but we could have done thing by using the symbolic method with chmod $9^{+}{\mathsf{s}}$ . 
What does this all mean? It means that we now have a directory, /usr/local/ share/Music that is owned by root and allows read and write access to group music. Group music has members janet and tony; thus, janet and tony can create files in directory /usr/local/share/Music. Other users can list the contents of the directory but cannot create files there. 
But we still have a problem. The default umask on this system is 0022, which prevents group members from writing files belonging to other members of the group. This would not be a problem if the shared directory contained only files, but since this directory will store music, and music is usually organized in a hierarchy of artists and albums, members of the group will need the ability to create files and directories inside directories created by other members. We need to change the umask used by janet and tony to 0002 instead. 
janet sets her umask to 0002, and creates a new test file and directory: 
[janet@linuxbox ~]$ umask 0002 
[janet@linuxbox ~]$ > /usr/local/share/Music/test_file 
[janet@linuxbox ~]$ mkdir /usr/local/share/Music/test_dir 
[janet@linuxbox ~]$ ls -l /usr/local/share/Music 
drwxrwsr-x 2 janet music 4096 2025-03-24 20:24 test_dir 
-rw-rw-r-- 1 janet music 0 2025-03-24 20:22 test_file 
[janet@linuxbox ~]$ 
Both files and directories are now created with the correct permissions to allow all members of the group music to create files and directories inside the Music directory. 
The one remaining issue is umask. The necessary setting only lasts until the end of session and must be reset. In Chapter 11, we'll look at making the change to umask permanent. 
# Changing Your Password 
The last topic we'll cover in this chapter is setting passwords for ourselves (and for other users if we have access to superuser privileges). To set or change a password, the passwd command is used. The command syntax looks like this: 
# passwd [user] 
To change our password, we just enter the passwd command. We will be prompted for our old password and our new password. 
[me@linuxbox ~]$ passwd (current) UNIX password: New UNIX password: 
The passwd command will try to enforce use of “strong” passwords. This means it will refuse to accept passwords that are too short, are too similar to previous passwords, are dictionary words, or are too easily guessed. 
[me@linuxbox ~]$ passwd 
(current) UNIX password: 
New UNIX password: 
BAD PASSWORD: is too similar to the old one 
New UNIX password: 
BAD PASSWORD: it is WAY too short 
New UNIX password: 
BAD PASSWORD: it is based on a dictionary word 
If we have superuser privileges, you can specify a username as an argument to the passwd command to set the password for another user. Other options are available to the superuser to allow account locking, password expiration, and so on. See the passwd man page for details. 
The passwd, addgroup, and usermod commands are part of a suite of commands in the shadow-utils package. Table 9-8 lists some of the commands contained in that package: 
Table 9-8: shadow-utils Commands 
<html><body><table><tr><td>Command</td><td>Description</td></tr><tr><td>lastlog</td><td>Reports the most recent login of all users or of a given user.</td></tr><tr><td>useradd</td><td>Create a new user or update default new user information.</td></tr><tr><td>userdel</td><td>Delete a user account and related files.</td></tr><tr><td>usermod</td><td>Modify a user account.</td></tr><tr><td>groupadd</td><td>Create a new group.</td></tr><tr><td>groupdel</td><td>Delete a group.</td></tr><tr><td>groupmod</td><td>.Modify a group definition on the system.</td></tr></table></body></html> 
We won’t be covering these commands in any detail as they fall a little outside the scope of this book. For further information, consult each command’s man page. 
# Summing Up 
In this chapter we saw how Unix-like systems such as Linux manage user permissions to allow the read, write, and execution access to files and directories. The basic ideas of this system of permissions date back to the early days of Unix and have stood up pretty well to the test of time. But the native permissions mechanism in Unix-like systems lacks the fine granularity of more modern systems. 
# Further Reading 
Wikipedia has a good article on malware: http://en.wikipedia.org/wiki/Malware 
# 10 – Processes 
Modern operating systems are usually multitasking, meaning they create the illusion of doing more than one thing at once by rapidly switching from one executing program to another. The Linux kernel manages this through the use of processes. Processes are how Linux organizes the different programs waiting for their turn at the CPU. 
Sometimes a computer will become sluggish or an application will stop responding. In this chapter, we will look at some of the tools available at the command line that let us examine what programs are doing and how to terminate processes that are misbehaving. 
This chapter will introduce the following commands: 
ps – Report a snapshot of current processes 
top – Display tasks 
jobs – List active jobs 
bg – Place a job in the background 
fg – Place a job in the foreground 
kill – Send a signal to a process 
killall – Kill processes by name 
nice - Run a program with modified scheduling priority 
renice - Alter priority of running processes 
nohup - Run a command immune to hangups 
halt/poweroff/reboot - Halt, power-off, or reboot the system 
shutdown – Shutdown or reboot the system 
# How a Process Works 
When a system starts up, the kernel initiates a few of its own activities as processes and launches a program called init. init, in turn, starts systemd which starts all the system services. In older Linux distributions init runs a series of shell scripts (located in /etc) called init scripts to perform a similar function. Many system services are implemented as daemon programs, programs that just sit in the background and do their thing without having any user interface. So, even if we are not logged in, the system is at least a little busy performing routine stuff. 
The fact that a program can launch other programs is expressed in the process scheme as a parent process producing a child process. 
The kernel maintains information about each process to help keep things organized. For example, each process is assigned a number called a process ID (PID). PIDs are assigned in ascending order, with init always getting PID 1. The kernel also keeps track of the memory assigned to each process, as well as the processes' readiness to resume execution. Like files, processes also have owners and user IDs, effective user IDs, etc. 
# Viewing Processes 
The most commonly used tool to view processes (there are several) is the ps command. The ps program has a lot of options, but in its simplest form it is used like this: 
<html><body><table><tr><td>[me@linuxbox ~]$ ps</td><td colspan="2"></td></tr><tr><td>PID TTY</td><td></td><td>TIME CMD</td></tr><tr><td>5198 pts/1</td><td>00:00:00 bash</td><td></td></tr><tr><td>10129 pts/1</td><td>00:00:00 ps</td><td></td></tr></table></body></html> 
The result in this example lists two processes, process 5198 and process 10129, which are bash and ps respectively. As we can see, by default, ps doesn't show us very much, just the processes associated with the current terminal session. To see more, we need to add some options, but before we do that, let's look at the other fields produced by ps. TTY is short for “teletype,” and refers to the controlling terminal for the process. Unix is showing its age here. The TIME field is the amount of CPU time consumed by the process. As we can see, neither process makes the computer work very hard. 
If we add an option, we can get a bigger picture of what the system is doing. 
<html><body><table><tr><td>[me@linuxbox ~]$ ps x</td><td colspan="3"></td></tr><tr><td>PID TTY</td><td>STAT</td><td>TIME COMMAND</td><td></td></tr><tr><td>2799 ?</td><td>Ssl</td><td></td><td>0:00 /usr/libexec/bonobo-activation-server -ac</td></tr><tr><td>2820 ?</td><td>Sl</td><td></td><td>0:01 /usr/libexec/evolution-data-server-1.10 -</td></tr><tr><td>15647 ？</td><td>Ss</td><td></td><td>0:00 /bin/sh /usr/bin/startkde</td></tr><tr><td>15751？</td><td>Ss</td><td></td><td>0:00 /usr/bin/ssh-agent /usr/bin/dbus-launch -</td></tr><tr><td>15754 ?</td><td>S</td><td></td><td>0:00 /usr/bin/dbus-launch --exit-with-session</td></tr><tr><td>15755？</td><td>Ss</td><td></td><td>0:01 /bin/dbus-daemon --fork --print-pid 4 -pr</td></tr></table></body></html> 
<html><body><table><tr><td>15774 ?</td><td>Ss</td><td></td><td>0:02 /usr/bin/gpg-agent -s -daemon</td></tr><tr><td>15793 ?</td><td>S</td><td></td><td>0:00 start_kdeinit --new-startup +kcminit_start</td></tr><tr><td>15794？</td><td>Ss</td><td></td><td>0:00 kdeinit Running...</td></tr><tr><td>15797？</td><td>S</td><td></td><td>0:00 dcopserver -nosid</td></tr><tr><td>and many more...</td><td></td><td></td><td></td></tr></table></body></html> 
Adding the “x” option (note that there is no leading dash) tells ps to show all of our processes regardless of what terminal (if any) they are controlled by. The presence of a “?” in the TTY column indicates no controlling terminal. Using this option, we see a list of every process that we own. 
Since the system is running a lot of processes, ps produces a long list. It is often helpful to pipe the output from ps into less for easier viewing. Some option combinations also produce long lines of output, so maximizing the terminal emulator window may be a good idea, too. 
A new column titled STAT has been added to the output. STAT is short for “state” and reveals the current status of the process, as shown in Table 10-1. 
Table 10-1: Process States 
<html><body><table><tr><td> State</td><td>Meaning</td></tr><tr><td>R</td><td>Running. This means that the process is running or ready to run.</td></tr><tr><td>S</td><td>Sleeping. The process is not running; rather, it is waiting for an event, such as a keystroke or network packet.</td></tr><tr><td>D</td><td>Uninterruptible sleep. The process is waiting for I/O such as a disk drive.</td></tr><tr><td>T</td><td>Stopped. The process has been instructed to stop. More on this later in the chapter.</td></tr><tr><td>Z</td><td>A defunct or “zombie" process. This is a child process that has terminated but has not been cleaned up by its parent.</td></tr><tr><td>Λ</td><td>A high-priority process. It's possible to grant more importance to a process, giving it more time on the CPU. This property of a process is called niceness.A process with high priority is said to be less nice because it's taking more of the CPU's time, which leaves less for everybody else.</td></tr><tr><td>N</td><td>A low-priority process.A process with low priority (a“nice" process) will get processor time only after other processes with higher priority have been serviced.</td></tr></table></body></html> 
The process state may be followed by other characters. These indicate various exotic process characteristics. See the ps man page for more detail. 
Another popular set of options is “aux” (without a leading dash). This gives us even more information. 
<html><body><table><tr><td>[me@linuxbox ~]$ ps aux</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>USER</td><td>PID</td><td>%CPU</td><td>%MEM</td><td>VSZ</td><td>RSS</td><td>TTY</td><td>STAT</td><td> START</td><td>TIME</td><td>COMMAND</td></tr><tr><td>root</td><td>1</td><td>0.0</td><td>0.0</td><td>2136</td><td>644 ?</td><td></td><td>Ss</td><td>Mar05</td><td>0:31</td><td>init</td></tr><tr><td>root</td><td>2</td><td>0.0</td><td>0.0</td><td>0</td><td></td><td></td><td>S<</td><td>Mar05</td><td>0:00</td><td>[kt]</td></tr><tr><td>root</td><td>3</td><td>0.0</td><td>0.0</td><td>0</td><td></td><td></td><td>S<</td><td>Mar05</td><td>0:00</td><td>[mi]</td></tr><tr><td>root</td><td>4</td><td>0.0</td><td>0.0</td><td>0</td><td>口？</td><td></td><td>S<</td><td>Mar05</td><td>0:00</td><td>[ks]</td></tr><tr><td>root</td><td>5</td><td>0.0</td><td>0.0</td><td>0</td><td></td><td></td><td>S<</td><td>Mar05</td><td>0:06</td><td>[wa]</td></tr><tr><td>root</td><td>6</td><td>0.0</td><td>0.0</td><td>0</td><td></td><td></td><td>S<</td><td>Mar05</td><td>0:36</td><td>[ev]</td></tr><tr><td>root</td><td>7</td><td>0.0</td><td>0.0</td><td>0</td><td></td><td></td><td>S<</td><td>Mar05</td><td>0:00</td><td>[kh]</td></tr><tr><td>and many more...</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table></body></html> 
This set of options displays the processes belonging to every user. Using the options without the leading dash invokes the command with “BSD style” behavior. The Linux version of ps can emulate the behavior of the ps program found in several different Unix implementations. The most popular BSD options are shown in Table 10-2. 
Table 10-2: Popular BSD Style ps Options 
<html><body><table><tr><td>Option</td><td>Function</td></tr><tr><td>X</td><td>List our running processes.</td></tr><tr><td>ax</td><td>List all running processes..</td></tr><tr><td>W</td><td>Include full command names.</td></tr><tr><td>u</td><td>Verbose listing.</td></tr></table></body></html> 
With the aux options, we get the additional columns shown in Table 10-3. 
Table 10-3: BSD Style ps Column Headers 
<html><body><table><tr><td> Header</td><td>Meaning</td></tr><tr><td>USER</td><td>User ID. This is the owner of the process.</td></tr><tr><td>%CPU</td><td>CPU usage in percent.</td></tr></table></body></html> 
<html><body><table><tr><td>%MEM Memory usage in percent.</td><td></td></tr><tr><td>VSZ</td><td>Virtual memory size.</td></tr><tr><td>RSS</td><td>Resident set size. This is the amount of physical memory (RAM) the process is using in kilobytes.</td></tr><tr><td>START</td><td>Time when the process started. For values over 24 hours, a date is used.</td></tr><tr><td>TIME</td><td>The amount of CPU time consumed by the process.</td></tr></table></body></html> 
It’s also possible to produce a detailed snapshot of a single process by including a PID as a command argument as shown in the example below. 
<html><body><table><tr><td colspan="10">[me@linuxbox ~]$ ps uw 44719</td></tr><tr><td>USER PID</td><td></td><td></td><td>%CPU %MEM VSZ</td><td></td><td>RSS TTY</td><td></td><td> STAT START</td><td></td><td>TIME COMMAND</td><td></td></tr><tr><td>me</td><td>44719</td><td>0.0</td><td>0.0</td><td>13480</td><td>6492 pts/1</td><td></td><td>S</td><td>15:57</td><td>0:00 bash</td><td></td></tr></table></body></html> 
# Viewing Processes Dynamically with top 
While the ps command can reveal a lot about what the machine is doing, it provides only a snapshot of the machine's state at the moment the ps command is executed. To see a more dynamic view of the machine's activity, we use the top command: 
top - 14:59:20 up 6:30, 2 users, load average: 0.07, 0.02, 0.00 Tasks: 109 total, 1 running, 106 sleeping, 0 stopped, 2 zombie Cpu(s): 0.7%us, 1.0%sy, 0.0%ni, 98.3%id, 0.0%wa, 0.0%hi, 0.0%si Mem: 319496k total, 314860k used, 4636k free, 19392k buff Swap: 875500k total, 149128k used, 726372k free, 114676k cach 
<html><body><table><tr><td>PID USER</td><td></td><td>PR NI</td><td>VIRT</td><td></td><td>RES</td><td></td><td>SHR S %CPU %MEM</td><td></td><td>TIME+</td><td>COMMAND</td></tr><tr><td>6244</td><td>me</td><td>39</td><td>19</td><td>31752</td><td>3124</td><td>2188 S</td><td>6.3</td><td>1.0</td><td>16:24.42</td><td>trackerd</td></tr><tr><td>11071</td><td>me</td><td>20</td><td>0</td><td>2304</td><td>1092</td><td>840 R</td><td>1.3</td><td>0.3</td><td>0:00.14</td><td>top</td></tr><tr><td>6180</td><td> me</td><td>20</td><td>0</td><td>2700</td><td>1100</td><td>772</td><td>S 0.7</td><td>0.3</td><td>0:03.66</td><td>dbus-dae</td></tr><tr><td>6321</td><td> me</td><td>20</td><td>0</td><td>20944</td><td>7248</td><td>6560</td><td>S 0.7</td><td>2.3</td><td>2:51.38</td><td>multiloa</td></tr><tr><td>4955</td><td>root</td><td>20</td><td>0</td><td>104m</td><td>9668</td><td>5776</td><td>S 0.3</td><td>3.0</td><td>2:19.39</td><td>Xorg</td></tr><tr><td>1</td><td>root</td><td>20</td><td>0</td><td>2976</td><td>528</td><td>476</td><td>S 0.0</td><td>0.2</td><td>0:03.14 init</td><td></td></tr><tr><td>2</td><td>root</td><td>15</td><td>-5</td><td>0</td><td>0</td><td>0</td><td>S 0.0</td><td>0.0</td><td>0:00.00</td><td>kthreadd</td></tr><tr><td>3</td><td>root</td><td>RT</td><td>-5</td><td>0</td><td>0</td><td>0</td><td>S 0.0</td><td>0.0</td><td>0:00.00</td><td>migratio</td></tr><tr><td></td><td>4 root</td><td>15</td><td>-5</td><td>0</td><td>0</td><td>0</td><td>S 0.0</td><td>0.0</td><td>0:00.72 ksoftirq</td><td></td></tr><tr><td>5</td><td>root</td><td>RT</td><td>-5</td><td>0</td><td>0</td><td>0</td><td>S 0.0</td><td>0.0</td><td>0:00.04 watchdog</td><td></td></tr><tr><td>6</td><td>root</td><td>15</td><td>-5</td><td>0</td><td>0</td><td>0</td><td>S 0.0</td><td>0.0</td><td></td><td>0:00.42 events/0</td></tr><tr><td>7</td><td>root</td><td>15</td><td>-5</td><td>0</td><td>0</td><td>0</td><td>S 0.0</td><td>0.0</td><td>0:00.06 khelper</td><td></td></tr><tr><td>41</td><td>root</td><td>15</td><td>-5</td><td>0</td><td>0</td><td>0</td><td>S 0.0</td><td>0.0</td><td>0:01.08 kblockd/</td><td></td></tr><tr><td>67</td><td>root</td><td>15</td><td>-5</td><td>0</td><td>0</td><td>0</td><td>S 0.0</td><td>0.0</td><td>0:00.00 kseriod</td><td></td></tr><tr><td>114</td><td>root</td><td>20</td><td>0</td><td>0</td><td>0</td><td>0</td><td>S 0.0</td><td>0.0</td><td>0:01.62 pdflush</td><td></td></tr><tr><td>116</td><td>root</td><td>15</td><td>-5</td><td>0</td><td>0</td><td>0</td><td>S 0.0</td><td>0.0</td><td></td><td>0:02.44 kswapd0</td></tr></table></body></html> 
The system summary contains a lot of good stuff. Here's a rundown: 
Table 10-4: top Information Fields 
<html><body><table><tr><td>Row</td><td>Field</td><td>Meaning</td></tr><tr><td>1</td><td>top</td><td>The name of the program.</td></tr><tr><td></td><td>14:59:20</td><td>The current time of day.</td></tr><tr><td></td><td>up 6:30</td><td>This is called uptime. It is the amount of time since the machine was last booted. In this example, the system has been up for six-and-a- half hours.</td></tr><tr><td></td><td>2 users</td><td>There are two users logged in.</td></tr><tr><td></td><td>load average:</td><td>Load average refers to the number of processes that are waiting to run, that is, the number of processes that are in a runnable state and are sharing the CPU. Three values are shown, each</td></tr></table></body></html> 
<html><body><table><tr><td></td><td>for a different period of time. The first is the average for the last 6O seconds, the next the previous 5 minutes, and finally the previous 15 minutes. Values less than 1.O indicate that the machine is not busy.</td></tr><tr><td>2 Tasks:</td><td>This summarizes the number of processes and their various process states.</td></tr><tr><td>3 Cpu(s):</td><td>This row describes the character of the activities that the CPU is performing.</td></tr><tr><td>0.7%us</td><td>0.7 percent of the CPU is being used for user processes. This means processes outside the kernel.</td></tr><tr><td>1.0%sy</td><td>1.0 percent of the CPU is being used for system (kernel) processes.</td></tr><tr><td>0.0%ni</td><td>0.0 percent of the CPU is being used by “nice" (low-priority) processes.</td></tr><tr><td>98.3%id</td><td>98.3 percent of the CPU is idle.</td></tr><tr><td>0.0%wa</td><td>0.0 percent of the CPU is waiting for I/O.</td></tr><tr><td>4 Mem:</td><td>This shows how physical RAM is being used.</td></tr><tr><td>5 Swap :</td><td>This shows how swap space (virtual memory) is being used.</td></tr></table></body></html> 
The top program accepts a number of keyboard commands. The two most interesting are h, which displays the program's help screen, and q, which quits top. 
Both major desktop environments provide graphical applications that display information similar to top (in much the same way that Task Manager in Windows works), but top is better than the graphical versions because it is faster and it consumes far fewer system resources. After all, our system monitor program shouldn't be the source of the system slowdown that we are trying to track. 
# Controlling Processes 
Now that we can see and monitor processes, let's gain some control over them. For our experiments, we're going to use a little program called xlogo as our guinea pig. The xlogo program is a sample program supplied with the X Window System (the underlying engine that makes the graphics on our display go though it’s going out of fashion in 
favor of Wayland), which simply displays a re-sizable window containing the X logo. 
First, we'll get to know our test subject. 
After entering the command, a small window containing the logo should appear somewhere on the screen. On some systems, xlogo may print a warning message, but it may be safely ignored. 
Tip: If your system does not include the xlogo program, try using gedit or kwrite instead. 
We can verify that xlogo is running by resizing its window. If the logo is redrawn in the new size, the program is running. 
Notice how our shell prompt has not returned? This is because the shell is waiting for the program to finish, just like all the other programs we have used so far. If we close the xlogo window, the prompt returns. 
![](https://cdn-mineru.openxlab.org.cn/extract/e52c4087-794e-43f3-821f-0f8eb1fdf822/bb5cd8d0c13781c1b43b5f27f39d459aadd6130ced359a530f9061bf0e168860.jpg) 
Figure 3: The xlogo program 
# Interrupting a Process 
Let's observe what happens when we run xlogo again. First, enter the xlogo command and verify that the program is running. Next, return to the terminal window and press Ctrl-c. 
[me@linuxbox ~]$ xlogo 
In a terminal, pressing Ctrl-c, interrupts a program. This means we are politely asking the program to terminate. After we pressed Ctrl-c, the xlogo window closed and the shell prompt returned. 
Many (but not all) command-line programs can be interrupted by using this technique. 
# Putting a Process in the Background 
Let's say we wanted to get the shell prompt back without terminating the xlogo program. We can do this by placing the program in the background. Think of the terminal as having a foreground (with stuff visible on the surface like the shell prompt) and a background (with stuff hidden behind the surface). To launch a program so that it is immediately placed in the background, we follow the command with an ampersand (&) character. 
<html><body><table><tr><td>[me@linuxbox ~]$ xlogo &</td></tr><tr><td></td></tr><tr><td>[1]28236</td></tr><tr><td>[me@linuxbox ~]$</td></tr></table></body></html> 
After entering the command, the xlogo window appeared and the shell prompt returned, but some funny numbers were printed too. This message is part of a shell feature called job control. With this message, the shell is telling us that we have started job number 1 ([1]) and that it has PID 28236. If we run ps, we can see our process. 
<html><body><table><tr><td colspan="2">[me@linuxbox ~]$ ps</td><td></td></tr><tr><td>PID TTY</td><td></td><td>TIME CMD</td></tr><tr><td>10603 pts/1</td><td></td><td>00:00:00 bash</td></tr><tr><td>28236 pts/1</td><td></td><td>00:00:00 xlogo</td></tr><tr><td>28239 pts/1</td><td></td><td>00:00:00 ps</td></tr></table></body></html> 
The shell's job control facility also gives us a way to list the jobs that have been launched from our terminal. Using the jobs command, we can see this list: 
<html><body><table><tr><td>[me@linuxbox ~]$ jobs [1]+ Running xlogo &</td></tr></table></body></html> 
The results show that we have one job, numbered 1, that it is running, and that the com 
mand was xlogo &. 
Note that we can put multiple commands in the background by using this shortcut as shown below. 
<html><body><table><tr><td>me@linuxbox:~$ xlogo & gedit &</td></tr><tr><td>[1]47211</td></tr><tr><td></td></tr><tr><td>[2]47212</td></tr></table></body></html> 
# Returning a Process to the Foreground 
A process in the background is immune from terminal keyboard input, including any attempt to interrupt it with $\tt c t r l-c$ . To return a process to the foreground, use the fg command in this way: 
<html><body><table><tr><td>[me@linuxbox ~]$ jobs</td><td></td></tr><tr><td>[1]+ Running</td><td>xlogo &</td></tr><tr><td>[me@linuxbox ~]$ fg %1 xlogo</td><td></td></tr></table></body></html> 
The $\mathsf{f}\mathsf{g}$ command followed by a percent sign and the job number (called a jobspec) does the trick. If we only have one background job, the jobspec is optional. To terminate xlogo, press Ctrl-c. 
# Stopping (Pausing) a Process 
Sometimes we'll want to stop a process without terminating it. This is often done to allow a foreground process to be moved to the background. To stop a foreground process and place it in the background, press Ctrl-z. Let's try it. At the command prompt, type xlogo, press the Enter key, and then press Ctrl-z: 
<html><body><table><tr><td>[me@linuxbox ~]$ xlogo</td></tr><tr><td>[1]+ Stopped</td></tr><tr><td>xlogo</td></tr><tr><td>[me@linuxbox ~]$</td></tr></table></body></html> 
After stopping xlogo, we can verify that the program has stopped by attempting to resize the xlogo window. We will see that it appears quite dead. We can either continue the program's execution in the foreground, using the $\mathsf{f}\mathsf{g}$ command, or resume the program's execution in the background with the bg command: 
[me@linuxbox ~]$ bg %1 [1]+ xlogo & [me@linuxbox ~]$ 
As with the fg command, the jobspec is optional if there is only one job. 
Moving a process from the foreground to the background is handy if we launch a graphical program from the command line, but forget to place it in the background by appending the trailing &. 
Why would we want to launch a graphical program from the command line? There are two reasons. 
The program we want to run might not be listed on the window manager's menus (such as xlogo). 
By launching a program from the command line, we might be able to see error messages that would otherwise be invisible if the program were launched graphically. Sometimes, a program will fail to start up when launched from the graphical menu. By launching it from the command line instead, we may see an error message that will reveal the problem. Also, some graphical programs have interesting and useful command line options. 
# Changing Process Priority 
As we saw in the output of the ps command (as well as top) there is a process attribute called “niceness” which refers to the scheduling priority given to a process. In certain circumstances such as when video transcoding or performing CPU-based ray tracing for example, we may want to give a process more priority (less niceness) or alternately if we want a process to use less CPU time we could give it more niceness. Niceness can be adjusted with the nice and renice commands. It is important to remember that only the superuser may increase the priority of a process and that regular users may only decrease the priority of processes that they own. 
The nice command launches a process with a specified niceness. Niceness adjustments are expressed from -20 (the most favorable) to 19 (the least favorable) with a default of value of zero (no adjustment). Let’s see how this works. Imagine we have a program called cpu-hog that we want to run at a lower priority than it’s normal 20. We can launch the program with nice as follows: 
Likewise if we have a program called must-run-fast that needs to be given more 
CPU priority, we (as the superuser) could do this: 
<html><body><table><tr><td>[me@linuxbox ~]$ sudo nice -n -10 must-run-fast</td></tr></table></body></html> 
It’s rarely necessary to run a command with increased priority and doing so runs the risk of starving essential system processes of needed CPU time, so be careful. 
The renice command adjusts the priority of a running process. For example, if we had launched the cpu-hog program and wanted to increase its niceness after the fact, we could do this: 
<html><body><table><tr><td colspan="2">[me@linuxbox ~]$ ps</td><td></td></tr><tr><td>PID TTY</td><td>TIME CMD</td><td></td></tr><tr><td>379087 pts/9</td><td></td><td>00:00: 00 bash</td></tr><tr><td>379215 pts/9</td><td>00:00:00 cpu-hog</td><td></td></tr><tr><td>379223 pts/9</td><td>00:00:00 ps</td><td></td></tr><tr><td></td><td>[me@linuxbox ~]$ renice -n 19 379215</td><td></td></tr></table></body></html> 
First, we run ps to determine the process id of the running cpu-hog program followed by the renice command with the desired niceness level and the process id. The niceness level of 19 (the maximum value) is useful as it makes the process only use CPU cycles when nothing else is waiting. 
# Signals 
The kill command is used to “kill” processes. This allows us to terminate programs that need killing (that is, some kind of pausing or termination). Here's an example: 
<html><body><table><tr><td>[me@linuxbox ~]$ xlogo &</td></tr><tr><td>[1] 28401 [me@linuxbox ~]$ kill 28401</td></tr><tr><td>[1]+ Terminated xlogo</td></tr></table></body></html> 
We first launch xlogo in the background. The shell prints the jobspec and the PID of the background process. Next, we use the kill command and specify the PID of the process we want to terminate. We could have also specified the process using a jobspec (for example, %1) instead of a PID. 
While this is all very straightforward, there is more to it than that. The kill command doesn't exactly “kill” processes: rather it sends them signals. Signals are one of several ways that the operating system communicates with programs. We have already seen signals in action with the use of $\tt c t r l-c$ and Ctrl-z. When the terminal receives one of these keystrokes, it sends a signal to the program in the foreground. In the case of Ctrlc, a signal called INT (interrupt) is sent; with Ctrl-z, a signal called TSTP (terminal stop) is sent. Programs, in turn, “listen” for signals and may act upon them as they are received. The fact that a program can listen and act upon signals allows a program to do things such as save work in progress when it is sent a termination signal. 
# Sending Signals to Processes with kill 
The kill command is used to send signals to programs. Its most common syntax looks like this: 
# kill [-signal] PID... 
If no signal is specified on the command line, then the TERM (terminate) signal is sent by default. The kill command is most often used to send the following signals: 
Table 10-5: Common Signals 
<html><body><table><tr><td>Number</td><td>Name</td><td>Meaning</td></tr><tr><td>1</td><td>HUP</td><td>Hangup. This is a vestige of the good old days when terminals were attached to remote computers with phone lines and modems. The signal is used to indicate to programs that the controlling terminal has “hung up." The effect of this signal can be demonstrated by closing a terminal session. The foreground program running on the terminal will be sent the signal and</td></tr><tr><td></td><td></td><td>This signal is also used by many daemon programs to cause a reinitialization. This means that when a daemon is sent this signal, it will restart and reread its configuration file. The</td></tr><tr><td></td><td></td><td>Apache web server is an example of a daemon that uses the HUP signal in this way. It's possible to make a process immune to the</td></tr></table></body></html> 
<html><body><table><tr><td colspan="2"></td><td>command which is discussed below.</td></tr><tr><td>2 9</td><td>INT</td><td>Interrupt. This performs the same function as a Ctrl-c sent from the terminal. It will usually terminate a program.</td></tr><tr><td></td><td>KILL</td><td>Kill. This signal is special. Whereas programs may choose to handle signals sent to them in different ways, including ignoring them all together, the KILL signal is never actually sent to the target program. Rather, the kernel immediately terminates the process. When a process is terminated in this manner, it is given no opportunity to “clean up" after itself or save its work. For this reason, the KILL signal should be used only as a last resort when other termination signals fail.</td></tr><tr><td>15</td><td>TERM</td><td>Terminate. This is the default signal sent by the kill command. If a program is still “alive" enough to receive signals, it will terminate.</td></tr><tr><td>18</td><td>CONT</td><td>Continue. This will restore a process after a STOP or TSTP signal. This signal is sent by the bg and fg commands.</td></tr><tr><td>19</td><td>STOP</td><td>Stop. This signal causes a process to pause without terminating. Like the KI LL signal, it is not sent to the target process, and thus it cannot be ignored.</td></tr><tr><td>20</td><td>TSTP</td><td>Terminal stop. This is the signal sent by the terminal when Ctrl - z is pressed. Unlike the STOP signal, the TSTP signal is received by the program, but the program may choose to ignore it.</td></tr></table></body></html> 
Let's try out the kill command: 
<html><body><table><tr><td>[me@linuxbox ~]$ xlogo &</td></tr><tr><td>[1]13546</td></tr><tr><td>[me@linuxbox ~]$ kill -1 13546</td></tr><tr><td>[1]+ Hangup xlogo</td></tr></table></body></html> 
In this example, we start the xlogo program in the background and then send it a HUP signal with kill. The xlogo program terminates, and the shell indicates that the background process has received a hangup signal. We may need to press the Enter key a couple of times before the message appears. Note that signals may be specified either by number or by name, including the name prefixed with the letters SIG. 
<html><body><table><tr><td>[me@linuxbox ~]$ xlogo &</td></tr><tr><td>[1]13601 [me@linuxbox ~]$ kill -INT 13601</td></tr><tr><td>[1]+ Interrupt xlogo</td></tr><tr><td>[me@linuxbox ~]$ xlogo &</td></tr><tr><td>[1] 13608</td></tr><tr><td>[me@linuxbox ~]$ kill -SIGINT 13608</td></tr><tr><td>[1]+ Interrupt xlogo</td></tr></table></body></html> 
Repeat the example above and try the other signals. Remember, we can also use jobspecs in place of PIDs. 
Processes, like files, have owners, and you must be the owner of a process (or the superuser) to send it signals with kill. 
In addition to the list of signals above, which are most often used with kill, there are other signals frequently used by the system as listed in Table 10-5. 
Table 10-6: Other Common Signals 
<html><body><table><tr><td>Number</td><td>Name</td><td>Meaning</td></tr><tr><td>3</td><td>QUIT</td><td>Quit.</td></tr><tr><td>11</td><td>SEGV</td><td>Segmentation violation. This signal is sent if a program makes illegal use of memory, that is, if it tried to write somewhere it was not allowed to write.</td></tr><tr><td>28</td><td>WINCH</td><td>Window change. This is the signal sent by the system when a window changes size. Some programs , such as top and less will respond to this signal by redrawing themselves to fit the new window dimensions.</td></tr></table></body></html> 
For the curious, a complete list of signals can be displayed with the following command: 
# Making a Process Hangup Proof 
As we discussed, above many command line programs will respond to the HUP signal by terminating when its controlling terminal “hangs up” (i.e. closes or disconnects). To prevent this behavior, we can launch the program with the nohup command. Here’s an example. 
<html><body><table><tr><td>[me@linuxbox ~]$ xlogo</td></tr></table></body></html> 
If we launch the xlogo program again then close our terminal window, the xlogo program will terminate because it is sent a HUP signal when its controlling terminal is closed. To prevent this we can launch xlogo with the nohup command like so: 
[me@linuxbox ~]$ nohup xlogo 
Now when we close the terminal window, xlogo will continue running. 
# Sending Signals to Multiple Processes with killall 
It's also possible to send signals to multiple processes matching a specified program or username by using the killall command. Here is the syntax: 
# killall [-u user] [-signal] name.. 
To demonstrate, we will start a couple of instances of the xlogo program and then terminate them. 
[me@linuxbox ~]$ xlogo & 
[1] 18801 
[me@linuxbox ~]$ xlogo & 
[2] 18802 
[me@linuxbox ~]$ killall xlogo 
[1]- Terminated xlogo 
$[2]+$ Terminated xlogo 
Remember, as with kill, we must have superuser privileges to send signals to processes that do not belong to us. 
# Shutting Down the System 
The process of shutting down the system involves the orderly termination of all the processes on the system, as well as performing some vital housekeeping chores (such as syncing all of the mounted file systems) before the system powers off. There are four commands that can perform this function. They are halt, poweroff, reboot, and shutdown. The first three are pretty self-explanatory and are generally used without any command line options. Here’s an example: 
The shutdown command is a bit more interesting. With it, we can specify which of the actions to perform (halt, power down, or reboot) and provide a time delay to the shutdown event. Most often it is used like this to halt the system: 
[me@linuxbox $-]$1$ sudo shutdown -h now or like this to reboot the system: 
[me@linuxbox $-]$1$ sudo shutdown -r now 
The delay can be specified in a variety of ways. See the shutdown man page for details. Once the shutdown command is executed, a message is “broadcast” to all logged-in users warning them of the impending event. 
# More Process-Related Commands 
Since monitoring processes is an important system administration task, there are a lot of commands for it. Table 10-6 lists some to play with: 
Table 10-7: Other Process Related Commands 
<html><body><table><tr><td>Command</td><td>Description</td></tr><tr><td>pstree</td><td>Outputs a process list arranged in a tree-like pattern showing the parent-child relationships between processes.</td></tr><tr><td>vmstat</td><td>Outputs a snapshot of system resource usage including, memory, swap, and disk I/O. To see a continuous display, follow the command with a time delay (in seconds) for updates. Here's an example: Vmstat 5. Terminate the output with Ctrl-c.</td></tr><tr><td>xload</td><td>A graphical program that draws a graph showing system load over time.</td></tr><tr><td>tload</td><td>Similar to the x load program but draws the graph in the terminal. Terminate the output with Ct rl-c.</td></tr></table></body></html> 
# Summing Up 
Most modern systems feature a mechanism for managing multiple processes. Linux provides a rich set of tools for this purpose. Given that Linux is the world's most deployed server operating system, this makes a lot of sense. However, unlike some other systems, Linux relies primarily on command line tools for process management. Though there are graphical process tools for Linux, the command line tools are greatly preferred because of their speed and light footprint. While the GUI tools may look pretty, they often create a lot of system load themselves, which somewhat defeats the purpose. 
## 1 What is the Shell
## 2 Navigation
## 3 Exploring the System
## 4 Manipulating Files and Directories
## 5 Working with Commands
## 6 Redirection
## 7 Seeing the World as the Shell Sees it
## 8 Advanced Keyboard Tricks
## 9 Permissions
## 10 Processes
# Part 2 Configuration and the Environment
## 11 The Environment
## 12 A Gentle Introduction to vi
## 13 Customizing the Prompt
# Part3 Common Tasks and Essential Tools
## 14 Package Management
## 15 Storage Media
## 16 Networking
## 17 Searching for Files
## 18 Archiving and Backup
## 19 Regular Expressions
## 20 Text Processing
## 21 Formatting Output
