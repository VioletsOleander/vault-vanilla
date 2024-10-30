# 1 Getting Started
## 1.1 About Version Control
版本控制系统即一个随时间记录文件修改的系统，我们可以用版本控制系统记录几乎计算机上几乎任何类型的文件的修改
### 1.1.1 Local Version Control Systems
RCS是一个较为流行的本地版本控制系统，RCS在磁盘上记录补丁集(patch sets)，即文件之间的差异，因此可以通过向原始文件添加补丁得到特定时间点的文件

### 1.1.2 Centralized Version Control Systems
中心化版本控制系统有一个中心服务器，服务器内包含所有版本的文件，客户端从服务器检出(check out)文件

### 1.1.3 Distributed Version Control Systems
分布式版本控制系统中，客户端不是仅仅只检出文件的最新快照(lastest snapshot)，而是完整镜像仓库(repository)，包括它的全部历史(history)
因此，即便服务器丢失信息，任意客户端都可以将仓库拷贝会服务器，将信息恢复
## 1.2 A Short History of Git
## 1.3 What is Git?
### 1.3.1 Snapshots, Not Differences
Git和其他版本控制系统的主要区别在于对待数据的方式
大多数其他版本控制系统存储的信息是一系列基于文件的修改，这些系统视它们存储的信息为一系列文件和随着时间对这些文件所做的修改
这一般被称为基于变化(delta-based)的版本控制
![[ProGit-Fig4.png]]

Git视数据为一个微型文件系统的一系列快照(snapshots)，Git中，每次提交，Git都会对我们的文件系统此时的样子拍个照，然后存下对这个快照的引用
如果文件没有被修改，Git不会重复存储，而是将其链接到之前已经存下的完全相同的文件
Git视自己的数据为快照流(stream of snapshots)
![[ProGit-Fig5.png]]

### 1.3.2 Nearly Every Operation Is Local
Git中大部分操作仅需要本地文件和资源即可完成，因为本地磁盘中有项目的整个历史，而中心化版本控制系统的大部分操作都有网络延迟开销
例如在Git中要查询一个文件的当前版本和一个月前的版本的差异，Git直接在本地查询一个月之前的文件，然后在本地做差异计算(difference calculation)，而不需向服务器请求

### 1.3.3 Git Has Integrity
Git中，文件在储存之前都会先计算校验和，并且之后通过该校验和索引，这意味着只要文件和目录有任何改动要储存下来，Git都会知道
这个功能内置于Git的最底层，由此构成了Git的完整性(integrity)，Git可以检测文件在传输时是否内容丢失或被污染

Git使用SHA-1哈希基于文件内容或目录结构计算校验和，SHA-1哈希生成包含40个字符的字符串，每个字符代表一位十六进制数字(0-9和a-f)
SHA-1哈希码举例：`24b9da6552252987aa493b52f8696cd6d3b00373`
Git不使用文件名索引文件，而是使用文件的哈希码索引文件
### 1.3.4 Git Generally Only Adds Data
对Git所做的大部分操作只会向Git的数据库中添加数据(add data to the Git database)，
因此，向Git提交一个快照后，它非常难以丢失，
同样，如果误操作了，也非常容易恢复(recover)
### 1.3.5 The Three Status
Git中的文件有三个主要状态：修改过的(modified)、暂存的(staged)、已提交的(committed)
- 修改过的即文件有修改，但尚未提交至数据库
- 暂存的即我们已经将一个修改过的文件标记(marked)为会进入我们的下一个提交快照(commit snapshot)
- 已提交的即数据已经安全储存于我们的本地数据库

一个Git项目有三个主要部分：工作树(working tree)、暂存区(staging area)、Git目录(Git directory)
![[ProGit-Fig6.png]]

工作树是项目的一个版本的检出(a single checkout of one version of the project)，即Git从Git目录中的压缩数据库(compressed database)中取出这些文件，置于磁盘中供我们使用或修改

暂存区是一个文件，通常也包含于Git目录中，用于存储我们的下一个提交(commit)会包含的信息

Git目录是存储我们的项目的元数据和对象数据库(object database)的地方，这是Git最重要的部分，当我们从其他主机克隆(clone)仓库时，就是复制了Git目录的内容

基本的Git工作流为
1. 工作树中的文件被修改
2. 选择性地选择一些想要被提交的修改到暂存区
3. 提交，这会将暂存区的文件取一个快照，然后在Git目录中永久存下这个快照

一个文件的特定版本存在了Git目录中，它就是已提交的(committed)，
一个文件被修改过并提交到了暂存区中，它就是暂存的(staged)，
一个文件被检出后进行了修改但未提交到暂存区，它就是修改过的(modified)
## 1.4 The Command Line
### 1.4.1 Installing Git
#### 1.4.1.1 Installing on Linux
Linux上，可以利用包管理工具(package management tool)进行下载
基于RPM的发布(distribution)，如Fedora、RHEL、CentOS使用 `dnf`
`sudo dnf install git-all`
基于Debian的发布，如Ubuntu、Debian使用 `apt`
`sudo apt intall git-all`
#### 1.4.1.2 Installing on macOS
#### 1.4.1.3 Insatlling on Windows
Windows上，需要下载Git for Windows
#### 1.4.1.4 Insatlling from Source
直接通过源代码下载Git方便得到最新的版本，因为二进制安装程序(binary installer)往往会慢于最新发布

要通过源代码下载Git，首先需要确认有Git需要的库，包括 `autotools` , `curl` , `zlib` , `openssl` , `expat` , `libiconv` 这些是编译Git需要的最小依赖
```shell
$sudo dnf install dh-autoreconf curl-devel expat-devel gettext-devel \
openssl-devel perl-devel zlib-devel
$sudo apt-get install df-autoreconf libcur14-gnutls-dev libexpat1-dev\
gettext libz-dev libssl-dev
```
如果需要通过不同格式阅读文档(doc, html, info)，需要下载额外依赖
```shell
$ sudo dnf install asciidoc xmlto docbook2X
$ sudo apt-get install asciidoc xmlto docbook2X install-info
```

之后，下载最新发布的源码包，本地编译即可
编译成功后，可以通过Git获取最新的Git源码以更新
`git clone https://git.kernel.org/pub/scm/git/git.git`
### 1.4.2 First-Time Git Setup
下载后，需要自定义Git环境(environment)，这些操作只需要执行一次，之后的更新会保持这些设置

Git包含了名为 `git config` 的工具，帮助我们设置和查看配置变量(configuration variable)，这些变量储存于三个不同的地方
1. `[path]/etc/gitconfig` 文件
	此处的设置应用于系统中所有用户和它们的仓库，
	对 `git config` 传递选项 `--system` 用于指明从这个文件读写配置
	这是个系统配置文件，读写需要管理员权限
2. `~/.gitconfig` 或 `~/.config/git/config` 文件
	此处的设置仅对自己的用户应用
	对 `git config` 传递选项 `--global` 用于指明从这个文件读写配置
	这里的配置影响该系统下自己的用户下所有的仓库
3. `.git/config` 文件(即自己的用户的任意仓库的Git目录下的 `config` 文件)
	此处的设置仅对自己的用户的特定仓库应用	
	对 `git config` 传递选项 `--local` 用于指明从这个文件读写配置，该选项为默认选项

低层级的配置会覆盖高层级的配置，例如 `local` 的配置优先级高于 `system` 的配置

Windows系统中，Git在 `$HOME` 目录中(一般是 `C:\Users\$USER` )寻找用户级别的配置文件 `.gitconfig` ，Git同样会在 `[path]/etc/gitconfig` 中寻找系统级别的配置文件，实际情况下这一般和Msys根有关，即我们运行安装程序时决定要将Git下载到的那个目录，Git for Windows 2.x以及以后的版本在 `C:\ProgramData\Git\config` 目录中也会有系统级别的配置文件
这些配置文件只能通过在相应权限下运行 `git config -f <file>` 修改

可以通过 `git config --list --show-origin` 查看所有的设置和它们的来源
#### 1.4.2.1 Your Identity
安装Git后首先需要设置用户名和邮箱地址，注意每次的提交都会用到这些信息
```shell
$ git config --global user.name "John Doe"
$ git config --global user.email johndoe@example.com
```
如果我们指定了 `--global` 选项，我们只需要设置这一次就好，Git会用这些信息标记我们自己的用户在系统中的一切行为
如果对某个特定的项目需要使用特定的信息，可以在项目中不指定 
`--global` 选项运行该命令即可

#### 1.4.2.2 Your Editor
设置好用户身份后，需要设置当Git需要我们输入信息时所使用的默认文本编辑器，例如 `$ git config --global core.editor emacs`
如果没有手动配置，Git会使用系统默认的编辑器

Windows中，如果需要设置编辑器，需要指明到其可执行文件的完整路径
#### 1.4.2.3 Your default branch name
默认情况下，Git会在我们用 `git init` 创建一个新的仓库时创造一个名为 `master` 的分支

版本2.28以后，可以自行设置默认初始分支的名字，例如
`git config --global init.defaultBranch main`
#### 1.4.2.4 Checking Your Settings
可以使用 `git config --list` 列出当下Git可以找到的所有配置，其输出的格式为 `key = value`

我们可能可以看到相同的key多次，因为Git会从不同的配置文件(如 `[path]/etc/gitconfig` , `~/.gitconfig` )中读到相同的key，Git会将key的最后遇到的值作为其最终值

可以通过 `$ git config <key>` 查看key对应的值，
例如 `git config user.name`

也可以查看最终是哪个配置文件决定了某个key的值
`git config --show-origin <key>` ，
其输出的格式为 `file: <file-path> <value>`
### 1.4.3 Getting Help
查看任意Git命令的手册可以通过
```shell
$ git help <verb>
$ git <verb> help
$ man git-<verb>
```
例如查看 `git config` 命令的手册，可以输入 `git help config`

使用 `git <command-name> -h` 可以查看较精简的用法说明
## 1.5 Summary
# 2 Git Basics
## 2.1 Getting a Git Repository
得到Git仓库有两种方式
1. 将一个并没有进行版本控制的本地目录转为一个Git仓库
2. 从其他地方克隆一个现存的Git仓库
### 2.1.1 Initializing a Repository in an Existing Directory
可以从一个现存的未经版本控制的目录初始化一个Git仓库

在需要初始化为Git仓库的目录下运行 `git init` ，这会在当前目录下创建一个新的名为 `.git` 的子目录，子目录内包含了所有我们必需的仓库文件——一个Git仓库框架(a Git repository skeleton)

目前，我们的项目中还没有文件被追踪(tracked)
如果需要开始对现存的文件进行版本控制，我们就需要开始追踪这些文件，并做一次提交
使用 `git add <file-name>` 用以指明需要追踪的文件，如
`git add *.c`
`git add LICENSE`
使用 `git commit` 以执行提交，如
`git commit -m 'Initial project version'`
### 2.1.2 Cloning an Existing Repository
如果我们需要一个现有Git仓库的拷贝，比如需要贡献一个项目，
可以使用 `git clone`
`git clone` 会将仓库进行一次完全的拷贝，默认情况下，项目的所有文件的所有版本历史都会被拉取，而不是仅仅得到一个正好可以工作的拷贝

克隆的用法是 `git clone <url>` ，例如
`git clone https://github.com/libgit2/libgit2`
会创建一个名为 `libgit2` 的目录，并在其中初始化一个 `.git` 子目录，然后将url对应仓库的所有数据拉取，最后检出一个最新版本的可工作的拷贝(working copy)

也可以指定目录名称，如
`git clone https://github.com/libgit2/libgit2 mylibgit`

Git可以使用不同的传输协议，之前的例子使用 `https://` 协议，而使用 `git://` , `user@server:path/to/repo.git` 则使用SSH协议
## 2.2 Recording Changes to the Repository
工作目录中的任何一个文件只能是两种状态其一：被追踪和未被追踪(tracked or untracked)

被追踪的文件即是包含在最新的快照(last snapshot)其中的文件，或者新暂存的文件，被追踪的文件的状态可以是未修改的、已修改的、暂存的，简单来说被追踪的文件就是Git知道要进行版本管理的文件
未被追踪的文件即工作目录中不包含在最新的快照和暂存区内的文件，当我们第一次克隆一个仓库时，仓库内的所有文件都是被追踪的且未修改的，因为Git仅仅是将它们检出，而我们尚未修改

我们对文件进行修改后，Git就视文件为已修改的，因为文件和上一次提交相比已经有了变化
我们选择性地暂存已修改的文件，然后提交
![[ProGit-Fig8.png]]
#### 2.2.1 Checking the Status of Your Files
`git status` 命令用于查看文件状态

如果在克隆后直接运行 `git status` ，Git会告诉我们工作树是干净的，即我们所追踪的文件都是未修改的，未追踪的文件对Git是不可见的
同时，Git会告诉我们现处的分支，一般是 `master` ，并且目前分支和服务器上的相同分支没有分歧(divergence)

现在添加一个新文件 `README` 至项目，再运行 `git status` ，Git会列出未被追踪的文件(Untracked files)
一个不在之前的快照/提交内的文件并且没有被暂存的文件会被Git视为未追踪的
### 2.2.2 Tracking New Files
`git add` 命令用于追踪新文件，如
`git add README`

现在 `git status` 将会告诉我们该文件已被追踪，并且已经暂存，等待提交(Changes to be committed)
`git add` 命令的参数是一个文件的路径或一个目录的路径，如果参数是目录，则递归追踪目录内所有文件
### 2.2.3 Staging Modified Files
如果对一个已追踪的文件例如 `CONTRIBUTING.md` 修改，`git status` 会告诉我们该文件已被追踪且已被修改，但尚未暂存(Changes not staged for commit)

要暂存该文件，同样使用 `git add` 命令，如
`git add CONTRIBUTING.md`
`git add` 是一个多目的命令，它可以用于追踪文件、暂存文件、标记合并冲突的文件已解决
其中的add可以理解为将文件添加至下一个提交

此时 `CONTRIBUTING.md` 在 `git status` 中显示为Changes to be committed，如果我们先不提交，而是再打开 `CONTRIBUTING.md` 进行一次修改，然后运行
`git status` ，
我们会发现 `CONTRIBUTING.md` 文件即在Changes to be committed中，也在Changes not staged for commit中

事实上，在我们运行 `git add` 把 `CONTRIBUTING.md` 暂存时，Git将该文件的快照置于暂存区，之后，即便我们在工作目录中对 `CONTRIBUTING.md` 进行了修改后再提交，提交的文件内容也会是之前运行 `git add` 时的快照内容，而不包括新的修改

因此，如果我们再运行了 `git add <file>` 后又对 `<file>` 进行了修改，我们需要在提交前再次运行 `git add` 将该文件的最新版本暂存
### 2.2.4 Short Status
如果认为 `git status` 的输出过于冗长，可以通过 `git status -s` 或 `git status --short` 得到更紧凑的输出，
其中，文件名之前是 `??` 表示文件是新文件，且尚未提交到暂存区，即尚未追踪
文件名之前是 ` M` 表示在工作目录已修改，尚未提交到暂存区
文件名之前是 `M ` 表示文件已修改，并提交到了暂存区
文件名之前是 `MM` 表示文件已修改，并提交到了暂存区，但之后又在工作区修改过
文件名之前是 `A ` 表示文件是新文件，并提交到了暂存区
可以发现输出前缀有两列，第一列表示暂存区状态，第二列表示工作树状态
### 2.2.5 Ignoring Files
一般会有一类我们不希望Git会自动追踪甚至Git会显示为未追踪状态的文件，这些文件通常是生成的日志文件或者编译系统构建的文件
这种情况下，我们可以将这类文件的模式写在 `.gitignore` 文件中
```shell
$ cat .gitignore
*.[oa]
*~
```
第一行告诉Git忽略一切以 `.o` 或 `.a` 结尾的文件，即作为编译系统输出的目标(object)文件和文档(archive)文件
第二行告诉Git忽略一切命名以飘号 `~` 结尾的文件，一般是文本编辑器创建的暂时文件

写好 `.gitignore` 方便我们避免意外将不必要的文件提交到我们的Git仓库中

`.gitignore` 中的模板(pattern)规则如下
- 空行以及以 `#` 开头的行会被忽略
- 标准的glob模式(pattern)适用，并且默认会被递归地在整个工作树内进行匹配
- 以 `/` 作为模式的开头以避免递归运用
- 以 `/` 作为模式的结尾以指明该模式匹配的是目录
- 以 `!` 作为一个模式的开头表示对它的否定

glob模式即shell所采用的简化的正则表达式，其中 `*` 匹配零到多个字符，`[]` 匹配框内的任意字符，例如 `[abc]` ，`?` 匹配单个字符，`[ - ]` 匹配短横范围内的任意字符，例如 `[0-9]` ，两个 `*` 可以用于匹配嵌套的目录，例如 `a/**/z` 匹配 `a/z` ，`a/b/z` ，`a/b/c/z`

一般情况下只需要在根目录维护一个 `.gitigonre` 文件即可，但也可以在子目录中维护额外的 `.gitigonre` ，子目录的 `.gitignore` 只对该目录及其子目录起作用
### 2.2.6 Viewing Your Staged and Unstaged Changes
`git status` 可以告诉我们哪些文件是我们修改过但还未暂存的，哪些文件是我们已经暂存但还未提交的，`git status` 会通过列出文件名指出相应的文件
而 `git diff` 则更具体地告诉我们文件中具体哪些内容发生了修改

假设我们修改了 `README` 并暂存，修改了 `CONTRIBUTING.md` 尚未暂存，
使用 `git status` ，我们可以发现 `README` 在Changes to be committed，同时 `CONTRIBUTING.md` 在Changes not staged for commit

如果此时我们想知道具体有哪些完成的但尚未暂存的修改，直接运行 `git diff` ，该命令就会比较工作目录中的文件(修改后的文件)和暂存区的文件(修改前的文件)，以此告诉我们哪些修改我们完成了但尚未暂存
如果所有的修改都已暂存，则 `git diff` 不会有输出
如果暂存了 `CONTRIBUTING.md` 之后，又进行了修改，且新修改没有暂存，
`git diff` 会显示仍未暂存的新修改

如果此时我们想知道具体有哪些已经暂存的但尚未提交(等待提交)的修改，运行
`git diff --staged` 或 `git diff --cached` ，该命令就会比较暂存区的文件和上一次提交的文件，以此告诉我们这次提交相对于上一次提交会有哪些修改
如果暂存了 `CONTRIBUTING.md` 之后，又进行了修改，且新修改没有暂存，
`git diff --staged` 的输出不会有变化
### 2.2.7 Committing Your Changes
`git commit` 用于将已暂存的修改提交，
直接运行 `git commit` 会根据我们的shell的 `EDITOR` 环境变量启动相应的编辑器，显示 `git status` 命令的输出，并提示输入提交信息(这些提示信息默认被注释)

如果运行 `git commit -v` ，提示信息还会包含diff的信息，让我们显式地看到这次会提交的修改信息

编辑好提交信息并退出编辑器，Git就会进行这次提交

在 `git commit` 命令中使用 `-m` 选项也可以加入提交信息

成功提交后，Git会在输出中显示我们提交到了哪一个分支(如master)，本次提交的SHA-1检验和(如463bcdf)，多少文件被修改过，以及本次提交中有多少行被加入，多少行被移除

提交即让Git记录了我们在暂存区设立的快照(snapshot)，没有暂存的文件是与提交无关的，我们利用提交记录我们项目目前的一个快照
### 2.2.8 Skipping the Staging Area
在 `git commit` 中添加 `-a` 选项可以让Git自动将目录中所有已经正在被追踪的文件被暂存，然后执行提交，相当于免去了我们手动执行 `git add` 暂存的麻烦
### 2.2.9 Removing Files
要在Git中移除一个文件，我们首先需要将其从我们已追踪文件的列表中移除(更准确地说即从暂存区移除)，然后执行提交

如果我们仅仅将其从工作目录中移除，Git会视其为一个未暂存的修改(Changes not staged for commit)
而我们运行 `git rm <file-name>` ，Git会先将其从工作目录移除，然后Git会将该移除(removal)暂存 

要移除一个已经提交到暂存区的文件，需要在 `git rm` 中添加 `-f` 选项，这是为了避免意外移除可能想要记录的文件

如果我们想要在工作树中保持文件，当想将其从暂存区移除，换句话说，我们希望文件仍保存在硬盘中，但不希望Git再追踪它，我们可以使用 `--cached` 选项，这在我们忘记把文件加入 `.gitignore` 或意外提交了文件时很有用

`git rm` 命令接收文件、目录、file-glob pattern作为参数，
例如 `git rm log/\*.log`
注意 `*` 前的 `\` 是必须的(This is necessary because Git does its own filename expansion in addition to your shell’s filename expansion)
该命令移除了 `log/` 目录中所有以 `.log` 为拓展名的文件
又例如 `git rm \*~` 会移除所有文件名以 `~` 结尾的文件
### 2.2.10 Moving Files
和其他版本控制系统不同，Git不显式追踪文件移动(movement)，如果我们在Git中重命名了一个文件，Git不会存在关于我们命名了文件的元数据

Git有 `mv` 命令，可以用于重命名文件，如
`git mv file_from file_to`
此时运行 `git status` 会发现Git将其标记为renamed的文件

运行 `git mv README.md README` 等价于运行 
```
mv README.md README
git rm README.md
git add README
```
Git会发现这隐式上就是一次重命名
## 2.3 Viewing the Commit History
`git log` 可以用于查询提交历史信息

没有添加额外参数时，`git log` 默认按逆时间序列出历史提交，关于提交的信息有它的SHA-1检验和，作者名和邮箱，日期，以及提交信息

添加 `-p` 或 `--patch` 选项可以展示每次提交的差异，添加 `-<number>` 选项可以控制需要展示的日志项(log entry)，如 `-2` 就限制只展示最后两次提交
添加 `--stat` 选项可以展示每次提交的一些统计信息，包括了被修改过的文件的名称，每个文件内有多少行被添加和移除，共有几个文件被改变等

选项 `--pretty` 用于改变日志的输出格式，我们可以选择一些预置的选项值，例如选项值 `oneline` 会使得每个提交仅占一行，又例如选项值 `short` , `full` , `fuller` 都大致以相同的格式展示日志，但分别有更少或更多的信息
选项值 `format` 允许我们指定自己喜好的输出格式

`git log --pretty=format` 的一些指示符(specifier)如下
![[ProGit-Table1.png]]
使用实例
```
git log --pretty=format:"%h - %an, %ar : %s
```

Git中，作者(author)和提交者(commiter)之间有所区别，作者是最初编写工作的人(originally wrote the work)，而提交者是最后应用工作的人(last applied the work)，例如，如果我们向一个项目发送了一个补丁，并且其中一个核心成员应用了该补丁，则我们就作为作者，核心成员作为提交者

`oneline` 和 `format` 的一些其他选项值配合 `--graph` 选项十分有用，`--graph` 选项会用ASCII字符绘图，展示我们的分支和合并历史，例如
```
git log --pretty=format:"%h %s" --graph
```

以上所介绍的只是 `git log` 指令的格式化输出选项中的一部分，一些常用的选项总结如下图
![[ProGit-Table2.png]]
### 2.3.1 Listing Log Output
除了输出格式化的选项，`git log` 还接收一些限制性选项(limiting)，即用于仅展示所有提交的一个子集(a subset of commits)的选项

`-2` 选项就是其中之一，用于仅展示最新的两个提交，我们可以使用 `-<n>` ，其中 `n` 是任意整数，以仅展示最新的$n$个提交
实际使用时，一般不会经常使用这种方法，因为默认情况下 Git 将所有的输出通过页导航展示(pipes all output through a pager)，因此一次只能看到一页的日志输出

但时间限制性选项则非常有用，例如 `--since` 和 `--until`
例如该命令用于得到最新两周的提交
```
git log --since=2.weeks
```
该命令接收许多种格式的时间，可以指明确切的日期例如 `"2008-01-15"` ，或一个相对的时间例如 `"2 years 1 day 3 minutes ago"`

限制性选项还可以用于根据特定的规则过滤出需要的提交记录，例如 `--author` 选项用于过滤出指定作者的提交，`--grep` 选项用于过滤出提交信息中含有匹配关键词的提交
向 `--author` 和 `--grep` 选项提供多个参数，过滤规则是满足其中任一即可，再额外添加 `--all-match` 选项，则过滤规则就是需要全部满足

还有一个有用的限制性选项是 `-S` ，该选项接收一个字符串，然后过滤出所有改变了该字符串的出现次数的提交，
例如，如果我们想要找到哪一次提交添加了或移除了对某个函数的引用，我们可以调用
```
git log -S function_name
```

我们还可以向 `git log` 传递路径作为过滤选项，如果我们指明了一个目录或文件名，日志输出就会只包含对这些文件进行了修改的提交，该选项一般放在最后，且路径前要有 `--` ，例如
```
git log -- path/to/file
```

一些常用的限制性选项总结如下
![[ProGit-Table3.png]]

有时我们的日志历史记录中相当大比例的提交可能只是合并提交，这通常不会提供很多信息，要防止合并提交显示，只需添加选项 `no-merge`
## 2.4 Undoing Things
Git提供了撤销操作，但要注意的是我们不能总是撤销我们的一些撤销操作，这也是Git中如果误操作就会失去一些工作的为数不多的几个领域之一

最常见的撤销操作发生于我们过早提交而忘记添加一些文件，或者我们写错了提交信息，如果我们需要重新进行一次提交，我们需要先完成需要的修改，然后将其暂存，然后再一次提交，使用 `--amend` 选项
```
git commit --amend
```
此时我们可以修改之前的提交信息，这次修改会将之前的提交信息覆盖写
如果我们只是想修改提交信息，就什么也不暂存，直接提交即可

示例
```
git commit -m 'Initail commit'
git add forgotten_file
git commit --amend
```

对上一次提交进行修改不会在日志中产生新的提交项
### 2.4.1 Unstaging a Staged File
对于已经暂存的文件，可以使用 `git reset HEAD <file>` 取消暂存(unstage)
例如 `git reset HEAD CONTIRBUTING.md`
### 2.4.2 Unmodifying a Modified File
如果我们想要取消对某个文件的修改(unmodifiy)，即将文件恢复为上次提交的时候的样子(或最初克隆下来的样子)，可以使用 `git checkout -- <file>`
例如 `git checkout -- CONTRIBUTING.md`

注意 `git checkout -- <file>` 实际上是一个危险的命令，这会让我们对文件做的任何本地修改丢失，Git会直接检出上次提交的文件，替换本地文件
任何已经提交到Git的文件几乎可以总是被恢复，即便是在已经删除的分支上的提交，又或者是用 `--amend` 覆盖写了的提交，对应的，任意没有提交的信息丢失了就再也找不回了
### 2.4.3 Undoing things with git restore
Git v2.23.0引入了 `git restore` 命令，该命令基本上是 `git reset` 的替代，从v2.23.0开始，Git的许多撤销操作将使用 `git restore` 而非 `git reset`  
#### 2.4.3.1 Unstaging a Staged File with git restore
`git restore --staged <file>` 可以用于取消暂存某个文件，例如
`git restore --staged CONTRIBUTING.md`
#### 2.4.3.2 Unmodifying a Modified File with git restore
`git restore <file>` 可以用于取消对某个文件的修改，例如
`git restore CONTRIBUTING.md`

注意 `git restore` 实际上是一个危险的命令，这会让我们对文件做的任何本地修改丢失，Git会直接检出上次提交的文件，替换本地文件
## 2.5 Working with Remotes
远程仓库(remote repositories)是托管在Internet上的Git仓库，远程仓库可以是只读的，也可以是可读写的
可以在远程仓库与他人协作，包括管理这些远程仓库，将我们需要共享的工作推送至远程仓库，以及从远程仓库拉取数据

对远程仓库进行管理的内容包括如何添加远程仓库，如何移除远程仓库，如何管理多个远程分支，以及定义远程分支是否被追踪

远程仓库事实上不一定不在本机上，以本机为主机也可以进行完全相同的远程仓库管理内容
### 2.5.1 Showing Your Remotes
`git remote` 命令用于查看配置了哪些远程服务器，它会列出我们指定的每个远程句柄的简称(shornames for each remote handles)
当我们从某个服务器克隆了一个远程仓库到本地时，Git给仓库的来源服务器的默认简称是 `origin`

如果指定了 `-v` 选项，可以看到简称实际对应的URL(在对远程仓库读写时，使用的就是URL)
一个Git仓库可以和多个远程仓库合作，`git remote` 会列出所有相关的远程仓库，这些远程仓库可以使用不同的协议(protocals)
### 2.5.2 Adding Remote Repositories
我们知道 `git clone` 会隐式地添加名为 `origin` 的远程仓库，我们也可以显示地添加远程仓库，使用 `git remote add <shorname> <url>`
例如 `git remote add pb https://github.com/paulboone/ticgit`

此时我们已经可以用字符串 `pb` 代替整个URL，例如我们想要获取仓库 `pb` ，可以使用 `git fetch pb`
此时 `pb` 仓库的 `master` 分支会变为本地仓库的 `pb/master` 分支，`pb` 仓库的 `ticgit` 分支会变为本地仓库的 `pb/ticgit` 分支
### 2.5.3 Fetching and Pulling from Your Remotes
`git fetch <remote>` 命令用于从远程仓库获取数据，该命令将远程仓库的所有本地仓库目前没有的数据拉取

我们知道 `git clone <remote>` 会自动添加 `<remote>` 且命名为 `origin` ，因此 `git fetch origin` 会将 `origin` 中自从上次克隆/拉取后新的提交拉取到本地，`git fetch` 只用于将数据下载到本地，不会对本地的其他数据有任何影响

如果我们将本地的当前分支设置为追踪(track)一个远程分支，则 `git pull` 命令会在拉取远程分支后自动将其与本地当前分支合并(merge)

`git clone` 命令会自动将本地的 `master` 设置为追踪远程的 `master` 分支
### 2.5.4 Pushing to Your Remotes
`git push <remote> <branch>` 用于将本地分支推送到远端
例如，如果我们想将 `master` 分支推送到 `origin` 服务端，可以运行
`git push origin master`
该命令只有在我们对 `origin` 具有写权利且与此同时没有其他人进行推送时才会生效，如果有其他人和我们同时 `clone` 了 `origin` ，且之后其他人先进行了一次 `push` ，则我们的 `push` 会被拒绝，我们需要先获取其他人推送的工作，才能被允许 `push` 
### 2.5.5 Inspecting a Remote
`git remote show <remote>` 用于展示远端的详细信息，例如
`git remote show origin`

该命令会列出远端仓库的URL，以及正在追踪的远端分支的信息，
该命令还会显示当我们在特定分支运行 `git push` 时，该分支会被 `push` 到远端的哪一个分支，以及当我们在特定分支运行 `git pull` 时，会自动拉取远端的哪个分支并自动与本地的当前分支合并，以及远端的哪些分支已经移除，而本地还存在，以及远端的哪些分支本地还没有
### 2.5.6 Renaming and Removing Remotes
`git remote rename` 用于重命名远端仓库，例如
`git remote rename pb paul` 将 `pb` 重命名为 `paul`

注意这会改变我们所有相关的正在远程追踪的分支的名称，例如分支名 `pb/master` 会改变为 `paul/master`

如果需要移除对某个远端仓库的引用(即本地移除)，可以使用 `git remote remove` 或 `git remote rm` ，例如 `git remove paul` ，这会删除所有的香断的正在远程追踪的分支，以及相关的配置设定
## 2.6 Tagging
和多数VCS一样，Git可以将仓库历史中的某个特定点标记为重要的，一般人们用该功能标记发布点(release points)(v1.0, v2.0等)
### 2.6.1 Listing Your Tags
`git tag [-l | --list]` 用于列出现存的标签，输出以字典序排序，因此排列顺序没有特殊含义

我们可以搜索匹配某个特定模式的标签，例如
`git tag -l "v1.8.5*"`
其中 `-l` 或 `--list` 选项是必需的
### 2.6.2 Creating Tags
Git支持两种类型的标签：轻量级的标签和带注释的标签(lightweight and annotated)
轻量级标签非常类似于不会改变的一个分支——它只是一个指向特定提交的指针(pointer to a specific commit)

带注释的标签，则作为完全的对象储存在Git数据库中，它们会被计算检验和；包括了标记者名称(tagger name)，邮箱，日期；有一条标记信息(tagging information)；可以被签名(signed)且被GNU Privacy Guard/GPG验证
因此一般推荐创建带注释的标签，但如果只想创建暂时的标签，或不想保存这些信息，就创建轻量级标签
#### 2.6.2.1 Annotated Tags
使用 `git tag -a` 可以创建带注释的标签，例如
`git tag -a v1.4 -m "my version 1.4"`
其中 `-m` 选项用于添加标记信息/注释，标记信息会和标签一起存储，如果没有使用 `-m` 选项，Git会启动默认的文本编辑器

使用 `git show` 命令可以查看标签相关数据，以及所标记的提交的信息，例如
`git show v1.4`
展示的信息包括标记者信息，提交被标记的日期，注释信息，以及提交本身的信息
#### 2.6.2.2 Lightweight Tags
轻量级标签可以基本上认为就是存在一个文件里的提交的检验和，轻量级标签不会保存其他任何信息
要创建一个轻量级标签，不要使用 `-m` , `-a` , `-s` 选项，只提供标签名称，
例如 `git tag v1.4-lw`

使用 `git show v1.4-lw` 时，我们不会看到多于的信息，只会看到提交本身的信息
### 2.6.3 Tagging Later
我们不仅可以对当前提交贴标签，也可以对历史提交贴标签，只需要在命令结尾指明对应提交的检验和(或检验和的一部分)，例如
`git tag -a v1.2 9fceb02`
### 2.6.4 Sharing Tags
默认情况下，`git push` 不会将标签迁移到远端服务器，在我们创建了标签后，我们需要显式地将标签推送到远端，命令格式和推送本地分支类似，即
`git push <remote> <tagname>` ，例如
`git push origin v1.5`

如果想要一次性推送多个标签，使用 `--tags` 选项，这会将本地有而远端没有的标签一次性都迁移到远端，例如
`git push origin --tags`

注意，`git push <remote> --tags` 会推送带注释的标签和轻量级标签，如果想只推送带注释的标签，使用 `git push <remote> --follow-tags` ，目前没有命令用于只推送轻量级标签
### 2.6.5 Deleting Tags
`git tag -d <tagname>` 用于删除本地仓库的标签
例如 `git tag -d v1.4-lw`
注意这不会删除任意远程服务器上的标签，要删除远程服务器上的标签，有两种方式：
第一种方式为 `git push <remote>:refs/tags/<tagname>`
例如 `git push origin :refs/tags/v1.4-lw` ，即将 `:` 前的空值推送到远端的标签上，因此删除了该标签
第二种方式为 `git push origin --delete <tagname>`
### 2.6.6 Checking out Tags
我们可以用 `git checkout <tagname>` 检出某个标签对应的提交版本，注意这会让我们的仓库变为“detached HEAD”状态
例如 `git checkout v2.0.0`

在“detached HEAD”状态，如果我们做了修改并进行了提交，我们的标签名不会变，但我们的新提交不会属于任何分支，且只能用提交哈希码索引，
因此，如果我们想要做出修改，例如我们要修补老版本中的一个bug，一般的实践是创建一个分支，在分支中进行提交
`git checkout -b version v2.0.0`
## 2.7 Git Aliases
Git允许我们通过 `git config` 为命令设置别名，例如
```
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
```
此时我们就可以用 `git ci` 替代 `git commit`

`git config` 还可以用于创造命令，例如
`git config --global alias.unstage 'reset HEAD--'`
此时 `git unstage fileA` 就等价于 `git reset HEAD-- fileA`

又例如 `git config --global alias.last 'log -1 HEAD'`
此时 `git last` 就可以列出最新的提交信息

如果要作别名的命令不是Git的子命令，而是外部的命令，需要在命令前加上 `!` 符号，例如 `git config --global alias.visual '!gitk'`
# 3 Git Branching
## 3.1 Branches in a NutShell
我们知道，Git不用一系列的修改集或差异集的方式存储数据，而是以一系列快照的方式存储数据

当执行一次提交后，Git会存储一个提交对象(commit object)，对象中包含了一个指针，指向我们暂存的内容的快照；包含了作者名和邮箱地址；包含了提交信息；包含了指向该提交之间的提交的指针(它的父提交)
初始提交没有父提交，普通的提交有一个父提交，由多个分支合并产生的提交有多个父提交

假设我们有一个包含了三个文件的目录，我们将它们全部暂存然后提交，
暂存文件时，Git做的事情包括：为每个文件计算SHA-1检验和，将该版本的文件存入Git仓库中(Git称其为二进制大对象/blobs)，然后将检验和加入暂存区

当我们用 `git commit` 提交时，Git会为每个子目录计算检验和(在本例中，只有项目根目录)，然后在Git仓库中存储一个树对象(tree object)，树对象包含了目录的内容(目录结构以及文件名)和指向各个blob的指针，之后创建一个包含了元数据和指向树对象的指针的提交对象

此时Git仓库中有5个对象：三个blobs，其中每个都包含了对应文件的内容；一个树对象，包含了目录的内容，并将文件名和对应的blob关联；一个提交对象，包含了元数据和指向树对象的指针
![[ProGit-Fig9.png]]

如果我们在此次提交后再进行了一些修改，然后再进行一次提交，下一次的提交会存储指向本次提交(它的前一个提交)的指针

Git中的分支本质上就是一个轻量的可移动的指针，指针指向这些提交中的其中一个提交，Git中默认的分支名称是 `master` ，我们在Git仓库开始进行提交时，就会得到一个指向我们最近的提交的名为 `master` 的分支，每次我们进行提交，`master` 分支指针也会自动向前移动

注意 `master` 分支和其他任何分支没有区别，不存在特殊性，近乎所有Git仓库都有 `master` 分支的原因是 `git init` 命令在默认情况下会创建名为 `master` 的分支
### 3.1.1 Creating a New Branch
在我们创建新分支的时候，Git会为我们创建一个可以四处移动的新指针，
例如，我们想创建名为 `testing` 的新分支，通过 `git branch` 命令：
`git branch testing`

这会创建一个新的指针，指向我们当前处于的提交
![[ProGit-Fig12.png]]

Git本身会维护一个特殊的指针，称为 `HEAD`，`HEAD` 指针指向我们目前所处的本地分支，因此Git通过 `HEAD` 指针知道我们当前所处的分支是什么
![[ProGit-Fig13.png]]


我们创建了 `testing` 分支以后，我们其实仍处于 `master` 分支上，因为 `git branch` 命令只创建一个新的分支，但不会切换到那个分支

我们可以用 `git log` 查看各个分支指针现在指向哪个提交：
`git log --oneline --decorate`
### 3.1.2 Switching Branches
`git checkout` 命令用于切换到一个现存的分支，例如：
`git checkout testing`
这使得 `HEAD` 指针指向 `testing` 分支
![[ProGit-Fig14.png]]
切换到新的分支以后，我们做一次新的提交
![[ProGit-Fig15.png]]
可以看到 `testing` 分支指向了新的提交，也就是向前移动了，而 `master` 分支还指向了我们运行 `git checkout` 时它所处的提交

注意 `git log` 命令默认情况下只会展示我们当前所处的分支的提交历史，如果我们需要查看特定分支的提交历史，则显式地使用 `git log <branch-name>` ，要展示所有分支的提交历史，添加 `--all` 选项

我们再切换回 `master` 分支：
`git checkout master`
![[ProGit-Fig16.png]]
此时该命令做了两件事：它将 `HEAD` 指针改变，使其指向了 `master` 分支；它将我们工作目录中的文件复原为 `master` 分支所指向的快照

此时我们再进行一些修改并进行提交，我们的项目就会出现分歧(diverge)，朝向两个不同的方向
![[ProGit-Fig17.png]]
`git log --oneline --decorate --graph --all` 可以在命令行展示类似的图像

在Git中，一个分支仅仅是一个包含了它所指向的提交的40个字符的SHA-1检验和的简单文件，因此创建或删除分支都是很廉价的，创建一个新分支等价于为一个文件写入41个字符(40个字符的检验和以及一个换行符)

另外，由于我们在提交时都会记录父提交，因此在合并时，Git可以自动找到恰当的合并基(merging base)，这些特性鼓励开发者多使用分支

如果我们要创建新分支，同时切换到这一分支，我们可以使用
`git checkout -b <newbranchname>`

Git 2.23版本以后，我们可以使用 `git switch` 替代 `git checkout` ：
- 切换到一个现存的分支：`git switch <branch-name>`
- 创建新分支并切换到它：`git switch -c <new-branch-name>` ，其中 `-c` 标志表示创建，我们也可以用全称 `--create`
- 回到之前检出(check out)的分支：`git switch -`
## 3.2 Basic Branching and Merging
### 3.2.1 Basic Branching
假设我们有一个项目，在 `master` 分支上已经有了几个提交
![[ProGit-Fig18.png]]

此时我们需要解决issue53，因此我们创建新的分支并切换到该分支
`git checkout -b iss53`
这等价于
`git branch iss53`
`git checkout iss53`

此时我们进行了一些修改，并提交，`iss53` 分支向前移动
![[ProGit-Fig20.png]]

此时，我们又需要解决一个新的需求，我们需要首先切换回 `master` 分支，
需要注意的是，在Git中，在切换分支的时候，如果我们的工作目录或暂存区域中有和我们即将检出的分支相冲突的修改，Git会阻止我们切换分支
因此，在切换分支的时候，需要保证有干净的工作状态(clean working state)，对应的方法有stash和commit amend

此时，因为我们已经提交了所有的修改，未做更多的修改，我们可以顺利切换到 `master` 分支
`git checkout master`
此时我们会发现工作目录的状态回到了 `master` 分支最后一次提交的状态

现在，我们创建一个新的分支
`git checkout -b hotfix`
然后我们进行一些修改然后提交
![[ProGit-Fig21.png]]

之后我们对该提交进行了测试，确认无误后，希望将这个分支合并到 `master` 分支，我们需要使用 `git merge` 命令
`git checkout master`
`git merge hotfix`

因为我们要合并的分支 `hotfix` 所指向的提交 `C4` 直接在我们当前所处的提交 `C2` 的前面，因此，Git所做的就是简单地将指针前移
![[ProGit-Fig22.png]]
在Git中，如果我们想要将一个提交合并入当前提交，如果要合并的提交可以直接通过提交历史回溯到当前提交，Git就会简单地将当前分支的指针前移到要合并的提交，因为此时不存在分歧的工作(divergent work)，这被称为“快速前移 fast forward”

在合并之后，我们可以删除 `hotfix` 分支了，因为已经不再需要它了，`master` 现在和它指向同一位置，我们通过 `-d` 选项删除分支
`git branch -d hotfix`

此时我们可以切换回 `iss53` 分支，继续之前的工作
`git checkout iss53`
![[ProGit-Fig23.png]]

值得注意的是我们在 `hotfix` 分支所做的工作并不会在 `iss53` 中包含，如果我们需要在 `iss53` 中也包含 `hotfix` 所做的工作，则需要将 `master` 分支合并到 `iss53` 分支，通过
`git merge master`
### 3.2.2 Basic Merging
假设我们完成了在 `iss53` 中的工作，决定将其合并入 `master` 分支，通过
`git checkout master`
`git merge iss53`

此时，由于我们的开发历史从某个点开始出现了分歧，我们现在所处的提交并不是我们要合并入的分支指向的提交的直接祖先，此时，Git会做一个三路的合并(three-way merge)，它合并了两个分支各自指向的提交和它们的共同祖先
![[ProGit-Fig24.png]]
Git根据这次三路合并创建一个新的快照，并创建一个指向这个快照的新的提交，称这个提交为合并提交(merge commit)，合并提交特殊的点在于它有多于一个的父提交
![[ProGit-Fig25.png]]

合并了我们的工作后，我们就不再需要 `iss53` 分支了，因此我们可以在我们的问题跟踪系统(issue-tracking system)中关闭这个issue，然后删除该分支：
`git branch -d iss53`
### 3.2.3 Basic Merge Conlicts
有时我们的合并难以直接顺利进行，例如我们在两个要合并的分支中以不同的方式更改了同一个文件的同一部分，Git就将无法干净地合并它们

例如，如果我们要将 `iss53` 合并入 `master` ，但两个分支存在冲突，我们就会得到：
```
$ git merge iss53
Auto-merging index.html
CONFLICT (content): Merge conflict in index.html
Automatic merge failed; fix conflicts and then commit the result
```

此时Git并没有自动创建一个新的合并提交，它停止了合并，等待用户解决冲突，我们可以用 `git status` 来查看在发生了合并冲突后，哪些文件是没有合并的：
```
$ git status
On branch master
You have unmerged paths.
  (fix conflicts and run "git commit")

Unmerged paths:
  (use "git add <file>..." to mark resolution)

    both modified:    index.html

no changes added to commit (use "git add" and/or "git commit -a")
```

任何存在合并冲突且尚未解决的文件都会被列为未合并(unmerged)，Git会在有冲突的文件中添加标准的冲突-解决标记(conflict-resolution markers)，因此我们可以手动打开这些文件并解决这些冲突，冲突-解决标记类似于：
```
<<<<<< HEAD:index.html
<div id="footer">contact : email.support@github.com</div>
======
<div id="footer">
 please contact us at support@github.com
</div>
>>>>>> iss53:index.html
```
`HEAD` 中的版本(即 `master` 分支 ，因为当我们运行合并命令时检出了 `master` )是那个块的上半部分(所有在 `=======` 之上的内容)，而 `iss53` 分支的版本是底部的所有内容

为了解决冲突，我们必须选择一边，或者自己手动合并内容，例如，可以通过用以下内容替换整个块来解决这个冲突：
```
<div id="footer">
 please contact us at email.support@github.com
</div>
```
这个解决方案包含了每个部分的一小部分，并且 `<<<<<<<` 、`=======` 和 `>>>>>>>` 这些行已经被完全移除

当我们在每个存在冲突的文件中都解决了这些部分之后，我们可以对每个文件运行 `git add` 命令以将其标记为已解决(mark it as resolved)，在Gti中，对文件进行暂存操作就是将其标记为已解决

如果我们想使用图形化工具来解决这些问题，可以运行 `git mergetool` ，它会启动一个适当的可视化合并工具，并引导我们完成冲突解决：
```
$ git mergetool

This message is displayed because 'merge.tool' is not configured.
See 'git mergetool --tool-help' or 'git help config' for more details.
'git mergetool' will now attempt to use one of the following tools:
opendiff kdiff3 tkdiff xxdiff meld tortoisemerge gvimdiff diffuse diffmerge ecmerge p4merge araxis bc3 codecompare vimdiff emerge
Merging:
index.html

Normal merge conflict for 'index.html':
  {local}: modified file
  {remote}: modified file
Hit return to start merge resolution tool (opendiff):
```

如果想使用除默认工具之外的合并工具(在这个例子中，因为命令是在macOS上运行的，所以Git选择了 `opendiff`)，我们可以在顶部“one of the following tools”之后看到所有支持的工具列表，只需输入其中我们更愿意使用的工具的名称

在我们退出合并工具后，Git会询问是否合并成功。如果你告诉它合并成功，它会暂存该文件，以标记它为已解决, 我们可以再次运行 `git status` 来验证所有冲突是否都已解决：
```
$ git status
On branch master
All confilcts fixed but you are still merging.
  (use "git commit" to conclude merge)

Changes to be committed:

    modified:    index.html
```

我们在确认所有有冲突的地方都已经解决并被暂存，就可以输入 `git commit` 来完成合并提交(finalize the merge commit)，默认的提交信息看起来可能像这样：
```
Merge branch `iss53`

Conflicts:
    index.html
#
# It looks like you may be commiting a merge.
# If this is not correct, please remove the file
#    .git/MERGE_HEAD
# and try again



# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit
# On branch master
# All conflicts fixed but you are still merging.
#
# Changes to be commited:
#    modified: index.html
#
```
如果我们需要对未来查看此次合并的人提供帮助，就可以修改这个提交信息，添加关于如何解决合并的详细信息，并解释为什么做出这些更改
## 3.3 Branch Management
让我们来介绍一些分支管理工具，当我们开始经常使用分支时，这些工具会非常有用

`git branch` 命令不仅仅是创建和删除分支，如果不带任何参数运行它，我们会得到一个当前分支的简单列表：
```
$ git branch
  iss53
  * master
  testing
```
注意 `*` 字符，它前缀了 `master` 分支：它表示我们当前检出(即 `HEAD` 指向)的分支，也就是如果我们此时提交，`master` 分支将随着我们的新工作向前移动

`--merged` 和 `--no-merged` 选项可以过滤这个列表，显示已经合并或尚未合并到当前所在分支的分支，例如要查看哪些分支已经合并到当前所在的分支，可以运行 `git branch --merged`：
```
$ git branch --merged
  iss53
  * master
```
因为之前已经合并了 `iss53`，我们就会在列表中看到它，列表中没有 `*` 在前面的分支通常可以用 `git branch -d` 安全地删除，因为我们已经将它们的工作合并到了另一个分支，所以不会丢失任何东西

要查看所有包含尚未合并的工作的分支，可以运行 `git branch --no-merged`：
```
$ git branch --no-merged
  testing
```
这显示了我们其他的分支，因为这些分支包含了我们尚未合并的工作，因此不能用 `git branch -d` 删除它们：
```
$ git branch -d testing
error: The branch 'testing' is not fully merged.
If your are sure you want to delete it, run `git branch -D testing'.
```
如果我们确实想要删除分支并放弃那个工作，可以使用 `-D` 强制执行

上述描述的选项，`--merged` 和 `--no-merged`，如果没有给定一个提交或分支名称作为参数，它们会分别显示已经合并或未合并到当前分支的内容，我们也可以提供额外的参数来询问相对于某个其他分支的合并状态，而不需要先检出那个分支，例如：
```
$ git checkout testing
$ git branch --no-merged master
  topicA
  featureB
```

要查看每个分支上的最后一次提交，可以运行 `git branch -v`：
```
$ git branch -v
  iss53 93b412c Fix javascript issue
  * master 7a98805 Merge branch 'iss53'
  testing 782fd34 Add scott to the author list in the readme
```
### 3.3.1 Changing a branch name
注意：不要重命名仍在被其他协作者使用的分支；以及最好不要重命名像 `master/main/mainline` 这样的分支

假设我们有一个名为 `bad-branch-name` 的分支，我们想要将其更改为 `corrected-branch-name`，同时保留所有历史记录(keeping all history)，我们还想更改远程(GitHub、GitLab 或其他服务器上的)分支名称

我们可以使用 `git branch --move` 命令在本地(locally)重命名分支：
```
$ git branch --move bad-branch-name correct-branch name
```
这个更改目前只是本地的，为了让其他人在远程仓库上看到更正后的分支，需要推送它：
```
$ git push --set-upstream origin corrected-branch-name
```

现在我们简要地看一下我们目前的情况：
```
$ git branch --all
* corrected-branch-name
main
remotes/origin/bad-branch-name
remotes/origin/corrected-branch-name
remotes/origin/main
```
注意我们当前所在的分支是 `corrected-branch-name`，它也在远程仓库上可用，然而，带有不良名称的分支仍然存在于远程，但我们可以通过执行以下命令来删除它：
```
$ git push origin --delete bad-branch-name
```
现在，不良的分支名称已经完全被更正后的分支名称所取代

**Changing the master branch name**
更改如 `master/main/mainline/default` 这样的分支名称将会破坏我们的代码库所依赖的集成、服务、辅助工具以及构建/发布脚本，在执行此操作之前，请确保与我们的合作者进行协商，此外，确保在代码库中进行全面搜索，并更新代码和脚本中对旧分支名称的所有引用(update any references to the old branch name)

使用以下命令将本地的 `master` 分支重命名为 `main`：
```
git branch --move master main
```
现在，本地不再有 `master` 分支了，因为它已被重命名为 `main` 分支，为了让其他人看到新的 `main` 分支，我们需要将它推送到远程仓库，使得重命名后的分支在远程仓库中可用：
```
git push --set-upstream origin main
```

现在我们的状态如下：
```
$ git branch --all
* main
remotes/origin/HEAD -> origin/master
remotes/origin/main
remotes/origin/master
```

我们的本地 `master` 分支已经没有了，它被 `main` 分支取代了，`main` 分支现在在远程仓库中，然而，旧的 `master` 分支仍然存在于远程仓库中，其他合作者将继续使用 `master` 分支作为他们工作的基准(as the base of their work))，直到我们进行进一步的更改

现在，我们还有一些额外的任务来完成过渡：
- 任何依赖于此项目的项目都需要更新他们的代码和/或配置
- 更新任何测试运行器配置文件(test-runner configuration files)
- 调整构建和发布脚本(adjust build and release scripts)
- 在代码库上重定向设置，例如代码库的默认分支、合并规则以及其他与分支名称匹配的事项(redirect settings on your repo host for things like the repo's default branch, merge rules, and other things that match branch names)
- 在文档中更新对旧分支的引用(update references to the old branch in the documentation)
- 关闭或合并任何针对旧分支的拉取请求(close or merge any pull requests that target the old branch)
完成了所有这些任务，并确定 `main` 分支的表现与 `master` 分支一样之后，我们可以删除 master 分支：
```
git push origin --delete master
```
## 3.4 Branching Workflows
在本节中，我们将介绍一些轻量级分支所能实现的常见工作流
#### 3.4.1 Long-Running Branches
因为 `Git` 使用简单的三路合并(three-way merge)，所以在长时间内多次将一个分支合并到另一个分支通常是容易做到的，这意味着我们可以有多个始终打开的分支，并用它们来完成开发周期的不同阶段；可以定期将一些分支合并到其他分支
许多 `Git` 开发者采用了这种工作流程，比如在他们的 `master` 分支中只保留完全稳定的代码——可能是已经发布或将要发布的代码；他们还有一个名为 `develop` 或 `next` 的并行分支，他们在这个分支工作或用来测试稳定性——这个分支不一定总是稳定的，但每当它达到一个稳定状态时，就可以合并到 `master` 分支中，这个分支也可以被用来在主题分支(topic branches)(短期分支，比如我们之前创建的 `iss53` 分支)准备好时拉入(pull in)它们，以确保它们通过所有测试并且不引入bugs

实际上，我们谈论的都是沿着我们所做的提交线移动的指针(pointers moving up the line of commits you're making)，稳定的分支在我们的提交历史中处于较后的位置，而最前沿的分支则处于较前的位置
![[ProGit-Fig26.png]]
通常更容易将它们视为工作隔离区(work silos)，当一组提交在经过完全测试后(fully tested)，它们会升级到一个更稳定的隔离区(graduate to a more stable silo)
![[ProGit-Fig27.png]]

我们可以为多个稳定性级别(for several levels of stability)重复这样做，一些较大的项目还有一个提议( `proposed` )或提议更新( `proposed updates` )分支，其中整合了可能还不适合进入 `next` 分支或 `master` 分支的分支
这种理念是，我们的分支处于不同的稳定性级别(at various levels of stability)，当它们达到更高稳定性级别时，就会合并到它们上面的分支中，再次强调，拥有多个长期运行的分支并不是必需的，但这通常很有帮助，特别是当处理的是非常大或复杂的项目时
#### 3.4.2 Topic Branches
主题分支(Topic branches)在任何大小的项目中都很有用，主题分支是一个短期存在(short-lived)的分支，我们创建它并用于单一特定功能或相关工作(for a single paticular feature or related work)，这可能是我们在其他版本控制系统中从未做过的事情，因为通常创建和合并分支的代价太高
但在 Git 中，一天内多次创建、工作、合并和删除分支是很常见的，例如在上一节中，我们创建了 `iss53` 和 `hotfix` 分支，在上面做了一些提交，然后在将它们合并到主分支后直接删除了它们
Git的这种技术允许我们迅速而彻底地进行上下文切换——因为工作被分隔开，那个分支中的所有更改都与那个主题有关(all the changes in that branch have to do with that topic)，所以在代码审查等过程中更容易看出发生了什么，我们可以将更改保留在那里几分钟、几天或几个月，并在它们准备好时合并它们，而不管它们是按什么顺序创建或工作的

考虑一个例子：在 `master` 分支上做一些工作，为一个问题(`iss91`)分叉(branching off)出去，在上面工作一段时间，从分叉到另一个分支，尝试用另一种方式处理同一个问题(`iss91v2`)，回到 master 分支再工作一段时间，然后从分叉出去做一些不确定是否是个好主意的工作(`dumbidea` 分支)，那么我们就会有这样的提交历史：
![[ProGit-Fig28.png]]
现在，假设我们决定使用issue91的第二个解决方案(`iss91v2`)；并且我们向同事展示了 `dumbidea` 分支，发现它其实非常出色，我们就可以丢弃原来的 `iss91` 分支(丢失提交 `C5` 和 `C6`)，并合并另外两个分支，则此时历史记录看起来像这样：
![[ProGit-Fig29.png]]
要记住，在进行所有这些操作时，这些分支完全是本地的，当进行进行分支和合并时，一切都是在本地Git仓库中完成的——与服务器没有任何通信
## 3.5 Remote Branches
远程引用(remote references)是我们远程仓库(in remote repositories)中的引用(指针 pointers)，包括分支(branches)、标签(tags)等，我们可以使用 `git ls-remote <remote>` 获取远程引用的完整列表(full list of remote references)，或者使用 `git remote show <remote>` 来获取远程分支以及更多信息
然而，更常见的方法是利用远程跟踪分支(remote-tracking branches)，远程跟踪分支是对远程分支状态的引用(references to the state of remote branches)，它们是我们不能移动的本地引用(local references that you can't move)；每当我们进行任何网络通信时，Git都会移动它们，以确保它们准确地表示远程仓库的状态，可以将它们想象为书签(bookmarks)，用来提醒我们上次连接到它们时远程仓库中的分支所在的位置(where the branches are in your remote repositories)

远程跟踪分支的命名采用 `<remote>/<branch>` ( `<远程仓库>/<分支>` ) 的形式，例如，如果我们想查看上次与 `origin` 远程仓库通信时 `master` 分支的状态，我们应该查看 `origin/master` 分支，如果我们与合作伙伴一起处理一个问题，并且他们推送了一个 `iss53` 分支，我们可能有自己的本地 `iss53` 分支，但服务器上的分支将由远程跟踪分支 `origin/iss53` 表示

让我们来看一个例子，假设我们在网络中有一个位于 `git.ourcompany.com` 的 Git 服务器，如果从这个服务器克隆，Git 的 `clone` 命令会自动将其命名为 `origin`，并拉取所有数据(pulls down all its data)，创建一个指向其 `master` 分支的指针，并在本地命名将其为 `origin/master` ，
Git 还会给出一个自己的本地 `master` 分支，从 `origin` 的 `master` 分支的同一位置开始

![[ProGit-Fig30.png]]

>***"origin" is not special***
>就像分支名称 "master" 在 Git 中没有任何特殊含义一样，"origin" 也是如此，"master" 是运行 `git init` 时的起始分支的默认名称，这也是它被广泛使用的唯一原因，而 "origin" 是运行 `git clone` 时远程仓库的默认名称
>如果我们运行 `git clone -o booyah`，那么就将有 `booyah/master` 作为默认远程分支(default remote branch)

如果我们在本地的 `master` 分支上进行了一些工作，同时，其他人推送到 git.ourcompany.com 并更新了它的 `master` 分支，那么我们的提交历史就会朝不同的方向发展，但只要我们不与 `origin` 服务器联系，我们的 `origin/master` 指针就不会移动。
![[ProGit-Fig31.png]]


为了与指定的远程仓库同步我们的工作，我们需要运行 `git fetch <remote>` 命令(在我们的例子中，就是 `git fetch origin`)，这个命令查找“origin”是哪个服务器（在这个例子中，就是 `git.ourcompany.com`)，然后获取我们尚未拥有的任何数据，并更新本地数据库，将我们的 `origin/master` 指针移动到它更新的位置
![[ProGit-Fig32.png]]

为了展示拥有多个远程服务器以及这些远程项目的远程分支看起来如何，让我们假设有另一个仅供我们的一个冲刺团队用于开发的内部Git服务器(internal Git server)，这个服务器位于 `git.team1.ourcompany.com` 
我们通过运行 `git remote add` 命令，将其作为新远程引用添加到当前正在工作的项目中，并将这个远程命名为 `teamone`，作为完整URL的简称：
```
git remote add teamon git://git.team1.ourcompany.com
```
![[ProGit-Fig33.png]]

现在，我们可以运行 `git fetch teamone` 来获取远程 `teamone` 服务器上所有我们还没有的数据，而由于那个服务器拥有的数据是我们现在 `origin` 服务器上数据的一个子集，Git 不会获取任何数据，但会设置一个远程跟踪分支叫做 `teamone/master`，指向 `teamone` 将其作为主分支的提交
![[ProGit-Fig34.png]]
#### 3.5.1 Pushing
当我们想要与世界分享一个分支时，我们需要将它推送到一个我们有写入权限的远程仓库，我们本地分支不会自动同步到我们写入的远程仓库——因此我们必须显式推送我们想要分享的分支，同时我们对于我们不想分享的工作，我们可以使用私有分支(private branches)，只推送我们想要合作的主题分支(push up only the topic branches you want to collaborate on)

如果我们有一个名为 `serverfix` 的分支想要与他人一起工作，我们运行 `git push <remote> <branch>` 推送它：
```
$ git push origin severfix
Counting objects: 24, done.
Delta compression using up to 8 threads.
Compressing objects: 100%(15/15), done.
Writing objects: 100%(24/24), 1.91KiB | 0 bytes/s, done.
Total 24 (delta 2), reused 0 (delta 0)
To https://github.com/schacon/simplegit
 * [new branch]    serverfix -> serverfix
```
该命令其实是一种简写，Git会自动将 `serverfix` 分支名扩展为 `refs/heads/serverfix:refs/heads/serverfix` ，这意味着“把我的本地 `serverfix` 分支推送到远程仓库以更新远程的 `serverfix` 分支”

我们将在Git Internals中详细解释 `refs/heads/`部分，但通常我们可以省略它，我们也可以执行 `git push origin serverfix:serverfix`，这会做同样的事情——它表示“把我的 `serverfix` 分支变成远程的 `serverfix` 分支”
我们可以使用这种格式将本地分支推送到命名不同的远程分支，例如如果我们不希望在远程仓库中称之为 `serverfix` ，可以运行 `git push origin serverfix:awesomebranch` ，将本地 `serverfix` 分支推送到远程项目的`awesomebranch` 分支

> ***Don't type your password every time***
> 如果使用HTTPS URL进行推送，Git服务器会要求我们输入用户名和密码进行身份验证，默认情况下，它会在终端中提示我们输入这些信息，以便服务器可以判断我们是否有权推送
>如果不想每次推送时都输入密码，可以设置一个“凭证缓存(credential cache)”，最简单的方法是将其仅在内存中保留几分钟，可以通过运行`git config --global credential.helper cache`来轻松设置

下一次我们的合作者之一从服务器取数据时，他们就会得到一个本地的 `origin/serverfix` 分支，作为对服务器端的 `serverfix` 分支的引用
```
$ git fetch origin
remote: Counting objects: 7, done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 3 (delta 0), reused 3 (delta 0)
Unpacking objects: 100% (3/3), done.
From https://github.com/schacon/simplegit
 * [new branch]    serverfix  ->  origin
```
值得注意的是，当我们执行一个获取操作(fetch)下载新的远程跟踪分支(remote-tracking branch)时，我们并不会自动拥有它们的本地可编辑副本，换句话说，在这种情况下，我们并没有一个新的 `serverfix` 分支——而是只有一个不能修改的 `origin/serverfix` 指针

要将这些工作合并到当前的工作分支中，可以运行 `git merge origin/serverfix` ，如果我们想拥有自己的 `serverfix` 分支以便工作，可以基于远程跟踪分支来创建它：
```
$ git checkout -b serverfix origin/serverfix
Branch serverfix set up to track remote branch serverfix from origin
Switched to a new branch 'serverfix'
```
这会给我们一个我们可以工作的本地分支 `serverfix` ， 它的起点即 `origin/serverfix`
#### 3.5.2 Tracking Branches
从远程跟踪分支检出一个本地分支会自动创建所谓的“跟踪分支(tracking branch)”(它跟踪的分支称为“上游分支 upstream branch”)，跟踪分支是与远程分支有直接关系(direct relationship)的本地分支
如果我们在跟踪分支上，并且输入 `git pull` ，Git会自动知道从哪个服务器获取数据以及合并哪个分支

当我们克隆一个仓库时，Git通常会自动创建一个跟踪 `origin/master` 的`master` 分支，我们可以设置其他跟踪分支——那些跟踪其他远程服务器上的分支的分支，或者不再跟踪 `master` 分支
我们刚才已经看到过了一个创建跟踪分支的简单的例子，即运行 `git checkout -b <branch> <remote>/<branch>`，Git为此提供了`--track` 的简写方式：
```
$ git checkout --track origin/serverfix
Branch serverfix set up to track remote branch serverfix from origin.
Switched to a new branch 'serverfix'
```
实际上，这个操作十分常用，以至于连 `--track` 的简写方式也有一个简写方式，如果我们尝试检出的分支(a)不存在，并且(b)只与一个远程分支的名称完全匹配(exactly matches a name on only one remote)，Git将为我们创建一个跟踪分支：
```
$ git checkout serverfix
Branch serverfix setup to track remote branch serverfix from origin
Switched to a new branch 'serverfix'
```
要设置一个与远程分支名称不同的本地分支，可以使用第一个版本，并指定一个不同的本地分支名称：
```
$ git checkout -b sf origin/serverfix
Branch sf set up to track remote branch serverfix from origin
Switched to a new branch 'sf'
```
现在，我们的本地分支 `sf` 会自动从 `origin/serverfix` 进行拉取(pull from)

如果我们已经有一个本地分支，并且想将其设置为刚刚拉取的远程分支，或者想要更改我们正在跟踪的上游分支，可以使用`git branch`命令的`-u`或`--set-upstream-to`选项，在任何时候显式地设置它：
```
$ git branch -u origin/serverfix
Branch serverfix set up to track remote barnch serverfix from origin.
```

> ***Upstream shorthand***
>当我们设置了一个跟踪分支后，可以使用 `@{upstream}` 或 `@{u}` 的简写来引用它的上游分支，所以如果我们在 `master` 分支上，并且它正在跟踪`origin/master` ，我们可以使用`git merge @{u}`而不是`git merge origin/master`，如果愿意的话

如果我们想要看到我们设定了什么跟踪分支，我们可以使用 `git branch` 的 `-vv` 选项，这会列出我们的本地分支并带有更多的信息，包括了正在跟踪哪个分支，以及本地分支是在前，在后或者二者皆有(ahead, behind or both)
```
$ git branch -vv
  iss53    7e424c3 [origin/iss53: ahead 2] Add forgotten brackets
  master   1ae2a45 [origin/master] Deploy index fix
* serverfix f8674d9 [teamon/server-fix-good: ahead 3, behind 1] This should do it 
  testing   5ea463a Tay something new
```
我们可以看到我们的 `iss53` 分支正在跟踪 `origin/iss53` ，并且“领先”两个，这意味着我们有两个本地提交尚未推送到服务器，我们还可以看到我们的`master` 分支正在跟踪 `origin/master` 并且是最新的，接下来我们可以看到我们的 `serverfix` 分支正在跟踪我们 `teamone` 服务器上的 `server-fix-good` 分支，并且领先三个，落后一个，这意味着服务器上有一个提交我们还没有合并进来，而我们有三个本地提交还没有推送，最后我们可以看到我们的`testing` 分支没有跟踪任何远程分支

要注意，这些数字只是自我们上次从每个服务器获取以来的情况，这个命令不会联系服务器，它展示的是它从这些服务器本地缓存的信息，如果想获得完全最新的领先和落后数字，我们需要在运行此命令之前从所有远程服务器获取最新数据(fetch from all you remotes)，可以这样做：
```
$ git fetch --all; git branch -vv
```
#### 3.5.3 Pulling
`git fetch`命令会获取服务器上所有我们还没有的更改，但它不会以任何方式修改我们的工作目录，它只会获取数据，然后让我们自己合并
然而，有一个叫做`git pull`的命令，在大多数情况下，它本质上是立即跟随 `git fetch`的 `git merge` ，如果我们像上一节演示的那样设置了跟踪分支，无论是通过显式设置还是通过克隆或检出命令，`git pull`会查找我们当前分支正在跟踪的服务器和分支，从那个服务器获取然后尝试合并那个远程分支

通常，最好只是显式使用`fetch`和`merge`命令，因为`git pull` 常常会让人感到困惑
#### 3.5.4 Deleting Remote Branches
假设我们完成了一个远程分支的工作——比如说完成了一个特性的开发，并且已经将其合并到了我们远程的 `master` 分支(或者我们的稳定代码线 stable codeline 所在的任何分支)，我们可以使用 `git push` 命令的 `--delete` 选项来删除远程分支
如果想从服务器上删除你的 `serverfix` 分支，可以运行以下命令：
```
$ git push origin --delete serverfix
To https://github.com/schacon/simplegit
 - [deleted]    serverfix
```
基本上该命令所做的所有事就是将该指针从服务器上移除，Git服务器通常会保持该数据一段时间，直到垃圾处理程序的运行，因此如果有时意外删除了，也比较容易恢复
## 3.6 Rebaseing
在Git中，将一个分支的更改整合到另一个分支主要有两大方式：合并(merge)和变基(rebase)
### 3.6.1 The Basic Rebase
![[ProGit-Fig35.png]]

整合分支的最简单方式，正如我们已经讨论过的，是使用 `merge` 命令，它在两个最新的分支快照( `C3` 和 `C4` )以及两者最近的共同祖先( `C2` )之间执行三方合并，创建一个新的快照(以及提交)
![[ProGit-Fig36.png]]

然而，还有另一种方式：我们可以取出 `C4` 中引入的变更部分(the patch of the change)，并在 `C3` 的基础上重新应用(reapply)它，在Git中，这被称为变基，使用 `rebase` 命令，我们可以取出在一个分支上提交的所有更改(all the changes that were committed on one brach)，并在另一个分支上重新应用它们

例如，我们检出 `experiment` 分支，然后将其变基到 `master` 分支(rebase it onto the `master` )，如下所示：
```
$ git checkout experiment
$ git rebase master
First, rewinding heada to replay your work on top of it...
Applying: added staged command
```
这个操作的工作原理是：找到我们正在变基的两个分支(当前所在的分支和正在变基到的分支)的共同祖先，获取当前所在的分支上每个提交引入的差异(diff)，将这些差异保存到临时文件中，然后将当前分支重置为指向与我们正在变基到的分支相同的提交(resetting the current branch to the same commit as the branch you are rebasing onto)，最后依次应用每个更改
![[ProGit-Fig37.png]]

此时，我们可以切换回 `master` 分支，并进行一次快速前移合并(fast-forward merge)
```
$ git checkout master
$ git merge experiment
```
![[ProGit-Fig38.png]]
现在，由 `C4'` 指向的快照与合并示例中 `C5` 指向的快照完全相同，集成的最终产品没有区别，但变基可以创建一个更干净的历史记录，如果我们检查一个变基分支的日志，它看起来像是一个线性历史：看起来所有的工作都是连续发生的，即使它最初是并行发生的

通常，我们会执行变基以确保我们的提交在远程分支上干净地应用——也许我们正在尝试贡献一个项目，但并不维护它，在这种情况下，我们会在一个分支上做我们的工作，然后当准备好将我们的补丁提交到主项目时，将我们的工作变基到 `origin/master` ，这样，维护者就不需要做任何集成工作——只需要一个快进或一个干净的应用

请注意，无论是通过变基得到的最后一个变基提交，还是合并后的最终合并提交，指向的快照是相同的——只有历史记录是不同的(only the history that is different)，变基是将一个工作线上的更改按它们引入的顺序重新播放到另一个工作线上，而合并是将端点合并在一起
### 3.6.2 More Interesting Rebases
我们还可以将变基操作重新应用到除变基目标分支之外的其他分支上，以一个具有从另一个主题分支分出的主题分支的历史(a history with a topic branch off another topic branch)为例，我们从一个主题分支(`server`)分出一个分支来为项目添加一些服务器端功能，并进行了一次提交，然后从那个分支分出另一个分支来进行客户端更改(`client`)，并多次提交，最后回到 `server` 分支并进行了更多的提交
![[ProGit-Fig39.png]]

假设我们决定将客户端更改合并到主线中进行发布，但想在进一步测试后再推迟服务器端的更改，我们可以取出在客户端但不在服务器上的更改( `C8` 和`C9` )，并使用`git rebase` 的 `--onto` 选项将它们重新应用到 `master` 分支上：
```
$ git rebase --onto master server client
```
这意味着：“取出 `client` 分支，找出它从 `server` 分支分叉以来的补丁，然后在 `master` 上重新应用这些补丁，就好像 `client` 是直接基于 `master` 分支一样
![[ProGit-Fig40.png]]
现在我们可以快速前移我们的 `master` 分支
```
$ git checkout master
$ git merge client
```
![[ProGit-Fig41.png]]

假设我们决定也拉取 `server` 分支，可以通过运行`git rebase <basebranch> <topicbranch>` 来将 `server` 分支变基到 `master` 分支，而无需先检出它——这将为我们检出主题分支(在本例中是 `server` )，并将其重新应用到基础分支( `master` )上：
```
git rebase master server
```

这将把我们 `server` 的工作重新应用到 `master` 的工作之上(on top of)
![[ProGit-Fig42.png]]

然后我们可以快速前移我们的基分支( `master` )：
```
$ git checkout master
$ git merge server
```

我们可以移除 `client` 和 `server` 分支因为所有的工作都已经集成，我们已经不再需要这些分支
```
$ git branch -d client
$ git branch -d server
```
![[ProGit-Fig43.png]]
### 3.6.3 The Perils of Rebasing
变基的并非没有缺点，这些缺点可以用一句话概括：
**不要变基那些存在于我们的仓库之外并且人们可能已经基于它们进行工作的提交(do not rebase commits that exist outside your repository and that people may have based work on)**

当我们变基时，我们是在放弃现有的提交，并创建新的类似但不同的提交，如果我们在某个地方推送了提交，其他人拉取了它们并基于它们进行工作，然后我们用 `git rebase` 重写了这些提交并再次推送，我们的合作者将不得不重新合并(re-merge)他们的工作，当我们尝试将他们的工作拉回到我们的工作中时，事情会变得混乱

让我们来看一个例子，看看变基我们已经公开的工作如何导致问题，假设我们从一个中央服务器克隆，然后在此基础上做了一些工作，提交历史看起来像这样：
![[ProGit-Fig44.png]]

现在，有人做了更多的工作，包括合并操作，并将这些工作推送到了中央服务器，我们获取了这些更改，并将新的远程分支合并到我们的工作中，使历史看起来像这样：
![[ProGit-Fig45.png]]

接下来，之前推送了工作的人决定回退并重新基于他们的工作进行变基，他们执行了`git push --force`来覆盖服务器上的历史记录，然后我们从那个服务器获取更新，带来了新的提交：
![[ProGit-Fig46.png]]

现在我们俩都陷入了困境，如果我们执行 `git pull`，我们将创建一个合并提交，它将包括两行历史记录，仓库将看起来像这样：
![[ProGit-Fig47.png]]

如果我们在历史记录看起来像这样的时候运行 `git log`，我们会看到两个具有相同作者、日期和消息的提交，这会让人困惑
此外，如果将这段历史推送回服务器，将重新引入所有那些已经变基过的提交到中央服务器，这可能会进一步让人们感到困惑，因为我们可以相当安全地假设，另一位开发者不希望 `C4` 和 `C6` 出现在历史记录中；这正是他们首先进行变基的原因
### 3.6.4 Rebase When You Rebase
如果我们团队中的某人强制推送了覆盖了我们基于其上工作的更改(force pushed changes that overwrite work that you've based work on)，我们的挑战就是要弄清楚什么是我们的工作，什么是他们重写的

除了提交的SHA-1校验和之外，Git还计算了一个仅基于提交引入的补丁的校验和，这被称为“补丁ID(patch-id)”，
如果我们拉取了被重写的工作，并在其上重新基于合作伙伴的新提交进行变基，Git通常能够成功地弄清楚什么是我们独特的工作，并将它们重新应用到新分支上

例如，在之前的场景中，如果我们在有人推送了变基提交(rebased commit)时，不是进行合并，而是放弃我们基于其上的工作，我们运行`git rebase teamone/master`，Git将会：
- 确定哪些工作是我们分支独有的( `C2, C3, C4, C6, C7` )
- 确定哪些不是合并提交( `C2, C3, C4` )
- 确定哪些没有被重写到目标分支中(只有 `C2` 和 `C3` ，因为 `C4` 与 `C4'` 是相同的补丁)
- 在目标分支 `teamone/master` 之上应用这些提交

我们得到的结果是这样的：
![[ProGit-Fig48.png]]

这只有在我们的合作伙伴创建的 `C4` 和 `C4'` 是几乎完全相同的补丁时才有效，否则，`git rebase` 将无法识别到它是重复的，并将添加另一个类似 `C4`的补丁(这可能会失败，因为更改至少已经部分存在)

我们也可以通过运行 `git pull --rebase` 而不是普通的 `git pull` 来简化这个过程，或者可以手动执行 `git fetch` ，然后执行 `git rebase teamone/master`，
如果使用 `git pull` 并希望将 `--rebase` 设置为默认值，我们可以使用类似`git config --global pull.rebase true` 的命令来设置 `pull.rebase` 配置值

如果我们仅变基不会推送到远端的提交，则没问题，如果我们变基已经推送的提交，但没有其他人正基于这个提交进行工作，则也没问题，如果我们变基已经推送的提交，且其他人可能已经基于该提交进行了工作，则会导致麻烦，如果我们在某个时候确实需要这样做，请确保我们以及合作伙伴的每个人都知道运行 `git pull --rebase`，以尝试在事情发生后让痛苦稍微简单一些\
### 3.6.4 Rebase vs. Merge
有人认为，仓库的提交历史是实际发生事情的记录，它是一个历史文件，本身具有价值，不应该被篡改
从这个角度来看，改变提交历史几乎是一种亵渎；如果有一系列混乱的合并提交，那又怎样？那就是事情的经过，仓库应该为后世保存这一点

相反的观点是，提交历史是你项目制作的故事，你不会出版一本书的第一稿，那么为什么要展示你的混乱工作呢？当你在做一个项目时，你可能需要记录你所有的错误和死胡同路径，但当你准备向世界展示你的工作时，你可能想要讲述一个更连贯的故事，说明如何从A到B。这个阵营的人使用像 `rebase` 和`filter-branch` 这样的工具，在将提交合并到主线分支之前重写他们的提交，以最适合未来读者的方式讲述故事。

现在，关于合并还是变基更好这个问题：希望你可以看到，这并不那么简单。Git是一个强大的工具，它允许你对历史做很多事情，但每个团队和每个项目都是不同的

你可以两全其美：在推送之前变基本地更改以清理你的工作，但永远不要变基你已经推送过的任何东西
# 4 Git on the Server
远程仓库通常是一个裸(bare)仓库——一个没有工作目录的Git仓库，因为仓库仅用作协作点，所以没有必要在磁盘上检出一个快照；它只是Git数据。用最简单的话来说，裸仓库就是你的项目的.git目录的内容，除此之外没有其他东西
## 4.1 The Protocals
### 4.1.1 Local Protocal
```
$ git clone /srv/git/project.git
$ git clone file:///srv/git/project.git
```
如果你在URL开头明确指定了 `file://` ，Git的操作会略有不同，如果你只指定路径，Git会尝试使用硬链接或直接复制它需要的文件，如果你指定了`file://` ，Git会启动它通常用于通过网络传输数据的过程，这通常效率要低得多，指定 `file://` 前缀的主要原因是，如果你想要一个干净的仓库副本，不包含多余的引用或对象(extraneous reference or objects)——通常是在从另一个版本控制系统导入或类似操作之后(见Git Internals中的维护任务)，我们将在这里使用普通路径，因为这样做几乎总是更快

要将一个本地仓库添加到现有的Git项目中，你可以运行类似于这样的命令：
```
$ git remote add local_proj /srv/git/project.git
```
然后，你可以通过你的新远程名称 `local_proj` 推送和拉取该远程仓库，就像你是通过网络进行操作一样
##### 4.1.1.1 The Pros
##### 4.1.1.2 The Cons
### 4.1.2 The HTTP Protocals
##### 4.1.2.1 Smart HTTP
它可能已经成为现在使用Git最受欢迎的方式，因为它可以像 `git://` 协议一样匿名设置服务，并且也可以像SSH协议一样在认证和加密之上进行推送，你不再需要为这些设置不同的URL，现在你可以使用单一的URL来实现两者
如果你尝试推送，并且仓库需要认证(它通常应该需要)，服务器可以提示输入用户名和密码，对于读取访问权限也是如此

实际上，对于像GitHub这样的服务，你用来在线查看仓库的URL(例如，https://github.com/schacon/simplegit)就是你可以用来克隆的URL，如果你有访问权限，也可以用来推送
##### 4.1.2.2 Dumb HTTP
##### 4.1.2.3 The Pros
##### 4.1.2.4 The Cons
如果你使用HTTP进行认证推送，提供你的凭证(credentials)有时比使用SSH上的密钥更复杂，然而，你可以使用几种凭证缓存工具，包括macOS上的Keychain访问和Windows上的凭证管理器，使这个过程相当无痛，阅读“凭证存储(Credential Storage)”以了解如何在你的系统上设置安全的HTTP密码缓存
### 4.1.3 The SSH Protocal
当自托管(self-hosting)Git服务时，一种常见的传输协议是通过SSH，这是因为在大多数地方已经设置了对服务器的SSH访问——如果没有，设置起来也很容易，SSH还是一种经过认证的网络协议，因为它无处不在，通常很容易设置和使用

要通过SSH克隆Git仓库，你可以指定一个像这样的 `ssh:// URL` ：
```
$ git clone ssh://[user@]server/project.git
```
也可以使用类似scp的语法
```
$ git clone [user@]server:project.git
```
以上两例中，如果没有指定用户名，Git则使用我们当前登录的用户名
##### 4.1.3.1 The Pros
##### 4.1.3.2 The Cons
SSH的一个缺点是它不支持对您的Git仓库的匿名访问，如果您使用SSH，人们必须拥有对您机器的SSH访问权限，即使是只读权限，这使得SSH并不适合那些人们可能只想克隆您的仓库来检查的开源项目，如果您只在公司内部网络中使用，SSH可能是您需要处理的唯一协议

如果您想允许对项目进行匿名只读访问，并且还想使用SSH，您将不得不设置SSH以便您进行推送，但为其他人设置其他方式来拉取
### 4.1.4 The Git Protocal
最后，我们有Git协议，这是Git自带的一个特殊守护进程，它在一个专用端口(9418)上监听，提供与SSH协议类似的服务，但完全没有认证或加密
为了让仓库通过Git协议提供服务，你必须创建一个 `git-daemon-export-ok` 文件——没有这个文件，守护进程不会提供仓库服务——但除此之外，没有安全措施，这意味着Git仓库要么对所有人都开放克隆，要么不开放，
这通常意味着没有人会设置可以通过这个协议进行推送，你可以启用推送访问权限，但鉴于缺乏认证，任何在互联网上找到你项目URL的人都可以推送到那个项目，简而言之，这是很少见的
#### 4.1.4.1 The Pros
#### 4.1.4.2 The Cons
由于缺乏TLS或其他加密措施，通过 `git://` 克隆可能会引发任意代码执行漏洞，因此除非你清楚自己在做什么，否则应该避免使用这种方式
- 如果你执行`git clone git://example.com/project.git`，控制你的路由器的攻击者可能会修改你刚刚克隆的仓库，插入恶意代码，如果你随后编译/运行了你刚刚克隆的代码，你将执行恶意代码，出于同样的原因，应该避免运行`git clone http://example.com/project.git`
- 执行`git clone https://example.com/project.git`不会遇到相同的问题(除非攻击者能为example.com提供TLS证书)，执行`git clone git@example.com:project.git`只有在你接受了一个错误的SSH密钥指纹时才会遇到这个问题
它也没有认证机制，即任何人都可以克隆仓库(尽管这往往是你想要的)，它也可能是最难设置的协议，它必须运行自己的守护进程，这需要 `xinetd` 或`systemd` 配置或类似的东西，这并不总是一件容易的事，它还需要防火墙访问端口9418，这不是企业防火墙总是允许的标准端口，在大型企业防火墙后面，这个不显眼的端口通常被阻止
## 4.2 Getting Git on a Server
为了初始设置任何Git服务器，你必须将现有的仓库导出到一个新的裸仓库中——一个不包含工作目录的仓库，这通常很容易做到，为了克隆你的仓库以创建一个新的裸仓库，你使用 `--bare` 选项运行克隆命令，按照惯例，裸仓库目录名以 `.git` 后缀结尾，例如：
```
$ git clone --bare my_project my_project.git
Cloning into bare repository 'my_project.git'
done.
```
现在你在你的 `my_project.git` 目录中应该有了一份Git目录数据的副本
这大致相当于：
```
$ cp -Rf my_project/.git my_project.git
```
### 4.2.1 Putting the Bare Repository on a Server
既然你已经有了仓库的裸版本，你所需要做的只是将其放在服务器上并设置你的协议，假设你已经设置了一个名为 `git.example.com` 的服务器，并且你有SSH访问权限，你想将所有的Git仓库存储在 `/srv/git` 目录下

假设 `/srv/git` 在那个服务器上已经存在，你可以通过复制你的裸仓库来设置你的新仓库：
```
$ scp -r my_project.git user@git.example.com:/srv/git
```
此时，拥有服务器上 `/srv/git` 目录基于SSH的读取权限的其他用户可以通过运行：
```
$ git clone user@git.example.com:/srv/git
```
来克隆你的仓库，如果一个用户通过SSH登录到服务器，并且对 `/srv/git/my_project.git` 目录有写入权限，他们也会自动拥有推送权限

如果你运行带有 `--shared` 选项的 `git init` 命令，Git会自动为仓库添加组写入权限，注意，运行这个命令不会在这个过程中破坏任何提交、引用等：
```
$ ssh user@git.example.com
$ cd /srv/git/my_project.git
$ git init --bare --shared
```

你可以看到，将Git仓库创建一个裸版本，并将其放在你和你协作者都有SSH访问权限的服务器上是多么容易，现在你已经准备好在同一个项目上进行协作了

重要的是要注意，这实际上就是你运行一个有用的Git服务器所需的一切——只需在服务器上添加SSH可访问的账户，并将裸仓库放在所有这些用户都有读写权限的地方，要与几个人在私有项目上进行协作，你所需要的只是一个SSH服务器和一个裸仓库
### 4.2.2 Small Setups
#### 4.2.2.1 SSH Access
## 4.3 Generating Your SSH Pubilc Key
许多Git服务器使用SSH公钥进行认证，为了提供公钥，如果你的系统中的每个用户还没有公钥，他们必须生成一个，这个过程在所有操作系统中都是相似的

首先，你应该检查确认你还没有密钥，默认情况下，用户的SSH密钥存储在用户的 `~/.ssh` 目录中，你可以通过进入那个目录并列出内容来轻松检查你是否已经有了密钥：
```
$ cd ~/.ssh
$ ls
authorized_keys  id_dsa  known_hosts
config           id_dsa.pub
```

你会看到像 `authorized_keys2、id_dsa、known_hosts、config、id_dsa.pub` 这样的文件
你正在寻找一对名为 `id_dsa` 或 `id_rsa` 之类的文件，以及一个带有 `.pub` 扩展名的匹配文件，`.pub` 文件是你的公钥，另一个文件是相应的私钥，如果你没有这些文件(或者你甚至没有 `.ssh` 目录)，你可以通过运行一个叫做 `ssh-keygen` 的程序来创建它们，这个程序随Linux/macOS系统的SSH包提供，并且随Windows的Git一起提供：
```
$ ssh-keygen -0
Generating public/private rsa key pair.
Enter file in which to save the key (/home/schacon/.ssh/id_rsa):
Creatng directory '/home/schacon/.ssh'.
Enter passphrase (empty for no passphrase)
Enter same passphrase again:
Your identification has been saved in /home/scharon/.ssh/id_rsa.
Your public key has been saved in /home/schacon/.ssh/id_rsa.pub.
The key fingerprint is:
d0:82:24:8e:d7:f1:bb:9b:33:53:96:93:49:da:9b:e3 schacon@mylaptop.local
```
首先，它会确认你想要保存密钥的位置(`.ssh/id_rsa` )，然后它会两次询问你密码短语，如果你不想在使用密钥(key)时输入密码(password)，你可以将其留空
然而，如果你确实使用了一个密码，请确保添加 `-o` 选项；它会以一种比默认格式更能抵抗暴力破解密码的方式保存私钥，你还可以使用 `ssh-agent` 工具来避免每次输入密码

现在，每个这样做的用户都必须将他们的公钥发送给你或管理Git服务器的任何人(假设你使用的是要求公钥的SSH服务器设置)，他们所要做的就是复制 `.pub` 文件的内容并通过电子邮件发送，公钥看起来像这样：
```
$ cat ~/.ssh/id_rsa.pub
ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAklOUpkDHrfHY17SbrmTIpNLTGK9Tjom/BWDSU GPl+nafzlHDTYW7hdI4yZ5ew18JH4JW9jbhUFrviQzM7xlELEVf4h9lFX5QVkbPppSwg0cda3 Pbv7kOdJ/MTyBlWXFCR+HAo3FXRitBqxiX1nKhXpHAZsMciLq8V6RjsNAQwdsdMFvSlVK/7XA t3FaoJoAsncM1Q9x5+3V0Ww68/eIFmb1zuUFljQJKprrX88XypNDvjYNby6vw/Pb0rwert/En mZ+AW4OZPnTPI89ZPmVMLuayrD2cE86Z/il8b+gw3r3+1nKatmIkjn2so1d01QraTlMqVSsbx NrRFi9wrf+M7Q== schacon@mylaptop.loca
```
## 4.4 Setting Up the Server
首先我们为用户创建一个 `git` 用户账号以及一个 `.ssh` 目录：
```
$ sudo adduser git
$ su git
$ cd
$ mkdir .ssh && chmod 700 .ssh
$ touch .ssh/authorized_keys && chmod 600 .ssh/authorized_keys
```

接着我们在 `git` 用户的 `authorized_keys` 文件中添加开发者们的公钥：
```
$ cat /tmp/id_rsa.john.pub >> ~./ssh/authorized_keys
$ cat /tmp/id_rsa.josie.pub >> ~./ssh/authorized_keys
$ cat /tmp/id_rsa.jessica.pub >> ~./ssh/authorized_keys
```

现在，我们为他们设定一个空仓库，使用 `git init --bare` ，初始化一个没有工作目录的仓库：
```
$ cd /srv/git
$ mkdir project.git
$ cd project.git
$ git init --bare
Initilized empty Git repository in /srv/git/project.gi/
```

然后，John、Josie或Jessica可以通过将其添加为远程仓库并推送一个分支，将他们项目的首个版本推送到那个仓库中
请注意，每次你想要添加一个项目时，都必须有人登录到机器上并创建一个裸仓库
让我们使用 `gitserver` 作为你已经设置好 `git` 用户和仓库的服务器的主机名，如果你在内部运行它，并且你为 `gitserver` 设置了DNS指向那台服务器，那么你可以几乎原样使用这些命令(假设 `myproject` 是一个已经存在的项目，里面有文件)：
```
# on John's computer 
$ cd myproject 
$ git init 
$ git add .
$ git commit -m 'Initial commit'
$ git remote add origin git@gitserver:/srv/git/project.git 
$ git push origin master
```
现在，其他人也可以拉取该仓库，修改并再推送
```
$ git clone git@gitserver:/srv/git/project.git
$ cd project
$ vim README
$ git commit -am 'Fix for README file'
$ git push origin master
```

你应该注意，目前所有这些用户也可以登录到服务器并以 `git` 用户的身份获得一个shell，如果你想限制这一点，你将不得不在 `/etc/passwd` 文件中将shell更改为其他内容
你可以使用一个名为 `git-shell` 的受限制(limited)shell工具来轻松限制 `git`用户账户仅进行与Git相关的活动，这个工具随Git一起提供，如果你将这个设置为 `git` 用户账户的登录shell，那么这个账户就不能对你的服务器进行正常的shell访问
要使用这个功能，你应该为那个账户的登录shell指定 `git-shell` 而不是 `bash` 或 `csh` ，为此，如果 `git-shell` 命令的完整路径还不在`/etc/shells`中，你必须首先将其添加进去：
```
$ cat /etc/shells # see if git-shell is already in there. If not...
$ which git-shell # make sure git-shell is installed on your system
$ sudo -e /etc/shells $ and add the path to git-shell from last command
```

然后我们使用 `chsh <username> -s <shell>` 修改用户的shell：
```
$sudo chsh git -s $(which git-shell)
```

现在 `git` 用户仍然可以使用SSH连接来推送和拉取Git仓库，但不能登录到机器上的shell，如果尝试，可以看到登录拒绝：
```
$ ssh git@gitserver
fatal: Interactive git shell is not enabled
hint: ~/git-shell-commands should exist and have read and execute access.
Connectino to gitserver closed.
```

此时，用户仍然可以使用SSH端口转发(port forwarding)访问git服务器能够到达的任何主机，如果你想阻止这种行为，你可以编辑`authorized_keys`文件，并在你想要限制的每个密钥前添加以下选项：
```
no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty
```

结果应该像这样：
```
$ cat ~/.ssh/authorized_keys

no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCB007n/ww+ouN4gSLKssMxXnBOvf9LGt4LojG6rs6hPB09j9R/T17/x4lhJA0F3FR1rP6kYBRsWj2aThGw6HXLm9/5zytK6Ztg3RPKK+4kYjh6541N YsnEAZuXz0jTTyAUfrtU3Z5E003C4oxOj6H0rfIF1kKI9MAQLMdpGW1GYEIgS9EzSdfd8AcC IicTDWbqLAcU4UpkaX8KyGlLwsNuuGztobF8m72ALC/nLF6JLtPofwFBlgc+myivO7TCUSBd LQlgMVOFq1I2uPWQOkOWQAHukEOmfjy2jctxSDBQ220ymjaNsHT4kgtZg2AYYgPqdAv8JggJ ICUvax2T9va5 gsg-keypair

no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDEwENNMomTboYI+LJieaAY16qiXiH3wuvENhBG...
```
## 4.5 Git Daemon
接下来我们将设置一个使用“Git”协议提供仓库服务的守护进程，这是快速、未经认证访问您的Git数据的常见选择，请记住，由于这不是一个需要经过认证的服务，您通过此协议提供的任何内容在其网络内都是公开的

如果您在防火墙外的服务器上运行此服务，它应该仅用于对全世界公开可见的项目；如果您在防火墙内的服务器上运行它，您可能会使用它在当您不想为每个用户添加SSH密钥时，来为大量人员或计算机(持续集成或构建服务器)提供只读访问权限的项目

Git协议相对容易设置，基本上，您需要以守护进程的方式运行此命令：
```
$ git daemon --reuseaddr --base-path=/srv/git/ /srv/git/
```
`--reuseaddr`选项允许服务器在不等待旧连接超时的情况下重新启动，而`--base-path`选项允许人们在不指定完整路径的情况下克隆项目，末尾的路径告诉Git守护进程在哪里查找要导出(export)的仓库
如果您正在运行防火墙，您还需要在设置此服务的机器上的9418端口上打一个洞

您可以根据您正在运行的操作系统采用多种方式来守护化此进程

由于`systemd`是现代Linux发行版中最常用的初始化系统，您可以用它来达到这个目的，只需在`/etc/systemd/system/`目录下放置一个名为`git-daemon.service`的文件，内容如下：
```
[Unit]
Description=Start Git Daemon

[Service]
ExecStart=/usr/bin/git daemon --reuseaddr --base-path=/srv/git/ /srv/git/
Restart=always
RestartSec=500ms

StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=git-daemon

User=git
Group=git

[Install]
WantedBy=multi-user.target
```

你可能已经注意到，这里启动Git守护进程时使用的是`git`作为组和用户，实际可以根据你的需求进行修改，并确保提供的用户在系统中存在，同时，检查Git二进制文件是否真的位于`/usr/bin/git`，并在必要时更改路径

最后，你将运行`systemctl enable git-daemon`来在启动时自动启动服务，并且可以使用`systemctl start git-daemon`和`systemctl stop git-daemon`分别启动和停止服务

在其他系统上，你可能想使用`xinetd` ，或 `sysvinit`系统中的一个脚本或其他方式——只要你能以某种方式守护化并监视那个命令(get that command daemonized and watched)

接下来，你必须告诉Git允许哪些仓库进行未经认证的基于Git服务器的访问，你可以通过在每个仓库中创建一个名为`git-daemon-export-ok`的文件来实现这一点：
```
$ cd /path/to/project.gi
$ touch git-daemon-export-ok
```
该文件的存在告诉Git可以在不进行身份验证的情况下为该项目提供服务
## 4.6 Smart HTTP
我们现在通过SSH有认证访问，并通过`git://`有非认证访问，但还有一个协议可以同时做到这两点

设置智能HTTP基本上就是启用一个随Git提供的叫做`git-http-backend`的CGI脚本在服务器上，这个CGI会读取由`git fetch`或`git push`发送到HTTP URL的路径和头部，并确定客户端是否可以通过HTTP通信(自1.6.6版本，以来对任何客户端来说判断都为true)

如果CGI发现客户端是智能的，它将智能地与其通信；否则它将回退到愚蠢的行为(因此它对使用旧客户端的读取是向后兼容的)

让我们通过一个非常基本的设置来了解这个过程，我们将使用Apache作为CGI服务器来设置这个，如果你没有设置Apache，你可以在Linux系统上使用类似下面的命令来设置：
```
$ sudo apt-get install apache2 apache2-utils
$ a2enmod cgi alias env
```
这也启用了 `mod_cgi, mod_alias, mod_env` 模块，这些模块对于它正确工作是需要的

您还需要将`/srv/git`目录的Unix用户组设置为`www-data`，以便您的Web服务器能够读取和写入仓库，因为运行CGI脚本的Apache实例(默认情况下)将作为该用户运行：
```
$ chgrp -R www-data /srv/git
```

接下来，我们需要在Apache配置中添加一些内容，以便将`git-http-backend`作为处理进入Web服务器`/git`路径的任何请求的处理程序(handler)：
```
SetEnv GIT_PROJECT_ROOT /srv/git
SetEnv GIT_HTTP_EXPORT_ALL
ScriptAlias /git/ /usr/lib/git-core/git-http-backend/
```
如果您不设置`GIT_HTTP_EXPORT_ALL`环境变量，那么Git将仅向未经认证的客户端提供包含`git-daemon-export-ok`文件的仓库，就像Git守护进程所做的那样

最后，您可能希望告诉Apache允许对`git-http-backend`的请求(requests)，并通过某种方式(可能使用Auth块)进行认证，例如：
```
<Files "git-http-backend">
    AuthType Basic
    AuthName "Git Access"
    AuthUserFile /srv/git/.htpasswd
    Require !(%{QUERY_STRING} -strmatch '*service=git-receive-pack*' || %{REQUEST_URI} =~ m#/git-receive-pack$#)
    Require valid-user
</Files>
```
这将需要我们创建包含了有效用户密码的 `.htpasswd` 文件，添加用户的命令为：
```
$ htpasswd -c /srv/git/.htpasswd schacon
```
有很多方法可以让Apache对用户进行认证，您需要选择并实现其中一种
这只是我们能想到的最简单的例子，您几乎肯定还希望通过SSL设置这些，以确保所有数据都被加密

我们不想深入Apache配置的具体细节，因为您可能使用不同的服务器或有不同的认证需求，关键在于Git附带了一个名为`git-http-backend`的CGI，当被调用时，它会进行所有协商，以通过HTTP发送和接收数据，它本身不实现任何认证，但这可以很容易地在调用它的Web服务器层进行控制，您可以使用几乎所有支持CGI的Web服务器来做到这一点，所以请选择您最熟悉的那一个
## 4.7 GitWeb
## 4.8 GitLab
另一种更解耦的协作方式是使用合并请求(merge request)，这个功能允许任何可以看到项目的用户以受控的方式为其做出贡献：
具有直接访问权限的用户可以简单地创建一个分支，将提交推送到该分支，并从他们的分支打开一个合并请求到主分支( `master` )或其他任何分支
没有仓库推送权限的用户可以“分叉(fork)”它以创建自己的副本，将提交推送到他们的副本，并从他们的分叉打开一个合并请求回到主项目
这种模型允许所有者完全控制什么内容以及何时进入仓库，同时允许来自不受信任用户的贡献

合并请求和问题(issues)是GitLab中长期讨论(long-lived discussion)的主要单元，每个合并请求都允许对提议的更改进行逐行讨论(a line-by-line discussion of the proposed change)(这支持一种轻量级的代码审查)，以及一个总体的讨论线程(a general overal discussion thread)，两者都可以分配给用户，或组织到里程碑中
## 4.8 Third Party Hosted Options
如果您不想经历设置自己的Git服务器所涉及的所有工作，您有几个选择，可以将您的Git项目托管在外部专用托管网站(hosting site)上，这样做提供了许多优势：托管网站通常设置快速，易于启动项目，并且不涉及服务器维护或监控，即使您在内部设置并运行自己的服务器，您可能仍然希望使用公共托管网站来托管您的开源代码——这对于开源社区来说通常更容易找到并帮助您
# 5 Distributed Git
## 5.1 Distributed Workflows
### 5.1.1 Centralized Workflow
![[ProGit-Fig53.png]]
这意味着如果两个开发者从中心仓库克隆代码并都进行了更改，第一个推送更改的开发者可以无问题地完成推送，第二个开发者在推送更改之前必须合并第一个开发者的工作，以避免覆盖第一个开发者的更改，这个概念在Git中和在Subversion(或任何CVCS)都存在，并且这种模型在Git中工作得非常好

如果你的公司或团队已经习惯了集中式工作流程，你可以很容易地使用Git继续这种工作流程，只需设置一个单一的仓库，并给团队中的每个人推送权限；Git不会让用户互相覆盖

假设John和Jessica同时开始工作，John完成了他的更改并将其推送到服务器，然后Jessica尝试推送她的更改，但服务器拒绝了，她被告知她正在尝试推送非快进更改(non-fast-forward changes)，这无法做到，直到她获取并合并更改，这种工作流程对很多人来说是有吸引力的，因为它是许多人熟悉和舒适的范式

这也不限于小团队，借助Git的分支模型，数百名开发者可以通过数十个分支同时成功地在单个项目上工作
### 5.1.2 Integration-Manager Workflow
因为Git允许你拥有多个远程仓库，所以可以实现这样的工作流程：每个开发者都可以对自己的公共仓库有写入权限，并对其他人的仓库有读取权限
这种情况通常包括一个代表“官方”项目的规范仓库(canonical repository)，要为该项目做出贡献，你需要创建自己的公共项目克隆(clone)，并将自己的更改推送到其中

然后，你可以向主要项目的维护者发送请求，让他们拉取你的更改，维护者随后可以将你的仓库添加为远程仓库，本地测试你的更改，将它们合并到他们的分支中，并将更改推送回他们的仓库

该流程的工作方式如下：
1. 项目维护者推送到他们的公共仓库
2. 贡献者克隆那个仓库并进行更改
3. 贡献者推送到他们自己的公共仓库副本(public copy)
4. 贡献者发送电子邮件给维护者，请求他们拉取更改(pull changes)
5. 维护者将贡献者的仓库添加为远程仓库并本地合并(locally)
6. 维护者将合并后的更改推送到主仓库(main repository)
![[ProGit-Fig54.png]]

这是一种在基于中心仓库的工具，如GitHub或GitLab中非常常见的工作流程，你可以很容易地分叉一个项目，并将你的更改推送到你的分叉中，让每个人都能看到
这种方法的一个主要优点是你可以继续工作，而主仓库的维护者随时都可以拉取你的更改，贡献者不必等待项目合并他们的更改——每一方都可以按照自己的节奏工作
### 5.1.3 Dictator and Lieutenants Workflow
独裁者与副手工作流是一种多仓库工作流程的变体，它通常被拥有数百名协作者的大型项目(huge projects)使用，一个著名的例子是Linux内核
不同的集成管理者负责仓库的某些部分，他们被称为副手(lieutenants)，所有副手都有一个集成管理者(inetgration manager)，被称为仁慈的独裁者，仁慈的独裁者从他们的目录推送到一个参考仓库(reference repository)，所有协作者也都需要从这个参考仓库拉取，该流程的工作方式如下(见仁慈的独裁者工作流程)：
1. 普通开发者在他们的主题分支上工作，并将他们的工作在 `master` 分支上进行变基，`master` 分支是仁慈的独裁者推送到的参考仓库的主分支
2. 副手将开发者的主题分支合并到他们的 `master` 分支中
3. 独裁者将副手的 `master` 分支合并到独裁者自己的 `master` 分支中
4. 最后，独裁者将那个 `master` 分支推送到参考仓库，以便其他开发者可以基于它进行变基
![[ProGit-Fig55.png]]
这种工作流程并不常见，但在非常大的项目或高度层级化的环境中可能很有用
它允许项目负责人(独裁者)将大部分工作委托出去，并在多个点收集大量的代码子集(collect large subsets of code at multiple points)，然后再将它们整合起来
## 5.2 Contributing to a Project
因为Git非常灵活，人们可以以许多不同的方式一起工作，每个项目的贡献方式都有点不同，涉及的一些变量包括活跃贡献者数量、选择的工作流程、你的提交权限，以及可能的外部贡献方法

第一个变量是活跃贡献者数量——有多少用户正在积极为这个项目贡献代码，以及他们贡献的频率如何？
在许多情况下，你可能有两三个开发者每天提交几次，或者对于有些休眠的项目来说可能更少，对于更大的公司或项目，开发者的数量可能达到数千人，每天有数百或数千次提交
因为随着开发者数量的增加，你会遇到更多确保你的代码可以干净地应用或容易合并的问题，你在工作时或你的更改等待被批准或应用时合并进来的工作可能会使你提交的更改变得过时或严重破坏
因此你要考虑如何保持你的代码始终是最新的，并且你的提交有效？

下一个变量是项目使用的工作流程
它是集中式的，每个开发者都有平等的写入主代码线的权限吗？项目是否有一个维护者或集成管理者检查所有的补丁(checks all the patches)？所有的补丁都是经过同行评审(peer-reviewed)和批准的吗？你是否参与了这个过程？是否有副手系统，你是否需要先将你的工作提交给他们？

下一个变量是你的提交权限
如果你有写入项目的权限，那么贡献到项目所需的工作流程会与没有权限时大不相同，如果你没有写入权限，项目如何接受贡献的工作？它甚至有政策吗？你一次贡献了多少工作？你多久贡献一次？

所有这些问题都可能影响你如何有效地为项目做出贡献，以及哪些工作流程对你来说是首选或可用的，我们将通过一系列用例来介绍这些方面的每个方面，从简单到更复杂，你应该能够从这些示例中构建出你在实践中需要的特定工作流程
### 5.2.1 Commit Guidelines
Git项目提供了一个文档，列出了创建提交以提交补丁的许多好建议——你可以在Git源代码的 `Documentation/SubmittingPatches` 文件中阅读它

首先，你的提交不应当包含任何空白错误(whitespace errors)，
Git提供了一个简单的方法来检查这一点——在提交之前，运行`git diff --check`，它将识别可能的空白错误并将它们列出

接下来，请尝试使每次提交都是一个逻辑上独立的变更集，
如果可能的话，尽量使你的更改易于理解——不要在周末连续编码五天解决五个不同的问题，然后在周一将它们全部作为一个巨大的提交提交；即使你在周末没有提交，也要在周一使用暂存区将你的工作至少分成每个问题(issue)一个提交，并为每个提交附上有用的信息
如果一些更改修改了同一个文件，请尝试使用`git add --patch`来部分暂存文件(partially stage files)，只要你在某个时候添加了所有的更改，无论你是做一次提交还是五次提交，项目快照在分支的尖端都是相同的，所以尽量让其他开发者在审查你的更改时更轻松
这种方法也使得在需要时更容易提取或回滚一个变更集，在将工作发送给其他人之前，请打造一个干净且易于理解的历史记录

最后需要记住的是提交信息,
养成创建高质量提交信息的习惯，可以使使用和与Git协作变得更加容易，
一般来说，你的消息应该以一行不超过大约50个字符的简短描述开始，这行描述应简洁地概述变更集，然后是一个空行，紧接着是更详细的解释，Git项目要求更详细的解释包括你变更的动机以及与之前行为的对比——这是一个很好的遵循准则；
以祈使句(imperative)写你的提交信息：“Fix bug”，而不是“Fixed bug”或“Fixes bug”；

这里有一个你可以遵循的模板，我们已从Tim Pope最初编写的版本中稍作调整：
```
Capitalized, short (50 chars or less) summary

More detailed explanaory text, if necessary. Warp it to about 72 characters or so. In some contexts, the first line is treated as the subject of an email and the rest of the text as the body. The blank line separating the summary frmo the body is critical (unless you omit the body entirely); tools like rebase will confuse you if you run the two together.

Write your commit message in the imperative: "Fix bug" and not "Fixed bug" or "Fixed bug." This conventino matches up with commit messages generated by commands like git merge and git revert.

Further paragraphs come after blank lines.

- Bullet points are okay, too

- Typically a hyphen or asterisk is used for the bullet, followed by a single space, with blank lines in between, but conventions vary here

- Use a hanging indent
```
### 5.2.2 Private Small Team
你可能遇到的最简单的设置是一个只有一两个其他开发者参与的私有项目
在这里，“私有”意味着闭源——对外界不可访问，你和其他开发者都有向仓库推送的权限

在这个环境中，你可以遵循一种类似于使用Subversion或其他集中式系统时的工作流程，你仍然可以获得诸如离线提交以及更简单的分支和合并等优势，但工作流程可能非常相似；主要的区别是合并发生在客户端而不是在提交时的服务器端

让我们看看当两个开发者开始使用共享仓库一起工作时可能是什么样子
第一个开发者，约翰，克隆了仓库，进行了更改，并在本地提交了，在这些示例中，协议消息已被替换为`...`以缩短它们：
```
# John's Machine

$ git clone john@githost:simplegit.git Cloning into 'simplegit'...

...

$ cd simplegit/ $ vim lib/simplegit.rb

$ git commit -am 'Remove invalid default value' [master 738ee87] Remove invalid default value

1 files changed, 1 insertions(+), 1 deletions(-)
```
第二个开发者杰西卡，同样克隆了仓库，并本地提交了修改：
```
# Jessica's Machine

$ git clone jessica@githost:simplegit.git Cloning into 'simplegit'...

...

$ cd simplegit/

$ vim TODO

$ git commit -am 'Add reset task' [master fbff5bc] Add reset task

1 files changed, 1 insertions(+), 0 deletions(-)
```
杰西卡将工作推送至服务器，没有问题：
```
# jessica's Machine
$ git push origin master
...
To jessica@githost:simplegit.git
    1edee6b..fbff5bc master -> master
```
上面的输出的最后一行显示了来自推送操作的有用返回消息
基本格式是 `<oldref>..<newref> fromref → toref`，其中 `oldref` 表示旧的引用，`newref` 表示新的引用，`fromref` 是正在推送的本地引用的名称，`toref` 是正在更新的远程引用的名称

在下面的讨论中，你会看到类似的输出，因此对这些含义有一个基本的了解将有助于理解仓库的各种状态，更多详细信息可在 `git-push` 的文档中找到

之后John也希望推送：
```
# John's Machine
$ git push origin master To john@githost:simplegit.git
! [rejected] master -> master (non-fast forward)
error: failed to push some refs to 'john@githost:simplegit.git'
```

在这种情况下，由于杰西卡早些时候推送了她的更改，约翰的推送失败了
两个开发者并没有编辑同一个文件，尽管Subversion会在服务器上自动合并不同文件的更改(if different files are edited)，但使用Git，你必须首先在本地合并提交，换句话说，在被允许推送之前，约翰必须先获取杰西卡的上游更改，并将它们合并到他的本地仓库中

作为第一步，约翰获取了杰西卡的工作(这只获取了杰西卡的上游工作，尚未将其合并到约翰的工作中)：
```
$ git fetch origin
...
From john@githost:simplegit
  + 049d078...fbff5bc master -> origin/master
```

此时John的本地仓库看起来像这样：
![[ProGit-Fig57.png]]
然后John就可以将Jessica的工作进行本地合并：
```
$ git merge origin/master
Merge made by the 'recursive' strategy.
  TODO |    1+
  1 files changed, 1 insertions(+), 0 deletions(-)
```
然后我们得到：
![[ProGit-Fig58.png]]

此时，约翰可能想要测试这段新代码，以确保杰西卡的工作不会影响他的任何工作，并且只要一切看起来都很好，他最终可以将新合并的工作推送到服务器上：
```
$ git push origin master
...
To john@githost:simplegit.git
   fbff5bc..72bbc59  master -> master
```

最后，约翰的提交历史将会是这样：
![[ProGit-Fig59.png]]

此时Jessica已经创建了一个新的主题分支 `issue54` ，并在该分支上进行了三次提交，她的提交历史看起来是：
![[ProGit-Fig60.png]]

Jessica可以获取服务器的新数据：
```
$ Jessica's Machine
$ git fetch origin
...
From jessica@githost:simplegit
   fbff5bc..72bbc59 master   ->    origin/master
```

此时Jessica的提交历史看起来是：
![[ProGit-Fig61.png]]

Jessica认为她的主题分支已经可以合并入主分支，为此她想要知道John的具体工作，因此使用 `git log` ：
```
$ git log --no-merges issue54..origin/master
commit  738ee872852dfaa9d6634e0dea7a324040193016
Author: John Smith <jsmith@example.com>
Date:  Fri May 29 16:01:27 2009 -0700

  Remove invalid default value
```
其中的 `issue54..origin/master` 语法是一个日志过滤器(log filter)，它使得Git仅展示在后面的分支( `origin/master` )上且不在前面的分支( `issue54` )上的提交

从上述输出中，我们可以看到有一个约翰所做的提交是杰西卡尚未合并到她的本地工作中的，杰西卡可以将她的专题工作合并到她的主分支中，将约翰的工作(`origin/master`)合并到她的主分支中，然后再推送回服务器

首先(在她的`issue54`专题分支上提交了所有工作后)，杰西卡切换回她的主分支，准备整合所有这些工作：
```
$ git checkout master
Switched to branch 'master'
Your branch is behind 'origin/master' by 2 commits, and can be fast forwarded
```

杰西卡可以先合并`origin/master`或`issue54`——它们都是上游分支，所以顺序无关紧要，无论她选择哪种顺序，最终的快照应该是相同的，只有历史记录会有所不同

她选择先合并`issue54`分支：
```
$ git merge issue54
Updating fbff5bc..4af4298 
Fast forward
  README | 1 +
  lib/simplegit.rb | 6 +++++-
  2 files changed, 6 insertions(+), 1 deletions(-)
```
没有出现问题；正如你看到的，这是一个简单的快进合并

杰西卡现在通过合并约翰之前获取的在`origin/master`分支中的工作来完成本地合并过程：
```
$ git merge origin/master 
Auto-merging lib/simplegit.rb
Merge made by the 'recursive' strategy.
  lib/simplegit.rb | 2 +-
  1 files changed, 1 insertions(+), 1 deletions(-)
```
一切都干净地合并了，杰西卡的历史记录现在看起来像这样：
![[ProGit-Fig62.png]]

现在`origin/master`可以从杰西卡的 `master` 分支到达，所以她应该能够成功推送(假设在此期间约翰没有推送更多的更改)：
```
$ git push origin master
...
To jessica@githost:simplegit.git 
  72bbc59..8059c15 master -> master
```
![[ProGit-Fig63.png]]

这是最简单的工作流程之一，你工作一段时间后(通常在一个主题分支上)，当工作准备好集成时，将其合并到你的主分支中，当你想要分享这项工作时，如果源仓库的主分支有变更，你拉取并合并你的主分支到源仓库的主分支，最后推送到服务器上的主分支
### 5.2.3 Private Managed Team
假设约翰和杰西卡正在共同开发一个功能(我们称之为“featureA”)，同时杰西卡和第三位开发者乔茜正在开发第二个功能(比如说，“featureB”)，在这种情况下，公司使用一种集成-管理者(integration-manager)的工作流程，其中只有特定的工程师才能集成各个小组的工作，并且主仓库的主分支只能由这些工程师更新，在这个场景中，所有的工作都是在基于团队的分支上完成的，然后由集成者稍后合并在一起

让我们跟随杰西卡的工作流程，看看她是如何在这个环境中与两位不同的开发者并行合作开发她的两个功能的，假设她已经克隆了她的仓库，她决定首先开始开发featureA，她为这个功能创建了一个新的分支，并在那里进行了一些工作：
```
# Jessica's Machine
$ git checkout -b featureA
Switched to a new branch 'featureA' 
$ vim lib/simplegit.rb
$ git commit -am 'Add limit to log function' 
[featureA 3300904] Add limit to log function
  1 files changed, 1 insertions(+), 1 deletions(-)
```
在这一点上，她需要与约翰共享她的工作，所以她将她的`featureA`分支提交推送到服务器上

杰西卡没有推送到主分支的权限——只有集成者才有——所以她必须推送到另一个分支，以便与约翰协作
```
$ git push -u origin featureA 
... 
To jessica@githost:simplegit.git
  * [new branch] featureA -> featureA
```

杰西卡通过电子邮件告诉约翰，她已经将一些工作推送到了一个名为`featureA`的分支，并且他现在可以查看，在等待约翰的反馈的同时，杰西卡决定开始和乔西一起工作在`featureB`上，为了开始，她创建了一个新的特性分支，并以服务器的主分支为基础：
```
# Jessica's Machine
$ git fetch origin
$ git checkout -b featureB origin/master 
Switched to a new branch 'featureB'
```

杰西卡在 `featureB` 上提交了几次工作之后，她的提交历史看起来是这样：
![[ProGit-Fig65.png]]
她准备推送她的工作，但收到了乔西的电子邮件，说一些初步的“featureB”工作已经被推送到服务器上的一个名为`featureBee`的分支
在杰西卡可以将她的工作推送到服务器之前，她需要将这些更改与自己的合并，杰西卡首先使用`git fetch`命令获取乔西的更改：
```
$ git fetch origin 
...
From jessica@githost:simplegit
  * [new branch] featureBee -> origin/featureBee
```

假设杰西卡仍然在她的检出(checked-out)的`featureB`分支上，她现在可以使用`git merge`命令将乔西的工作合并到那个分支中：
```
$ git merge origin/featureBee 
Auto-merging lib/simplegit.rb
Merge made by the 'recursive' strategy.
  lib/simplegit.rb | 4 ++++
  1 files changed, 4 insertions(+), 0 deletions(-)
```

此时，杰西卡想要将所有合并后的“featureB”工作推送回服务器，但她不想仅仅推送自己的`featureB`分支，相反，由于乔西已经创建了一个上游的`featureBee`分支，杰西卡希望推送到那个分支，她通过以下命令来实现：
```
git push origin featureB:featureBee
...
To jessica@githost:simplegit.git
  fba9af8..cd685d1 featureB -> featureBee
```
这被称为引用规范(refspec)，还要注意`-u`标志；这是`--set-upstream`的简写，它配置了分支，以便以后更容易地推送和拉取

突然，杰西卡收到了约翰的电子邮件，他告诉她他已经推送了一些更改到他们正在合作的`featureA`分支上，并要求杰西卡看看它们，再次，杰西卡运行了一个简单的`git fetch`来获取服务器上的所有新内容，包括(当然)约翰的最新工作：
```
git fetch origin
...
From jessica@githost:simplegit
  3300904..aad881d featureA -> origin/featureA
```

杰西卡可以通过比较新获取的`featureA`分支的内容和她本地相同分支的副本来显示出约翰的新工作： 
```
git log featureA..origin/featureA
commit aad881d154acdaeb2b6b18ea0e827ed8a6d671e6
Author: John Smith <jsmith@example.com>
Date: Fri May 29 19:57:33 2009 -0700
   
  Increase log output to 30 from 25
```
这条命令会显示从远程分支`origin/featureA`到本地分支`featureA`的所有提交记录，即约翰所做的更改

如果杰西卡认为该工作可以，她就会将其合并入自己的 `featureA` 分支：
```
$ git checkout featureA
Switched to branch 'featureA'
$ git merge origin/featureA
Updating 3300904..aad881d 
Fast forward
  lib/simplegit.rb | 10 +++++++++-
1 files changed, 9 insertions(+), 1 deletions(-)
```

最后，杰西卡可能想要对所有合并后的内容做一些小的修改，所以她可以自由地进行这些修改，将它们提交到她的本地`featureA`分支，并将最终结果推送回服务器：
```
$ git commit -am 'Add small tweak to merged content' 
[featureA 774b3ed] Add small tweak to merged content
  1 files changed, 1 insertions(+), 1 deletions(-) 
$ git push
...
To jessica@githost:simplegit.git
  3300904..774b3ed featureA -> featureA
```
这些命令会将杰西卡对合并后内容所做的修改添加到暂存区，提交到本地的`featureA`分支，并推送到服务器上的同名分支

此时她的提交历史看起来是：
![[ProGit-Fig66.png]]

在某个时候，杰西卡、乔西和约翰通知集成人员，服务器上的`featureA`和`featureBee`分支已经准备好合并到主线中，集成人员将这些分支合并到主线后，执行`fetch`操作会拉取新的合并提交，使历史记录看起来像这样：
![[ProGit-Fig67.png]]
许多团队因为Git具有这种能力而转向使用它：让多个团队并行工作，在流程的后期合并不同的工作线，小团队或团队的子组能够通过远程分支协作，而不必一定要涉及或妨碍整个团队，这是Git的一个巨大优势
### 5.2.4 Forked Public Project
向公共项目贡献代码略有不同，因为你没有权限直接更新项目上的分支，你必须通过其他方式将工作提交给维护者

第一个例子描述了在支持简单分叉的Git托管服务上通过分叉来贡献，许多托管站点支持这一点(包括GitHub、BitBucket、repo.or.cz等)，并且许多项目维护者期望这种样式的贡献
下一节将讨论那些更倾向于通过电子邮件接受贡献补丁的项目

首先，你可能会想要克隆主仓库，为你计划贡献的补丁或补丁系列创建一个主题分支，并在那里进行你的工作，基本的步骤如下：
```
$ git clone <url>
$ cd project
$ git checkout -b featureA
  ... work ...
$ git commit
  ... work ...
$ git commit
```

当你的分支工作(branch work)完成，准备将其贡献回维护者时，前往原始项目页面并点击“Fork”按钮，创建你自己可写的项目分叉，然后你需要将这个仓库的URL添加为你本地仓库的一个新远程仓库；在这个例子中，我们称它为`myfork`：
```
$ git remote add myfork <url>
```

然后，你需要将你的新工作推送到这个仓库，最简单的方法是直接将你正在工作的主题分支推送到你分叉的仓库，而不是将这项工作合并到你的主分支再推送它，
这样做的话，如果你的工作没有被接受或者被挑选(cherry-picked)，你就不必回滚(rewind)你的主分支，如果维护者合并 `merge` 、变基 `rebase` 或挑选`cherry-pick` 了你的工作，你最终无论如何都会通过从他们的仓库拉取来得到它

你可以使用以下命令推送你的工作：
```
$ git push -u myfork featureA
```
一旦你的工作被推送到你对该仓库的分支上，你需要通知原始项目的维护者，你有一些工作希望他们合并，这通常被称为“拉取请求(pull request)”，你通常可以通过网站生成这样的请求——GitHub有自己的“拉取请求”机制——或者你可以运行`git request-pull`命令，并将随后的输出手动通过电子邮件发送给项目维护者

`git request-pull`命令需要你指定你希望将主题分支(topic branch)合并到的基础分支(base branch)，以及你希望他们从中拉取的Git仓库URL，并生成你要求被拉取的所有更改的摘要

例如，如果Jessica想要向John发送一个拉取请求，并且她刚刚推送了两个提交到她的主题分支上，她可以运行以下命令：
```
$ git request-pull origin/master myfork
The following changes since commit 1edee6b1d61823a2de3b09c160d7080b8d1b3a40:
Jessica Smith (1):
        Create new function

are available in the git repository at:

    https://githost/simplegit.git featureA

Jessica Smith (2):
        Add limit to log function
        Increase log output to 30 from 25

    lib/simplegit.rb |    10 ++++++++-
    1 files changes, 9 insertions(+), 1 deletions(-)
```
这个输出可以发送给维护者——它告诉他们工作是从哪里分支出来的，总结了提交的内容，并识别了新工作将从哪里拉取

在我们不是维护者的项目中，我们一般会有一个分支 `master` 始终追踪 `origin/master` ，并且我们会在主题分支中工作，
这样做的好处在于，如果这些主题分支被维护者拒绝，我们也可以直接丢弃它，将工作主题隔离到主题分支中，也使我们更容易在主仓库的提交尖端在我们工作期间移动的情况下，变基我们的新工作

例如，如果您想向项目提交第二项工作主题，不要继续在您刚刚推送的主题分支上工作，而是从主存储库的`master`分支重新开始：
```
$ git checkout -b featureB origin/master
  ...work...
$ git commit
$ git push myfork featureB
$ git request-pull origin/master myfork
  ... email generated request pull to maintainer ...
$ git fetch origin
```
现在，你的每个主题都被包含在一个独立的信息孤岛中——类似于补丁队列(patch queue)——你可以重写、变基和修改它们，而不必担心主题之间相互干扰或相互依赖，就像这样：
![[ProGit-Fig69.png]]

让我们假设项目维护者已经合并了一批其他的补丁，并尝试了你的第一条分支，但由于他已经合并了其他的补丁，现在已经无法直接干净地合并了
在这种情况下，你可以尝试将该分支变基到`origin/master`之上，为维护者解决冲突，然后重新提交你的更改：
```
$ git checkout featureA
$ git rebase origin/master
$ git push -f myfork featureA
```
此时我们的历史看起来是：
![[ProGit-Fig70.png]]

因为你变基了分支，你需要在推送命令中指定`-f`参数，以便能够用不是其后代的提交来替换服务器上的 `featureA` 分支
另一种选择是将这个新的工作推送到服务器上的一个不同分支(可能叫做`featureAv2`)

让我们再来看一个可能的场景：维护者查看了你第二个分支中的工作，并喜欢这个概念，但希望你改变一个实现细节，你也会利用这个机会将工作基于项目当前的 `master` 分支
你从当前的`origin/master`分支开始一个新的分支，将 `featureB` 的更改压缩到那里，解决任何冲突，做出实现上的改变，然后将其作为一个新的分支进行推送：
```
$ git checkout -b featureBv2 origin/master
$ git merge --squash featureB
  ... change implementation ...
$ git commit
$ git push myfork featureBv2
```
`--squash` 选项会将被合并分支(merged branch)上的所有工作压缩成一个更改集(changeset)，产生的状态就像实际发生了合并一样，但实际上并没有创建合并提交
这意味着你未来的提交将只有一个父提交，并允许你引入另一个分支的所有更改，并且可以在记录新的提交之前再进行更多的更改

此外，在默认合并过程中(in the case of the default merge process)， 如果需要延迟合并提交(delay the merge commit)，`--no-commit` 选项可能会很有用

此时，你可以通知维护者你已经完成了所请求的更改，并且他们可以在你的 `featureBv2` 分支中找到这些更改

![[ProGit-Fig71.png]]
### 5.2.5 Public Project over Email
许多项目都有接受补丁的既定程序——你需要检查每个项目的具体规则，因为它们会有所不同，有一些较老、较大的项目通过开发者邮件列表接受补丁，我们现在将举一个这样的例子

工作流程与之前的用例类似——你为每个补丁系列(patch series)创建主题分支，不同之处在于你如何将它们提交给项目，在本例中，你不再需要分叉项目并推送到你自己可写的版本(push to your own writable version)，而是为每个提交系列(commit serise)生成电子邮件版本，并将它们发送到开发者邮件列表(developer mailing list)：
```
$ git checkout -b topicA
  ... work ...
$ git commit
  ... work ...
$ git commit
```
现在，你有两次提交想要发送到邮件列表，你可以使用 `git format-patch` 命令生成 mbox 格式的文件，然后你可以将这些文件作为电子邮件发送到邮件列表——它将每个提交转换为一封电子邮件，以提交信息的第一行为主题，其余的信息加上提交引入的补丁作为正文

这种方式的好处是，使用 `format-patch` 生成的电子邮件中的补丁应用后，可以正确地保留所有的提交信息
```
$ git format-patch -M origin/master
0001-add-limit-to-log-function.patch
0002-increase-log-output-to-30-from-25.patch
```
`format-patch` 命令会打印出它创建的补丁文件的名称。`-M` 开关告诉 Git 寻找重命名的文件

最终生成的文件看起来像这样：
```
$ cat 0001-add-limit-to-log-function.patch
From 330090432754092d704da8e76ca5c05c198e71a8 Mon Sep 17 00:00:00 2001
From: Jessica Smith <jessica@example.com>
Date: Sun, 6 Apr 2008 10:17:23 -0700
Subject: [PATCH 1/2] Add limit to log function

Limit log functionality to the first 20

---
  lib/simplegit.rb | 2 +-
  1 files changed, 1 insertions(+), 1 deletions(-)

diff --git a/lib/simplegit.rb b/lib/simplegit.rb
index 76f47bc..f9815f1 100644
--- a/lib/simplegit.rb
+++ b/lib/simple.rb
@@ -14,7 +14,7 @@ class SimpleGit
   end

   def log(treeish = 'master')
-      command("git log #{treeish}")
+      command("git log -n 20 #{treeish}")
   end

   def ls_tree(treeish = 'master')
---
2.1.0
```
你还可以编辑这些补丁文件，以添加更多你不希望出现在提交信息中的信息到邮件列表，如果你在 `---` 行和补丁开始的地方(即 `diff --git` 行)之间添加文本，开发者可以阅读它，但这些内容会被补丁应用过程忽略

要将这个邮件发送到邮件列表，你可以直接将文件粘贴到你的电子邮件程序中，或者通过命令行程序发送
粘贴文本经常会导致格式问题，特别是使用“更智能”的客户端时，它们可能不会适当地保留换行符和其他空白，幸运的是，Git 提供了一个工具来帮助你通过 IMAP 发送正确格式化的补丁，这可能对你来说更简单

我们将演示如何通过 Gmail 发送补丁，这是我们最熟悉的电子邮件代理；你可以在 Git 源代码中的 `Documentation/SubmittingPatches` 文件的末尾阅读到许多邮件程序的详细说明

首先，你需要在 `~/.gitconfig` 文件中设置 imap 部分，你可以使用一系列 `git config` 命令单独设置每个值，或者你可以手动添加它们，但最终你的配置文件应该看起来像这样：
```
[imap]
  folder = "[Gmail]/Drafts"
  host = imaps://imap.gmail.com
  user = user@gmail.com
  pass = YXJ8g76G_2^sFbd
  port = 993
  sslverify = false
```
如果你的 IMAP 服务器不使用 SSL，最后两行可能就没有必要了，主机值(host value)将是 `imap://` 而不是 `imaps://`

设置完成后，你可以使用 `git imap-send` 将补丁系列放置在指定 IMAP 服务器的草稿文件夹(Drafts folder)中：
```
$ cat *.patch | git imap-send
Resolving imap.gmail.com... ok
Connecting to [74.125.142.109]:993... ok
Logging in...
sendding 2 messages
100% (2/2) done
```
此时，你应该能够进入你的草稿文件夹，将收件人字段更改为你发送补丁的邮件列表，可能还要抄送(carbon copy)给维护者或负责该部分的人，然后发送出去

你也可以通过 SMTP 服务器发送补丁，和之前一样，你可以使用一系列 `git config` 命令单独设置每个值，或者你可以手动在 `~/.gitconfig` 文件的 `sendemail` 部分添加它们：
```
[sendemail]
  smtpencryption = tls
  smtpserver = smtp.gmail.com
  smtpuser = user@gmail.com
  smtpserverport = 587
```
此时，你可以使用 `git send-email` 发送我们的补丁
```
$ git send-email *.patch
0001-add-limit-to-log-function.patch 
0002-increase-log-output-to-30-from-25.patch
Who should the emails appear to be from? [Jessica Smith <jessica@example.com>]
Emails will be sent from: Jessica Smith <jessica@example.com> Who should the emails be sent to? jessica@example.com 
Message-ID to be used as In-Reply-To for the first email? y
```

然后，Git 会为你要发送的每个补丁输出一堆日志信息，看起来像这样：
```
(mbox) Adding cc: Jessica Smith <jessica@example.com> from
  \line 'From: Jessica Smith <jessica@example.com>'
OK. Log says:
Sendmail: /usr/sbin/sendmail -i jessica@example.com 
From: Jessica Smith <jessica@example.com>
To: jessica@example.com
Subject: [PATCH 1/2] Add limit to log function 
Date: Sat, 30 May 2009 13:29:15 -0700
Message-Id: <1243715356-61726-1-git-send-email-jessica@example.com> 
X-Mailer: git-send-email 1.6.2.rc1.20.g8c5b.dirty
In-Reply-To: <y> 
References: <y>

Result: OK
```
## 5.3 Maintaining a Project
除了知道如何有效地为项目做出贡献外，你很可能还需要知道如何维护一个项目，这可能包括接受和应用通过 `format-patch` 生成并发送到你邮箱的补丁，或者整合(integrate changes)远程仓库中的远程分支的更改

无论你是维护一个官方仓库(canonical repository)，还是想通过验证或批准补丁来提供帮助，你都需要知道如何以一种对其他贡献者最清晰、对你长期而言可持续的方式来接受工作(accept work)
### 5.3.1 Working in Topic Branches
当你考虑集成新工作时，通常最好在一个主题分支中尝试——即专门为尝试新工作而创建的临时分支，这样，你可以轻松地单独调整补丁，并在它不起作用时暂时抛弃它，直到你有时间回来处理它

为了方便你在不得不暂时放弃它并在以后回来，你可以根据你将要尝试的工作的主题创建一个简单的分支名称，比如 `ruby_client` 或其他类似的描述性名称，以方便记忆

Git 项目的维护者也倾向于对这些分支进行命名空间划分——比如 `sc/ruby_client`，其中 `sc` 是贡献工作的人的简称

你可以基于你的主分支这样创建分支：
```
$ git branch sc/ruby_client master
```

或者，如果你想立即切换到它，你可以使用 `checkout -b` 选项：
```
$ git checkout -b sc/ruby_client master
```
现在你已经准备好将你收到的贡献工作添加到这个主题分支中，并确定是否要将它合并到你的长期分支中
### 5.3.2 Applying Patches from Email
如果你通过电子邮件收到一个你需要集成到你的项目中的补丁，你需要在你的主题分支上应用这个补丁来评估它

应用电子邮件发送的补丁有两种方式：使用`git apply`或者使用`git am`

**Applying a Patch with `am`**
如果贡献者是一名Git用户，并且足够细心地使用`format-patch`命令来生成他们的补丁，那么你的工作就更容易了，因为补丁包含了作者信息和提交信息供你使用

如果可能的话，鼓励你的贡献者使用`format-patch`而不是`diff`来为你生成补丁，以方便你只需要在遇到遗留补丁(legacy patches)和其他类似的东西的时候才使用`git apply`

要应用由`format-patch`生成的补丁，你应该使用`git am`(这个命令之所以命名为`am`，是因为它用于“从一个邮箱中应用一系列补丁 apply a series of patches from a mailbox”)，技术上，`git am`是为了读取mbox文件而构建的，mbox是一个简单的纯文本(plain-text)格式，用于在一个文本文件中存储一个或多个电子邮件消息(email messages)，它看起来像这样：
```
From 330090432754092d704da8e76ca5c05c198e71a8 Mon Sep 17 00:00:00 2001
From: Jessica Smith <jessica@example.com>
Date: Sun, 6 Apr 2008 10:17:23 -0700
Subject: [PATCH 1/2] Add limit to log function

Limit log functionality to the first 20
```
这是`git format-patch`命令输出的开始；它也代表了有效的mbox电子邮件格式

如果有人使用`git send-email`正确地将补丁通过电子邮件发送给你，并且你将其下载为mbox格式，那么你可以让`git am`指向该mbox文件，它将开始应用它所看到的所有补丁，如果你运行的邮件客户端可以将多个电子邮件保存为mbox格式，你可以将整个补丁系列(entire patch series)保存到一个文件中，然后使用`git am`逐个应用它们

然而，如果有人是首先通过`git format-patch`生成了补丁文件，再将其上传到票务系统或类似的东西中(而不是用 `git send-email` )，你可以将该文件保存在本地，然后将保存在磁盘上的该文件传递给`git am`以应用它：
```
$ git am 0001-limit-log-function.patch
Applying: Add limit to log function
```

你可以看到它被干净地应用了，并且自动为你创建了新的提交，提交的作者信息是从电子邮件的`From`和`Date`头部获取的，提交的消息是从电子邮件的`Subject`和正文(在补丁之前)获取的

例如，如果这个补丁是从上面的mbox示例应用的，生成的提交看起来会像这样：
```
$ git log --pretty=fuller -1
commit 6c5e70b984a60b3cecd395edd5b48a7575bf58e0
Author: Jessica Smith <jessica@example.com>
AuthorDate: Sun Apr 6 10:17:23 2008 -0700
Commit: Scott Chacon <schacon@gmail.com>
CommitDate: Thu Apr 9 09:19:06 2009 -0700

  Add limit to log function

  Limit log functionality to the first 20
```
`Commit` 信息显示了应用补丁的人和应用的时间，`Author` 信息是最初创建补丁的个人以及最初创建补丁的时间

但有可能补丁不会干净地应用，也许你的主分支与创建补丁的分支(the branch the patch was built from)差异太大，或者补丁依赖于你尚未应用的其他补丁，在这种情况下，`git am`过程将失败，并询问你想做什么：
```
$ git am 0001-see-if-this-helps-the-gem.patch
Applying: See if this helps the gem
error: patch failed: ticgit.gemspec:1 
error: ticgit.gemspec: patch does not apply 
Patch failed at 0001.
When you have resolved this problem run "git am --resolved".
If you would prefer to skip this patch, instead run "git am --skip". To restore the original branch and stop patching run "git am --abort".
```
这个命令会在它遇到问题的任何文件中放置冲突标记(confilct markers)，就像一个有冲突的合并或变基操作一样

你解决这个问题的方式也大致相同——编辑文件以解决冲突，将新文件暂存，然后运行`git am --resolved`以继续应用下一个补丁：
```
$ (fix the file)
$ git add ticgit.gemspec
$ git am --resolved
Applying: See if this helps the gem
```
如果你想让Git更智能地尝试解决冲突，你可以向它传递一个`-3`选项，这会让Git尝试进行三方合并，这个选项默认不开启，因为如果补丁所基于的提交不在你的仓库中，它就不起作用，如果你确实有那个提交——如果补丁是基于一个公共提交——那么`-3`选项在应用有冲突的补丁时通常会更聪明：
```
$ git am -3 0001-see-if-this-helps-the-gem.patch
Applying: See if this helps the gem
error: patch failed: ticgit.gemspec:1
error: ticgit.gemspec: patch does not apply 
Using index info to reconstruct a base tree... 
Falling back to patching base and 3-way merge... 
No changes -- Patch already applied.
```
在这种情况下，如果没有使用 `-3` 选项，补丁会被认为存在冲突，由于使用了 `-3` 选项，补丁被干净地应用了

如果你正在从 mbox 文件中应用多个补丁，你也可以以交互模式运行 `am` 命令，它会在找到每个补丁时停下来，并询问你是否想要应用它：
```
$ git am -3 -i mbox
Commit Body is:
-------------------------- 
See if this helps the gem 
--------------------------
Apply? [y]es/[n]o/[e]dit/[v]iew patch/[a]ccept all
```
这在你保存了多个补丁时很有用，因为如果你不记得它是什么，你可以先查看补丁，或者如果你已经应用过了，可以选择不应用它

当你的主题的所有补丁都被应用并提交到你的分支后，你可以选择是否以及如何将它们集成到一个长期运行的分支中
### 5.3.3 Checking Out Remote Branches
如果你的贡献来自一个设置了自己的仓库、推送了一些更改进去，然后发送给你他的仓库的URL和他的更改所在的远程分支名称的Git用户，你可以将该仓库添加为远程仓库并进行本地合并

例如，如果Jessica通过电子邮件告诉你，她在她的仓库的`ruby-client`分支中有一个很棒的新特性，你可以通过添加远程仓库并本地检出该分支来测试它：
```
$ git remote add jessica https://github.com/jessica/myproject.git
$ git fetch jessica
$ git checkout -b rubyclient jessica/ruby-client
```
如果她稍后通过电子邮件再次发送另一个分支，其中包含另一个很棒的特性，你可以直接 `fetch` 并 `checkout` ，因为你将她的仓库设置为了远程仓库之一

这在你与一个人持续合作时最有用，如果有人偶尔只有单个补丁要贡献，那么通过电子邮件接受它可能比要求每个人都运行自己的服务器，并且不得不不断地添加和删除远程仓库来获取几个补丁要节省时间，你也可能不想拥有数百个远程仓库，每个仓库都是为只贡献一两个补丁的人准备的

然而，脚本和托管服务(hosted services)可能会使这变得更容易——这在很大程度上取决于你的开发方式以及你的贡献者的开发方式

这种方法的另一个优点是你可以获取提交的历史记录，在遇到合并问题时，你就可以知道他们的工作是基于你历史的哪个部分；一个正确的三方合并是可以默认执行的，而不是必须提供 `-3` 并希望补丁是从你可以访问的公共提交生成的

如果你不是持续与某人合作，但仍然想以这种方式从他们那里拉取，你可以将远程仓库的URL提供给 `git pull` 命令，这会执行一次性的拉取，并且不会将URL保存为远程引用(remote reference)：
```
$ git pull https://github.com/onetimeguy/project
From https://github.com/onttimeguy/project
 * branch        HEAD        -> FETCH_HEAD
Merge made by the 'recursive' strategy
```
### 5.3.4 Determinging What Is Introduced
现在你有一个包含了贡献工作的主题分支，此时，你可以决定你想用它做什么，本节介绍了几个命令，用于方便你看到如果你将这个合并到你的主分支中，你将引入什么

通常，回顾一下这个分支中所有不在你 `master` 分支中的提交是有帮助的，你可以通过在分支名称前添加 `--not` 选项来排除主分支中的提交，这和我们之前使用的 `master..contrib` 格式做的事情是一样的

例如，如果你的贡献者给你发送了两个补丁，你创建了一个叫做 `contrib` 的分支，并在其上应用了这些补丁，你可以运行这个命令：
```
$ git log contrib --not master
commit 5b6235bd297351589efc4d73316f0a68d484f118
Author: Scott Chacon <schacon@gmail.com>
Date: Fri Oct 24 09:53:59 2008 -0700

    See if this helps the gem

commit 7482e0d16d04bea79d0dba8988cc78df655f16a0
Author: Scott Chacon <schacon@gmail.com>
Date: Mon Oct 22 19:38:36 2008 -0700

    Update gemspec to hopefully work better
```
要查看每个提交引入了哪些更改，请记住你可以向 `git log` 传递 `-p` 选项，它将附加每个提交引入的差异(the diff introduced to each commit)

如果你想查看如果将这个主题分支与另一个分支合并会发生什么变化的完整差异，你可能需要使用一个小技巧来获得正确的结果，你可能会想到运行这个命令：
```
$ git diff master
```
这个命令会给你一个差异，但它可能会误导你，如果你的主分支自你从它创建主题分支以来已经向前移动，那么你会得到看似奇怪的结果，这是因为 Git 直接比较你所在的主题分支的最后一个提交的快照和主分支上最后一个提交的快照

例如，如果你在主分支上的文件中添加了一行，直接比较快照会看起来像是主题分支将要删除那行(主题分支基于的主分支已经不是最新的了)，如果主分支是你的专题分支的直接祖先，则不会有问题；但如果两个历史已经分叉，差异会看起来像是你在添加专题分支中的所有新东西，并移除主分支中所有独特的东西

你真正想要看到的是添加到主题分支的更改——即如果你将这个分支与主分支合并，你将引入的工作，你可以通过让 Git 比较你的主题分支上的最后一个提交与它与主分支拥有的第一个共同祖先(common ancestor)来做到这一点

技术上，你可以通过明确找出共同祖先，然后运行你的 `diff` 命令来实现这一点：
```
$ git merge-base master topic-branch
36c7dba2c95e6bbb78dfa822519ecfec6e1ca649
$ git diff 36c7db
```
或者更具体地：
```
$ git diff $(git merge-base contrib master)
```
然而，这些方法都不是特别方便，所以 Git 提供了另一种简写来做同样的事情：三点点语法(triple-dot syntax)

在使用 `git diff` 命令的上下文中，你可以在另一个分支名称后加上三个点来执行一个差异比较，这个比较是在你当前所在的分支(`contrib`)的最后一个提交和它与另一个分支的共同祖先之间进行的：
```
$ git diff master...contrib
```
该命令会展示当前的主题分支和它与 `master` 分支的共同祖先合并时，会引入的差异，这是一个很有用的语法
### 5.3.5 Integrating Contributed Work
当您的专题分支中的所有工作都准备好要集成到一个更主流的分支时，应该如何去做？此外，您想要使用什么样的整体工作流程(overall workflow)来维护您的项目？您有多种选择，我们将介绍其中的一些

**Merging Workflows**
一种基本的工作流程是直接将所有工作合并到您的主分支中，在这种情况下，您有一个包含基本稳定代码(basically stable code)的主分支，当您在专题分支中有认为已经完成的工作，或者有其他人贡献的工作并且您已经验证过，您将其合并到主分支中，删除刚刚合并的专题分支，然后重复这个过程

例如，如果我们有一个仓库，有两个名为 `ruby_client` 和 `php_client` 的分支，我们先合并 `ruby_client`，然后是 `php_client`，您的历史记录最终会看起来像：
![[ProGit-Fig72.png]]
这可能是最简单的工作流程，但如果你要处理的是更大或更稳定的项目，并且你想要非常小心地引入什么，它可能会有问题

如果你有一个更重要的项目，你可能想使用一个两阶段的合并周期(two-phase merge cycle)，在这种情况下，你有两个长期运行的分支，`master` 分支和 `develop` 分支，你决定只有当一个非常稳定的版本要发布出来时，`master` 分支才会更新，所有新代码都集成到 `develop` 分支中

你定期将这两个分支推送到公共仓库，每次你有一个新专题分支要合并时，你将其合并到 `develop` 分支中；然后，当你标记一个版本(tag a release)时，你将 `master` 分支快进(fast-forward)到现在已经稳定的 `develop` 分支所在的位置：
![[ProGit-Fig74.png]]
这样，当人们克隆你的项目仓库时，他们可以选择检出主分支来构建最新的稳定版本，并轻松地保持最新状态，或者他们可以检出开发分支，这个分支包含更前沿的内容(cutting-edge content)

你还可以扩展这个概念，通过设置一个集成分支 `integrate`，其中所有的工作都合并在一起，然后，当集成分支上的代码库稳定并通过测试时，你将它合并到开发分支中；当开发分支经过一段时间的考验证明是稳定的，你将主分支快进到那个点

**Large-Merging Workflows**
Git 项目有四个长期运行的分支：主分支(`master`)、下一个分支(`next`)、已见分支(`seen`，以前称为 `pu` — 拟议更新 proposed updates)，用于新工作，以及维护分支(`maint`)用于维护回溯(maintenance backports)

当贡献者引入新工作时，它们被收集到维护者仓库中的专题分支(topic branches in the maintainer's reposiroty)，这些专题分支会被评估以确定它们是否安全、准备就绪，或者是否需要更多的工作，如果它们是安全的，它们就合并到 `next` ，然后该分支被推送，使得集成在一起的主题分支被每个人看到
![[ProGit-Fig77.png]]

如果这些专题还需要工作，它们会被合并到 `seen` 分支中，当确定它们完全稳定时，这些专题会被重新合并(re-merge)到 `master` 分支中

然后，`next` 分支和 `seen` 分支会从 `master` 分支重新构建，这意味着 `master` 分支几乎总是在向前移动，`next` 分支偶尔会进行变基，而 `seen` 分支则更频繁地进行变基：

![[ProGit-Fig78.png]]
当一个专题分支最终被合并到主分支后，它就从仓库中移除了

Git 项目还有一个维护分支(`maint`)，它是从上一个发布分支出来的(forked from the last release)，以便在需要维护版本时提供回溯补丁(backported patches in case a maintenance release is required)

因此，当你克隆 Git 仓库时，你有四个分支可以检出，以评估项目在不同开发阶段的情况，这取决于你想要多么前沿或你想要如何贡献；而Git项目的维护者也有一个结构化的工作流程来帮助他们审查新的贡献

**Rebasing and Cherry-Picking Workflows**
一些维护者更喜欢在他们的主分支之上变基(`rebase`)或挑选(`cherry-pick`)贡献的工作，而不是合并它，以保持一个大体上是线性的历史记录

当你在专题分支中有工作，并且确定你想要集成它时，你切换到那个分支并运行 `rebase` 命令，以在当前的主分支(或开发分支，等等)之上重建变更，如果一切顺利，你可以快进你的主分支，你最终会得到一个线性的项目历史

将引入的工作从一个分支移动到另一个分支的另一种方式是挑选它，在 Git 中，`cherry-pick` 类似于单个提交的变基(a rebase for a single commit)，它取出在提交中引入的补丁，并尝试重新应用到你当前所在的分支上
这在你在一个专题分支上有许多提交并且你只想集成其中之一时很有用，或者如果你在一个专题分支上只有一个提交并且你更倾向于挑选它而不是运行`rebase`

例如，假设你有一个看起来像这样的项目：
![[ProGit-Fig79.png]]

如果你想要拉取提交 `e43a6` 到你的 `master` 分支，运行：
```
$ git cherry-pick e43a6
Finished one cherry-pick
[master]: created a0a41a9: "More friendly message when locking the index fails."
3 files changed, 17 insertions(+), 3 deletions(-)
```
这引入了与`e43a6`相同的更改，但您会得到一个新的提交SHA-1值，因为应用的日期不同，现在您的历史记录看起来像这样：
![[ProGit-Fig80.png]]
现在你可以移除你的主题分支，并且丢弃你不想拉取的提交

**Rerere**
如果你正在进行大量的合并和变基操作，或者你正在维护一个长期存在的主题分支，Git有一个名为“rerere”的功能可以帮助你

Rerere代表“重用已记录的解决方案 reuse recorded resolution”——它是一种简化手动冲突解决的方法(shortcutting manual conflict resolution)，当启用rerere时，Git会保留一组成功合并的前后映像(pre- and post-images from successful merges)，如果它注意到有一个看起来和你以前已经解决过的冲突完全一样的冲突，它就会直接使用上次的解决方案，而不会打扰你

这个功能包括两个部分：一个配置设置(configuration setting)和一个命令(a command)，配置设置是`rerere.enabled`，它可以放在你的全局配置中：
```
$ git config --global rerere.enabled true
```
现在，每当你执行合并以解决冲突时(do a merge that resolves conflicts)，解决方案将被记录在缓存中，以防你将来需要它

如果需要，你可以使用`git rerere`命令与`rerere`缓存交互，当它被单独调用时，Git会检查其解决方案数据库，并尝试找到与任何当前合并冲突匹配的解决方案并解决它们(如果将`rerere.enabled`设置为`true`，则会自动执行此操作)，还有一些子命令可以查看将记录的内容，从缓存中擦除特定的解决方案，以及清除整个缓存
### 5.3.6 Tagging Your Releases
当你决定发布一个版本(cut a release)时，你可能会想要分配一个标签(assign a tag)，这样你就可以在将来的任何时候重新创建该版本(re-create that release)，如果你决定作为维护者签署标签(sign the tag as the maintainer)，那么标签可能看起来像这样：
```
$ git tag -s v1.5 -m 'my signed 1.5 tag'
You need a passphrase to unlock the secret key for
user: "Scott Chacon <schacon@gmail.com>"
1024-bit DSA key, ID F721C45A, created 2009-02-09
```
如果你确实签署了你的标签，你可能会遇到分发用于签署标签的公共PGP密钥的问题(the problem of distributing the public PGP key used to sign your tags)

Git项目的维护者通过将他们的公钥作为仓库中的一个blob包含在仓库中，然后添加一个直接指向该内容的标签(tag)来解决这个问题，要做到这一点，你可以通过运行`gpg --list-keys`来确定你想要使用哪个密钥：
```
$ gpg --list-keys
/Users/schacon/.gunpg/pubring.gpg
---------------------------------
pub 1024D/F721C45A 2009-02-09 [expires: 2010-02-09]
uid Scott Chacon <schacon@gmail.com>
sub 2048g/45D02282 2009-02-09 [expires: 2010-02-09]
```

然后，您可以直接将密钥导入到Git数据库中(import the key into the Git database)，方法是将其导出并通过管道传给 `git hash-object`(exporting it and piping that through `git hash-object`)，这将在Git中写入一个包含这些内容的新blob，并返回blob的SHA-1：
```
$ gpg -a --export F721C45A | git hash-object -w --stdin
```

既然您已经在Git中拥有了密钥的内容，您可以通过指定`hash-object`命令给您的新SHA-1值来创建一个直接指向它的标签：
```
$ git tag -a maintainer-pgp-pub 659ef797d181633c87ec71ac3f9ba29fe5775b92
```

如果你再运行 `git push --tags`，`maintainer-pgp-pub` 标签将与所有人共享，如果有人想要验证一个标签，他们可以直接从数据库中直接提取 blob 并将其导入到 GPG 中，以导入你的 PGP 密钥：
```
$ git show maintainer-pgp-pub | gpg --import
```

他们可以使用该密钥验证您签名的所有标签，此外，如果您在标签消息中包含说明(include instructions in the tag message)，运行 `git show <tag>` 将允许您向最终用户提供更具体的标签验证说明
### 5.3.7 Generating a Build Number
因为Git没有像“v123”这样的单调递增数字来标识每个提交，所以如果您想要为提交指定一个易于阅读的名称，您可以在该提交上运行`git describe`，Git会生成一个字符串，由以下几部分组成：首先是比该提交更早的最新标签的名称(the name of the most recent tag earlier than that commit)，然后是自该标签以来的提交数量(the number of commits since that tag)，最后是被描述的提交的部分SHA-1值(以字母“g”为前缀，表示Git)：
```
$ git describe master
v1.6.2-rc1-20-g8c5b85c
```
这样，您可以导出一个快照或构建并将其命名为人们容易理解的名称

实际上，如果您从Git仓库克隆的源代码构建Git，`git --version`就会给您一个看起来像这样的东西

如果您正在描述一个您直接标记的提交，它会直接给您标记名称

默认情况下，`git describe`命令需要注释标签(annotated tags)(使用`-a`或`-s`标志创建的标签)；如果您还想要利用轻量级(非注释)标签，可以在命令中添加`--tags`选项

您还可以使用`git describe`给出的字符串作为`git checkout`或`git show`命令的目标，尽管它依赖于末尾的缩写SHA-1值，所以可能不会永远有效
例如，Linux内核最近从8个字符跳到10个字符以确保SHA-1对象的唯一性，因此旧的`git describe`输出名称已失效
### 5.3.8 Preparing a Release
现在你想发布一个构建，你想要做的其中一件事是为你的代码创建一个最新快照的存档(an archive of the latest snapshot of your code)，以供那些不使用Git的人使用，执行此操作的命令是`git archive`：

```
$ git archive master --prefix='project/' | gzip > `git describe master`.tar.gz 
$ ls *.tar.gz
v1.6.2-rc1-20-g8c5b85c.tar.gz
```

如果有人打开那个tarball，他们会在 `project` 目录下获得你项目的最新快照，你也可以以类似的方式创建一个zip存档，但需要将`--format=zip`选项传递给`git archive`：
```
$ git archive master --prefix='project/' --format=zip > `git describe master`.zip
```

现在你有一个漂亮的tarball和一个zip存档，你可以将你的项目发布上传到你的网站或通过电子邮件发送给人们
### 5.3.9 The Shorlog
是时候给想知道你的项目中发生了什么的人们的邮件列表发送邮件了，一种快速获取自上次发布或发送邮件以来你的项目中添加了哪些内容的变更日志(changelog)的方法是使用`git shortlog`命令，它会总结你给出范围内的所有提交(all commits in the range you give it)
例如，以下命令会给你一个自上次发布v1.0.1以来的所有提交的摘要：
```
$ git shortlog --no-merges master --not v1.0.1
Chris Wanstrath (6):
        Add support for annotated tags to Grit::Tag
        Add packed-refs annotated tag support.
        Add Grit::Commit#to_patch
        Update version and History.txt
        Remove stray 'puts'
        Make ls_tree ignore nils

Tom Preston-Werner (4):
        fix dates in history
        dynamic version method
        Version bump to 1.0.2
        Regenerated gemspec for version 1.0.2
```
您得到了一个自`v1.0.1`版本以来所有提交的清晰摘要，按作者分组，您可以将其通过电子邮件发送到您的列表
# 6 GitHub
## 6.1 Account Setup and Configuration
### 6.1.1 SSH Access
### 6.1.2 Your Avatar
### 6.1.3 Your Email Addresses
### 6.1.4 Two Factor Authentication
## 6.2 Contributing to a Project
### 6.2.1 Forking Projects
### 6.2.2 The Github Flow
1. Fork the project
2. Create a topic branch from `master`
3. Make some commits to improve the project
4. Push this branch to your GitHub project
5. Open a Pull Request on GitHub
6. Discuss, and optionally continue committing
7. The project owner merges or closes the Pull Request
8. Sync the updated `master` back to your fork

**Creating a Pull Request**

**Iterating on a Pull Request**
### 6.2.3 Advanced Pull Requests
**Pull Requests as Patches**

**Keeping up with Upstream**
```
$ git remote add upstream https://github.com/schacon/blink

$ git fetch upstream
remote: Counting objects: 3, done.
remote: Compressing objects: 100% (3/3), done. 
Unpacking objects: 100% (3/3), done.
remote: Total 3 (delta 0), reused 0 (delta 0) 
From https://github.com/schacon/blink
 * [new branch] master -> upstream/master

$ git merge upstream/master
Auto-merging blink.ino
CONFLICT (content): Merge conflict in blink.ino
Automatic merge failed; fix conflicts and then commit the result.

$ vim blink.ino
$ git add blink.ino
$ git commit
[slow-blink 3c8d735] Merge remote-tracking branch 'upstream/master' into slower-blink

$ git push origin slow-blink
Counting objects: 6, done.
Delta compression using up to 8 threads. 
Compressing objects: 100% (6/6), done.
Writing objects: 100% (6/6), 682 bytes | 0 bytes/s, done. 
Total 6 (delta 2), reused 0 (delta 0) 
To https://github.com/tonychacon/blink
  ef4725c..3c8d735 slower-blink -> slow-blink
```

1. Add the original repository as a remote named `upstream`
2. Fetch the newest work from that remote
3. Merge the main branch of that repository into your topic branch
4. Fix the confilct that occurred
5. Push back up to that same topic branch

**References**
GitHub有许多种方式可以引用你可以在GitHub上写入的几乎任何东西

让我们从如何交叉引用另一个拉取请求(pull request)或问题(issue)开始，所有的拉取请求和问题都被分配了编号，它们在项目中是唯一的(unique within the project)，例如，你不能同时拥有Pull Request#3和Issue#3，如果你想在任何其他请求或问题中引用任何拉取请求或问题，你只需在任何评论或描述中简单地写上`#<num>`

如果问题或拉取请求位于其他地方，你也可以更具体地引用；如果你要引用的是你所在的仓库的分支(fork)中的一个问题或拉取请求，请写上`username#<num>`，或者使用`username/repo#<num>`来引用另一个仓库中的某个问题或拉取请求

除了issue编号，您还可以通过SHA-1引用特定的提交，您必须指定一个完整的40个字符的SHA-1，GitHub在会将评论中的SHA-1值直接链接到该提交，同样，您可以像处理issue一样引用分叉或其他存储库中的提交
### 6.2.4 GitHub Flavored Markdown
GitHub支持在评论(comment)或描述(description)以及几乎所有文本框(text box)中自动渲染Markdown格式的文本

**Task Lists**
You can create a task list like this:
```
- [X] Wriet the code
- [ ] Write all the tests
- [ ] Document the code
```
The really cool part is that you can simply click the checkboxes to update the comment——you don't have to edit the Markdown directly to check tasks

**Code Snippets**
您还可以在注释中添加代码片段，这在您想要展示一些在实际将其作为提交实施到您的分支之前可以尝试做的事情时特别有用，这通常也用于添加示例代码，说明什么不起作用或者这个拉取请求可以实现什么

要添加代码片段，您需要用反引号将其“围起来”：
```java
for (int i = 0; i < 5; i++)
{
    System.out.println("i is :" + i);
}
```
If you add a language name like we did there with 'java', GitHub will also try to syntax highlight the snippet.

**Quoting**
如果你在回复一个长篇评论中的一小部分，你可以通过在行前加上`>`字符来选择性地引用其他评论，实际上，这种做法非常常见且非常有用，以至于有一个键盘快捷键可以实现这个功能，如果你在评论中高亮你想要直接回复的文本，然后按下`r`键，它就会在评论框中为你引用那段文本

引用看起来是这样的：
```
> Whether 'tis Nobler in the mind to suffer 
> The Slings and Arrows of outrageous Fortune,

How big are these slings and in particular, these arrows?
```

**Emoji**
There is even an emoji helper in GitHub. If you are typing a comment and you start with a `:` character, an autocompleter will help you find what you’re looking for

Emojis take the form of `:<name>:` anywhere in the comment. For instance, you could write something like this:
```
I :eyes: that :bug: and I :cold_sweat:.

:trophy: for :microscope: it.

:+1: and :sparkless: on this :ship:, it's :fire::poop:!

:clap::tada::panda_face:
```

**Images**

### 6.2.5 Keep your GitHub public repository up-to-date
一旦你分叉了一个GitHub仓库，你的仓库(你的“分叉”)就独立于原始仓库存在
特别是，当原始仓库有新的提交时，GitHub会通过类似以下的消息通知你：
```
This branch is 5 commits behind progit:master.
```

GitHub仓库不会被GitHub自动更新，你需要手动更新你的仓库：
```
$ git check out master
$ git pull https://github.com/progit/progit2.git
$ git push origin master
```
1. If you were on another branch, return to `master`
2. Fetch changes from https://github.com/progit/progit2.git and merge them into master
3. Push your `master` branch to `origin`

This works, but it is a little tedious having to spell out the fetch URL every time. You can automate this work with a bit of configuration:
```
$ git remote add progit https://github.com/progit/progit2.git
$ git fetch progit
$ git branch --set-upstream-to=progit/master master
$ git config --local remote.pushDefault origin
```
1. Add the source repository and give it a name
2. Get a reference on progit's branches, in particular `master`
3. Set your `master` to fetch from the `progit` remote
4. Define the default push repository to `origin` 

Once this is done, the workflow becomes much simpler:
```
$ git checkout master
$ git pull
$ git push
```
1. If you were on another branch, return to `master`
2. Fetch changes from `progit` and merge changes into `master`
3. Push your `master` branch to `origin`

这种方法可能很有用，但它并非没有缺点，Git会默默地为您完成这项工作，但如果您在本地`master`上提交，从`progit`拉取，然后推送到`origin` ，所有这些操作在这个设置中都是有效的，所以您必须小心，永远不要直接提交到`master`，因为那个分支实际上属于上游仓库
(否则在拉取的时候，由于Git会自动将远端的更新合并入本地的 `master` 分支内，合并冲突就有可能发生，因此我们应该在除 `master` 分支以外的分支工作)
## 6.3 Maintaining a Project
### 6.3.1 Creating a New Repository
GitHub上所有的项目都可以通过 HTTPS 协议访问，URL形式为 `https://github.com/<user>/<project_name>` ，或者通过 SSH 形式为 `git@github.com:<user>/<project_name>`

通常，分享基于HTTPS的URL对于公共项目来说更可取，因为用户无需拥有GitHub帐户即可访问它以进行克隆，如果您给他们SSH URL，用户将需要拥有一个帐户并上传SSH密钥才能访问您的项目，HTTPS URL也是他们将粘贴到浏览器中以在那里查看项目的完全相同的URL
### 6.3.2 Adding Collaborators
如果你正在与其他你想要给予提交访问权限的人一起工作，你需要将他们添加为“协作者”，如果Ben、Jeff和Louise都在GitHub上注册了账户，你想要给他们对你的仓库的推送访问权限(push access)，你可以将他们添加到你的项目中(add them to your project)，这意味着他们对项目和Git仓库都有读写访问权限
### 6.3.3 Managing Pull Requests

**Email Notifications**
`git pull <url> <branch> 是一种简单的方法，可以在不需要添加远程仓库的情况下合并远程分支，如果您愿意，您可以创建并切换到一个主题分支，然后运行此命令以合并拉取请求的更改

其他有趣的URL是`.diff`和`.patch` URL，正如您可能猜到的，它们提供了拉取请求的统一差异和补丁版本(unified diff and patch versions of the Pull Request)，从技术上讲，您可以使用类似以下内容合并拉取请求的工作：
```
$ curl https://github.com/tonychacon/fade/pull/1.patch | git am
```

**Collaborating on the Pull Request**
每次其他人在拉取请求上发表评论时，您将继续收到电子邮件通知，以便您知道有活动发生，这些邮件将各自有一个链接到拉取请求的活动发生的地方，您也可以直接回复电子邮件以在拉取请求线程上发表评论(Responses to emails are included in the thread)

一旦Pull Request的代码已经是您喜欢并希望合并的了，您可以选择通过我们之前看到的`git pull <url> <branch>`语法将代码拉取下来并本地合并，或者通过将fork仓库添加为远程仓库并获取其分支并合并

如果合并是简单的(trivial)，您也可以直接在GitHub网站上点击“合并”按钮。这将执行一个“非快进”合并，即使可以进行快进合并，也会创建一个合并提交，这意味着无论何时，每次您点击合并按钮，都会创建一个合并提交，如果您点击提示链接，GitHub会为您提供所有这些信息。
![[ProGit-Fig116.png]]
如果您决定不想合并它，您也可以直接关闭拉取请求，发起请求的人将会收到通知

**Pull Request Refs**
如果你正在处理大量的拉取请求，并且不想添加许多远程仓库或每次都进行一次性拉取，GitHub允许你使用一个巧妙的技巧

实际上，GitHub将仓库的拉取请求分支(Pull Request branches for a repository)表示为服务器上的伪分支(pesudo-branches on the server)，默认情况下，当你克隆时不会获得它们，但它们以一种隐蔽的方式存在，你可以很容易地访问它们

为了演示这一点，我们将使用一个低级(low-level)命令(通常被称为“管道 plumb”命令)`ls-remote`，这个命令通常不在日常的Git操作中使用，但它对我们展示服务器上存在的引用非常有用(show us what references are present on the server)

如果我们对我们之前使用的“blink”仓库运行这个命令，我们将获得仓库中所有分支、标签和其他引用的列表(a list of all the branches and tags and other references in the repository)：
```
$ git ls-remote https://github.com/schacon/blink
10d539600d86723087810ec636870a504f4fee4d HEAD 10d539600d86723087810ec636870a504f4fee4d refs/heads/master 6a83107c62950be9453aac297bb0193fd743cd6e refs/pull/1/head afe83c2d1a70674c9505cc1d8b7d380d5e076ed3 refs/pull/1/merge 3c8d735ee16296c242be7a9742ebfbc2665adec1 refs/pull/2/head 15c9f4f80973a2758462ab2066b6ad9fe8dcf03d refs/pull/2/merge a5a7751a33b7e86c5e9bb07b26001bb17d775d1a refs/pull/4/head 31a45fc257e8433c8d8804e3e848cf61c9d3166c refs/pull/4/merge
```
当然，如果你在仓库中运行`git ls-remote origin`或 `git ls-remote` + 任何你想要检查的远程仓库，它同样会显示类似于这样的内容

如果仓库在GitHub上，并且你已经打开了任何拉取请求，你将得到这些以`refs/pull/`为前缀的引用，这些基本上都是分支，但由于它们不在`refs/heads/`下，所以当你从服务器克隆或获取时，通常不会得到它们——获取过程通常会忽略它们(the process of fetching ignores them normally)

每个拉取请求有两个引用——以`/head`结尾的那个引用指向拉取请求分支中最后一个提交(the last commit in the Pull Request branch)，所以，如果有人在我们的仓库中打开了一个拉取请求，他们的分支名为`bug-fix`并且指向提交`a5a775`，虽然我们的仓库中将不会有`bug-fix`分支，但我们会有`pull/<pr#>/head`指向`a5a775`，这意味着我们可以很容易地一次性拉取每个拉取请求分支(pull down every Pull Request branch)，而不必添加一堆远程仓库

现在，你可以通过这样直接获取引用：
```
$ git fetch origin refs/pull/958/head
From https://github.com/libgit2/libgit2
 * branch        refs/pull/958/head -> FETCH_HEAD
```
该命令Git连接到`origin`远程，并下载名为`refs/pull/958/head`的引用
Git下载了您需要构建该引用的所有内容，并将指向您想要的提交的指针放在了`.git/FETCH_HEAD`下，您可以在要测试的分支中使用`git merge FETCH_HEAD`进行后续操作，但该合并提交消息会看起来有点奇怪，此外，如果您正在审查很多拉取请求，这会变得繁琐

还有一种方法可以获取所有拉取请求，并在每次连接到远程时保持它们的最新状态；用编辑器打开`.git/config`，并查找`origin`远程 ，它应该看起来有点像这样：
```
[remote "origin"]
    url = https://github.com/libgit2/libgit2
    fetch = +refs/heads/*:refs/remotes/origin/*
```
以`fetch =`开头的那一行是一个“refspec”，它是一种将远程上的名称映射到本地`.git`目录中的名称的方法，这个特定的refspec告诉Git：“在远程上位于`refs/heads`下的内容应该放在我的本地仓库下的`refs/remotes/origin`下
，”您可以修改此部分以添加另一个refspec：
```
[remote "origin"]
    url = https://github.com/libgit2/libgit2.git
    fetch = +refs/heads/*:refs/remotes/origin/*
    fetch = +refs/pull/*/head:refs/remotes/origin/pr/*
```
最后一行告诉Git：“所有类似于 `refs/pull/123/head` 的分支应该本地存储于 `refs/remotes/origin/pr/123`”

我们保存修改，然后运行 `git fetch`：
```
$ git fetch
# ...
  * [new ref]        refs/pull/1/head -> origin/pr/1
  * [new ref]        refs/pull/2/head -> origin/pr/2
  * [new ref]        refs/pull/4/head -> origin/pr/4
# ...
```
现在，所有的远程拉取请求都以本地引用的形式表示，这些引用的行为很像跟踪分支；它们是只读的，并且在执行获取操作时会更新，这使得在本地尝试拉取请求中的代码变得非常容易：
```
$ git checkout pr/2
Checking out files: 100% (3769/3769), done.
Branch pr/2 set up to track remote branch pr/2 from origin. Switched to a new branch 'pr/2'
```
我们会注意到refspec的远端部分的末尾的 `head`，在GitHub方面，还有一个`refs/pull/#/merge`引用，它代表了如果你在网站上点击“合并”按钮将会产生的提交，这可以让你在点击按钮之前测试该合并提交

**Pull Requests on Pull Requests**
你不仅可以开启针对主分支的拉取请求(open Pull Requests that target the main or `master` branch)，实际上，你还可以开启针对网络中任何分支的拉取请求，你甚至可以针对另一个拉取请求开启拉取请求

如果你看到一个拉取请求正朝着正确的方向发展，并且你有一个依赖于它的变更想法，或者你不确定这是否是一个好主意，或者你只是没有推送到目标分支的权限，你可以直接向它打开一个拉取请求(open a Pull Request directly to it)

当你要打开一个拉取请求时，页面顶部有一个框，指定你请求拉取到的分支和你请求被拉取的分支，如果你点击该框右侧的"Edit"按钮，你可以更改目标的的fork和branch
![[ProGit-Fig117.png]]
在这里，您可以相对容易地指定将您的新分支合并到另一个拉取请求或项目的另一个分支中

### 6.3.4 Mentions and Notifications
GitHub 还内置了一个相当不错的通知系统，当你有问题或需要特定个人或团队的反馈时，这个系统可能会非常有用

在任何评论中，你可以开始输入一个`@`字符，它将开始自动补全项目中的协作者或贡献者的姓名和用户名

一旦你发表了一个带有用户提及的评论，该用户将收到通知，这意味着这可以是一个非常有效的方式将人们引入对话，而不是让他们投票，在GitHub上的拉取请求中，人们经常会将他们团队或公司中的其他人拉进来审查问题或拉取请求

如果某人在拉取请求(Pull Request)或问题(Issue)中被提及，他们将被“订阅”到这个请求或问题上，并且会在任何活动发生时继续收到通知，如果你打开了这个请求或问题，或者你在关注(watching)这个仓库，或者你对某个问题发表了评论，你也会被订阅，如果你不再希望收到通知，你可以点击页面上的“取消订阅”按钮，以停止接收更新

**The Notification Page**
There is also a fair amount of metadata embedded in the headers of the emails that GitHub sends you, which can be really helpful for setting up custom filters and rules
```
To: tonychacon/fade <fade@noreply.github.com>
Message-ID: <tonychacon/fade/pull/1@github.com>
Subject: [fade] Wait longer to see the dimming effect better (#1) 
X-GitHub-Recipient: tonychacon 
List-ID: tonychacon/fade <fade.tonychacon.github.com> 
List-Archive: https://github.com/tonychacon/fade 
List-Post: <mailto:reply+i-4XXX@reply.github.com> 
List-Unsubscribe: <mailto:unsub+i-XXX@reply.github.com>,... 
X-GitHub-Recipient-Address: tchacon@example.com
```
`Message-ID`以`<user>/<project>/<type>/<id>`的格式提供信息，帮助我们快速定向到特定的项目或甚至拉取请求，例如，如果这是一个Issue，`<type>`字段将是“issues”，而不是“pull”

`List-Post`和`List-Unsubscribe`字段意味着，如果你有一个可用的邮件客户端，你可以轻松地利用该字段信息发布到列表或从线程中“取消订阅”，这与在通知的网络版本上点击“静音”按钮或在问题或拉取请求页面本身上点击“取消订阅”相同

同样值得注意的是，如果你同时启用了电子邮件和网络通知，并且你阅读了电子邮件版本的提醒，如果你的邮件客户端允许显示图片，网络版本也会被标记为已读
### 6.3.5 Special Files
GitHub会识别许多仓库中的特殊文件

**README**
`README`文件可以是许多格式，例如，它可以是`README`、`README.md`、`README.asciidoc`等，如果GitHub在您的源代码中看到一个`README`文件，它将在项目的首页上呈现它(render it on the landing page of the project)

许多团队使用这个文件来保存所有与项目相关的信息(all the relevant project information)，供可能对仓库或项目不熟悉的人使用，这通常包括以下内容：
• 项目的目的
• 如何配置和安装它
• 如何使用它或启动它的一个示例
• 项目提供的许可证(license)
• 如何为该项目做出贡献
由于GitHub将自动渲染此文件，因此您可以在其中嵌入图像或链接，以便于读者理解

**CONTRIBUTING**
如果仓库内有名为 `CONTRIBUTING` 的文件，后缀任意，GitHub会在有人开启拉取请求时展示如下窗口：
![[ProGit-FIg122.png]]
这里的想法是，您可以在发送到您项目的拉取请求中指定您想要或不想要的特定事项，这样，人们可能会在打开拉取请求之前真正阅读指南(read the guidelines)
### 6.3.6 Project Administration
通常情况下，对于一个单独的项目，你可以做的管理事务并不多，但有一些项可能会引起你的兴趣

**Changing the Default Branch**
如果您希望人们默认打开拉取请求或默认查看的分支不是“master”，您可以在您的仓库设置页面的“Options”标签下进行更改

在下拉菜单中更改默认分支之后，所有主要操作都将使用该默认分支，包括当有人克隆存储库时，默认检出的分支

**Transferring a Project**
如果您想在GitHub上将一个项目转移到另一个用户或组织，您可以在您的仓库设置页面的“Options”标签底部找到“Transfer ownership”选项，通过这个选项您可以实现项目转移

这在以下情况下很有帮助：如果你要放弃一个项目，而有人想要接手；或者如果你的项目正在扩大，你想要将其转移到一个组织中

这不仅会将仓库及其所有关注者和星标转移到另一个地方，还会从你的URL设置一个重定向到新位置，它还会重定向来自Git的克隆和获取操作，而不仅仅是Web请求
## 6.4 Managing an organization
除了单个用户帐户之外，GitHub还有所谓的“组织”，与个人帐户一样，组织帐户具有一个命名空间，其中包含他们所有的项目，但许多其他方面是不同的
这些帐户代表一群拥有项目共享所有权的人，并且有许多工具可以管理这些人的子组，通常，这些帐户用于开源组织(例如“perl”或“rails”)或公司(例如“google”或“twitter”)
### 6.4.1 Orgranization Basics
### 6.4.2 Teams
Additionally, team `@mentions` (such as `@acmecorp/frontend`) work much the same as they do with individual users, except that all members of the team are then subscribed to the thread. This is useful if you want the attention from someone on a team, but you don’t know exactly who to ask.
### 6.4.3 Audit Log
## 6.5 Scripting GitHub
### 6.5.1 Services and Hooks
GitHub仓库管理中的Hooks和Services部分是让GitHub与外部系统交互的最简单方法

**Services**
In this case, if we hit the “Add service” button, the email address we specified will get an email every time someone pushes to the repository. Services can listen for lots of different types of events, but most only listen for push events and then do something with that data.

If there is a system you are using that you would like to integrate with GitHub, you should check here to see if there is an existing service integration available. For example, if you’re using Jenkins to run tests on your codebase, you can enable the Jenkins builtin service integration to kick off a test run every time someone pushes to your repository.

**Hooks**
If you need something more specific or you want to integrate with a service or site that is not included in this list, you can instead use the more generic hooks system. GitHub repository hooks are pretty simple. You specify a URL and GitHub will post an HTTP payload to that URL on any event you want.

Generally the way this works is you can setup a small web service to listen for a GitHub hook payload and then do something with the data when it is received.

Webhook的配置非常简单，在大多数情况下，您只需输入一个URL和一个密钥，然后点击“添加Webhook”，有一些选项可以让您选择希望GitHub在哪些事件下发送payload，默认情况下，只有在有人将新代码推送到您的存储库的任何分支时，才会收到推送事件的有效载荷

Let’s see a small example of a web service you may set up to handle a web hook. We’ll use the Ruby web framework Sinatra since it’s fairly concise and you should be able to easily see what we’re doing.

Let’s say we want to get an email if a specific person pushes to a specific branch of our project modifying a specific file. We could fairly easily do that with code like this:
```
require 'sinatra'
require 'json'
require 'mail'

post '/payload' do
  push = JSON.parse(request.body.read) # parse the JSON

  # gather the data we're looking for
  pusher = push["pusher"]["name"]
  branch = push["ref"]

  # get a list of all the files touched
  files = push["commits"].map do |commit|
    commit['added'] + commit['modified'] + commit['removed']
  end
  files = files.flatten.uniq

  # check for our criteria
  if pusher == 'schacon' &&
     branch == 'ref/haeds/special-branch' &&
     files.include?('special-file.txt')
 
     Mail.deliver do
       from 'tchacon@example.com'
       to 'tchacon@example.com'
       subject 'Scott Changed the File' 
       body "ALARM"
     end
  end
end
```
在这里，我们正在获取GitHub提供给我们的JSON有效载荷，并查找谁推送了它，他们推送到了哪个分支，以及所有推送的提交中触及了哪些文件，然后我们将其与我们的标准进行核对，如果匹配，就发送一封电子邮件

为了开发和测试这样的东西，您在设置钩子的同一屏幕上拥有一个不错的开发人员控制台，您可以查看GitHub尝试为该webhook进行的最后几次交付(deliveries)，对于每个钩子，您可以深入了解它何时被交付，是否成功，以及请求和响应的正文和标题，这使得测试和调试您的钩子变得非常容易

这个的另一个重要特点是，您可以重新传递(redeliver)任何有效载荷，以轻松测试您的服务
### 6.5.2 The GitHub API
Services and hooks give you a way to receive push notifications about events that happen on your repositories, but what if you need more information about these events? What if you need to automate something like adding collaborators or labeling issues?

This is where the GitHub API comes in handy. GitHub has tons of API endpoints for doing nearly anything you can do on the website in an automated fashion. In this section we’ll learn how to authenticate and connect to the API, how to comment on an issue and how to change the status of a Pull Request through the API.

**Basic Usage**
您可以做的最基本事情是对不需要身份验证的端点进行简单的GET请求(a simple GET request on an endpoint that doesn't require authentication)，这个端点可以是用户或开源项目上的只读信息，例如，如果我们想要了解更多关于名为“schacon”的用户，我们可以运行类似这样的操作：
```
$ curl https://api.github.com/users/schacon
{
  "login": "schacon",
  "id": 70,
  "avatar_url": "https://avatars.githubusercontent.com/u/70"
# ...
  "name": "Scott Chacon",
  "company": "GitHub",
  "following": 19,
  "created_at": "2008-01-27T17:19:28Z",
  "updated_at": "2014-06-10T02:37:23Z"
}
```
有大量的类似这样的端点，可以获取有关组织、项目、问题、提交等的信息——几乎可以公开在GitHub上看到的所有内容，您甚至可以使用API来渲染任意的Markdown或查找`.gitignore`模板：
```
$ curl https://api.github.com/gitignore/templates/Java
{
  "name": "Java", 
  "source": "*.class

# Mobile Tools for Java (J2ME) 
  .mtj.tmp/

# Package Files #
*.jar 
*.war 
*.ear

# virtual machine crash logs, see https://www.java.com/en/download/help/error_hotspot.xml hs_err_pid*
"
}
```

**Commenting on an Issue**
如果您想在网站进行某个操作，例如对问题或拉取请求发表评论，或者如果您想查看或与私有内容(private content)进行交互，您需要进行身份验证

有几种身份验证方法，您可以采用仅使用用户名和密码的基本身份验证，但通常使用个人访问令牌(personal access token)是一个更好的主意，您可以从设置页面的“Application”选项卡生成此令牌

它会询问您希望这个令牌用于哪些作用域以及一个描述，请确保使用一个好的描述，这样当您的脚本或应用程序不再使用时，您可以轻松地删除令牌

GitHub只会向您展示一次令牌，所以请确保复制它，现在，您可以使用它来在脚本中进行身份验证，而不是使用用户名和密码，使用令牌可以方便您限制您想要执行的操作的作用域，并且令牌是可以撤销的

这还有一个额外的好处，就是可以增加您的速率限制，如果不进行身份验证，您将被限制在每小时60个请求，如果您进行身份验证，您可以每小时进行多达5,000个请求

让我们使用它在我们的一个问题上发表评论，假设我们想要在特定问题上留下评论，问题编号为6，为此，我们需要向`repos/<user>/<repo>/issues/<num>/comments` 发送一个HTTP POST请求，并将我们刚刚生成的令牌作为Authorization头：
```
$ curl -H "Content-Type: application/json" \ 
       -H "Authorization: token TOKEN" \
       --data '{"body":"A new comment, :+1:"}' \
       https://api.github.com/repos/schacon/blink/issues/6/comments"
{
    "id": 58322100,
    "html_url": "https://github.com/schacon/blink/issues/6#issuecomment-58322100", 
    ...
    "user": {
        "login": "tonychacon", "id": 7874698,
        "avatar_url": "https://avatars.githubusercontent.com/u/7874698?v=2", 
        "type": "User",
},
    "created_at": "2014-10-08T07:48:19Z", 
    "updated_at": "2014-10-08T07:48:19Z", 
    "body": "A new comment, :+1:"
}
```
您可以使用API来完成网站上几乎所有您可以做的事情——创建和设置里程碑，将人员分配给问题和拉取请求，创建和更改标签，访问提交数据，创建新的提交和分支，打开、关闭或合并拉取请求，创建和编辑团队，在拉取请求中对代码行进行评论，搜索网站等等

**Changing the Status of a Pull Request**
每个提交可以有一个或多个与之关联的状态，并且有一个API可以添加和查询该状态

大多数持续集成和测试服务都利用这个API来响应推送，通过测试推送的代码，然后报告该提交是否通过了所有测试，您还可以使用此功能来检查提交消息是否正确格式化，提交者是否遵循了您所有的贡献指南，提交是否有效签名等

假设您在存储库上设置了一个webhook，该webhook触发一个小型web服务，该服务检查提交消息中的`Signed-off-by`字符串：
```
require 'httparty' 
require 'sinatra' 
require 'json'

post '/payload' do
  push = JSON.parse(request.body.read) # parse the JSON  
  repo_name = push['repository']['full_name']

  # look through each commit message 
  push["commits"].each do |commit|

    # look for a Signed-off-by string 
    if /Signed-off-by/.match commit['message']
      state = 'success'
      description = 'Successfully signed off!' 
    else
      state = 'failure'
      description = 'No signoff found.' 
    end

  # post status to GitHub 
  sha = commit["id"]
  status_url = "https://api.github.com/repos/#{repo_name}/statuses/#{sha}"

  status = {
      "state" => state,
      "description" => description,
      "target_url" => "http://example.com/how-to-signoff",
      "context" => "validate/signoff"
  }
  HTTParty.post(status_url, 
  :body => status.to_json, 
  :headers => {
      'Content-Type' => 'application/json',
      'User-Agent' => 'tonychacon/signoff',
      'Authorization' => "token #{ENV['TOKEN']}" } 
   )
   end 
end
```
在这个webhook处理程序(handler)中，我们查看刚刚推送的每个提交，我们在提交消息中查找“Signed-off-by”字符串，最后我们通过HTTP POST到`/repos/<user>/<repo>/statuses/<commit_sha>` API端点，以设置状态

在这种情况下，您可以发送一个状态('success'，'failure'，'error')，一个关于发生了什么的描述，一个目标URL以方便用户可以访问以获取更多信息，以及一个“上下文”，以防单个提交有多个状态，例如，测试服务可能提供状态，像这样的验证服务也可能提供状态——它们之间通过“上下文”字段进行区分
# 7 Git Tools
## 7.1 Revision Selection
Git allows you to refer to a single commit, set of commits, or range of commits in a number of ways.
### 7.1.1 Single Revision
You can obviously refer to any single commit by its full, 40-character SHA-1 hash
**Short SHA-1**
Git is smart enough to figure out what commit you’re referring to if you provide the first few characters of the SHA-1 hash, as long as that partial hash is at least four characters long and unambiguous; that is, no other object in the object database can have a hash that begins with the same prefix. ( 多数情况下，SHA-1 hash 的前4个字符足够索引 commit )

In this case, say you’re interested in the commit whose hash begins with `1c002dd…​`. You can inspect that commit with any of the following variations of `git show` (assuming the shorter versions are unambiguous):

```console
$ git show 1c002dd4b536e7479fe34593e72e6c6c1819e53b
$ git show 1c002dd4b536e7479f
$ git show 1c002d
```
( 用 git show 审查提交信息 )

Git can figure out a short, unique abbreviation for your SHA-1 values. If you pass `--abbrev-commit` to the `git log` command, the output will use shorter values but keep them unique; it defaults to using seven characters but makes them longer if necessary to keep the SHA-1 unambiguous: ( `git log --avvrev-commit` 可以让 git 输出短的且唯一的 SHA-1值，默认7位 )
```console
$ git log --abbrev-commit --pretty=oneline
ca82a6d Change the version number
085bb3b Remove unnecessary test code
a11bef0 Initial commit
```
Generally, eight to ten characters are more than enough to be unique within a project. For example, as of February 2019, the Linux kernel (which is a fairly sizable project) has over 875,000 commits and almost seven million objects in its object database, with no two objects whose SHA-1s are identical in the first 12 characters. ( SHA-1冲突的概率几乎不可能 )
#### Branch References
One straightforward way to refer to a particular commit is if it’s the commit at the tip of a branch; in that case, you can simply use the branch name in any Git command that expects a reference to a commit. For instance, if you want to examine the last commit object on a branch, the following commands are equivalent, assuming that the `topic1` branch points to commit `ca82a6d…​`: ( branch name 在许多情况等价于该分支上最新的提交 )
```console
$ git show ca82a6dff817ec66f44342007202690a93763949
$ git show topic1
```

If you want to see which specific SHA-1 a branch points to, or if you want to see what any of these examples boils down to in terms of SHA-1s, you can use a Git plumbing tool called `rev-parse`. You can see [Git Internals](https://git-scm.com/book/en/v2/ch00/ch10-git-internals) for more information about plumbing tools; basically, `rev-parse` exists for lower-level operations and isn’t designed to be used in day-to-day operations. However, it can be helpful sometimes when you need to see what’s really going on. Here you can run `rev-parse` on your branch.
```console
$ git rev-parse topic1
ca82a6dff817ec66f44342007202690a93763949
```
( `git rev-parse` 可以用于查看 branch 最新提交的 SHA-1 )
#### RefLog Shortnames
One of the things Git does in the background while you’re working away is keep a “reflog” — a log of where your HEAD and branch references have been for the last few months. ( git 的 reflog 记录了我们的 HEAD 以及各个分支在一段时间内指向的位置 )

You can see your reflog by using `git reflog`:
```console
$ git reflog
734713b HEAD@{0}: commit: Fix refs handling, add gc auto, update tests
d921970 HEAD@{1}: merge phedders/rdocs: Merge made by the 'recursive' strategy.
1c002dd HEAD@{2}: commit: Add some blame and merge stuff
1c36188 HEAD@{3}: rebase -i (squash): updating HEAD
95df984 HEAD@{4}: commit: # This is a combination of two commits.
1c36188 HEAD@{5}: rebase -i (squash): updating HEAD
7e05da5 HEAD@{6}: rebase -i (pick): updating HEAD
```

Every time your branch tip is updated for any reason, Git stores that information for you in this temporary history. You can use your reflog data to refer to older commits as well. For example, if you want to see the fifth prior value of the HEAD of your repository, you can use the `@{5}` reference that you see in the reflog output: ( 分支头部更新，git 就会存储这些信息)
```console
$ git show HEAD@{5}
```
( `git show HEAD@{5}` 展示 reflog 中正数第五个 HEAD 所指向的提交的信息 )

You can also use this syntax to see where a branch was some specific amount of time ago. For instance, to see where your `master` branch was yesterday, you can type:
```console
$ git show master@{yesterday}
```
( `@{yesterday}` 展示 `master` 分支昨天所在的位置 )

This technique only works for data that’s still in your reflog, so you can’t use it to look for commits older than a few months. ( reflog 中的数据会过期，因此太早之前的不能用这种方式引用 )

To see reflog information formatted like the `git log` output, you can run `git log -g`:
```console
$ git log -g master
commit 734713bc047d87bf7eac9674765ae793478c50d3
Reflog: master@{0} (Scott Chacon <schacon@gmail.com>)
Reflog message: commit: Fix refs handling, add gc auto, update tests
Author: Scott Chacon <schacon@gmail.com>
Date:   Fri Jan 2 18:32:33 2009 -0800

    Fix refs handling, add gc auto, update tests

commit d921970aadf03b3cf0e71becdaab3147ba71cdef
Reflog: master@{1} (Scott Chacon <schacon@gmail.com>)
Reflog message: merge phedders/rdocs: Merge made by recursive.
Author: Scott Chacon <schacon@gmail.com>
Date:   Thu Dec 11 15:08:43 2008 -0800

    Merge commit 'phedders/rdocs'
```

It’s important to note that reflog information is strictly local — it’s a log only of what _you’ve_ done in _your_ repository. ( reflog 的信息完全局部，仅记录本地下各分支的引用历史 )

The references won’t be the same on someone else’s copy of the repository; also, right after you initially clone a repository, you’ll have an empty reflog, as no activity has occurred yet in your repository. Running `git show HEAD@{2.months.ago}` will show you the matching commit only if you cloned the project at least two months ago — if you cloned it any more recently than that, you’ll see only your first local commit. ( 刚 clone 来的仓库的 reflog 为空 )
#### Ancestry Reference
The other main way to specify a commit is via its ancestry. If you place a `^` (caret) at the end of a reference, Git resolves it to mean the parent of that commit. Suppose you look at the history of your project:
```console
$ git log --pretty=format:'%h %s' --graph
* 734713b Fix refs handling, add gc auto, update tests
*   d921970 Merge commit 'phedders/rdocs'
|\
| * 35cfb2b Some rdoc changes
* | 1c002dd Add some blame and merge stuff
|/
* 1c36188 Ignore *.gem
* 9b29157 Add open3_detach to gemspec file list
```
Then, you can see the previous commit by specifying `HEAD^`, which means “the parent of HEAD”: ( `HEAD^` 意味着 HEAD 的父提交 )
```console
$ git show HEAD^
commit d921970aadf03b3cf0e71becdaab3147ba71cdef
Merge: 1c002dd... 35cfb2b...
Author: Scott Chacon <schacon@gmail.com>
Date:   Thu Dec 11 15:08:43 2008 -0800

    Merge commit 'phedders/rdocs'
```

You can also specify a number after the `^` to identify _which_ parent you want; for example, `d921970^2` means “the second parent of d921970.” This syntax is useful only for merge commits, which have more than one parent — the _first_ parent of a merge commit is from the branch you were on when you merged (frequently `master`), while the _second_ parent of a merge commit is from the branch that was merged (say, `topic`): ( 对于 merge 提交，它可能有多个父提交，可以在 `^` 后添加数字引用，merge 提交的第一个父提交是执行 git merge 时所在的提交，一般是 `master` ，第二个父提交即作为 git merge 参数的提交 )
```console
$ git show d921970^
commit 1c002dd4b536e7479fe34593e72e6c6c1819e53b
Author: Scott Chacon <schacon@gmail.com>
Date:   Thu Dec 11 14:58:32 2008 -0800

    Add some blame and merge stuff

$ git show d921970^2
commit 35cfb2b795a55793d7cc56a6cc2060b4bb732548
Author: Paul Hedderly <paul+git@mjr.org>
Date:   Wed Dec 10 22:22:03 2008 +0000

    Some rdoc changes
```

The other main ancestry specification is the `~` (tilde). This also refers to the first parent, so `HEAD~` and `HEAD^` are equivalent. The difference becomes apparent when you specify a number. `HEAD~2` means “the first parent of the first parent,” or “the grandparent” — it traverses the first parents the number of times you specify. For example, in the history listed earlier, `HEAD~3` would be:
```console
$ git show HEAD~3
commit 1c3618887afb5fbcbea25b7c013f4e2114448b8d
Author: Tom Preston-Werner <tom@mojombo.com>
Date:   Fri Nov 7 13:47:59 2008 -0500

    Ignore *.gem
```
This can also be written `HEAD~~~`, which again is the first parent of the first parent of the first parent:
```console
$ git show HEAD~~~
commit 1c3618887afb5fbcbea25b7c013f4e2114448b8d
Author: Tom Preston-Werner <tom@mojombo.com>
Date:   Fri Nov 7 13:47:59 2008 -0500

    Ignore *.gem
```
( `HEAD~` 表示 HEAD 的第一个父提交，`HEAD~~` 表示 HEAD 的第一个父提交的第一个父提交，以此类推 )

You can also combine these syntaxes — you can get the second parent of the previous reference (assuming it was a merge commit) by using `HEAD~3^2`, and so on. ( `^` , `~` 可以结合使用 )
### 7.1.2 Commit Ranges
Now that you can specify individual commits, let’s see how to specify ranges of commits. This is particularly useful for managing your branches — if you have a lot of branches, you can use range specifications to answer questions such as, “What work is on this branch that I haven’t yet merged into my main branch?” ( 指定一个范围内的提交 )
#### Double Dot
The most common range specification is the double-dot syntax. This basically asks Git to resolve a range of commits that are reachable from one commit but aren’t reachable from another. For example, say you have a commit history that looks like [Example history for range selection](https://git-scm.com/book/en/v2/ch00/double_dot).
![[ProGit-Fig136.png]]

Say you want to see what is in your `experiment` branch that hasn’t yet been merged into your `master` branch. You can ask Git to show you a log of just those commits with `master..experiment` — that means “all commits reachable from `experiment` that aren’t reachable from `master`.”  ( `master..experiment` 表示 `experiment` 包含的 `master` 不包含的所有提交 )

For the sake of brevity and clarity in these examples, the letters of the commit objects from the diagram are used in place of the actual log output in the order that they would display:
```console
$ git log master..experiment
D
C
```

If, on the other hand, you want to see the opposite — all commits in `master` that aren’t in `experiment` — you can reverse the branch names. `experiment..master` shows you everything in `master` not reachable from `experiment`:
```console
$ git log experiment..master
F
E
```
( `experiment..master` 即 `master` 包含但 `experiment` 不包含的提交 )

This is useful if you want to keep the `experiment` branch up to date and preview what you’re about to merge. ( 该语法便于我们查看有什么 commit 是会被 merge 的 )

Another frequent use of this syntax is to see what you’re about to push to a remote: 
```console
$ git log origin/master..HEAD
```
This command shows you any commits in your current branch that aren’t in the `master` branch on your `origin` remote. ( `origin/master..HEAD` 表示在当前分支即 HEAD 中但不在 `origin/master` 中的提交，它方便我们查看有什么 commit 是会被 push 的 )

If you run a `git push` and your current branch is tracking `origin/master`, the commits listed by `git log origin/master..HEAD` are the commits that will be transferred to the server. 

You can also leave off one side of the syntax to have Git assume `HEAD`. For example, you can get the same results as in the previous example by typing `git log origin/master..` — Git substitutes `HEAD` if one side is missing. ( 可以省略写为 `origin/master..` git 默认添加 HEAD 在没有参数的一边 )
####  Multiple Points
The double-dot syntax is useful as a shorthand, but perhaps you want to specify more than two branches to indicate your revision, such as seeing what commits are in any of several branches that aren’t in the branch you’re currently on. Git allows you to do this by using either the `^` character or `--not` before any reference from which you don’t want to see reachable commits. Thus, the following three commands are equivalent:
```console
$ git log refA..refB
$ git log ^refA refB
$ git log refB --not refA
```
( 三个命令的意思都为审查在 `refB` 中但不在 `refA` 中的提交 )

This is nice because with this syntax you can specify more than two references in your query, which you cannot do with the double-dot syntax. For instance, if you want to see all commits that are reachable from `refA` or `refB` but not from `refC`, you can use either of:
```console
$ git log refA refB ^refC 
$ git log refA refB --not refC
```
( 在 refA 和 refB 中但不在 refC 中 )
#### Triple Dots
The last major range-selection syntax is the triple-dot syntax, which specifies all the commits that are reachable by _either_ of two references but not by both of them. Look back at the example commit history in [Example history for range selection](https://git-scm.com/book/en/v2/ch00/double_dot). If you want to see what is in `master` or `experiment` but not any common references, you can run:
```console
$ git log master...experiment
F
E
D
C
```
( `master...expreiment` 表示不同时在两个分支内，但也存在于其中一个分支内的提交 )
Again, this gives you normal `log` output but shows you only the commit information for those four commits, appearing in the traditional commit date ordering. ( 排序为日期排序 )

A common switch to use with the `log` command in this case is `--left-right`, which shows you which side of the range each commit is in. This helps make the output more useful:
```console
$ git log --left-right master...experiment
< F
< E
> D
> C
```
( `--left-right` 进一步展示这些提交是属于左边的分支还是右边的分支)
## 7.2 Git Tools - Interactive Staging
### 7.2.1 Interactive Staging
In this section, you’ll look at a few interactive Git commands that can help you craft your commits to include only certain combinations and parts of files. These tools are helpful if you modify a number of files extensively, then decide that you want those changes to be partitioned into several focused commits rather than one big messy commit. This way, you can make sure your commits are logically separate changesets and can be reviewed easily by the developers working with you.
> 本节介绍一些交互式 Git 命令，帮助我们构造我们的提交，使其仅包含文件的部分内容的组合
> 这些工具在我们大量修改文件，并且希望将这些修改分为几次提交的时候很有用，以使得我们的提交在逻辑上是分离的变更集合

If you run `git add` with the `-i` or `--interactive` option, Git enters an interactive shell mode, displaying something like this:
> `git add -i` 进入交互模式

```console
$ git add -i
           staged     unstaged path
  1:    unchanged        +0/-1 TODO
  2:    unchanged        +1/-1 index.html
  3:    unchanged        +5/-1 lib/simplegit.rb

*** Commands ***
  1: [s]tatus     2: [u]pdate      3: [r]evert     4: [a]dd untracked
  5: [p]atch      6: [d]iff        7: [q]uit       8: [h]elp
What now>
```
You can see that this command shows you a much different view of your staging area than you’re probably used to — basically, the same information you get with `git status` but a bit more succinct and informative. It lists the changes you’ve staged on the left and unstaged changes on the right.
> 它在左边列出了我们已经暂存的改变，在右边列出了尚未暂存的改变

After this comes a “Commands” section, which allows you to do a number of things like staging and unstaging files, staging parts of files, adding untracked files, and displaying diffs of what has been staged.
### 7.2.2 Staging and Unstaging Files
If you type `u` or `2` (for update) at the `What now>` prompt, you’re prompted for which files you want to stage:
```console
What now> u
           staged     unstaged path
  1:    unchanged        +0/-1 TODO
  2:    unchanged        +1/-1 index.html
  3:    unchanged        +5/-1 lib/simplegit.rb
Update>>
```
> 键入 `u/2` ，我们会被提示需要暂存哪些文件

To stage the `TODO` and `index.html` files, you can type the numbers:
```console
Update>> 1,2
           staged     unstaged path
* 1:    unchanged        +0/-1 TODO
* 2:    unchanged        +1/-1 index.html
  3:    unchanged        +5/-1 lib/simplegit.rb
Update>>
```

The `*` next to each file means the file is selected to be staged. If you press Enter after typing nothing at the `Update>>` prompt, Git takes anything selected and stages it for you:
```console
Update>>
updated 2 paths

*** Commands ***
  1: [s]tatus     2: [u]pdate      3: [r]evert     4: [a]dd untracked
  5: [p]atch      6: [d]iff        7: [q]uit       8: [h]elp
What now> s
           staged     unstaged path
  1:        +0/-1      nothing TODO
  2:        +1/-1      nothing index.html
  3:    unchanged        +5/-1 lib/simplegit.rb
```

Now you can see that the `TODO` and `index.html` files are staged and the `simplegit.rb` file is still unstaged. If you want to unstage the `TODO` file at this point, you use the `r` or `3` (for revert) option:
```console
*** Commands ***
  1: [s]tatus     2: [u]pdate      3: [r]evert     4: [a]dd untracked
  5: [p]atch      6: [d]iff        7: [q]uit       8: [h]elp
What now> r
           staged     unstaged path
  1:        +0/-1      nothing TODO
  2:        +1/-1      nothing index.html
  3:    unchanged        +5/-1 lib/simplegit.rb
Revert>> 1
           staged     unstaged path
* 1:        +0/-1      nothing TODO
  2:        +1/-1      nothing index.html
  3:    unchanged        +5/-1 lib/simplegit.rb
Revert>> [enter]
reverted one path
```
> 输入 `r/3` 可以取消暂存文件

Looking at your Git status again, you can see that you’ve unstaged the `TODO` file:
```console
*** Commands ***
  1: [s]tatus     2: [u]pdate      3: [r]evert     4: [a]dd untracked
  5: [p]atch      6: [d]iff        7: [q]uit       8: [h]elp
What now> s
           staged     unstaged path
  1:    unchanged        +0/-1 TODO
  2:        +1/-1      nothing index.html
  3:    unchanged        +5/-1 lib/simplegit.rb
```

To see the diff of what you’ve staged, you can use the `d` or `6` (for diff) command. It shows you a list of your staged files, and you can select the ones for which you would like to see the staged diff. This is much like specifying `git diff --cached` on the command line:
> 输入 `d/6` 可以查看暂存的文件的 diff

```console
*** Commands ***
  1: [s]tatus     2: [u]pdate      3: [r]evert     4: [a]dd untracked
  5: [p]atch      6: [d]iff        7: [q]uit       8: [h]elp
What now> d
           staged     unstaged path
  1:        +1/-1      nothing index.html
Review diff>> 1
diff --git a/index.html b/index.html
index 4d07108..4335f49 100644
--- a/index.html
+++ b/index.html
@@ -16,7 +16,7 @@ Date Finder

 <p id="out">...</p>

-<div id="footer">contact : support@github.com</div>
+<div id="footer">contact : email.support@github.com</div>

 <script type="text/javascript">
```

With these basic commands, you can use the interactive add mode to deal with your staging area a little more easily.
### 7.2.3 Staging Patches
It’s also possible for Git to stage certain _parts_ of files and not the rest. For example, if you make two changes to your `simplegit.rb` file and want to stage one of them and not the other, doing so is very easy in Git. From the same interactive prompt explained in the previous section, type `p` or `5` (for patch). Git will ask you which files you would like to partially stage; then, for each section of the selected files, it will display hunks of the file diff and ask if you would like to stage them, one by one:
> Git 支持仅暂存文件的一部分
> 输入 `p/5` ，Git 会问我们哪个文件需要部分暂存


```console
diff --git a/lib/simplegit.rb b/lib/simplegit.rb
index dd5ecc4..57399e0 100644
--- a/lib/simplegit.rb
+++ b/lib/simplegit.rb
@@ -22,7 +22,7 @@ class SimpleGit
   end

   def log(treeish = 'master')
-    command("git log -n 25 #{treeish}")
+    command("git log -n 30 #{treeish}")
   end

   def blame(path)
Stage this hunk [y,n,a,d,/,j,J,g,e,?]?
```

You have a lot of options at this point. Typing `?` shows a list of what you can do:

```console
Stage this hunk [y,n,a,d,/,j,J,g,e,?]? ?
y - stage this hunk
n - do not stage this hunk
a - stage this and all the remaining hunks in the file
d - do not stage this hunk nor any of the remaining hunks in the file
g - select a hunk to go to
/ - search for a hunk matching the given regex
j - leave this hunk undecided, see next undecided hunk
J - leave this hunk undecided, see next hunk
k - leave this hunk undecided, see previous undecided hunk
K - leave this hunk undecided, see previous hunk
s - split the current hunk into smaller hunks
e - manually edit the current hunk
? - print help
```

Generally, you’ll type `y` or `n` if you want to stage each hunk, but staging all of them in certain files or skipping a hunk decision until later can be helpful too. If you stage one part of the file and leave another part unstaged, your status output will look like this:

```console
What now> 1
           staged     unstaged path
  1:    unchanged        +0/-1 TODO
  2:        +1/-1      nothing index.html
  3:        +1/-1        +4/-0 lib/simplegit.rb
```

The status of the `simplegit.rb` file is interesting. It shows you that a couple of lines are staged and a couple are unstaged. You’ve partially staged this file. At this point, you can exit the interactive adding script and run `git commit` to commit the partially staged files.

You also don’t need to be in interactive add mode to do the partial-file staging — you can start the same script by using `git add -p` or `git add --patch` on the command line.

Furthermore, you can use patch mode for partially resetting files with the `git reset --patch` command, for checking out parts of files with the `git checkout --patch` command and for stashing parts of files with the `git stash save --patch` command. We’ll go into more details on each of these as we get to more advanced usages of these commands.