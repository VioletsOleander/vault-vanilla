# 前言
LaTeX [1] 是一个文档准备系统 (Document Preparing System)，它非常适用于生成高印刷质量的科技类和数学类文档。它也能够生成所有其他种类的文档，小到简单的信件，大到完整的书籍。LaTeX 使用 TeX [6] 作为它的排版引擎。这份短小的手册描述了 LaTeX2e 的使用，对 LaTeX 的大多数应用来说应该是足够了。参考文献[1, 2] 对 LaTeX 系统提供了完整的描述。 

本手册在英文版 lshort 的基础上进行了适当的重新编排，共有八章和两篇附录：
- 第一章
    讲述 LaTeX 的来源，源代码的基本结构，以及如何编译源代码生成文档。
- 第二章
    讲述在 LaTeX 中如何书写文字，包括中文。 
- 第三章
    讲述文档排版的基本元素——标题、目录、列表、图片、表格等等。结合前一章的内容，你应当能够制作内容较为丰富的文档了。 
- 第四章
    LaTeX 排版公式的能力是众人皆知的。本章的内容涉及了一些排版公式经常用到的命令、环境和符号。章节末尾列出了 LaTeX 常见的数学符号。 
- 第五章
    介绍了如何修改文档的一些基本样式，包括字体、段落、页面尺寸、页眉页脚等。 
- 第六章
    介绍了 LaTeX 的一些扩展功能：排版参考文献、排版索引、排版带有颜色和超链接的电子文档。 
- 第七章
    介绍了如何在 LaTeX 里使用 TikZ 绘图。作为入门手册，这一部分点到为止。 
- 第八章
    当你相当熟悉前面几章的内容，需要自己编写命令和宏包扩展 LaTeX 的功能时，本章介绍了一些基本的命令满足你的需求。 
- 附录 A 
    介绍了如何安装 TeX 发行版和更新宏包。 
- 附录 B 
    当新手遇到错误和需要寻求更多帮助时，本章提供了一些基本的参考。 

这些章节是循序渐进的，建议刚刚熟悉 LaTeX 的读者按顺序阅读。一定要认真阅读例子的源代码，它们贯穿全篇手册，包含了很多的信息。 

如果你已经对 LaTeX 较为熟练，本手册的资源已不足够解决你的问题时，请访问“Compre-hensive $\mathrm{TCX}$ Archive Network ” (CTAN) 站点，主页是 https://www.ctan.org 。所有的宏包也可以从 https://mirrors.ctan.org 和遍布全球的各个镜像站点中获得。

在本书中你会找到其他引用 CTAN 的地方，形式为 CTAN:// 和之后的树状结构。引用本身是一个超链接，点击后将打开内容在 CTAN 上相应位置的页面。

要在自己的电脑上安装 TeX 发行版，请参考附录 A 中的内容。各个操作系统下的 TeX 发行版位于 CTAN://systems。 

如果你有意在这份文档中增加、删除或者改变一些内容，请通知作者。作者对 LaTeX 初学者的反馈特别感兴趣，尤其是关于这份介绍哪些内容很容易理解，哪些内容可能需要更好地解释，而哪些内容由于太过难以理解、非常不常用而不适宜放在本手册。 

CTeX 开发小组 https://github.com/CTeX-org 

# 1 LaTeX 的基本概念
欢迎使用 LaTeX！本章开头用简短的篇幅介绍了 LaTeX 的来源，然后介绍了 LaTeX 源代码的写法，编译 LaTeX 源代码生成文档的方法，以及理解接下来的章节的一些必要知识。 
## 1.1 概述 
### 1.1.1 TeX 
TeX 是高德纳 (Donald E. Knuth) 为排版文字和数学公式而开发的软件[6]。1977 年，正在编写《计算机程序设计艺术》的高德纳意识到每况愈下的排版质量将影响其著作的发行，为扭转这种状况，他着手开发 TeX ，发掘当时刚刚用于出版工业的数字印刷设备的潜力。1982 年，高德纳发布 TeX 排版引擎，而后在 1989 年又为更好地支持 8-bit 字符和多语言排版而予以改进。 TeX 以其卓越的稳定性、跨平台能力和几乎没有 bug 的特性而著称。它的版本号不断趋近于 $\pi$ ，当前为 3.141592653。 

TeX 读作“Tech”，与汉字“泰赫”的发音相近，其中“ch”的发音类似于“h”。TeX 的拼写来自希腊词语τεχνική (technique，技术) 开头的几个字母，在 ASCII 字符环境中写作 TeX。 

### 1.1.2 LaTeX
LaTeX 是一种使用 TeX 程序作为排版引擎的格式 (format)，可以粗略地将它理解成是对 TeX 的一层封装。LaTeX 最初的设计目标是分离内容与格式，以便作者能够专注于内容创作而非版式设计，并能以此得到高质量排版的作品。LaTeX 起初由 Leslie Lamport 博士[1] 开发，目前由 LaTeX 工作组进行维护。 

LaTeX 读作“Lah-tech”或者“Lay-tech”，与汉字“拉泰赫”或“雷泰赫”的发音相近，在 ASCII 字符环境写作 LaTeX。 LaTeX2e 是 LaTeX 的当前版本，意思是超出了第二版，但还远未达到第三版，在 ASCII 字符环境写作 LaTeX2e。 

### 1.1.3 LaTeX 的优缺点 
经常有人喜欢对比 LaTeX 和以 Microsoft Office Word 为代表的“所见即所得”(What You See Is What You Get) 字处理工具。这种对比是没有意义的，因为 TeX 是一个排版引擎，LaTeX 是其封装，而 Word 是字处理工具。二者的设计目标不一致，也各自有自己的适用范围。 

不过，这里仍旧总结 LaTeX 的一些优点： 
- 具有专业的排版输出能力，产生的文档看上去就像“印刷品”一样。
- 具有方便而强大的数学公式排版能力，无出其右者。 
- 绝大多数时候，用户只需专注于一些组织文档结构的基础命令，无需 (或很少) 操心文档的版面设计。
- 很容易生成复杂的专业排版元素，如脚注、交叉引用、参考文献、目录等。
- 强大的可扩展性。世界各地的人开发了数以千计的 LaTeX 宏包用于补充和扩展 LaTeX 的功能。一些常用宏包列在了本手册附录中的 B.3 小节。更多的宏包参考 The LaTeX companion[2]。
- 能够促使用户写出结构良好的文档——而这也是 LaTeX 存在的初衷。
- LaTeX 和 TeX 及相关软件是跨平台、免费、开源的。无论用户使用的是 Windows，macOS (OS X)，GNU/Linux 还是 FreeBSD 等操作系统，都能轻松获得和使用这一强大的排版工具，并且获得稳定的输出。

LaTeX 的缺点也是显而易见的：
- 入门门槛高。本手册的副标题叫做“111 分钟了解 LaTex2e”，实际上“111”是本手册正文部分 (包括附录) 的页数。如果真的以平均一页一分钟的速度看完了本手册，你只是粗窥门径而已，离学会它还很远。
- 不容易排查错误。LaTeX 作为一个依靠编写代码工作的排版工具，其使用的宏语言比 C++ 或 Python 等程序设计语言在错误排查方面困难得多。它虽然能够提示错误，但不提供调试的机制，有时错误提示还很难理解。
- 不容易定制样式。LaTeX 提供了一个基本上良好的样式，为了让用户不去关注样式而专注于文档结构。但如果想要改进 LaTeX 生成的文档样式则是十分困难的。
- 相比“所见即所得”的模式有一些不便，为了查看生成文档的效果，用户总要不停地编译。 

### 1.1.4 命令行基础 
LaTeX 和 TeX 及相关软件大多仅提供了命令行接口，而不像 Word、Adobe InDesign 一样有图形用户界面。命令行程序的结构往往比较简单，它们接受用户输入，读取相关文件，进行一些操作和运算后输出目标文件，有时还会将提示信息、运行结果显示在屏幕上。在 Windows 系统上，如需进入命令行，可在开始菜单中搜索“命令提示符”，也可在“运行”窗口中输入 cmd 打开；Linux 或 macOS 等 \*nix 系统中可搜索 “Terminal” 打开终端。部分系统也提供了一些快捷方式，具体请参考相关手册。 

与常规软件类似，命令行程序也都是可执行程序，在 Windows 上后缀名为. exe，而在类 Unix 系统上则需要带有 x 权限。在大多数命令行环境中，系统会根据环境变量 PATH 中存储的路径来搜索可供执行的程序。因此在运行之前，需确保 LaTeX 、TeX 及相关程序所在路径已包含在 PATH 中。 

在命令行中运行程序时，需要先输入程序名，其后可加一系列用空格分隔的参数，并按下 Enter 键执行。一般情况下，命令行程序执行完毕会自行退出。若遇到错误或中断，可输入 Ctrl+C 以强制结束。 

使用命令行程序输入、输出文件时，需确保文件路径正确。通常需要先切换到文件所在目录，再执行有关程序。切换路径可以执行 

```
cd <path>
```

注意 `<path>`  中的多级目录在 Windows 系统上使用反斜线 \ 分隔，而在类 Unix 系统上使用正斜线 / 分隔。如果 `<path>`  中带有空格，则需加上引号 "。此外，在 Windows 系统上如果要切换到其他分区，还需加上 /d 选项，例如 `cd /d "C:\Program Files (x86)\"` 。 

许多用户会使用 TeXworks 或 TeXstudio 等编辑器来编写 LaTeX 文档。这些编辑器提供的编译功能，实际上只是对特定命令行程序的封装，而并非魔法。 

## 1.2 第一次使用 LaTeX 
源代码 1.1 是一份最短的 LaTeX 源代码示例。 

```latex
\document class{article}
\begin{document}
``Hello world!'' from \LaTeX.
\end{document} 
```

源代码 1.1: LaTeX 的一个最简单的源代码示例。 

这里首先介绍如何编译使用这份源代码，在后续小节中再介绍源代码的细节。你可以将这份源代码保存为 `helloworld.TeX` ，而后编译。具体来说 : 

- 如果使用 TeXworks 或 TeXstudio 等编辑器，你可以使用编辑器提供的“编译”按钮或者“排版”按钮。建议使用 pdfLaTeX 或 XeLaTeX 作为默认的编译方式 (不同编译方式的差别，见 1.7 节)。 
- 如果使用命令行方式进行编译，则需打开 Windows 命令提示符或者\*nix 的终端，在源代码所在的目录下输入： 

```
pdfLaTeX helloworld 
```

或者 

```
xeLaTeX helloworld 
```

如果编译成功，可以在 `helloworld.TeX` 所在目录看到生成的 `helloworld.pdf` 以及一些其它文件。 

源代码 1.2 是在 LaTeX 排版中文的一个最简示例。编译的方式与上一份源代码相同，但需使用 XeLaTeX 编译方式。中文支持的详细内容见 2.2 节。

```latex
\document class{cTeXart} 
\begin{document} 
“你好，世界！” 来自 \LaTeX{} 的问候。
\end{document} 
```

源代码 1.2: 在 LaTeX 中排版中文的最简源代码示例。 

## 1.3 LaTeX 命令和代码结构 
LaTeX 的源代码为文本文件。这些文本除了文字本身，还包括各种命令，用在排版公式、划分文档结构、控制样式等等不同的地方。 

>  LaTex 源码为文本文件

### 1.3.1 LaTeX 命令和环境 
LaTeX 中命令 (控制序列) 以反斜线 \ 开头，为以下两种形式之一： 
- 反斜线和后面的一串字母，如 \LaTeX。它们以任意非字母符号 (空格、数字、标点等) 为界限。 
- 反斜线和后面的单个非字母符号，如 \\$ 。 

>  两种形式的 LaTex 命令：
>  1. `\[a-zA-Z]+` ，以任意非字母符号为界限
>  2. `\` + 单个非字母符号

要注意 LaTeX 命令是对大小写敏感的，比如输入 \LaTeX 命令可以生成错落有致的 LaTeX 字母组合，但输入 \LaTeX 或者 \LaTeX 什么都得不到，还会报错；它们与 \LaTeX 是不同的命令。 

字母形式的 LaTeX 命令忽略其后的所有连续空格。如果要人为引入空格，需要在命令后面加一对花括号阻止其忽略空格： 

>  字母形式的 LaTex 命令后面的空格都会被忽略
>  要添加空格，可以添加花括号 `\Tex{}` 或使用 `\ ` 插入一个间距，即 `\Tex\ `

```latex
Shall we call ourselves
\Tex users

or \Tex{} users?
```

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/cceb697b4ddf5a5da1dc6d185deb4948942364a124eabc69e03eacaf8342b91a.jpg) 

一些 LaTeX 命令可以接收一些参数，参数的内容会影响命令的效果。LaTeX 的参数分为可选参数和必选参数。可选参数以方括号 \[ 和 \] 包裹；必选参数一般以花括号 { 和 } 包裹 (以单个字符作为命令的参数时，可以不加括号，例如 `frac 1 2` 等价于 `\frac {1} {2}` )。还有些命令可以带一个星号 \* ，带星号和不带星号的命令效果有一定差异。初次接触这些概念时，可以粗略地把星号看作一种特殊的可选参数。 

>  LaTex 命令的可选参数用 `[]` 包括，必选参数用 `{}` 包括，`*` 也可以视作特殊的可选参数

LaTeX 中还包括环境，用以令一些效果在局部生效，或是生成特殊的文档元素。LaTeX 环境的用法为一对命令 \begin 和 \end： 

```latex
\begin{<environment name>}[<optional arguments>]{<mandatory arguments>}
...
\end{<environment name>}
```

其中 ⟨environment name⟩ 为环境名，\begin 和\end 中填写的环境名应当一致。类似命令，{ ⟨ mandatory arguments ⟩ } 和 \[ ⟨ optional arguments ⟩ \] 为环境所需的必选和可选参数。 LaTex 环境可能需要一个或多个必选/可选参数，也可能完全不需要参数。部分环境允许嵌套使用。 

>  LaTex 用 `\begin, \end` 包裹，环境也可以有可选参数和必选参数，环境名可以视作必选参数

有些命令 (如 \bfseries) 会对其后所有内容产生作用。若要限制其作用范围，则需要使用分组。LaTeX 使用一对花括号 { 和 } 作为分组，在分组中使用的命令被限制在分组内，不会影响到分组外的内容。上文提到的 LaTeX 环境隐含了一个分组，在环境中的命令被包裹在分组内。5.1.1 和 5.1.2 小节中介绍的修改字体和字号的命令用法，即属此类。 

>  一些命令会对其后的所有内容产生作用，需要使用分组限制其作用范围
>  LaTex 使用 `{}` 划分分组 (一些命令在分组内也会产生全局作用)
>  LaTex 环境本身也隐含了分组，环境中的命令的作用范围都限制在了分组中

### 1.3.2 LaTeX 源代码结构 
LaTeX 源代码以一个 \documentclass 命令作为开头，它指定了文档使用的文档类。document 环境当中的内容是文档正文。 

>  LaTex 源码以 `\documentclass` 命令作为开头，指定该 LaTex 文档使用的文档类
>  文档的正文在 `document` 环境中，`document` 环境之后的内容都会被忽略

在 \document class 和 \begin{document} 之间的位置称为导言区。在导言区中常会使用 \usepackage 命令调用宏包，还会进行文档的全局设置。

>  `\documentclass` 命令和 `document` 环境的开始 `\begin{document}` 之间的位置称为导言区
>  导言区一般使用 `\usepackage` 命令调用宏包，以及进行文档的全局设置

```latex
\documentclass{...} % ... 为某文档类
 % 导言区
\begin{document}
 % 正文内容
\end{document}
 % 此后内容会被忽略 
```

## 1.4 LaTex 宏包和文档类 
本节将仔细解释在 1.3.2 小节中出现的宏包和文档类的概念以及详细用法。 

### 1.4.1 文档类 
文档类规定了 LaTeX 源代码所要生成的文档的性质——普通文章、书籍、演示文稿、个人简历等等。LaTeX 源代码的开头须用 \documentclass 指定文档类： 

>  `\documentclass` 命令的格式如下，LaTex 源码必须在开头指定文档类，文档类决定了 LaTex 要生成的文档的性质

```latex
\documentclass[<options>]{<class-name>}
```

其中 ⟨class-name⟩ 为文档类的名称，如 LaTeX 提供的 article、report、book，在其基础上派生的一些文档类，如支持中文排版的 cTeXart、cTeXrep、cTeXbook，或者有其它功能的一些文档类，如 moderncv、beamer 等。 LaTeX 提供的基础文档类见表 1.1，其中前三个习惯上称为“标准文档类”。 

>  必选参数 `<class-name>` 为文档类的名称，LaTex 提供了标准文档类 `artical, report, book`，其基础上还派生了 `cTeXart, cTeXrep, cTeXbook` 等，以及一些有其他功能的文档类 `moderncv, beamer` 等

表 1.1: LaTeX 提供的基础文档类

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/b51c6ab795c0e4ce4473593c760dcaf389c1fbe23a154a3f779a88a698cd806b.jpg) 

可选参数 ⟨options⟩ 为文档类指定选项，以全局地规定一些排版的参数，如字号、纸张大小、单双面等等。比如调用 article 文档类排版文章，指定纸张为 A4 大小，基本字号为 11pt，双面排版： 

>  可选参数 `<options>` 为文档类指定选项，全局规定文档的排版参数

```latex
\documentclass[11pt, twoside, a4paper]{artical}
```

LaTeX 的三个标准文档类可指定的选项包括： 

- 10pt, 11pt, 12pt 
    指定文档的基本字号。默认为 $10\mathrm{pt}$ 。 
- a4paper, letter paper, …
    指定纸张大小，默认为美式信纸 letterpaper ( $8.5\,\mathrm{in}\times11$ in，大约相当于 $21.6\,\mathrm{cm}\times28.0\,\mathrm{cm})$ 。可指定选项还包括 a5paper，b5paper，executive paper 和 legalpaper。有关纸张大小的更多细节，请参考 5.4.1 小节。 
- twoside, oneside 
    指定单面/双面排版。双面排版时，奇偶页的页眉页脚、页边距不同。article 和 report 默认为 oneside，book 默认为 twoside。 
- onecolumn, twocolumn 
    指定单栏/双栏排版。默认为 onecolumn。 
- openright, openany 
    指定新的一章 \chapter 是在奇数页 (右侧) 开始，还是直接紧跟着上一页开始。report 默认为 openany，book 默认为 openright。对 article 无效。 
- landscape 
    指定横向排版。默认为纵向。 
- titlepage, notitlepage 
    指定标题命令 \maketitle 是否生成单独的标题页。article 默认为 no title page，report 和 book 默认为 titlepage。 
- fleqn 
    令行间公式左对齐。默认为居中对齐。 
- leqno 
    将公式编号放在左边。默认为右边。 
- draft, final 
    指定草稿/终稿模式。草稿模式下，断行不良 (溢出) 的地方会在行尾添加一个黑色方块；插图、超链接等功能也会受这一组选项影响，具体见后文。默认为 final。 

>  三个标准文档类可指定的选项包括
>  - `xxpt` 指定字号，默认为 10pt
>  - `xxpaper` 指定纸张，默认为美式信纸 letterpaper
>  - `xxside` 指定单面还是双面排版，单面排版时，奇偶页的页眉页脚页边距都相同，双面排版则不同。`artical, report` 默认单面，`book` 默认双面
>  - `xxcolumn` 指定单栏/双栏排版，默认单栏
>  - `openxx` 指定新的一章 `\chapter` 是否强制在奇数页 (右侧) 开始，`report` 默认为 `openright` ，`book` 不强制，对 `artical` 无效
>  - `landscape` 指定文档为横向排版，默认为纵向
>  - `(no)titlepage` 指定 `\maketile` 是否生成单独的标题页，`artical` 默认不生产，`report, book` 默认生成
>  - `fleqn` 指定 inline 公式左对齐，默认为居中对齐
>  - `leqno` 将公式编号放在左边，默认为右边
>  - `draft/final` 草稿模式下编译更快，但会影响插图、超链接的渲染，以及短行不良的地方的解析

### 1.4.2 宏包 
在使用 LaTeX 时，时常需要依赖一些扩展来增强或补充 LaTeX 的功能，比如排版复杂的表格、插入图片、增加颜色甚至超链接等等。这些扩展称为宏包。调用宏包的方法非常类似调用文档类的方法： 
\usepackage [ ⟨ options ⟩ ] { ⟨ package-name ⟩ } 
\usepackage 可以一次性调用多个宏包，在⟨package-name⟩中用逗号隔开。这种用法一般不要指定选项 8： 
% 一次性调用三个排版表格常用的宏包
\usepackage{tabularx, makecell, multirow} 
附录B.3 汇总了常用的一些宏包。我们在手册接下来的章节中，也会穿插介绍一些最常用的宏包的使用方法。 
在使用宏包和文档类之前，一定要首先确认它们是否安装在你的计算机中，否则\use-package 等命令会报错误。详见附录A.2。 
宏包 (包括前面所说的文档类) 可能定义了许多命令和环境，或者修改了 LaTeX 已有的命令和环境。它们的用法说明记在相应宏包和文档类的帮助文档。在 Windows 命令提示符或者 Linux 终端下输入命令可查阅相应文档： 
TeXdoc ⟨ pkg-name ⟩ 
其中⟨pkg-name⟩是宏包或者文档类的名称。更多获得帮助的方法见附录B.2。 
## 1.5 LaTeX 用到的文件一览 
除了源代码文件. TeX 以外，我们在使用 LaTeX 时还可能接触到各种格式的文件。本节简单介绍一下在使用 LaTeX 时能够经常见到的文件。 
每个宏包和文档类都是带特定扩展名的文件，除此之外也有一些文件出现于 LaTeX 模板中： 
.sty 宏包文件。宏包的名称与文件名一致。 
.cls 文档类文件。文档类名称与文件名一致。 
.bib BIBTeX 参考文献数据库文件。 
.bst $\mathrm{BiotaEX}$ 用到的参考文献格式模板。详见 6.1.4 小节。 
LaTeX 在编译过程中除了生成. dvi 或. pdf 格式的文档外 9，还可能会生成相当多的辅助文件和日志。一些功能如交叉引用、参考文献、目录、索引等，需要先通过编译生成辅助文件，然后再次编译时读入辅助文件得到正确的结果，所以复杂的 LaTeX 源代码可能要编译多次： 
.log 排版引擎生成的日志文件，供排查错误使用。 
.aux LaTeX 生成的主辅助文件，记录交叉引用、目录、参考文献的引用等。 
.toc LaTeX 生成的目录记录文件。 
.lot LaTeX 生成的表格目录记录文件。 
.bbl BIBTeX 生成的参考文献记录文件。 
.blg BIBTeX 生成的日志文件。 
.idx LaTeX 生成的供 makeindex 处理的索引记录文件。 
.ind makeindex 处理. idx 生成的用于排版的格式化索引文件。 
.ilg makeindex 生成的日志文件。 
.out hyperref 宏包生成的 PDF 书签记录文件。 
# 1.6 文件的组织方式 
当编写长篇文档时，例如当编写书籍、毕业论文时，单个源文件会使修改、校对变得十分困难。将源文件分割成若干个文件，例如将每章内容单独写在一个文件中，会大大简化修改和校对的工作。可参考源代码 3.1 的写法。 
LaTeX 提供了命令\include 用来在源代码里插入文件： 
\include{ ⟨ filename ⟩ } 
⟨filename⟩为文件名 (不带. TeX 扩展名) 10，如果和要编译的主文件不在一个目录中，则要加上相对或绝对路径，例如： 
\include{chapters/file} % 相对路径 
\include{/home/Bob/file} % \*nix (包含 Linux、macOS) 绝对路径
\include{D:/file} % Windows 绝对路径，用正斜线 
值得注意的是\include 在读入⟨filename⟩之前会另起一页。有的时候我们并不需要这样，而是用\input 命令，它纯粹是把文件里的内容插入： 
\input{ ⟨ filename ⟩ } 
当导言区内容较多时，常常将其单独放置在一个. TeX 文件中，再用\input 命令插入。复杂的图、表、代码等也会用类似的手段处理。 
LaTeX 还提供了一个\includeonly 命令来组织文件，用于导言区，指定只载入某些文件。导言区使用了\includeonly 后，正文中不在其列表范围的\include 命令不会起效： 
\includeonly{ ⟨ filename1 ⟩ , ⟨ filename2 ⟩ ,…} 
需要注意的是，使用\include 和\input 命令载入的文件名最好不要加空格和特殊字符，也尽量避免使用中文名，否则很可能会出错 11。 
最后介绍一个实用的工具宏包 syntonly。加载这个宏包后，在导言区使用\syntaxonly 命 
令，可令 LaTeX 编译后不生成 DVI 或者 PDF 文档，只排查错误，编译速度会快不少：
\usepackage{syntonly}
 \syntaxonly 
如果想生成文档，则用% 注释掉\syntaxonly 命令即可。 
# 1.7 L A TeX 和 $\mathbf{TeX}$ 相关的术语和概念 
在本章的最后有必要澄清几个概念： 
引擎全称为排版引擎，是编译源代码并生成文档的程序，如 pdfTeX、XeTeX 等。有时也称为编译器。 
格式是定义了一组命令的代码集。LaTeX 就是最广泛应用的一个格式，高德纳本人还编写了一个简单的 plain TeX 格式，没有定义诸如\document class 和\section 等等命令。 
编译命令是实际调用的、结合了引擎和格式的命令。如 xeLaTeX 命令是结合 XeTeX 引擎和 LaTeX 格式的一个编译命令。 
常见的引擎、格式和编译命令的关系总结于表 1.2。 
LaTeX 编译命令和 LaTeX 格式往往容易混淆，在讨论关于 LaTeX 的时候需要明确。为避免混淆，本手册中的 LaTeX 一律指的是格式，编译命令则用等宽字体 LaTeX 表示。 
在此介绍一下几个编译命令的基本特点： 
LaTeX 虽然名为 LaTeX 命令，底层调用的引擎其实是 pdfTeX。该命令生成 dvi (Device Inde-pendent) 格式的文档，用 dvipdfmx 命令可以将其转为 pdf。 
pdfLaTeX 底层调用的引擎也是 pdfTeX，可以直接生成 pdf 格式的文档。 
表 1.2: TeX 引擎、格式及其对应的编译命令。
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/6caaa3cd51678f8cf5756c206181f6afb065ffbcb8d853028ab7ccf185d41101.jpg) 
xeLaTeX 底层调用的引擎是 $\mathrm {{X}_{\mathrm{{eff}E X}} }$ ，支持 UTF-8 编码和对 TrueType/OpenType 字体的调用。当前较为方便的中文排版解决方案基于 xeLaTeX，详见 2.2 节。 
luaLaTeX 底层调用的引擎是 $\mathrm {{L}u a T E X^{12}} $ ，这个引擎在 pdfTeX 引擎基础上发展而来，除了支持 UTF-8 编码和对 TrueType/OpenType 字体的调用外，还支持通过 Lua 语言扩展 TeX 的功能。luaLaTeX 编译命令下的中文排版支持需要借助 luaTeXja 宏包。 
# 2 用 LaTeX 排版文字
文字是排版的基础。本章主要介绍如何在 LaTeX 中输入各种文字符号，包括标点符号、连字符、重音等，以及控制文字断行和断页的方式。本章简要介绍了在 LaTeX 中排版中文的方法。随着 LaTeX 和底层 TeX 引擎的发展，旧方式 (CCT、CJK 等) 日渐退出舞台，xeLaTeX 和 luaLaTeX 编译命令配合 cTeX 宏包/文档类的方式成为当前的主流中文排版支持方式。 
# 3 文档元素
# 4 排版数学公式
# 5 排版样式设定
# 6 特色工具和功能
# 7 绘图功能
# 8 自定义 LaTeX 命令和功能
# A 安装 LaTeX 发行版
# B 排除错误、寻求帮助
