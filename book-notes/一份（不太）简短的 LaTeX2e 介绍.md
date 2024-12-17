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

>  Windows 系统中，多级目录用反斜线/转义符号 `\` 分割，类 Unix 系统中，用正斜线/除号 `/` 分割

许多用户会使用 TeXworks 或 TeXstudio 等编辑器来编写 LaTeX 文档。这些编辑器提供的编译功能，实际上只是对特定命令行程序的封装，而并非魔法。 

## 1.2 第一次使用 LaTeX 
源代码 1.1 是一份最短的 LaTeX 源代码示例。 

```latex
\documentclass{article}
\begin{document}
``Hello world!'' from \LaTeX.
\end{document} 
```

源代码 1.1: LaTeX 的一个最简单的源代码示例。 

这里首先介绍如何编译使用这份源代码，在后续小节中再介绍源代码的细节。你可以将这份源代码保存为 `helloworld.tex` ，而后编译。具体来说 : 

- 如果使用 TeXworks 或 TeXstudio 等编辑器，你可以使用编辑器提供的“编译”按钮或者“排版”按钮。建议使用 pdfLaTeX 或 XeLaTeX 作为默认的编译方式 (不同编译方式的差别，见 1.7 节)。 
- 如果使用命令行方式进行编译，则需打开 Windows 命令提示符或者\*nix 的终端，在源代码所在的目录下输入： 

```
pdflatex helloworld 
```

或者

```
xelatex helloworld 
```

如果编译成功，可以在 `helloworld.tex` 所在目录看到生成的 `helloworld.pdf` 以及一些其它文件。 

源代码 1.2 是在 LaTeX 排版中文的一个最简示例。编译的方式与上一份源代码相同，但需使用 `xelatex` 编译方式。中文支持的详细内容见 2.2 节。

```latex
\documentclass{ctexart} 
\begin{document} 
“你好，世界！” 来自 \LaTeX{} 的问候。
\end{document} 
```

源代码 1.2: 在 LaTeX 中排版中文的最简源代码示例。 

>  注意文档类为 `ctexart` ，文件应该保存为 UTF-8 编码，以及使用 XeTex 引擎编译

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

在 \documentclass 和 \begin{document} 之间的位置称为导言区。在导言区中常会使用 \usepackage 命令调用宏包，还会进行文档的全局设置。

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

## 1.4 LaTeX 宏包和文档类 
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

>  LaTex 中的拓展称为宏包
>  宏包通过 `\usepackage` 命令调用，其中包名是必选参数

```latex
\usepackage[<options>]{<package-name>}
```

\usepackage 可以一次性调用多个宏包，在 ⟨package-name⟩ 中用逗号隔开。这种用法一般不要指定选项： 

```latex
% 一次性调用三个排版表格常用的宏包
\usepackage{tabularx, makecell, multirow} 
```

>  可以一次指定多个包名，用逗号隔开，但此时一般不要指定可选参数 (选项)
>  因为此时执行选项相当于对每个宏包都指定相同选项，如果有宏包不识别就会出错

附录 B.3 汇总了常用的一些宏包。我们在手册接下来的章节中，也会穿插介绍一些最常用的宏包的使用方法。 

在使用宏包和文档类之前，一定要首先确认它们是否安装在你的计算机中，否则\use-package 等命令会报错误。详见附录 A.2。 

宏包 (包括前面所说的文档类) 可能定义了许多命令和环境，或者修改了 LaTeX 已有的命令和环境。它们的用法说明记在相应宏包和文档类的帮助文档。在 Windows 命令提示符或者 Linux 终端下输入命令可查阅相应文档： 

>  宏包以及文档类会自己定义许多命令和环境，有的还会修改 LaTex 已有的命令和环境

```latex
texdoc <pkg-name>
```

其中 ⟨pkg-name⟩ 是宏包或者文档类的名称。更多获得帮助的方法见附录 B.2。 

## 1.5 LaTeX 用到的文件一览 
除了源代码文件 `.tex` 以外，我们在使用 LaTeX 时还可能接触到各种格式的文件。本节简单介绍一下在使用 LaTeX 时能够经常见到的文件。 

每个宏包和文档类都是带特定扩展名的文件，除此之外也有一些文件出现于 LaTeX 模板中： 
-  `.sty` 宏包文件。宏包的名称与文件名一致。 
-  `.cls` 文档类文件。文档类名称与文件名一致。 
-  `.bib` BiBTeX 参考文献数据库文件。 
-  `.bst` BiBTex 用到的参考文献格式模板。详见 6.1.4 小节。 

>  每个宏包和文档类本质都是带有特定拓展名的文件，宏包文件的拓展名为 `.sty` ，文档类文件的拓展名为 `.cls`

LaTeX 在编译过程中除了生成 `.dvi` 或 `.pdf` 格式的文档外，还可能会生成相当多的辅助文件和日志。一些功能如交叉引用、参考文献、目录、索引等，需要先通过编译生成辅助文件，然后再次编译时读入辅助文件得到正确的结果，所以复杂的 LaTeX 源代码可能要编译多次： 

>  复杂的 LaTex 源码需要多次编译，先编译生成辅助文件，第二次编译读入辅助文件再编译得到最终文件

-  `.log` 排版引擎生成的日志文件，供排查错误使用。 
-  `.aux` LaTeX 生成的主辅助文件，记录交叉引用、目录、参考文献的引用等。 
-  `.toc` LaTeX 生成的目录记录文件。 
-  `.lot` LaTeX 生成的表格目录记录文件。 
-  `.bbl` BiBTeX 生成的参考文献记录文件。 
-  `.blg` BiBTeX 生成的日志文件。 
-  `.idx` LaTeX 生成的供 makeindex 处理的索引记录文件。 
-  `.ind` makeindex 处理. idx 生成的用于排版的格式化索引文件。 
-  `.ilg` makeindex 生成的日志文件。 
-  `.out` hyperref 宏包生成的 PDF 书签记录文件。 

## 1.6 文件的组织方式 
当编写长篇文档时，例如当编写书籍、毕业论文时，单个源文件会使修改、校对变得十分困难。将源文件分割成若干个文件，例如将每章内容单独写在一个文件中，会大大简化修改和校对的工作。可参考源代码 3.1 的写法。 

>  将源文件划分为多个文件可以避免修改带来的不必要的编译

LaTeX 提供了命令 \include 用来在源代码里插入文件： 

```latex
\include{<filename>}
```

>  `\include` 命令用于在源码中插入文件，它只有一个必选参数为文件名

⟨filename⟩ 为文件名 (不带 `.tex` 扩展名)，如果和要编译的主文件不在一个目录中，则要加上相对或绝对路径，例如： 

```latex
\include{chapters/file} % 相对路径 
\include{/home/Bob/file} % \*nix (包含 Linux、macOS) 绝对路径
\include{D:/file} % Windows 绝对路径，用正斜线 
```

>  如果 `include` 的文件和主文件不在同一目录，需要将文件名改为绝对路径或相对路径

值得注意的是 \include 在读入 ⟨filename⟩ 之前会另起一页。有的时候我们并不需要这样，而是用 \input 命令，它纯粹是把文件里的内容插入： 

```latex
\input{<fliename>}
```

>  `\input` 命令也可以插入文件内容，差异在于 `\include` 会先令起一页再插入文件内容，而 `\input` 则直接插入

当导言区内容较多时，常常将其单独放置在一个 `.tex` 文件中，再用 \input 命令插入。复杂的图、表、代码等也会用类似的手段处理。 

>  例如，导言区内容较多时，就可以将其单独放在一个 `.tex` 文件中，用 `\input` 插入，复杂的图、表、代码也可以单独放在文件中，用 `\input` 插入

LaTeX 还提供了一个 \includeonly 命令来组织文件，用于导言区，指定只载入某些文件。导言区使用了 \includeonly 后，正文中不在其列表范围的 \include 命令不会起效： 

```latex
\includeonly{<filename1>,<filename2>,...}
```

>  `\includeonly` 命令用于导言区，其作用是指定能 `\include` 的文件范围
>  正文中 `\include` 的文件如果不在 `\includeonly` 中，就不会起效

需要注意的是，使用 \include 和 \input 命令载入的文件名最好不要加空格和特殊字符，也尽量避免使用中文名，否则很可能会出错。 

>  `\include, \input` 载入的文件最好使用 ASCII 字符组成文件名，防止出错

最后介绍一个实用的工具宏包 syntonly。加载这个宏包后，在导言区使用\syntaxonly 命令，可令 LaTeX 编译后不生成 DVI 或者 PDF 文档，只排查错误，编译速度会快不少：

```latex
\usepackage{syntonly}
\syntaxonly
```

如果想生成文档，则用 % 注释掉 \syntaxonly 命令即可。 

>  `syntonly` 宏包提供了 `\syntaxonly` 命令，在导言区使用时，它会令 LaTex 编译后不生产文档，只排查错误，可以用于快速排查错误

## 1.7  LaTeX 和 TeX 相关的术语和概念 
在本章的最后有必要澄清几个概念： 
- **引擎** 全称为排版引擎，是编译源代码并生成文档的程序，如 pdfTeX、XeTeX 等。有时也称为编译器。 
- **格式** 是定义了一组命令的代码集。LaTeX 就是最广泛应用的一个格式，高德纳本人还编写了一个简单的 plain TeX 格式，没有定义诸如 \documentclass 和\section 等等命令。 
- **编译命令** 是实际调用的、结合了引擎和格式的命令。如 `xelatex` 命令是结合 XeTeX 引擎和 LaTeX 格式的一个编译命令。 

常见的引擎、格式和编译命令的关系总结于表 1.2。 

LaTeX 编译命令和 LaTeX 格式往往容易混淆，在讨论关于 LaTeX 的时候需要明确。为避免混淆，本手册中的 LaTeX 一律指的是格式，编译命令则用等宽字体表 `latex` 表示。 

在此介绍一下几个编译命令的基本特点： 
- `latex` 虽然名为 `latex` 命令，底层调用的引擎其实是 pdfTeX。该命令生成 dvi (Device Inde-pendent) 格式的文档，用 `dvipdfmx` 命令可以将其转为 pdf。 
- `pdflatex` 底层调用的引擎也是 pdfTeX，可以直接生成 pdf 格式的文档。 
- `xelatex` 底层调用的引擎是 XeTex ，支持 UTF-8 编码和对 TrueType/OpenType 字体的调用。当前较为方便的中文排版解决方案基于 `xelatex`，详见 2.2 节。 
- `lualatex` 底层调用的引擎是 LuaTex ，这个引擎在 pdfTeX 引擎基础上发展而来，除了支持 UTF-8 编码和对 TrueType/OpenType 字体的调用外，还支持通过 Lua 语言扩展 TeX 的功能。`lualatex` 编译命令下的中文排版支持需要借助 `luatexja` 宏包。 

表 1.2: TeX 引擎、格式及其对应的编译命令。

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/6caaa3cd51678f8cf5756c206181f6afb065ffbcb8d853028ab7ccf185d41101.jpg) 

# 2 用 LaTeX 排版文字
文字是排版的基础。本章主要介绍如何在 LaTeX 中输入各种文字符号，包括标点符号、连字符、重音等，以及控制文字断行和断页的方式。本章简要介绍了在 LaTeX 中排版中文的方法。随着 LaTeX 和底层 TeX 引擎的发展，旧方式 (CCT、CJK 等) 日渐退出舞台，`xelatex` 和 `lualatex` 编译命令配合 `ctex` 宏包/文档类的方式成为当前的主流中文排版支持方式。 

## 2.1 语言文字和编码 
LaTeX 源代码为文本文件，而文本文件的一个至关重要的性质是它的编码。在此用尽量短的篇幅介绍一下。 

### 2.1.1 ASCII 编码 
计算机的基本存储单位是字节 (byte)，每个字节为八位 (8-bit)，范围用十六进制写作 `0x00-0xFF` 。ASCII (美国通用信息交换码) 使用 `0x00-0x7F` 对文字编码，也就是 7-bit，覆盖了基本的拉丁字母、数字和符号，以及一些不可打印的控制字符 (如换行符、制表符等)。 

>  每个字节的范围为 `0x00-0xFF`，ASCII 使用的范围是 `0x00-0x7F` ，也就是仅使用 7 个 bit，ASCII 覆盖了基本的字母、数字、符号以及一些控制字符

由于 TeX 最初设计用于排版以英文为主的西文文档，ASCII 编码完全够用，因而早期版本的 TeX 只支持 7-bit 和 ASCII 编码。排版扩展拉丁字符必须使用后文所述的各种符号和重音命令，如 Möbius 必须通过输入 M \\"obius 得到。 

>  早期 TeX 仅支持 7 bit 的 ASCII 编码，要排版拓展拉丁字符需要使用符号命令和重音命令

### 2.1.2 扩展编码 
在 ASCII 之后，各种语言文字都发展了自己的编码，比如西欧语言的 Latin-1、日本的 Shift-JIS、中国大陆的 GB 2312—80 和 GBK 等。它们中的绝大多数都向下兼容 ASCII，因此无论是在哪种编码下， TeX 以及 LaTeX 的命令和符号都能用。 

>  各种语言的编码对 ASCII 进行了拓展，中文的编码有 GB 2321——80 和 GBK 等

TeX 从 3.0 版开始支持 8-bit，能够处理编码在 `0x80-0xFF` 范围内的字符。西欧 (拉丁字母)、俄语系 (西里尔字母) 等语言文字的编码方案大都利用了 `0x80-0xFF` 这个范围，处理起来较为方便。使用 `latex` 或 `pdflatex` 编译命令时，对源代码的编码处理由 inputenc 宏包支持。比如将源代码保存为 Latin-1 编码，并在导言区调用 inputenc 宏包并指定 latin1 选项后，Möbius 这样的词语就可以直接通过 (用适当输入法) 输入 Möbius 得到了。 

>  TeX 3.0 开始支持 8-bit 编码，ASCII 没有使用到范围 `0x80-0xFF` 可以被映射到其他的字符 (取决于特定的编码方案是否这么做)
>  使用 `latex/paflatex` 命令时 (调用 pdfTeX 引擎编译)，文件源码的编码处理由 `inputenc` 宏包支持，例如，我们首先将源文件保存为 Latin-1 编码，然后在导言区使用 `inputenc` 宏包，指定 `latin1` 选项，pdfTeX 引擎就可以直接将 Latin-1 编码的文件正确编译

用于汉字的 GBK 等编码是多字节编码，ASCII 字符为一个字节，汉字等非 ASCII 字符为两个字节，使用 `latex` 或 `pdflatex` 编译命令时需要借助一些宏包进行较为复杂的判断和处理。早期排版中文须使用 CJK 宏包，它是一个用于处理中、日、韩等东亚语言文字编码和字体配置的宏包。但 CJK 宏包的使用非常不方便，目前已不再推荐直接使用。 

>  GBK 编码是多字节编码，ASCII 字符占一个字节 (`0x00-0x7F`)，汉字等非 ASCII 字符占两个字节
>  对于使用 GBK 编码的源文件，使用 `latex/pdflatex` 调用 pdfTeX 引擎编译时同样需要使用额外的宏包，例如早期的 CJK 宏包

### 2.1.3 UTF-8 编码 
Unicode 是一个多国字符的集合，覆盖了几乎全球范围内的语言文字。UTF-8 是 Unicode 的一套编码方案，一个字符由一个到四个字节编码，其中单字节字符的编码与 ASCII 编码兼容。现行版本的 LaTeX 使用 UTF-8 作为默认编码。将使用拉丁字母的文档保存为 UTF-8 编码后，可以用 `pdflatex` 直接编译，比如： 

>  Unicode 字符集合包含了大量的多语言字符，UTF-8 对 Unicode 字符集进行了编码，UTF-8 对一个字符采用 1 个到 4 个字节编码，ASCII 字符仍然采用单字节，与 ASCII 编码方式兼容
>  目前的 LaTeX 使用 UTF-8 编码作为默认编码方式，因此我们将源文件保存为 UTF-8 编码方式后，可以直接用 `pdflatex` 编译

```latex
\documentclass{artical}
\begin{document}
Français Português Español Føroyskt
\end{document}
```

但是非拉丁字母仍然无法直接在 LaTeX 中使用，如西里尔字母 (俄文)、希腊字母、阿拉伯字母以及东亚文字等。 

>  但 LaTeX 对 UTF-8 的支持不完全，例如中文字符有被 UTF-8 编码，但包含中文字符的 UTF-8 源码文件无法直接被编译 (本质上是 pdfTeX 引擎对 UTF-8 的支持不完全)

较为现代的 TeX 引擎，如 XeTeX 和 LuaTeX，它们均原生支持 UTF-8 编码。使用 `xelatex` 和 `lualatex` 排版时，将源代码保存为 UTF-8 编码，并借助 fontspec 宏包 (见 5.1.5 小节) 调用适当的字体，原则上就可以在源代码中输入任意语言的文字。注意此时不再适用 inputenc 宏包。但一些复杂语言 (如印地语、阿拉伯语等) 的排版需要考虑到断词规则、文字方向、标点禁则等诸多细节，因此需要更多的宏包支持，如 babel、polyglossia 等，此处不再涉及。 

>  更现代的 TeX 引擎例如 XeTeX 和 LuaTeX 完全支持 UTF-8 编码
>  我们将包含 Unicode 字符的源文件保存为 UTF-8 编码，并借助 `fontspec` 宏包调节字体，使用 `xelatex/lualatex` 编译即可

## 2.2 排版中文 
用 LaTeX 排版中文需要解决两方面问题，一方面是对中文字体的支持，另一方面是对中文排版中的一些细节的处理，包括在汉字之间控制断行、标点符号的禁则 (如句号、逗号不允许出现在行首)、中英文之间插入间距等。CJK 宏包对中文字体的支持比较麻烦，已经不再推荐直接使用。XeTeX 和 LuaTeX 除了直接支持 UTF-8 编码外，还支持直接调用 TrueType/OpenType 格式的字体。 `xeCJK` 及 `luaTeXja` 宏包则在此基础上封装了对汉字排版细节的处理功能。 

>  XeTeX 和 LuaTeX 支持排版 UTF-8 编码的 Unicode 字符，但是汉字排版仍需要进一步的细节处理，包括对中文字体的支持和一些排版细节
>  XeTeX 和 LuaTex 直接支持直接调用 TrueType/OpenType 格式的字体，`xeCJK` 和 `luatexja` 宏包封装了这一功能，还进一步封装了对中文排版细节的处理功能

`ctex` 宏包和文档类进一步封装了 `CJK`、`xeCJK`、`luatexja` 等宏包，使得用户在排版中文时不用再考虑排版引擎等细节。`ctex` 宏包本身用于配合各种文档类排版中文，而 `ctex` 文档类对 LaTeX 的标准文档类进行了封装，对一些排版根据中文排版习惯做了调整，包括 `ctexart`、`ctexrep`、`ctexbook` 等。`ctex` 宏包和文档类能够识别操作系统和 TeX 发行版中安装的中文字体，因此基本无需额外配置即可排版中文文档。下面举一个使用 `ctex` 文档类排版中文的最简例子： 

>  `ctex` 宏包进一步封装了这些宏包，使得用户不需要考虑具体的排版引擎
>  `ctex` 文档类包括 `ctexart\ctexrep\ctexbook` 等，它封装了标准文档类，对中文排版进行了优化
>  `ctex` 宏包可以和其他文档类配合排版中文，当然最好和 `ctex` 文档类配合

```latex
\documentclass{ctexart}
\begin{document}
在 \LaTeX{}中排版中文。汉字和 English 单词混排，通常不需要在中英文之间添加额外的空格。
当然，为了代码的可读性，加上汉字和 English 之间的空格也无妨。
汉字换行时不会引入多余的空格。 
\end{document} 
```

注意源代码须保存为 UTF-8 编码，并使用 `xelatex` 或 `lualatex` 命令编译。虽然 `ctex` 宏包和文档类保留了对 GBK 编码以及 `latex` 和 `pdflatex` 编译命令的兼容，但我们并不推荐这样做。 

## 2.3 LaTeX 中的字符 
### 2.3.1 空格和分段 
LaTeX 源代码中，空格键和 Tab 键输入的空白字符视为“空格”。连续的若干个空白字符视为一个空格。一行开头的空格忽略不计。行末的换行符视为一个空格；但连续两个换行符，也就是空行，会将文字分段。多个空行被视为一个空行。也可以在行末使用 \par 命令分段。 

>  LaTeX 源码忽略行首的空白字符，连续的空白字符视为一个
>  连续的两个换行符 (一个空行) 会将文字分段，多个空行视为一个空行
>  `\par` 命令也可以用于分段

```latex
several spaces     equal one.
    Front space are ignored.
    
A empty line stats a new paragraph.\par
A \verb|\par| does the same
```

### 2.3.2 注释 
LaTex 用 % 字符作为注释。在这个字符之后直到行末，所有的字符都被忽略，行末的换行符也不引入空格。 

>  `%` 为注释字符，注释字符到行末的所有字符都被忽略，包括行末的换行符

```latex
This is an % short comment
% ---
% Long and organized
% comments
% ---
example: Comments do not bre%
ak a work
```

### 2.3.3 特殊字符 
以下字符在 LaTex 里有特殊用途，如 % 表示注释， \$ 、^、_ 等用于排版数学公式，& 用于排版表格，等等。直接输入这些字符得不到对应的符号，还往往会出错： 

>  LaTeX 的保留字符如下

```
# $ % & { } _ ^  ~  \ 
```

如果想要输入以上符号，需要使用以下带反斜线的形式输入，类似编程语言里的“转义”符号： 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/5d543d1c7346e23c4685639114376a6f23eea082fbb966eaa1dab68a2db3780a.jpg) 

这些“转义”符号事实上是一些 LaTeX 命令。其中 `\^` 和 `\~` 两个命令需要一个参数，加一对花括号的写法相当于提供了空的参数，否则它们可能会将后面的字符作为参数，形成重音效果 (详见 2.3.6 节)。 `\\` 被直接定义成了手动换行的命令，输入反斜线就需要用 `\textbackslash`。 

>  要输入保留字符，需要进行“转义”

```latex
\# \$ \% \& \{ \} \_
\^{} \~{} \textbackslash
```

>  这些“转义”符号本质上也是 LaTeX 命令 (`\` + 单个非字母符号)
>  其中 `\^` 和 `\~` 命令还需要一个必选参数，如果使用空的 `{}` ，表示提供空的参数，否则它们会将后面的字符作为参数
>  `\\` 被定义为手动换行的命令，命令 `\textbackslash` 用于输入反斜线

### 2.3.4 连字 
西文排版中经常会出现连字 (ligatures)，常见的有 ff/fi/fl/ffi/ffl。 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/7318c3ef7aa6e8af5f6c808cc53088a7575fbc65165451f7eead02a82ab62fea.jpg) 

```latex
It's difficult to find \ldots \\
It's dif{}f{}icult to f{}ind \ldots
```

>  英文排版时，对于连续的 `ff/fi/fl/ffi/ffl` 会出现连字
>  在字母之间添加空的 `{}` 可以防止连字

### 2.3.5 标点符号 
中文的标点符号 (绝大多数为非 ASCII 字符) 使用中文输入法输入即可，一般不需要过多留意。而输入西文标点符号时，有不少地方需要留意。 

**引号** 
LaTeX 中单引号‘和’分别用 \` 和' 输入；双引号“ 和” 分别用 \` \` 和'' 输入 (" 可以输入后双引号，但没有直接输入前双引号的字符，习惯上用'' 输入以和 \` \` 更好地对应)。 

>  英文中的左单引号 `'` 使用 \`  输入，右单引号 `'`  使用 ` ' ` 输入 
>  左双引号 `"` 使用 \`\` 输入，右双引号 `"` 使用 ` '' ` 或者 `"` 输入

```latex
``Please press the `x' key. ''
```

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/139eb8fe990ac549fee5cce86a5e85684e15caa8c0fa5d3fb045dfc194fde063.jpg) 

中文的引号‘’和“ ”与西文的引号实际上是同一组符号，但由于中西文通常用不同的字体显示，它们的具体形状和宽度可能有所不同。在使用 `ctex` 宏包或文档类的情况下，中文引号可以通过输入法直接输入。 

>  中文的引号和英文的引号在 Unicode 中使用同一码位，因为实际上是同一组符号
>  其具体的形状和宽度的不同是因为中西文的字体不同
>  使用 `ctex` 宏包或文档类时，直接输入中文引号即可正确编译

**连字号和破折号** 
LaTeX 中有三种长度的“横线”可用：连字号 (hyphen)、短破折号 (en-dash) 和长破折号 (em-dash)。它们分别有不同的用途：连字号 - 用来组成复合词；短破折号 – 用来连接数字表示范围；长破折号 — 用来连接单词，语义上类似中文的破折号。 

```latex
daughter-in-law, X-rated\\
pages 13--67\\
yes---or no?
```

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/e9dd1664a6afc329e03c0e2023bebf212f04b4ae381ca827d378784bd171da89.jpg) 

>  连字号 `-` 最短，用于组成复合词
>  短破折号 `--` 稍长，用于连接数字表示范围
>  长破折号 `---` 最长，用来连接单词，语义上类似中文的破折号

**省略号** 
LaTeX 提供了 \ldots 命令表示省略号，相对于直接输入三个点的方式更为合理。 \dots 与 \ldots 命令等效。 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/2d2e8130a01f4c7a4c82676956b4f86f3e8a7f3af9cd6f6787a13adafb2a7e9f.jpg) 

```latex
one, two, three, \ldots{} one hundred
```

>  `\ldots` 命令和 `\dots` 命令等效

**波浪号** 
我们在 2.3.3 小节中了解了 \ \~ 命令，它可以用来输入波浪号，但位置靠顶端 ( \ \~ 命令主要用作重音，参考下一小节)。西文中较少将波浪号作为标点符号使用，在中文环境中一般直接使用全角波浪号 (～)。 

>  中文环境要使用波浪号直接输入全角波浪号 ~ 即可
>  西文环境中波浪号命令 `\~` 一般用于标注重音

### 2.3.6 拉丁文扩展与重音 
LaTeX 支持用命令输入西欧语言中使用的各种拉丁文扩展字符，主要为带重音的字母： 

```latex
H\^otel,na\"i ve,\'el\`eve,\\
sm\o rrebr\o d, !`Se\ notrita!,\\
Sch\"onbruner Schlo\ss{}
Stra\ss e
```

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/a16a713490fdaba7aa48549b84f0a4c22d9abbcc2d708464325c45d59d39e4d4.jpg) 

更多可用的符号和重音见表 2.1。注意与 4.3.7 小节的数学重音区分开来。 

表 2.1: LaTeX 文本中的重音和特殊字符
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/19ee9078281eb44b065e4b6e5ebbd31da6fe183d23156b5d5656aaa345a71cca.jpg) 

前四行实际上都是带一个参数的命令。 `\^o`  也可以写作 `\^{o}`，以此类推。 

>  LaTex 提供了命令用于输入一些拓展拉丁字符，包括带重音的字母

### 2.3.7 其它符号 
LaTeX 预定义了其它一些文本模式的符号，部分符号可参考表 4.4。 

```latex
\P{} \S{} \dag{} \ddag{}
\copyright{} \pounds{}

\textasteriskcentered
\textperiodcenterd
\textbullet

\textregistered{} \texttrademark
```

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/2d78cd69b44419eb88897e723957739b87f89aed9b486342f80413b4a69f8955.jpg) 

更多的符号多由特定的宏包支持。参考文献[14] 搜集了所有在 TeX 发行版中可用的符号，使用时要留意每个符号所依赖的宏包。 

>  LaTex 也提供了输入其他一些符号的命令，如上所示

### 2.3.8 LaTeX 标志 
我们见到的所有错落有致的 LaTeX 标志都是由以下命令输入的： 

```latex
\TeX
\LaTex
\LaTeXe
```

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/09738b0b7ef46a51031b1fb06599269de9999ec843ee44b462813cfd408d9012.jpg) 

## 2.4 断行和断页 
LaTeX 将文字段落在合适的位置进行断行，尽可能做到每行的疏密程度匀称，单词间距不会过宽或过窄。文字段落和公式、图表等内容从上到下顺序排布，并在合适的位置断页，分割成匀称的页面。在绝大多数时候，我们无需自己操心断行和断页。但偶尔会遇到需要手工调整的地方。 

>  LaTeX 会帮助我们自动断行断页，保持单词间距和页面内容匀称
>  大多数时候不需要我们手工调整

### 2.4.1 单词间距 
在西文排版实践中，断行的位置优先选取在两个单词之间，也就是在源代码中输入的“空格”。“空格”本身通常生成一个间距，它会根据行宽和上下文自动调整，文字密一些的地方，单词间距就略窄，反之略宽。 

>  西文排版中，一般优先在单词之间 (的空格处) 断行 (中文排版要实现汉字之间断航需要借助宏包和特殊排版引擎的支持)
>  在一行内，单词之间的空格本身会被 LaTeX 用于生成间距，具体的间距会根据行宽和上下文自动调整，例如文字密就间距窄
>  注意这里的“空格”都是指空白字符，包括了换行符等

文字在单词间的“空格”处断行时，“空格”生成的间距随之舍去。我们可以使用字符 \~ 输入一个不会断行的空格 (高德纳称之为 tie，“带子”)，通常用在英文人名、图表名称等上下文环境： 

>  空格被断行时，它不会生成间距
>  保留字符 `~` 表示一个不会断行的空格

```latex
Fig.~2d \\
Donald~E. Knuth
```

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/0a67d704b0ba0f6ae346e3819b22b459ccfa9929bd9181e830548590915e45af.jpg) 

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/3e35d883cdde15383166739d84f3c78fc1d8e8a5c056024985cc1c6a91b3b016.jpg) 

### 2.4.2 手动断行和断页 
如果我们确实需要手动断行，可使用如下命令： 

```latex
\\[<length>]    \\*[<length>]
\newline
```

它们有两点区别：一是 `\\` 可以带可选参数 ⟨length⟩，用于在断行处向下增加垂直间距 (见 5.3.5 小节)，而 \newline 不带可选参数；二是 `\\` 也在表格、公式等地方用于换行，而 \newline 只用于文本段落中。带星号的 `\\` 表示禁止在断行处分页。 

>  命令 `\\[<length>], \\*[<length>], \newline`  用于手动断行
>  `\\, \\*` 命令的可选参数用于在断行处向下增加垂直间距，`\newline` 不能带参数
>  `\\, \\*` 可以在表格和公式环境中引入断行，`\newline` 只能用于文本段落中
>  `\\*` 表示禁止在断行处分页

```latex
另外需要注意的是，使用 \verb|\\| 
断行命令 \\ 不会令内容另起一段，
而是在段落中直接开始新的一行。 
```

>  `\\` 不会另起一段，仅仅是开启新的一行，段落不变

断页的命令有两个： 

```latex
\newpage 
\clearpage 
```

通常情况下两个命令都起到另起一页的作用，区别在于：第一，在双栏排版模式中 \newpage 起到另起一栏的作用， \clearpage 则能够另起一页；第二，在涉及浮动体的排版上行为不同。后文的 3.9 节以及 5.4.3 小节会更详细地介绍相关内容。 

>  `\newpage, \clearpage` 用于断页
>  `\newpage` 在双栏排版中功能转为另起一栏，`\clearpage` 另起一页

有时候我们不满足于 LaTeX 默认的断行和断页位置，需要进行微调，可以用以下命令告诉 LaTeX 哪些地方适合断行或断页，哪些地方不适合： 

```latex
\linebreak[<n>]    \nolinebreak[<n>]
\pagebreak[<n>]    \nopagebreak[<n>]
```

以上命令都带一个可选参数，用数字 `<n>` 代表适合/不适合的程度，取值范围为 0–4，不带可选参数时，缺省为 4。比如 `\linebreak` 或者 `\linebreak[4]` 意味着此处需要强行断行； `\nopagebreak` 或 `\nopagebreak[4]` 意味着禁止在此处断页。 

>  `\linebreak, \pagebreak` 也可以用于断行断页，其可选参数表示强制程度，默认为 4，即最大值，表示强制断行断页
>  `\nolinebreak, \nopagebreak` 作用相反，默认为强制不断行断页

以上命令适合给出优先考虑断行断页/禁止断行断页的位置，但不适合直接拿来断行或断页，使用 \newline 或 \newpage 等命令是更好的选择。因为 \newline 和 \newpage 会在断行/断页位置填充适当的间距，但 \linebreak 和 \pagebreak 不能，使用这些命令强行断行/断页可能会制造出糟糕的排版效果，并导致 LaTeX 报 Underfull  \hbox 等警告。 

>  `\newline, \newpage` 等命令在断行断页处填充适当间距，使得上一行的字词间距能保持得较为合理

```latex
使用 \verb|\newline| 断行的效果
\newline
与使用 \verb|\linebreak| 断行的效果
\linebreak
进行对比
```

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/a453b1cd780e5a770923350dde05973d895f6d158c1ffc546b868fcbfd006d4d.jpg) 

### 2.4.3 断词 
如果 LaTeX 遇到了很长的英文单词，仅在单词之间的“空格”处断行无法生成疏密程度匀称的段落时，就会考虑从单词中间断开。对于绝大多数单词， LaTeX 能够找到合适的断词位置，在断开的行尾加上连字符 -。 

>  LaTeX 遇到很长的单词会考虑在单词中间断开，断开的行尾加上连字符 `-`

如果一些单词没能自动断词，我们可以在单词内手动使用 \- 命令指定断词的位置： 

```latex
I think this is: su\-per\-cal\-%
i\-frag\-i\-lis\-tic\-ex\-pi\-%
al\-i\-do\-cious.
```

>  `\-` 命令用于手动在单词内部指定断词的位置

# 3 文档元素 
在知道了如何输入文字后，我们将在本章了解一个结构化的文档所依赖的各种元素——章节、目录、列表、图表、交叉引用、脚注等等。 
## 3.1 章节和目录 
### 3.1.1 章节标题 
一篇结构化的、条理清晰文档一定是层次分明的，通过不同的命令分割为章、节、小节。三个标准文档类 article、report 和 book提供了划分章节的命令： 

```latex
\chapter{<title>}  \section{<title>}  \subsection{<title>}
\subsubsection{<title>}  \paragraph{<title>}  \subparagraph{<title>}
```

其中 \chapter 只在 report 和 book 文档类有定义。这些命令生成章节标题，并能够自动编号。除此之外 LaTeX 还提供了 \part 命令，用来将整个文档分割为大的分块，但不影响 \chapter 或 \section 等的编号。 

>  标准文档类 (article, report, book) 中划分章节的命令为 `\chapter`，划分节和小节的命令为 `\section, \subsection, \subsubsection` ，划分段落的命令为 `\paragraph, \subparagraph` ，这些命令都有一个必选参数 `<title>`
>  其中划分章节的命令 `\chapter` 只有在 report 和 book 中有定义
>  这些命令都会自动编号
>  `\part` 命令可以将文档分割为大的分块，但不影响以上命令的编号
>  注意这些划分章、节、小节的命令仅在标准文档类中有定义，其他文档类中不一定

上述命令除了生成带编号的标题之外，还向目录中添加条目，并影响页眉页脚的内容 (详见 5.5 节)。每个命令有两种变体： 

- 带可选参数的变体：  `\section[<short title>]{<title>}` 标题使用 ⟨title⟩ 参数，在目录和页眉页脚中使用 ⟨short title⟩ 参数；
- 带星号的变体：  `\section*{<title>}` 标题不带编号，也不生成目录项和页眉页脚。 

>  这些划分章和节的命令除了生成带编号的标题以外，还会将章/节名作为条目添加到目录中，同时也会影响页眉页脚的内容
>  这些命令各自存在两种变体：
>  - 带可选参数 `[<short title>]` 的变体，`<short title>` 的内容将用作目录中和页眉页脚中
>  - 带星号的变体，例如 `\section*{<title>}` ，带星号时，标题不会编号，并且也不生成目录条目和页眉页脚

较低层次如  \paragraph 和  \subparagraph 即使不用带星号的变体，生成的标题默认也不带编号，事实上，除 \part 外： 

- article 文档类带编号的层级为 \section、 \subsection、 \subsubsection 三级；
- report 和 book 文档类带编号的层级为 \chapter、 \section、 \subsection 三级。 

>  `\paragraph, \subparagraph` 即使不带星号，生成的标题也没有编号
>  article 类带编号的层级为 `\[sub][sub]section`
>  report 和 book 类带编号的层级为 `\chapter, \[sub]section`

对此的详细解释和调整方法见 8.3.3 小节。

LaTeX 及标准文档类并未提供为 \section 等章节命令定制格式的功能，这一功能由 titlesec 宏包提供。详情请参考宏包的帮助文档。 

### 3.1.2 目录 
在 LaTeX 中生成目录非常容易，只需在合适的地方使用命令： 

```latex
\tableofcontents
```

这个命令会生成单独的一章 (report/book) 或一节 (article)，标题默认为“Contents”，可通过 8.4 节给出的方法定制标题。 \tableofcontents 生成的章节默认不写入目录 ( \section\* 或 \chapter\*)，可使用 tocbibind 等宏包修改设置。 

>  `\tableofcontents` 用于生成目录，它在 report/book 中会单独生成一个 chapter ，在 article 中会单独生成一个 section，标题为 “Contents“
>  目录中不会包含目录章节 (等价于 `\chapter*{Contents}, \section*{Contents}`)

正确生成目录项，一般需要编译两次源代码。 

有时我们使用了 \chapter\* 或 \section\*  这样不生成目录项的章节标题命令，而又想手动生成该章节的目录项，可以在标题命令后面使用： 

```latex
\addcontentsline{toc}{<level>}{<title>}
```

其中 `<level>`  为章节层次 chapter 或 section 等，`<title>`  为出现于目录项的章节标题。

>  `\addcontentsline{toc}{<level>}{<title>}` 可以用于为 `\chapter*, \section*` 的无目录项的章节标题命令手动生成目录项

titletoc、tocloft 等宏包提供了具体定制目录项格式的功能，详情请参考宏包的帮助文档。 

### 3.1.3 文档结构的划分 
所有标准文档类都提供了一个 \appendix 命令将正文和附录分开，使用 \appendix 后，最高一级章节改为使用拉丁字母编号，从 A 开始。

>  标准文档类都使用 `\appendix` 划分正文和附录
>  附录的最高一级章节编号为拉丁字母

book 文档类还提供了前言、正文、后记结构的划分命令：

-  `\frontmatter` 前言部分，页码使用小写罗马数字；其后的 \chapter 不编号。
-  `\mainmatter` 正文部分，页码使用阿拉伯数字，从 1 开始计数；其后的章节编号正常。
-  `\backmatter`  后记部分，页码格式不变，继续正常计数；其后的 \chapter 不编号。 

>  book 文档类还有 `\frontmatter, \mainmatter, \backmatter` 命令用于划分前言、正文、后记

以上三个命令还可和 \appendix 命令结合，生成有前言、正文、附录、后记四部分的文档。源代码 3.1 结合 1.6 节的 \include 命令和其它一些命令示意了一份完整的文档结构。 

```latex
\documentclass{book} 

% 导言区，加载宏包和各项设置，包括参考文献、索引等
\usepackage{makeidx}% 调用 makeidx 宏包，用来处理索引
\makeindex % 开启索引的收集
\bibliography style{plain}% 指定参考文献样式为 plain 

\begin{document} 

\frontmatter% 前言部分
\maketitle% 标题页
\include{preface} % 前言章节 preface.TeX
\tableofcontents 

\mainmatter% 正文部分
\include{chapter1} % 第一章 chapter1.TeX
\include{chapter2} % 第二章 chapter2.TeX 
...
\appendix% 附录
\include{appendixA} % 附录 A appendixA.TeX
 ... 
 
\backmatter% 后记部分
\include{epilogue} % 后记 epilogue.TeX
\bibliography{books} % 利用 BibTeX 工具从数据库文件 books. bib 生成参考文献
\printindex% 利用 makeindex 工具生成索引 
\end{document} 
```

源代码 3.1: book 文档类的文档结构示例。 

## 3.2 标题页 
LaTeX 支持生成简单的标题页。首先需要给定标题和作者等信息： 

```latex
\title{<title>}    \author{<author>}    \date{<date>}
```

其中前两个命令是必须的 (不用 \title 会报错；不用 \author 会警告)， \date 命令可选。LaTeX 还提供了一个 \today 命令自动生成当前日期， \date 默认使用 \today。在 \title、 \author 等命令内可以使用 \thanks 命令生成标题页的脚注，用 \and 隔开多个人名。 

>  要生成标题页，首先需要通过 `\title, \author, \date` 给定信息
>  其中 `\date` 可选，`\title, \author` 必须
>  `\date` 默认使用 `\today` 命令生成当前日期，即 `\date{\today}`
>  `\title, \author` 中可以用 `\thanks` 在标题页生成脚注，用 `\and` 隔开人名

在信息给定后，就可以使用 \maketitle 命令生成一个简单的标题页了。源代码 3.2 给出了一个标题页的示例和大致效果。article 文档类的标题默认不单独成页，而 report 和 book 默认单独成页。可在 \documentclass 命令调用文档类时指定 titlepage 或 no title page 选项以修改默认的行为。 

>  使用以上命令给定信息后，调用 `\maketitle` 命令即可生成标题页

```latex
\title{Test title}
\author{ Mary \thanks{E-mail: *****@*****.com} 
    \and Ted \thanks{Corresponding author}  
    \and Louis}
\date{\today} 
```

源代码 3.2: LaTeX 默认的标题页示例和效果。

LaTeX 标准类还提供了一个简单的 titlepage 环境，生成不带页眉页脚的一页。用户可以在这个环境中使用各种排版元素自由发挥，生成自定义的标题页以替代 \maketitle 命令。甚至可以利用 titlepage 环境重新定义 \maketitle： 

>  LaTeX 标准类提供 `titlepage` 环境生成不带页眉页脚的一页
>  可以用该环境替代 `\maketitle` 或用该环境重新定义 `\maketitle`


```latex
\renewcommand{\maketitle}{\begin{titlepage} 
# ... % 用户自定义命令 
\end{titlepage}} 
``` 

事实上，为标准文档类指定了 titlepage 选项以后，使用 \maketitle 命令生成的标题页就是一个 titlepage 环境。 

>  如果使用标准文档类指定了 `titlepage` 选项后 (book 和 report 默认即 `titlepage` ，article 默认为 `notitlepage` )，`\maketitle` 命令生成的标题页就是一个 `\begin{titlepage}\end{titlepage}` 环境

以上是 LaTeX 标准文档类的标题页相关命令用法。在各种文档模板中经常有自定义的标题页，有可能需要除了 \title 和 \author 以外的命令给定信息，用法也可能与标准文档类的不一致 (甚至有些模板可能没有定义 titlepage 等环境)。使用文档模板前一定要仔细阅读文档模板的帮助文档。 

## 3.3 交叉引用 
交叉引用是 LaTeX 强大的自动排版功能的体现之一。在能够被交叉引用的地方，如章节、公式、图表、定理等位置使用 \label 命令： 

```latex
\label{<label-name>}
```

之后可以在别处使用 \ref 或 \pageref 命令，分别生成交叉引用的编号和页码： 

```latex
\ref{<label-name>}    \pageref{<label-name>}
```

```latex
A reference to this subsection
\label{sec:this} looks like:
``see section~\ref{sec:this} on
page~\pageref{sec:this}.''
```

为了生成正确的交叉引用，一般也需要多次编译源代码。 

>  在需要标记的地方使用 `\label{<label-name>}` 标记，之后可以使用 `ref{<label-name>}` 生成 `<label-name>` 所处的位置的信息 (具体看 `\label` 标记的是什么)
>  或者 `\pageref{<label-name>}` 生成 `<label-name>` 所处的页码
>  本例中的 `~` 用于生成不断行的空格

 \label 命令可用于记录各种类型的交叉引用，使用位置分别为： 
- 章节标题
    在章节标题命令 \section 等之后紧接着使用。 
- 行间公式
    单行公式在公式内任意位置使用；多行公式在每一行公式的任意位置使用。 
- 有序列表
    在 enumerate 环境的每个 \item 命令之后、下一个 \item 命令之前任意位置使用。 
- 图表标题
    在图表标题命令 \caption 之后紧接着使用。 
- 定理环境
    在定理环境内部任意位置使用。 

>  `\label` 可以使用的位置包括
>  章节标题 - 紧接着 `\section` 等命令之后
>  公式 - 单行公式在公式中任意位置，多行公式在每一行公式的任意位置
>  有序列表 - `enumerate` 环境中，每个 `\item` 命令之后，下一个 `\item` 命令之前的任意位置
>  图标标题 - 紧接着 `\caption` 命令之后
>  定理环境 - 定理环境内部任意位置

在使用不记编号的命令形式 ( \section\*、 \caption\* 、带可选参数的 \item 命令等) 时不要使用 \label 命令，否则生成的引用编号不正确。 

## 3.4 脚注和边注 
使用 \footnote 命令可以在页面底部生成一个脚注： 

```latex
\footnote{<footnote>}
```

>  `\footnote{<footnote>}` 用于生成脚注

假如我们输入以下文字和命令： 

```latex
“天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。” \footnote{出自《千字文》。}
```

在正文中则为：“天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。” 

有些情况下 (比如在表格环境、各种盒子内) 使用 \footnote 并不能正确生成脚注。我们可以分两步进行，先使用 \footnotemark 为脚注计数，再在合适的位置用 \footnotetext 生成脚注。比如： 

```latex
\begin{tabular}{l} 
\hline 
“天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。” \footnotemark  \\
\hline 
\footnotetext{表格里的名句出自《千字文》。} 
```

效果为：“天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。”

>  在表格环境和各种盒子内使用 `\footnote` 不能正确生成脚注，可以先用 `\footnotemark` 为脚注计数，然后在合适的位置用 `\footnotetext` 生成脚注

使用 \marginpar 命令可在边栏位置生成边注： 

```latex
\marginpar[<left-margin>]{<right-margin>}
```

如果只给定了 `<right-margin>` ，那么边注在奇偶数页文字相同；如果同时给定了 `<left-margin>` ，则偶数页使用 `<left-margin>`  的文字。 
例如以下代码： 

```latex
 \marginpar{\footnotesize 边注较窄，不要写过多文字，最好设置较小的字号。} 
```
 
其效果见边栏。 

>  `\marginpar[<left-margin>]{<right-margin>}` 用于生成边注
>  - 


## 3.5 特殊环境 
### 3.5.1 列表 
LaTeX 提供了基本的有序和无序列表环境 enumerate 和 itemize，两者的用法很类似，都用 \item 标明每个列表项。enumerate 环境会自动对列表项编号。 

```latex
\begin{enumerate}
\item ...
\end{enumerate}
```

>  `enumerate` 和 `itemize` 环境表示有序和无序列表环境
>  在环境中，`\item` 命令用于表明列表项
>  `enumerate` 环境会自动为列表项编号

其中 \item 可带一个可选参数，将有序列表的计数或者无序列表的符号替换成自定义的符号。
>  `\item` 命令可以带可选参数，将列表的计数符号替换为自定义的符号

列表可以嵌套使用，最多嵌套四层。 
>  列表环境可以嵌套

```latex
\begin{enumerate}
    \item An item.
    \begin{enumerate}
        \item A nested item.\label{itref}
        \item[*] A starred item.
    \end{enumerate}
    \item Reference(\ref{itref}).
\end{enumerate}
\begin{itemize}
    \item An item.
    \begin{itemize}
        \item A nested item.
        \item[+] A `plus' item.
        \item Another item.
    \end{itemsize}
    \item Go back to upper level
\end{itemize}
```

![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/29894cdaf86eb1309b050d3dc5d66fb96429b4e29df874c231ddf268f5d86244.jpg) 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/2f8994d087635fe68ff2a18b8bca9d5d6d6780aed70766fbfa2b3661bd27b43f.jpg) 

关键字环境 description 的用法与以上两者类似，不同的是 \item 后的可选参数用来写关键字，以粗体显示，一般是必填的：

```latex
\begin{description}
\item[<item title>]
\end{description}
```

```latex
\begin{description}
    \item[Enumerate] Numbered list.
    \teim[Itemize] Non-numbered list.
\end{description}
```

各级无序列表的符号由命令 \labelitemi 到 \label item iv 定义，可以简单地重新定义它 \renew command{ \labelitemi}{ \ddag}
  \renew command{ \label item ii}{ \dag}
  \begin{itemize}  \item First item  \begin{itemize}  \item Subitem  \item Subitem  \end{itemize}  \item Second item
  \end{itemize} 
‡ First item † Subitem † Subitem
 ‡ Second item 有序列表的符号由命令 \labelenumi 到 \label e num iv 定义，重新定义这些命令需要用到 8.3 节的计数器相关命令： 
 \renew command{ \labelenumi}% { \Alph{enumi}>}
 \begin{enumerate}  \item First item  \item Second item
  \end{enumerate} 
$ \mathrm{A} \!>$ First item
 $ \mathrm{B} \!>$ Second item 默认的列表间距比较宽， $ \mathrm{emptyset}$ 本身也未提供方便的定制功能，可用 enumitem 宏包定制各种列表间距。enumitem 宏包还提供了对列表标签、引用等的定制。有兴趣的读者可参考其帮助文档。 
 
### 3.5.2 对齐环境 
center、flushleft 和 flushright 环境分别用于生成居中、左对齐和右对齐的文本环境。 
 \begin{center} …  \end{center}  \begin{flushleft} …  \end{flushleft}  \begin{flushright} …  \end{flushright} 
 \begin{center} Centered TeXt using a
  \verb|center| environment.
  \end{center}
  \begin{flushleft} Left-aligned TeXt using a
  \verb|flushleft| environment.
  \end{flushleft}
  \begin{flushright} Right-aligned TeXt using a
  \verb|flushright| environment.
  \end{flushright} 
Centered TeXt using a center environment. Left-aligned TeXt using a flushleft environment. 
Right-aligned TeXt using a flushright 除此之外，还可以用以下命令直接改变文字的对齐方式： 
 \centering  \ragged right  \raggedleft 
 \centering Centered TeXt paragraph.  \ragged right Left-aligned TeXt paragraph.  \raggedleft Right-aligned TeXt paragraph. 
Centered TeXt paragraph. Left-aligned TeXt paragraph. Right-aligned TeXt paragraph. 
三个命令和对应的环境经常被误用，有直接用所谓 \flushleft 命令或者 ragged right 环境的，都是不甚严格的用法 (即使它们可能有效)。有一点可以将两者区分开来：center 等环境会在上下文产生一个额外间距，而 \centering 等命令不产生，只是改变对齐方式。比如在浮动体环境 table 或 figure 内实现居中对齐，用 \centering 命令即可，没必要再用 center 环境。 
### 3.5.3 引用环境 
LaTeX 提供了两种引用的环境：quote 用于引用较短的文字，首行不缩进；quotation 用于引用若干段文字，首行缩进。引用环境较一般文字有额外的左右缩进。 
Francis Bacon says:  \begin{quote} Knowledge is power.  \end{quote} 
Francis Bacon says: Knowledge is power. 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/698fcd43db7b71c4a1f23ecbb581a32d640c257e2c37b2def7216912b35ecc3e.jpg) 
verse 用于排版诗歌，与 quotation 恰好相反，verse 是首行悬挂缩进的。 
Rabindranath Tagore's short poem:  \begin{verse} Beauty is truth's smile when she beholds her own face in a perfect mirror.  \end{verse} 
Rabindranath Tagore’s short poem: Beauty is truth’s smile when she beholds her own face in a perfect mirror. 
# 3.5.4 摘要环境 
摘要环境 abstract 默认只在标准文档类中的 article 和 report 文档类可用，一般用于紧跟 \maketitle 命令之后介绍文档的摘要。如果文档类指定了 titlepage 选项，则单独成页；反之，单栏排版时相当于一个居中的小标题加一个 quotation 环境，双栏排版时相当于 \section \* 定义的一节。 
# 3.5.5 代码环境 
有时我们需要将一段代码原样转义输出，这就要用到代码环境 verbatim，它以等宽字体排版代码，回车和空格也分别起到换行和空位的作用；带星号的版本更进一步将空格显示成“␣”。 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/9408d574f66f3efa1a3b5d8d1b203c637fae02012d80f617e10b3fbf4c8f7769.jpg) 
要排版简短的代码或关键字，可使用 \verb 命令： 
 \verb ⟨ delim ⟩⟨ code ⟩⟨ delim ⟩ 
⟨delim⟩标明代码的分界位置，前后必须一致，除字母、空格或星号外，可任意选择使得不与代码本身冲突，习惯上使用| 符号。 
同 verbatim 环境， \verb 后也可以带一个星号，以显示空格： 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/37d04ff55f2b5daefdf9eb7c9a661a6b906486ee490a3b1189a423e564b796d1.jpg) 
 \verb 命令对符号的处理比较复杂，一般不能用在其它命令的参数里，否则多半会出错。verbatim 宏包优化了 verbatim 环境的内部命令，并提供了 \verbatim input 命令用来直接读入文件生成代码环境。fancyvrb 宏包提供了可定制格式的 Verbatim 环境；listings 宏包更进一步，可生成关键字高亮的代码环境，支持各种程序设计语言的语法和关键字。详情请参考各自的帮助文档。 
# 3.6 表格 
LaTeX 里排版表格不如 Word 等所见即所得的工具简便和自由，不过对于不太复杂的表格来讲，完全能够胜任。 
排版表格最基本的 tabular 环境用法为： 
 \begin{tabular} [ ⟨ align ⟩ ] { ⟨ column-spec ⟩ } ⟨ item1 ⟩ & ⟨ item2 ⟩ & …  \ \  \hline ⟨ item1 ⟩ & ⟨ item2 ⟩ & …  \ \  \end{tabular} 
其中⟨column-spec⟩是列格式标记，在接下来的内容将仔细介绍；& 用来分隔单元格； \ \ 用来换行； \hline 用来在行与行之间绘制横线。 
直接使用 tabular 环境的话，会和周围的文字混排。此时可用一个可选参数⟨align⟩控制垂直对齐：t 和 b 分别表示按表格顶部、底部对齐，其他参数或省略不写 (默认) 表示居中对齐。 
 \begin{tabular}{|c|} center- \ \ aligned  \ \
  \end{tabular},
  \begin{tabular}[t]{|c|} top- \ \ aligned  \ \
  \end{tabular},
  \begin{tabular}[b]{|c|} bottom- \ \ aligned \ \
  \end{tabular} tabulars. bottom 
center, top, aligned tabulars. aligned aligned 
但是通常情况下 tabular 环境很少与文字直接混排，而是会放在 table 浮动体环境中，并用 \caption 命令加标题。 
# 3.6.1 列格式 
tabular 环境使用⟨column-spec⟩参数指定表格的列数以及每列的格式。基本的列格式见表 3.1。 
表 3.1: LaTeX 表格列格式
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/c84a5ee1a67de6794fd45b915d4eed8be709bdfca5c9e05e450f8891d5aa1d0d.jpg) 
 \begin{tabular}{lcr|p{6em}}
  \hline left & center & right & par box with fixed width \ \ L & C & R & P  \ \
  \hline
  \end{tabular} 
left center right par box with fixed width L C R P 
表格中每行的单元格数目不能多于列格式里 $ \updownarrow/{ \tt c}/{ \tt r}/{ \tt p}$ 的总数 (可以少于这个总数)，否则出错。 
@ 格式可在单元格前后插入任意的文本，但同时它也消除了单元格前后额外添加的间距。 $ \circledcirc$ 格式可以适当使用以充当“竖线”。特别地， $ \mathbb{Q} \{ \}$ 可直接用来消除单元格前后的间距： 
 \begin{tabular}{@{} r@{:}lr @{}}  \hline 1 & 1 & one  \ \ 11 & 3 & eleven  \ \  \hline
  \end{tabular} 1:1 one 11:3 eleven 
另外 $ \mathrm{emptyset}$ 还提供了简便的将格式参数重复的写法 $* \{ \langle n \rangle \} \{ \langle c o l u m n \cdot s p e c \rangle \}$ ，比如以下两种写法是等效的： 
 \begin{tabular}{|c|c|c|c|c|p{4em}|p{4em}|}
  \begin{tabular}{| \*{5}{c|} \*{2}{p{4em}|}} 
有时需要为整列修饰格式，比如整列改变为粗体，如果每个单元格都加上 \bfseries 命令会比较麻烦。array 宏包提供了辅助格式 $>$ 和 $<$ ，用于给列格式前后加上修饰命令： 
%  \usepackage{array}
  \begin{tabular}{>{ \itshape} $ \cdot{ \tt r}{<} \{* \}{ \tt I} \}$  \hline italic & normal  \ \ column & column  \ \  \hline
  \end{tabular} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/bb01ff2cdc7b556c69ced0e16773e4d3a8e359da961fb3237cd29e0d443befee.jpg) 
辅助格式甚至支持插入 \centering 等命令改变 p 列格式的对齐方式，一般还要加额外的命令 \array backslash 以免出错 6： 
%  \usepackage{array}
  \begin{tabular}%
 {>{ \centering \array backslash}p{9em}}  \hline Some center-aligned long TeXt.  \ \  \hline
  \end{tabular} 
Some center-aligned long TeXt. 
array 宏包还提供了类似 p 格式的 $ \mathtt{m}$ 格式和 b 格式，三者分别在垂直方向上靠顶端对齐、居中以及底端对齐。 
%  \usepackage{array}
  \newcommand \txt{a b c d e f g h i}
  \begin{tabular}{cp{2em}m{2em}b{2em}}  \hline pos &  \txt &  \txt &  \txt  \ \  \hline
  \end{tabular} 
a b c a b c d e f pos a b c d e f g h i d e f g h i g h i 
# 3.6.2 列宽 
在控制列宽方面， $ \mathrm{emptyset}$ 表格有着明显的不足： $ \mathtt{L}/ \mathtt{c}/ \mathtt{r}$ 格式的列宽是由文字内容的自然宽度决定的，而 p 格式给定了列宽却不好控制对齐 (可用 array 宏包的辅助格式)，更何况列与列之间通常还有间距，所以直接生成给定总宽度的表格并不容易。 
LaTeX 本身提供了 tabular $^{ \ast}$ 环境用来排版定宽表格，但是不太方便使用，比如要用到 $ \circledcirc$ 格式插入额外命令，令单元格之间的间距为 \fill，但即使这样仍然有瑕疵： 
 \begin{tabular \*}{14em}%
 {@{ \extra col sep{ \fill}}|c|c|c|c|}  \hline A & B & C & D  \ \  \hline a & b & c & d  \ \  \hline
  \end{tabular \*} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/00c818fa660dbcd3811cc5dc5e3590ae3c756a9e8fe5117786657f45c7374877.jpg) 
tabularx 宏包为我们提供了方便的解决方案。它引入了一个 X 列格式，类似 p 列格式，不过会根据表格宽度自动计算列宽，多个 X 列格式平均分配列宽。X 列格式也可以用 array 里的辅助格式修饰对齐方式： 
%  \usepackage{array, tabularx}
  \begin{tabularx}{14em}%
 {| \*{4}{>{ \centering \array backslash}X|}}  \hline A & B & C & D  \ \  \hline a & b & c & d  \ \  \hline
  \end{tabularx} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/3c05169cb09d2def2023e148f335fa98c0de17caf5258969d4cd425e52eb2d50.jpg) 
# 3.6.3 横线 
我们已经在之前的例子见过许多次绘制表格线的 \hline 命令。另外 \cline{ $ \mathbf{ \check{ \langle}}i \mathbf{ \rangle} \mathbf{-} \mathbf{ \langle}j \mathbf{ \rangle} \mathbf{ \rangle}$ 用来绘制跨越部分单元格的横线： 
 \begin{tabular}{|c|c|c|}  \hline 4 & 9 & 2  \ \  \cline{2-3} 3 & 5 & 7  \ \  \cline{1-1} 8 & 1 & 6  \ \  \hline
  \end{tabular} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/8856d8366e9a17db82bf25448af6b0da137510a734957664b40cf87f3ff62623.jpg) 
在科技论文排版中广泛应用的表格形式是三线表，形式干净简明。三线表由 booktabs 宏包支持，它提供了 \toprule、 \midrule 和 \bottomrule 命令用以排版三线表的三条线，以及和 \cline 对应的 \cmidrule。除此之外，最好不要用其它横线以及竖线： 
%  \usepackage{booktabs}
  \begin{tabular}{cccc}  \toprule &  \multi column{3}{c}{Numbers}  \ \  \cmidrule{2-4} & 1 & 2 & 3  \ \  \midrule Alphabet & A & B & C  \ \ Roman & I & II& III  \ \  \bottomrule
  \end{tabular} 
Numbers 1 2 3 Alphabet A B C Roman I II III 
# 3.6.4 合并单元格 
$ \mathrm{emptyset}$ 是一行一行排版表格的，横向合并单元格较为容易，由 \multi column 命令实现： $ \mathsf{ \backslash m u l t i c o l u m n \{}} \{ \langle n \rangle \} \{ \langle c o l u m n \cdot s p e c \rangle \} \{ \langle i t e m \rangle \}$ 
其中 $ \langle n \rangle$ 为要合并的列数，⟨column-spec⟩为合并单元格后的列格式，只允许出现一个 l/c/r 或 p 格式。如果合并前的单元格前后带表格线|，合并后的列格式也要带| 以使得表格的竖线一致。 
 \begin{tabular}{|c|c|c|}  \hline 1 & 2 & Center  \ \  \hline  \multicolumn{2}{|c|}{3} &  \multi column{1}{r|}{Right}  \ \  \hline 4 &  \multi column{2}{c|}{C}  \ \  \hline 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/bd3735e76153e7ed367bf2f15b2210370b8bb2a300dd0b7ef2ed43b92ac39c18.jpg) 
上面的例子还体现了，形如 \multi column{1}{⟨column-spec⟩}{⟨item⟩} 的命令可以用来修改某一个单元格的列格式。 
纵向合并单元格需要用到 multirow 宏包提供的 \multirow 命令： $ \mathtt{ \backslash m u l t i r o w \{ \langle n \rangle \} \{ \langle w i d t h \rangle \} \{ \langle i t e m \rangle \}}$ 
⟨width⟩为合并后单元格的宽度，可以填 $^*$ 以使用自然宽度。我们看一个结合 \cline、 \multi column 和 \multirow 命令的例子： 
%  \usepackage{multirow}
  \begin{tabular}{ccc}  \hline  \multirow{2}{ \*}{Item} &  \multi column{2}{c}{Value}  \ \  \cline{2-3} & First & Second  \ \  \hline A & 1 & 2  \ \  \hline
  \end{tabular} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/3fe4a0c48427c49da427f874884d190f106e7869c45a832f60eea2edadf1d55e.jpg) 
# 3.6.5 嵌套表格 
相对于合并单元格，拆分单元格对于 $ \mathrm{emptyset}$ 来说并非易事。在单元格中嵌套一个小表格可以起到“拆分单元格”的效果。在以下的例子中，注意要用 \multi column 命令配合 $ \mathbb{Q} \{ \}$ 格式把单元格的额外边距去掉，使得嵌套的表格线能和外层的表格线正确相连： 
 \begin{tabular}{|c|c|c|}
  \hline a & b & c  \ \  \hline a &  \multicolumn{1}{@{}c@{}|} { \begin{tabular}{c|c} e & f  \ \  \hline e & f  \ \  \end{tabular}} & c  \ \  \hline a & b & c  \ \  \hline  \end{tabular} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/cc3c0e07845b733a5e3e9eb55dfd6057da450b0538d6ff1c7dab16bedb3f6249.jpg) 
如果不需要为“拆分的单元格”画线，并且只在垂直方向“拆分”的话，makecell 宏包提供的 \makecell 命令是一个简单的解决方案： 
%  \usepackage{makecell}
  \begin{tabular}{|c|c|}
  \hline a &  \makecell{d1  \ \ d2}  \ \ \hline b & c  \ \
  \hline
  \end{tabular} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/54f55c695843eeb260fc6fe22ffad32bfbf746bef8d0822a138a200509d826c4.jpg) 
# 3.6.6 行距控制 
$ \mathrm{emptyset}$ 生成的表格看起来通常比较紧凑。修改参数 \array stretch 可以得到行距更加宽松的表格 (相关命令参考 8.1.1 小节)： 
 \renew command \array stretch{1.8}
  \begin{tabular}{|c|}  \hline Really loose  \ \  \hline tabular rows. \ \  \hline
  \end{tabular} 
Really loose tabular rows. 
另一种增加间距的办法是给换行命令 \ \ 添加可选参数，在这一行下面加额外的间距，适合用于在行间不加横线的表格： 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/99fe0d1eb419d6f1ce051b2ee24d373cb7821e533e02024bacfb3ef0e91231c2.jpg) 
但是这种换行方式的存在导致了一个缺陷——从第二行开始，表格的首个单元格不能直接使用中括号[]，否则 \ \ 往往会将下一行的中括号当作自己的可选参数，因而出错。如果要使用中括号，应当放在花括号{} 里面。或者也可以选择将换行命令写成 \ \[0pt]。 
# 3.7 图片 
LaTeX 本身不支持插图功能，需要由 graphicx 宏包辅助支持。 
使用 laTeX $^+$ dvipdfmx 编译命令时，调用 graphicx 宏包时要指定 dvipdfmx 选项 7；而使用 pdflaTeX 或 xelaTeX 命令编译时不需要。 
读者可能听说过“LaTeX 只能插入. eps 格式的图片，需要把. jpg 转成. eps 格式”的观点。LaTeX 发展到今天，这个观点早已过时。事实上不同编译命令支持的图片格式种类各异，见表 3.2。这个表格也能解答诸如“为什么. eps 格式图片在 pdflaTeX 编译命令下出错”之类的问题。本表格也再一次说明，使用 xelaTeX 命令是我们最推荐的方式。 
表 3.2: 各种编译方式支持的主流图片格式
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/b96e847f394cee9254f514b2bdc6a5abc5e49d8972db30a1ac29d3be702d30ad.jpg) 
注：在较新的 $ \mathrm{TeX}$ 发行版中，laTeX $^+$ dvipdfmx 和 pdflaTeX 命令可不依赖宏包，支持原来需要宏包扩展的图片格式 (但 pdflaTeX 命令仍不支持. bmp 格式的位图)。 
在调用了 graphicx 宏包以后，就可以使用 \include graphics 命令加载图片了： 
 \include graphics [ ⟨ options ⟩ ] { ⟨ filename ⟩ } 
其中⟨filename⟩为图片文件名，与 \include 命令的用法类似，文件名可能需要用相对路径或绝对路径表示 (见 1.6 节)。图片文件的扩展名一般可不写。另外一定要注意，文件名里既不要有空格 (类似 \include)，也不要有多余的英文点号，否则宏包在解析文件名的过程中会出错。 
另外 graphicx 宏包还提供了 \graphics path 命令，用于声明一个或多个图片文件存放的目录，使用这些目录里的图片时可不用写路径： 
% 假设主要的图片放在 figures 子目录下，标志放在 logo 子目录下
 \graphics path {{figures/}{logo/}} 
在 \include graphics 命令的可选参数⟨options⟩中可以使用 $ \langle k e y \rangle{=} \langle v a l u e \rangle$ 的形式，常用的参数如下： 
表 3.3:  \include graphics 命令的可选参数
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/1ded61cd8eb666e7cb3842abeab822087ce3550fbd82a5115e6cd819b39c5423.jpg) 
graphicx 宏包也支持 draft/final 选项。当 graphicx 宏包或文档类指定 draft 选项时，图片将不会被实际插入，取而代之的是一个包含文件名的与原图片等大的方框。 
# 3.8 盒子 
盒子是 LaTeX 排版的基础单元，虽然解释略有抽象：每一行是一个盒子，里面的文字从左到右依次排列；每一页也是一个盒子，各行文字从上到下依次排布⋯⋯颇有一些活字印刷术的味道。 
不管如何，LaTeX 提供了一些命令让我们手动生成一些有特定用途的盒子。 
# 3.8.1 水平盒子 
生成水平盒子的命令如下： 
 \mbox{…}  \makebox [ ⟨ width ⟩ ][ ⟨ align ⟩ ] {…} 
 \mbox 生成一个基本的水平盒子，内容只有一行，不允许分段 (除非嵌套其它盒子，比如后文的垂直盒子)。外表看上去， \mbox 的内容与正常的文本无二，不过断行时文字不会从盒子里断开。 
 \makebox 更进一步，可以加上可选参数用于控制盒子的宽度⟨width⟩，以及内容的对齐方式⟨align⟩，可选居中 $ \mathtt{C}$ (默认值)、左对齐 l、右对齐 $ \mathtt{r}$ 和分散对齐 ${ \mathsf{s}}^{8}$ 。 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/a0473c9e4f5401734d31153338b45fd899156eb54107dd5a2e5d3401c1e9375d.jpg) 
# 3.8.2 带框的水平盒子 
 \fbox 和 \framebox 让我们可以为水平盒子添加边框。使用的语法与 \mbox 和 \makebox 一模一样： 
 \fbox{…}
  \framebox [ ⟨ width ⟩ ][ ⟨ align ⟩ ] {…} 
可以通过 \setlength 命令 (见 5.3.1 小节) 调节边框的宽度 \fboxrule 和内边距 \fboxsep \framebox[10em][r]{Test box} \ \[1ex]
  \setlength{ \fboxrule}{1.6pt}
  \setlength{ \fboxsep}{1em}
  \framebox[10em][r]{Test box} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/34067d852d79d017feddb5035b0098933f10f985e201bf7afa48f701e4166d62.jpg) 
# 3.8.3 垂直盒子 
如果需要排版一个文字可以换行的盒子， $ \mathrm{emptyset}$ 提供了两种方式： \parbox [ ⟨ align ⟩ ][ ⟨ height ⟩ ][ ⟨ inner-align ⟩ ] { ⟨ width ⟩ }{…}  \begin{minipage} [ ⟨ align ⟩ ][ ⟨ height ⟩ ][ ⟨ inner-align ⟩ ] { ⟨ width ⟩ } …  \end{minipage} 
其中⟨align⟩为盒子和周围文字的对齐情况 (类似 tabular 环境)；⟨height⟩和⟨inner-align⟩设置盒子的高度和内容的对齐方式，类似水平盒子 \makebox 的设置，不过⟨inner-align⟩接受的参数是顶部 t、底部 b、居中 c 和分散对齐 s。 
三字经： \parbox[t]{3em}%
{ 人之初性本善性相近习相远 }
  \quad 千字文： \begin{minipage}[b][8ex][t]{4em} 天地玄黄宇宙洪荒 \end{minipage} 
天地玄黄宇宙洪荒三字经：人之初千字文：性本善性相近习相远 
如果在 minipage 里使用 \footnote 命令，生成的脚注会出现在盒子底部，编号是独立的，并且使用小写字母编号。这也是 minipage 环境之被称为“迷你页”(Mini-page) 的原因。而在
 \parbox 里无法正常使用 \footnote 命令，只能在盒子里使用 \footnote mark，在盒子外使用
 \footnote TeXt。 
 \fbox{ \begin{minipage}{15em}% 这是一个垂直盒子的测试。 \footnote{ 脚注来自 minipage 。 }
  \end{minipage}} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/15a27c3e0cbff2ea66126fe38fa9f4e9b3270615e0d3645900ff2d5a5c25e7c9.jpg) 
# 3.8.4 标尺盒子 
 \rule 命令用来画一个实心的矩形盒子，也可适当调整以用来画线 (标尺)：  \rule [ ⟨ raise ⟩ ] { ⟨ width ⟩ }{ ⟨ height ⟩ } 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/43b3fcbb87e8b366d56493ec07320ef5c68b3ba087f8a54e2a36d8bf291fd417.jpg) 
# 3.9 浮动体 
内容丰富的文章或者书籍往往包含许多图片和表格等内容。这些内容的尺寸往往太大，导致分页困难。LaTeX 为此引入了浮动体的机制，令大块的内容可以脱离上下文，放置在合适的位置。 
LaTeX 预定义了两类浮动体环境 figure 和 table。习惯上 figure 里放图片，table 里放表格，但并没有严格限制，可以在任何一个浮动体里放置文字、公式、表格、图片等等任意内容。 
以 table 环境的用法举例，figure 同理： 
 \begin{table} [ ⟨ placement ⟩ ] …  \end{table} 
⟨placement⟩参数提供了一些符号用来表示浮动体允许排版的位置，如 hbp 允许浮动体排版在当前位置、底部或者单独成页。table 和 figure 浮动体的默认设置为 tbp。 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/8f4427b85f84560503281f12605becb4dc0db9a0f9666a359adc2f5d6e0013cf.jpg) 
注 1：排版位置的选取与参数里符号的顺序无关， $ \mathrm{emptyset}$ 总是以 h-t-b-p 的优先级顺序决定浮动体位置。也就是说[!htp] 和[ph! t] 没有区别。注 2：限制包括浮动体个数 (除单独成页外，默认每页不超过 3 个浮动体，其中顶部不超过 2 个，底部不超过 1 个) 以及浮动体空间占页面的百分比 (默认顶部不超过 70%，底部不超过 30%)。 
双栏排版环境下，LaTeX 提供了 table $^*$ 和 figure \* 环境用来排版跨栏的浮动体。它们的用法与 table 和 figure 一样，不同之处为双栏的⟨placement⟩参数只能用 tp 两个位置。 
浮动体的位置选取受到先后顺序的限制。如果某个浮动体由于参数限制、空间限制等原因在当前页无法放置，就要推迟到之后处理，并使得之后的同类浮动体一并推迟。 \clearpage 命令会在另起一页之前，先将所有推迟处理的浮动体排版成页，此时 htbp 等位置限制被完全忽略。 
float 宏包为浮动体提供了 H 位置参数，不与 htbp 及! 混用。使用 H 位置参数时，会取消浮动机制，将浮动体视为一般的盒子插入当前位置。这在一些特殊情况下很有用 (如使用 multicol 宏包排版分栏内容的时候)，但尺寸过大的浮动体可能使得分页比较困难。 
# 3.9.1 浮动体的标题 
图表等浮动体提供了 \caption 命令加标题，并且自动给浮动体编号： 
 \caption{…} 
 \caption 的用法非常类似于 \section 等命令，可以用带星号的命令 \caption \*9 生成不带编号的标题，也可以使用带可选参数的形式 \caption[…]{…}，使得在目录里使用短标题。 \caption 命令之后还可以紧跟 \label 命令标记交叉引用。 
 \caption 生成的标题形如“Figure 1: …”(figure 环境) 或“Table 1: …”(table 环境)。可通过修改 \figurename 和 \tablename 的内容来修改标题的前缀 (详见第 8.4 节)。标题样式的定制功能由 caption 宏包提供，详见该宏包的帮助文档，在此不作赘述。 
table 和 figure 两种浮动体分别有各自的生成目录的命令： 
 \list of tables  \list of figures 
它们类似 \table of contents 生成单独的章节。 
# 3.9.2 并排和子图表 
我们时常有在一个浮动体里面放置多张图的用法。最简单的用法就是直接并排放置，也可以通过分段或者换行命令 \ \ 排版多行多列的图片。以下为示意代码，效果大致如图 3.1 所示。 
 \begin{figure}[htbp]  \centering  \include graphics[width $ \it{.}=$ ...]{...}  \qquad  \include graphics[width $ \it{.}=$ ...]{...}  \ \[... pt]  \include graphics[width $ \it{.}=$ ...]{...}  \caption{...}
  \end{figure} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/edd43cc62b4987496d40f9b27045d5c5c653d156fcf0e6cc45bb9827f3249605.jpg) 
图 3.1: 并排放置图片的示意。 
由于标题是横跨一行的，用 \caption 命令为每个图片单独生成标题就需要借助前文提到的
 \parbox 或者 minipage 环境，将标题限制在盒子内。效果见图 3.2 和图 3.3。 
 \begin{figure}[htbp]  \centering  \begin{minipage}{...} 
 \centering  \include graphics[width $ \overbar{ \ }$ ...]{...}  \caption{...}  \end{minipage}  \qquad  \begin{minipage}{...}  \centering  \include graphics[width $ \overbar{ \ }$ ...]{...}  \caption{...}  \end{minipage}  \end{figure} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/9bcb2f78811d679035d33e6dc4ebebd239ef48cc6e73769352b7423a82bb6e6b.jpg) 
图 3.2: 并排图 1 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/508063d58ac034bd0a8ddefcd052d09fe25eb4df02efbf35c4c64f82319225e3.jpg) 
图 3.3: 并排图 2 
当我们需要更进一步，给每个图片定义小标题时，就要用到 subcaption 宏包的功能了。这里仅举一例，效果见图 3.4a 和 $3.4 \mathrm{b}$ 。更详细的用法请参考 subcaption 宏包的帮助文档。 
 \begin{figure}[htbp]  \centering  \begin{subfigure}{...}  \centering  \include graphics[width $ \overbar{ \ }$ ...]{...}  \caption{...}  \end{subfigure}  \qquad  \begin{subfigure}{...}  \centering  \include graphics[width $ \overbar{ \ }$ ...]{...}  \caption{...}  \end{subfigure}
  \end{figure} 
![](https://cdn-mineru.openxlab.org.cn/model-mineru/prod/2f8f8d9b9c730a5bb36cc6db27154bea99f4c7f6008db4997efccfd5c8450856.jpg) 
图 3.4: 使用 subcaption 宏包的 subfigure 环境排版子图。subcaption 依赖上文提到过的 caption 宏包，因此也支持子图表标题样式的定制。并排子图 
表的功能也可通过 subfig 宏包的 \subfloat 命令实现，具体请参考宏包文档。 
# 4 排版数学公式
# 5 排版样式设定
# 6 特色工具和功能
# 7 绘图功能
# 8 自定义 LaTeX 命令和功能
# A 安装 LaTeX 发行版
# B 排除错误、寻求帮助
