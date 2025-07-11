---
completed: true
version: 0.13.1
---
# Writing in Typst
Let's get started! Suppose you got assigned to write a technical report for university. It will contain prose, maths, headings, and figures. To get started, you create a new project on the Typst app. You'll be taken to the editor where you see two panels: A source panel where you compose your document and a preview panel where you see the rendered document.

![Typst app screenshot](https://typst.app/assets/docs/1-writing-app.png)

You already have a good angle for your report in mind. So let's start by writing the introduction. Enter some text in the editor panel. You'll notice that the text immediately appears on the previewed page.

```
In this report, we will explore the
various factors that influence fluid
dynamics in glaciers and how they
contribute to the formation and
behaviour of these natural structures.
```

![Preview](https://typst.app/assets/docs/ePl1U-2a7w8qkmb3CLl_oAAAAAAAAAAA.png)

_Throughout this tutorial, we'll show code examples like this one. Just like in the app, the first panel contains markup and the second panel shows a preview. We shrunk the page to fit the examples so you can see what's going on._

The next step is to add a heading and emphasize some text. Typst uses simple markup for the most common formatting tasks. To add a heading, enter the `=` character and to emphasize some text with italics, enclose it in `_underscores_`.

>  使用 `=` 表示标题，使用 `_text_` 表示斜体

```
= Introduction
In this report, we will explore the
various factors that influence _fluid
dynamics_ in glaciers and how they
contribute to the formation and
behaviour of these natural structures.
```

![Preview](https://typst.app/assets/docs/p75v-z7QqVChplB2N8HZfwAAAAAAAAAA.png)

That was easy! To add a new paragraph, just add a blank line in between two lines of text. If that paragraph needs a subheading, produce it by typing `==` instead of `=`. The number of `=` characters determines the nesting level of the heading.
>  由空行隔开的两段文本就是两个单独的 paragraph
>  子标题就是添加 `=` 的数量，例如 `==`

Now we want to list a few of the circumstances that influence glacier dynamics. To do that, we use a numbered list. For each item of the list, we type a `+` character at the beginning of the line. Typst will automatically number the items.
>  有序列表通过在行首使用 `+` 表示

```
+ The climate
+ The topography
+ The geology
```

![Preview](https://typst.app/assets/docs/U3IHQbhSNQ8ndkXIv_gPrgAAAAAAAAAA.png)

If we wanted to add a bulleted list, we would use the `-` character instead of the `+` character. We can also nest lists: For example, we can add a sub-list to the first item of the list above by indenting it.
>  无序列表使用 `-`
>  列表可以嵌套，通过添加缩进即可

```
+ The climate
  - Temperature
  - Precipitation
+ The topography
+ The geology
```

![Preview](https://typst.app/assets/docs/xmS-BPiM_gDHkWk9_uhE_gAAAAAAAAAA.png)

## Adding a figure
You think that your report would benefit from a figure. Let's add one. Typst supports images in the formats PNG, JPEG, GIF, and SVG. To add an image file to your project, first open the _file panel_ by clicking the box icon in the left sidebar. Here, you can see a list of all files in your project. Currently, there is only one: The main Typst file you are writing in. To upload another file, click the button with the arrow in the top-right corner. This opens the upload dialog, in which you can pick files to upload from your computer. Select an image file for your report.
>  Typst 支持 PNG, JEPG, GIF, SVG 格式的图像

![Upload dialog](https://typst.app/assets/docs/1-writing-upload.png)

We have seen before that specific symbols (called _markup_) have specific meaning in Typst. We can use `=`, `-`, `+`, and `_` to create headings, lists and emphasized text, respectively. However, having a special symbol for everything we want to insert into our document would soon become cryptic and unwieldy. For this reason, Typst reserves markup symbols only for the most common things. Everything else is inserted with _functions._ For our image to show up on the page, we use Typst's [`image`](https://typst.app/docs/reference/visualize/image/ "`image`") function.

```
#image("glacier.jpg")
```

>  Typst 仅保留了最常用的一些标记符，其他的一些表示通过 “函数” 实现
>  要插入图像，需要使用 `image` 函数: `#image(path to image)`

![Preview](https://typst.app/assets/docs/KwKlYCVb2uFZqZ8abt3-ggAAAAAAAAAA.png)

In general, a function produces some output for a set of _arguments_. When you _call_ a function within markup, you provide the arguments and Typst inserts the result (the function's _return value_) into the document. In our case, the `image` function takes one argument: The path to the image file.
>   简单来说，函数为接收的一组参数产生输出
>   调用函数时，我们提供一组参数，Typst 会将函数的返回值插入到文档中

 To call a function in markup, we first need to type the `#` character, immediately followed by the name of the function. Then, we enclose the arguments in parentheses. Typst recognizes many different data types within argument lists. Our file path is a short [string of text](https://typst.app/docs/reference/foundations/str/), so we need to enclose it in double quotes.
>  要调用函数，我们需要使用 `#<function_name>(arguments)` 的语法
>  Typst 对函数参数可以识别不同的数据类型，例如字符串等

The inserted image uses the whole width of the page. To change that, pass the `width` argument to the `image` function. This is a _named_ argument and therefore specified as a `name: value` pair. If there are multiple arguments, they are separated by commas, so we first need to put a comma behind the path.
>  `image` 函数还接收命名参数，例如 `width: 70%`
>  命名参数通过 `name: value` 指定
>  参数之间用逗号隔开

```
#image("glacier.jpg", width: 70%)
```

![Preview](https://typst.app/assets/docs/lpadKIOzcEsf_MGoSeZghAAAAAAAAAAA.png)

The `width` argument is a [relative length](https://typst.app/docs/reference/layout/relative/). In our case, we specified a percentage, determining that the image shall take up `70%` of the page's width. We also could have specified an absolute value like `1cm` or `0.7in`.

Just like text, the image is now aligned at the left side of the page by default. It's also lacking a caption. Let's fix that by using the [figure](https://typst.app/docs/reference/model/figure/ "figure") function. This function takes the figure's contents as a positional argument and an optional caption as a named argument.
>  `figure` 函数接收图像的内容作为位置参数，并接收一个可选参数作为 caption

Within the argument list of the `figure` function, Typst is already in code mode. This means, you now have to remove the hash before the image function call. The hash is only needed directly in markup (to disambiguate text from function calls).
>  注意在 `figure` 的参数列表中，Typst 已经处于 code mode，这意味着我们需要移除 `image` 函数调用之前的 `#`
>  `#` 只有直接在 markup 中使用时是需要的 (以区分文本和函数调用)

The caption consists of arbitrary markup. To give markup to a function, we enclose it in square brackets. This construct is called a _content block._
>  caption 可以接收任意的 markup，注意要传递 markup 时，需要用 `[]` 框起，这个构造称为 content block

```
#figure(
  image("glacier.jpg", width: 70%),
  caption: [
    _Glaciers_ form an important part
    of the earth's climate system.
  ],
)
```

![Preview](https://typst.app/assets/docs/v5OnReUO8fD5Rfj2aJZVyQAAAAAAAAAA.png)

You continue to write your report and now want to reference the figure. To do that, first attach a label to figure. A label uniquely identifies an element in your document. Add one after the figure by enclosing some name in angle brackets. You can then reference the figure in your text by writing an `@` symbol followed by that name. Headings and equations can also be labelled to make them referenceable.
>  要引用某个 figure，我们需要为 figure 添加 label, label 唯一地识别了文档中的一个元素
>  我们通过在 figure 后添加 `<label>` 来进行标记
>  在之后的文本里，就可以使用 `@label` 来引用特定的符号
>  标题和公式也可以被标记并引用

```
Glaciers as the one shown in
@glaciers will cease to exist if
we don't take action soon!

#figure(
  image("glacier.jpg", width: 70%),
  caption: [
    _Glaciers_ form an important part
    of the earth's climate system.
  ],
) <glaciers>
```

![Preview](https://typst.app/assets/docs/cwZ12iQ39B4L-_wQwhO2TAAAAAAAAAAA.png)

INFO
So far, we've passed content blocks (markup in square brackets) and strings (text in double quotes) to our functions. Both seem to contain text. What's the difference?

A content block can contain text, but also any other kind of markup, function calls, and more, whereas a string is really just a _sequence of characters_ and nothing else.

>  content block 和 string 都可以作为参数传递给函数，差异在于 content block 可以包含文本、任意类型的 markup、函数调用等等，而 string 只是字符序列

For example, the image function expects a path to an image file. It would not make sense to pass, e.g., a paragraph of text or another image as the image's path parameter. That's why only strings are allowed here. On the contrary, strings work wherever content is expected because text is a valid kind of content.

## Adding a bibliography
As you write up your report, you need to back up some of your claims. You can add a bibliography to your document with the [`bibliography`](https://typst.app/docs/reference/model/bibliography/ "`bibliography`") function. This function expects a path to a bibliography file.
>  文献目录通过 `bibliography` 函数添加，该函数期待接收一个 bibliography 文件的路径

Typst's native bibliography format is [Hayagriva](https://github.com/typst/hayagriva/blob/main/docs/file-format.md), but for compatibility you can also use BibLaTeX files. As your classmate has already done a literature survey and sent you a `.bib` file, you'll use that one. Upload the file through the file panel to access it in Typst.
>  Typst 的原生 bibliography 格式为 Hayagriva，也兼容 BibLaTeX (`.bib`)

Once the document contains a bibliography, you can start citing from it. Citations use the same syntax as references to a label. As soon as you cite a source for the first time, it will appear in the bibliography section of your document. Typst supports different citation and bibliography styles. Consult the [reference](https://typst.app/docs/reference/model/bibliography/#parameters-style) for more details.

```
= Methods
We follow the glacier melting models
established in @glacier-melt.

#bibliography("works.bib")
```

>  使用 `bibliography` 导入了文献目录之后，就可以引用其中的文献
>  引用文献的格式和引用 label 的格式相同，即使用 `@`，只要我们在文中的某处引用了文献，它就同时会出现在结尾的 Bibliography 中

![Preview](https://typst.app/assets/docs/QGPHT14ksdea0r_8vy01WAAAAAAAAAAA.png)

## Maths
After fleshing out the methods section, you move on to the meat of the document: Your equations. Typst has built-in mathematical typesetting and uses its own math notation. Let's start with a simple equation. We wrap it in `$` signs to let Typst know it should expect a mathematical expression:
>  Typst 的内建数学排版引擎使用其自己的数学符号
>  简单的公式用 `$` 包围

```
The equation $Q = rho A v + C$
defines the glacial flow rate.
```

![Preview](https://typst.app/assets/docs/_u5BjLoMFBZU2zg1OWULdgAAAAAAAAAA.png)

The equation is typeset inline, on the same line as the surrounding text. If you want to have it on its own line instead, you should insert a single space at its start and end:

```
The flow rate of a glacier is
defined by the following equation:

$ Q = rho A v + C $
```

![Preview](https://typst.app/assets/docs/GXI0mvGOqqSC165iRTK-QwAAAAAAAAAA.png)

We can see that Typst displayed the single letters `Q`, `A`, `v`, and `C` as-is, while it translated `rho` into a Greek letter. Math mode will always show single letters verbatim. Multiple letters, however, are interpreted as symbols, variables, or function names. To imply a multiplication between single letters, put spaces between them.
>  Typst 将 `rho` 翻译为了希腊字母
>  Typst 的 math mode 会按照原样展示单个字符，但多个字符会被视为 symbols, variables, function names
>  在两个字符之间表示乘法只需要在二者之间放一个空格

If you want to have a variable that consists of multiple letters, you can enclose it in quotes:

```
The flow rate of a glacier is given
by the following equation:

$ Q = rho A v + "time offset" $
```

![Preview](https://typst.app/assets/docs/JSaojGBiKH-FLbIYqeWSgAAAAAAAAAAA.png)

>  要在公式中插入文本 (Typst 称为 variable)，只需要使用 `""` 

You'll also need a sum formula in your paper. We can use the `sum` symbol and then specify the range of the summation in sub-and superscripts:

```
Total displaced soil by glacial flow:

$ 7.32 beta +
  sum_(i=0)^nabla Q_i / 2 $
```

>  表示求和时，使用 `sum` symbol，以及可以有额外的上下标

![Preview](https://typst.app/assets/docs/rTDyTGxJlXKPRHJub3ALRgAAAAAAAAAA.png)

To add a subscript to a symbol or variable, type a `_` character and then the subscript. Similarly, use the `^` character for a superscript. If your sub-or superscript consists of multiple things, you must enclose them in round parentheses.
>  如果上下标有多个字符，需要用 `()` 括起来

The above example also showed us how to insert fractions: Simply put a `/` character between the numerator and the denominator and Typst will automatically turn it into a fraction. Parentheses are smartly resolved, so you can enter your expression as you would into a calculator and Typst will replace parenthesized sub-expressions with the appropriate notation.
>  要插入分数，使用 `/`，括号会自动解析

```
Total displaced soil by glacial flow:

$ 7.32 beta +
  sum_(i=0)^nabla
    (Q_i (a_i - epsilon)) / 2 $
```

![Preview](https://typst.app/assets/docs/HgeB2Bx5Lh3a5NPfF6WEdwAAAAAAAAAA.png)

Not all math constructs have special syntax. Instead, we use functions, just like the `image` function we have seen before. For example, to insert a column vector, we can use the [`vec`](https://typst.app/docs/reference/math/vec/) function. Within math mode, function calls don't need to start with the `#` character.

```
$ v := vec(x_1, x_2, x_3) $
```

>  Typst 没有将所有的数学结构映射为特殊语法，一些数学结构使用函数表示
>  例如，要插入列向量，需要使用 `vec` 函数
>  在 math mode 中，函数调用不需要使用 `#` 符号

![Preview](https://typst.app/assets/docs/nj0pMnkuoX2t5FZ3_x6YKwAAAAAAAAAA.png)

Some functions are only available within math mode. For example, the [`cal`](https://typst.app/docs/reference/math/variants/#functions-cal) function is used to typeset calligraphic letters commonly used for sets. The [math section of the reference](https://typst.app/docs/reference/math/) provides a complete list of all functions that math mode makes available.
>  一些函数只能在 math mode 中使用，例如 `cal` 函数

One more thing: Many symbols, such as the arrow, have a lot of variants. You can select among these variants by appending a dot and a modifier name to a symbol's name:
>  许多 symbols，例如箭头，具有许多变体，我们可以通过 `.<modifier_name>` 来选择特定的变体

```
$ a arrow.squiggly b $
```

![Preview](https://typst.app/assets/docs/0GgQitNz41j-75F3FS6iAwAAAAAAAAAA.png)

This notation is also available in markup mode, but the symbol name must be preceded with `#sym.` there. See the [symbols section](https://typst.app/docs/reference/symbols/sym/) for a list of all available symbols.
>  一些 math mode 中的记号可以在 markup mode 中使用，但是要以 `#sym.` 开头，即 `#sym.<symbol_name>`

## Review
You have now seen how to write a basic document in Typst. You learned how to emphasize text, write lists, insert images, align content, and typeset mathematical expressions. You also learned about Typst's functions. There are many more kinds of content that Typst lets you insert into your document, such as [tables](https://typst.app/docs/reference/model/table/), [shapes](https://typst.app/docs/reference/visualize/), and [code blocks](https://typst.app/docs/reference/text/raw/). You can peruse the [reference](https://typst.app/docs/reference/ "reference") to learn more about these and other features.

For the moment, you have completed writing your report. You have already saved a PDF by clicking on the download button in the top right corner. However, you think the report could look a bit less plain. In the next section, we'll learn how to customize the look of our document.