---
completed: true
version: 0.13.1
---
# Formatting
So far, you have written a report with some text, a few equations and images. However, it still looks very plain. Your teaching assistant does not yet know that you are using a new typesetting system, and you want your report to fit in with the other student's submissions. In this chapter, we will see how to format your report using Typst's styling system.

## Set rules
As we have seen in the previous chapter, Typst has functions that _insert_ content (e.g. the [`image`](https://typst.app/docs/reference/visualize/image/ "`image`") function) and others that _manipulate_ content that they received as arguments (e.g. the [`align`](https://typst.app/docs/reference/layout/align/ "`align`") function). The first impulse you might have when you want, for example, to change the font, could be to look for a function that does that and wrap the complete document in it.
>  Typst 提供了一些用于调节内容的函数，例如 `align, text`
>  例如下例中使用了 `text` 函数来调节整段内容的字体

```
#text(font: "New Computer Modern")[
  = Background
  In the case of glaciers, fluid
  dynamics principles can be used
  to understand how the movement
  and behaviour of the ice is
  influenced by factors such as
  temperature, pressure, and the
  presence of other fluids (such as
  water).
]
```

![Preview](https://typst.app/assets/docs/Mxs7D0bmcIimpDtvVr74rgAAAAAAAAAA.png)

Wait, shouldn't all arguments of a function be specified within parentheses? Why is there a second set of square brackets with content _after_ the parentheses? The answer is that, as passing content to a function is such a common thing to do in Typst, there is special syntax for it: Instead of putting the content inside of the argument list, you can write it in square brackets directly after the normal arguments, saving on punctuation.
>  可以看到上面的函数调用语法和我们之前遇到的不太一样，其内容参数是在 `)` 之后传递进去的
>  因为向函数传递内容在 Typst 中非常常见，因此 Typst 为它提供了特殊的语法: 不将 content 放在 argument list 中，而是在参数列表后的 `[]` 中提供

As seen above, that works. With the [`text`](https://typst.app/docs/reference/text/text/ "`text`") function, we can adjust the font for all text within it. However, wrapping the document in countless functions and applying styles selectively and in-situ can quickly become cumbersome.

Fortunately, Typst has a more elegant solution. With _set rules,_ you can apply style properties to all occurrences of some kind of content. You write a set rule by entering the `set` keyword, followed by the name of the function whose properties you want to set, and a list of arguments in parentheses.
>  Typst 的 `set` 用于设定规则，设定之后，将对文档中所有的 content 生效
>  `set` 的使用方法是 `#set + <name_of_function>(arguments)` 即 `#set` 加上需要设定其属性的函数名称，以及其参数列表

```
#set text(
  font: "New Computer Modern"
)

= Background
In the case of glaciers, fluid
dynamics principles can be used
to understand how the movement
and behaviour of the ice is
influenced by factors such as
temperature, pressure, and the
presence of other fluids (such as
water).
```

![Preview](https://typst.app/assets/docs/yASdcoySH-jqyIxxwTDlJQAAAAAAAAAA.png)

INFO
Want to know in more technical terms what is happening here?

Set rules can be conceptualized as setting default values for some of the parameters of a function for all future uses of that function.

>  set rules 可以被视为为一个函数的未来所有使用设定了它某些参数的默认值

## The autocomplete panel
If you followed along and tried a few things in the app, you might have noticed that always after you enter a `#` character, a panel pops up to show you the available functions, and, within an argument list, the available parameters. That's the autocomplete panel. It can be very useful while you are writing your document: You can apply its suggestions by hitting the Return key or navigate to the desired completion with the arrow keys. The panel can be dismissed by hitting the Escape key and opened again by typing `#` or hitting Ctrl + Space. Use the autocomplete panel to discover the right arguments for functions. Most suggestions come with a small description of what they do.

![Autocomplete panel](https://typst.app/assets/docs/2-formatting-autocomplete.png)

## Set up the page
Back to set rules: When writing a rule, you choose the function depending on what type of element you want to style. Here is a list of some functions that are commonly used in set rules:

- [`text`](https://typst.app/docs/reference/text/text/ "`text`") to set font family, size, color, and other properties of text
- [`page`](https://typst.app/docs/reference/layout/page/ "`page`") to set the page size, margins, headers, enable columns, and footers
- [`par`](https://typst.app/docs/reference/model/par/ "`par`") to justify paragraphs, set line spacing, and more
- [`heading`](https://typst.app/docs/reference/model/heading/ "`heading`") to set the appearance of headings and enable numbering
- [`document`](https://typst.app/docs/reference/model/document/ "`document`") to set the metadata contained in the PDF output, such as title and author

>  set rules 时选择的函数取决于我们想要定义什么元素的风格，常用于 set rules 的函数有:
>  - `text`: 设置字体、大小、颜色等文本属性
>  - `page`: 设置页面大小、页边际、标题、启用列和脚注
>  - `par`: 对齐段落、设置行间距等
>  - `heading`: 设置标题的外观、启用计数
>  - `document`: 设置 PDF 输出中包含的元数据，例如标题和作者

Not all function parameters can be set. In general, only parameters that tell a function _how_ to do something can be set, not those that tell it _what_ to do it with. The function reference pages indicate which parameters are settable.
>  不是所有的函数参数可以被 set rules 设定
>  通常可以 set 的参数是告诉函数 how to do it 的参数，而不是 what to do it with 的参数

Let's add a few more styles to our document. We want larger margins and a serif font. For the purposes of the example, we'll also set another page size.

```
#set page(
  paper: "a6",
  margin: (x: 1.8cm, y: 1.5cm),
)
#set text(
  font: "New Computer Modern",
  size: 10pt
)
#set par(
  justify: true,
  leading: 0.52em,
)

= Introduction
In this report, we will explore the
various factors that influence fluid
dynamics in glaciers and how they
contribute to the formation and
behaviour of these natural structures.

...

#align(center + bottom)[
  #image("glacier.jpg", width: 70%)

  *Glaciers form an important
  part of the earth's climate
  system.*
]
```

![Preview](https://typst.app/assets/docs/vXvjGwfGgpk5eo7U4CVWMQAAAAAAAAAA.png)

There are a few things of note here.

First is the [`page`](https://typst.app/docs/reference/layout/page/ "`page`") set rule. It receives two arguments: the page size and margins for the page. The page size is a string. Typst accepts [many standard page sizes,](https://typst.app/docs/reference/layout/page/#parameters-paper) but you can also specify a custom page size. The margins are specified as a [dictionary.](https://typst.app/docs/reference/foundations/dictionary/) Dictionaries are a collection of key-value pairs. In this case, the keys are `x` and `y`, and the values are the horizontal and vertical margins, respectively. We could also have specified separate margins for each side by passing a dictionary with the keys `left`, `right`, `top`, and `bottom`.
>  `page` 函数接受了两个参数: page size 和 page margin
>  page size 为 string
>  page margin 为 dictionary, 可以是 `x, y` 指定水平和垂直，也可以是 `left, right, top, bottom` 分别指定上下左右

Next is the set [`text`](https://typst.app/docs/reference/text/text/ "`text`") set rule. Here, we set the font size to `10pt` and font family to `"New Computer Modern"`. The Typst app comes with many fonts that you can try for your document. When you are in the text function's argument list, you can discover the available fonts in the autocomplete panel.
>  `text` 函数接受了字号和字体

We have also set the spacing between lines (a.k.a. leading): It is specified as a [length](https://typst.app/docs/reference/layout/length/ "length") value, and we used the `em` unit to specify the leading relative to the size of the font: `1em` is equivalent to the current font size (which defaults to `11pt`).
>  `par` 函数设定了行间距 (leading)，其 value 类型是 "length"，其中 `1em` 等价于当前字体大小 (默认为 `11pt`)

Finally, we have bottom aligned our image by adding a vertical alignment to our center alignment. Vertical and horizontal alignments can be combined with the `+` operator to yield a 2D alignment.
>  `align` 中，水平和垂直的对齐可以通过 `+` 结合

## A hint of sophistication
To structure our document more clearly, we now want to number our headings. We can do this by setting the `numbering` parameter of the [`heading`](https://typst.app/docs/reference/model/heading/ "`heading`") function.
>  设定 `heading(numbering)` 参数可以为标题添加数字

```
#set heading(numbering: "1.")

= Introduction
#lorem(10)

== Background
#lorem(12)

== Methods
#lorem(15)
```

![Preview](https://typst.app/assets/docs/4WtF0u81AczurIYrwpRdcwAAAAAAAAAA.png)

We specified the string `"1."` as the numbering parameter. This tells Typst to number the headings with arabic numerals and to put a dot between the number of each level. We can also use [letters, roman numerals, and symbols](https://typst.app/docs/reference/model/numbering/) for our headings:
>  设定为 `"1."` 告诉 Typst 使用阿拉伯数字，并且在不同层级之间放 `.`

```
#set heading(numbering: "1.a")

= Introduction
#lorem(10)

== Background
#lorem(12)

== Methods
#lorem(15)
```

![Preview](https://typst.app/assets/docs/Llv0DrZ6U-QKf1vju2wP4QAAAAAAAAAA.png)

This example also uses the [`lorem`](https://typst.app/docs/reference/text/lorem/ "`lorem`") function to generate some placeholder text. This function takes a number as an argument and generates that many words of _Lorem Ipsum_ text.
>  `#lorem` 函数用于生成随意的问题，接收一个数字作为参数

INFO
Did you wonder why the headings and text set rules apply to all text and headings, even if they are not produced with the respective functions?

Typst internally calls the `heading` function every time you write `= Conclusion`. In fact, the function call `#heading[Conclusion]` is equivalent to the heading markup above. Other markup elements work similarly, they are only _syntax sugar_ for the corresponding function calls.
>  Typst 内部会在每次我们写下 `= ...` 的时候调用 `heading` 函数，实际上，函数调用 `#heading[...]` 等价于 `= ...`
>  其他的 markup 元素也是类似的，它们实际上可以认为是针对对应函数调用的语法糖

## Show rules
You are already pretty happy with how this turned out. But one last thing needs to be fixed: The report you are writing is intended for a larger project and that project's name should always be accompanied by a logo, even in prose.

You consider your options. You could add an `#image("logo.svg")` call before every instance of the logo using search and replace. That sounds very tedious. Instead, you could maybe [define a custom function](https://typst.app/docs/reference/foundations/function/#defining-functions) that always yields the logo with its image. However, there is an even easier way:

With show rules, you can redefine how Typst displays certain elements. You specify which elements Typst should show differently and how they should look. Show rules can be applied to instances of text, many functions, and even the whole document.
>  可以通过 `#show` 规则重新定义 Typst 如何展示特定的元素
>  `#show` rules 可以用于文本实例、函数、甚至整个文档

```
#show "ArtosFlow": name => box[
  #box(image(
    "logo.svg",
    height: 0.7em,
  ))
  #name
]

This report is embedded in the
ArtosFlow project. ArtosFlow is a
project of the Artos Institute.
```

![Preview](https://typst.app/assets/docs/349_Itx4-rTeNxzJmNofvgAAAAAAAAAA.png)

There is a lot of new syntax in this example: We write the `show` keyword, followed by a string of text we want to show differently and a colon. Then, we write a function that takes the content that shall be shown as an argument. Here, we called that argument `name`. We can now use the `name` variable in the function's body to print the ArtosFlow name. Our show rule adds the logo image in front of the name and puts the result into a box to prevent linebreaks from occurring between logo and name. The image is also put inside of a box, so that it does not appear in its own paragraph.
>  示例中，我们在 `show` 关键字后添加了我们想要特殊展示的文本字符串，然后是一个 `:`
>  之后，我们写下一个接收应该被作为参数展示的内容 `name`，我们可以在函数体内使用 `name`
>  函数体使用了 `box` 来避免在图像和文本之间出现换行，在 `box` 内的元素不会自成一段

The calls to the first box function and the image function did not require a leading `#` because they were not embedded directly in markup. When Typst expects code instead of markup, the leading `#` is not needed to access functions, keywords, and variables. This can be observed in parameter lists, function definitions, and [code blocks](https://typst.app/docs/reference/scripting/).
>  对第一个 `box` 函数和 `image` 函数的调用不需要使用 `#`，因为它们不是直接嵌入在 markup 中
>  当 Typst 期待 code 而不是 markup 时，就不需要 `#` 来访问函数、关键字、变量，一般这种情景会在参数列表、函数定义、code block 中出现

## Review
You now know how to apply basic formatting to your Typst documents. You learned how to set the font, justify your paragraphs, change the page dimensions, and add numbering to your headings with set rules. You also learned how to use a basic show rule to change how text appears throughout your document.

You have handed in your report. Your supervisor was so happy with it that they want to adapt it into a conference paper! In the next section, we will learn how to format your document as a paper using more advanced show rules and functions.
