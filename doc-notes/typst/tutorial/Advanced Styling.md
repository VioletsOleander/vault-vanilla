---
completed: true
version: 0.13.1
---
In the previous two chapters of this tutorial, you have learned how to write a document in Typst and how to change its formatting. The report you wrote throughout the last two chapters got a straight A and your supervisor wants to base a conference paper on it! The report will of course have to comply with the conference's style guide. Let's see how we can achieve that.

Before we start, let's create a team, invite your supervisor and add them to the team. You can do this by going back to the app dashboard with the back icon in the top left corner of the editor. Then, choose the plus icon in the left toolbar and create a team. Finally, click on the new team and go to its settings by clicking 'manage team' next to the team name. Now you can invite your supervisor by email.

![The team settings](https://typst.app/assets/docs/3-advanced-team-settings.png)

Next, move your project into the team: Open it, going to its settings by choosing the gear icon in the left toolbar and selecting your new team from the owners dropdown. Don't forget to save your changes!

Now, your supervisor can also edit the project and you can both see the changes in real time. You can join our [Discord server](https://discord.gg/2uDybryKPe) to find other users and try teams with them!

## The conference guidelines
The layout guidelines are available on the conference website. Let's take a look at them:

- The font should be an 11pt serif font
- The title should be in 17pt and bold
- The paper contains a single-column abstract and two-column main text
- The abstract should be centered
- The main text should be justified
- First level section headings should be 13pt, centered, and rendered in small capitals
- Second level headings are run-ins, italicized and have the same size as the body text
- Finally, the pages should be US letter sized, numbered in the center of the footer and the top right corner of each page should contain the title of the paper

We already know how to do many of these things, but for some of them, we'll need to learn some new tricks.

## Writing the right set rules
Let's start by writing some set rules for the document.

```
#set page(
  paper: "us-letter",
  header: align(right)[
    A fluid dynamic model for
    glacier flow
  ],
  numbering: "1",
)
#set par(justify: true)
#set text(
  font: "Libertinus Serif",
  size: 11pt,
)

#lorem(600)
```

>  `#set page` 设定了纸张大小、页面 header 格式、页码格式
>  `#set par` 设定了段落对齐方式
>  `#set text` 设定了文本字体、大小

![Preview](https://typst.app/assets/docs/p6Vtj1ockTIzscSwa5_kewAAAAAAAAAA.png)

You are already familiar with most of what is going on here. We set the text size to `11pt` and the font to Libertinus Serif. We also enable paragraph justification and set the page size to US letter.

The `header` argument is new: With it, we can provide content to fill the top margin of every page. In the header, we specify our paper's title as requested by the conference style guide. We use the `align` function to align the text to the right.
>  `#set page()` 中的 `header` 参数用于为每个页面的顶部边际添加内容，例如指定为 paper 的 title
>  我们还使用了 `align` 函数来将文本对齐到右边

Last but not least is the `numbering` argument. Here, we can provide a [numbering pattern](https://typst.app/docs/reference/model/numbering/) that defines how to number the pages. By setting into to `"1"`, Typst only displays the bare page number. Setting it to `"(1/1)"` would have displayed the current page and total number of pages surrounded by parentheses. And we could even have provided a completely custom function here to format things to our liking.
>  `#set page()` 中的 `numbering` 参数用于定义页码形式，例如 `"1"`
>  设定为 `"(1/1)"` 会展示当前页码和总页码，且以括号包围

## Creating a title and abstract
Now, let's add a title and an abstract. We'll start with the title. We center align it and increase its font weight by enclosing it in `*stars*`.

```
#align(center, text(17pt)[
  *A fluid dynamic model
  for glacier flow*
])
```

>  我们通过 `#align` 函数来居中标题，同时使用 `text` 函数设定了字号，并且用 `**` 表示加粗

![Preview](https://typst.app/assets/docs/EYkMw9AAwHWkDqrGblkKBgAAAAAAAAAA.png)

This looks right. We used the `text` function to override the previous text set rule locally, increasing the size to 17pt for the function's argument. 

Let's also add the author list: Since we are writing this paper together with our supervisor, we'll add our own and their name.

```
#grid(
  columns: (1fr, 1fr),
  align(center)[
    Therese Tungsten \
    Artos Institute \
    #link("mailto:tung@artos.edu")
  ],
  align(center)[
    Dr. John Doe \
    Artos Institute \
    #link("mailto:doe@artos.edu")
  ]
)
```

![Preview](https://typst.app/assets/docs/Iwl_3LT7ijX6dcpL71YOWAAAAAAAAAAA.png)

The two author blocks are laid out next to each other. We use the [`grid`](https://typst.app/docs/reference/layout/grid/ "`grid`") function to create this layout. With a grid, we can control exactly how large each column is and which content goes into which cell. The `columns` argument takes an array of [relative lengths](https://typst.app/docs/reference/layout/relative/) or [fractions](https://typst.app/docs/reference/layout/fraction/). In this case, we passed it two equal fractional sizes, telling it to split the available space into two equal columns. We then passed two content arguments to the grid function. The first with our own details, and the second with our supervisors'. We again use the `align` function to center the content within the column. The grid takes an arbitrary number of content arguments specifying the cells. Rows are added automatically, but they can also be manually sized with the `rows` argument.
>  我们使用 `grid` 创建网格，其中我们通过 `columns` 参数控制了每个列的大小，然后传递了两个 content 参数，其中使用了 `align` 来居中
>  `grid` 接收任意数量的参数，会随着参数增加而自动添加行

Now, let's add the abstract. Remember that the conference wants the abstract to be set ragged and centered.

```
...

#align(center)[
  #set par(justify: false)
  *Abstract* \
  #lorem(80)
]
```

![Preview](https://typst.app/assets/docs/4IdrVTeq86rbgvB-RNog6gAAAAAAAAAA.png)

Well done! One notable thing is that we used a set rule within the content argument of `align` to turn off justification for the abstract. This does not affect the remainder of the document even though it was specified after the first set rule because content blocks _scope_ styling. Anything set within a content block will only affect the content within that block.
>  我们在 `align` 的 content 参数中使用了一个 set rule 来关闭对齐，这个 rule 仅在 content block 的 scope 内生效

Another tweak could be to save the paper title in a variable, so that we do not have to type it twice, for header and title. We can do that with the `let` keyword:
>  `#let` 可以用于定义变量

```
#let title = [
  A fluid dynamic model
  for glacier flow
]

...

#set page(
  header: align(
    right + horizon,
    title
  ),
  ...
)

#align(center, text(17pt)[
  *#title*
])

...

```

![Preview](https://typst.app/assets/docs/ZKsZ2Eei-RUgPTLeJrMEYgAAAAAAAAAA.png)

After we bound the content to the `title` variable, we can use it in functions and also within markup (prefixed by `#`, like functions). This way, if we decide on another title, we can easily change it in one place.
>  在设定了变量之后，可以在 markup 中添加前缀 `#` 使用，以及在函数中直接使用

## Adding columns and headings
The paper above unfortunately looks like a wall of lead. To fix that, let's add some headings and switch our paper to a two-column layout. Fortunately, that's easy to do: We just need to amend our `page` set rule with the `columns` argument.
>  通过在 `#set page` rule 中添加 `columns` 选项，可以将页面改为多栏排版

By adding `columns: 2` to the argument list, we have wrapped the whole document in two columns. However, that would also affect the title and authors overview. To keep them spanning the whole page, we can wrap them in a function call to [`place`](https://typst.app/docs/reference/layout/place/). Place expects an alignment and the content it should place as positional arguments. Using the named `scope` argument, we can decide if the items should be placed relative to the current column or its parent (the page). There is one more thing to configure: If no other arguments are provided, `place` takes its content out of the flow of the document and positions it over the other content without affecting the layout of other content in its container:
>  为了避免双栏排版影响标题和作者，我们可以将它们放在 `place` 函数调用中
>  `place` 函数的位置参数是一个对齐参数和 content 参数，命名参数 `scope` 则用于决定 content 应该相对于当前列还是其父 (page) 放置
>  如果没有提供的命名参数，`place` 会将 content 从 document flow 中取出，将其放在其他 content 之上，不影响其他 content 的布局

```
#place(
  top + center,
  rect(fill: black),
)
#lorem(30)
```

![Preview](https://typst.app/assets/docs/30s7cU9X36lW286rJXE3RwAAAAAAAAAA.png)

If we hadn't used `place` here, the square would be in its own line, but here it overlaps the few lines of text following it. Likewise, that text acts like as if there was no square. To change this behavior, we can pass the argument `float: true` to ensure that the space taken up by the placed item at the top or bottom of the page is not occupied by any other content.
>  而指定 `float: true` 可以让 `place` 占据的位置不被任意其他 content 占据

```
#set page(
  paper: "us-letter",
  header: align(
    right + horizon,
    title
  ),
  numbering: "1",
  columns: 2,
)

#place(
  top + center,
  float: true,
  scope: "parent",
  clearance: 2em,
)[
  ...

  #par(justify: false)[
    *Abstract* \
    #lorem(80)
  ]
]

= Introduction
#lorem(300)

= Related Work
#lorem(200)
```

![Preview](https://typst.app/assets/docs/dAJVP8paZmMvnK23cMA_0AAAAAAAAAAA.png)

In this example, we also used the `clearance` argument of the `place` function to provide the space between it and the body instead of using the [`v`](https://typst.app/docs/reference/layout/v/) function. We can also remove the explicit `align(center, ..)` calls around the various parts since they inherit the center alignment from the placement.
>  我们还使用了 `clearance` 参数来为 place 区域和其他区域添加额外空格
>  我们也移除了额外的 `align` 调用，因为 content 会继承 placement 的 alignment 属性

Now there is only one thing left to do: Style our headings. We need to make them centered and use small capitals. Because the `heading` function does not offer a way to set any of that, we need to write our own heading show rule.
>  我们最后需要格式 heading，我们可以通过 `#show heading` 来提供 heading 的格式化方式

```
#show heading: it => [
  #set align(center)
  #set text(13pt, weight: "regular")
  #block(smallcaps(it.body))
]

...
```

![Preview](https://typst.app/assets/docs/ZJxJWdUySZNKlj1_Vn1NWgAAAAAAAAAA.png)

This looks great! We used a show rule that applies to all headings. We give it a function that gets passed the heading as a parameter. That parameter can be used as content but it also has some fields like `title`, `numbers`, and `level` from which we can compose a custom look. Here, we are center-aligning, setting the font weight to `"regular"` because headings are bold by default, and use the [`smallcaps`](https://typst.app/docs/reference/text/smallcaps/ "`smallcaps`") function to render the heading's title in small capitals.
>  在这个 rule 中，我们定义了一个函数，它接收 headings 作为参数

The only remaining problem is that all headings look the same now. The "Motivation" and "Problem Statement" subsections ought to be italic run in headers, but right now, they look indistinguishable from the section headings. We can fix that by using a `where` selector on our set rule: This is a [method](https://typst.app/docs/reference/scripting/#methods) we can call on headings (and other elements) that allows us to filter them by their level. We can use it to differentiate between section and subsection headings:
>  为了进一步完善 rule，可以使用 `where` selector

```
#show heading.where(
  level: 1
): it => block(width: 100%)[
  #set align(center)
  #set text(13pt, weight: "regular")
  #smallcaps(it.body)
]

#show heading.where(
  level: 2
): it => text(
  size: 11pt,
  weight: "regular",
  style: "italic",
  it.body + [.],
)
```

![Preview](https://typst.app/assets/docs/eBNymJDskGFkYVAXkF9cuAAAAAAAAAAA.png)

This looks great! We wrote two show rules that each selectively apply to the first and second level headings. We used a `where` selector to filter the headings by their level. We then rendered the subsection headings as run-ins. We also automatically add a period to the end of the subsection headings.

Let's review the conference's style guide:

- The font should be an 11pt serif font ✓
- The title should be in 17pt and bold ✓
- The paper contains a single-column abstract and two-column main text ✓
- The abstract should be centered ✓
- The main text should be justified ✓
- First level section headings should be centered, rendered in small caps and in 13pt ✓
- Second level headings are run-ins, italicized and have the same size as the body text ✓
- Finally, the pages should be US letter sized, numbered in the center and the top right corner of each page should contain the title of the paper ✓

We are now in compliance with all of these styles and can submit the paper to the conference! The finished paper looks like this:

![The finished paper](https://typst.app/assets/docs/3-advanced-paper.png)

## Review
You have now learned how to create headers and footers, how to use functions and scopes to locally override styles, how to create more complex layouts with the [`grid`](https://typst.app/docs/reference/layout/grid/ "`grid`") function and how to write show rules for individual functions, and the whole document. You also learned how to use the [`where` selector](https://typst.app/docs/reference/styling/#show-rules) to filter the headings by their level.

The paper was a great success! You've met a lot of like-minded researchers at the conference and are planning a project which you hope to publish at the same venue next year. You'll need to write a new paper using the same style guide though, so maybe now you want to create a time-saving template for you and your team?

In the next section, we will learn how to create templates that can be reused in multiple documents. This is a more advanced topic, so feel free to come back to it later if you don't feel up to it right now.