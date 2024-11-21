# Basic usage
## Keyboard Bindings
Modifier keys are specified as `<c-x>`, `<m-x>`, and `<a-x>` for ctrl+x, meta+x, and alt+x respectively. For shift+x and ctrl-shift-x, just type `X` and `<c-X>`. See the next section for how to customize these bindings.
> 辅助按键/修饰按键分别用 `<c-x>/<m-x>/<a-x>` 表示 ctrl/meta/alt+x
> shift+x 和 ctrl+shift+x 对应 `X` 和 `<c-X>`

Once you have Vimium installed, you can see this list of key bindings at any time by typing `?`.

Navigating the current page:

```
?       show the help dialog for a list of all available keys
h       scroll left
j       scroll down
k       scroll up
l       scroll right
gg      scroll to top of the page
G       scroll to bottom of the page
d       scroll down half a page
u       scroll up half a page
f       open a link in the current tab
F       open a link in a new tab
r       reload
gs      view source
i       enter insert mode -- all commands will be ignored until you hit Esc to exit
yy      copy the current url to the clipboard
yf      copy a link url to the clipboard
gf      cycle forward to the next frame
gF      focus the main/top frame
```

> 在当前页面中导航
> ? - 帮助页面
> h/j/k/l - 向左/下/上/右滚动
> gg/G - 页面顶/底部
>  d/u - 向下/上半个页面 (down/up) 
> f/F - 在当前/新 tab 打开一个链接 (find)
> r - 重新加载页面 (reload)
> gs - 查看网页源代码 (source)
> i - 进入 insert 模式 (忽略所有命令，直到使用 esc 退出)
> yy - 拷贝当前页面的 url 到剪贴板 (yank)
> yf - 拷贝一个链接到剪贴板 (find and yank)
> gf - 循环前进到下一个帧 (frame)
> gF - 聚焦到顶部/主帧

Navigating to new pages:

```
o       Open URL, bookmark, or history entry
O       Open URL, bookmark, history entry in a new tab
b       Open bookmark
B       Open bookmark in a new tab
```

> 导航到新的页面
> o/O - 打开 URL、书签、历史条目在当前/新的 tab (open)
> b/B - 打开书签在当前/新的 tab (bookmark)

Using find:

```
/       enter find mode
          -- type your search query and hit enter to search, or Esc to cancel
n       cycle forward to the next find match
N       cycle backward to the previous find match
```

> 使用查找
> / - 进入查找模式，键入 query 之后，键入 enter 进行查找，键入 esc 退出
> n/N - 循环前进/退后到下一个/上一个匹配项

For advanced usage, see [regular expressions](https://github.com/philc/vimium/wiki/Find-Mode) on the wiki.

Navigating your history:

```
H       go back in history
L       go forward in history
```

> 在页面跳转历史中导航
> H/L - 在页面跳转历史中后退/前进

Manipulating tabs:

```
J, gT   go one tab left
K, gt   go one tab right
g0      go to the first tab. Use ng0 to go to n-th tab
g$      go to the last tab
^       visit the previously-visited tab
t       create tab
yt      duplicate current tab
x       close current tab
X       restore closed tab (i.e. unwind the 'x' command)
T       search through your open tabs
W       move current tab to new window
<a-p>   pin/unpin current tab
```

> 操纵标签页
> J/gT - 跳到左边的标签页
> K/gt - 跳到右边的标签页
> g0 - 跳到第一个标签页，使用 ng0 跳到第 n 个标签页
> g$ - 跳到最后一个标签页
> ^ - 回到上一个访问的标签页
> t - 创建标签页 (tab)
> yt - 复制当前标签页 (yank and tab)
> x - 关闭当前标签页
> X - 恢复关闭的标签页
> T - 在打开的标签页中搜索 (Tab)
> W - 将当前标签页移动到新的窗口 (Window)
> alt+p - 固定/取消固定当前标签页

Using marks:

```
ma, mA  set local mark "a" (global mark "A")
`a, `A  jump to local mark "a" (global mark "A")
``      jump back to the position before the previous jump
          -- that is, before the previous gg, G, n, N, / or `a
```

> 使用标记
> ma/mA - 设定局部/全局标记 "a"/"A" (mark)
> \` a/A - 跳转到局部/全局标记 "a"/"A"
> \`\` - 回到跳跃之前的位置

Additional advanced browsing commands:

```
]], [[  Follow the link labeled 'next' or '>' ('previous' or '<')
          - helpful for browsing paginated sites
<a-f>   open multiple links in a new tab
gi      focus the first (or n-th) text input box on the page. Use <tab> to cycle through options.
gu      go up one level in the URL hierarchy
gU      go up to root of the URL hierarchy
ge      edit the current URL
gE      edit the current URL and open in a new tab
zH      scroll all the way left
zL      scroll all the way right
v       enter visual mode; use p/P to paste-and-go, use y to yank
V       enter visual line mode
R       Hard reload the page (skip the cache)
```

> 另外的高级浏览命令
> ]]/\[\[ - 跟随由 `next/previous` 或 `>/<` 标记的链接，在浏览分页的网站中有用
> alt+f - 在新的标签页中打开多个(两个)链接
> gi - 聚焦到页面的第一个 (或第 n 个) 输入框，使用 tab 进行循环跳转 (go input)
> gu - 在 URL 层级中向上跳跃一级  (go up)
> gU - 跳跃到 URL 层级的根部
> ge - 编辑当前 URL
> gE - 编辑当前 URL 并且在新标签页打开
> zH - 滚动到最左边
> zL - 滚动到最右边
> v - 进入 visual 模式，使用 p/P 进行粘贴并跳转 (使用默认搜索引擎搜索剪贴板的的内容)，使用 y 进行拷贝 
> V - 进入 visual line 模式
> R - 硬刷新页面 (跳过缓存)，也就是强制从服务器获取所有资源，而不是从浏览器的缓存中加载

Vimium supports command repetition so, for example, hitting `5t` will open 5 tabs in rapid succession. `<Esc>` (or `<c-[>`) will clear any partial commands in the queue and will also exit insert and find modes.
> Vimium 支持命令重复，例如 `5t` 会连续打开 5 个标签页
> `esc` 或 `ctrl + [` 会清除队列中的任意部分命令，同时也会退出插入和查找模式

There are some advanced commands which aren't documented here; refer to the help dialog (type `?`) for a full list.

## Custom Key Mappings
You may remap or unmap any of the default key bindings in the "Custom key mappings" on the options page.

Enter one of the following key mapping commands per line:

- `map key command`: Maps a key to a Vimium command. Overrides Chrome's default behavior (if any).
- `unmap key`: Unmaps a key and restores Chrome's default behavior (if any).
- `unmapAll`: Unmaps all bindings. This is useful if you want to completely wipe Vimium's defaults and start from scratch with your own setup.

> 自定义按键映射的命令格式为：`map key command` ，这会覆盖 Chrome 的默认行为；`unmap key` 用于解除某个按键的映射，恢复为 Chrome 的默认行为；`unampAll` 则解除所有映射，一般用于自定义 Vimium 的所有行为

Examples:

- `map <c-d> scrollPageDown` maps ctrl+d to scrolling the page down. Chrome's default behavior of bringing up a bookmark dialog is suppressed.
- `map r reload` maps the r key to reloading the page.
- `unmap <c-d>` removes any mapping for ctrl+d and restores Chrome's default behavior.
- `unmap r` removes any mapping for the r key.

Available Vimium commands can be found via the "Show available commands" link near the key mapping box on the options page. The command name appears to the right of the description in parenthesis.

You can add comments to key mappings by starting a line with `"` or `#`.

The following special keys are available for mapping:

- `<c-*>`, `<a-*>`, `<s-*>`, `<m-*>` for ctrl, alt, shift, and meta (command on Mac) respectively with any key. Replace `*` with the key of choice.
- `<left>`, `<right>`, `<up>`, `<down>` for the arrow keys.
- `<f1>` through `<f12>` for the function keys.
- `<space>` for the space key.
- `<tab>`, `<enter>`, `<delete>`, `<backspace>`, `<insert>`, `<home>` and `<end>` for the corresponding non-printable keys.

Shifts are automatically detected so, for example, `<c-&>` corresponds to ctrl+shift+7 on an English keyboard.