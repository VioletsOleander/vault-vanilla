# 1. JavaScript编程语言
## 1.1 简介
### 1.1.1 Javascript简介
**什么是Javascript？**
由javascript写出的程序称为脚本(scripts)，它们可以直接写在网页的HTML中，在页面加载时自动执行
脚本以纯文本形式提供，不需要特殊的准备或编译即可运行

javascript有自己的语言规范ECMAScript，现在它和Java没有任何关系

javascript可以在浏览器，服务器等任意搭载了JavaScript引擎的设备中运行

浏览器内嵌了JavaScript引擎，可以称为JavaScript虚拟机
各种不同的引擎有：
- V8 —— Chrome，Opera，Edga的JS引擎
- SpiderMonkey —— Firefox的JS引擎
- Chakra用于IE，JavaScriptCore、Nitro、SquirrelFish用于Safari

引擎的工作方式：
- 读取/解析脚本
- 将脚本转化/编译为机器语言
- 执行机器语言
引擎会对流程中的每个阶段进行优化，它甚至可以在编译的脚本运行时监视它，分析流经该脚本的数据，并根据获得的信息进一步优化机器代码

**浏览器中的 JavaScript 能做什么？**
JS不提供对内存和CPU的底层访问，因此是“安全”的
JS的能力取决于运行环境，如Node.js支持允许 JavaScript 读取/写入任意文件，执行网络请求等的函数
浏览器中的JS可以做与网页操作、用户交互和 Web 服务器相关的所有事情：
- 在网页中添加新的 HTML，修改网页已有内容和网页的样式
- 响应用户的行为，响应鼠标的点击，指针的移动，按键的按动
- 向远程服务器发送网络请求，下载和上传文件([AJAX](https://en.wikipedia.org/wiki/Ajax_(programming)) 和 [COMET](https://en.wikipedia.org/wiki/Comet_(programming)) 技术)
- 获取或设置 cookie，向访问者提出问题或发送消息
- 记住客户端的数据(“本地存储”)

**浏览器中的 JavaScript 不能做什么？**
在浏览器中的JS的能力是受限的，目的是防止恶意网页获取用户私人信息或损害用户数据
限制包括：
- 网页中的 JavaScript 不能读、写、复制和执行硬盘上的任意文件，它没有直接访问操作系统的功能
	现代浏览器允许 JavaScript 做一些文件相关的操作，但是这个操作是受到限制的，仅当用户做出特定的行为，JavaScript 才能操作这个文件
	例如，用户把文件“拖放”到浏览器中，或者通过 `<input>` 标签选择了文件
	有很多与相机/麦克风和其它设备进行交互的方式，但是这些都需要获得用户的明确许可
- 不同的标签页(tabs)/窗口(windows)之间通常互不了解
	有时候，也会有一些联系，例如一个标签页通过 JavaScript 打开的另外一个标签页，但即使在这种情况下，如果两个标签页打开的不是同一个网站(域名、协议或者端口任一不相同的网站)，它们都不能相互通信
	这就是所谓的“同源策略”，为了通过“同源策略”相互通信，两个标签页必须都包含一些处理这个问题的特定的JS代码，并均允许数据交换
	这个限制也是为了用户的信息安全，例如，用户打开的 `http://anysite.com` 网页必须不能访问 `http://gmail.com`(另外一个标签页打开的网页)因此也不能从那里窃取信息
- JavaScript 可以轻松地通过互联网与当前页面所在的服务器进行通信。但是从其他网站/域的服务器中接收数据的能力被削弱了
	实际上可以，但是需要来自远程服务器的明确协议(在 HTTP header 中)，这也是为了用户的信息安全

如果在浏览器环境外，如在服务器上使用 JavaScript，则不存在此类限制
现代浏览器还允许安装可能会要求扩展权限的插件/扩展

**是什么使得 JavaScript 与众不同？**
- 与 HTML/CSS 完全集成
- 简单的事，简单地完成
- 被所有的主流浏览器支持，并且默认开启
JS是用于创建浏览器界面的使用最广泛的工具，此外，JS还可用于创建服务器、移动端应用程序等

**JavaScript “上层”的语言**
现代化的工具使得编译速度非常快且透明，实际上允许开发者使用另一种语言编写代码并会将其“自动转换”为 JavaScript
这些语言在浏览器中执行之前，都会被 编译/转化成 JavaScript：
- [CoffeeScript](https://coffeescript.org/) 
	是 JavaScript 的一种语法糖，它引入了更加简短的语法，使我们可以编写更清晰简洁的代码，通常，Ruby 开发者喜欢它。
- [TypeScript](https://www.typescriptlang.org/) 
	专注于添加“严格的数据类型”以简化开发，以更好地支持复杂系统的开发
	由微软开发
- [Flow](https://flow.org/) 
	也添加了数据类型，但是以一种不同的方式
	由 Facebook 开发
- [Dart](https://www.dartlang.org/) 
	是一门独立的语言，它拥有自己的引擎，该引擎可以在非浏览器环境中运行，例如手机应用
	它也可以被编译成 JavaScript
	由 Google 开发。
- [Brython](https://brython.info/) 
	是一个 Python 到 JavaScript 的转译器，让我们可以在不使用 JavaScript 的情况下，以纯 Python 编写应用程序。
- [Kotlin](https://kotlinlang.org/docs/reference/js-overview.html) 
	是一个现代、简洁且安全的编程语言，编写出的应用程序可以在浏览器和 Node 环境中运行

**总结**
- JavaScript 最开始是专门为浏览器设计的一门语言，但是现在也被用于很多其他的环境
- JavaScript 作为被应用最广泛的浏览器语言，且与 HTML/CSS 完全集成，具有独特的地位
- 有很多其他的语言可以被转译成 JavaScript，这些语言还提供了更多的功能

### 1.1.2 手册与规范
**规范**
ECMA-262 规范包含了大部分深入的、详细的、规范化的关于 JavaScript 的信息，这份规范明确地定义了这门语言
每年都会发布一个新版本的规范，最新的规范草案请见 [https://tc39.es/ecma262/](https://tc39.es/ecma262/)

**手册**
- MDN(Mozilla) JavaScript 索引 
	是一个带有用例和其他信息的主要的手册，它是一个获取关于个别语言函数、方法等深入信息的很好的信息来源
    可以在 [https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference) 阅读

**兼容性表**
JavaScript 是一门还在发展中的语言，定期会添加一些新的功能，要查看它们在基于浏览器的引擎及其他引擎中的支持情况，请看：
- [每个功能的支持表](https://caniuse.com/) 
	例如，查看哪个引擎支持现代加密(cryptography)函数：[https://caniuse.com/#feat=cryptography](https://caniuse.com/#feat=cryptography)
- [一份列有语言功能以及引擎是否支持这些功能的表格](https://kangax.github.io/compat-table)

### 1.1.3 代码编辑器
IDE或Lightweight Editors

### 1.1.4 开发者控制台
为了发现错误并获得一些与脚本相关且有用的信息，浏览器内置了“开发者工具”

通常，开发者倾向于使用 Chrome 或 Firefox 进行开发，因为它们有最好的开发者工具

## 1.2 JavaScript基础知识
### 1.2.1 Hello, world!
首先，让我们看看如何将脚本添加到网页上

**“script” 标签**
我们可以使用 `<script>` 标签将JS程序插入到 HTML 文档的几乎任何位置：
```html
<!DOCTYPE HTML>
<html>
<body>
  <p> Before the script</p>
  <script>
    alert("Hello,world!");
  </script>
  <p> After the script</p>
</body>
</html>
```
`<script>` 标签中包裹了 JavaScript 代码，当浏览器遇到 `<script>` 标签，代码会自动运行

`<script>` 标签有一些现在很少用到的特性(attribute)：
- `type` 特性：`<script type=…>`
	在老的 HTML4 标准中，要求 script 标签有 `type` 特性，通常是 `type="text/javascript"`
	这样的特性声明现在已经不再需要，而且，现代 HTML 标准已经完全改变了此特性的含义
- `language` 特性：`<script language=…>`
	这个特性是为了显示脚本所用的语言
	这个特性现在已经没有任何意义，因为语言默认就是 JavaScript

**外部脚本**
如果我们有大量的 JavaScript 代码，我们可以将它放入一个单独的文件

而脚本文件则可以通过 `src` 特性(attribute)添加到 HTML 文件中：
```html
<script src="/path/to/script.js"></script>
```
这里的 `/path/to/script.js` 是从网站根目录开始的绝对路径
当然也可以提供相对于当前页面的相对路径，例如 `src ="script.js"`，就像 `src="./script.js"`，表示当前文件夹中的 `"script.js"` 文件

我们也可以提供一个完整的 URL 地址，例如：
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.11/lodash.js"></script>
```

要附加多个脚本，使用多个标签即可：
```html
<script src="/js/script1.js"></script>
<script src="/js/script2.js"></script>
```

一般来说，只有最简单的脚本才嵌入到 HTML 中，更复杂的脚本存放在单独的文件中，浏览器会下载独立文件，然后将它保存到浏览器的缓存中，之后，其他页面想要相同的脚本就可以直接缓存中获取，所以文件实际上只会下载一次，以节省流量，并使得页面加载更快。

如果设置了 `src` 特性，`script` 标签内容将会被忽略，如：
```html
<script src="file.js">
  alert(1); // 此内容会被忽略，因为设定了 src
</script>
```
我们必须进行选择，要么使用外部的 `<script src="…">`，要么使用正常包裹代码的 `<script>`
为了让上面的例子工作，我们可以将它分成两个 `<script>` 标签：
```html
<script src="file.js"></script>
<script>
  alert(1);
</script>
```

**总结**
- 我们可以使用一个 `<script>` 标签将 JavaScript 代码添加到页面中
- type 和 language 特性(attribute)不是必需的
- 外部的脚本可以通过 `<script src="path/to/script.js"></script>` 的方式插入

### 1.2.2 代码结构
**语句(Statements)**
语句是执行行为(action)的语法结构和命令，如`alert('Hello, world!')` 是用来显示消息的语句，语句之间可以使用分号进行分割

通常，每条语句独占一行，以提高代码的可读性：
```javascript
alert('Hello');
alert('World');
```

**分号**
当存在换行符(line break)时，在大多数情况下也可以省略分号，如：
```javascript
alert('Hello')
alert('World')
```
JavaScript 将换行符理解成“隐式”的分号，这也被称为“自动分号插入”

在大多数情况下，换行意味着一个分号，但也有很多换行并不是分号的例子：
```javascript
alert(3 +
1
+ 2);
```
代码输出 `6`，因为 JavaScript 并没有在这里插入分号，因为如果一行以加号 `"+"` 结尾，那么显然这是一个“不完整的表达式”(incomplete expression)，不需要分号
所以，这个例子得到了预期的结果

但存在 JavaScript 无法确定是否真的需要自动插入分号的情况，如：
```javascript
alert("Hello");

[1, 2].forEach(alert);
```
这段代码的运行结果：先显示 `Hello`，然后显示 `1`，然后 `2`

如果删除 alert 语句后的分号：
```javascript
alert("Hello")

[1, 2].forEach(alert);
```
这次 JavaScript 引擎并没有假设在方括号 `[...]` 前有一个分号，对于引擎来说，它是这样的：`alert("Hello")[1, 2].forEach(alert);`

因此我们需要在 `alert` 后面加一个分号，代码才能正常运行

因此即使语句被换行符分隔了，依然建议在它们之间加分号，这个规则被社区广泛采用

**注释**
我们可以在脚本的任何地方添加注释，它们并不会影响代码的执行，因为引擎会直接忽略它们

单行注释以两个正斜杠字符 `//` 开始：
```javascript
// 这行注释独占一行
alert('Hello');

alert('World'); // 这行注释跟随在语句后面
```

多行注释以一个正斜杠和星号开始 “/\*” 并以一个星号和正斜杠结束 “\*/”：
```javascript
/* 
这是一个多行注释
*/
alert('Hello');
alert('World');
```
所以如果我们在 `/* … */` 中放入代码，并不会执行，可以利用注释很方便地临时禁用代码

在大多数的编辑器中，一行代码可以使用 Ctrl+/ 快捷键进行单行注释，使用 Ctrl+Shift+/ 的快捷键可以进行多行注释

不要在 /*...*/ 内嵌套另一个 /*...*/，下面这段代码会报错而无法执行：
```javascript
/*
  /* 嵌套注释 ?!? */
*/
alert( 'World' );
```

注释会增加代码总量，但有很多工具可以帮你在把代码部署到服务器之前缩减代码，这些工具会移除注释，这样注释就不会出现在发布的脚本中，所以注释对我们的生产没有任何负面影响

### 1.2.3 现代模式，"use strict"
长久以来，JavaScript 不断向前发展且并未带来任何兼容性问题，因为新的特性被加入，旧的功能也没有改变，这么做有利于兼容旧代码，但缺点是 JavaScript 创造者的任何错误或不完善的决定也将永远被保留

这种情况一直持续到 2009 年 ECMAScript 5 (ES5) 的出现，ES5 规范增加了新的语言特性
并且修改了一些已经存在的特性
而为了保证旧的功能能够使用，大部分的修改是默认不生效的
我们可以使用一个特殊的指令 —— `"use strict"` 来明确地激活这些特性

**“use strict”**
这个指令看上去像一个字符串 `"use strict"`或者 '`use strict'`
当它处于脚本文件的顶部时，则整个脚本文件都将以“现代”模式进行工作。

比如：
```javascript
"use strict";

// 代码以现代模式工作
...
```
`"use strict"` 可以被放在函数体的开头，这样则可以只在该函数中启用严格模式，但通常人们会在整个脚本中启用严格模式

注意：
(1) 要确保 `"use strict"` 出现在脚本的最顶部，否则严格模式可能无法启用，如这里的严格模式就没有被启用：
```javascript
alert("some code");
// 下面的 "use strict" 会被忽略，必须在最顶部。

"use strict";

// 严格模式没有被激活
```
只有注释可以出现在 `"use strict"` 的上面

(2) 没有类似于 `"no use strict"` 这样的指令可以使程序返回默认模式，一旦进入了严格模式，就没有回头路了

**浏览器控制台**
当使用开发者控制台(developer console)运行代码时，它默认是不启动 `use strict` 的

可以尝试搭配使用 Shift+Enter 按键去输入多行代码，然后将 `use strict`放在代码最顶部，就像这样：
```console
'use strict'; <Shift+Enter 换行>
//  ...你的代码
<按下 Enter 以运行>
```

**我们应该使用 “use strict” 吗？**
现代 JavaScript 支持 “class” 和 “module” —— 高级语言结构，它们会自动启用 `use strict` ，因此，如果我们使用它们，则无需添加 `"use strict"` 指令

因此，目前我们欢迎将 `"use strict";` 写在脚本的顶部
当之后我们的代码全都写在了 class 和 module 中时，则可以将 `"use strict";` 这行代码省略掉

本教程的所有例子都默认采用严格模式，除非特别指定（非常少）

### 1.2.4 变量
**变量**
在 JavaScript 中创建一个变量，我们需要用到 `let` 关键字

创建/声明/定义一个名称为 “message” 的变量：`let message;`
通过赋值运算符 = 为变量添加一些数据：
```javascript
let message;

message = 'Hello'; // 将字符串 'Hello' 保存在名为 message 的变量中
```
现在这个字符串已经保存到与该变量相关联的内存区域了，我们可以通过使用该变量名称访问它：
```js
let message;
message = 'Hello!';

alert(message); // 显示变量内容
```
我们可以将变量定义和赋值合并成一行：
```js
let message = 'Hello!'; // 定义变量，并且赋值

alert(message); // Hello!
```
可以在一行中声明多个变量：`let user = 'John', age = 25, message = 'Hello';`
多行变量声明有点长，但更容易阅读：
```js
let user = 'John';
let age = 25;
let message = 'Hello';
```

**一个现实生活的类比**
将变量想象成一个“数据”的盒子，盒子上有一个唯一的标注盒子名字的贴纸

例如，变量 `message` 可以被想象成一个标有 `"message"` 的盒子，盒子里面的值为 `"Hello!"`
我们可以在盒子内放入任何值，并且，这个盒子的值，我们想改变多少次，就可以改变多少次

注意，当值改变的时候，之前的数据就被从变量中删除了

我们还可以声明两个变量，然后将其中一个变量的数据拷贝到另一个变量：
```js
let hello = 'Hello world!';

let message;

// 将字符串 'Hello world' 从变量 hello 拷贝到 message
message = hello;

// 现在两个变量保存着相同的数据
alert(hello); // Hello world!
alert(message); // Hello world!
```

注意：
一个变量应该只被声明一次，对同一个变量进行重复声明会触发 error：
```js
let message = "This";

// 重复 'let' 会导致 error
let message = "That"; // SyntaxError: 'message' has already been declared
```
因此，我们对同一个变量应该只声明一次，之后在不使用 let 的情况下对其进行引用

**变量命名**
JavaScript 的变量命名有两个限制：
1. 变量名称必须仅包含字母、数字、符号 `$` 和 `_`
2. 首字符必须非数字
如果命名包括多个单词，通常采用驼峰式命名法(camelCase)

下面的命名是有效的：
```js
let $ = 1; // 使用 "$" 声明一个变量
let _ = 2; // 现在用 "_" 声明一个变量

alert($ + _); // 3
```
下面的变量命名不正确：
```js
let 1a; // 不能以数字开始

let my-name; // 连字符 '-' 不允许用于变量命名
```

变量命区分大小写，命名为 `apple` 和 `APPLE` 的变量是不同的两个变量

保留字无法用作变量命名，因为它们被用于编程语言本身了

**常量**
声明一个常数（不变）变量，可以使用 `const` 而非 `let`：
`const myBirthday = '18.04.1982';`
常量不能被修改，如果尝试修改就会发现报错

一个普遍的做法是将常量用作别名，以便记住那些在执行之前就已知的难以记住的值，一般使用大写字母和下划线来命名这些常量，例如，让我们以所谓的“web”（十六进制）格式为颜色声明常量：
```js
const COLOR_RED = "#F00";
const COLOR_GREEN = "#0F0";
const COLOR_BLUE = "#00F";
const COLOR_ORANGE = "#FF7F00";

// ……当我们需要选择一个颜色
let color = COLOR_ORANGE;
alert(color); // #FF7F00
```
大写命名的常量仅用作“硬编码(hard-coded)”值的别名，这些常量的值在一般代码执行之前就已知了(比如红色的十六进制值)

在执行期间被“计算”出来，但初始赋值之后就不会改变的常量采用常规命名，例如：`const pageLoadTime = /* 网页加载所需的时间 */;` 中 `pageLoadTime` 的值在页面加载之前是未知的，所以采用常规命名，但是它仍然是个常量，因为赋值之后不会改变

注意：
在 JavaScript 中，倾向于声明一个新的变量，而不是重用现有的变量，因为常常重用现有的变量会节省了一点变量声明的时间，但容易让我们却在调试代码的时候损失数十倍时间，因此额外声明一个变量绝对是利大于弊的，而现代的 JavaScript 压缩器和浏览器都能够很好地对代码进行优化，所以不会产生性能问题，为不同的值使用不同的变量可以帮助引擎对代码进行优化

**总结**
我们可以使用 `var`、`let `或 `const` 声明变量来存储数据
- `let` — 现代的变量声明方式
- `var` — 老旧的变量声明方式，一般情况下，我们不会再使用它
- `const` — 类似于 `let`，但是变量的值无法被修改
变量应当以一种容易理解变量内部是什么的方式进行命名

### 1.2.5 数据类型
在 JavaScript 中有 8 种基本的数据类型(7 种原始类型和 1 种复杂类型)

我们可以将任何类型的值存入变量，例如，一个变量可以在前一刻是个字符串，下一刻就存储一个数字：
```js
// 没有错误
let message = "hello";
message = 123456;
```
允许这种操作的编程语言，例如 JavaScript，被称为“动态类型”(dynamically typed)的编程语言
在动态类型的编程语言中，虽然有不同的数据类型，但是我们定义的变量并不会在定义后被限制为某一数据类型

**Number类型**
_number_ 类型代表整数和浮点数：
```js
let n = 123;
n = 12.345;
```

除了常规的数字，所谓的“特殊数值(special numeric values)”，包括`Infinity`、`-Infinity` 和 `NaN` 也属于 _number_ 类型
- `Infinity` 代表数学概念中的无穷大，是一个比任何数字都大的特殊值
	我们可以通过除以 0 来得到它：`alert( 1 / 0 ); // Infinity`
	或者在代码中直接使用它：`alert( Infinity ); // Infinity`
- `NaN` 代表一个计算错误，它是一个不正确的或者一个未定义的数学操作所得到的结果
	如：`alert( "not a number" / 2 ); // NaN，这样的除法是错误的`
	
	`NaN` 是粘性的。任何对 `NaN` 的进一步数学运算都会返回 `NaN`：
	```js
	alert( NaN + 1 ); // NaN
	alert( 3 * NaN ); // NaN
	alert( "not a number" / 2 - 1 ); // NaN
	```
	所以，如果在数学表达式中有一个 `NaN`，会被传播到最终结果(只有一个例外：`NaN ** 0` 结果为 `1` )
	
	在 JavaScript 中做数学运算是安全的，即我们可以做任何事：除0，将非数字字符串视为数字等等，而脚本永远不会因为一个fatal error而停止，最坏的情况下，我们会也得到 `NaN` 的结果

**BigInt 类型**
在 JavaScript 中，“number” 类型无法安全地表示大于$(2^{53}-1)$或小于$-(2^{53}-1)$的整数
`BigInt` 类型是最近被添加到 JavaScript 语言中的，用于表示任意长度的整数，可以通过将 `n` 附加到整数字段的末尾来创建 `BigInt` 值：
```js
// 尾部的 "n" 表示这是一个 BigInt 类型
const bigInt = 1234567890123456789012345678901234567890n;
```

**String 类型**
JavaScript 中的字符串必须被括在引号里，有三种包含字符串的方式：
1. 双引号：`"Hello"`
2. 单引号：`'Hello'`
3. 反引号：`` `Hello` ``
```js
let str = "Hello";
let str2 = 'Single quotes are ok too';
let phrase = `can embed another ${str}`;
```

其中反引号是”功能扩展“引号，它允许我们通过将变量和表达式包装在 `${…}` 中，来将表达式的值嵌入到字符串中：
```js
let name = "John";

// 嵌入一个变量
alert( `Hello, ${name}!` ); // Hello, John!

// 嵌入一个表达式
alert( `the result is ${1 + 2}` ); // the result is 3
```
`${…}` 内的表达式会被计算，计算结果会成为字符串的一部分
可以在 `${…}` 内放置任何东西，诸如名为 `name` 的变量，或者诸如 `1 + 2` 的算数表达式

该格式仅仅在反引号内有效，其他引号不会解析这种嵌入

在一些语言中，单个字符有一个特殊的 “character” 类型，在 C 语言和 Java 语言中被称为 “char”，但在 JavaScript 中没有这种类型，只有一种 `string` 类型，一个字符串可以包含零个(为空)、一个或多个字符

**Boolean 类型(逻辑类型)**
boolean 类型仅包含两个值：`true` 和 `false`

如：
```js
let nameFieldChecked = true; // yes, name field is checked
let ageFieldChecked = false; // no, age field is not checked
```

比较的结果是布尔值：
```js
let isGreater = 4 > 1;

alert( isGreater ); // true
```


**null 值**
特殊的 `null` 值不属于上述任何一种类型，它自己构成了一个独立的类型，只包含 `null` 值

如：`let age = null;` ，表示 `age` 的值是未知的

相比较于其他编程语言，JavaScript 中的 `null` 仅仅是一个代表“无”、“空”或“值未知”的特殊值，而不是一个“对不存在的 `object` 的引用”或者 “null 指针”


**undefined 值**
特殊值 `undefined` 和 `null` 一样自己构成了一个类型，`undefined` 的含义是未被赋值，如果一个变量已被声明，但未被赋值，那么它的值就是 `undefined`：
```js
let age;

alert(age); // 弹出 "undefined"
```

可以显式地将 undefined 赋值给变量：
```js
let age = 100;

// 将值修改为 undefined
age = undefined;

alert(age); // "undefined"
```
但是不建议这样做

通常，使用 null 将一个“空”或者“未知”的值写入变量中，而 undefined 则保留作为未进行初始化的事物的默认初始值

**Object 类型和 Symbol 类型**
其他所有的数据类型都被称为“原始类型”，因为它们的值只包含一个单独的内容(字符串、数字或者其他)，`object` 则用于储存数据集合和更复杂的实体

`symbol` 类型用于创建对象的唯一标识符

**typeof 运算符**
`typeof` 运算符返回参数的类型(以字符串的形式返回参数的数据类型)：
```js
typeof undefined // "undefined"

typeof 0 // "number"

typeof 10n // "bigint"

typeof true // "boolean"

typeof "foo" // "string"

typeof Symbol("id") // "symbol"

typeof Math // "object"  (1)

typeof null // "object"  (2)

typeof alert // "function"  (3)
```
其中：
- `Math` 是一个提供数学运算的内建 `object`
- `typeof null` 的结果为 `"object"`，这是官方承认的 `typeof` 的错误
	这个问题来自于 JavaScript 语言的早期阶段，并为了兼容性而保留了下来，因此需要注意 `null` 绝对不是一个 `object` ， `null` 有自己的类型，它是一个特殊值，`typeof` 的行为在这里是错误的
- `typeof alert` 的结果是 `"function"` ， 因为 `alert` 在 JavaScript 语言中是一个函数
	实际上在 JavaScript 语言中没有一个特别的 “function” 类型。函数隶属于 `object` 类型
	但是 `typeof` 会对函数区分对待，并返回 `"function"`。这也是来自于 JavaScript 语言早期的问题，从技术上讲，这种行为是不正确的，但在实际编程中却非常方便

另一种语法：`typeof(x)` 与 `typeof x` 的作用是相同的

简单地说， `typeof`是一个操作符，不是一个函数，因此这里的括号事实上不是 `typeof` 的一部分，它是数学运算分组的括号

通常，这样的括号里包含的是一个数学表达式，例如 `(2 + 2)`，但这里它只包含一个参数 `(x)`
从语法上讲，加上括号允许我们在 `typeof` 运算符和其参数之间不打空格，有些人喜欢这样的风格

一般 `typeof x` 语法更为常见

**总结**
JavaScript 中有八种基本的数据类型(前七种为基本数据类型，也称为原始数据类型，而 `object` 为复杂数据类型)

- 七种原始数据类型：
    - `number` 用于任何类型的数字：整数或浮点数，整数的范围在 $\pm(2^{53}-1)$内
    - `bigint` 用于任意长度的整数
    - `string` 用于字符串：一个字符串可以包含 0 个或多个字符，所以没有单独的单字符类型
    - `boolean` 用于 `true` 和 `false`
    - `null` 用于未知的值 —— 只有一个 `null` 值的独立类型
    - `undefined` 用于未定义的值 —— 只有一个 `undefined` 值的独立类型
    - `symbol` 用于唯一的标识符
- 以及一种非原始数据类型：
    - `object` 用于更复杂的数据结构

我们可以通过 `typeof` 运算符查看存储在变量中的数据的类型是什么：
- 通常用 `typeof x`，但 `typeof(x)` 也可行
- `typeof` 运算符会以字符串的形式返回类型名称，例如 `"string"`
- `typeof null` 会返回 `"object"` —— 这是 JavaScript 编程语言的一个错误，实际上它并不是一个 `object`

### 1.2.6 交互：alert、prompt 和 confirm
与用户交互的函数：`alert`，`prompt` 和`confirm`

**alert**
该函数用于显示信息，它会显示一条信息弹窗，并等待用户按下 “确认”
例如：
```js
alert("Hello");
```
弹出的这个带有信息的小窗口被称为模态窗(modal window)
“模态” 意味着在用户处理完模态窗口之前，用户不能与页面的其他部分进行交互，例如点击其他按钮等

在本例中，是指在用户点击“确定”按钮之前

**prompt**
该函数用于接收输入，`prompt` 函数接收两个参数：
```js
result = prompt(title, [default]);
```
`title` ：显示给用户的文本
`default` ：可选的第二个参数，指定 input 框的初始值

>**语法中的方括号 `[...]`**
>上述语法中 `default` 周围的方括号表示该参数是可选的，不是必需的

`prompt` 用于提示用户输入，它会使得浏览器显示一个带有文本消息(输入提示)的模态窗口，还有 input 框和确定/取消按钮

用户在提示输入栏中输入内容，然后按“确定”键，该文本就会存储于`result` 中
用户按取消键或按 Esc 键取消输入，`result` 为 `null`

如：
```js
let age = prompt('How old are you?', 100);

alert(`You are ${age} years old!`); // You are 100 years old!
```

注意 `prompt` 以字符串的形式返回用户的输入
**confirm**
该函数用于提示用户确认，语法：
```js
result = confirm(question);
```
`confirm` 函数显示一个带有 `question` 以及确定/取消两个按钮的模态窗口

用户点击确定，函数返回 `true`，点击取消，函数返回 `false`

如：
```js
let isBoss = confirm("Are you the boss?");

alert( isBoss ); // 如果“确定”按钮被按下，则显示 true
```

**总结**
与用户交互的3个浏览器的特定函数：
- `alert` 
	显示信息
- `prompt`
	显示信息要求用户输入文本
	点击确定返回文本，点击取消或按下 Esc 键返回 `null`
- `confirm`
	显示信息等待用户点击确定或取消
	点击确定返回 `true`，点击取消或按下 Esc 键返回 `false`

这些方法都是模态的：它们暂停脚本的执行，并且不允许用户与该页面的其余部分进行交互，直到窗口被解除

上述所有方法共有两个限制：
1. 模态窗口的确切位置由浏览器决定，通常在页面中心
2. 窗口的确切外观也取决于浏览器，我们不能修改它

### 1.2.7 类型转换
大多数情况下，运算符和函数会自动将赋予它们的值转换为正确的类型
比如：
- `alert` 会自动将任何值都转换为字符串类型以进行显示
- 算术运算符会将值转换为数字类型
	比如，当把除法 `/` 用于非 number 类型：`alert( "6" / "2" ); 
	得到3， 在除法中，string类型的值被自动转换成number类型后进行计算`

在某些情况下，我们需要将值显式地转换为我们期望的类型

**字符串转换**
转换发生在输出内容的时候
我们也可以显式地调用 `String(value)` 来将 `value` 转换为字符串类型：
```js
let value = true;
alert(typeof value); // boolean

value = String(value); // 现在，值是一个字符串形式的 "true"
alert(typeof value); // string
```
还比如：`false` 变成 `"false"`，`null` 变成 `"null"` 等

**数字类型转换**
转换发生在进行算术操作时
可以使用 `Number(value)` 显式地将这个 `value` 转换为 number 类型：
```js
let str = "123";
alert(typeof str); // string

let num = Number(str); // 变成 number 类型， 123

alert(typeof num); // number
```

如果该字符串不是一个有效的数字，转换的结果会是 `NaN` ：
```js
let age = Number("an arbitrary string instead of a number");

alert(age); // NaN，转换失败
```

number 类型转换规则：

| 值              | 变成……                                                                                                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `undefined`     | `NaN`                                                                                                                                                                                            |
| `null`          | `0`                                                                                                                                                                                              |
| `true 和 false` | `1`和`0`                                                                                                                                                                                      |
| `string`        | 去掉首尾空白字符（空格、换行符 `\n`、制表符 `\t` 等）后的纯数字字符串中含有的数字<br>如果剩余字符串为空，则转换结果为 `0` <br>如果剩余字符串不是纯数字字符串，类型转换出现 error ，返回 `NaN` |

如：
```js
alert( Number("   123   ") ); // 123
alert( Number("123z") );      // NaN
alert( Number(true) );        // 1
alert( Number(false) );       // 0
```

大多数数学运算符都执行这种转换

**布尔类型转换**
转换发生在进行逻辑操作时
也可以通过调用 Boolean(value) 显式地进行转换

转换规则如下：
- 直观上为“空”的值( `0`、空字符串、`null`、`undefined` 和 `NaN` )将变为 `false`
- 其他值变成 `true`

比如：
```js
alert( Boolean(1) ); // true
alert( Boolean(0) ); // false

alert( Boolean("hello") ); // true
alert( Boolean("") ); // false
```


PHP视 `"0"` 为 `false`，在 JavaScript 中，非空的字符串总是 `true`
```js
alert( Boolean("0") ); // true
alert( Boolean(" ") ); // 空格，也是 true（任何非空字符串都是 true）
```

### 1.2.8 基础运算符，数学运算

**术语：“一元运算符(unary)”，“二元运算符(binary)”，“运算元(operand)”**
- 运算元(operand)
	运算符应用的对象
	比如说乘法运算 `5 * 2`，有两个运算元：左运算元 `5` 和右运算元 `2`
	有时候人们也称其为“参数(argument)”而不是“运算元”
- 一元运算符(unary)
	如果一个运算符对应的只有一个运算元，那么它是一元运算符
	比如说一元负号运算符(unary negation) `-`，它的作用是对数字进行正负转换：
	`let x = 1; x = -x; alert(x); // 显示-1，，一元负号运算符生效`
- 二元运算符(binary)
	如果一个运算符拥有两个运算元，那么它是 **二元运算符**
	减号还存在二元运算符形式：
	`let x = 1, y = 3; alert( y - x ); // 2，二元运算符减号做减运算`
在上面的示例中，我们使用一个相同的符号表征了两个不同的运算符：
负号运算符，即反转符号的一元运算符
减法运算符，是从另一个数减去一个数的二元运算符

**数学运算**
JavaScript以下数学运算：
- 加法 `+`
- 减法 `-`
- 乘法 `*`
- 除法 `/`
- 取余 `%`
- 求幂 `**`

**取余 %**
`a % b` 的结果是 `a` 整除 `b` 的余数
例如：
```js
alert( 5 % 2 ); // 1，5 除以 2 的余数
alert( 8 % 3 ); // 2，8 除以 3 的余数
```


**求幂 \*\***
求幂运算 `a ** b` 将 `a` 提升至 `a` 的 `b` 次幂
在数学运算中我们将其表示为$a^b$
例如：
```js
alert( 2 ** 2 ); // 2² = 4
alert( 2 ** 3 ); // 2³ = 8
alert( 2 ** 4 ); // 2⁴ = 16
```

就像在数学运算中一样，幂运算也适用于非整数
例如，平方根是指数为$\frac 1 2$的幂运算：
```js
alert( 4 ** (1/2) ); // 2（1/2 次方与平方根相同)
alert( 8 ** (1/3) ); // 2（1/3 次方与立方根相同)
```

**用二元运算符 + 连接字符串**
如果加号 `+` 被应用于字符串，它将合并/连接各个字符串：
```js
let s = "my" + "string";
alert(s); // mystring
```

注意：只要任意一个运算元是字符串，那么另一个运算元也将被转化为字符串
举个例子：
```js
alert( '1' + 2 ); // "12"
alert( 2 + '1' ); // "21"
```

下面是一个更复杂的例子：
```js
alert(2 + 2 + '1' ); // "41"，不是 "221"
```
在这里，运算符是按从左到右的顺序工作，第一个 `+` 将两个数字相加，所以返回 `4`，然后下一个 `+` 将字符串 `1` 加入其中，所以就是 `4 + '1' = '41'`

另一个类似的例子：
```js
alert('1' + 2 + 2); // "122"，不是 "14"
```
这里，第一个操作数是一个字符串，所以编译器将其他两个操作数也视为了字符串，`2` 被与 `'1'` 连接到了一起，也就是像 `'1' + 2 = "12"` 然后 `"12" + 2 = "122" `这样

二元运算符 `+` 是唯一一个以这种方式支持字符串的运算符
其他算术运算符只对数字起作用，并且总是将其运算元转换为数字
下面是减法和除法运算的示例：
```js
alert( 6 - '2' ); // 4，将 '2' 转换为数字
alert( '6' / '2' ); // 3，将两个运算元都转换为数字
```

**数字转化，一元运算符 +**
加号 `+` 有两种形式，一种是上面我们刚刚讨论的二元运算符，还有一种是一元运算符

一元运算符加号，或者说加号 `+` 应用于单个值时，对数字没有任何作用，但是如果运算元不是数字，加号 `+` 则会将其转化为数字
例如：
```js
// 对数字无效
let x = 1;
alert( +x ); // 1

let y = -2;
alert( +y ); // -2

// 转化非数字
alert( +true ); // 1
alert( +"" );   // 0
```
也就是说，它的效果和 `Number(...)` 相同，但是更加简短

我们经常会有将字符串转化为数字的需求
比如，如果我们正在从 HTML 表单中取值，通常得到的都是字符串，如果我们想对它们求和，该怎么办？

二元运算符加号会把它们合并成字符串：
```js
let apples = "2";
let oranges = "3";

alert( apples + oranges ); // "23"，二元运算符加号合并字符串
```

如果我们想把它们当做数字对待，我们需要转化它们，然后再求和：
```js
let apples = "2";
let oranges = "3";

// 在二元运算符加号起作用之前，所有的值都被转化为了数字
alert( +apples + +oranges ); // 5

// 更长的写法
// alert( Number(apples) + Number(oranges) ); // 5
```
本例中，一元运算符加号首先起作用，它们将字符串转为数字，然后二元运算符加号对它们进行求和

为什么一元运算符先于二元运算符作用于运算元？这是由于它们拥有更高的优先级

**运算符优先级**
如果一个表达式拥有超过一个运算符，执行的顺序则由**优先级**决定，所有的运算符中都隐含着优先级顺序
在 JavaScript 中有众多运算符，每个运算符都有对应的优先级数字，数字越大，越先执行，如果优先级相同，则按照由左至右的顺序执行

摘抄自 Mozilla 的部分优先级表：

| 优先级 | 名称     | 符号 |
| ------ | -------- | ---- |
| …      | …        | …    |
| 15     | 一元加号 | `+`  |
| 15     | 一元负号 | `-`  |
| 14     | 求幂     | `**` |
| 13     | 乘号     | `*`  |
| 13     | 除号     | `/`  |
| 12     | 加号     | `+`  |
| 12     | 减号     | `-`  |
| …      | …        | …    |
| 2      | 赋值符   | `=`  |
| …      | …        | …    |

很显然一元运算符优先级高于二元运算符

**赋值运算符**
赋值符号 = 也是一个运算符，它的优先级非常低，只有 2
这也是为什么当我们赋值时，比如 `x = 2 * 2 + 1`，所有的计算先执行，然后 `=` 才执行，将计算结果存储到 `x`
```js
let x = 2 * 2 + 1;

alert( x ); // 5
```

**赋值 = 返回一个值**
在 JavaScript 中，所有运算符都会返回一个值，这对于 `+` 和 `-` 来说是显而易见的，但对于 `=` 来说也是如此
语句 `x = value` 将值 `value` 写入 `x` 然后返回 `value`

下面是一个在复杂语句中使用赋值的例子：
```js
let a = 1;
let b = 2;

let c = 3 - (a = b + 1);

alert( a ); // 3
alert( c ); // 0
```
上面这个例子，`(a = b + 1)` 的结果是赋给 `a` 的值，也就是 `3` ，然后该值被用于进一步的运算

注意这样的技巧不会使代码变得更清晰或可读

**链式赋值(Chaining assignments)**
JavaScript 中一个有趣的特性是链式赋值：
```js
let a, b, c;

a = b = c = 2 + 2;

alert( a ); // 4
alert( b ); // 4
alert( c ); // 4
```
链式赋值从右到左进行计算，首先，对最右边的表达式 `2 + 2` 求值，然后将其赋给左边的变量：`c`、`b` 和 `a` ，最后，所有的变量共享一个值

出于可读性，最好将这种代码分成几行：
```js
c = 2 + 2;
b = c;
a = c;
```

**原地修改**
我们经常需要对一个变量做运算，并将新的结果存储在同一个变量中

例如：
```js
let n = 2;
n = n + 5;
n = n * 2;
```

可以使用运算符 `+=` 和 `*=` 来缩写这种表示：
```js
let n = 2;
n += 5; // 现在 n = 7（等同于 n = n + 5）
n *= 2; // 现在 n = 14（等同于 n = n * 2）

alert( n ); // 14
```

所有算术和位运算符都有对应的“修改并赋值”运算符，如：`/=` 和 `-=` 等

注意这类运算符的优先级与普通赋值运算符的优先级相同，所以它们在大多数其他运算之后执行：
```js
let n = 2;

n *= 3 + 5;

alert( n ); // 16 （右边部分先被计算，等同于 n *= 8）
```

**自增/自减**
对一个数进行加一、减一是最常见的数学运算符之一
所以，对此有一些专门的运算符：

自增 `++` 将变量与 `1` 相加：
```js
let counter = 2;
counter++;      // 和 counter = counter + 1 效果一样，但是更简洁
alert( counter ); // 3
```

自减 `--` 将变量与 `1` 相减：
```js
let counter = 2;
counter--;      // 和 counter = counter - 1 效果一样，但是更简洁
alert( counter ); // 1
```
注意：
自增/自减只能应用于变量，将其应用于数值常量(比如 5++)则会报错

运算符 `++` 和 `--` 可以置于变量前，也可以置于变量后
- 当运算符置于变量后，被称为“后置形式”：`counter++`
- 当运算符置于变量前，被称为“前置形式”：`++counter`
两者都做同一件事：将变量 `counter` 与 `1` 相加

那么它们有区别吗？有，但只有当我们使用 `++/--` 的返回值时才能看到区别。
我们知道，所有的运算符都有返回值，自增/自减也不例外，而前置形式返回的是运算后的值，但后置返回的是运算前的值

为了直观看到区别，看下面的例子：
```js
let counter = 1;
let a = ++counter; // (*)

alert(a); // 2
```
`(*)` 所在的行是前置形式 `++counter`，对 `counter` 做自增运算，返回的是新的值 `2` ，因此 `alert` 显示的是 `2`

下面让我们看看后置形式：
```js
let counter = 1;
let a = counter++; // (*) 将 ++counter 改为 counter++

alert(a); // 1
```
`(*)`所在的行是后置形式 `counter++`，它同样对 `counter` 做加法，但是返回的是旧值/做加法之前的值，因此 `alert` 显示的是 `1`

总结：
- 如果自增/自减的返回值不会被使用，那么两者形式没有区别：
```js
let counter = 0;
counter++;
++counter;
alert( counter ); // 2，以上两行作用相同
```

-  如果我们想要对变量进行自增操作，并且需要立刻使用自增后的值，那么我们需要使用前置形式：
```js
let counter = 0;
alert( ++counter ); // 1
```

- 如果我们想要将一个数加一，但是我们想使用其自增之前的值，那么我们需要使用后置形式：
```js
let counter = 0;
alert( counter++ ); // 0
```

**自增/自减和其它运算符的对比**
`++/--` 运算符同样可以在表达式内部使用，并且它们的优先级比绝大部分的算数运算符要高

举个例子：
```js
let counter = 1;
alert( 2 * ++counter ); // 4
```

与下方例子对比：
```js
let counter = 1;
alert( 2 * counter++ ); // 2，因为 counter++ 返回的是“旧值”
```
尽管从技术层面上来说可行，但是这样的写法会降低代码的可阅读性
在一行上做多个操作 —— 这样并不好
当阅读代码时，快速的视觉“纵向”扫描会很容易漏掉 counter++，这样的自增操作并不明显。
建议用“一行一个行为”的模式：
```js
let counter = 1;
alert( 2 * counter );
counter++;
```

**位运算符**
位运算符把运算元当做 `32` 位整数，并在它们的二进制表现形式上操作
大部分的编程语言都支持这些运算符

下面是位运算符：
- 按位与 ( `&` )
- 按位或 ( `|` )
- 按位异或 ( `^` )
- 按位非 ( `~` )
- 左移 ( `<<` )
- 右移 ( `>>` )
- 无符号右移 ( `>>>` )
在 Web 开发中很少使用它们，但在某些特殊领域中，例如密码学，它们很有用

**逗号运算符**
逗号运算符 `,` 是最少见最不常使用的运算符之一

可以使用逗号运算符处理多个表达式：使用 `,` 将它们分开
每个表达式都会被执行，但是只有最后一个的结果会被返回

举个例子：
```js
let a = (1 + 2, 3 + 4);

alert( a ); // 7 (the result of 3 + 4)
```
这里，第一个表达式 `1 + 2` 运行了，但是它的结果被丢弃了，随后计算 `3 + 4`，并且该计算结果被返回

注意逗号运算符的优先级非常低，比 `=` 还要低，因此上面的例子中圆括号非常重要，如果没有圆括号：`a = 1 + 2, 3 + 4` 会先执行 `+`，将数值相加得到 `a = 3, 7`，然后赋值运算符 `=` 执行 `a = 3`，整个表达式的值是 `7` 

有时候，人们会使用它以将几个行为放在同一行上
举个例子：
```js
// 同一行上有三个操作
for (a = 1, b = 3, c = a * b; a < 10; a++) {
 ...
}
```
这样的技巧在许多 JavaScript 框架中都有使用，但是通常它并不能提升代码的可读性

### 1.2.9 值的比较
**比较结果为 Boolean 类型**
注意所有比较运算符均返回布尔值：
- `true` 
- `false`

示例：
```js
alert( 2 > 1 );  // true（正确）
alert( 2 == 1 ); // false（错误）
alert( 2 != 1 ); // true（正确）
```

比较的结果是布尔值，因此也可以被赋值给任意变量：
```js
let result = 5 > 4; // 把比较的结果赋值给 result
alert( result ); // true
```

**字符串比较**
在比较字符串的大小时，JavaScript 会使用“字典(dictionary)”或“词典(lexicographical)”顺序进行判定，换言之，字符串是按字符/字母逐个进行比较的

例如：
```js
alert( 'Z' > 'A' ); // true
alert( 'Glow' > 'Glee' ); // true
alert( 'Bee' > 'Be' ); // true
```

字符串的比较算法：
1. 首先比较两个字符串的首位字符大小
2. 如果一方字符较大/或较小，则该字符串大于/或小于另一个字符串，结束
3. 否则，如果两个字符串的首位字符相等，则继续取出两个字符串各自的后一位字符进行比较
4. 重复上述步骤进行比较，直到比较完成某字符串的所有字符为止
5. 如果两个字符串的字符同时用完，那么则判定它们相等，否则未结束(还有未比较的字符)的字符串更大

程序中的字典序非真正的字典顺序，而是 Unicode 编码顺序，JavaScript在比较字符时，实际比较的是字符在Unicode中的索引值，比如 `a > A` ， 因为在 JavaScript 使用的内部编码表中(Unicode)，小写字母的字符索引值更大

**不同类型间的比较**
当对不同类型的值进行比较时，JavaScript 会首先将其转化为数字/number类型再判定大小

例如：
```js
alert( '2' > 1 ); // true，字符串 '2' 会被转化为数字 2
alert( '01' == 1 ); // true，字符串 '01' 会被转化为数字 1
```

对于布尔类型值，`true` 会被转化为 `1`、`false` 转化为 `0`
例如：
```js
alert( true == 1 ); // true
alert( false == 0 ); // true
```

**严格相等**
普通的相等性检查 == 存在一个问题，它不能区分出 0 和 false：
```js
alert( 0 == false ); // true
```

也同样无法区分空字符串和 false：
```js
alert( '' == false ); // true
```

因为在比较不同类型的值时，处于相等判断符号 `==` 两侧的值会先被转化为数字，空字符串和 `false` 转化后它们都为数字 0

而严格相等运算符 `===` 在进行比较时不会做任何的类型转换，如果 `a` 和 `b`  属于不同的数据类型，那么 `a === b` 不会做任何的类型转换而立刻返回 `false`

让我们试试：
```js
alert( 0 === false ); // false，因为被比较值的数据类型不同
```

同样的，与“不相等”符号 `!=` 类似，“严格不相等”表示为 `!==`
严格相等的运算符虽然写起来稍微长一些，但是它能够很清楚地显示代码意图，降低犯错的可能性

**对 null 和 undefined 进行比较**
当使用严格相等 `===` 比较二者时：
它们不相等，因为它们属于不同的类型
```js
alert( null === undefined ); // false
```

当使用非严格相等 `==` 比较二者时：
JavaScript 存在一个特殊的规则，会判定它们相等，且 `null` 和 `undefined` 仅仅等于对方而不等于其他任何的值(只在非严格相等下成立)
```js
alert( null == undefined ); // true
```

当使用其他比较方法如 `< > <= >=` 时：
`null/undefined` 会被转化为数字：`null` 被转化为 `0`，`undefined` 被转化为 `NaN`

这些规则会带来一些奇怪的现象
**奇怪的结果：null vs 0**
通过比较 `null` 和 0 可得：
```js
alert( null > 0 );  // (1) false
alert( null == 0 ); // (2) false
alert( null >= 0 ); // (3) true
```
为什么会出现这种反常结果，这是因为相等性检查 `==` 和普通比较符 `> < >= <=` 的代码逻辑是相互独立的

进行与数值的比较时，`null` 会被转化为数字，因此它被转化为了 `0` ，因此 `null >= 0` 返回值是 true， `null > 0` 返回值是 false
在相等性检查 `==` 中，`undefined` 和 `null`不会进行任何的类型转换，它们有自己独立的比较规则，即除了它们之间互等外，不会等于任何其他的值，因此 `null == 0` 会返回 false

**特立独行的 undefined**
`undefined` 与其他值进行比较：
```js
alert( undefined > 0 ); // false (1)
alert( undefined < 0 ); // false (2)
alert( undefined == 0 ); // false (3)
```
解释如下：
- `(1)` 和 `(2)` 都返回 `false` 是因为 `undefined` 在比较中被转换为了 `NaN`，而 `NaN` 是一个特殊的数值型值，它与任何值进行比较都会返回 `false`
- `(3)` 返回 `false` 是因为这是一个相等性检查，而 `undefined` 只与 `null` 相等，不会与其他值相等

我们需要避免潜在的问题：
- 除了严格相等 `===` 外，其他但凡是有 `undefined/null` 参与的比较，我们都需要格外小心
- 除非你非常清楚自己在做什么，否则永远不要使用 `>= > < <=` 去比较一个可能为 `null/undefined` 的变量。对于取值可能是 `null/undefined` 的变量，请按需要分别检查它的取值情况

### 1.2.10 条件分支：if 和 '?'
**“if” 语句**
`if(...)` 语句计算括号里的条件表达式，如果计算结果是 `true`，就会执行对应的代码块，如：
```js
let year = prompt('In which year was ECMAScript-2015 specification published?', '');

if (year == 2015) alert( 'You are right!' );
```
在上面这个例子中，条件是一个相等性检查：`year == 2015`

如果有多个语句要执行，我们必须将要执行的代码块封装在大括号内：
```js
if (year == 2015) {
  alert( "That's correct!" );
  alert( "You're so smart!" );
}
```

一般建议每次使用 if 语句都用大括号 `{}` 来包装代码块，以提高代码可读性

**布尔转换**
`if (…)` 语句会计算圆括号内的表达式，如果计算结果不是布尔型，则会将计算结果转换为布尔型

回顾布尔型的转换规则：
- 数字 `0`、空字符串 `""`、`null`、`undefined` 和 `NaN` 都会被转换成 `false`，它们被称为“假值(falsy)”。
- 其他值被转换为 `true`，它们被称为“真值(truthy)”

所以，下面这个条件下的代码永远不会执行：
```js
if (0) { // 0 是假值（falsy）
  ...
}
```

而下面的条件始终有效：
```js
if (1) { // 1 是真值（truthy）
  ...
}
```

也可以将预先计算的布尔值传入 `if` 语句，像这样：
```js
let cond = (year == 2015); // 相等运算符的结果是 true 或 false

if (cond) {
  ...
}
```

**“else” 语句**
`if` 语句有时会包含一个可选的 “else” 块，如果判断条件不成立，就会执行它内部的代码：
```js
let year = prompt('In which year was ECMAScript-2015 specification published?', '');

if (year == 2015) {
  alert( 'You guessed it right!' );
} else {
  alert( 'How can you be so wrong?' ); // 2015 以外的任何值
}
```

**多个条件：“else if”**
有时我们需要测试一个条件的几个变体。我们可以通过使用 `else if` 子句实现，例如：
```js
let year = prompt('In which year was ECMAScript-2015 specification published?', '');

if (year < 2015) {
  alert( 'Too early...' );
} else if (year > 2015) {
  alert( 'Too late' );
} else {
  alert( 'Exactly!' );
}
```
可以有更多的 else if 块，结尾的 else 是可选的

**条件运算符 ‘?’**
有时我们需要根据一个条件去赋值一个变量，如下所示：
```js
let accessAllowed;
let age = prompt('How old are you?', '');

if (age > 18) {
  accessAllowed = true;
} else {
  accessAllowed = false;
}

alert(accessAllowed);
```

“条件”或“问号”运算符让我们可以更简短地达到目的，这个运算符通过问号 `?` 表示，有时它被称为三元运算符，因为该运算符中有三个操作数，它是 JavaScript 中唯一一个有这么多操作数的运算符

语法：
```js
let result = condition ? value1 : value2;
```
该运算符计算条件 `condition` 的结果，如果结果为真，则返回 `value1`，否则返回 `value2`

例如：
```js
let accessAllowed = (age > 18) ? true : false;
```
技术上讲，我们可以省略` age > 18` 外面的括号，因为问号运算符的优先级较低，所以它会在比较运算符 `>` 后执行

下面这个示例会执行和前面那个示例相同的操作：
```js
// 比较运算符 "age > 18" 首先执行
//（不需要将其包含在括号中）
let accessAllowed = age > 18 ? true : false;
```
但括号可以使代码可读性更强，所以建议使用它们

在上面的例子中，也可以不使用问号运算符，因为比较运算符本身就返回 `true/false`：
```js
// 下面代码同样可以实现
let accessAllowed = age > 18;
```
