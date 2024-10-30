# 1 Getting Started
## 1.1 Installation
rustup：用于管理Rust版本和相关工具的命令行工具

查看rust compiler版本：`rustc --version`
升级rustup：`rustup update`
卸载rustup：`rustup self uninstall`
阅读文档：`rustup doc`

## 1.2 Hello, World!
rust文件后缀为*.rs*
rust文件的命名习惯是用下划线划分单词，如：*hello_world.rs*

编译文件：`rustc main.rs`
运行：`.\main.exe`

rust程序从main函数开始执行(rustfmt可以用于标准格式化rust代码)
rust风格以四个空格为缩进
println!调用了rust宏，rust宏与函数是不同的
"Hello, World!"字符串作为了println!的参数
rust代码以 ; 作为一行代码行的结尾

main.pdb包含了debug信息

## 1.3 Hello, Cargo!
Cargo是rust的的构建系统和包管理工具，常用于管理rust项目
Cargo可以帮助构建代码，下载依赖库

查看Cargo版本：`cargo --version`

用Cargo来创建项目：
`cargo new hello_cargo`
`cd hello_cargo`
Cargo为我们创造了一个*Cargo.toml*文件和一个src目录，src中包含了*main.rs*
Cargo同时初始化了一个git仓库，我们可以看到*.gitignore*文件

*Cargo.toml*是Cargo的配置文件，后缀名全称Tom's Obvious, Minimal Language
Cargo希望我们把源文件都放在src目录里，而README，Lisense，配置文件等与代码无关的文件则放在项目目录里

用Cargo来构建项目：
在hello_cargo目录中输入`cargo build`
*hello_cargo.exe*将被创建在.\\target\\debug\\hello_cargo.exe，因为默认构建是debug构建，Cargo将可执行文件放在了debug目录
Cargo同时创建了*Cargo.lock*在hello_cargo目录，该文件对项目依赖的版本进行追踪，该文件由Cargo自动管理

用Cargo来构建并运行项目：
在hello_cargo目录中输入`cargo run`可以让Cargo帮助我们编译代码并运行，`cargo run`替代了`cargo build`+`.\target\debug\hello_cargo.exe`，当然Cargo在识别该项目已经成功构建且没有改动后不会重复构建，会直接运行可执行文件
如果对源代码进行了修改，`cargo run`会重新构建项目并运行可执行文件

如果我们需要编译，而不需要产生可执行文件，可以使用`cargo check`
因为不产生可执行文件，`cargo check`的效率要高于`cargo build`
在编写项目时，我们希望可以定期检查之前的代码可以正确编译，这时可以使用`cargo check`，方便我们快速确认之前代码的正确性，以及快速发现bug，在完成了项目的全部工作后，我们再使用`cargo build`得到可执行文件

用Cargo来构建用于发布的项目：
当项目已经可以用于发布时，`cargo build --release`可以对项目进行构建的同时对其优化，该命令将创建target/release目录而不是target/debug目录，优化会让编译的时间更长

# 2 Programming a Guessing Game
用Cargo构造项目并试运行：
```
> cargo new guessing_game
> cargo run
Hello, world!
```

## 2.1 Reading Inputs
猜数游戏的第一部分需要请求用户输入一个数字，并检查数值的值是否为所想要的：
```rust
use std::io;  

fn main() { 
	println!("Guess the number!"); 
	
	println!("Please input your guess."); 
	
	let mut guess = String::new(); 
	
	io::stdin() 
		.read_line(&mut guess) 
		.expect("Failed to read line"); 
		
	println!("You guessed: {guess}"); }
```
解释：
`use std:: io`：将标准库中的io库引入作用域，rust会预导入std中的一些库，但有些库在需要时要手动导入，如io库
`fn main(){`：main函数是程序的入口，由fn关键字声明函数
`println!("")`：println是将字符串打印到屏幕的宏

`let mut guess = String::new()`：创造一个变量来存储用户输入

let语句常用于创造变量，如：`let apple = 5`
在rust中，变量实际上是默认不可变的，如果需要值可变的变量，就需要加上mut关键字，如：`let mut banana = 5`

`String::new()`是返回了String实例的一个函数，String是由标准库提供的字符串类型，由utf-8编码，可变长度
`::`表明了`new()`函数是和String类型相关联的函数，关联函数是针对类型实现的，而不是实例，可以理解为静态方法
`new()`函数在很多类型中都有实现，在这里它返回了一个新的空字符串

因此总地来说，`let mut guess = String::new()`创造了一个可变变量guess，将它绑定到了String类型的一个新的空实例上

`io.stdin().read_line(&mut guess)`：调用了io模块中的stdin函数，stdin函数会返回一个`std::io::Stdin`的实例，这个类型代表了终端的标准输入句柄
`read_line(&mut guess)`调用了该类型的read_line方法，该方法将用户输入追加到参数字符串中，参数字符串应该是可变的
`&`说明了参数是一个引用，避免了内存中的数据拷贝，在rust中，引用默认为常引用，因此如果要对变量进行改变，需要加上mut关键字

`.expect("Falied to read line")`：read_line方法会返回一个Result值，Result是一个枚举类型，枚举类型变量的值可能是多种可能状态中的一个，每种可能的状态称为一个枚举成员
Result类型的枚举成员有Ok和Err，Ok成员表示操作成功，Ok内包含成功产生的值，Err成员表示操作失败，Err内包含失败信息
Result类型的实例有expect方法，如果类型的值是Err，调用expect方法时，程序终止并打印我们传入的参数，如果类型的值时Ok，expect会获取Ok的值并直接返回，在本例中返回的是用户输入的字节数

如果不使用expect方法，编译时会产生警告，rust会警告我们没有使用由read_line返回的Result值，说明该程序无法解决可能出现的错误

用`cargo run`编译并运行，发现程序可以正确编译，读取用户输入并打印，至此我们完成了第一部分

## 2.2 Adding Dependencies
rust生成随机数的功能并不在std库内，rust提供了rand crate执行这个功能
crate是一个rust代码包，是rust源代码文件的集合，我们正在构建的项目是一个二进制crate，它生成一个可执行文件，而rand carte是一个库crate，库crate包含了能被任意其他程序使用的代码，但是不能自己执行

在我们使用rand crate编写代码之前，我们需要修改`Cargo.toml`文件，引入一个rand依赖：
```
[dependcies]
rand = "0.8.5"
```
*Cargo.toml*文件中的`[dependencies]`部分用于指定本项目依赖的crates及其版本，本例中我们用“0.8.5”指定了rand的语义化版本，“0.8.5”实际上是“^0.8.5"的缩写，表示任何至少是0.8.5但少于0.9.0的版本，Cargo认为这些版本的公有api和0.8.5版本的是兼容的，这使得我们可以确定得到能使本章代码正常编译的，含有最新补丁的版本，任何大于等于0.9.0的版本的api则可能会改变

`cargo build`进行编译

Cargo会从registry上获取所有包的最新版本信息，这是一份对Crates.io上的数据的拷贝，Crates.io是rust生态环境中的开发者贡献开源项目的地方
Cargo会在更新完registry后检查`[dependencies]`片段，并下载列表中包含但还未下载的crates，在本例中，Cargo会将rand依赖的其他crates一并下载，下载完依赖后，Cargo先编译依赖，然后编译项目

之后若修改了源代码，没有添加新的依赖，`cargo build`只会重新编译源代码，不会重复编译依赖

在第一次构建项目时，Cargo所有依赖的符合要求的版本并将其写入*Cargo.lock*文件，在之后重新构建项目时，Cargo会发现*Cargo.lock*已存在并使用其中指定的版本，Cargo以此保证项目的可重复构建，因此*Cargo.lock*也会被纳入版本管理系统

当我们确实需要升级crate时，使用`cargo update`，Cargo会忽略*Cargo.lock*文件，并计算符合*Cargo.toml*声明的最新版本，然后更新*Cargo.lock*，如在本例中，rand发布了0.8.6版本，`cargo update`会对其进行升级，如果想要使用0.9.0以上的版本，则需要显式地修改*Cargo.toml*，在下一次的`cargo build`时，Cargo就会从registry上获取符合要求的最新的crate

## 2.3 Generating Random Numbers
修改*main.rs*：
```rust
use std::io;
use rand::Rng;
fn main() {
	println!("Guess the number!"); 
	
	let secret_number = rand::thread_rng().gen_range(1..=100);
	
	 println!("The secret number is: {secret_number}"); 
	 
	 println!("Please input your guess."); 
	 
	 let mut guess = String::new(); 
	 
	 io::stdin() 
		 .read_line(&mut guess) 
		 .expect("Failed to read line"); 
	 
	 println!("You guessed: {guess}"); 
}
```
解释：
`rand::Rng`是一个trait，定义了随机数生成应该实现的方法，我们将这个trait加入作用域，`rand::thread_rng()`函数提供了随机数生成器，它位于当前执行线程的本地环境中，从操作系统获取seed，`gen_range()`是生成器的一个方法，该方法由`rand::Rng`这个trait定义，范围表达式`1..=100`作为其参数，注意这个范围表达式包含了上下界

注：每个crate都有文档，`cargo doc --open`会构建所有本地依赖提供的文档，在浏览器打开，我们可以借此查看rand的文档

`cargo run`运行程序

## 2.4 Comparing Guessing Number And Random Number
修改*main.rs*：
```rust
use rand::Rng; 
use std::cmp::Ordering; 
use std::io; 

fn main() { 
// --snip-- 
	println!("You guessed: {guess}"); 
	
	match guess.cmp(&secret_number) { 
		Ordering::Less => println!("Too small!"), 
		Ordering::Greater => println!("Too big!"), 
		Ordering::Equal => println!("You win!"), 
	} 
}
```
解释：
`use std::cmp::Ordering`将标准库的Ordering类型代入作用域，Ordering也是一个枚举类型，有三个成员Less，Greater，Equal
String实例的cmp方法用于比较两个值，它的参数是一个被比较值的引用，即secret_number，并返回一个Ordering类型的成员，我们使用match表达式，根据返回的是哪一个成员，作出相应的动作

一个match表达式由多个分支构成，每个分支都包含要匹配的模式以及模式匹配时的代码，match会顺序进行匹配，并在得到第一个匹配后执行动作并退出

但代码现在还不能编译通过，错误状态为mismatched types，即不匹配的类型，rust有静态强类型系统，也有类型推断，在`let mut guess = String::new()`中，rust推断guess为String类型，而secret_number为数字类型，默认是i32，该错误的原因是String类型和i32类型不能比较

我们需要将String类型转换为数字类型：`let guess: u32 = guess.trim().parse().expect("Please type a number!")`，我们创造了一个新的变量guess，它其实是用一个新值隐藏了guess旧的值，这个特性常用于类型转换

String实例的`trim()`方法会去除字符串开头和结尾的任意空白字符，因为在输入的时候用户必须输入Enter键才可以让`read_line()`返回，而`read_line()`会对其进行读取，在Windows中，字符串后会被添加上\\r\\n，因此需要将其去除

String实例的`pares()`方法用于将String转化为其他类型，我们通过`let guess: u32`指定了类型，之后rust会将secret_numbert也推断为u32类型，就可以进行比较了

`parse()`方法也会返回一个Result类型以进行错误处理，如果成功转换，返回Ok类型，`expect()`方法会返回Ok类型内含的值，即转换后的数字，否则打印错误信息并返回

## 2.5 Using Loop to Allow Multiple Attempts
可以利用loop关键字创造无限循环：
```rust
// --snip-- 
	println!("The secret number is: {secret_number}"); 
	loop { 
		println!("Please input your guess."); 
		// --snip-- 
		match guess.cmp(&secret_number) { 
			Ordering::Less => println!("Too small!"), 
			Ordering::Greater => println!("Too big!"), 
			Ordering::Equal => println!("You win!"), 
		} 
	} 
}
```
用户现在可以ctrl+c退出循环，或可以利用对`parse()`的`expect()`处理来退出循环

我们可以用break关键字，在猜测成功后退出循环：
```rust
// --snip-- 
		match guess.cmp(&secret_number) { 
			Ordering::Less => println!("Too small!"), 
			Ordering::Greater => println!("Too big!"), 
			Ordering::Equal => { 
				println!("You win!"); 
				break; 
			} 
		} 
	} 
}
```

对程序改进，在处理无效输入时，忽略用户的无效输入，让用户继续猜测，而不是终止程序：
```rust
        // --snip--

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        println!("You guessed: {guess}");

        // --snip--

```
我们使用match语句进行错误处理，因为`parse()`返回的是Result类型，如果返回一个包含数字结果的Ok，返回数字，如果返回一个包含错误信息的Err，\_是一个通配符，本例中用于匹配所有的Err值，continue进入下一个循环

最后删除将sercret_number打印的语句即可完成这个游戏

# 3 Common Programming Concepts
## 3.1 Variables and Mutability
### 3.1.1 Variables
默认情况下，变量是不可改变的

`cargo new variables`生成新项目
修改*main.rs*：
```rust
fn main() {
    let x = 5;
    println!("The value of x is: {x}");
    x = 6;
    println!("The value of x is: {x}");
}
```
`cargo run`发现不能通过编译，因为不能对不可变变量x二次赋值

添加mut关键字就可以构造可变变量
```rust
fn main() {
    let mut x = 5;
    println!("The value of x is: {x}");
    x = 6;
    println!("The value of x is: {x}");
}
```
此时可以正常编译运行

### 3.1.2 Constants
常量也是绑定到一个名称的不可改变的值，但它和变量有所区别：
(1) 不允许对常量使用mut关键字，常量只能用const关键字声明，且必须注明值的类型
(2) 常量可以在任何作用域声明，包括全局作用域
(3) 常量的值只能由常量表达式设定，不能在运行时计算

声明一个常量：
`const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;`
rust对常量的命名规范是全大写，单词间下划线隔开，编译器可以在编译时计算一组有限的操作，因此我们可以用计算表达式的方式给常量赋值

常量在它的被声明的作用域内在整个程序周期都是有效的，


### 3.1.3 Shadowing
我们可以声明一个和之前声明过的变量完全同名的变量，在rust中，这称为第一个变量被第二个变量隐藏了，此时任何使用该变量的行为都会被认为是在使用第二个变量，直到第二个变量自己也被隐藏或是作用域结束

比如：
```rust
fn main() {
    let x = 5;

    let x = x + 1;

    {
        let x = x * 2;
        println!("The value of x in the inner scope is: {x}");
    }

    println!("The value of x is: {x}");
}
```
我们先将x的值绑定为5，然后创建了一个新的x变量，值为6，然后在花括号的内部作用域中，第三个let语句创建了新的x变量，值为12，该作用域结束后，隐藏结束，x的值变为6

隐藏是用let关键字完成的，和声明一个mutable变量不同，用let关键字定义的变量仍然是不可变的
在我们再次使用let时，实际上时创建了一个新的变量，因此变量的类型可以完全不同，如：
```rust
let spaces = "    "; // string type
let spaces = spaces.len(); // number type
```
如果使用mut关键字，则不行：
```rust
let mut spaces = "    ";
spaces = spaces.len();
```
编译器会告诉我们不能改变变量的类型

## 3.2 Data Types
rust有两个数据类型子集：标量(scalar)和复合(compound)

rust是一个静态类型语言，即在编译时就要知道所有变量的类型，编译器通常可以通过变量的值和使用方式推断出变量的类型，但如果多种类型都有可能时，我们需要显式注明类型，如用`parse()`将String转换为数字时：
`let guess: u32 = "42".parse().expect("Not a Number!");`
否则编译器会报错

### 3.2.1 Scalar Types
标量类型代表一个单独的值，rust的四个基本标量类型：整型，浮点数，布尔类型，字符类型

**Integer Types：**
rust的内置整型类型：
i8，i16，i32，i64，i128，isize
u8，u16，u32，u64，u128，usize
有符号数以补码形式存储

有符号数的存储范围是$[-2^{n-1}, 2^{n-1}-1]$
无符号是的存储范围是$[0, 2^n-1]$

isize，usize的存储位数依赖于计算机架构，32位架构就是32位，64位架构就是64位

rust中的整型字面值：
十进制如：`98_222`
十六进制如：`0xff`
八进制如：`0o77`
二进制如：`0b1111_0000`
单字节字符(仅限u8)如：`b'A'`

rust默认整型类型是i32

rust遭遇整型溢出：
1. 在debug模式编译时：
rust会检查这类问题并使程序panic，rust用panic表示程序因错误而退出
2. 使用`--release`flag编译时：
rust不会检查整型溢出，溢出时，因为补码的缘故，整型的值不会是期望的值

**Floating-Point Types：**
rust的浮点数类型包括f32和f64，默认为f64，所有浮点型都是有符号的
如：
```rust
fn main() {
    let x = 2.0; // 默认f64，双精度

    let y: f32 = 3.0; // f32，单精度
}
```

补充rust的数值运算：
```rust
fn main() {
    // addition
    let sum = 5 + 10;

    // subtraction
    let difference = 95.5 - 4.3;

    // multiplication
    let product = 4 * 30;

    // division
    let quotient = 56.7 / 32.2;
    let truncated = -5 / 3; // 结果为 -1，整数除法会向零取整

    // remainder
    let remainder = 43 % 5;
}

```

**Boolean Type：**
只有两个可能的值：true和false
如：
```rust
fn main() {
    let t = true;

    let f: bool = false; // with explicit type annotation
}
```

(4) Character Type
如：
```rust
fn main() {
    let c = 'z';
    let z: char = 'ℤ'; // with explicit type annotation
    let heart_eyed_cat = '😻';
}
```
rust的char类型大小为4个字节32位，代表了一个Unicode标量值，rust中，带变音符号的字母，中文，日文，韩文，emoji和0长度的空白字符都是有效的chart值，Unicode标量值的范围从`U+0000`到`U+D7FF`和从`U+E000`到`U+10FFFF`

### 3.2.2 Compound Types
rust的复合类型包括元组(tuple)和数组(array)

**Tuple Type：**
元组的长度固定，一旦声明，不会变化
元组中可以包含不同类型的值，如：
`let tup: (i32, f64, u8) = (500, 6.4, 1);`

我们可以用模式匹配来解构元组的值，如：
`let (x, y, z) = tup;`

可以通过索引访问元组的值，如：
`let five_hundred = x.0;`
`let six_point_four = x.1;`
`let one = x.2;`

不带任何值的元组叫单元组(unit)，它的值和对应的类型都写作`()`，表示空的值或空的返回类型，如果表达式不返回其它任何值，则会隐式地返回单元值(unit value)

**Array Type：**
数组只能包含同类型的值，rust中数组的长度也是固定的
数组在栈上为程序分配空间，是已知大小的单个内存块

数组的声明：
`let a = [1, 2, 3, 4, 5];`
`let a: [i32; 5] = [1, 2, 3, 4, 5];`i32是每个元素的类型，5是数组的长度
`let a = [3; 5];`等价于`let a = [3, 3, 3, 3, 3];`

访问数组元素：
`let first = a[0]`
`let second = a[1]`

如果在运行时发生了数组越界，会导致运行时错误(runtime error)，程序会携带错误信息退出，rust会检查指定的索引是否小于数组的长度，如果超过了，rust会panic，这种检查必须在运行时进行，因为编译器不可能知道用户在以后运行代码时将输入什么值

## 3.3 Functions
rust使用fn关键字声明函数，函数命名遵循全部小写并下划线隔开的规则
函数定义示例：
```rust
fn main() {
    println!("Hello, world!");

    another_function();
}

fn another_function() {
    println!("Another function.");
}
```
rust中函数定义的位置没有影响，只要函数被定义在了调用该函数者可见的作用域内

**Parameters：**
参数也是函数签名的一部分
参数使用示例：
```rust
fn main() {
    another_function(5);
}

fn another_function(x: i32) {
    println!("The value of x is: {x}");
}
```

rust规定在函数签名中我们必须显式声明每个参数的类型

多个参数用逗号隔开

**Statements and Expressions：**
函数体是由一系列的语句和一个可选的的结尾表达式构成的

在rust中，语句和表达式是不同的：
语句是执行一系列操作但不返回值的指令
表达式是则会计算并产生一个结果值

使用let关键字创造一个变量并绑定一个值是语句，如：`let y = 5;`
函数定义也是一个语句，如：
```rust
fn main(){
	let y = 5; 
}
```

语句不会返回值，因此不能将let语句赋值给另一个变量，如：
`let x = (let y = 5);`
会导致编译错误

表达式会计算出一个值，如`5 + 6`，这是一个会计算出值11的表达式，表达式可以是语句的一部分，如`let y = 5;`中的`5`就是一个表达式，值是5
函数调用也是表达式 ，宏调用也是表达式
用大括号创造一个新的作用域也是表达式，如：
```rust
fn main(){
	let y = {
		let x = 3;
		x + 1;
	}

	println!("The value of y is: {y}");
}
```
这个表达式：
```rust
{
    let x = 3;
    x + 1
}
```
是一个代码块，值是4，这个表达式是let语句的一部分，表达式的结尾没有分号，如果加上了分号，会将其变成语句，而语句不会返回值

**Functions with Return Values：**
如果函数有返回值，我们需要声明它的类型，如：
```rust
fn five() -> i32 {
    5
}

fn main() {
    let x = five();

    println!("The value of x is: {x}");
}
```
rust中函数的返回值等同于函数体最后一个表达式的值，用return关键字也可以提前返回

运行这样的代码会导致编译错误：
```rust
fn main() {
    let x = plus_one(5);

    println!("The value of x is: {x}");
}

fn plus_one(x: i32) -> i32 {
    x + 1;
}
```
函数`plus_one()`应返回一个i32的值，但语句使用了单位类型`()`表示不返回值，和函数定义矛盾，产生了mismatched type错误

## 3.4 Comments
示例：
```rust
fn main() {
    // I’m feeling lucky today
    let lucky_number = 7;
}

fn main() {
    let lucky_number = 7; // I’m feeling lucky today
}

// So we’re doing something complicated here, long enough that we need
// multiple lines of comments to do it! Whew! Hopefully, this comment will
// explain what’s going on.
```
注释可以放在代码上或代码后

## 3.5 Control Flow
### 3.5.1 if Expressions
示例：
```rust
fn main() {
    let number = 3;

    if number < 5 {
        println!("condition was true");
    } else {
        println!("condition was false");
    }
}
```

需要注意条件一定要是布尔值，否则会错误，如：
```rust
fn main() {
    let number = 3;

    if number {
        println!("number was three");
    }
}
```
rust不会隐式地将非布尔值转换为布尔值

是用else if处理多重条件：
```rust
fn main() {
    let number = 6;

    if number % 4 == 0 {
        println!("number is divisible by 4");
    } else if number % 3 == 0 {
        println!("number is divisible by 3");
    } else if number % 2 == 0 {
        println!("number is divisible by 2");
    } else {
        println!("number is not divisible by 4, 3, or 2");
    }
}
```

if是一个表达式，我们可以在let中使用它：
```rust
fn main() {
    let condition = true;
    let number = if condition { 5 } else { 6 };

    println!("The value of number is: {number}");
}
```
注意代码块的值就是最后一个表达式的值，因此if表达式的值取决于哪一个代码块被执行，这意味着if的每个分支的可能返回值都必须是同一个类型，如果不是的话，会编译出错，如：
```rust
fn main() {
    let condition = true;

    let number = if condition { 5 } else { "six" };

    println!("The value of number is: {number}");
}
```
因为变量必须有一个类型，rust必须要在编译时就知道变量number的类型，以验证变量number在其他处的有效使用

### 3.5.2 Repetition with Loops
rust包含三种循环：loop，while，for

**loop：**
示例：
```rust
fn main() {
    loop {
        println!("again!");
    }
}
```

loop可以用于检查可能会失败的操作，如检查线程是否完成了任务，如果需要将操作的结果传递给loop以外的代码，我们可以将返回值加入用于停止循环的break表达式，它会被停止的循环返回，如：
```rust
fn main() {
    let mut counter = 0;

    let result = loop {
        counter += 1;

        if counter == 10 {
            break counter * 2;
        }
    };

    println!("The result is {result}");
}
```
注意loop表达式是let语句的一部分

用循环标签在多个循环间消除歧义
如果存在嵌套循环，break和continue一般只作用于内层循环，但我们可以对一个loop指定一个loop label，将该标签与break或continue一起使用以指明关键字作用的循环是哪一个，如：
```rust
fn main() {
    let mut count = 0;
    'counting_up: loop {
        println!("count = {count}");
        let mut remaining = 10;

        loop {
            println!("remaining = {remaining}");
            if remaining == 9 {
                break;
            }
            if count == 2 {
                break 'counting_up;
            }
            remaining -= 1;
        }

        count += 1;
    }
    println!("End count = {count}");
}
```
外层循环的标签是`'counting_up`，内层循环的第一个break会退出内层循环，而`break 'counting_up`则会退出外层循环

**while：**
示例：
```rust
fn main() {
    let mut number = 3;

    while number != 0 {
        println!("{number}!");

        number -= 1;
    }

    println!("LIFTOFF!!!");
}
```

**for：**
使用for循环对一个集合的每个元素执行一些代码：
```rust
fn main() {
    let a = [10, 20, 30, 40, 50];

    for element in a {
        println!("the value is: {element}");
    }
}
```

用for循环来倒计时：
```rust
fn main() {
    for number in (1..4).rev() {
        println!("{number}!");
    }
    println!("LIFTOFF!!!");
}
```
其中的`rev()`方法用于反转`Range`

# 4 Understanding Ownership
### 4.1 What Is Ownership
所有权是一系列rust用于关于程序内存的规则，编译器在编译时会进行检查，如果其中的任意一个规则被违反，则编译不会成功，因此所有权系统的任何功能都不会在运行时减慢程序

补充：关于堆和栈
堆和栈都是运行时可供代码使用的内存
栈遵循后进先出，栈中的所有数据都必须占用已知且固定的大小，在编译时位置大小或大小可变的数据则要存放在堆上，在向堆存放数据时，我们请求一定大小的空间，内存分配器在堆上找到一处足够大的空间，将它标记为已使用，并返回它的指针，指针的大小是固定的，可以存放在栈上
入栈比在堆上分配内存要快
访问栈上的数据比访问堆上的数据要快
当调用一个函数时，传递给函数的值和函数的局部变量被压入栈中，当函数结束，这些值被弹出栈

所有权系统解决的就是跟踪哪部分代码正在使用堆上的哪部分数据，最大限度减小堆上的重复数据的数量，以及清理堆上不再使用的数据确保不会耗尽空间，也就是说，所有权的主要目的就是管理堆数据

**Ownership Rules**
- rust中的每个值都有一个所有者
- 值在任意时刻有且仅有一个所有者
- 当所有者(变量)离开作用域，这个值将被丢弃

**Variable Space**
作用域是一个项(item)在程序中有效的范围，如对于`let s = "hello"`，变量s被绑定到了一个字符串字面值，这个字符串值是硬编码进我们的程序代码中的，变量s从它被声明到当前作用域结束时都是有效的，如：
```rust
    {                      // s尚未被声明，无效
        let s = "hello";   // s从此处起有效

        // 使用s
    }                      // 作用域结束，s无效
```
两个重要的时间点：
- 当s进入作用域，它是有效的
- 这一直持续到它离开作用域为止

**The String Type**
rust的字符串类型除了字符串字面值，还有`String`类型，这个类型管理被分配到堆上的数据，能够存储在编译时未知大小的文本

我们可以用`from()`函数从字符串字面值创建`String`，如：`let s = String::from("hello");`
运算符`::`表示使用`String`类型的命名空间下的`from()`函数

`String`类型的字符串是允许被修改的：
```rust
    let mut s = String::from("hello");

    s.push_str(", world!"); // push_str()在字符串后追加字面值

    println!("{}", s); // 将打印`hello, world!`

```

**Memory and Allocation**
字符串字面值在编译时就知道其内容，文本被直接硬编码进最后的可执行文件中

对于`String`类型，为了支持一个可变的文本片段，我们需要在堆上分配一个编译时未知大小的内存来存放内容，它包括两步：
- 在运行时，向内存分配器请求内存
- 在我们处理完`String`后，向内存分配器返回内存

我们调用了`String::from()`就会导致运行时的内存请求，请求内存的方法在各个编程语言中是通用的
但释放内存的机制有不同的实现，有的编程语言使用垃圾回收机制自动记录并清理不再使用的内存，有的编程语言需要手动清理，需要精确地对一个`allocate`配对一个`free`
在Rust中，内存在拥有它的变量离开作用域后被自动释放，如：
```rust
    {
        let s = String::from("hello"); // 从此处起，s 是有效的

        // 使用 s
    }                                  // 此作用域已结束，
                                       // s 不再有效
```
当变量`s`离开了作用域，Rust调用了`drop()`函数，`String`类型的编写者可以在`drop()`函数中写上释放内存的代码，Rust在结尾的`}`处自动调用了`drop()`

**Variables and Data Interacting with Move**
Rust中，多个变量可以采用不同的方式与同一数据进行交互
如：
```rust
	let x = 5;
	let y = x;
```
我们将值`5`绑定到`x`，然后生成一个`x`的值的拷贝并绑定给`y`，现在有两个变量`x, y`，且值都是`5`，而整型是有固定大小的简单值，这两个`5`值被放入了栈中

如：
```rust
	let s1 = String::from("hello");
	let s2 = s1;
```
这里的运行方式则与上面不同

`s1`是一个`String`类型变量，它的值是`"hello"`
一个`String`由三部分组成：ptr, len, capacity，ptr是一个指向存放字符串内容的内存的指针，len是长度，capacity是容量，这些数据都存储在栈上，字符串的内容存储在堆上
长度表示`String`当前内容的字节数，容量是`String`从内存分配器获得的多少字节的内存
当我们把`s1`赋值给`s2`，栈上的`String`的数据被复制了，即我们拷贝了ptr, len, capacity的值，但堆上的值没有被拷贝

我们知道当一个变量离开作用域后，Rust会自动`drop()`函数清理变量的堆内存，显然`s1, s2`的指针指向同一处内存，我们要避免二次释放的问题

Rust会在`let s2 = s1;`之后认为`s1`不再有效，因此Rust不需要在`s1`离开其作用域后清理任何东西
我们可以尝试在`let s2 = s1;`后使用`s1`：
```rust
    let s1 = String::from("hello");
    let s2 = s1;

    println!("{}, world!", s1);
```
Rust禁止我们使用无效的引用，编译器会报错：borrow of moved value

Rust称`let s2 = s1;`的操作为移动(move)，**移动可以认为是浅拷贝+无效第一个变量**，我们认为`s1`被移动到了`s2`中

现在只有`s2`是有效的，而在`s2`离开其作用域时，就会释放自己的内存

Rust不会自动创造深拷贝

**Variables and Data Interacting with Clone**
深拷贝堆上的数据需要调用`clone()`方法，如：
```rust
    let s1 = String::from("hello");
    let s2 = s1.clone();

    println!("s1 = {}, s2 = {}", s1, s2);
```

**Stack Only Data: Copy**
回忆之前的一个例子：
```rust
	let x = 5;
	let y = x;

	println("x = {}, y = {}", x, y);
```
这段代码中，没有调用clone，x在赋值后依然有效，没有被移动到y中，这是因为像整型这种在编译时已知大小的类型是完全存储在栈上的，对于其实际的值进行拷贝也是非常快速的，不涉及到深浅拷贝的区别，都是深拷贝

Rust有叫做Copy trait的特殊注解，可以用在类似整型这样的存储在栈上的类型上，如果一个类型实现了Copy trait，那么一个旧的变量在将其赋值给其他变量后仍然可用

Rust不允许我们对自身或其任何部分实现了Drop trait的类型注解Copy trait
如果我们对一个其值离开作用域需要进行特殊处理的类型进行Copy trait注解，将会导致一个编译时错误

任何一组简单标量值的组合都可以实现Copy trait，任何不需要内存分配或某种形式资源的类型都可以实现Copy trait，如：
- 所有的整型，如`u32`
- 布尔类型`bool`
- 所有的浮点型，如`f32`
- 字符类型`char`
- 元组，当且仅当它包含的类型都实现了Copy trait，如`(i32, i32)`

**Ownership and Functions**
将值传递给函数与给变量赋值的原理类似，像函数传递值也可能会移动或拷贝，如赋值一样，如：
```rust
fn main() {
	let s = String::from("hello"); // s进入作用域

	take_ownership(s);             // s的值被移动(move)到函数中
	                               // 因此s到这里已经不再有效

	let x = 5;                     // x进入作用域

	makes_copy(x);                 // x应该被移动到函数中
	                               // 但i32是Copy的
	                               // 因此之后仍然可以使用x
	
} // 在此处，x先离开作用域，然后是s，但是s已经无效，所以没有特别的事发生

fn takes_ownership(some_string: String) { // some_string进入作用域
	println!("{}", some_string);
} // 在此处，some_string离开作用域，drop方法被调用，堆内存被释放

fn makes_copy(some_integer: i32) { // some_integer进入作用域
	println!("{}", some_integer); 
} // 在此处，some_integer离开作用域，没有特别的事发生
```
如果我们在尝试调用`take_ownership(s)`后再使用s时，会导致编译错误

**Return Values and Scope**
返回值也可以转移所有权，如：
```rust
fn main() {
	let s1 = gives_ownership();     // gives_ownership()将返回值
	                                // 移动(move)给s1

	let s2 = String::from("hello"); // s2进入作用域

	let s3 = takes_and_gives_back(s2); // s2被移动(move)到函数
	                                   // takes_and_gives_back()中
	                                   // 函数也将返回值移动(move)给
	                                   // s3
} // 在此处，s3离开作用域，被drop，s2已经被移动，无事发生，s1离开作用域，被drop

fn gives_ownership() -> String {      // gives_ownership()会将
                                      // 返回值移动给调用它的函数
    let some_string = String::from("yours"); // some_string
                                             // 进入作用域
    some_string                       // some_string被返回
                                      // 并被移动给调用的函数
}

fn takes_and_gives_back(a_string: String) -> String { // a_string
                                                      // 进入作用域
    a_string                          // a_string被返回
                                      // 并被移动给调用的函数
}
```

变量的所有权总是遵循相同的模式： 将值赋值给另一个变量时进行移动，当持有堆中数据值的变量离开作用域时，它的值被drop清理，除非对于这个数据值的所有权已经被移动至另一个变量

如果我们需要让函数使用一个值但不带走它的所有权，可以使用引用

## 4.2 References and Borrowing
函数可以使用元组返回多个值：
```rust
fn main() {
    let s1 = String::from("hello");

    let (s2, len) = calculate_length(s1);

    println!("The length of '{}' is {}.", s2, len);
}

fn calculate_length(s: String) -> (String, usize) {
    let length = s.len(); // len() returns the length of a String

    (s, length)
}
```
这段代码为了在调用函数之后可以再使用`String`，不得不将`String`再返回

我们可以提供一个对`String`值的引用，这个值属于其他变量，但我们可以通过对它的引用对其进行访问，与指针不同的是，引用确保指向某个特定类型的有效值

引用传参：
```rust
fn main() {
	let s1 = String::from("hello");

	let len = calculate_length(&s1);

	println!("The length of '{}' is {}.", s1, len);
}

fn calculate_length(s: &String) -> usize { // s是String的引用
	s.len()
} // 在此处s离开了作用域，但它不拥有对引用值的所有权，因此没有发生drop
```
我们定义了以一个对象的引用作为参数的函数，它不会获取值的所有权，引用允许我们使用一些值但不获取它们的使用权

`&s1`创建了一个指向s1的值的引用，但引用不拥有这个值
函数签名中的`&`也表示了参数的类型应该是一个引用

函数`calculate_length()`中，变量s有效的作用域和函数的参数的作用域是一样的，但s离开作用域时不会drop值，因为s没有所有权，因此也就无需归还所有权，事实上变量s仅包含了一个成员ptr，ptr存储了指向变量s1的指针

我们称创造一个引用的过程为借用

我们不能通过引用来修改值：
```rust
fn main() {
    let s = String::from("hello");

    change(&s);
}

fn change(some_string: &String) {
    some_string.push_str(", world");
}
```
引用默认是不可变的

**Mutable Reference**
可变引用允许我们修改借用的值：
```rust
fn main() {
    let mut s = String::from("hello");

    change(&mut s);
}

fn change(some_string: &mut String) {
    some_string.push_str(", world");
}
```
我们用`&mut s`创建了一个对s的值的可变引用，同时我们也修改了函数签名，使其接受一个可变引用作为参数

可变引用有一个限制：当我们对一个值创建了一个可变引用，我们就不能对它再创建其他的引用，如：
```rust
    let mut s = String::from("hello");

    let r1 = &mut s;
    let r2 = &mut s;

    println!("{}, {}", r1, r2);
```
会导致编译错误，我们不能在同一时间将s多次作为可变变量借出

这一限制限制我们用一种非常谨慎的方式使用引用的可变性，Rust的这一特性是为了在编译时就避免数据竞争(data race)，数据竞争发生的情况是：
- 两个及其以上数量的指针同时访问同一数据
- 至少有一个指针被用来写入数据
- 没有同步数据访问的机制
数据竞争会导致未定义的行为，并且难以在运行时追踪，Rust通过拒绝编译可能会存在数据竞争的代码避免数据竞争的出现

我们可以用大括号创建一个新的作用域，以允许多个可变引用，只是不能同时拥有：
```rust
fn main() {
    let mut s = String::from("hello");

    {
        let r1 = &mut s;
    } // r1 在这里离开了作用域，所以我们完全可以创建一个新的引用

    let r2 = &mut s;
}
```

相似地，结合使用可变引用和不可变引用也会导致问题：
```rust
    let mut s = String::from("hello");

    let r1 = &s; // no problem
    let r2 = &s; // no problem
    let r3 = &mut s; // BIG PROBLEM

    println!("{}, {}, and {}", r1, r2, r3);
```
当我们对一个值创建了不可变引用时，不能再创建对它的可变引用，但可以创建更多的不可变引用

一个引用的作用域从声明的地方开始一直持续到最后一次使用为止，如：
```rust
	let mut s = String::from("hello");

	let r1 = &s; // 没问题
	let r2 = &s; // 没问题
	println!("{} and {}", r1, r2);
	// 在这之后r1, r2没有再被使用

	let r3 = &mut s; // 没问题
	println!("{}", r3);
```
这段代码可以正常编译，因为最后一次对不可变引用的使用发生在对可变引用的声明之前，不可变引用的作用域在`println()`之后结束，而可变引用还未被创建，作用域没有重叠

**Dangling References**
悬垂指针指指针指向了被释放的内存，这些内存甚至可能被分配给了其他的持有者，在Rust中，编译器会确保引用永远不会变为悬垂状态，即如果我们拥有了对一些数据的引用，编译器会确保在引用离开它的作用域之前，数据不会离开它的作用域

在Rust中创建悬垂引用会导致编译错误：
```rust
fn main() {
    let reference_to_nothing = dangle();
}

fn dangle() -> &String {
    let s = String::from("hello");

    &s
}
```

为代码添加注释：
```rust
fn main() {
    let reference_to_nothing = dangle();
}

fn dangle() -> &String { // dangle() 返回一个字符串的引用

    let s = String::from("hello"); // s 是一个新字符串

    &s // 返回字符串 s 的引用
} // 这里 s 离开作用域并被drop，其内存被释放。
  // 危险！
```
变量`s`在`dangle()`内被创建，当该函数结束，`s`被释放，但我们尝试返回它的引用，那么引用将会指向一个无效的`String`

如果直接返回`s`，则没问题：
```rust
fn no_dangle() -> String {
    let s = String::from("hello");

    s
}
```
函数的返回会让`s`对值的所有权被移动出去，`s`的作用域结束后，没有值会被释放

**The Rules of References**
- 在任意给定时间，要么只能有一个可变引用，要么可以有多个不可变引用
- 引用必须总是有效的

## 4.3 The Slice Type
slice允许我们引用集合中一段连续的元素序列，而不用引用整个集合
slice是一类引用，因此没有所有权

解决一个问题：编写一个函数，该函数接收一个用空格分隔单词的字符串，并返回在该字符串中找到的第一个单词。如果函数在该字符串中并未找到空格，则整个字符串就是一个单词，此时应该返回整个字符串

如果不使用slice：
```rust
fn first_word(s: &String) -> usize {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return i;
        }
    }

    s.len()
}
```
解释：
我们需要逐个元素地检查`String`中的值是否为空格，因此用`as_bytes()`方法将其转化为字节数组
`for (i, &item) in bytes.iter().enumerate() {`在字节数组上创建了一个迭代器，其中`iter()`方法返回集合中的每一个元素，`enumerate()`方法包装了`iter()`的结果，返回元组
我们使用模式来结构`enumerate()`返回的元组
在循环中，我们使用字节的字面值语法来寻找代表空格的字节，如果找到了空格，返回其索引，否则返回字符串的长度

我们返回了一个独立的`usize`，但这个数值只有在`&String`的上下文菜才能保证它是有意义的，它是一个与`String`相分离的值，无法保证它将来仍然有效，如：
```rust
fn main() {
    let mut s = String::from("hello world");

    let word = first_word(&s); // word 的值为 5

    s.clear(); // 这清空了字符串，使其等于 ""

    // word 在此处的值仍然是 5，
    // 但是没有更多的字符串让我们可以有效地应用数值 5, word 的值现在完全无效！
}
```

如果编写一个`second_word()`函数，函数签名则需要写成这样：
`fn second_word(s: &String) -> (usize, usize) {`
我们需要跟踪一个开始索引和结尾索引，同时有了更多的从数据的某个状态计算而来的值，却完全没有跟这个状态相绑定，这些和数据的状态不相关的变量需要和数据的状态保持同步

**String Slices**
字符串slice是对`String`中一部份值的引用，如：
```rust
fn main() {
    let s = String::from("hello world");

    let hello = &s[0..5];
    let world = &s[6..11];
}
```
slice的数据结构有两部分，ptr和len，分别存储了slice的开始位置和slice的长度，如`let world = &s[6..11]`，`world`将包含一个指向`s`的第六个索引上的字节的指针，以及长度值`5`

从零开始可以忽略`starting_index`，如：
```rust
#![allow(unused)]
fn main() {
	let s = String::from("hello");
	
	let slice = &s[0..2];
	let slice = &s[..2];
}
```

以`String`的尾部结束可以忽略`ending_index`，如：
```rust
#![allow(unused)]
fn main() {
	let s = String::from("hello");
	
	let len = s.len();
	
	let slice = &s[3..len];
	let slice = &s[3..];
}
```

获取整个字符串的slice：
```rust
#![allow(unused)]
fn main() {
	let s = String::from("hello");
	
	let len = s.len();
	
	let slice = &s[0..len];
	let slice = &s[..];
}
```

注：不能从一个多字节字符的中间创建slice，索引必须在有效的UTF-8字符边界内

使用slice重写`first_word()`：
```rust
fn first_word(s: &String) -> &str {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }

    &s[..]
}
```
字符串slice的类型声明写作`&str`，现在函数返回的是与底层数据关联的单个值

函数`second_word()`的签名也可以写成`fn second_word(s: &String) -> &str {`

用slice版本尝试修改值会导致编译错误：
```rust
fn main() {
    let mut s = String::from("hello world");

    let word = first_word(&s);

    s.clear(); // error!

    println!("the first word is: {}", word);
}
```
`clear()`试图清空`String`，因此它会尝试获取一个可变引用，此时不可变引用仍在作用域内，因此Rust禁止了可变引用的获取

**String Literals as Slices**
之前我们提到字符串的字面值被存储在二进制文件中：
`let s = "Hello, world!";`
这里`s`的类型实际上是`&str`，它是一个指向二进制文件特定位置的slice，而由于`&str`是一个不可变引用，字符串字面值也是不可变的

**String Slices as Parameters**
改进`first_word()`的函数签名：
`fn first_word(s: &str) -> &str {`
它可以接受字符串的slice，如果需要传递整个字符串，就传递整个字符串的slice，也可以直接传递对`String`的引用，如：
```rust
fn main() {
    let my_string = String::from("hello world");

    // `first_word` 适用于 `String`的 slice，部分或全部
    let word = first_word(&my_string[0..6]);
    let word = first_word(&my_string[..]);
    // `first_word` 也适用于 `String` 的引用，
    // 这等价于整个 `String` 的 slice
    let word = first_word(&my_string);

    let my_string_literal = "hello world";

    // `first_word` 适用于字符串字面值的slice，部分或全部
    let word = first_word(&my_string_literal[0..6]);
    let word = first_word(&my_string_literal[..]);

    // 因为字符串字面值已经是字符串 slice 了，
    // 这也是适用的，无需 slice 语法！
    let word = first_word(my_string_literal);
}
```

**Other Slices**
其他类型同样可以创造slice：
```rust
let a = [1, 2, 3, 4, 5];

let slice = &a[1..3];

assert_eq!(slice, &[2, 3]);
```
这个slice的类型是`&[i32]`，它同样存储了指向其开始元素的指针和长度

# 5 Using Structs to Structure Related Data
## 5.1 Defining and Instantiating Structs
我们使用 `struct` 关键字定义结构体，结构体中的每一字段包括了数据名称及其类型：
```rust
struct User {
	active: bool,
	username: String,
	email: String,
	sign_in_count: u64,
}
```

定义了结构体之后，我们通过为每个字段指定具体的值为它创造一个实例，用 `key: value` 的形式进行指定，如：
```rust
fn main() {
	let user = User {
		active: true,
		username: String::from("someusername123"),
		email: String::from("someone@example.com"),
		sign_in_count: 1,
	};
}
```

对结构体内部值的访问可以使用 `.` 
如果实例是可变的，可以直接改变字段的值，如：
```rust
fn main() {
    let mut user1 = User {
        active: true,
        username: String::from("someusername123"),
        email: String::from("someone@example.com"),
        sign_in_count: 1,
    };

    user1.email = String::from("anotheremail@example.com");
}
```
注意整个实例都是可变的，Rust不允许单独标记某个字段值为可变的

我们可以在函数的结尾构造一个结构体的新实例作为最后的表达式，以返回这个实例：
```rust
fn build_user(email: String, username: String) -> User {
	User {
		active: true,
		username: username,
		email: email,
		sign_in_count: 1,
	}
}
```

**Using the Field Init Shorthand**
如果参数名称和结构体字段名称相同，我们可以字段初始化简写语法：
```rust
fn build_user(email: String, username: String) -> User {
    User {
        active: true,
        username, // 代替了 username: username
        email,  // 代替了 email: email
        sign_in_count: 1,
    }
}
```

**Creating Instance from Other Instance with Struct Update Syntax**
我们可以使用结构体更新语法来使用旧实例的大部分值创建新实例，如果不使用结构体更新语法：
```rust
fn main() {
    // --snip--
    let user2 = User {
        active: user1.active,
        username: user1.username,
        email: String::from("another@example.com"),
        sign_in_count: user1.sign_in_count,
    };
}
```

使用结构体更新语法：
```rust
fn main() {
	// --snip--
	let user2 = User {
		email: String::from("another@example.com"),
		..user1
	};
}
```
语法 `..` 指定了剩余没有显式设置值的字段的值和给定实例的对应字段相同，注意该语法必须放在最后

注意结构体更新语法和使用 `=` 的赋值是一样的，将会导致所有权的移动，在此例中，我们在创建了 `user2` 之后就不能再使用 `user1` 了，因为 `user1` 中的 `username` 字段中的 `String` 被移动给了 `user2` 中的 `username`，如果我们给 `user2` 的 `email` 和 `username` 都赋予新的 `String` 值，从而只使用 `user1` 的 `active` 和 `sign_in_count` 值，那么 `user1` 在创建 `user2` 后仍然有效，因为`active` 和 `sign_in_count` 的类型都是实现 `Copy` trait 的类型

**Using Tuple Structs Without Named Fields to Create Different Types**
Rust也支持定义和元组类似的结构体，称为元组结构体，元组结构体没有字段名，只有字段的类型

定义元组结构体：
```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
	let black = Color(0, 0, 0);
	let origin = Point(0, 0, 0);
}
```
注意变量black和origin的值的类型是不同的，因为它们是不同元组结构体的实例，我们定义的每个结构体都有其自己的类型，即使结构体的字段可能有着相同的类型
元组结构体可以用解构语法进行解构，也可以用 `.` 加索引访问单独的值

**Unit-Like Structs Without Any Fields**
没有任何字段的结构体称为类单元结构体，因为它们类似于单元类型 `()` (unit类型)，当我们需要在某个类型上实现一个trait，但不想要在这个类型上存储任何数据时，可以使用类单元结构体

声明和实例化类单元结构体：
```rust
struct AlwaysEqual;

fn main {
	let subject = AlwaysEqual; // 实例化
}
```

**Ownership of Struct Data**
在结构体定义：
```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}
```
中，我们使用了对值拥有所有权的 `String` 类型，而不是 `&str` (字符串slice)类型，因为我们想要结构体的每个实例都拥有自己的数据，并且只要该实例有效，数据就有效

但结构体也可以存储指向由其他对象拥有的数据的引用，但我们需要用上生命周期，生命周期用于保证结构体引用的数据在结构体有效时一定是有效的，如果我们需要在结构体中存储一个引用，但不指定声明周期，编译器将报错：
```rust
struct User {
    active: bool,
    username: &str,
    email: &str,
    sign_in_count: u64,
}

fn main() {
    let user1 = User {
        active: true,
        username: "someusername123",
        email: "someone@example.com",
        sign_in_count: 1,
    };
}
```

## 5.2 An Example Program Using Structs
一个简单的程序：
```rust
fn main() {
    let width1 = 30;
    let height1 = 50;

    println!(
        "The area of the rectangle is {} square pixels.",
        area(width1, height1)
    );
}

fn area(width: u32, height: u32) -> u32 {
    width * height
}
```

注意 `area` 函数的签名：`fn area(width: u32, height: u32) -> u32 {`
函数 `area` 应该用于计算出一个长方形的面积，它的两个参数没有表现它们本身的关联性

**Refactoring with Tuples**
可以采用元组传参：
```rust
fn main() {
    let rect1 = (30, 50);

    println!(
        "The area of the rectangle is {} square pixels.",
        area(rect1)
    );
}

fn area(dimensions: (u32, u32)) -> u32 {
    dimensions.0 * dimensions.1
}
```
元组增加了结构性，但没有给出元素的名称，只能以下标索引

**Refactoring with Structs: Adding More Meaning**
采用结构体传参：
```rust
struct Rectangle {
    width: u32,
    height: u32,
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };

    println!(
        "The area of the rectangle is {} square pixels.",
        area(&rect1)
    );
}

fn area(rectangle: &Rectangle) -> u32 {
    rectangle.width * rectangle.height
}
```
为了保持所有权，采用了引用传参
采用结构体，代码有了更强的解释性，函数的签名更加清晰

**Adding Useful Functionality with Derived Traits**
`println!` 宏不能用于直接打印结构体：
```rust
struct Rectangle {
    width: u32,
    height: u32,
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };

    println!("rect1 is {}", rect1);
}
```
编译器报错：
`error[E0277]: Rectangle doesn't implement std::fmt::Display`

`prinln!` 宏能处理很多类型的格式，默认情况下，`{}` 会告诉 `println!` 使用 `Display` 格式，即直接提供给终端用户查看的输出，目前所见的基本类型默认都实现了 `Display` ，因为它们的显示方式是明确的，但是对于结构体，可以有不同的显示方式，因此Rust没有对结构体实现 `Display` 

如果我们将语句改成：`println!("rect1 is {:?}", rect1);` ，`{:?}` 会告诉 `println!` 使用 `Debug` 输出格式， `Debug` trait允许我们用一种方便开发者dubug的方式打印结构体，Rust对结构体默认实现了 `Debug`

我们需要在结构体定义之前显式加上外部属性 `[derive(Debug)]` 来为结构体显式选择这个功能：
```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };

    println!("rect1 is {:?}", rect1);
}
```

也可以使用 `{:#?}`

另一种可以用 `Debug` 格式打印数据的方法是 `dbg!` 宏，`dbg!` 宏会接收一个表达式的所有权( `println!` 宏接收的是引用)，打印出调用该宏时所在的文件和行号，以及表达式的值，然后返回对该值的所有权

`dbg!` 会打印到 `stderr` 即标准错误控制台流， `println!` 会打印到 `stdout` 即标准输出控制台流

使用 `dbg!` 宏：
```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

fn main() {
    let scale = 2;
    let rect1 = Rectangle {
        width: dbg!(30 * scale),
        height: 50,
    };

    dbg!(&rect1);
}
```
我们可以将 `dbg!` 放在表达式 `(30 * scale)` 周围，因为它会返回所有权，因此 `width` 会得到相同的值，和没有 `dbg!` 是一样的
我们不希望 `rect1` 的所有权被夺走，因此也可以传递引用

结果：
```cmd
> cargo run
   Compiling rectangles v0.1.0 (file:///projects/rectangles)
    Finished dev [unoptimized + debuginfo] target(s) in 0.61s
     Running `target/debug/rectangles`
[src/main.rs:10] 30 * scale = 60
[src/main.rs:14] &rect1 = Rectangle {
    width: 60,
    height: 50,
}
```
我们可以看到文件名，行数，表达式，以及它的值( `Debug` 格式)

Rust提供了很多可以通过使用 `derive` 属性使用的 trait，为我们的自定义类型增加行为

## 5.3 Method Syntax
方法在结构体的上下文中被定义(或者是枚举或trait对象的上下文)，它们的第一个参数总是 `self` ，表示调用它们的结构体实例

**Defining Methods**
我们可以定义 `Rectangle` 结构体的 `area` 方法：
```rust
#[derive(Debug)]
struct Rectangle {
	width: u32,
	height: u32
}

impl Rectangle {
	fn area(&self) -> u32 {
		self.width * self.width
	}
}

fn main() {
	let rect1 = Rectangle {
		width: 30,
		height: 50,
	};
	
	println!("The area of the rectangle is {} square pixels.", 
	rect.area());
}
```
为了将方法定义在结构体的上下文内，我们开始了一个 `impl` 块，位于块内的所有内容都将与 `Rectangle` 内容关联
方法函数的第一个参数是 `self`
方法使用方法语法调用，即一个实例后加一个 `.`

在方法函数的签名中，我们使用了 `&self` 以替代 `rectangle: &Rectangle`，`&self` 实际上是 `self: &Self` 的缩写，而在一个 `impl` 块中，`Self` 类型是 `impl` 块类型的别名
方法函数必须有一个名叫 `self` 的参数，参数的类型是 `Self`，我们使用 `self` 作为其缩写，如果我们不希望方法函数夺取所有权，我们需要加上 `&` 表示借用
仅仅使用 `self` 作为第一个参数，而不是使用引用的情况是很少见的，通常用在该方法是将 `self` 转换为别的实例，我们希望防止调用者在转换后使用原始的实例

方法的名称和结构体中字段的名称可以相同，如：
```rust
impl Rectangle {
    fn width(&self) -> bool {
        self.width > 0
    }
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };

    if rect1.width() {
        println!("The rectangle has a nonzero width; it is {}", rect1.width);
    }
}
```
Rust编译器将通过是否 `.width` 后面有 `()` 来区分

不过与字段同名的方法一般将被定义为只返回字段的值，而不执行其他操作，称这样的方法为 `getters` ，Rust不会对其自动实现，`getters` 的使用场景在于我们将字段定义为私有，方法为公有，因此对字段的只读访问作为该类型的公共API的一部分

Rust的自动引用和解引用：
当我们使用 `object.something()` 调用方法时，Rust会自动为 `object` 添加 `&`, `&mut`, `*`，以便使 `object` 匹配方法签名，也就是说，这些代码是等价的：
```rust
p1.distance(&p2)
(&p1).distance(&p2)
```
该自动引用方法之所以有效，是因为方法总是有一个明确的接收者类型，即 `self` 的类型，因此Rust可以明确地计算出方法是仅仅读取(`&self`)，作出修改(`&mut self`)，或是获取所有权(`self`)

**Methods with More Parameters**
我们希望定义一个可以判断一个长方形是否完全包含另一个长方形的方法：
```rust
fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };
    let rect2 = Rectangle {
        width: 10,
        height: 40,
    };
    let rect3 = Rectangle {
        width: 60,
        height: 45,
    };

    println!("Can rect1 hold rect2? {}", rect1.can_hold(&rect2));
    println!("Can rect1 hold rect3? {}", rect1.can_hold(&rect3));
}
```
显然方法需要一个 `Rectangle` 类的不可变引用的类型的参数，我们在 `impl` 块中定义这个新的方法：
```rust
impl Rectangle {
	fn area(&self) -> u32 {
		self.width * self.height
	}
	fn can_hold(&self, other: &Rectangle) -> bool {
		self.width > other.width && self.height > other.height
	}
}
```

**Associated Functions**
所有在 `impl` 块中定义的函数都称为关联函数，因为它们与 `impl` 后面命名的类型相关，我们可以定义不以 `self` 为第一参数的关联函数(因此不是方法)，因为这些方法并不作用于结构体的某一特定实例
在 `String` 类型上定义的 `String::from` 函数就是一个关联函数

不作为方法的关联函数常用作放回一个结构体新实例的构造函数，名称一般为 `new` ，如：
```rust
impl Rectangle {
	fn square(size: u32) -> Self {
		Self {
			width: size,
			height: size,
		}
	}
}
```
在 `impl` 块中，关键词 `Self` 指代 `impl` 关键字后出现的类型，此处指代 `Rectangle`

我们使用结构体名和 `::` 来调用关联函数，如`let sq = Rectangle::square(3);` ，该函数位于结构体的命名空间中

`::` 语法用于关联函数和模块创造的命名空间

**Multiple impl Blocks**
每个结构体都允许拥有多个 `impl` 块：
```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}
impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

impl Rectangle {
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}
```

# 6 Enums and Pattern Matching
## 6.1 Defining an Enum
任何一个IP地址只有两种可能的形式，IPv4或IPv6，我们可以用枚举类型定义一个 `IpAddKind` ，并列出它的所用可能值 `V4` 和 `V6` ，它们称为枚举的成员
```rust
enum IpAddKind {
	 V4,
	 V6,
}
```
**Enum Values**
我们可以创建 `IpAddKind` 两个不同成员的实例：
```rust
let four = IpAddKind::V4;
let six = IpAddKind::V6;
```
枚举的成员都位于其标识符的命名空间中，注意 `IpAddKind::V4` 和 `IpAddKind::V6` 都是同一类型的：
```rust
fn route(ip_kind : IpAddKind) {}

route(IpAddKind::V4);
route(IpAddKind::V6);
```

如果我们需要存储IP地址的类型和具体的值，可以：
```rust
enum IpAddKind {
	 V4,
	 V6,
}

struct IpAddr {
	kind: IpAddrKind,
	address: String,
}

let home = IpAddr {
	kind: IpAddrKind::V4,
	address: String::from("127.0.0.1"),
}

let loopback = IpAddr {
	kind: IpAddrKind::V6,
	address: String::From("::1"),
}
```
我们使用了结构体使枚举成员和值关联

我们可以用更简洁的方式表示相同的概念，即仅仅使用枚举类型，并且将数据直接放进每个枚举成员：
```rust
enum IpAddr {
	V4(String),
	V6(String),
}

let home = IpAddr::V4(String::from("127.0.0.1"));
let loopback = IpAddr::V6(String::from("::1"));
```
我们将两个成员都关联了 `String` 值，将数据直接附加到了枚举的每个成员上，我们可以发现每个我们定义的枚举成员的名字同时也成为了构建枚举实例的函数，如 `IpAddr::V4()` 接受一个 `String` 类型参数，返回一个 `IpAddr` 类型的实例，这些构造函数会在定义枚举类型的时候自动被定义

枚举类型的每个成员可以处理不同类型和数量的数据，如：
```rust
enum IpAddr {
	V4(u8, u8, u8, u8),
	V6(String),
}

let home = IpAddr::V4(127, 0, 0, 1);

let loopback = IpAddr::V6(String::from("::1"));
```

Rust的标准库提供了用于存储和编码IP地址的类型，标准库定义了枚举类型 `IpAddr` ，而成员关联的数据的类型是自定义的结构体类型：
```rust
struct Ipv4Addr {
    // --snip--
}

struct Ipv6Addr {
    // --snip--
}

enum IpAddr {
    V4(Ipv4Addr),
    V6(Ipv6Addr),
}
```
这说明我们可以将任意类型的数据放入枚举类型的成员中，甚至另一个枚举类型

注意即使标准库有 `IpAddr` 的定义，但如果我们没有将标准库的定义引入作用域，就不会引发冲突

成员中内嵌了多种类型的枚举类型：
```rust
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}
```
该枚举类型有四个不同类型的成员：
- `Quit` 没有关联任何数据
- `Move` 包含了命名的字段，类似结构体
- `Write` 包含了一个 `String`
- `ChangeColor` 包含了3个 `i32`

事实上定义枚举类型的成员就像定义了多个不同类型的结构体，如：
```rust
struct QuitMessage; // unit struct 类单元结构体
struct MoveMessge {
	x: i32,
	y: i32,
}
struct WriteMessage(String); // tuple struct 元组结构体
struct ChangeColorMessage(i32, i32, i32); //tuple struct 元组结构体
```

结构体和枚举的另一个相似点是我们也可以在枚举类型上定义方法，如：
```rust
impl Message {
	fn call(&self) {
		// method body
	}
}
let m = Message::Write(String::from("hello"));
m.call()
```
本例中，我们创建了一个变量 `m` ，它的值是 `Message::Write(String::from("hello"))` ，当 `m.call()` 调用时，方法体通过 `&self` 参数获取了这个值，即获取了调用这个方法的值

**The Option Enum and Its Advantages Over Null Values**
`Option` 是标准库定义的一个枚举类型

在Rust中，不存在空值的概念，在有空值的语言中，变量总是处于这两种状态之一：空值和非空值
但空值的问题在于当我们尝试像使用一个非空值一样使用一个空值，会导致某种形式的错误
空值尝试表达的概念是：空值是一个因为某种原因目前无效或缺失的值

Rust没有空值，而是拥有一个可以编码存在或不存在的概念的枚举类型，即 `Option<T>` ，定义于标准库中：
```rust
enum Option<T> {
	None,
	Some(T),
}
```
`Option<T>` 包含在了preclude中，因此不需要显式地将其代入作用域，`Option<T>` 的成员也是如此，因此我们可以不使用 `Option::` 前缀来使用 `Some` 和 `None` 

`<T>` 语法表示这是一个泛型类型参数，`<T>` 意味着 `Option` 枚举中的 `Some` 成员可以包含任意类型的数据，不同的 `<T>` 可以让 `Option<T>` 成为不同的类型，如：
```rust
let some_number = Some(5);
let some_char = Some('e');

let absent_number: Option<i32> = None;
```
`some_number` 的类型是 `Option<i32>` ，`some_char` 的类型是 `Option<char>` ，我们在 `Some` 成员中指定了值，因此Rust可以推断其类型，而 `absent_number` 无法的类型无法从 `None` 推断，因此需要显式指定类型

因为 `Option<T>` 和 `T` 是完全不同的类型，编译器不允许我们像这是一个肯定有效的值一样使用 `Option<T>` ，如：
```rust
let x: i8 = 5;
let y: Option<i8> = Some(5);

let sum = x + y;
```
会编译错误，因为 `Option<i8>` 和 `i8` 的类型不同，Rust不允许将其相加，当在Rust中拥有一个 `i8` 类型的值，编译器会确保它的值总是有效的，我们无需对它做空值检查，而当我们使用 `Option<i8>` 类型时，我们需要担心可能没有值，而编译器会确保我们在使用它的值之前处理了为空的情况

因此我们在对 `Option<T>` 进行运算之前，必须将它转化为 `T` ，而这可以帮助我们捕获到空值最常见的问题之一：假设某值不为空而实际上为空

Rust通过这样消除了错误地假设一个非空值的危险，为了拥有一个可能为空的值，我们需要显式地将其类型声明为 `Option<T>` ，而当我们要使用 `Option<T>` 时，需要明确处理值为空的情况，若一个值的类型不是 `Option<T>` ，我们可以安全地认为它的值不为空

通常我们用 `match` 表达式处理枚举类型，通过枚举类型的不同成员运行不同的代码

## 6.2 The match Control Flow Construct
`match` 控制流运算符允许我们将一个值与一系列的模式相比较，并根据匹配的模式执行相应代码，模式可以是字面值，变量名，通配符等等，使用 `match` 时，编译器检查会确保所有可能的情况都得到了处理

使用 `match` 的例子：
```rust
enum Coin {
	Penny,
	Nickel,
	Dime,
	Quarter,
}

fn value_in_cents(coin: Coin) -> u8 {
	match coin {
		Coin::Penny => 1,
		Coin::Nickel => 5,
		Coin::Dime => 10,
		Coin::Quarter => 25,
	}
}
```
我们在 `match` 关键字后跟上了一个表达式，本例中是 `coin` 的值，它和 `if` 关键字的区别在于 `if` 后跟的条件表达式必须返回一个布尔类型的值，而 `match` 后的表达式的值可以是任意类型，本例中的类型是 `Coin` 枚举类型

之后我们跟上了 `match` 分支，一个 `match` 分支由两部分组成：模式+代码，第一个分支的模式是值 `Coin::Penny` ，代码是值 `1` ，每个分支由逗号隔开

`match` 表达式执行时，它将值与模式一一匹配，如果成功匹配，则执行代码，而这些代码应是表达式，该表达式的结果值将作为整个 `match` 表达式的返回值

如果需要在分支中运行多行代码，需要使用大括号，并且分支后的逗号可以省略，分支的返回值是代码块中最后一个表达式的值：
```rust
fn value_in_cents(coin: Coin) -> u8 {
	match coin {
		Coin::Penny => {
			println!("Lucky penny!");
			1	
		}
		Coin::Nickel => 5, 
		Coin::Dime => 10, 
		Coin::Quarter => 25,
	}
}
```

**Patterns That Bind to Values**
`match` 分支的另一个作用是绑定匹配模式的部分值，这也是如何从枚举成员中提取值的：
```rust
#[derive(Debug)] // 这样可以立刻看到州的名称
enum UsState {
    Alabama,
    Alaska,
    // --snip--
}

enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),
}
```
我们在枚举类型 `Coin` 的成员 `Quarter` 的中存放一个类型为 `UsState` 的值

在 `match` 表达式中，我们在匹配 `Coin::Quarter` 成员的分支的模式中增加了一个 `state` 变量，当成员 `Coin::Quarter` 匹配成功，变量 `state` 将和该成员相关联的值绑定：
```rust
fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}!", state);
            25
        }
    }
}
```
如果调用 `value_in_cents(Coin::Quarter(UsState::Alaska))`，参数变量 `coin` 将是 `Coin::Quarter(UsState::Alaska)` ，匹配时，`state` 将会与值 `UsState::Alaska` 绑定，之后我们可以通过 `state` 使用值 `UsState::Alaska`

**Matching with Option\<T\>**
`Option<T>` 也是枚举类型，因此也可以搭配 `match` 关键字：
```rust
fn plus_one(x: Option<i32>) -> Option<i32> {
	match x {
		None => None,
		Some(i) => Some(i+1),
	}
}
let five = Some(5);
let six = plus_one(five);
let none = plus_one(None);
```
当我们调用 `plus_one(five)` ，`plus_one` 函数体中的 `x` 将会是值 `Some(5)` ，然后和 `Some(i)` 匹配，因为它们是相同的成员，然后 `i` 绑定了 `Some` 中包含的值，即 `i` 拥有了值 `5` ，之后我们构建了一个含有新的值 `6` 的 `Some` 并返回

**Matches Are Exhaustive**
`match` 的分支必须包含所有可能性，如：
```rust
fn main() {
    fn plus_one(x: Option<i32>) -> Option<i32> {
        match x {
            Some(i) => Some(i + 1),
        }
    }

    let five = Some(5);
    let six = plus_one(five);
    let none = plus_one(None);
}
```
会编译错误，Rust会告诉我们哪些模式缺失了
因此Rust中的匹配是可穷尽的，必须包含所有的可能性

**Catch-all Patterns and the _ Placeholder**
在 `match` 中，我们也可以实现对特定的值采取特殊操作，而对其他的值采取默认操作：
```rust
let dice_roll = 9;
match dice_roll {
	3 => add_fancy_hat(),
	7 => remove_fancy_hat(),
	other => move_player(other),
}
fn add_fancy_hat() {}
fn remove_fancy_hat() {}
fn move_player(num_spaces: u8) {}
```
前两个分支的匹配模式是字面值 `3` 和 `7`  ，最后一个分支涵盖了 `u8` 的所有其他可能的值，其模式是名为 `other` 变量，`move_player` 函数使用了这个变量
通配模式必须放在最后

如果我们需要使用通配模式，但不希望使用其获取的值，我们可以使用模式 `_` ，这个模式可以匹配任意值但不会进行绑定，这告诉Rust我们不会使用这个值，因此Rust也不会警告我们存在未使用的变量：
```rust
fn main() {
    let dice_roll = 9;
    match dice_roll {
        3 => add_fancy_hat(),
        7 => remove_fancy_hat(),
        _ => reroll(),
    }

    fn add_fancy_hat() {}
    fn remove_fancy_hat() {}
    fn reroll() {}
}
```

如果我们只是希望满足 `match` 的可穷举性，但通配模式下不会执行任何代码，则：
```rust
fn main() {
    let dice_roll = 9;
    match dice_roll {
        3 => add_fancy_hat(),
        7 => remove_fancy_hat(),
        _ => (),
    }

    fn add_fancy_hat() {}
    fn remove_fancy_hat() {}
}
```
我们使用单元值(空元组)作为 `_` 分支的代码，明确告诉Rust我们不会使用与前面模式不匹配的值，并且不会在这种情况下运行任何代码

## 6.3 Concise Control Flow with if let
`if let` 语法用于处理只匹配一个模式的值而忽略其他模式的情况，如：
```rust
let config_max = Some(3u8);
match config_max {
	Some(max) => println!("The maximum is configured to be {}",max),
	_ => (),
}
```
该程序匹配 `config_max` 变量中的 `Option<u8>` 类型的值，并且只有在该值为 `Some` 成员时执行代码

如果使用 `if let` 语法：
```rust
let config_max = Some(3u8);
if let Some(max) = config_max {
	println!("The maximum is configured to be {}", max);
}
```
`if let` 语法中，模式和表达式通过等号分隔，它的工作方式和 `match` 相同，等号后的表达式对应 `match` 后的表达式，等号前的模式对应 `match` 中的模式，如果模式不匹配，`if let` 后的代码块不会执行

可以认为 `if let` 语法是 `match` 语法的语法糖

可以在 `if let` 语法中添加 `else` ，`else` 块中的代码和 `match` 表达式中 `_` 块中的代码相同
以下两种方式是等价的：
```rust
let mut count = 0;
match coin {
	Coin::Quarter(state) => println!("State quarter from {:?}!", state),
	_ => count += 1,
}
```
```rust
let mut count = 0;
if let Coin::Quarter(state) = coin {
	println!("State quarter from {:?}!", state);
} else {
	count += 1;
}
```

# 7 Managing Growing Projects with Packages, Crates, and Modules
目前我们编写的代码都处于一个文件的一个模块中
一个包(package)可以包含多个二进制crate项和一个可选的crate库，随着包的增长，我们可以将包中的部分代码提取出来做成独立的crate，这些crate就会作为外部依赖项
对于由一系列互相关联的包组成的超大型项目，Cargo提供了工作空间功能

封装使得我们在高的级别重用代码，当我们实现一个操作后，其他代码可以通过该代码的公共接口进行调用，而不需要它是如何实现的

作用域：代码所在的嵌套上下文中有一组定义为“in scope”的名称，同一个作用域不能有两个名称相同的项

Rust的模块系统包括：
- 包(Packages)：Cargo的一个功能，允许我们构建，测试和分享crates
- Crates：一个模块树，形成了库和二进制可执行项目
- 模块(Modules)和 use：允许我们控制作用域和路径的私有性
- 路径(Paths)：一种命名结构体、函数、模块等项的方式

## 7.1 Packages and Crates
一个crate是Rust在编译时的最小单位，如果我们用 `rustc` 命令编译，传递一个源代码文件给它，编译器会将这个文件视为一个crate
crate可以包含模块，而模块可以定义在其他文件，然后和crate一起编译

crate有两种形式：二进制项(二进制crate)或库(库crate)
二进制项可以被编译为可执行程序，如一个命令行程序或一个服务器，因此他们必须有一个 `main` 函数定义程序执行时需要做的事，我们目前创建的crate都是二进制项
库没有 `main` 函数，它们也不会被编译成可执行程序，它们提供一些函数的定义以便其他项目使用，如 `rand` crate就提供了生成随机数的函数，大多数时候，crate指的是库，和其他编程语言的library概念一致

`crate root` 是一个源文件，Rust编译器以它为起始点，并构建我们的crate的根模块

包(package)是提供一系列功能的一个或多个crate，一个包会包含一个 `Cargo.toml` 文件来描述如何构建这些crate

Cargo实际上就是一个包，它包含了一些二进制项，这些二进制项实现了Cargo的命令行指令，我们使用这些指令来构建我们的代码，同时，Cargo也包含了这些二进制项的所依赖的库，我们甚至可以利用Cargo中的这些库自己实现Cargo的命令行工具

一个包只能包含至多一个库，但可以包含任意多个二进制项
一个包不允许是空的，必须至少包含一个crate(无论是库还是二进制项)

创建一个包：
```cmd
> cargo new my-project
	Created binary (application) `my-project` package
	
> dir my-project
Cargo.toml
src

> dir my-projuct/src
main.rs
```
当我们运行了 `cargo new` ， Cargo为我们创建了一个叫做 `my-project` 的包，同时项目根目录下有一个 `Cargo.toml` 文件，可以发现 `Cargo.toml` 文件中没有提到 `src/main.rs` 文件，因为Cargo遵循 `src/main.rs` 就是与包同名的二进制项的crate根，同样地，如果Cargo发现包目录中包含了 `src/lib.rs` ，则Cargo会认为包中包含了与包同名的库，且它的crate根是 `src/lib.rs` ，crate根文件将由Cargo传给 `rustc` 来构建库或二进制项

在本例中，我们有了只包含 `src/main.rs` 的包，说明它只包含了一个名为 `my-project` 的二进制项(二进制crate)，如果一个包包含了 `src/main.rs` 和 `src/lib.rs` 说明它包含了两个crate，一个二进制项，一个库，且名字都与包相同

一个包可以有多个二进制项，其他的二进制项存储在 `src/bin` 目录下，该目录下的每个文件都会编译成独立的二进制项

## 7.2 Defining Modules to Control Scope and Privacy
**Modules Cheat Sheet**
- 从crate根节点开始
	当编译一个crate时，编译器首先在crate根文件(通常对一个库crate而言是 `src/lib.rs` ，对一个二进制crate而言是 `src/main.rs`)中寻找需要被编译的代码
- 声明模块
	在crate根文件中，我们可以声明新的模块，比如我们用 `mod garden` 声明了一个叫 `garden` 的模块，编译器会在下列路径寻找模块代码：
	- 内联：如果 `mod garden` 后是大括号而不是分号，在大括号中寻找
	- 文件 `src/garden.rs` 
	- 文件 `src/garden/mod.rs`
- 声明子模块
	在除了crate根文件以外的其他文件中，我们可以定义子模块，比如我们可以在 `src/garden.rs` 中定义 `mod vegetables` ，编译器会在以父模块命名的目录中寻找子模块代码：
	-  内联：如果 `mod vegetables` 后是大括号而不是分号，在大括号中寻找
	- 文件 `src/garden/vegetables.rs`
	- 文件 `src/garden/vegetables/mod.rs`
- 模块中的代码路径
	一旦一个模块是我们crate的一部分，我们可以在隐私规则允许的前提下，从同一个crate内的任意地方，通过代码路径引用该模块的代码
	比如一个garden vegetables模块下的 `Asparagus` 类型可以通过 `crate::garden::vegetables::Asparagus` 被找到
- 私有vs公有
	一个模块里的代码默认对其父模块私有，在声明时用 `pub mod` 以使得模块公有，若要使一个公有模块里的成员公有，需要在它们的声明前加上 `pub` 关键字
- `use` 关键字
	在一个作用域内，`use` 关键字创建了一个成员的快捷方式，用于减少长路径的重复
	如我们可以在一个作用域内通过
	`use crate::garden::vegetables::Asparagus` 创建一个快捷方式，然后在该作用域内通过 `Asparagus` 来使用该类型

**Grouping Related Code in Modules**
模块让我们可以对一个crate中的代码进行分组，以提高可读性和重用性，我们也可以利用模块来控制项的私有性，一个模块中的代码默认是私有的，私有项是不可为外部使用的内在详细实现，我们也可以将模块和其中的项标记为公开的，以允许外部代码使用和依赖它们

创建一个 `restaurant` 库，定义一些模块和函数：
`cargo new --lib restaurant`

定义一个包含了其他内置了函数的模块的模块 `front_of_house` ：
```rust
mod front_of_house {
    mod hosting {
        fn add_to_waitlist() {}

        fn seat_at_table() {}
    }

    mod serving {
        fn take_order() {}

        fn serve_order() {}

        fn take_payment() {}
    }
}
```
我们通过 `mod` 关键字定义一个模块，模块的主体在花括号内，我们可以在模块内定义其他模块，模块内还可以保存一些对其他项的定义，如结构体，枚举类型，常量，特性(trait)，函数

我们通过使用模块将相关的定义分组到一起，我们便可以基于分组对代码进行导航和使用，保持程序的组织性

`src/lib.rs` 和 `src/main.rs` 称为crate根，因为这两个文件的内容都分别在crate模块结构(模块树)的根部形成了一个名为crate的模块，以及其他的子模块，按树形组织，如：
```cmd
crate
 └── front_of_house
     ├── hosting
     │   ├── add_to_waitlist
     │   └── seat_at_table
     └── serving
         ├── take_order
         ├── serve_order
         └── take_payment
```
模块树展示了模块的父子/嵌套结构和兄弟结构，注意根是名为 `crate` 的隐式模块

## 7.3 Paths for Referring to an Item in the Module Tree
Rust使用路径在模块树中定位一个项，路径有两种形式：
- 绝对路径
	以crate根(root)开头的全路径，若是用于访问外部crate的代码的绝对路径，则以其crate名开头，若是用于访问当前crate的代码的绝对路径，则以字面值 `crate` 开头
- 相对路径
	以 `self`、`super` 或当前模块的标识符开头，路径从当前模块开始
路径中的标识符由双冒号 `::` 分割

在crate根定义一个新函数 `eat_at_restaurant`，`eat_at_restaurant` 函数是我们crate库的一个公共API，所以使用 `pub` 关键字来标记它，我们需要在其中调用 `add_to_waitlist` 函数，有两种方法：
```rust
mod front_of_house {
	mod hosting {
		fn add_to_waitlist() {}
	}
}

pub fn eat_at_restaurant() {
	// 绝对路径
	crate::front_of_house::hosting::add_to_waitlist();
	// 相对路径
	front_of_house::hosting::add_to_waitlist();
}
```
第一种方式是绝对路径，因为 `add_to_waitlist` 函数与 `eat_at_restaurant` 被定义在同一crate中，可以使用 `crate` 关键字为起始的绝对路径
第二种方式是相对路径，这个路径以 `front_of_house` 为起始，因为这个模块在模块树中，与 `eat_at_restaurant` 定义在同一层级，以模块名开头意味着该路径是相对路径

一般倾向于使用绝对路径

这段代码会产生一个编译错误，错误信息告诉我们 `hosting` 模块是私有的，在 Rust中，默认所有项(函数、方法、结构体、枚举、模块和常量)对父模块都是私有的，不允许通过 `front_of_house` 访问

Rust中，父模块中的项不能使用子模块中的私有项，但是子模块中的项可以使用它们父模块中的项，这是因为子模块的功能是封装并隐藏父模块中某个项的实现详情，因此子模块需要看到这个项定义的上下文

Rust选择以默认隐藏内部实现细节的方式来实现模块系统功能，方便更改内部代码而不会破坏外部代码，但Rust也提供了通过使用 `pub` 关键字来创建公共项，使子模块的内部部分暴露给上级模块

**Exposing Paths with the `pub` Keyword**
我们想让父模块中的 `eat_at_restaurant` 函数可以访问子模块中的 `add_to_waitlist` 函数，需要使用 `pub` 关键字来标记 `hosting` 模块：
```rust
mod front_of_house {
    pub mod hosting {
        fn add_to_waitlist() {}
    }
}

pub fn eat_at_restaurant() {
    // 绝对路径
    crate::front_of_house::hosting::add_to_waitlist();

    // 相对路径
    front_of_house::hosting::add_to_waitlist();
}
```
我们在 `mod hosting` 前添加了 `pub` 关键字，使其变成公有的，现在我们可以通过 `front_of_house` 访问 `hosting` ，即如果我们可以访问 `front_of_house`，那我们也可以访问 `hosting`
但编译仍会显示报错，因为Rust中，默认所有项(函数、方法、结构体、枚举、模块和常量)对父模块都是私有的，即 `hosting` 中的项对它是私有的，使模块公有并不使其内容也是公有的(私有性规则不但应用于模块，还应用于结构体、枚举、函数和方法)

因此，模块上的 `pub` 关键字只允许其父模块引用它，而不允许访问内部代码
模块仅是一个容器，因此我们在将模块变为公有的同时，需要更深入地选择将模块中的一个或多个项变为公有

将 `pub` 关键字放置在 `add_to_waitlist` 函数的定义之前，使其变成公有：
```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

pub fn eat_at_restaurant() {
    // 绝对路径
    crate::front_of_house::hosting::add_to_waitlist();

    // 相对路径
    front_of_house::hosting::add_to_waitlist();
}
```
现在代码可以编译通过了

在绝对路径中，我们从 `crate` 也就是crate根开始，crate 根中定义了 `front_of_house` 模块，虽然 `front_of_house` 模块对其父模块是私有的，不过因为 `eat_at_restaurant` 函数与 `front_of_house` 定义于同一模块中(即`eat_at_restaurant` 和 `front_of_house` 是兄弟)，我们可以从 `eat_at_restaurant` 中引用 `front_of_house`，因为`hosting` 模块使用了 `pub` 标记，我们可以访问 `hosting` 的父模块，所以可以访问 `hosting`，最后由于`add_to_waitlist` 函数被标记为 `pub` ，我们可以访问其父模块，所以这个函数调用是有效的

在相对路径，其逻辑与绝对路径相同，除了第一步，不同于从 crate 根开始，路径从 `front_of_house` 开始，因为 `front_of_house` 模块与 `eat_at_restaurant` 定义于同一模块，所以从 `eat_at_restaurant` 中开始定义的该模块相对路径是有效的，接下来因为 `hosting` 和 `add_to_waitlist` 被标记为 `pub`，路径其余的部分也是有效的，因此函数调用也是有效的

**Best Practices for Packages with a Binary and a Library**
我们知道包可以同时包含一个 _src/main.rs_ 二进制crate根和一个 _src/lib.rs_ 库crate根，且这两个crate默认以包名来命名

通常，这类同时包含二进制crate和库crate的包，在二进制crate中会包含刚好足够的代码来启动一个可执行文件，可执行文件调用库crate的代码(作为示例)

这类的包的模块树一般定义在 _src/lib.rs_ 中，我们通过以包名开头的路径在二进制crate中使用公有项
事实上，包中的二进制crate同其它外部crate一样，是库crate的用户，只能使用公有API

**Starting Relative Paths with super**
如果我们想要从父模块开始构建相对路径，而不是从当前模块开始，可以通过在相对路径的开头使用 `super` ，类似以 `..` 语法开始一个文件系统路径

`back_of_house` 模块中的定义的 `fix_incorrect_order` 函数通过指定的 `super` 起始的 `serve_order` 路径，来调用父模块中的 `deliver_order` 函数：
```rust
fn deliver_order() {}

mod back_of_house {
	fn fix_incorrect_order() {
		cook_order();
		super::deliver_order();
	}
	fn cook_order() {}
}
```
`fix_incorrect_order` 函数在 `back_of_house` 模块中，我们使用 `super` 进入了 `back_of_house` 的父模块，即本例中的 `crate` 根，以找到定义于其中的 `deliver_order` 

**Making Structs and Enums Public**
`pub` 同样可以用于创建公有的结构体和枚举类型，但需要注意的是，在一个结构体定义的前面使用了 `pub` ，这个结构体会变成公有的，但是这个结构体的字段仍然是私有的，我们需要根据情况决定每个字段是否公有

定义了一个公有结构体 `back_of_house:Breakfast`，其中有一个公有字段 `toast` 和私有字段 `seasonal_fruit` ：
```rust
mod back_of_house {
	pub struct Breakfast {
		pub toast: String,
		seasonal_fruit: String,
	}
	impl Breakfase {
		pub in summer(toast: &str) -> Breakfast {
			Breakfast {
				toast: String::from(toast),
				seasonal_fruit: String::from("peaches"),
			}
		}
	}
}

pub fn ead_at_restaurant() {
	// 在夏天订购一个黑麦土司作为早餐
	let mut meal = back_of_house::Breakfast::summer("Rye");
	// 改变主意更换想要面包的类型
	meal.toast = String::from("Wheat");
	println!("I'd like {} toast please", meal.toast);

	// 如果取消下一行的注释代码不能编译； 
	// 不允许查看或修改早餐附带的季节水果 
	// meal.seasonal_fruit = String::from("blueberries");
}
```
因为 `back_of_house::Breakfast` 结构体的 `toast` 字段是公有的，所以我们可以在 `eat_at_restaurant` 中使用点号来随意的读写 `toast` 字段，但因为`seasonal_fruit` 是私有的，我们不能在 `eat_at_restaurant` 中使用 `seasonal_fruit` 字段

另外要注意的是，因为 `back_of_house::Breakfast` 具有私有字段，所以这个结构体需要提供一个公共的关联函数来构造 `Breakfast` 的实例(在本例中是 `summer`)
如果 `Breakfast` 没有这样的函数，我们将无法在 `eat_at_restaurant` 中创建 `Breakfast` 实例，因为我们不能在 `eat_at_restaurant` 中设置私有字段 `seasonal_fruit` 的值

而对于枚举类型来说，如果我们在 `enum` 关键字前面加上 `pub`将其设为公有，则它的所有成员都将变为公有，如：
```rust
mod back_of_house {
    pub enum Appetizer {
        Soup,
        Salad,
    }
}

pub fn eat_at_restaurant() {
    let order1 = back_of_house::Appetizer::Soup;
    let order2 = back_of_house::Appetizer::Salad;
}
```

因为枚举类型的使用大多数是使用它的成员，因此枚举成员默认就是公有的
结构体通常使用时不一定涉及到字段内部，因此结构体的字段默认都是私有的

## 7.4 Bring Paths into Scope with the use Keyword
在一个作用域内，我们可以使用 `use` 关键字创建一个短路径，然后就可以在作用域中的任何地方使用这个更短的名字

用 `use crate::front_of_house::hosting` 将 `crate::front_of_house::hosting` 模块引入
`eat_at_restaurant` 函数的所处的作用域：
```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```
之后只需要指定 `hosting::add_to_waitlist` 即可在 `eat_at_restaurant` 中调用 `add_to_waitlist` 函数

在作用域中使用 `use` 将路径引入作用域类似于在文件系统中创建软连接(符号连接/symbolic link)

本例中，我们通过在 crate 根增加 `use crate::front_of_house::hosting`，使得 `hosting`在作用域中成为有效的名称，如同 `hosting` 模块被定义于 crate 根一样

通过 `use` 将路径引入作用域时也会检查私有性

注意 `use` 创建的名称只有在 `use` 关键字所在的作用域内才有效/`use` 语句只适用于其所在的作用域：
```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

use crate::front_of_house::hosting;

mod customer {
    pub fn eat_at_restaurant() {
        hosting::add_to_waitlist();
    }
}
```
将 `eat_at_restaurant` 函数移动到 `customer` 子模块，这时 `eat_at_restaurant` 函数所处的作用域是和 `use` 语句所处的作用域不同的，因此会导致编译错误

但是在子模块 `customer` 内通过 `super::hosting` 引用父模块中的这个短路径则可以，说明 `use` 语句引入的短路径是只读处于同一作用域的兄弟可见的，因此子模块可以通过父模块进行访问

**Creating Idiomatic `use` Paths**
指定 `use crate::front_of_house::hosting` ，然后在 `eat_at_restaurant`中调用 `hosting::add_to_waitlist` 是惯用的做法

但是直接将 `add_to_waitlist` 引入作用域也是可行的：
```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

use crate::front_of_house::hosting::add_to_waitlist;

pub fn eat_at_restaurant() {
    add_to_waitlist();
}
```

但是不推荐这个做法，使用 `use` 将函数引入作用域的习惯用法一般是先使用 `use` 将函数的父模块引入作用域，然后在调用函数时指定父模块，因为这样可以清晰地表明函数不是在本地定义的

而使用 `use` 引入结构体、枚举和其他项时，习惯是指定它们的完整路径，直接将它们引入作用域，如将 `HashMap` 结构体引入二进制 crate 作用域：
```rust
use std::collections::HashMap;

fn main() {
    let mut map = HashMap::new();
    map.insert(1, 2);
}
```

但由于父模块名称不同但自身名称相同的项可能产生冲突，所以应该灵活使用，如将两个具有相同名称但不同父模块的 `Result` 类型引入作用域，并引用它们：
```rust
use std::fmt;
use std::io;

fn function1() -> fmt::Result {
    // --snip--
    Ok(())
}

fn function2() -> io::Result<()> {
    // --snip--
    Ok(())
}
```
我们通过使用父模块以区分这两个 `Result` 类型

**Providing New Names with the `as` Keyword**
使用 `use` 将两个同名类型引入同一作用域这个问题还有另一个解决办法：在这个类型的路径后面，我们使用 `as` 指定一个新的本地名称或者别名：
```rust
use std::io::Result as IoResult;
use std::fmt::Result;

fn function1() -> Result {
    // --snip--
    Ok(())
}

fn function2() -> IoResult<()> {
    // --snip--
    Ok(())
}
```

**Re-exporting Names with `pub use`**
当我们用 `use` 将一个名称带入作用域时，这个新的名字默认会被认为是这个作用域内的一个私有的名字，因此虽然可以被同作用域内的项使用，但是对于作用域外是不可见的
使用 `pub use` 可以使这个名字成为公有的名字，这种技术被称为 “_重导出_(_re-exporting_)”，我们不仅将一个名称导入了当前作用域，还允许别人把它导入他们自己的作用域：
```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

在这个修改之前，外部代码需要使用路径 `restaurant::front_of_house::hosting::add_to_waitlist()` 来调用 `add_to_waitlist` 函数
现在，外部代码现在可以使用路径 `restaurant::hosting::add_to_waitlist` 来调用 `add_to_waitlist` 函数，即 `pub use` 从根模块重导出了 `hosting` 模块

使用 `pub use`，我们可以使用一种结构编写代码，却将不同的结构形式暴露出来

**Using External Packages**
在第二章中我们编写了一个猜猜看游戏，那个项目使用了一个外部包，`rand`，来生成随机数。为了在项目中使用 `rand`，在 _Cargo.toml_ 中加入了如下行：
`rand = "0.8.5"`
将 `rand` 作为依赖项在 _Cargo.toml_ 中加入告诉了 Cargo 要从 crates.io 下载 `rand` 和其依赖，以使 `rand` 可在我们的项目代码中使用

接着，我们使用 `use::rand(包名)::项名` 将我们需要的项引入作用域
第二章的 “生成一个随机数” 部分中，我们曾将 `Rng` trait 引入作用域并调用了 `rand::thread_rng` 函数：
```rust
use std::io;
use rand::Rng;

fn main() {
    let secret_number = rand::thread_rng().gen_range(1..=100);
}
```

crates.io 上有很多 Rust 社区成员发布的包，将其引入自己的项目都需要相同的步骤：在 _Cargo.toml_ 列出所需要的包并通过 `use` 将其中定义的项引入项目包的作用域中

注意 `std` 标准库对于我们的包来说也是外部 crate
因为标准库随 Rust 语言一同分发，无需修改 _Cargo.toml_ 来引入 `std`，不过也需要通过 `use` 将标准库中定义的项引入项目包的作用域中来引用它们，比如我们使用的 `HashMap`：`use std::collections::HashMap;`
这是一个以标准库 crate 名 `std` 为开头的的绝对路径

**Using Nested Paths to Clean Up Large `use` Lists**
当需要引入很多定义于相同包或相同模块的项时，为每一项单独列出一行会占用源码很大的空间，如：
```rust
use rand::Rng;
// --snip--
use std::cmp::Ordering;
use std::io;
// --snip--
```
我们可以使用嵌套路径将定义于相同包或相同模块的项在一行中引入作用域
这么做需要指定路径的相同部分，接着是 `::` ，接着是大括号中的各自不同的路径部分：
```rust
use rand::Rng;
// --snip--
use std::{cmp::Ordering, io};
// --snip--
```

可以在路径的任何层级使用嵌套路径：
```rust
use std::io;
use std::io::Write;
```
等价于
```rust
use std::io::{self, Write};
```

**The Glob Operator**
如果我们希望将一个路径下的所有公有项都引入作用域，可以指定路径后跟glob 运算符 `*` ：
```rust
use std::collections::*;
```
将 `std::collections` 中定义的所有公有项引入当前作用域

glob 运算符需要谨慎使用，因为它会使得我们难以推导作用域中有什么名称以及它们是在何处定义的

## 7.5 Seperating Modules into Different Files
目前为止的所有的例子都仅在一个文件(crate 根文件中，对于库crate，根文件是 _src/lib.rs_ ，对于二进制crate，根文件是 _src/main.rs_ )中定义多个模块，当模块变得更大时，需要将它们的定义移动到单独的文件中，从而使代码更容易阅读

将 `front_of_house` 模块提取到其自己的文件中的流程：
```rust
mod front_of_house;

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```
删除 `front_of_house` 模块的大括号中的代码，只留下 `mod front_of_house;`声明
现在直到创建 _src/front_of_house.rs_ 文件之前，代码都不能编译，在此只对模块进行了声明，而没有实现

模块的代码要放入 _src/front_of_house.rs_ 的文件中，Rust编译器找到了 crate 根中名叫 `front_of_house` 的模块声明，就会在 _src_ 目录下搜寻和模块名同名的文件：
```rust
pub mod hosting {
    pub fn add_to_waitlist() {}
}
```

在模块树中，我们只需要用 `mod` 声明一次就加载该模块的文件，一旦编译器知道了这个文件是项目的一部分(通过我们的声明知道)，它就确定了该模块在模块树中的具体位置，因此项目中的其他文件应该使用该模块声明位置的路径来引用该模块，和模块文件的路径无关，因此 `mod` 并不等价于其他编程语言中的 `include` 操作

将 `hosting` 模块提取到自己的文件中的流程：
因为 `hosting` 不是根模块，而是 `front_of_house` 的子模块，因此我们需要将 `hosting` 的文件放在与它父模块同名的目录 _src/front_of_house/_ 中

我们修改 _src/front_of_house.rs_ 使之仅包含 `hosting` 模块的声明：
`pub mod hosting;`

将模块的定义写在 _src/front_of_house/hosting.rs_ 文件中：
`pub fn add_to_waitlist() {}`

**Alternate File Paths**
Rust 编译器除了支持以上介绍的文件路径外，仍然支持一种更老的文件路径，比如：
对于声明于 crate 根的 `front_of_house` 模块，编译器会在如下位置查找模块代码：
- _src/front_of_house.rs_
- _src/front_of_house/mod.rs_ (老风格)
对于 `front_of_house` 的子模块 `hosting`，编译器会在如下位置查找模块代码：
- _src/front_of_house/hosting.rs_
- _src/front_of_house/hosting/mod.rs_ (老风格)

如果对同一模块同时使用这两种路径风格，会得到一个编译错误
在同一项目中的不同模块混用不同的路径风格是允许的，但不推荐
使用 _mod.rs_ 这一文件名的风格的主要缺点是会导致项目中出现很多 _mod.rs_ 文件

**Summary**
Rust 提供了将包分成多个 crate，将 crate 分成模块，以及通过指定绝对或相对路径从一个模块引用另一个模块中定义的项的方式
可以通过使用 `use` 语句将相关路径下的项引入作用域，`use` 支持嵌套路径
模块定义的代码默认是私有的，可以选择增加 `pub` 关键字使其定义变为公有

# 8 Common Collections
Rust标准库中包含了许多集合数据类型，与内建的数组和元组数据类型不同的是，这些集合数据类型所指向的数据都是存储在堆上的，因此数据的数量不要求在编译的时候就已知，也可以在运行时增加或减少

Rust常用的集合数据类型包括：
- 向量(vector)
- 字符串(string)
- 哈希表(hash map)

## 8.1 Storing Lists of Values with Vectors
向量类型在Rust中写为 `Vec<T>` ，向量中的数据在内存中是连续紧挨着排列的，向量类型只能储存相同类型的值

**Creating a New Vector**
调用 `Vec::new` 函数创建一个新的空向量：
```rust
let v: Vec<i32> = Vec::new();
```
我们对变量 `v` 增加了类型注解，因为需要在没有往向量中插入值的情况下告诉编译器向量需要存储的数据的类型
向量类型是用泛型实现的，即 `Vec<T>` 是个泛型

如果我们使用初始值来创建一个向量，我们可以使用 `vec!` 宏，让Rust自动推导向量的类型：
```rust
let v = vec![1, 2, 3];
```
此例中，Rust会推断向量的类型为 `Vec<i32>` ，因为 `i32` 是默认的整数类型，我们提供了 `i32` 类型的初始值，Rust就会创建 `Vec<i32>` 类型的向量

**Updating a Vector**
`push` 方法可以为向量新增元素：
```rust
let mut v = Vec::new();
v.push(5);
v.push(6);
v.push(7);
v.push(8);
```
注意向量也属于变量，如果需要它可变，需要在声明时写上 `mut` 关键字
Rust可以根据 `push` 入的数据的类型推断向量的类型，本例中，因为 `push` 入的数据是 `i32` ，Rust推断向量的类型是 `Vec<i32>` ，因此省略了类型注解

**Reading Elements of Vectors**
可以通过索引或 `get` 方法引用向量中储存的值：
```rust
let v = vec![1, 2, 3, 4, 5];

let third: &i32 = &v[2]; // 返回对特定位置元素的引用
println!("The third element is {third}");

let third: Option<&i32> = v.get(2); //返回 Option<&T>
match third {
	Some(third) => println!("The third element is {third}"),
	None => println!("There is no third element"),
}
```

显然，在尝试获取超出数组范围的元素时，通过下标索引和通过 `get` 方法的行为是不一样的：
```rust
let v = vec![1, 2, 3, 4, 5];

let does_not_exist = &v[100];
let does_not_exist = v.get(100);
```
对于直接使用 `[]` 索引，当引用一个不存在的元素时，Rust会panic，程序崩溃
如果我们认为尝试访问超过向量范围的元素是一个严重错误的情况，这时应该使程序崩溃，则采用 `[]` 直接索引即可

对于使用 `get` 方法，当它被传递了一个数组范围外的索引时，它会返回 `Option<&T>::None` ，如果我们认为偶尔尝试访问超过向量范围的元素是一个正常情况，并且在代码添加了处理 `Some(&element)` 和 `None` 的逻辑时，则采用 `get` 方法

比如处理用户输入时，采用 `get` 方法，并添加错误处理的逻辑，比简单让程序崩溃更友好

在Rust中，我们已经知道，每当程序要获取一个有效的引用，借用检查器(borrow checker)就会根据所有权和引用规则进行检查，确保这个引用是符合规则且有效的
我们已经知道，在同一个作用域中，不能同时获取对一个值的可变引用和不可变引用，这个规则对向量类型也适用：
```rust
let mut v = vec![1, 2, 3, 4, 5];

let first = &v[0];

v.push(6);

println!("The first element is: {first}");
```
上述代码会导致编译错误，我们在 `let first = &v[0]` 中获取了对向量的第一个值的不可变引用，这个引用最后使用于 `println!` 函数，因此在这个不可变引用的作用域中，我们在 `v.push(6)` 中获取了整个向量，或者说向量中所有值的可变引用，显然发生了冲突

`v.push` 需要获取向量中所有值的可变引用的原因是当我们往向量结尾增加新的元素的时候，可能内存中没有足够的连续空间可以在之前的结尾添加这个元素，因此需要请求OS分配新的内存块，并将数据全部拷贝到新内存块中，这时原来的引用就指向了被释放的内存，成为悬空引用

**Iterating over the Values in a Vector**
可以使用 `for` 循环遍历向量中的元素：
```rust
let v = vec![100, 32, 57];
for i in &v {
	println!("{i}");
}
```
本例中，我们使用 `for` 循环来获取了向量中每个 `i32` 类型的值的不可变因引用

同样可以用 `for` 循环获取每个值的可变引用：
```rust
let mut v = vec![100, 32, 57];
for i in &mut v {
	*i += 50;
}
```
要访问可变引用所指向的值，需要使用解引用运算符 `*` 

同样，如果我们在 `for` 循环中给向量插入或删除值，会导致编译错误，因为对向量的修改涉及到对向量中所有的值进行可变引用，而在 `for` 循环结束前，对向量中的各个值的引用仍处于作用域内

**Using an Enum to Store Multiple Types**
枚举类型的成员都属于同一个枚举类型，但是它们可以关联不同类型的值，因此我们可以利用枚举类型实现在向量中存储不同类型的值：
```rust
enum SpreadsheetCell {
	Int(i32),
	Float(f64),
	Text(String),
}

let row = vec![
	SpreadsheetCell::Int(3),
	SpreadSheetCell::Text(String::from("blue")),
	SpreadsheetCell::Float(10.12),
];
```
注意本例中，成员 `Int` , `Text` , `Float` 都属于枚举类型 `SpreadsheetCell`

Rust之所以需要在编译时就知道向量中值的特定类型是因为这可以使得它准确知道存储一个元素需要分配多少内存，另外，确定的类型也方便Rust检查对于该向量中的元素的操作是否合法

如果已经知道了向量中可能存储的所有可能类型，也可采用存储枚举类型的方式达到使得在向量中存储多个类型的值的目的，对于枚举类型，Rust编译器会确保所有的情况都在 `match` 中得到了处理

**Dropping a Vector Drops Its Elements**
类似于任何其他的 `struct`，向量在其离开作用域时会被释放：
```rust
{
	let v = vec![1, 2, 3, 4];

	// do stuff with v
} // v goes out of scope and is freed here
```
当向量被释放，它包含的所有的值都会被释放
借用检查器也保证了任何对向量的值的引用只有在向量本身有效时才有效

## 8.2 Storing UTF-8 Encoded Text with Strings


