---
edition: "2024"
---
This version of the text assumes you’re using Rust 1.85.0 (released 2025-02-17) or later with `edition = "2024"` in the Cargo.toml file of all projects to configure them to use Rust 2024 edition idioms. See the [“Installation” section of Chapter 1](https://doc.rust-lang.org/book/ch01-01-installation.html) to install or update Rust.

The HTML format is available online at [https://doc.rust-lang.org/stable/book/](https://doc.rust-lang.org/stable/book/) and offline with installations of Rust made with `rustup`; run `rustup doc --book` to open.

Several community [translations](https://doc.rust-lang.org/book/appendix-06-translation.html) are also available.

This text is available in [paperback and ebook format from No Starch Press](https://nostarch.com/rust-programming-language-2nd-edition).

# Introduction
Note: This edition of the book is the same as [The Rust Programming Language](https://nostarch.com/rust-programming-language-2nd-edition) available in print and ebook format from [No Starch Press](https://nostarch.com/).

Welcome to _The Rust Programming Language_, an introductory book about Rust. The Rust programming language helps you write faster, more reliable software. High-level ergonomics and low-level control are often at odds in programming language design; Rust challenges that conflict. Through balancing powerful technical capacity and a great developer experience, Rust gives you the option to control low-level details (such as memory usage) without all the hassle traditionally associated with such control.

## Who Rust Is For
Rust is ideal for many people for a variety of reasons. Let’s look at a few of the most important groups.

### Teams of Developers
Rust is proving to be a productive tool for collaborating among large teams of developers with varying levels of systems programming knowledge. Low-level code is prone to various subtle bugs, which in most other languages can be caught only through extensive testing and careful code review by experienced developers. In Rust, the compiler plays a gatekeeper role by refusing to compile code with these elusive bugs, including concurrency bugs. By working alongside the compiler, the team can spend their time focusing on the program’s logic rather than chasing down bugs.

Rust also brings contemporary developer tools to the systems programming world:

- Cargo, the included dependency manager and build tool, makes adding, compiling, and managing dependencies painless and consistent across the Rust ecosystem.
- The Rustfmt formatting tool ensures a consistent coding style across developers.
- The rust-analyzer powers Integrated Development Environment (IDE) integration for code completion and inline error messages.

By using these and other tools in the Rust ecosystem, developers can be productive while writing systems-level code.

### Students
Rust is for students and those who are interested in learning about systems concepts. Using Rust, many people have learned about topics like operating systems development. The community is very welcoming and happy to answer student questions. Through efforts such as this book, the Rust teams want to make systems concepts more accessible to more people, especially those new to programming.

### Companies
Hundreds of companies, large and small, use Rust in production for a variety of tasks, including command line tools, web services, DevOps tooling, embedded devices, audio and video analysis and transcoding, cryptocurrencies, bioinformatics, search engines, Internet of Things applications, machine learning, and even major parts of the Firefox web browser.

### Open Source Developers
Rust is for people who want to build the Rust programming language, community, developer tools, and libraries. We’d love to have you contribute to the Rust language.

### People Who Value Speed and Stability
Rust is for people who crave speed and stability in a language. By speed, we mean both how quickly Rust code can run and the speed at which Rust lets you write programs. The Rust compiler’s checks ensure stability through feature additions and refactoring. This is in contrast to the brittle legacy code in languages without these checks, which developers are often afraid to modify. By striving for zero-cost abstractions—higher-level features that compile to lower-level code as fast as code written manually—Rust endeavors to make safe code be fast code as well.

The Rust language hopes to support many other users as well; those mentioned here are merely some of the biggest stakeholders. Overall, Rust’s greatest ambition is to eliminate the trade-offs that programmers have accepted for decades by providing safety _and_ productivity, speed _and_ ergonomics. Give Rust a try and see if its choices work for you.

## Who This Book Is For
This book assumes that you’ve written code in another programming language but doesn’t make any assumptions about which one. We’ve tried to make the material broadly accessible to those from a wide variety of programming backgrounds. We don’t spend a lot of time talking about what programming _is_ or how to think about it. If you’re entirely new to programming, you would be better served by reading a book that specifically provides an introduction to programming.

## How to Use This Book
In general, this book assumes that you’re reading it in sequence from front to back. Later chapters build on concepts in earlier chapters, and earlier chapters might not delve into details on a particular topic but will revisit the topic in a later chapter.

You’ll find two kinds of chapters in this book: concept chapters and project chapters. In concept chapters, you’ll learn about an aspect of Rust. In project chapters, we’ll build small programs together, applying what you’ve learned so far. Chapters 2, 12, and 21 are project chapters; the rest are concept chapters.

Chapter 1 explains how to install Rust, how to write a “Hello, world!” program, and how to use Cargo, Rust’s package manager and build tool. Chapter 2 is a hands-on introduction to writing a program in Rust, having you build up a number guessing game. Here we cover concepts at a high level, and later chapters will provide additional detail. If you want to get your hands dirty right away, Chapter 2 is the place for that. Chapter 3 covers Rust features that are similar to those of other programming languages, and in Chapter 4 you’ll learn about Rust’s ownership system. If you’re a particularly meticulous learner who prefers to learn every detail before moving on to the next, you might want to skip Chapter 2 and go straight to Chapter 3, returning to Chapter 2 when you’d like to work on a project applying the details you’ve learned.

Chapter 5 discusses structs and methods, and Chapter 6 covers enums, `match` expressions, and the `if let` control flow construct. You’ll use structs and enums to make custom types in Rust.

In Chapter 7, you’ll learn about Rust’s module system and about privacy rules for organizing your code and its public Application Programming Interface (API). Chapter 8 discusses some common collection data structures that the standard library provides, such as vectors, strings, and hash maps. Chapter 9 explores Rust’s error-handling philosophy and techniques.

Chapter 10 digs into generics, traits, and lifetimes, which give you the power to define code that applies to multiple types. Chapter 11 is all about testing, which even with Rust’s safety guarantees is necessary to ensure your program’s logic is correct. In Chapter 12, we’ll build our own implementation of a subset of functionality from the `grep` command line tool that searches for text within files. For this, we’ll use many of the concepts we discussed in the previous chapters.

Chapter 13 explores closures and iterators: features of Rust that come from functional programming languages. In Chapter 14, we’ll examine Cargo in more depth and talk about best practices for sharing your libraries with others. Chapter 15 discusses smart pointers that the standard library provides and the traits that enable their functionality.

In Chapter 16, we’ll walk through different models of concurrent programming and talk about how Rust helps you to program in multiple threads fearlessly. In Chapter 17, we build on that by exploring Rust’s async and await syntax, along with tasks, futures, and streams, and the lightweight concurrency model they enable.

Chapter 18 looks at how Rust idioms compare to object-oriented programming principles you might be familiar with. Chapter 19 is a reference on patterns and pattern matching, which are powerful ways of expressing ideas throughout Rust programs. Chapter 20 contains a smorgasbord of advanced topics of interest, including unsafe Rust, macros, and more about lifetimes, traits, types, functions, and closures.

In Chapter 21, we’ll complete a project in which we’ll implement a low-level multithreaded web server!

Finally, some appendixes contain useful information about the language in a more reference-like format. **Appendix A** covers Rust’s keywords, **Appendix B** covers Rust’s operators and symbols, **Appendix C** covers derivable traits provided by the standard library, **Appendix D** covers some useful development tools, and **Appendix E** explains Rust editions. In **Appendix F**, you can find translations of the book, and in **Appendix G** we’ll cover how Rust is made and what nightly Rust is.

There is no wrong way to read this book: if you want to skip ahead, go for it! You might have to jump back to earlier chapters if you experience any confusion. But do whatever works for you.

An important part of the process of learning Rust is learning how to read the error messages the compiler displays: these will guide you toward working code. As such, we’ll provide many examples that don’t compile along with the error message the compiler will show you in each situation. Know that if you enter and run a random example, it may not compile! Make sure you read the surrounding text to see whether the example you’re trying to run is meant to error. Ferris will also help you distinguish code that isn’t meant to work:

| Ferris                                                                                                    | Meaning                                          |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| ![Ferris with a question mark](https://doc.rust-lang.org/book/img/ferris/does_not_compile.svg)            | This code does not compile!                      |
| ![Ferris throwing up their hands](https://doc.rust-lang.org/book/img/ferris/panics.svg)                   | This code panics!                                |
| ![Ferris with one claw up, shrugging](https://doc.rust-lang.org/book/img/ferris/not_desired_behavior.svg) | This code does not produce the desired behavior. |

In most situations, we’ll lead you to the correct version of any code that doesn’t compile.

## Source Code
The source files from which this book is generated can be found on [GitHub](https://github.com/rust-lang/book/tree/main/src).

# 1 Getting Started
Let’s start your Rust journey! There’s a lot to learn, but every journey starts somewhere. In this chapter, we’ll discuss:

- Installing Rust on Linux, macOS, and Windows
- Writing a program that prints `Hello, world!`
- Using `cargo`, Rust’s package manager and build system

## 1.1 Installation
The first step is to install Rust. We’ll download Rust through `rustup`, a command line tool for managing Rust versions and associated tools. You’ll need an internet connection for the download.
>  我们通过 `rustup` —— 一个管理 Rust 版本和相关工具的命令行工具，来下载 Rust

Note: If you prefer not to use `rustup` for some reason, please see the [Other Rust Installation Methods page](https://forge.rust-lang.org/infra/other-installation-methods.html) for more options.

The following steps install the latest stable version of the Rust compiler. Rust’s stability guarantees ensure that all the examples in the book that compile will continue to compile with newer Rust versions. The output might differ slightly between versions because Rust often improves error messages and warnings. In other words, any newer, stable version of Rust you install using these steps should work as expected with the content of this book.
>  我们将下载 Rust 编译器的最新稳定版
>  Rust 的稳定性保证所有书中的代码都可以被更新版本的 Rust 编译器编译

### Command Line Notation
In this chapter and throughout the book, we’ll show some commands used in the terminal. Lines that you should enter in a terminal all start with `$`. You don’t need to type the `$` character; it’s the command line prompt shown to indicate the start of each command. Lines that don’t start with `$` typically show the output of the previous command. Additionally, PowerShell-specific examples will use `>` rather than `$`.

### Installing `rustup` on Linux or macOS
If you’re using Linux or macOS, open a terminal and enter the following command:

`$ curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh`

The command downloads a script and starts the installation of the `rustup` tool, which installs the latest stable version of Rust. You might be prompted for your password. If the install is successful, the following line will appear:

`Rust is installed now. Great!`

You will also need a _linker_, which is a program that Rust uses to join its compiled outputs into one file. It is likely you already have one. If you get linker errors, you should install a C compiler, which will typically include a linker. A C compiler is also useful because some common Rust packages depend on C code and will need a C compiler.

On macOS, you can get a C compiler by running:

`$ xcode-select --install`

Linux users should generally install GCC or Clang, according to their distribution’s documentation. For example, if you use Ubuntu, you can install the `build-essential` package.

### Installing `rustup` on Windows
On Windows, go to [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install) and follow the instructions for installing Rust. At some point in the installation, you’ll be prompted to install Visual Studio. This provides a linker and the native libraries needed to compile programs. If you need more help with this step, see [https://rust-lang.github.io/rustup/installation/windows-msvc.html](https://rust-lang.github.io/rustup/installation/windows-msvc.html)

The rest of this book uses commands that work in both _cmd.exe_ and PowerShell. If there are specific differences, we’ll explain which to use.

### Troubleshooting
To check whether you have Rust installed correctly, open a shell and enter this line:

`$ rustc --version`

You should see the version number, commit hash, and commit date for the latest stable version that has been released, in the following format:

`rustc x.y.z (abcabcabc yyyy-mm-dd)`

If you see this information, you have installed Rust successfully! If you don’t see this information, check that Rust is in your `%PATH%` system variable as follows.

In Windows CMD, use:

`> echo %PATH%`

In PowerShell, use:

`> echo $env:Path`

In Linux and macOS, use:

`$ echo $PATH`

If that’s all correct and Rust still isn’t working, there are a number of places you can get help. Find out how to get in touch with other Rustaceans (a silly nickname we call ourselves) on [the community page](https://www.rust-lang.org/community).

### Updating and Uninstalling
Once Rust is installed via `rustup`, updating to a newly released version is easy. From your shell, run the following update script:

`$ rustup update`

To uninstall Rust and `rustup`, run the following uninstall script from your shell:

`$ rustup self uninstall`

>  `rustup update` 即可更新 Rust 编译器
>  `rustup self uninstall` 即可卸载 Rust 和 `rustup`

### Local Documentation
The installation of Rust also includes a local copy of the documentation so that you can read it offline. Run `rustup doc` to open the local documentation in your browser.
>  `rustup doc` 打开文档

Any time a type or function is provided by the standard library and you’re not sure what it does or how to use it, use the application programming interface (API) documentation to find out!

### Text Editors and Integrated Development Environments
This book makes no assumptions about what tools you use to author Rust code. Just about any text editor will get the job done! However, many text editors and integrated development environments (IDEs) have built-in support for Rust. You can always find a fairly current list of many editors and IDEs on [the tools page](https://www.rust-lang.org/tools) on the Rust website.

### Working Offline with This Book
In several examples, we will use Rust packages beyond the standard library. To work through those examples, you will either need to have an internet connection or to have downloaded those dependencies ahead of time. To download the dependencies ahead of time, you can run the following commands. (We’ll explain what `cargo` is and what each of these commands does in detail later.)

```
$ cargo new get-dependencies 
$ cd get-dependencies 
$ cargo add rand@0.8.5 trpl@0.2.0`
```

This will cache the downloads for these packages so you will not need to download them later. Once you have run this command, you do not need to keep the `get-dependencies` folder. If you have run this command, you can use the `--offline` flag with all `cargo` commands in the rest of the book to use these cached versions instead of attempting to use the network.

## 1.2 Hello, World!
Now that you’ve installed Rust, it’s time to write your first Rust program.
It’s traditional when learning a new language to write a little program that
prints the text `Hello, world!` to the screen, so we’ll do the same here!

> Note: This book assumes basic familiarity with the command line. Rust makes
> no specific demands about your editing or tooling or where your code lives, so
> if you prefer to use an integrated development environment (IDE) instead of
> the command line, feel free to use your favorite IDE. Many IDEs now have some
> degree of Rust support; check the IDE’s documentation for details. The Rust
> team has been focusing on enabling great IDE support via `rust-analyzer`. See Appendix D for more details.

### Creating a Project Directory
You’ll start by making a directory to store your Rust code. It doesn’t matter
to Rust where your code lives, but for the exercises and projects in this book,
we suggest making a _projects_ directory in your home directory and keeping all your projects there.

Open a terminal and enter the following commands to make a _projects_ directory and a directory for the “Hello, world!” project within the _projects_ directory.

For Linux, macOS, and PowerShell on Windows, enter this:

```console
$ mkdir ~/projects
$ cd ~/projects
$ mkdir hello_world
$ cd hello_world
```

For Windows CMD, enter this:

```cmd
> mkdir "%USERPROFILE%\projects"
> cd /d "%USERPROFILE%\projects"
> mkdir hello_world
> cd hello_world
```

### Writing and Running a Rust Program
Next, make a new source file and call it _main.rs_. Rust files always end with
the _. rs_ extension. If you’re using more than one word in your filename, the
convention is to use an underscore to separate them. For example, use _hello_world. rs_ rather than _helloworld. rs_.

>  Rust 文件后缀为 `.rs`
>  如果文件名中有多个单词，使用下划线隔开

Now open the _main. rs_ file you just created and enter the code in Listing 1-1.

```rust
fn main () {
    println!("Hello, world!");
}
```

[Listing 1-1](https://doc.rust-lang.org/book/ch01-02-hello-world.html#listing-1-1): A program that prints `Hello, world!`

Save the file and go back to your terminal window in the
_~/projects/hello_world_ directory. On Linux or macOS, enter the following
commands to compile and run the file:

```console
$ rustc main.rs
$ ./main
Hello, world!
```

>  `rustc main.rs` 即可编译

On Windows, enter the command `.\main` instead of `./main`:

```powershell
> rustc main.rs
> .\main
Hello, world!
```

Regardless of your operating system, the string `Hello, world!` should print to the terminal. If you don’t see this output, refer back to the [“Troubleshooting”][troubleshooting] part of the Installation section for ways to get help.

If `Hello, world!` did print, congratulations! You’ve officially written a Rust
program. That makes you a Rust programmer—welcome!

### Anatomy of a Rust Program
Let’s review this “Hello, world!” program in detail. Here’s the first piece of
the puzzle:

```rust
fn main() {

}
```

These lines define a function named `main`. The `main` function is special: it
is always the first code that runs in every executable Rust program. Here, the
first line declares a function named `main` that has no parameters and returns
nothing. If there were parameters, they would go inside the parentheses `()`.

>  `main` 函数是 Rust 可执行程序的入口

The function body is wrapped in `{}`. Rust requires curly brackets around all
function bodies. It’s good style to place the opening curly bracket on the same line as the function declaration, adding one space in between.
>  Rust 要求函数体用 `{}` 包围，通常的风格是 `{` 不换行

> Note: If you want to stick to a standard style across Rust projects, you can
> use an automatic formatter tool called `rustfmt` to format your code in a
> particular style (more on `rustfmt` in [Appendix D][devtools]). The Rust team has included this tool with the standard Rust distribution, as `rustc` is, so it should already be installed on your computer!

>  `rustfmt` 可以用于格式化代码

The body of the `main` function holds the following code:

```rust
println!("Hello, world!");
```

This line does all the work in this little program: it prints text to the
screen. There are three important details to notice here.

First, `println!` calls a Rust macro. If it had called a function instead, it
would be entered as `println` (without the `!`). Rust macros are a way to write code that generates code to extend Rust syntax, and we’ll discuss them in more detail in [Chapter 20][ch20-macros]. For now, you just need to know that using a `!` means that you’re calling a macro instead of a normal function and that macros don’t always follow the same rules as functions.

>  使用 `!` 是在调用一个 Rust macro 而不是一个普通的函数

Second, you see the `"Hello, world!"` string. We pass this string as an argument to `println!`, and the string is printed to the screen.

Third, we end the line with a semicolon (`;`), which indicates that this
expression is over and the next one is ready to begin. Most lines of Rust code
end with a semicolon.
>  Rust 代码需要以 `;` 作为行尾

### Compiling and Running Are Separate Steps
You’ve just run a newly created program, so let’s examine each step in the
process.

Before running a Rust program, you must compile it using the Rust compiler by entering the `rustc` command and passing it the name of your source file, like this:

```console
$ rustc main.rs
```

If you have a C or C++ background, you’ll notice that this is similar to `gcc`
or `clang`. After compiling successfully, Rust outputs a binary executable.

On Linux, macOS, and PowerShell on Windows, you can see the executable by entering the `ls` command in your shell:

```console
$ ls
main  main.rs
```

On Linux and macOS, you’ll see two files. With PowerShell on Windows, you’ll
see the same three files that you would see using CMD. With CMD on Windows, you would enter the following:

```cmd
> dir /B %= the /B option says to only show the file names =%
main.exe
main.pdb
main.rs
```

This shows the source code file with the _. rs_ extension, the executable file
(_main. exe_ on Windows, but _main_ on all other platforms), and, when using
Windows, a file containing debugging information with the _. pdb_ extension.
From here, you run the _main_ or _main. exe_ file, like this:

```console
$ ./main # or .\main on Windows
```

If your _main. rs_ is your “Hello, world!” program, this line prints `Hello,
world! ` to your terminal.

If you’re more familiar with a dynamic language, such as Ruby, Python, or JavaScript, you might not be used to compiling and running a program as
separate steps. Rust is an _ahead-of-time compiled_ language, meaning you can compile a program and give the executable to someone else, and they can run it even without having Rust installed. If you give someone a _. rb_, _. py_, or _. js_ file, they need to have a Ruby, Python, or JavaScript implementation installed (respectively). But in those languages, you only need one command to compile and run your program. Everything is a trade-off in language design.

Just compiling with `rustc` is fine for simple programs, but as your project
grows, you’ll want to manage all the options and make it easy to share your
code. Next, we’ll introduce you to the Cargo tool, which will help you write real-world Rust programs.

## 1.3 Hello, Cargo!
Cargo is Rust’s build system and package manager. Most Rustaceans use this tool to manage their Rust projects because Cargo handles a lot of tasks for you, such as building your code, downloading the libraries your code depends on, and building those libraries. (We call the libraries that your code needs _dependencies_.)
>  Cargo 是 Rust 的构建系统和包管理器
>  Cargo 负责构建代码、下载依赖库、构建其他库等任务

The simplest Rust programs, like the one we’ve written so far, don’t have any
dependencies. If we had built the “Hello, world!” project with Cargo, it would
only use the part of Cargo that handles building your code. As you write more
complex Rust programs, you’ll add dependencies, and if you start a project
using Cargo, adding dependencies will be much easier to do.

Because the vast majority of Rust projects use Cargo, the rest of this book
assumes that you’re using Cargo too. Cargo comes installed with Rust if you
used the official installers discussed in the [“Installation”][installation] section. If you installed Rust through some other means, check whether Cargo is installed by entering the following in your terminal:

```console
$ cargo --version
```

If you see a version number, you have it! If you see an error, such as `command not found `, look at the documentation for your method of installation to
determine how to install Cargo separately.

### Creating a Project with Cargo
Let’s create a new project using Cargo and look at how it differs from our
original “Hello, world!” project. Navigate back to your _projects_ directory (or wherever you decided to store your code). Then, on any operating system, run the following:

```console
$ cargo new hello_cargo
$ cd hello_cargo
```

The first command creates a new directory and project called _hello_cargo_.
We’ve named our project _hello_cargo_, and Cargo creates its files in a
directory of the same name.

Go into the _hello_cargo_ directory and list the files. You’ll see that Cargo
has generated two files and one directory for us: a _Cargo. toml_ file and a
_src_ directory with a _main. rs_ file inside.

>  `cargo new xxx` 会创建一个新目录，并且会创建一些文件，包括了 `Cargo.toml` 和 `src/main.rs` 以及 `.gitignore`

It has also initialized a new Git repository along with a _. gitignore_ file.
Git files won’t be generated if you run `cargo new` within an existing Git
repository; you can override this behavior by using `cargo new --vcs=git`.

> Note: Git is a common version control system. You can change `cargo new` to use a different version control system or no version control system by using the `--vcs` flag. Run `cargo new --help` to see the available options.

Open _Cargo. toml_ in your text editor of choice. It should look similar to the
code in Listing 1-2.

```toml
[package]
name = "hello_cargo"
version = "0.1.0"
edition = "2024"

[dependencies]
```

[Listing 1-2](https://doc.rust-lang.org/book/ch01-03-hello-cargo.html#listing-1-2): Contents of _Cargo.toml_ generated by `cargo new`

This file is in the [_TOML_][toml] (_Tom’s Obvious, Minimal Language_) format, which is Cargo’s configuration format.

The first line, `[package]`, is a section heading that indicates that the
following statements are configuring a package. As we add more information to this file, we’ll add other sections.

>  Cargo 使用 TOML 作为配置文件格式
>  `[package]` section 用于配置 package

The next three lines set the configuration information Cargo needs to compile
your program: the name, the version, and the edition of Rust to use. We’ll talk
about the `edition` key in [Appendix E][appendix-e].

The last line, `[dependencies]`, is the start of a section for you to list any of your project’s dependencies. In Rust, packages of code are referred to as _crates_. We won’t need any other crates for this project, but we will in the first project in Chapter 2, so we’ll use this dependencies section then.

>  `[dependencies]` section 用于列出项目依赖
>   Rust 将 packages of code 称为 crates

Now open _src/main. rs_ and take a look:

<span class="filename">Filename: src/main. rs</span>

```rust
fn main() {
    println!("Hello, world!");
}
```

Cargo has generated a “Hello, world!” program for you, just like the one we
wrote in Listing 1-1! So far, the differences between our project and the
project Cargo generated are that Cargo placed the code in the _src_ directory
and we have a _Cargo. toml_ configuration file in the top directory.

Cargo expects your source files to live inside the _src_ directory. The top-level project directory is just for README files, license information, configuration files, and anything else not related to your code. Using Cargo helps you organize your projects. There’s a place for everything, and everything is in its place.

>  Cargo 要求源文件都放在 `src` 目录，顶级目录只用于放 README, license, 配置文件，以及和代码无关的东西

If you started a project that doesn’t use Cargo, as we did with the “Hello,
world!” project, you can convert it to a project that does use Cargo. Move the project code into the _src_ directory and create an appropriate _Cargo. toml_
file. One easy way to get that _Cargo. toml_ file is to run `cargo init`, which
will create it for you automatically.

>  `cargo init` 可以为项目自动生成 `Cargo.toml` 文件

### Building and Running a Cargo Project
Now let’s look at what’s different when we build and run the “Hello, world!”
program with Cargo! From your _hello_cargo_ directory, build your project by entering the following command:

```console
$ cargo build
   Compiling hello_cargo v0.1.0 (file:///projects/hello_cargo)
    Finished dev [unoptimized + debuginfo] target(s) in 2.85 secs
```

This command creates an executable file in _target/debug/hello_cargo_ (or
_target\debug\hello_cargo. exe_ on Windows) rather than in your current
directory. 

>  `cargo build` 会构建项目，项目会被构建在 `target/debug/...` 中 (默认构建为 debug)

Because the default build is a debug build, Cargo puts the binary in a directory named _debug_. You can run the executable with this command:

```console
$ ./target/debug/hello_cargo # or .\target\debug\hello_cargo.exe on Windows
Hello, world!
```

If all goes well, `Hello, world!` should print to the terminal. Running `cargo
build for the first time also causes Cargo to create a new file at the top level: _Cargo. lock_. This file keeps track of the exact versions of dependencies in your project. This project doesn’t have dependencies, so the file is a bit sparse. You won’t ever need to change this file manually; Cargo manages its contents for you.

>  第一次运行 `cargo build` 会在顶级目录创建 `Cargo.lock`
>  该文件追踪项目中的依赖版本，该文件应该由 Cargo 自动管理

We just built a project with `cargo build` and ran it with `./target/debug/hello_cargo`, but we can also use `cargo run` to compile the code and then run the resultant executable all in one command:

```console
$ cargo run
    Finished dev [unoptimized + debuginfo] target(s) in 0.0 secs
     Running `target/debug/hello_cargo`
Hello, world!
```

Using `cargo run` is more convenient than having to remember to run `cargo
build and then use the whole path to the binary, so most developers use ` cargo run `.

>  `cargo run` 用于运行可执行文件

Notice that this time we didn’t see output indicating that Cargo was compiling
`hello_cargo`. Cargo figured out that the files hadn’t changed, so it didn’t
rebuild but just ran the binary. If you had modified your source code, Cargo
would have rebuilt the project before running it, and you would have seen this
output:

>  `cargo run` 时，如果 Cargo 发现源文件没有更改，就不会重新构建，如果有更改，则会重新构建，再运行

```console
$ cargo run
   Compiling hello_cargo v0.1.0 (file:///projects/hello_cargo)
    Finished dev [unoptimized + debuginfo] target(s) in 0.33 secs
     Running `target/debug/hello_cargo`
Hello, world!
```

Cargo also provides a command called `cargo check`. This command quickly checks your code to make sure it compiles but doesn’t produce an executable:

```console
$ cargo check
   Checking hello_cargo v0.1.0 (file:///projects/hello_cargo)
    Finished dev [unoptimized + debuginfo] target(s) in 0.32 secs
```

>  `cargo check` 用于快速检查代码是否可以正确编译

Why would you not want an executable? Often, `cargo check` is much faster than `cargo build` because it skips the step of producing an executable. If you’re continually checking your work while writing the code, using `cargo check` will speed up the process of letting you know if your project is still compiling! As such, many Rustaceans run `cargo check` periodically as they write their program to make sure it compiles. Then they run `cargo build` when they’re ready to use the executable.

>  通常 `cargo check` 要比 `cargo build` 快，因为它跳过了生成可执行文件的过程
>  因此尽量用 `cargo check` 来检查

Let’s recap what we’ve learned so far about Cargo:

- We can create a project using `cargo new`.
- We can build a project using `cargo build`.
- We can build and run a project in one step using `cargo run`.
- We can build a project without producing a binary to check for errors using
  `cargo check`.
- Instead of saving the result of the build in the same directory as our code,
  Cargo stores it in the _target/debug_ directory.

An additional advantage of using Cargo is that the commands are the same no matter which operating system you’re working on. So, at this point, we’ll no longer provide specific instructions for Linux and macOS versus Windows.

### Building for Release
When your project is finally ready for release, you can use `cargo build --release ` to compile it with optimizations. This command will create an executable in _target/release_ instead of _target/debug_. The optimizations make your Rust code run faster, but turning them on lengthens the time it takes for your program to compile. This is why there are two different profiles: one for development, when you want to rebuild quickly and often, and another for building the final program you’ll give to a user that won’t be rebuilt repeatedly and that will run as fast as possible. If you’re benchmarking your code’s running time, be sure to run `cargo build --release` and benchmark with the executable in _target/release_.

>  如果准备发布了，可以 `cargo build --release`，Cargo 会编译且优化项目
>  编译结果会在 `target/release/` 中

### Cargo as Convention
With simple projects, Cargo doesn’t provide a lot of value over just using `rustc`, but it will prove its worth as your programs become more intricate. Once programs grow to multiple files or need a dependency, it’s much easier to let Cargo coordinate the build.

Even though the `hello_cargo` project is simple, it now uses much of the real tooling you’ll use in the rest of your Rust career. In fact, to work on any existing projects, you can use the following commands to check out the code using Git, change to that project’s directory, and build:

```console
$ git clone example.org/someproject
$ cd someproject
$ cargo build
```

For more information about Cargo, check out [its documentation][cargo].

## Summary
You’re already off to a great start on your Rust journey! In this chapter,
you’ve learned how to:

- Install the latest stable version of Rust using `rustup`
- Update to a newer Rust version
- Open locally installed documentation
- Write and run a “Hello, world!” program using `rustc` directly
- Create and run a new project using the conventions of Cargo

This is a great time to build a more substantial program to get used to reading
and writing Rust code. So, in Chapter 2, we’ll build a guessing game program. If you would rather start by learning how common programming concepts work in Rust, see Chapter 3 and then return to Chapter 2.

# 2 Programming a Guessing Game
Let’s jump into Rust by working through a hands-on project together! This chapter introduces you to a few common Rust concepts by showing you how to use them in a real program. You’ll learn about `let`, `match`, methods, associated functions, external crates, and more! In the following chapters, we’ll explore these ideas in more detail. In this chapter, you’ll just practice the fundamentals.

We’ll implement a classic beginner programming problem: a guessing game. Here’s how it works: the program will generate a random integer between 1 and 100. It will then prompt the player to enter a guess. After a guess is entered, the program will indicate whether the guess is too low or too high. If the guess is correct, the game will print a congratulatory message and exit.

## Setting Up a New Project
To set up a new project, go to the _projects_ directory that you created in Chapter 1 and make a new project using Cargo, like so:

```console
$ cargo new guessing_game
$ cd guessing_game
```

The first command, `cargo new`, takes the name of the project (`guessing_game`) as the first argument. The second command changes to the new project’s directory. 

Look at the generated _Cargo.toml_ file:

<span class="filename">Filename: Cargo.toml</span>

```toml
[package]
name = "guessing_game"
version = "0.1.0"
edition = "2024"

[dependencies]
```

As you saw in Chapter 1, `cargo new` generates a “Hello, world!” program for you. Check out the _src/main.rs_ file:

<span class="filename">Filename: src/main.rs</span>

```rust
fn main() {
    println!("Hello, world!");
}
```

Now let’s compile this “Hello, world!” program and run it in the same step using the `cargo run` command:

```console
$ cargo run
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.08s
     Running `target/debug/guessing_game`
Hello, world!
```

The `run` command comes in handy when you need to rapidly iterate on a project, as we’ll do in this game, quickly testing each iteration before moving on to the next one.

Reopen the _src/main.rs_ file. You’ll be writing all the code in this file.

## Processing a Guess
The first part of the guessing game program will ask for user input, process that input, and check that the input is in the expected form. To start, we’ll allow the player to input a guess. Enter the code in Listing 2-1 into _src/main.rs_.

Filename: src/main.rs

```rust
use std::io;

fn main() {
    println!("Guess the number!");

    println!("Please input your guess.");

    let mut guess = String::new();

    io::stdin()
        .read_line(&mut guess)
        .expect("Failed to read line");

    println!("You guessed: {guess}");
}
```

[Listing 2-1](https://doc.rust-lang.org/book/ch02-00-guessing-game-tutorial.html#listing-2-1): Code that gets a guess from the user and prints it

This code contains a lot of information, so let’s go over it line by line. To obtain user input and then print the result as output, we need to bring the `io` input/output library into scope. The `io` library comes from the standard library, known as `std`:
>  为了处理用户输入，并打印结果，需要将 `io` 库加入作用域中
>  `io` 库来自于标准库，即 `std`

```rust
use std::io
```

By default, Rust has a set of items defined in the standard library that it brings into the scope of every program. This set is called the _prelude_, and you can see everything in it [in the standard library documentation](https://doc.rust-lang.org/std/prelude/index.html).
>  Rust 默认会将一组定义在标准库中的项自动引入所有程序的作用域中，这组项被称为 prelude

If a type you want to use isn’t in the prelude, you have to bring that type into scope explicitly with a `use` statement. Using the `std::io` library provides you with a number of useful features, including the ability to accept user input.
>  如果我们想要使用的项不在 prelude 中，我们需要显式使用 `use` 语句将它加入作用域，例如 `use std::io`

As you saw in Chapter 1, the `main` function is the entry point into the program:

```rust
fn main() {
```

The `fn` syntax declares a new function; the parentheses, `()`, indicate there are no parameters; and the curly bracket, `{`, starts the body of the function.
>  `fn` 用于声明函数

As you also learned in Chapter 1, `println!` is a macro that prints a string to the screen:

```rust
    println!("Guess the number!");
    println!("Please input your guess.");
```

This code is printing a prompt stating what the game is and requesting input from the user.

>  `println!` 是打印字符串的宏

### Storing Values with Variables
Next, we’ll create a _variable_ to store the user input, like this:

```rust
    let mut guess = String::new();
```

Now the program is getting interesting! There’s a lot going on in this little line. We use the `let` statement to create the variable. Here’s another example:

```rust
let apple = 5
```

This line creates a new variable named `apples` and binds it to the value 5. In Rust, variables are immutable by default, meaning once we give the variable a value, the value won’t change. We’ll be discussing this concept in detail in the [“Variables and Mutability”](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html#variables-and-mutability) section in Chapter 3. 

>  `let` 语句用于创建变量
>  例如 `let apple = 5` 创建名为 `apple` 的变量，然后将它绑定到值 5 
>  Rust 中的变量默认不可变，意味着一旦我们为变量赋予一个值后，这个值就不会变了 `

To make a variable mutable, we add `mut` before the variable name:

```rust
let apples = 5; // immutable 
let mut bananas = 5; // mutable
```

>  要创建一个可变的变量，需要在变量名之前添加 `mut`

Note: The `//` syntax starts a comment that continues until the end of the line. Rust ignores everything in comments. We’ll discuss comments in more detail in [Chapter 3](https://doc.rust-lang.org/book/ch03-04-comments.html).
>  `//` 用于创建注释

Returning to the guessing game program, you now know that `let mut guess` will introduce a mutable variable named `guess`. The equal sign (`=`) tells Rust we want to bind something to the variable now. On the right of the equal sign is the value that `guess` is bound to, which is the result of calling `String::new`, a function that returns a new instance of a `String`. [`String`](https://doc.rust-lang.org/std/string/struct.String.html) is a string type provided by the standard library that is a growable, UTF-8 encoded bit of text.
>  `let mut guess =` 中的 `=` 告诉 Rust 我们将某个东西绑定到新引入的变量 `guess`
>  `=` 右边的值就是要绑定的目标，在上例中，这个目标是调用 `String::new` 函数的返回值，它会返回 `String` 的一个新实例
>  `String` 是标准库提供的字符串类型，它是可变的，使用 UTF-8 编码

The `::` syntax in the `::new` line indicates that `new` is an associated function of the `String` type. An _associated function_ is a function that’s implemented on a type, in this case `String`. This `new` function creates a new, empty string. You’ll find a `new` function on many types because it’s a common name for a function that makes a new value of some kind.
>  `String::new` 中的 `::new` 意味着 `new` 是 `String` 类型的关联函数
>  关联函数指的是在某个类型上实现的函数
>  `new` 函数创建一个新的空字符串
>  许多类型都有自己的 `new` 函数，它是用于创建新值的常见函数名

In full, the `let mut guess = String::new();` line has created a mutable variable that is currently bound to a new, empty instance of a `String`. Whew!

### Receiving User Input
Recall that we included the input/output functionality from the standard library with `use std::io;` on the first line of the program. Now we’ll call the `stdin` function from the `io` module, which will allow us to handle user input:

```rust
    io::stdin()
        .read_line(&mut guess)
```

If we hadn’t imported the `io` module with `use std::io;` at the beginning of the program, we could still use the function by writing this function call as `std::io::stdin`. The `stdin` function returns an instance of [`std::io::Stdin`](https://doc.rust-lang.org/std/io/struct.Stdin.html), which is a type that represents a handle to the standard input for your terminal.

>  我们调用 `io` 模块的 `stdin` 函数来处理用户输入
>  如果我们没有用 `use std::io`，我们也可以通过 `std::io::stdin` 来调用该函数
>  `stdin` 函数返回 `std::io::Stdin` 的一个实例，`std::io::Stdin` 是一个表示终端标准输入的句柄的类型

Next, the line `.read_line(&mut guess)` calls the [`read_line`](https://doc.rust-lang.org/std/io/struct.Stdin.html#method.read_line) method on the standard input handle to get input from the user. We’re also passing `&mut guess` as the argument to `read_line` to tell it what string to store the user input in. The full job of `read_line` is to take whatever the user types into standard input and append that into a string (without overwriting its contents), so we therefore pass that string as an argument. The string argument needs to be mutable so the method can change the string’s content.
>  `.read_line(&mut guess)` 调用了标准输入句柄的 `read_line` 函数，以读取用户输入
>  我们还出传入了 `&mut guess` 作为参数，该参数会存储用户输入
>  `read_line` 的工作是将用户键入标准输入的东西追加到一个字符串上 (不会覆盖它的内容)，因此我们需要传入一个可变的字符串类型作为参数

The `&` indicates that this argument is a _reference_, which gives you a way to let multiple parts of your code access one piece of data without needing to copy that data into memory multiple times. References are a complex feature, and one of Rust’s major advantages is how safe and easy it is to use references. You don’t need to know a lot of those details to finish this program. For now, all you need to know is that, like variables, references are immutable by default. Hence, you need to write `&mut guess` rather than `&guess` to make it mutable. (Chapter 4 will explain references more thoroughly.)
>  `&` 表示该参数是一个引用，这避免了将数据多次拷贝到内存
>  和变量一样，Rust 中的引用默认是不可变的，因此我们需要写为 `&mut guess` 而不是 `&guess`，以让引用可变

### Handling Potential Failure with `Result`
We’re still working on this line of code. We’re now discussing a third line of text, but note that it’s still part of a single logical line of code. The next part is this method:

```rust
        .expect("Failed to read line");
```

We could have written this code as:

```rust
io::stdin().read_line(&mut guess).expect("Failed to read line");
```

However, one long line is difficult to read, so it’s best to divide it. It’s often wise to introduce a newline and other whitespace to help break up long lines when you call a method with the `.method_name()` syntax. 
>  当我们用 `.method_name()` 的语法调用方法时，最好引入换行符和一些空格来确保可读性

Now let’s discuss what this line does.

As mentioned earlier, `read_line` puts whatever the user enters into the string we pass to it, but it also returns a `Result` value. [`Result`](https://doc.rust-lang.org/std/result/enum.Result.html) is an [_enumeration_](https://doc.rust-lang.org/book/ch06-00-enums.html), often called an _enum_, which is a type that can be in one of multiple possible states. We call each possible state a _variant_.
>  `read_line` 会将用于输入追加到它的参数中，它也同时会返回一个 `Result` 值
>  `Result` 类型是一种枚举 (通常称为 enum)，也就是可能处于多种不同状态中的一种的类型，我们把每种可能的状态称为一个变体 (variant)

[Chapter 6](https://doc.rust-lang.org/book/ch06-00-enums.html) will cover enums in more detail. The purpose of these `Result` types is to encode error-handling information.
>  这些 `Result` 类型的目的是编码错误处理信息

`Result` ’s variants are `Ok` and `Err`. The `Ok` variant indicates the operation was successful, and it contains the successfully generated value. The `Err` variant means the operation failed, and it contains information about how or why the operation failed.
>  `Result` 的变体是 `Ok` 和 `Err`
>  `Ok` 表示操作成功，并且会包含成功生成的值
>  `Err` 表示操作失败，并且会包含操作是如何并且为什么失败的信息

Values of the `Result` type, like values of any type, have methods defined on them. An instance of `Result` has an [`expect` method](https://doc.rust-lang.org/std/result/enum.Result.html#method.expect) that you can call. If this instance of `Result` is an `Err` value, `expect` will cause the program to crash and display the message that you passed as an argument to `expect`. If the `read_line` method returns an `Err`, it would likely be the result of an error coming from the underlying operating system. If this instance of `Result` is an `Ok` value, `expect` will take the return value that `Ok` is holding and return just that value to you so you can use it. In this case, that value is the number of bytes in the user’s input.
>  `Result` 类型的值，和任何其他类型的值一样，也有定义在其上的方法
>  `Result` 类型的实例具有可以调用的 `expect` 方法，如果 `Result` 实例是一个 `Err` 值，`expect` 会让程序崩溃，并显示我们作为参数传递给 `expect` 的消息
>  如果 `read_line` 返回 `Err`，很可能是由于底层 OS 出现错误导致的
>  如果这个 `Result` 实例是一个 `Ok` 值，`expect` 会取出 `Ok` 所包含的返回值，并将其直接返回，便于我们使用
>  在上例的情况下，这个值就是用户输入的字节数

If you don’t call `expect`, the program will compile, but you’ll get a warning:

```shell
$ cargo build
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
warning: unused `Result` that must be used
  --> src/main.rs:10:5
   |
10 |     io::stdin().read_line(&mut guess);
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: this `Result` may be an `Err` variant, which should be handled
   = note: `#[warn(unused_must_use)]` on by default
help: use `let _ = ...` to ignore the resulting value
   |
10 |     let _ = io::stdin().read_line(&mut guess);
   |     +++++++

warning: `guessing_game` (bin "guessing_game") generated 1 warning
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.59s

```

Rust warns that you haven’t used the `Result` value returned from `read_line`, indicating that the program hasn’t handled a possible error.

>  如果我们不对返回的 `Result` 实例调用 `expect`，在编译时会得到警告，表明程序没有处理潜在的错误

The right way to suppress the warning is to actually write error-handling code, but in our case we just want to crash this program when a problem occurs, so we can use `expect`. You’ll learn about recovering from errors in [Chapter 9](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html).

### Printing Values with `println!` Placeholders
Aside from the closing curly bracket, there’s only one more line to discuss in the code so far:

```rust
    println!("You guessed: {guess}");
```

This line prints the string that now contains the user’s input. The `{}` set of curly brackets is a placeholder: think of `{}` as little crab pincers that hold a value in place. When printing the value of a variable, the variable name can go inside the curly brackets. When printing the result of evaluating an expression, place empty curly brackets in the format string, then follow the format string with a comma-separated list of expressions to print in each empty curly bracket placeholder in the same order. 

>  `println!` 中的 `{}` 是占位符，我们可以将变量的值放在 `{}` 内
>  我们也可以在格式字符串中放上 `{}`，然后在字符串后跟上一个逗号分隔的表达式列表

Printing a variable and the result of an expression in one call to `println!` would look like this:

```rust
let x = 5; 
let y = 10;  

println!("x = {x} and y + 2 = {}", y + 2);
```

This code would print `x = 5 and y + 2 = 12`.

### Testing the First Part
Let’s test the first part of the guessing game. Run it using `cargo run`:

```shell
$ cargo run
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 6.44s
     Running `target/debug/guessing_game`
Guess the number!
Please input your guess.
6
You guessed: 6
```


## Generating a Secret Number
Next, we need to generate a secret number that the user will try to guess. The secret number should be different every time so the game is fun to play more than once. We’ll use a random number between 1 and 100 so the game isn’t too difficult. Rust doesn’t yet include random number functionality in its standard library. However, the Rust team does provide a [`rand` crate](https://crates.io/crates/rand) with said functionality.
>  Rust 标准库尚没有提供随机数生成的功能，但 Rust team 提供了 `rand` crate 来实现类似的功能

### Using a Crate to Get More Functionality
Remember that a crate is a collection of Rust source code files. The project we’ve been building is a _binary crate_, which is an executable. The `rand` crate is a _library crate_, which contains code that is intended to be used in other programs and can’t be executed on its own.
>  crate 就是一组 Rust 源文件的集合
>  我们构建的项目是二进制 crate，它是可执行的
>  `rand` crate 是库 crate，它包含了意在被使用的代码，crate 本身是不可执行的 (没有提供 `main` 函数)

Cargo’s coordination of external crates is where Cargo really shines. Before we can write code that uses `rand`, we need to modify the _Cargo.toml_ file to include the `rand` crate as a dependency. 
>  Cargo 对外部 crate 的协调是它真正出色的地方
>  在我们编写使用 `rand` crate 的代码之前，我们需要修改 `Cargo.toml` 文件，将 `rand` crate 添加为依赖项

Open that file now and add the following line to the bottom, beneath the `[dependencies]` section header that Cargo created for you. Be sure to specify `rand` exactly as we have here, with this version number, or the code examples in this tutorial may not work:
>  我们需要在 `[dependencies]` 下添加依赖项，同时可以指定依赖版本

Filename: Cargo.toml

```toml
[dependencies] 
rand = "0.8.5"
```

In the _Cargo.toml_ file, everything that follows a header is part of that section that continues until another section starts. In `[dependencies]` you tell Cargo which external crates your project depends on and which versions of those crates you require. In this case, we specify the `rand` crate with the semantic version specifier `0.8.5`. Cargo understands [Semantic Versioning](http://semver.org/) (sometimes called _SemVer_), which is a standard for writing version numbers. The specifier `0.8.5` is actually shorthand for `^0.8.5`, which means any version that is at least 0.8.5 but below 0.9.0.
>  Cargo 理解 Semantic Versioning
>  `0.8.5` 实际上是 `^0.8.5` 的缩写，表示版本至少是 `0.8.5`，但小于 `0.9.0`

Cargo considers these versions to have public APIs compatible with version 0.8.5, and this specification ensures you’ll get the latest patch release that will still compile with the code in this chapter. Any version 0.9.0 or greater is not guaranteed to have the same API as what the following examples use.
>  Cargo 认为版本在 `0.8.5` 和 `0.9.0` 之间的版本会有和 `0.8.5` 版本兼容的 API，以确保我们可以的得到能够正常编译的最新版本
>  任意大于 `0.9.0` 的版本都不保证会有相同的 API

Now, without changing any of the code, let’s build the project, as shown in Listing 2-2.

```shell
$ cargo build
  Updating crates.io index
   Locking 15 packages to latest Rust 1.85.0 compatible versions
    Adding rand v0.8.5 (available: v0.9.0)
 Compiling proc-macro2 v1.0.93
 Compiling unicode-ident v1.0.17
 Compiling libc v0.2.170
 Compiling cfg-if v1.0.0
 Compiling byteorder v1.5.0
 Compiling getrandom v0.2.15
 Compiling rand_core v0.6.4
 Compiling quote v1.0.38
 Compiling syn v2.0.98
 Compiling zerocopy-derive v0.7.35
 Compiling zerocopy v0.7.35
 Compiling ppv-lite86 v0.2.20
 Compiling rand_chacha v0.3.1
 Compiling rand v0.8.5
 Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
  Finished `dev` profile [unoptimized + debuginfo] target(s) in 2.48s

```

[Listing 2-2](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#listing-2-2): The output from running `cargo build` after adding the rand crate as a dependency

You may see different version numbers (but they will all be compatible with the code, thanks to SemVer!) and different lines (depending on the operating system), and the lines may be in a different order.

When we include an external dependency, Cargo fetches the latest versions of everything that dependency needs from the _registry_, which is a copy of data from [Crates.io](https://crates.io/). Crates.io is where people in the Rust ecosystem post their open source Rust projects for others to use.
>  当我们加入了外部依赖后，Cargo 会在构建时从 registry 获取依赖所需的所有最新版本
>  registry 是 `Crates.io` 的副本，`Crates.io` 是 Rust 生态系统中人们发布开源 Rust 项目供他人使用的平台

After updating the registry, Cargo checks the `[dependencies]` section and downloads any crates listed that aren’t already downloaded. In this case, although we only listed `rand` as a dependency, Cargo also grabbed other crates that `rand` depends on to work. After downloading the crates, Rust compiles them and then compiles the project with the dependencies available.
>  在更新完 registry 之后，Cargo 会检查 `[dependencies]` 部分，然后下载任意还没有下载的 crates
>  注意到 Cargo 还下载了 `rand` 所依赖的 crates
>  在下载之后，Cargo 会将编译它们，然后编译项目

If you immediately run `cargo build` again without making any changes, you won’t get any output aside from the `Finished` line. Cargo knows it has already downloaded and compiled the dependencies, and you haven’t changed anything about them in your _Cargo.toml_ file. Cargo also knows that you haven’t changed anything about your code, so it doesn’t recompile that either. With nothing to do, it simply exits.
>  如果我们再运行一次 `cargo build`，我们将直接看到 `Finished`

If you open the _src/main.rs_ file, make a trivial change, and then save it and build again, you’ll only see two lines of output:

```shell
$ cargo build
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.13s
```

These lines show that Cargo only updates the build with your tiny change to the _src/main.rs_ file. Your dependencies haven’t changed, so Cargo knows it can reuse what it has already downloaded and compiled for those.
>  Cargo 会识别具体哪些修改需要重新编译

#### Ensuring Reproducible Builds with the _Cargo.lock_ File
Cargo has a mechanism that ensures you can rebuild the same artifact every time you or anyone else builds your code: Cargo will use only the versions of the dependencies you specified until you indicate otherwise. For example, say that next week version 0.8.6 of the `rand` crate comes out, and that version contains an important bug fix, but it also contains a regression that will break your code. To handle this, Rust creates the _Cargo.lock_ file the first time you run `cargo build`, so we now have this in the _guessing_game_ directory.
>  Cargo 具有一个机制，可以确保每次我们或其他人构建代码的时候都能得到相同的产物: Cargo 将只使用我们指定的依赖版本，直到我们另有只是
>  例如，加入下周 `rand` 的 0.8.6 版本发布了，这个版本包含一个重要的错误修复，但也包含一个会破坏代码的回归问题
>  为了解决这个问题，Rust 会第一次运行 `cargo build` 的时候生成一个 `Cargo.lock` 文件

When you build a project for the first time, Cargo figures out all the versions of the dependencies that fit the criteria and then writes them to the _Cargo.lock_ file. When you build your project in the future, Cargo will see that the _Cargo.lock_ file exists and will use the versions specified there rather than doing all the work of figuring out versions again. This lets you have a reproducible build automatically. In other words, your project will remain at 0.8.5 until you explicitly upgrade, thanks to the _Cargo.lock_ file. Because the _Cargo.lock_ file is important for reproducible builds, it’s often checked into source control with the rest of the code in your project.
>  当我们第一次构建一个项目时，Cargo 会确定所有符合要求的依赖版本，并将它们写入 `Cargo.lock` 文件
>  在以后构建项目的时候，Cargo 会使用 `Cargo.lock` 中指定的版本，而不是重新计算所有依赖的版本，这样我们的构建就是可重复的
>  换句话说，除非我们显式升级，项目将一直将 `rand` 保持在 0.8.5 版本
>  因为 `Cargo.lock` 对于可重复构建非常重要，通常会和其他代码一起被提交到版本控制中

#### Updating a Crate to Get a New Version
When you _do_ want to update a crate, Cargo provides the command `update`, which will ignore the _Cargo.lock_ file and figure out all the latest versions that fit your specifications in _Cargo.toml_. Cargo will then write those versions to the _Cargo.lock_ file. In this case, Cargo will only look for versions greater than 0.8.5 and less than 0.9.0. 
>  当我们需要更新 crate 时，可以使用 `cargo update`
>  `cargo update` 会忽略 `Cargo.lock` 文件，然后根据 `Cargo.toml` 中的信息确定需要获取的最新版本，然后将这些版本写入 `Cargo.lock` 文件中
>  例如我们直接 `cargo update` 后，Cargo 将寻找大于 `0.8.5` 但小于 `0.9.0` 的版本

If the `rand` crate has released the two new versions 0.8.6 and 0.9.0, you would see the following if you ran `cargo update`:

```shell
$ cargo update
    Updating crates.io index
     Locking 1 package to latest Rust 1.85.0 compatible version
    Updating rand v0.8.5 -> v0.8.6 (available: v0.9.0)
```

Cargo ignores the 0.9.0 release. At this point, you would also notice a change in your _Cargo.lock_ file noting that the version of the `rand` crate you are now using is 0.8.6. 

To use `rand` version 0.9.0 or any version in the 0.9._x_ series, you’d have to update the _Cargo.toml_ file to look like this instead:
>  要使用 0.9.0 以上的版本，我们需要在 `Cargo.toml` 中显式指定

```toml
[dependencies]
rand = "0.9.0"
```

The next time you run `cargo build`, Cargo will update the registry of crates available and reevaluate your `rand` requirements according to the new version you have specified.

There’s a lot more to say about [Cargo](https://doc.rust-lang.org/cargo/) and [its ecosystem](https://doc.rust-lang.org/cargo/reference/publishing.html), which we’ll discuss in Chapter 14, but for now, that’s all you need to know. Cargo makes it very easy to reuse libraries, so Rustaceans are able to write smaller projects that are assembled from a number of packages.

### Generating a Random Number
Let’s start using `rand` to generate a number to guess. The next step is to update _src/main.rs_, as shown in Listing 2-3.

Filename: src/main.rs

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

[Listing 2-3](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#listing-2-3): Adding code to generate a random number

First we add the line `use rand::Rng;`. The `Rng` trait defines methods that random number generators implement, and this trait must be in scope for us to use those methods. Chapter 10 will cover traits in detail.
>  我们先添加了 `use rand::Rng`，`Rng` trait 定义了随机数生成器需要实现的方法，且这个 trait 必须在作用域中，我们才能使用这些方法

Next, we’re adding two lines in the middle. In the first line, we call the `rand::thread_rng` function that gives us the particular random number generator we’re going to use: one that is local to the current thread of execution and is seeded by the operating system. Then we call the `gen_range` method on the random number generator. This method is defined by the `Rng` trait that we brought into scope with the `use rand::Rng;` statement. The `gen_range` method takes a range expression as an argument and generates a random number in the range. The kind of range expression we’re using here takes the form `start..=end` and is inclusive on the lower and upper bounds, so we need to specify `1..=100` to request a number between 1 and 100.
>  之后，我们在中间加入了两行，第一行调用了 `rand::thread_rng` 函数，该函数会给我们一个特定的随机数生成器: 这个生成器是当前执行线程本地的，并由操作系统初始化
>  然后我们在该随机数生成器上调用 `gen_range` 方法，该方法由 `Rng` trait 定义，该方法接收一个范围表达式作为参数，生成该范围内的一个随机数
>  我们这里使用的范围表达式形式为 `start..=end`，上下限包含

Note: You won’t just know which traits to use and which methods and functions to call from a crate, so each crate has documentation with instructions for using it. Another neat feature of Cargo is that running the `cargo doc --open` command will build documentation provided by all your dependencies locally and open it in your browser. If you’re interested in other functionality in the `rand` crate, for example, run `cargo doc --open` and click `rand` in the sidebar on the left.
>  `cargo doc --open` 会本地构建所有依赖项提供的文档，并在浏览器中打开

The second new line prints the secret number. This is useful while we’re developing the program to be able to test it, but we’ll delete it from the final version. It’s not much of a game if the program prints the answer as soon as it starts!

Try running the program a few times:

```shell
$ cargo run
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.02s
     Running `target/debug/guessing_game`
Guess the number!
The secret number is: 7
Please input your guess.
4
You guessed: 4

$ cargo run
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.02s
     Running `target/debug/guessing_game`
Guess the number!
The secret number is: 83
Please input your guess.
5
You guessed: 5
```

You should get different random numbers, and they should all be numbers between 1 and 100. Great job!

## Comparing the Guess to the Secret Number
Now that we have user input and a random number, we can compare them. That step is shown in Listing 2-4. Note that this code won’t compile just yet, as we will explain.

Filename: src/main.rs

```rust
use std::cmp::Ordering;
use std::io;

use rand::Rng;

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

[Listing 2-4](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#listing-2-4): Handling the possible return values of comparing two numbers

First we add another `use` statement, bringing a type called `std::cmp::Ordering` into scope from the standard library. The `Ordering` type is another enum and has the variants `Less`, `Greater`, and `Equal`. These are the three outcomes that are possible when you compare two values.
>  有了用户输入和随机数之后，我们可以对其进行比较
>  我们首先添加另一个 `use` 语句，将类型 `std::cmp::Ordering` 带入作用域
>  `Ordering` 类型是另一个 enum 类型，它变体有 `Less, Greater, Equal`，表示了比较两个值的时候，可能的三个结果

Then we add five new lines at the bottom that use the `Ordering` type. The `cmp` method compares two values and can be called on anything that can be compared. It takes a reference to whatever you want to compare with: here it’s comparing `guess` to `secret_number`. Then it returns a variant of the `Ordering` enum we brought into scope with the `use` statement. We use a [`match`](https://doc.rust-lang.org/stable/book/ch06-02-match.html) expression to decide what to do next based on which variant of `Ordering` was returned from the call to `cmp` with the values in `guess` and `secret_number`.
>  然后，我们使用了 `cmp` 方法来比较两个值，该方法可以接受一个想要比较的内容的引用，然后返回一个 `Ordering` 枚举的变体
>  我们使用 `match` 表达式，根据返回的变体决定下一步的操作

A `match` expression is made up of _arms_. An arm consists of a _pattern_ to match against, and the code that should be run if the value given to `match` fits that arm’s pattern. Rust takes the value given to `match` and looks through each arm’s pattern in turn. Patterns and the `match` construct are powerful Rust features: they let you express a variety of situations your code might encounter and they make sure you handle them all. These features will be covered in detail in Chapter 6 and Chapter 19, respectively.
>  `match` 表示由多个分支 (arms) 组成，每个分支包含一个模式 (pattern)，代码会将 `match` 给定的值对模式一一比较，如果发现相同，就执行分支的代码，然后退出
>  模式和 `match` 结构是 Rust 中非常强大的特性: 它们允许你表达代码中可能遇到的各种情况，并确保你处理所有情况

Let’s walk through an example with the `match` expression we use here. Say that the user has guessed 50 and the randomly generated secret number this time is 38.

When the code compares 50 to 38, the `cmp` method will return `Ordering::Greater` because 50 is greater than 38. The `match` expression gets the `Ordering::Greater` value and starts checking each arm’s pattern. It looks at the first arm’s pattern, `Ordering::Less`, and sees that the value `Ordering::Greater` does not match `Ordering::Less`, so it ignores the code in that arm and moves to the next arm. The next arm’s pattern is `Ordering::Greater`, which _does_ match `Ordering::Greater`! The associated code in that arm will execute and print `Too big!` to the screen. The `match` expression ends after the first successful match, so it won’t look at the last arm in this scenario.
>  注意 `match` 在第一次成功比较之后就会结束

However, the code in Listing 2-4 won’t compile yet. Let’s try it:

```shell
$ cargo build
   Compiling libc v0.2.86
   Compiling getrandom v0.2.2
   Compiling cfg-if v1.0.0
   Compiling ppv-lite86 v0.2.10
   Compiling rand_core v0.6.2
   Compiling rand_chacha v0.3.0
   Compiling rand v0.8.5
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
error[E0308]: mismatched types
  --> src/main.rs:23:21
   |
23 |     match guess.cmp(&secret_number) {
   |                 --- ^^^^^^^^^^^^^^ expected `&String`, found `&{integer}`
   |                 |
   |                 arguments to this method are incorrect
   |
   = note: expected reference `&String`
              found reference `&{integer}`
note: method defined here
  --> /rustc/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/cmp.rs:964:8

For more information about this error, try `rustc --explain E0308`.
error: could not compile `guessing_game` (bin "guessing_game") due to 1 previous error
```

The core of the error states that there are _mismatched types_. Rust has a strong, static type system. However, it also has type inference. When we wrote `let mut guess = String::new()`, Rust was able to infer that `guess` should be a `String` and didn’t make us write the type. The `secret_number`, on the other hand, is a number type. A few of Rust’s number types can have a value between 1 and 100: `i32`, a 32-bit number; `u32`, an unsigned 32-bit number; `i64`, a 64-bit number; as well as others. Unless otherwise specified, Rust defaults to an `i32`, which is the type of `secret_number` unless you add type information elsewhere that would cause Rust to infer a different numerical type. The reason for the error is that Rust cannot compare a string and a number type.
>  但是目前的代码是无法编译的
>  错误的核心信息是类型不匹配，Rust 有一个强大的静态类型系统，并且它也支持类型推断
>  当我们编写 `let mut guess = String::new()` 时，Rust 能够推断出 `guess` 应该是一个 `String` 类型，而不需要我们显式地写出类型
>  然而 `secret_number` 是一个数字类型，Rust 的很多数字类型可以表示 1 到 100 之间的值，例如 `i32` (32 位有符号整数)， `u32` (32 位无符号整数)，`i64` (64 位有符号整数) 等
>  除非另有指定，Rust 默认使用 `i32` 类型
>  这里的错误就是无法比较 `String` 和 `i32` 类型

Ultimately, we want to convert the `String` the program reads as input into a number type so we can compare it numerically to the secret number. We do so by adding this line to the `main` function body:

Filename: src/main.rs

```rust
// --snip--

let mut guess = String::new();

io::stdin()
    .read_line(&mut guess)
    .expect("Failed to read line");

let guess: u32 = guess.trim().parse().expect("Please type a number!");

println!("You guessed: {guess}");

match guess.cmp(&secret_number) {
    Ordering::Less => println!("Too small!"),
    Ordering::Greater => println!("Too big!"),
    Ordering::Equal => println!("You win!"),
}
```

The line is:

```rust
let guess: u32 = guess.trim().parse().expect("Please type a number!");
```

We create a variable named `guess`. But wait, doesn’t the program already have a variable named `guess`? It does, but helpfully Rust allows us to shadow the previous value of `guess` with a new one. _Shadowing_ lets us reuse the `guess` variable name rather than forcing us to create two unique variables, such as `guess_str` and `guess`, for example. We’ll cover this in more detail in [Chapter 3](https://doc.rust-lang.org/stable/book/ch03-01-variables-and-mutability.html#shadowing), but for now, know that this feature is often used when you want to convert a value from one type to another type.
>  我们额外添加一行将 `guess` 转换为数字类型
>  我们在这一行中，创建了一个名为 `guess` 的变量，虽然之前已经有了 `guess`，但 Rust 会用新的值来遮蔽 (shadow) 之前的值
>  遮蔽允许我们重复使用变量名，而不需要创建不同的变量
>  shadowing 的特定在将一个值从一个类型转化到另一个类型非常常见

We bind this new variable to the expression `guess.trim().parse()`. The `guess` in the expression refers to the original `guess` variable that contained the input as a string. The `trim` method on a `String` instance will eliminate any whitespace at the beginning and end, which we must do before we can convert the string to a `u32`, which can only contain numerical data. The user must press enter to satisfy `read_line` and input their guess, which adds a newline character to the string. For example, if the user types 5 and presses enter, `guess` looks like this: `5\n`. The `\n` represents “newline.” (On Windows, pressing enter results in a carriage return and a newline, `\r\n`.) The `trim` method eliminates `\n` or `\r\n`, resulting in just `5`.
>  我们将变量 `guess` 绑定到 `guess.trim().parse()`
>  `trim` 方法会删除开头和结尾的空白字符，这一步是必须的，在将字符串转化为 `u32` 之前，我们必须确保字符串中只包含数字数据
>  因为用户必须按下回车来结束 `read_line`，故输入字符中总会有换行符，例如 `\n` 或 `\r\n`

The [`parse` method on strings](https://doc.rust-lang.org/stable/std/primitive.str.html#method.parse) converts a string to another type. Here, we use it to convert from a string to a number. We need to tell Rust the exact number type we want by using `let guess: u32`. The colon (`:`) after `guess` tells Rust we’ll annotate the variable’s type. Rust has a few built-in number types; the `u32` seen here is an unsigned, 32-bit integer. It’s a good default choice for a small positive number. You’ll learn about other number types in [Chapter 3](https://doc.rust-lang.org/stable/book/ch03-02-data-types.html#integer-types).
>  `parse` 方法将字符串转化为另一个类型，我们通过使用 `let guess: u32` 来告诉 Rust 我们需要的类型，即通过了类型标注

Additionally, the `u32` annotation in this example program and the comparison with `secret_number` means Rust will infer that `secret_number` should be a `u32` as well. So now the comparison will be between two values of the same type!
>  此外，Rust 还会通过比较代码，推断出 `secret_number` 的类型也应该是 `u32`，故比较最终会成功

The `parse` method will only work on characters that can logically be converted into numbers and so can easily cause errors. If, for example, the string contained `A👍%`, there would be no way to convert that to a number. Because it might fail, the `parse` method returns a `Result` type, much as the `read_line` method does (discussed earlier in [“Handling Potential Failure with `Result`”](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#handling-potential-failure-with-result)). We’ll treat this `Result` the same way by using the `expect` method again. If `parse` returns an `Err` `Result` variant because it couldn’t create a number from the string, the `expect` call will crash the game and print the message we give it. If `parse` can successfully convert the string to a number, it will return the `Ok` variant of `Result`, and `expect` will return the number that we want from the `Ok` value.
>  `parse` 方法只能对逻辑上可以转换为数字的字符起作用，因此很容易引发错误
>  由于可能引发失败，`parse` 方法返回的是一个 `Result` 类型，和 `read_line` 方法类似，我们同样使用 `expect` 方法来处理这个 `Result`
>  如果 `parse` 返回了 `Err` 变体，那么 `expect` 调用将让程序崩溃，并打印提供的信息
>  如果返回了 `Ok` 变体，则 `expect` 从 `Ok` 值中返回我们想要的数字

Let’s run the program now:

```shell
$ cargo run
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.26s
     Running `target/debug/guessing_game`
Guess the number!
The secret number is: 58
Please input your guess.
  76
You guessed: 76
Too big!
```

Nice! Even though spaces were added before the guess, the program still figured out that the user guessed 76. Run the program a few times to verify the different behavior with different kinds of input: guess the number correctly, guess a number that is too high, and guess a number that is too low.

We have most of the game working now, but the user can make only one guess. Let’s change that by adding a loop!

## Allowing Multiple Guesses with Looping
The `loop` keyword creates an infinite loop. We’ll add a loop to give users more chances at guessing the number:
>  `loop` 关键词用于创建无限循环

Filename: src/main.rs

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

As you can see, we’ve moved everything from the guess input prompt onward into a loop. Be sure to indent the lines inside the loop another four spaces each and run the program again. The program will now ask for another guess forever, which actually introduces a new problem. It doesn’t seem like the user can quit!

The user could always interrupt the program by using the keyboard shortcut ctrl-c. But there’s another way to escape this insatiable monster, as mentioned in the `parse` discussion in [“Comparing the Guess to the Secret Number”](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#comparing-the-guess-to-the-secret-number): if the user enters a non-number answer, the program will crash. We can take advantage of that to allow the user to quit, as shown here:
>  我们可以利用程序在接收非数字输入时就退出的特性来帮助用户退出程序

```shell
$ cargo run
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.23s
     Running `target/debug/guessing_game`
Guess the number!
The secret number is: 59
Please input your guess.
45
You guessed: 45
Too small!
Please input your guess.
60
You guessed: 60
Too big!
Please input your guess.
59
You guessed: 59
You win!
Please input your guess.
quit

thread 'main' panicked at src/main.rs:28:47:
Please type a number!: ParseIntError { kind: InvalidDigit }
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

Typing `quit` will quit the game, but as you’ll notice, so will entering any other non-number input. This is suboptimal, to say the least; we want the game to also stop when the correct number is guessed.

### Quitting After a Correct Guess
Let’s program the game to quit when the user wins by adding a `break` statement:

Filename: src/main.rs

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

Adding the `break` line after `You win!` makes the program exit the loop when the user guesses the secret number correctly. Exiting the loop also means exiting the program, because the loop is the last part of `main`.
>  我们添加 `break` 语句让成功匹配时退出循环

### Handling Invalid Input
To further refine the game’s behavior, rather than crashing the program when the user inputs a non-number, let’s make the game ignore a non-number so the user can continue guessing. We can do that by altering the line where `guess` is converted from a `String` to a `u32`, as shown in Listing 2-5.

Filename: src/main.rs

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

[Listing 2-5](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#listing-2-5): Ignoring a non-number guess and asking for another guess instead of crashing the program

We switch from an `expect` call to a `match` expression to move from crashing on an error to handling the error. Remember that `parse` returns a `Result` type and `Result` is an enum that has the variants `Ok` and `Err`. We’re using a `match` expression here, as we did with the `Ordering` result of the `cmp` method.
>  我们将原来的错误处理: 调用 `expect` 改为使用 `match` 表达式
>  回忆一下，`parse` 返回 `Result` 类型，`Result` 类型是一个 enum 类型，有两个变体 `Ok, Err`
>  我们使用 `match` 表达式来匹配这两个变体

If `parse` is able to successfully turn the string into a number, it will return an `Ok` value that contains the resultant number. That `Ok` value will match the first arm’s pattern, and the `match` expression will just return the `num` value that `parse` produced and put inside the `Ok` value. That number will end up right where we want it in the new `guess` variable we’re creating.
>  如果 `parse` 成功将字符串转化为数字，它将返回 `Ok` 值，且包含了结果数字，这会匹配到第一个分支，`match` 表达式就会返回 `num`

If `parse` is _not_ able to turn the string into a number, it will return an `Err` value that contains more information about the error. The `Err` value does not match the `Ok(num)` pattern in the first `match` arm, but it does match the `Err(_)` pattern in the second arm. The underscore, `_`, is a catch-all value; in this example, we’re saying we want to match all `Err` values, no matter what information they have inside them. So the program will execute the second arm’s code, `continue`, which tells the program to go to the next iteration of the `loop` and ask for another guess. So, effectively, the program ignores all errors that `parse` might encounter!
>  如果 `parse` 失败，它会返回 `Err` 值，包含了错误信息
>  `Err(_)` 中的 `_` 是一个通配符，表示匹配所有可能的值，即无论 `Err` 中包含什么信息，都会匹配，进而执行 `continue`，跳转到 `loop` 的下一次循环

Now everything in the program should work as expected. Let’s try it:

```shell
$ cargo run
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.13s
     Running `target/debug/guessing_game`
Guess the number!
The secret number is: 61
Please input your guess.
10
You guessed: 10
Too small!
Please input your guess.
99
You guessed: 99
Too big!
Please input your guess.
foo
Please input your guess.
61
You guessed: 61
You win!
```

Awesome! With one tiny final tweak, we will finish the guessing game. Recall that the program is still printing the secret number. That worked well for testing, but it ruins the game. Let’s delete the `println!` that outputs the secret number. Listing 2-6 shows the final code.

Filename: src/main.rs

```rust
use std::cmp::Ordering;
use std::io;

use rand::Rng;

fn main() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..=100);

    loop {
        println!("Please input your guess.");

        let mut guess = String::new();

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };

        println!("You guessed: {guess}");

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

[Listing 2-6](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#listing-2-6): Complete guessing game code

At this point, you’ve successfully built the guessing game. Congratulations!

## Summary
This project was a hands-on way to introduce you to many new Rust concepts: `let`, `match`, functions, the use of external crates, and more. In the next few chapters, you’ll learn about these concepts in more detail. Chapter 3 covers concepts that most programming languages have, such as variables, data types, and functions, and shows how to use them in Rust. Chapter 4 explores ownership, a feature that makes Rust different from other languages. Chapter 5 discusses structs and method syntax, and Chapter 6 explains how enums work.

# 3 Common Programming Concepts
This chapter covers concepts that appear in almost every programming language and how they work in Rust. Many programming languages have much in common at their core. None of the concepts presented in this chapter are unique to Rust, but we’ll discuss them in the context of Rust and explain the conventions around using these concepts.

Specifically, you’ll learn about variables, basic types, functions, comments, and control flow. These foundations will be in every Rust program, and learning them early will give you a strong core to start from.

**Keywords**
The Rust language has a set of _keywords_ that are reserved for use by the language only, much as in other languages. Keep in mind that you cannot use these words as names of variables or functions. Most of the keywords have special meanings, and you’ll be using them to do various tasks in your Rust programs; a few have no current functionality associated with them but have been reserved for functionality that might be added to Rust in the future. You can find a list of the keywords in [Appendix A](https://doc.rust-lang.org/stable/book/appendix-01-keywords.html).

## 3.1 Variables and Mutability
As mentioned in the [“Storing Values with Variables”](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#storing-values-with-variables) section, by default, variables are immutable. This is one of many nudges Rust gives you to write your code in a way that takes advantage of the safety and easy concurrency that Rust offers. However, you still have the option to make your variables mutable. Let’s explore how and why Rust encourages you to favor immutability and why sometimes you might want to opt out.
>  Rust 中的变量默认是不可变的

When a variable is immutable, once a value is bound to a name, you can’t change that value. To illustrate this, generate a new project called _variables_ in your _projects_ directory by using `cargo new variables`.
>  如果变量不可变，当 value 绑定到 name 的时候，我们就不能改变 value

Then, in your new _variables_ directory, open _src/main.rs_ and replace its code with the following code, which won’t compile just yet:
>  如果我们对不可变的变量重新赋值，编译不会通过

Filename: src/main.rs

```rust
fn main() {
    let x = 5;
    println!("The value of x is: {x}");
    x = 6;
    println!("The value of x is: {x}");
}
```

Save and run the program using `cargo run`. You should receive an error message regarding an immutability error, as shown in this output:

```shell
$ cargo run
   Compiling variables v0.1.0 (file:///projects/variables)
error[E0384]: cannot assign twice to immutable variable `x`
 --> src/main.rs:4:5
  |
2 |     let x = 5;
  |         - first assignment to `x`
3 |     println!("The value of x is: {x}");
4 |     x = 6;
  |     ^^^^^ cannot assign twice to immutable variable
  |
help: consider making this binding mutable
  |
2 |     let mut x = 5;
  |         +++

For more information about this error, try `rustc --explain E0384`.
error: could not compile `variables` (bin "variables") due to 1 previous error
```

This example shows how the compiler helps you find errors in your programs. Compiler errors can be frustrating, but really they only mean your program isn’t safely doing what you want it to do yet; they do _not_ mean that you’re not a good programmer! Experienced Rustaceans still get compiler errors.

You received the error message `` cannot assign twice to immutable variable `x` `` because you tried to assign a second value to the immutable `x` variable.

It’s important that we get compile-time errors when we attempt to change a value that’s designated as immutable because this very situation can lead to bugs. If one part of our code operates on the assumption that a value will never change and another part of our code changes that value, it’s possible that the first part of the code won’t do what it was designed to do. The cause of this kind of bug can be difficult to track down after the fact, especially when the second piece of code changes the value only _sometimes_. The Rust compiler guarantees that when you state that a value won’t change, it really won’t change, so you don’t have to keep track of it yourself. Your code is thus easier to reason through.
>  Rust 编译器不允许这样的情况出现，是为了保证程序符合设计语义，如果代码的一部分假设一个值永远不会改变，但另一部分修改了这个值，那么第一部分代码就可能无法按照设计执行

But mutability can be very useful, and can make code more convenient to write. Although variables are immutable by default, you can make them mutable by adding `mut` in front of the variable name as you did in [Chapter 2](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#storing-values-with-variables). Adding `mut` also conveys intent to future readers of the code by indicating that other parts of the code will be changing this variable’s value.
>  添加 `mut` 将变量声明为可变

For example, let’s change _src/main.rs_ to the following:

Filename: src/main.rs

```rust
fn main() {
    let mut x = 5;
    println!("The value of x is: {x}");
    x = 6;
    println!("The value of x is: {x}");
}
```

When we run the program now, we get this:

```shell
$ cargo run
   Compiling variables v0.1.0 (file:///projects/variables)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.30s
     Running `target/debug/variables`
The value of x is: 5
The value of x is: 6
```

We’re allowed to change the value bound to `x` from `5` to `6` when `mut` is used. Ultimately, deciding whether to use mutability or not is up to you and depends on what you think is clearest in that particular situation.

### Constant
Like immutable variables, _constants_ are values that are bound to a name and are not allowed to change, but there are a few differences between constants and variables.
>  和不可变变量类似，常量指绑定到一个名字且不允许变化的值
>  常量和变量之间存在一些差异

First, you aren’t allowed to use `mut` with constants. Constants aren’t just immutable by default—they’re always immutable. You declare constants using the `const` keyword instead of the `let` keyword, and the type of the value _must_ be annotated. We’ll cover types and type annotations in the next section, [“Data Types”](https://doc.rust-lang.org/stable/book/ch03-02-data-types.html#data-types), so don’t worry about the details right now. Just know that you must always annotate the type.
>  首先，不允许对常量使用 `mut` —— 常量不是默认不可变，而是总是不可变
>  另外，我们通过 `const` 关键字声明常量，而不是 `let` 关键字，并且我们必须为常量提供类型标注

Constants can be declared in any scope, including the global scope, which makes them useful for values that many parts of code need to know about.
>  其次，常量可以在任意作用域声明，包括全局作用域

The last difference is that constants may be set only to a constant expression, not the result of a value that could only be computed at runtime.
>  最后，常量只可以被设定为一个常量表达式，而不是在运行时计算的一个结果值

Here’s an example of a constant declaration:

```rust
const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
```

The constant’s name is `THREE_HOURS_IN_SECONDS` and its value is set to the result of multiplying 60 (the number of seconds in a minute) by 60 (the number of minutes in an hour) by 3 (the number of hours we want to count in this program). Rust’s naming convention for constants is to use all uppercase with underscores between words. The compiler is able to evaluate a limited set of operations at compile time, which lets us choose to write out this value in a way that’s easier to understand and verify, rather than setting this constant to the value 10,800. See the [Rust Reference’s section on constant evaluation](https://doc.rust-lang.org/stable/reference/const_eval.html) for more information on what operations can be used when declaring constants.
>  Rust 对常量的命名风格是使用全大写，单词之间使用下划线
>  Rust 的编译器可以在编译时计算一组有限的操作，这使我们能够以更易于理解和验证的方式写出这个值，而不是直接将该常量设为数值 10800

Constants are valid for the entire time a program runs, within the scope in which they were declared. This property makes constants useful for values in your application domain that multiple parts of the program might need to know about, such as the maximum number of points any player of a game is allowed to earn, or the speed of light.
>  常量在程序运行的整个期间都有效，在声明它们的作用域内可用

Naming hardcoded values used throughout your program as constants is useful in conveying the meaning of that value to future maintainers of the code. It also helps to have only one place in your code you would need to change if the hardcoded value needed to be updated in the future.
>  尽量将程序中广泛使用的硬编码值命名为常量

### Shadowing
As you saw in the guessing game tutorial in [Chapter 2](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#comparing-the-guess-to-the-secret-number), you can declare a new variable with the same name as a previous variable. Rustaceans say that the first variable is _shadowed_ by the second, which means that the second variable is what the compiler will see when you use the name of the variable. In effect, the second variable overshadows the first, taking any uses of the variable name to itself until either it itself is shadowed or the scope ends. We can shadow a variable by using the same variable’s name and repeating the use of the `let` keyword as follows:
>  我们可以用和之前变量相同的名字声明一个新的变量，我们称第一个变量被第二个变量 “遮蔽” 了
>  这意味着当我们使用这个变量名时，编译器看到的是第二个变量，效果上，第二个变量会覆盖第一个变量，直到第二个变量自身被遮蔽或其作用域结束之前，所有对变量名的使用都会指向它
>  我们可以通过使用相同的变量名并重复使用 `let` 关键字来遮蔽一个变量，如下所示:

Filename: src/main.rs

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

This program first binds `x` to a value of `5`. Then it creates a new variable `x` by repeating `let x =`, taking the original value and adding `1` so the value of `x` is then `6`. Then, within an inner scope created with the curly brackets, the third `let` statement also shadows `x` and creates a new variable, multiplying the previous value by `2` to give `x` a value of `12`. When that scope is over, the inner shadowing ends and `x` returns to being `6`. When we run this program, it will output the following:

```shell
$ cargo run
   Compiling variables v0.1.0 (file:///projects/variables)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.31s
     Running `target/debug/variables`
The value of x in the inner scope is: 12
The value of x is: 6
```

Shadowing is different from marking a variable as `mut` because we’ll get a compile-time error if we accidentally try to reassign to this variable without using the `let` keyword. By using `let`, we can perform a few transformations on a value but have the variable be immutable after those transformations have been completed.
>  遮蔽与将变量标记为 `mut` 不同，因为如果我们不小心在没有使用 `let` 关键字的情况下尝试重新赋值给这个变量，编译器会报错
>  通过使用 `let`，我们可以在不改变变量本身的情况下对一个值进行一些转换，并且在这些转换完成后，变量仍然是不可变的

The other difference between `mut` and shadowing is that because we’re effectively creating a new variable when we use the `let` keyword again, we can change the type of the value but reuse the same name. For example, say our program asks a user to show how many spaces they want between some text by inputting space characters, and then we want to store that input as a number:
>  遮蔽和 `mut` 的另一个区别是: 当我们再次使用 `let` 关键字时，我们实际上是在创建一个新的变量，因此我们可以改变值的类型，同时重复使用相同的变量名

```rust
    let spaces = "   ";
    let spaces = spaces.len();
```

The first `spaces` variable is a string type and the second `spaces` variable is a number type. Shadowing thus spares us from having to come up with different names, such as `spaces_str` and `spaces_num`; instead, we can reuse the simpler `spaces` name. However, if we try to use `mut` for this, as shown here, we’ll get a compile-time error:

```rust
    let mut spaces = "   ";
    spaces = spaces.len();
```

The error says we’re not allowed to mutate a variable’s type:

```shell
$ cargo run
   Compiling variables v0.1.0 (file:///projects/variables)
error[E0308]: mismatched types
 --> src/main.rs:3:14
  |
2 |     let mut spaces = "   ";
  |                      ----- expected due to this value
3 |     spaces = spaces.len();
  |              ^^^^^^^^^^^^ expected `&str`, found `usize`

For more information about this error, try `rustc --explain E0308`.
error: could not compile `variables` (bin "variables") due to 1 previous error
```

Now that we’ve explored how variables work, let’s look at more data types they can have.

## 3.2 Data Types
Every value in Rust is of a certain _data type_, which tells Rust what kind of data is being specified so it knows how to work with that data. We’ll look at two data type subsets: scalar and compound.
>  Rust 中的每个值都有特定的数据类型
>  我们将介绍两个数据类型子集: 标量类型和复合类型

Keep in mind that Rust is a _statically typed_ language, which means that it must know the types of all variables at compile time. The compiler can usually infer what type we want to use based on the value and how we use it. In cases when many types are possible, such as when we converted a `String` to a numeric type using `parse` in the [“Comparing the Guess to the Secret Number”](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#comparing-the-guess-to-the-secret-number) section in Chapter 2, we must add a type annotation, like this:
>  Rust 是静态类型语言，这意味着 Rust 必须在编译时知道所有变量的类型
>  通常，编译器可以根据值以及我们如何使用该值来推断值的类型
>  但在某些情况下，可能有多种可能的类型可以选择
>  例如在 CH2 中，我们使用 `parse` 将 `String` 转换为数值类型时，我们必须添加类型注解，如下所示:

```rust
let guess: u32 = "42".parse().expect("Not a number!");
```

If we don’t add the `: u32` type annotation shown in the preceding code, Rust will display the following error, which means the compiler needs more information from us to know which type we want to use:
>  如果没有足够的类型注解，导致编译器无法推断一些数据类型时，编译器会报错

```shell
$ cargo build
   Compiling no_type_annotations v0.1.0 (file:///projects/no_type_annotations)
error[E0284]: type annotations needed
 --> src/main.rs:2:9
  |
2 |     let guess = "42".parse().expect("Not a number!");
  |         ^^^^^        ----- type must be known at this point
  |
  = note: cannot satisfy `<_ as FromStr>::Err == _`
help: consider giving `guess` an explicit type
  |
2 |     let guess: /* Type */ = "42".parse().expect("Not a number!");
  |              ++++++++++++

For more information about this error, try `rustc --explain E0284`.
error: could not compile `no_type_annotations` (bin "no_type_annotations") due to 1 previous error
```

You’ll see different type annotations for other data types.

### Scalar Types
A _scalar_ type represents a single value. Rust has four primary scalar types: integers, floating-point numbers, Booleans, and characters. You may recognize these from other programming languages. Let’s jump into how they work in Rust.
>  标量类型表示单个值，Rust 有四个主要的标量类型: 整数、浮点数、布尔值、字符

#### Integer Types
An _integer_ is a number without a fractional component. We used one integer type in Chapter 2, the `u32` type. This type declaration indicates that the value it’s associated with should be an unsigned integer (signed integer types start with `i` instead of `u`) that takes up 32 bits of space. Table 3-1 shows the built-in integer types in Rust. We can use any of these variants to declare the type of an integer value.
>  整数即没有小数部分的数字，我们在 chapter 2 使用了一个整数类型 `u32`，它表示关联的值是一个无符号整数 (有符号整数应该以 `i` 开头而不是 `u`)，并且占据 32 bit 的空间
>  Rust 的内建整数类型如下所示

Table 3-1: Integer Types in Rust

|Length|Signed|Unsigned|
|---|---|---|
|8-bit|`i8`|`u8`|
|16-bit|`i16`|`u16`|
|32-bit|`i32`|`u32`|
|64-bit|`i64`|`u64`|
|128-bit|`i128`|`u128`|
|arch|`isize`|`usize`|

Each variant can be either signed or unsigned and has an explicit size. _Signed_ and _unsigned_ refer to whether it’s possible for the number to be negative—in other words, whether the number needs to have a sign with it (signed) or whether it will only ever be positive and can therefore be represented without a sign (unsigned). It’s like writing numbers on paper: when the sign matters, a number is shown with a plus sign or a minus sign; however, when it’s safe to assume the number is positive, it’s shown with no sign. Signed numbers are stored using [two’s complement](https://en.wikipedia.org/wiki/Two%27s_complement) representation.
>  每种变体都可以是有符号的或无符号的，且具有明确的大小
>  有符号数使用补码表示法进行存储

Each signed variant can store numbers from $−(2^{n − 1})$ to $2^{n − 1} − 1$ inclusive, where _n_ is the number of bits that variant uses. So an `i8` can store numbers from $−(2^7)$ to $2^7 − 1$, which equals $−128$ to $127$. Unsigned variants can store numbers from $0$ to $2^{n − 1}$, so a `u8` can store numbers from $0$ to $2^8 − 1$, which equals 0 to 255.

Additionally, the `isize` and `usize` types depend on the architecture of the computer your program is running on, which is denoted in the table as “arch”: 64 bits if you’re on a 64-bit architecture and 32 bits if you’re on a 32-bit architecture.
>  `isize, usize` 类型依赖于程序所运行的计算机架构，如果是 64 位架构，则为 64 位，如果是 32 位架构，则为 32 位

You can write integer literals in any of the forms shown in Table 3-2. Note that number literals that can be multiple numeric types allow a type suffix, such as `57u8`, to designate the type. Number literals can also use `_` as a visual separator to make the number easier to read, such as `1_000`, which will have the same value as if you had specified `1000`.
>  我们可以使用 Table 3-2 中的任意形式表示整数字面量
>  注意，可以属于多种数据类型的数字字面量可以使用类型后缀，例如 `57u8`，以指定其类型
>  数字字面量可以使用 `_` 作为视觉分隔符，是其更加易读，例如 `1_000`，它的值和写成 `1000` 是相同的

Table 3-2: Integer Literals in Rust

|Number literals|Example|
|---|---|
|Decimal|`98_222`|
|Hex|`0xff`|
|Octal|`0o77`|
|Binary|`0b1111_0000`|
|Byte (`u8` only)|`b'A'`|

So how do you know which type of integer to use? If you’re unsure, Rust’s defaults are generally good places to start: integer types default to `i32`. The primary situation in which you’d use `isize` or `usize` is when indexing some sort of collection.
>  Rust 的默认整数类型是 `i32`，一般我们在对某种集合进行索引时才会使用 `isize` 或 `usize`

**Integer Overflow**
Let’s say you have a variable of type `u8` that can hold values between 0 and 255. If you try to change the variable to a value outside that range, such as 256, _integer overflow_ will occur, which can result in one of two behaviors. When you’re compiling in debug mode, Rust includes checks for integer overflow that cause your program to _panic_ at runtime if this behavior occurs. Rust uses the term _panicking_ when a program exits with an error; we’ll discuss panics in more depth in the [“Unrecoverable Errors with `panic!`”](https://doc.rust-lang.org/stable/book/ch09-01-unrecoverable-errors-with-panic.html) section in Chapter 9.
>  Rust 的 debug mode 编译包含了对可以在程序运行时引发 panic 的整数溢出的检查

When you’re compiling in release mode with the `--release` flag, Rust does _not_ include checks for integer overflow that cause panics. Instead, if overflow occurs, Rust performs _two’s complement wrapping_. In short, values greater than the maximum value the type can hold “wrap around” to the minimum of the values the type can hold. In the case of a `u8`, the value 256 becomes 0, the value 257 becomes 1, and so on. The program won’t panic, but the variable will have a value that probably isn’t what you were expecting it to have. Relying on integer overflow’s wrapping behavior is considered an error.
>  在 release mode 编译下，Rust 不会对导致 panic 的整数溢出进行检查
>  如果发生溢出，Rust 会执行补码环绕，简而言之，超过该类型的最大值的值会被 “环绕” 到该类型可以存储的最小值
>  例如，对于 `u8` 类型来说，值 256 会变成 0，值 257 会变成 1，依次类推
>  程序不会 panic，但变量的值可能与你预期的完全不同
>  注意编程时不要依赖整数溢出的环绕行为

To explicitly handle the possibility of overflow, you can use these families of methods provided by the standard library for primitive numeric types:

- Wrap in all modes with the `wrapping_*` methods, such as `wrapping_add`.
- Return the `None` value if there is overflow with the `checked_*` methods.
- Return the value and a Boolean indicating whether there was overflow with the `overflowing_*` methods.
- Saturate at the value’s minimum or maximum values with the `saturating_*` methods.

>  要显式处理溢出的可能性，可以使用标准库为原始数值类型提供的以下方法族:
>  - 使用 `wrapping_*` 方法 (例如 `warpping_add`) 在所有模式下进行环绕处理
>  - 使用 `checked_*` 方法在发生溢出时返回 `None` 值
>  - 使用 `overflowing_*` 方法返回一个值以及布尔值，表示是否发生了溢出
>  - 使用 `saturating_*` 方法在达到该类型的最小值或最大值时进行饱和处理

#### Floating-Point Types
Rust also has two primitive types for _floating-point numbers_, which are numbers with decimal points. Rust’s floating-point types are `f32` and `f64`, which are 32 bits and 64 bits in size, respectively. The default type is `f64` because on modern CPUs, it’s roughly the same speed as `f32` but is capable of more precision. All floating-point types are signed.
>  Rust 还有两种用于浮点数的原始类型 (浮点数即带有小数点的数字): `f32, f64`
>  默认的类型是 `f64`，因为在现代 CPU 上，它的速度与 `f32` 相当，但能提高更高的精度
>  所有的浮点类型都是有符号的

Here’s an example that shows floating-point numbers in action:

Filename: src/main.rs

```rust
fn main() {
    let x = 2.0; // f64

    let y: f32 = 3.0; // f32
}
```

Floating-point numbers are represented according to the IEEE-754 standard.
>  浮点数使用 IEEE-754 标准表示

#### Numeric Operations
Rust supports the basic mathematical operations you’d expect for all the number types: addition, subtraction, multiplication, division, and remainder. Integer division truncates toward zero to the nearest integer. 
>  Rust 支持所有数字类型都具备的基本数学运算: 加法、减法、乘法、除法和取余
>  整数除法会向零方向阶段，得到最近的整数

The following code shows how you’d use each numeric operation in a `let` statement:

Filename: src/main.rs

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
    let truncated = -5 / 3; // Results in -1

    // remainder
    let remainder = 43 % 5;
}
```

Each expression in these statements uses a mathematical operator and evaluates to a single value, which is then bound to a variable. [Appendix B](https://doc.rust-lang.org/stable/book/appendix-02-operators.html) contains a list of all operators that Rust provides.

#### The Boolean Type
As in most other programming languages, a Boolean type in Rust has two possible values: `true` and `false`. Booleans are one byte in size. The Boolean type in Rust is specified using `bool`. For example:
>  Rust 中，布尔类型有两个可能值: `true, false`
>  布尔类型的大小为单字节
>  布尔类型通过 `bool` 注解

Filename: src/main.rs

```rust
fn main() {
    let t = true;

    let f: bool = false; // with explicit type annotation
}
```

The main way to use Boolean values is through conditionals, such as an `if` expression. We’ll cover how `if` expressions work in Rust in the [“Control Flow”](https://doc.rust-lang.org/stable/book/ch03-05-control-flow.html#control-flow) section.

#### The Character Type
Rust’s `char` type is the language’s most primitive alphabetic type. Here are some examples of declaring `char` values:
>  Rust 的 `char` 类型是语言中最基本的字母类型

Filename: src/main.rs

```rust
fn main() {
    let c = 'z';
    let z: char = 'ℤ'; // with explicit type annotation
    let heart_eyed_cat = '😻';
}
```

Note that we specify `char` literals with single quotes, as opposed to string literals, which use double quotes. Rust’s `char` type is four bytes in size and represents a Unicode Scalar Value, which means it can represent a lot more than just ASCII. Accented letters; Chinese, Japanese, and Korean characters; emoji; and zero-width spaces are all valid `char` values in Rust. 
>  我们通过单引号来表示 `char` 字面量，而字符串字面量则使用双引号
>  `char` 类型的大小是四字节，表示一个 Unicode 标量值，这意味着它不仅可以表示 ASCII 字符，也可以表示更多字符
>  带变音符号的字母、中文、日文、韩文、emoji、零宽度空格在 Rust 中都是合法的 `char` 值

Unicode Scalar Values range from `U+0000` to `U+D7FF` and `U+E000` to `U+10FFFF` inclusive. However, a “character” isn’t really a concept in Unicode, so your human intuition for what a “character” is may not match up with what a `char` is in Rust. We’ll discuss this topic in detail in [“Storing UTF-8 Encoded Text with Strings”](https://doc.rust-lang.org/stable/book/ch08-02-strings.html#storing-utf-8-encoded-text-with-strings) in Chapter 8.
>  Unicode 标量值的范围是从 `U+0000` 到 `U+D7FF`，以及 `U+E000` 到 `U+10FFFF`
>  但是 “字符” 并不是 Unicode 中的一个明确概念，因此人类对 “字符” 的直觉可能与 Rust 中 `char` 的定义不太一致

### Compound Types
_Compound types_ can group multiple values into one type. Rust has two primitive compound types: tuples and arrays.
>  复合类型将多个值聚集到一个类型中，Rust 有两个原始复合类型: tuple, array

#### The Tuple Type
A _tuple_ is a general way of grouping together a number of values with a variety of types into one compound type. Tuples have a fixed length: once declared, they cannot grow or shrink in size.
>  tuple 可以将多种类型的值组合到一个复合类型中
>  tuple 的长度固定，一旦声明，就不能变化

We create a tuple by writing a comma-separated list of values inside parentheses. Each position in the tuple has a type, and the types of the different values in the tuple don’t have to be the same. We’ve added optional type annotations in this example:
>  我们通过在括号内编写用逗号分隔的值列表来创建 tuple
>  tuple 中的每个位置都有一个类型
>  在这个例子中，我们添加了可选的类型注释:

Filename: src/main.rs

```rust
fn main() {
    let tup: (i32, f64, u8) = (500, 6.4, 1);
}
```

The variable `tup` binds to the entire tuple because a tuple is considered a single compound element. To get the individual values out of a tuple, we can use pattern matching to destructure a tuple value, like this:
>  上例中，变量 `tup` 绑定了整个 tuple，因为一个 tuple 被视作单个复合元素
>  为了得到 tuple 中的单个值，我们可以使用模式匹配来结构元组值，例如:

Filename: src/main.rs

```rust
fn main() {
    let tup = (500, 6.4, 1);

    let (x, y, z) = tup;

    println!("The value of y is: {y}");
}
```

This program first creates a tuple and binds it to the variable `tup`. It then uses a pattern with `let` to take `tup` and turn it into three separate variables, `x`, `y`, and `z`. This is called _destructuring_ because it breaks the single tuple into three parts. Finally, the program prints the value of `y`, which is `6.4`.
>  这个程序首先创建一个 tuple 并将它绑定到变量 `tup`
>  然后，使用 `let` 进行模式匹配，将 `tup` 拆分为三个独立的变量，这称为 destructuring，因为它将单个 tuple 拆分为了三个部分

We can also access a tuple element directly by using a period (`.`) followed by the index of the value we want to access. For example:
>  我们也可以通过 `.<index>` 来访问 tuple 元素，如下所示:

Filename: src/main.rs

```rust
fn main() {
    let x: (i32, f64, u8) = (500, 6.4, 1);

    let five_hundred = x.0;

    let six_point_four = x.1;

    let one = x.2;
}
```

This program creates the tuple `x` and then accesses each element of the tuple using their respective indices. As with most programming languages, the first index in a tuple is 0.

The tuple without any values has a special name, _unit_. This value and its corresponding type are both written `()` and represent an empty value or an empty return type. Expressions implicitly return the unit value if they don’t return any other value.
>  没有任何值的 tuple 称为 unit，这个值及其对应的类型都写为 `()`，表示空值或空返回类型
>  Rust 中，如果表达式不返回其他任何值，它们就会隐式地返回 unit 值

#### The Array Type
Another way to have a collection of multiple values is with an _array_. Unlike a tuple, every element of an array must have the same type. Unlike arrays in some other languages, arrays in Rust have a fixed length.
>  另一种集合一组值的方式是 array
>  和 tuple 不同，array 中的所有元素的类型都相同
>  Rust 中，array 也是固定长度的

We write the values in an array as a comma-separated list inside square brackets:
>  array 通过方括号中逗号分隔的值创建，如下所示:

Filename: src/main.rs

```rust
fn main() {
    let a = [1, 2, 3, 4, 5];
}
```

Arrays are useful when you want your data allocated on the stack, the same as the other types we have seen so far, rather than the heap (we will discuss the stack and the heap more in [Chapter 4](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html#the-stack-and-the-heap)) or when you want to ensure you always have a fixed number of elements. An array isn’t as flexible as the vector type, though. A _vector_ is a similar collection type provided by the standard library that _is_ allowed to grow or shrink in size. If you’re unsure whether to use an array or a vector, chances are you should use a vector. [Chapter 8](https://doc.rust-lang.org/stable/book/ch08-01-vectors.html) discusses vectors in more detail.
>  和 array 类似的类型是 vector 类型，vector 是由标准库提供的集合类型，vector 类型允许大小增大或减小
>  如果我们不确定使用 array 或 vector，那么很可能应该使用 vector

However, arrays are more useful when you know the number of elements will not need to change. For example, if you were using the names of the month in a program, you would probably use an array rather than a vector because you know it will always contain 12 elements:

```rust
let months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]; 
```

You write an array’s type using square brackets with the type of each element, a semicolon, and then the number of elements in the array, like so:
>  可以使用 `[dtype, ele_num]` 作为数组的类型注释，例如:

```rust
let a: [i32; 5] = [1, 2, 3, 4, 5];
```

Here, `i32` is the type of each element. After the semicolon, the number `5` indicates the array contains five elements.

You can also initialize an array to contain the same value for each element by specifying the initial value, followed by a semicolon, and then the length of the array in square brackets, as shown here:

```rust
let a = [3; 5]; 
```

The array named `a` will contain `5` elements that will all be set to the value `3` initially. This is the same as writing `let a = [3, 3, 3, 3, 3];` but in a more concise way.

>  像 `let a = [3; 5];` 这样的写法将初始化一个长度为 `5`，所有元素都是 `3` 的数组

##### Accessing Array Elements
An array is a single chunk of memory of a known, fixed size that can be allocated on the stack. You can access elements of an array using indexing, like this:
>  array 是分配在栈上的一块已知且固定大小的内存，可以使用索引访问 array 元素:

Filename: src/main.rs

```rust
fn main() {
    let a = [1, 2, 3, 4, 5];

    let first = a[0];
    let second = a[1];
}
```

In this example, the variable named `first` will get the value `1` because that is the value at index `[0]` in the array. The variable named `second` will get the value `2` from index `[1]` in the array.

##### Invalid Array Element Access
Let’s see what happens if you try to access an element of an array that is past the end of the array. Say you run this code, similar to the guessing game in Chapter 2, to get an array index from the user:

Filename: src/main.rs

```rust
use std::io;

fn main() {
    let a = [1, 2, 3, 4, 5];

    println!("Please enter an array index.");

    let mut index = String::new();

    io::stdin()
        .read_line(&mut index)
        .expect("Failed to read line");

    let index: usize = index
        .trim()
        .parse()
        .expect("Index entered was not a number");

    let element = a[index];

    println!("The value of the element at index {index} is: {element}");
}
```

This code compiles successfully. If you run this code using `cargo run` and enter `0`, `1`, `2`, `3`, or `4`, the program will print out the corresponding value at that index in the array. If you instead enter a number past the end of the array, such as `10`, you’ll see output like this:

```
thread 'main' panicked at src/main.rs:19:19:
index out of bounds: the len is 5 but the index is 10
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

The program resulted in a _runtime_ error at the point of using an invalid value in the indexing operation. The program exited with an error message and didn’t execute the final `println!` statement. When you attempt to access an element using indexing, Rust will check that the index you’ve specified is less than the array length. If the index is greater than or equal to the length, Rust will panic. This check has to happen at runtime, especially in this case, because the compiler can’t possibly know what value a user will enter when they run the code later.
>  如果 array 访问越界，将导致一个运行时错误，程序会立刻退出
>  当我们用索引访问元素时，Rust 会检查指定的索引是否小于数组长度，如果索引大于等于数组长度，Rust 会 panic
>  这种检查必须在运行时执行，因为编译时无法确定具体的索引数值

This is an example of Rust’s memory safety principles in action. In many low-level languages, this kind of check is not done, and when you provide an incorrect index, invalid memory can be accessed. Rust protects you against this kind of error by immediately exiting instead of allowing the memory access and continuing. Chapter 9 discusses more of Rust’s error handling and how you can write readable, safe code that neither panics nor allows invalid memory access.
>  这是 Rust 的内存安全原则实际应用的例子，在许多低级语言中，这个检查通常不会运行，故我们可能会访问到无效的内存
>  Rust 通过立即退出程序来保护我们免受此类错误的影响，而不是允许内存访问并继续执行

## 3.3 Functions
Functions are prevalent in Rust code. You’ve already seen one of the most important functions in the language: the `main` function, which is the entry point of many programs. You’ve also seen the `fn` keyword, which allows you to declare new functions.

Rust code uses _snake case_ as the conventional style for function and variable names, in which all letters are lowercase and underscores separate words. Here’s a program that contains an example function definition:
>  Rust 使用 snake case 用于函数和变量名

Filename: src/main.rs

```rust
fn main() {
    println!("Hello, world!");

    another_function();
}

fn another_function() {
    println!("Another function.");
}
```

We define a function in Rust by entering `fn` followed by a function name and a set of parentheses. The curly brackets tell the compiler where the function body begins and ends.

We can call any function we’ve defined by entering its name followed by a set of parentheses. Because `another_function` is defined in the program, it can be called from inside the `main` function. Note that we defined `another_function` _after_ the `main` function in the source code; we could have defined it before as well. Rust doesn’t care where you define your functions, only that they’re defined somewhere in a scope that can be seen by the caller.
>  Rust 对于函数定义的位置没有要求，只要定义在调用者可以看到的作用域内接口

Let’s start a new binary project named _functions_ to explore functions further. Place the `another_function` example in _src/main.rs_ and run it. You should see the following output:

```
$ cargo run
   Compiling functions v0.1.0 (file:///projects/functions)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.28s
     Running `target/debug/functions`
Hello, world!
Another function.
```

The lines execute in the order in which they appear in the `main` function. First the “Hello, world!” message prints, and then `another_function` is called and its message is printed.

### Parameters
We can define functions to have _parameters_, which are special variables that are part of a function’s signature. When a function has parameters, you can provide it with concrete values for those parameters. Technically, the concrete values are called _arguments_, but in casual conversation, people tend to use the words _parameter_ and _argument_ interchangeably for either the variables in a function’s definition or the concrete values passed in when you call a function.
>  parameter 是作为函数签名一部分的特殊变量
>  我们为 parameter 提供具体的值，具体的值称为 arguments

In this version of `another_function` we add a parameter:

Filename: src/main.rs

```rust
fn main() {
    another_function(5);
}

fn another_function(x: i32) {
    println!("The value of x is: {x}");
}
```

Try running this program; you should get the following output:

```
$ cargo run
   Compiling functions v0.1.0 (file:///projects/functions)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.21s
     Running `target/debug/functions`
The value of x is: 5
```

The declaration of `another_function` has one parameter named `x`. The type of `x` is specified as `i32`. When we pass `5` in to `another_function`, the `println!` macro puts `5` where the pair of curly brackets containing `x` was in the format string.

In function signatures, you _must_ declare the type of each parameter. This is a deliberate decision in Rust’s design: requiring type annotations in function definitions means the compiler almost never needs you to use them elsewhere in the code to figure out what type you mean. The compiler is also able to give more helpful error messages if it knows what types the function expects.
>  函数签名中，形参必须声明类型，即要有类型注解

When defining multiple parameters, separate the parameter declarations with commas, like this:

Filename: src/main.rs

```rust
fn main() {
    print_labeled_measurement(5, 'h');
}

fn print_labeled_measurement(value: i32, unit_label: char) {
    println!("The measurement is: {value}{unit_label}");
}
```

This example creates a function named `print_labeled_measurement` with two parameters. The first parameter is named `value` and is an `i32`. The second is named `unit_label` and is type `char`. The function then prints text containing both the `value` and the `unit_label`.

Let’s try running this code. Replace the program currently in your _functions_ project’s _src/main.rs_ file with the preceding example and run it using `cargo run`:

```
$ cargo run
   Compiling functions v0.1.0 (file:///projects/functions)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.31s
     Running `target/debug/functions`
The measurement is: 5h
```

Because we called the function with `5` as the value for `value` and `'h'` as the value for `unit_label`, the program output contains those values.

### Statements and Expressions
Function bodies are made up of a series of statements optionally ending in an expression. So far, the functions we’ve covered haven’t included an ending expression, but you have seen an expression as part of a statement. Because Rust is an expression-based language, this is an important distinction to understand. Other languages don’t have the same distinctions, so let’s look at what statements and expressions are and how their differences affect the bodies of functions.
>  函数体包含一系列语句，函数体可以以一个表达式为结束
>  Rust 是一个基于表达式的语言

- **Statements** are instructions that perform some action and do not return a value.
- **Expressions** evaluate to a resultant value. Let’s look at some examples.

>  语句: 执行动作，不返回值
>  表达式: 评估值，并返回值

We’ve actually already used statements and expressions. Creating a variable and assigning a value to it with the `let` keyword is a statement. In Listing 3-1, `let y = 6;` is a statement.
>  使用 `let` 创建变量为语句

Filename: src/main.rs

```rust
fn main() {
    let y = 6;
}
```

[Listing 3-1](https://doc.rust-lang.org/stable/book/ch03-03-how-functions-work.html#listing-3-1): A `main` function declaration containing one statement

Function definitions are also statements; the entire preceding example is a statement in itself. (As we will see below, _calling_ a function is not a statement.)
>  函数定义也是语句

Statements do not return values. Therefore, you can’t assign a `let` statement to another variable, as the following code tries to do; you’ll get an error:

Filename: src/main.rs

```rust
fn main() {
    let x = (let y = 6);
}
```

When you run this program, the error you’ll get looks like this:

```
$ cargo run
   Compiling functions v0.1.0 (file:///projects/functions)
error: expected expression, found `let` statement
 --> src/main.rs:2:14
  |
2 |     let x = (let y = 6);
  |              ^^^
  |
  = note: only supported directly in conditions of `if` and `while` expressions

warning: unnecessary parentheses around assigned value
 --> src/main.rs:2:13
  |
2 |     let x = (let y = 6);
  |             ^         ^
  |
  = note: `#[warn(unused_parens)]` on by default
help: remove these parentheses
  |
2 -     let x = (let y = 6);
2 +     let x = let y = 6;
  |

warning: `functions` (bin "functions") generated 1 warning
error: could not compile `functions` (bin "functions") due to 1 previous error; 1 warning emitted
```

The `let y = 6` statement does not return a value, so there isn’t anything for `x` to bind to. This is different from what happens in other languages, such as C and Ruby, where the assignment returns the value of the assignment. In those languages, you can write `x = y = 6` and have both `x` and `y` have the value `6`; that is not the case in Rust.

Expressions evaluate to a value and make up most of the rest of the code that you’ll write in Rust. Consider a math operation, such as `5 + 6`, which is an expression that evaluates to the value `11`. Expressions can be part of statements: in Listing 3-1, the `6` in the statement `let y = 6;` is an expression that evaluates to the value `6`. Calling a function is an expression. Calling a macro is an expression. A new scope block created with curly brackets is an expression, for example:
>  表达式可以是语句的一部分
>  函数调用是表达式
>  宏调用是表达式
>  创建一个新的作用域是表达式

Filename: src/main.rs

```rust
fn main() {
    let y = {
        let x = 3;
        x + 1
    };

    println!("The value of y is: {y}");
}
```

This expression:

```rust
{
    let x = 3;
    x + 1
}
```

is a block that, in this case, evaluates to `4`. That value gets bound to `y` as part of the `let` statement. Note that the `x + 1` line doesn’t have a semicolon at the end, which is unlike most of the lines you’ve seen so far. Expressions do not include ending semicolons. If you add a semicolon to the end of an expression, you turn it into a statement, and it will then not return a value. Keep this in mind as you explore function return values and expressions next.
>  `{}` 的结尾表达式没有 `;`，因为表达式本身不需要以 `;` 结尾，只有语句需要以 `;` 结尾

### Functions with Return Values
Functions can return values to the code that calls them. We don’t name return values, but we must declare their type after an arrow (`->`). In Rust, the return value of the function is synonymous with the value of the final expression in the block of the body of a function. You can return early from a function by using the `return` keyword and specifying a value, but most functions return the last expression implicitly. Here’s an example of a function that returns a value:
>  如果函数有返回值，需要通过 `->` 声明返回值类型
>  Rust 中，函数的返回值等于函数体块中最后一个表达式的值，也可以通过 `return` 关键字返回特定的值，但大多数函数隐式返回最后一个表达式的值

Filename: src/main.rs

```rust
fn five() -> i32 {
    5
}

fn main() {
    let x = five();

    println!("The value of x is: {x}");
}
```

There are no function calls, macros, or even `let` statements in the `five` function—just the number `5` by itself. That’s a perfectly valid function in Rust. Note that the function’s return type is specified too, as `-> i32`. Try running this code; the output should look like this:

```
$ cargo run
   Compiling functions v0.1.0 (file:///projects/functions)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.30s
     Running `target/debug/functions`
The value of x is: 5
```

The `5` in `five` is the function’s return value, which is why the return type is `i32`. Let’s examine this in more detail. There are two important bits: first, the line `let x = five();` shows that we’re using the return value of a function to initialize a variable. Because the function `five` returns a `5`, that line is the same as the following:

```rust
let x = 5;
```

Second, the `five` function has no parameters and defines the type of the return value, but the body of the function is a lonely `5` with no semicolon because it’s an expression whose value we want to return.

Let’s look at another example:

Filename: src/main.rs

```rust
fn main() {
    let x = plus_one(5);

    println!("The value of x is: {x}");
}

fn plus_one(x: i32) -> i32 {
    x + 1
}
```

Running this code will print `The value of x is: 6`. But if we place a semicolon at the end of the line containing `x + 1`, changing it from an expression to a statement, we’ll get an error:

Compiling this code produces an error, as follows:

```
$ cargo run
   Compiling functions v0.1.0 (file:///projects/functions)
error[E0308]: mismatched types
 --> src/main.rs:7:24
  |
7 | fn plus_one(x: i32) -> i32 {
  |    --------            ^^^ expected `i32`, found `()`
  |    |
  |    implicitly returns `()` as its body has no tail or `return` expression
8 |     x + 1;
  |          - help: remove this semicolon to return this value

For more information about this error, try `rustc --explain E0308`.
error: could not compile `functions` (bin "functions") due to 1 previous error
```

The main error message, `mismatched types`, reveals the core issue with this code. The definition of the function `plus_one` says that it will return an `i32`, but statements don’t evaluate to a value, which is expressed by `()`, the unit type. Therefore, nothing is returned, which contradicts the function definition and results in an error. In this output, Rust provides a message to possibly help rectify this issue: it suggests removing the semicolon, which would fix the error.
>  表达式不会返回任何值，或者说等价于返回单元类型 `()`

## 3.4 Comments
All programmers strive to make their code easy to understand, but sometimes extra explanation is warranted. In these cases, programmers leave _comments_ in their source code that the compiler will ignore but people reading the source code may find useful.

Here’s a simple comment:

```
// hello, world
```

In Rust, the idiomatic comment style starts a comment with two slashes, and the comment continues until the end of the line. For comments that extend beyond a single line, you’ll need to include `//` on each line, like this:

```
// So we're doing something complicated here, long enough that we need
// multiple lines of comments to do it! Whew! Hopefully, this comment will
// explain what's going on.
```

Comments can also be placed at the end of lines containing code:

Filename: src/main.rs

```rust
fn main() {
    let lucky_number = 7; // I'm feeling lucky today
}
```

But you’ll more often see them used in this format, with the comment on a separate line above the code it’s annotating:

Filename: src/main.rs

```rust
fn main() {
    // I'm feeling lucky today
    let lucky_number = 7;
}
```

Rust also has another kind of comment, documentation comments, which we’ll discuss in the [“Publishing a Crate to Crates.io”](https://doc.rust-lang.org/stable/book/ch14-02-publishing-to-crates-io.html) section of Chapter 14.

## 3.5 Control Flow
The ability to run some code depending on whether a condition is `true` and to run some code repeatedly while a condition is `true` are basic building blocks in most programming languages. The most common constructs that let you control the flow of execution of Rust code are `if` expressions and loops.

### `if` Expressions
An `if` expression allows you to branch your code depending on conditions. You provide a condition and then state, “If this condition is met, run this block of code. If the condition is not met, do not run this block of code.”
>  `if` 表达式用于基于条件分支代码

Create a new project called _branches_ in your _projects_ directory to explore the `if` expression. In the _src/main.rs_ file, input the following:

Filename: src/main.rs

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

All `if` expressions start with the keyword `if`, followed by a condition. In this case, the condition checks whether or not the variable `number` has a value less than 5. We place the block of code to execute if the condition is `true` immediately after the condition inside curly brackets. Blocks of code associated with the conditions in `if` expressions are sometimes called _arms_, just like the arms in `match` expressions that we discussed in the [“Comparing the Guess to the Secret Number”](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#comparing-the-guess-to-the-secret-number) section of Chapter 2.
>  和 `if` 表达式相关的 blocks of code 称为 arms

Optionally, we can also include an `else` expression, which we chose to do here, to give the program an alternative block of code to execute should the condition evaluate to `false`. If you don’t provide an `else` expression and the condition is `false`, the program will just skip the `if` block and move on to the next bit of code.

Try running this code; you should see the following output:

```
$ cargo run
   Compiling branches v0.1.0 (file:///projects/branches)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.31s
     Running `target/debug/branches`
condition was true
```

Let’s try changing the value of `number` to a value that makes the condition `false` to see what happens:

```rust
let number = 7; 
```

Run the program again, and look at the output:

```
$ cargo run
   Compiling branches v0.1.0 (file:///projects/branches)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.31s
     Running `target/debug/branches`
condition was false
```

It’s also worth noting that the condition in this code _must_ be a `bool`. If the condition isn’t a `bool`, we’ll get an error. For example, try running the following code:
>  注意 `if` 表达式的条件必须是 `bool`

Filename: src/main.rs

```rust
fn main() {
    let number = 3;

    if number {
        println!("number was three");
    }
}
```

The `if` condition evaluates to a value of `3` this time, and Rust throws an error:

```
$ cargo run
   Compiling branches v0.1.0 (file:///projects/branches)
error[E0308]: mismatched types
 --> src/main.rs:4:8
  |
4 |     if number {
  |        ^^^^^^ expected `bool`, found integer

For more information about this error, try `rustc --explain E0308`.
error: could not compile `branches` (bin "branches") due to 1 previous error
```

The error indicates that Rust expected a `bool` but got an integer. Unlike languages such as Ruby and JavaScript, Rust will not automatically try to convert non-Boolean types to a Boolean. You must be explicit and always provide `if` with a Boolean as its condition. If we want the `if` code block to run only when a number is not equal to `0`, for example, we can change the `if` expression to the following:
>  Rust 不会自动将非 bool 变量转化为 bool，我们必须显式提供 bool 作为条件

Filename: src/main.rs

```rust
fn main() {
    let number = 3;

    if number != 0 {
        println!("number was something other than zero");
    }
}
```

Running this code will print `number was something other than zero`.

#### Handling Multiple Conditions with `else if`
You can use multiple conditions by combining `if` and `else` in an `else if` expression. For example:
>  结合 `if, else, else if` 表达式，可以组成多分支

Filename: src/main.rs

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

This program has four possible paths it can take. After running it, you should see the following output:

```
$ cargo run
   Compiling branches v0.1.0 (file:///projects/branches)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.31s
     Running `target/debug/branches`
number is divisible by 3
```

When this program executes, it checks each `if` expression in turn and executes the first body for which the condition evaluates to `true`. Note that even though 6 is divisible by 2, we don’t see the output `number is divisible by 2`, nor do we see the `number is not divisible by 4, 3, or 2` text from the `else` block. That’s because Rust only executes the block for the first `true` condition, and once it finds one, it doesn’t even check the rest.

Using too many `else if` expressions can clutter your code, so if you have more than one, you might want to refactor your code. Chapter 6 describes a powerful Rust branching construct called `match` for these cases.
>  如果有多于一个的 `else if`，可以考虑使用 `match`

#### Using `if` in a `let` Statement
Because `if` is an expression, we can use it on the right side of a `let` statement to assign the outcome to a variable, as in Listing 3-2.

Filename: src/main.rs

```rust
fn main() {
    let condition = true;
    let number = if condition { 5 } else { 6 };

    println!("The value of number is: {number}");
}
```

[Listing 3-2](https://doc.rust-lang.org/stable/book/ch03-05-control-flow.html#listing-3-2): Assigning the result of an `if` expression to a variable

The `number` variable will be bound to a value based on the outcome of the `if` expression. Run this code to see what happens:

```
$ cargo run
   Compiling branches v0.1.0 (file:///projects/branches)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.30s
     Running `target/debug/branches`
The value of number is: 5
```

Remember that blocks of code evaluate to the last expression in them, and numbers by themselves are also expressions. In this case, the value of the whole `if` expression depends on which block of code executes. This means the values that have the potential to be results from each arm of the `if` must be the same type; in Listing 3-2, the results of both the `if` arm and the `else` arm were `i32` integers. If the types are mismatched, as in the following example, we’ll get an error:
>  `if` 是一个表达式，它的返回值是它执行的 code block 的返回值
>  code block 的返回值为它的最后一个表达式
>  这要求 `if` 的每一个可能执行的 code block 都返回相同的类型

Filename: src/main.rs

```rust
fn main() {
    let condition = true;

    let number = if condition { 5 } else { "six" };

    println!("The value of number is: {number}");
}
```

When we try to compile this code, we’ll get an error. The `if` and `else` arms have value types that are incompatible, and Rust indicates exactly where to find the problem in the program:

```
$ cargo run
   Compiling branches v0.1.0 (file:///projects/branches)
error[E0308]: `if` and `else` have incompatible types
 --> src/main.rs:4:44
  |
4 |     let number = if condition { 5 } else { "six" };
  |                                 -          ^^^^^ expected integer, found `&str`
  |                                 |
  |                                 expected because of this

For more information about this error, try `rustc --explain E0308`.
error: could not compile `branches` (bin "branches") due to 1 previous error
```

The expression in the `if` block evaluates to an integer, and the expression in the `else` block evaluates to a string. This won’t work because variables must have a single type, and Rust needs to know at compile time what type the `number` variable is, definitively. Knowing the type of `number` lets the compiler verify the type is valid everywhere we use `number`. Rust wouldn’t be able to do that if the type of `number` was only determined at runtime; the compiler would be more complex and would make fewer guarantees about the code if it had to keep track of multiple hypothetical types for any variable.
>  这样的设计是为了让 Rust 在编译时推导出所有的变量类型，因此不能让变量的类型推迟到 runtime 决定

### Repetition with Loops
It’s often useful to execute a block of code more than once. For this task, Rust provides several _loops_, which will run through the code inside the loop body to the end and then start immediately back at the beginning. To experiment with loops, let’s make a new project called _loops_.

Rust has three kinds of loops: `loop`, `while`, and `for`. Let’s try each one.

#### Repeating Code with `loop`
The `loop` keyword tells Rust to execute a block of code over and over again forever or until you explicitly tell it to stop.
>  `loop` 关键字用于创建无限循环

As an example, change the _src/main.rs_ file in your _loops_ directory to look like this:

Filename: src/main.rs

```rust
fn main() {
    loop {
        println!("again!");
    }
}
```

When we run this program, we’ll see `again!` printed over and over continuously until we stop the program manually. Most terminals support the keyboard shortcut ctrl-c to interrupt a program that is stuck in a continual loop. Give it a try:

```
$ cargo run
   Compiling loops v0.1.0 (file:///projects/loops)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.08s
     Running `target/debug/loops`
again!
again!
again!
again!
^Cagain!
```

The symbol `^C` represents where you pressed ctrl-c. You may or may not see the word `again!` printed after the `^C`, depending on where the code was in the loop when it received the interrupt signal.

Fortunately, Rust also provides a way to break out of a loop using code. You can place the `break` keyword within the loop to tell the program when to stop executing the loop. Recall that we did this in the guessing game in the [“Quitting After a Correct Guess”](https://doc.rust-lang.org/stable/book/ch02-00-guessing-game-tutorial.html#quitting-after-a-correct-guess) section of Chapter 2 to exit the program when the user won the game by guessing the correct number.
>  `break` 关键字用于跳出循环

We also used `continue` in the guessing game, which in a loop tells the program to skip over any remaining code in this iteration of the loop and go to the next iteration.
>  `continue` 关键字用于跳出当次循环

#### Returning Values from Loops
One of the uses of a `loop` is to retry an operation you know might fail, such as checking whether a thread has completed its job. You might also need to pass the result of that operation out of the loop to the rest of your code. To do this, you can add the value you want returned after the `break` expression you use to stop the loop; that value will be returned out of the loop so you can use it, as shown here:
>  在 `break` 语句添加需要返回的值，可以作为 `loop` 表达式的返回值

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

Before the loop, we declare a variable named `counter` and initialize it to `0`. Then we declare a variable named `result` to hold the value returned from the loop. On every iteration of the loop, we add `1` to the `counter` variable, and then check whether the `counter` is equal to `10`. When it is, we use the `break` keyword with the value `counter * 2`. After the loop, we use a semicolon to end the statement that assigns the value to `result`. Finally, we print the value in `result`, which in this case is `20`.

You can also `return` from inside a loop. While `break` only exits the current loop, `return` always exits the current function.

#### Loop Labels to Disambiguate Between Multiple Loops
If you have loops within loops, `break` and `continue` apply to the innermost loop at that point. You can optionally specify a _loop label_ on a loop that you can then use with `break` or `continue` to specify that those keywords apply to the labeled loop instead of the innermost loop. Loop labels must begin with a single quote. Here’s an example with two nested loops:
>  可以指定 loop 标签，用于 `break, continue`
>  loop 标签以单引号开始

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

The outer loop has the label `'counting_up`, and it will count up from 0 to 2. The inner loop without a label counts down from 10 to 9. The first `break` that doesn’t specify a label will exit the inner loop only. The `break 'counting_up;` statement will exit the outer loop. This code prints:

```
$ cargo run
   Compiling loops v0.1.0 (file:///projects/loops)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.58s
     Running `target/debug/loops`
count = 0
remaining = 10
remaining = 9
count = 1
remaining = 10
remaining = 9
count = 2
remaining = 10
End count = 2
```

#### Conditional Loops with `while`
A program will often need to evaluate a condition within a loop. While the condition is `true`, the loop runs. When the condition ceases to be `true`, the program calls `break`, stopping the loop. It’s possible to implement behavior like this using a combination of `loop`, `if`, `else`, and `break`; you could try that now in a program, if you’d like. However, this pattern is so common that Rust has a built-in language construct for it, called a `while` loop. In Listing 3-3, we use `while` to loop the program three times, counting down each time, and then, after the loop, print a message and exit.

Filename: src/main.rs

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

[Listing 3-3](https://doc.rust-lang.org/stable/book/ch03-05-control-flow.html#listing-3-3): Using a `while` loop to run code while a condition evaluates to `true`

This construct eliminates a lot of nesting that would be necessary if you used `loop`, `if`, `else`, and `break`, and it’s clearer. While a condition evaluates to `true`, the code runs; otherwise, it exits the loop.

#### Looping Through a Collection with `for`
You can also use the `while` construct to loop over the elements of a collection, such as an array. For example, the loop in Listing 3-4 prints each element in the array `a`.

Filename: src/main.rs

```rust
fn main() {
    let a = [10, 20, 30, 40, 50];
    let mut index = 0;

    while index < 5 {
        println!("the value is: {}", a[index]);

        index += 1;
    }
}
```

[Listing 3-4](https://doc.rust-lang.org/stable/book/ch03-05-control-flow.html#listing-3-4): Looping through each element of a collection using a `while` loop

Here, the code counts up through the elements in the array. It starts at index `0`, and then loops until it reaches the final index in the array (that is, when `index < 5` is no longer `true`). Running this code will print every element in the array:

```
$ cargo run
   Compiling loops v0.1.0 (file:///projects/loops)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.32s
     Running `target/debug/loops`
the value is: 10
the value is: 20
the value is: 30
the value is: 40
the value is: 50
```

All five array values appear in the terminal, as expected. Even though `index` will reach a value of `5` at some point, the loop stops executing before trying to fetch a sixth value from the array.

However, this approach is error prone; we could cause the program to panic if the index value or test condition is incorrect. For example, if you changed the definition of the `a` array to have four elements but forgot to update the condition to `while index < 4`, the code would panic. It’s also slow, because the compiler adds runtime code to perform the conditional check of whether the index is within the bounds of the array on every iteration through the loop.
>  使用 `while` 遍历数组易错，且需要编译器添加 runtime code 来执行是否索引在每个迭代都没有越界的条件检查

As a more concise alternative, you can use a `for` loop and execute some code for each item in a collection. A `for` loop looks like the code in Listing 3-5.

Filename: src/main.rs

```rust
fn main() {
    let a = [10, 20, 30, 40, 50];

    for element in a {
        println!("the value is: {element}");
    }
}
```

[Listing 3-5](https://doc.rust-lang.org/stable/book/ch03-05-control-flow.html#listing-3-5): Looping through each element of a collection using a `for` loop

When we run this code, we’ll see the same output as in Listing 3-4. More importantly, we’ve now increased the safety of the code and eliminated the chance of bugs that might result from going beyond the end of the array or not going far enough and missing some items.

Using the `for` loop, you wouldn’t need to remember to change any other code if you changed the number of values in the array, as you would with the method used in Listing 3-4.

The safety and conciseness of `for` loops make them the most commonly used loop construct in Rust. Even in situations in which you want to run some code a certain number of times, as in the countdown example that used a `while` loop in Listing 3-3, most Rustaceans would use a `for` loop. The way to do that would be to use a `Range`, provided by the standard library, which generates all numbers in sequence starting from one number and ending before another number.
>  `for` 是最常用的循环结构

Here’s what the countdown would look like using a `for` loop and another method we’ve not yet talked about, `rev`, to reverse the range:
>  `for` 配合 `range` 表达式可以实现 countdown
>  `rev` 用于反转 range

Filename: src/main.rs

```rust
fn main() {
    for number in (1..4).rev() {
        println!("{number}!");
    }
    println!("LIFTOFF!!!");
}
```

This code is a bit nicer, isn’t it?

## Summary
You made it! This was a sizable chapter: you learned about variables, scalar and compound data types, functions, comments, `if` expressions, and loops! To practice with the concepts discussed in this chapter, try building programs to do the following:

- Convert temperatures between Fahrenheit and Celsius.
- Generate the _n_th Fibonacci number.
- Print the lyrics to the Christmas carol “The Twelve Days of Christmas,” taking advantage of the repetition in the song.

When you’re ready to move on, we’ll talk about a concept in Rust that _doesn’t_ commonly exist in other programming languages: ownership.

# 4 Understanding Ownership
Ownership is Rust’s most unique feature and has deep implications for the rest of the language. It enables Rust to make memory safety guarantees without needing a garbage collector, so it’s important to understand how ownership works. In this chapter, we’ll talk about ownership as well as several related features: borrowing, slices, and how Rust lays data out in memory.
>  所有权机制让 Rust 不需要 GC 的情况下保证内存安全

## 4.1 What Is Ownership?
_Ownership_ is a set of rules that govern how a Rust program manages memory. All programs have to manage the way they use a computer’s memory while running. Some languages have garbage collection that regularly looks for no-longer-used memory as the program runs; in other languages, the programmer must explicitly allocate and free the memory. Rust uses a third approach: memory is managed through a system of ownership with a set of rules that the compiler checks. If any of the rules are violated, the program won’t compile. None of the features of ownership will slow down your program while it’s running.
>  所有权是 Rust 管理内存的一系列规则，之前的内存管理方式无非两种:
>  - GC，周期性地收集程序运行中不再使用的内存
>  - 人为分配和释放
>  Rust 提出第三种方法: 通过所有权系统管理，所有权系统由一组规则组成，编译器在编译时验证程序是否符合这些规则，如果违反了规则，编译失败，因此所有权规则完全不涉及程序的运行效率

Because ownership is a new concept for many programmers, it does take some time to get used to. The good news is that the more experienced you become with Rust and the rules of the ownership system, the easier you’ll find it to naturally develop code that is safe and efficient. Keep at it!

When you understand ownership, you’ll have a solid foundation for understanding the features that make Rust unique. In this chapter, you’ll learn ownership by working through some examples that focus on a very common data structure: strings.

**The Stack and the Heap**
Many programming languages don’t require you to think about the stack and the heap very often. But in a systems programming language like Rust, whether a value is on the stack or the heap affects how the language behaves and why you have to make certain decisions. Parts of ownership will be described in relation to the stack and the heap later in this chapter, so here is a brief explanation in preparation.

Both the stack and the heap are parts of memory available to your code to use at runtime, but they are structured in different ways. The stack stores values in the order it gets them and removes the values in the opposite order. This is referred to as _last in, first out_. Think of a stack of plates: when you add more plates, you put them on top of the pile, and when you need a plate, you take one off the top. Adding or removing plates from the middle or bottom wouldn’t work as well! Adding data is called _pushing onto the stack_, and removing data is called _popping off the stack_. All data stored on the stack must have a known, fixed size. Data with an unknown size at compile time or a size that might change must be stored on the heap instead.
>  stack, heap 都是程序运行时可以访问的内存
>  stack 顺序存储，后进先出，所有存储在 stack 上的数据必须是编译时已知的固定大小，编译时大小未知的数据，或者大小会在运行时改变的数据必须存储在 heap

The heap is less organized: when you put data on the heap, you request a certain amount of space. The memory allocator finds an empty spot in the heap that is big enough, marks it as being in use, and returns a _pointer_, which is the address of that location. This process is called _allocating on the heap_ and is sometimes abbreviated as just _allocating_ (pushing values onto the stack is not considered allocating). Because the pointer to the heap is a known, fixed size, you can store the pointer on the stack, but when you want the actual data, you must follow the pointer. Think of being seated at a restaurant. When you enter, you state the number of people in your group, and the host finds an empty table that fits everyone and leads you there. If someone in your group comes late, they can ask where you’ve been seated to find you.
>  heap 由 memory allocator 管理，allocator 在运行时找到足够大的 empty spot，将其标记为使用，然后返回其地址
>  这个过程就是在 heap 上 allocate memory 的过程 (push value to stack 不属于 allocate)
>  注意，pointer 是固定大小的，因此 pointer 可以存储在 stack 上

Pushing to the stack is faster than allocating on the heap because the allocator never has to search for a place to store new data; that location is always at the top of the stack. Comparatively, allocating space on the heap requires more work because the allocator must first find a big enough space to hold the data and then perform bookkeeping to prepare for the next allocation.
>  push to stack 快于 allocate on heap，因为不需要寻找空位置，push 到 stack top 即可
>  allocate on heap 要求寻找空位，并且 allocator 需要做 bookkeeping

Accessing data in the heap is slower than accessing data on the stack because you have to follow a pointer to get there. Contemporary processors are faster if they jump around less in memory. Continuing the analogy, consider a server at a restaurant taking orders from many tables. It’s most efficient to get all the orders at one table before moving on to the next table. Taking an order from table A, then an order from table B, then one from A again, and then one from B again would be a much slower process. By the same token, a processor can do its job better if it works on data that’s close to other data (as it is on the stack) rather than farther away (as it can be on the heap).
>  访问 heap 数据也比访问 stack 数据慢，因为需要通过 pointer 间接访问
>  目前的处理器如果在 memory 中的跳跃范围较小，可以较快地访问，如果处理器在一串比较相近的数据中跳跃 (stack 上)，就比它在比较随机远离的数据中跳跃 (heap 上) 更快

When your code calls a function, the values passed into the function (including, potentially, pointers to data on the heap) and the function’s local variables get pushed onto the stack. When the function is over, those values get popped off the stack.
>  调用函数时，传递给函数的 values 和函数的局部变量都会 push onto stack，函数结束时，values pop off the stack

Keeping track of what parts of code are using what data on the heap, minimizing the amount of duplicate data on the heap, and cleaning up unused data on the heap so you don’t run out of space are all problems that ownership addresses. Once you understand ownership, you won’t need to think about the stack and the heap very often, but knowing that the main purpose of ownership is to manage heap data can help explain why it works the way it does.
>  所有权机制解决了: 追踪那部分代码在 heap 上使用数据、最小化 heap 上的重复数据、清除 heap 上不用的数据避免空间用完，等问题

### Ownership Rules
First, let’s take a look at the ownership rules. Keep these rules in mind as we work through the examples that illustrate them:

- Each value in Rust has an _owner_.
- There can only be one owner at a time.
- When the owner goes out of scope, the value will be dropped.

>  记住这些概念 (所有权规则):
>  - Rust 中的每个 value 都有一个 owner
>  - 每个 value 一次只能有一个 owner
>  - 如果 owner 离开作用域，value 必须被丢弃

### Variable Scope
Now that we’re past basic Rust syntax, we won’t include all the `fn main() {` code in examples, so if you’re following along, make sure to put the following examples inside a `main` function manually. As a result, our examples will be a bit more concise, letting us focus on the actual details rather than boilerplate code.

As a first example of ownership, we’ll look at the _scope_ of some variables. A scope is the range within a program for which an item is valid. Take the following variable:
>  考虑变量的作用域
>  作用域是程序中的一个范围，该范围内，一个 item 是有效的

```rust
let s = "hello";
```

The variable `s` refers to a string literal, where the value of the string is hardcoded into the text of our program. The variable is valid from the point at which it’s declared until the end of the current _scope_. Listing 4-1 shows a program with comments annotating where the variable `s` would be valid.
>  变量从它的声明点，到当前作用域结束，保持有效

```rust
{                      // s is not valid here, it’s not yet declared
    let s = "hello";   // s is valid from this point forward

    // do stuff with s
}                      // this scope is now over, and s is no longer valid
```

[Listing 4-1](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html#listing-4-1): A variable and the scope in which it is valid

In other words, there are two important points in time here:

- When `s` comes _into_ scope, it is valid.
- It remains valid until it goes _out of_ scope.

>  也就是两点规则:
>  - 变量进入作用域，开始有效
>  - 变量离开作用域之前，奥驰有效

At this point, the relationship between scopes and when variables are valid is similar to that in other programming languages. Now we’ll build on top of this understanding by introducing the `String` type.

### The `String` Type
To illustrate the rules of ownership, we need a data type that is more complex than those we covered in the [“Data Types”](https://doc.rust-lang.org/stable/book/ch03-02-data-types.html#data-types) section of Chapter 3. The types covered previously are of a known size, can be stored on the stack and popped off the stack when their scope is over, and can be quickly and trivially copied to make a new, independent instance if another part of code needs to use the same value in a different scope. But we want to look at data that is stored on the heap and explore how Rust knows when to clean up that data, and the `String` type is a great example.
>  目前考虑的类型都是固定大小，在进入作用域时入栈，在离开作用域时出栈，因此也可以在其他代码在其他作用域使用相同的 value 时被快速且轻量地 copy 以创建一个新的独立 instance

We’ll concentrate on the parts of `String` that relate to ownership. These aspects also apply to other complex data types, whether they are provided by the standard library or created by you. We’ll discuss `String` in more depth in [Chapter 8](https://doc.rust-lang.org/stable/book/ch08-02-strings.html).

We’ve already seen string literals, where a string value is hardcoded into our program. String literals are convenient, but they aren’t suitable for every situation in which we may want to use text. One reason is that they’re immutable. Another is that not every string value can be known when we write our code: for example, what if we want to take user input and store it? For these situations, Rust has a second string type, `String`. This type manages data allocated on the heap and as such is able to store an amount of text that is unknown to us at compile time. You can create a `String` from a string literal using the `from` function, like so:
>  `String` 类型管理 heap 上的数据，存储编译时未知的文本数据 
>  `String::from` 用于从字面值创建 `String`

```rust
let s = String::from("hello");
```

The double colon `::` operator allows us to namespace this particular `from` function under the `String` type rather than using some sort of name like `string_from`. We’ll discuss this syntax more in the [“Method Syntax”](https://doc.rust-lang.org/stable/book/ch05-03-method-syntax.html#method-syntax) section of Chapter 5, and when we talk about namespacing with modules in [“Paths for Referring to an Item in the Module Tree”](https://doc.rust-lang.org/stable/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html) in Chapter 7.

This kind of string _can_ be mutated:

```rust
let mut s = String::from("hello");

s.push_str(", world!"); // push_str() appends a literal to a String

println!("{s}"); // This will print `hello, world!`
```

So, what’s the difference here? Why can `String` be mutated but literals cannot? The difference is in how these two types deal with memory.

### Memory and Allocation
In the case of a string literal, we know the contents at compile time, so the text is hardcoded directly into the final executable. This is why string literals are fast and efficient. But these properties only come from the string literal’s immutability. Unfortunately, we can’t put a blob of memory into the binary for each piece of text whose size is unknown at compile time and whose size might change while running the program.
>  string literal 的值在编译时已知，故其 value 会被硬编码到最终的 exe
>  因此 string literal 是快速且高效的，但这都依赖于 string literal 的不可变性

With the `String` type, in order to support a mutable, growable piece of text, we need to allocate an amount of memory on the heap, unknown at compile time, to hold the contents. This means:

- The memory must be requested from the memory allocator at runtime.
- We need a way of returning this memory to the allocator when we’re done with our `String`.

>  `String` 支持运行时可变的值，意味着:
>  - 需要在运行时由 memory allocator 请求 memory
>  - 用完 `String` 之后，需要一种将 memory 返回给 allocator 的方式

That first part is done by us: when we call `String::from`, its implementation requests the memory it needs. This is pretty much universal in programming languages.
>  第一个部分由我们完成: `String::from` 请求了 memory，各个语言都一样

However, the second part is different. In languages with a _garbage collector (GC)_, the GC keeps track of and cleans up memory that isn’t being used anymore, and we don’t need to think about it. In most languages without a GC, it’s our responsibility to identify when memory is no longer being used and to call code to explicitly free it, just as we did to request it. Doing this correctly has historically been a difficult programming problem. If we forget, we’ll waste memory. If we do it too early, we’ll have an invalid variable. If we do it twice, that’s a bug too. We need to pair exactly one `allocate` with exactly one `free`.
>  第二个部分则各个语言不同

Rust takes a different path: the memory is automatically returned once the variable that owns it goes out of scope. Here’s a version of our scope example from Listing 4-1 using a `String` instead of a string literal:
>  Rust 中，memory 会在拥有了它的变量离开作用域的时候，被自动归还 (非常合理，变量离开了作用域，说明不能再使用，其资源就必须归还，否则拿着也没用)

```rust
{
    let s = String::from("hello"); // s is valid from this point forward

    // do stuff with s
}                                  // this scope is now over, and s is no longer valid
```

There is a natural point at which we can return the memory our `String` needs to the allocator: when `s` goes out of scope. When a variable goes out of scope, Rust calls a special function for us. This function is called [`drop`](https://doc.rust-lang.org/stable/std/ops/trait.Drop.html#tymethod.drop), and it’s where the author of `String` can put the code to return the memory. Rust calls `drop` automatically at the closing curly bracket.
>  当变量离开作用域，Rust 会调用变量的 `drop` 函数，`String` 的作者需要将返回 memory 的代码写在 `drop` 函数中
>  Rust 会在 `}` 自动调用 `drop` 函数

Note: In C++, this pattern of deallocating resources at the end of an item’s lifetime is sometimes called _Resource Acquisition Is Initialization (RAII)_. The `drop` function in Rust will be familiar to you if you’ve used RAII patterns.
>  C++ 中，这个在某个 item 的生命周期结束时释放资源的模式有时称为资源获取即初始化 (RAII) 
>  RAII 即把资源获取写在构造函数，把资源释放写在析构函数，使得资源的生命周期和 item 的生命周期相同

This pattern has a profound impact on the way Rust code is written. It may seem simple right now, but the behavior of code can be unexpected in more complicated situations when we want to have multiple variables use the data we’ve allocated on the heap. Let’s explore some of those situations now.

#### Variables and Data Interacting with Move
Multiple variables can interact with the same data in different ways in Rust. Let’s look at an example using an integer in Listing 4-2.

```rust
let x = 5;
let y = x;
```

[Listing 4-2](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html#listing-4-2): Assigning the integer value of variable `x` to `y`

We can probably guess what this is doing: “bind the value `5` to `x`; then make a copy of the value in `x` and bind it to `y`.” We now have two variables, `x` and `y`, and both equal `5`. This is indeed what is happening, because integers are simple values with a known, fixed size, and these two `5` values are pushed onto the stack.
>  上述代码先将 value 5 绑定到 x，然后拷贝了 x 的 value，将其绑定到 y
>  我们现在有两个变量 x, y，都等于 5
>  因为整数 value 的大小是固定的，故这两个 `5` value 都会入栈

Now let’s look at the `String` version:

```rust
let s1 = String::from("hello");
let s2 = s1;
```

This looks very similar, so we might assume that the way it works would be the same: that is, the second line would make a copy of the value in `s1` and bind it to `s2`. But this isn’t quite what happens.

Take a look at Figure 4-1 to see what is happening to `String` under the covers. A `String` is made up of three parts, shown on the left: a pointer to the memory that holds the contents of the string, a length, and a capacity. This group of data is stored on the stack. On the right is the memory on the heap that holds the contents.
>  `String` 由三个部分组成: pointer, length, capacity
>  这三个部分的数据都存在 stack 上

![Two tables: the first table contains the representation of s1 on the stack, consisting of its length (5), capacity (5), and a pointer to the first value in the second table. The second table contains the representation of the string data on the heap, byte by byte.](https://doc.rust-lang.org/stable/book/img/trpl04-01.svg)

Figure 4-1: Representation in memory of a `String` holding the value `"hello"` bound to `s1`

The length is how much memory, in bytes, the contents of the `String` are currently using. The capacity is the total amount of memory, in bytes, that the `String` has received from the allocator. The difference between length and capacity matters, but not in this context, so for now, it’s fine to ignore the capacity.
>  length, capacity 的单位为 bytes
>  length 表示 `String` 内容的长度，capacity 表示 allocator 赋予的大小

When we assign `s1` to `s2`, the `String` data is copied, meaning we copy the pointer, the length, and the capacity that are on the stack. We do not copy the data on the heap that the pointer refers to. In other words, the data representation in memory looks like Figure 4-2.
>  直接赋值仅仅会拷贝 stack 上的 `String` 数据，即 pointer, length, capacity，也就是浅拷贝

![Three tables: tables s1 and s2 representing those strings on the stack, respectively, and both pointing to the same string data on the heap.](https://doc.rust-lang.org/stable/book/img/trpl04-02.svg)

Figure 4-2: Representation in memory of the variable `s2` that has a copy of the pointer, length, and capacity of `s1`

The representation does _not_ look like Figure 4-3, which is what memory would look like if Rust instead copied the heap data as well. If Rust did this, the operation `s2 = s1` could be very expensive in terms of runtime performance if the data on the heap were large.

![Four tables: two tables representing the stack data for s1 and s2, and each points to its own copy of string data on the heap.](https://doc.rust-lang.org/stable/book/img/trpl04-03.svg)

Figure 4-3: Another possibility for what `s2 = s1` might do if Rust copied the heap data as well

Earlier, we said that when a variable goes out of scope, Rust automatically calls the `drop` function and cleans up the heap memory for that variable. But Figure 4-2 shows both data pointers pointing to the same location. This is a problem: when `s2` and `s1` go out of scope, they will both try to free the same memory. This is known as a _double free_ error and is one of the memory safety bugs we mentioned previously. Freeing memory twice can lead to memory corruption, which can potentially lead to security vulnerabilities.
>  我们知道 Rust 在变量离开作用域时会调用 `drop` 清理该变量的 heap memory
>  显然，如果像上例一样直接赋值，就会导致重复释放 memory

To ensure memory safety, after the line `let s2 = s1;`, Rust considers `s1` as no longer valid. Therefore, Rust doesn’t need to free anything when `s1` goes out of scope. Check out what happens when you try to use `s1` after `s2` is created; it won’t work:
>  为了确保 memory safety，在 `let s2 = s1` 之后，Rust 认为 `s1` 不再有效，因此不会在 `s1` 离开作用域后释放 memory

```rust
let s1 = String::from("hello");
let s2 = s1;

println!("{s1}, world!");
```

You’ll get an error like this because Rust prevents you from using the invalidated reference:
>  在该赋值之后再使用 `s1` 将触发编译错误，因为不允许使用无效的引用

```
$ cargo run
   Compiling ownership v0.1.0 (file:///projects/ownership)
error[E0382]: borrow of moved value: `s1`
 --> src/main.rs:5:15
  |
2 |     let s1 = String::from("hello");
  |         -- move occurs because `s1` has type `String`, which does not implement the `Copy` trait
3 |     let s2 = s1;
  |              -- value moved here
4 |
5 |     println!("{s1}, world!");
  |               ^^^^ value borrowed here after move
  |
  = note: this error originates in the macro `$crate::format_args_nl` which comes from the expansion of the macro `println` (in Nightly builds, run with -Z macro-backtrace for more info)
help: consider cloning the value if the performance cost is acceptable
  |
3 |     let s2 = s1.clone();
  |                ++++++++

For more information about this error, try `rustc --explain E0382`.
error: could not compile `ownership` (bin "ownership") due to 1 previous error
```

If you’ve heard the terms _shallow copy_ and _deep copy_ while working with other languages, the concept of copying the pointer, length, and capacity without copying the data probably sounds like making a shallow copy. But because Rust also invalidates the first variable, instead of being called a shallow copy, it’s known as a _move_. In this example, we would say that `s1` was _moved_ into `s2`. So, what actually happens is shown in Figure 4-4.
>  实际上，因为 Rust 在赋值时还无效化了上一个变量，故赋值操作不仅仅是执行一个 shallow copy，而是执行了 move
>  也就是 `s1` 被 move 到了 `s2`

![Three tables: tables s1 and s2 representing those strings on the stack, respectively, and both pointing to the same string data on the heap. Table s1 is grayed out be-cause s1 is no longer valid; only s2 can be used to access the heap data.](https://doc.rust-lang.org/stable/book/img/trpl04-04.svg)

Figure 4-4: Representation in memory after `s1` has been invalidated

That solves our problem! With only `s2` valid, when it goes out of scope it alone will free the memory, and we’re done.

In addition, there’s a design choice that’s implied by this: Rust will never automatically create “deep” copies of your data. Therefore, any _automatic_ copying can be assumed to be inexpensive in terms of runtime performance.
>  Rust 永远不会自动 deep copy，任意的自动拷贝都可以认为是对于 runtime 性能轻量的

#### Scope and Assignment
The inverse of this is true for the relationship between scoping, ownership, and memory being freed via the `drop` function as well. When you assign a completely new value to an existing variable, Rust will call `drop` and free the original value’s memory immediately. Consider this code, for example:
>  当我们将一个新的 value 赋值给已经存在的变量时，Rust 会对原来的 value 调用 `drop`，立即释放其 memory (因为没有其他对原来的 value 的引用了)

```rust
let mut s = String::from("hello");
s = String::from("ahoy");

println!("{s}, world!");
```

We initially declare a variable `s` and bind it to a `String` with the value `"hello"`. Then we immediately create a new `String` with the value `"ahoy"` and assign it to `s`. At this point, nothing is referring to the original value on the heap at all.

![One table s representing the string value on the stack, pointing to the second piece of string data (ahoy) on the heap, with the original string data (hello) grayed out because it cannot be accessed anymore.](https://doc.rust-lang.org/stable/book/img/trpl04-05.svg)

Figure 4-5: Representation in memory after the initial value has been replaced in its entirety.

The original string thus immediately goes out of scope. Rust will run the `drop` function on it and its memory will be freed right away. When we print the value at the end, it will be `"ahoy, world!"`.
>  在没有任何对原来 value 的引用后，原来的 value 立即离开了作用域，Rust 进而对它调用 `drop`

#### Variables and Data Interacting with Clone
If we _do_ want to deeply copy the heap data of the `String`, not just the stack data, we can use a common method called `clone`. We’ll discuss method syntax in Chapter 5, but because methods are a common feature in many programming languages, you’ve probably seen them before.
>  deep copy 需要使用 `clone` 方法

Here’s an example of the `clone` method in action:

```rust
let s1 = String::from("hello");
let s2 = s1.clone();

println!("s1 = {s1}, s2 = {s2}");
```

This works just fine and explicitly produces the behavior shown in Figure 4-3, where the heap data _does_ get copied.

When you see a call to `clone`, you know that some arbitrary code is being executed and that code may be expensive. It’s a visual indicator that something different is going on.

#### Stack-Only Data: Copy
There’s another wrinkle we haven’t talked about yet. This code using integers—part of which was shown in Listing 4-2—works and is valid:

```rust
let x = 5;
let y = x;

println!("x = {x}, y = {y}");
```

But this code seems to contradict what we just learned: we don’t have a call to `clone`, but `x` is still valid and wasn’t moved into `y`.

The reason is that types such as integers that have a known size at compile time are stored entirely on the stack, so copies of the actual values are quick to make. That means there’s no reason we would want to prevent `x` from being valid after we create the variable `y`. In other words, there’s no difference between deep and shallow copying here, so calling `clone` wouldn’t do anything different from the usual shallow copying, and we can leave it out.
>  注意对于编译时已知大小，存储在 stack 上的数据，因为对它的 copy 是快速的操作，因此不会有赋值时的 move 出现，都是 copy

Rust has a special annotation called the `Copy` trait that we can place on types that are stored on the stack, as integers are (we’ll talk more about traits in [Chapter 10](https://doc.rust-lang.org/stable/book/ch10-02-traits.html)). If a type implements the `Copy` trait, variables that use it do not move, but rather are trivially copied, making them still valid after assignment to another variable.
>  Rust 有一个特殊的注解称为 `Copy` trait，我们可以将其放在存储在 stack 的类型上
>  如果一个类型实现了 `Copy` trait，那么使用这个类型的变量在对另一个变量赋值时就不会 move，而是 copy

Rust won’t let us annotate a type with `Copy` if the type, or any of its parts, has implemented the `Drop` trait. If the type needs something special to happen when the value goes out of scope and we add the `Copy` annotation to that type, we’ll get a compile-time error. To learn about how to add the `Copy` annotation to your type to implement the trait, see [“Derivable Traits”](https://doc.rust-lang.org/stable/book/appendix-03-derivable-traits.html) in Appendix C.
>  Rust 不允许对已经实现了 `Drop` trait 的类型实现 `Copy` trait
>  如果一个类型需要在 value 离开作用域后做一些特殊操作，而我们又为它添加了 `Copy` 注解，就会得到编译时错误

So, what types implement the `Copy` trait? You can check the documentation for the given type to be sure, but as a general rule, any group of simple scalar values can implement `Copy`, and nothing that requires allocation or is some form of resource can implement `Copy`. Here are some of the types that implement `Copy`:

- All the integer types, such as `u32`.
- The Boolean type, `bool`, with values `true` and `false`.
- All the floating-point types, such as `f64`.
- The character type, `char`.
- Tuples, if they only contain types that also implement `Copy`. For example, `(i32, i32)` implements `Copy`, but `(i32, String)` does not.

>  任意的简单标量值组合都可以实现 `Copy`，任何需要分配内存或某种资源的类型都不能实现 `Copy`，实现了 `Copy` 的类型包括:
>  - 整数类型
>  - 布尔类型
>  - 浮点类型
>  - 字符类型
>  - 元组类型 (仅包含实现了 `Copy` 的 types)

### Ownership and Functions
The mechanics of passing a value to a function are similar to those when assigning a value to a variable. Passing a variable to a function will move or copy, just as assignment does. Listing 4-3 has an example with some annotations showing where variables go into and out of scope.
>  对函数传递 value 的机制和将 value 赋值给变量的机制类似: 对函数传递 value 同样会 move or copy

Filename: src/main.rs

```rust
fn main() {
    let s = String::from("hello");  // s comes into scope

    takes_ownership(s);             // s's value moves into the function...
                                    // ... and so is no longer valid here

    let x = 5;                      // x comes into scope

    makes_copy(x);                  // because i32 implements the Copy trait,
                                    // x does NOT move into the function,
    println!("{}", x);              // so it's okay to use x afterward

} // Here, x goes out of scope, then s. But because s's value was moved, nothing
  // special happens.

fn takes_ownership(some_string: String) { // some_string comes into scope
    println!("{some_string}");
} // Here, some_string goes out of scope and `drop` is called. The backing
  // memory is freed.

fn makes_copy(some_integer: i32) { // some_integer comes into scope
    println!("{some_integer}");
} // Here, some_integer goes out of scope. Nothing special happens.
```

[Listing 4-3](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html#listing-4-3): Functions with ownership and scope annotated

If we tried to use `s` after the call to `takes_ownership`, Rust would throw a compile-time error. These static checks protect us from mistakes. Try adding code to `main` that uses `s` and `x` to see where you can use them and where the ownership rules prevent you from doing so.

### Return Values and Scope
Returning values can also transfer ownership. Listing 4-4 shows an example of a function that returns some value, with similar annotations as those in Listing 4-3.
>  返回 value 同样会转移所有权

Filename: src/main.rs

```rust
fn main() {
    let s1 = gives_ownership();        // gives_ownership moves its return
                                       // value into s1

    let s2 = String::from("hello");    // s2 comes into scope

    let s3 = takes_and_gives_back(s2); // s2 is moved into
                                       // takes_and_gives_back, which also
                                       // moves its return value into s3
} // Here, s3 goes out of scope and is dropped. s2 was moved, so nothing
  // happens. s1 goes out of scope and is dropped.

fn gives_ownership() -> String {       // gives_ownership will move its
                                       // return value into the function
                                       // that calls it

    let some_string = String::from("yours"); // some_string comes into scope

    some_string                        // some_string is returned and
                                       // moves out to the calling
                                       // function
}

// This function takes a String and returns a String.
fn takes_and_gives_back(a_string: String) -> String {
    // a_string comes into
    // scope

    a_string  // a_string is returned and moves out to the calling function
}
```


[Listing 4-4](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html#listing-4-4): Transferring ownership of return values

The ownership of a variable follows the same pattern every time: assigning a value to another variable moves it. When a variable that includes data on the heap goes out of scope, the value will be cleaned up by `drop` unless ownership of the data has been moved to another variable.
>  变量的所有权永远遵循相同的模式: 将 value 赋予给另一个变量会移动所有权，当变量离开作用域，value 会被 `drop` 清理，除非该变量对数据的所有权已经被移动走

While this works, taking ownership and then returning ownership with every function is a bit tedious. What if we want to let a function use a value but not take ownership? It’s quite annoying that anything we pass in also needs to be passed back if we want to use it again, in addition to any data resulting from the body of the function that we might want to return as well.

Rust does let us return multiple values using a tuple, as shown in Listing 4-5.

Filename: src/main.rs

```rust
fn main() {
    let s1 = String::from("hello");

    let (s2, len) = calculate_length(s1);

    println!("The length of '{s2}' is {len}.");
}

fn calculate_length(s: String) -> (String, usize) {
    let length = s.len(); // len() returns the length of a String

    (s, length)
}
```

[Listing 4-5](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html#listing-4-5): Returning ownership of parameters

But this is too much ceremony and a lot of work for a concept that should be common. Luckily for us, Rust has a feature for using a value without transferring ownership, called _references_.

## 4.2 References and Borrowing
The issue with the tuple code in Listing 4-5 is that we have to return the `String` to the calling function so we can still use the `String` after the call to `calculate_length`, because the `String` was moved into `calculate_length`. Instead, we can provide a reference to the `String` value. A _reference_ is like a pointer in that it’s an address we can follow to access the data stored at that address; that data is owned by some other variable. Unlike a pointer, a reference is guaranteed to point to a valid value of a particular type for the life of that reference.
>  引用类似于指针，它是一个地址，我们通过该地址访问存储在该地址的数据，而这些数据由其他变量所有
>  与指针不同的是，引用保证在其生命周期内始终指向特定类型的一个有效值

Here is how you would define and use a `calculate_length` function that has a reference to an object as a parameter instead of taking ownership of the value:

Filename: src/main.rs

```rust
fn main() {
    let s1 = String::from("hello");

    let len = calculate_length(&s1);

    println!("The length of '{s1}' is {len}.");
}

fn calculate_length(s: &String) -> usize {
    s.len()
}
```

First, notice that all the tuple code in the variable declaration and the function return value is gone. Second, note that we pass `&s1` into `calculate_length` and, in its definition, we take `&String` rather than `String`. These ampersands represent _references_, and they allow you to refer to some value without taking ownership of it. Figure 4-6 depicts this concept.
>  我们在函数签名中用 `&` 表示引用，并且在传递实参时使用 `&` 表示取引用，这使得传递参数不会转移所有权

![Three tables: the table for s contains only a pointer to the table for s1. The table for s1 contains the stack data for s1 and points to the string data on the heap.](https://doc.rust-lang.org/stable/book/img/trpl04-06.svg)

Figure 4-6: A diagram of `&String s` pointing at `String s1`

Note: The opposite of referencing by using `&` is _dereferencing_, which is accomplished with the dereference operator, `*`. We’ll see some uses of the dereference operator in Chapter 8 and discuss details of dereferencing in Chapter 15.

Let’s take a closer look at the function call here:

```rust
let s1 = String::from("hello");

let len = calculate_length(&s1);
```

The `&s1` syntax lets us create a reference that _refers_ to the value of `s1` but does not own it. Because the reference does not own it, the value it points to will not be dropped when the reference stops being used.
>  `&s1` 创建了一个引用了 `s1` 的 value ，但不拥有该 value 的引用
>  因为引用不拥有 value，故 value 不会在引用不再被使用后被 drop

Likewise, the signature of the function uses `&` to indicate that the type of the parameter `s` is a reference. Let’s add some explanatory annotations:

```rust
fn calculate_length(s: &String) -> usize { // s is a reference to a String
    s.len()
} // Here, s goes out of scope. But because s does not have ownership of what
  // it refers to, the value is not dropped.
```

The scope in which the variable `s` is valid is the same as any function parameter’s scope, but the value pointed to by the reference is not dropped when `s` stops being used, because `s` doesn’t have ownership. When functions have references as parameters instead of the actual values, we won’t need to return the values in order to give back ownership, because we never had ownership.
>  函数内的引用变量在函数结束后离开作用域，但它不拥有 value，没有所有权，故 value 不会被 drop

We call the action of creating a reference _borrowing_. As in real life, if a person owns something, you can borrow it from them. When you’re done, you have to give it back. You don’t own it.
>  我们称创建引用的动作为 “借用”

So, what happens if we try to modify something we’re borrowing? Try the code in Listing 4-6. Spoiler alert: it doesn’t work!

Filename: src/main.rs

```rust
fn main() {
    let s = String::from("hello");

    change(&s);
}

fn change(some_string: &String) {
    some_string.push_str(", world");
}
```

[Listing 4-6](https://doc.rust-lang.org/stable/book/ch04-02-references-and-borrowing.html#listing-4-6): Attempting to modify a borrowed value

Here’s the error:

```
$ cargo run
   Compiling ownership v0.1.0 (file:///projects/ownership)
error[E0596]: cannot borrow `*some_string` as mutable, as it is behind a `&` reference
 --> src/main.rs:8:5
  |
8 |     some_string.push_str(", world");
  |     ^^^^^^^^^^^ `some_string` is a `&` reference, so the data it refers to cannot be borrowed as mutable
  |
help: consider changing this to be a mutable reference
  |
7 | fn change(some_string: &mut String) {
  |                         +++

For more information about this error, try `rustc --explain E0596`.
error: could not compile `ownership` (bin "ownership") due to 1 previous error
```

Just as variables are immutable by default, so are references. We’re not allowed to modify something we have a reference to.
>   引用默认不可变

### Mutable References
We can fix the code from Listing 4-6 to allow us to modify a borrowed value with just a few small tweaks that use, instead, a _mutable reference_:

Filename: src/main.rs

```rust
fn main() {
    let mut s = String::from("hello");

    change(&mut s);
}

fn change(some_string: &mut String) {
    some_string.push_str(", world");
}
```

First we change `s` to be `mut`. Then we create a mutable reference with `&mut s` where we call the `change` function, and update the function signature to accept a mutable reference with `some_string: &mut String`. This makes it very clear that the `change` function will mutate the value it borrows.
>  为了使用可变引用，我们需要:
>  - 原变量声明为 `mut`
>  - 使用 `&mut` 创建可变引用
>  - 函数签名使用 `&mut`

Mutable references have one big restriction: if you have a mutable reference to a value, you can have no other references to that value. This code that attempts to create two mutable references to `s` will fail:
>  可变引用存在限制: 如果对一个 value 现存一个可变引用，不允许创建其他引用

Filename: src/main.rs

```rust
let mut s = String::from("hello");

let r1 = &mut s;
let r2 = &mut s;

println!("{}, {}", r1, r2);
```

Here’s the error:

```
$ cargo run
   Compiling ownership v0.1.0 (file:///projects/ownership)
error[E0499]: cannot borrow `s` as mutable more than once at a time
 --> src/main.rs:5:14
  |
4 |     let r1 = &mut s;
  |              ------ first mutable borrow occurs here
5 |     let r2 = &mut s;
  |              ^^^^^^ second mutable borrow occurs here
6 |
7 |     println!("{}, {}", r1, r2);
  |                        -- first borrow later used here

For more information about this error, try `rustc --explain E0499`.
error: could not compile `ownership` (bin "ownership") due to 1 previous error
```

This error says that this code is invalid because we cannot borrow `s` as mutable more than once at a time. The first mutable borrow is in `r1` and must last until it’s used in the `println!`, but between the creation of that mutable reference and its usage, we tried to create another mutable reference in `r2` that borrows the same data as `r1`.
>  也就是不能以可变的形式多次借用一个变量

The restriction preventing multiple mutable references to the same data at the same time allows for mutation but in a very controlled fashion. It’s something that new Rustaceans struggle with because most languages let you mutate whenever you’d like. The benefit of having this restriction is that Rust can prevent data races at compile time. A _data race_ is similar to a race condition and happens when these three behaviors occur:

- Two or more pointers access the same data at the same time.
- At least one of the pointers is being used to write to the data.
- There’s no mechanism being used to synchronize access to the data.

>  这样的设计决策原因在于: 在编译时防止数据竞争
>  数据竞争在以下三种行为同时发生时出现:
>  - 多个指针同时访问相同的数据
>  - 至少一个指针会用于写入
>  - 没有同步机制协调对数据的访问

Data races cause undefined behavior and can be difficult to diagnose and fix when you’re trying to track them down at runtime; Rust prevents this problem by refusing to compile code with data races!
>  Rust 直接拒绝编译可能存在数据竞争的代码

As always, we can use curly brackets to create a new scope, allowing for multiple mutable references, just not _simultaneous_ ones:

```rust
let mut s = String::from("hello");

{
    let r1 = &mut s;
} // r1 goes out of scope here, so we can make a new reference with no problems.

let r2 = &mut s;
```

>  我们可以通过创建新的作用域来实现创建多个可变引用，本质是避免同时创建多个引用 (在上一个引用仍然被使用时创建新引用)

Rust enforces a similar rule for combining mutable and immutable references. This code results in an error:

```rust
let mut s = String::from("hello");

let r1 = &s; // no problem
let r2 = &s; // no problem
let r3 = &mut s; // BIG PROBLEM

println!("{}, {}, and {}", r1, r2, r3);
```

Here’s the error:

```
$ cargo run
   Compiling ownership v0.1.0 (file:///projects/ownership)
error[E0502]: cannot borrow `s` as mutable because it is also borrowed as immutable
 --> src/main.rs:6:14
  |
4 |     let r1 = &s; // no problem
  |              -- immutable borrow occurs here
5 |     let r2 = &s; // no problem
6 |     let r3 = &mut s; // BIG PROBLEM
  |              ^^^^^^ mutable borrow occurs here
7 |
8 |     println!("{}, {}, and {}", r1, r2, r3);
  |                                -- immutable borrow later used here

For more information about this error, try `rustc --explain E0502`.
error: could not compile `ownership` (bin "ownership") due to 1 previous error
```

Whew! We _also_ cannot have a mutable reference while we have an immutable one to the same value.

Users of an immutable reference don’t expect the value to suddenly change out from under them! However, multiple immutable references are allowed because no one who is just reading the data has the ability to affect anyone else’s reading of the data.

Note that a reference’s scope starts from where it is introduced and continues through the last time that reference is used. For instance, this code will compile because the last usage of the immutable references is in the `println!`, before the mutable reference is introduced:

```rust
let mut s = String::from("hello");

let r1 = &s; // no problem
let r2 = &s; // no problem
println!("{r1} and {r2}");
// Variables r1 and r2 will not be used after this point.

let r3 = &mut s; // no problem
println!("{r3}");
```

The scopes of the immutable references `r1` and `r2` end after the `println!` where they are last used, which is before the mutable reference `r3` is created. These scopes don’t overlap, so this code is allowed: the compiler can tell that the reference is no longer being used at a point before the end of the scope.

>  一个引用的作用域从它被创建时开始，直到它被最后一次使用结束

Even though borrowing errors may be frustrating at times, remember that it’s the Rust compiler pointing out a potential bug early (at compile time rather than at runtime) and showing you exactly where the problem is. Then you don’t have to track down why your data isn’t what you thought it was.

### Dangling References
In languages with pointers, it’s easy to erroneously create a _dangling pointer_—a pointer that references a location in memory that may have been given to someone else—by freeing some memory while preserving a pointer to that memory. In Rust, by contrast, the compiler guarantees that references will never be dangling references: if you have a reference to some data, the compiler will ensure that the data will not go out of scope before the reference to the data does.
>  Rust 编译器确保不会出现悬挂引用: 如果有对特定数据的引用，编译器确保引用离开作用域的时间一定在数据离开作用域的时间之前

Let’s try to create a dangling reference to see how Rust prevents them with a compile-time error:

Filename: src/main.rs

```rust
fn main() {
    let reference_to_nothing = dangle();
}

fn dangle() -> &String {
    let s = String::from("hello");

    &s
}
```

Here’s the error:

```
$ cargo run
   Compiling ownership v0.1.0 (file:///projects/ownership)
error[E0106]: missing lifetime specifier
 --> src/main.rs:5:16
  |
5 | fn dangle() -> &String {
  |                ^ expected named lifetime parameter
  |
  = help: this function's return type contains a borrowed value, but there is no value for it to be borrowed from
help: consider using the `'static` lifetime, but this is uncommon unless you're returning a borrowed value from a `const` or a `static`
  |
5 | fn dangle() -> &'static String {
  |                 +++++++
help: instead, you are more likely to want to return an owned value
  |
5 - fn dangle() -> &String {
5 + fn dangle() -> String {
  |

error[E0515]: cannot return reference to local variable `s`
 --> src/main.rs:8:5
  |
8 |     &s
  |     ^^ returns a reference to data owned by the current function

Some errors have detailed explanations: E0106, E0515.
For more information about an error, try `rustc --explain E0106`.
error: could not compile `ownership` (bin "ownership") due to 2 previous errors
```

This error message refers to a feature we haven’t covered yet: lifetimes. We’ll discuss lifetimes in detail in Chapter 10. But, if you disregard the parts about lifetimes, the message does contain the key to why this code is a problem:

```
this function's return type contains a borrowed value, but there is no value for it to be borrowed from
```

Let’s take a closer look at exactly what’s happening at each stage of our `dangle` code:

Filename: src/main.rs

```rust
fn dangle() -> &String { // dangle returns a reference to a String

    let s = String::from("hello"); // s is a new String

    &s // we return a reference to the String, s
} // Here, s goes out of scope, and is dropped, so its memory goes away.
  // Danger!
```

Because `s` is created inside `dangle`, when the code of `dangle` is finished, `s` will be deallocated. But we tried to return a reference to it. That means this reference would be pointing to an invalid `String`. That’s no good! Rust won’t let us do this.

The solution here is to return the `String` directly:

```rust
fn no_dangle() -> String {
    let s = String::from("hello");

    s
}
```

This works without any problems. Ownership is moved out, and nothing is deallocated.

>  不允许在函数内创建 value，然后返回其引用
>  只能直接返回 value，转移所有权，不会 deallocate memory

### The Rules of References
Let’s recap what we’ve discussed about references:

- At any given time, you can have _either_ one mutable reference _or_ any number of immutable references.
- References must always be valid.

>  引用的规则:
>  - 任意时间内，要么有一个可变引用，要么有多个不可变引用
>  - 引用必须总是有效

Next, we’ll look at a different kind of reference: slices.

## 4.3 The Slice Type
_Slices_ let you reference a contiguous sequence of elements in a [collection](https://doc.rust-lang.org/stable/book/ch08-00-common-collections.html) rather than the whole collection. A slice is a kind of reference, so it does not have ownership.
>  slide 也是一类 reference，故并没有所有权

Here’s a small programming problem: write a function that takes a string of words separated by spaces and returns the first word it finds in that string. If the function doesn’t find a space in the string, the whole string must be one word, so the entire string should be returned.

Let’s work through how we’d write the signature of this function without using slices, to understand the problem that slices will solve:

```rust
fn first_word(s: &String) -> ?
```

The `first_word` function has a `&String` as a parameter. We don’t need ownership, so this is fine. (In idiomatic Rust, functions do not take ownership of their arguments unless they need to, and the reasons for that will become clear as we keep going!) But what should we return? We don’t really have a way to talk about part of a string. However, we could return the index of the end of the word, indicated by a space. Let’s try that, as shown in Listing 4-7.
>  Rust 中，函数通常不会获取参数的所有权，除非确实需要

Filename: src/main.rs

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

[Listing 4-7](https://doc.rust-lang.org/stable/book/ch04-03-slices.html#listing-4-7): The `first_word` function that returns a byte index value into the `String` parameter

Because we need to go through the `String` element by element and check whether a value is a space, we’ll convert our `String` to an array of bytes using the `as_bytes` method.
>  我们用 `as_bytes` 方法将 `String` 转化为 array of bytes

```rust
let bytes = s.as_bytes();
```

Next, we create an iterator over the array of bytes using the `iter` method:
>  我们用 `iter` 方法创建迭代器

```rust
for (i, &item) in bytes.iter().enumerate() {
```

We’ll discuss iterators in more detail in [Chapter 13](https://doc.rust-lang.org/stable/book/ch13-02-iterators.html). For now, know that `iter` is a method that returns each element in a collection and that `enumerate` wraps the result of `iter` and returns each element as part of a tuple instead. The first element of the tuple returned from `enumerate` is the index, and the second element is a reference to the element. This is a bit more convenient than calculating the index ourselves.
>  `enumerate` 包装了 `iter` 的结果，返回一个 tuple，包含了索引和对元素的引用

Because the `enumerate` method returns a tuple, we can use patterns to destructure that tuple. We’ll be discussing patterns more in [Chapter 6](https://doc.rust-lang.org/stable/book/ch06-02-match.html#patterns-that-bind-to-values). In the `for` loop, we specify a pattern that has `i` for the index in the tuple and `&item` for the single byte in the tuple. Because we get a reference to the element from `.iter().enumerate()`, we use `&` in the pattern.

Inside the `for` loop, we search for the byte that represents the space by using the byte literal syntax. If we find a space, we return the position. Otherwise, we return the length of the string by using `s.len()`.
>  我们使用 byte 字面量语法 `b' '` 表示空格

```rust
    if item == b' ' {
        return i;
    }
}

s.len()
```

We now have a way to find out the index of the end of the first word in the string, but there’s a problem. We’re returning a `usize` on its own, but it’s only a meaningful number in the context of the `&String`. In other words, because it’s a separate value from the `String`, there’s no guarantee that it will still be valid in the future. Consider the program in Listing 4-8 that uses the `first_word` function from Listing 4-7.

Filename: src/main.rs

```rust
fn main() {
    let mut s = String::from("hello world");

    let word = first_word(&s); // word will get the value 5

    s.clear(); // this empties the String, making it equal to ""

    // `word` still has the value `5` here, but `s` no longer has any content
    // that we could meaningfully use with the value `5`, so `word` is now
    // totally invalid!
}
```

[Listing 4-8](https://doc.rust-lang.org/stable/book/ch04-03-slices.html#listing-4-8): Storing the result from calling the `first_word` function and then changing the `String` contents

This program compiles without any errors and would also do so if we used `word` after calling `s.clear()`. Because `word` isn’t connected to the state of `s` at all, `word` still contains the value `5`. We could use that value `5` with the variable `s` to try to extract the first word out, but this would be a bug because the contents of `s` have changed since we saved `5` in `word`.
>  但意识到关于 `String` 的索引是需要 `String` 本身的内容作为上下文才是有意义的

Having to worry about the index in `word` getting out of sync with the data in `s` is tedious and error prone! Managing these indices is even more brittle if we write a `second_word` function. Its signature would have to look like this:

```rust
fn second_word(s: &String) -> (usize, usize) {
```

Now we’re tracking a starting _and_ an ending index, and we have even more values that were calculated from data in a particular state but aren’t tied to that state at all. We have three unrelated variables floating around that need to be kept in sync.

Luckily, Rust has a solution to this problem: string slices.

### String Slices
A _string slice_ is a reference to part of a `String`, and it looks like this:

```rust
let s = String::from("hello world");

let hello = &s[0..5];
let world = &s[6..11];
```

Rather than a reference to the entire `String`, `hello` is a reference to a portion of the `String`, specified in the extra `[0..5]` bit. We create slices using a range within brackets by specifying `[starting_index..ending_index]`, where _`starting_index`_ is the first position in the slice and _`ending_index`_ is one more than the last position in the slice. Internally, the slice data structure stores the starting position and the length of the slice, which corresponds to _`ending_index`_ minus _`starting_index`_. So, in the case of `let world = &s[6..11];`, `world` would be a slice that contains a pointer to the byte at index 6 of `s` with a length value of `5`.

>  string slice 是不是对整个 `String` 的引用，而是对 `String` 的一部分的引用
>  我们使用 range 创建 slice: `[string_index..ending_index]`
>  slice 数据结构实际上存储了 slice 的起始位置和长度

Figure 4-7 shows this in a diagram.

![Three tables: a table representing the stack data of s, which points to the byte at index 0 in a table of the string data "hello world" on the heap. The third table rep-resents the stack data of the slice world, which has a length value of 5 and points to byte 6 of the heap data table.](https://doc.rust-lang.org/stable/book/img/trpl04-07.svg)

Figure 4-7: String slice referring to part of a `String`

With Rust’s `..` range syntax, if you want to start at index 0, you can drop the value before the two periods. In other words, these are equal:
>  如果没有指定起始，range 默认从 0 开始

```rust
let s = String::from("hello");

let slice = &s[0..2];
let slice = &s[..2];
```

By the same token, if your slice includes the last byte of the `String`, you can drop the trailing number. That means these are equal:
>  如果没有指定结尾，range 默认取到 string 的 length

```rust
let s = String::from("hello");

let len = s.len();

let slice = &s[3..len];
let slice = &s[3..];
```

You can also drop both values to take a slice of the entire string. So these are equal:
>  起始和结束都没有，默认取整个 string

```rust
let s = String::from("hello");

let len = s.len();

let slice = &s[0..len];
let slice = &s[..];
```

Note: String slice range indices must occur at valid UTF-8 character boundaries. If you attempt to create a string slice in the middle of a multibyte character, your program will exit with an error. For the purposes of introducing string slices, we are assuming ASCII only in this section; a more thorough discussion of UTF-8 handling is in the [“Storing UTF-8 Encoded Text with Strings”](https://doc.rust-lang.org/stable/book/ch08-02-strings.html#storing-utf-8-encoded-text-with-strings) section of Chapter 8.
>  注意，string slice range 索引必须在有效的 UTF-8 字符边界

With all this information in mind, let’s rewrite `first_word` to return a slice. The type that signifies “string slice” is written as `&str`:
>  string slice 的类型记作 `&str`

Filename: src/main.rs

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

We get the index for the end of the word the same way we did in Listing 4-7, by looking for the first occurrence of a space. When we find a space, we return a string slice using the start of the string and the index of the space as the starting and ending indices.

Now when we call `first_word`, we get back a single value that is tied to the underlying data. The value is made up of a reference to the starting point of the slice and the number of elements in the slice.

Returning a slice would also work for a `second_word` function:

```rust
fn second_word(s: &String) -> &str {
```

We now have a straightforward API that’s much harder to mess up because the compiler will ensure the references into the `String` remain valid. Remember the bug in the program in Listing 4-8, when we got the index to the end of the first word but then cleared the string so our index was invalid? That code was logically incorrect but didn’t show any immediate errors. The problems would show up later if we kept trying to use the first word index with an emptied string. Slices make this bug impossible and let us know we have a problem with our code much sooner. Using the slice version of `first_word` will throw a compile-time error:

Filename: src/main.rs

```rust
fn main() {
    let mut s = String::from("hello world");

    let word = first_word(&s);

    s.clear(); // error!

    println!("the first word is: {word}");
}
```

Here’s the compiler error:

```
$ cargo run
   Compiling ownership v0.1.0 (file:///projects/ownership)
error[E0502]: cannot borrow `s` as mutable because it is also borrowed as immutable
  --> src/main.rs:18:5
   |
16 |     let word = first_word(&s);
   |                           -- immutable borrow occurs here
17 |
18 |     s.clear(); // error!
   |     ^^^^^^^^^ mutable borrow occurs here
19 |
20 |     println!("the first word is: {word}");
   |                                  ------ immutable borrow later used here

For more information about this error, try `rustc --explain E0502`.
error: could not compile `ownership` (bin "ownership") due to 1 previous error
```

Recall from the borrowing rules that if we have an immutable reference to something, we cannot also take a mutable reference. Because `clear` needs to truncate the `String`, it needs to get a mutable reference. The `println!` after the call to `clear` uses the reference in `word`, so the immutable reference must still be active at that point. Rust disallows the mutable reference in `clear` and the immutable reference in `word` from existing at the same time, and compilation fails. Not only has Rust made our API easier to use, but it has also eliminated an entire class of errors at compile time!
>  `clear` 需要截断 `String`，故需要获取可变引用
>  `println!` 使用了 `word` 中的引用，故不可变引用的作用域一直到这一个点
>  Rust 不允许可变引用和不可变引用同时存在，故编译失败

#### String Literals as Slices
Recall that we talked about string literals being stored inside the binary. Now that we know about slices, we can properly understand string literals:

```rust
let s = "Hello, world!";
```

The type of `s` here is `&str`: it’s a slice pointing to that specific point of the binary. This is also why string literals are immutable; `&str` is an immutable reference.

>  string 字面值的类型也是 `&str`，即它是一个指向二进制文件中特定位置的 slice，是不可变引用

#### String Slices as Parameters
Knowing that you can take slices of literals and `String` values leads us to one more improvement on `first_word`, and that’s its signature:

```rust
fn first_word(s: &String) -> &str {
```

A more experienced Rustacean would write the signature shown in Listing 4-9 instead because it allows us to use the same function on both `&String` values and `&str` values.

```rust
fn first_word(s: &str) -> &str {
```

[Listing 4-9](https://doc.rust-lang.org/stable/book/ch04-03-slices.html#listing-4-9): Improving the `first_word` function by using a string slice for the type of the `s` parameter

If we have a string slice, we can pass that directly. If we have a `String`, we can pass a slice of the `String` or a reference to the `String`. This flexibility takes advantage of _deref coercions_, a feature we will cover in the [“Implicit Deref Coercions with Functions and Methods”](https://doc.rust-lang.org/stable/book/ch15-02-deref.html#implicit-deref-coercions-with-functions-and-methods) section of Chapter 15.
>  函数接收 `&str` 的参数时，它既可以接收 `&str`，也可以接收 `&String`
>  故将参数记作 `&str` 会更加灵活

Defining a function to take a string slice instead of a reference to a `String` makes our API more general and useful without losing any functionality:

Filename: src/main.rs

```rust
fn main() {
    let my_string = String::from("hello world");

    // `first_word` works on slices of `String`s, whether partial or whole.
    let word = first_word(&my_string[0..6]);
    let word = first_word(&my_string[..]);
    // `first_word` also works on references to `String`s, which are equivalent
    // to whole slices of `String`s.
    let word = first_word(&my_string);

    let my_string_literal = "hello world";

    // `first_word` works on slices of string literals, whether partial or
    // whole.
    let word = first_word(&my_string_literal[0..6]);
    let word = first_word(&my_string_literal[..]);

    // Because string literals *are* string slices already,
    // this works too, without the slice syntax!
    let word = first_word(my_string_literal);
}
```

### Other Slices
String slices, as you might imagine, are specific to strings. But there’s a more general slice type too. Consider this array:

```rust
let a = [1, 2, 3, 4, 5]; 
```

Just as we might want to refer to part of a string, we might want to refer to part of an array. We’d do so like this:

```rust
let a = [1, 2, 3, 4, 5];  
let slice = &a[1..3];  
assert_eq!(slice, &[2, 3]); 
```

This slice has the type `&[i32]`. It works the same way as string slices do, by storing a reference to the first element and a length. You’ll use this kind of slice for all sorts of other collections. We’ll discuss these collections in detail when we talk about vectors in Chapter 8.
>  array 也可以有 slice，上例中 slice 的类型为 `&[i32]`，其工作方式和 string slice 相同

## Summary
The concepts of ownership, borrowing, and slices ensure memory safety in Rust programs at compile time. The Rust language gives you control over your memory usage in the same way as other systems programming languages, but having the owner of data automatically clean up that data when the owner goes out of scope means you don’t have to write and debug extra code to get this control.

Ownership affects how lots of other parts of Rust work, so we’ll talk about these concepts further throughout the rest of the book. Let’s move on to Chapter 5 and look at grouping pieces of data together in a `struct`.