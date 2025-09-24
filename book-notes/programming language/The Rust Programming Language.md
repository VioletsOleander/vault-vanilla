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
>  整数除法会向零方向截断，得到最近的整数

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
>  可以使用 `[dtype; ele_num]` 作为数组的类型注释，例如:

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
>  Rust 对于函数定义的位置没有要求，只要定义在调用者可以看到的作用域内即可

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
>  语句不会返回任何值，或者说等价于返回单元类型 `()`

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
>  - 变量离开作用域之前，保持有效

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

# 5 Using Structs to Structure Related Data
A _struct_, or _structure_, is a custom data type that lets you package together and name multiple related values that make up a meaningful group. If you’re familiar with an object-oriented language, a _struct_ is like an object’s data attributes. In this chapter, we’ll compare and contrast tuples with structs to build on what you already know and demonstrate when structs are a better way to group data.
>  struct 是自定义数据类型，用于将多个相关的值组合在一起

We’ll demonstrate how to define and instantiate structs. We’ll discuss how to define associated functions, especially the kind of associated functions called _methods_, to specify behavior associated with a struct type. Structs and enums (discussed in Chapter 6) are the building blocks for creating new types in your program’s domain to take full advantage of Rust’s compile-time type checking.

## 5.1 Defining and Instantiating Structs
Structs are similar to tuples, discussed in [“The Tuple Type”](https://doc.rust-lang.org/stable/book/ch03-02-data-types.html#the-tuple-type) section, in that both hold multiple related values. Like tuples, the pieces of a struct can be different types. Unlike with tuples, in a struct you’ll name each piece of data so it’s clear what the values mean. Adding these names means that structs are more flexible than tuples: you don’t have to rely on the order of the data to specify or access the values of an instance.
>  struct 类似于 tuple，和 tuple 一样，struct 内可以有多个类型
>  和 tuple 不同的是，struct 内可以为每个 field 命名，进而可以不依赖顺序来访问 struct 内的 value

To define a struct, we enter the keyword `struct` and name the entire struct. A struct’s name should describe the significance of the pieces of data being grouped together. Then, inside curly brackets, we define the names and types of the pieces of data, which we call _fields_. For example, Listing 5-1 shows a struct that stores information about a user account.
>  struct 通过关键字 `struct` 定义，形式为 `struct <struct-name>`
>  struct 内需要定义名称和类型，我们称为 fields

Filename: src/main.rs

```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}
```

[Listing 5-1](https://doc.rust-lang.org/stable/book/ch05-01-defining-structs.html#listing-5-1): A `User` struct definition

To use a struct after we’ve defined it, we create an _instance_ of that struct by specifying concrete values for each of the fields. We create an instance by stating the name of the struct and then add curly brackets containing _ `key: value` _ pairs, where the keys are the names of the fields and the values are the data we want to store in those fields. We don’t have to specify the fields in the same order in which we declared them in the struct. In other words, the struct definition is like a general template for the type, and instances fill in that template with particular data to create values of the type. For example, we can declare a particular user as shown in Listing 5-2.
>  我们通过为每个 fields 指定具体的值来为 struct 创建 instance
>  instance 创建的格式为 `<struct-name> {name: value, ...}`

Filename: src/main.rs

```rust
fn main() {
    let user1 = User {
        active: true,
        username: String::from("someusername123"),
        email: String::from("someone@example.com"),
        sign_in_count: 1,
    };
}
```

[Listing 5-2](https://doc.rust-lang.org/stable/book/ch05-01-defining-structs.html#listing-5-2): Creating an instance of the `User` struct

To get a specific value from a struct, we use dot notation. For example, to access this user’s email address, we use `user1.email`. If the instance is mutable, we can change a value by using the dot notation and assigning into a particular field. Listing 5-3 shows how to change the value in the `email` field of a mutable `User` instance.
>  访问 fields 的格式为 `<struct-name>.<field-name>`
>  如果该 instance 是可变的，value 可以重新赋值

Filename: src/main.rs

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

[Listing 5-3](https://doc.rust-lang.org/stable/book/ch05-01-defining-structs.html#listing-5-3): Changing the value in the `email` field of a `User` instance

Note that the entire instance must be mutable; Rust doesn’t allow us to mark only certain fields as mutable. As with any expression, we can construct a new instance of the struct as the last expression in the function body to implicitly return that new instance.
>  Rust 不允许将特定的 field 标记为可变
>  我们可以在函数体的最后一个表达式构造 struct 的 instance，作为返回值

Listing 5-4 shows a `build_user` function that returns a `User` instance with the given email and username. The `active` field gets the value of `true`, and the `sign_in_count` gets a value of `1`.

Filename: src/main.rs

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

[Listing 5-4](https://doc.rust-lang.org/stable/book/ch05-01-defining-structs.html#listing-5-4): A `build_user` function that takes an email and username and returns a `User` instance

It makes sense to name the function parameters with the same name as the struct fields, but having to repeat the `email` and `username` field names and variables is a bit tedious. If the struct had more fields, repeating each name would get even more annoying. Luckily, there’s a convenient shorthand!

### Using the Field Init Shorthand
Because the parameter names and the struct field names are exactly the same in Listing 5-4, we can use the _field init shorthand_ syntax to rewrite `build_user` so it behaves exactly the same but doesn’t have the repetition of `username` and `email`, as shown in Listing 5-5.
>  如果变量名和 field name 完全相同，我们可以使用 field init shorthand 语法进行简写，这使得我们不需要重复 field name

Filename: src/main.rs

```rust
fn build_user(email: String, username: String) -> User {
    User {
        active: true,
        username,
        email,
        sign_in_count: 1,
    }
}
```

[Listing 5-5](https://doc.rust-lang.org/stable/book/ch05-01-defining-structs.html#listing-5-5): A `build_user` function that uses field init shorthand because the `username` and `email` parameters have the same name as struct fields

Here, we’re creating a new instance of the `User` struct, which has a field named `email`. We want to set the `email` field’s value to the value in the `email` parameter of the `build_user` function. Because the `email` field and the `email` parameter have the same name, we only need to write `email` rather than `email: email`.

### Creating Instances from Other Instances with Struct Update Syntax
It’s often useful to create a new instance of a struct that includes most of the values from another instance, but changes some. You can do this using _struct update syntax_.
>  如果要通过一个 instance 创建另一个 instance，可以使用 struct update syntax

First, in Listing 5-6 we show how to create a new `User` instance in `user2` regularly, without the update syntax. We set a new value for `email` but otherwise use the same values from `user1` that we created in Listing 5-2.

Filename: src/main.rs

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

[Listing 5-6](https://doc.rust-lang.org/stable/book/ch05-01-defining-structs.html#listing-5-6): Creating a new `User` instance using all but one of the values from `user1`

Using struct update syntax, we can achieve the same effect with less code, as shown in Listing 5-7. The syntax `..` specifies that the remaining fields not explicitly set should have the same value as the fields in the given instance.
>  在 struct update syntax 中，我们指定需要修改的 fields，然后通过 `..` 指定剩余的 fields 都具有和给定实例相同的 value

Filename: src/main.rs

```rust
fn main() {
    // --snip--

    let user2 = User {
        email: String::from("another@example.com"),
        ..user1
    };
}
```

[Listing 5-7](https://doc.rust-lang.org/stable/book/ch05-01-defining-structs.html#listing-5-7): Using struct update syntax to set a new `email` value for a `User` instance but to use the rest of the values from `user1`

The code in Listing 5-7 also creates an instance in `user2` that has a different value for `email` but has the same values for the `username`, `active`, and `sign_in_count` fields from `user1`. The `..user1` must come last to specify that any remaining fields should get their values from the corresponding fields in `user1`, but we can choose to specify values for as many fields as we want in any order, regardless of the order of the fields in the struct’s definition.
>  `..<instance-name>` 必须在最后指定

Note that the struct update syntax uses `=` like an assignment; this is because it moves the data, just as we saw in the [“Variables and Data Interacting with Move”](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html#variables-and-data-interacting-with-move) section. In this example, we can no longer use `user1` after creating `user2` because the `String` in the `username` field of `user1` was moved into `user2`. If we had given `user2` new `String` values for both `email` and `username`, and thus only used the `active` and `sign_in_count` values from `user1`, then `user1` would still be valid after creating `user2`. Both `active` and `sign_in_count` are types that implement the `Copy` trait, so the behavior we discussed in the [“Stack-Only Data: Copy”](https://doc.rust-lang.org/stable/book/ch04-01-what-is-ownership.html#stack-only-data-copy) section would apply. We can still use `user1.email` in this example, because its value was _not_ moved out.
>  注意，struct update syntax 类似赋值一样，使用 `=`，这是因为 struct update syntax 移动了数据
>  在上例中，我们不能在创建了 `user2` 之后再使用 `user1`，因为 `user1` 中的 `String` 都被移动了
>  而如果我们在 struct update syntax 没有移动涉及 `String` 的值，则仍然可以在创建了 ` user2 ` 之后使用 `user1`
>  这实际上是因为其他的类型都实现了 `Copy` trait，故它们的数据会被拷贝
>  注意没有被 move out 的数据仍然可以被访问，例如 `user1.email`

### Using Tuple Structs Without Named Fields to Create Different Types
Rust also supports structs that look similar to tuples, called _tuple structs_. Tuple structs have the added meaning the struct name provides but don’t have names associated with their fields; rather, they just have the types of the fields. Tuple structs are useful when you want to give the whole tuple a name and make the tuple a different type from other tuples, and when naming each field as in a regular struct would be verbose or redundant.
>  Rust 也支持看起来像 tuple 的 struct，称为 tuple structs
>  tuple structs 的字段没有各自的名称，只有字段的类型

To define a tuple struct, start with the `struct` keyword and the struct name followed by the types in the tuple. For example, here we define and use two tuple structs named `Color` and `Point`:
>  tuple struct 的定义语法为 `struct <struct-name> (type1, type2, ...)`

Filename: src/main.rs

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
}
```

Note that the `black` and `origin` values are different types because they’re instances of different tuple structs. Each struct you define is its own type, even though the fields within the struct might have the same types. For example, a function that takes a parameter of type `Color` cannot take a `Point` as an argument, even though both types are made up of three `i32` values. Otherwise, tuple struct instances are similar to tuples in that you can destructure them into their individual pieces, and you can use a `.` followed by the index to access an individual value. Unlike tuples, tuple structs require you to name the type of the struct when you destructure them. For example, we would write `let Point(x, y, z) = point`.
>  注意每个自定义的 struct 都属于自己的类型，和其 fields 的类型无关
>  tuple struct instance 可以向 tuple 一样被 destructure，且通过 `.<index>` 访问 fields
>  注意 tuple struct 在 destructure 时，需要提供结构目标的 struct 的名字，例如 `let Point(x, y, z) = point` 而不是 `let x, y, z = point`

### Unit-Like Structs Without Any Fields
You can also define structs that don’t have any fields! These are called _unit-like structs_ because they behave similarly to `()`, the unit type that we mentioned in [“The Tuple Type”](https://doc.rust-lang.org/stable/book/ch03-02-data-types.html#the-tuple-type) section. Unit-like structs can be useful when you need to implement a trait on some type but don’t have any data that you want to store in the type itself. We’ll discuss traits in Chapter 10. Here’s an example of declaring and instantiating a unit struct named `AlwaysEqual`:
>  没有任何 fields 的 struct 称为 unit-like struct，它们的行为和 unit type `()` 类似
>  unit-like struct 在我们想要在某个类型上实现一个 trait，但并不想存储任何数据在该类型上时十分有用

Filename: src/main.rs

```rust
struct AlwaysEqual;

fn main() {
    let subject = AlwaysEqual;
}
```

To define `AlwaysEqual`, we use the `struct` keyword, the name we want, and then a semicolon. No need for curly brackets or parentheses! Then we can get an instance of `AlwaysEqual` in the `subject` variable in a similar way: using the name we defined, without any curly brackets or parentheses. 
>  unit-like struct 的定义为 `struct <struct-name>;`
>  并且直接使用 `<struct-name>` 就可以得到其 instance

Imagine that later we’ll implement behavior for this type such that every instance of `AlwaysEqual` is always equal to every instance of any other type, perhaps to have a known result for testing purposes. We wouldn’t need any data to implement that behavior! You’ll see in Chapter 10 how to define traits and implement them on any type, including unit-like structs.

### Ownership of Struct Data
In the `User` struct definition in Listing 5-1, we used the owned `String` type rather than the `&str` string slice type. This is a deliberate choice because we want each instance of this struct to own all of its data and for that data to be valid for as long as the entire struct is valid.
>  我们通常希望 struct 的每个 instance 都拥有其所有的数据，并且这些数据在 instance 实例有效下都有效

It’s also possible for structs to store references to data owned by something else, but to do so requires the use of _lifetimes_, a Rust feature that we’ll discuss in Chapter 10. Lifetimes ensure that the data referenced by a struct is valid for as long as the struct is. Let’s say you try to store a reference in a struct without specifying lifetimes, like the following; this won’t work:
>  struct 可以存储引用，但要做到这一点需要使用生命周期
>  生命周期确保结构体引用的数据在结构体有效期间一直有效
>  在结构体中存储引用而不指定声明周期会无法编译

Filename: src/main.rs

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

The compiler will complain that it needs lifetime specifiers:

```
$ cargo run
   Compiling structs v0.1.0 (file:///projects/structs)
error[E0106]: missing lifetime specifier
 --> src/main.rs:3:15
  |
3 |     username: &str,
  |               ^ expected named lifetime parameter
  |
help: consider introducing a named lifetime parameter
  |
1 ~ struct User<'a> {
2 |     active: bool,
3 ~     username: &'a str,
  |

error[E0106]: missing lifetime specifier
 --> src/main.rs:4:12
  |
4 |     email: &str,
  |            ^ expected named lifetime parameter
  |
help: consider introducing a named lifetime parameter
  |
1 ~ struct User<'a> {
2 |     active: bool,
3 |     username: &str,
4 ~     email: &'a str,
  |

For more information about this error, try `rustc --explain E0106`.
error: could not compile `structs` (bin "structs") due to 2 previous errors
```

In Chapter 10, we’ll discuss how to fix these errors so you can store references in structs, but for now, we’ll fix errors like these using owned types like `String` instead of references like `&str`.

## 5.2 An Example Program Using Structs
To understand when we might want to use structs, let’s write a program that calculates the area of a rectangle. We’ll start by using single variables, and then refactor the program until we’re using structs instead.

Let’s make a new binary project with Cargo called _rectangles_ that will take the width and height of a rectangle specified in pixels and calculate the area of the rectangle. Listing 5-8 shows a short program with one way of doing exactly that in our project’s _src/main.rs_.

Filename: src/main.rs

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

[Listing 5-8](https://doc.rust-lang.org/stable/book/ch05-02-example-structs.html#listing-5-8): Calculating the area of a rectangle specified by separate width and height variables

Now, run this program using `cargo run`:

```
$ cargo run
   Compiling rectangles v0.1.0 (file:///projects/rectangles)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.42s
     Running `target/debug/rectangles`
The area of the rectangle is 1500 square pixels.
```

This code succeeds in figuring out the area of the rectangle by calling the `area` function with each dimension, but we can do more to make this code clear and readable.

The issue with this code is evident in the signature of `area`:

```rust
fn area(width: u32, height: u32) -> u32 {
```

The `area` function is supposed to calculate the area of one rectangle, but the function we wrote has two parameters, and it’s not clear anywhere in our program that the parameters are related. It would be more readable and more manageable to group width and height together. We’ve already discussed one way we might do that in [“The Tuple Type”](https://doc.rust-lang.org/stable/book/ch03-02-data-types.html#the-tuple-type) section of Chapter 3: by using tuples.
>  `area` 函数的功能是计算长方形的面积，但是通过其参数实际上并没有传达这些信息
>  比如，我们传递的两个参数是分离的，没有表现出它们互相关联

### Refactoring with Tuples
Listing 5-9 shows another version of our program that uses tuples.

Filename: src/main.rs

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

[Listing 5-9](https://doc.rust-lang.org/stable/book/ch05-02-example-structs.html#listing-5-9): Specifying the width and height of the rectangle with a tuple

In one way, this program is better. Tuples let us add a bit of structure, and we’re now passing just one argument. But in another way, this version is less clear: tuples don’t name their elements, so we have to index into the parts of the tuple, making our calculation less obvious.
>  简单改进之后，我们可以传递 tuple
>  但此时的问题是 tuple 的 field 没有名字，我们需要使用索引访问

Mixing up the width and height wouldn’t matter for the area calculation, but if we want to draw the rectangle on the screen, it would matter! We would have to keep in mind that `width` is the tuple index `0` and `height` is the tuple index `1`. This would be even harder for someone else to figure out and keep in mind if they were to use our code. Because we haven’t conveyed the meaning of our data in our code, it’s now easier to introduce errors.

### Refactoring with Structs: Adding More Meaning
We use structs to add meaning by labeling the data. We can transform the tuple we’re using into a struct with a name for the whole as well as names for the parts, as shown in Listing 5-10.
>  进一步改进后，我们使用 struct

Filename: src/main.rs

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

[Listing 5-10](https://doc.rust-lang.org/stable/book/ch05-02-example-structs.html#listing-5-10): Defining a `Rectangle` struct

Here, we’ve defined a struct and named it `Rectangle`. Inside the curly brackets, we defined the fields as `width` and `height`, both of which have type `u32`. Then, in `main`, we created a particular instance of `Rectangle` that has a width of `30` and a height of `50`.

Our `area` function is now defined with one parameter, which we’ve named `rectangle`, whose type is an immutable borrow of a struct `Rectangle` instance. As mentioned in Chapter 4, we want to borrow the struct rather than take ownership of it. This way, `main` retains its ownership and can continue using `rect1`, which is the reason we use the `&` in the function signature and where we call the function.
>  函数的参数是对 `Rectangle` 的不可变引用

The `area` function accesses the `width` and `height` fields of the `Rectangle` instance (note that accessing fields of a borrowed struct instance does not move the field values, which is why you often see borrows of structs). Our function signature for `area` now says exactly what we mean: calculate the area of `Rectangle`, using its `width` and `height` fields. This conveys that the width and height are related to each other, and it gives descriptive names to the values rather than using the tuple index values of `0` and `1`. This is a win for clarity.
>  此时函数的功能从参数和函数体来看就更加清晰

### Adding Useful Functionality with Derived Traits
It’d be useful to be able to print an instance of `Rectangle` while we’re debugging our program and see the values for all its fields. Listing 5-11 tries using the [`println!` macro](https://doc.rust-lang.org/stable/std/macro.println.html) as we have used in previous chapters. This won’t work, however.

Filename: src/main.rs

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

    println!("rect1 is {rect1}");
}
```

[Listing 5-11](https://doc.rust-lang.org/stable/book/ch05-02-example-structs.html#listing-5-11): Attempting to print a `Rectangle` instance

>  直接使用 `println!` 试图打印 `Rectangle` 类型会出现编译错误

When we compile this code, we get an error with this core message:

```
error[E0277]: `Rectangle` doesn't implement `std::fmt::Display`
```

The `println!` macro can do many kinds of formatting, and by default, the curly brackets tell `println!` to use formatting known as `Display`: output intended for direct end user consumption. The primitive types we’ve seen so far implement `Display` by default because there’s only one way you’d want to show a `1` or any other primitive type to a user. But with structs, the way `println!` should format the output is less clear because there are more display possibilities: Do you want commas or not? Do you want to print the curly brackets? Should all the fields be shown? Due to this ambiguity, Rust doesn’t try to guess what we want, and structs don’t have a provided implementation of `Display` to use with `println!` and the `{}` placeholder.
>  `println!` 宏可以做许多格式化工作，且默认情况下 `{}` 告诉它使用一种称为 `Display` 的格式: 这种格式是为直接面向最终用户的输出设计的
>  基本类型默认都实现了 `Display`，但对于结构体来说，我们需要根据自己的理解，实现它面型最终用户的输出

If we continue reading the errors, we’ll find this helpful note:

```
= help: the trait `std::fmt::Display` is not implemented for `Rectangle`
= note: in format strings you may be able to use `{:?}` (or {:#?} for pretty-print) instead
```

Let’s try it! The `println!` macro call will now look like `println!("rect1 is {rect1:?}");`. Putting the specifier `:?` inside the curly brackets tells `println!` we want to use an output format called `Debug`. The `Debug` trait enables us to print our struct in a way that is useful for developers so we can see its value while we’re debugging our code.
>  如果我们使用 `{rect1:?}`，其中的 `:?` 告诉 `println!` 我们想要使用格式为 `Debug` 的输出格式

Compile the code with this change. Drat! We still get an error:

```
error[E0277]: ` Rectangle ` doesn't implement ` Debug `
```

But again, the compiler gives us a helpful note:

```
= help: the trait `Debug` is not implemented for `Rectangle`
= note: add `#[derive(Debug)]` to `Rectangle` or manually `impl Debug for Rectangle`
```

Rust _does_ include functionality to print out debugging information, but we have to explicitly opt in to make that functionality available for our struct. To do that, we add the outer attribute `#[derive(Debug)]` just before the struct definition, as shown in Listing 5-12.
>  Rust 本身提供了打印 debug 信息的功能，但我们需要显式让该功能对于我们的结构体可用
>  为此，我们需要在结构体定义之前添加属性 `#[derive(Debug)]`

Filename: src/main.rs

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

    println!("rect1 is {rect1:?}");
}
```

[Listing 5-12](https://doc.rust-lang.org/stable/book/ch05-02-example-structs.html#listing-5-12): Adding the attribute to derive the `Debug` trait and printing the `Rectangle` instance using debug formatting

Now when we run the program, we won’t get any errors, and we’ll see the following output:

```
$ cargo run
   Compiling rectangles v0.1.0 (file:///projects/rectangles)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.48s
     Running `target/debug/rectangles`
rect1 is Rectangle { width: 30, height: 50 }
```

Nice! It’s not the prettiest output, but it shows the values of all the fields for this instance, which would definitely help during debugging. When we have larger structs, it’s useful to have output that’s a bit easier to read; in those cases, we can use `{:#?}` instead of `{:?}` in the `println!` string. In this example, using the `{:#?}` style will output the following:
>  如果使用 `{:#?}` 而不是 `{:?}`，可以获得更可读的方式

```
$ cargo run
   Compiling rectangles v0.1.0 (file:///projects/rectangles)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.48s
     Running `target/debug/rectangles`
rect1 is Rectangle {
    width: 30,
    height: 50,
}
```

Another way to print out a value using the `Debug` format is to use the [`dbg!` macro](https://doc.rust-lang.org/stable/std/macro.dbg.html), which takes ownership of an expression (as opposed to `println!`, which takes a reference), prints the file and line number of where that `dbg!` macro call occurs in your code along with the resultant value of that expression, and returns ownership of the value.
>  使用 `Debug` 格式打印 value 的另一种方式是使用 `dbg!` macro，它会获得表达式的所有权 (`println!` 则只获取引用)，打印出该 dbg macro 在代码中出现的文件和行号，以及该表达式的结果值，然后返回该值的所有权

Note: Calling the `dbg!` macro prints to the standard error console stream (`stderr`), as opposed to `println!`, which prints to the standard output console stream (`stdout`). We’ll talk more about `stderr` and `stdout` in the [“Writing Error Messages to Standard Error Instead of Standard Output” section in Chapter 12](https://doc.rust-lang.org/stable/book/ch12-06-writing-to-stderr-instead-of-stdout.html).
>  dbg macro 打印到标准错误控制台流 stderr 而不是标准输出控制台流 stdout

Here’s an example where we’re interested in the value that gets assigned to the `width` field, as well as the value of the whole struct in `rect1`:

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

We can put `dbg!` around the expression `30 * scale` and, because `dbg!` returns ownership of the expression’s value, the `width` field will get the same value as if we didn’t have the `dbg!` call there. We don’t want `dbg!` to take ownership of `rect1`, so we use a reference to `rect1` in the next call. Here’s what the output of this example looks like:
>  我们可以在需要打印的地方使用 `dbg!` 包围，因为 dbg macro 会返回所有权
>  如果我们不希望 dbg 获取所有权，可以传递引用

```
$ cargo run
   Compiling rectangles v0.1.0 (file:///projects/rectangles)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.61s
     Running `target/debug/rectangles`
[src/main.rs:10:16] 30 * scale = 60
[src/main.rs:14:5] &rect1 = Rectangle {
    width: 60,
    height: 50,
}
```

We can see the first bit of output came from _src/main.rs_ line 10 where we’re debugging the expression `30 * scale`, and its resultant value is `60` (the `Debug` formatting implemented for integers is to print only their value). The `dbg!` call on line 14 of _src/main.rs_ outputs the value of `&rect1`, which is the `Rectangle` struct. This output uses the pretty `Debug` formatting of the `Rectangle` type. The `dbg!` macro can be really helpful when you’re trying to figure out what your code is doing!

In addition to the `Debug` trait, Rust has provided a number of traits for us to use with the `derive` attribute that can add useful behavior to our custom types. Those traits and their behaviors are listed in [Appendix C](https://doc.rust-lang.org/stable/book/appendix-03-derivable-traits.html). We’ll cover how to implement these traits with custom behavior as well as how to create your own traits in Chapter 10. There are also many attributes other than `derive`; for more information, see [the “Attributes” section of the Rust Reference](https://doc.rust-lang.org/stable/reference/attributes.html).
>  除了 Debug trait 以外，Rust 还提供了许多其他 traits，我们可以直接使用 derive attribute 用以为我们的自定义类型添加行为

Our `area` function is very specific: it only computes the area of rectangles. It would be helpful to tie this behavior more closely to our `Rectangle` struct because it won’t work with any other type. Let’s look at how we can continue to refactor this code by turning the `area` function into an `area` _method_ defined on our `Rectangle` type.

## 5.3 Method Syntax
_Methods_ are similar to functions: we declare them with the `fn` keyword and a name, they can have parameters and a return value, and they contain some code that’s run when the method is called from somewhere else. Unlike functions, methods are defined within the context of a struct (or an enum or a trait object, which we cover in [Chapter 6](https://doc.rust-lang.org/book/ch06-00-enums.html) and [Chapter 18](https://doc.rust-lang.org/book/ch18-02-trait-objects.html), respectively), and their first parameter is always `self`, which represents the instance of the struct the method is being called on.
>  方法类似于函数: 我们同样用 `fn` 声明方法，方法可以有参数和返回值
>  方法和函数不同，方法需要定义在 struct 的上下文内 (或者 enum, trait object)，并且方法的第一个参数永远是 `self`，表示调用方法的 instance

### Defining Methods
Let’s change the `area` function that has a `Rectangle` instance as a parameter and instead make an `area` method defined on the `Rectangle` struct, as shown in Listing 5-13.

Filename: src/main.rs

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

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };

    println!(
        "The area of the rectangle is {} square pixels.",
        rect1.area()
    );
}
```

[Listing 5-13](https://doc.rust-lang.org/book/ch05-03-method-syntax.html#listing-5-13): Defining an `area` method on the `Rectangle` struct

To define the function within the context of `Rectangle`, we start an `impl` (implementation) block for `Rectangle`. Everything within this `impl` block will be associated with the `Rectangle` type. Then we move the `area` function within the `impl` curly brackets and change the first (and in this case, only) parameter to be `self` in the signature and everywhere within the body. In `main`, where we called the `area` function and passed `rect1` as an argument, we can instead use _method syntax_ to call the `area` method on our `Rectangle` instance. The method syntax goes after an instance: we add a dot followed by the method name, parentheses, and any arguments.
>  要在 `Rectangle` 的上下文内定义函数，我们需要为 `Rectangle` 启动一个 `impl` 块，在该块内的内容都会和 `Rectangle` 类型关联
>  我们将函数签名的第一个参数改为 `self`
>  在调用方法时，使用 `.<method-name>()` 即可

In the signature for `area`, we use `&self` instead of `rectangle: &Rectangle`. The `&self` is actually short for `self: &Self`. Within an `impl` block, the type `Self` is an alias for the type that the `impl` block is for. Methods must have a parameter named `self` of type `Self` for their first parameter, so Rust lets you abbreviate this with only the name `self` in the first parameter spot. Note that we still need to use the `&` in front of the `self` shorthand to indicate that this method borrows the `Self` instance, just as we did in `rectangle: &Rectangle`. Methods can take ownership of `self`, borrow `self` immutably, as we’ve done here, or borrow `self` mutably, just as they can any other parameter.
>  方法签名中，我们使用 `&self` 替代了 `rectangle: &Rectangle`，它实际上是对 `self: &Self` 的简写
>  在 `impl` 块内，类型 `Self` 就是对 `impl` 块关联的类型的别名
>  方法可以获取 `self` 的所有权，或者获取不可变引用，或者获取可变引用

We chose `&self` here for the same reason we used `&Rectangle` in the function version: we don’t want to take ownership, and we just want to read the data in the struct, not write to it. If we wanted to change the instance that we’ve called the method on as part of what the method does, we’d use `&mut self` as the first parameter. Having a method that takes ownership of the instance by using just `self` as the first parameter is rare; this technique is usually used when the method transforms `self` into something else and you want to prevent the caller from using the original instance after the transformation.
>  如果要获取可变引用，则写为 `&mut self`
>  获取所有权的方法很少见，通常用于方法将 `self` 转化为其他东西，并且调用者不能在转化后使用原来的实例

The main reason for using methods instead of functions, in addition to providing method syntax and not having to repeat the type of `self` in every method’s signature, is for organization. We’ve put all the things we can do with an instance of a type in one `impl` block rather than making future users of our code search for capabilities of `Rectangle` in various places in the library we provide.

Note that we can choose to give a method the same name as one of the struct’s fields. For example, we can define a method on `Rectangle` that is also named `width`:
>  方法的名称可以和 struct fild 名称相同

Filename: src/main.rs

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

Here, we’re choosing to make the `width` method return `true` if the value in the instance’s `width` field is greater than `0` and `false` if the value is `0`: we can use a field within a method of the same name for any purpose. In `main`, when we follow `rect1.width` with parentheses, Rust knows we mean the method `width`. When we don’t use parentheses, Rust knows we mean the field `width`.

Often, but not always, when we give a method the same name as a field we want it to only return the value in the field and do nothing else. Methods like this are called _getters_, and Rust does not implement them automatically for struct fields as some other languages do. Getters are useful because you can make the field private but the method public, and thus enable read-only access to that field as part of the type’s public API. We will discuss what public and private are and how to designate a field or method as public or private in [Chapter 7](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html#exposing-paths-with-the-pub-keyword).
>  通常和 field 有相同名字的 method 将仅仅返回 value，不做其他的事
>  这样的 methods 称为 getters, Rust 不会自动为 fields 实现 getters
>  getters 用于让 fields 私有，方法共有，即让 fields 只读

### Where’s the `->` Operator?
In C and C++, two different operators are used for calling methods: you use `.` if you’re calling a method on the object directly and `->` if you’re calling the method on a pointer to the object and need to dereference the pointer first. In other words, if `object` is a pointer, `object->something()` is similar to `(*object).something()`.

Rust doesn’t have an equivalent to the `->` operator; instead, Rust has a feature called _automatic referencing and dereferencing_. Calling methods is one of the few places in Rust with this behavior.
>  Rust 具有自动引用和解引用的机制

Here’s how it works: when you call a method with `object.something()`, Rust automatically adds in `&`, `&mut`, or `*` so `object` matches the signature of the method. In other words, the following are the same:
>  当我们通过 `object.something()` 调用方法时，Rust 会自动添加 `&, &mut, *`，使得方法调用者匹配方法的 `self` 参数类型

```rust
p1.distance(&p2); (&p1).distance(&p2); 
```

The first one looks much cleaner. This automatic referencing behavior works because methods have a clear receiver—the type of `self`. Given the receiver and name of a method, Rust can figure out definitively whether the method is reading (`&self`), mutating (`&mut self`), or consuming (`self`). The fact that Rust makes borrowing implicit for method receivers is a big part of making ownership ergonomic in practice.
>  自动引用的行为可以工作是因为方法的参数类型指定了需要接收的实例类型，即 `self` 的类型

### Methods with More Parameters
Let’s practice using methods by implementing a second method on the `Rectangle` struct. This time we want an instance of `Rectangle` to take another instance of `Rectangle` and return `true` if the second `Rectangle` can fit completely within `self` (the first `Rectangle`); otherwise, it should return `false`. That is, once we’ve defined the `can_hold` method, we want to be able to write the program shown in Listing 5-14.

Filename: src/main.rs

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

[Listing 5-14](https://doc.rust-lang.org/book/ch05-03-method-syntax.html#listing-5-14): Using the as-yet-unwritten `can_hold` method

The expected output would look like the following because both dimensions of `rect2` are smaller than the dimensions of `rect1`, but `rect3` is wider than `rect1`:

```
Can rect1 hold rect2? true
Can rect1 hold rect3? false
```

We know we want to define a method, so it will be within the `impl Rectangle` block. The method name will be `can_hold`, and it will take an immutable borrow of another `Rectangle` as a parameter. We can tell what the type of the parameter will be by looking at the code that calls the method: `rect1.can_hold(&rect2)` passes in `&rect2`, which is an immutable borrow to `rect2`, an instance of `Rectangle`. This makes sense because we only need to read `rect2` (rather than write, which would mean we’d need a mutable borrow), and we want `main` to retain ownership of `rect2` so we can use it again after calling the `can_hold` method. The return value of `can_hold` will be a Boolean, and the implementation will check whether the width and height of `self` are greater than the width and height of the other `Rectangle`, respectively. Let’s add the new `can_hold` method to the `impl` block from Listing 5-13, shown in Listing 5-15.

Filename: src/main.rs

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

[Listing 5-15](https://doc.rust-lang.org/book/ch05-03-method-syntax.html#listing-5-15): Implementing the `can_hold` method on `Rectangle` that takes another `Rectangle` instance as a parameter

When we run this code with the `main` function in Listing 5-14, we’ll get our desired output. Methods can take multiple parameters that we add to the signature after the `self` parameter, and those parameters work just like parameters in functions.

### Associated Functions
All functions defined within an `impl` block are called _associated functions_ because they’re associated with the type named after the `impl`. We can define associated functions that don’t have `self` as their first parameter (and thus are not methods) because they don’t need an instance of the type to work with. We’ve already used one function like this: the `String::from` function that’s defined on the `String` type.
>  所有在 `impl` 块内定义的函数都称为关联函数，因为他们与 `impl` 的类型关联
>  我们可以定义一些不以 `self` 为第一个参数的相关函数 (因此不是方法)，他们不需要一个实例就可以工作
>  `String::from` 就是例子，它是定义在 `String` 类型上的

Associated functions that aren’t methods are often used for constructors that will return a new instance of the struct. These are often called `new`, but `new` isn’t a special name and isn’t built into the language. For example, we could choose to provide an associated function named `square` that would have one dimension parameter and use that as both width and height, thus making it easier to create a square `Rectangle` rather than having to specify the same value twice:
>  不是方法的相关函数通常用于构造函数，这些函数通常命名为 `new`，会返回一个新实例

Filename: src/main.rs

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

The `Self` keywords in the return type and in the body of the function are aliases for the type that appears after the `impl` keyword, which in this case is `Rectangle`.

To call this associated function, we use the `::` syntax with the struct name; `let sq = Rectangle::square(3);` is an example. This function is namespaced by the struct: the `::` syntax is used for both associated functions and namespaces created by modules. We’ll discuss modules in [Chapter 7](https://doc.rust-lang.org/book/ch07-02-defining-modules-to-control-scope-and-privacy.html).
>  调用相关函数需要使用 `<struct-name>::<function-name>()` 的语法
>  该函数在 struct 的命名空间下，`::` 语法即用于相关函数，也用于模块创建的命名空间

### Multiple `impl` Blocks
Each struct is allowed to have multiple `impl` blocks. For example, Listing 5-15 is equivalent to the code shown in Listing 5-16, which has each method in its own `impl` block.

```rust
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

[Listing 5-16](https://doc.rust-lang.org/book/ch05-03-method-syntax.html#listing-5-16): Rewriting Listing 5-15 using multiple `impl` blocks

There’s no reason to separate these methods into multiple `impl` blocks here, but this is valid syntax. We’ll see a case in which multiple `impl` blocks are useful in Chapter 10, where we discuss generic types and traits.

## Summary
Structs let you create custom types that are meaningful for your domain. By using structs, you can keep associated pieces of data connected to each other and name each piece to make your code clear. In `impl` blocks, you can define functions that are associated with your type, and methods are a kind of associated function that let you specify the behavior that instances of your structs have.

But structs aren’t the only way you can create custom types: let’s turn to Rust’s enum feature to add another tool to your toolbox.

# 6 Enums and Pattern Matching
In this chapter, we’ll look at _enumerations_, also referred to as _enums_. Enums allow you to define a type by enumerating its possible _variants_. First we’ll define and use an enum to show how an enum can encode meaning along with data. Next, we’ll explore a particularly useful enum, called `Option`, which expresses that a value can be either something or nothing. Then we’ll look at how pattern matching in the `match` expression makes it easy to run different code for different values of an enum. Finally, we’ll cover how the `if let` construct is another convenient and concise idiom available to handle enums in your code.
>  枚举允许我们通过列举可能的变体来定义一种类型
>  枚举类型 `Option` 表示一个 value 可以是某个值或者没有值

## 6.1 Defining an Enum
Where structs give you a way of grouping together related fields and data, like a `Rectangle` with its `width` and `height`, enums give you a way of saying a value is one of a possible set of values. For example, we may want to say that `Rectangle` is one of a set of possible shapes that also includes `Circle` and `Triangle`. To do this, Rust allows us to encode these possibilities as an enum.

Let’s look at a situation we might want to express in code and see why enums are useful and more appropriate than structs in this case. Say we need to work with IP addresses. Currently, two major standards are used for IP addresses: version four and version six. Because these are the only possibilities for an IP address that our program will come across, we can _enumerate_ all possible variants, which is where enumeration gets its name.
>  假设我们要处理 IP 地址，IP 地址有两种标准: v4, v6
>  因为给定了两种确定的可能值，故我们可以枚举所有可能的变体

Any IP address can be either a version four or a version six address, but not both at the same time. That property of IP addresses makes the enum data structure appropriate because an enum value can only be one of its variants. Both version four and version six addresses are still fundamentally IP addresses, so they should be treated as the same type when the code is handling situations that apply to any kind of IP address.
>  任意 IP 地址只能是 v4 or v6，故很适合用枚举类型编码这一特性

We can express this concept in code by defining an `IpAddrKind` enumeration and listing the possible kinds an IP address can be, `V4` and `V6`. These are the variants of the enum:
>  枚举类型的定义形式为 `enum <type-name> { variant1, variant2, ...}`

```rust
enum IpAddrKind {
    V4,
    V6,
}
```

`IpAddrKind` is now a custom data type that we can use elsewhere in our code.

### Enum Values
We can create instances of each of the two variants of `IpAddrKind` like this:

```rust
let four = IpAddrKind::V4;
let six = IpAddrKind::V6;
```

Note that the variants of the enum are namespaced under its identifier, and we use a double colon to separate the two. This is useful because now both values `IpAddrKind::V4` and `IpAddrKind::V6` are of the same type: `IpAddrKind`. We can then, for instance, define a function that takes any `IpAddrKind`:
>  我们通过 `::` 表示枚举的变体在枚举类型的命名空间下
>  枚举的变体值的类型都是枚举类型

```rust
fn route(ip_kind: IpAddrKind) {}
```

And we can call this function with either variant:

```rust
route(IpAddrKind::V4);
route(IpAddrKind::V6);
```

Using enums has even more advantages. Thinking more about our IP address type, at the moment we don’t have a way to store the actual IP address _data_; we only know what _kind_ it is. Given that you just learned about structs in Chapter 5, you might be tempted to tackle this problem with structs as shown in Listing 6-1.

```rust
enum IpAddrKind {
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
};

let loopback = IpAddr {
    kind: IpAddrKind::V6,
    address: String::from("::1"),
};

```

[Listing 6-1](https://doc.rust-lang.org/book/ch06-01-defining-an-enum.html#listing-6-1): Storing the data and `IpAddrKind` variant of an IP address using a `struct`

Here, we’ve defined a struct `IpAddr` that has two fields: a `kind` field that is of type `IpAddrKind` (the enum we defined previously) and an `address` field of type `String`. We have two instances of this struct. The first is `home`, and it has the value `IpAddrKind::V4` as its `kind` with associated address data of `127.0.0.1`. The second instance is `loopback`. It has the other variant of `IpAddrKind` as its `kind` value, `V6`, and has address `::1` associated with it. We’ve used a struct to bundle the `kind` and `address` values together, so now the variant is associated with the value.

However, representing the same concept using just an enum is more concise: rather than an enum inside a struct, we can put data directly into each enum variant. This new definition of the `IpAddr` enum says that both `V4` and `V6` variants will have associated `String` values:
>  Rust 中，每个枚举变体可以之间存储数据，或者说关联数据

```rust
enum IpAddr {
    V4(String),
    V6(String),
}

let home = IpAddr::V4(String::from("127.0.0.1"));

let loopback = IpAddr::V6(String::from("::1"));
```

We attach data to each variant of the enum directly, so there is no need for an extra struct. Here, it’s also easier to see another detail of how enums work: the name of each enum variant that we define also becomes a function that constructs an instance of the enum. That is, `IpAddr::V4()` is a function call that takes a `String` argument and returns an instance of the `IpAddr` type. We automatically get this constructor function defined as a result of defining the enum.
>  我们可以看到我们定义的枚举变体的名称本身也会成为一个构造枚举实例的函数 (或者理解为枚举类型的一个构造函数)
>  `IpAddr::V4()` 是一个函数调用，接收 `String` 参数，返回一个 `IpAddr` 类型的实例
>  这个构造函数是 Rust 自动为我们生成的

There’s another advantage to using an enum rather than a struct: each variant can have different types and amounts of associated data. Version four IP addresses will always have four numeric components that will have values between 0 and 255. If we wanted to store `V4` addresses as four `u8` values but still express `V6` addresses as one `String` value, we wouldn’t be able to with a struct. Enums handle this case with ease:
>  不同的变体可以关联不同类型、数量的数据
>  例如 IPv4 的地址可以用 4 个 0 到 255 的数值成分表示

```rust
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

let home = IpAddr::V4(127, 0, 0, 1);

let loopback = IpAddr::V6(String::from("::1"));
```

We’ve shown several different ways to define data structures to store version four and version six IP addresses. However, as it turns out, wanting to store IP addresses and encode which kind they are is so common that [the standard library has a definition we can use!](https://doc.rust-lang.org/std/net/enum.IpAddr.html) Let’s look at how the standard library defines `IpAddr`: it has the exact enum and variants that we’ve defined and used, but it embeds the address data inside the variants in the form of two different structs, which are defined differently for each variant:
>  Rust 标准库也提供了用于存储 IP 地址的类型 `IpAddr`
>  `IpAddr` 是一个枚举类型，有两个变体，变体分别关联了两个结构体

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

This code illustrates that you can put any kind of data inside an enum variant: strings, numeric types, or structs, for example. You can even include another enum! Also, standard library types are often not much more complicated than what you might come up with.
>  我们可以将任意类型的数据关联到变体，甚至另外一个枚举类型的值

Note that even though the standard library contains a definition for `IpAddr`, we can still create and use our own definition without conflict because we haven’t brought the standard library’s definition into our scope. We’ll talk more about bringing types into scope in Chapter 7.
>  注意如果我们不将标准库定义代入作用域，我们可以自行定义 `IpAddr` 类型而不发生冲突

Let’s look at another example of an enum in Listing 6-2: this one has a wide variety of types embedded in its variants.

```rust
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}
```

[Listing 6-2](https://doc.rust-lang.org/book/ch06-01-defining-an-enum.html#listing-6-2): A `Message` enum whose variants each store different amounts and types of values

This enum has four variants with different types:

- `Quit`: Has no data associated with it at all
- `Move`: Has named fields, like a struct does
- `Write`: Includes a single `String`
- `ChangeColor`: Includes three `i32` values

Defining an enum with variants such as the ones in Listing 6-2 is similar to defining different kinds of struct definitions, except the enum doesn’t use the `struct` keyword and all the variants are grouped together under the `Message` type. The following structs could hold the same data that the preceding enum variants hold:

```rust
struct QuitMessage; // unit struct
struct MoveMessage {
    x: i32,
    y: i32,
}
struct WriteMessage(String); // tuple struct
struct ChangeColorMessage(i32, i32, i32); // tuple struct
```

But if we used the different structs, each of which has its own type, we couldn’t as easily define a function to take any of these kinds of messages as we could with the `Message` enum defined in Listing 6-2, which is a single type.

There is one more similarity between enums and structs: just as we’re able to define methods on structs using `impl`, we’re also able to define methods on enums. Here’s a method named `call` that we could define on our `Message` enum:

```rust
impl Message {
    fn call(&self) {
        // method body would be defined here
    }
}

let m = Message::Write(String::from("hello"));
m.call();
```

The body of the method would use `self` to get the value that we called the method on. In this example, we’ve created a variable `m` that has the value `Message::Write(String::from("hello"))`, and that is what `self` will be in the body of the `call` method when `m.call()` runs.

>  我们可以为枚举类型实现方法，和结构体是类似的

Let’s look at another enum in the standard library that is very common and useful: `Option`.

### The `Option` Enum and Its Advantages Over Null Values
This section explores a case study of `Option`, which is another enum defined by the standard library. The `Option` type encodes the very common scenario in which a value could be something or it could be nothing.

For example, if you request the first item in a non-empty list, you would get a value. If you request the first item in an empty list, you would get nothing. Expressing this concept in terms of the type system means the compiler can check whether you’ve handled all the cases you should be handling; this functionality can prevent bugs that are extremely common in other programming languages.
>  Rust 通过 `Option` 类型在类型系统中表示了 value 存在或为空的概念
>  这使得编译器可以检查我们是否处理了所有应该处理的情况，以避免在其他语言中常见的 bug

Programming language design is often thought of in terms of which features you include, but the features you exclude are important too. Rust doesn’t have the null feature that many other languages have. _Null_ is a value that means there is no value there. In languages with null, variables can always be in one of two states: null or not-null.
>  Rust 没有其他语言中常见的空值特性
>  空值表示那里没有值，在支持空值的语言中，变量总是处于两种状态之间: 空或非空

In his 2009 presentation “Null References: The Billion Dollar Mistake,” Tony Hoare, the inventor of null, had this to say:

> I call it my billion-dollar mistake. At that time, I was designing the first comprehensive type system for references in an object-oriented language. My goal was to ensure that all use of references should be absolutely safe, with checking performed automatically by the compiler. But I couldn’t resist the temptation to put in a null reference, simply because it was so easy to implement. This has led to innumerable errors, vulnerabilities, and system crashes, which have probably caused a billion dollars of pain and damage in the last forty years.

The problem with null values is that if you try to use a null value as a not-null value, you’ll get an error of some kind. Because this null or not-null property is pervasive, it’s extremely easy to make this kind of error.
>  空值的问题在于: 如果试图将空值作为非空值使用，就会引发错误
>  由于 “是否为空” 这一属性无处不在，这种错误非常容易发生

However, the concept that null is trying to express is still a useful one: a null is a value that is currently invalid or absent for some reason.
>  但 null 值本身想要表达的概念仍然是有用的: null 表示一个值由于某种原因当前无效或缺失

The problem isn’t really with the concept but with the particular implementation. As such, Rust does not have nulls, but it does have an enum that can encode the concept of a value being present or absent. This enum is `Option<T>`, and it is [defined by the standard library](https://doc.rust-lang.org/std/option/enum.Option.html) as follows:
>  问题不在于概念本身，而在于实现
>  因此 Rust 没有 null，但有一个枚举类型来表示值的存在或不存在，这个枚举类型就是 `Option<T>`，它在标准库的定义如下

```rust
enum Option<T> {
    None,
    Some(T),
}
```

The `Option<T>` enum is so useful that it’s even included in the prelude; you don’t need to bring it into scope explicitly. Its variants are also included in the prelude: you can use `Some` and `None` directly without the `Option::` prefix. The `Option<T>` enum is still just a regular enum, and `Some(T)` and `None` are still variants of type `Option<T>`.
>  `Option<T>` 类型属于 prelude，我们不需要将它显示带入作用域
>  `Option<T>` 的变体也在 prelude 中，我们可以不使用 `Option::` 前缀，直接使用 `Some, None`

The `<T>` syntax is a feature of Rust we haven’t talked about yet. It’s a generic type parameter, and we’ll cover generics in more detail in Chapter 10. For now, all you need to know is that `<T>` means that the `Some` variant of the `Option` enum can hold one piece of data of any type, and that each concrete type that gets used in place of `T` makes the overall `Option<T>` type a different type. Here are some examples of using `Option` values to hold number types and char types:
>  `<T>` 语法是一个泛型类型参数，它意味着 `Some` 变体可以存储任意类型的数据，每个用于替代 `T` 的类型都会使得整个 `Option<T>` 成为一种不同的类型

```rust
let some_number = Some(5);
let some_char = Some('e');

let absent_number: Option<i32> = None;
```

The type of `some_number` is `Option<i32>`. The type of `some_char` is `Option<char>`, which is a different type. Rust can infer these types because we’ve specified a value inside the `Some` variant. For `absent_number`, Rust requires us to annotate the overall `Option` type: the compiler can’t infer the type that the corresponding `Some` variant will hold by looking only at a `None` value. Here, we tell Rust that we mean for `absent_number` to be of type `Option<i32>`.
>  上例中，`some_number` 的类型为 `Option<i32>`，`some_char` 的类型为 `Option<char>`，他们由 Rust 自动推导得到
>  `absent_number` 则需要自行指定类型

When we have a `Some` value, we know that a value is present and the value is held within the `Some`. When we have a `None` value, in some sense it means the same thing as null: we don’t have a valid value. So why is having `Option<T>` any better than having null?
>  `None` 变体的含义就和 null 完全相同，为什么 `Option<T>` 优于 `None`?

In short, because `Option<T>` and `T` (where `T` can be any type) are different types, the compiler won’t let us use an `Option<T>` value as if it were definitely a valid value. For example, this code won’t compile, because it’s trying to add an `i8` to an `Option<i8>`:
>  简单地说，是因为 `Option<T>` 和 `T` 是不同的类型，故编译器不会让我们将一个类型为 `Option<T>` 类型的值当作一个 `T` 类型的有效值
>  例如下面的代码就无法编译

```rust
let x: i8 = 5;
let y: Option<i8> = Some(5);

let sum = x + y;
```

If we run this code, we get an error message like this one:

```
$ cargo run
   Compiling enums v0.1.0 (file:///projects/enums)
error[E0277]: cannot add `Option<i8>` to `i8`
 --> src/main.rs:5:17
  |
5 |     let sum = x + y;
  |                 ^ no implementation for `i8 + Option<i8>`
  |
  = help: the trait `Add<Option<i8>>` is not implemented for `i8`
  = help: the following other types implement trait `Add<Rhs>`:
            `&i8` implements `Add<i8>`
            `&i8` implements `Add`
            `i8` implements `Add<&i8>`
            `i8` implements `Add`

For more information about this error, try `rustc --explain E0277`.
error: could not compile `enums` (bin "enums") due to 1 previous error
```

Intense! In effect, this error message means that Rust doesn’t understand how to add an `i8` and an `Option<i8>`, because they’re different types. When we have a value of a type like `i8` in Rust, the compiler will ensure that we always have a valid value. We can proceed confidently without having to check for null before using that value. Only when we have an `Option<i8>` (or whatever type of value we’re working with) do we have to worry about possibly not having a value, and the compiler will make sure we handle that case before using the value.
>  Rust 中，如果我们有一个例如 `i8` 类型的 value，编译器会保证这个 value 一定是有效的
>  只有当我们有一个为 `Option<i8>` 类型的 value 时，我们需要担心可能值是无效的，编译器则会通过编译失败来提醒我们处理这一点

In other words, you have to convert an `Option<T>` to a `T` before you can perform `T` operations with it. Generally, this helps catch one of the most common issues with null: assuming that something isn’t null when it actually is.
>  换句话说，我们需要将 `Option<T>` 转化为 `T`，才能对其进行 `T` 相关的操作
>  这就有助于处理关于 null 最常见的问题: 认为某个 value 不是 null，但它实际上是 null

Eliminating the risk of incorrectly assuming a not-null value helps you to be more confident in your code. In order to have a value that can possibly be null, you must explicitly opt in by making the type of that value `Option<T>`. Then, when you use that value, you are required to explicitly handle the case when the value is null. Everywhere that a value has a type that isn’t an `Option<T>`, you _can_ safely assume that the value isn’t null. This was a deliberate design decision for Rust to limit null’s pervasiveness and increase the safety of Rust code.
>  消除了这一个风险后，我们将对代码更加自信
>  如果要有一个可能为 null 的 value，我们必须显式将其类型声明为 `Option<T>`，当使用这个 value 时，就需要显式处理 value 可能为 null 的情况
>  而如果 value 不是 `Option<T>` 类型，我们可以安全地认为 value 不是 null
>  这是一个特意的设计决策

So how do you get the `T` value out of a `Some` variant when you have a value of type `Option<T>` so that you can use that value? The `Option<T>` enum has a large number of methods that are useful in a variety of situations; you can check them out in [its documentation](https://doc.rust-lang.org/std/option/enum.Option.html). Becoming familiar with the methods on `Option<T>` will be extremely useful in your journey with Rust.

In general, in order to use an `Option<T>` value, you want to have code that will handle each variant. You want some code that will run only when you have a `Some(T)` value, and this code is allowed to use the inner `T`. You want some other code to run only if you have a `None` value, and that code doesn’t have a `T` value available. The `match` expression is a control flow construct that does just this when used with enums: it will run different code depending on which variant of the enum it has, and that code can use the data inside the matching value.
>  要使用 `Option<T>` value，我们需要一个处理每个变体的代码，例如 `match` 表达式

## 6.2 The `match` Control Flow Construct
Rust has an extremely powerful control flow construct called `match` that allows you to compare a value against a series of patterns and then execute code based on which pattern matches. Patterns can be made up of literal values, variable names, wildcards, and many other things; [Chapter 19](https://doc.rust-lang.org/book/ch19-00-patterns.html) covers all the different kinds of patterns and what they do. The power of `match` comes from the expressiveness of the patterns and the fact that the compiler confirms that all possible cases are handled.
>  控制流结构 `match` 允许我们将 value 和一系列 pattern 比较，然后执行代码
>  pattern 可以是字面值，变量名，通配符等
>  `match` 的 power 来自于 pattern 的表示能力，以及编译器会确保所有可能的情况都被处理

Think of a `match` expression as being like a coin-sorting machine: coins slide down a track with variously sized holes along it, and each coin falls through the first hole it encounters that it fits into. In the same way, values go through each pattern in a `match`, and at the first pattern the value “fits,” the value falls into the associated code block to be used during execution.

Speaking of coins, let’s use them as an example using `match`! We can write a function that takes an unknown US coin and, in a similar way as the counting machine, determines which coin it is and returns its value in cents, as shown in Listing 6-3.

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

[Listing 6-3](https://doc.rust-lang.org/book/ch06-02-match.html#listing-6-3): An enum and a `match` expression that has the variants of the enum as its patterns

Let’s break down the `match` in the `value_in_cents` function. First we list the `match` keyword followed by an expression, which in this case is the value `coin`. This seems very similar to a conditional expression used with `if`, but there’s a big difference: with `if`, the condition needs to evaluate to a Boolean value, but here it can be any type. The type of `coin` in this example is the `Coin` enum that we defined on the first line.
>  `if` 中，condition 必须是布尔值，而 `match` 中的 value 可以是任意类型

Next are the `match` arms. An arm has two parts: a pattern and some code. The first arm here has a pattern that is the value `Coin::Penny` and then the `=>` operator that separates the pattern and the code to run. The code in this case is just the value `1`. Each arm is separated from the next with a comma.
>  `match` 分支有两部分: pattern 和 code
>  `=>` 运算符分离了 pattern 和 code
>  arms 之间用逗号隔开

When the `match` expression executes, it compares the resultant value against the pattern of each arm, in order. If a pattern matches the value, the code associated with that pattern is executed. If that pattern doesn’t match the value, execution continues to the next arm, much as in a coin-sorting machine. We can have as many arms as we need: in Listing 6-3, our `match` has four arms.

The code associated with each arm is an expression, and the resultant value of the expression in the matching arm is the value that gets returned for the entire `match` expression.
>  每个 arm 关联的代码是表达式，其 value 就是整个 `match` 表达式返回的 value

We don’t typically use curly brackets if the match arm code is short, as it is in Listing 6-3 where each arm just returns a value. If you want to run multiple lines of code in a match arm, you must use curly brackets, and the comma following the arm is then optional. For example, the following code prints “Lucky penny!” every time the method is called with a `Coin::Penny`, but still returns the last value of the block, `1`:
>  arm 也可以关联一个 block，该 block 的返回值就是该 arm 的 value
>  如果关联的 block，最后的 `,` 可以省略

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

### Patterns That Bind to Values
Another useful feature of match arms is that they can bind to the parts of the values that match the pattern. This is how we can extract values out of enum variants.

As an example, let’s change one of our enum variants to hold data inside it. From 1999 through 2008, the United States minted quarters with different designs for each of the 50 states on one side. No other coins got state designs, so only quarters have this extra value. We can add this information to our `enum` by changing the `Quarter` variant to include a `UsState` value stored inside it, which we’ve done in Listing 6-4.

```rust
#[derive(Debug)] // so we can inspect the state in a minute
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

[Listing 6-4](https://doc.rust-lang.org/book/ch06-02-match.html#listing-6-4): A `Coin` enum in which the `Quarter` variant also holds a `UsState` value

Let’s imagine that a friend is trying to collect all 50 state quarters. While we sort our loose change by coin type, we’ll also call out the name of the state associated with each quarter so that if it’s one our friend doesn’t have, they can add it to their collection.

In the match expression for this code, we add a variable called `state` to the pattern that matches values of the variant `Coin::Quarter`. When a `Coin::Quarter` matches, the `state` variable will bind to the value of that quarter’s state. Then we can use `state` in the code for that arm, like so:
>  match arm 可以对 enum variant 所关联的值进行绑定，进而可以使用这个值，示例的形式如下

```rust
fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {state:?}!");
            25
        }
    }
}
```

If we were to call `value_in_cents(Coin::Quarter(UsState::Alaska))`, `coin` would be `Coin::Quarter(UsState::Alaska)`. When we compare that value with each of the match arms, none of them match until we reach `Coin::Quarter(state)`. At that point, the binding for `state` will be the value `UsState::Alaska`. We can then use that binding in the `println!` expression, thus getting the inner state value out of the `Coin` enum variant for `Quarter`.

### Matching with `Option<T>`
In the previous section, we wanted to get the inner `T` value out of the `Some` case when using `Option<T>`; we can also handle `Option<T>` using `match`, as we did with the `Coin` enum! Instead of comparing coins, we’ll compare the variants of `Option<T>`, but the way the `match` expression works remains the same.

Let’s say we want to write a function that takes an `Option<i32>` and, if there’s a value inside, adds 1 to that value. If there isn’t a value inside, the function should return the `None` value and not attempt to perform any operations.

This function is very easy to write, thanks to `match`, and will look like Listing 6-5.

```rust
fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
    }
}

let five = Some(5);
let six = plus_one(five);
let none = plus_one(None);
```

[Listing 6-5](https://doc.rust-lang.org/book/ch06-02-match.html#listing-6-5): A function that uses a `match` expression on an `Option<i32>`

>  `match` 表达式可以用于处理 `Option<T>` 类型，示例形式如上

Let’s examine the first execution of `plus_one` in more detail. When we call `plus_one(five)`, the variable `x` in the body of `plus_one` will have the value `Some(5)`. We then compare that against each match arm:

```rust
None => None,
```

The `Some(5)` value doesn’t match the pattern `None`, so we continue to the next arm:

```rust
Some(i) => Some(i + 1),
```

Does `Some(5)` match `Some(i)`? It does! We have the same variant. The `i` binds to the value contained in `Some`, so `i` takes the value `5`. The code in the match arm is then executed, so we add 1 to the value of `i` and create a new `Some` value with our total `6` inside.
>  `Some(i) => Some(i+1)` 中，我们的 variant `Some(5)` 会被匹配，然后 `i` 会绑定到 `5`，然后 arm 中的代码会被执行，即构造了一个新的 variant

Now let’s consider the second call of `plus_one` in Listing 6-5, where `x` is `None`. We enter the `match` and compare to the first arm:

```rust
None => None,
```

It matches! There’s no value to add to, so the program stops and returns the `None` value on the right side of `=>`. Because the first arm matched, no other arms are compared.

Combining `match` and enums is useful in many situations. You’ll see this pattern a lot in Rust code: `match` against an enum, bind a variable to the data inside, and then execute code based on it. It’s a bit tricky at first, but once you get used to it, you’ll wish you had it in all languages. It’s consistently a user favorite.

### Matches Are Exhaustive
There’s one other aspect of `match` we need to discuss: the arms’ patterns must cover all possibilities. Consider this version of our `plus_one` function, which has a bug and won’t compile:
>  `match` 中的 patterns 必须覆盖所有情况，否则无法编译

```rust
fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        Some(i) => Some(i + 1),
    }
}
```

We didn’t handle the `None` case, so this code will cause a bug. Luckily, it’s a bug Rust knows how to catch. If we try to compile this code, we’ll get this error:

```
$ cargo run
   Compiling enums v0.1.0 (file:///projects/enums)
error[E0004]: non-exhaustive patterns: `None` not covered
 --> src/main.rs:3:15
  |
3 |         match x {
  |               ^ pattern `None` not covered
  |
note: `Option<i32>` defined here
 --> /rustc/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/option.rs:572:1
 ::: /rustc/4eb161250e340c8f48f66e2b929ef4a5bed7c181/library/core/src/option.rs:576:5
  |
  = note: not covered
  = note: the matched value is of type `Option<i32>`
help: ensure that all possible cases are being handled by adding a match arm with a wildcard pattern or an explicit pattern as shown
  |
4 ~             Some(i) => Some(i + 1),
5 ~             None => todo!(),
  |

For more information about this error, try `rustc --explain E0004`.
error: could not compile `enums` (bin "enums") due to 1 previous error
```

Rust knows that we didn’t cover every possible case, and even knows which pattern we forgot! Matches in Rust are _exhaustive_: we must exhaust every last possibility in order for the code to be valid. Especially in the case of `Option<T>`, when Rust prevents us from forgetting to explicitly handle the `None` case, it protects us from assuming that we have a value when we might have null, thus making the billion-dollar mistake discussed earlier impossible.

### Catch-All Patterns and the `_` Placeholder
Using enums, we can also take special actions for a few particular values, but for all other values take one default action. Imagine we’re implementing a game where, if you roll a 3 on a dice roll, your player doesn’t move, but instead gets a new fancy hat. If you roll a 7, your player loses a fancy hat. For all other values, your player moves that number of spaces on the game board. Here’s a `match` that implements that logic, with the result of the dice roll hardcoded rather than a random value, and all other logic represented by functions without bodies because actually implementing them is out of scope for this example:

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

For the first two arms, the patterns are the literal values `3` and `7`. For the last arm that covers every other possible value, the pattern is the variable we’ve chosen to name `other`. The code that runs for the `other` arm uses the variable by passing it to the `move_player` function.
>  我们可以用一个变量名例如 `other` 绑定所有上面的 patterns 都不匹配的可能结果，然后在 arm 中使用它

This code compiles, even though we haven’t listed all the possible values a `u8` can have, because the last pattern will match all values not specifically listed. This catch-all pattern meets the requirement that `match` must be exhaustive. Note that we have to put the catch-all arm last because the patterns are evaluated in order. If we put the catch-all arm earlier, the other arms would never run, so Rust will warn us if we add arms after a catch-all!
>  catch-all arm 必须放在最后

Rust also has a pattern we can use when we want a catch-all but don’t want to _use_ the value in the catch-all pattern: `_` is a special pattern that matches any value and does not bind to that value. This tells Rust we aren’t going to use the value, so Rust won’t warn us about an unused variable.
>  使用 pattern `_` 也可以指定 catch-all pattern，但不绑定 value

Let’s change the rules of the game: now, if you roll anything other than a 3 or a 7, you must roll again. We no longer need to use the catch-all value, so we can change our code to use `_` instead of the variable named `other`:

```rust
let dice_roll = 9;
match dice_roll {
    3 => add_fancy_hat(),
    7 => remove_fancy_hat(),
    _ => reroll(),
}

fn add_fancy_hat() {}
fn remove_fancy_hat() {}
fn reroll() {}
```

This example also meets the exhaustiveness requirement because we’re explicitly ignoring all other values in the last arm; we haven’t forgotten anything.

Finally, we’ll change the rules of the game one more time so that nothing else happens on your turn if you roll anything other than a 3 or a 7. We can express that by using the unit value (the empty tuple type we mentioned in [“The Tuple Type”](https://doc.rust-lang.org/book/ch03-02-data-types.html#the-tuple-type) section) as the code that goes with the `_` arm:
>  如果我们在 arm 中不想返回任何值，可以直接写 unit value (empty tuple type)

```rust
let dice_roll = 9;
match dice_roll {
    3 => add_fancy_hat(),
    7 => remove_fancy_hat(),
    _ => (),
}

fn add_fancy_hat() {}
fn remove_fancy_hat() {}
```

Here, we’re telling Rust explicitly that we aren’t going to use any other value that doesn’t match a pattern in an earlier arm, and we don’t want to run any code in this case.

There’s more about patterns and matching that we’ll cover in [Chapter 19](https://doc.rust-lang.org/book/ch19-00-patterns.html). For now, we’re going to move on to the `if let` syntax, which can be useful in situations where the `match` expression is a bit wordy.

## 6.3 Concise Control Flow with `if let` and `let else`
The `if let` syntax lets you combine `if` and `let` into a less verbose way to handle values that match one pattern while ignoring the rest. Consider the program in Listing 6-6 that matches on an `Option<u8>` value in the `config_max` variable but only wants to execute code if the value is the `Some` variant.

```rust
let config_max = Some(3u8);
match config_max {
    Some(max) => println!("The maximum is configured to be {max}"),
    _ => (),
}
```

[Listing 6-6](https://doc.rust-lang.org/book/ch06-03-if-let.html#listing-6-6): A `match` that only cares about executing code when the value is `Some`

If the value is `Some`, we print out the value in the `Some` variant by binding the value to the variable `max` in the pattern. We don’t want to do anything with the `None` value. To satisfy the `match` expression, we have to add `_ => ()` after processing just one variant, which is annoying boilerplate code to add.

Instead, we could write this in a shorter way using `if let`. The following code behaves the same as the `match` in Listing 6-6:

```rust
let config_max = Some(3u8);
if let Some(max) = config_max {
    println!("The maximum is configured to be {max}");
}
```

The syntax `if let` takes a pattern and an expression separated by an equal sign. It works the same way as a `match`, where the expression is given to the `match` and the pattern is its first arm. In this case, the pattern is `Some(max)`, and the `max` binds to the value inside the `Some`. We can then use `max` in the body of the `if let` block in the same way we used `max` in the corresponding `match` arm. The code in the `if let` block only runs if the value matches the pattern.
>  `if let` 接收一个 pattern 和一个表达式，二者用 `=` 隔开
>  如果表达式匹配 pattern，执行 code

Using `if let` means less typing, less indentation, and less boilerplate code. However, you lose the exhaustive checking `match` enforces that ensures you aren’t forgetting to handle any cases. Choosing between `match` and `if let` depends on what you’re doing in your particular situation and whether gaining conciseness is an appropriate trade-off for losing exhaustive checking.

In other words, you can think of `if let` as syntax sugar for a `match` that runs code when the value matches one pattern and then ignores all other values.
>  可以将 `if let` 视作 `match` 的一个语法糖，它在 value 匹配 pattern 时运行代码，在 value 不匹配时什么事也不做

We can include an `else` with an `if let`. The block of code that goes with the `else` is the same as the block of code that would go with the `_` case in the `match` expression that is equivalent to the `if let` and `else`. Recall the `Coin` enum definition in Listing 6-4, where the `Quarter` variant also held a `UsState` value. If we wanted to count all non-quarter coins we see while also announcing the state of the quarters, we could do that with a `match` expression, like this:
>  `if let` 也可以搭配 `else`，等价于 `match` 表达式中进入 `_` 的情况

```rust
let mut count = 0;
match coin {
    Coin::Quarter(state) => println!("State quarter from {state:?}!"),
    _ => count += 1,
}
```

Or we could use an `if let` and `else` expression, like this:

```rust
let mut count = 0;
if let Coin::Quarter(state) = coin {
    println!("State quarter from {state:?}!");
} else {
    count += 1;
}
```

### Staying on the “Happy Path” with `let...else`
The common pattern is to perform some computation when a value is present and return a default value otherwise. Continuing on with our example of coins with a `UsState` value, if we wanted to say something funny depending on how old the state on the quarter was, we might introduce a method on `UsState` to check the age of a state, like so:

```rust
impl UsState {
    fn existed_in(&self, year: u16) -> bool {
        match self {
            UsState::Alabama => year >= 1819,
            UsState::Alaska => year >= 1959,
            // -- snip --
        }
    }
}

```

Then we might use `if let` to match on the type of coin, introducing a `state` variable within the body of the condition, as in Listing 6-7.

```rust
fn describe_state_quarter(coin: Coin) -> Option<String> {
    if let Coin::Quarter(state) = coin {
        if state.existed_in(1900) {
            Some(format!("{state:?} is pretty old, for America!"))
        } else {
            Some(format!("{state:?} is relatively new."))
        }
    } else {
        None
    }
}
```

[Listing 6-7](https://doc.rust-lang.org/book/ch06-03-if-let.html#listing-6-7): Checking whether a state existed in 1900 by using conditionals nested inside an `if let`.

That gets the job done, but it has pushed the work into the body of the `if let` statement, and if the work to be done is more complicated, it might be hard to follow exactly how the top-level branches relate. We could also take advantage of the fact that expressions produce a value either to produce the `state` from the `if let` or to return early, as in Listing 6-8. (You could do similar with a `match`, too.)

```rust
fn describe_state_quarter(coin: Coin) -> Option<String> {
    let state = if let Coin::Quarter(state) = coin {
        state
    } else {
        return None;
    };

    if state.existed_in(1900) {
        Some(format!("{state:?} is pretty old, for America!"))
    } else {
        Some(format!("{state:?} is relatively new."))
    }
}
```

[Listing 6-8](https://doc.rust-lang.org/book/ch06-03-if-let.html#listing-6-8): Using `if let` to produce a value or return early.

This is a bit annoying to follow in its own way, though! One branch of the `if let` produces a value, and the other one returns from the function entirely.

To make this common pattern nicer to express, Rust has `let...else`. The `let...else` syntax takes a pattern on the left side and an expression on the right, very similar to `if let`, but it does not have an `if` branch, only an `else` branch. If the pattern matches, it will bind the value from the pattern in the outer scope. If the pattern does _not_ match, the program will flow into the `else` arm, which must return from the function.
>  `let...else` 语法接收左边的一个 pattern 和右边的一个表达式，非常类似于 `if let`，但它没有 `if` 分支，只有 `else` 分支
>  如果 pattern 匹配表达式，该语法将 pattern 中的值绑定到外部作用域中
>  如果 pattern 不匹配表达式，程序执行 `else` 分支，该分支必须从函数中返回

In Listing 6-9, you can see how Listing 6-8 looks when using `let...else` in place of `if let`.

```rust
fn describe_state_quarter(coin: Coin) -> Option<String> {
    let Coin::Quarter(state) = coin else {
        return None;
    };

    if state.existed_in(1900) {
        Some(format!("{state:?} is pretty old, for America!"))
    } else {
        Some(format!("{state:?} is relatively new."))
    }
}
```

[Listing 6-9](https://doc.rust-lang.org/book/ch06-03-if-let.html#listing-6-9): Using `let...else` to clarify the flow through the function.

Notice that it stays “on the happy path” in the main body of the function this way, without having significantly different control flow for two branches the way the `if let` did.

If you have a situation in which your program has logic that is too verbose to express using a `match`, remember that `if let` and `let...else` are in your Rust toolbox as well.

## Summary
We’ve now covered how to use enums to create custom types that can be one of a set of enumerated values. We’ve shown how the standard library’s `Option<T>` type helps you use the type system to prevent errors. When enum values have data inside them, you can use `match` or `if let` to extract and use those values, depending on how many cases you need to handle.

Your Rust programs can now express concepts in your domain using structs and enums. Creating custom types to use in your API ensures type safety: the compiler will make certain your functions only get values of the type each function expects.

In order to provide a well-organized API to your users that is straightforward to use and only exposes exactly what your users will need, let’s now turn to Rust’s modules.

# 7 Managing Growing Projects with Packages, Crates, and Modules
As you write large programs, organizing your code will become increasingly important. By grouping related functionality and separating code with distinct features, you’ll clarify where to find code that implements a particular feature and where to go to change how a feature works.

The programs we’ve written so far have been in one module in one file. As a project grows, you should organize code by splitting it into multiple modules and then multiple files. A package can contain multiple binary crates and optionally one library crate. As a package grows, you can extract parts into separate crates that become external dependencies. This chapter covers all these techniques. For very large projects comprising a set of interrelated packages that evolve together, Cargo provides _workspaces_, which we’ll cover in [“Cargo Workspaces”](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html) in Chapter 14.
>  我们目前写的程序都位于单个模块，单个文件中
>  随着项目增长，我们应该将代码分到多个模块和多个文件中
>  一个 package 可以包含多个 binary crates，以及一个 library crate
>  随着 package 增大，我们可以将 package 的部分提取为单独的 cates，作为外部依赖
>  对于非常大的，包含了一组互相关联的 packages 的项目，Cargo 提供了 workspaces

We’ll also discuss encapsulating implementation details, which lets you reuse code at a higher level: once you’ve implemented an operation, other code can call your code via its public interface without having to know how the implementation works. The way you write code defines which parts are public for other code to use and which parts are private implementation details that you reserve the right to change. This is another way to limit the amount of detail you have to keep in your head.

A related concept is scope: the nested context in which code is written has a set of names that are defined as “in scope.” When reading, writing, and compiling code, programmers and compilers need to know whether a particular name at a particular spot refers to a variable, function, struct, enum, module, constant, or other item and what that item means. You can create scopes and change which names are in or out of scope. You can’t have two items with the same name in the same scope; tools are available to resolve name conflicts.

Rust has a number of features that allow you to manage your code’s organization, including which details are exposed, which details are private, and what names are in each scope in your programs. These features, sometimes collectively referred to as the _module system_, include:

- **Packages**: A Cargo feature that lets you build, test, and share crates
- **Crates**: A tree of modules that produces a library or executable
- **Modules and use**: Let you control the organization, scope, and privacy of paths
- **Paths**: A way of naming an item, such as a struct, function, or module

>  Rust 提供的用于组织代码的特性被称为模块系统，它包括了:
>  - packages: 用于构建、测试、共享 crates
>  - crates: a tree of modules，被编译生成一个库或可执行文件
>  - modules: 用于控制 paths 的组织，作用域和私有性
>  - paths: 命名一个 item 的方式，例如 struct, function, module

In this chapter, we’ll cover all these features, discuss how they interact, and explain how to use them to manage scope. By the end, you should have a solid understanding of the module system and be able to work with scopes like a pro!

## 7.1 Packages and Crates
The first parts of the module system we’ll cover are packages and crates.

A _crate_ is the smallest amount of code that the Rust compiler considers at a time. Even if you run `rustc` rather than `cargo` and pass a single source code file (as we did all the way back in “Writing and Running a Rust Program” in Chapter 1), the compiler considers that file to be a crate. Crates can contain modules, and the modules may be defined in other files that get compiled with the crate, as we’ll see in the coming sections.
>  crate 是 Rust 编译器一次处理的最小代码单元
>  当我们使用 `rustc`，并传递单个源文件时，编译器就将这个文件视作一个 crate
>  crate 可以包含 modules, modules 也可以定义在和 crate 一起编译的其他文件中

A crate can come in one of two forms: a binary crate or a library crate. _Binary crates_ are programs you can compile to an executable that you can run, such as a command line program or a server. Each must have a function called `main` that defines what happens when the executable runs. All the crates we’ve created so far have been binary crates.
>  crate 有两种形式: 二进制 crate 或库 crate
>  二进制 crate 是可以编译为可执行文件的程序，它必须有一个 `main` 函数，定义程序运行时的行为

_Library crates_ don’t have a `main` function, and they don’t compile to an executable. Instead, they define functionality intended to be shared with multiple projects. For example, the `rand` crate we used in [Chapter 2](https://doc.rust-lang.org/book/ch02-00-guessing-game-tutorial.html#generating-a-random-number) provides functionality that generates random numbers. Most of the time when Rustaceans say “crate,” they mean library crate, and they use “crate” interchangeably with the general programming concept of a “library.”
>  库 crate 没有 `main` 函数，不会编译为一个可执行文件，它们定义了旨在多个项目之间共享的功能
>  例如 `rand` crate 提供了生成随机数的功能，大多数情况下，crate 指库 crate，且 crate 和 library 同义

The _crate root_ is a source file that the Rust compiler starts from and makes up the root module of your crate (we’ll explain modules in depth in [“Defining Modules to Control Scope and Privacy”](https://doc.rust-lang.org/book/ch07-02-defining-modules-to-control-scope-and-privacy.html)).
>  crate root 为 Rust 编译器**开始编译**的源文件
>  它构成了我们 crate 的 root module

A _package_ is a bundle of one or more crates that provides a set of functionality. A package contains a _Cargo.toml_ file that describes how to build those crates. Cargo is actually a package that contains the binary crate for the command line tool you’ve been using to build your code. The Cargo package also contains a library crate that the binary crate depends on. Other projects can depend on the Cargo library crate to use the same logic the Cargo command line tool uses.
>  package 是一个或多个 crate 的功能集合
>  package 包含一个 `Cargo.toml` 文件，描述了如何构建这些 crates
>  Cargo 本身就是一个 package，包含了我们用于构建代码的命令行工具，也包含了一个 libaray crate，被它的 binary crate 所依赖
>  其他项目可以依赖于 Cargo package 的 library crate，以使用和 Cargo 命令行工具相同的逻辑

A package can contain as many binary crates as you like, but at most only one library crate. A package must contain at least one crate, whether that’s a library or binary crate.
>  package 可以包含任意多的 binary crate，但只能包含一个 library crate
>  package 至少需要包含一个 crate

Let’s walk through what happens when we create a package. First we enter the command `cargo new my-project`:

```
$ cargo new my-project
     Created binary (application) `my-project` package
$ ls my-project
Cargo.toml
src
$ ls my-project/src
main.rs
```

After we run `cargo new my-project`, we use `ls` to see what Cargo creates. In the project directory, there’s a _Cargo.toml_ file, giving us a package. There’s also a _src_ directory that contains _main.rs_. Open _Cargo.toml_ in your text editor, and note there’s no mention of _src/main.rs_. Cargo follows a convention that _src/main.rs_ is the crate root of a binary crate with the same name as the package. Likewise, Cargo knows that if the package directory contains _src/lib.rs_, the package contains a library crate with the same name as the package, and _src/lib.rs_ is its crate root. Cargo passes the crate root files to `rustc` to build the library or binary.
>  `Cargo.toml` 中没有提到 `src/main.rs`，Cargo 默认认为 `src/main.rs` 是一个和 package 同名的 binary crate 的 crate root
>  类似地，Crago 默认认为 `src/lib.rs` 是一个和 package 同名的 lib crate 的 crate root
>  Cargo 会将 crate root 文件传递给 `rustc` 来构建 library 或 binary

Here, we have a package that only contains _src/main.rs_, meaning it only contains a binary crate named `my-project`. If a package contains _src/main.rs_ and _src/lib.rs_, it has two crates: a binary and a library, both with the same name as the package. A package can have multiple binary crates by placing files in the _src/bin_ directory: each file will be a separate binary crate.
>  如果 package 包含了 `src/main.rs, src/lib.rs`，它就包含了两个 crate，各自名字都和 package 相同
>  `src/bin` 目录下的每个文件都被视作一个单独的 binary crate

## 7.2 Defining Modules to Control Scope and Privacy
In this section, we’ll talk about modules and other parts of the module system, namely _paths_, which allow you to name items; the `use` keyword that brings a path into scope; and the `pub` keyword to make items public. We’ll also discuss the `as` keyword, external packages, and the glob operator.
>  paths 允许我们命名 item
>  `use` 关键字将 path 代入作用域，`pub` 关键字让 item 公有

### Modules Cheat Sheet
Before we get to the details of modules and paths, here we provide a quick reference on how modules, paths, the `use` keyword, and the `pub` keyword work in the compiler, and how most developers organize their code. We’ll be going through examples of each of these rules throughout this chapter, but this is a great place to refer to as a reminder of how modules work.

- **Start from the crate root**: When compiling a crate, the compiler first looks in the crate root file (usually _src/lib.rs_ for a library crate or _src/main.rs_ for a binary crate) for code to compile.
>  编译 crate 时，编译器首先查看 crate root file，通常是 `src/lib.rs, src/main.rs`

- **Declaring modules**: In the crate root file, you can declare new modules; say you declare a “garden” module with `mod garden;`. The compiler will look for the module’s code in these places:
    - Inline, within curly brackets that replace the semicolon following `mod garden`
    - In the file _src/garden.rs_
    - In the file _src/garden/mod.rs_
>  crate root file 中可以声明新的 modules，例如 `mod xxx`
>  编译器会在这些地方查找该 module 的代码:
>  - 内联: 在 `mod xxx{}` 中的 `{}` 查找
>  - 在 `src/xxx.rs` 中查找
>  - 在 `src/xxx/mod.rs` 中查找

- **Declaring submodules**: In any file other than the crate root, you can declare submodules. For example, you might declare `mod vegetables;` in _src/garden.rs_. The compiler will look for the submodule’s code within the directory named for the parent module in these places:
    - Inline, directly following `mod vegetables`, within curly brackets instead of the semicolon
    - In the file _src/garden/vegetables.rs_
    - In the file _src/garden/vegetables/mod.rs_
>  在除了 crate root 的其他文件中，可以声明 submodules
>  例如可以在 `src/xxx.rs` 中声明 `mod yyy`，编译器会在其 parent module 的名字相同的目录中的以下地方寻找 submodule 的代码:
>  - 内联: 在 `mod yyy{}` 中的 `{}` 查找
>  - 在文件 `src/xxx/yyy.rs` 中
>  - 在文件 `src/xxx/yyy/mod.rs` 中

- **Paths to code in modules**: Once a module is part of your crate, you can refer to code in that module from anywhere else in that same crate, as long as the privacy rules allow, using the path to the code. For example, an `Asparagus` type in the garden vegetables module would be found at `crate::garden::vegetables::Asparagus`.
>  如果 module 是 crate 的一部分时，我们可以在 crate 的任意地方，使用 path 引用 module 中的代码
>  例如 `xxx/yyy` module 中的 `zzz` 类型可以被 `crate::xxx::yyy::zzz` 访问

- **Private vs. public**: Code within a module is private from its parent modules by default. To make a module public, declare it with `pub mod` instead of `mod`. To make items within a public module public as well, use `pub` before their declarations.
>  module 中的代码默认对于它的 parent module 是私有的
>  要让 module 公有，需要用 `pub mod` 声明而不是 `mod`
>  要让公有 module 中的 items 公有，需要在它的声明之前使用 `pub`

- **The `use` keyword**: Within a scope, the `use` keyword creates shortcuts to items to reduce repetition of long paths. In any scope that can refer to `crate::garden::vegetables::Asparagus`, you can create a shortcut with `use crate::garden::vegetables::Asparagus;` and from then on you only need to write `Asparagus` to make use of that type in the scope.
>  在一个作用域内，`use` 关键字会创建对某个 item 的捷径，而不需要写长路径
>  例如，我们使用 `use crate::xxx::yyy::zzz`，之后我们就可以直接在作用域中使用 `zzz`

Here, we create a binary crate named `backyard` that illustrates these rules. The crate’s directory, also named `backyard`, contains these files and directories:

```
backyard
├── Cargo.lock
├── Cargo.toml
└── src
    ├── garden
    │   └── vegetables.rs
    ├── garden.rs
    └── main.rs
```

The crate root file in this case is _src/main.rs_, and it contains:

Filename: src/main.rs

```rust
use crate::garden::vegetables::Asparagus;

pub mod garden;

fn main() {
    let plant = Asparagus {};
    println!("I'm growing {plant:?}!");
}
```

The `pub mod garden;` line tells the compiler to include the code it finds in _src/garden.rs_, which is:

Filename: src/garden.rs

```rust
pub mod vegetables; 
```

Here, `pub mod vegetables;` means the code in _src/garden/vegetables.rs_ is included too. That code is:

```rust
#[derive(Debug)] 
pub struct Asparagus {}
```

Now let’s get into the details of these rules and demonstrate them in action!

### Grouping Related Code in Modules
_Modules_ let us organize code within a crate for readability and easy reuse. Modules also allow us to control the _privacy_ of items because code within a module is private by default. Private items are internal implementation details not available for outside use. We can choose to make modules and the items within them public, which exposes them to allow external code to use and depend on them.
>  modules 除了用于组织 crate 内的代码以外，还允许我们控制 items 的私有性
>  module 内的代码默认是私有，即不提供外部使用接口的内部实现
>  我们可以让 modules 和 module 内的 item 公有

As an example, let’s write a library crate that provides the functionality of a restaurant. We’ll define the signatures of functions but leave their bodies empty to concentrate on the organization of the code rather than the implementation of a restaurant.

In the restaurant industry, some parts of a restaurant are referred to as _front of house_ and others as _back of house_. Front of house is where customers are; this encompasses where the hosts seat customers, servers take orders and payment, and bartenders make drinks. Back of house is where the chefs and cooks work in the kitchen, dishwashers clean up, and managers do administrative work.

To structure our crate in this way, we can organize its functions into nested modules. Create a new library named `restaurant` by running `cargo new restaurant --lib`. Then enter the code in Listing 7-1 into _src/lib.rs_ to define some modules and function signatures; this code is the front of house section.

Filename: src/lib.rs

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

[Listing 7-1](https://doc.rust-lang.org/book/ch07-02-defining-modules-to-control-scope-and-privacy.html#listing-7-1): A `front_of_house` module containing other modules that then contain functions

We define a module with the `mod` keyword followed by the name of the module (in this case, `front_of_house`). The body of the module then goes inside curly brackets. Inside modules, we can place other modules, as in this case with the modules `hosting` and `serving`. Modules can also hold definitions for other items, such as structs, enums, constants, traits, and as in Listing 7-1, functions.
>  我们通过 `mod + module_name` 来定义 module
>  并且我们直接在 `{}` 中提供了 module 的代码
>  在 module 内，我们可以定义其他 module

By using modules, we can group related definitions together and name why they’re related. Programmers using this code can navigate the code based on the groups rather than having to read through all the definitions, making it easier to find the definitions relevant to them. Programmers adding new functionality to this code would know where to place the code to keep the program organized.

Earlier, we mentioned that _src/main.rs_ and _src/lib.rs_ are called crate roots. The reason for their name is that the contents of either of these two files form a module named `crate` at the root of the crate’s module structure, known as the _module tree_.
>  `src/main.rs, src/lib.rs` 被称为 crate roots 的原因是这两个文件的内容会在 crate 的 module structure/module tree 的根处构成一个名为 `crate` 的 module

Listing 7-2 shows the module tree for the structure in Listing 7-1.

```
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

[Listing 7-2](https://doc.rust-lang.org/book/ch07-02-defining-modules-to-control-scope-and-privacy.html#listing-7-2): The module tree for the code in Listing 7-1

This tree shows how some of the modules nest inside other modules; for example, `hosting` nests inside `front_of_house`. The tree also shows that some modules are _siblings_, meaning they’re defined in the same module; `hosting` and `serving` are siblings defined within `front_of_house`. If module A is contained inside module B, we say that module A is the _child_ of module B and that module B is the _parent_ of module A. Notice that the entire module tree is rooted under the implicit module named `crate`.
>  crate 的整个 module tree 都位于隐式的 module `crate` 下

The module tree might remind you of the filesystem’s directory tree on your computer; this is a very apt comparison! Just like directories in a filesystem, you use modules to organize your code. And just like files in a directory, we need a way to find our modules.

## 7.3 Paths for Referring to an Item in the Module Tree
To show Rust where to find an item in a module tree, we use a path in the same way we use a path when navigating a filesystem. To call a function, we need to know its path.
>  为了让 Rust 知道在如何在 module tree 中找到 item，我们使用 path，形式和在文件系统中使用路径是类似的

A path can take two forms:

- An _absolute path_ is the full path starting from a crate root; for code from an external crate, the absolute path begins with the crate name, and for code from the current crate, it starts with the literal `crate`.
- A _relative path_ starts from the current module and uses `self`, `super`, or an identifier in the current module.

>  path 可以是
>  - 绝对路径: 从 `crate` 根开始，对于外部 crate 的代码，绝对路径从其 crate 的名字开始
>  - 相对路径: 从当前 module 开始，使用 `self, super` 或者当前 module 中的标识符

Both absolute and relative paths are followed by one or more identifiers separated by double colons (`::`).

Returning to Listing 7-1, say we want to call the `add_to_waitlist` function. This is the same as asking: what’s the path of the `add_to_waitlist` function? Listing 7-3 contains Listing 7-1 with some of the modules and functions removed.

We’ll show two ways to call the `add_to_waitlist` function from a new function, `eat_at_restaurant`, defined in the crate root. These paths are correct, but there’s another problem remaining that will prevent this example from compiling as is. We’ll explain why in a bit.

The `eat_at_restaurant` function is part of our library crate’s public API, so we mark it with the `pub` keyword. In the [“Exposing Paths with the `pub` Keyword”](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html#exposing-paths-with-the-pub-keyword) section, we’ll go into more detail about `pub`.

Filename: src/lib.rs

```rust
mod front_of_house {
    mod hosting {
        fn add_to_waitlist() {}
    }
}

pub fn eat_at_restaurant() {
    // Absolute path
    crate::front_of_house::hosting::add_to_waitlist();

    // Relative path
    front_of_house::hosting::add_to_waitlist();
}
```

[Listing 7-3](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html#listing-7-3): Calling the `add_to_waitlist` function using absolute and relative paths

The first time we call the `add_to_waitlist` function in `eat_at_restaurant`, we use an absolute path. The `add_to_waitlist` function is defined in the same crate as `eat_at_restaurant`, which means we can use the `crate` keyword to start an absolute path. We then include each of the successive modules until we make our way to `add_to_waitlist`. You can imagine a filesystem with the same structure: we’d specify the path `/front_of_house/hosting/add_to_waitlist` to run the `add_to_waitlist` program; using the `crate` name to start from the crate root is like using `/` to start from the filesystem root in your shell.

The second time we call `add_to_waitlist` in `eat_at_restaurant`, we use a relative path. The path starts with `front_of_house`, the name of the module defined at the same level of the module tree as `eat_at_restaurant`. Here the filesystem equivalent would be using the path `front_of_house/hosting/add_to_waitlist`. Starting with a module name means that the path is relative.
>  从一个 module name 开始，就意味着 path 是相对路径

Choosing whether to use a relative or absolute path is a decision you’ll make based on your project, and it depends on whether you’re more likely to move item definition code separately from or together with the code that uses the item. For example, if we moved the `front_of_house` module and the `eat_at_restaurant` function into a module named `customer_experience`, we’d need to update the absolute path to `add_to_waitlist`, but the relative path would still be valid. However, if we moved the `eat_at_restaurant` function separately into a module named `dining`, the absolute path to the `add_to_waitlist` call would stay the same, but the relative path would need to be updated. Our preference in general is to specify absolute paths because it’s more likely we’ll want to move code definitions and item calls independently of each other.
>  通常我们倾向于使用绝对路径，因为我们通常不会将代码定义和 item 调用一起移动，而是独立移动

Let’s try to compile Listing 7-3 and find out why it won’t compile yet! The errors we get are shown in Listing 7-4.

```
$ cargo build
   Compiling restaurant v0.1.0 (file:///projects/restaurant)
error[E0603]: module `hosting` is private
 --> src/lib.rs:9:28
  |
9 |     crate::front_of_house::hosting::add_to_waitlist();
  |                            ^^^^^^^  --------------- function `add_to_waitlist` is not publicly re-exported
  |                            |
  |                            private module
  |
note: the module `hosting` is defined here
 --> src/lib.rs:2:5
  |
2 |     mod hosting {
  |     ^^^^^^^^^^^

error[E0603]: module `hosting` is private
  --> src/lib.rs:12:21
   |
12 |     front_of_house::hosting::add_to_waitlist();
   |                     ^^^^^^^  --------------- function `add_to_waitlist` is not publicly re-exported
   |                     |
   |                     private module
   |
note: the module `hosting` is defined here
  --> src/lib.rs:2:5
   |
2  |     mod hosting {
   |     ^^^^^^^^^^^

For more information about this error, try `rustc --explain E0603`.
error: could not compile `restaurant` (lib) due to 2 previous errors
```

[Listing 7-4](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html#listing-7-4): Compiler errors from building the code in Listing 7-3

The error messages say that module `hosting` is private. In other words, we have the correct paths for the `hosting` module and the `add_to_waitlist` function, but Rust won’t let us use them because it doesn’t have access to the private sections. In Rust, all items (functions, methods, structs, enums, modules, and constants) are private to parent modules by default. If you want to make an item like a function or struct private, you put it in a module.
>  Rust 中，所有的 items (functions, methods, structs, enums, modules, constants) 默认对于 parent module 都是私有的

Items in a parent module can’t use the private items inside child modules, but items in child modules can use the items in their ancestor modules. This is because child modules wrap and hide their implementation details, but the child modules can see the context in which they’re defined. To continue with our metaphor, think of the privacy rules as being like the back office of a restaurant: what goes on in there is private to restaurant customers, but office managers can see and do everything in the restaurant they operate.
>  parent module 的 items 不能使用 child module 中的私有 item，但 child module 中的 items 可以使用其祖先 module 的 items，因为 child module 封装了其实现细节，但是可以看到它被定义的上下文

Rust chose to have the module system function this way so that hiding inner implementation details is the default. That way, you know which parts of the inner code you can change without breaking outer code. However, Rust does give you the option to expose inner parts of child modules’ code to outer ancestor modules by using the `pub` keyword to make an item public.

### Exposing Paths with the `pub` Keyword
Let’s return to the error in Listing 7-4 that told us the `hosting` module is private. We want the `eat_at_restaurant` function in the parent module to have access to the `add_to_waitlist` function in the child module, so we mark the `hosting` module with the `pub` keyword, as shown in Listing 7-5.
>  如果我们希望 parent module 访问 child module 中的 item，我们需要将 child module 首先标记为 `pub`

Filename: src/lib.rs

```rust
mod front_of_house {
    pub mod hosting {
        fn add_to_waitlist() {}
    }
}

// -- snip --
```

[Listing 7-5](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html#listing-7-5): Declaring the `hosting` module as `pub` to use it from `eat_at_restaurant`

Unfortunately, the code in Listing 7-5 still results in compiler errors, as shown in Listing 7-6.

```
$ cargo build
   Compiling restaurant v0.1.0 (file:///projects/restaurant)
error[E0603]: function `add_to_waitlist` is private
  --> src/lib.rs:10:37
   |
10 |     crate::front_of_house::hosting::add_to_waitlist();
   |                                     ^^^^^^^^^^^^^^^ private function
   |
note: the function `add_to_waitlist` is defined here
  --> src/lib.rs:3:9
   |
3  |         fn add_to_waitlist() {}
   |         ^^^^^^^^^^^^^^^^^^^^

error[E0603]: function `add_to_waitlist` is private
  --> src/lib.rs:13:30
   |
13 |     front_of_house::hosting::add_to_waitlist();
   |                              ^^^^^^^^^^^^^^^ private function
   |
note: the function `add_to_waitlist` is defined here
  --> src/lib.rs:3:9
   |
3  |         fn add_to_waitlist() {}
   |         ^^^^^^^^^^^^^^^^^^^^

For more information about this error, try `rustc --explain E0603`.
error: could not compile `restaurant` (lib) due to 2 previous errors
```

[Listing 7-6](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html#listing-7-6): Compiler errors from building the code in Listing 7-5

What happened? Adding the `pub` keyword in front of `mod hosting` makes the module public. With this change, if we can access `front_of_house`, we can access `hosting`. But the _contents_ of `hosting` are still private; making the module public doesn’t make its contents public. The `pub` keyword on a module only lets code in its ancestor modules refer to it, not access its inner code. Because modules are containers, there’s not much we can do by only making the module public; we need to go further and choose to make one or more of the items within the module public as well.
>  但是仅仅让 module 为公有还不够，我们还需要让 module 的内容公有
>  因此我们还需要将 `hosting` 函数记作 `pub`

The errors in Listing 7-6 say that the `add_to_waitlist` function is private. The privacy rules apply to structs, enums, functions, and methods as well as modules.

Let’s also make the `add_to_waitlist` function public by adding the `pub` keyword before its definition, as in Listing 7-7.

Filename: src/lib.rs

```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

// -- snip --
```

[Listing 7-7](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html#listing-7-7): Adding the `pub` keyword to `mod hosting` and `fn add_to_waitlist` lets us call the function from `eat_at_restaurant`

Now the code will compile! To see why adding the `pub` keyword lets us use these paths in `eat_at_restaurant` with respect to the privacy rules, let’s look at the absolute and the relative paths.

In the absolute path, we start with `crate`, the root of our crate’s module tree. The `front_of_house` module is defined in the crate root. While `front_of_house` isn’t public, because the `eat_at_restaurant` function is defined in the same module as `front_of_house` (that is, `eat_at_restaurant` and `front_of_house` are siblings), we can refer to `front_of_house` from `eat_at_restaurant`. Next is the `hosting` module marked with `pub`. We can access the parent module of `hosting`, so we can access `hosting`. Finally, the `add_to_waitlist` function is marked with `pub` and we can access its parent module, so this function call works!

In the relative path, the logic is the same as the absolute path except for the first step: rather than starting from the crate root, the path starts from `front_of_house`. The `front_of_house` module is defined within the same module as `eat_at_restaurant`, so the relative path starting from the module in which `eat_at_restaurant` is defined works. Then, because `hosting` and `add_to_waitlist` are marked with `pub`, the rest of the path works, and this function call is valid!

If you plan on sharing your library crate so other projects can use your code, your public API is your contract with users of your crate that determines how they can interact with your code. There are many considerations around managing changes to your public API to make it easier for people to depend on your crate. These considerations are beyond the scope of this book; if you’re interested in this topic, see [The Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).

#### Best Practices for Packages with a Binary and a Library
We mentioned that a package can contain both a _src/main.rs_ binary crate root as well as a _src/lib.rs_ library crate root, and both crates will have the package name by default. Typically, packages with this pattern of containing both a library and a binary crate will have just enough code in the binary crate to start an executable that calls code defined in the library crate. This lets other projects benefit from the most functionality that the package provides because the library crate’s code can be shared.
>  package 可以同时包含一个 `src/main.rs` binary crate root 和 `src/lib.rs` library crate root，这两个 crate 默认的名字都是 package name
>  通常同时有 binary crate, library crate 的 package 中，它的 binary crate 的代码通常只是调用 library crate 中的功能，大部分功能定义位于 library crate，这样其他的项目也可以利用这个 library crate

The module tree should be defined in _src/lib.rs_. Then, any public items can be used in the binary crate by starting paths with the name of the package. The binary crate becomes a user of the library crate just like a completely external crate would use the library crate: it can only use the public API. This helps you design a good API; not only are you the author, you’re also a client!
>  对于同时有 binary crate, library crate 的 package，其 module tree 应该定义在 `src/lib.rs`，那么 binary crate 中，要使用 items，应该以 package 的名称作为 path 的起始
>  binary crate 就成为了 library crate 的用户，就像一个完全外部的 crate: 仅仅使用公共 API
>  这可以帮助我们设计好的 API，即我们不仅仅是 author，也要是 client

In [Chapter 12](https://doc.rust-lang.org/book/ch12-00-an-io-project.html), we’ll demonstrate this organizational practice with a command line program that will contain both a binary crate and a library crate.

### Starting Relative Paths with `super`
We can construct relative paths that begin in the parent module, rather than the current module or the crate root, by using `super` at the start of the path. This is like starting a filesystem path with the `..` syntax that means to go to the parent directory. Using `super` allows us to reference an item that we know is in the parent module, which can make rearranging the module tree easier when the module is closely related to the parent but the parent might be moved elsewhere in the module tree someday.
>  我们可以通过以 `super` 为路径起点，构建以 parent module 为起始的相对路径
>  使用 `super` 允许我们访问 parent module 中的 item

Consider the code in Listing 7-8 that models the situation in which a chef fixes an incorrect order and personally brings it out to the customer. The function `fix_incorrect_order` defined in the `back_of_house` module calls the function `deliver_order` defined in the parent module by specifying the path to `deliver_order`, starting with `super`.

Filename: src/lib.rs

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

[Listing 7-8](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html#listing-7-8): Calling a function using a relative path starting with `super`

The `fix_incorrect_order` function is in the `back_of_house` module, so we can use `super` to go to the parent module of `back_of_house`, which in this case is `crate`, the root. From there, we look for `deliver_order` and find it. Success! We think the `back_of_house` module and the `deliver_order` function are likely to stay in the same relationship to each other and get moved together should we decide to reorganize the crate’s module tree. Therefore, we used `super` so we’ll have fewer places to update code in the future if this code gets moved to a different module.

### Making Structs and Enums Public
We can also use `pub` to designate structs and enums as public, but there are a few extra details to the usage of `pub` with structs and enums. If we use `pub` before a struct definition, we make the struct public, but the struct’s fields will still be private. We can make each field public or not on a case-by-case basis. In Listing 7-9, we’ve defined a public `back_of_house::Breakfast` struct with a public `toast` field but a private `seasonal_fruit` field. This models the case in a restaurant where the customer can pick the type of bread that comes with a meal, but the chef decides which fruit accompanies the meal based on what’s in season and in stock. The available fruit changes quickly, so customers can’t choose the fruit or even see which fruit they’ll get.
>  让 struct 本身 `pub` 还不够，还需要将需要暴露的 field 也标记为 `pub`

Filename: src/lib.rs

```rust
mod back_of_house {
    pub struct Breakfast {
        pub toast: String,
        seasonal_fruit: String,
    }

    impl Breakfast {
        pub fn summer(toast: &str) -> Breakfast {
            Breakfast {
                toast: String::from(toast),
                seasonal_fruit: String::from("peaches"),
            }
        }
    }
}

pub fn eat_at_restaurant() {
    // Order a breakfast in the summer with Rye toast.
    let mut meal = back_of_house::Breakfast::summer("Rye");
    // Change our mind about what bread we'd like.
    meal.toast = String::from("Wheat");
    println!("I'd like {} toast please", meal.toast);

    // The next line won't compile if we uncomment it; we're not allowed
    // to see or modify the seasonal fruit that comes with the meal.
    // meal.seasonal_fruit = String::from("blueberries");
}
```

[Listing 7-9](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html#listing-7-9): A struct with some public fields and some private fields

Because the `toast` field in the `back_of_house::Breakfast` struct is public, in `eat_at_restaurant` we can write and read to the `toast` field using dot notation. Notice that we can’t use the `seasonal_fruit` field in `eat_at_restaurant`, because `seasonal_fruit` is private. Try uncommenting the line modifying the `seasonal_fruit` field value to see what error you get!

Also, note that because `back_of_house::Breakfast` has a private field, the struct needs to provide a public associated function that constructs an instance of `Breakfast` (we’ve named it `summer` here). If `Breakfast` didn’t have such a function, we couldn’t create an instance of `Breakfast` in `eat_at_restaurant` because we couldn’t set the value of the private `seasonal_fruit` field in `eat_at_restaurant`.
>  如果 struct 有 private filed，该 struct 必须提供一个 public 函数用于构造该 struct 的 instance，否则我们无法构造该 struct 的 instance，因为我们无法设定它的 private field

In contrast, if we make an enum public, all of its variants are then public. We only need the `pub` before the `enum` keyword, as shown in Listing 7-10.
>  但是相较之下，如果我们让枚举类型 public，则它的所有变体都会是 public

Filename: src/lib.rs

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

[Listing 7-10](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html#listing-7-10): Designating an enum as public makes all its variants public.

Because we made the `Appetizer` enum public, we can use the `Soup` and `Salad` variants in `eat_at_restaurant`.

Enums aren’t very useful unless their variants are public; it would be annoying to have to annotate all enum variants with `pub` in every case, so the default for enum variants is to be public. Structs are often useful without their fields being public, so struct fields follow the general rule of everything being private by default unless annotated with `pub`.
>  这是因为 enum 的用意一般就是让它的所有变体可见，因此默认 enum 的变体都是 public 的，而 struct 则相反，需要一些 field 不可见作为实现细节

There’s one more situation involving `pub` that we haven’t covered, and that is our last module system feature: the `use` keyword. We’ll cover `use` by itself first, and then we’ll show how to combine `pub` and `use`.

## 7.4 Bringing Paths into Scope with the `use` Keyword
Having to write out the paths to call functions can feel inconvenient and repetitive. In Listing 7-7, whether we chose the absolute or relative path to the `add_to_waitlist` function, every time we wanted to call `add_to_waitlist` we had to specify `front_of_house` and `hosting` too. Fortunately, there’s a way to simplify this process: we can create a shortcut to a path with the `use` keyword once, and then use the shorter name everywhere else in the scope.

In Listing 7-11, we bring the `crate::front_of_house::hosting` module into the scope of the `eat_at_restaurant` function so we only have to specify `hosting::add_to_waitlist` to call the `add_to_waitlist` function in `eat_at_restaurant`.
>  我们可以使用 `use` 将特定 module 代入作用域，以简化之后的 path 形式

Filename: src/lib.rs

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

[Listing 7-11](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#listing-7-11): Bringing a module into scope with `use`

Adding `use` and a path in a scope is similar to creating a symbolic link in the filesystem. By adding `use crate::front_of_house::hosting` in the crate root, `hosting` is now a valid name in that scope, just as though the `hosting` module had been defined in the crate root. Paths brought into scope with `use` also check privacy, like any other paths.
>  使用 `use` 之后，module name 就是当前作用域的一个有效名字

Note that `use` only creates the shortcut for the particular scope in which the `use` occurs. Listing 7-12 moves the `eat_at_restaurant` function into a new child module named `customer`, which is then a different scope than the `use` statement, so the function body won’t compile.
>  注意 `use` 仅在当前作用域有效

Filename: src/lib.rs

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

[Listing 7-12](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#listing-7-12): A `use` statement only applies in the scope it’s in.

The compiler error shows that the shortcut no longer applies within the `customer` module:

```
$ cargo build
   Compiling restaurant v0.1.0 (file:///projects/restaurant)
error[E0433]: failed to resolve: use of undeclared crate or module `hosting`
  --> src/lib.rs:11:9
   |
11 |         hosting::add_to_waitlist();
   |         ^^^^^^^ use of undeclared crate or module `hosting`
   |
help: consider importing this module through its public re-export
   |
10 +     use crate::hosting;
   |

warning: unused import: `crate::front_of_house::hosting`
 --> src/lib.rs:7:5
  |
7 | use crate::front_of_house::hosting;
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` on by default

For more information about this error, try `rustc --explain E0433`.
warning: `restaurant` (lib) generated 1 warning
error: could not compile `restaurant` (lib) due to 1 previous error; 1 warning emitted
```

Notice there’s also a warning that the `use` is no longer used in its scope! To fix this problem, move the `use` within the `customer` module too, or reference the shortcut in the parent module with `super::hosting` within the child `customer` module.

### Creating Idiomatic `use` Paths
In Listing 7-11, you might have wondered why we specified `use crate::front_of_house::hosting` and then called `hosting::add_to_waitlist` in `eat_at_restaurant`, rather than specifying the `use` path all the way out to the `add_to_waitlist` function to achieve the same result, as in Listing 7-13.

Filename: src/lib.rs

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

[Listing 7-13](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#listing-7-13): Bringing the `add_to_waitlist` function into scope with `use`, which is unidiomatic

Although both Listing 7-11 and Listing 7-13 accomplish the same task, Listing 7-11 is the idiomatic way to bring a function into scope with `use`. Bringing the function’s parent module into scope with `use` means we have to specify the parent module when calling the function. Specifying the parent module when calling the function makes it clear that the function isn’t locally defined while still minimizing repetition of the full path. The code in Listing 7-13 is unclear as to where `add_to_waitlist` is defined.
>  通常将函数引入作用域的惯用方式是将其 parent module 引入作用域，这是为了在使用函数时明确表示函数不是本地定义的

On the other hand, when bringing in structs, enums, and other items with `use`, it’s idiomatic to specify the full path. Listing 7-14 shows the idiomatic way to bring the standard library’s `HashMap` struct into the scope of a binary crate.
>  另一方面，将 structs, enums, 和其他 items 代入作用域时，通常会指定 `use` 完整路径

Filename: src/main.rs

```rust
use std::collections::HashMap;

fn main() {
    let mut map = HashMap::new();
    map.insert(1, 2);
}
```

[Listing 7-14](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#listing-7-14): Bringing `HashMap` into scope in an idiomatic way

There’s no strong reason behind this idiom: it’s just the convention that has emerged, and folks have gotten used to reading and writing Rust code this way.

The exception to this idiom is if we’re bringing two items with the same name into scope with `use` statements, because Rust doesn’t allow that. Listing 7-15 shows how to bring two `Result` types into scope that have the same name but different parent modules, and how to refer to them.
>  但如果两个 item 具有相同的名字，为了避免冲突，还是需要先 `use` 其 parent module，再使用这些 items

Filename: src/lib.rs

```rust
use std::fmt;
use std::io;

fn function1() -> fmt::Result {
    // --snip--
}

fn function2() -> io::Result<()> {
    // --snip--
}
```

[Listing 7-15](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#listing-7-15): Bringing two types with the same name into the same scope requires using their parent modules.

As you can see, using the parent modules distinguishes the two `Result` types. If instead we specified `use std::fmt::Result` and `use std::io::Result`, we’d have two `Result` types in the same scope, and Rust wouldn’t know which one we meant when we used `Result`.

### Providing New Names with the `as` Keyword
There’s another solution to the problem of bringing two types of the same name into the same scope with `use`: after the path, we can specify `as` and a new local name, or _alias_, for the type. Listing 7-16 shows another way to write the code in Listing 7-15 by renaming one of the two `Result` types using `as`.
>  使用 `use` 时，我们可以通过 `as` 来指定别名，这也是避免命名冲突的方式

Filename: src/lib.rs

```rust
use std::fmt::Result;
use std::io::Result as IoResult;

fn function1() -> Result {
    // --snip--
}

fn function2() -> IoResult<()> {
    // --snip--
}
```

[Listing 7-16](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#listing-7-16): Renaming a type when it’s brought into scope with the `as` keyword

In the second `use` statement, we chose the new name `IoResult` for the `std::io::Result` type, which won’t conflict with the `Result` from `std::fmt` that we’ve also brought into scope. Listing 7-15 and Listing 7-16 are considered idiomatic, so the choice is up to you!

### Re-exporting Names with `pub use`
When we bring a name into scope with the `use` keyword, the name is private to the scope into which we imported it. To enable code outside that scope to refer to that name as if it had been defined in that scope, we can combine `pub` and `use`. This technique is called _re-exporting_ because we’re bringing an item into scope but also making that item available for others to bring into their scope.
>  使用 `use` 将名字代入作用域时，该名字是私有于该作用域的
>  如果要让作用域之外也看到这个名字，需要使用 `pub use`
>  这个技巧称为重导出，因为我们将一个 item 带入作用域，同时让这个 item 对于其他地方可见，使得其他地方可以将这个 item 代入它们的作用域

Listing 7-17 shows the code in Listing 7-11 with `use` in the root module changed to `pub use`.

Filename: src/lib.rs

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

[Listing 7-17](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#listing-7-17): Making a name available for any code to use from a new scope with `pub use`

Before this change, external code would have to call the `add_to_waitlist` function by using the path `restaurant::front_of_house::hosting::add_to_waitlist()`, which also would have required the `front_of_house` module to be marked as `pub`. Now that this `pub use` has re-exported the `hosting` module from the root module, external code can use the path `restaurant::hosting::add_to_waitlist()` instead.

Re-exporting is useful when the internal structure of your code is different from how programmers calling your code would think about the domain. For example, in this restaurant metaphor, the people running the restaurant think about “front of house” and “back of house.” But customers visiting a restaurant probably won’t think about the parts of the restaurant in those terms. With `pub use`, we can write our code with one structure but expose a different structure. Doing so makes our library well organized for programmers working on the library and programmers calling the library. We’ll look at another example of `pub use` and how it affects your crate’s documentation in [“Exporting a Convenient Public API with `pub use`”](https://doc.rust-lang.org/book/ch14-02-publishing-to-crates-io.html#exporting-a-convenient-public-api-with-pub-use) in Chapter 14.
>  重导出在我们的代码内部结构和调用我们的代码的程序员对于领域的理解不同时非常有用
>  例如，餐厅的工作人员会考虑前台和后台，但来用餐的顾客不会这样来思考
>  使用 `pub use`，我们可以使用一种结构来编写代码，但暴露另一种结构，这使得我们的库对于库的开发者和库的使用者都更加清晰

### Using External Packages
In Chapter 2, we programmed a guessing game project that used an external package called `rand` to get random numbers. To use `rand` in our project, we added this line to _Cargo.toml_:

Filename: Cargo.toml

```rust
rand = "0.8.5"
```

Adding `rand` as a dependency in _Cargo.toml_ tells Cargo to download the `rand` package and any dependencies from [crates.io](https://crates.io/) and make `rand` available to our project.

>  要使用外部的 package 时，我们需要在 `Cargo.toml` 中将它添加为依赖，Cargo 会从 `crates.io` 自动下载依赖包

Then, to bring `rand` definitions into the scope of our package, we added a `use` line starting with the name of the crate, `rand`, and listed the items we wanted to bring into scope. Recall that in [“Generating a Random Number”](https://doc.rust-lang.org/book/ch02-00-guessing-game-tutorial.html#generating-a-random-number) in Chapter 2, we brought the `Rng` trait into scope and called the `rand::thread_rng` function:
>  之后，我们使用 `use <crate-name>` 就可以使用其中的内容了

```rust
use rand::Rng;

fn main() {
    let secret_number = rand::thread_rng().gen_range(1..=100);
}
```

Members of the Rust community have made many packages available at [crates.io](https://crates.io/), and pulling any of them into your package involves these same steps: listing them in your package’s _Cargo.toml_ file and using `use` to bring items from their crates into scope.
>  使用 `crates.io` 中的 package 的步骤都是一样的: 在 `Cargo.toml` 中列出 package，在代码中使用 `use` 将 item 从它们定义的 crate 带入当前作用域中

Note that the standard `std` library is also a crate that’s external to our package. Because the standard library is shipped with the Rust language, we don’t need to change _Cargo.toml_ to include `std`. But we do need to refer to it with `use` to bring items from there into our package’s scope. For example, with `HashMap` we would use this line:
>  标准库 `std` 也是相对于我们当前 package 的外部 crate，但使用标准库不需要我们在 `Cargo.toml` 中特意指定

```rust
use std::collections::HashMap; 
```

This is an absolute path starting with `std`, the name of the standard library crate.

### Using Nested Paths to Clean Up Large `use` Lists
If we’re using multiple items defined in the same crate or same module, listing each item on its own line can take up a lot of vertical space in our files. For example, these two `use` statements we had in the guessing game in Listing 2-4 bring items from `std` into scope:

Filename: src/main.rs

```rust
// --snip--
use std::cmp::Ordering;
use std::io;
// --snip--
```

Instead, we can use nested paths to bring the same items into scope in one line. We do this by specifying the common part of the path, followed by two colons, and then curly brackets around a list of the parts of the paths that differ, as shown in Listing 7-18.
>  我们可以用以下的语法同时将多个 itesm 带入我们的作用域

Filename: src/main.rs

```rust
// --snip--
use std::{cmp::Ordering, io};
// --snip--
```

[Listing 7-18](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#listing-7-18): Specifying a nested path to bring multiple items with the same prefix into scope

In bigger programs, bringing many items into scope from the same crate or module using nested paths can reduce the number of separate `use` statements needed by a lot!

We can use a nested path at any level in a path, which is useful when combining two `use` statements that share a subpath. For example, Listing 7-19 shows two `use` statements: one that brings `std::io` into scope and one that brings `std::io::Write` into scope.

Filename: src/lib.rs

```rust
use std::io;
use std::io::Write;
```

[Listing 7-19](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#listing-7-19): Two `use` statements where one is a subpath of the other

The common part of these two paths is `std::io`, and that’s the complete first path. To merge these two paths into one `use` statement, we can use `self` in the nested path, as shown in Listing 7-20.

Filename: src/lib.rs

```rust
use std::io::{self, Write};
```

[Listing 7-20](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#listing-7-20): Combining the paths in Listing 7-19 into one `use` statement

This line brings `std::io` and `std::io::Write` into scope.

### The Glob Operator
If we want to bring _all_ public items defined in a path into scope, we can specify that path followed by the `*` glob operator:
>  如果我们需要将所有的公有 item 带入作用域，可以使用通配符 `*`

```rust
use std::collections::*;
```

This `use` statement brings all public items defined in `std::collections` into the current scope. Be careful when using the glob operator! Glob can make it harder to tell what names are in scope and where a name used in your program was defined. Additionally, if the dependency changes its definitions, what you’ve imported changes as well, which may lead to compiler errors when you upgrade the dependency if the dependency adds a definition with the same name as a definition of yours in the same scope, for example.

The glob operator is often used when testing to bring everything under test into the `tests` module; we’ll talk about that in [“How to Write Tests”](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#how-to-write-tests) in Chapter 11. The glob operator is also sometimes used as part of the prelude pattern: see [the standard library documentation](https://doc.rust-lang.org/std/prelude/index.html#other-preludes) for more information on that pattern.
>  这个方式主要是在 `tests` module 中测试将所有东西带入测试的场景

## 7.5 Separating Modules into Different Files
So far, all the examples in this chapter defined multiple modules in one file. When modules get large, you might want to move their definitions to a separate file to make the code easier to navigate.

For example, let’s start from the code in Listing 7-17 that had multiple restaurant modules. We’ll extract modules into files instead of having all the modules defined in the crate root file. In this case, the crate root file is _src/lib.rs_, but this procedure also works with binary crates whose crate root file is _src/main.rs_.

First we’ll extract the `front_of_house` module to its own file. Remove the code inside the curly brackets for the `front_of_house` module, leaving only the `mod front_of_house;` declaration, so that _src/lib.rs_ contains the code shown in Listing 7-21. Note that this won’t compile until we create the _src/front_of_house.rs_ file in Listing 7-22.
>  我们可以不用把所有 modules 都定义在 crate root module 的文件中
>  我们只需要在 crate root file 中留好声明，然后在和 module 相同名字的文件中写下定义即可

Filename: src/lib.rs

```rust
mod front_of_house;

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
}
```

[Listing 7-21](https://doc.rust-lang.org/book/ch07-05-separating-modules-into-different-files.html#listing-7-21): Declaring the `front_of_house` module whose body will be in _src/front_of_house.rs_

Next, place the code that was in the curly brackets into a new file named _src/front_of_house.rs_, as shown in Listing 7-22. The compiler knows to look in this file because it came across the module declaration in the crate root with the name `front_of_house`.
>  编译器会根据 crate root 中的声明寻找 module 文件

Filename: src/front_of_house.rs

```rust
pub mod hosting {
    pub fn add_to_waitlist() {}
}
```

[Listing 7-22](https://doc.rust-lang.org/book/ch07-05-separating-modules-into-different-files.html#listing-7-22): Definitions inside the `front_of_house` module in _src/front_of_house.rs_

Note that you only need to load a file using a `mod` declaration _once_ in your module tree. Once the compiler knows the file is part of the project (and knows where in the module tree the code resides because of where you’ve put the `mod` statement), other files in your project should refer to the loaded file’s code using a path to where it was declared, as covered in the [“Paths for Referring to an Item in the Module Tree”](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html) section. In other words, `mod` is _not_ an “include” operation that you may have seen in other programming languages.
>  使用 `mod` 声明来加载 module 文件只需要在 module tree 中执行一次
>  编译器知道了 module 文件的位置之后，我们项目中的其他文件就可以直接根据该 module 声明的位置来访问加载的 module
>  也就是 `mod` 并不等同于 `#include`

Next, we’ll extract the `hosting` module to its own file. The process is a bit different because `hosting` is a child module of `front_of_house`, not of the root module. We’ll place the file for `hosting` in a new directory that will be named for its ancestors in the module tree, in this case _src/front_of_house_.

To start moving `hosting`, we change _src/front_of_house.rs_ to contain only the declaration of the `hosting` module:
>  child module 的文件需要放在指定名字的子目录中，同时父 module 文件中仅留下声明

Filename: src/front_of_house.rs

```rust
pub mod hosting;
```

Then we create a _src/front_of_house_ directory and a _hosting.rs_ file to contain the definitions made in the `hosting` module:

Filename: src/front_of_house/hosting.rs

```rust
pub fn add_to_waitlist() {}
```

If we instead put _hosting.rs_ in the _src_ directory, the compiler would expect the _hosting.rs_ code to be in a `hosting` module declared in the crate root, and not declared as a child of the `front_of_house` module. The compiler’s rules for which files to check for which modules’ code mean the directories and files more closely match the module tree.

### Alternate File Paths
So far we’ve covered the most idiomatic file paths the Rust compiler uses, but Rust also supports an older style of file path. For a module named `front_of_house` declared in the crate root, the compiler will look for the module’s code in:

- _src/front_of_house.rs_ (what we covered)
- _src/front_of_house/mod.rs_ (older style, still supported path)

For a module named `hosting` that is a submodule of `front_of_house`, the compiler will look for the module’s code in:

- _src/front_of_house/hosting.rs_ (what we covered)
- _src/front_of_house/hosting/mod.rs_ (older style, still supported path)

If you use both styles for the same module, you’ll get a compiler error. Using a mix of both styles for different modules in the same project is allowed, but might be confusing for people navigating your project.
>  除了之前介绍的组织文件路径的方式，另一种旧方式也仍然支持，但如果在相同 module 中使用两种风格，则会编译失败
>  但在一个项目中为不同的 modules 使用不同的风格是允许的

The main downside to the style that uses files named _mod.rs_ is that your project can end up with many files named _mod.rs_, which can get confusing when you have them open in your editor at the same time.

We’ve moved each module’s code to a separate file, and the module tree remains the same. The function calls in `eat_at_restaurant` will work without any modification, even though the definitions live in different files. This technique lets you move modules to new files as they grow in size.

Note that the `pub use crate::front_of_house::hosting` statement in _src/lib.rs_ also hasn’t changed, nor does `use` have any impact on what files are compiled as part of the crate. The `mod` keyword declares modules, and Rust looks in a file with the same name as the module for the code that goes into that module.

## Summary
Rust lets you split a package into multiple crates and a crate into modules so you can refer to items defined in one module from another module. You can do this by specifying absolute or relative paths. These paths can be brought into scope with a `use` statement so you can use a shorter path for multiple uses of the item in that scope. Module code is private by default, but you can make definitions public by adding the `pub` keyword.

In the next chapter, we’ll look at some collection data structures in the standard library that you can use in your neatly organized code.

# 8 Common Collections
Rust’s standard library includes a number of very useful data structures called _collections_. Most other data types represent one specific value, but collections can contain multiple values. Unlike the built-in array and tuple types, the data that these collections point to is stored on the heap, which means the amount of data does not need to be known at compile time and can grow or shrink as the program runs. Each kind of collection has different capabilities and costs, and choosing an appropriate one for your current situation is a skill you’ll develop over time. In this chapter, we’ll discuss three collections that are used very often in Rust programs:

- A _vector_ allows you to store a variable number of values next to each other.
- A _string_ is a collection of characters. We’ve mentioned the `String` type previously, but in this chapter we’ll talk about it in depth.
- A _hash map_ allows you to associate a value with a specific key. It’s a particular implementation of the more general data structure called a _map_.

>  Rust 标准库包含了很多有用的数据结构，称为集合
>  大多数其他数据类型表示一个特定的值，但集合可以包含多个值
>  和内建的数组和元组类型不同，集合类型在堆上存储数据，这意味着不需要在编译时知道数据大小，并且数据大小可以在程序运行时变化
>  我们讨论三种常用的集合:
>  - vector
>  - string
>  - hash map, 它是更通用数据结构 map 的一种具体实现

To learn about the other kinds of collections provided by the standard library, see [the documentation](https://doc.rust-lang.org/std/collections/index.html).

We’ll discuss how to create and update vectors, strings, and hash maps, as well as what makes each special.

## 8.1 Storing Lists of Values with Vectors
The first collection type we’ll look at is `Vec<T>`, also known as a _vector_. Vectors allow you to store more than one value in a single data structure that puts all the values next to each other in memory. Vectors can only store values of the same type. They are useful when you have a list of items, such as the lines of text in a file or the prices of items in a shopping cart.
>  vector 只能存储相同类型的 value, vector 将 values 临近地紧密存储

### Creating a New Vector
To create a new empty vector, we call the `Vec::new` function, as shown in Listing 8-1.

```rust
let v: Vec<i32> = Vec::new();
```

[Listing 8-1](https://doc.rust-lang.org/book/ch08-01-vectors.html#listing-8-1): Creating a new, empty vector to hold values of type `i32`

>  `Vec::new` 用于构造 vector

Note that we added a type annotation here. Because we aren’t inserting any values into this vector, Rust doesn’t know what kind of elements we intend to store. This is an important point. Vectors are implemented using generics; we’ll cover how to use generics with your own types in Chapter 10. For now, know that the `Vec<T>` type provided by the standard library can hold any type. When we create a vector to hold a specific type, we can specify the type within angle brackets. In Listing 8-1, we’ve told Rust that the `Vec<T>` in `v` will hold elements of the `i32` type.
>  注意我们这里添加了类型注释，因为 vectors 是使用泛型实现的

More often, you’ll create a `Vec<T>` with initial values and Rust will infer the type of value you want to store, so you rarely need to do this type annotation. Rust conveniently provides the `vec!` macro, which will create a new vector that holds the values you give it. Listing 8-2 creates a new `Vec<i32>` that holds the values `1`, `2`, and `3`. The integer type is `i32` because that’s the default integer type, as we discussed in the [“Data Types”](https://doc.rust-lang.org/book/ch03-02-data-types.html#data-types) section of Chapter 3.
>  通常使用初始值构造 vector 可以让 Rust 推导类型
>  使用初始值构造 vector 需要使用 `vec!` macro

```rust
let v = vec![1, 2, 3];
```

[Listing 8-2](https://doc.rust-lang.org/book/ch08-01-vectors.html#listing-8-2): Creating a new vector containing values

Because we’ve given initial `i32` values, Rust can infer that the type of `v` is `Vec<i32>`, and the type annotation isn’t necessary. Next, we’ll look at how to modify a vector.

### Updating a Vector
To create a vector and then add elements to it, we can use the `push` method, as shown in Listing 8-3.

```rust
let mut v = Vec::new();

v.push(5);
v.push(6);
v.push(7);
v.push(8);
```

[Listing 8-3](https://doc.rust-lang.org/book/ch08-01-vectors.html#listing-8-3): Using the `push` method to add values to a vector

As with any variable, if we want to be able to change its value, we need to make it mutable using the `mut` keyword, as discussed in Chapter 3. The numbers we place inside are all of type `i32`, and Rust infers this from the data, so we don’t need the `Vec<i32>` annotation.

>  要额外添加元素，可以使用 `push` 方法
>  但注意，和任意变量一样，如果我们想要让 vector 可变，需要使用 `mut`

### Reading Elements of Vectors
There are two ways to reference a value stored in a vector: via indexing or by using the `get` method. In the following examples, we’ve annotated the types of the values that are returned from these functions for extra clarity.
>  访问 vector 中元素的方法有两种，通过索引或者通过 `get` 方法

Listing 8-4 shows both methods of accessing a value in a vector, with indexing syntax and the `get` method.

```rust
let v = vec![1, 2, 3, 4, 5];

let third: &i32 = &v[2];
println!("The third element is {third}");

let third: Option<&i32> = v.get(2);
match third {
    Some(third) => println!("The third element is {third}"),
    None => println!("There is no third element."),
}
```

[Listing 8-4](https://doc.rust-lang.org/book/ch08-01-vectors.html#listing-8-4): Using indexing syntax and using the `get` method to access an item in a vector

Note a few details here. We use the index value of `2` to get the third element because vectors are indexed by number, starting at zero. Using `&` and `[]` gives us a reference to the element at the index value. When we use the `get` method with the index passed as an argument, we get an `Option<&T>` that we can use with `match`.
>  注意使用 `get` 方法返回的结果类型为 `Option<&T>`

Rust provides these two ways to reference an element so you can choose how the program behaves when you try to use an index value outside the range of existing elements. As an example, let’s see what happens when we have a vector of five elements and then we try to access an element at index 100 with each technique, as shown in Listing 8-5.

```rust
let v = vec![1, 2, 3, 4, 5];

let does_not_exist = &v[100];
let does_not_exist = v.get(100);
```

[Listing 8-5](https://doc.rust-lang.org/book/ch08-01-vectors.html#listing-8-5): Attempting to access the element at index 100 in a vector containing five elements

When we run this code, the first `[]` method will cause the program to panic because it references a nonexistent element. This method is best used when you want your program to crash if there’s an attempt to access an element past the end of the vector.

When the `get` method is passed an index that is outside the vector, it returns `None` without panicking. You would use this method if accessing an element beyond the range of the vector may happen occasionally under normal circumstances. Your code will then have logic to handle having either `Some(&element)` or `None`, as discussed in Chapter 6. For example, the index could be coming from a person entering a number. If they accidentally enter a number that’s too large and the program gets a `None` value, you could tell the user how many items are in the current vector and give them another chance to enter a valid value. That would be more user-friendly than crashing the program due to a typo!

>  使用 `[]` 访问越界元素时，程序会直接 panic
>  使用 `get` 访问越界元素时，方法会返回 `None`，如果没有访问越界元素，则返回 `Some(&element)`

When the program has a valid reference, the borrow checker enforces the ownership and borrowing rules (covered in Chapter 4) to ensure this reference and any other references to the contents of the vector remain valid. Recall the rule that states you can’t have mutable and immutable references in the same scope. That rule applies in Listing 8-6, where we hold an immutable reference to the first element in a vector and try to add an element to the end. This program won’t work if we also try to refer to that element later in the function.
>  当程序有有效的引用时，借用检查器通过所有权和借用规则确保该引用是有效的
>  当我们具有一个对第一个元素的不可变引用，并且尝试为 vector 尾部添加元素，程序就无法编译

```rust
let mut v = vec![1, 2, 3, 4, 5];

let first = &v[0];

v.push(6);

println!("The first element is: {first}");
```

[Listing 8-6](https://doc.rust-lang.org/book/ch08-01-vectors.html#listing-8-6): Attempting to add an element to a vector while holding a reference to an item

Compiling this code will result in this error:

```
$ cargo run
   Compiling collections v0.1.0 (file:///projects/collections)
error[E0502]: cannot borrow `v` as mutable because it is also borrowed as immutable
 --> src/main.rs:6:5
  |
4 |     let first = &v[0];
  |                  - immutable borrow occurs here
5 |
6 |     v.push(6);
  |     ^^^^^^^^^ mutable borrow occurs here
7 |
8 |     println!("The first element is: {first}");
  |                                     ------- immutable borrow later used here

For more information about this error, try `rustc --explain E0502`.
error: could not compile `collections` (bin "collections") due to 1 previous error
```

The code in Listing 8-6 might look like it should work: why should a reference to the first element care about changes at the end of the vector? This error is due to the way vectors work: because vectors put the values next to each other in memory, adding a new element onto the end of the vector might require allocating new memory and copying the old elements to the new space, if there isn’t enough room to put all the elements next to each other where the vector is currently stored. In that case, the reference to the first element would be pointing to deallocated memory. The borrowing rules prevent programs from ending up in that situation.
>  这个错误的原因来自于 vector 的工作方式: vector 紧密排布元素，因此添加新元素可能导致 vector 重新分配内存，并且将旧元素拷贝到新空间，此时，对第一个元素的引用就会指向已经释放的内存
>  借用规则防止了这个情况的发生

Note: For more on the implementation details of the `Vec<T>` type, see [“The Rustonomicon”](https://doc.rust-lang.org/nomicon/vec/vec.html).

### Iterating Over the Values in a Vector
To access each element in a vector in turn, we would iterate through all of the elements rather than use indices to access one at a time. Listing 8-7 shows how to use a `for` loop to get immutable references to each element in a vector of `i32` values and print them.

```rust
let v = vec![100, 32, 57];
for i in &v {
    println!("{i}");
}
```

[Listing 8-7](https://doc.rust-lang.org/book/ch08-01-vectors.html#listing-8-7): Printing each element in a vector by iterating over the elements using a `for` loop

>  `for x in &vec` 可以迭代 vector 中的元素，这样获取的是不可变引用

We can also iterate over mutable references to each element in a mutable vector in order to make changes to all the elements. The `for` loop in Listing 8-8 will add `50` to each element.
>  `for x in &mut v` 则获取可变引用

```rust
let mut v = vec![100, 32, 57];
for i in &mut v {
    *i += 50;
}
```

[Listing 8-8](https://doc.rust-lang.org/book/ch08-01-vectors.html#listing-8-8): Iterating over mutable references to elements in a vector

To change the value that the mutable reference refers to, we have to use the `*` dereference operator to get to the value in `i` before we can use the `+=` operator. We’ll talk more about the dereference operator in the [“Following the Reference to the Value”](https://doc.rust-lang.org/book/ch15-02-deref.html#following-the-pointer-to-the-value-with-the-dereference-operator) section of Chapter 15.
>  要获取 value，需要使用 `*i` 进行解引用

Iterating over a vector, whether immutably or mutably, is safe because of the borrow checker’s rules. If we attempted to insert or remove items in the `for` loop bodies in Listing 8-7 and Listing 8-8, we would get a compiler error similar to the one we got with the code in Listing 8-6. The reference to the vector that the `for` loop holds prevents simultaneous modification of the whole vector.
>  对 vector 迭代同样受借用检查器的保护，如果我们尝试在 `for` 中插入或移除 item，将得到编译错误，因为我们的引用防止了同时对整个 vector 的修改

### Using an Enum to Store Multiple Types
Vectors can only store values that are of the same type. This can be inconvenient; there are definitely use cases for needing to store a list of items of different types. Fortunately, the variants of an enum are defined under the same enum type, so when we need one type to represent elements of different types, we can define and use an enum!

For example, say we want to get values from a row in a spreadsheet in which some of the columns in the row contain integers, some floating-point numbers, and some strings. We can define an enum whose variants will hold the different value types, and all the enum variants will be considered the same type: that of the enum. Then we can create a vector to hold that enum and so, ultimately, hold different types. We’ve demonstrated this in Listing 8-9.
>  可以利用 vector 存储 enum 类型来实现存储不同类型的值，因为 enum 变体可以关联不同类型的值，但是 enum 变体都被视为相同类型

```rust
enum SpreadsheetCell {
    Int(i32),
    Float(f64),
    Text(String),
}

let row = vec![
    SpreadsheetCell::Int(3),
    SpreadsheetCell::Text(String::from("blue")),
    SpreadsheetCell::Float(10.12),
];
```

[Listing 8-9](https://doc.rust-lang.org/book/ch08-01-vectors.html#listing-8-9): Defining an `enum` to store values of different types in one vector

Rust needs to know what types will be in the vector at compile time so it knows exactly how much memory on the heap will be needed to store each element. We must also be explicit about what types are allowed in this vector. If Rust allowed a vector to hold any type, there would be a chance that one or more of the types would cause errors with the operations performed on the elements of the vector. Using an enum plus a `match` expression means that Rust will ensure at compile time that every possible case is handled, as discussed in Chapter 6.
>  Rust 需要在编译时知道 vector 中将包含哪些类型，以便确切知道堆上应该分配多少内存来存储每个元素
>  使用 enum 加上 `match` 表达式意味着 Rust 在编译时处理了所有可能的情况

If you don’t know the exhaustive set of types a program will get at runtime to store in a vector, the enum technique won’t work. Instead, you can use a trait object, which we’ll cover in Chapter 18.

Now that we’ve discussed some of the most common ways to use vectors, be sure to review [the API documentation](https://doc.rust-lang.org/std/vec/struct.Vec.html) for all of the many useful methods defined on `Vec<T>` by the standard library. For example, in addition to `push`, a `pop` method removes and returns the last element.

### Dropping a Vector Drops Its Elements
Like any other `struct`, a vector is freed when it goes out of scope, as annotated in Listing 8-10.

```rust
{
    let v = vec![1, 2, 3, 4];

    // do stuff with v
} // <- v goes out of scope and is freed here
```

[Listing 8-10](https://doc.rust-lang.org/book/ch08-01-vectors.html#listing-8-10): Showing where the vector and its elements are dropped

>  和任意其他 `struct` 一样，vector 在离开作用域之后会被释放

When the vector gets dropped, all of its contents are also dropped, meaning the integers it holds will be cleaned up. The borrow checker ensures that any references to contents of a vector are only used while the vector itself is valid.
>  vector 被释放后，其所有内容都会被释放

Let’s move on to the next collection type: `String`!

## 8.2 Storing UTF-8 Encoded Text with Strings
We talked about strings in Chapter 4, but we’ll look at them in more depth now. New Rustaceans commonly get stuck on strings for a combination of three reasons: Rust’s propensity for exposing possible errors, strings being a more complicated data structure than many programmers give them credit for, and UTF-8. These factors combine in a way that can seem difficult when you’re coming from other programming languages.

We discuss strings in the context of collections because strings are implemented as a collection of bytes, plus some methods to provide useful functionality when those bytes are interpreted as text. In this section, we’ll talk about the operations on `String` that every collection type has, such as creating, updating, and reading. We’ll also discuss the ways in which `String` is different from the other collections, namely how indexing into a `String` is complicated by the differences between how people and computers interpret `String` data.
>  string 实际上就是存储 bytes 的集合类型，以及添加了一些将 bytes 解释为 text 的方法

### What Is a String?
We’ll first define what we mean by the term _string_. Rust has only one string type in the core language, which is the string slice `str` that is usually seen in its borrowed form `&str`. In Chapter 4, we talked about _string slices_, which are references to some UTF-8 encoded string data stored elsewhere. String literals, for example, are stored in the program’s binary and are therefore string slices.
>  Rust 在核心语言中只有一种字符串类型，即字符串切片 `str`，通常以借用形式 `&str` 出现
>  字符串切片是对其他地方存储的 UTF-8 编码的字符串数据的引用
>  存储在程序的二进制文件中的字符串字面量也是字符串切片

The `String` type, which is provided by Rust’s standard library rather than coded into the core language, is a growable, mutable, owned, UTF-8 encoded string type. When Rustaceans refer to “strings” in Rust, they might be referring to either the `String` or the string slice `&str` types, not just one of those types. Although this section is largely about `String`, both types are used heavily in Rust’s standard library, and both `String` and string slices are UTF-8 encoded.
>  `String` 类型由 Rust 标准库提供，而不是内置于核心语言中
>  `String` 类型是一个可增长、可变、拥有所有权的 UTF-8 编码的字符串类型
>  Rust 中的 "strings" 既可以指 `String` 也可以指 `&str` 类型，两个类型都是 UTF-8 编码的

### Creating a New String
Many of the same operations available with `Vec<T>` are available with `String` as well because `String` is actually implemented as a wrapper around a vector of bytes with some extra guarantees, restrictions, and capabilities. An example of a function that works the same way with `Vec<T>` and `String` is the `new` function to create an instance, shown in Listing 8-11.
>  许多 `Vec<T>` 中的方法对于 `String` 也适用，因为 `String` 实际上被实现为对 vector of bytes 的包装器，带有一些额外的保证、限制和能力

```rust
let mut s = String::new();
```

[Listing 8-11](https://doc.rust-lang.org/book/ch08-02-strings.html#listing-8-11): Creating a new, empty `String`

This line creates a new, empty string called `s`, into which we can then load data. Often, we’ll have some initial data with which we want to start the string. For that, we use the `to_string` method, which is available on any type that implements the `Display` trait, as string literals do. Listing 8-12 shows two examples.
>  实现了 `Display` trait 的类型 (例如字符串字面值) 都有 `to_string` 方法，可以通过它获取 `String` 类型

```rust
let data = "initial contents";

let s = data.to_string();

// The method also works on a literal directly:
let s = "initial contents".to_string();
```

[Listing 8-12](https://doc.rust-lang.org/book/ch08-02-strings.html#listing-8-12): Using the `to_string` method to create a `String` from a string literal

This code creates a string containing `initial contents`.

We can also use the function `String::from` to create a `String` from a string literal. The code in Listing 8-13 is equivalent to the code in Listing 8-12 that uses `to_string`.

```rust
let s = String::from("initial contents");
```

[Listing 8-13](https://doc.rust-lang.org/book/ch08-02-strings.html#listing-8-13): Using the `String::from` function to create a `String` from a string literal

Because strings are used for so many things, we can use many different generic APIs for strings, providing us with a lot of options. Some of them can seem redundant, but they all have their place! In this case, `String::from` and `to_string` do the same thing, so which one you choose is a matter of style and readability.

Remember that strings are UTF-8 encoded, so we can include any properly encoded data in them, as shown in Listing 8-14.

```rust
let hello = String::from("السلام عليكم");
let hello = String::from("Dobrý den");
let hello = String::from("Hello");
let hello = String::from("שלום");
let hello = String::from("नमस्ते");
let hello = String::from("こんにちは");
let hello = String::from("안녕하세요");
let hello = String::from("你好");
let hello = String::from("Olá");
let hello = String::from("Здравствуйте");
let hello = String::from("Hola");
```

[Listing 8-14](https://doc.rust-lang.org/book/ch08-02-strings.html#listing-8-14): Storing greetings in different languages in strings

All of these are valid `String` values.

### Updating a String
A `String` can grow in size and its contents can change, just like the contents of a `Vec<T>`, if you push more data into it. In addition, you can conveniently use the `+` operator or the `format!` macro to concatenate `String` values.
>  `String` 类型可以使用 `+` operator 或者 `format!` macro 来拼接

#### Appending to a String with `push_str` and `push`
We can grow a `String` by using the `push_str` method to append a string slice, as shown in Listing 8-15.
>  可以通过 `String` 的 `push_str` 方法，将一个 string slice 追加到 `String` 中

```rust
let mut s = String::from("foo");
s.push_str("bar");
```

[Listing 8-15](https://doc.rust-lang.org/book/ch08-02-strings.html#listing-8-15): Appending a string slice to a `String` using the `push_str` method

After these two lines, `s` will contain `foobar`. The `push_str` method takes a string slice because we don’t necessarily want to take ownership of the parameter. For example, in the code in Listing 8-16, we want to be able to use `s2` after appending its contents to `s1`.
>  `push_str` 接收 string slice 是因为我们不想获得参数的所有权

```rust
let mut s1 = String::from("foo");
let s2 = "bar";
s1.push_str(s2);
println!("s2 is {s2}");
```

[Listing 8-16](https://doc.rust-lang.org/book/ch08-02-strings.html#listing-8-16): Using a string slice after appending its contents to a `String`

If the `push_str` method took ownership of `s2`, we wouldn’t be able to print its value on the last line. However, this code works as we’d expect!

The `push` method takes a single character as a parameter and adds it to the `String`. Listing 8-17 adds the letter _l_ to a `String` using the `push` method.
>  `push` 方法则接收单个字符作为参数

```rust
let mut s = String::from("lo");
s.push('l');
```

[Listing 8-17](https://doc.rust-lang.org/book/ch08-02-strings.html#listing-8-17): Adding one character to a `String` value using `push`

As a result, `s` will contain `lol`.

#### Concatenation with the `+` Operator or the `format!` Macro
Often, you’ll want to combine two existing strings. One way to do so is to use the `+` operator, as shown in Listing 8-18.
>  拼接两个 `String` 的方式是使用 `+` 运算符

```rust
let s1 = String::from("Hello, ");
let s2 = String::from("world!");
let s3 = s1 + &s2; // note s1 has been moved here and can no longer be used
```

[Listing 8-18](https://doc.rust-lang.org/book/ch08-02-strings.html#listing-8-18): Using the `+` operator to combine two `String` values into a new `String` value

The string `s3` will contain `Hello, world!`. The reason `s1` is no longer valid after the addition, and the reason we used a reference to `s2`, has to do with the signature of the method that’s called when we use the `+` operator. The `+` operator uses the `add` method, whose signature looks something like this:

```rust
fn add(self, s: &str) -> String {
```

>  `s1` 在运算后不再有效，以及我们使用对 `s2` 的引用的原因和 `+` 运算符的方法签名有关
>  `+` 关联的是 `add` 方法，签名如上

In the standard library, you’ll see `add` defined using generics and associated types. Here, we’ve substituted in concrete types, which is what happens when we call this method with `String` values. We’ll discuss generics in Chapter 10. This signature gives us the clues we need in order to understand the tricky bits of the `+` operator.
>  在标准库中，`add` 是使用泛型定义的
>  如果我们用 `String` values 调用该方法，泛型就会被替换为具体类型

First, `s2` has an `&`, meaning that we’re adding a _reference_ of the second string to the first string. This is because of the `s` parameter in the `add` function: we can only add a `&str` to a `String`; we can’t add two `String` values together. But wait—the type of `&s2` is `&String`, not `&str`, as specified in the second parameter to `add`. So why does Listing 8-18 compile?
>  从 `add` 中可以看到，我们只能将 `&str` 加到 `String` 上，而不能将两个 `String` 相加
>  但 `&s2` 的类型实际上是 `&String`

The reason we’re able to use `&s2` in the call to `add` is that the compiler can _coerce_ the `&String` argument into a `&str`. When we call the `add` method, Rust uses a _deref coercion_, which here turns `&s2` into `&s2[..]`. We’ll discuss deref coercion in more depth in Chapter 15. Because `add` does not take ownership of the `s` parameter, `s2` will still be a valid `String` after this operation.
>  这能够正确执行的原因是编译器将 `&String` 强制转化为了 `&str`
>  当我们调用 `add` 方法时，Rust 使用了 deref corecion，将 `&s2` 转化为 `&s2[..]`

Second, we can see in the signature that `add` takes ownership of `self` because `self` does _not_ have an `&`. This means `s1` in Listing 8-18 will be moved into the `add` call and will no longer be valid after that. So, although `let s3 = s1 + &s2;` looks like it will copy both strings and create a new one, this statement actually takes ownership of `s1`, appends a copy of the contents of `s2`, and then returns ownership of the result. In other words, it looks like it’s making a lot of copies, but it isn’t; the implementation is more efficient than copying.
>  `add` 获取了 `self` 的所有权，因为 `self` 没有 `&`
>  故 `s1` 会被移动到 `add` 调用中，且之后不再有效
>  `add` 返回了结果的所有权

If we need to concatenate multiple strings, the behavior of the `+` operator gets unwieldy:

```rust
let s1 = String::from("tic");
let s2 = String::from("tac");
let s3 = String::from("toe");

let s = s1 + "-" + &s2 + "-" + &s3;
```

At this point, `s` will be `tic-tac-toe`. With all of the `+` and `"` characters, it’s difficult to see what’s going on. For combining strings in more complicated ways, we can instead use the `format!` macro:

```rust
let s1 = String::from("tic");
let s2 = String::from("tac");
let s3 = String::from("toe");

let s = format!("{s1}-{s2}-{s3}");
```

This code also sets `s` to `tic-tac-toe`. The `format!` macro works like `println!`, but instead of printing the output to the screen, it returns a `String` with the contents. The version of the code using `format!` is much easier to read, and the code generated by the `format!` macro uses references so that this call doesn’t take ownership of any of its parameters.

>  如果要连续使用 `+`，我们可能会不容易看出究竟发生了什么
>  更简单的方式是使用 `format!` macro，它和 `println!` 类似，只不过是返回一个 `String` 而不是打印
>  `format!` 会使用引用，因此它的所有参数的所有权都不会被获取

### Indexing into Strings
In many other programming languages, accessing individual characters in a string by referencing them by index is a valid and common operation. However, if you try to access parts of a `String` using indexing syntax in Rust, you’ll get an error. Consider the invalid code in Listing 8-19.

```rust
let s1 = String::from("hi");
let h = s1[0];
```

[Listing 8-19](https://doc.rust-lang.org/book/ch08-02-strings.html#listing-8-19): Attempting to use indexing syntax with a String

This code will result in the following error:

```
$ cargo run
   Compiling collections v0.1.0 (file:///projects/collections)
error[E0277]: the type `str` cannot be indexed by `{integer}`
 --> src/main.rs:3:16
  |
3 |     let h = s1[0];
  |                ^ string indices are ranges of `usize`
  |
  = note: you can use `.chars().nth()` or `.bytes().nth()`
          for more information, see chapter 8 in The Book: <https://doc.rust-lang.org/book/ch08-02-strings.html#indexing-into-strings>
  = help: the trait `SliceIndex<str>` is not implemented for `{integer}`
          but trait `SliceIndex<[_]>` is implemented for `usize`
  = help: for that trait implementation, expected `[_]`, found `str`
  = note: required for `String` to implement `Index<{integer}>`

For more information about this error, try `rustc --explain E0277`.
error: could not compile `collections` (bin "collections") due to 1 previous error
```

The error and the note tell the story: Rust strings don’t support indexing. But why not? To answer that question, we need to discuss how Rust stores strings in memory.
>  `String` 不支持索引

#### Internal Representation
A `String` is a wrapper over a `Vec<u8>`. Let’s look at some of our properly encoded UTF-8 example strings from Listing 8-14. First, this one:
>  `String` 实质上是 `Vec<u8>` 的包装器

```rust
let hello = String::from("Hola");
```

In this case, `len` will be `4`, which means the vector storing the string `"Hola"` is 4 bytes long. Each of these letters takes one byte when encoded in UTF-8. 
>  上例中，`Vec<u8>` 的 `len=4`，即 vector 存储了四个字节，每个字节存储一个字母的 UTF-8 编码

The following line, however, may surprise you (note that this string begins with the capital Cyrillic letter _Ze_, not the number 3):

```rust
let hello = String::from("Здравствуйте"); 
```

If you were asked how long the string is, you might say 12. In fact, Rust’s answer is 24: that’s the number of bytes it takes to encode “Здравствуйте” in UTF-8, because each Unicode scalar value in that string takes 2 bytes of storage. 
>  上例的 `len=24`，因为非 ASCII 字符的 UTF-8 编码需要两个字节存储

Therefore, an index into the string’s bytes will not always correlate to a valid Unicode scalar value. To demonstrate, consider this invalid Rust code:

```rust
let hello = "Здравствуйте";
let answer = &hello[0]; 
```

You already know that `answer` will not be `З`, the first letter. When encoded in UTF-8, the first byte of `З` is `208` and the second is `151`, so it would seem that `answer` should in fact be `208`, but `208` is not a valid character on its own. Returning `208` is likely not what a user would want if they asked for the first letter of this string; however, that’s the only data that Rust has at byte index 0. Users generally don’t want the byte value returned, even if the string contains only Latin letters: if `&"hi"[0]` were valid code that returned the byte value, it would return `104`, not `h`.
>  因此，直接索引访问 `String` 存储的 byte 可能不会关联到有效的 Unicode scalar value

The answer, then, is that to avoid returning an unexpected value and causing bugs that might not be discovered immediately, Rust doesn’t compile this code at all and prevents misunderstandings early in the development process.

#### Bytes and Scalar Values and Grapheme Clusters! Oh My!
Another point about UTF-8 is that there are actually three relevant ways to look at strings from Rust’s perspective: as bytes, scalar values, and grapheme clusters (the closest thing to what we would call _letters_).
>  从 Rust 的角度来看，有三种方式来看待 strings: as bytes, scalar values, 字型簇 (类似字母)

If we look at the Hindi word “नमस्ते” written in the Devanagari script, it is stored as a vector of `u8` values that looks like this:

```
[224, 164, 168, 224, 164, 174, 224, 164, 184, 224, 165, 141, 224, 164, 164, 224, 165, 135]
```

That’s 18 bytes and is how computers ultimately store this data. If we look at them as Unicode scalar values, which are what Rust’s `char` type is, those bytes look like this:

```
['न', 'म', 'स', '्', 'त', 'े']
```

There are six `char` values here, but the fourth and sixth are not letters: they’re diacritics that don’t make sense on their own. Finally, if we look at them as grapheme clusters, we’d get what a person would call the four letters that make up the Hindi word:

```
["न", "म", "स्", "ते"]
```

Rust provides different ways of interpreting the raw string data that computers store so that each program can choose the interpretation it needs, no matter what human language the data is in.
>  Rust 提供了多种方式来解释计算机存储的原始字符串数据，程序可以根据需要选择合适的解释方式

A final reason Rust doesn’t allow us to index into a `String` to get a character is that indexing operations are expected to always take constant time (O(1)). But it isn’t possible to guarantee that performance with a `String`, because Rust would have to walk through the contents from the beginning to the index to determine how many valid characters there were.

### Slicing Strings
Indexing into a string is often a bad idea because it’s not clear what the return type of the string-indexing operation should be: a byte value, a character, a grapheme cluster, or a string slice. If you really need to use indices to create string slices, therefore, Rust asks you to be more specific.
>  因为 Rust 有三种方式看待 string，故对 string 进行索引通常不确定会返回的是 byte value, character, grapheme cluter 还是 string slice
>  如果我们真的想创建 string slices, Rust 要求我们更加具体

Rather than indexing using `[]` with a single number, you can use `[]` with a range to create a string slice containing particular bytes:

```rust
let hello = "Здравствуйте";

let s = &hello[0..4];
```

Here, `s` will be a `&str` that contains the first four bytes of the string. Earlier, we mentioned that each of these characters was two bytes, which means `s` will be `Зд`.
>  我们需要在 `[]` 提供一个 range 来创建包含特定 bytes 的 string slice
>  这里，`s` 的类型将是 `&str`，包含了 string 的前 4 个 bytes

If we were to try to slice only part of a character’s bytes with something like `&hello[0..1]`, Rust would panic at runtime in the same way as if an invalid index were accessed in a vector:

```
$ cargo run
   Compiling collections v0.1.0 (file:///projects/collections)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.43s
     Running `target/debug/collections`

thread 'main' panicked at src/main.rs:4:19:
byte index 1 is not a char boundary; it is inside 'З' (bytes 0..2) of `Здравствуйте`
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

You should use caution when creating string slices with ranges, because doing so can crash your program.
>  使用 ranges 创建 string slices 也要小心，如果没有索引在有效字符边界处，程序会崩溃

### Methods for Iterating Over Strings
The best way to operate on pieces of strings is to be explicit about whether you want characters or bytes. For individual Unicode scalar values, use the `chars` method. Calling `chars` on “Зд” separates out and returns two values of type `char`, and you can iterate over the result to access each element:
>  要对 string 的片段进行操作的最好方式是明确我们要 characters 还是 bytes
>  对于独立的 Unicode scalar values，使用 `chars` 方法，它将拆分 string，返回可以迭代的 `char` 类型列表

```rust
for c in "Зд".chars() {
    println!("{c}");
}
```

This code will print the following:

```
З
д
```

Alternatively, the `bytes` method returns each raw byte, which might be appropriate for your domain:
>  `bytes` 则返回 byte 列表

```rust
for b in "Зд".bytes() {
    println!("{b}");
}
```

This code will print the four bytes that make up this string:

```
208
151
208
180
```

But be sure to remember that valid Unicode scalar values may be made up of more than one byte.
>  明确记住有效的 Unicode scalar values 可以由多个 bytes 组成

Getting grapheme clusters from strings, as with the Devanagari script, is complex, so this functionality is not provided by the standard library. Crates are available on [crates.io](https://crates.io/) if this is the functionality you need.
>  标准库没有提供从 string 获取 grapheme clusters 的方法

### Strings Are Not So Simple
To summarize, strings are complicated. Different programming languages make different choices about how to present this complexity to the programmer. Rust has chosen to make the correct handling of `String` data the default behavior for all Rust programs, which means programmers have to put more thought into handling UTF-8 data up front. This trade-off exposes more of the complexity of strings than is apparent in other programming languages, but it prevents you from having to handle errors involving non-ASCII characters later in your development life cycle.
>  Rust 暴露了对 UTF-8 数据的处理，即暴露了 string 的复杂性，但也避免了在开发过程中出现 non-ASCII 字符相关的错误

The good news is that the standard library offers a lot of functionality built off the `String` and `&str` types to help handle these complex situations correctly. Be sure to check out the documentation for useful methods like `contains` for searching in a string and `replace` for substituting parts of a string with another string.

Let’s switch to something a bit less complex: hash maps!

## 8.3 Storing Keys with Associated Values in Hash Maps
The last of our common collections is the _hash map_. The type `HashMap<K, V>` stores a mapping of keys of type `K` to values of type `V` using a _hashing function_, which determines how it places these keys and values into memory. Many programming languages support this kind of data structure, but they often use a different name, such as _hash_, _map_, _object_, _hash table_, _dictionary_, or _associative array_, just to name a few.
>  类型 `HashMap<K, V>` 使用哈希函数存储类型为 `K` 的 keys 到类型为 `V` 的 values 的映射，哈希函数决定了如何将 keys, values 放入内存

Hash maps are useful when you want to look up data not by using an index, as you can with vectors, but by using a key that can be of any type. For example, in a game, you could keep track of each team’s score in a hash map in which each key is a team’s name and the values are each team’s score. Given a team name, you can retrieve its score.

We’ll go over the basic API of hash maps in this section, but many more goodies are hiding in the functions defined on `HashMap<K, V>` by the standard library. As always, check the standard library documentation for more information.

### Creating a New Hash Map
One way to create an empty hash map is to use `new` and to add elements with `insert`. In Listing 8-20, we’re keeping track of the scores of two teams whose names are _Blue_ and _Yellow_. The Blue team starts with 10 points, and the Yellow team starts with 50.
>  使用 `new()` 创建，使用 `insert()` 加入元素

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();

scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);
```

[Listing 8-20](https://doc.rust-lang.org/book/ch08-03-hash-maps.html#listing-8-20): Creating a new hash map and inserting some keys and values

Note that we need to first `use` the `HashMap` from the collections portion of the standard library. Of our three common collections, this one is the least often used, so it’s not included in the features brought into scope automatically in the prelude. Hash maps also have less support from the standard library; there’s no built-in macro to construct them, for example.

Just like vectors, hash maps store their data on the heap. This `HashMap` has keys of type `String` and values of type `i32`. Like vectors, hash maps are homogeneous: all of the keys must have the same type, and all of the values must have the same type.
>  hash map 在堆上存储数据
>  hash map 所有的 keys, values 要求类型相同

### Accessing Values in a Hash Map
We can get a value out of the hash map by providing its key to the `get` method, as shown in Listing 8-21.
>  `get()` 通过 key 获取 value

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();

scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);

let team_name = String::from("Blue");
let score = scores.get(&team_name).copied().unwrap_or(0);
```

[Listing 8-21](https://doc.rust-lang.org/book/ch08-03-hash-maps.html#listing-8-21): Accessing the score for the Blue team stored in the hash map

Here, `score` will have the value that’s associated with the Blue team, and the result will be `10`. The `get` method returns an `Option<&V>`; if there’s no value for that key in the hash map, `get` will return `None`. This program handles the `Option` by calling `copied` to get an `Option<i32>` rather than an `Option<&i32>`, then `unwrap_or` to set `score` to zero if `scores` doesn’t have an entry for the key.
>  `get` 返回的是 `Option<&V>`，如果没有找到 key，`get` 会返回 `None`
>  对 `Option<&i32>` 调用 `copied` 获取 `Option<i32>`，调用 `unwarp_or` 在其中的值为 `None` 时设置为 0

We can iterate over each key-value pair in a hash map in a similar manner as we do with vectors, using a `for` loop:

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();

scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);

for (key, value) in &scores {
    println!("{key}: {value}");
}
```

>  hash map 的迭代没有顺序保证

This code will print each pair in an arbitrary order:

```
Yellow: 50
Blue: 10
```

### Hash Maps and Ownership
For types that implement the `Copy` trait, like `i32`, the values are copied into the hash map. For owned values like `String`, the values will be moved and the hash map will be the owner of those values, as demonstrated in Listing 8-22.
>  如果类型实现了 `Copy` trait，其 value 会被拷贝到 hash map
>  对于 owned values，则会发生所有权转移，hash map 获得所有权

```rust
use std::collections::HashMap;

let field_name = String::from("Favorite color");
let field_value = String::from("Blue");

let mut map = HashMap::new();
map.insert(field_name, field_value);
// field_name and field_value are invalid at this point, try using them and
// see what compiler error you get!
```

[Listing 8-22](https://doc.rust-lang.org/book/ch08-03-hash-maps.html#listing-8-22): Showing that keys and values are owned by the hash map once they’re inserted

We aren’t able to use the variables `field_name` and `field_value` after they’ve been moved into the hash map with the call to `insert`.
>  在 `insert` 之后，我们就不再能使用 `field_name, field_value`

If we insert references to values into the hash map, the values won’t be moved into the hash map. The values that the references point to must be valid for at least as long as the hash map is valid. We’ll talk more about these issues in [“Validating References with Lifetimes”](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#validating-references-with-lifetimes) in Chapter 10.
>  如果我们 `insert` 的是引用，则不会发生所有权移动，但在 hash map 有效期间，引用的 value 必须保证有效

### Updating a Hash Map
Although the number of key and value pairs is growable, each unique key can only have one value associated with it at a time (but not vice versa: for example, both the Blue team and the Yellow team could have the value `10` stored in the `scores` hash map).

When you want to change the data in a hash map, you have to decide how to handle the case when a key already has a value assigned. You could replace the old value with the new value, completely disregarding the old value. You could keep the old value and ignore the new value, only adding the new value if the key _doesn’t_ already have a value. Or you could combine the old value and the new value. Let’s look at how to do each of these!

#### Overwriting a Value
If we insert a key and a value into a hash map and then insert that same key with a different value, the value associated with that key will be replaced. Even though the code in Listing 8-23 calls `insert` twice, the hash map will only contain one key-value pair because we’re inserting the value for the Blue team’s key both times.
>  如果 `insert` 了相同的 key 但是有不同的 value，则新 value 会替代旧 value

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();

scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Blue"), 25);

println!("{scores:?}");
```

[Listing 8-23](https://doc.rust-lang.org/book/ch08-03-hash-maps.html#listing-8-23): Replacing a value stored with a particular key

This code will print `{"Blue": 25}`. The original value of `10` has been overwritten.

#### Adding a Key and Value Only If a Key Isn’t Present
It’s common to check whether a particular key already exists in the hash map with a value and then to take the following actions: if the key does exist in the hash map, the existing value should remain the way it is; if the key doesn’t exist, insert it and a value for it.

Hash maps have a special API for this called `entry` that takes the key you want to check as a parameter. The return value of the `entry` method is an enum called `Entry` that represents a value that might or might not exist. Let’s say we want to check whether the key for the Yellow team has a value associated with it. If it doesn’t, we want to insert the value `50`, and the same for the Blue team. Using the `entry` API, the code looks like Listing 8-24.
>  `entry` 接收一个 key，检查 key 是否存在，其返回 `Entry` enum，表示一个值可能存在或不存在

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);

scores.entry(String::from("Yellow")).or_insert(50);
scores.entry(String::from("Blue")).or_insert(50);

println!("{scores:?}");
```

[Listing 8-24](https://doc.rust-lang.org/book/ch08-03-hash-maps.html#listing-8-24): Using the `entry` method to only insert if the key does not already have a value

The `or_insert` method on `Entry` is defined to return a mutable reference to the value for the corresponding `Entry` key if that key exists, and if not, it inserts the parameter as the new value for this key and returns a mutable reference to the new value. This technique is much cleaner than writing the logic ourselves and, in addition, plays more nicely with the borrow checker.
>  `Entry` 的 `or_insert` 方法在如果 key 存在时，返回对 value 的可变引用
>  如果 key 不存在，则将它的参数插入作为该 key 的 value，返回对该 value 的可变引用

Running the code in Listing 8-24 will print `{"Yellow": 50, "Blue": 10}`. The first call to `entry` will insert the key for the Yellow team with the value `50` because the Yellow team doesn’t have a value already. The second call to `entry` will not change the hash map because the Blue team already has the value `10`.

#### Updating a Value Based on the Old Value
Another common use case for hash maps is to look up a key’s value and then update it based on the old value. For instance, Listing 8-25 shows code that counts how many times each word appears in some text. We use a hash map with the words as keys and increment the value to keep track of how many times we’ve seen that word. If it’s the first time we’ve seen a word, we’ll first insert the value `0`.

```rust
use std::collections::HashMap;

let text = "hello world wonderful world";

let mut map = HashMap::new();

for word in text.split_whitespace() {
    let count = map.entry(word).or_insert(0);
    *count += 1;
}

println!("{map:?}");
```

[Listing 8-25](https://doc.rust-lang.org/book/ch08-03-hash-maps.html#listing-8-25): Counting occurrences of words using a hash map that stores words and counts

This code will print `{"world": 2, "hello": 1, "wonderful": 1}`. You might see the same key-value pairs printed in a different order: recall from [“Accessing Values in a Hash Map”](https://doc.rust-lang.org/book/ch08-03-hash-maps.html#accessing-values-in-a-hash-map) that iterating over a hash map happens in an arbitrary order.

The `split_whitespace` method returns an iterator over subslices, separated by whitespace, of the value in `text`. The `or_insert` method returns a mutable reference (`&mut V`) to the value for the specified key. Here, we store that mutable reference in the `count` variable, so in order to assign to that value, we must first dereference `count` using the asterisk (`*`). The mutable reference goes out of scope at the end of the `for` loop, so all of these changes are safe and allowed by the borrowing rules.
>  `split_whitespace` 方法返回对 subslices 的迭代器，由空格分开
>  `or_insert` 方法返回对指定 key 的 value 的可变引用 `&mut V`

### Hashing Functions
By default, `HashMap` uses a hashing function called _SipHash_ that can provide resistance to denial-of-service (DoS) attacks involving hash tables [1](https://doc.rust-lang.org/book/ch08-03-hash-maps.html#footnote-siphash). This is not the fastest hashing algorithm available, but the trade-off for better security that comes with the drop in performance is worth it. If you profile your code and find that the default hash function is too slow for your purposes, you can switch to another function by specifying a different hasher. A _hasher_ is a type that implements the `BuildHasher` trait. We’ll talk about traits and how to implement them in [Chapter 10](https://doc.rust-lang.org/book/ch10-02-traits.html). You don’t necessarily have to implement your own hasher from scratch; [crates.io](https://crates.io/) has libraries shared by other Rust users that provide hashers implementing many common hashing algorithms.
>  hasher 是实现了 `BuildHasher` trait 的类型

## Summary
Vectors, strings, and hash maps will provide a large amount of functionality necessary in programs when you need to store, access, and modify data. Here are some exercises you should now be equipped to solve:

1. Given a list of integers, use a vector and return the median (when sorted, the value in the middle position) and mode (the value that occurs most often; a hash map will be helpful here) of the list.
2. Convert strings to pig latin. The first consonant of each word is moved to the end of the word and _ay_ is added, so _first_ becomes _irst-fay_. Words that start with a vowel have _hay_ added to the end instead (_apple_ becomes _apple-hay_). Keep in mind the details about UTF-8 encoding!
3. Using a hash map and vectors, create a text interface to allow a user to add employee names to a department in a company; for example, “Add Sally to Engineering” or “Add Amir to Sales.” Then let the user retrieve a list of all people in a department or all people in the company by department, sorted alphabetically.

The standard library API documentation describes methods that vectors, strings, and hash maps have that will be helpful for these exercises!

We’re getting into more complex programs in which operations can fail, so it’s a perfect time to discuss error handling. We’ll do that next!

---

1. [https://en.wikipedia.org/wiki/SipHash](https://en.wikipedia.org/wiki/SipHash) [↩](https://doc.rust-lang.org/book/ch08-03-hash-maps.html#fr-siphash-1)

# 9 Error Handling
Errors are a fact of life in software, so Rust has a number of features for handling situations in which something goes wrong. In many cases, Rust requires you to acknowledge the possibility of an error and take some action before your code will compile. This requirement makes your program more robust by ensuring that you’ll discover errors and handle them appropriately before deploying your code to production!

Rust groups errors into two major categories: _recoverable_ and _unrecoverable_ errors. For a recoverable error, such as a _file not found_ error, we most likely just want to report the problem to the user and retry the operation. Unrecoverable errors are always symptoms of bugs, such as trying to access a location beyond the end of an array, and so we want to immediately stop the program.
>  Rust 将 errors 分为两类: 可恢复错误和不可恢复错误
>  可恢复错误例如 file not found error，一般要报告给用户，并重试
>  不可恢复错误通常来自于 bugs，例如数组访问越界，此时需要停止程序

Most languages don’t distinguish between these two kinds of errors and handle both in the same way, using mechanisms such as exceptions. Rust doesn’t have exceptions. Instead, it has the type `Result<T, E>` for recoverable errors and the `panic!` macro that stops execution when the program encounters an unrecoverable error. This chapter covers calling `panic!` first and then talks about returning `Result<T, E>` values. Additionally, we’ll explore considerations when deciding whether to try to recover from an error or to stop execution.
>  Rust 没有异常，对于可恢复错误，提供了类型 `Result<T, E>`，对于不可恢复错误，提供了 `panic!` macro 来停止程序运行

## 9.1 Unrecoverable Errors with `panic!`
Sometimes bad things happen in your code, and there’s nothing you can do about it. In these cases, Rust has the `panic!` macro. There are two ways to cause a panic in practice: by taking an action that causes our code to panic (such as accessing an array past the end) or by explicitly calling the `panic!` macro. In both cases, we cause a panic in our program. By default, these panics will print a failure message, unwind, clean up the stack, and quit. Via an environment variable, you can also have Rust display the call stack when a panic occurs to make it easier to track down the source of the panic.
>  实践中有两种方式导致程序 panic: 执行导致我们代码 panic 的行为 (例如数组访问越界) 或者显式调用 `panic!` macro
>  panic 默认会打印错误信息，展开并清理栈，然后退出程序
>  通过设置一个环境变量，可以让 Rust 在 panic 发生时展示调用栈

***Unwinding the Stack or Aborting in Response to a Panic***
By default, when a panic occurs the program starts _unwinding_, which means Rust walks back up the stack and cleans up the data from each function it encounters. However, walking back and cleaning up is a lot of work. Rust, therefore, allows you to choose the alternative of immediately _aborting_, which ends the program without cleaning up.
>  默认情况下，panic 发生时，程序会执行栈展开，即 Rust 沿着调用栈向上回溯，清理它遇到的每个函数的数据
>  但这一过程需要巨大开销，Rust 允许我们选择立即终止 (aborting)，即程序不进行清理直接退出

Memory that the program was using will then need to be cleaned up by the operating system. If in your project you need to make the resultant binary as small as possible, you can switch from unwinding to aborting upon a panic by adding `panic = 'abort'` to the appropriate `[profile]` sections in your _Cargo.toml_ file. For example, if you want to abort on panic in release mode, add this:
>  该程序使用的内存则需要由 OS 清理
>  如果我们希望项目的二进制文件越小越好，我们可以将 unwinding 改为 aborting，通过在 `[profile]` section 添加以下部分

```toml
[profile.release] 
panic = 'abort' 
```

Let’s try calling `panic!` in a simple program:

Filename: src/main.rs

```rust
fn main() {
    panic!("crash and burn");
}
```

When you run the program, you’ll see something like this:

```
$ cargo run
   Compiling panic v0.1.0 (file:///projects/panic)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.25s
     Running `target/debug/panic`

thread 'main' panicked at src/main.rs:2:5:
crash and burn
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

The call to `panic!` causes the error message contained in the last two lines. The first line shows our panic message and the place in our source code where the panic occurred: _src/main.rs:2:5_ indicates that it’s the second line, fifth character of our _src/main.rs_ file.
>  对 `panic!` 的调用导致了以上的最后两行错误信息
>  其中第一行显示了 panic message 和源码位置 (`src/main.rs` 的第二行，第五个字符)

In this case, the line indicated is part of our code, and if we go to that line, we see the `panic!` macro call. In other cases, the `panic!` call might be in code that our code calls, and the filename and line number reported by the error message will be someone else’s code where the `panic!` macro is called, not the line of our code that eventually led to the `panic!` call.

We can use the backtrace of the functions the `panic!` call came from to figure out the part of our code that is causing the problem. To understand how to use a `panic!` backtrace, let’s look at another example and see what it’s like when a `panic!` call comes from a library because of a bug in our code instead of from our code calling the macro directly. Listing 9-1 has some code that attempts to access an index in a vector beyond the range of valid indexes.
>  我们可以使用 `panic!` 函数调用的栈回溯来发现到底是什么导致了 panic

```rust
fn main() {
    let v = vec![1, 2, 3];

    v[99];
}
```

[Listing 9-1](https://doc.rust-lang.org/book/ch09-01-unrecoverable-errors-with-panic.html#listing-9-1): Attempting to access an element beyond the end of a vector, which will cause a call to `panic!`

Here, we’re attempting to access the 100th element of our vector (which is at index 99 because indexing starts at zero), but the vector has only three elements. In this situation, Rust will panic. Using `[]` is supposed to return an element, but if you pass an invalid index, there’s no element that Rust could return here that would be correct.

In C, attempting to read beyond the end of a data structure is undefined behavior. You might get whatever is at the location in memory that would correspond to that element in the data structure, even though the memory doesn’t belong to that structure. This is called a _buffer overread_ and can lead to security vulnerabilities if an attacker is able to manipulate the index in such a way as to read data they shouldn’t be allowed to that is stored after the data structure.
>  C 中尝试读取数据结构以外的数据属于未定义行为，即越界缓冲读取，可能引发安全漏洞

To protect your program from this sort of vulnerability, if you try to read an element at an index that doesn’t exist, Rust will stop execution and refuse to continue. Let’s try it and see:

```
$ cargo run
   Compiling panic v0.1.0 (file:///projects/panic)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.27s
     Running `target/debug/panic`

thread 'main' panicked at src/main.rs:4:6:
index out of bounds: the len is 3 but the index is 99
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

This error points at line 4 of our _main.rs_ where we attempt to access index `99` of the vector in `v`.
>  Rust 中，这会触发 panic

The `note:` line tells us that we can set the `RUST_BACKTRACE` environment variable to get a backtrace of exactly what happened to cause the error. A _backtrace_ is a list of all the functions that have been called to get to this point. Backtraces in Rust work as they do in other languages: the key to reading the backtrace is to start from the top and read until you see files you wrote. That’s the spot where the problem originated. The lines above that spot are code that your code has called; the lines below are code that called your code. These before-and-after lines might include core Rust code, standard library code, or crates that you’re using. 
>  如果设置了 `RUST_BACKTRACE=1`，我们可以获取导致 panic 的调用栈
>  backtrace (调用栈) 是指从程序执行开始到当前出错位置所调用的函数的列表
>  阅读调用栈的关键是从顶部开始查看，知道看到我们编写的文件为止，这个位置通常就是问题的根源
>  这个位置上面的代码是我们的代码所调用的函数，这个位置下面的代码是调用我们代码的函数，这些前后代码可能包含 Rust 核心代码、标准库代码、以及我们使用的 crates

Let’s try getting a backtrace by setting the `RUST_BACKTRACE` environment variable to any value except `0`. Listing 9-2 shows output similar to what you’ll see.

```
$ RUST_BACKTRACE=1 cargo run
thread 'main' panicked at src/main.rs:4:6:
index out of bounds: the len is 3 but the index is 99
stack backtrace:
   0: rust_begin_unwind at /rustc/4d91de4e48198da2e33413efdcd9cd2cc0c46688/library/std/src/panicking.rs:692:5
   1: core::panicking::panic_fmt at /rustc/4d91de4e48198da2e33413efdcd9cd2cc0c46688/library/core/src/panicking.rs:75:14
   2: core::panicking::panic_bounds_check at /rustc/4d91de4e48198da2e33413efdcd9cd2cc0c46688/library/core/src/panicking.rs:273:5
   3: <usize as core::slice::index::SliceIndex<[T]>>::index at file:///home/.rustup/toolchains/1.85/lib/rustlib/src/rust/library/core/src/slice/index.rs:274:10
   4: core::slice::index::<impl core::ops::index::Index<I> for [T]>::index at file:///home/.rustup/toolchains/1.85/lib/rustlib/src/rust/library/core/src/slice/index.rs:16:9
   5: <alloc::vec::Vec<T,A> as core::ops::index::Index<I>>::index at file:///home/.rustup/toolchains/1.85/lib/rustlib/src/rust/library/alloc/src/vec/mod.rs:3361:9
   6: panic::main at ./src/main.rs:4:6
   7: core::ops::function::FnOnce::call_once at file:///home/.rustup/toolchains/1.85/lib/rustlib/src/rust/library/core/src/ops/function.rs:250:5
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
```

[Listing 9-2](https://doc.rust-lang.org/book/ch09-01-unrecoverable-errors-with-panic.html#listing-9-2): The backtrace generated by a call to `panic!` displayed when the environment variable `RUST_BACKTRACE` is set

That’s a lot of output! The exact output you see might be different depending on your operating system and Rust version. In order to get backtraces with this information, debug symbols must be enabled. Debug symbols are enabled by default when using `cargo build` or `cargo run` without the `--release` flag, as we have here.
>  注意为了得到调用栈信息，需要启用 debug symbols，即 debug 编译默认启动的 symbols

In the output in Listing 9-2, line 6 of the backtrace points to the line in our project that’s causing the problem: line 4 of _src/main.rs_. If we don’t want our program to panic, we should start our investigation at the location pointed to by the first line mentioning a file we wrote. In Listing 9-1, where we deliberately wrote code that would panic, the way to fix the panic is to not request an element beyond the range of the vector indexes. When your code panics in the future, you’ll need to figure out what action the code is taking with what values to cause the panic and what the code should do instead.

We’ll come back to `panic!` and when we should and should not use `panic!` to handle error conditions in the [“To `panic!` or Not to `panic!`”](https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html#to-panic-or-not-to-panic) section later in this chapter. Next, we’ll look at how to recover from an error using `Result`.

## 9.2 Recoverable Errors with `Result`
Most errors aren’t serious enough to require the program to stop entirely. Sometimes when a function fails it’s for a reason that you can easily interpret and respond to. For example, if you try to open a file and that operation fails because the file doesn’t exist, you might want to create the file instead of terminating the process.
>  大多数错误并不足以完全停止程序，一些错误可以直接解决，例如尝试打开不存在的文件时，可以直接创建该文件而不是终止程序

Recall from [“Handling Potential Failure with `Result`”](https://doc.rust-lang.org/book/ch02-00-guessing-game-tutorial.html#handling-potential-failure-with-result) in Chapter 2 that the `Result` enum is defined as having two variants, `Ok` and `Err`, as follows:

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

The `T` and `E` are generic type parameters: we’ll discuss generics in more detail in Chapter 10. What you need to know right now is that `T` represents the type of the value that will be returned in a success case within the `Ok` variant, and `E` represents the type of the error that will be returned in a failure case within the `Err` variant. Because `Result` has these generic type parameters, we can use the `Result` type and the functions defined on it in many different situations where the success value and error value we want to return may differ.

>  `Result` enum 有两个变体 `Ok, Err`
>  `T, E` 是泛型参数，分表示两个变体绑定的值的类型

Let’s call a function that returns a `Result` value because the function could fail. In Listing 9-3 we try to open a file.

Filename: src/main.rs

```rust
use std::fs::File;

fn main() {
    let greeting_file_result = File::open("hello.txt");
}
```

[Listing 9-3](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#listing-9-3): Opening a file

The return type of `File::open` is a `Result<T, E>`. The generic parameter `T` has been filled in by the implementation of `File::open` with the type of the success value, `std::fs::File`, which is a file handle. The type of `E` used in the error value is `std::io::Error`. This return type means the call to `File::open` might succeed and return a file handle that we can read from or write to. The function call also might fail: for example, the file might not exist, or we might not have permission to access the file. The `File::open` function needs to have a way to tell us whether it succeeded or failed and at the same time give us either the file handle or error information. This information is exactly what the `Result` enum conveys.
>  函数返回 `Result` 类型表示了函数可能失败
>  `File::open` 的返回类型是 `Result<std::fs::File, std::io::Error>`，意味着 `File::open` 成功就返回一个文件句柄，否则返回错误 (例如文件不存在或者无权限访问时)

In the case where `File::open` succeeds, the value in the variable `greeting_file_result` will be an instance of `Ok` that contains a file handle. In the case where it fails, the value in `greeting_file_result` will be an instance of `Err` that contains more information about the kind of error that occurred.

We need to add to the code in Listing 9-3 to take different actions depending on the value `File::open` returns. Listing 9-4 shows one way to handle the `Result` using a basic tool, the `match` expression that we discussed in Chapter 6.
>  我们对结果使用 `match` 表达式，根据是否成功执行不同的路径

Filename: src/main.rs

```rust
use std::fs::File;

fn main() {
    let greeting_file_result = File::open("hello.txt");

    let greeting_file = match greeting_file_result {
        Ok(file) => file,
        Err(error) => panic!("Problem opening the file: {error:?}"),
    };
}
```


[Listing 9-4](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#listing-9-4): Using a `match` expression to handle the `Result` variants that might be returned

Note that, like the `Option` enum, the `Result` enum and its variants have been brought into scope by the prelude, so we don’t need to specify `Result::` before the `Ok` and `Err` variants in the `match` arms.
>  和 `Option` enum 一样，`Result` enum 和其变体都在 prelude 中

When the result is `Ok`, this code will return the inner `file` value out of the `Ok` variant, and we then assign that file handle value to the variable `greeting_file`. After the `match`, we can use the file handle for reading or writing.

The other arm of the `match` handles the case where we get an `Err` value from `File::open`. In this example, we’ve chosen to call the `panic!` macro. If there’s no file named _hello.txt_ in our current directory and we run this code, we’ll see the following output from the `panic!` macro:

```
$ cargo run
   Compiling error-handling v0.1.0 (file:///projects/error-handling)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.73s
     Running `target/debug/error-handling`

thread 'main' panicked at src/main.rs:8:23:
Problem opening the file: Os { code: 2, kind: NotFound, message: "No such file or directory" }
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

As usual, this output tells us exactly what has gone wrong.

### Matching on Different Errors
The code in Listing 9-4 will `panic!` no matter why `File::open` failed. However, we want to take different actions for different failure reasons. If `File::open` failed because the file doesn’t exist, we want to create the file and return the handle to the new file. If `File::open` failed for any other reason—for example, because we didn’t have permission to open the file—we still want the code to `panic!` in the same way it did in Listing 9-4. For this, we add an inner `match` expression, shown in Listing 9-5.
>  我们进一步考虑我们想要根据不同的失败原因执行不同的动作

Filename: src/main.rs

```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    let greeting_file_result = File::open("hello.txt");

    let greeting_file = match greeting_file_result {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("Problem creating the file: {e:?}"),
            },
            _ => {
                panic!("Problem opening the file: {error:?}");
            }
        },
    };
}
```

[Listing 9-5](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#listing-9-5): Handling different kinds of errors in different ways

The type of the value that `File::open` returns inside the `Err` variant is `io::Error`, which is a struct provided by the standard library. This struct has a method `kind` that we can call to get an `io::ErrorKind` value. The enum `io::ErrorKind` is provided by the standard library and has variants representing the different kinds of errors that might result from an `io` operation. The variant we want to use is `ErrorKind::NotFound`, which indicates the file we’re trying to open doesn’t exist yet. So we match on `greeting_file_result`, but we also have an inner match on `error.kind()`.
>  `io::Error` 类型是一个标准库提供的结构体，该结构体有 `kind` 方法，返回 `io::ErrorKind` value 
>  enum `io::ErrorKind` 由标准库提供，具有表示 IO 操作的各种错误的变体
>  `ErrorKind::NotFound` 表示要打开的文件不存在

The condition we want to check in the inner match is whether the value returned by `error.kind()` is the `NotFound` variant of the `ErrorKind` enum. If it is, we try to create the file with `File::create`. However, because `File::create` could also fail, we need a second arm in the inner `match` expression. When the file can’t be created, a different error message is printed. The second arm of the outer `match` stays the same, so the program panics on any error besides the missing file error.

***Alternatives to Using `match` with `Result<T, E>`***
That’s a lot of `match`! The `match` expression is very useful but also very much a primitive. In Chapter 13, you’ll learn about closures, which are used with many of the methods defined on `Result<T, E>`. These methods can be more concise than using `match` when handling `Result<T, E>` values in your code.
>  closures 会和许多定义在 `Result<T, E>` 上的方法一起使用，使用 closure 会比使用 `match` 更精简

For example, here’s another way to write the same logic as shown in Listing 9-5, this time using closures and the `unwrap_or_else` method:

```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    let greeting_file = File::open("hello.txt").unwrap_or_else(|error| {
        if error.kind() == ErrorKind::NotFound {
            File::create("hello.txt").unwrap_or_else(|error| {
                panic!("Problem creating the file: {error:?}");
            })
        } else {
            panic!("Problem opening the file: {error:?}");
        }
    });
}
```

Although this code has the same behavior as Listing 9-5, it doesn’t contain any `match` expressions and is cleaner to read. Come back to this example after you’ve read Chapter 13, and look up the `unwrap_or_else` method in the standard library documentation. Many more of these methods can clean up huge nested `match` expressions when you’re dealing with errors.

#### Shortcuts for Panic on Error: `unwrap` and `expect`
Using `match` works well enough, but it can be a bit verbose and doesn’t always communicate intent well. The `Result<T, E>` type has many helper methods defined on it to do various, more specific tasks. The `unwrap` method is a shortcut method implemented just like the `match` expression we wrote in Listing 9-4. If the `Result` value is the `Ok` variant, `unwrap` will return the value inside the `Ok`. If the `Result` is the `Err` variant, `unwrap` will call the `panic!` macro for us. Here is an example of `unwrap` in action:
>  `Result<T, E>` 类型有许多 helper 方法
>  `unwarp` 方法类似于 `match` 表达式的 shortcut，如果 `Result` value 为 `Ok`，`unwarp` 会返回 `Ok` 中的值，如果 `Result` value 为 `Err`，`unwarp` 则会调用 `panic!`

Filename: src/main.rs

```rust
use std::fs::File;

fn main() {
    let greeting_file = File::open("hello.txt").unwrap();
}
```

If we run this code without a _hello.txt_ file, we’ll see an error message from the `panic!` call that the `unwrap` method makes:

```
thread 'main' panicked at src/main.rs:4:49:
called `Result::unwrap()` on an `Err` value: Os { code: 2, kind: NotFound, message: "No such file or directory" }
```

Similarly, the `expect` method lets us also choose the `panic!` error message. Using `expect` instead of `unwrap` and providing good error messages can convey your intent and make tracking down the source of a panic easier. The syntax of `expect` looks like this:
>  `expect` 方法类似于 `unwarp`，但是还允许我们提供 `panic!` 错误信息

Filename: src/main.rs

```rust
use std::fs::File;

fn main() {
    let greeting_file = File::open("hello.txt")
        .expect("hello.txt should be included in this project");
}
```

We use `expect` in the same way as `unwrap`: to return the file handle or call the `panic!` macro. The error message used by `expect` in its call to `panic!` will be the parameter that we pass to `expect`, rather than the default `panic!` message that `unwrap` uses. Here’s what it looks like:

```
thread 'main' panicked at src/main.rs:5:10:
hello.txt should be included in this project: Os { code: 2, kind: NotFound, message: "No such file or directory" }
```

In production-quality code, most Rustaceans choose `expect` rather than `unwrap` and give more context about why the operation is expected to always succeed. That way, if your assumptions are ever proven wrong, you have more information to use in debugging.
>  通常偏好 `expect` 而不是 `unwarp`

### Propagating Errors
When a function’s implementation calls something that might fail, instead of handling the error within the function itself, you can return the error to the calling code so that it can decide what to do. This is known as _propagating_ the error and gives more control to the calling code, where there might be more information or logic that dictates how the error should be handled than what you have available in the context of your code.
>  函数除了可以自己处理错误以外，还可以将错误返回给调用者，即错误传播
>  调用者可能有更多关于如何处理错误的上下文信息

For example, Listing 9-6 shows a function that reads a username from a file. If the file doesn’t exist or can’t be read, this function will return those errors to the code that called the function.
>  例如下面的函数就定义为返回 `Result<String, io::Error>`，如果函数中出现错误，会直接返回 `Result` 的变体

Filename: src/main.rs

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let username_file_result = File::open("hello.txt");

    let mut username_file = match username_file_result {
        Ok(file) => file,
        Err(e) => return Err(e),
    };

    let mut username = String::new();

    match username_file.read_to_string(&mut username) {
        Ok(_) => Ok(username),
        Err(e) => Err(e),
    }
}
```

[Listing 9-6](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#listing-9-6): A function that returns errors to the calling code using `match`

This function can be written in a much shorter way, but we’re going to start by doing a lot of it manually in order to explore error handling; at the end, we’ll show the shorter way. Let’s look at the return type of the function first: `Result<String, io::Error>`. This means the function is returning a value of the type `Result<T, E>`, where the generic parameter `T` has been filled in with the concrete type `String` and the generic type `E` has been filled in with the concrete type `io::Error`.
>  返回类型是 `Result<T, E>` 的具体类型 `Result<String, io::Error>`

If this function succeeds without any problems, the code that calls this function will receive an `Ok` value that holds a `String`—the `username` that this function read from the file. If this function encounters any problems, the calling code will receive an `Err` value that holds an instance of `io::Error` that contains more information about what the problems were. We chose `io::Error` as the return type of this function because that happens to be the type of the error value returned from both of the operations we’re calling in this function’s body that might fail: the `File::open` function and the `read_to_string` method.

The body of the function starts by calling the `File::open` function. Then we handle the `Result` value with a `match` similar to the `match` in Listing 9-4. If `File::open` succeeds, the file handle in the pattern variable `file` becomes the value in the mutable variable `username_file` and the function continues. In the `Err` case, instead of calling `panic!`, we use the `return` keyword to return early out of the function entirely and pass the error value from `File::open`, now in the pattern variable `e`, back to the calling code as this function’s error value.

So, if we have a file handle in `username_file`, the function then creates a new `String` in variable `username` and calls the `read_to_string` method on the file handle in `username_file` to read the contents of the file into `username`. The `read_to_string` method also returns a `Result` because it might fail, even though `File::open` succeeded. So we need another `match` to handle that `Result`: if `read_to_string` succeeds, then our function has succeeded, and we return the username from the file that’s now in `username` wrapped in an `Ok`. If `read_to_string` fails, we return the error value in the same way that we returned the error value in the `match` that handled the return value of `File::open`. However, we don’t need to explicitly say `return`, because this is the last expression in the function.
>  file handle 的 `read_to_string` 方法会将文件内容读入 `username`，该函数的返回类型也是 `Result`，因为它可能失败

The code that calls this code will then handle getting either an `Ok` value that contains a username or an `Err` value that contains an `io::Error`. It’s up to the calling code to decide what to do with those values. If the calling code gets an `Err` value, it could call `panic!` and crash the program, use a default username, or look up the username from somewhere other than a file, for example. We don’t have enough information on what the calling code is actually trying to do, so we propagate all the success or error information upward for it to handle appropriately.

This pattern of propagating errors is so common in Rust that Rust provides the question mark operator `?` to make this easier.

#### A Shortcut for Propagating Errors: The `?` Operator
Listing 9-7 shows an implementation of `read_username_from_file` that has the same functionality as in Listing 9-6, but this implementation uses the `?` operator.

Filename: src/main.rs

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut username_file = File::open("hello.txt")?;
    let mut username = String::new();
    username_file.read_to_string(&mut username)?;
    Ok(username)
}
```

[Listing 9-7](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#listing-9-7): A function that returns errors to the calling code using the `?` operator

The `?` placed after a `Result` value is defined to work in almost the same way as the `match` expressions we defined to handle the `Result` values in Listing 9-6. If the value of the `Result` is an `Ok`, the value inside the `Ok` will get returned from this expression, and the program will continue. If the value is an `Err`, the `Err` will be returned from the whole function as if we had used the `return` keyword so the error value gets propagated to the calling code.
>  在一个 `Result` value 之后的 `?` 运算符的工作方式和 `match` 几乎一致
>  如果 `Result` value 是 `Ok`，则该表达式返回的就是 `Ok` 中的 value，程序会继续执行
>  如果 value 是 `Err`，则函数会直接返回 `Err` value，就好像我们使用了 ` return ` 让 error value 传播到调用者

There is a difference between what the `match` expression from Listing 9-6 does and what the `?` operator does: error values that have the `?` operator called on them go through the `from` function, defined in the `From` trait in the standard library, which is used to convert values from one type into another. When the `?` operator calls the `from` function, the error type received is converted into the error type defined in the return type of the current function. This is useful when a function returns one error type to represent all the ways a function might fail, even if parts might fail for many different reasons.
>  `match` 表达式和 `?` 运算符的关键区别是: 在 error values 上调用 `?` 会调用标准库中定义的 `From` trait 的 `from` 函数，该函数用于将一种类型的 value 转化为另一种类型
>  当 `?` 调用了 `from` 函数，它接收到的 error type 会被转化为定义在当前函数的返回类型中的 error type
>  这在函数返回单一的错误类型以表示函数所有可能失败的情况时很有用

For example, we could change the `read_username_from_file` function in Listing 9-7 to return a custom error type named `OurError` that we define. If we also define `impl From<io::Error> for OurError` to construct an instance of `OurError` from an `io::Error`, then the `?` operator calls in the body of `read_username_from_file` will call `from` and convert the error types without needing to add any more code to the function.
>  例如，我们可以改变 `read_username_from_file` 函数，让它返回自定义类型 `OurError`
>  如果我们还定义了 `From<io::Error> for OurError` 来从 `io::Error` 构造 `OurError` 的实例，那么 `?` 就会调用该 `from` 方法将错误类型转换

In the context of Listing 9-7, the `?` at the end of the `File::open` call will return the value inside an `Ok` to the variable `username_file`. If an error occurs, the `?` operator will return early out of the whole function and give any `Err` value to the calling code. The same thing applies to the `?` at the end of the `read_to_string` call.

The `?` operator eliminates a lot of boilerplate and makes this function’s implementation simpler. We could even shorten this code further by chaining method calls immediately after the `?`, as shown in Listing 9-8.

Filename: src/main.rs

```rust
use std::fs::File;
use std::io::{self, Read};

fn read_username_from_file() -> Result<String, io::Error> {
    let mut username = String::new();

    File::open("hello.txt")?.read_to_string(&mut username)?;

    Ok(username)
}
```

[Listing 9-8](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#listing-9-8): Chaining method calls after the `?` operator

We’ve moved the creation of the new `String` in `username` to the beginning of the function; that part hasn’t changed. Instead of creating a variable `username_file`, we’ve chained the call to `read_to_string` directly onto the result of `File::open("hello.txt")?`. We still have a `?` at the end of the `read_to_string` call, and we still return an `Ok` value containing `username` when both `File::open` and `read_to_string` succeed rather than returning errors. The functionality is again the same as in Listing 9-6 and Listing 9-7; this is just a different, more ergonomic way to write it.

Listing 9-9 shows a way to make this even shorter using `fs::read_to_string`.

Filename: src/main.rs

```rust
use std::fs;
use std::io;

fn read_username_from_file() -> Result<String, io::Error> {
    fs::read_to_string("hello.txt")
}
```

[Listing 9-9](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#listing-9-9): Using `fs::read_to_string` instead of opening and then reading the file

Reading a file into a string is a fairly common operation, so the standard library provides the convenient `fs::read_to_string` function that opens the file, creates a new `String`, reads the contents of the file, puts the contents into that `String`, and returns it. Of course, using `fs::read_to_string` doesn’t give us the opportunity to explain all the error handling, so we did it the longer way first.
>  标准库提供了 `fs::read_to_string` 函数，它打开文件，创造一个新 `String`，从文件中读取内容，将内容放入该 `String`，然后返回它

#### Where the `?` Operator Can Be Used
The `?` operator can only be used in functions whose return type is compatible with the value the `?` is used on. This is because the `?` operator is defined to perform an early return of a value out of the function, in the same manner as the `match` expression we defined in Listing 9-6. In Listing 9-6, the `match` was using a `Result` value, and the early return arm returned an `Err(e)` value. The return type of the function has to be a `Result` so that it’s compatible with this `return`.

In Listing 9-10, let’s look at the error we’ll get if we use the `?` operator in a `main` function with a return type that is incompatible with the type of the value we use `?` on.

Filename: src/main.rs

```rust
use std::fs::File;

fn main() {
    let greeting_file = File::open("hello.txt")?;
}
```

[Listing 9-10](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#listing-9-10): Attempting to use the `?` in the `main` function that returns `()` won’t compile.

This code opens a file, which might fail. The `?` operator follows the `Result` value returned by `File::open`, but this `main` function has the return type of `()`, not `Result`. When we compile this code, we get the following error message:

```
$ cargo run
   Compiling error-handling v0.1.0 (file:///projects/error-handling)
error[E0277]: the `?` operator can only be used in a function that returns `Result` or `Option` (or another type that implements `FromResidual`)
 --> src/main.rs:4:48
  |
3 | fn main() {
  | --------- this function should return `Result` or `Option` to accept `?`
4 |     let greeting_file = File::open("hello.txt")?;
  |                                                ^ cannot use the `?` operator in a function that returns `()`
  |
  = help: the trait `FromResidual<Result<Infallible, std::io::Error>>` is not implemented for `()`
help: consider adding return type
  |
3 ~ fn main() -> Result<(), Box<dyn std::error::Error>> {
4 |     let greeting_file = File::open("hello.txt")?;
5 +     Ok(())
  |

For more information about this error, try `rustc --explain E0277`.
error: could not compile `error-handling` (bin "error-handling") due to 1 previous error
```

This error points out that we’re only allowed to use the `?` operator in a function that returns `Result`, `Option`, or another type that implements `FromResidual`.

To fix the error, you have two choices. One choice is to change the return type of your function to be compatible with the value you’re using the `?` operator on as long as you have no restrictions preventing that. The other choice is to use a `match` or one of the `Result<T, E>` methods to handle the `Result<T, E>` in whatever way is appropriate.

The error message also mentioned that `?` can be used with `Option<T>` values as well. As with using `?` on `Result`, you can only use `?` on `Option` in a function that returns an `Option`. The behavior of the `?` operator when called on an `Option<T>` is similar to its behavior when called on a `Result<T, E>`: if the value is `None`, the `None` will be returned early from the function at that point. If the value is `Some`, the value inside the `Some` is the resultant value of the expression, and the function continues. Listing 9-11 has an example of a function that finds the last character of the first line in the given text.
>  `?` 也和 `Option<T>` 类型兼容: 如果 value 是 `None`，则返回 `None`，如果是 `Some`，则返回 `Some` 中的 value

```rust
fn last_char_of_first_line(text: &str) -> Option<char> {
    text.lines().next()?.chars().last()
}
```

[Listing 9-11](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#listing-9-11): Using the `?` operator on an `Option<T>` value

This function returns `Option<char>` because it’s possible that there is a character there, but it’s also possible that there isn’t. This code takes the `text` string slice argument and calls the `lines` method on it, which returns an iterator over the lines in the string. Because this function wants to examine the first line, it calls `next` on the iterator to get the first value from the iterator. If `text` is the empty string, this call to `next` will return `None`, in which case we use `?` to stop and return `None` from `last_char_of_first_line`. If `text` is not the empty string, `next` will return a `Some` value containing a string slice of the first line in `text`.
>  该函数接收 `&str`，并调用 `lines` 方法，该方法会返回一个迭代器
>  对迭代器调用 `next` 会得到第一个 value，如果没有 value 了，会返回 `None` (也就是返回类型依旧是 `Option<T>`)，我们使用 `? ` 来判断这一点

The `?` extracts the string slice, and we can call `chars` on that string slice to get an iterator of its characters. We’re interested in the last character in this first line, so we call `last` to return the last item in the iterator. This is an `Option` because it’s possible that the first line is the empty string; for example, if `text` starts with a blank line but has characters on other lines, as in `"\nhi"`. However, if there is a last character on the first line, it will be returned in the `Some` variant. The `?` operator in the middle gives us a concise way to express this logic, allowing us to implement the function in one line. If we couldn’t use the `?` operator on `Option`, we’d have to implement this logic using more method calls or a `match` expression.
>  如果迭代器顺利返回 value (string slice)，我们对它调用 `chars` 得到对字符的迭代器
>  要获得最后一个字符，我们调用 `last` 获取迭代器的最后一个 item (`last` 返回的也是 `Option<T>`，因为有可能 `chars` 得到的迭代器没有 value 可迭代)

Note that you can use the `?` operator on a `Result` in a function that returns `Result`, and you can use the `?` operator on an `Option` in a function that returns `Option`, but you can’t mix and match. The `?` operator won’t automatically convert a `Result` to an `Option` or vice versa; in those cases, you can use methods like the `ok` method on `Result` or the `ok_or` method on `Option` to do the conversion explicitly.
>  我们可以在返回 `Result` 的函数中对 `Result` 使用 `?`，也可以在返回 `Option` 的函数中对 `Option` 使用 `?`，但不能混合
>  在这种情况下，我们可以使用 `Resuslt` 的 `ok` 方法或 `Option` 的 `ok_or` 方法来进行显式类型转换

So far, all the `main` functions we’ve used return `()`. The `main` function is special because it’s the entry point and exit point of an executable program, and there are restrictions on what its return type can be for the program to behave as expected.
>  目前为止，我们的 `main` 都返回 `()`
>  对 `main` 函数的返回类型是有限制的

Luckily, `main` can also return a `Result<(), E>`. Listing 9-12 has the code from Listing 9-10, but we’ve changed the return type of `main` to be `Result<(), Box<dyn Error>>` and added a return value `Ok(())` to the end. This code will now compile.
>  `main` 可以返回 `Result<(), E>`，但不能返回 `Result<(), Box<dyn Error>>`

Filename: src/main.rs

```rust
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    let greeting_file = File::open("hello.txt")?;

    Ok(())
}
```

[Listing 9-12](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#listing-9-12): Changing `main` to return `Result<(), E>` allows the use of the `?` operator on `Result` values.

The `Box<dyn Error>` type is a _trait object_, which we’ll talk about in [“Using Trait Objects That Allow for Values of Different Types”](https://doc.rust-lang.org/book/ch18-02-trait-objects.html#using-trait-objects-that-allow-for-values-of-different-types) in Chapter 18. For now, you can read `Box<dyn Error>` to mean “any kind of error.” Using `?` on a `Result` value in a `main` function with the error type `Box<dyn Error>` is allowed because it allows any `Err` value to be returned early. Even though the body of this `main` function will only ever return errors of type `std::io::Error`, by specifying `Box<dyn Error>`, this signature will continue to be correct even if more code that returns other errors is added to the body of `main`.
>  `Box<dyn Error>` 是一个特质对象，它可以理解为任意类型的错误
>  在 `main` 中对 error type 为 `Box<dyn Error>` 的 `Err` value 使用 `?` 是允许的，但 `main` 函数本身只能返回 `std::io::Error`

When a `main` function returns a `Result<(), E>`, the executable will exit with a value of `0` if `main` returns `Ok(())` and will exit with a nonzero value if `main` returns an `Err` value. Executables written in C return integers when they exit: programs that exit successfully return the integer `0`, and programs that error return some integer other than `0`. Rust also returns integers from executables to be compatible with this convention.
>  如果 `main` 的返回值是 `Result<(), E>`，可执行文件的退出值要么是 `0` (`main` 返回 `Ok(())`)，要么是非零 (`main` 返回 `Err`)，这些值是整型值

The `main` function may return any types that implement [the `std::process::Termination` trait](https://doc.rust-lang.org/std/process/trait.Termination.html), which contains a function `report` that returns an `ExitCode`. Consult the standard library documentation for more information on implementing the `Termination` trait for your own types.
>  `main` 可以返回任意实现了 `std::process::Termination` trait 的类型，该 trait 包含了函数 `report`，返回 `ExitCode`

Now that we’ve discussed the details of calling `panic!` or returning `Result`, let’s return to the topic of how to decide which is appropriate to use in which cases.

## 9.3 To `panic!` or Not to `panic!`
So how do you decide when you should call `panic!` and when you should return `Result`? When code panics, there’s no way to recover. You could call `panic!` for any error situation, whether there’s a possible way to recover or not, but then you’re making the decision that a situation is unrecoverable on behalf of the calling code. When you choose to return a `Result` value, you give the calling code options. The calling code could choose to attempt to recover in a way that’s appropriate for its situation, or it could decide that an `Err` value in this case is unrecoverable, so it can call `panic!` and turn your recoverable error into an unrecoverable one. Therefore, returning `Result` is a good default choice when you’re defining a function that might fail.
>  当我们定义一个可能失败的函数时，返回 `Result` 是一个好的默认选择

In situations such as examples, prototype code, and tests, it’s more appropriate to write code that panics instead of returning a `Result`. Let’s explore why, then discuss situations in which the compiler can’t tell that failure is impossible, but you as a human can. The chapter will conclude with some general guidelines on how to decide whether to panic in library code.
>  对于示例代码、原型代码和测试场景，则更适合写 `panic` 的代码而不是返回 `Result`

### Examples, Prototype Code, and Tests
When you’re writing an example to illustrate some concept, also including robust error-handling code can make the example less clear. In examples, it’s understood that a call to a method like `unwrap` that could panic is meant as a placeholder for the way you’d want your application to handle errors, which can differ based on what the rest of your code is doing.
>  当编写示例来阐述某个概念时，引入健壮的错误处理代码反而会让示例更不清晰
>  在示例中，对例如 `unwarp` 等会导致 panic 的方法调用通常被视作一种占位符，表示这个地方我们希望应用程序以某种方式处理错误，但处理方式取决于我们剩下的代码做什么

Similarly, the `unwrap` and `expect` methods are very handy when prototyping, before you’re ready to decide how to handle errors. They leave clear markers in your code for when you’re ready to make your program more robust.
>  因此，`unwarp, expect` 方法在写原型时非常有用，在我们没决定好如何处理错误时，它们在代码中留下清晰的标记，等待我们后续的决定

If a method call fails in a test, you’d want the whole test to fail, even if that method isn’t the functionality under test. Because `panic!` is how a test is marked as a failure, calling `unwrap` or `expect` is exactly what should happen.
>  如果方法调用在测试中失败，我们希望整个测试失败，即使该方法不是该测试的核心功能
>  因为 `panic!` 就是标记测试失败的标准方式，因此调用 `unwarp, expect` 就是我们想要的

### Cases in Which You Have More Information Than the Compiler
It would also be appropriate to call `expect` when you have some other logic that ensures the `Result` will have an `Ok` value, but the logic isn’t something the compiler understands. You’ll still have a `Result` value that you need to handle: whatever operation you’re calling still has the possibility of failing in general, even though it’s logically impossible in your particular situation. If you can ensure by manually inspecting the code that you’ll never have an `Err` variant, it’s perfectly acceptable to call `expect` and document the reason you think you’ll never have an `Err` variant in the argument text. Here’s an example:
>  当我们有其他逻辑可以确保 `Result` 会有 `Ok` value，但这个逻辑不是编译器可以理解的，此时调用 `expect` 方法也是合适的，同时可以在文档中写下我们不处理 `Err` 变体的原因

```rust
use std::net::IpAddr;

let home: IpAddr = "127.0.0.1"
    .parse()
    .expect("Hardcoded IP address should be valid");

```

We’re creating an `IpAddr` instance by parsing a hardcoded string. We can see that `127.0.0.1` is a valid IP address, so it’s acceptable to use `expect` here. However, having a hardcoded, valid string doesn’t change the return type of the `parse` method: we still get a `Result` value, and the compiler will still make us handle the `Result` as if the `Err` variant is a possibility because the compiler isn’t smart enough to see that this string is always a valid IP address. If the IP address string came from a user rather than being hardcoded into the program and therefore _did_ have a possibility of failure, we’d definitely want to handle the `Result` in a more robust way instead. Mentioning the assumption that this IP address is hardcoded will prompt us to change `expect` to better error-handling code if, in the future, we need to get the IP address from some other source instead.

### Guidelines for Error Handling
It’s advisable to have your code panic when it’s possible that your code could end up in a bad state. In this context, a _bad state_ is when some assumption, guarantee, contract, or invariant has been broken, such as when invalid values, contradictory values, or missing values are passed to your code—plus one or more of the following:

- The bad state is something that is unexpected, as opposed to something that will likely happen occasionally, like a user entering data in the wrong format.
- Your code after this point needs to rely on not being in this bad state, rather than checking for the problem at every step.
- There’s not a good way to encode this information in the types you use. We’ll work through an example of what we mean in [“Encoding States and Behavior as Types”](https://doc.rust-lang.org/book/ch18-03-oo-design-patterns.html#encoding-states-and-behavior-as-types) in Chapter 18.

>  当我们的代码有可能位于不安全状态下，应该使用 `panic!` 
>  不安全状态即某些假设、保证、不变式被破坏，例如无效值、冲突值或缺失值，此外还要满足以下的一个或多个条件:
>  - 这种不安全状态是出乎意料的，而不是一些偶尔可能发生的情况，例如用户输入格式错误
>  - 在这一点之后的代码需要依赖于 “不处于该不安全状态”，它不会在每一步都检查问题
>  - 我们使用的类型无法很好的表示这种信息

If someone calls your code and passes in values that don’t make sense, it’s best to return an error if you can so the user of the library can decide what they want to do in that case. However, in cases where continuing could be insecure or harmful, the best choice might be to call `panic!` and alert the person using your library to the bug in their code so they can fix it during development. Similarly, `panic!` is often appropriate if you’re calling external code that is out of your control and it returns an invalid state that you have no way of fixing.
>  如果是代码被调用并传入了一些无意义的值，则最后返回错误，让用户可以决定如何修改，如果是继续执行可能导致不安全或有害，则最好调用 `panic!`
>  类似地，如果调用外部代码，它返回了我们无法处理的无效状态，也适合调用 `panic!`

However, when failure is expected, it’s more appropriate to return a `Result` than to make a `panic!` call. Examples include a parser being given malformed data or an HTTP request returning a status that indicates you have hit a rate limit. In these cases, returning a `Result` indicates that failure is an expected possibility that the calling code must decide how to handle.
>  如果失败是有预期的，则更适合返回 `Result` 而不是 `panic!`

When your code performs an operation that could put a user at risk if it’s called using invalid values, your code should verify the values are valid first and panic if the values aren’t valid. This is mostly for safety reasons: attempting to operate on invalid data can expose your code to vulnerabilities. This is the main reason the standard library will call `panic!` if you attempt an out-of-bounds memory access: trying to access memory that doesn’t belong to the current data structure is a common security problem. Functions often have _contracts_: their behavior is only guaranteed if the inputs meet particular requirements. Panicking when the contract is violated makes sense because a contract violation always indicates a caller-side bug, and it’s not a kind of error you want the calling code to have to explicitly handle. In fact, there’s no reasonable way for calling code to recover; the calling _programmers_ need to fix the code. Contracts for a function, especially when a violation will cause a panic, should be explained in the API documentation for the function.
>  如果代码对无效值进行操作会导致风险，则代码就应该先验证值的有效性，如果值无效，就 `panic!`，这主要是为了安全考虑
>  例如标准库的内存访问越界会 `panic!`
>  函数通常具有契约: 只有在输入满足特定要求，其行为才具有保证
>  当契约被违反时，panic 是合理的，因为契约被违反意味着调用方存在错误，且这类错误不应该由调用方代码显式处理，这应该由调用方程序员来亲自处理
>  函数的契约，尤其违反时会导致 panic 的契约，应该在 API 文档中解释

However, having lots of error checks in all of your functions would be verbose and annoying. Fortunately, you can use Rust’s type system (and thus the type checking done by the compiler) to do many of the checks for you. If your function has a particular type as a parameter, you can proceed with your code’s logic knowing that the compiler has already ensured you have a valid value. For example, if you have a type rather than an `Option`, your program expects to have _something_ rather than _nothing_. Your code then doesn’t have to handle two cases for the `Some` and `None` variants: it will only have one case for definitely having a value. Code trying to pass nothing to your function won’t even compile, so your function doesn’t have to check for that case at runtime. Another example is using an unsigned integer type such as `u32`, which ensures the parameter is never negative.
>  我们可以利用 Rust 的类型系统，通过编译器的类型检查帮助实现错误检查
>  例如如果参数类型不是 `Option`，函数内就不需要处理 `Some, None` 变体
>  又例如如果类型是 `u32`，就不需要考虑负数的情况

### Creating Custom Types for Validation
Let’s take the idea of using Rust’s type system to ensure we have a valid value one step further and look at creating a custom type for validation. Recall the guessing game in Chapter 2 in which our code asked the user to guess a number between 1 and 100. We never validated that the user’s guess was between those numbers before checking it against our secret number; we only validated that the guess was positive. In this case, the consequences were not very dire: our output of “Too high” or “Too low” would still be correct. But it would be a useful enhancement to guide the user toward valid guesses and have different behavior when the user guesses a number that’s out of range versus when the user types, for example, letters instead.
>  让我们进一步利用 Rust 的类型系统来确保我们具有一个有效的值，并尝试创建一个自定义类型来实现验证功能

One way to do this would be to parse the guess as an `i32` instead of only a `u32` to allow potentially negative numbers, and then add a check for the number being in range, like so:

Filename: src/main.rs

```rust
loop {
    // --snip--

    let guess: i32 = match guess.trim().parse() {
        Ok(num) => num,
        Err(_) => continue,
    };

    if guess < 1 || guess > 100 {
        println!("The secret number will be between 1 and 100.");
        continue;
    }

    match guess.cmp(&secret_number) {
        // --snip--
}
```

The `if` expression checks whether our value is out of range, tells the user about the problem, and calls `continue` to start the next iteration of the loop and ask for another guess. After the `if` expression, we can proceed with the comparisons between `guess` and the secret number knowing that `guess` is between 1 and 100.

However, this is not an ideal solution: if it were absolutely critical that the program only operated on values between 1 and 100, and it had many functions with this requirement, having a check like this in every function would be tedious (and might impact performance).

Instead, we can make a new type in a dedicated module and put the validations in a function to create an instance of the type rather than repeating the validations everywhere. That way, it’s safe for functions to use the new type in their signatures and confidently use the values they receive. Listing 9-13 shows one way to define a `Guess` type that will only create an instance of `Guess` if the `new` function receives a value between 1 and 100.
>  我们可以在一个专用的 module 中定义一个新类型，并将验证逻辑封装在一个用于创建该类型的实例的函数中，这样就不需要在许多地方重复验证逻辑
>  这样一来，在函数签名中使用了这个类型后，函数就可以放心地使用接收到的值

Filename: src/guessing_game.rs

```rust
pub struct Guess {
    value: i32,
}

impl Guess {
    pub fn new(value: i32) -> Guess {
        if value < 1 || value > 100 {
            panic!("Guess value must be between 1 and 100, got {value}.");
        }

        Guess { value }
    }

    pub fn value(&self) -> i32 {
        self.value
    }
}
```

[Listing 9-13](https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html#listing-9-13): A `Guess` type that will only continue with values between 1 and 100

Note that this code in _src/guessing_game.rs_ depends on adding a module declaration `mod guessing_game;` in _src/lib.rs_ that we haven’t shown here. Within this new module’s file, we define a struct in that module named `Guess` that has a field named `value` that holds an `i32`. This is where the number will be stored.

Then we implement an associated function named `new` on `Guess` that creates instances of `Guess` values. The `new` function is defined to have one parameter named `value` of type `i32` and to return a `Guess`. The code in the body of the `new` function tests `value` to make sure it’s between 1 and 100. If `value` doesn’t pass this test, we make a `panic!` call, which will alert the programmer who is writing the calling code that they have a bug they need to fix, because creating a `Guess` with a `value` outside this range would violate the contract that `Guess::new` is relying on. The conditions in which `Guess::new` might panic should be discussed in its public-facing API documentation; we’ll cover documentation conventions indicating the possibility of a `panic!` in the API documentation that you create in Chapter 14. If `value` does pass the test, we create a new `Guess` with its `value` field set to the `value` parameter and return the `Guess`.
>  我们在 `Guess::new` 中写上了判断逻辑，并且在逻辑不满足是 `panic!`，因为逻辑不满足即不满足 `Guess::new` 依赖的契约
>  `Guess::new` 的契约，以及它会 `panic!` 的情况应该在 API 文档中写明

Next, we implement a method named `value` that borrows `self`, doesn’t have any other parameters, and returns an `i32`. This kind of method is sometimes called a _getter_ because its purpose is to get some data from its fields and return it. This public method is necessary because the `value` field of the `Guess` struct is private. It’s important that the `value` field be private so code using the `Guess` struct is not allowed to set `value` directly: code outside the `guessing_game` module _must_ use the `Guess::new` function to create an instance of `Guess`, thereby ensuring there’s no way for a `Guess` to have a `value` that hasn’t been checked by the conditions in the `Guess::new` function.
>  `value` 方法是一个 getter 方法，方法为公有，返回私有的 field

A function that has a parameter or returns only numbers between 1 and 100 could then declare in its signature that it takes or returns a `Guess` rather than an `i32` and wouldn’t need to do any additional checks in its body.

## Summary
Rust’s error-handling features are designed to help you write more robust code. The `panic!` macro signals that your program is in a state it can’t handle and lets you tell the process to stop instead of trying to proceed with invalid or incorrect values. The `Result` enum uses Rust’s type system to indicate that operations might fail in a way that your code could recover from. You can use `Result` to tell code that calls your code that it needs to handle potential success or failure as well. Using `panic!` and `Result` in the appropriate situations will make your code more reliable in the face of inevitable problems.
>  `panic!` 宏用于表明程序处于无法处理的状态，它会通知程序停止运行，而不是继续尝试使用无效或错误的值
>  `Result` enum 使用 Rust 的类型系统明确表示某些操作会失效，但我们的代码也有能力从失败恢复

Now that you’ve seen useful ways that the standard library uses generics with the `Option` and `Result` enums, we’ll talk about how generics work and how you can use them in your code.

# 10 Generic Types, Traits, and Lifetimes
Every programming language has tools for effectively handling the duplication of concepts. In Rust, one such tool is _generics_: abstract stand-ins for concrete types or other properties. We can express the behavior of generics or how they relate to other generics without knowing what will be in their place when compiling and running the code.
>  Rust 处理概念的重复的工具就是泛型: 它是具体类型或其他属性的抽象占位符
>  我们可以在不预先知道泛型在编译时和运行时实际将被替换成什么的情况下，表示泛型的行为或它们彼此之间的关系

Functions can take parameters of some generic type, instead of a concrete type like `i32` or `String`, in the same way they take parameters with unknown values to run the same code on multiple concrete values. In fact, we’ve already used generics in Chapter 6 with `Option<T>`, in Chapter 8 with `Vec<T>` and `HashMap<K, V>`, and in Chapter 9 with `Result<T, E>`. In this chapter, you’ll explore how to define your own types, functions, and methods with generics!
>  函数可以接收泛型类型的参数，而不是具体类型，这和函数接收带有未知值的参数并在实际的多个具体值上运行的逻辑类似

First we’ll review how to extract a function to reduce code duplication. We’ll then use the same technique to make a generic function from two functions that differ only in the types of their parameters. We’ll also explain how to use generic types in struct and enum definitions.

Then you’ll learn how to use _traits_ to define behavior in a generic way. You can combine traits with generic types to constrain a generic type to accept only those types that have a particular behavior, as opposed to just any type.
>  我们将学习使用 traits 以通用的方法定义行为
>  我们可以将 traits 和泛型结合，来限制一个泛型类型只能接收具有特定行为的类型，而不是任意类型

Finally, we’ll discuss _lifetimes_: a variety of generics that give the compiler information about how references relate to each other. Lifetimes allow us to give the compiler enough information about borrowed values so that it can ensure references will be valid in more situations than it could without our help.
>  我们将讨论生命周期，这是一种特殊的泛型机制，用于向编译器提供关于引用之间的关系的信息
>  生命周期使得编译器可以获得足够多的关于借用值的信息，以确保引用将在更多的情况下都保持有效

***Removing Duplication by Extracting a Function***
Generics allow us to replace specific types with a placeholder that represents multiple types to remove code duplication. Before diving into generics syntax, let’s first look at how to remove duplication in a way that doesn’t involve generic types by extracting a function that replaces specific values with a placeholder that represents multiple values. Then we’ll apply the same technique to extract a generic function! By looking at how to recognize duplicated code you can extract into a function, you’ll start to recognize duplicated code that can use generics.

We’ll begin with the short program in Listing 10-1 that finds the largest number in a list.

Filename: src/main.rs

```rust
fn main() {
    let number_list = vec![34, 50, 25, 100, 65];

    let mut largest = &number_list[0];

    for number in &number_list {
        if number > largest {
            largest = number;
        }
    }

    println!("The largest number is {largest}");
}
```

[Listing 10-1](https://doc.rust-lang.org/book/ch10-00-generics.html#listing-10-1): Finding the largest number in a list of numbers

We store a list of integers in the variable `number_list` and place a reference to the first number in the list in a variable named `largest`. We then iterate through all the numbers in the list, and if the current number is greater than the number stored in `largest`, we replace the reference in that variable. However, if the current number is less than or equal to the largest number seen so far, the variable doesn’t change, and the code moves on to the next number in the list. After considering all the numbers in the list, `largest` should refer to the largest number, which in this case is 100.

We’ve now been tasked with finding the largest number in two different lists of numbers. To do so, we can choose to duplicate the code in Listing 10-1 and use the same logic at two different places in the program, as shown in Listing 10-2.

Filename: src/main.rs

```rust
fn main() {
    let number_list = vec![34, 50, 25, 100, 65];

    let mut largest = &number_list[0];

    for number in &number_list {
        if number > largest {
            largest = number;
        }
    }

    println!("The largest number is {largest}");

    let number_list = vec![102, 34, 6000, 89, 54, 2, 43, 8];

    let mut largest = &number_list[0];

    for number in &number_list {
        if number > largest {
            largest = number;
        }
    }

    println!("The largest number is {largest}");
}
```

[Listing 10-2](https://doc.rust-lang.org/book/ch10-00-generics.html#listing-10-2): Code to find the largest number in _two_ lists of numbers

Although this code works, duplicating code is tedious and error prone. We also have to remember to update the code in multiple places when we want to change it.

To eliminate this duplication, we’ll create an abstraction by defining a function that operates on any list of integers passed in as a parameter. This solution makes our code clearer and lets us express the concept of finding the largest number in a list abstractly.

In Listing 10-3, we extract the code that finds the largest number into a function named `largest`. Then we call the function to find the largest number in the two lists from Listing 10-2. We could also use the function on any other list of `i32` values we might have in the future.

Filename: src/main.rs

```rust
fn largest(list: &[i32]) -> &i32 {
    let mut largest = &list[0];

    for item in list {
        if item > largest {
            largest = item;
        }
    }

    largest
}

fn main() {
    let number_list = vec![34, 50, 25, 100, 65];

    let result = largest(&number_list);
    println!("The largest number is {result}");

    let number_list = vec![102, 34, 6000, 89, 54, 2, 43, 8];

    let result = largest(&number_list);
    println!("The largest number is {result}");
}
```

[Listing 10-3](https://doc.rust-lang.org/book/ch10-00-generics.html#listing-10-3): Abstracted code to find the largest number in two lists

The `largest` function has a parameter called `list`, which represents any concrete slice of `i32` values we might pass into the function. As a result, when we call the function, the code runs on the specific values that we pass in.

In summary, here are the steps we took to change the code from Listing 10-2 to Listing 10-3:

1. Identify duplicate code.
2. Extract the duplicate code into the body of the function, and specify the inputs and return values of that code in the function signature.
3. Update the two instances of duplicated code to call the function instead.

Next, we’ll use these same steps with generics to reduce code duplication. In the same way that the function body can operate on an abstract `list` instead of specific values, generics allow code to operate on abstract types.

For example, say we had two functions: one that finds the largest item in a slice of `i32` values and one that finds the largest item in a slice of `char` values. How would we eliminate that duplication? Let’s find out!

## 10.1 Generic Data Types
We use generics to create definitions for items like function signatures or structs, which we can then use with many different concrete data types. Let’s first look at how to define functions, structs, enums, and methods using generics. Then we’ll discuss how generics affect code performance.
>  函数签名和结构体都可以用泛型进行定义

### In Function Definitions
When defining a function that uses generics, we place the generics in the signature of the function where we would usually specify the data types of the parameters and return value. Doing so makes our code more flexible and provides more functionality to callers of our function while preventing code duplication.

Continuing with our `largest` function, Listing 10-4 shows two functions that both find the largest value in a slice. We’ll then combine these into a single function that uses generics.

Filename: src/main.rs

```rust
fn largest_i32(list: &[i32]) -> &i32 {
    let mut largest = &list[0];

    for item in list {
        if item > largest {
            largest = item;
        }
    }

    largest
}

fn largest_char(list: &[char]) -> &char {
    let mut largest = &list[0];

    for item in list {
        if item > largest {
            largest = item;
        }
    }

    largest
}

fn main() {
    let number_list = vec![34, 50, 25, 100, 65];

    let result = largest_i32(&number_list);
    println!("The largest number is {result}");

    let char_list = vec!['y', 'm', 'a', 'q'];

    let result = largest_char(&char_list);
    println!("The largest char is {result}");
}
```

[Listing 10-4](https://doc.rust-lang.org/book/ch10-01-syntax.html#listing-10-4): Two functions that differ only in their names and in the types in their signatures

The `largest_i32` function is the one we extracted in Listing 10-3 that finds the largest `i32` in a slice. The `largest_char` function finds the largest `char` in a slice. The function bodies have the same code, so let’s eliminate the duplication by introducing a generic type parameter in a single function.

To parameterize the types in a new single function, we need to name the type parameter, just as we do for the value parameters to a function. You can use any identifier as a type parameter name. But we’ll use `T` because, by convention, type parameter names in Rust are short, often just one letter, and Rust’s type-naming convention is CamelCase. Short for _type_, `T` is the default choice of most Rust programmers.
>  Rust 通常用单字母表示类型参数名，例如 `T`

When we use a parameter in the body of the function, we have to declare the parameter name in the signature so the compiler knows what that name means. Similarly, when we use a type parameter name in a function signature, we have to declare the type parameter name before we use it. To define the generic `largest` function, we place type name declarations inside angle brackets, `<>`, between the name of the function and the parameter list, like this:
>  当我们在函数体中使用参数时，我们必须在函数签名中声明了参数名称，编译器才知道这个名称的含义
>  类似地，当我们在函数签名中使用一个类型参数名称时，我们需要在使用它之前声明它，因此我们在函数和参数之前放上 `<T>`

```rust
fn largest<T>(list: &[T]) -> &T {
```

We read this definition as: the function `largest` is generic over some type `T`. This function has one parameter named `list`, which is a slice of values of type `T`. The `largest` function will return a reference to a value of the same type `T`.
>  上述定义下，函数 `largest` 是一个在某个类型 `T` 上的泛型
>  该函数有一个参数 `list`，这是一个 values 为 `T` 的 slice
>  该函数会返回一个对类型为 `T` 的 value 的引用

Listing 10-5 shows the combined `largest` function definition using the generic data type in its signature. The listing also shows how we can call the function with either a slice of `i32` values or `char` values. Note that this code won’t compile yet.

Filename: src/main.rs

```rust
fn largest<T>(list: &[T]) -> &T {
    let mut largest = &list[0];

    for item in list {
        if item > largest {
            largest = item;
        }
    }

    largest
}

fn main() {
    let number_list = vec![34, 50, 25, 100, 65];

    let result = largest(&number_list);
    println!("The largest number is {result}");

    let char_list = vec!['y', 'm', 'a', 'q'];

    let result = largest(&char_list);
    println!("The largest char is {result}");
}
```

[Listing 10-5](https://doc.rust-lang.org/book/ch10-01-syntax.html#listing-10-5): The `largest` function using generic type parameters; this doesn’t compile yet

If we compile this code right now, we’ll get this error:

```
$ cargo run
   Compiling chapter10 v0.1.0 (file:///projects/chapter10)
error[E0369]: binary operation `>` cannot be applied to type `&T`
 --> src/main.rs:5:17
  |
5 |         if item > largest {
  |            ---- ^ ------- &T
  |            |
  |            &T
  |
help: consider restricting type parameter `T` with trait `PartialOrd`
  |
1 | fn largest<T: std::cmp::PartialOrd>(list: &[T]) -> &T {
  |             ++++++++++++++++++++++

For more information about this error, try `rustc --explain E0369`.
error: could not compile `chapter10` (bin "chapter10") due to 1 previous error
```

The help text mentions `std::cmp::PartialOrd`, which is a _trait_, and we’re going to talk about traits in the next section. For now, know that this error states that the body of `largest` won’t work for all possible types that `T` could be. Because we want to compare values of type `T` in the body, we can only use types whose values can be ordered. To enable comparisons, the standard library has the `std::cmp::PartialOrd` trait that you can implement on types (see Appendix C for more on this trait). 
>  上述信息告诉我们 `largest` 的函数体无法适用于所有可能的 `T` 类型
>  这是因为我们在函数体中比较了类型 `T` 的 values
>  能够比较的类型需要实现标准库的 trait `std::cmp::PartialOrd`

To fix Listing 10-5, we can follow the help text’s suggestion and restrict the types valid for `T` to only those that implement `PartialOrd`. The listing will then compile, because the standard library implements `PartialOrd` on both `i32` and `char`.
>  我们遵循帮助文本的建议，限制对 `T` 有效的类型为实现了 `PartialOrd` 的类型
>  这样代码可以正常编译，因为标准库已经为 `i32, char` 实现了 `PartialOrd`

### In Struct Definitions
We can also define structs to use a generic type parameter in one or more fields using the `<>` syntax. Listing 10-6 defines a `Point<T>` struct to hold `x` and `y` coordinate values of any type.
>  结构体泛型的定义类似，同样使用 `<>` 语法

Filename: src/main.rs

```rust
struct Point<T> {
    x: T,
    y: T,
}

fn main() {
    let integer = Point { x: 5, y: 10 };
    let float = Point { x: 1.0, y: 4.0 };
}
```

[Listing 10-6](https://doc.rust-lang.org/book/ch10-01-syntax.html#listing-10-6): A `Point<T>` struct that holds `x` and `y` values of type `T`

The syntax for using generics in struct definitions is similar to that used in function definitions. First we declare the name of the type parameter inside angle brackets just after the name of the struct. Then we use the generic type in the struct definition where we would otherwise specify concrete data types.
>  我们首先在结构体名称后面声明类型参数的名称，然后在结构体定义中使用类型参数

Note that because we’ve used only one generic type to define `Point<T>`, this definition says that the `Point<T>` struct is generic over some type `T`, and the fields `x` and `y` are _both_ that same type, whatever that type may be. If we create an instance of a `Point<T>` that has values of different types, as in Listing 10-7, our code won’t compile.

Filename: src/main.rs

```rust
struct Point<T> {
    x: T,
    y: T,
}

fn main() {
    let wont_work = Point { x: 5, y: 4.0 };
}
```

[Listing 10-7](https://doc.rust-lang.org/book/ch10-01-syntax.html#listing-10-7): The fields `x` and `y` must be the same type because both have the same generic data type `T`.

In this example, when we assign the integer value `5` to `x`, we let the compiler know that the generic type `T` will be an integer for this instance of `Point<T>`. Then when we specify `4.0` for `y`, which we’ve defined to have the same type as `x`, we’ll get a type mismatch error like this:

```
$ cargo run
   Compiling chapter10 v0.1.0 (file:///projects/chapter10)
error[E0308]: mismatched types
 --> src/main.rs:7:38
  |
7 |     let wont_work = Point { x: 5, y: 4.0 };
  |                                      ^^^ expected integer, found floating-point number

For more information about this error, try `rustc --explain E0308`.
error: could not compile `chapter10` (bin "chapter10") due to 1 previous error
```

To define a `Point` struct where `x` and `y` are both generics but could have different types, we can use multiple generic type parameters. For example, in Listing 10-8, we change the definition of `Point` to be generic over types `T` and `U` where `x` is of type `T` and `y` is of type `U`.

Filename: src/main.rs

```rust
struct Point<T, U> {
    x: T,
    y: U,
}

fn main() {
    let both_integer = Point { x: 5, y: 10 };
    let both_float = Point { x: 1.0, y: 4.0 };
    let integer_and_float = Point { x: 5, y: 4.0 };
}
```

[Listing 10-8](https://doc.rust-lang.org/book/ch10-01-syntax.html#listing-10-8): A `Point<T, U>` generic over two types so that `x` and `y` can be values of different types

Now all the instances of `Point` shown are allowed! You can use as many generic type parameters in a definition as you want, but using more than a few makes your code hard to read. If you’re finding you need lots of generic types in your code, it could indicate that your code needs restructuring into smaller pieces.
>  使用更多的泛型也是允许的，但注意如果我们发现我们的代码需要使用很多泛型类型，这说明我们的代码需要重新组织成更小的模块

### In Enum Definitions
As we did with structs, we can define enums to hold generic data types in their variants. Let’s take another look at the `Option<T>` enum that the standard library provides, which we used in Chapter 6:
>  enum 同样可以作为泛型

```rust
enum Option<T> {
    Some(T),
    None,
}
```

This definition should now make more sense to you. As you can see, the `Option<T>` enum is generic over type `T` and has two variants: `Some`, which holds one value of type `T`, and a `None` variant that doesn’t hold any value. By using the `Option<T>` enum, we can express the abstract concept of an optional value, and because `Option<T>` is generic, we can use this abstraction no matter what the type of the optional value is.

Enums can use multiple generic types as well. The definition of the `Result` enum that we used in Chapter 9 is one example:

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

The `Result` enum is generic over two types, `T` and `E`, and has two variants: `Ok`, which holds a value of type `T`, and `Err`, which holds a value of type `E`. This definition makes it convenient to use the `Result` enum anywhere we have an operation that might succeed (return a value of some type `T`) or fail (return an error of some type `E`). In fact, this is what we used to open a file in Listing 9-3, where `T` was filled in with the type `std::fs::File` when the file was opened successfully and `E` was filled in with the type `std::io::Error` when there were problems opening the file.

When you recognize situations in your code with multiple struct or enum definitions that differ only in the types of the values they hold, you can avoid duplication by using generic types instead.

### In Method Definitions
We can implement methods on structs and enums (as we did in Chapter 5) and use generic types in their definitions too. Listing 10-9 shows the `Point<T>` struct we defined in Listing 10-6 with a method named `x` implemented on it.
>  为泛型 struct, enum 实现方法时，可以使用泛型类型

Filename: src/main.rs

```rust
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

fn main() {
    let p = Point { x: 5, y: 10 };

    println!("p.x = {}", p.x());
}
```

[Listing 10-9](https://doc.rust-lang.org/book/ch10-01-syntax.html#listing-10-9): Implementing a method named `x` on the `Point<T>` struct that will return a reference to the `x` field of type `T`

Here, we’ve defined a method named `x` on `Point<T>` that returns a reference to the data in the field `x`.

Note that we have to declare `T` just after `impl` so we can use `T` to specify that we’re implementing methods on the type `Point<T>`. By declaring `T` as a generic type after `impl`, Rust can identify that the type in the angle brackets in `Point` is a generic type rather than a concrete type. We could have chosen a different name for this generic parameter than the generic parameter declared in the struct definition, but using the same name is conventional. 
>  我们在 `impl` 之后声明了 `T`，以使用 `T` 来表示我们正在为类型 `Point<T>` 实现方法
>  只有在 `impl` 之后写了 `<T>`，Rust 才知道 `Point<T>` 中的 `T` 是一个泛型类型，而不是具体类型
>  我们可以在这里为泛型类型选择一个新名字，但通常会使用类型泛型声明中的名字

If you write a method within an `impl` that declares a generic type, that method will be defined on any instance of the type, no matter what concrete type ends up substituting for the generic type.
>  如果我们在 `impl` 实现了一个声明了泛型类型的方法，则这个方法会为泛型 struct, enum 的任意实例都实现，无论最终使用什么具体类型来实例化

We can also specify constraints on generic types when defining methods on the type. We could, for example, implement methods only on `Point<f32>` instances rather than on `Point<T>` instances with any generic type. In Listing 10-10 we use the concrete type `f32`, meaning we don’t declare any types after `impl`.
>  在为泛型类型定义方法时，我们也可以声明约束
>  例如我们可以仅为 `Point<f32>` 实例类型实现方法，而不是为 `Point<T>` 泛型类型实现
>  为实例类型实现方法时，我们不需要为 `impl` 声明泛型类型，只需要明确是为实例类型 `Point<f32>` `impl` 即可

Filename: src/main.rs

```rust
impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}
```

[Listing 10-10](https://doc.rust-lang.org/book/ch10-01-syntax.html#listing-10-10): An `impl` block that only applies to a struct with a particular concrete type for the generic type parameter `T`

This code means the type `Point<f32>` will have a `distance_from_origin` method; other instances of `Point<T>` where `T` is not of type `f32` will not have this method defined. The method measures how far our point is from the point at coordinates (0.0, 0.0) and uses mathematical operations that are available only for floating-point types.

Generic type parameters in a struct definition aren’t always the same as those you use in that same struct’s method signatures. Listing 10-11 uses the generic types `X1` and `Y1` for the `Point` struct and `X2` `Y2` for the `mixup` method signature to make the example clearer. The method creates a new `Point` instance with the `x` value from the `self` `Point` (of type `X1`) and the `y` value from the passed-in `Point` (of type `Y2`).
>  struct 定义中的泛型类型参数并不总是和 struct 方法签名中的泛型类型参数相同
>  例如下例为泛型类型 `Point<X1, Y1>` 实现了泛型方法 `mixup<X2, Y2>`，这个方法又会返回一个新的泛型 struct `Point<X1, Y2>`

>  反正就是各种泛型，只要不冲突，到最后编译时再一一实例化即可

Filename: src/main.rs

```rust
struct Point<X1, Y1> {
    x: X1,
    y: Y1,
}

impl<X1, Y1> Point<X1, Y1> {
    fn mixup<X2, Y2>(self, other: Point<X2, Y2>) -> Point<X1, Y2> {
        Point {
            x: self.x,
            y: other.y,
        }
    }
}

fn main() {
    let p1 = Point { x: 5, y: 10.4 };
    let p2 = Point { x: "Hello", y: 'c' };

    let p3 = p1.mixup(p2);

    println!("p3.x = {}, p3.y = {}", p3.x, p3.y);
}
```

[Listing 10-11](https://doc.rust-lang.org/book/ch10-01-syntax.html#listing-10-11): A method that uses generic types different from its struct’s definition

In `main`, we’ve defined a `Point` that has an `i32` for `x` (with value `5`) and an `f64` for `y` (with value `10.4`). The `p2` variable is a `Point` struct that has a string slice for `x` (with value `"Hello"`) and a `char` for `y` (with value `c`). Calling `mixup` on `p1` with the argument `p2` gives us `p3`, which will have an `i32` for `x` because `x` came from `p1`. The `p3` variable will have a `char` for `y` because `y` came from `p2`. The `println!` macro call will print `p3.x = 5, p3.y = c`.

The purpose of this example is to demonstrate a situation in which some generic parameters are declared with `impl` and some are declared with the method definition. Here, the generic parameters `X1` and `Y1` are declared after `impl` because they go with the struct definition. The generic parameters `X2` and `Y2` are declared after `fn mixup` because they’re only relevant to the method.
>  这个例子展示了有时有些泛型参数是在 `impl` 中声明，有些泛型参数则在方法定义中声明
>  注意在 `impl` 中声明的泛型类型可以在 `impl` 中使用，在方法定义中声明的泛型类型则只能在方法中使用

### Performance of Code Using Generics
You might be wondering whether there is a runtime cost when using generic type parameters. The good news is that using generic types won’t make your program run any slower than it would with concrete types.

Rust accomplishes this by performing monomorphization of the code using generics at compile time. _Monomorphization_ is the process of turning generic code into specific code by filling in the concrete types that are used when compiled. In this process, the compiler does the opposite of the steps we used to create the generic function in Listing 10-5: the compiler looks at all the places where generic code is called and generates code for the concrete types the generic code is called with.
>  泛型不会带来运行时开销
>  Rust 会在编译时为泛型代码执行单态化，即在编译时使用具体类型来填充泛型类型
>  编译器会查找所有泛型代码的调用处，使用调用处的具体类型来生成使用具体类型的代码

Let’s look at how this works by using the standard library’s generic `Option<T>` enum:

```rust
let integer = Some(5); 
let float = Some(5.0);
```

When Rust compiles this code, it performs monomorphization. During that process, the compiler reads the values that have been used in `Option<T>` instances and identifies two kinds of `Option<T>`: one is `i32` and the other is `f64`. As such, it expands the generic definition of `Option<T>` into two definitions specialized to `i32` and `f64`, thereby replacing the generic definition with the specific ones.

The monomorphized version of the code looks similar to the following (the compiler uses different names than what we’re using here for illustration):

Filename: src/main.rs

```rust
enum Option_i32 {
    Some(i32),
    None,
}

enum Option_f64 {
    Some(f64),
    None,
}

fn main() {
    let integer = Option_i32::Some(5);
    let float = Option_f64::Some(5.0);
}
```

The generic `Option<T>` is replaced with the specific definitions created by the compiler. Because Rust compiles generic code into code that specifies the type in each instance, we pay no runtime cost for using generics. When the code runs, it performs just as it would if we had duplicated each definition by hand. The process of monomorphization makes Rust’s generics extremely efficient at runtime.

## 10.2 Traits: Defining Shared Behavior
A _trait_ defines the functionality a particular type has and can share with other types. We can use traits to define shared behavior in an abstract way. We can use _trait bounds_ to specify that a generic type can be any type that has certain behavior.
>  特质定义了某个特定类型所具有的功能，该功能可以和其他类型共享
>  我们使用特质来抽象地定义共享的行为
>  我们使用特质约束来指定泛型类型可以是任何具备特定行为的类型

Note: Traits are similar to a feature often called _interfaces_ in other languages, although with some differences.
>  特质类似于其他语言中的接口，但存在一些差异

### Defining a Trait
A type’s behavior consists of the methods we can call on that type. Different types share the same behavior if we can call the same methods on all of those types. Trait definitions are a way to group method signatures together to define a set of behaviors necessary to accomplish some purpose.
>  类型的行为由我们可以对该类型调用的方法组成
>  如果我们可以为不同的类型调用相同的方法，它们就共享相同的行为
>  特质定义是一种将方法签名组合在一起并定义完成某个目的所必须的一组行为的一种方式

For example, let’s say we have multiple structs that hold various kinds and amounts of text: a `NewsArticle` struct that holds a news story filed in a particular location and a `SocialPost` that can have, at most, 280 characters along with metadata that indicates whether it was a new post, a repost, or a reply to another post.
>  例如，假设我们有多个结构体，用于存储不同种类和数量的文本
>  `NewsArtical` struct 存储某个地方的新闻，`SoclaiPost` struct 存储帖子以及它的元数据

We want to make a media aggregator library crate named `aggregator` that can display summaries of data that might be stored in a `NewsArticle` or `SocialPost` instance. To do this, we need a summary from each type, and we’ll request that summary by calling a `summarize` method on an instance. Listing 10-12 shows the definition of a public `Summary` trait that expresses this behavior.
>  我们想编写一个 `aggregator` crate，展示存储在 `NewsArticle` 或 `SocialPost` 中的数据总结
>  为此，我们希望这两个类型都具有 `summarize` 方法，我们通过定义一个公有特质 ` Summay ` 表示这个行为

Filename: src/lib.rs

```rust
pub trait Summary {
    fn summarize(&self) -> String;
}
```

[Listing 10-12](https://doc.rust-lang.org/book/ch10-02-traits.html#listing-10-12): A `Summary` trait that consists of the behavior provided by a `summarize` method

Here, we declare a trait using the `trait` keyword and then the trait’s name, which is `Summary` in this case. We also declare the trait as `pub` so that crates depending on this crate can make use of this trait too, as we’ll see in a few examples. Inside the curly brackets, we declare the method signatures that describe the behaviors of the types that implement this trait, which in this case is `fn summarize(&self) -> String`.
>  我们使用 `trait` 关键字声明特质，并且将它声明为 `pub`，使得其他 crate 可以使用这个特质
>  我们在特质中声明了方法签名，描述了实现该特质的类型的行为，既 `fn summarize(&self) -> String`

After the method signature, instead of providing an implementation within curly brackets, we use a semicolon. Each type implementing this trait must provide its own custom behavior for the body of the method. The compiler will enforce that any type that has the `Summary` trait will have the method `summarize` defined with this signature exactly.
>  编译器会要求所有实现了这个特质的类型都必须提供特质指定的方法的实现，即实现了具有相同签名的方法

A trait can have multiple methods in its body: the method signatures are listed one per line, and each line ends in a semicolon.
>  特质可以有多个方法

### Implementing a Trait on a Type
Now that we’ve defined the desired signatures of the `Summary` trait’s methods, we can implement it on the types in our media aggregator. Listing 10-13 shows an implementation of the `Summary` trait on the `NewsArticle` struct that uses the headline, the author, and the location to create the return value of `summarize`. For the `SocialPost` struct, we define `summarize` as the username followed by the entire text of the post, assuming that the post content is already limited to 280 characters.

Filename: src/lib.rs

```rust
pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}

pub struct SocialPost {
    pub username: String,
    pub content: String,
    pub reply: bool,
    pub repost: bool,
}

impl Summary for SocialPost {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}
```

[Listing 10-13](https://doc.rust-lang.org/book/ch10-02-traits.html#listing-10-13): Implementing the `Summary` trait on the `NewsArticle` and `SocialPost` types

Implementing a trait on a type is similar to implementing regular methods. The difference is that after `impl`, we put the trait name we want to implement, then use the `for` keyword, and then specify the name of the type we want to implement the trait for. Within the `impl` block, we put the method signatures that the trait definition has defined. Instead of adding a semicolon after each signature, we use curly brackets and fill in the method body with the specific behavior that we want the methods of the trait to have for the particular type.
>  在类型上实现特质类似于实现常规方法
>  差异在于我们在 `impl` 之后写下我们要实现的特质名称，然后使用 `for` 关键字，指定实现该特质的类型

Now that the library has implemented the `Summary` trait on `NewsArticle` and `SocialPost`, users of the crate can call the trait methods on instances of `NewsArticle` and `SocialPost` in the same way we call regular methods. The only difference is that the user must bring the trait into scope as well as the types. Here’s an example of how a binary crate could use our `aggregator` library crate:
>  实现了特质之后，我们就可以像调用常规方法一样调用特质方法
>  差异在于用户必须将特质带入作用域，就和类型一样

```rust
use aggregator::{SocialPost, Summary};

fn main() {
    let post = SocialPost {
        username: String::from("horse_ebooks"),
        content: String::from(
            "of course, as you probably already know, people",
        ),
        reply: false,
        repost: false,
    };

    println!("1 new post: {}", post.summarize());
}
```

This code prints `1 new post: horse_ebooks: of course, as you probably already know, people`.

Other crates that depend on the `aggregator` crate can also bring the `Summary` trait into scope to implement `Summary` on their own types. One restriction to note is that we can implement a trait on a type only if either the trait or the type, or both, are local to our crate. For example, we can implement standard library traits like `Display` on a custom type like `SocialPost` as part of our `aggregator` crate functionality because the type `SocialPost` is local to our `aggregator` crate. We can also implement `Summary` on `Vec<T>` in our `aggregator` crate because the trait `Summary` is local to our `aggregator` crate.
>  其他依赖于 `aggregator` crate 的 crates 也可以将 `Summary` trait 带入作用域，并为它们的类型实现 `Summary` trait
>  Rust 编译器会执行 “孤儿规则”，它要求我们这个特质或这个类型，或两者，是在当前 crate 中定义的，才能在这个类型上实现这个特质
>  例如，我们可以在我们 crate 内定义的类型 `SocialPost` 上实现标准库特质 ` Dispaly `，因为 `SocialPost` 是定义在当前 crate 中的，我们也可以在我们的 crate 中为标准库类型 `Vec<T>` 实现特质 `Summary`，因为特质 `Summary` 是定义在当前 crate 中的

But we can’t implement external traits on external types. For example, we can’t implement the `Display` trait on `Vec<T>` within our `aggregator` crate because `Display` and `Vec<T>` are both defined in the standard library and aren’t local to our `aggregator` crate. This restriction is part of a property called _coherence_, and more specifically the _orphan rule_, so named because the parent type is not present. This rule ensures that other people’s code can’t break your code and vice versa. Without the rule, two crates could implement the same trait for the same type, and Rust wouldn’t know which implementation to use.
>  但我们不能为外部类型实现外部特质
>  例如我们不能在 `aggregator` crate 中为 `Vec<T>` 实现 `Display` ，因为二者都定义在标准库中，不是本地的
>  这个限制是一致性原则的一部分，即孤儿规则，之所以这么命名，是因为其父类型 (即所属的 crate) 不在当前上下文中
>  这个规则确保不会破坏其他人的代码，如果没有这个规则，两个 crates 可以为相同类型实现相同的特质，Rust 就不知道使用哪一个

### Default Implementations
Sometimes it’s useful to have default behavior for some or all of the methods in a trait instead of requiring implementations for all methods on every type. Then, as we implement the trait on a particular type, we can keep or override each method’s default behavior.

In Listing 10-14, we specify a default string for the `summarize` method of the `Summary` trait instead of only defining the method signature, as we did in Listing 10-12.
>  我们可以为特质方法提供默认实现

Filename: src/lib.rs

```rust
pub trait Summary {
    fn summarize(&self) -> String {
        String::from("(Read more...)")
    }
}
```

[Listing 10-14](https://doc.rust-lang.org/book/ch10-02-traits.html#listing-10-14): Defining a `Summary` trait with a default implementation of the `summarize` method

To use a default implementation to summarize instances of `NewsArticle`, we specify an empty `impl` block with `impl Summary for NewsArticle {}`.
>  如果我们要让类型实现特质，但不想提供自己的实现而是依赖默认实现，`impl` 一个空的 block 即可

Even though we’re no longer defining the `summarize` method on `NewsArticle` directly, we’ve provided a default implementation and specified that `NewsArticle` implements the `Summary` trait. As a result, we can still call the `summarize` method on an instance of `NewsArticle`, like this:

```rust
let article = NewsArticle {
    headline: String::from("Penguins win the Stanley Cup Championship!"),
    location: String::from("Pittsburgh, PA, USA"),
    author: String::from("Iceburgh"),
    content: String::from(
        "The Pittsburgh Penguins once again are the best \
         hockey team in the NHL.",
    ),
};

println!("New article available! {}", article.summarize());
```

This code prints `New article available! (Read more...)`.

Creating a default implementation doesn’t require us to change anything about the implementation of `Summary` on `SocialPost` in Listing 10-13. The reason is that the syntax for overriding a default implementation is the same as the syntax for implementing a trait method that doesn’t have a default implementation.

Default implementations can call other methods in the same trait, even if those other methods don’t have a default implementation. In this way, a trait can provide a lot of useful functionality and only require implementors to specify a small part of it. For example, we could define the `Summary` trait to have a `summarize_author` method whose implementation is required, and then define a `summarize` method that has a default implementation that calls the `summarize_author` method:
>  特质的默认方法可以调用相同特质中的其他方法，即便这些方法没有默认实现
>  通过这种方式，特质可以提供许多有用的方法，并仅要求实现者实现其中的一小部分
>  例如，我们可以定义 `Summary` 特质具有 `summarize_author` 方法，要求实现者实现，并定义 `summarize` 方法，具有一个调用 `summarize_author` 方法的默认实现

To use this version of `Summary`, we only need to define `summarize_author` when we implement the trait on a type:

```rust
impl Summary for SocialPost {
    fn summarize_author(&self) -> String {
        format!("@{}", self.username)
    }
}
```

After we define `summarize_author`, we can call `summarize` on instances of the `SocialPost` struct, and the default implementation of `summarize` will call the definition of `summarize_author` that we’ve provided. Because we’ve implemented `summarize_author`, the `Summary` trait has given us the behavior of the `summarize` method without requiring us to write any more code. Here’s what that looks like:

```rust
let post = SocialPost {
    username: String::from("horse_ebooks"),
    content: String::from(
        "of course, as you probably already know, people",
    ),
    reply: false,
    repost: false,
};

println!("1 new post: {}", post.summarize());
```

This code prints `1 new post: (Read more from @horse_ebooks...)`.

Note that it isn’t possible to call the default implementation from an overriding implementation of that same method.
>  注意不能从某个默认实现的覆盖实现中调用默认实现

### Traits as Parameters
Now that you know how to define and implement traits, we can explore how to use traits to define functions that accept many different types. We’ll use the `Summary` trait we implemented on the `NewsArticle` and `SocialPost` types in Listing 10-13 to define a `notify` function that calls the `summarize` method on its `item` parameter, which is of some type that implements the `Summary` trait. To do this, we use the `impl Trait` syntax, like this:
>  我们可以使用特质来定义接收不同类型参数的函数

```rust
pub fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize());
}
```

Instead of a concrete type for the `item` parameter, we specify the `impl` keyword and the trait name. This parameter accepts any type that implements the specified trait. In the body of `notify`, we can call any methods on `item` that come from the `Summary` trait, such as `summarize`. We can call `notify` and pass in any instance of `NewsArticle` or `SocialPost`. Code that calls the function with any other type, such as a `String` or an `i32`, won’t compile because those types don’t implement `Summary`.
>  上述函数的参数 `item` 没有使用具体类型，其类型为 `impl Trait`
>  `impl` 关键字 + 特质名称说明了该函数接收任意实现了该特质的类型，我们在函数体中可以对该类型调用特质中的方法

#### Trait Bound Syntax
The `impl Trait` syntax works for straightforward cases but is actually syntax sugar for a longer form known as a _trait bound_; it looks like this:

```rust
pub fn notify<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}
```

This longer form is equivalent to the example in the previous section but is more verbose. We place trait bounds with the declaration of the generic type parameter after a colon and inside angle brackets.
>  `impl Trait` 语法实际上是一种更长形式的语法糖
>  完整的形式称为特质约束，其写法如上，可以看到实际上函数接收的是一种泛型，并且这个泛型应该满足特质约束

The `impl Trait` syntax is convenient and makes for more concise code in simple cases, while the fuller trait bound syntax can express more complexity in other cases. For example, we can have two parameters that implement `Summary`. Doing so with the `impl Trait` syntax looks like this:

```rust
pub fn notify(item1: &impl Summary, item2: &impl Summary) {
```

Using `impl Trait` is appropriate if we want this function to allow `item1` and `item2` to have different types (as long as both types implement `Summary`). If we want to force both parameters to have the same type, however, we must use a trait bound, like this:

```rust
pub fn notify<T: Summary>(item1: &T, item2: &T) {
```

The generic type `T` specified as the type of the `item1` and `item2` parameters constrains the function such that the concrete type of the value passed as an argument for `item1` and `item2` must be the same.

#### Specifying Multiple Trait Bounds with the `+` Syntax
We can also specify more than one trait bound. Say we wanted `notify` to use display formatting as well as `summarize` on `item`: we specify in the `notify` definition that `item` must implement both `Display` and `Summary`. We can do so using the `+` syntax:

```rust
pub fn notify(item: &(impl Summary + Display)) {
```

The `+` syntax is also valid with trait bounds on generic types:

```rust
pub fn notify<T: Summary + Display>(item: &T) {
```

With the two trait bounds specified, the body of `notify` can call `summarize` and use `{}` to format `item`.
>  要指定多个特质约束，我们可以使用 `+`

#### Clearer Trait Bounds with `where` Clauses
Using too many trait bounds has its downsides. Each generic has its own trait bounds, so functions with multiple generic type parameters can contain lots of trait bound information between the function’s name and its parameter list, making the function signature hard to read. For this reason, Rust has alternate syntax for specifying trait bounds inside a `where` clause after the function signature. So, instead of writing this:

```rust
fn some_function<T: Display + Clone, U: Clone + Debug>(t: &T, u: &U) -> i32 {
```

we can use a `where` clause, like this:

```rust
fn some_function<T, U>(t: &T, u: &U) -> i32
where
    T: Display + Clone,
    U: Clone + Debug,
{
```

This function’s signature is less cluttered: the function name, parameter list, and return type are close together, similar to a function without lots of trait bounds.

>  为泛型指定特质约束的另一种语法是在函数签名后使用 `where` 子句

### Returning Types That Implement Traits
We can also use the `impl Trait` syntax in the return position to return a value of some type that implements a trait, as shown here:

```rust
fn returns_summarizable() -> impl Summary {
    SocialPost {
        username: String::from("horse_ebooks"),
        content: String::from(
            "of course, as you probably already know, people",
        ),
        reply: false,
        repost: false,
    }
}
```

By using `impl Summary` for the return type, we specify that the `returns_summarizable` function returns some type that implements the `Summary` trait without naming the concrete type. In this case, `returns_summarizable` returns a `SocialPost`, but the code calling this function doesn’t need to know that.
>  特质约束同样可以用于函数的返回类型

The ability to specify a return type only by the trait it implements is especially useful in the context of closures and iterators, which we cover in Chapter 13. Closures and iterators create types that only the compiler knows or types that are very long to specify. The `impl Trait` syntax lets you concisely specify that a function returns some type that implements the `Iterator` trait without needing to write out a very long type.
>  通过特质指定返回类型在闭包和迭代器中十分有用
>  闭包和迭代器会生成只有编译器才知道的类型，这些类型名称非常长，难以书写
>  而 `impl Trait` 语法可以让我们简洁地指定实现了 `Iterator` 特质的返回类型，无须写出很长的类型名称

However, you can only use `impl Trait` if you’re returning a single type. For example, this code that returns either a `NewsArticle` or a `SocialPost` with the return type specified as `impl Summary` wouldn’t work:

```rust
fn returns_summarizable(switch: bool) -> impl Summary {
    if switch {
        NewsArticle {
            headline: String::from(
                "Penguins win the Stanley Cup Championship!",
            ),
            location: String::from("Pittsburgh, PA, USA"),
            author: String::from("Iceburgh"),
            content: String::from(
                "The Pittsburgh Penguins once again are the best \
                 hockey team in the NHL.",
            ),
        }
    } else {
        SocialPost {
            username: String::from("horse_ebooks"),
            content: String::from(
                "of course, as you probably already know, people",
            ),
            reply: false,
            repost: false,
        }
    }
}
```

Returning either a `NewsArticle` or a `SocialPost` isn’t allowed due to restrictions around how the `impl Trait` syntax is implemented in the compiler. We’ll cover how to write a function with this behavior in the [“Using Trait Objects That Allow for Values of Different Types”](https://doc.rust-lang.org/book/ch18-02-trait-objects.html#using-trait-objects-that-allow-for-values-of-different-types) section of Chapter 18.
>  但注意返回 `impl Trait` 的函数仍然只能返回单个类型

### Using Trait Bounds to Conditionally Implement Methods
By using a trait bound with an `impl` block that uses generic type parameters, we can implement methods conditionally for types that implement the specified traits. For example, the type `Pair<T>` in Listing 10-15 always implements the `new` function to return a new instance of `Pair<T>` (recall from the [“Defining Methods”](https://doc.rust-lang.org/book/ch05-03-method-syntax.html#defining-methods) section of Chapter 5 that `Self` is a type alias for the type of the `impl` block, which in this case is `Pair<T>`). But in the next `impl` block, `Pair<T>` only implements the `cmp_display` method if its inner type `T` implements the `PartialOrd` trait that enables comparison _and_ the `Display` trait that enables printing.
>  在 `impl` block 中为泛型类型参数添加特质约束，可以表示为实现了特质的泛型类型实现方法
>  例如，下面的例子说明 `Pair<T>` 只有在 `T` 实现了 `PartialOrd, Display` 特质的情况下才实现 `cmp_display` 方法

Filename: src/lib.rs

```rust
use std::fmt::Display;

struct Pair<T> {
    x: T,
    y: T,
}

impl<T> Pair<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T: Display + PartialOrd> Pair<T> {
    fn cmp_display(&self) {
        if self.x >= self.y {
            println!("The largest member is x = {}", self.x);
        } else {
            println!("The largest member is y = {}", self.y);
        }
    }
}
```

[Listing 10-15](https://doc.rust-lang.org/book/ch10-02-traits.html#listing-10-15): Conditionally implementing methods on a generic type depending on trait bounds

We can also conditionally implement a trait for any type that implements another trait. Implementations of a trait on any type that satisfies the trait bounds are called _blanket implementations_ and are used extensively in the Rust standard library. For example, the standard library implements the `ToString` trait on any type that implements the `Display` trait. The `impl` block in the standard library looks similar to this code:
>  我们也可以使用特质约束，指明为已经实现了某个特质的泛型类型实现新的特质
>  这种在满足特质约束的任意类型上实现特质的方法称为空白实现，在 Rust 标准库中广泛使用
>  例如，标准库为任意实现了 `Display` 特质的类型实现了 `ToString` 特质，其代码形式类似于:

>  注意为泛型类型实现特质时必须有特质约束，否则无法编译
>  因为这样就表示为所有类型实现特质，违反了 “泛化需要有限度” 的原则，我们可以为一个特定类型实现特质，可以为满足约束的一组类型实现特质，但不能为所有类型实现特质

```rust
impl<T: Display> ToString for T {
    // --snip--
}
```

Because the standard library has this blanket implementation, we can call the `to_string` method defined by the `ToString` trait on any type that implements the `Display` trait. For example, we can turn integers into their corresponding `String` values like this because integers implement `Display`:
>  因为这样的空白实现，我们可以为任意实现了 `Display` 特质的类型调用特质 `ToString` 类型下的 `to_string` 方法

```rust
let s = 3.to_string(); 
```

Blanket implementations appear in the documentation for the trait in the “Implementors” section.

Traits and trait bounds let us write code that uses generic type parameters to reduce duplication but also specify to the compiler that we want the generic type to have particular behavior. The compiler can then use the trait bound information to check that all the concrete types used with our code provide the correct behavior. In dynamically typed languages, we would get an error at runtime if we called a method on a type which didn’t define the method. But Rust moves these errors to compile time so we’re forced to fix the problems before our code is even able to run. Additionally, we don’t have to write code that checks for behavior at runtime because we’ve already checked at compile time. Doing so improves performance without having to give up the flexibility of generics.
>  编译器可以使用特质约束信息来检查代码中使用的具体类型是否提供正确行为
>  在动态类型语言中，我们将在为某个类型调用它未定义的方法时获得运行时错误 (但是这个一般静态检查器也可以检查出来吧)，但 Rust 在编译时移除了这些错误，并且，我们可以不用编写运行时检查行为的代码，因为编译时已经保证了
>  这即提高了性能，也没有放弃泛型的灵活性

## 10.3 Validating References with Lifetimes
Lifetimes are another kind of generic that we’ve already been using. Rather than ensuring that a type has the behavior we want, lifetimes ensure that references are valid as long as we need them to be.
>  生命周期也是一类泛型，与确保类型具备我们期望的行为不同，生命周期确保引用在其需要的有效期内始终有效

One detail we didn’t discuss in the [“References and Borrowing”](https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html#references-and-borrowing) section in Chapter 4 is that every reference in Rust has a _lifetime_, which is the scope for which that reference is valid. Most of the time, lifetimes are implicit and inferred, just like most of the time, types are inferred. We are only required to annotate types when multiple types are possible. In a similar way, we have to annotate lifetimes when the lifetimes of references could be related in a few different ways. Rust requires us to annotate the relationships using generic lifetime parameters to ensure the actual references used at runtime will definitely be valid.
>  Rust 中每个引用都有生命周期，它表示该引用有效的范围
>  大多数情况下，生命周期是隐式推断的，就像大多数情况下类型也是推断的，我们只需要具有多种可能类型时才需要标记类型
>  类似地，我们只需要在引用的生命周期可能以不同的方式相互关联时，才需要显示标注生命周期
>  Rust 要求我们使用泛型生命周期参数来标注这些关系，确保在运行时实际使用的引用一定是有效的

Annotating lifetimes is not even a concept most other programming languages have, so this is going to feel unfamiliar. Although we won’t cover lifetimes in their entirety in this chapter, we’ll discuss common ways you might encounter lifetime syntax so you can get comfortable with the concept.

### Preventing Dangling References with Lifetimes
The main aim of lifetimes is to prevent _dangling references_, which cause a program to reference data other than the data it’s intended to reference. Consider the program in Listing 10-16, which has an outer scope and an inner scope.
>  生命周期的主要目标是避免悬挂引用，这类引用会导致程序引用到不是它想引用的数据

```rust
fn main() {
    let r;

    {
        let x = 5;
        r = &x;
    }

    println!("r: {r}");
}
```

[Listing 10-16](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#listing-10-16): An attempt to use a reference whose value has gone out of scope

Note: The examples in Listings 10-16, 10-17, and 10-23 declare variables without giving them an initial value, so the variable name exists in the outer scope. At first glance, this might appear to be in conflict with Rust’s having no null values. However, if we try to use a variable before giving it a value, we’ll get a compile-time error, which shows that Rust indeed does not allow null values.
>  上述示例声明了变量但没有赋初始值，因此变量名存在于外部作用域
>  乍一看，这似乎与 Rust 不支持空值的特性相冲突，但如果我们在尝试给变量赋值之前使用它，编译器会报错，这恰恰说明了 Rust 不允许空值存在

The outer scope declares a variable named `r` with no initial value, and the inner scope declares a variable named `x` with the initial value of `5`. Inside the inner scope, we attempt to set the value of `r` as a reference to `x`. Then the inner scope ends, and we attempt to print the value in `r`. This code won’t compile because the value that `r` is referring to has gone out of scope before we try to use it. Here is the error message:
>  外部作用域声明了名为 `r` 没有初始值的变量
>  内部作用域声明了名为 `x` 初始值为 `5` 的变量
>  在内部作用域，我们尝试将 `r` 的值设为对 `x` 的引用，内部作用域结束后，我们尝试打印 `r` 中的值
>  这将编译失败，因为 `r` 引用的值在我们使用它之前就离开了作用域

```
$ cargo run
   Compiling chapter10 v0.1.0 (file:///projects/chapter10)
error[E0597]: `x` does not live long enough
 --> src/main.rs:6:13
  |
5 |         let x = 5;
  |             - binding `x` declared here
6 |         r = &x;
  |             ^^ borrowed value does not live long enough
7 |     }
  |     - `x` dropped here while still borrowed
8 |
9 |     println!("r: {r}");
  |                  --- borrow later used here

For more information about this error, try `rustc --explain E0597`.
error: could not compile `chapter10` (bin "chapter10") due to 1 previous error
```

The error message says that the variable `x` “does not live long enough.” The reason is that `x` will be out of scope when the inner scope ends on line 7. But `r` is still valid for the outer scope; because its scope is larger, we say that it “lives longer.” If Rust allowed this code to work, `r` would be referencing memory that was deallocated when `x` went out of scope, and anything we tried to do with `r` wouldn’t work correctly. So how does Rust determine that this code is invalid? It uses a borrow checker.
>  如果我们让这部分代码编译，那么 `r` 所引用的内存将在 `x` 离开作用域之后就被释放，导致 bug
>  Rust 使用借用检查器来决定这样的代码是否有效

### The Borrow Checker
The Rust compiler has a _borrow checker_ that compares scopes to determine whether all borrows are valid. Listing 10-17 shows the same code as Listing 10-16 but with annotations showing the lifetimes of the variables.
>  Rust 编译器具有借用检查器，它会比较作用域，来决定所有的借用是否有效
>  下例展示了所有变量的声明周期

```rust
fn main() {
    let r;                // ---------+-- 'a
                          //          |
    {                     //          |
        let x = 5;        // -+-- 'b  |
        r = &x;           //  |       |
    }                     // -+       |
                          //          |
    println!("r: {r}");   //          |
}                         // ---------+
```

[Listing 10-17](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#listing-10-17): Annotations of the lifetimes of `r` and `x`, named `'a` and `'b`, respectively

Here, we’ve annotated the lifetime of `r` with `'a` and the lifetime of `x` with `'b`. As you can see, the inner `'b` block is much smaller than the outer `'a` lifetime block. At compile time, Rust compares the size of the two lifetimes and sees that `r` has a lifetime of `'a` but that it refers to memory with a lifetime of `'b`. The program is rejected because `'b` is shorter than `'a`: the subject of the reference doesn’t live as long as the reference.
>  在编译时，Rust 会比较引用本身的生命周期和它所引用变量的生命周期，如果它所引用变量的生命周期比引用本身短，就编译失败

Listing 10-18 fixes the code so it doesn’t have a dangling reference and it compiles without any errors.

```rust
fn main() {
    let x = 5;            // ----------+-- 'b
                          //           |
    let r = &x;           // --+-- 'a  |
                          //   |       |
    println!("r: {r}");   //   |       |
                          // --+       |
}                         // ----------+
```

[Listing 10-18](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#listing-10-18): A valid reference because the data has a longer lifetime than the reference

Here, `x` has the lifetime `'b`, which in this case is larger than `'a`. This means `r` can reference `x` because Rust knows that the reference in `r` will always be valid while `x` is valid.

Now that you know where the lifetimes of references are and how Rust analyzes lifetimes to ensure references will always be valid, let’s explore generic lifetimes of parameters and return values in the context of functions.

### Generic Lifetimes in Functions
We’ll write a function that returns the longer of two string slices. This function will take two string slices and return a single string slice. After we’ve implemented the `longest` function, the code in Listing 10-19 should print `The longest string is abcd`.
>  我们实现一个函数，它接收两个 string slices，返回二者中较长的 string slice

Filename: src/main.rs

```rust
fn main() {
    let string1 = String::from("abcd");
    let string2 = "xyz";

    let result = longest(string1.as_str(), string2);
    println!("The longest string is {result}");
}
```

[Listing 10-19](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#listing-10-19): A `main` function that calls the `longest` function to find the longer of two string slices

Note that we want the function to take string slices, which are references, rather than strings, because we don’t want the `longest` function to take ownership of its parameters. Refer to [“String Slices as Parameters”](https://doc.rust-lang.org/book/ch04-03-slices.html#string-slices-as-parameters) in Chapter 4 for more discussion about why the parameters we use in Listing 10-19 are the ones we want.

If we try to implement the `longest` function as shown in Listing 10-20, it won’t compile.

Filename: src/main.rs

```rust
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() { x } else { y }
}
```

[Listing 10-20](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#listing-10-20): An implementation of the `longest` function that returns the longer of two string slices but does not yet compile

Instead, we get the following error that talks about lifetimes:

```
$ cargo run
   Compiling chapter10 v0.1.0 (file:///projects/chapter10)
error[E0106]: missing lifetime specifier
 --> src/main.rs:9:33
  |
9 | fn longest(x: &str, y: &str) -> &str {
  |               ----     ----     ^ expected named lifetime parameter
  |
  = help: this function's return type contains a borrowed value, but the signature does not say whether it is borrowed from `x` or `y`
help: consider introducing a named lifetime parameter
  |
9 | fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
  |           ++++     ++          ++          ++

For more information about this error, try `rustc --explain E0106`.
error: could not compile `chapter10` (bin "chapter10") due to 1 previous error
```

The help text reveals that the return type needs a generic lifetime parameter on it because Rust can’t tell whether the reference being returned refers to `x` or `y`. Actually, we don’t know either, because the `if` block in the body of this function returns a reference to `x` and the `else` block returns a reference to `y`!
>  朴素地实现这个函数将不能编译，帮助信息说明返回类型需要有一个泛型生命周期参数，因为 Rust 无法判断函数所返回的引用是指向 `x` 还是 `y` 的

When we’re defining this function, we don’t know the concrete values that will be passed into this function, so we don’t know whether the `if` case or the `else` case will execute. We also don’t know the concrete lifetimes of the references that will be passed in, so we can’t look at the scopes as we did in Listings 10-17 and 10-18 to determine whether the reference we return will always be valid. The borrow checker can’t determine this either, because it doesn’t know how the lifetimes of `x` and `y` relate to the lifetime of the return value. To fix this error, we’ll add generic lifetime parameters that define the relationship between the references so the borrow checker can perform its analysis.
>  我们在定义该函数时，无法确定传给函数的具体值，因此我们不知道是否是执行 `if` 还是 `else`
>  我们也不知道传入的引用的具体生命周期，因此我们无法通过作用域判断是否我们返回的引用总是有效的
>  借用检查器也无法判断，因为它不知道 `x,y` 的生命周期是如何和返回值的生命周期相关联的
>  为此，我们需要添加泛型生命周期参数，它定义了引用之间的关系，便于借用检查器执行它的分析

>  Rust 的借用检查器不能静态判断返回的引用生命周期和哪个输入的生命周期相关联，也就是它不知道返回的引用是 “借自” `x` 还是 `y`

>  假设 Rust 允许这样写

```rust
fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() { x } else { y }
}
```

>  然后我们这样调用：

```rust
fn main() {
    let string1 = String::from("long string");
    let result;
    {
        let string2 = String::from("xyz");
        result = longest(string1.as_str(), string2.as_str());
        // 👆 string2 在这里结束作用域！
    }
    println!("The longest string is {}", result); // ❗️使用 result
}
```

>  `string2` 是一个局部变量，在 `{}` 结束时被释放，而 `longest` 函数可能返回的是 `string2.as_str()` —— 也就是指向 `string2` 的引用。
>  但 `string2` 已经被销毁了，因此在 `{}` 之外， `result` 就是一个悬挂引用，在 `println!` 时访问它，就是未定义行为

### Lifetime Annotation Syntax
Lifetime annotations don’t change how long any of the references live. Rather, they describe the relationships of the lifetimes of multiple references to each other without affecting the lifetimes. Just as functions can accept any type when the signature specifies a generic type parameter, functions can accept references with any lifetime by specifying a generic lifetime parameter.
>  生命周期标注不会改变任何引用的实际存活时间，它们只是描述多个引用的生命周期之间的关系
>  类似于函数可以在签名为泛型类型参数时接收任意类型，函数可以在指定泛型生命周期参数时接收具有任意生命周期的引用

Lifetime annotations have a slightly unusual syntax: the names of lifetime parameters must start with an apostrophe (`'`) and are usually all lowercase and very short, like generic types. Most people use the name `'a` for the first lifetime annotation. We place lifetime parameter annotations after the `&` of a reference, using a space to separate the annotation from the reference’s type.
>  生命周期参数的名字必须以 `'` 开头，通常为小写且非常短，类似于泛型类型
>  我们在一个引用的 `&` 之后写下生命周期参数标注，使用空格分开引用的类型和引用的生命周期参数

Here are some examples: a reference to an `i32` without a lifetime parameter, a reference to an `i32` that has a lifetime parameter named `'a`, and a mutable reference to an `i32` that also has the lifetime `'a`.

```rust
&i32        // a reference
&'a i32     // a reference with an explicit lifetime
&'a mut i32 // a mutable reference with an explicit lifetime
```

One lifetime annotation by itself doesn’t have much meaning because the annotations are meant to tell Rust how generic lifetime parameters of multiple references relate to each other. Let’s examine how the lifetime annotations relate to each other in the context of the `longest` function.
>  仅有一个生命周期标注没有太多意义，因为这些标注的目的是告诉 Rust 多个引用的泛型生命周期参数是如何互相关联的

### Lifetime Annotations in Function Signatures
To use lifetime annotations in function signatures, we need to declare the generic _lifetime_ parameters inside angle brackets between the function name and the parameter list, just as we did with generic _type_ parameters.
>  要在函数签名中使用生命周期标注，我们需要在 `<>` 中声明泛型生命周期参数，和声明泛型类型参数类似

We want the signature to express the following constraint: the returned reference will be valid as long as both the parameters are valid. This is the relationship between lifetimes of the parameters and the return value. We’ll name the lifetime `'a` and then add it to each reference, as shown in Listing 10-21.
>  我们希望签名表示以下的约束:
>  返回的引用的将和参数的有效期保持一致，这就是参数的生命周期和返回值的生命周期之间的关系
>  我们将泛型生命周期参数命名为 `'a`，并为每个引用都加上这个泛型生命周期参数

Filename: src/main.rs

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

[Listing 10-21](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#listing-10-21): The `longest` function definition specifying that all the references in the signature must have the same lifetime `'a`

This code should compile and produce the result we want when we use it with the `main` function in Listing 10-19.

The function signature now tells Rust that for some lifetime `'a`, the function takes two parameters, both of which are string slices that live at least as long as lifetime `'a`. The function signature also tells Rust that the string slice returned from the function will live at least as long as lifetime `'a`. In practice, it means that the lifetime of the reference returned by the `longest` function is the same as the smaller of the lifetimes of the values referred to by the function arguments. These relationships are what we want Rust to use when analyzing this code.
>  现在的函数签名告诉 Rust: 对于某个生命周期 `'a`，该函数接受两个参数，这两个参数都是至少与生命周期 `'a` 存活时间一样长的 string slice
>  该函数签名还告诉 Rust 该函数返回的 string slice  的存活时间也至少和生命周期 `'a` 一样长
>  实际上，这意味着 `longest` 函数返回的引用的生命周期和它的参数所引用的值的生命周期中较短的那个相同
>  这正是我们希望 Rust 分析代码时使用的关系

Remember, when we specify the lifetime parameters in this function signature, we’re not changing the lifetimes of any values passed in or returned. Rather, we’re specifying that the borrow checker should reject any values that don’t adhere to these constraints. Note that the `longest` function doesn’t need to know exactly how long `x` and `y` will live, only that some scope can be substituted for `'a` that will satisfy this signature.
>  我们在函数签名中指定生命周期参数时，我们并不会改变传入或传出的值的生命周期，我们是在告诉借用检查器: 拒绝不遵循这些约束的值
>  注意 `longest` 函数不需要准确知道 `x,y` 具体会存活多久，它只需要知道**存在某个作用域可以替换 `'a` 来满足签名中的要求**

When annotating lifetimes in functions, the annotations go in the function signature, not in the function body. The lifetime annotations become part of the contract of the function, much like the types in the signature. Having function signatures contain the lifetime contract means the analysis the Rust compiler does can be simpler. If there’s a problem with the way a function is annotated or the way it is called, the compiler errors can point to the part of our code and the constraints more precisely. 
>  当在函数中注解生命周期时，标注应该放在函数签名中，而不是函数体中
>  生命周期注解构成了函数的契约的一部分，就像函数签名中的类型
>  让函数签名包含生命周期契约意味着 Rust 编译器做的分析可以更简单，如果函数的标注或其调用的方式存在问题，编译器错误就可以更精确地指向代码中的具体部分和约束条件

If, instead, the Rust compiler made more inferences about what we intended the relationships of the lifetimes to be, the compiler might only be able to point to a use of our code many steps away from the cause of the problem.
>  相反，如果让 Rust 编译器自行对生命周期的关系进行推断，编译器可能指出距离问题根源更远的代码位置，使得调试更困难

When we pass concrete references to `longest`, the concrete lifetime that is substituted for `'a` is the part of the scope of `x` that overlaps with the scope of `y`. In other words, the generic lifetime `'a` will get the concrete lifetime that is equal to the smaller of the lifetimes of `x` and `y`. Because we’ve annotated the returned reference with the same lifetime parameter `'a`, the returned reference will also be valid for the length of the smaller of the lifetimes of `x` and `y`.
>  当我们向 `longest` 传入具体的引用时，将 `'a` 替换掉的具体生命周期将会是 `x` 的生命周期和 `y` 的生命周期重叠的那一部分
>  换句话说，泛型生命周期 `'a` 将获得的具体生命周期等于 `x,y` 的生命周期中较小的那个
>  因为我们已经将返回引用的生命周期标记为和生命周期参数 `'a` 相同，返回的引用的有效期就等于 `x,y` 的生命周期中较小的那个

Let’s look at how the lifetime annotations restrict the `longest` function by passing in references that have different concrete lifetimes. Listing 10-22 is a straightforward example.

Filename: src/main.rs

```rust
fn main() {
    let string1 = String::from("long string is long");

    {
        let string2 = String::from("xyz");
        let result = longest(string1.as_str(), string2.as_str());
        println!("The longest string is {result}");
    }
}
```

[Listing 10-22](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#listing-10-22): Using the `longest` function with references to `String` values that have different concrete lifetimes

In this example, `string1` is valid until the end of the outer scope, `string2` is valid until the end of the inner scope, and `result` references something that is valid until the end of the inner scope. Run this code and you’ll see that the borrow checker approves; it will compile and print `The longest string is long string is long`.

Next, let’s try an example that shows that the lifetime of the reference in `result` must be the smaller lifetime of the two arguments. We’ll move the declaration of the `result` variable outside the inner scope but leave the assignment of the value to the `result` variable inside the scope with `string2`. Then we’ll move the `println!` that uses `result` to outside the inner scope, after the inner scope has ended. The code in Listing 10-23 will not compile.

Filename: src/main.rs

```rust
fn main() {
    let string1 = String::from("long string is long");
    let result;
    {
        let string2 = String::from("xyz");
        result = longest(string1.as_str(), string2.as_str());
    }
    println!("The longest string is {result}");
}
```

[Listing 10-23](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#listing-10-23): Attempting to use `result` after `string2` has gone out of scope

When we try to compile this code, we get this error:

```
$ cargo run
   Compiling chapter10 v0.1.0 (file:///projects/chapter10)
error[E0597]: `string2` does not live long enough
 --> src/main.rs:6:44
  |
5 |         let string2 = String::from("xyz");
  |             ------- binding `string2` declared here
6 |         result = longest(string1.as_str(), string2.as_str());
  |                                            ^^^^^^^ borrowed value does not live long enough
7 |     }
  |     - `string2` dropped here while still borrowed
8 |     println!("The longest string is {result}");
  |                                     -------- borrow later used here

For more information about this error, try `rustc --explain E0597`.
error: could not compile `chapter10` (bin "chapter10") due to 1 previous error
```

The error shows that for `result` to be valid for the `println!` statement, `string2` would need to be valid until the end of the outer scope. Rust knows this because we annotated the lifetimes of the function parameters and return values using the same lifetime parameter `'a`.

As humans, we can look at this code and see that `string1` is longer than `string2`, and therefore, `result` will contain a reference to `string1`. Because `string1` has not gone out of scope yet, a reference to `string1` will still be valid for the `println!` statement. However, the compiler can’t see that the reference is valid in this case. We’ve told Rust that the lifetime of the reference returned by the `longest` function is the same as the smaller of the lifetimes of the references passed in. Therefore, the borrow checker disallows the code in Listing 10-23 as possibly having an invalid reference.

Try designing more experiments that vary the values and lifetimes of the references passed in to the `longest` function and how the returned reference is used. Make hypotheses about whether or not your experiments will pass the borrow checker before you compile; then check to see if you’re right!

### Thinking in Terms of Lifetimes
The way in which you need to specify lifetime parameters depends on what your function is doing. For example, if we changed the implementation of the `longest` function to always return the first parameter rather than the longest string slice, we wouldn’t need to specify a lifetime on the `y` parameter. The following code will compile:
>  我们需要如何指定生命周期参数取决于我们函数具体的功能，例如如果我们修改了 `longest` 的实现，让他永远返回第一个参数而不是更长的 string slice，我们就不需要为 `y` 指定生命周期

Filename: src/main.rs

```rust
fn longest<'a>(x: &'a str, y: &str) -> &'a str {
    x
}
```

We’ve specified a lifetime parameter `'a` for the parameter `x` and the return type, but not for the parameter `y`, because the lifetime of `y` does not have any relationship with the lifetime of `x` or the return value.

When returning a reference from a function, the lifetime parameter for the return type needs to match the lifetime parameter for one of the parameters. If the reference returned does _not_ refer to one of the parameters, it must refer to a value created within this function. However, this would be a dangling reference because the value will go out of scope at the end of the function. Consider this attempted implementation of the `longest` function that won’t compile:
>  当函数返回一个引用时，返回类型的生命周期参数必须与函数的某个参数的生命周期参数匹配
>  如果返回的引用并不引用函数的任何一个参数，那么它只能指向函数内创建的值，而这就会导致悬挂引用

Filename: src/main.rs

```rust
fn longest<'a>(x: &str, y: &str) -> &'a str {
    let result = String::from("really long string");
    result.as_str()
}
```

Here, even though we’ve specified a lifetime parameter `'a` for the return type, this implementation will fail to compile because the return value lifetime is not related to the lifetime of the parameters at all. Here is the error message we get:
>  上例中，即使我们指定了生命周期参数，该实现也无法编译，因为返回值的生命周期和参数的生命周期没有关系

```
$ cargo run
   Compiling chapter10 v0.1.0 (file:///projects/chapter10)
error[E0515]: cannot return value referencing local variable `result`
  --> src/main.rs:11:5
   |
11 |     result.as_str()
   |     ------^^^^^^^^^
   |     |
   |     returns a value referencing data owned by the current function
   |     `result` is borrowed here

For more information about this error, try `rustc --explain E0515`.
error: could not compile `chapter10` (bin "chapter10") due to 1 previous error
```

The problem is that `result` goes out of scope and gets cleaned up at the end of the `longest` function. We’re also trying to return a reference to `result` from the function. There is no way we can specify lifetime parameters that would change the dangling reference, and Rust won’t let us create a dangling reference. In this case, the best fix would be to return an owned data type rather than a reference so the calling function is then responsible for cleaning up the value.
>  如果我们要返回函数内创建的值，最好的方式是返回一个拥有所有权的数据类型，而不是引用，让 Rust 为我们转移所有权

Ultimately, lifetime syntax is about connecting the lifetimes of various parameters and return values of functions. Once they’re connected, Rust has enough information to allow memory-safe operations and disallow operations that would create dangling pointers or otherwise violate memory safety.
>  归根结底，生命周期语法是关于将函数的各种参数和返回值的生命周期关联起来，一旦建立了这种关联，Rust 就有了足够的信息，以允许内存安全的操作，并禁止会创建悬挂指针或者违反内存安全的操作

### Lifetime Annotations in Struct Definitions
So far, the structs we’ve defined all hold owned types. We can define structs to hold references, but in that case we would need to add a lifetime annotation on every reference in the struct’s definition. Listing 10-24 has a struct named `ImportantExcerpt` that holds a string slice.
>  目前为止，我们定义的所有结构体都拥有具有所有权的类型
>  我们可以定义持有引用的结构体，要求是我们需要为结构体中的每个引用加上生命周期标记

Filename: src/main.rs

```rust
struct ImportantExcerpt<'a> {
    part: &'a str,
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence = novel.split('.').next().unwrap();
    let i = ImportantExcerpt {
        part: first_sentence,
    };
}
```

[Listing 10-24](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#listing-10-24): A struct that holds a reference, requiring a lifetime annotation

This struct has the single field `part` that holds a string slice, which is a reference. As with generic data types, we declare the name of the generic lifetime parameter inside angle brackets after the name of the struct so we can use the lifetime parameter in the body of the struct definition. This annotation means an instance of `ImportantExcerpt` can’t outlive the reference it holds in its `part` field.
>  上例中的结构体的 field `part` 是 string slice，即引用
>  和泛型数据类型一样，我们在 `<>` 中声明泛型生命周期参数，这样我们就可以在结构体定义中使用使用该生命周期参数
>  这个生命周期的含义是结构体 `ImportantExcerpt` 的实例的存活时间不能比其 `part` 字段引用的值的存活时间更久 `

The `main` function here creates an instance of the `ImportantExcerpt` struct that holds a reference to the first sentence of the `String` owned by the variable `novel`. The data in `novel` exists before the `ImportantExcerpt` instance is created. In addition, `novel` doesn’t go out of scope until after the `ImportantExcerpt` goes out of scope, so the reference in the `ImportantExcerpt` instance is valid.

### Lifetime Elision
You’ve learned that every reference has a lifetime and that you need to specify lifetime parameters for functions or structs that use references. However, we had a function in Listing 4-9, shown again in Listing 10-25, that compiled without lifetime annotations.
>  我们已经知道，每个引用都有生命周期，而在使用引用的函数和结构体中，我们需要指定生命周期参数
>  但是下面的函数不需要指定生命周期参数也可以编译

Filename: src/lib.rs

```rust
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }

    &s[..]
}
```

[Listing 10-25](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#listing-10-25): A function we defined in Listing 4-9 that compiled without lifetime annotations, even though the parameter and return type are references

The reason this function compiles without lifetime annotations is historical: in early versions (pre-1.0) of Rust, this code wouldn’t have compiled because every reference needed an explicit lifetime. At that time, the function signature would have been written like this:
>  该函数在没有生命周期标注的情况下可以编译是出于历史原因，在 Rust 早期版本，这段代码无法编译，因为每个引用都需要显式地指定生命周期，故函数签名应该被写为:

```rust
fn first_word<'a>(s: &'a str) -> &'a str {
```

After writing a lot of Rust code, the Rust team found that Rust programmers were entering the same lifetime annotations over and over in particular situations. These situations were predictable and followed a few deterministic patterns. The developers programmed these patterns into the compiler’s code so the borrow checker could infer the lifetimes in these situations and wouldn’t need explicit annotations.
>  后来 Rust 团队发现 Rust 程序员在特定情况下反复输入相同的生命周期标注
>  这些情况是可以预测的，并遵循一些特定的模式，开发者将这些模式加入了编译器中，故借用检查器可以在这些情况下自己推导生命周期，不需要显式标注

This piece of Rust history is relevant because it’s possible that more deterministic patterns will emerge and be added to the compiler. In the future, even fewer lifetime annotations might be required.
>  在将来，可能会出现更多确定性的模式，并被加入到编译器中，因此在将来或许要编写的生命周期标注会更少

The patterns programmed into Rust’s analysis of references are called the _lifetime elision rules_. These aren’t rules for programmers to follow; they’re a set of particular cases that the compiler will consider, and if your code fits these cases, you don’t need to write the lifetimes explicitly.
>  这些被嵌入到 Rust 引用分析中的模式被称为生命周期省略规则，这些规则不是要求程序员遵循的规则，而是编译器在分析代码时会考虑的一组特定情况，如果我们的代码符合这些情况，我们就不需要显式写生命周期

The elision rules don’t provide full inference. If there is still ambiguity about what lifetimes the references have after Rust applies the rules, the compiler won’t guess what the lifetime of the remaining references should be. Instead of guessing, the compiler will give you an error that you can resolve by adding the lifetime annotations.
>  省略规则不提供完整的推断，如果 Rust 编译器在应用了这些规则之后，引用的生命周期仍然存在歧义，编译器不会猜测，而会报错，让我们通过生命周期标注指定

Lifetimes on function or method parameters are called _input lifetimes_, and lifetimes on return values are called _output lifetimes_.
>  函数或方法参数上的生命周期称为输入生命周期，返回值上的生命周期称为输出生命周期

The compiler uses three rules to figure out the lifetimes of the references when there aren’t explicit annotations. The first rule applies to input lifetimes, and the second and third rules apply to output lifetimes. If the compiler gets to the end of the three rules and there are still references for which it can’t figure out lifetimes, the compiler will stop with an error. These rules apply to `fn` definitions as well as `impl` blocks.
>  当没有显式标注时，编译器会使用三个规则来推断引用的生命周期
>  第一条规则适用于输入生命周期，第二条规则和第三条规则适用于输出生命周期
>  如果编译器用完了三条规则后仍然存在不清楚的生命周期，它就会报错
>  这些规则既适用于 `fn` 定义，也适用于 `impl` 块

The first rule is that the compiler assigns a lifetime parameter to each parameter that’s a reference. In other words, a function with one parameter gets one lifetime parameter: `fn foo<'a>(x: &'a i32)`; a function with two parameters gets two separate lifetime parameters: `fn foo<'a, 'b>(x: &'a i32, y: &'b i32)`; and so on.
>  第一条规则是编译器为每个是引用的参数赋予一个生命周期参数
>  换句话说，只有一个类型的函数获得一个生命周期参数: `fn foo<'a>(x: &'a i32)`，有两个参数的函数得到两个独立的生命周期参数: `fn foo<'a, 'b>(x: &'a i32, y: &'b i32)`

The second rule is that, if there is exactly one input lifetime parameter, that lifetime is assigned to all output lifetime parameters: `fn foo<'a>(x: &'a i32) -> &'a i32`.
>  第二个规则是如果正好有一个输入生命周期参数，它会被赋予给所有输出生命周期参数: `fn foo<'a>(x: &'a i32) -> &'a i32`

The third rule is that, if there are multiple input lifetime parameters, but one of them is `&self` or `&mut self` because this is a method, the lifetime of `self` is assigned to all output lifetime parameters. This third rule makes methods much nicer to read and write because fewer symbols are necessary.
>  第三条规则是如果有多个输入生命周期参数，但其中一个是 `&self or &mut sef` (因为这个一个方法)，则 `self` 的生命周期会被赋予所有输出生命周期参数
>  这个规则使得方法更容易编写和阅读

Let’s pretend we’re the compiler. We’ll apply these rules to figure out the lifetimes of the references in the signature of the `first_word` function in Listing 10-25. The signature starts without any lifetimes associated with the references:

```rust
fn first_word(s: &str) -> &str {
```

Then the compiler applies the first rule, which specifies that each parameter gets its own lifetime. We’ll call it `'a` as usual, so now the signature is this:

```rust
fn first_word<'a>(s: &'a str) -> &str {
```

The second rule applies because there is exactly one input lifetime. The second rule specifies that the lifetime of the one input parameter gets assigned to the output lifetime, so the signature is now this:

```rust
fn first_word<'a>(s: &'a str) -> &'a str {
```

Now all the references in this function signature have lifetimes, and the compiler can continue its analysis without needing the programmer to annotate the lifetimes in this function signature.

Let’s look at another example, this time using the `longest` function that had no lifetime parameters when we started working with it in Listing 10-20:

```rust
fn longest(x: &str, y: &str) -> &str {
```

Let’s apply the first rule: each parameter gets its own lifetime. This time we have two parameters instead of one, so we have two lifetimes:

```rust
fn longest<'a, 'b>(x: &'a str, y: &'b str) -> &str {
```

You can see that the second rule doesn’t apply because there is more than one input lifetime. The third rule doesn’t apply either, because `longest` is a function rather than a method, so none of the parameters are `self`. After working through all three rules, we still haven’t figured out what the return type’s lifetime is. This is why we got an error trying to compile the code in Listing 10-20: the compiler worked through the lifetime elision rules but still couldn’t figure out all the lifetimes of the references in the signature.

Because the third rule really only applies in method signatures, we’ll look at lifetimes in that context next to see why the third rule means we don’t have to annotate lifetimes in method signatures very often.

### Lifetime Annotations in Method Definitions
When we implement methods on a struct with lifetimes, we use the same syntax as that of generic type parameters, as shown in Listing 10-11. Where we declare and use the lifetime parameters depends on whether they’re related to the struct fields or the method parameters and return values.
>  当我们在具有生命周期的结构体上实现方法时，我们使用和泛型类型参数相同的语法
>  生命周期参数声明和使用的位置取决于它们是和结构体字段相关还是和方法参数或返回类型相关

Lifetime names for struct fields always need to be declared after the `impl` keyword and then used after the struct’s name because those lifetimes are part of the struct’s type.
>  结构体字段的声明周期名称必须在 `impl` 关键字之后声明，然后在结构体名称后使用，因为这些生命周期是**结构体类型的一部分**

In method signatures inside the `impl` block, references might be tied to the lifetime of references in the struct’s fields, or they might be independent. In addition, the lifetime elision rules often make it so that lifetime annotations aren’t necessary in method signatures. Let’s look at some examples using the struct named `ImportantExcerpt` that we defined in Listing 10-24.
>  在 `impl` 块内的方法签名中的引用可能和结构体字段中的引用的生命周期关联，也可能独立
>  此外，生命周期省略规则经常使得方法签名中无需添加生命周期标注

First we’ll use a method named `level` whose only parameter is a reference to `self` and whose return value is an `i32`, which is not a reference to anything:

```rust
impl<'a> ImportantExcerpt<'a> {
    fn level(&self) -> i32 {
        3
    }
}
```

The lifetime parameter declaration after `impl` and its use after the type name are required, but we’re not required to annotate the lifetime of the reference to `self` because of the first elision rule.
>  因为第一条规则，虽然我们是接收 `&self`，我们不需要标记生命周期

Here is an example where the third lifetime elision rule applies:

```rust
impl<'a> ImportantExcerpt<'a> {
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("Attention please: {announcement}");
        self.part
    }
}
```

There are two input lifetimes, so Rust applies the first lifetime elision rule and gives both `&self` and `announcement` their own lifetimes. Then, because one of the parameters is `&self`, the return type gets the lifetime of `&self`, and all lifetimes have been accounted for.

>  因为结构体存储了引用，故结构体实例的生命周期是由它内部字段的生命周期决定的，因此如果返回的是结构体字段引用，可以直接定义它的生命周期是结构体生命周期
>  注意如果返回 `announcement`，则不能编译，需要额外标注，必须声明 “我返回的引用至少活到 `announcement` 死为止”
>  生命周期就是要告诉调用者我返回的东西的生命周期是什么，Rust 编译器函数明确这个信息，因为函数里面具体做什么它不知道，Rust 编译器通过这个信息对代码进行分析，判断是否存在悬空引用

### The Static Lifetime
One special lifetime we need to discuss is `'static`, which denotes that the affected reference _can_ live for the entire duration of the program. All string literals have the `'static` lifetime, which we can annotate as follows:
>  `'static` 表示引用的存活时间和整个程序一样
>  所有的字符串字面值的生命周期都是 `'static`

```rust
let s: &'static str = "I have a static lifetime.";
```

The text of this string is stored directly in the program’s binary, which is always available. Therefore, the lifetime of all string literals is `'static`.
>  字符串字面值的文本存储在程序的二进制文件中，故总是可用，因此所有字符串字面值的生命周期都是 `'static`

You might see suggestions in error messages to use the `'static` lifetime. But before specifying `'static` as the lifetime for a reference, think about whether the reference you have actually lives the entire lifetime of your program or not, and whether you want it to. Most of the time, an error message suggesting the `'static` lifetime results from attempting to create a dangling reference or a mismatch of the available lifetimes. In such cases, the solution is to fix those problems, not to specify the `'static` lifetime.

## Generic Type Parameters, Trait Bounds, and Lifetimes Together
Let’s briefly look at the syntax of specifying generic type parameters, trait bounds, and lifetimes all in one function!

```rust
use std::fmt::Display;

fn longest_with_an_announcement<'a, T>(
    x: &'a str,
    y: &'a str,
    ann: T,
) -> &'a str
where
    T: Display,
{
    println!("Announcement! {ann}");
    if x.len() > y.len() { x } else { y }
}
```

This is the `longest` function from Listing 10-21 that returns the longer of two string slices. But now it has an extra parameter named `ann` of the generic type `T`, which can be filled in by any type that implements the `Display` trait as specified by the `where` clause. This extra parameter will be printed using `{}`, which is why the `Display` trait bound is necessary. Because lifetimes are a type of generic, the declarations of the lifetime parameter `'a` and the generic type parameter `T` go in the same list inside the angle brackets after the function name.
>  因为生命周期也是一种泛型，因此生命周期参数和泛型类型参数都位于 `<>` 中

## Summary
We covered a lot in this chapter! Now that you know about generic type parameters, traits and trait bounds, and generic lifetime parameters, you’re ready to write code without repetition that works in many different situations. Generic type parameters let you apply the code to different types. Traits and trait bounds ensure that even though the types are generic, they’ll have the behavior the code needs. You learned how to use lifetime annotations to ensure that this flexible code won’t have any dangling references. And all of this analysis happens at compile time, which doesn’t affect runtime performance!

Believe it or not, there is much more to learn on the topics we discussed in this chapter: Chapter 18 discusses trait objects, which are another way to use traits. There are also more complex scenarios involving lifetime annotations that you will only need in very advanced scenarios; for those, you should read the [Rust Reference](https://doc.rust-lang.org/reference/index.html). But next, you’ll learn how to write tests in Rust so you can make sure your code is working the way it should.

# 11 Writing Automated Tests
In his 1972 essay “The Humble Programmer,” Edsger W. Dijkstra said that “program testing can be a very effective way to show the presence of bugs, but it is hopelessly inadequate for showing their absence.” That doesn’t mean we shouldn’t try to test as much as we can!

Correctness in our programs is the extent to which our code does what we intend it to do. Rust is designed with a high degree of concern about the correctness of programs, but correctness is complex and not easy to prove. Rust’s type system shoulders a huge part of this burden, but the type system cannot catch everything. As such, Rust includes support for writing automated software tests.

Say we write a function `add_two` that adds 2 to whatever number is passed to it. This function’s signature accepts an integer as a parameter and returns an integer as a result. When we implement and compile that function, Rust does all the type checking and borrow checking that you’ve learned so far to ensure that, for instance, we aren’t passing a `String` value or an invalid reference to this function. But Rust _can’t_ check that this function will do precisely what we intend, which is return the parameter plus 2 rather than, say, the parameter plus 10 or the parameter minus 50! That’s where tests come in.

We can write tests that assert, for example, that when we pass `3` to the `add_two` function, the returned value is `5`. We can run these tests whenever we make changes to our code to make sure any existing correct behavior has not changed.

Testing is a complex skill: although we can’t cover in one chapter every detail about how to write good tests, in this chapter we will discuss the mechanics of Rust’s testing facilities. We’ll talk about the annotations and macros available to you when writing your tests, the default behavior and options provided for running your tests, and how to organize tests into unit tests and integration tests.

## 11.1 How to Write Tests
Tests are Rust functions that verify that the non-test code is functioning in the expected manner. The bodies of test functions typically perform these three actions:

- Set up any needed data or state.
- Run the code you want to test.
- Assert that the results are what you expect.

>  测试函数的函数体通常执行以下动作:
>  - 设定好需要的数据和状态
>  - 运行需要测试的代码
>  - 断言结果是所期望的

Let’s look at the features Rust provides specifically for writing tests that take these actions, which include the `test` attribute, a few macros, and the `should_panic` attribute.

### The Anatomy of a Test Function
At its simplest, a test in Rust is a function that’s annotated with the `test` attribute. Attributes are metadata about pieces of Rust code; one example is the `derive` attribute we used with structs in Chapter 5. To change a function into a test function, add `#[test]` on the line before `fn`. When you run your tests with the `cargo test` command, Rust builds a test runner binary that runs the annotated functions and reports on whether each test function passes or fails.
>  Rust 中的 test 是一个用 `test` 属性注解的函数
>  属性是关于 Rust 代码片段的**元数据**，例如我们之前使用过 `derive` 属性
>  要将函数变为测试函数，我们在 `fn` 上添加 `#[test]`
>  使用 `cargo test` 时，Rust 会构造运行测试的二进制函数，运行标记的函数，并报告是否每个测试函数通过

Whenever we make a new library project with Cargo, a test module with a test function in it is automatically generated for us. This module gives you a template for writing your tests so you don’t have to look up the exact structure and syntax every time you start a new project. You can add as many additional test functions and as many test modules as you want!
>  使用 Cargo 构造新的库项目时，都会自动生成带有 test 函数的 test module
>  该 module 会提供编写测试的模板代码
>  我们可以自行添加 test modules 和 test functions

We’ll explore some aspects of how tests work by experimenting with the template test before we actually test any code. Then we’ll write some real-world tests that call some code that we’ve written and assert that its behavior is correct.

Let’s create a new library project called `adder` that will add two numbers:

```
$ cargo new adder --lib
     Created library `adder` project
$ cd adder
```

The contents of the _src/lib.rs_ file in your `adder` library should look like Listing 11-1.

Filename: src/lib.rs

```rust
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
```

[Listing 11-1](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#listing-11-1): The code generated automatically by `cargo new`

The file starts with an example `add` function, so that we have something to test.

For now, let’s focus solely on the `it_works` function. Note the `#[test]` annotation: this attribute indicates this is a test function, so the test runner knows to treat this function as a test. We might also have non-test functions in the `tests` module to help set up common scenarios or perform common operations, so we always need to indicate which functions are tests.
>  `#[test]` 属性表示函数为测试函数
>  在 `tests` module 中也可以有不是测试函数的函数

The example function body uses the `assert_eq!` macro to assert that `result`, which contains the result of calling `add` with 2 and 2, equals 4. This assertion serves as an example of the format for a typical test. Let’s run it to see that this test passes.

The `cargo test` command runs all tests in our project, as shown in Listing 11-2.

```
$ cargo test
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.57s
     Running unittests src/lib.rs (target/debug/deps/adder-01ad14159ff659ab)

running 1 test
test tests::it_works ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

[Listing 11-2](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#listing-11-2): The output from running the automatically generated test

Cargo compiled and ran the test. We see the line `running 1 test`. The next line shows the name of the generated test function, called `tests::it_works`, and that the result of running that test is `ok`. The overall summary `test result: ok.` means that all the tests passed, and the portion that reads `1 passed; 0 failed` totals the number of tests that passed or failed.

It’s possible to mark a test as ignored so it doesn’t run in a particular instance; we’ll cover that in the [“Ignoring Some Tests Unless Specifically Requested”](https://doc.rust-lang.org/book/ch11-02-running-tests.html#ignoring-some-tests-unless-specifically-requested) section later in this chapter. Because we haven’t done that here, the summary shows `0 ignored`. We can also pass an argument to the `cargo test` command to run only tests whose name matches a string; this is called _filtering_ and we’ll cover it in the [“Running a Subset of Tests by Name”](https://doc.rust-lang.org/book/ch11-02-running-tests.html#running-a-subset-of-tests-by-name) section. Here we haven’t filtered the tests being run, so the end of the summary shows `0 filtered out`.

The `0 measured` statistic is for benchmark tests that measure performance. Benchmark tests are, as of this writing, only available in nightly Rust. See [the documentation about benchmark tests](https://doc.rust-lang.org/unstable-book/library-features/test.html) to learn more.

The next part of the test output starting at `Doc-tests adder` is for the results of any documentation tests. We don’t have any documentation tests yet, but Rust can compile any code examples that appear in our API documentation. This feature helps keep your docs and your code in sync! We’ll discuss how to write documentation tests in the [“Documentation Comments as Tests”](https://doc.rust-lang.org/book/ch14-02-publishing-to-crates-io.html#documentation-comments-as-tests) section of Chapter 14. For now, we’ll ignore the `Doc-tests` output.
>  Rust 可以编译出现在我们 API 文档中的代码示例，doc-test 保持我们的文档和代码是一致的

Let’s start to customize the test to our own needs. First, change the name of the `it_works` function to a different name, such as `exploration`, like so:

Filename: src/lib.rs

```rust
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exploration() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
```

Then run `cargo test` again. The output now shows `exploration` instead of `it_works`:

```
$ cargo test
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.59s
     Running unittests src/lib.rs (target/debug/deps/adder-92948b65e88960b4)

running 1 test
test tests::exploration ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Now we’ll add another test, but this time we’ll make a test that fails! Tests fail when something in the test function panics. Each test is run in a new thread, and when the main thread sees that a test thread has died, the test is marked as failed. In Chapter 9, we talked about how the simplest way to panic is to call the `panic!` macro. Enter the new test as a function named `another`, so your _src/lib.rs_ file looks like Listing 11-3.
>  如果 test function panic，测试失败
>  每个 test 都在新线程中运行，主线程发现测试线程失败，就将测试标记为失败

Filename: src/lib.rs

```rust
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exploration() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn another() {
        panic!("Make this test fail");
    }
}
```

[Listing 11-3](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#listing-11-3): Adding a second test that will fail because we call the `panic!` macro

Run the tests again using `cargo test`. The output should look like Listing 11-4, which shows that our `exploration` test passed and `another` failed.

```
$ cargo test
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.72s
     Running unittests src/lib.rs (target/debug/deps/adder-92948b65e88960b4)

running 2 tests
test tests::another ... FAILED
test tests::exploration ... ok

failures:

---- tests::another stdout ----

thread 'tests::another' panicked at src/lib.rs:17:9:
Make this test fail
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::another

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
```

[Listing 11-4](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#listing-11-4): Test results when one test passes and one test fails

Instead of `ok`, the line `test tests::another` shows `FAILED`. Two new sections appear between the individual results and the summary: the first displays the detailed reason for each test failure. In this case, we get the details that `tests::another` failed because it panicked with the message `Make this test fail` on line 17 in the _src/lib.rs_ file. The next section lists just the names of all the failing tests, which is useful when there are lots of tests and lots of detailed failing test output. We can use the name of a failing test to run just that test to more easily debug it; we’ll talk more about ways to run tests in the [“Controlling How Tests Are Run”](https://doc.rust-lang.org/book/ch11-02-running-tests.html#controlling-how-tests-are-run) section.

The summary line displays at the end: overall, our test result is `FAILED`. We had one test pass and one test fail.

Now that you’ve seen what the test results look like in different scenarios, let’s look at some macros other than `panic!` that are useful in tests.

### Checking Results with the `assert!` Macro
The `assert!` macro, provided by the standard library, is useful when you want to ensure that some condition in a test evaluates to `true`. We give the `assert!` macro an argument that evaluates to a Boolean. If the value is `true`, nothing happens and the test passes. If the value is `false`, the `assert!` macro calls `panic!` to cause the test to fail. Using the `assert!` macro helps us check that our code is functioning in the way we intend.
>  `assert!` macro 接受评估为布尔值的参数，如果值为 `false`，会调用 `panic!`

In Chapter 5, Listing 5-15, we used a `Rectangle` struct and a `can_hold` method, which are repeated here in Listing 11-5. Let’s put this code in the _src/lib.rs_ file, then write some tests for it using the `assert!` macro.

Filename: src/lib.rs

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}
```

[Listing 11-5](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#listing-11-5): The `Rectangle` struct and its `can_hold` method from Chapter 5

The `can_hold` method returns a Boolean, which means it’s a perfect use case for the `assert!` macro. In Listing 11-6, we write a test that exercises the `can_hold` method by creating a `Rectangle` instance that has a width of 8 and a height of 7 and asserting that it can hold another `Rectangle` instance that has a width of 5 and a height of 1.

Filename: src/lib.rs

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn larger_can_hold_smaller() {
        let larger = Rectangle {
            width: 8,
            height: 7,
        };
        let smaller = Rectangle {
            width: 5,
            height: 1,
        };

        assert!(larger.can_hold(&smaller));
    }
}
```

[Listing 11-6](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#listing-11-6): A test for `can_hold` that checks whether a larger rectangle can indeed hold a smaller rectangle

Note the `use super::*;` line inside the `tests` module. The `tests` module is a regular module that follows the usual visibility rules we covered in Chapter 7 in the [“Paths for Referring to an Item in the Module Tree”](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html) section. Because the `tests` module is an inner module, we need to bring the code under test in the outer module into the scope of the inner module. We use a glob here, so anything we define in the outer module is available to this `tests` module.
>  `tests` module 中的 `use super::*` 表示将外部 module 的代码带入内部 module

We’ve named our test `larger_can_hold_smaller`, and we’ve created the two `Rectangle` instances that we need. Then we called the `assert!` macro and passed it the result of calling `larger.can_hold(&smaller)`. This expression is supposed to return `true`, so our test should pass. Let’s find out!

```
$ cargo test
   Compiling rectangle v0.1.0 (file:///projects/rectangle)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.66s
     Running unittests src/lib.rs (target/debug/deps/rectangle-6584c4561e48942e)

running 1 test
test tests::larger_can_hold_smaller ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests rectangle

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

It does pass! Let’s add another test, this time asserting that a smaller rectangle cannot hold a larger rectangle:

Filename: src/lib.rs

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn larger_can_hold_smaller() {
        // --snip--
    }

    #[test]
    fn smaller_cannot_hold_larger() {
        let larger = Rectangle {
            width: 8,
            height: 7,
        };
        let smaller = Rectangle {
            width: 5,
            height: 1,
        };

        assert!(!smaller.can_hold(&larger));
    }
}
```

Because the correct result of the `can_hold` function in this case is `false`, we need to negate that result before we pass it to the `assert!` macro. As a result, our test will pass if `can_hold` returns `false`:

```
$ cargo test
   Compiling rectangle v0.1.0 (file:///projects/rectangle)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.66s
     Running unittests src/lib.rs (target/debug/deps/rectangle-6584c4561e48942e)

running 2 tests
test tests::larger_can_hold_smaller ... ok
test tests::smaller_cannot_hold_larger ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests rectangle

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Two tests that pass! Now let’s see what happens to our test results when we introduce a bug in our code. We’ll change the implementation of the `can_hold` method by replacing the greater-than sign with a less-than sign when it compares the widths:

```rust
// --snip--
impl Rectangle {
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width < other.width && self.height > other.height
    }
}
```

Running the tests now produces the following:

```
$ cargo test
   Compiling rectangle v0.1.0 (file:///projects/rectangle)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.66s
     Running unittests src/lib.rs (target/debug/deps/rectangle-6584c4561e48942e)

running 2 tests
test tests::larger_can_hold_smaller ... FAILED
test tests::smaller_cannot_hold_larger ... ok

failures:

---- tests::larger_can_hold_smaller stdout ----

thread 'tests::larger_can_hold_smaller' panicked at src/lib.rs:28:9:
assertion failed: larger.can_hold(&smaller)
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::larger_can_hold_smaller

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
```

Our tests caught the bug! Because `larger.width` is `8` and `smaller.width` is `5`, the comparison of the widths in `can_hold` now returns `false`: 8 is not less than 5.

### Testing Equality with the `assert_eq!` and `assert_ne!` Macros
A common way to verify functionality is to test for equality between the result of the code under test and the value you expect the code to return. You could do this by using the `assert!` macro and passing it an expression using the `==` operator. However, this is such a common test that the standard library provides a pair of macros— `assert_eq!` and `assert_ne!` —to perform this test more conveniently. These macros compare two arguments for equality or inequality, respectively. 

They’ll also print the two values if the assertion fails, which makes it easier to see _why_ the test failed; conversely, the `assert!` macro only indicates that it got a `false` value for the `==` expression, without printing the values that led to the `false` value.
>  `assert_eq!, assert_ne!` 会打印出两个 values 如果 assertion 失败

In Listing 11-7, we write a function named `add_two` that adds `2` to its parameter, then we test this function using the `assert_eq!` macro.

Filename: src/lib.rs

```rust
pub fn add_two(a: u64) -> u64 {
    a + 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_adds_two() {
        let result = add_two(2);
        assert_eq!(result, 4);
    }
}
```

[Listing 11-7](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#listing-11-7): Testing the function `add_two` using the `assert_eq!` macro

Let’s check that it passes!

```
$ cargo test
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.58s
     Running unittests src/lib.rs (target/debug/deps/adder-92948b65e88960b4)

running 1 test
test tests::it_adds_two ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

We create a variable named `result` that holds the result of calling `add_two(2)`. Then we pass `result` and `4` as the arguments to the `assert_eq!` macro. The output line for this test is `test tests::it_adds_two ... ok`, and the `ok` text indicates that our test passed!

Let’s introduce a bug into our code to see what `assert_eq!` looks like when it fails. Change the implementation of the `add_two` function to instead add `3`:

```rust
pub fn add_two(a: u64) -> u64 {
    a + 3
}
```

Run the tests again:

```
$ cargo test
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.61s
     Running unittests src/lib.rs (target/debug/deps/adder-92948b65e88960b4)

running 1 test
test tests::it_adds_two ... FAILED

failures:

---- tests::it_adds_two stdout ----

thread 'tests::it_adds_two' panicked at src/lib.rs:12:9:
assertion `left == right` failed
  left: 5
 right: 4
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::it_adds_two

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
```

Our test caught the bug! The `tests::it_adds_two` test failed, and the message tells us that the assertion that failed was `left == right` and what the `left` and `right` values are. This message helps us start debugging: the `left` argument, where we had the result of calling `add_two(2)`, was `5` but the `right` argument was `4`. You can imagine that this would be especially helpful when we have a lot of tests going on.

Note that in some languages and test frameworks, the parameters to equality assertion functions are called `expected` and `actual`, and the order in which we specify the arguments matters. However, in Rust, they’re called `left` and `right`, and the order in which we specify the value we expect and the value the code produces doesn’t matter. We could write the assertion in this test as `assert_eq!(4, result)`, which would result in the same failure message that displays ``assertion `left == right` failed``.

The `assert_ne!` macro will pass if the two values we give it are not equal and fail if they’re equal. This macro is most useful for cases when we’re not sure what a value _will_ be, but we know what the value definitely _shouldn’t_ be. For example, if we’re testing a function that is guaranteed to change its input in some way, but the way in which the input is changed depends on the day of the week that we run our tests, the best thing to assert might be that the output of the function is not equal to the input.

Under the surface, the `assert_eq!` and `assert_ne!` macros use the operators `==` and `!=`, respectively. When the assertions fail, these macros print their arguments using debug formatting, which means the values being compared must implement the `PartialEq` and `Debug` traits. All primitive types and most of the standard library types implement these traits. For structs and enums that you define yourself, you’ll need to implement `PartialEq` to assert equality of those types. You’ll also need to implement `Debug` to print the values when the assertion fails. Because both traits are derivable traits, as mentioned in Listing 5-12 in Chapter 5, this is usually as straightforward as adding the `#[derive(PartialEq, Debug)]` annotation to your struct or enum definition. See Appendix C, [“Derivable Traits,”](https://doc.rust-lang.org/book/appendix-03-derivable-traits.html) for more details about these and other derivable traits.
>  `assert_eq!, assert_ne!` macro 使用的是 `==, !=` 运算符
>  断言失败时，它们会以 debug 格式打印参数
>  因此比较的 values 必须实现 `PartialEq, Debug` 特质，所有的 primitive 类型和大多数标准库类型都实现了这两个特质
>  对于自己定义的 structs, enums，我们需要自行实现 `PartialEq` 和 `Debug`，但这两个特质都是可自动推导的，我们可以直接在定义上加 `#[derive(PartialEq, Debug)]` 标记

### Adding Custom Failure Messages
You can also add a custom message to be printed with the failure message as optional arguments to the `assert!`, `assert_eq!`, and `assert_ne!` macros. Any arguments specified after the required arguments are passed along to the `format!` macro (discussed in [“Concatenation with the `+` Operator or the `format!` Macro”](https://doc.rust-lang.org/book/ch08-02-strings.html#concatenation-with-the--operator-or-the-format-macro) in Chapter 8), so you can pass a format string that contains `{}` placeholders and values to go in those placeholders. Custom messages are useful for documenting what an assertion means; when a test fails, you’ll have a better idea of what the problem is with the code.

For example, let’s say we have a function that greets people by name and we want to test that the name we pass into the function appears in the output:

Filename: src/lib.rs

```rust
pub fn greeting(name: &str) -> String {
    format!("Hello {name}!")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_contains_name() {
        let result = greeting("Carol");
        assert!(result.contains("Carol"));
    }
}
```

The requirements for this program haven’t been agreed upon yet, and we’re pretty sure the `Hello` text at the beginning of the greeting will change. We decided we don’t want to have to update the test when the requirements change, so instead of checking for exact equality to the value returned from the `greeting` function, we’ll just assert that the output contains the text of the input parameter.

Now let’s introduce a bug into this code by changing `greeting` to exclude `name` to see what the default test failure looks like:

```rust
pub fn greeting(name: &str) -> String {
    String::from("Hello!")
}
```

Running this test produces the following:

```
$ cargo test
   Compiling greeter v0.1.0 (file:///projects/greeter)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.91s
     Running unittests src/lib.rs (target/debug/deps/greeter-170b942eb5bf5e3a)

running 1 test
test tests::greeting_contains_name ... FAILED

failures:

---- tests::greeting_contains_name stdout ----

thread 'tests::greeting_contains_name' panicked at src/lib.rs:12:9:
assertion failed: result.contains("Carol")
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::greeting_contains_name

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
```


This result just indicates that the assertion failed and which line the assertion is on. A more useful failure message would print the value from the `greeting` function. Let’s add a custom failure message composed of a format string with a placeholder filled in with the actual value we got from the `greeting` function:

```rust
#[test]
fn greeting_contains_name() {
    let result = greeting("Carol");
    assert!(
        result.contains("Carol"),
        "Greeting did not contain name, value was `{result}`"
    );
}
```

Now when we run the test, we’ll get a more informative error message:

```
$ cargo test
   Compiling greeter v0.1.0 (file:///projects/greeter)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.93s
     Running unittests src/lib.rs (target/debug/deps/greeter-170b942eb5bf5e3a)

running 1 test
test tests::greeting_contains_name ... FAILED

failures:

---- tests::greeting_contains_name stdout ----

thread 'tests::greeting_contains_name' panicked at src/lib.rs:12:9:
Greeting did not contain name, value was `Hello!`
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::greeting_contains_name

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
```

We can see the value we actually got in the test output, which would help us debug what happened instead of what we were expecting to happen.

### Checking for Panics with `should_panic`
In addition to checking return values, it’s important to check that our code handles error conditions as we expect. For example, consider the `Guess` type that we created in Chapter 9, Listing 9-13. Other code that uses `Guess` depends on the guarantee that `Guess` instances will contain only values between 1 and 100. We can write a test that ensures that attempting to create a `Guess` instance with a value outside that range panics.

We do this by adding the attribute `should_panic` to our test function. The test passes if the code inside the function panics; the test fails if the code inside the function doesn’t panic.
>  要测试是否函数可以处理错误情况，可以在测试函数上添加属性 `should_panic`，如果函数内代码 panic，则测试通过，否则测试失败

Listing 11-8 shows a test that checks that the error conditions of `Guess::new` happen when we expect them to.

Filename: src/lib.rs

```rust
pub struct Guess {
    value: i32,
}

impl Guess {
    pub fn new(value: i32) -> Guess {
        if value < 1 || value > 100 {
            panic!("Guess value must be between 1 and 100, got {value}.");
        }

        Guess { value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn greater_than_100() {
        Guess::new(200);
    }
}
```

[Listing 11-8](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#listing-11-8): Testing that a condition will cause a `panic!`

We place the `#[should_panic]` attribute after the `#[test]` attribute and before the test function it applies to. Let’s look at the result when this test passes:

```
$ cargo test
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.58s
     Running unittests src/lib.rs (target/debug/deps/guessing_game-57d70c3acb738f4d)

running 1 test
test tests::greater_than_100 - should panic ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests guessing_game

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Looks good! Now let’s introduce a bug in our code by removing the condition that the `new` function will panic if the value is greater than 100:

```rust
// --snip--
impl Guess {
    pub fn new(value: i32) -> Guess {
        if value < 1 {
            panic!("Guess value must be between 1 and 100, got {value}.");
        }

        Guess { value }
    }
}
```

When we run the test in Listing 11-8, it will fail:

```
$ cargo test
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.62s
     Running unittests src/lib.rs (target/debug/deps/guessing_game-57d70c3acb738f4d)

running 1 test
test tests::greater_than_100 - should panic ... FAILED

failures:

---- tests::greater_than_100 stdout ----
note: test did not panic as expected

failures:
    tests::greater_than_100

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
```

We don’t get a very helpful message in this case, but when we look at the test function, we see that it’s annotated with `#[should_panic]`. The failure we got means that the code in the test function did not cause a panic.

Tests that use `should_panic` can be imprecise. A `should_panic` test would pass even if the test panics for a different reason from the one we were expecting. To make `should_panic` tests more precise, we can add an optional `expected` parameter to the `should_panic` attribute. The test harness will make sure that the failure message contains the provided text. For example, consider the modified code for `Guess` in Listing 11-9 where the `new` function panics with different messages depending on whether the value is too small or too large.
> 要让 `should_panic` 更精确，我们可以为 `should_panic` 属性添加 `expected` 参数，这会让测试确保失败信息包含所提供的文本，确保函数是因为正确的原因 panic

Filename: src/lib.rs

```rust
// --snip--

impl Guess {
    pub fn new(value: i32) -> Guess {
        if value < 1 {
            panic!(
                "Guess value must be greater than or equal to 1, got {value}."
            );
        } else if value > 100 {
            panic!(
                "Guess value must be less than or equal to 100, got {value}."
            );
        }

        Guess { value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "less than or equal to 100")]
    fn greater_than_100() {
        Guess::new(200);
    }
}
```

[Listing 11-9](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#listing-11-9): Testing for a `panic!` with a panic message containing a specified substring

This test will pass because the value we put in the `should_panic` attribute’s `expected` parameter is a substring of the message that the `Guess::new` function panics with. We could have specified the entire panic message that we expect, which in this case would be `Guess value must be less than or equal to 100, got 200`. What you choose to specify depends on how much of the panic message is unique or dynamic and how precise you want your test to be. In this case, a substring of the panic message is enough to ensure that the code in the test function executes the `else if value > 100` case.
>  我们也可以指定我们期望的完整 panic message

To see what happens when a `should_panic` test with an `expected` message fails, let’s again introduce a bug into our code by swapping the bodies of the `if value < 1` and the `else if value > 100` blocks:

```rust
if value < 1 {
    panic!(
        "Guess value must be less than or equal to 100, got {value}."
    );
} else if value > 100 {
    panic!(
        "Guess value must be greater than or equal to 1, got {value}."
    );
}
```

This time when we run the `should_panic` test, it will fail:

```
$ cargo test
   Compiling guessing_game v0.1.0 (file:///projects/guessing_game)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.66s
     Running unittests src/lib.rs (target/debug/deps/guessing_game-57d70c3acb738f4d)

running 1 test
test tests::greater_than_100 - should panic ... FAILED

failures:

---- tests::greater_than_100 stdout ----

thread 'tests::greater_than_100' panicked at src/lib.rs:12:13:
Guess value must be greater than or equal to 1, got 200.
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
note: panic did not contain expected string
      panic message: `"Guess value must be greater than or equal to 1, got 200."`,
 expected substring: `"less than or equal to 100"`

failures:
    tests::greater_than_100

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
```

The failure message indicates that this test did indeed panic as we expected, but the panic message did not include the expected string `less than or equal to 100`. The panic message that we did get in this case was `Guess value must be greater than or equal to 1, got 200.` Now we can start figuring out where our bug is!

### Using `Result<T, E>` in Tests
Our tests so far all panic when they fail. We can also write tests that use `Result<T, E>`! Here’s the test from Listing 11-1, rewritten to use `Result<T, E>` and return an `Err` instead of panicking:

```rust
#[test]
fn it_works() -> Result<(), String> {
    let result = add(2, 2);

    if result == 4 {
        Ok(())
    } else {
        Err(String::from("two plus two does not equal four"))
    }
}
```

The `it_works` function now has the `Result<(), String>` return type. In the body of the function, rather than calling the `assert_eq!` macro, we return `Ok(())` when the test passes and an `Err` with a `String` inside when the test fails.
>  测试函数也可以返回 `Result<(), String>` 表示测试是否通过，而不是使用 panic

Writing tests so they return a `Result<T, E>` enables you to use the question mark operator in the body of tests, which can be a convenient way to write tests that should fail if any operation within them returns an `Err` variant.

You can’t use the `#[should_panic]` annotation on tests that use `Result<T, E>`. To assert that an operation returns an `Err` variant, _don’t_ use the question mark operator on the `Result<T, E>` value. Instead, use `assert!(value.is_err())`.
>  使用 `Result<T, E>` 的测试函数不能使用 `#[should_panic]` 标记
>  要断言一个操作返回 `Err` 变体，不要在 `Result<T, E>` value 上使用 `?`，而是使用 `assert!(value.is_err())`

Now that you know several ways to write tests, let’s look at what is happening when we run our tests and explore the different options we can use with `cargo test`.

## 11.2 Controlling How Tests Are Run
Just as `cargo run` compiles your code and then runs the resultant binary, `cargo test` compiles your code in test mode and runs the resultant test binary. The default behavior of the binary produced by `cargo test` is to run all the tests in parallel and capture output generated during test runs, preventing the output from being displayed and making it easier to read the output related to the test results. You can, however, specify command line options to change this default behavior.
>  `cargo test` 以测试模式编译我们的代码，运行得到的二进制
>  `cargo test` 产生的二进制的默认行为是并行运行所有测试，并在测试运行时捕获生成的输出，避免输出被展示并且让测试结果相关的输出更易读

Some command line options go to `cargo test`, and some go to the resultant test binary. To separate these two types of arguments, you list the arguments that go to `cargo test` followed by the separator `--` and then the ones that go to the test binary. Running `cargo test --help` displays the options you can use with `cargo test`, and running `cargo test -- --help` displays the options you can use after the separator. Those options are also documented in [the “Tests” section](https://doc.rust-lang.org/rustc/tests/index.html) of the [the rustc book](https://doc.rust-lang.org/rustc/index.html).
>  一些命令行选项传递给 `cargo test`，一些传递给生成的测试二进制文件
>  `cargo test` 之后跟传递给 `cargo test` 的选项，`--` 之后跟传递给二进制文件的选项

### Running Tests in Parallel or Consecutively
When you run multiple tests, by default they run in parallel using threads, meaning they finish running faster and you get feedback quicker. Because the tests are running at the same time, you must make sure your tests don’t depend on each other or on any shared state, including a shared environment, such as the current working directory or environment variables.
>  运行多个测试时，它们默认多线程并行运行，因此要注意测试不会相互依赖或依赖于共享状态 (包括了共享环境，例如当前工作目录或环境变量)

For example, say each of your tests runs some code that creates a file on disk named _test-output.txt_ and writes some data to that file. Then each test reads the data in that file and asserts that the file contains a particular value, which is different in each test. Because the tests run at the same time, one test might overwrite the file in the time between another test writing and reading the file. The second test will then fail, not because the code is incorrect but because the tests have interfered with each other while running in parallel. One solution is to make sure each test writes to a different file; another solution is to run the tests one at a time.

If you don’t want to run the tests in parallel or if you want more fine-grained control over the number of threads used, you can send the `--test-threads` flag and the number of threads you want to use to the test binary. Take a look at the following example:

```
$ cargo test -- --test-threads=1 
```

>  如果要控制运行测试的线程数，可以如上指定
>  设置为 1 时，就顺序运行

We set the number of test threads to `1`, telling the program not to use any parallelism. Running the tests using one thread will take longer than running them in parallel, but the tests won’t interfere with each other if they share state.

### Showing Function Output
By default, if a test passes, Rust’s test library captures anything printed to standard output. For example, if we call `println!` in a test and the test passes, we won’t see the `println!` output in the terminal; we’ll see only the line that indicates the test passed. If a test fails, we’ll see whatever was printed to standard output with the rest of the failure message.
>  测试库或捕获所有对标准输出的打印，如果测试成功，我们不会看到它们，如果测试失败，我们会看到它们

As an example, Listing 11-10 has a silly function that prints the value of its parameter and returns 10, as well as a test that passes and a test that fails.

Filename: src/lib.rs

```rust
fn prints_and_returns_10(a: i32) -> i32 {
    println!("I got the value {a}");
    10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_test_will_pass() {
        let value = prints_and_returns_10(4);
        assert_eq!(value, 10);
    }

    #[test]
    fn this_test_will_fail() {
        let value = prints_and_returns_10(8);
        assert_eq!(value, 5);
    }
}
```

[Listing 11-10](https://doc.rust-lang.org/book/ch11-02-running-tests.html#listing-11-10): Tests for a function that calls `println!`

When we run these tests with `cargo test`, we’ll see the following output:

```
$ cargo test
   Compiling silly-function v0.1.0 (file:///projects/silly-function)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.58s
     Running unittests src/lib.rs (target/debug/deps/silly_function-160869f38cff9166)

running 2 tests
test tests::this_test_will_fail ... FAILED
test tests::this_test_will_pass ... ok

failures:

---- tests::this_test_will_fail stdout ----
I got the value 8

thread 'tests::this_test_will_fail' panicked at src/lib.rs:19:9:
assertion `left == right` failed
  left: 10
 right: 5
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::this_test_will_fail

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
```

Note that nowhere in this output do we see `I got the value 4`, which is printed when the test that passes runs. That output has been captured. The output from the test that failed, `I got the value 8`, appears in the section of the test summary output, which also shows the cause of the test failure.

If we want to see printed values for passing tests as well, we can tell Rust to also show the output of successful tests with `--show-output`:

```
$ cargo test -- --show-output 
```

>  如果我们想在测试成功时也看到输出，可以如上指定

When we run the tests in Listing 11-10 again with the `--show-output` flag, we see the following output:

```
$ cargo test -- --show-output
   Compiling silly-function v0.1.0 (file:///projects/silly-function)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.60s
     Running unittests src/lib.rs (target/debug/deps/silly_function-160869f38cff9166)

running 2 tests
test tests::this_test_will_fail ... FAILED
test tests::this_test_will_pass ... ok

successes:

---- tests::this_test_will_pass stdout ----
I got the value 4


successes:
    tests::this_test_will_pass

failures:

---- tests::this_test_will_fail stdout ----
I got the value 8

thread 'tests::this_test_will_fail' panicked at src/lib.rs:19:9:
assertion `left == right` failed
  left: 10
 right: 5
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    tests::this_test_will_fail

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `--lib`
```

### Running a Subset of Tests by Name
Sometimes, running a full test suite can take a long time. If you’re working on code in a particular area, you might want to run only the tests pertaining to that code. You can choose which tests to run by passing `cargo test` the name or names of the test(s) you want to run as an argument.

To demonstrate how to run a subset of tests, we’ll first create three tests for our `add_two` function, as shown in Listing 11-11, and choose which ones to run.

Filename: src/lib.rs

```rust
pub fn add_two(a: u64) -> u64 {
    a + 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_two_and_two() {
        let result = add_two(2);
        assert_eq!(result, 4);
    }

    #[test]
    fn add_three_and_two() {
        let result = add_two(3);
        assert_eq!(result, 5);
    }

    #[test]
    fn one_hundred() {
        let result = add_two(100);
        assert_eq!(result, 102);
    }
}
```

[Listing 11-11](https://doc.rust-lang.org/book/ch11-02-running-tests.html#listing-11-11): Three tests with three different names

If we run the tests without passing any arguments, as we saw earlier, all the tests will run in parallel:

```
$ cargo test
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.62s
     Running unittests src/lib.rs (target/debug/deps/adder-92948b65e88960b4)

running 3 tests
test tests::add_three_and_two ... ok
test tests::add_two_and_two ... ok
test tests::one_hundred ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

#### Running Single Tests
We can pass the name of any test function to `cargo test` to run only that test:

```
$ cargo test one_hundred
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.69s
     Running unittests src/lib.rs (target/debug/deps/adder-92948b65e88960b4)

running 1 test
test tests::one_hundred ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.00s
```

Only the test with the name `one_hundred` ran; the other two tests didn’t match that name. The test output lets us know we had more tests that didn’t run by displaying `2 filtered out` at the end.

>  可以传递测试名称使得只运行一个测试

We can’t specify the names of multiple tests in this way; only the first value given to `cargo test` will be used. But there is a way to run multiple tests.

#### Filtering to Run Multiple Tests
We can specify part of a test name, and any test whose name matches that value will be run. For example, because two of our tests’ names contain `add`, we can run those two by running `cargo test add`:

```
$ cargo test add
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.61s
     Running unittests src/lib.rs (target/debug/deps/adder-92948b65e88960b4)

running 2 tests
test tests::add_three_and_two ... ok
test tests::add_two_and_two ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.00s
```

This command ran all tests with `add` in the name and filtered out the test named `one_hundred`. Also note that the module in which a test appears becomes part of the test’s name, so we can run all the tests in a module by filtering on the module’s name.

>  实际上包含了传递的字符串的测试名都会被运行
>  如果要运行 module 内的所有测试，指定 module name 即可

### Ignoring Some Tests Unless Specifically Requested
Sometimes a few specific tests can be very time-consuming to execute, so you might want to exclude them during most runs of `cargo test`. Rather than listing as arguments all tests you do want to run, you can instead annotate the time-consuming tests using the `ignore` attribute to exclude them, as shown here:

Filename: src/lib.rs

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    #[ignore]
    fn expensive_test() {
        // code that takes an hour to run
    }
}
```

After `#[test]`, we add the `#[ignore]` line to the test we want to exclude. Now when we run our tests, `it_works` runs, but `expensive_test` doesn’t:
>  使用 `#[ignore]` 属性可以忽略一些测试

```
$ cargo test
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.60s
     Running unittests src/lib.rs (target/debug/deps/adder-92948b65e88960b4)

running 2 tests
test tests::expensive_test ... ignored
test tests::it_works ... ok

test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

The `expensive_test` function is listed as `ignored`. If we want to run only the ignored tests, we can use `cargo test -- --ignored`:

```
$ cargo test -- --ignored
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.61s
     Running unittests src/lib.rs (target/debug/deps/adder-92948b65e88960b4)

running 1 test
test tests::expensive_test ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.00s

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

By controlling which tests run, you can make sure your `cargo test` results will be returned quickly. When you’re at a point where it makes sense to check the results of the `ignored` tests and you have time to wait for the results, you can run `cargo test -- --ignored` instead. If you want to run all tests whether they’re ignored or not, you can run `cargo test -- --include-ignored`.
>  也可以仅运行被忽略的测试: `cargo test -- --ignored`
>  如果要运行所有: `cargo test -- --include-ignored`

## 11.3 Test Organization
As mentioned at the start of the chapter, testing is a complex discipline, and different people use different terminology and organization. The Rust community thinks about tests in terms of two main categories: unit tests and integration tests. _Unit tests_ are small and more focused, testing one module in isolation at a time, and can test private interfaces. _Integration tests_ are entirely external to your library and use your code in the same way any other external code would, using only the public interface and potentially exercising multiple modules per test.
>  Rust 社区主要将测试分为两类: 单元测试和集成测试
>  单元测试通常规模小且集中，一次对一个模块进行隔离测试，且可以测试私有接口
>  集成测试完全位于库之外，如同使用其他代码一样使用我们的代码，仅使用贡藕给你接口，并且每个测试可能涉及多个模块

Writing both kinds of tests is important to ensure that the pieces of your library are doing what you expect them to, separately and together.

### Unit Tests
The purpose of unit tests is to test each unit of code in isolation from the rest of the code to quickly pinpoint where code is and isn’t working as expected. You’ll put unit tests in the _src_ directory in each file with the code that they’re testing. The convention is to create a module named `tests` in each file to contain the test functions and to annotate the module with `cfg(test)`.
>  单元测试的目的是独立测试每个代码单元，以快速定位哪里的代码正常/不正常工作
>  单元测试可以放在 `src` 目录下的各个文件中，与被测试代码位于同一个文件
>  通常的做法是在每个文件中创建要给名为 `tests` 的 module，包含测试函数，并且使用 `cfg(test)` 标注该 module

#### The Tests Module and `#[cfg(test)]`
The `#[cfg(test)]` annotation on the `tests` module tells Rust to compile and run the test code only when you run `cargo test`, not when you run `cargo build`. This saves compile time when you only want to build the library and saves space in the resultant compiled artifact because the tests are not included. You’ll see that because integration tests go in a different directory, they don’t need the `#[cfg(test)]` annotation. However, because unit tests go in the same files as the code, you’ll use `#[cfg(test)]` to specify that they shouldn’t be included in the compiled result.
>  `#[cfg(test)]` 标记告诉 Rust 仅在使用 `cargo test` 时编译并运行测试代码
>  在正常 `cargo build` 的产物中不会包含测试相关内容
>  因为集成测试位于不同的目录，它们就不需要 `#[cfg(test)]` 标记，但单元测试位于同一文件中，因此标记是有必要的

Recall that when we generated the new `adder` project in the first section of this chapter, Cargo generated this code for us:

Filename: src/lib.rs

```rust
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
```

On the automatically generated `tests` module, the attribute `cfg` stands for _configuration_ and tells Rust that the following item should only be included given a certain configuration option. In this case, the configuration option is `test`, which is provided by Rust for compiling and running tests. By using the `cfg` attribute, Cargo compiles our test code only if we actively run the tests with `cargo test`. This includes any helper functions that might be within this module, in addition to the functions annotated with `#[test]`.
>  属性 `cfg` 表示配置，它告诉 Rust 下面的 item 应该在给定特定配置选项时才被包含
>  配置选项为 `test` 时，就是在编译和运行测试时才包含

#### Testing Private Functions
There’s debate within the testing community about whether or not private functions should be tested directly, and other languages make it difficult or impossible to test private functions. Regardless of which testing ideology you adhere to, Rust’s privacy rules do allow you to test private functions. Consider the code in Listing 11-12 with the private function `internal_adder`.
>  Rust 的隐私规则允许我们测试私有函数

Filename: src/lib.rs

```rust
pub fn add_two(a: u64) -> u64 {
    internal_adder(a, 2)
}

fn internal_adder(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal() {
        let result = internal_adder(2, 2);
        assert_eq!(result, 4);
    }
}
```

[Listing 11-12](https://doc.rust-lang.org/book/ch11-03-test-organization.html#listing-11-12): Testing a private function

Note that the `internal_adder` function is not marked as `pub`. Tests are just Rust code, and the `tests` module is just another module. As we discussed in [“Paths for Referring to an Item in the Module Tree”](https://doc.rust-lang.org/book/ch07-03-paths-for-referring-to-an-item-in-the-module-tree.html), items in child modules can use the items in their ancestor modules. In this test, we bring all of the `tests` module’s parent’s items into scope with `use super::*`, and then the test can call `internal_adder`. If you don’t think private functions should be tested, there’s nothing in Rust that will compel you to do so.
>  `tests` module 是另外的 module，但子 module 可以使用祖先 module 的东西

### Integration Tests
In Rust, integration tests are entirely external to your library. They use your library in the same way any other code would, which means they can only call functions that are part of your library’s public API. Their purpose is to test whether many parts of your library work together correctly. Units of code that work correctly on their own could have problems when integrated, so test coverage of the integrated code is important as well. To create integration tests, you first need a _tests_ directory.
>  集成测试完全在我们的库之外，它们只调用库的公共 API，它们的目的是测试是否库的各个组件能一同工作，有时各个组件独立工作没问题，但是协同会有问题，因此对集成代码的测试覆盖也很重要

#### The _tests_ Directory
We create a _tests_ directory at the top level of our project directory, next to _src_. Cargo knows to look for integration test files in this directory. We can then make as many test files as we want, and Cargo will compile each of the files as an individual crate.
>  我们需要在项目目录顶部创建 `tests` 目录，Cargo 默认在这个目录寻找集成测试文件
>  我们在里面添加测试文件，Cargo 会将它们逐一编译为单独的 crate

Let’s create an integration test. With the code in Listing 11-12 still in the _src/lib.rs_ file, make a _tests_ directory, and create a new file named _tests/integration_test.rs_. Your directory structure should look like this:

```
adder
├── Cargo.lock
├── Cargo.toml
├── src
│   └── lib.rs
└── tests
    └── integration_test.rs
```

Enter the code in Listing 11-13 into the _tests/integration_test.rs_ file.

Filename: tests/integration_test.rs

```rust
use adder::add_two;

#[test]
fn it_adds_two() {
    let result = add_two(2);
    assert_eq!(result, 4);
}
```

[Listing 11-13](https://doc.rust-lang.org/book/ch11-03-test-organization.html#listing-11-13): An integration test of a function in the `adder` crate

Each file in the _tests_ directory is a separate crate, so we need to bring our library into each test crate’s scope. For that reason we add `use adder::add_two;` at the top of the code, which we didn’t need in the unit tests.
>  `tests` 目录下的每个文件都是独立的 crate，因此我们需要将库带入 test crate 的作用域

We don’t need to annotate any code in _tests/integration_test.rs_ with `#[cfg(test)]`. Cargo treats the _tests_ directory specially and compiles files in this directory only when we run `cargo test`. Run `cargo test` now:

```
$ cargo test
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 1.31s
     Running unittests src/lib.rs (target/debug/deps/adder-1082c4b063a8fbe6)

running 1 test
test tests::internal ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/integration_test.rs (target/debug/deps/integration_test-1082c4b063a8fbe6)

running 1 test
test it_adds_two ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

>  Cargo 只会在 `cargo test` 编译 `tests` 下的文件，因此不需要特别注明 `#[cfg(test)]`

The three sections of output include the unit tests, the integration test, and the doc tests. Note that if any test in a section fails, the following sections will not be run. For example, if a unit test fails, there won’t be any output for integration and doc tests because those tests will only be run if all unit tests are passing.
>  测试的三部分包括单元测试、集成测试、文档测试
>  如果任意测试部分失败，后续的测试部分不会运行

The first section for the unit tests is the same as we’ve been seeing: one line for each unit test (one named `internal` that we added in Listing 11-12) and then a summary line for the unit tests.

The integration tests section starts with the line `Running tests/integration_test.rs`. Next, there is a line for each test function in that integration test and a summary line for the results of the integration test just before the `Doc-tests adder` section starts.

Each integration test file has its own section, so if we add more files in the _tests_ directory, there will be more integration test sections.

We can still run a particular integration test function by specifying the test function’s name as an argument to `cargo test`. To run all the tests in a particular integration test file, use the `--test` argument of `cargo test` followed by the name of the file:
>  使用 `--test` 可以指定集成测试文件

```
$ cargo test --test integration_test
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.64s
     Running tests/integration_test.rs (target/debug/deps/integration_test-82e7799c1bc62298)

running 1 test
test it_adds_two ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

This command runs only the tests in the _tests/integration_test.rs_ file.

#### Submodules in Integration Tests
As you add more integration tests, you might want to make more files in the _tests_ directory to help organize them; for example, you can group the test functions by the functionality they’re testing. As mentioned earlier, each file in the _tests_ directory is compiled as its own separate crate, which is useful for creating separate scopes to more closely imitate the way end users will be using your crate. However, this means files in the _tests_ directory don’t share the same behavior as files in _src_ do, as you learned in Chapter 7 regarding how to separate code into modules and files.

The different behavior of _tests_ directory files is most noticeable when you have a set of helper functions to use in multiple integration test files and you try to follow the steps in the [“Separating Modules into Different Files”](https://doc.rust-lang.org/book/ch07-05-separating-modules-into-different-files.html) section of Chapter 7 to extract them into a common module. For example, if we create _tests/common.rs_ and place a function named `setup` in it, we can add some code to `setup` that we want to call from multiple test functions in multiple test files:

Filename: tests/common.rs

```rust
pub fn setup() {
    // setup code specific to your library's tests would go here
}
```

When we run the tests again, we’ll see a new section in the test output for the _common.rs_ file, even though this file doesn’t contain any test functions nor did we call the `setup` function from anywhere:

```
$ cargo test
   Compiling adder v0.1.0 (file:///projects/adder)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.89s
     Running unittests src/lib.rs (target/debug/deps/adder-92948b65e88960b4)

running 1 test
test tests::internal ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/common.rs (target/debug/deps/common-92948b65e88960b4)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/integration_test.rs (target/debug/deps/integration_test-92948b65e88960b4)

running 1 test
test it_adds_two ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests adder

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Having `common` appear in the test results with `running 0 tests` displayed for it is not what we wanted. We just wanted to share some code with the other integration test files. To avoid having `common` appear in the test output, instead of creating _tests/common.rs_, we’ll create _tests/common/mod.rs_. The project directory now looks like this:

```
├── Cargo.lock
├── Cargo.toml
├── src
│   └── lib.rs
└── tests
    ├── common
    │   └── mod.rs
    └── integration_test.rs
```

This is the older naming convention that Rust also understands that we mentioned in [“Alternate File Paths”](https://doc.rust-lang.org/book/ch07-05-separating-modules-into-different-files.html#alternate-file-paths) in Chapter 7. Naming the file this way tells Rust not to treat the `common` module as an integration test file. When we move the `setup` function code into _tests/common/mod.rs_ and delete the _tests/common.rs_ file, the section in the test output will no longer appear. Files in subdirectories of the _tests_ directory don’t get compiled as separate crates or have sections in the test output.
>  `tests` 的子目录中的文件不会被编译为独立的 crate 或在 test 输出中有 section

After we’ve created _tests/common/mod.rs_, we can use it from any of the integration test files as a module. Here’s an example of calling the `setup` function from the `it_adds_two` test in _tests/integration_test.rs_:

Filename: tests/integration_test.rs

```rust
use adder::add_two;

mod common;

#[test]
fn it_adds_two() {
    common::setup();

    let result = add_two(2);
    assert_eq!(result, 4);
}
```

Note that the `mod common;` declaration is the same as the module declaration we demonstrated in Listing 7-21. Then, in the test function, we can call the `common::setup()` function.

#### Integration Tests for Binary Crates
If our project is a binary crate that only contains a _src/main.rs_ file and doesn’t have a _src/lib.rs_ file, we can’t create integration tests in the _tests_ directory and bring functions defined in the _src/main.rs_ file into scope with a `use` statement. Only library crates expose functions that other crates can use; binary crates are meant to be run on their own.
>  只有库 crate 可以向其他 crate 暴露 API

This is one of the reasons Rust projects that provide a binary have a straightforward _src/main.rs_ file that calls logic that lives in the _src/lib.rs_ file. Using that structure, integration tests _can_ test the library crate with `use` to make the important functionality available. If the important functionality works, the small amount of code in the _src/main.rs_ file will work as well, and that small amount of code doesn’t need to be tested.

## Summary
Rust’s testing features provide a way to specify how code should function to ensure it continues to work as you expect, even as you make changes. Unit tests exercise different parts of a library separately and can test private implementation details. Integration tests check that many parts of the library work together correctly, and they use the library’s public API to test the code in the same way external code will use it. Even though Rust’s type system and ownership rules help prevent some kinds of bugs, tests are still important to reduce logic bugs having to do with how your code is expected to behave.

Let’s combine the knowledge you learned in this chapter and in previous chapters to work on a project!

# 12 An I/O Project: Building a Command Line Program
This chapter is a recap of the many skills you’ve learned so far and an exploration of a few more standard library features. We’ll build a command line tool that interacts with file and command line input/output to practice some of the Rust concepts you now have under your belt.

Rust’s speed, safety, single binary output, and cross-platform support make it an ideal language for creating command line tools, so for our project, we’ll make our own version of the classic command line search tool `grep` (**g**lobally search a **r**egular **e**xpression and **p**rint). In the simplest use case, `grep` searches a specified file for a specified string. To do so, `grep` takes as its arguments a file path and a string. Then it reads the file, finds lines in that file that contain the string argument, and prints those lines.

Along the way, we’ll show how to make our command line tool use the terminal features that many other command line tools use. We’ll read the value of an environment variable to allow the user to configure the behavior of our tool. We’ll also print error messages to the standard error console stream (`stderr`) instead of standard output (`stdout`) so that, for example, the user can redirect successful output to a file while still seeing error messages onscreen.

One Rust community member, Andrew Gallant, has already created a fully featured, very fast version of `grep`, called `ripgrep`. By comparison, our version will be fairly simple, but this chapter will give you some of the background knowledge you need to understand a real-world project such as `ripgrep`.

Our `grep` project will combine a number of concepts you’ve learned so far:

- Organizing code ([Chapter 7](https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html))
- Using vectors and strings ([Chapter 8](https://doc.rust-lang.org/book/ch08-00-common-collections.html))
- Handling errors ([Chapter 9](https://doc.rust-lang.org/book/ch09-00-error-handling.html))
- Using traits and lifetimes where appropriate ([Chapter 10](https://doc.rust-lang.org/book/ch10-00-generics.html))
- Writing tests ([Chapter 11](https://doc.rust-lang.org/book/ch11-00-testing.html))

We’ll also briefly introduce closures, iterators, and trait objects, which [Chapter 13](https://doc.rust-lang.org/book/ch13-00-functional-features.html) and [Chapter 18](https://doc.rust-lang.org/book/ch18-00-oop.html) will cover in detail.

## 12.1 Accepting Command Line Arguments
Let’s create a new project with, as always, `cargo new`. We’ll call our project `minigrep` to distinguish it from the `grep` tool that you might already have on your system.

```
$ cargo new minigrep
     Created binary (application) `minigrep` project
$ cd minigrep
```

The first task is to make `minigrep` accept its two command line arguments: the file path and a string to search for. That is, we want to be able to run our program with `cargo run`, two hyphens to indicate the following arguments are for our program rather than for `cargo`, a string to search for, and a path to a file to search in, like so:

```
$ cargo run -- searchstring example-filename.txt 
```

>  使用 `--` 作为分隔符，后面的参数会交给二进制文件

Right now, the program generated by `cargo new` cannot process arguments we give it. Some existing libraries on [crates.io](https://crates.io/) can help with writing a program that accepts command line arguments, but because you’re just learning this concept, let’s implement this capability ourselves.

### Reading the Argument Values
To enable `minigrep` to read the values of command line arguments we pass to it, we’ll need the `std::env::args` function provided in Rust’s standard library. This function returns an iterator of the command line arguments passed to `minigrep`. We’ll cover iterators fully in [Chapter 13](https://doc.rust-lang.org/book/ch13-00-functional-features.html). For now, you only need to know two details about iterators: iterators produce a series of values, and we can call the `collect` method on an iterator to turn it into a collection, such as a vector, that contains all the elements the iterator produces.
>  `std::env::args` 函数返回传递给二进制文件的命令行参数的迭代器
>  我们对迭代器调用 `collect` 方法将它转为集合类型

The code in Listing 12-1 allows your `minigrep` program to read any command line arguments passed to it, and then collect the values into a vector.

Filename: src/main.rs

```rust
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    dbg!(args);
}
```

[Listing 12-1](https://doc.rust-lang.org/book/ch12-01-accepting-command-line-arguments.html#listing-12-1): Collecting the command line arguments into a vector and printing them

First we bring the `std::env` module into scope with a `use` statement so we can use its `args` function. Notice that the `std::env::args` function is nested in two levels of modules. As we discussed in [Chapter 7](https://doc.rust-lang.org/book/ch07-04-bringing-paths-into-scope-with-the-use-keyword.html#creating-idiomatic-use-paths), in cases where the desired function is nested in more than one module, we’ve chosen to bring the parent module into scope rather than the function. By doing so, we can easily use other functions from `std::env`. It’s also less ambiguous than adding `use std::env::args` and then calling the function with just `args`, because `args` might easily be mistaken for a function that’s defined in the current module.

***The `args` Function and Invalid Unicode***
Note that `std::env::args` will panic if any argument contains invalid Unicode. If your program needs to accept arguments containing invalid Unicode, use `std::env::args_os` instead. That function returns an iterator that produces `OsString` values instead of `String` values. We’ve chosen to use `std::env::args` here for simplicity because `OsString` values differ per platform and are more complex to work with than `String` values.
>  `std::env::args` 会在命令行参数包含无效 Unicode 字符时 panic

On the first line of `main`, we call `env::args`, and we immediately use `collect` to turn the iterator into a vector containing all the values produced by the iterator. We can use the `collect` function to create many kinds of collections, so we explicitly annotate the type of `args` to specify that we want a vector of strings. Although you very rarely need to annotate types in Rust, `collect` is one function you do often need to annotate because Rust isn’t able to infer the kind of collection you want.
>  `collect` 要求我们注解返回类型，以确定我们要哪一类集合类型

Finally, we print the vector using the debug macro. Let’s try running the code first with no arguments and then with two arguments:

```
$ cargo run
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.61s
     Running `target/debug/minigrep`
[src/main.rs:5:5] args = [
    "target/debug/minigrep",
]
```

```
$ cargo run -- needle haystack
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.57s
     Running `target/debug/minigrep needle haystack`
[src/main.rs:5:5] args = [
    "target/debug/minigrep",
    "needle",
    "haystack",
]
```

Notice that the first value in the vector is `"target/debug/minigrep"`, which is the name of our binary. This matches the behavior of the arguments list in C, letting programs use the name by which they were invoked in their execution. It’s often convenient to have access to the program name in case you want to print it in messages or change the behavior of the program based on what command line alias was used to invoke the program. But for the purposes of this chapter, we’ll ignore it and save only the two arguments we need.
>  第一个参数是二进制文件的名称

### Saving the Argument Values in Variables
The program is currently able to access the values specified as command line arguments. Now we need to save the values of the two arguments in variables so we can use the values throughout the rest of the program. We do that in Listing 12-2.

Filename: src/main.rs

```rust
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let query = &args[1];
    let file_path = &args[2];

    println!("Searching for {query}");
    println!("In file {file_path}");
}
```

[Listing 12-2](https://doc.rust-lang.org/book/ch12-01-accepting-command-line-arguments.html#listing-12-2): Creating variables to hold the query argument and file path argument

As we saw when we printed the vector, the program’s name takes up the first value in the vector at `args[0]`, so we’re starting arguments at index 1. The first argument `minigrep` takes is the string we’re searching for, so we put a reference to the first argument in the variable `query`. The second argument will be the file path, so we put a reference to the second argument in the variable `file_path`.

We temporarily print the values of these variables to prove that the code is working as we intend. Let’s run this program again with the arguments `test` and `sample.txt`:

```
$ cargo run -- test sample.txt
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.0s
     Running `target/debug/minigrep test sample.txt`
Searching for test
In file sample.txt
```

Great, the program is working! The values of the arguments we need are being saved into the right variables. Later we’ll add some error handling to deal with certain potential erroneous situations, such as when the user provides no arguments; for now, we’ll ignore that situation and work on adding file-reading capabilities instead.

## 12.2 Reading a File
Now we’ll add functionality to read the file specified in the `file_path` argument. First we need a sample file to test it with: we’ll use a file with a small amount of text over multiple lines with some repeated words. Listing 12-3 has an Emily Dickinson poem that will work well! Create a file called _poem.txt_ at the root level of your project, and enter the poem “I’m Nobody! Who are you?”

Filename: poem.txt

```
I'm nobody! Who are you?
Are you nobody, too?
Then there's a pair of us - don't tell!
They'd banish us, you know.

How dreary to be somebody!
How public, like a frog
To tell your name the livelong day
To an admiring bog!
```

[Listing 12-3](https://doc.rust-lang.org/book/ch12-02-reading-a-file.html#listing-12-3): A poem by Emily Dickinson makes a good test case.

With the text in place, edit _src/main.rs_ and add code to read the file, as shown in Listing 12-4.

Filename: src/main.rs

```rust
use std::env;
use std::fs;

fn main() {
    // --snip--
    println!("In file {file_path}");

    let contents = fs::read_to_string(file_path)
        .expect("Should have been able to read the file");

    println!("With text:\n{contents}");
}
```

[Listing 12-4](https://doc.rust-lang.org/book/ch12-02-reading-a-file.html#listing-12-4): Reading the contents of the file specified by the second argument

First we bring in a relevant part of the standard library with a `use` statement: we need `std::fs` to handle files.

In `main`, the new statement `fs::read_to_string` takes the `file_path`, opens that file, and returns a value of type `std::io::Result<String>` that contains the file’s contents.
>  `fs::read_to_string` 接受文件路径，打开文件，返回包含文件内容的 `std::io::Result<String>`

After that, we again add a temporary `println!` statement that prints the value of `contents` after the file is read, so we can check that the program is working so far.

Let’s run this code with any string as the first command line argument (because we haven’t implemented the searching part yet) and the _poem.txt_ file as the second argument:

```
$ cargo run -- the poem.txt
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.0s
     Running `target/debug/minigrep the poem.txt`
Searching for the
In file poem.txt
With text:
I'm nobody! Who are you?
Are you nobody, too?
Then there's a pair of us - don't tell!
They'd banish us, you know.

How dreary to be somebody!
How public, like a frog
To tell your name the livelong day
To an admiring bog!
```

Great! The code read and then printed the contents of the file. But the code has a few flaws. At the moment, the `main` function has multiple responsibilities: generally, functions are clearer and easier to maintain if each function is responsible for only one idea. The other problem is that we’re not handling errors as well as we could. The program is still small, so these flaws aren’t a big problem, but as the program grows, it will be harder to fix them cleanly. It’s a good practice to begin refactoring early on when developing a program because it’s much easier to refactor smaller amounts of code. We’ll do that next.

## 12.3 Refactoring to Improve Modularity and Error Handling
To improve our program, we’ll fix four problems that have to do with the program’s structure and how it’s handling potential errors. First, our `main` function now performs two tasks: it parses arguments and reads files. As our program grows, the number of separate tasks the `main` function handles will increase. As a function gains responsibilities, it becomes more difficult to reason about, harder to test, and harder to change without breaking one of its parts. It’s best to separate functionality so each function is responsible for one task.

This issue also ties into the second problem: although `query` and `file_path` are configuration variables to our program, variables like `contents` are used to perform the program’s logic. The longer `main` becomes, the more variables we’ll need to bring into scope; the more variables we have in scope, the harder it will be to keep track of the purpose of each. It’s best to group the configuration variables into one structure to make their purpose clear.

The third problem is that we’ve used `expect` to print an error message when reading the file fails, but the error message just prints `Should have been able to read the file`. Reading a file can fail in a number of ways: for example, the file could be missing, or we might not have permission to open it. Right now, regardless of the situation, we’d print the same error message for everything, which wouldn’t give the user any information!

Fourth, we use `expect` to handle an error, and if the user runs our program without specifying enough arguments, they’ll get an `index out of bounds` error from Rust that doesn’t clearly explain the problem. It would be best if all the error-handling code were in one place so future maintainers had only one place to consult the code if the error-handling logic needed to change. Having all the error-handling code in one place will also ensure that we’re printing messages that will be meaningful to our end users.

Let’s address these four problems by refactoring our project.

### Separation of Concerns for Binary Projects
The organizational problem of allocating responsibility for multiple tasks to the `main` function is common to many binary projects. As a result, many Rust programmers find it useful to split up the separate concerns of a binary program when the `main` function starts getting large. This process has the following steps:

- Split your program into a _main.rs_ file and a _lib.rs_ file and move your program’s logic to _lib.rs_.
- As long as your command line parsing logic is small, it can remain in the `main` function.
- When the command line parsing logic starts getting complicated, extract it from the `main` function into other functions or types.

>  分离程序逻辑的方法:
>  - 将程序分为 `main.rs, lib.rs`，将逻辑移动到 `lib.rs`
>  - 逻辑复杂时，将逻辑从 `main` 函数提取出来成为其他函数

The responsibilities that remain in the `main` function after this process should be limited to the following:

- Calling the command line parsing logic with the argument values
- Setting up any other configuration
- Calling a `run` function in _lib.rs_
- Handling the error if `run` returns an error

This pattern is about separating concerns: _main.rs_ handles running the program and _lib.rs_ handles all the logic of the task at hand. Because you can’t test the `main` function directly, this structure lets you test all of your program’s logic by moving it out of the `main` function. The code that remains in the `main` function will be small enough to verify its correctness by reading it. Let’s rework our program by following this process.
>  `main.rs` 运行程序，`lib.rs` 处理任务逻辑

#### Extracting the Argument Parser
We’ll extract the functionality for parsing arguments into a function that `main` will call. Listing 12-5 shows the new start of the `main` function that calls a new function `parse_config`, which we’ll define in _src/main.rs_.

Filename: src/main.rs

```rust
fn main() {
    let args: Vec<String> = env::args().collect();

    let (query, file_path) = parse_config(&args);

    // --snip--
}

fn parse_config(args: &[String]) -> (&str, &str) {
    let query = &args[1];
    let file_path = &args[2];

    (query, file_path)
}
```

[Listing 12-5](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#listing-12-5): Extracting a `parse_config` function from `main`

We’re still collecting the command line arguments into a vector, but instead of assigning the argument value at index 1 to the variable `query` and the argument value at index 2 to the variable `file_path` within the `main` function, we pass the whole vector to the `parse_config` function. The `parse_config` function then holds the logic that determines which argument goes in which variable and passes the values back to `main`. We still create the `query` and `file_path` variables in `main`, but `main` no longer has the responsibility of determining how the command line arguments and variables correspond.

This rework may seem like overkill for our small program, but we’re refactoring in small, incremental steps. After making this change, run the program again to verify that the argument parsing still works. It’s good to check your progress often, to help identify the cause of problems when they occur.

#### Grouping Configuration Values
We can take another small step to improve the `parse_config` function further. At the moment, we’re returning a tuple, but then we immediately break that tuple into individual parts again. This is a sign that perhaps we don’t have the right abstraction yet.

Another indicator that shows there’s room for improvement is the `config` part of `parse_config`, which implies that the two values we return are related and are both part of one configuration value. We’re not currently conveying this meaning in the structure of the data other than by grouping the two values into a tuple; we’ll instead put the two values into one struct and give each of the struct fields a meaningful name. Doing so will make it easier for future maintainers of this code to understand how the different values relate to each other and what their purpose is.

Listing 12-6 shows the improvements to the `parse_config` function.

Filename: src/main.rs

```rust
fn main() {
    let args: Vec<String> = env::args().collect();

    let config = parse_config(&args);

    println!("Searching for {}", config.query);
    println!("In file {}", config.file_path);

    let contents = fs::read_to_string(config.file_path)
        .expect("Should have been able to read the file");

    // --snip--
}

struct Config {
    query: String,
    file_path: String,
}

fn parse_config(args: &[String]) -> Config {
    let query = args[1].clone();
    let file_path = args[2].clone();

    Config { query, file_path }
}
```

[Listing 12-6](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#listing-12-6): Refactoring `parse_config` to return an instance of a `Config` struct

We’ve added a struct named `Config` defined to have fields named `query` and `file_path`. The signature of `parse_config` now indicates that it returns a `Config` value. In the body of `parse_config`, where we used to return string slices that reference `String` values in `args`, we now define `Config` to contain owned `String` values. The `args` variable in `main` is the owner of the argument values and is only letting the `parse_config` function borrow them, which means we’d violate Rust’s borrowing rules if `Config` tried to take ownership of the values in `args`.

There are a number of ways we could manage the `String` data; the easiest, though somewhat inefficient, route is to call the `clone` method on the values. This will make a full copy of the data for the `Config` instance to own, which takes more time and memory than storing a reference to the string data. However, cloning the data also makes our code very straightforward because we don’t have to manage the lifetimes of the references; in this circumstance, giving up a little performance to gain simplicity is a worthwhile trade-off.

### The Trade-Offs of Using `clone`
There’s a tendency among many Rustaceans to avoid using `clone` to fix ownership problems because of its runtime cost. In [Chapter 13](https://doc.rust-lang.org/book/ch13-00-functional-features.html), you’ll learn how to use more efficient methods in this type of situation. But for now, it’s okay to copy a few strings to continue making progress because you’ll make these copies only once and your file path and query string are very small. It’s better to have a working program that’s a bit inefficient than to try to hyperoptimize code on your first pass. As you become more experienced with Rust, it’ll be easier to start with the most efficient solution, but for now, it’s perfectly acceptable to call `clone`.

We’ve updated `main` so it places the instance of `Config` returned by `parse_config` into a variable named `config`, and we updated the code that previously used the separate `query` and `file_path` variables so it now uses the fields on the `Config` struct instead.

Now our code more clearly conveys that `query` and `file_path` are related and that their purpose is to configure how the program will work. Any code that uses these values knows to find them in the `config` instance in the fields named for their purpose.

#### Creating a Constructor for `Config`
So far, we’ve extracted the logic responsible for parsing the command line arguments from `main` and placed it in the `parse_config` function. Doing so helped us see that the `query` and `file_path` values were related, and that relationship should be conveyed in our code. We then added a `Config` struct to name the related purpose of `query` and `file_path` and to be able to return the values’ names as struct field names from the `parse_config` function.

So now that the purpose of the `parse_config` function is to create a `Config` instance, we can change `parse_config` from a plain function to a function named `new` that is associated with the `Config` struct. Making this change will make the code more idiomatic. We can create instances of types in the standard library, such as `String`, by calling `String::new`. Similarly, by changing `parse_config` into a `new` function associated with `Config`, we’ll be able to create instances of `Config` by calling `Config::new`. Listing 12-7 shows the changes we need to make.

Filename: src/main.rs

```rust
fn main() {
    let args: Vec<String> = env::args().collect();

    let config = Config::new(&args);

    // --snip--
}

// --snip--

impl Config {
    fn new(args: &[String]) -> Config {
        let query = args[1].clone();
        let file_path = args[2].clone();

        Config { query, file_path }
    }
}
```

[Listing 12-7](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#listing-12-7): Changing `parse_config` into `Config::new`

We’ve updated `main` where we were calling `parse_config` to instead call `Config::new`. We’ve changed the name of `parse_config` to `new` and moved it within an `impl` block, which associates the `new` function with `Config`. Try compiling this code again to make sure it works.

### Fixing the Error Handling
Now we’ll work on fixing our error handling. Recall that attempting to access the values in the `args` vector at index 1 or index 2 will cause the program to panic if the vector contains fewer than three items. Try running the program without any arguments; it will look like this:

```
$ cargo run
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.0s
     Running `target/debug/minigrep`

thread 'main' panicked at src/main.rs:27:21:
index out of bounds: the len is 1 but the index is 1
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

The line `index out of bounds: the len is 1 but the index is 1` is an error message intended for programmers. It won’t help our end users understand what they should do instead. Let’s fix that now.

#### Improving the Error Message
In Listing 12-8, we add a check in the `new` function that will verify that the slice is long enough before accessing index 1 and index 2. If the slice isn’t long enough, the program panics and displays a better error message.

Filename: src/main.rs

```rust
// --snip--
fn new(args: &[String]) -> Config {
    if args.len() < 3 {
        panic!("not enough arguments");
    }
    // --snip--
```

[Listing 12-8](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#listing-12-8): Adding a check for the number of arguments

This code is similar to [the `Guess::new` function we wrote in Listing 9-13](https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html#creating-custom-types-for-validation), where we called `panic!` when the `value` argument was out of the range of valid values. Instead of checking for a range of values here, we’re checking that the length of `args` is at least `3` and the rest of the function can operate under the assumption that this condition has been met. If `args` has fewer than three items, this condition will be `true`, and we call the `panic!` macro to end the program immediately.

With these extra few lines of code in `new`, let’s run the program without any arguments again to see what the error looks like now:

```
$ cargo run
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.0s
     Running `target/debug/minigrep`

thread 'main' panicked at src/main.rs:26:13:
not enough arguments
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

This output is better: we now have a reasonable error message. However, we also have extraneous information we don’t want to give to our users. Perhaps the technique we used in Listing 9-13 isn’t the best one to use here: a call to `panic!` is more appropriate for a programming problem than a usage problem, [as discussed in Chapter 9](https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html#guidelines-for-error-handling). Instead, we’ll use the other technique you learned about in Chapter 9— [returning a `Result`](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html) that indicates either success or an error.
>  对 `panic!` 的调用更适合于针对编程问题而不是使用问题

#### Returning a `Result` Instead of Calling `panic!`
We can instead return a `Result` value that will contain a `Config` instance in the successful case and will describe the problem in the error case. We’re also going to change the function name from `new` to `build` because many programmers expect `new` functions to never fail. When `Config::build` is communicating to `main`, we can use the `Result` type to signal there was a problem. Then we can change `main` to convert an `Err` variant into a more practical error for our users without the surrounding text about `thread 'main'` and `RUST_BACKTRACE` that a call to `panic!` causes.
>  `new` 通常永不失败，故我们需要定义 `build`

Listing 12-9 shows the changes we need to make to the return value of the function we’re now calling `Config::build` and the body of the function needed to return a `Result`. Note that this won’t compile until we update `main` as well, which we’ll do in the next listing.

Filename: src/main.rs

```rust
impl Config {
    fn build(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 3 {
            return Err("not enough arguments");
        }

        let query = args[1].clone();
        let file_path = args[2].clone();

        Ok(Config { query, file_path })
    }
}
```

[Listing 12-9](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#listing-12-9): Returning a `Result` from `Config::build`

Our `build` function returns a `Result` with a `Config` instance in the success case and a string literal in the error case. Our error values will always be string literals that have the `'static` lifetime.

We’ve made two changes in the body of the function: instead of calling `panic!` when the user doesn’t pass enough arguments, we now return an `Err` value, and we’ve wrapped the `Config` return value in an `Ok`. These changes make the function conform to its new type signature.

Returning an `Err` value from `Config::build` allows the `main` function to handle the `Result` value returned from the `build` function and exit the process more cleanly in the error case.

#### Calling `Config::build` and Handling Errors
To handle the error case and print a user-friendly message, we need to update `main` to handle the `Result` being returned by `Config::build`, as shown in Listing 12-10. We’ll also take the responsibility of exiting the command line tool with a nonzero error code away from `panic!` and instead implement it by hand. A nonzero exit status is a convention to signal to the process that called our program that the program exited with an error state.

Filename: src/main.rs

```rust
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    let config = Config::build(&args).unwrap_or_else(|err| {
        println!("Problem parsing arguments: {err}");
        process::exit(1);
    });

    // --snip--
```

[Listing 12-10](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#listing-12-10): Exiting with an error code if building a `Config` fails

In this listing, we’ve used a method we haven’t covered in detail yet: `unwrap_or_else`, which is defined on `Result<T, E>` by the standard library. Using `unwrap_or_else` allows us to define some custom, non- `panic!` error handling. If the `Result` is an `Ok` value, this method’s behavior is similar to `unwrap`: it returns the inner value that `Ok` is wrapping. However, if the value is an `Err` value, this method calls the code in the _closure_, which is an anonymous function we define and pass as an argument to `unwrap_or_else`. We’ll cover closures in more detail in [Chapter 13](https://doc.rust-lang.org/book/ch13-00-functional-features.html). For now, you just need to know that `unwrap_or_else` will pass the inner value of the `Err`, which in this case is the static string `"not enough arguments"` that we added in Listing 12-9, to our closure in the argument `err` that appears between the vertical pipes. The code in the closure can then use the `err` value when it runs.
>  `unwrap_or_else` 方法允许我们自定义一些非 `panic!` 错误处理
>  如果 `Result` 是 `Ok`，则方法行为类似于 `unwrap`: 返回内部的值
>  如果是 `Err`，则方法调用闭包中的代码，闭包是一个匿名函数，作为参数传给 `unwrap_or_else`
>  闭包的写法如上所示，它捕获 `err` 参数，执行逻辑

We’ve added a new `use` line to bring `process` from the standard library into scope. The code in the closure that will be run in the error case is only two lines: we print the `err` value and then call `process::exit`. The `process::exit` function will stop the program immediately and return the number that was passed as the exit status code. This is similar to the `panic!` -based handling we used in Listing 12-8, but we no longer get all the extra output. Let’s try it:
>  `process:exit` 函数会停止程序，返回数字

```
$ cargo run
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.48s
     Running `target/debug/minigrep`
Problem parsing arguments: not enough arguments
```

Great! This output is much friendlier for our users.

### Extracting Logic from the `main` Function
Now that we’ve finished refactoring the configuration parsing, let’s turn to the program’s logic. As we stated in [“Separation of Concerns for Binary Projects”](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#separation-of-concerns-for-binary-projects), we’ll extract a function named `run` that will hold all the logic currently in the `main` function that isn’t involved with setting up configuration or handling errors. When we’re done, the `main` function will be concise and easy to verify by inspection, and we’ll be able to write tests for all the other logic.
>  `main` 负责设置配置或处理错误，逻辑交给 `run` 函数

Listing 12-11 shows the small, incremental improvement of extracting a `run` function.

Filename: src/main.rs

```rust
fn main() {
    // --snip--

    println!("Searching for {}", config.query);
    println!("In file {}", config.file_path);

    run(config);
}

fn run(config: Config) {
    let contents = fs::read_to_string(config.file_path)
        .expect("Should have been able to read the file");

    println!("With text:\n{contents}");
}

// --snip--
```

[Listing 12-11](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#listing-12-11): Extracting a `run` function containing the rest of the program logic

The `run` function now contains all the remaining logic from `main`, starting from reading the file. The `run` function takes the `Config` instance as an argument.

#### Returning Errors from the `run` Function
With the remaining program logic separated into the `run` function, we can improve the error handling, as we did with `Config::build` in Listing 12-9. Instead of allowing the program to panic by calling `expect`, the `run` function will return a `Result<T, E>` when something goes wrong. This will let us further consolidate the logic around handling errors into `main` in a user-friendly way. Listing 12-12 shows the changes we need to make to the signature and body of `run`.

Filename: src/main.rs

```rust
use std::error::Error;

// --snip--

fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let contents = fs::read_to_string(config.file_path)?;

    println!("With text:\n{contents}");

    Ok(())
}
```

[Listing 12-12](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#listing-12-12): Changing the `run` function to return `Result`

We’ve made three significant changes here. First, we changed the return type of the `run` function to `Result<(), Box<dyn Error>>`. This function previously returned the unit type, `()`, and we keep that as the value returned in the `Ok` case.

For the error type, we used the _trait object_ `Box<dyn Error>` (and we’ve brought `std::error::Error` into scope with a `use` statement at the top). We’ll cover trait objects in [Chapter 18](https://doc.rust-lang.org/book/ch18-00-oop.html). For now, just know that `Box<dyn Error>` means the function will return a type that implements the `Error` trait, but we don’t have to specify what particular type the return value will be. This gives us flexibility to return error values that may be of different types in different error cases. The `dyn` keyword is short for _dynamic_.

>  我们让 `run` 返回 `Result<(), Box<dyn Error>>`
>  该函数之前返回单元类型 `()`，我们让 `Ok` 绑定单元类型
>  错误类型使用了特质对象 `Box<dyn Error>`，它的意思是函数会返回一个实现了 `Error` 特质的类型，`dyn` 关键字表示动态的

Second, we’ve removed the call to `expect` in favor of the `?` operator, as we talked about in [Chapter 9](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#a-shortcut-for-propagating-errors-the--operator). Rather than `panic!` on an error, `?` will return the error value from the current function for the caller to handle.

Third, the `run` function now returns an `Ok` value in the success case. We’ve declared the `run` function’s success type as `()` in the signature, which means we need to wrap the unit type value in the `Ok` value. This `Ok(())` syntax might look a bit strange at first, but using `()` like this is the idiomatic way to indicate that we’re calling `run` for its side effects only; it doesn’t return a value we need.
>  我们用 `Ok(())` 表示 `Ok` 包含了一个单元类型值，这也是一种惯用做法，表示我们调用 `run` 仅仅是为了它的副作用，它并不返回我们需要使用的值

When you run this code, it will compile but will display a warning:

```
$ cargo run -- the poem.txt
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
warning: unused `Result` that must be used
  --> src/main.rs:19:5
   |
19 |     run(config);
   |     ^^^^^^^^^^^
   |
   = note: this `Result` may be an `Err` variant, which should be handled
   = note: `#[warn(unused_must_use)]` on by default
help: use `let _ = ...` to ignore the resulting value
   |
19 |     let _ = run(config);
   |     +++++++

warning: `minigrep` (bin "minigrep") generated 1 warning
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.71s
     Running `target/debug/minigrep the poem.txt`
Searching for the
In file poem.txt
With text:
I'm nobody! Who are you?
Are you nobody, too?
Then there's a pair of us - don't tell!
They'd banish us, you know.

How dreary to be somebody!
How public, like a frog
To tell your name the livelong day
To an admiring bog!
```

Rust tells us that our code ignored the `Result` value and the `Result` value might indicate that an error occurred. But we’re not checking to see whether or not there was an error, and the compiler reminds us that we probably meant to have some error-handling code here! Let’s rectify that problem now.

#### Handling Errors Returned from `run` in `main`
We’ll check for errors and handle them using a technique similar to one we used with `Config::build` in Listing 12-10, but with a slight difference:

Filename: src/main.rs

```rust
fn main() {
    // --snip--

    println!("Searching for {}", config.query);
    println!("In file {}", config.file_path);

    if let Err(e) = run(config) {
        println!("Application error: {e}");
        process::exit(1);
    }
}
```

We use `if let` rather than `unwrap_or_else` to check whether `run` returns an `Err` value and to call `process::exit(1)` if it does. The `run` function doesn’t return a value that we want to `unwrap` in the same way that `Config::build` returns the `Config` instance. Because `run` returns `()` in the success case, we only care about detecting an error, so we don’t need `unwrap_or_else` to return the unwrapped value, which would only be `()`.
>  我们使用 `if let` 而不是 `unwarp_or_else` 来检查是否 `run` 返回了 `Err` value
>  因为我们并不想 `unwarp` `run` 的返回值 (单元类型值 `()`)，因此我们不需要使用 `unwarp_or_else`

The bodies of the `if let` and the `unwrap_or_else` functions are the same in both cases: we print the error and exit.
>  我们在 `if let` 和 `unwarp_or_else` 中的函数体使用一样的逻辑: 打印错误并退出

### Splitting Code into a Library Crate
Our `minigrep` project is looking good so far! Now we’ll split the _src/main.rs_ file and put some code into the _src/lib.rs_ file. That way, we can test the code and have a _src/main.rs_ file with fewer responsibilities.

Let’s define the code responsible for searching text in _src/lib.rs_ rather than in _src/main.rs_, which will let us (or anyone else using our `minigrep` library) call the searching function from more contexts than our `minigrep` binary.

First, let’s define the `search` function signature in _src/lib.rs_ as shown in Listing 12-13, with a body that calls the `unimplemented!` macro. We’ll explain the signature in more detail when we fill in the implementation.

Filename: src/lib.rs

```rust
pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    unimplemented!();
}
```

[Listing 12-13](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#listing-12-13): Defining the `search` function in _src/lib.rs_

We’ve used the `pub` keyword on the function definition to designate `search` as part of our library crate’s public API. We now have a library crate that we can use from our binary crate and that we can test!

Now we need to bring the code defined in _src/lib.rs_ into the scope of the binary crate in _src/main.rs_ and call it, as shown in Listing 12-14.

Filename: src/main.rs

```rust
// --snip--
use minigrep::search;

fn main() {
    // --snip--
}

// --snip--

fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let contents = fs::read_to_string(config.file_path)?;

    for line in search(&config.query, &contents) {
        println!("{line}");
    }

    Ok(())
}
```

[Listing 12-14](https://doc.rust-lang.org/book/ch12-03-improving-error-handling-and-modularity.html#listing-12-14): Using the `minigrep` library crate’s `search` function in _src/main.rs_

We add a `use minigrep::search` line to bring the `search` function from the library crate into the binary crate’s scope. Then, in the `run` function, rather than printing out the contents of the file, we call the `search` function and pass the `config.query` value and `contents` as arguments. Then `run` will use a `for` loop to print each line returned from `search` that matched the query. 
>  我们在 `lib.rs` 中定义 `search` 函数，然后将它从 lib crate 带入 binary crate 的作用域
>  在 `run` 函数中，我们调用 `search`，传入参数，使用循环打印出 `search` 中返回的，contents 中匹配 query 的行

This is also a good time to remove the `println!` calls in the `main` function that displayed the query and the file path so that our program only prints the search results (if no errors occur).

Note that the search function will be collecting all the results into a vector it returns before any printing happens. This implementation could be slow to display results when searching large files because results aren’t printed as they’re found; we’ll discuss a possible way to fix this using iterators in Chapter 13.

Whew! That was a lot of work, but we’ve set ourselves up for success in the future. Now it’s much easier to handle errors, and we’ve made the code more modular. Almost all of our work will be done in _src/lib.rs_ from here on out.

Let’s take advantage of this newfound modularity by doing something that would have been difficult with the old code but is easy with the new code: we’ll write some tests!

## 12.4 Developing the Library’s Functionality with Test-Driven Development
Now that we have the search logic in _src/lib.rs_ separate from the `main` function, it’s much easier to write tests for the core functionality of our code. We can call functions directly with various arguments and check return values without having to call our binary from the command line.

In this section, we’ll add the searching logic to the `minigrep` program using the test-driven development (TDD) process with the following steps:

1. Write a test that fails and run it to make sure it fails for the reason you expect.
2. Write or modify just enough code to make the new test pass.
3. Refactor the code you just added or changed and make sure the tests continue to pass.
4. Repeat from step 1!

>  TDD 测试驱动开发:
>  1. 编写一个测试，该测试会失败，并且按照我们期望的方式失败
>  2. 修改部分代码，让恰好这个新测试可以通过
>  3. 重构刚才加入的代码，确保所有测试可以通过
>  4. 从 step 1 开始重复

Though it’s just one of many ways to write software, TDD can help drive code design. Writing the test before you write the code that makes the test pass helps to maintain high test coverage throughout the process.
>  TDD 可以帮助驱动代码设计，在我们编写让测试通过的代码之前编写测试，可以帮助我们在整个过程维护高的测试覆盖率

We’ll test-drive the implementation of the functionality that will actually do the searching for the query string in the file contents and produce a list of lines that match the query. We’ll add this functionality in a function called `search`.

### Writing a Failing Test
In _src/lib.rs_, we’ll add a `tests` module with a test function, as we did in [Chapter 11](https://doc.rust-lang.org/book/ch11-01-writing-tests.html#the-anatomy-of-a-test-function). The test function specifies the behavior we want the `search` function to have: it will take a query and the text to search, and it will return only the lines from the text that contain the query. Listing 12-15 shows this test.
>  我们在 `src/lib.rs` 添加 `tests` module 以及测试函数
>  测试函数指定了我们想要 `search` 拥有的行为

Filename: src/lib.rs

```rust
// --snip--

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_result() {
        let query = "duct";
        let contents = "\
Rust:
safe, fast, productive.
Pick three.";

        assert_eq!(vec!["safe, fast, productive."], search(query, contents));
    }
}
```

[Listing 12-15](https://doc.rust-lang.org/book/ch12-04-testing-the-librarys-functionality.html#listing-12-15): Creating a failing test for the `search` function for the functionality we wish we had

This test searches for the string `"duct"`. The text we’re searching is three lines, only one of which contains `"duct"` (note that the backslash after the opening double quote tells Rust not to put a newline character at the beginning of the contents of this string literal). We assert that the value returned from the `search` function contains only the line we expect.
>  `\` 告诉 Rust 不要将之后的换行放在字符串字面值中

If we run this test, it will currently fail because the `unimplemented!` macro panics with the message “not implemented”. In accordance with TDD principles, we’ll take a small step of adding just enough code to get the test to not panic when calling the function by defining the `search` function to always return an empty vector, as shown in Listing 12-16. Then the test should compile and fail because an empty vector doesn’t match a vector containing the line `"safe, fast, productive."`

Filename: src/lib.rs

```rust
pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    vec![]
}
```

[Listing 12-16](https://doc.rust-lang.org/book/ch12-04-testing-the-librarys-functionality.html#listing-12-16): Defining just enough of the `search` function so calling it won’t panic

Now let’s discuss why we need to define an explicit lifetime `'a` in the signature of `search` and use that lifetime with the `contents` argument and the return value. Recall in [Chapter 10](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html) that the lifetime parameters specify which argument lifetime is connected to the lifetime of the return value. In this case, we indicate that the returned vector should contain string slices that reference slices of the argument `contents` (rather than the argument `query`).
>  上述的生命周期表示了返回的 vector 包含的应该是引用了 `contents` 的 string slices，而不是 `query` 的

In other words, we tell Rust that the data returned by the `search` function will live as long as the data passed into the `search` function in the `contents` argument. This is important! The data referenced _by_ a slice needs to be valid for the reference to be valid; if the compiler assumes we’re making string slices of `query` rather than `contents`, it will do its safety checking incorrectly.

If we forget the lifetime annotations and try to compile this function, we’ll get this error:

```
$ cargo build
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
error[E0106]: missing lifetime specifier
 --> src/lib.rs:1:51
  |
1 | pub fn search(query: &str, contents: &str) -> Vec<&str> {
  |                      ----            ----         ^ expected named lifetime parameter
  |
  = help: this function's return type contains a borrowed value, but the signature does not say whether it is borrowed from `query` or `contents`
help: consider introducing a named lifetime parameter
  |
1 | pub fn search<'a>(query: &'a str, contents: &'a str) -> Vec<&'a str> {
  |              ++++         ++                 ++              ++

For more information about this error, try `rustc --explain E0106`.
error: could not compile `minigrep` (lib) due to 1 previous error
```

Rust can’t know which of the two parameters we need for the output, so we need to tell it explicitly. Note that the help text suggests specifying the same lifetime parameter for all the parameters and the output type, which is incorrect! Because `contents` is the parameter that contains all of our text and we want to return the parts of that text that match, we know `contents` is the only parameter that should be connected to the return value using the lifetime syntax.

Other programming languages don’t require you to connect arguments to return values in the signature, but this practice will get easier over time. You might want to compare this example with the examples in the [“Validating References with Lifetimes”](https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html#validating-references-with-lifetimes) section in Chapter 10.

### Writing Code to Pass the Test
Currently, our test is failing because we always return an empty vector. To fix that and implement `search`, our program needs to follow these steps:

1. Iterate through each line of the contents.
2. Check whether the line contains our query string.
3. If it does, add it to the list of values we’re returning.
4. If it doesn’t, do nothing.
5. Return the list of results that match.

Let’s work through each step, starting with iterating through lines.

#### Iterating Through Lines with the `lines` Method
Rust has a helpful method to handle line-by-line iteration of strings, conveniently named `lines`, that works as shown in Listing 12-17. Note that this won’t compile yet.

Filename: src/lib.rs

```rust
pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    for line in contents.lines() {
        // do something with line
    }
}
```

[Listing 12-17](https://doc.rust-lang.org/book/ch12-04-testing-the-librarys-functionality.html#listing-12-17): Iterating through each line in `contents`

The `lines` method returns an iterator. We’ll talk about iterators in depth in [Chapter 13](https://doc.rust-lang.org/book/ch13-02-iterators.html), but recall that you saw this way of using an iterator in [Listing 3-5](https://doc.rust-lang.org/book/ch03-05-control-flow.html#looping-through-a-collection-with-for), where we used a `for` loop with an iterator to run some code on each item in a collection.
>  `&str` 的 `lines` 方法返回一个迭代器，会逐行迭代内容

#### Searching Each Line for the Query
Next, we’ll check whether the current line contains our query string. Fortunately, strings have a helpful method named `contains` that does this for us! Add a call to the `contains` method in the `search` function, as shown in Listing 12-18. Note that this still won’t compile yet.

Filename: src/lib.rs

```rust
pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    for line in contents.lines() {
        if line.contains(query) {
            // do something with line
        }
    }
}
```

[Listing 12-18](https://doc.rust-lang.org/book/ch12-04-testing-the-librarys-functionality.html#listing-12-18): Adding functionality to see whether the line contains the string in `query`

At the moment, we’re building up functionality. To get the code to compile, we need to return a value from the body as we indicated we would in the function signature.
>  `contains` 方法判断字符串中是否包含子字符串

#### Storing Matching Lines
To finish this function, we need a way to store the matching lines that we want to return. For that, we can make a mutable vector before the `for` loop and call the `push` method to store a `line` in the vector. After the `for` loop, we return the vector, as shown in Listing 12-19.
>  我们构造一个可变 vector，使用 `push` 方法存入新数据

Filename: src/lib.rs

```rust
pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    let mut results = Vec::new();

    for line in contents.lines() {
        if line.contains(query) {
            results.push(line);
        }
    }

    results
}
```

[Listing 12-19](https://doc.rust-lang.org/book/ch12-04-testing-the-librarys-functionality.html#listing-12-19): Storing the lines that match so we can return them

Now the `search` function should return only the lines that contain `query`, and our test should pass. Let’s run the test:

```
$ cargo test
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 1.22s
     Running unittests src/lib.rs (target/debug/deps/minigrep-9cd200e5fac0fc94)

running 1 test
test tests::one_result ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/main.rs (target/debug/deps/minigrep-9cd200e5fac0fc94)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests minigrep

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Our test passed, so we know it works!

At this point, we could consider opportunities for refactoring the implementation of the search function while keeping the tests passing to maintain the same functionality. The code in the search function isn’t too bad, but it doesn’t take advantage of some useful features of iterators. We’ll return to this example in [Chapter 13](https://doc.rust-lang.org/book/ch13-02-iterators.html), where we’ll explore iterators in detail, and look at how to improve it.

Now the entire program should work! Let’s try it out, first with a word that should return exactly one line from the Emily Dickinson poem: _frog_.

```
$ cargo run -- frog poem.txt
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.38s
     Running `target/debug/minigrep frog poem.txt`
How public, like a frog
```

Cool! Now let’s try a word that will match multiple lines, like _body_:

```
$ cargo run -- body poem.txt
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.0s
     Running `target/debug/minigrep body poem.txt`
I'm nobody! Who are you?
Are you nobody, too?
How dreary to be somebody!
```

And finally, let’s make sure that we don’t get any lines when we search for a word that isn’t anywhere in the poem, such as _monomorphization_:

```
$ cargo run -- monomorphization poem.txt
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.0s
     Running `target/debug/minigrep monomorphization poem.txt`
```

Excellent! We’ve built our own mini version of a classic tool and learned a lot about how to structure applications. We’ve also learned a bit about file input and output, lifetimes, testing, and command line parsing.

To round out this project, we’ll briefly demonstrate how to work with environment variables and how to print to standard error, both of which are useful when you’re writing command line programs.

## 12.5 Working with Environment Variables
We’ll improve the `minigrep` binary by adding an extra feature: an option for case-insensitive searching that the user can turn on via an environment variable. We could make this feature a command line option and require that users enter it each time they want it to apply, but by instead making it an environment variable, we allow our users to set the environment variable once and have all their searches be case insensitive in that terminal session.

### Writing a Failing Test for the Case-Insensitive `search` Function
We first add a new `search_case_insensitive` function to the `minigrep` library that will be called when the environment variable has a value. We’ll continue to follow the TDD process, so the first step is again to write a failing test. We’ll add a new test for the new `search_case_insensitive` function and rename our old test from `one_result` to `case_sensitive` to clarify the differences between the two tests, as shown in Listing 12-20.

Filename: src/lib.rs

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_sensitive() {
        let query = "duct";
        let contents = "\
Rust:
safe, fast, productive.
Pick three.
Duct tape.";

        assert_eq!(vec!["safe, fast, productive."], search(query, contents));
    }

    #[test]
    fn case_insensitive() {
        let query = "rUsT";
        let contents = "\
Rust:
safe, fast, productive.
Pick three.
Trust me.";

        assert_eq!(
            vec!["Rust:", "Trust me."],
            search_case_insensitive(query, contents)
        );
    }
}
```

[Listing 12-20](https://doc.rust-lang.org/book/ch12-05-working-with-environment-variables.html#listing-12-20): Adding a new failing test for the case-insensitive function we’re about to add

Note that we’ve edited the old test’s `contents` too. We’ve added a new line with the text `"Duct tape."` using a capital _D_ that shouldn’t match the query `"duct"` when we’re searching in a case-sensitive manner. Changing the old test in this way helps ensure that we don’t accidentally break the case-sensitive search functionality that we’ve already implemented. This test should pass now and should continue to pass as we work on the case-insensitive search.

The new test for the case-_insensitive_ search uses `"rUsT"` as its query. In the `search_case_insensitive` function we’re about to add, the query `"rUsT"` should match the line containing `"Rust:"` with a capital _R_ and match the line `"Trust me."` even though both have different casing from the query. This is our failing test, and it will fail to compile because we haven’t yet defined the `search_case_insensitive` function. Feel free to add a skeleton implementation that always returns an empty vector, similar to the way we did for the `search` function in Listing 12-16 to see the test compile and fail.

### Implementing the `search_case_insensitive` Function
The `search_case_insensitive` function, shown in Listing 12-21, will be almost the same as the `search` function. The only difference is that we’ll lowercase the `query` and each `line` so that whatever the case of the input arguments, they’ll be the same case when we check whether the line contains the query.

Filename: src/lib.rs

```rust
pub fn search_case_insensitive<'a>(
    query: &str,
    contents: &'a str,
) -> Vec<&'a str> {
    let query = query.to_lowercase();
    let mut results = Vec::new();

    for line in contents.lines() {
        if line.to_lowercase().contains(&query) {
            results.push(line);
        }
    }

    results
}
```

[Listing 12-21](https://doc.rust-lang.org/book/ch12-05-working-with-environment-variables.html#listing-12-21): Defining the `search_case_insensitive` function to lowercase the query and the line before comparing them

First we lowercase the `query` string and store it in a new variable with the same name, shadowing the original `query`. Calling `to_lowercase` on the query is necessary so that no matter whether the user’s query is `"rust"`, `"RUST"`, `"Rust"`, or `"``rUsT``"`, we’ll treat the query as if it were `"rust"` and be insensitive to the case. While `to_lowercase` will handle basic Unicode, it won’t be 100 percent accurate. If we were writing a real application, we’d want to do a bit more work here, but this section is about environment variables, not Unicode, so we’ll leave it at that here.

Note that `query` is now a `String` rather than a string slice because calling `to_lowercase` creates new data rather than referencing existing data. Say the query is `"rUsT"`, as an example: that string slice doesn’t contain a lowercase `u` or `t` for us to use, so we have to allocate a new `String` containing `"rust"`. When we pass `query` as an argument to the `contains` method now, we need to add an ampersand because the signature of `contains` is defined to take a string slice.

Next, we add a call to `to_lowercase` on each `line` to lowercase all characters. Now that we’ve converted `line` and `query` to lowercase, we’ll find matches no matter what the case of the query is.

Let’s see if this implementation passes the tests:

```
$ cargo test
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 1.33s
     Running unittests src/lib.rs (target/debug/deps/minigrep-9cd200e5fac0fc94)

running 2 tests
test tests::case_insensitive ... ok
test tests::case_sensitive ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/main.rs (target/debug/deps/minigrep-9cd200e5fac0fc94)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests minigrep

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Great! They passed. Now, let’s call the new `search_case_insensitive` function from the `run` function. First we’ll add a configuration option to the `Config` struct to switch between case-sensitive and case-insensitive search. Adding this field will cause compiler errors because we aren’t initializing this field anywhere yet:

Filename: src/main.rs

```rust
pub struct Config {
    pub query: String,
    pub file_path: String,
    pub ignore_case: bool,
}
```

We added the `ignore_case` field that holds a Boolean. Next, we need the `run` function to check the `ignore_case` field’s value and use that to decide whether to call the `search` function or the `search_case_insensitive` function, as shown in Listing 12-22. This still won’t compile yet.

Filename: src/main.rs

```rust
use minigrep::{search, search_case_insensitive};

// --snip--

fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let contents = fs::read_to_string(config.file_path)?;

    let results = if config.ignore_case {
        search_case_insensitive(&config.query, &contents)
    } else {
        search(&config.query, &contents)
    };

    for line in results {
        println!("{line}");
    }

    Ok(())
}
```

[Listing 12-22](https://doc.rust-lang.org/book/ch12-05-working-with-environment-variables.html#listing-12-22): Calling either `search` or `search_case_insensitive` based on the value in `config.ignore_case`

Finally, we need to check for the environment variable. The functions for working with environment variables are in the `env` module in the standard library, which is already in scope at the top of _src/main.rs_. We’ll use the `var` function from the `env` module to check to see if any value has been set for an environment variable named `IGNORE_CASE`, as shown in Listing 12-23.

Filename: src/main.rs

```rust
impl Config {
    fn build(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 3 {
            return Err("not enough arguments");
        }

        let query = args[1].clone();
        let file_path = args[2].clone();

        let ignore_case = env::var("IGNORE_CASE").is_ok();

        Ok(Config {
            query,
            file_path,
            ignore_case,
        })
    }
}
```

[Listing 12-23](https://doc.rust-lang.org/book/ch12-05-working-with-environment-variables.html#listing-12-23): Checking for any value in an environment variable named `IGNORE_CASE`

Here, we create a new variable, `ignore_case`. To set its value, we call the `env::var` function and pass it the name of the `IGNORE_CASE` environment variable. The `env::var` function returns a `Result` that will be the successful `Ok` variant that contains the value of the environment variable if the environment variable is set to any value. It will return the `Err` variant if the environment variable is not set.

We’re using the `is_ok` method on the `Result` to check whether the environment variable is set, which means the program should do a case-insensitive search. If the `IGNORE_CASE` environment variable isn’t set to anything, `is_ok` will return `false` and the program will perform a case-sensitive search. We don’t care about the _value_ of the environment variable, just whether it’s set or unset, so we’re checking `is_ok` rather than using `unwrap`, `expect`, or any of the other methods we’ve seen on `Result`.

We pass the value in the `ignore_case` variable to the `Config` instance so the `run` function can read that value and decide whether to call `search_case_insensitive` or `search`, as we implemented in Listing 12-22.

Let’s give it a try! First we’ll run our program without the environment variable set and with the query `to`, which should match any line that contains the word _to_ in all lowercase:

```
$ cargo run -- to poem.txt
   Compiling minigrep v0.1.0 (file:///projects/minigrep)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.0s
     Running `target/debug/minigrep to poem.txt`
Are you nobody, too?
How dreary to be somebody!
```

Looks like that still works! Now let’s run the program with `IGNORE_CASE` set to `1` but with the same query _to_:

```
$ IGNORE_CASE=1 cargo run -- to poem.txt
```

If you’re using PowerShell, you will need to set the environment variable and run the program as separate commands:

```
PS> $Env:IGNORE_CASE=1; cargo run -- to poem.txt
```

This will make `IGNORE_CASE` persist for the remainder of your shell session. It can be unset with the `Remove-Item` cmdlet:

```
PS> Remove-Item Env:IGNORE_CASE 
```

We should get lines that contain _to_ that might have uppercase letters:

```
Are you nobody, too?
How dreary to be somebody!
To tell your name the livelong day
To an admiring bog!
```

Excellent, we also got lines containing _To_! Our `minigrep` program can now do case-insensitive searching controlled by an environment variable. Now you know how to manage options set using either command line arguments or environment variables.

Some programs allow arguments _and_ environment variables for the same configuration. In those cases, the programs decide that one or the other takes precedence. For another exercise on your own, try controlling case sensitivity through either a command line argument or an environment variable. Decide whether the command line argument or the environment variable should take precedence if the program is run with one set to case sensitive and one set to ignore case.

The `std::env` module contains many more useful features for dealing with environment variables: check out its documentation to see what is available.

## 12.6 Writing Error Messages to Standard Error Instead of Standard Output
At the moment, we’re writing all of our output to the terminal using the `println!` macro. In most terminals, there are two kinds of output: _standard output_ (`stdout`) for general information and _standard error_ (`stderr`) for error messages. This distinction enables users to choose to direct the successful output of a program to a file but still print error messages to the screen.

The `println!` macro is only capable of printing to standard output, so we have to use something else to print to standard error.
>  `println!` marco 只能打印到标准输出

### Checking Where Errors Are Written
First let’s observe how the content printed by `minigrep` is currently being written to standard output, including any error messages we want to write to standard error instead. We’ll do that by redirecting the standard output stream to a file while intentionally causing an error. We won’t redirect the standard error stream, so any content sent to standard error will continue to display on the screen.

Command line programs are expected to send error messages to the standard error stream so we can still see error messages on the screen even if we redirect the standard output stream to a file. Our program is not currently well behaved: we’re about to see that it saves the error message output to a file instead!
>  命令行程序通常要将错误信息发送到标准错误流，这使得我们在把标准输出流重定向到文件中时，仍然可以看到屏幕上的错误信息

To demonstrate this behavior, we’ll run the program with `>` and the file path, _output.txt_, that we want to redirect the standard output stream to. We won’t pass any arguments, which should cause an error:

```
$ cargo run > output.txt
```

The `>` syntax tells the shell to write the contents of standard output to _output.txt_ instead of the screen. We didn’t see the error message we were expecting printed to the screen, so that means it must have ended up in the file. This is what _output.txt_ contains:

```
Problem parsing arguments: not enough arguments 
```

Yup, our error message is being printed to standard output. It’s much more useful for error messages like this to be printed to standard error so only data from a successful run ends up in the file. We’ll change that.

### Printing Errors to Standard Error
We’ll use the code in Listing 12-24 to change how error messages are printed. Because of the refactoring we did earlier in this chapter, all the code that prints error messages is in one function, `main`. The standard library provides the `eprintln!` macro that prints to the standard error stream, so let’s change the two places we were calling `println!` to print errors to use `eprintln!` instead.
>  `eprintln!` macro 打印到标准错误流

Filename: src/main.rs

```rust
fn main() {
    let args: Vec<String> = env::args().collect();

    let config = Config::build(&args).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {err}");
        process::exit(1);
    });

    if let Err(e) = run(config) {
        eprintln!("Application error: {e}");
        process::exit(1);
    }
}
```

[Listing 12-24](https://doc.rust-lang.org/book/ch12-06-writing-to-stderr-instead-of-stdout.html#listing-12-24): Writing error messages to standard error instead of standard output using `eprintln!`

Let’s now run the program again in the same way, without any arguments and redirecting standard output with `>`:

```
$ cargo run > output.txt
Problem parsing arguments: not enough arguments
```

Now we see the error onscreen and _output.txt_ contains nothing, which is the behavior we expect of command line programs.

Let’s run the program again with arguments that don’t cause an error but still redirect standard output to a file, like so:

`$ cargo run -- to poem.txt > output.txt`

We won’t see any output to the terminal, and _output.txt_ will contain our results:

Filename: output.txt

```
Are you nobody, too?
How dreary to be somebody!
```

This demonstrates that we’re now using standard output for successful output and standard error for error output as appropriate.

## Summary
This chapter recapped some of the major concepts you’ve learned so far and covered how to perform common I/O operations in Rust. By using command line arguments, files, environment variables, and the `eprintln!` macro for printing errors, you’re now prepared to write command line applications. Combined with the concepts in previous chapters, your code will be well organized, store data effectively in the appropriate data structures, handle errors nicely, and be well tested.

Next, we’ll explore some Rust features that were influenced by functional languages: closures and iterators.
