# Part 1 Basic Concepts
## CH1 How to Write a Simple Makefile
The `make` program is intended to automate the mundane aspects of transforming source code into an executable. The advantages of `make` over scripts is that you can specify the relationships between the elements of your program to `make`, and it knows through these relationships and timestamps exactly what steps need to be redone to produce the desired program each time. Using this information, `make` can also optimize the build process avoiding unnecessary steps.

`make` defines a language for describing the relationships between source code, intermediate files, and executables. ( `make` 定义了一种描述源文件、中间文件、可执行文件之间关系的语言 )

The specification that `make` uses is generally saved in a file named `makefile`. Here is a `makefile` to build the traditional “Hello, World” program:
```makefile
hello: hello.c
        gcc hello.c -o hello
```
To build the program execute `make` by typing:
```
$ make
```
at the command prompt of your favorite shell. This will cause the `make` program to read the `makefile` and build the first target it finds there:
```
$ make
gcc hello.c -o hello
```
( `make` 将自己读取 `makefile` ，然后构建它找到的第一个目标 )

If a target is included as a command-line argument, that target is updated. If no command-line targets are given, then the first target in the file is used, called the `default goal`. ( 作为参数指定的目标会被构建，没有传入参数则自动构建第一个目标，即默认目标 )

Typically the default goal in most `makefiles` is to build a program. This usually involves many steps. Often the source code for the program is incomplete and the source must be generated using utilities such as `flex` or `bison`. Next the source is compiled into binary object files (`.o` files for C/C++, `.class` files for Java, etc.). Then, for C/C++, the object files are bound together by a linker (usually invoked through the compiler, `gcc`) to form an executable program.

Modifying any of the source files and reinvoking `make` will cause some, but usually not all, of these commands to be repeated so the source code changes are properly incorporated into the executable.

The specification file, or `makefile`, describes the relationshipbetween the source, intermediate, and executable program files so that make can perform the minimum amount of work necessary to update the executable.

So the principle value of `make` comes from its ability to perform the complex series of commands necessary to build an application and to optimize these operations when possible to reduce the time taken by the edit-compile-debug cycle. ( `make` 可以执行构建一个程序所需的一系列命令，并在源码更新时采用尽可能少的工作重新构建程序 ) Furthermore, `make` is flexible enough to be used anywhere one kind of file depends on another from traditional programming in C/C++ to Java, TEX, database management, and more.
### 1.1 Targets and Prerequisites
Essentially a `makefile` contains a set of rules used to build an application. The first rule seen by `make` is used as the `default rule`.  ( `makefile` 包含构建程序的一系列规则，`make` 所看到的第一条规则会被用作默认规则 ) A rule consists of three parts: the target, its prerequisites, and the command(s) to perform:
```
target: prereq_1 prereq_2
        commands
```
( 规则包含了目标、前提、命令三部分 )

The `target` is the file or thing that must be made. The `prerequisites or dependents` are those files that must exist before the target can be successfully created. And the `commands` are those shell commands that will create the target from the prerequisites. ( `commands` 即使用前提构建目标的shell命令 )

The command script usually appears on the following lines and is preceded by a tab character.

When `make` is asked to evaluate a rule, it begins by finding the files indicated by the prerequisites and target. If any of the prerequisites has an associated rule, `make` attempts to update those first. ( 构建目标时，如果任意前提有关联的规则，`make` 会优先评估这些规则，以更新前提 ) Next, the target file is considered. If any prerequisite is newer than the target, the target is remade by executing the commands. ( 如果任意前提比目标更新，`make` 会重新构建目标 ) Each command line is passed to the shell and is executed in its own subshell. If any of the commands generates an error, the building of the target is terminated and `make` exits. ( `make` 会将命令传递给subshell以执行 ) One file is considered newer than another if it has been modified more recently.

The *makefile* for this program is also quite simple:
```
count_words: count_words.o lexer.o -lfl 
    gcc count_words.o lexer.o -lfl -o count_words

count_words.o: count_words.c 
    gcc -c count_words.c

lexer.o: lexer.c 
    gcc -c lexer.c

lexer.c: lexer.l
    flex -t lexer.l > lexer.c
```
When this *makefile* is executed for the first time, we see:
```
$ make
gcc -c count_words.c 
flex -t lexer.l > lexer.c 
gcc -c lexer.c
gcc count_words.o lexer.o -lfl -o count_words
```

As you look at the `makefile` and sample execution, you may notice that the order in which commands are executed by make are nearly the opposite to the order they occur in the `makefile`. This *top-down* style is common in `makefiles`. Usually the most general form of target is specified first in the makefile and the details are left for later. ( 通常先指定最广泛的目标，细节留后 ) The `make` program supports this style in many ways. Chief among these is make’s twophase execution model and recursive variables
### 1.2 Dependency Checking
First `make` notices that the command line contains no targets so it decides to make the default goal, `count_words`. It checks for prerequisites and sees three: `count_words.o`, `lexer.o`, and `-lfl`. `make` now considers how to build `count_words.o` and sees a rule for it. Again, it checks the prerequisites, notices that `count_words.c` has no rules but that the file exists, so `make` executes the commands to transform `count_words.c` into `count_words.o` by executing the command:
```
gcc -c count_word.c
```
This “chaining” of targets to prerequisites to targets to prerequisites is typical of how make analyzes a makefile to decide the commands to be performed

Finally, make examines `-lfl`. The `-l` option to `gcc` indicates a system library that must be linked into the application. The actual library name indicated by “fl” is `libfl.a`. GNU `make` includes special support for this syntax. When a prerequisite of the form `l<NAME>` is seen, `make` searches for a file of the form `libNAME.so`; if no match is found, it then searches for `libNAME.a`. Here `make` finds `/usr/lib/libfl.a` and proceeds with the final action, linking. ( 前提中的 `-lfl` 意为需要构建目标需要动态链接库文件 `libfl.so` 或静态链接库文件 `libfl.a` ，`make` 会在 `/usr/lib/` 中寻找这些文件 )
### 1.3 Minimizing Rebuilds
After editing this file we need to rebuild the application to test our fix:
```
$ make
flex -t lexer.l > lexer.c 
gcc -c lexer.c
gcc count_words.o lexer.o -lfl -ocount_words
```

Notice this time the file `count_words.c` was not recompiled. When `make` analyzed the rule, it discovered that `count_words.o` existed and was newer than its prerequisite `count_words.c` so no action was necessary to bring the file upto date. ( 目标存在，并且比前提更新，说明前提没有变化，因此不需要重新构建目标 ) While analyzing `lexer.c`, however, `make` saw that the prerequisite l`exer.l` was newer than its target `lexer.c` so `make` must update `lexer.c`. ( 前提更新了，目标也要重新构建以更新 ) This, in turn, caused the update of `lexer.o` and then `count_words`. 
### 1.4 Invoking make
The previous examples assume that:
• All the project source code and the make description file are stored in a single directory.
• The make description file is called `makefile`, `Makefile`, or `GNUMakefile`.
• The makefile resides in the user’s current directory when executing the `make` command. 

When `make` is invoked under these conditions, it automatically creates the first target it sees. To update a different target (or to update more than one target) include the target name on the command line:
```
$ make lexer.c
```
( make 默认构建它看到的第一个目标，可以通过参数指定它需要构建/更新的目标 )

If the target you specify is already up to date, make will say so and immediately exit, doing nothing else:
```
$ make lexer.c
make: `lexer.c` is up to date.
```

If you specify a target that is not in the makefile and for which there is no implicit rule (discussed in Chapter 2), make will respond with:
```
$ make non-existent-target
make: *** No rule to make target `non-existent-target'. Stop.
```
make has many command-line options. One of the most useful is `--just-print` (or `-n`) which tells make to display the commands it would execute for a particular target without actually executing them. This is particularly valuable while writing makefiles. It is also possible to set almost any makefile variable on the command line to override the default value or the value set in the makefile. ( makefile变量可以在命令行中对其修改或覆写 )
### 1.4 Basic Makefile Syntax
Makefiles are usually structured top-down so that the most general target, often called `all`, is updated by default. ( 自顶向下编写makefile以保证 make 以最广泛的目标作为默认目标 ) More and more detailed targets follow with targets for program maintenance, ( 用于程序维护的目标写在最后 ) such as a `clean` target to delete unwanted temporary files, coming last. As you can guess from these target names, targets do not have to be actual files, any name will do. ( 目标不必要是文件 )

The more complete (but still not quite complete) form of a rule is:
```
target_1 target_2 target_3 : prerequisite_1 prerequisite_2
    command_1
    command_2
    command_3
```
One or more targets appear to the left of the colon and zero or more prerequisites can appear to the right of the colon. ( 目标数大于等于1，前提数大于等于0 ) If no prerequisites are listed to the right, then only the target(s) that do not exist are updated. ( 没有前提时，仅构建尚不存在的目标 ) The set of commands executed to update a target are sometimes called the command script, but most often just the commands.

Each command *must* begin with a tab character. This (obscure) syntax tells make that the characters that follow the tab are to be passed to a subshell for execution. If you accidentally insert a tab as the first character of a noncommand line, make will interpret the following text as a command under most circumstances. ( `make` 会将tab字符后的字符都解释为命令，传递给subshell ) If you’re lucky and your errant tab character is recognized as a syntax error you will receive the message:
```
$ make
Makefile:6:*** commands commence before first target. Stop.
```

The comment character for make is the hash or pound sign, `#`. All text from the pound sign to the end of line is ignored. Comments can be indented and leading whitespace is ignored.

The comment character `#` does not introduce a make comment in the text of commands. The entire line, including the # and subsequent characters, is passed to the shell for execution. How it is handled there depends on your shell. ( 在makefile的命令line中写注释是无效的 )

Long lines can be continued using the standard Unix escape character backslash (`\`) ( 行继续符号 ). It is common for commands to be continued in this way. It is also common for lists of prerequisites to be continued with backslash.
## CH2 Rules
Since the target of one rule can be referenced as a prerequisite in another rule, the set of targets and prerequisites form a chain or graph of *dependencies* (short for “dependency graph”). Building and processing this dependency graph to update the requested target is what `make` is all about. ( `make` 的职责就是构建依赖图以更新目标 )

there are a number of different kinds of rules. 
*Explicit rules*, like the ones in the previous chapter, indicate a specific target to be updated if it is out of date with respect to any of its prerequisites. This is the most common type of rule you will be writing. 
*Pattern rules* use wildcards instead of explicit filenames. This allows make to apply the rule any time a target file matching the pattern needs to updated. *Implicit rules* are either pattern rules or suffix rules found in the rules database built-in to make. ( 在内建规则数据库中找到的规则 ) Having a built-in database of rules makes writing makefiles easier since for many common tasks make already knows the file types, suffixes, and programs for updating targets. 
*Static pattern rules* are like regular pattern rules except they apply only to a specific list of target files.

GNU make can be used as a “dropin” replacement for many other versions of make and includes several features specifically for compatibility. *Suffix rules* were make’s original means for writing general rules. GNU make includes support for suffix rules, but they are considered obsolete having been replaced by pattern rules that are clearer and more general. ( GNU make中 suffix rules已经被pattern rule替代 )
### 2.1 Explicit Rules
A rule can have more than one target. This means that each target has the same set of prerequisites as the others.For instance: 
```
vpath.o variable.o: make.h config.h getopt.h gettext.h dep.h
``` 
This indicates that both vpath.o and variable.o depend on the same set of C header files. This line has the same effect as:
```
vpath.o: make.h config.h getopt.h gettext.h dep.h 
variable.o: make.h config.h getopt.h gettext.h dep.h
```
The two targets are handled independently.

A rule does not have to be defined “all at once.” Each time make sees a target file it adds the target and prerequisites to the dependency graph. If a target has already been seen and exists in the graph, any additional prerequisites are appended to the target file entry in make’s dependency graph. In the simple case, this is useful for breaking long lines naturally to improve the readability of the makefile:
```
vpath.o: vpath.c make.h config.h getopt.h gettext.h dep.h 
vpath.o: filedef.h hash.h job.h commands.h variable.h vpath.h
```

In more complex cases, the prerequisite list can be composed of files that are managed very differently:
```
# Make sure lexer.c is created before vpath.c is compiled.
vpath.o: lexer.c 
...
# Compile vpath.c with special flags. 
vpath.o: vpath.c $(COMPILE.c) $(RULE_FLAGS) $(OUTPUT_OPTION) $< 
...
# Include dependencies generated by a program. 
include auto-generated-dependencies.d
```
This rule might be placed near the rules for managing lexer.c so developers are reminded of this subtle relationship ( `vpath.o: lexer.c` 揭示了二者的依赖关系，一般和管理 `lexer.c` 的规则相接近 )

Later, the compilation rule for `vpath.o` is placed among other compilation rules. The command for this rule uses three `make` variables. for now you just need to know that a variable is either a dollar sign followed by a single character or a dollar sign followed by a word in parentheses. ( `make` 变量的形式是 `$()`  )

Finally, the `.o/.h` dependencies are included in the makefile from a separate file managed by an external program.
#### 2.1.1 Wildcards
A makefile often contains long lists of files. To simplify this process make supports wildcards (also known as globbing). make’s wildcards are identical to the Bourne shell’s: `~, *, ?, [...],` and `[^...]`. For instance, `*.*` expands to all the files containing a period. A question mark represents any single character, and `[...]` represents a character class. To select the “opposite” (negated) character class use `[^...]`.

A tilde(`~`) followed by a user name represents that user’s home directory.

Wildcards are automatically expanded by make whenever a wildcard appears in a target, prerequisite, or command script context. Wildcards can be very useful for creating more adaptable makefiles. For instance, instead of listing all the files in a program explicitly, you can use wildcards:`*`
```
prog: *.c
    $(CC) -o $@ $^
```

It is easy to misuse them as the following example shows:
```
*.o: constants.h
```
The intent is clear: all object files depend on the header file constants.h, but consider how this expands on a clean directory without any object files:
```
: constants.h
```
This is a legal make expression and will not produce an error by itself, but it will also not provide the dependency the user wants. The proper way to implement this rule is to perform a wildcard on the source files (since they are always present) and transform that into a list of object files. We will cover this technique when we discuss make functions in Chapter 4.

When mp_make expands a wildcard (or indeed when mp_make looks for any file), it reads and caches the directory contents. This caching improves mp_make’s performance considerably. However, once mp_make has read and cached the directory contents, mp_make will not “see” any changes made to the directory. This can be a mysterious source of errors in a mp_makefile. The issue can sometimes be resolved by using a sub-shell and globbing (e.g., shell wildcards) rather than mp_make’s own wildcards, but occasionally, this is not possible and we must resort to bizarre hacks.

Finally, it is worth noting that wildcard expansion is performed by `make` when the pattern appears as a target or prerequisite. However, when the pattern appears in a command, the expansion is performed by the subshell. ( 在目标或前提中的通配符是由 `make` 展开的，在命令中的通配符是由subshell展开的 ) This can occasionally be important because make will expand the wildcards immediately upon reading the makefile, but the shell will expand the wildcards in commands much later when the command is executed. When a lot of complex file manipulation is being done, the two wildcard expansions can be quite different. ( 因此目标或前提中的通配符与命令中的通配符被展开的时间也是不同的 )
#### 2.1.2 Phony Targets
it is often useful for a target to be just a label representing a command script. For instance, earlier we noted that a standard first target in many makefiles is called `all`. Targets that do not represent files are known as phony targets. Another standard phony target is `clean`: ( 两个较常用的标准假目标是 `all`和 `clean` )
```
clean:
    rm -f *.o lexer.c
```
Normally, phony targets will always be executed because the commands associated with the rule do not create the target name. ( 假目标一般总会执行，因为目标名称一般不会被命令创建，`make` 每次遇到假目标便会判断需要执行以创建假目标 )

It is important to note that make cannot distinguish between a file target and phony target. If by chance the name of a phony target exists as a file, make will associate the file with the phony target name in its dependency graph. If, for example, the file clean happened to be created running make clean would yield the confusing message:
```
$ make clean
make: `clean' is up to date.
```
Since most phony targets do not have prerequisites, the clean target would always be considered up to date and would never execute.

To avoid this problem, GNU make includes a special target, .PHONY, to tell make that a target is not a real file. Any target can be declared phony by including it as a prerequisite of `.PHONY`: ( 显式声明假目标，防止它和同名文件混淆 )
```
.PHONY: clean
clean:
    rm -f *.o lexer.c
```
Now make will always execute the commands associated with clean even if a file named clean exists. In addition to marking a target as always out of date, specifying that a target is phony tells make that this file does not follow the normal rules for making a target file from a source file. Therefore, make can optimize its normal rule search to improve performance. ( 声明的好处有两点，一点是不和文件混淆，`make` 每次都会执行假目标相关命令，另一点是让该规则不是常规规则，`make` 在处理其他规则时，不会将这条规则纳入搜索范围 )

It rarely makes sense to use a phony target as a prerequisite of a real file since the phony is always out of date and will always cause the target file to be remade. ( 将假目标放在前提时，由于假目标每次都会重新执行，相关的目标文件每次也会被重新创建 )
However, it is often useful to give phony targets prerequisites. For instance, the all target is usually given the list of programs to be built:
```
.PHONY: all
all: bash bashbug
```
Here the `all` target creates the `bash` shell program and the `bashbug` error reporting tool. ( `all` 一定会被执行，因此它的前提一定需要被创建，利用这一点，可以将所需要构建的程序放在 `all` 的前提中 )

Phony targets can also be thought of as shell scripts embedded in a makefile. Making a phony target a prerequisite of another target will invoke the phony target script before making the actual target. ( 假目标用在前提还可以用于在构建过程中嵌入shell脚本的执行 ) Suppose we are tight on disk space and before executing a disk-intensive task we want to display available disk space. We could write:
```
.PHONY: make-documentation 
make-documentation:
    df -k . | awk 'NR = = 2 { printf( "%d available\n", $$4 ) }'
    javadoc ...
```

There are a number of other good uses for phony targets. The output of make can be confusing to read and debug. There are several reasons for this: makefiles are written top-down but the commands are executed by make bottom-up; also, there is no indication which rule is currently being evaluated. The output of make can be made much easier to read if major targets are commented in the make output. Phony targets are a useful way to accomplish this. Here is an example taken from the `bash` makefile: ( 假目标也可以用于提高 `make` 的可读性 )
```
$(Program): build_msg $(OBJECTS) $(BUILTINS_DEP) $(LIBDEP) 
    $(RM) $@
    $(CC) $(LDFLAGS) -o $(Program) $(OBJECTS) $(LIBS) 
    ls -l $(Program)
    size $(Program)

.PHONY: build_msg 
build_msg:
    @printf "#\n# Building $(Program)\n#\n"
```
Because the printf is in a phony target, the message is printed immediately before any prerequisites are updated. If the build message were instead placed as the first command of the $(Program) command script, then it would be executed after all compilation and dependency generation. ( `build_msg` 假目标允许我们在构建任何前提之前打印信息，而命令部分的 `printf` 则只会在所有前提构建完毕之后才会执行 ) It is important to note that because phony targets are always out of date, the phony `build_msg` target causes $(Program) to be regenerated even when it is not out of date. In this case, it seems a reasonable choice since most of the computation is performed while compiling the object files so only the final link will always be performed. ( 但要注意的是以假目标为前提的目标会每次都被重新构建 )

Phony targets can also be used to improve the “user interface” of a makefile. Often targets are complex strings containing directory path elements, additional filename components (such as version numbers) and standard suffixes. This can make specifying a target filename on the command line a challenge. The problem can be avoided by adding a simple phony target whose prerequisite is the actual target file.

By convention there are a set of more or less standard phony targets that many makefiles include.
![[Managing Projects with GNU Make-Table2.1.png]]
The target TAGS is not really a phony since the output of the `ctags` and `etags` programs is a file named TAGS. It is included here because it is the only standard nonphony target we know of.\
#### 2.1.3 Empty Targets
Phony targets are always out of date, so they always execute and they always cause their dependent (the target associated with the prerequisite) to be remade.

But suppose we have some command, with no output file, that needs to be performed only occasionally and we don’t want our dependents updated? For this, we can make a rule whose target is an empty file (sometimes referred to as a cookie):
```
prog: size prog.o
    $(CC) $(LDFLAGS) -o $@ $^

size: prog.o 
    size $^
    touch size
```
Notice that the size rule uses touch to create an empty file named size after it completes. This empty file is used for its timestamp so that make will execute the size rule only when prog.o has been updated. Furthermore, the size prerequisite to prog will not force an update of prog unless its object file is also newer. ( `size` 规则只会在 `prog.o` 更新后才执行，`size` 文件的时间戳和 `prog.o` 的时间戳是近一致的，它不会导致 `prog` 每次都需要重新构建，仅在 `prog.o` 更新后才重新构建 )


Empty files are particularly useful when combined with the automatic variable `$?`. Within the command script part of a rule, make defines the variable `$?` to be the set of prerequisites that are newer than the target. ( `$?` 定义为比目标更新的前提 ) Here is a rule to print all the files that have changed since the last time make print was executed:
```
print: *.[hc] 
    lpr $? 
    touch $@
```
Generally, empty files can be used to mark the last time a particular event has taken place. ( 空文件一般用作时间戳 )
### 2.2 Variables
The simplest ones have the syntax:
```
$(variable-name)
```
This indicates that we want to expand the variable whose name is variable-name.
Variables can contain almost any text, and variable names can contain most characters including punctuation. ( 变量名可以包含标点符号 ) The variable containing the C compile command is `COMPILE.c`, for example. In general, a variable name must be surrounded by `$( )` or `${ }` to be recognized by make. As a special case, a single character variable name does not require the parentheses.

A makefile will typically define many variables, but there are also many special variables defined automatically by make. Some can be set by the user to control make’s behavior while others are set by make to communicate with the user’s makefile. ( `make` 自己会定义一些变量 )
#### 2.2.1 Automatic Variables
Automatic variables are set by make after a rule is matched.( `make` 在匹配到一个规则后，会自己设定自动变量 ) They provide access to elements from the target and prerequisite lists so you don’t have to explicitly specify any filenames. They are very useful for avoiding code duplication, but are critical when defining more general pattern rules (discussed later)

There are seven “core” automatic variables: 
`$@` The filename representing the target. ( 目标文件名 )
`$%` The filename element of an archive member specification. 
`$<` The filename of the first prerequisite. ( 第一个前提文件名 )
`$?` The names of all prerequisites that are newer than the target, separated by spaces. ( 所有更新前提文件名 )
`$^` The filenames of all the prerequisites, separated by spaces. This list has duplicate filenames removed since for most uses, such as compiling, copying, etc., duplicates are not wanted. ( 所有前提文件名，去重 )
`$+` Similar to `$^`, this is the names of all the prerequisites separated by spaces, except that `$+` includes duplicates. This variable was created for specific situations such as arguments to linkers where duplicate values have meaning. 
`$*` The stem of the target filename. A stem is typically a filename without its suffix. (We’ll discuss how stems are computed later in the section “Pattern Rules.”) Its use outside of pattern rules is discouraged.

In addition, each of the above variables has two variants for compatibility with other makes. ( 以上自动变量各有两个变体 ) One variant returns only the directory portion of the value. This is indicated by appending a “D” to the symbol, `$(@D), $(<D)`, etc. The other variant returns only the file portion of the value. This is indicated by appending an F to the symbol, `$(@F), $(<F)`, etc. Note that these variant names are more than one character long and so must be enclosed in parentheses. ( 注意变量名称超过一个字符需要用括号括起来 ) GNU make provides a more readable alternative with the `dir` and `notdir` functions. We will discuss functions in Chapter 4.
### 2.3 Finding Files with VPATH and vpath
Our examples so far have been simple enough that the makefile and sources all lived in a single directory. Let’s refactor our example and create a more realistic file layout. We can modify our word counting program by refactoring main into a function called counter.

In a traditional source tree layout the header files are placed in an `include` directory and the source is placed in a `src` directory. We’ll do this and put our makefile in the parent directory.
![[Managing Projects with GNU Make-Figure2.1.png]]
Since our source files now include header files, these new dependencies should be recorded in our makefile so that when a header file is modified, the corresponding object file will be updated. ( 在makefile中体现对头文件的依赖 )
```
count_words: count_words.o counter.c lexer.o -lfl
    gcc $^ -o $@
count_words.o: count_words.c include/counter.h 
    gcc -c $<
counter.o: counter.c include/counter.h include/lexer.h 
    gcc -c $<
lexer.o: lexer.c include/lexer.h 
    gcc -c $<
lexer.c: lexer.l
    flex -t $< > $@
```
Now when we run our makefile, we get:
```
$ make
make: *** No rule to make target `count_words.c', needed by `count_words.o'. Stop.
```
why can’t make find the source file? Because the source file is in the src directory not the current directory. Unless you direct it otherwise, make will look only in the current directory for its targets and prerequisites. ( `make` 默认仅在当前目录寻找目标和前提文件 )

You can tell make to look in different directories for its source files using the `VPATH` and `vpath` features. To fix our immediate problem, we can add a `VPATH` assignment to the makefile:
```
VPATH = src
```
This indicates that make should look in the directory src if the files it wants are not in the current directory. Now when we run our makefile, we get:
```
$ make
gcc -c src/count_words.c -o count_words.o
src/count_words.c:2:21: counter.h: No such file or directory
make: *** [count_words.o] Error 1
```
Notice that make now successfully tries to compile the first file, filling in correctly the relative path to the source. This is another reason to use automatic variables: make cannot use the appropriate path to the source if you hardcode the filename. ( `make` 在当前目录没有找到文件 `count_words.c` ，根据VPATH，在 `src` 目录中找到了，注意 `make` 自动在传递给subshell的命令中，将 `$<` 替换为了 `src/count_words.c` )
Unfortunately, the compilation dies because gcc can’t find the include file. We can fix this latest problem by “customizing” the implicit compilation rule ( 隐式编译规则 ) with the appropriate `-I` option:
```
CPPFLAGS = -I include
```
and changing occurrences of `gcc` to `gcc$(CPPFLAGS)`. Now the build succeeds: ( 定义变量为 `gcc` 传递flag，以让 `gcc` 找到在不同目录中的头文件 )
```
$ make
gcc -I include -c src/count_words.c -o count_words.o 
gcc -I include -c src/counter.c -o counter.o
flex -t src/lexer.l > lexer.c
gcc -I include -c lexer.c -o lexer.o
gcc count_words.o counter.o lexer.o /lib/libfl.a -o count_words
```
The VPATH variable consists of a list of directories to search when make needs a file. The list will be searched for targets as well as prerequisites, but not for files mentioned in command scripts. ( `make` 在目标和前提中遇到不在当前目录的文件会根据VPATH寻找，但命令中的文件则不会 ) The list of directories can be separated by spaces or colons on Unix and separated by spaces or semicolons on Windows.

The VPATH variable is good because it solved our searching problem above, but it is a rather large hammer. make will search each directory for *any* file it needs. If a file of the same name exists in multiple places in the VPATH list, make grabs the first one. Sometimes this can be a problem. ( VPATH增大了 `make` 的文件搜索范围，如果多个目录中存在同名文件，可能会引起混淆 )

The vpath directive is a more precise way to achieve our goals. The syntax of this directive is:
```
vpath pattern directory-list
```
So our previous VPATH use can be rewritten as:
```
vpath %.c src 
vpath %.l src 
vpath %.h include
```
Now we’ve told make that it should search for .c and .l files in the src directory and we’ve also added a line to search for .h files in the include directory (so we can remove the include/ from our header file prerequisites). In more complex applications, this control can save a lot of headache and debugging time. ( vpath 对 `make` 的搜索行为进行了更细致的控制 )

Here we used vpath to handle the problem of finding source that is distributed among several directories. There is a related but different problem of how to build an application so that the object files are written into a “binary tree” while the source files live in a separate “source tree.” Proper use of vpath can also helpto solve this new problem, but the task quickly becomes complex and vpath alone is not sufficient. We’ll discuss this problem in detail in later sections.
### 2.4 Pattern Rules
Many programs that read one file type and output another conform to standard conventions. For instance, all C compilers assume that files that have a `.c` suffix contain C source code and that the object filename can be derived by replacing the `.c` suffix with `.o` (or `.obj` for some Windows compilers). In the previous chapter, we noticed that flex input files use the `.l` suffix and that flex generates `.c` files。

These conventions allow make to simplify rule creation by recognizing common filename patterns and providing built-in rules for processing them. For example, by using these built-in rules our 17-line makefile can be reduced to
```
VPATH = src include 
CPPFLAGS = -I include

count_words: counter.o lexer.o -lfl 
count_words.o: counter.h 
counter.o: counter.h lexer.h 
lexer.o: lexer.h
```
( `make` 会识别常用的文件名模式，并调用内建规则处理它们 )

The built-in rules are all instances of pattern rules. A pattern rule looks like the normal rules you have already seen except the stem of the file (the portion before the suffix) is represented by a `%` character. ( 内建规则都属于模式规则，模式规则中，文件的stem都用 `%` 表示 ) This makefile works because of three built-in rules. The first specifies how to compile a `.o` file from a `.c` file:
```
%.o: %.c
    $(COMPILE.c) $(OUTPUT_OPTION) $<
```
The second specifies how to make a `.c` file from a `.l` file:
```
%.c: %.l
    @$(RM) $@ 
    $(LEX.l) $< > $@
```
Finally, there is a special rule to generate a file with no suffix (always an executable) from a `.c` file:
```
%: %.o
    $(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@
```

let’s go over make’s output carefully and see how make applies these built-in rules. When we run make on our seven-line makefile, the output is:
```
$ make
gcc -I include -c -o count_words.o src/count_words.c
gcc -I include -c -o counter.o src/counter.c 
flex -t src/lexer.l > lexer.c
gcc -I include -c -o lexer.o lexer.c
gcc count_words.o counter.o lexer.o /lib/libfl.a -o count_words 
rm lexer.c
```
First, make reads the makefile and sets the default goal to `count_words` since there are no command-line targets specified. Looking at the default goal, make identifies four prerequisites: `count_words.o` (this prerequisite is missing from the makefile, but is provided by the implicit rule), `counter.o`, `lexer.o`, and `-lfl`. It then tries to update each prerequisite in turn. ( 默认目标 `count_words` 的前提中没有 `count_words.o` ，但 `make` 根据隐式规则会自己寻找它 )

When make examines the first prerequisite, count_words.o, make finds no explicit rule for it but discovers the implicit rule. Looking in the local directory, make cannot find the source, so it begins searching the VPATH and finds a matching source file in src. Since src/count_words.c has no prerequisites, make is free to update count_words.o so it runs the commands for the implicit rule. ( `make` 同样根据隐式规则寻找 `count_words.o` 的前提 `count_words.c` ，根据VPATH找到了它，根据隐式规则的命令编译它 ) `counter.o` is similar.

When make considers `lexer.o`, it cannot find a corresponding source file (even in `src`) so it assumes this (nonexistent source) is an intermediate file and looks for a way to make `lexer.c` from some other source file. ( `make` 根据隐式规则没有找到 `lexer.c` ，因此继续寻找根据隐式规则能够生成 `lexer.c` 的文件 ) It discovers a rule to create a .c file from a .l file and notices that lexer.l exists. There is no action required to update lexer.l, so it moves on to the command for updating lexer.c, which yields the `flex` command line. Next make updates the object file from the C source. Using sequences of rules like this to update a target is called *rule chaining*.

Next, make examines the library specification `-lfl`. It searches the standard library directories for the system and discovers `/lib/libfl.a`.

Now make has all the prerequisites for updating `count_words`, so it executes the final gcc command. Lastly, make realizes it created an intermediate file that is not necessary to keep so it cleans it up.

As you can see, using rules in makefiles allows you to omit a lot of detail. Rules can have complex interactions that yield very powerful behaviors. In particular, having a built-in database of common rules makes many types of makefile specifications very simple.

The built-in rules can be customized by changing the values of the variables in the command scripts. A typical rule has a host of variables, beginning with the program to execute, and including variables to set major groupings of command-line options, such as the output file, optimization, debugging, etc. You can look at make’s default set of rules (and variables) by running make --`print-data-base`. ( 通过改变内建规则的变量值，可以自定义内建规则 )
#### 2.4.1 The Patterns
The percent character in a pattern rule is roughly equivalent to `*` in a Unix shell. It represents any number of any characters. The percent character can be placed anywhere within the pattern but can occur only once. Here are some valid uses of percent:( 模式规则中，`%` 为通配符，匹配任意数量的任意字符 )
```
%,v
s%.o 
wrapper_%
```

characters other than percent match literally within a filename. A pattern can contain a prefix or a suffix or both. When `make` searches for a pattern rule to apply, it first looks for a matching pattern rule target. The pattern rule target must start with the prefix and end with the suffix (if they exist). If a match is found, the characters between the prefix and suffix are taken as the stem of the name. Next make looks at the prerequisites of the pattern rule by substituting the stem into the prerequisite pattern. If the resulting filename exists or can be made by applying another rule, a match is made and the rule is applied. The stem word must contain at least one character. ( 对于一个目标，`make` 要应用模式规则时，先寻找库中的规则的目标模式是否有和它匹配的，找到后，`make` 去除前缀后缀得到stem，并根据模式规则的前提，寻找是否有与嵌入stem的前提模式相对应的文件 )

It is also possible to have a pattern containing only a percent character. The most common use of this pattern is to build a Unix executable program. ( 单 `%` 的模式常用于构建可执行程序 ) For instance, here are several pattern rules GNU make includes for building programs:
```
%: %.mod
    $(COMPILE.mod) -o $@ -e $@ $^
%: %.cpp
    $(LINK.cpp) $^ $(LOADLIBES) $(LDLIBS) -o $@
%: %.sh
    cat $< >$@ chmod a+x $@
```
These patterns will be used to generate an executable from a Modula source file, a preprocessed C source file, and a Bourne shell script, respectively. We will see many more implicit rules in the section “The Implicit Rules Database.”
#### 2.4.2 Static Pattern Rules
A static pattern rule is one that applies only to a specific list of targets.
```
$(OBJECTS): %.o: %.c
    $(CC) -c $(CFLAGS) $< -o $@
```
The only difference between this rule and an ordinary pattern rule is the initial `$(OBJECTS):` specification. This limits the rule to the files listed in the `$(OBJECTS)` variable ( 静态模式规则的匹配范围由变量指定 )

This is very similar to a pattern rule. Each object file in `$(OBJECTS)` is matched against the pattern `%.o` and its stem is extracted. The stem is then substituted into the pattern `%.c` to yield the target’s prerequisite. If the target pattern does not exist, make issues a warning.

Use static pattern rules whenever it is easier to list the target files explicitly than to identify them by a suffix or other pattern. ( 在更容易可以显式列出目标文件时，可以使用静态模式规则 )
#### 2.4.3 Suffix Rules
Suffix rules are the original (and obsolete) way of defining implicit rules. Because other versions of `make` may not support GNU make’s pattern rule syntax, you will still see suffix rules in makefiles intended for a wide distribution ( GUN `make` 已经用pattern rule替代了suffix rule )

Suffix rules consist of one or two suffixes concatenated and used as a target: ( 后缀规则将两个拼接起来的后缀作为目标 )
```
.c.o:
    $(COMPILE.c) $(OUTPUT_OPTION) $<
```
This is a little confusing because the prerequisite suffix comes first and the target suffix second. ( 第一个后缀是前提的后缀，第二个后缀是目标的后缀 ) This rule matches the same set of targets and prerequisites as:
```
%.o: %.c
    $(COMPILE.c) $(OUTPUT_OPTION) $<
```
The suffix rule forms the stem of the file by removing the target suffix. It forms the prerequisite by replacing the target suffix with the prerequisite suffix.

The suffix rule is recognized by make only if the two suffixes are in a list of known suffixes.

The above suffix rule is known as a double-suffix rule since it contains two suffixes. There are also single-suffix rules. These rules are used to create executables since Unix executables do not have a suffix: ( 单后缀的后缀规则一般用于构建可执行文件 )
```
.p:
    $(LINK.p) $^ $(LOADLIBES) $(LDLIBS) -o $@
```
This rule produces an executable image from a Pascal source file. This is completely analogous to the pattern rule:
```
%: %.p
    $(LINK.p) $^ $(LOADLIBES) $(LDLIBS) -o $@
```

The known suffix list is the strangest part of the syntax. A special target, `.SUFFIXES`, is used to set the list of known suffixes. Here is the first part of the default `.SUFFIXES` definition: ( `.SUFFIXES` 目标列出了已知的后缀表，以下是它的默认定义 )
```
.SUFFIXES: .out .a .ln .o .c .cc .C .cpp .p .f .F .r .y .l
```

You can add your own suffixes by simply adding a `.SUFFIXES` rule to your makefile: ( 我们在 makefile 中可以用 `.SUFFIX` 目标添加后缀 )
```
.SUFFIXES: .pdf .fo .html .xml
```

If you want to delete all the known suffixes (because they are interfering with your special suffixes) simply specify no prerequisites:
```
.SUFFIXES:
```

You can also use the command-line option `--no-builtin-rules` (or `-r`).
### 2.5 The Implicit Rules Database
GNU make 3.80 has about 90 built-in implicit rules. An implicit rule is either a pattern rule or a suffix rule. There are built-in pattern rules for C, C++, Pascal, FORTRAN, ratfor, Modula, Texinfo, TEX (including Tangle and Weave), Emacs Lisp, RCS, and SCCS. In addition, there are rules for supporting programs for these languages, such as `cpp`, `as`, `yacc`, `lex`, `tangle`, `weave` and `dvi` tools

To examine the rules database built into make, use the `--print-data-base` commandline option (`-p` for short). 

make prints its variable definitions each one preceded by a comment indicating the “origin” of the definition. For instance, variables can be environment variables, default values, automatic variables, etc. After the variables, come the rules. The actual format used by GNU make is:
```
%: %.C
# commands to execute (built-in):
    $(LINK.C) $^ $(LOADLIBES) $(LDLIBS) -o $@
```

For rules defined by the makefile, the comment will include the file and line where the rule was defined: ( 在makefile中定义的规则会在注释中被列出该规则来自于哪个文件以及哪一行 )
```
%.html: %.xml
# commands to execute (from `Makefile', line 168): 
    $(XMLTO) $(XMLTO_FLAGS) html-nochunks $<
```
#### 2.5.1 Working with Implicit Rules
The built-in implicit rules are applied whenever a target is being considered and there is no explicit rule to update it. So using an implicit rule is easy: simply do not specify a command script when adding your target to the makefile. This causes make to search its built-in database to satisfy the target. ( 在目标被识别，并且没有显式规则以构建它时，`make` 就会调用隐式规则 )

you can delete the two rules concerning flex from the built-in rule base like this:
```
%.o: %.l 
%.c: %.l
```
A pattern with no command script will remove the rule from make’s database. ( 我们可以写下没有命令的模式规则，以移除 `make` 数据库中对应的隐式规则 ) In practice, situations such as this are very rare.

We have seen several examples of how make will “chain” rules together while trying to update a target. This can lead to some complexity, which we’ll examine here. When make considers how to update a target, it searches the implicit rules for a target pattern that matches the target in hand. ( `make` 首先搜索数据库匹配目标 ) For each target pattern that matches the target file, make will look for an existing matching prerequisite. That is, after matching the target pattern, make immediately looks for the prerequisite “source” file. ( 找到匹配的目标后， `make` 根据规则中的前提模式寻找源文件 ) If the prerequisite is found, the rule is used. ( 若可以找到源文件，就应用这个规则 ) For some target patterns, there are many possible source files. For instance, a `.o` file can be made from `.c`, `.cc`, `.cpp`, `.p`, `.f`, `.r`, `.s`, and `.mod` files. 

But what if the source is not found after searching all possible rules? In this case, make will search the rules again, this time assuming that the matching source file should be considered as a new target for updating. ( 如果 `make` 没有找到匹配前提模式的源文件，`make` 递归地将该前提模式设定为目标，寻找匹配该前提模式的隐式规则 ) By performing this search recursively, make can find a “chain” of rules that allows updating a target.

One of the more impressive sequences that make can produce automatically from its database is shown here. First, we setup our experiment by creating an empty yacc source file and registering with RCS using `ci` (that is, we want a version-controlled yacc source file):
```
$ touch foo.y 
$ ci foo.y
foo.y,v <-- foo.y 
.
initial revision: 1.1 done
```
Now, we ask make how it would create the executable `foo`. The `--just-print` (or `-n`) option tells make to report what actions it would perform without actually running them. Notice that we have no makefile and no “source” code, only an RCS file:
```
$ make -n foo
co foo.y,v foo.y
foo.y,v --> foo.y
revision 1.1
done
bison -y foo.y
mv -f f.tab.c foo.c
gcc -c -o foo.o foo.c
gcc foo.o -o foo
rm foo.c foo.o foo.y
```
Following the chain of implicit rules and prerequisites, make determined it could create the executable, foo, if it had the object file foo.o. It could create foo.o if it had the C source file foo.c. It could create foo.c if it had the yacc source file foo.y. Finally, it realized it could create foo.y by checking out the file from the RCS file `foo.y,v`, which it actually has.
Once make has formulated this plan, it executes it by checking out foo. y with co, transforming it into foo.c with bison, compiling it into foo.o with gcc, and linking it to form foo again with gcc. All this from the implicit rules database.

The files generated by chaining rules are called *intermediate* files and are treated specially by make. First, since intermediate files do not occur in targets (otherwise they would not be intermediate), make will never simply update an intermediate file. Second, because make creates intermediate files itself as a side effect of updating a target, make will delete the intermediates before exiting. ( `make` 会自动删除中间文件 )

The implicit rules database contains a lot of rules and even very large projects rarely use most of it. Because it contains such a wide variety, rules you didn’t expect to fire may be used by make in unexpected ways. As a preventative measure, some large projects choose to discard the implicit rules entirely in favor of their own handcrafted rules. You can do this easily with the `--no-builtin-rules` (or `-r`) option. (I have never had to use this option even on the largest projects.) If you use this option, you may also want to consider using `--no-builtin-variables` (or `-R`).
#### 2.5.2 Rule Structure
The built-in rules have a standard structure intended to make them easily customizable. Here is the (by now familiar) rule for updating an object file from its C source:
```
%.o: %.c
    $(COMPILE.c) $(OUTPUT_OPTION) $<
```
The customization of this rule is controlled entirely by the set of variables it uses. ( 对内建规则的自定义完全由它使用的变量控制 ) We see two variables here, but `COMPILE.c` in particular is defined in terms of several other variables: ( 变量 `COMPILE.c` 进一步由其他几个变量定义 )
```
COMPILE.c = $(CC) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
CC = gcc
OUTPUT_OPTION = -o $@
```
The C compiler itself can be changed by altering the value of the `CC` ( 改变 `CC` 变量可以改变使用的C编译器 ) variable. The other variables are used for setting compilation options (`CFLAGS`) ( `CFLAGS` 用于设置编译选项 ), preprocessor options (`CPPFLAGS`) ( `CPPFLAGS` 用于设置预处理选项 ), and architecture-specific options (`TARGET_ARCH`) ( `TARGET_ARCH` 用于设置针对结构的选项 ).

The variables in a built-in rule are intended to make customizing the rule as easy as possible. For that reason, it is important to be very careful when setting these variables in your makefile. If you set these variables in a naive way, you destroy the end user’s ability to customize them. For instance, given this assignment in a makefile:
```
CPPFLAGS = -I project/include
```
If the user wanted to add a CPP define to the command line, they would normally invoke `make` like:
```
$ make CPPFLAGS=-DDEBUG
```
But in so doing they would accidentally remove the `-I` option that is (presumably) required for compiling. Variables set on the command line override all other assignments to the variable. ( 在命令行设定的变量值有最高优先级，会覆盖其他变量赋值 ) So, setting `CPPFLAGS` inappropriately in the makefile “broke” a customization feature that most users would expect to work. Instead of using simple assignment, consider redefining the compilation variable to include your own variables:
```
COMPILE.c = $(CC) $(CFLAGS) $(INCLUDES) $(CPPFLAGS) $(TARGET_ARCH) -c INCLUDES = -I project/include
```
Or you can use append-style assignment, which is discussed in the section “Other Types of Assignment” in Chapter 3.
#### 2.5.3 Implicit Rules for Source Control
make knows about two source code control systems, RCS and SCCS, and supports their use with built-in implicit rules. I’ve never found a use for the source control support in make, nor have I seen it used in other production software. I do not recommend the use of this feature. 
#### 2.5.4 A Simple Help Command
### 2.6 Speical Targets
A *special target* is a built-in phony target used to change make’s default behavior. ( 特殊目标是内建的假目标 )For instance, `.PHONY`, a special target, which we’ve already seen, declares that its prerequisite does not refer to an actual file and should always be considered out of date. The `.PHONY` target is the most common special target you will see, but there are others as well.

These special targets follow the syntax of normal targets, that is `target: prerequisite`, but the target is not a file or even a normal phony. They are really more like directives for modifying make’s internal algorithms. ( 特殊目标的作用更类似于修改 `make` 内部行为的指令 )

There are twelve special targets. They fall into three categories: as we’ve just said many are used to alter the behavior of make when updating a target, another set act simply as global flags to make and ignore their targets, finally the `.SUFFIXES` special target is used when specifying old-fashioned suffix rules

The most useful target modifiers (aside from `.PHONY`) are:
.`INTERMEDIATE`
    Prerequisites of this special target are treated as intermediate files. If make creates the file while updating another target, the file will be deleted automatically when make exits. If the file already exists when make considers updating the file, the file will not be deleted. ( 标记中间文件，作为其前提的文件如果在 `make` 运行前不存在且运行时被生成就会被最后删除 )
`.SECONDARY`
    Prerequisites of this special target are treated as intermediate files but are never automatically deleted. The most common use of `.SECONDARY` is to mark object files stored in libraries. Normally these object files will be deleted as soon as they are added to an archive. Sometimes it is more convenient during development to keep these object files, but still use the make support for updating archives. ( 标记中间文件，但作为其前提的文件不会被删除 )
`.PRECIOUS`
    When make is interrupted during execution, it may delete the target file it is updating if the file was modified since make started. This is so make doesn’t leave a partially constructed (possibly corrupt) file laying around in the build tree. There are times when you don’t want this behavior, particularly if the file is large and computationally expensive to create. If you mark the file as precious, make will never delete the file if interrupted.
    Use of .PRECIOUS is relatively rare, but when it is needed it is often a life saver. Note that make will not perform an automatic delete if the commands of a rule generate an error. It does so only when interrupted by a signal. ( `make` 在被信号中断的时候会删除未完全构建的目标文件，用 `.PRECIOUS` 标记可以阻止这一行为，注意 `make` 在出错的时候不会删除未完全构建的文件 )
`.DELETE_ON_ERROR`
    This is sort of the opposite of .PRECIOUS. Marking a target as .DELETE_ON_ERROR says that make should delete the target if any of the commands associated with the rule generates an error. make normally only deletes the target if it is interrupted by a signal. ( 该标记令 `make` 在遇到和被标记文件相关的错误时，会删除该文件 )
### 2.7 Automatic Dependency Generation
Let’s use a program to identify the relationships between files and maybe even have this program write out these dependencies in makefile syntax. As you have probably guessed, such a program already exists—at least for C/C++. There is a option to gcc and many other C/C++ compilers that will read the source and write makefile dependencies. For instance, here is how I found the dependencies for stdio.h: ( `gcc` 存在选项可以用于阅读源文件，并以makefile语法输出依赖 )
```
$ echo "#include <stdio.h>" > stdio.c
$ gcc -M stdio.c
stdio.o: stdio.c /usr/include/stdio.h /usr/include/_ansi.h \
    /usr/include/newlib.h /usr/include/sys/config.h \
    /usr/include/machine/ieeefp.h /usr/include/cygwin/config.h \
    /usr/lib/gcc-lib/i686-pc-cygwin/3.2/include/stddef.h \
    /usr/lib/gcc-lib/i686-pc-cygwin/3.2/include/stdarg.h \
    /usr/include/sys/reent.h /usr/include/sys/_types.h \
    /usr/include/sys/types.h /usr/include/machine/types.h \
    /usr/include/sys/features.h /usr/include/cygwin/types.h \
    /usr/include/sys/sysmacros.h /usr/include/stdint.h \
    /usr/include/sys/stdio.h
```

for including automatically generated dependencies into makefiles. we need to add an `include` directive to the make program.

So, the trick is to write a makefile target whose action runs gcc over all your source with the -M option, saves the results in a dependency file, and then re-runs make including the generated dependency file in the makefile so it can trigger the updates we need. ( 我们在makefile中定义相关目标，它的命令是对于我们所有的源文件运行 `gcc -M` 生成它们的依赖，将这些结果储存在一个依赖文件中，我们运行一次 `make` 执行这些动作，然后再运行 `make` 借由依赖文件，构建目标 ) Before GNU make, this is exactly what was done and the rule looked like:
```
depend: count_words.c lexer.c counter.c
    $(CC) -M $(CPPFLAGS) $^ > $@
include depend
```
Before running make to build the program, you would first execute `make depend` to generate the dependencies. ( 需要先执行 `make depend` 生成依赖文件 `depend` ) This was good as far as it went, but often people would add or remove dependencies from their source without regenerating the depend file. Then source wouldn’t get recompiled and the whole mess started again.

GNU make solved this last niggling problem with a cool feature and a simple algorithm. First, the algorithm. If we generated each source file’s dependencies into its own dependency file with, say, a .d suffix and added the .d file itself as a target to this dependency rule, then make could know that the .d needed to be updated (along with the object file) when the source file changed:
```
counter.o counter.d: src/counter.c include/counter.h include/lexer.h
```
( 我们为每个目标文件设定一个依赖文件 `.d` 将它也作为目标，以便 `make` 知道，当源文件更新时，依赖文件需要重新生成 )

