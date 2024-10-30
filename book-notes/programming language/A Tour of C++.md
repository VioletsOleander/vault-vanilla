# 1 The Basics
## 1.1 Introduction
This chapter informally presents the notation of $\mathrm{C++}$ , $\mathrm{C++}$ ’s model of memory and computation, and the basic mechanisms for organizing code into a program. These are the language facilities supporting the styles most often seen in C and sometimes called procedural programming .
> 本章介绍 C++ 的概念、C++的存储和计算模型，以及基础的代码组织机制
## 1.2 Programs
$\mathrm{C++}$ is a compiled language. For a program to run, its source text has to be processed by a compiler, producing object files, which are combined by a linker yielding an executable program. A $\mathrm{C++}$ program typically consists of many source code files (usually simply called source files ).

![[A Tour of C++-FIg1.png]]

An executable program is created for a specific hardware/system combination; it is not portable, say, from a Mac to a Windows PC. When we talk about portability of $\mathrm{C++}$ programs, we usually mean portability of source code; that is, the source code can be successfully compiled and run on a variety of systems.
> 可执行程序是不可移植的，我们讨论的可移植性指的是源程序，即源程序可以在不同系统上编译且运行

The ISO $\mathrm{C++}$ standard defines two kinds of entities: 
- *Core language features* , such as built-in types (e.g., char and int ) and loops (e.g., for -state- ments and while -statements) 
- *Standard-library components* , such as containers (e.g., vector and map ) and I/O operations (e.g., $<<$ and getline () )
> ISO C++标准定义了两类实体：
> - 核心语言特征：例如内建类型、循环（ `char/int` ; `for/while` ）
> - 标准库成分：例如容器、IO 操作（ `vector/map` ; `<</getline()` ）

The standard-library components are perfectly ordinary $\mathrm{C++}$ code provided by every $\mathrm{C++}$ implementation. That is, the $\mathrm{C++}$ standard library can be implemented in $\mathrm{C++}$ itself and is (with very minor uses of machine code for things such as thread context switching). This implies that $\mathrm{C++}$ is sufficiently expressive and efficient for the most demanding systems programming tasks.
> C++的标准库同样使用 C++编写（仅针对线程上下文切换用了极少数的机器码），且 C++标准库在所有的 C++实现中都有提供

$\mathrm{C++}$ is a statically typed language. That is, the type of every entity (e.g., object, value, name, and expression) must be known to the compiler at its point of use. The type of an object determines the set of operations applicable to it.
> C++是静态类型语言，即编译器在使用任意实体时，都必须知道它的类型
### 1.2.1 Hello, World!
The minimal $\mathrm{C++}$ program is

```
int main () { } // the minimal $C++$ program
```

This defines a function called main , which takes no arguments and does nothing.

Curly braces, $\{\,\}.$ , express grouping in $\mathrm{C++}$ . Here, they indicate the start and end of the function body. The double slash, // , begins a comment that extends to the end of the line. A comment is for the human reader; the compiler ignores comments.

Every $\mathrm{C++}$ program must have exactly one global function named main () . The program starts by executing that function. The int integer value returned by main () , if any, is the program’s return value to ‘‘the system.’’ If no value is returned, the system will receive a value indicating successful completion. A nonzero value from main () indicates failure. Not ev ery operating system and execution environment make use of that return value: Linux/Unix-based environments do, but Windows- based environments rarely do.
> 所有的 C++程序必须有且仅有一个全局函数 `main`
> `main` 函数没有指定返回值时，系统在程序成功执行后仍会受到0表示执行成功，一个非零的返回值则表示失败
> 一般仅有 Linux/Unix 系统会利用这个返回值

Typically, a program produces some output. Here is a program that writes Hello, World! :

```
#include <iostream>

int main () { std:: cout $<<$ "Hello, World!\n";
```

The line `#include <iostream>` instructs the compiler to include the declarations of the standard stream I/O facilities as found in iostream . Without these declarations, the expression
> `#include <iostream>` 指示编译器 include `iostream` 中的所有声明

```
std:: cout << "Hello, World!\n"
```

would make no sense. The operator $<<$ (‘‘put to’’) writes its second argument onto its first. In this case, the string literal "Hello, World!\n" is written onto the standard output stream std:: cout . A string literal is a sequence of characters surrounded by double quotes. In a string literal, the backslash character \ followed by another character denotes a single ‘‘special character.’’ In this case, $\mathtt{\backslash n}$ is the newline character, so that the characters written are Hello, World! followed by a newline.
> 运算符 `<<` 将它的第二个参数写入第一个参数，本例中将字符串字面值写入了 `std::cout` 

The std:: specifies that the name cout is to be found in the standard-library namespace (§3.4). I usually leave out the std:: when discussing standard features; $\S3.4$ shows how to make names from a namespace visible without explicit qualification.
> `std::` 指定了名称 `cout` 应该在标准库命名空间中找到

Essentially all executable code is placed in functions and called directly or indirectly from main () . For example:
> 所有的可执行码都处于函数中，由 `main()` 直接或间接调用

A ‘‘return type’’ void indicates that a function does not return a value.
## 1.3 Functions
The main way of getting something done in a $\mathrm{C++}$ program is to call a function to do it. Defining a function is the way you specify how an operation is to be done. A function cannot be called unless it has been previously declared.
> 函数在没有被先前声明的情况下不能被调用

A function declaration gives the name of the function, the type of the value returned (if any), and the number and types of the arguments that must be supplied in a call. For example:
> 函数声明给出函数名、返回值类型、参数数量和类型

In a function declaration, the return type comes before the name of the function and the argument types come after the name enclosed in parentheses.

The semantics of argument passing are identical to the semantics of initialization (§3.6.1). That is, argument types are checked and implicit argument type conversion takes place when necessary (§1.4). For example:
> 参数传递的语义和初始化的语义相同，也就是会检查实参类型，并且如果可能会进行隐式类型转化
> 类型检查和转换是在编译时进行的

The value of such compile-time checking and type conversion should not be underestimated.

A function declaration may contain argument names. This can be a help to the reader of a pro- gram, but unless the declaration is also a function definition, the compiler simply ignores such names. For example:
> 函数声明时可以包含参数名称，但编译器会忽略它们

The type of a function consists of its return type and the sequence of its argument types. For exam- ple:
> 函数的类型由它的返回类型和参数类型构成，例如
> `double(const vector<double>&, int)`

```cpp
double get (const vector<double>& vec, int index); // type: double (const vector<double>&, int)
```

A function can be a member of a class $(\S2.3,\S4.2.1)$ . For such a member function , the name of its class is also part of the function type. For example:
> 函数作为类的成员函数时，类的名称也是函数类型的一部分，例如
> `char& String::(int)`

```cpp
char& String:: operator[](int index); // type: char& String::(int)
```

We want our code to be comprehensible, because that is the first step on the way to maintainability. The first step to comprehensibility is to break computational tasks into meaningful chunks (repre- sented as functions and classes) and name those. Such functions then provide the basic vocabulary of computation, just as the types (built-in and user-defined) provide the basic vocabulary of data. 

The $\mathrm{C++}$ standard algorithms (e.g., find , sort , and iota ) provide a good start (Chapter 12). Next, we can compose functions representing common or specialized tasks into larger computations.

The number of errors in code correlates strongly with the amount of code and the complexity of the code. Both problems can be addressed by using more and shorter functions. Using a function to do a specific task often saves us from writing a specific piece of code in the middle of other code; making it a function forces us to name the activity and document its dependencies.
> 多使用函数来做特定的任务

If two functions are defined with the same name, but with different argument types, the com- piler will choose the most appropriate function to invoke for each call. For example:
> 具有相同函数但参数类型不同的函数是允许的，编译器会根据实参类型决定要调用的具体函数是哪一个 （事实上这些函数的函数类型是不同的）

If two alternative functions could be called, but neither is better than the other, the call is deemed ambiguous and the compiler gives an error. For example:
> 如果存在多个候选，且无法区分哪个更适合，编译器会报错

```cpp
void print (int, double); 
void print (double ,int);
void user2 () 
{ 
    print (0,0); // error : ambiguous
}
```

Defining multiple functions with the same name is known as function overloading and is one of the essential parts of generic programming $(\S7.2)$ . When a function is overloaded, each function of the same name should implement the same semantics. The print () functions are an example of this; each print () prints its argument.
> 使用函数重载时，注意名称相同的函数应该实现相同的语义
## 1.4 Types, Variables, and Arithmetic
Every name and every expression has a type that determines the operations that may be performed on it. For example, the declaration

```
int inch;
```

specifies that inch is of type int ; that is, inch is an integer variable. 

A declaration is a statement that introduces an entity into the program. It specifies a type for the entity: 
• A type defines a set of possible values and a set of operations (for an object). 
• An object is some memory that holds a value of some type. 
• A value is a set of bits interpreted according to a type. 
• A variable is a named object.
> C++通过声明语句为程序引入实体，它为实体指定一个类型
> 类型定义了实体的可能值集合，以及可以对其进行的运算
> 对象即存储了某个类型值的内存空间
> 值即根据类型翻译的 bits 集合
> 变量即有名称的对象

$\mathrm{C++}$ offers a small zoo of fundamental types, but since I’m not a zoologist, I will not list them all. You can find them all in reference sources, such as [Stroustrup, 2013] or the [Cppreference] on the Web. Examples are:

Each fundamental type corresponds directly to hardware facilities and has a fixed size that deter- mines the range of values that can be stored in it:
> C++提供了基本类型，每个基本类型直接对应了硬件特性，且大小固定（决定了可以存储的值范围）

A char variable is of the natural size to hold a character on a given machine (typically an 8-bit byte), and the sizes of other types are multiples of the size of a char . The size of a type is imple- mentation-defined (i.e., it can vary among different machines) and can be obtained by the siz eof operator; for example, siz eof (char) equals 1 and siz eof (int) is often 4 .
> `sizeof()` 运算符用于获取类型的大小

Numbers can be ﬂoating-point or integers. 
• Floating-point numbers are recognized by a decimal point (e.g., 3.14 ) or by an exponent (e.g., 3e−2 ). 
• Integer literals are by default decimal (e.g., 42 means forty-two). A 0b prefix indicates a binary (base 2) integer literal (e.g., 0b10101010 ). A ${\bf0x}$ prefix indicates a hexadecimal (base 16) integer literal (e.g., 0xBAD1234 ). A 0 prefix indicates an octal (base 8) integer literal (e.g., 0334 ).
> 数：浮点数、整数
> 浮点数由小数点或 e 表示
> 整数字面值默认为十进制，前缀0b 表示二进制，0x 表示16进制，0表示8进制

To make long literals more readable for humans, we can use a single quote ( ' ) as a digit separator. For example, $\pi$ is about `3.14159'26535'89793'23846'26433'83279'50288` or if you prefer hexadecimal `0x3.243F'6A88'85A3'08D3` .
### 1.4.1 Arithmetic
The arithmetic operators can be used for appropriate combinations of the fundamental types:

So can the comparison operators:

Furthermore, logical operators are provided:

A bitwise logical operator yields a result of the operand type for which the operation has been per- formed on each bit. The logical operators && and $||$ simply return true or false depending on the values of their operands.
> 按位的逻辑运算符的返回结果类型和 operand 的类型一致，计算就按照按位进行计算
> 逻辑运算符 `&&, ||` 则只返回 `true or false`

In assignments and in arithmetic operations, $\mathrm{C++}$ performs all meaningful conversions between the basic types so that they can be mixed freely:
> C++会在算数运算和赋值时进行必要的基本类型之间的转换

The conversions used in expressions are called *the usual arithmetic conversions* and aim to ensure that expressions are computed at the highest precision of its operands. For example, an addition of a double and an int is calculated using double-precision ﬂoating-point arithmetic.
> C++在表达式中自动执行的转换称为 usual arithmetic conversion，目标是表达式以其操作数可以达到的最高精度进行计算

Note that $=$ is the assignment operator and $==$ tests equality.

In addition to the conventional arithmetic and logical operators, $\mathrm{C++}$ offers more specific opera- tions for modifying a variable:

$$
\begin{array}{l l}{{x+=y}}&{{\quad I I\ x=x+y}}\\ {{\quad+x\quad}}&{{\quad I I\ i n c r e m e n t;\ x=x+I}}\\ {{x-=y}}&{{\quad I I\ x=x-y}}\\ {{\quad-x\quad}}&{{\quad I I\ d e c r e m e n t;\ x=x-I}}\\ {{x*=y}}&{{\quad I I\ s c a l i n g;\ x=x^{*}y}}\\ {{x/=y}}&{{\quad I I\ s c a l i n g;\ x=x/y}}\\ {{x\%=y}}&{{\quad I I\ x=x\%y}}\end{array}
$$

These operators are concise, convenient, and very frequently used.

The order of evaluation of expressions is left to right, except for assignments, which are right- to-left. The order of evaluation of function arguments is unfortunately unspecified.
> 表达式估值的顺序是从左到右
> 赋值表达式是例外，从右到左
> 函数参数估值的顺序一般未定义
### 1.4.2 Initialization
Before an object can be used, it must be given a value. $\mathrm{C++}$ offers a variety of notations for expressing initialization, such as the $=$ used above, and a universal form based on curly-brace- delimited initializer lists:
> 对象被使用之前必须被给定一个值
> C++提供了 `=` 以及 `{}` 用于初始化

The $=$ form is traditional and dates back to C, but if in doubt, use the general $\{\}$ -list form. If nothing else, it saves you from conversions that lose information:
> `{}` 形式的初始化会在类型不匹配时报错，而 `=` 则会执行隐式类型转化，因此可能会导致损失信息

```c
int i1 = 7.8; // i1 becomes 7
int i2 {7.8}; // error: floating point to interger conversion
```

Unfortunately, conversions that lose information, narrowing conversions , such as double to int and int to char , are allowed and implicitly applied when you use $=$ (but not when you use $\{\}$ ). The prob- lems caused by implicit narrowing conversions are a price paid for C compatibility (§16.3).

A constant (§1.6) cannot be left uninitialized and a variable should only be left uninitialized in extremely rare circumstances. Don’t introduce a name until you have a suitable value for it. User- defined types (such as string , vector , Matrix , Motor controller , and Orc_warrior ) can be defined to be implicitly initialized (§4.2.1).
> 常量必须初始化，变量也只有在极少数情况不初始化，不要在没有为某个名字准备好合适的值的时候引入这个名字
> 用户定义的类型可以隐式初始化（构造函数）

When defining a variable, you don’t need to state its type explicitly when it can be deduced from the initializer:
> 如果在定义变量时，变量的类型可以从初始化值中推导时，我们可以不需要显式声明其类型

```c
auto b = true; // bool
auto bb {true}; // bool
auto i = 1; // int
auto z = sqrt(y); // z has the type whatever sqrt() returns
```

With auto , we tend to use the $=$ because there is no potentially troublesome type conversion involved, but if you prefer to use $\{\}$ initialization consistently, you can do that instead.
> 使用 `auto` 时，我们倾向于使用 `=` ，因为变量的类型会自动推导，故不会存在潜在的类型转换

We use auto where we don’t hav e a specific reason to mention the type explicitly. ‘‘Specific reasons’’ include: 
• The definition is in a large scope where we want to make the type clearly visible to readers of our code. 
• We want to be explicit about a variable’s range or precision (e.g., double rather than ﬂoat ).

> `auto` 可以在我们没有明确的需要显式提及类型的原因时使用，需要显式提及的明确原因包括：
>  - 定义处于一个大的作用域，我们需要让 code reviewer 明确知道类型
>  - 我们希望明确该变量的范围或精度 (`double/float` )

Using auto , we avoid redundancy and writing long type names. This is especially important in generic programming where the exact type of an object can be hard for the programmer to know and the type names can be quite long (§12.2).
> `auto` 可以避免写过长的类型名称，在泛型编程中，程序员较难以知道对象明确的类型时，`auto` 十分有用
## 1.5 Scope and Lifetime
A declaration introduces its name into a scope:
> 声明向当前作用域引入一个名字

• *Local scope* : A name declared in a function (§1.3) or lambda (§6.3.2) is called a local name . Its scope extends from its point of declaration to the end of the block in which its declara- tion occurs. A block is delimited by a $\{\,\}$ pair. Function argument names are considered local names. 
> 局部作用域：声明在一个函数或 lambda 中的名字称为局部名字，其作用域从声明语句到当前 block 结束，block 使用 `{}` 指定
> 函数参数名称被视作局部名字

• Class scope : A name is called a member name (or a class member name ) if it is defined in a class (§2.2, $\S2.3$ , Chapter 4), outside any function (§1.3), lambda $(\S6.3.2)$ , or enum class (§2.5). Its scope extends from the opening $\{$ of its enclosing declaration to the end of that declaration. 
> 类作用域：定义于类中的，在任意函数/lambda/enmu 类外的名字称为成员名字/类成员名字，它的作用域从包含它的声明的 `{` 开始到 `}` 结束

• Namespace scope : A name is called a namespace member name if it is defined in a name- space (§3.4) outside any function, lambda (§6.3.2), class (§2.2, $\S2.3$ , Chapter 4), or enum class (§2.5). Its scope extends from the point of declaration to the end of its namespace. 
> 命名空间作用域：定义域命名空间中，在任意函数/lambda/类/enum 类外的名字称为命名空间成员名字，其作用域从声明开始到命名空间结束

A name not declared inside any other construct is called a global name and is said to be in the global namespace . In addition, we can have objects without names, such as temporaries and objects created using new (§4.2.2). For example: 
> 未定义于任意其他结构的名字称为全局名字，处于全局命名空间
> 也可以存在没有名字的对象，例如暂时对象以及使用 `new` 创建的对象

```c
vector<int> vec; // vec is global (a global vector of integers)
struct Record { 
    string name; // name is a member or Record (a string member) 
    // ... 
}; 

void fct (int arg) // fct is global (a global function) 
                   // arg is local (an integer argument) 
{ 
    string motto {"Who dares wins"}; // motto is local 
    auto p = new Record{"Hume"}; // p points to an unnamed Record (created by new) // ... 
}
```

An object must be constructed (initialized) before it is used and will be destroyed at the end of its scope. For a namespace object the point of destruction is the end of the program. For a member, the point of destruction is determined by the point of destruction of the object of which it is a mem- ber. An object created by new ‘‘lives’’ until destroyed by delete (§4.2.2).
> 一个对象必须在它被使用和在作用域结束被摧毁之前被构建/初始化
> 命名空间对象被摧毁的时候就是程序结束的时候
> 成员对象被摧毁的时候取决于它归属的对象
> 由 `new` 创建的对象直到使用 `delete` 之间都一直存在
## 1.6 Constants
$\mathrm{C++}$ supports two notions of immutability:

- `const` : meaning roughly ‘‘I promise not to change this value.’’ This is used primarily to specify interfaces so that data can be passed to functions using pointers and references without fear of it being modified. The compiler enforces the promise made by const . The value of a const can be calculated at run time.

- `constexpr` : meaning roughly ‘‘to be evaluated at compile time.’’ This is used primarily to specify constants, to allow placement of data in read-only memory (where it is unlikely to be corrupted), and for performance. The value of a constexpr must be calculated by the compiler.

> C++支持两种不可变的概念：
> - `const` : 意思是 “保证该值不被修改” ，主要用于指定接口，因此使用指针或引用向函数中传入的参数就不需要担心被修改，编译器会静态检测 `const` 修饰的是否被修改，`const` 变量的初始值可以在运行时被计算
> - `constexpr` : 意思是 “在编译时被估值”，`constexpr` 主要用于指定常数，使得数据可以放在只读内存中（因此不可能被污染，并且可以提高性能），`constexpr` 的值必须由编译器计算

For example:

```cpp
constexpr dmv = 17; // dmv is a named constant
int var = 17; // var is not a constant
const double sq = sqrt(17); // sq is a named constant, possibly evaluated at run time

double sum(const vector<int>&); // sum will not modify its argument

vector<double> v {1.2, 1.4, 1.5}; // v is not a constant
const double s1 = sum(v); // ok: sum(v) is evaluated at run time
constexpr double s2 = sum(v); // error: sum(v) is not a constant experession
```

For a function to be usable in a constant expression, that is, in an expression that will be evaluated by the compiler, it must be defined constexpr. For example:
> 如果函数需要能在常量表达式中被使用，即在一个编译时被估值的表达式中被使用，则函数的返回值必须定义为 `constexpr`

```cpp
constexpr double square(double x) { return x * x; }

constexpr double n1 = 1.4 * square(17); // ok: 1.4 * square(17) is a constant expression
constexpr double n2 = square(var); //error: square(var) is not a constant expresion
const double n3 = square(var); // ok: may be evaluated at run time
```

A constexpr function can be used for non-constant arguments, but when that is done the result is not a constant expression. We allow a constexpr function to be called with non-constant-expression arguments in contexts that do not require constant expressions. That way, we don’t have to define essentially the same function twice: once for constant expressions and once for variables.
> 一个 `constexpr` 函数可以接受非常量的参数，但此时函数结果就不是一个常量表达式

To be constexpr , a function must be rather simple and cannot have side effects and can only use information passed to it as arguments. In particular, it cannot modify non-local variables, but it can have loops and use its own local variables. For example:
> 要满足 `constexpr` ，函数必须要很简单，并且不能有 side effects，以及只能使用作为参数传入给它的信息，特别地，它不能修改非局部变量，但允许有循环并且使用自己的局部变量

In a few places, constant expressions are required by language rules (e.g., array bounds (§1.7), case labels $(\S1.8)$ , template value arguments (§6.2), and constants declared using constexpr ). In other cases, compile-time evaluation is important for performance. Independently of performance issues, the notion of immutability (an object with an unchangeable state) is an important design concern.
> C++的语言规则要求在一些地方使用常量表达式，例如数组界、case label、模板值参数、使用 `constexpr` 声明的量
> 同时编译时的估值对于性能也是很重要的
## 1.7 Pointers, Arrays, and References
The most fundamental collection of data is a contiguously allocated sequence of elements of the same type, called an array . This is basically what the hardware offers. An array of elements of type char can be declared like this:
> 连续分配的相同类型元素的序列称为数组 

```
char v[6]; // array of 6 characters
```

Similarly, a pointer can be declared like this:

```
char ∗ p; // pointer to character
```

In declarations, [ ] means ‘‘array of’’ and $^*$ means ‘‘pointer to.’’ All arrays have 0 as their lower bound, so v has six elements, v[0] to v[5] . The size of an array must be a constant expression (§1.6). A pointer variable can hold the address of an object of the appropriate type:
> 数组都以 0 为 lower bound，数组的大小必须是常量表达式
> 指针变量可以存储对应类型的对象的地址

In an expression, prefix unary $^*$ means ‘‘contents of’’ and prefix unary & means ‘‘address of.’’ We can represent the result of that initialized definition graphically:
> 表达式中的前缀 `*` 表示解引用，`&` 前缀表示取地址

Consider copying ten elements from one array to another:

This for -statement can be read as ‘‘set i to zero; while i is not 10 , copy the i th element and increment i .’’ When applied to an integer or ﬂoating-point variable, the increment operator, $^{++}$ , simply adds 1 . 
> 对于整形和浮点变量，运算符 `++` 即自加1

$\mathrm{C++}$ also offers a simpler for -statement, called a range- for -statement, for loops that traverse a sequence in the simplest way:
> C++还提供了 range-for 语句，便于遍历一个序列

```cpp
void print()
{
    int v[] = {1,2, 3, 4};

    for (auto x: v)
        cout << x << endl;

    for (auto x: {1, 2, 3, 4})
        cout << x <<endl;
}
```

The first range- for -statement can be read as ‘‘for every element of $\mathbf{v}$ , from the first to the last, place a copy in $\pmb{x}$ and print it.’’ Note that we don’t hav e to specify an array bound when we initialize it with a list. The range- for -statement can be used for any sequence of elements $(\S12.1)$ .
> range-for 语句可以被解释为遍历序列中的各个元素，将元素的值拷贝给 `x` ，然后执行对应逻辑
> 同时注意我们直接用一个 list 初始化数组时，我们不需要指定数组的界
> range-for 语句可以用于包含了任意元素类型的序列

If we didn’t want to copy the values from $\mathbf{v}$ into the variable $\pmb{x}$ , but rather just have $\mathbf{x}$ refer to an element, we could write:
> 如果我们不希望 `x` 是值拷贝，可以写为

```cpp
void increment()
{
    int v[] = {1, 2, 3, 4};

    for (auto& x: v)
        x++;
}
```

In a declaration, the unary suffix & means ‘‘reference to.’’ A reference is similar to a pointer, except that you don’t need to use a prefix $^*$ to access the value referred to by the reference. Also, a reference cannot be made to refer to a different object after its initialization.
> 声明时的 `&` 前缀表示引用，引用和指针的差异在于引用不能在它的初始化之后再被用于指向另一个不同的对象

References are particularly useful for specifying function arguments. For example: 

```
void sort (vector<double>& v); // sor t v (v is a vector of doubles)
```

By using a reference, we ensure that for a call sort (my_vec) , we do not copy my_vec and that it really is my_vec that is sorted and not a copy of it. 

When we don’t want to modify an argument but still don’t want the cost of copying, we use a const reference (§1.6). For example:
> 如果我们不想拷贝参数，同时也不想修改参数，可以使用常引用

```cpp
double sum (const vector<double>&)
```

Functions taking const references are very common.
> 使用 `const` 引用的函数非常常见

When used in declarations, operators (such as & , $^*,$ , and [ ] ) are called declarator operators :
> 当用在声明语句时，运算符（例如 `&/*/[]`）被称为声明运算符

```cpp
T a[n]; // T[n]: a is an array of T
T* a; // T*: a is a pointer to T
T& a; // T&: a is a reference to T
T f(A); // T(A): f is a function taking argument of type A and return result of type T
```
### 1.7.1 The Null Pointer
We try to ensure that a pointer always points to an object so that dereferencing it is valid. When we don’t hav e an object to point to or if we need to represent the notion of ‘‘no object available’’ (e.g., for an end of a list), we give the pointer the value nullptr (‘‘the null pointer’’). There is only one nullptr shared by all pointer types:
> 我们需要尽量保证对指针的解引用是合法的
> 如果需要表达指针没有指向任何对象，可以将指针赋值 `nullptr`，值 `nullptr` 由所有的指针类型共享

```cpp
double* p = nullptr;
Link<Record>* ls = nullptr;
int x = nullptr; // error: nullptr is a pointer not an integer
```

It is often wise to check that a pointer argument actually points to something:

Note how we can advance a pointer to point to the next element of an array using $^{++}$ and that we can leave out the initializer in a for -statement if we don’t need it.
> 我们可以通过 `++` 让指针指向数组中的下一个元素

The definition of count_x() assumes that the char ∗ is a $C$ -style string , that is, that the pointer points to a zero-terminated array of char . The characters in a string literal are immutable, so to han- dle count_x ("Hello!") , I declared count_x() a const char ∗ argument.
> C 风格的 string 即以 `\0` (ASCII 值就是0) 为结尾的 `char` 数组

In older code, 0 or NULL is typically used instead of nullptr . Howev er, using nullptr eliminates potential confusion between integers (such as 0 or NULL ) and pointers (such as nullptr ).
> 在较老的代码中，`0/NULL` 使用的较多，但使用 `nullptr` 可以避免指针和整数混淆

In the count_x() example, we are not using the initializer part of the for -statement, so we can use the simpler while -statement:

The while -statement executes until its condition becomes false .

A test of a numeric value (e.g., while $({}^{*}{\mathfrak{p}})$ in count\_ ${\pmb x}()$ ) is equivalent to comparing the value to 0 (e.g., while $({\tt s p l}{=}0)$ ). A test of a pointer value (e.g., if (p) ) is equivalent to comparing the value to nullptr (e.g., if (p!=nullptr) ).
> 对一个数值的测试，例如 `while(*p)` 等价于和 `0` 比较，即 `while(*p!=0)`
> 对一个指针的测试，例如 `if(p)` 等价于和 `nullptr` 比较，即 `if(p!=nullptr)`

There is no ‘‘null reference.’’ A reference must refer to a valid object (and implementations assume that it does). There are obscure and clever ways to violate that rule; don’t do that.
## 1.8 Tests
$\mathrm{C++}$ provides a conventional set of statements for expressing selection and looping, such as if - statements, switch -statements, while -loops, and for -loops. For example, here is a simple function that prompts the user and returns a Boolean indicating the response:

```cpp
bool accept()
{
    cout << "Do you want to proceed";
    char answer;
    cin >> answer;

    if (answer == 'y')
        return ture;
    return false;
}
```

To match the $<<$ output operator (‘‘put to’’), the ${}>>{}$ operator (‘‘get from’’) is used for input; cin is the standard input stream (Chapter 10). The type of the right-hand operand of ${}>>{}$ determines what input is accepted, and its right-hand operand is the target of the input operation. The $\mathtt{\backslash n}$ character at the end of the output string represents a newline (§1.2.1).
> `>>` 运算符用于获取输出，本例中，`cin` 是标准输入流，运算符 `>>` 的右边的 operand 的类型取决于接收到的输入

Note that the definition of answer appears where it is needed (and not before that). A declara- tion can appear anywhere a statement can. The example could be improved by taking an n (for ‘‘no’’) answer into account:
> 声明语句可以在任何其他语句允许出现的地方出现

A switch -statement tests a value against a set of constants. Those constants, called case -labels, must be distinct, and if the value tested does not match any of them, the default is chosen. If the value doesn’t match any case -label and no default is provided, no action is taken .

We don’t hav e to exit a case by returning from the function that contains its switch -statement. Often, we just want to continue execution with the statement following the switch -statement. We can do that using a break statement. As an example, consider an overly clever, yet primitive, parser for a trivial command video game:

Like a for -statement (§1.7), an if -statement can introduce a variable and test it. For example:
> 和 `for` 语句类似，`if` 语句也可以引入一个变量并对其进行测试

```cpp
void do_something(vector<int>& v)
{
    if(auto n=v.size(); n!=0)
        // .. we get here if n!=0
}
```

Here, the integer n is defined for use within the if -statement , initialized with v.siz e () , and immedi- ately tested by the ${\mathfrak{n l}}{=}0$ condition after the semicolon. A name declared in a condition is in scope on both branches of the if -statement.
> 这里的整数 `n` 是为在 `if` 语句中使用而定义，并以 `v.size()` 初始化
> 定义在条件中的名称的作用域在 `if` 的两个分支中都有效

As with the for -statement, the purpose of declaring a name in the condition of an if -statement is to keep the scope of the variable limited to improve readability and minimize errors.
> 和 `for` 语句一样，在条件中声明名字的目的是限制作用域以提高可读性和减少错误

The most common case is testing a variable against 0 (or the nullptr ). To do that, simply leave out the explicit mention of the condition. For example:
> 如果 `if` 条件中的测试是相对于 `0` 或者 `nullptr` ，则测试语句可以忽略

```cpp
void do_something(vector<int>& v)
{
    if(auto n = v.size())
        // .. we get here if n!=0
}
```

Prefer to use this terser and simpler form when you can.
> 尽量多用这类精简语法
## 1.9 Mapping to Hardware
$\mathrm{C++}$ offers a direct mapping to hardware. When you use one of the fundamental operations, the implementation is what the hardware offers, typically a single machine operation. For example, adding two int s, $\tt x+y$ executes an integer add machine instruction.
> C++提供了对硬件的直接映射，当我们使用 fundamental operations，其实现就是硬件直接提供的，一般就是单个 machine operation

A $\mathrm{C++}$ implementation sees a machine’s memory as a sequence of memory locations into which it can place (typed) objects and address them using pointers:

A pointer is represented in memory as a machine address, so the numeric value of $\mathsf{p}$ in this figure would be 3 . If this looks much like an array (§1.7), that’s because an array is $\mathrm{C++}^{\ast}$ s basic abstrac- tion of ‘‘a contiguous sequence of objects in memory.’’

The simple mapping of fundamental language constructs to hardware is crucial for the raw low- level performance for which C and $\mathrm{C++}$ have been famous for decades. The basic machine model of C and $\mathrm{C++}$ is based on computer hardware, rather than some form of mathematics.
### 1.9.1 Assignment
An assignment of a built-in type is a simple machine copy operation. Consider:
> 内建类型的赋值就是简单的 machine copy operation

This is obvious. We can graphically represent that like this:

Note that the two objects are independent. We can change the value of $\mathsf{y}$ without affecting the value of $\pmb{x}$ . For example $\scriptstyle{\pmb{\chi}=\pmb{99}}$ will not change the value of y . Unlike Java, C# , and other languages, but like C, that is true for all types, not just for int s.

If we want different objects to refer to the same (shared) value, we must say so. We could use pointers:

We can represent that graphically like this:

I arbitrarily chose 88 and 92 as the addresses of the int s. Again, we can see that the assigned-to object gets the value from the assigned object, yielding two independent objects (here, pointers), with the same value. That is, ${\tt p}{\tt=}{\tt q}$ gives ${\tt p}{\tt==}{\tt q}$ . After ${\tt p}{\tt=}{\tt q}$ , both pointers point to y .

A reference and a pointer both refer/point to an object and both are represented in memory as a machine address. However, the language rules for using them differ. Assignment to a reference does not change what the reference refers to but assigns to the referenced object:

We can represent that graphically like this:

To access the value pointed to by a pointer, you use $^*$ ; that is automatically (implicitly) done for a reference.

After $\tt x{=}y$ , we have `x==y` for every built-in type and well-designed user-defined type (Chapter 2) that offers $=$ (assignment) and $==$ (equality comparison).
### 1.9.2 Initialization
Initialization differs from assignment. In general, for an assignment to work correctly, the assigned-to object must have a value. On the other hand, the task of initialization is to make an uninitialized piece of memory into a valid object. For almost all types, the effect of reading from or writing to an uninitialized variable is undefined. For built-in types, that’s most obvious for refer- ences:
> 初始化和赋值是不同的
> 总的来说，如果赋值需要正常工作，则被赋值的对象必须有一个值，另一方面，初始化的工作是让一块未被初始化的内存变为一个有效的对象
> 对于几乎所有的类型，对未被初始化的变量的读和写都是未定义的
> 对于内建类型，引用最为明显 ：

```cpp
int x = 7;
int& r {x}; // bind r to x(r refers to x)
r = 7; // assign to whatever r refers to

int& r2; //error: uninitialized reference
r2 = 99; // assign to whatever r2 refers to
```

Fortunately, we cannot have an uninitialized reference; if we could, then that $r{\pmb2}{=}{\pmb9}{\pmb9}$ would assign 99 to some unspecified memory location; the result would eventually lead to bad results or a crash.
> 因为我们不能有未被初始化的引用，因此 `r2=99` 不会执行，否则就会向某个未指定的内存区域进行赋值

You can use $=$ to initialize a reference but please don’t let that confuse you. For example:
> 也可以使用 `=` 来初始化引用

```cpp
int& r = x; // bind r to x (r refers to x)
```

This is still initialization and binds r to $\pmb{x}$ , rather than any form of value copy.
> 这仍然是对 `r` 的初始化，将 `r` 绑定到 `x` ，而不是值拷贝

The distinction between initialization and assignment is also crucial to many user-defined types, such as string and vector , where an assigned-to object owns a resource that needs to eventually be released (§5.3).
> 对于用户定义的类型，初始化和赋值的差异也很重要，例如 `string/vector`

The basic semantics of argument passing and function value return are that of initialization (§3.6). For example, that’s how we get pass-by-reference.
> 函数的参数传递和值返回的基本语义是初始化，例如通过引用传参就是初始化了引用
## 1.10 Advice
The advice here is a subset of the $\mathrm{C++}$ Core Guidelines [Stroustrup, 2015]. References to guide- lines look like this [CG: ES. 23], meaning the $23\mathrm{rd}$ rule in the Expressions and Statement section. Generally, a core guideline offers further rationale and examples.

[1] Don’t panic! All will become clear in time; $\S1.1$ ; [CG: In. 0].
[2] Don’t use the built-in features exclusively or on their own. On the contrary, the fundamental (built-in) features are usually best used indirectly through libraries, such as the ISO $\mathrm{C++}$ standard library (Chapters 8–15); [CG: P.10].
[3] You don’t hav e to know every detail of $\mathrm{C++}$ to write good programs.
[4] Focus on programming techniques, not on language features.
[5] For the final word on language definition issues, see the ISO $\mathrm{C++}$ standard; $\S16.1.3$ ; [CG: P.2].
[6] ‘‘Package’’ meaningful operations as carefully named functions; $\S1.3$ ; [CG: F.1].
[7] A function should perform a single logical operation; $\S1.3$ [CG: F.2].
[8] Keep functions short; $\S1.3$ ; [CG: F.3].
[9] Use overloading when functions perform conceptually the same task on different types; $\S1.3$ .
[10] If a function may have to be evaluated at compile time, declare it constexpr ; $\S1.6$ ; [CG: F.4].
[11] Understand how language primitives map to hardware; $\S1.4$ , $\S1.7$ , $\S1.9$ , §2.3, §4.2.2, §4.4.
[12] Use digit separators to make large literals readable; $\S1.4$ ; [CG: NL. 11].
[13] Avoid complicated expressions; [CG: ES. 40].
[14] Avoid narrowing conversions; $\S1.4.2$ ; [CG: ES. 46].
[15] Minimize the scope of a variable; $\S1.5$ .
[16] Avoid ‘‘magic constants’’; use symbolic constants; $\S1.6$ ; [CG: ES. 45].
[17] Prefer immutable data; $\S1.6$ ; [CG: P.10].
[18] Declare one name (only) per declaration; [CG: ES. 10].
[19] Keep common and local names short, and keep uncommon and nonlocal names longer; [CG: ES. 7].
[20] Avoid similar-looking names; [CG: ES. 8].
[21] Avoid ALL_CAPS names; [CG: ES. 9].
[22] Prefer the $\{\}$ -initializer syntax for declarations with a named type; $\S1.4$ ; [CG: ES. 23].
[23] Use auto to avoid repeating type names; $\S1.4.2$ ; [CG: ES. 11].
[24] Avoid uninitialized variables; $\S1.4$ ; [CG: ES. 20].
[25] Keep scopes small; $\S1.5$ ; [CG: ES. 5].
[26] When declaring a variable in the condition of an if -statement, prefer the version with the implicit test against 0 ; $\S1.8$ .
[27] Use unsigned for bit manipulation only; $\S1.4$ ; [CG: ES. 101] [CG: ES. 106].
[28] Keep use of pointers simple and straightforward; $\S1.7$ ; [CG: ES. 42].
[29] Use nullptr rather than 0 or NULL ; $\S1.7$ ; [CG: ES. 47].
[30] Don’t declare a variable until you have a value to initialize it with; §1.7, §1.8; [CG: ES. 21].
[31] Don’t say in comments what can be clearly stated in code; [CG: NL. 1].
[32] State intent in comments; [CG: NL. 2].
[33] Maintain a consistent indentation style; [CG: NL. 4].

> 建议：
> 1. Don't panic!
> 2. 不要完全只有内建的特征，最好是通过库间接使用它们
> 3. 我们并不需要知道 C++ 的一切才能写出好程序
> 4. 集中于编程技巧，而不是语言特性
> 5. 语言问题的最终解释权在 ISO C++标准
> 6. 将有意义的操作打包为仔细命名的函数
> 7. 一个函数应该执行单个逻辑操作
> 8. 保持函数简短
> 9. 在不同的类型执行概念上相同任务时，使用重载
> 10. 如果函数需要在编译时评估，使用 `constexpr`
> 11. 理解语言原语如何映射到硬件
> 12. 使用 digital seperators 让大的字面值更可读
> 13. 避免复杂表达式
> 14. 避免损失精度的类型转换
> 15. 最小化变量的作用域
> 16. 避免“魔法常量”，使用符号常量
> 17. 偏好不可变数据
> 18. 每一次声明仅声明一个名字
> 19. 保持常用和局部的名字较短，保持不常用和非局部的名字更长
> 20. 避免看起来类似的名字
> 21. 避免全大写的名字
> 22. 对于有名字的类型进行初始化时，偏好 `{}` 语法
> 23. 使用 `auto` 避免重复的类型名称
> 24. 避免未初始化的变量
> 25. 保持作用域 small
> 26. 在 `if` 语句中声明变量时，偏好隐式 test against 0 的版本
> 27. 仅在位操作时使用 `unsigned`
> 28. 保持指针的使用简单且直接
> 29. 使用 `nullptr` 而不是 `0/NULL`
> 30. 在没有准备好初始化值时，不要声明变量
# 2 User-Defined Types
## 2.1 Introduction
We call the types that can be built from the fundamental types (§1.4), the const modifier (§1.6), and the declarator operators (§1.7) built-in types . $\mathrm{C++}^{\prime}$ s set of built-in types and operations is rich, but deliberately low-level. They directly and efficiently reﬂect the capabilities of conventional com- puter hardware. However, they don’t provide the programmer with high-level facilities to conve- niently write advanced applications. Instead, $\mathrm{C++}$ augments the built-in types and operations with a sophisticated set of abstraction mechanisms out of which programmers can build such high-level facilities.
> 我们称可以从 fundamental 类型、 `const` 修饰符、 declarator operator 中构建的类型为内建类型

The $\mathrm{C++}$ abstraction mechanisms are primarily designed to let programmers design and imple- ment their own types, with suitable representations and operations, and for programmers to simply and elegantly use such types. Types built out of other types using $\mathbf{C++}^{\dagger}$ ’s abstraction mechanisms are called user-defined types . They are referred to as classes and enumerations . User defined types can be built out of both built-in types and other user-defined types. Most of this book is devoted to the design, implementation, and use of user-defined types. User-defined types are often preferred over built-in types because they are easier to use, less error-prone, and typically as efficient for what they do as direct use of built-in types, or even faster.
> C++提供了抽象机制，帮助程序员设计和实现自己的类型，即用户定义的类型，一般指的是 classes 和 enumerations
> 相对于内建类型，我们一般偏好用户定义的类型

The rest of this chapter presents the simplest and most fundamental facilities for defining and using types. Chapters 4–7 are a more complete description of the abstraction mechanisms and the programming styles they support. Chapters 8–15 present an overview of the standard library, and since the standard library mainly consists of user-defined types, they provide examples of what can be built using the language facilities and programming techniques presented in Chapters 1–7.
## 2.2 Structures
The first step in building a new type is often to organize the elements it needs into a data structure, a struct :

```cpp
struct Vector {
    int sz;
    double* elem;
};
```

This first version of Vector consists of an int and a double ∗ 
A variable of type Vector can be defined like this:

```cpp
Vector v;
```

However, by itself that is not of much use because v ’s elem pointer doesn’t point to anything. For it to be useful, we must give $\mathbf{v}$ some elements to point to. For example, we can construct a Vector like this:

```cpp
void vector_init(Vector& v, int s)
{
    v.elem = new double[s]; // allocate an array of s doubles
    v.sz = s;
}
```

That is, v’s elem member gets a pointer produced by the new operator and v’s sz member gets the number of elements. The & in Vector& indicates that we pass v by non-const reference (§1.7); that way, vector_init() can modify the vector passed to it.

The new operator allocates memory from an area called the free store (also known as dynamic memory and heap). Objects allocated on the free store are independent of the scope from which they are created and ‘‘live’’ until they are destroyed using the delete operator (§4.2.2). 
> 分配于堆中的对象独立于它们所被创建的作用域，并会存活到被 `delete` 运算符摧毁

A simple use of Vector looks like this:

```cpp
double read_and_sum(int s) {
    // read integers from cin and return their sum; s is assumed to be positive

    Vector v;
    vector_init(v, s); // allocate s elements for v

    for (int i = 0; i!=s; i++)
        cin>>v.elem[i];

    double sum = 0;
    for (int i = 0; i!=s; i++)
        sum += v.elem[i];
    return sum;
}
```


There is a long way to go before our Vector is as elegant and ﬂexible as the standard-library vector . In particular, a user of Vector has to know every detail of Vector ’s representation. The rest of this chapter and the next two gradually improve Vector as an example of language features and tech- niques. Chapter 11 presents the standard-library vector , which contains many nice improvements.

I use vector and other standard-library components as examples
- to illustrate language features and design techniques, and 
- to help you learn and use the standard-library components.

Don’t reinvent standard-library components such as vector and string ; use them. 

We use . (dot) to access struct members through a name (and through a reference) and $->$ to access struct members through a pointer. For example: 

```cpp
void f(Vector v, Vector& rv, Vector* pv) {
    int i1 = v.sz; // access through name
    int i2 = rv.sz; // access through reference
    int i3 = pv->sz; // access through pointer
}
```
## 2.3 Classes
Having the data specified separately from the operations on it has advantages, such as the ability to use the data in arbitrary ways. However, a tighter connection between the representation and the operations is needed for a user-defined type to have all the properties expected of a ‘‘real type.’’ In particular, we often want to keep the representation inaccessible to users so as to ease use, guaran- tee consistent use of the data, and allow us to later improve the representation. To do that we have to distinguish between the interface to a type (to be used by all) and its implementation (which has access to the otherwise inaccessible data). The language mechanism for that is called a class . A class has a set of members , which can be data, function, or type members. The interface is defined by the public members of a class, and private members are accessible only through that interface. For example:
> 我们希望对用户隐藏表示，提供接口
> 我们需要区分类型的接口和实现，对应的语言机制称为类，类的接口由类的 `public` 成员定义，而 `private` 成员必须通过这一接口访问

```cpp
class Vector {
public:
    Vector(int s) : elem{new double[s]}, sz{s} {} // construct a Vector
    double& operator[] (int i) { return elem[i]; } // element access: subscripting
    int size() { return sz; }
private:
    double* elem; // pointer to the elements
    int sz; // the number of elements
};
```

Given that, we can define a variable of our new type Vector :

```cpp
Vector v(6); // a Vector with 6 elements
```

We can illustrate a Vector object graphically:

Basically, the Vector object is a ‘‘handle’’ containing a pointer to the elements ( elem ) and the num- ber of elements ( sz ). The number of elements (6 in the example) can vary from Vector object to Vector object, and a Vector object can have a different number of elements at different times (§4.2.3). However, the Vector object itself is always the same size. This is the basic technique for handling varying amounts of information in $\mathrm{C++}$ : a fixed-size handle referring to a variable amount of data ‘‘elsewhere’’ (e.g., on the free store allocated by new ; $\S4.2.2)$ . How to design and use such objects is the main topic of Chapter 4.
> `Vector` 对象本身的大小是固定的，但不同的 `Vector` 对象可以指向不同大小的数据
> C++处理 varying amounts of information 的基本技术是：一个固定大小的 handle 指向另一个地方的可变数量的数据

Here, the representation of a Vector (the members elem and sz ) is accessible only through the interface provided by the public members: Vector () , operator[]() , and siz e () . The read_and_sum () example from $\S2.2$ simplifies to:
> 此时 `Vector` 的表示（成员 `elem` 和 `sz` ）仅可以通过由 `public` 成员提供的接口访问：`Vector()` , `operator[]()`, `size()`

```cpp
double read_and_sum(int s) {
    Vector v(s);
    for (int i = 0; i!=v.size(); i++)
        cin>>v[i];

    double sum = 0;
    for (int i = 0; i!=v.size(); i++)
        sum+=v[i];
    return sum;
}
```

A member ‘‘function’’ with the same name as its class is called a constructor , that is, a function used to construct objects of a class. So, the constructor, Vector () , replaces vector_init () from $\S2.2$ . Unlike an ordinary function, a constructor is guaranteed to be used to initialize objects of its class. Thus, defining a constructor eliminates the problem of uninitialized variables for a class.
> 和类名称相同的成员函数称为构造函数，用于构造该类的对象
> 构造函数保证会被用于初始化该类的对象，定义构造函数就可以解决类成员未初始化的问题

Vector (int) defines how objects of type Vector are constructed. In particular, it states that it needs an integer to do that. That integer is used as the number of elements. The constructor initializes the Vector members using a member initializer list:
> 本例中，构造函数通过初始化列表来初始化类成员

```cpp
: elem{new double[s]}, sz{s}
```

That is, we first initialize elem with a pointer to s elements of type double obtained from the free store. Then, we initialize sz to s .

Access to elements is provided by a subscript function, called operator[] . It returns a reference to the appropriate element (a double& allowing both reading and writing).

The `size()` function is supplied to give users the number of elements.

Obviously, error handling is completely missing, but we’ll return to that in $\S3.5$ . Similarly, we did not provide a mechanism to ‘‘give back’’ the array of double s acquired by new ; $\S4.2.2$ shows how to use a destructor to elegantly do that.

There is no fundamental difference between a struct and a class ; a struct is simply a class with members public by default. For example, you can define constructors and other member functions for a struct .
> `struct/class` 直接并没有本质的差别，`struct` 仅仅是所有成员默认为 `public` 的 `class` 
> 我们同样可以为 `struct` 定义构造函数以及其他成员函数
## 2.4 Unions
A union is a struct in which all members are allocated at the same address so that the union occupies only as much space as its largest member. Naturally, a union can hold a value for only one member at a time. 
> `union` 是一类特殊的 `struct` ，其中所有的成员都分配在同一个地址，因此 `union` 占据的空间仅仅和它最大的成员占据的空间相同

For example, consider a symbol table entry that holds a name and a value. The value can either be a Node ∗ or an int :

```cpp
enum Type {ptr, num};// a Type can hold values ptr and num (§2.5)

struct Entry {
    string name; // string is a standard library type
    Type t;
    Node* p; // use p if p==ptr
    int i; // use i if t==num
};

void f(Entry* pe) {
    if (pe->t == num)
        cout << pe->i;
}
```

The members p and i are never used at the same time, so space is wasted. 
> 上例中，成员 `p/i` 永远不会被同时使用，因此浪费了空间

It can be easily recovered by specifying that both should be members of a union , like this:

```cpp
union Value {
    Node* p;
    int i;
};
```

The language doesn’t keep track of which kind of value is held by a union , so the programmer must do that:
> C++不会追踪 union 存储的是哪一类值

```cpp
struct Entry {
    string name;
    Type t;
    Value v; // ues v.p if t==ptr; use v.i is t==num
}

void f(Entry* pe) {
    if (pe->t == num)
        cout << pe->v.i
}
```

Maintaining the correspondence between a *type field* (here, t ) and the type held in a union is error- prone. To avoid errors, we can enforce that correspondence by encapsulating the union and the type field in a class and offer access only through member functions that use the union correctly. At the application level, abstractions relying on such *tagged unions* are common and useful. The use of ‘‘naked’’ union s is best minimized.
> 维护一个类型域和 union 中存储的类型是很易错的
> 为了避免错误，我们可以将 union 和类型域封装在类中，并且仅仅通过成员函数提供正确使用 union 的访问
> 这类 tagged union 在应用层次的使用是很常见的，naked union 的使用应该最小化

The standard library type, `variant` , can be used to eliminate most direct uses of unions. A `variant` stores a value of one of a set of alternative types (§13.5.1). For example, a `variant<Node ∗ ,int>` can hold either a Node ∗ or an int .
> 标准库类型 `variant` 可以用于提到大多数 union 的直接使用

Using variant , the Entr y example could be written as: 

```cpp
struct Entry {
    string name;
    variant<Node*, int> v;
};

void f(Entry* pe) {
    if(holds_alternative<int>(pe->v)) //does *pe hold an int?
        cout << get<int>(pe->v);
}
```

For many uses, a variant is simpler and safer to use than a union .
> 多数情况下，`variant` 比 union 更简单且安全
## 2.5 Enumerations
In addition to classes, $\mathrm{C++}$ supports a simple form of user-defined type for which we can enumerate the values:
> 除了类以外，C++还支持一类用户定义类型：枚举类型

```cpp
enum class Color {red, blue, green};
enum class Traffic_light {green, yellow, red};

Color col = Color::red;
Traffic_light light = Traffic_ligth::red
```

Note that enumerators (e.g., red ) are in the scope of their enum class , so that they can be used repeatedly in different enum class es without confusion. For example, Color:: red is Color ’s red which is different from Traffic_light:: red .
> 枚举变量例如 `red` 处于它们的枚举类的作用域内，因此不同的枚举类中可以使用相同的名字

Enumerations are used to represent small sets of integer values. They are used to make code more readable and less error-prone than it would have been had the symbolic (and mnemonic) enu- merator names not been used.
> 枚举类型一般用于表示一个小的整数集合，用于编写更可读和更不易错的代码，因为枚举类型使用的是符号名称

The class after the enum specifies that an enumeration is strongly typed and that its enumerators are scoped. Being separate types, enum class es help prevent accidental misuses of constants. In particular, we cannot mix Traffic_light and Color values:
> `enum` 之后的 `class` 指定了枚举是强类型的，以及它的枚举变量是位于自身作用域内的，`enum class` 实际上是 C++11引入的新枚举类型，它用于避免常量的意外误用
> 传统的枚举值名称在全局作用域可见，并且类型不安全，可以被赋值为任意整数值以及与其他整数值进行比较和计算

```cpp
Color x= red; // error: which red?
Color y = Traffic_light::red; // error: that red is not a Color
Color z = Color::red; // OK
```

Similarly, we cannot implicitly mix Color and integer values:

```cpp
int i = Color::red; // error: Color::red is not an int
Color c = 3; // initialization error: 3 is not a Color
```

Catching attempted conversions to an enum is a good defense against errors, but often we want to initialize an enum with a value from its underlying type (by default, that’s int ), so that’s allowed, as is explicit conversion from the underlying type:
> 使用 `enum` 类型的内在的类型值（默认是 `int`）来初始化 `enum` 是允许的

```cpp
Color x = Color{5}; // OK, but verbose
Color y {6}; // also OK
```

By default, an enum class has only assignment, initialization, and comparisons (e.g., $==\mathrm{and}<;\S1.4)$ defined. However, an enumeration is a user-defined type, so we can define operators for it:
> 默认情况下，`enum class` 仅定义了赋值、初始化和比较
> 但枚举类型也是用户定义的类型，我们可以为它定义运算符

```cpp
Traffic_light& operator++(Traffic_light& t)
    // prefic increment;
{
    switch(t) {
        case Traffic_light::green: return t = Traffic_light::yellow;
        case Traffic_light::yellow: return t = Traffic_light::red;
        case Traffic_light::red: return t= Traffic_light::green;
    }

}

Traffic_light next = ++light; // next becomes Traffic_light::green
```

If you don’t want to explicitly qualify enumerator names and want enumerator values to be int s (without the need for an explicit conversion), you can remove the class from enum class to get a ‘‘plain’’ enum . The enumerators from a ‘‘plain’’ enum are entered into the same scope as the name of their enum and implicitly converts to their integer value. For example:
> 如果我们不想显式限定枚举名称，并且希望枚举值为 `int` （即不再需要显式的转换），我们可以使用 `enum`
> `enum` 中的枚举名称的作用域和 `enum` 的名称的作用域相同，并且它们的名字会被隐式转化为整数值

```cpp
enum Color { red, green, blue };
int col = green;
```

Here col gets the value 1 . By default, the integer values of enumerators start with 0 and increase by one for each additional enumerator. The ‘‘plain’’ enum s hav e been in $\mathrm{C++}$ (and C) since the earliest days, so even though they are less well behaved, they are common in current code.
> 默认情况下，枚举名称的值从0开始，依次加1
## 2.6 Advice
[1] Prefer well-defined user-defined types over built-in types when the built-in types are too low- level; $\S2.1$ .
[2] Organize related data into structures ( struct s or class es); $\S2.2$ ; [CG: C.1].
[3] Represent the distinction between an interface and an implementation using a class ; $\S2.3$ ; [CG: C.3].
[4] A struct is simply a class with its members public by default; $\S2.3$ .
[5] Define constructors to guarantee and simplify initialization of class es; $\S2.3$ ; [CG: C.2].
[6] Avoid ‘‘naked’’ union s; wrap them in a class together with a type field; $\S2.4$ ; [CG: C.181].
[7] Use enumerations to represent sets of named constants; $\S2.5$ ; [CG: Enum. 2].
[8] Prefer class enum s over ‘‘plain’’ enum s to minimize surprises; $\S2.5$ ; [CG: Enum. 3].
[9] Define operations on enumerations for safe and simple use; $\S2.5$ ; [CG: Enum. 4].
> - 内建类型过于 low-level 时，偏好定义好的用户定义类型
> - 相关数据组织为结构 (`struct/class`)
> - 使用类来区分接口和实现
> - 定义构造函数，简化类的初始化
> - 避免 naked Union，将其包含在类内，类内维护类型域
> - 使用枚举表示有名称的常量
> - 偏好 `enum class` to `enum`
> - 为枚举定义运算
# 3 Modularity
## 3.1 Introduction
A $\mathrm{C++}$ program consists of many separately developed parts, such as functions (§1.2.1), user- defined types (Chapter 2), class hierarchies (§4.5), and templates (Chapter 6). 
The key to managing this is to clearly define the interactions among those parts. The first and most important step is to distinguish between the interface to a part and its implementation. 
> 管理 C++程序的分离开发的部分的关键是清晰定义它们之间的交互，其中重要的一步就是区分接口和实现

At the language level, $\mathrm{C++}$ represents interfaces by declarations. A declaration specifies all that’s needed to use a function or a type. For example:
> 在语言层面，C++使用声明表示接口
> 声明指定了一个函数或者一个类型所需要使用的所有东西，例如：

```cpp
double sqrt(double); // the square root function takes a double and return a double

class Vector {
public:
    Vector(int s);
    double& operator[](int i);
    int size();
private:
    double* elem; // elem points to an arrya of sz doubles
    int sz;
};
```

The key point here is that the function bodies, the function definitions , are ‘‘elsewhere.’’ For this example, we might like for the representation of Vector to be ‘‘elsewhere’’ also, but we will deal with that later (abstract types; $\S4.3)$ . The definition of sqrt () will look like this:
> 处理了声明之后，我们要聚焦的关键就是函数体/函数定义，函数定义是 “elsewhere” 的，也就是和声明处于不同的地方

```cpp
double sqrt(double d) // definition of sqrt()
{
    // ..algorithm as found in math textbook..
}
```

For `Vector` , we need to define all three member functions: 

```cpp
Vector::Vector(int s) // definition of the constructor
    :elem{new double[s]}, sz{s}
{
}

double& Vector::operator[](int i) // definition of subscripting
{
    return elem[i];
}

int Vector::size() // definition of size()
{
    return sz;
}
```

We must define Vector ’s functions, but not sqr t () because it is part of the standard library. Howev er, that makes no real difference: a library is simply ‘‘some other code we happen to use’’ written with the same language facilities we use.
> C++的库也就是 C++写的一些 facility

There can be many declarations for an entity, such as a function, but only one definition.
> 一个实体（例如函数）可以有许多声明，但仅能有一个定义
## 3.2 Separate Compilation
$\mathrm{C++}$ supports a notion of separate compilation where user code sees only declarations of the types and functions used. The definitions of those types and functions are in separate source files and are compiled separately. This can be used to organize a program into a set of semi-independent code fragments. Such separation can be used to minimize compilation times and to strictly enforce seperation of logically distinct parts of a program (thus minimizing the chance of errors). A library is often a collection of separately compiled code fragments (e.g., functions).
> C++支持分离编译，其中用户代码仅见到所需要使用的类型和函数的声明
> 而这些类型和函数的定义则在分离的源文件中，被分离编译
> 这帮助我们将程序组织为多个半独立的代码片段，也可以用于最小化编译时间，以及严格实现程序中的逻辑上不同的部分的分离（以此最小化错误的出现）
> 库常常就是一系列分离编译的代码片段（例如函数）

Typically, we place the declarations that specify the interface to a module in a file with a name indicating its intended use. For example:
> 一般我们会将指定了对于一个模块接口的全部声明放在一个头文件中

```cpp
// Vector.h
class Vector {
public:
    Vector(int s);
    double& operator[](int i);
    int size();
private:
    double* elem; // elem points to an array of sz doubles
    int sz;
};
```

This declaration would be placed in a file Vector. h . Users then include that file, called a header file , to access that interface. For example:
> 用户通过 include 该头文件来访问该接口

```cpp
// user.cpp
#include "Vector.h" // get Vector's interface
#include <cmath> // get the standard-library math function interface including sqrt()

double sqrt_sum(Vector& v)
{
    double sum = 0;
    for(int i = 0; i!=v.size(); i++)
        sum += std::sqrt(v[i]);
    return sum;
}
```

To help the compiler ensure consistency, the .cpp file providing the implementation of Vector will also include the .h file providing its interface:
> 为了帮助编译器保证一致性，提供了 `Vector` 实现的 `.cpp` 文件也需要 include 提供了它的接口的头文件

```cpp
// Vector.cpp
#include "Vector.h" // get Vector's interface
Vector::Vector(int s) // intialize members
    :elem{new double[s]}, sz{s}
{
}

double& Vector::operator[](int i)
{
    return elem[i];
}

int Vector::size()
{
    return sz;
}
```

The code in user. cpp and Vector. cpp shares the Vector interface information presented in Vector. h , but the two files are otherwise independent and can be separately compiled. Graphically, the program fragments can be represented like this:
> `user.cpp/Vector.cpp` 中的代码共享了定义于 `Vector.h` 中的 Vector 接口，但这两个文件事实上是分离的，因此可以被分别编译

![[A Tour of C++-Fig2.png]]

Strictly speaking, using separate compilation isn’t a language issue; it is an issue of how best to take advantage of a particular language implementation. However, it is of great practical importance. The best approach to program organization is to think of the program as a set of modules with well-defined dependencies, represent that modularity logically through language features, and then exploit the modularity physically through files for effective separate compilation.
> 严格地说，使用分离编译并不是一个和语言相关地问题，而是一个如何最好的利用某个特定语言实现的问题
> 编程组织的最好方法是将程序思考为一系列具有良定义的依赖的模块，然后通过语言特性逻辑上表示该模块性，以及在物理上通过文件和分离编译利用该模块性

A .cpp file that is compiled by itself (including the h files it `#include` s) is called a translation unit . A program can consist of many thousand translation units.
> 一个单独编译的 `.cpp` 文件（包括它 `#include` 的 `.h` 文件）被称为一个编译单元，一个程序可以有数千个编译单元构成
## 3.3 Modules(C++20)
The use of `#include` s is a very old, error-prone, and rather expensive way of composing programs out of parts. If you `#include` header. h in 101 translation units, the text of header. h will be processed by the compiler 101 times. If you `#include` header1. h before header2. h the declarations and macros in header1. h might affect the meaning of the code in header2. h . If instead you `#include` header2. h before header1. h , it is header2. h that might affect the code in header1. h . Obviously, this is not ideal, and in fact it has been a major source of cost and bugs since 1972 when this mechanism was first introduced into C.
> 如果我们在101个编译单元中 `#include <header.h>` ，则 `header.h` 中的文本会被编译器处理101次
> 如果在 `header1.h` include 之后再 include `header2.h`，则 `header1.h` 中定义的宏和声明可能会影响 `header2.h` 中的内容

We are finally about to get a better way of expressing physical modules in $\mathrm{C++}$ . The language feature, called module s is not yet ISO $\mathrm{C++}$ , but it is an ISO Technical Specification [ModulesTS] and will be part of $_{\mathrm{C++20}}$ . Implementations are in use, so I risk recommending it here even though details are likely to change and it may be years before everybody can use it in production code. Old code, in this case code using `#include` , can ‘‘live’’ for a very long time because it can be costly and time consuming to update.
> `module` 特性暂时不是 ISO C++，但是是 ISO Techinial Specification，并且将会是 C++20的一部分

Consider how to express the Vector and `sqrt_sum()` example from $\S3.2$ using `module` s:

```cpp
// file Vector.cpp:

module; // this compilation will define a module
    
// ... here we put stuff that Vector might need for its implementation ...

export module Vector; // defining the module called "Vector"

export class Vector {
public:
    Vector(int s);
    double& operator[](int i);
    int size ();
private:
    double ∗ elem; // elem points to an array of sz doubles
    int sz;
};

Vector::Vector (int s)
    :elem{new double[s]}, sz{s} // initialize members
{ 
}
double& Vector::operator[](int i)
{
    return elem[i];
} 
int Vector::size()
{
    return sz;
}

export int size(const Vector& v) {return v.size();}
```

This defines a module called Vector , which exports the class Vector , all its member functions, and the non-member function size () .
> 上例定义了一个名为 `Vector` 的 `module`，该模块导出了类 `Vector` ，包括了它的所有成员函数，还导出了一个函数 `size()`

The way we use this module is to import it where we need it. For example:
> 我们在需要的时候导入要使用的模块

```cpp
// file user.cpp
import Vector; // get Vector's interface
#include<cmath> // get the standard-library math function interface including sqrt()

double sqrt_sum(Vector& v)
{
    double sum = 0;
    for (int i = 0; i != v.size(); i++)
        sum += std::sqrt(v[i]);
    return sum;
}
```

I could have impor ed the standard library mathematical functions also, but I used the old-fashioned `#include` just to show that you can mix old and new. Such mixing is essential for gradually upgrading older code from using `#include` to import.
> 标准库也可以被导入，本例展示了 include 方法和 import 方法是可以混用的

The differences between headers and modules are not just syntactic. 
- A module is compiled once only (rather than in each translation unit in which it is used). 
- Two modules can be impored in either order without changing their meaning. 
- If you import something into a module, users of your module do not implicitly gain access to (and are not bothered by) what you imported: import is not transitive.

The effects on maintainability and compile-time performance can be spectacular.
> 模块和头文件之间的区别不仅仅在语法上：
> - 模块仅会被编译一次（而不是在每个被使用的编译模块中都被编译）
> - 两个模块可以用任意顺序 import 而不改变它们的语义
> - 如果我们向模块中 import 一些东西，我们的模块的用户是不会隐式地得到对于这些东西的访问权限的，也就是 import 不是可传递的
## 3.4 Namespaces
In addition to functions (§1.3), classes (§2.3), and enumerations $(\S2.5)$ , $\mathrm{C++}$ offers namespaces as a mechanism for expressing that some declarations belong together and that their names shouldn’t clash with other names. For example, I might want to experiment with my own complex number type $(\S4.2.1,\S14.4)$ :
> 除了函数、类、枚举以外，C++还提供了命名空间，来将一些声明放在一起，避免名字冲突

```cpp
namespace My_code {
    class complex {
        // ...
    };

    complex sqrt(complex);
    // ..

    int main();
}

int My_code::main()
{
    complex z {1,2};
    auto z2 = sqrt(z);
    std::cout << '{' << z2.real() << ',' << z2.imag() << '}\n';
}

int main()
{
    return My_code::main();
}
```

By putting my code into the namespace My_code , I make sure that my names do not conﬂict with the standard-library names in namespace std (§3.4). That precaution is wise, because the standard library does provide support for complex arithmetic $(\S4.2.1,\S14.4)$ . The simplest way to access a name in another namespace is to qualify it with the namespace name (e.g., std::cout and My_code::main ). The ‘‘real main () ’’ is defined in the global namespace, that is, not local to a defined namespace, class, or function. If repeatedly qualifying a name becomes tedious or distracting, we can bring the name into a scope with a using -declaration:
> 访问另一个命名空间中名字的最简单的方式就是使用命名空间修饰名字，例如 `std::cout`
> 真正的 `main()` 定义于全局命名空间
> 使用 `using` 声明可以将名字带入作用域

```cpp
void my_code(vector<int>& x, vector<int>& y)
{
    using std::swap; // use the standard-library swap
    // ...
    swap(x,y); // std::swap()
    other::swap(x,y); // some other swap()
}
```

A using-declaration makes a name from a namespace usable as if it was declared in the scope in which it appears. After using std:: swap , it is exactly as if swap had been declared in `my_code()` . To gain access to all names in the standard-library namespace, we can use a using -directive:
> `using` 声明让命名空间中的名字在直接当前作用域内可见
> `using namespace` 则让命名空间内所有名字在当前作用域内可见

```cpp
using namespace std;
```

A using-directive makes unqualified names from the named namespace accessible from the scope in which we placed the directive. So after the using -directive for std , we can simply write cout rather than std:: cout . By using a using-directive, we lose the ability to selectively use names from that namespace, so this facility should be used carefully, usually for a library that’s pervasive in an application (e.g., std ) or during a transition for an application that didn’t use `namespace` s.

Namespaces are primarily used to organize larger program components, such as libraries. They simplify the composition of a program out of separately developed parts.
> 命名空间主要用于组织更大的程序成分，例如库
> 它们将一个程序的组成简化为多个开发部分
## 3.5 Error Handling
Error handling is a large and complex topic with concerns and ramifications that go far beyond language facilities into programming techniques and tools. However, $\mathrm{C++}$ provides a few features to help. The major tool is the type system itself. Instead of painstakingly building up our applications from the built-in types (e.g., `char` , `int`, and `double` ) and statements (e.g., `if` , `while` and `for` ), we build types (e.g., `string` , `map` , and `regex` ) and algorithms (e.g., `sort()` , `find_if()` , and `draw_all()` ) that are appropriate for our applications. Such higher-level constructs simplify our programming, limit our opportunities for mistakes (e.g., you are unlikely to try to apply a tree traversal to a dialog box), and increase the compiler’s chances of catching errors. 
> C++提供了一些特征来帮助错误处理，主要的工具就是类型系统本身
> 构建应用时，我们应该多用合适的用户定义类型（例如 `string/map/regex` ）以及算法（`sort()/find_if()/draw_all()`），也有助于编译器捕获错误

The majority of $\mathrm{C++}$ language constructs are dedicated to the design and implementation of elegant and efficient abstractions (e.g., user-defined types and algorithms using them). One effect of such abstraction is that the point where a run-time error can be detected is separated from the point where it can be handled. As programs grow, and especially when libraries are used extensively, standards for handling errors become important. It is a good idea to articulate a strategy for error handling early on in the development of a program.
> C++语言的大部分构造都是为了设计和实现优雅且高效的抽象（例如用户定义的类型和使用它们的算法）
> 这种抽象的一个效果是，检测运行时错误的位置与处理这些错误的位置分离开来
> 随着程序的增长，特别是在大量使用库的情况下，处理错误的标准变得很重要
> 在程序开发的早期阶段明确一个错误处理策略是一个好主意
### 3.5.1 Exceptions
Consider again the Vector example. What ought to be done when we try to access an element that is out of range for the vector from $\S2.3?$

- The writer of Vector doesn’t know what the user would like to have done in this case (the writer of Vector typically doesn’t even know in which program the vector will be running).
- The user of Vector cannot consistently detect the problem (if the user could, the out-of-range access wouldn’t happen in the first place).

Assuming that out-of-range access is a kind of error that we want to recover from, the solution is for the Vector implementer to detect the attempted out-of-range access and tell the user about it. The user can then take appropriate action. 
> `Vector` 应该可以检测出越界访问，并且警告用户

For example, Vector::operator[]() can detect an attempted out-of-range access and throw an `out_of_range` exception:

```cpp
double& Vector::operator[](int i)
{
    if (i < 0 || size() <= i)
        throw out_of_range{"Vector::operator[]"};
    return elem[i];
}
```

The `throw` transfers control to a handler for exceptions of type out_of_rang e in some function that directly or indirectly called Vector::operator[]() . To do that, the implementation will unwind the function call stack as needed to get back to the context of that caller. That is, the exception handling mechanism will exit scopes and functions as needed to get back to a caller that has expressed interest in handling that kind of exception, invoking destructors (§4.2.2) along the way as needed. For example:
> `throw` 会将控制权移交给在某个直接或者间接调用了 `Vector::operator[]` 的函数的对于异常类型为 `our_of_range` 的错误处理程序
> 为此，需要可以回溯函数调用栈，以回到调用者的上下文信息，也就是说，异常处理机制会按照需要退出作用域和函数，回到具有处理该类型异常能力的调用者上，并且在回溯的过程中会按照需要调用析构函数

```cpp
void f(Vector& v)
{
    //...
    try { // exceptions here are handled be the handler define below
        v[v.size()] = 7; // try to access beyond the end of v
    }
    catch (out_of_range& err) {
        // ..handle range error...
        cerr << err.what() << '\n';
    }
}
```

We put code for which we are interested in handling exceptions into a try-block. The attempted assignment to `v[v.size()]` will fail. Therefore, the catch -clause providing a handler for exceptions of type out_of_range will be entered. The out_of_rang e type is defined in the standard library (in `<stdexcept>` ) and is in fact used by some standard-library container access functions.
> 我们将可能发生异常的语句放在了 `try` -block，`catch` -子句提供异常处理程序，处理类型为 `out_of_range` 的异常
> `out_of_range` 类型定义在标准库 `<stdexcept>` 中，并且实际上被一些标准库容器访问函数使用

I caught the exception by reference to avoid copying and used the what () function to print the error message put into it at the throw -point.
> 我们使用引用捕获异常，并且使用 `what()` 函数来打印在 `throw` 时给异常类型放入的错误信息

Use of the exception-handling mechanisms can make error handling simpler, more systematic, and more readable. To achieve that, don’t overuse tr y -statements. The main technique for making error handling simple and systematic (called Resource Acquisition Is Initialization; RAII ) is explained in $\S4.2.2$ . The basic idea behind RAII is for a constructor to acquire all resources necessary for a class to operate and have the destructor release all resources, thus making resource release guaranteed and implicit.
> 并不要过度使用 `try` 语句
> 让错误处理简单且系统的主要技巧是 RAII，RAII 的基本思想是让构造函数获取一个类所必须的全部资源，然后让析构函数释放全部资源

A function that should never throw an exception can be declared `noexcept` . 
> 应该永远不会抛出异常的函数可以被声明为 `noexcept` 

For example:

```cpp
void user(int sz) noexcept
{
    Vector v(sz);
    iota(&v[0], &v[sz], 1); // fill v with 1,2,3,4...
}
```

If all good intent and planning fails, so that `user()` still throws, `std:: terminate()` is called to immediately terminate the program.
> 如果被声明为 `noexcept` 的函数内抛出了异常，`std::terminate()` 会被直接调用，程序会被终止
### 3.5.2 Invariants
The use of exceptions to signal out-of-range access is an example of a function checking its argument and refusing to act because a basic assumption, a *precondition* , didn’t hold. 
> 使用异常来 signal 越界访问是函数检查参数，并且在其基本假设/预条件不满足时拒绝操作的一个实例

Had we formally specified Vector ’s subscript operator, we would have said something like ‘‘the index must be in the `[0 : size()` ) range,’’ and that was in fact what we tested in our operator[]() . The `[a:b` ) notation specifies a half-open range, meaning that a is part of the range, but b is not. Whenever we define a function, we should consider what its preconditions are and consider whether to test them (§3.5.3). For most applications it is a good idea to test simple invariants; see also $\S3.5.4$ .
> 当我们定义一个函数时，我们需要考虑该函数的预条件是什么，并且需要考虑是否需要测试它们
> 对于多数程序，我们最好测试简单的不变式

However, `operator[]()` operates on objects of type `Vector` and nothing it does makes any sense unless the members of `Vector` have ‘‘reasonable’’ values. In particular, we did say ‘‘ `elem` points to an array of $\mathtt{s z}$ doubles’’ but we only said that in a comment. Such a statement of what is assumed to be true for a class is called a class invariant , or simply an invariant . 
> 一个类的不变式是指在对象的声明周期中总是保持正确的条件，例如“`elem` 指向具有 `sz` 个 doubles 的数组”，无论何时调用对象的方法，都应该假设这一点是真的

It is the job of a constructor to establish the invariant for its class (so that the member functions can rely on it) and for the member functions to make sure that the invariant holds when they exit. Unfortunately, our Vector con- structor only partially did its job. It properly initialized the Vector members, but it failed to check that the arguments passed to it made sense. Consider:
> 建立类不变式的责任属于构造函数，其他成员函数要依赖于这个类不变式
> 其他成员函数需要保证它们退出时，类不变式是保持的
> 我们的 `Vector` 的构造函数正确初始化了成员，但没有正确检查参数，例如：

```cpp
Vector v(-27);
```

This is likely to cause chaos.
Here is a more appropriate definition: 

```cpp
Vector::Vector(int s)
{
    if (s < 0) 
        throw length_error{"Vector constructor: negative size"};
    elem = new double[s];
    sz = s;
}
```

I use the standard-library exception length_error to report a non-positive number of elements because some standard-library operations use that exception to report problems of this kind. If operator new can’t find memory to allocate, it throws a `std:: bad_alloc` . 
> 我们使用标准库异常 `length_error` 来报告负数的元素数量，C++的一些标准库也是这么做的
> 如果运算符 `new` 找不到可以分配的空闲内存空间，它会抛出 `std::bac_alloc`

We can now write:

```cpp
void test()
{
    try {
        Vector v(-27);
    }
    catch (std::length_error& err) {
        // handle negative size
    }
    catch (std::bad_alloc& err) {
        // handle memory exhaution
    }
}
```

You can define your own classes to be used as exceptions and have them carry arbitrary information from a point where an error is detected to a point where it can be handled (§3.5.1).
> 我们可以定义自己的异常类，并指定它们要为异常处理程序携带多少异常发生地的信息

Often, a function has no way of completing its assigned task after an exception is thrown. Then, ‘‘handling’’ an exception means doing some minimal local cleanup and rethrowing the exception. For example:
> 函数在发生异常时一般是难以再继续任务的，因此“处理”异常意味着进行一些 minimal local cleanup，然后重新抛出该异常

```cpp
void test()
{
    try {
        Vector v(-27);
    }
    catch (std::length_err& err) {
        // do something and rethrow
        cerr << "test failed: length error\n";
        throw; // rethrow
    }
    catch (std::bad_alloc& ) {
        // ouch! this program is not designed to handle memory exhausion
        std::terminate(); // terminate the program
    }
}
```

In well-designed code try-blocks are rare. Avoid overuse by systematically using the RAII technique $(\S4.2.2,\S5.3)$ .
> 如果代码设计良好，try-block 实际上会很少出现，因此我们需要系统地使用 RAII 技巧来避免过度使用 try-block

The notion of invariants is central to the design of classes, and preconditions serve a similar role in the design of functions. Invariants
-  help us to understand precisely what we want 
-  force us to be specific; that gives us a better chance of getting our code correct (after debugging and testing).
> 类的设计中，不变式是其中心思想，就例如预条件之于函数
> 类不变式帮助我们知道我们需要什么，并且聚焦到具体

The notion of invariants underlies C++'s notions of resource management is supported by constructors (Chapter 4) and destructors $(\S4.2.2,\S13.2)$ .
> C++通过构造函数和析构函数处理资源管理的类不变式
### 3.5.3 Error-Handling Alternatives
Error handling is a major issue in all real-world software, so naturally there are a variety of approaches. If an error is detected and it cannot be handled locally in a function, the function must somehow communicate the problem to some caller. Throwing an exception is $C++$ ’s most general mechanism for that.
> 错误在函数内发生，且不能直接处理时，函数需要一个与调用者沟通该错误的方式，C++中，最通用的方法就是抛出异常

There are languages where exceptions are designed simply to provide an alternate mechanism for returning values. $\mathrm{C++}$ is not such a language: exceptions are designed to be used to report failure to complete a given task. Exceptions are integrated with constructors and destructors to provide a coherent framework for error handling and resource management (§4.2.2, §5.3). Compilers are optimized to make returning a value much cheaper than throwing the same value as an exception.
> C++的编译器被设计为直接返回值要比抛出值相同的异常更高效，因此 C++中的异常并不是用于返回值的一个替代方式，而应该用于报告 failure

Throwing an exception is not the only way of reporting an error that cannot be handled locally. A function can indicate that it cannot perform its allotted task by:

- throwing an exception 
- somehow return a value indicating failure 
- terminating the program (by invoking a function like terminate() , exit() , or abort() ). 

> 一个函数通过：
> - 抛出异常
> - 返回 failure indicating value
> - 终止程序
> 来报告任务的失败

We return an error indicator (an ‘‘error code’’) when: 
- A failure is normal and expected. For example, it is quite normal for a request to open a file to fail (maybe there is no file of that name or maybe the file cannot be opened with the permissions requested). 
- An immediate caller can reasonably be expected to handle the failure. 
> 不是抛出异常/终止程序，而是仅返回 error indicator/error code 的情况有：
> - failure 是常见并且可预见的，例如打开文件失败
> - 存在一个中间调用者是预期可以处理该 failure 的

We throw an exception when: 
- An error is so rare that a programmer is likely to forget to check for it. For example, when did you last check the return value of printf () ?  
- An error cannot be handled by an immediate caller. Instead, the error has to percolate back to an ultimate caller. For example, it is infeasible to have every function in an application reliably handle every allocation failure or network outage. 
- New kinds of errors can be added in lower-modules of an application so that higher-level modules are not written to cope with such errors. For example, when a previously single-threaded application is modified to use multiple threads or resources are placed remotely to be accessed over a network. 
- No suitable return path for errors codes are available. For example, a constructor does not have a return value for a ‘‘caller’’ to check. In particular, constructors may be invoked for several local variables or in a partially constructed complex object so that clean-up based on error codes would be quite complicated. 
- The return path of a function is made more complicated or expensive by a need to pass both a value and an error indicator back (e.g., a pair ; $\S13.4.3$ ), possibly leading to the use of out- parameters, non-local error-status indicators, or other workarounds. 
- The error has to be transmitted up a call chain to an ‘‘ultimate caller.’’ Repeatedly checking an error-code would be tedious, expensive, and error-prone. 
- The recovery from errors depends on the results of several function calls, leading to the need to maintain local state between calls and complicated control structures. 
- The function that found the error was a callback (a function argument), so the immediate caller may not even know what function was called. 
- An error implies that some ‘‘undo action’’ is needed. 
> 抛出异常的情况：
> - 错误非常罕见，程序员也极少特意检查它（例如 `printf`）
> - 中间调用者无法处理，错误需要回溯到最高调用者（例如 allocation failure, network outage，它们不能期望每个中间调用者都有能力处理）
> - 应用的低层模块添加了新的错误类型，高层模块还没有写处理这些错误的代码（例如单线程程序改多线程，例如改为访问远程资源）
> - error code 没有合适的返回路径（例如构造函数）
> - 要同时返回 value 和 error indicator 时，函数的返回路径会过于复杂
> - error 需要被传递给最高的调用者，此时在回溯栈中反复检查 error code 会过于麻烦
> - error recovery 依赖于多个函数调用的结果，因此需要维护调用之间的局部状态
> - 出现 error 的函数是一个回调函数（作为参数的函数），因此直接调用者可能不知道到底调用了哪个函数
> - 出现的 error 表示此时需要一些“撤回操作”

We terminate when 
- An error is of a kind from which we cannot recover. For example, for many – but not all – systems there is no reasonable way to recover from memory exhaustion. 
- The system is one where error-handling is based on restarting a thread, process, or computer whenever a non-trivial error is detected.
> 终止程序的情况：
> - 出现了我们无法恢复的错误，例如大多数系统都无法从内存耗尽中恢复
> - 所处的系统在检测到 non-trivial 错误的时候，其恢复方式是重启线程、进程

One way to ensure termination is to add `noexcept` to a function so that a throw from anywhere in the function’s implementation will turn into a terminate () . Note that there are applications that can’t accept unconditional terminations, so alternatives must be used.
> 一种保证函数可以终止程序的方法是添加 `nonexcept`
> 此时在函数内部的 `throw` 都会导致 `terminate()` 

Don’t believe that all error codes or all exceptions are bad; there are clear uses for both. Furthermore, do not believe the myth that exception handling is slow; it is often faster than correct handling of complex or rare error conditions, and of repeated tests of error codes.

RAII (§4.2.2, §5.3) is essential for simple and efficient error-handling using exceptions. Code littered with try-blocks often simply reﬂects the worst aspects of error-handling strategies conceived for error codes.
> 要简单且高效地使用异常进行错误处理，RAII 是必要的
> 到处都是 try block 的代码实际上是最差的错误处理策略
### 3.5.4 Contracts
There is currently no general and standard way of writing optional run-time tests of invariants, pre-conditions, etc. A contract mechanism is proposed for C++20 [Garcia, 2016] [Garcia, 2018]. The aim is to support users who want to rely on testing to get programs right – running with extensive run-time checks – but then deploy code with minimal checks. This is popular in high-performance applications in organizations that rely on systematic and extensive checking.
> 目前没有通用和标准的方法来写对于不变式、预条件的可选的运行时测试
> C++20提出了合约机制，来支持以大量的运行时检查运行，以最少的检查部署

For now, we have to rely on ad hoc mechanisms. For example, we could use a command-line macro to control a run-time check: 
> 合约机制暂时未实现，我们需要依赖临时的机制
> 例如，使用命令行宏来控制运行时检查：

```cpp
double& Vector::operator[](int i)
{
    if (RANGE_CHECK && (i < 0 || size() <= i))
        throw out_of_range{"Vector::operator[]"};
    return elem[i];
}
```

The standard library offers the debug macro, `assert()`, to assert that a condition must hold at run time. For example:
> C++标准库提供了 debug 宏以及 `assert()`，用以断言某个条件必须在运行时成立

```cpp
void f(const char* p)
{
    assert(p != nullptr); // p must not be the nullptr
    // ..
}
```

If the condition of an `assert()` fails in ‘‘debug mode,’’ the program terminates. If we are not in debug mode, the `assert()` is not checked. That’s pretty crude and inﬂexible, but often sufficient.
> 如果在 debug 模式下，某个 `assert()` fail，程序终止
> 在非 debug 模式下，`assert()` 不会被检查
> 这种方法很粗略且不灵活，但往往够用
### 3.5.5 Static Assertions
Exceptions report errors found at run time. If an error can be found at compile time, it is usually preferable to do so. That’s what much of the type system and the facilities for specifying the interfaces to user-defined types are for. However, we can also perform simple checks on most properties that are known at compile time and report failures to meet our expectations as compiler error messages. For example:
> 异常用以报告运行时找到的错误，但如果在编译时能找到错误当然更好，而这就是类型系统所要做的事
> 并且，我们也可以在编译时对一些在编译时就知道的大多数性质进行简单的检查，然后以编译错误信息的方式报告 failure

```cpp
static_assert(4 <= sizeof(int), "integers are too small"); // check integer size
```

This will write integers are too small if $4\mathrm{<=}$ sizeof (int) does not hold; that is, if an int on this system does not have at least 4 bytes.  We call such statements of expectations *assertions* .

The static_assert mechanism can be used for anything that can be expressed in terms of constant expressions $(\S1.6)$ . 
> `static_assert` 机制可以用于任意可以表示为常量表达式的条件

For example:

```cpp
constexpr double C = 299792.458; // km/s
void f(double speed)
{
    constexpr double local_max = 160.0 / (60 * 60); // 160km/h == 160.0/(60*60) km/s

    static_assert(speed < C, "cant't go that fast"); // error: speed must be a constant

    static_assert(local_max < C, "cant't go that fast"); // OK
}
```

In general, `static_assert(A, S)` prints S as a compiler error message if A is not true .  If you don’t want a specific message printed, leave out the S and the compiler will supply a default message:
> 一般地，`static_arrset(A, S)` 会在 `A` 非 true 时打印编译器错误信息 `S`
> 如果 `S` 留空，则编译器会打印默认信息

```cpp
static_assert(4 <= sizeof(int)); // use default message
```

The default message is typically the source location of the static_asser plus a character representa- tion of the asserted predicate. 
> 默认信息一般就是 `static_assert` 在源文件中的位置，加上一个表示断言的谓词的字符

The most important uses of static_asser come when we make assertions about types used as parameters in generic programming $(\S7.2,\S13.9)$ .
> `static_assert` 在泛型编程时对作为形参的类型进行断言时非常有用
## 3.6 Function Arguments and Return Values
The primary and recommended way of passing information from one part of a program to another is through a function call. Information needed to perform a task is passed as arguments to a function and the results produced are passed back as return values. For example:
> 从程序的不同部分之间传递信息的推荐方式是通过函数调用
> 函数通过参数接受处理任务需要的信息，然后返回值

```cpp
int sum(const vector<int>& v)
{
    int s = 0;
    for (const int i : v)
        s += i;
    return s;
}

vector fib = {1, 2, 3, 5, 8, 13, 21};

int x = sum(fib); // x becomes 53
```

There are other paths through which information can be passed between functions, such as global variables (§1.5), pointer and reference parameters (§3.6.1), and shared state in a class object (Chapter 4). Global variables are strongly discouraged as a known source of errors, and state should typi- cally be shared only between functions jointly implementing a well-defined abstraction (e.g., mem- ber functions of a class; $\S2.3)$ .
> 函数之间传递信息的方式还有：
> 全局变量、指针和引用参数、类对象中的共享状态
> 全局变量不推荐使用，状态应该仅在共同实现了一个良定义的抽象的函数之间共享（例如类的成员函数）

Given the importance of passing information to and from functions, it is not surprising that there are a variety of ways of doing it. Ke y concerns are:

- Is an object copied or shared? i
- If an object is shared, is it mutable? 
- Is an object moved, leaving an ‘‘empty object’’ behind (§5.2.2)?
> 在函数之间传递信息是，需要考虑：
> - 对象是拷贝还是共享
> - 如果是共享，对象可变吗
> - 对象是否被移动，导致留下了“空对象”

The default behavior for both argument passing and value return is ‘‘copy’’ (§1.9), but some copies can implicitly be optimized to moves.
> 默认的实参传递和值返回的行为是“拷贝”（初始化），但一些拷贝可以被隐式优化为“移动”

In the sum() example, the resulting int is copied out of sum () but it would be inefficient and pointless to copy the potentially very large vector into sum () , so the argument is passed by reference (indicated by the & ; $\S1.7$ ).

The sum() has no reason to modify its argument. This immutability is indicated by declaring the vector argument const (§1.6), so the vector is passed by const reference.
### 3.6.1 Argument Passing
First consider how to get values into a function. By default we copy (‘‘pass-by-value’’) and if we want to refer to an object in the caller’s environment, we use a reference (‘‘pass-by-reference’’). For example:

```cpp
void test(vector<int> v, vector<int>& rv)
    // v is passed by value; rv is passed by reference
{
    v[1] = 99; // modify v(a local variable)
    rv[2] = 66; // modify whatever rv refers to
}

int main()
{
    vector fib = {1,2,3,5,8,13,21};
    test(fib, fib);
    cout << fib[1] << ' ' << fib[2] <<'\n'; // prints 2 66
}
```

When we care about performance, we usually pass small values by-value and larger ones by-refer- ence. Here ‘‘small’’ means ‘‘something that’s really cheap to copy.’’ Exactly what ‘‘small’’ means depends on machine architecture, but ‘‘the size of two or three pointers or less’’ is a good rule of thumb. 
> 大小小于两个或三个指针大小的对象可以称为 “small”

If we want to pass by reference for performance reasons but don’t need to modify the argument, we pass-by- const -reference as in the sum () example. This is by far the most common case in ordinary good code: it is fast and not error-prone. 
> 传递常引用是非常常见的 goob practice

It is not uncommon for a function argument to have a default value; that is, a value that is con- sidered preferred or just the most common. We can specify such a default by a default function argument . For example:
> 函数可以有默认实参

```cpp
void print(int value, int base = 10); // print value in base "base"

print(x, 16); // hexadecimal
print(x, 60); // sexagesimal
print(x); // use the default: decimal
```

This is a notationally simpler alternative to overloading:
> 可以把函数默认参数看为对以下重载代码的替代

```cpp
void print(int value, int base); // print value in base "base"

void print(int value)
    // print value in base 10
{
    print(value, 10);
}
```

### 3.6.2 Value Return
Once we have computed a result, we need to get it out of the function and back to the caller. Again, the default for value return is to copy and for small objects that’s ideal.  We return ‘‘by reference’’ only when we want to grant a caller access to something that is not local to the function. For example:
> 我们仅在我们能对于调用者保证返回的对象对于返回函数是非局部时才可以以引用返回

```cpp
class Vector {
public:
    // ...
    double& operator[](int i) { return elem[i];} // return reference to the i-th element
private:
    double* elem; // elem points to an array of sz
    // ...
}
```

The i th element of a Vector exists independently of the call of the subscript operator, so we can return a reference to it. 
> 例如 `operator[]` 的调用和 `elem[i]` 的存在是独立的，因此我们可以在 `operator[]` 中返回对于 `elem[i]` 的引用

On the other hand, a local variable disappears when the function returns, so we should not return a pointer or reference to it: 
> 局部变量在函数返回值即消失，因此我们不应该返回指向它的指针或者引用

```cpp
int& bad()
{
    int x;
    // ..
    return x; // bad: return a reference to the local variable x
}
```

Fortunately, all major $\mathrm{C++}$ compilers will catch the obvious error in bad() . Returning a reference or a value of a ‘‘small’’ type is efficient, but how do we pass large amounts of information out of a function? Consider:

```cpp
Matrix operator+(const Matrix& x, const Matrix& y)
{
    Matrix res;
    // ...for all res[i,j], res[i,j] = x[i,j] + y[i,j]
    return res;
}

Matrix m1, m2;
// ...
Matrix m3 = m1 + m2; // no copy
```

A Matrix may be very large and expensive to copy even on modern hardware. So we don’t copy, we give Matrix a move constructor (§5.2.2) and very cheaply move the Matrix out of operator+() . 
> 对于大的对象，我们需要写好移动构造函数，避免返回时的拷贝

We do not need to regress to using manual memory management:

```cpp
Matrix* add(const Matrix& x; const Matrix& y)
    // complicated and error-prone 20th century style
{
    Matrix* p = new Matrix;
    // for all *p[i,j], *p[i,j] = x[i,j] + y[i,j]
    return p;
}

Matrix m1, m2;
// ...
Matrix* m3 = m1 + m2; // just copy a pointer
// ...
delete m3; // easliy forgotten
```

Unfortunately, returning large objects by returning a pointer to it is common in older code and a major source of hard-to-find errors. Don’t write such code.
> 返回指针的做法已经过时，且易错，因此不推荐

 Note that operator+() is as efficient as add () , but far easier to define, easier to use, and less error-prone.
 > 注意 `operator+()` 和 `add()` 的效率是一致的，因此重载运算符是更好的实践

If a function cannot perform its required task, it can throw an exception (§3.5.1). This can help avoid code from being littered with error-code tests for ‘‘exceptional problems.’’

The return type of a function can be deduced from its return value. For example:
> 函数的返回类型可以从它的返回值中推导

```cpp
auto mul(int i, double d) { return i * d; } // here, auto means deduce the return type
```

This can be convenient, especially for generic functions (function templates; $\S6.3.1$ ) and lambdas (§6.3.3), but should be used carefully because a deduced type does not offer a stable interface: a change to the implementation of the function (or lambda) can change the type.
> 这对于泛型函数（函数模板）和匿名函数很方便，但注意这不会提供一个稳定的接口：对函数实现的改变会改变其返回类型
### 3.6.3 Structured Binding
A function can return only a single value, but that value can be a class object with many members. This allows us to efficiently return many values. For example:
> 可以利用结构体、类对象来从函数中返回多个值

```cpp
struct Entry {
    string name;
    int value;
};

Entry read_entry(istream& is) // naive read function, for a better version, be 10.5
{
   string s;
   int i;
   is >> s >> i;
   return {s, i};
}

auto e = read_entry(cin);
cout << e.name << e.value;
```

Here, {s, i} is used to construct the Entry return value. 
> 本例中，`{s, i}` 被用于构造 `Entry` 返回值

Similarly, we can ‘‘unpack’’ an Entry ’s members into local variables:
> 我们也可以解包 `Entry` 的成员到局部变量

```cpp
auto [n, v] = read_entry(is);
cout << n << v;
```

The `auto [n, v]` declares two local variables n and $\mathbf{v}$ with their types deduced from read_entry () ’s return type. This mechanism for giving local names to members of a class object is called structured binding .
> `auto [n, v]` 声明了两个局部变量 `n ,v` ，其类型从 `read_entry()` 的返回类型中推断
> 这类将类对象的成员赋予局部名字的机制称为结构化绑定

Consider another example:

```cpp
map<string, int> m;
// .. fill m
for (const auto [key, value]: m)
    cout << key << value;
```

As usual, we can decorate auto with const and & . For example:
> `auto` 类型也可以添加 `const` 和 `&` 修饰

```cpp
void incr(map<string, int>& m) // increment the value of each element of m
{
    for (auto& [key, value]: m)
        ++value;
}
```

When structured binding is used for a class with no private data, it is easy to see how the binding is done: there must be the same number of names defined for the binding as there are nonstatic data members of the class, and each name introduced in the binding names the corresponding member. There will not be any difference in the object code quality compared to explicitly using a composite object; the use of structured binding is all about how best to express an idea.
> 当结构化绑定用于没有 private 数据的类时，要求绑定的名字和类的非静态数据成员数量相同，结构化绑定并不会和显示使用一个合成对象在代码质量上有任何差异

It is also possible to handle classes where access is through member functions. For example:
> 对于需要通过成员函数访问的数据对象，结构化绑定也是可行的

```cpp
complex<double> z = {1, 2};
auto [re, im] = z + 2; // re = 3, im = 2
```

A complex has two data members, but its interface consists of access functions, such as `real()` and `imag()` . Mapping a `complex<double>` to two local variables, such as re and im is feasible and efficient, but the technique for doing so is beyond the scope of this book.
> `complex` 有两个数据成员，但其接口是两个访问函数 `real()/imag()`
## 3.7 Advice
[1] Distinguish between declarations (used as interfaces) and definitions (used as implementa- tions); $\S3.1$ .
[2] Use header files to represent interfaces and to emphasize logical structure; $\S3.2$ ; [CG: SF. 3].
[3] `#include` a header in the source file that implements its functions; $\S3.2$ ; [CG: SF. 5].
[4] Avoid non-inline function definitions in headers; $\S3.2$ ; [CG: SF. 2].
[5] Prefer module s over headers (where module s are supported); $\S3.3$ .
[6] Use namespaces to express logical structure; $\S3.4$ ; [CG: SF. 20].
[7] Use `using` -directives for transition, for foundational libraries (such as std ), or within a local scope; $\S3.4$ ; [CG: SF. 6] [CG: SF. 7].
[8] Don’t put a using -directive in a header file; $\S3.4$ ; [CG: SF. 7].
[9] Throw an exception to indicate that you cannot perform an assigned task; $\S3.5$ ; [CG: E.2].
[10] Use exceptions for error handling only; $\S3.5.3$ ; [CG: E.3].
[11] Use error codes when an immediate caller is expected to handle the error; $\S3.5.3$ .
[12] Throw an exception if the error is expected to percolate up through many function calls; $\S3.5.3$ .
[13] If in doubt whether to use an exception or an error code, prefer exceptions; $\S3.5.3$ .
[14] Develop an error-handling strategy early in a design; $\S3.5$ ; [CG: E.12].
[15] Use purpose-designed user-defined types as exceptions (not built-in types); $\S3.5.1$ .
[16] Don’t try to catch every exception in every function; $\S3.5$ ; [CG: E.7].
[17] Prefer RAII to explicit tr y -blocks; $\S3.5.1$ , $\S3.5.2$ ; [CG: E.6].
[18] If your function may not throw, declare it noexcept ; $\S3.5$ ; [CG: E.12].
[19] Let a constructor establish an invariant, and throw if it cannot; $\S3.5.2$ ; [CG: E.5].
[20] Design your error-handling strategy around invariants; $\S3.5.2$ ; [CG: E.4].
[21] What can be checked at compile time is usually best checked at compile time; $\S3.5.5$ [CG: P.4] [CG: P.5].
[22] Pass ‘‘small’’ values by value and ‘‘large‘‘ values by references; $\S3.6.1$ ; [CG: F.16].
[23] Prefer pass-by- const -reference over plain pass-by-reference; $\S3.6.1$ ; [CG: F.17].
[24] Return values as function-return values (rather than by out-parameters); $\S3.6.2$ ; [CG: F.20] [CG: F.21].
[25] Don’t overuse return-type deduction; $\S3.6.2$ .
[26] Don’t overuse structured binding; using a named return type is often clearer documentation; $\S3.6.3$ .
> 1. 区分声明（用于接口）和定义（用于实现）
> 2. 使用头文件表示接口并强调逻辑结构
> 3. 在实现了函数的源文件中，`#include` 头文件
> 4. 避免头文件中的非内联函数定义
> 5. 如果有 `modules` ，偏好 modules
> 6. 使用命名空间来表达逻辑结构
> 7. 可以在局部作用域内使用 `using` 指令，或者对基础的库如 `std` 使用 `using`，或者用于过渡
> 8. 不要在头文件中使用 `using` 指令
> 9. 如果不能完成指定的任务，抛出一个异常
> 10. 仅在错误处理时抛出异常
> 11. 如果中间调用者可以处理错误，则使用 error code
> 12. 如果错误会传递经过过多函数调用，则抛出异常
> 13. 不知道该用异常还是 error code 时，使用异常
> 14. 在设计早期就开发一个错误处理策略
> 15. 使用设计好的用户定义类型作为异常，而不是内建类型
> 16. 不要在每个函数捕获每个异常
> 17. prefer RAII to try-blocks
> 18. 如果函数不会 throw，声明为 noexcept
> 19. 让构造函数建立不变式，如果不能构建的话， throw 异常
> 20. 围绕不变式设计错误处理策略
> 21. 能在编译时检查，就在编译时检查
> 22. “小”值使用值传递，“大”值使用引用传递
> 23. 偏好常引用
> 24. 通过函数返回值返回值，而不是通过外部参数
> 25. 不要滥用返回类型推导
> 26. 不要滥用结构化绑定
# 4 Classes
## 4.1 Introduction
This chapter and the next three aim to give you an idea of $C++^{\prime}\mathrm{s}$ support for abstraction and resource management without going into a lot of detail:

- This chapter informally presents ways of defining and using new types ( user-defined types ). In particular, it presents the basic properties, implementation techniques, and language facilities used for concrete classes , abstract classes , and class hierarchies . 
- Chapter 5 presents the operations that have defined meaning in $\mathrm{C++}$ , such as constructors, destructors, and assignments. It outlines the rules for using those in combination to control the life cycle of objects and to support simple, efficient, and complete resource management. 
- Chapter 6 introduces templates as a mechanism for parameterizing types and algorithms with (other) types and algorithms. Computations on user-defined and built-in types are rep- resented as functions, sometimes generalized to template functions and function objects . 
- Chapter 7 gives an overview of the concepts, techniques, and language features that underlie generic programming. The focus is on the definition and use of concepts for precisely specifying interfaces to templates and guide design. Variadic templates are introduced for specifying the most general and most ﬂexible interfaces.

These are the language facilities supporting the programming styles known as object-oriented programming and generic programming . 
> Chapter4-7介绍的是 C++对于面向对象编程和泛型编程的支持

Chapters 8–15 follow up by presenting examples of standard-library facilities and their use.

The central language feature of $\mathrm{C++}$ is the class . A class is a user-defined type provided to represent a concept in the code of a program. Whenever our design for a program has a useful concept, idea, entity, etc., we try to represent it as a class in the program so that the idea is there in the code, rather than just in our heads, in a design document, or in some comments. A program built out of a well-chosen set of classes is far easier to understand and get right than one that builds everything directly in terms of the built-in types. In particular, classes are often what libraries offer.

Essentially all language facilities beyond the fundamental types, operators, and statements exist to help define better classes or to use them more conveniently. By ‘‘better,’’ I mean more correct, easier to maintain, more efficient, more elegant, easier to use, easier to read, and easier to reason about. Most programming techniques rely on the design and implementation of specific kinds of classes. The needs and tastes of programmers vary immensely. Consequently, the support for classes is extensive. Here, we will just consider the basic support for three important kinds of classes:
> 我们仅考虑对于三种类：具体类、抽象类、类层次中的类，的基本支持

- Concrete classes (§4.2) 
- Abstract classes (§4.3) 
- Classes in class hierarchies (§4.5)

An astounding number of useful classes turn out to be of one of these three kinds. Even more classes can be seen as simple variants of these kinds or are implemented using combinations of the techniques used for these.

## 4.2 Concrete Types
The basic idea of concrete classes is that they behave ‘‘just like built-in types.’’ For example, a complex number type and an infinite-precision integer are much like built-in int , except of course that they hav e their own semantics and sets of operations. Similarly, a vector and a string are much like built-in arrays, except that they are better behaved (§9.2, $\S10.3$ , $\S11.2)$ .
> 具体类的基本思想就是：它们的行为就像内建类型
> 例如复数类型、无限精度整数类型就类似于 int
> 类似地，向量和字符串也类似内建数组

The defining characteristic of a concrete type is that its representation is part of its definition. In many important cases, such as a vector , that representation is only one or more pointers to data stored elsewhere, but that representation is present in each object of a concrete class. That allows implementations to be optimally efficient in time and space. 
> 具体类型的一个定义特点就是它的表示也是它的定义的一部分，例如 vector 的表示实质上就是一个或者多个指向存储数据的指针，而对于 vector 类的每一个实例，该表示都成立

In particular, it allows us to

- place objects of concrete types on the stack, in statically allocated memory, and in other objects $(\S1.5)$ ; 
- refer to objects directly (and not just through pointers or references); 
- initialize objects immediately and completely (e.g., using constructors; $\S2.3)$ ; and 
- copy and move objects (§5.2).
> 因为具体类型的表示是其定义的一部分，是确定的，故可以帮助我们：
> - 将具体类型的对象放在栈中、静态分配的内存中，以及其他对象中
> - 直接引用对象（而不是只能通过指针或引用）
> - 直接且完全初始化对象
> - 拷贝和移动对象

The representation can be private (as it is for Vector ; $\S2.3)$ and accessible only through the member functions, but it is present. Therefore, if the representation changes in any significant way, a user must recompile. This is the price to pay for having concrete types behave exactly like built-in types. For types that don’t change often, and where local variables provide much-needed clarity and efficiency, this is acceptable and often ideal. To increase ﬂexibility, a concrete type can keep major parts of its representation on the free store (dynamic memory, heap) and access them through the part stored in the class object itself. That’s the way vector and string are implemented; they can be considered resource handles with carefully crafted interfaces.
> 表示可以是私有的，仅通过成员函数访问（例如 vector）
> 但表示是存在的，因此如果表示改变，用户需要重新编译，这是具体类型为了能表现得和内建类型一样而付出的代价
> 对于不会经常改变的类型，适宜使用具体类型
> 如果要提高灵活性，具体类型需要将它表示的主要部分保存在动态内存，然后通过存储在类对象本身的指针/引用访问它（例如 vector/string）

### 4.2.1 An Arithmetic Type
The ‘‘classical user-defined arithmetic type’’ is complex :

```cpp
class complex {
    double re, im; // representation: two doubles
public:
    complex(double r, double i): re{r}, im{i} {} // construct complex from two scalar
    complex(double r): re{r}, im{0} {} // construct complex from one scalar
    complex(): re{0}, im{0} {} // default complex: {0, 0}

    double real() const { return re; }
    void real(double d) { re = d; }
    double imag() const { return im; }
    void imag(double d) { im = d; }

    complex& operator+=(complex z) 
    {
        re += z.re; // add to re and im
        im += z.im;
        retuurn *this;
    }

    complex& operator-=(complex z) 
    {
        re -= z.re; 
        im -= z.im;
        retuurn *this;
    }

    complex& operator*=(complex z); // defined out of class somewhere
    complex& operator/=(complex z);
};
```

This is a slightly simplified version of the standard-library complex $(\S14.4)$ . The class definition itself contains only the operations requiring access to the representation. The representation is simple and conventional. For practical reasons, it has to be compatible with what Fortran provided 60 years ago, and we need a conventional set of operators. In addition to the logical demands, complex must be efficient or it will remain unused. This implies that simple operations must be inlined. That is, simple operations (such as constructors, $\scriptstyle+=$ , and imag () ) must be implemented without func- tion calls in the generated machine code. Functions defined in a class are inlined by default. It is possible to explicitly request inlining by preceding a function declaration with the keyword inline .
> 该示例是标准库 complex 的简化版本
> 其类定义本身仅包含了需要访问其表示的操作
> 除了逻辑上的实现以外，我们还需要保证 complex 类应该是高效的，因此简单的操作应该是内联的（例如构造函数，`+=`，`imag()`），也就是在对应的机器码中不应该有函数调用
> 直接定义在类里面的函数默认就是内联的
> 我们也可以在函数声明前添加关键字 `inline` 来显式请求内联

An industrial-strength complex (like the standard-library one) is carefully implemented to do appropriate inlining.

A constructor that can be invoked without an argument is called a default constructor . Thus, complex() is complex ’s default constructor. By defining a default constructor you eliminate the pos- sibility of uninitialized variables of that type.
> 可以无参数调用的构造函数是默认构造函数，定义默认构造函数可以消除该类变量不被初始化的可能性

The const specifiers on the functions returning the real and imaginary parts indicate that these functions do not modify the object for which they are called. A const member function can be invoked for both const and non- const objects, but a non- const member function can only be invoked for non- const objects. For example:
> 函数定义之前的 `const` 修饰符表明函数不会修改调用它们的对象
> `const` 函数可以被 `const` 对象调用，也可以被非 `const` 对象调用
> 但非 `const` 成员函数仅能被非 `const` 对象调用（`const` 对象只能调用 `const` 成员函数）

```cpp
complex z = {1,0};
const complex cz {1,3};
z = cz; // OK: assigning to a non-const variable
cz = z; // error: complex::operator=() is a non-const member function
x = z.real(); // OK: complex::real() is a const member function
```

Many useful operations do not require direct access to the representation of complex , so they can be defined separately from the class definition:
> 许多操作不需要直接访问 complex 的表示，因此它们的定义可以与类的定义分离开

```cpp
complex operator+(complex a, complex b) { return a+= b; }
complex operator-(complex a, complex b) { return a-=b; }
complex operator-(complex a) { return {-a.real(), -a.imag()}; } // unary minus
complex operator*(complex a, complex b) { return a*=b; }
complex operator/(complex a, complex b) { return a/=b; }
```

Here, I use the fact that an argument passed by value is copied so that I can modify an argument without affecting the caller’s copy and use the result as the return value.

The definition of `==` and `!=` are straightforward:

```cpp
bool operator==(complex a, complex b) // equal
{
    return a.real() == b.real() && a.imag() == b.imag();
}
bool operator!=(complex a, complex b) // not equal
{
    return !(a==b);
}
complex sqrt(complex); // the definition is elsewhere
```

Class `complex` can be used like this:

```cpp
void f(complex z)
{
    complex a {2.3}; // construct {2.3, 0.0} from 2.3
    complex b {1/a};
    complex c {a + z * complex{1, 2.3}};

    // ...

    if (c != b)
        c = -(b/a) + 2 * b;

}
```

The compiler converts operators involving complex numbers into appropriate function calls. For example, ${\mathfrak{c l}}{=}{\mathfrak{b}}$ means operator!=(c, b) and 1/a means operator/(complex{1}, a) .
> 编译器会将关于 complex 的运算符转化为合适的函数调用，例如 `c!=b` 意味着 `operator!=(c,b)` ，`1/a` 意味着 `operator/(complex{1}, a)`

User-defined operators (‘‘overloaded operators’’) should be used cautiously and conventionally. The syntax is fixed by the language, so you can’t define a unary / . Also, it is not possible to change the meaning of an operator for built-in types, so you can’t redefine $^+$ to subtract int s.
> 不能重载内建类型的运算符
### 4.2.2 A Container
A container is an object holding a collection of elements. We call class Vector a container because objects of type Vector are containers. As defined in $\S2.3$ , Vector isn’t an unreasonable container of double s: it is simple to understand, establishes a useful invariant (§3.5.2), provides range-checked access (§3.5.1), and provides siz e () to allow us to iterate over its elements. However, it does have a fatal ﬂaw: it allocates elements using new but nev er deallocates them. That’s not a good idea because although $\mathrm{C++}$ defines an interface for a garbage collector (§5.3), it is not guaranteed that one is available to make unused memory available for new objects. In some environments you can’t use a collector, and often you prefer more precise control of destruction for logical or performance reasons. We need a mechanism to ensure that the memory allocated by the constructor is deallocated; that mechanism is a destructor :
> 容器即存储一系列元素的对象
> 类 `Vector` 也是容器，因为它实例化的对象就是容器
> 我们定义的 `Vector` 还存在一个致命的缺陷，即从不释放 `new` 的数据
> 我们需要确保通过构造函数分配的内存在析构函数中释放

```cpp
class Vector {
public:
    Vector(int s): elem{new double[s]}, sz{s} // constructor: acquire resources
    {
        for (int i = 0; i != s; i++) // initialize elements
            elem[i] = 0;
    }

    ~Vector() { delete[] elem; } // destructor: release resources

    double& operator[](int i);
    int size() const;
private:
    double* elem; // elem points to an array of sz doubles
    int sz;
}
```

The name of a destructor is the complement operator, ˜ , followed by the name of the class; it is the complement of a constructor. Vector ’s constructor allocates some memory on the free store (also called the heap or dynamic store ) using the new operator. The destructor cleans up by freeing that memory using the delete[] operator. Plain delete deletes an individual object, delete[] deletes an array.
> 析构函数是构造函数的补
> 构造函数在堆中通过 `new` 运算符分配内存
> 析构函数使用 `delete[]` 运算符释放内存
> `delete` 释放单个对象
> `delete[]` 释放一个数组

This is all done without intervention by users of Vector. The users simply create and use Vector s much as they would variables of built-in types. For example:
> 这些过程的完成都不需要用户干涉

```cpp
void fct(int n)
{
    Vector v(n);
    // ..use v
    {
        Vector v2(2*n);
        // .. use v and v2..
    } // v2 is destroyed here

} // v is destoryed here
```

Vector obeys the same rules for naming, scope, allocation, lifetime, etc. (§1.5), as does a built-in type, such as int and char . This Vector has been simplified by leaving out error handling; see $\S3.5$ .
> 我们定义的 `Vector` 和内建类型例如 `int/char` 遵循相同的命名、作用域、分配、声明周期规则

The constructor/destructor combination is the basis of many elegant techniques. In particular, it is the basis for most $\mathrm{C++}$ general resource management techniques $(\S5.3,\ \S13.2)$ . 
> 构造函数和析构函数是大多数 C++通用资源管理的基础

Consider a graphical illustration of a Vector :

The constructor allocates the elements and initializes the Vector members appropriately. The de- structor deallocates the elements. This handle-to-data model is very commonly used to manage data that can vary in size during the lifetime of an object. The technique of acquiring resources in a constructor and releasing them in a destructor, known as Resource Acquisition Is Initialization or RAII , allows us to eliminate ‘‘naked new operations,’’ that is, to avoid allocations in general code and keep them buried inside the implementation of well-behaved abstractions. Similarly, ‘‘naked delete operations’’ should be avoided. Avoiding naked new and naked delete makes code far less error-prone and far easier to keep free of resource leaks (§13.2).
> 构造函数分配元素，并初始化成员，析构函数释放元素
> 这是一种句柄到数据模型，经常用于管理在对象的声明周期中其大小会变化的数据
> 在构造函数中获取数据以及在析构函数中释放数据的技术称为 RAII，该技术方便我们消除裸的 `new` 操作，即避免在通常的代码中进行分配，而是将它们封装在类的实现中
> 同样的，裸的 `delete` 也需要避免

### 4.2.3 Initializing Containers
A container exists to hold elements, so obviously we need convenient ways of getting elements into a container. We can create a Vector with an appropriate number of elements and then assign to them, but typically other ways are more elegant. Here, I just mention two favorites:

- Initializer-list constructor : Initialize with a list of elements. 
- push_back () : Add a new element at the end of (at the back of) the sequence.

These can be declared like this:

```cpp
class Vector {
public:
    Vector(std::initializer_list<double>); // initialize with a list of doubles
    void push_back(double); // add element at end, increasing size by one
};
```

The `push_back()` is useful for input of arbitrary numbers of elements. For example:

```cpp
Vector read(istream& is) {
    Vector v;
    for (double d; is >> d;) // read floating-point values into d
        v.push_back(d); // add d to v
    return v;
}
```

The input loop is terminated by an end-of-file or a formatting error. Until that happens, each number read is added to the Vector so that at the end, v ’s size is the number of elements read. I used a for -statement rather than the more conventional while -statement to keep the scope of d limited to the loop. 
> 输入循环由 EOF 或格式化错误结束
> 我们使用 `for` 而不是 `while` 以保持 `d` 的作用域限制在循环内

The way to provide Vector with a move constructor, so that returning a potentially huge amount of data from read () is cheap, is explained in $\S5.2.2$ :

```cpp
Vector v = read(cin); // no copy of Vector elements here
```

The way that `std:: vector` is represented to make `push_back()` and other operations that change a vector's size efficient is presented in $\S11.2$ .

The `std::initializer_list` used to define the initializer-list constructor is a standard-library type known to the compiler: when we use a $\{\}$ -list, such as $^{\{1,2,3,4\}}$ , the compiler will create an object of type initializ er_list to give to the program. So, we can write:
> 示例中用于定义初始化列表构造函数的 `std::initializer_list` 是一个标准库类型，并且编译器知道它，当我们使用 `{}` 列表时，例如 `{1,2,3,4}` ，编译器会根据该列表，为程序创建一个类型为 `initializere_list` 的对象

```cpp
Vector v1 = {1,2,3,4,5}; // v1 has 5 elements
Vector v2 = {1.2,2.3,3.4,4.5}; // v2 has 4 elements
```

Vector ’s initializer-list constructor might be defined like this:

```cpp
Vector::Vector(std::initializer_list<double> lst) // initialize with a list
    :elem{new double[lst.size()]}, sz{static_cast<int>(lst.size())}
{
    copy(lst.begin(), lst.end(), elem); // copy from lst into elem
}
```

Unfortunately, the standard-library uses unsigned integers for sizes and subscripts, so I need to use the ugly static_cast to explicitly convert the size of the initializer list to an int . This is pedantic because the chance that the number of elements in a handwritten list is larger than the largest integer (32,767 for 16-bit integers and 2,147,483,647 for 32-bit integers) is rather low. Howev er, the type system has no common sense. It knows about the possible values of variables, rather than actual values, so it might complain where there is no actual violation. Such warnings can occasionally save the programmer from a bad error.
> 标准库使用的是 `unsigned` 的整数作为大小和下标，因此需要使用 `static_cast` 将其显式转化为 `int`
> 事实上，一个手写的列表的 size 大于 `int` 类型支持的最大值的概率很小，但不这么写，类型系统就会发出警告，因为 `unsigned` 的正整数范围是比 `int` 大的

A static_cast does not check the value it is converting; the programmer is trusted to use it cor- rectly. This is not always a good assumption, so if in doubt, check the value. Explicit type conver- sions (often called casts to remind you that they are used to prop up something broken) are best avoided. Try to use unchecked casts only for the lowest level of a system. They are error-prone.
> `static_cast` 不会检查它转换的值是多少，这需要由程序员来做
> 最好避免显式的类型转换，尝试尽量仅在系统的最底层使用未检查的转换，减少在高层逻辑中引入错误的可能性

Other casts are reinterpret_cast for treating an object as simply a sequence of bytes and const_cast for ‘‘casting away const .’’ Judicious use of the type system and well-designed libraries allow us to eliminate unchecked casts in higher-level software.
> `reinterpret_cast` 仅仅将对象视作字节序列
> `const_cast` 一般用于去掉 `const` 
## 4.3 Abstract Types
Types such as complex and Vector are called concrete types because their representation is part of their definition. In that, they resemble built-in types. In contrast, an abstract type is a type that completely insulates a user from implementation details. To do that, we decouple the interface from the representation and give up genuine local variables. Since we don’t know anything about the representation of an abstract type (not even its size), we must allocate objects on the free store (§4.2.2) and access them through references or pointers $(\S1.7,\S13.2.1)$ .
> 类似于 `complex` 和 `Vector` 的类型称为具体类型，因为它们的表示是它们定义的一部分
> 抽象类型是完全将用户和实现细节隔离的类型，为此，我们需要完全隔离接口和表示，并且不定义真正的局部变量，因为抽象类型的表示，甚至其大小，应该是完全未知的
> 我们必须在堆中分配对象，然后通过引用或指针访问

First, we define the interface of a class Container , which we will design as a more abstract version of our Vector :

```cpp
class Container {
public:
    virtual double& operator[](int) = 0; // pure virtual function
    virtual int size() const = 0; // const member function
    virtual ~Container() {} // destructor
};
```

This class is a pure interface to specific containers defined later. The word vir tual means ‘‘may be redefined later in a class derived from this one.’’ Unsurprisingly, a function declared virtual is called a virtual function . A class derived from Container provides an implementation for the Con- tainer interface. The curious $\tt=0$ syntax says the function is pure virtual ; that is, some class derived from Container must define the function. Thus, it is not possible to define an object that is just a Container . For example:
> 该类是对之后定义的具体容器类型的纯接口，`virtual` 意思是该函数可能会在该类的衍生类中重定义，声明为 `virtual` 的函数称为虚函数
> 由 `Container` 类衍生的类提供对 `Container` 接口的实现
> `=0` 语法表明该函数是纯虚的，即 `Container` 的衍生类必须重定义该函数
> 因此，我们不可能定义一个类型仅为 `Container` 的对象

```cpp
Container c; // error: there can be no objects of an abstract
Container* p = new Vector_container(10); // OK: Container is an interface
```

A Container can only serve as the interface to a class that implements its operator[]() and siz e () functions. A class with a pure virtual function is called an abstract class . 
> `Container` 仅能作为实现了它的 `operator[]()` 和 `size()` 的类的接口
> 具有纯虚函数的类称为抽象类

This Container can be used like this: 

```cpp
void use(Container& c)
{
    const int sz = c.size();
    for (int i = 0; i!=sz; i++)
        cout<<c[i];
}
```

Note how use () uses the Container interface in complete ignorance of implementation details. It uses size () and [ ] without any idea of exactly which type provides their implementation. A class that provides the interface to a variety of other classes is often called a polymorphic type .
> 注意在上例中，我们直接使用 `Container` 定义的接口（包括 `size()` 和 `operator[]()` ），而完全不需要知道它具体的实现
> 为一系列其他类提供了接口的类一般被称为多态类型

As is common for abstract classes, Container does not have a constructor. After all, it does not have any data to initialize. On the other hand, Container does have a destructor and that destructor is vir tual , so that classes derived from Container can provide implementations. Again, that is com- mon for abstract classes because they tend to be manipulated through references or pointers, and someone destroying a Container through a pointer has no idea what resources are owned by its implementation; see also $\S4.5$ .
> 抽象类一般没有构造函数，因为没有需要初始化的数据
> 但抽象类一般有析构函数，并且析构函数是虚函数，因此析构函数由其衍生类提供具体实现
> 析构函数定义为虚函数对于抽象类是十分平常的，因为抽象类一般通过指针或引用被操纵，而用户通过指针摧毁抽象类对象时是不知道它的实现具体占据了多少资源的，资源销毁需要由对象自己实现

The abstract class Container defines only an interface and no implementation. For Container to be useful, we have to implement a container that implements the functions required by its interface. For that, we could use the concrete class Vector :
> 抽象类 `Container` 仅定义了接口，没有任何实现
> 我们需要实现一个实现了该接口需要的函数的容器类，例如具体类 `Vector`

```cpp
class Vector_container: public Container  { // Vector_container implements Container
public:
    Vector_container(int s): v(s) {} // Vector of s elements
    ~Vector_container() {}
    double& operator[](int i) override { return v[i]; }
    int size() const override { return v.size(); }
private:
    Vector v;
}
```

The `:public` can be read as ‘‘is derived from’’ or ‘‘is a subtype of.’’  Class Vector container is said to be derived from class Container , and class Container is said to be a base of class Vector container . An alternative terminology calls Vector container and Container subclass and superclass , respectively. The derived class is said to inherit members from its base class, so the use of base and derived classes is commonly referred to as inheritance .
> 衍生类从基类中继承成员

The members operator[]() and size() are said to override the corresponding members in the base class Container . I used the explicit override to make clear what’s intended. The use of override is optional, but being explicit allows the compiler to catch mistakes, such as misspellings of function names or slight differences between the type of a virtual function and its intended overrider. The explicit use of override is particularly useful in larger class hiearchies where it can otherwise be hard to know what is supposed to override what.
> 成员 `operator[]()/size()` 覆盖了基类中的对应成员
> 显式使用 `override` 是可选的，但显式写出可以帮助编译器捕获错误，例如误拼写函数名，或者原函数和覆盖函数类型不一致
> 在更大的类结构中，显式使用 `override` 十分有用，帮助清晰表示

The destructor ( ˜Vector_container () ) overrides the base class destructor ( ˜Container () ). Note that the member destructor ( ˜Vector () ) is implicitly invoked by its class’s destructor ( ˜Vector_container () ).
> 析构函数 `~Vector_container()` 覆盖了其基类析构函数 `~Container()`
> 另外类成员的析构函数 (`~Vector()`) 会由 `~Vector_container()` 隐式调用

For a function like `use(Container&)` to use a Container in complete ignorance of implementation details, some other function will have to make an object on which it can operate. For example:
> 要让类似 `use(Container&)` 的函数完全依靠接口操纵 Container，其他函数需要为其提供对象

```cpp
void g()
{
    Vector_container vc(10); // Vector of ten elements
    // .. fill vc
    use(vc);
}
```

Since use () doesn’t know about Vector container s but only knows the Container interface, it will work just as well for a different implementation of a Container . For example:
> 因为 `use()` 仅仅依赖于 `Container` 的接口，我们完全可以为它提供不同的实现

```cpp
class List_container : public Container { // List_container implements Container
public:
    List_container() {} // empty List
    List_container(initializer_list<double> il): ld{il} {}
    ~List_container() {}

    double& operator[](int i) override;
    int size() const override { return ld.size(); }
private:
    std::list<double> ld; // standard library of doubles
};

double& List_container::operator[](int i)
{
    for (auto& x : ld) {
        if (i==0)
            return x;
        i--;
    }
    throw out_of_range{"List container"};
}
```

Here, the representation is a standard-library `list<double>` . Usually, I would not implement a container with a subscript operation using a list , because performance of list subscripting is atrocious compared to vector subscripting. However, here I just wanted to show an implementation that is radically different from the usual one.

A function can create a List_container and have use () use it: 

```cpp
void h()
{
    List_container lc = {1, 2, 3, 4};
    use(lc);
}
```

The point is that use (Container&) has no idea if its argument is a Vector container , a List_container , or some other kind of container; it doesn’t need to know. It can use any kind of Container . It knows only the interface defined by Container . Consequently, use (Container&) needn’t be recompiled if the implementation of List_container changes or a brand-new class derived from Container is used.
> `use(Container&)` 并不涉及容器的实现，仅仅使用接口，因此即便其实现改变，`use(Container&)` 都不需要重新编译

The ﬂip side of this ﬂexibility is that objects must be manipulated through pointers or references $(\S5.2,\S13.2.1$ ).
> 但为了这样的灵活性，对象必须通过指针或者引用操纵

## 4.4 Virtual Functions
Consider again the use of Container :

```cpp
void use(Container& c)
{
    const int sz = c.size();
    for (int i = 0; i != sz; i++)
        cout << c[i];
}
```

How is the call c[i] in use () resolved to the right operator[]() ? When h () calls use () , List_container ’s operator[]() must be called. When ${\mathfrak{g}}()$ calls use () , Vector container ’s operator[]() must be called. To achieve this resolution, a Container object must contain information to allow it to select the right function to call at run time. 
> 调用虚函数时，例如 `operator[]()` ，实际应该调用对应的重载实现，为此，一个 `Container` 对象必须包含能让它在运行时选择正确函数调用的信息

The usual implementation technique is for the compiler to convert the name of a virtual function into an index into a table of pointers to functions. That table is usually called the virtual function table or simply the vtbl . Each class with virtual functions has its own vtbl identifying its virtual functions. This can be represented graphically like this:
> 常见的实现是让编译器将虚函数名作为虚函数表（包含了指向各个函数的指针）的索引
> 每个具有虚函数的类都有自己的虚函数表，标识自己的虚函数

![[A Tour of C++-Virtual Function.png]]

The functions in the vtbl allow the object to be used correctly even when the size of the object and the layout of its data are unknown to the caller. The implementation of the caller needs only to know the location of the pointer to the vtbl in a Container and the index used for each virtual function. This virtual call mechanism can be made almost as efficient as the ‘‘normal function call’’ mechanism (within $25\%$ ). Its space overhead is one pointer in each object of a class with virtual functions plus one vtbl for each such class.
> 虚函数表指向的都是对应虚函数在自己所属的类中的实际实现，这使得调用者可以正确调用对应的虚函数
> 调用者仅需要知道对应容器类型指向 vtbl 的指针，以及用于每个虚函数的索引
> 该虚函数调用方式和正常的函数调用的效率近似，其空间开销仅仅是每个对象中多存储一个指向其对应类的 vtbl 的指针

## 4.5 Class Hierarchies
The Container example is a very simple example of a class hierarchy. A class hierarchy is a set of classes ordered in a lattice created by derivation (e.g., : public ). We use class hierarchies to represent concepts that have hierarchical relationships, such as ‘‘A fire engine is a kind of a truck which is a kind of a vehicle’’ and ‘‘A smiley face is a kind of a circle which is a kind of a shape.’’ Huge hierarchies, with hundreds of classes, that are both deep and wide are common. As a semi-realistic classic example, let’s consider shapes on a screen:
> 我们使用类层次来表示具有层次关系的概念

The arrows represent inheritance relationships. For example, class Circle is derived from class Shape . A class hierachy is conventionally drawn growing down from the most basic class, the root, towards the (later defined) derived classes. To represent that simple diagram in code, we must first specify a class that defines the general properties of all shapes:
> 我们先指定所有 shapes 的通用性质

```cpp
class Shape {
public:
    virtual Point center() const = 0; // pure virtual 
    virtual void move(Point to) = 0;
    virtual void draw() const = 0; // draw on current "Canvas"
    virtual void rotate(int angle) = 0;
    virtual ~Shape() {} // destructor
};
```

Naturally, this interface is an abstract class: as far as representation is concerned, nothing (except the location of the pointer to the vtbl ) is common for every Shape . 
> 类 `Shape` 作为抽象类，其作用是作为接口，而它的具体表示对于每个 `Shape` 都是不同的（除了指向 `vtbl` 的指针的位置是相同的）

Given this definition, we can write general functions manipulating vectors of pointers to shapes:
> 给定其接口定义后，我们已经可以为由指向 `Shape` 的指针构成的 vector 写通用的函数了

```cpp
void rotate_all(vector<Shape*>& v, int angle) // rotate v's elements by angle degrees
{
    for (auto p : v)
        p->rotate(angle);
}
```

To define a particular shape, we must say that it is a Shape and specify its particular properties (including its virtual functions):
> 要定义特定的 `Shape` ，我们需要继承它，并指定其特定的属性（包括虚函数）

```cpp
class Circle: public Shape {
public:
    Circle(Point p, int rad); // constructor
    Point center() const override { return x; }
    void move(Point to) override { x = to; }
    void draw() const override;
    void rotate(int) override {} 
private:
    Points x; // center
    int r; // radius
};
```

So far, the Shape and Circle example provides nothing new compared to the Container and Vector container example, but we can build further:

```cpp
class Smiley: public Circle { // use the circle as the base for a face
public:
    Smiley(Point p, int rad): Circle{p, rad}, mouth{nullptr} {}
    ~Smiley() 
    { 
        delete mouth; 
        for(auto p: eyes)
            delete p;
    }
    void move(Point to) override;
    void draw() const override;
    void rotate(int) override;
    void add_eye(Shape* s)
    {
        eyes.push_back(s);
    }
    void set_mouth(Shape* s);
    virtual void wink(int i); // wink eye number i
private:
    vector<Shape*> eyes; // usually two eyes
    Shape* mouth;
}
```

The push_back () member of vector copies its argument into the vector (here, eyes ) as the last element, increasing that vector’s size by one. 

We can now define Smiley:: draw() using calls to Smiley ’s base and member draw() s: 

```cpp
void Smiley::draw() const {
    Circle::draw();
    for(auto p: eyes)
        p->draw();
    mouth->draw();
}
```

Note the way that Smiley keeps its eyes in a standard-library vector and deletes them in its destructor. Shape ’s destructor is vir tual and Smiley ’s destructor overrides it. A virtual destructor is essential for an abstract class because an object of a derived class is usually manipulated through the interface provided by its abstract base class. In particular, it may be deleted through a pointer to a base class. Then, the virtual function call mechanism ensures that the proper destructor is called. That destructor then implicitly invokes the destructors of its bases and members.
> `Shape` 的析构函数是虚函数，`Smiley` 的析构函数重载了它
> 虚析构函数对于抽象类是必须的，因为衍生类对象往往需要通过由抽象类提供的接口操纵，尤其是它会被通过一个指向基类的指针 `delete`，因此虚函数机制保证正确的析构函数被调用，而该析构函数也会隐式地调用其成员类和基类的析构函数

In this simplified example, it is the programmer’s task to place the eyes and mouth appropriately within the circle representing the face. We can add data members, operations, or both as we define a new class by derivation. This gives great ﬂexibility with corresponding opportunities for confusion and poor design.

### 4.5.1 Benefits from Hierarchies
A class hierarchy offers two kinds of benefits:

- Interface inheritance : An object of a derived class can be used wherever an object of a base class is required. That is, the base class acts as an interface for the derived class. The Container and Shape classes are examples. Such classes are often abstract classes. 
- Implementation inheritance : A base class provides functions or data that simplifies the implementation of derived classes. Smiley ’s uses of Circle ’s constructor and of Circle:: draw () are examples. Such base classes often have data members and constructors.

> 类的继承机制提供了两种优势
> - 接口继承：当需要基类对象时，总是可以使用衍生类类对象代替，即基类作为衍生类的接口，提供接口的基类一般是抽象类
> - 实现继承：基类可以为衍生类提供函数或数据，这类基类一般有数据成员和构造函数

Concrete classes – especially classes with small representations – are much like built-in types: we define them as local variables, access them using their names, copy them around, etc. Classes in class hierarchies are different: we tend to allocate them on the free store using new , and we access them through pointers or references. For example, consider a function that reads data describing shapes from an input stream and constructs the appropriate Shape objects:
> 一般来说，对于具体类，我们对其的操作和内建类型是一致的，都是将它们直接定义为局部变量，直接访问它们存储在栈中的表示
> 但对于处于类继承层次中的类，我们倾向于用 `new` 在堆中分配它们，然后使用指针或引用访问它们

```cpp
enum class Kind {circle, triangle, smiley};
Shape* read_shape(istream& is) // read shape descriptions from input stream
    // 注意 `read_shape()` 函数返回的是 `Shape*` 
{
    // .. read shape header from is and its Kind k
    switch(k) {
        case Kind::circle: 
            // read circle data {Point, int} into p and r
            return new Circle{p, r};
        case Kind::triangle:
            // read triangle data {Point, Point, Point} into p1, p2, and p3
            return new Triangle{p1, p2, p3};
        case Kind::smiley:
            // read smiley data {Point, int, Shape, Shape, Shape} into p, r, e1, e2 and m
            Smiley* ps = new Smiley{p, r};
            ps->add_eye(e1);
            ps->add_eye(e2);
            ps->set_mouth(m);
            return ps;
    }
}
```

A program may use that shape reader like this:

```cpp
void user()
{
    std::vector<Shape*> v;
    while(cin)
        v.push_back(read_shape(cin));
    draw_all(v); // call draw() for each element
    rotate_all(v, 45); // call rotate(45) for each element
    for(auto p: v) // remember delete elements
        delete p;
}
```

Obviously, the example is simplified – especially with respect to error handling – but it vividly illustrates that user () has absolutely no idea of which kinds of shapes it manipulates. The user () code can be compiled once and later used for new Shape s added to the program. Note that there are no pointers to the shapes outside user () , so user () is responsible for deallocating them. This is done with the delete operator and relies critically on Shape ’s virtual destructor. Because that destructor is virtual, delete invokes the destructor for the most derived class. This is crucial because a derived class may have acquired all kinds of resources (such as file handles, locks, and output streams) that need to be released. In this case, a Smiley deletes its eyes and mouth objects. Once it has done that, it calls Circle ’s destructor. Objects are constructed ‘‘bottom up’’ (base first) by constructors and destroyed ‘‘top down’’ (derived first) by destructors.
> 示例中 `user()` 完全不会知道它操纵的具体类型是什么，`user()` 的代码仅需要编译一次，之后再加入新的 `Shape` 衍生类型也不会影响
> 注意 `user()` 之外没有再指向 `Shape` 的指针，因此 `user()` 需要负责释放它们，即需要通过 `delete` 删除指针，`delete` 会调用对应对象的析构函数
> 释放资源是很关键的，因为衍生类会获取很多类型的资源，例如文件句柄、锁、输出流等
> 对象构造的顺序是从顶层到底层，析构的顺序是从底层到顶层

### 4.5.2 Hierarchy Navigation
The read_shape () function returns `Shape∗` so that we can treat all Shapes alike. However, what can we do if we want to use a member function that is only provided by a particular derived class, such as Smiley ’s wink () ? We can ask ‘‘is this Shape a kind of Smiley ?’’ using the `dynamic_cast` operator:
> 如果我们需要使用 `Shape` 接口之外的成员函数，也就是具体的衍生类的成员函数，我们需要使用 `dynamic_cast` 运算符

```cpp
Shape* ps {read_shape(cin)};
if(Smiley* p = dynamic_cast<Smiley*>(ps)) { // does ps point to a Smiley
    // ...
}
else {
    // ...
}
```

If at run time the object pointed to by the argument of dynamic_cast (here, ps ) is not of the expected(here, Smiley) or a class derived from the expected type, dynamic_cast returns nullptr.
> 如果 `ps` 实际指向的对象不是 `Smiley*` 类型的，且也不是 `Smiley` 的衍生类，则 `dynamic_cast` 返回 `nullptr`

We use dynamic_cast to a pointer type when a pointer to an object of a different derived class is a valid argument. We then test whether the result is nullptr . This test can often conveniently be placed in the initialization of a variable in a condition. 

When a different type is unacceptable, we can simply dynamic_cast to a reference type. If the object is not of the expected type, dynamic_cast throws a bad_cast exception:
> 除了转化为另一个指针类型以外，`dynamic_cast` 还可以将指针类型转化为引用类型，转化为引用类型时，如果被转化指针的实际类型不匹配的话，会抛出 `std::bad_cast` 异常

```cpp
Shape* ps {read_shape(cin)};
Smiley& r{dynamic_cast<Smiley&>(*ps)}; // somewhere, catch std::bad_cast
```

Code is cleaner when dynamic_cast is used with restraint. If we can avoid using type information, we can write simpler and more efficient code, but occasionally type information is lost and must be recovered. This typically happens when we pass an object to some system that accepts an interface specified by a base class. When that system later passes the object back to us, we might have to recover the original type. Operations similar to dynamic_cast are known as ‘‘is kind of’’ and ‘‘is instance of’’ operations.

### 4.5.3 Avoiding Resource Leaks
Experienced programmers will have noticed that I left open three opportunities for mistakes:

- The implementer of Smiley may fail to delete the pointer to mouth . 
- A user of `read_shape()` might fail to delete the pointer returned. 
- The owner of a container of Shape pointers might fail to delete the objects pointed to.

In that sense, pointers to objects allocated on the free store is dangerous: a ‘‘plain old pointer’’ should not be used to represent ownership. For example:

```cpp
void user(int x)
{
    Shape* p = new Circle{Point{0, 0}, 10};
    // ...
    if (x < 0) throw Bad_x{}; // potential leak 
    if (x == 0) return; // potential leak
    // ...
    delete p;
}
```

This will leak unless $x$ is positive. Assigning the result of new to a ‘‘naked pointer’’ is asking for trouble.
> 显然，`user(int x)` 中存在内存泄漏的可能性，只有在 `x` 为正数时才可能避免泄露
> 因此，将 `new` 的结果赋值给一个裸露的指针是存在问题的

One simple solution to such problems is to use a standard-library unique_ptr (§13.2.1) rather than a ‘‘naked pointer’’ when deletion is required:
> 一个简单的方法是在有 `delete` 需求时，使用标准库提供的 ` unique_ptr ` 替代传统的指针

```cpp
class Smiley: public Circle {
    // ...
private:
    vector<unique_ptr<Shape>> eyes; // usually two eyes
    unique_ptr<Shape> mouth;
};
```

This is an example of a simple, general, and efficient technique for resource management (§5.3). 

As a pleasant side effect of this change, we no longer need to define a destructor for Smiley . The compiler will implicitly generate one that does the required destruction of the `unique_ptr` s (§5.3) in the vector . The code using unique_ptr will be exactly as efficient as code using the raw pointers correctly. 
> 将 `Smiley` 中的有 `delete` 需求的指针都替换为 `unique_ptr` 之后，我们就不再需要为 `Smiley` 定义析构函数了，编译器会隐式地为 `unique_ptr` 生成析构函数
> 并且使用 `unique_ptr` 的效率和使用普通指针的效率是相当的

Now consider users of `read_shape()` : 

```cpp
unique_ptr<Shape> read_shape (istream& is) // read shape descriptions from input stream
{
    // .. read shape header from is and its Kind k
    switch (k) {
        case Kind::circle: 
            // read circle data {Point, int} into p and r
            return unique_ptr<Shape>{new Circle{p, r}};
        case Kind::triangle:
            // read triangle data {Point, Point, Point} into p1, p2, and p3
            return unique_ptr<Shape>{new Triangle{p1, p2, p3}};
            // ...
    }
}

void user()
{
    std::vector<unique_ptr<Shape>> v;
    while(cin)
        v.push_back(read_shape(cin));
    draw_all(v); // call draw() for each element
    rotate_all(v, 45); // call rotate(45) for each element
} // all Shapes implicitly destroyed
```

Now each object is owned by a unique_ptr that will delete the object when it is no longer needed, that is, when its unique_ptr goes out of scope.
> 每个被 `unique_ptr` 指向的对象都会自动在 `unique_ptr` 超出其作用域后被 `delete`

For the unique_ptr version of user () to work, we need versions of draw_all () and rotate_all () that accept `vector<unique_ptr<Shape>> s`. Writing many such `_all()` functions could become tedious, so $\S6.3.2$ shows an alternative.

## 4.6 Advice
[1] Express ideas directly in code; $\S4.1$ ; [CG: P.1].
[2] A concrete type is the simplest kind of class. Where applicable, prefer a concrete type over more complicated classes and over plain data structures; $\S4.2$ ; [CG: C.10].
[3] Use concrete classes to represent simple concepts; $\S4.2$ .
[4] Prefer concrete classes over class hierarchies for performance-critical components; $\S4.2$ .
[5] Define constructors to handle initialization of objects; $\S4.2.1$ , $\S5.1.1$ ; [CG: C.40] [CG: C.41].
[6] Make a function a member only if it needs direct access to the representation of a class; $\S4.2.1$ ; [CG: C.4].
[7] Define operators primarily to mimic conventional usage; $\S4.2.1$ ; [CG: C.160].
[8] Use nonmember functions for symmetric operators; $\S4.2.1$ ; [CG: C.161].
[9] Declare a member function that does not modify the state of its object const ; $\S4.2.1$ .
[10] If a constructor acquires a resource, its class needs a destructor to release the resource; $\S4.2.2$ ; [CG: C.20].
[11] Avoid ‘‘naked’’ new and delete operations; $\S4.2.2$ ; [CG: R.11].
[12] Use resource handles and RAII to manage resources; $\S4.2.2$ ; [CG: R.1].
[13] If a class is a container, give it an initializer-list constructor; $\S4.2.3$ ; [CG: C.103].
[14] Use abstract classes as interfaces when complete separation of interface and implementation is needed; $\S4.3$ ; [CG: C.122].
[15] Access polymorphic objects through pointers and references; $\S4.3$ .
[16] An abstract class typically doesn’t need a constructor; $\S4.3$ ; [CG: C.126].
[17] Use class hierarchies to represent concepts with inherent hierarchical structure; $\S4.5$ .
[18] A class with a virtual function should have a virtual destructor; $\S4.5$ ; [CG: C.127].
[19] Use override to make overriding explicit in large class hierarchies; $\S4.5.1$ ; [CG: C.128].
[20] When designing a class hierarchy, distinguish between implementation inheritance and inter- face inheritance; $\S4.5.1$ ; [CG: C.129].
[21] Use dynamic_cast where class hierarchy navigation is unavoidable; $\S4.5.2$ ; [CG: C.146].
[22] Use dynamic_cast to a reference type when failure to find the required class is considered a failure; $\S4.5.2$ ; [CG: C.147].
[23] Use dynamic_cast to a pointer type when failure to find the required class is considered a valid alternative; $\S4.5.2$ ; [CG: C.148].
[24] Use unique_ptr or shared_ptr to avoid forgetting to delete objects created using new ; $\S4.5.3$ ; [CG: C.149].
> 1. Express ideas directly in code
> 2. 偏好具体类 to 更复杂的类型或单纯的数据结构
> 3. 使用具体类表示简单概念
> 4. 对于性能关键的组件，偏好具体类 to 类层次结构
> 5. 定义构造函数处理对象的初始化
> 6. 只有函数需要直接访问类表示时，才将其作为成员函数
> 7. Define operators primarily to mimic conventional usage
> 8. 使用非成员函数来重定义对称的运算符
> 9. 将不修改对象状态的成员函数声明为 `const`
> 10. If a constructor acquires resources, its class needs a dectuctor to release the resource
> 11. 避免裸的 `new` 和 `delete` 操作
> 12. Use resource handles and RAII to manage resources
> 13. 定义容器类时，为其定义初始化列表构造函数
> 14. 需要完全隔离实现和接口时，使用抽象类作为接口
> 15. 通过指针和引用访问多态对象
> 16. 抽象类不需要构造函数
> 17. Use class hierarchies to represent concepts with inherent hierarchical structure
> 18. 有虚函数的类就要有虚析构函数
> 19. 覆盖时，使用 `override` 以清晰表示
> 20. 设计类层次时，区分好实现继承和接口继承
> 21. Use `dynamic_cast` when class hierarchy navigation is unavoidable
> 22. 如果将找不到对应的具体类型视为 failure 时，使用 `dynamic_cast` 将指针转化为引用
> 23. 如果将找不到对应的具体类型视为 valid alternative 时，使用 `dynamic_cast` 将指针转化为指针
> 24. 使用 `unique_ptr/shared_ptr` 来避免忘记 `delete` `new` 出来的对象

# 5 Essential Operations
## 5.1 Introduction
Some operations, such as initialization, assignment, copy, and move, are fundamental in the sense that language rules make assumptions about them. Other operations, such as $==$ and $<<$ , have conventional meanings that are perilous to ignore.
> 一些操作，如初始化、赋值、拷贝和移动，是根本性的，因为 C++的语言规则对它们做出了假设
> 其他操作，如 `==` 和 `<<`，具有约定的含义，忽略这些含义是危险的

### 5.1.1 Essential Operations
Construction of objects plays a key role in many designs. This wide variety of uses is reﬂected in the range and ﬂexibility of the language features supporting initialization.

Constructors, destructors, and copy and move operations for a type are not logically separate. We must define them as a matched set or suffer logical or performance problems. If a class $\mathbf{x}$ has a destructor that performs a nontrivial task, such as free-store deallocation or lock release, the class is likely to need the full complement of functions:
> 一个类型的构造函数、析构函数、拷贝和移动操作并不是逻辑上分离的，如果不想要出现逻辑上或性能上的问题，我们必须将它们定义为一个匹配的集合
> 如果类 `X` 具有一个执行非平凡任务的析构函数，例如释放内存、锁，则类很可能需要以下函数的完整补全

```cpp
class X {
public:
    X(Sometype); // ordinary constructor: create an object X
    X(); // default constructor
    X(const X&); // copy constructor
    X(X&&); // move constructor
    X& operator=(const X&); // copy assignment: clean up target and copy
    X& operator=(X&&); // move assignment: clean up target and move
    ~X();
    // ...
}
```

There are five situations in which an object can be copied or moved:

- As the source of an assignment 
- As an object initializer 
- As a function argument 
- As a function return value 
- As an exception

> 对象需要被拷贝或移动的情况有：
> - 作为赋值的源操作数
> - 作为对象初始化值
> - 作为函数实参
> - 作为函数返回值
> - 作为异常

An assignment uses a copy or move assignment operator. In principle, the other cases use a copy or move constructor. Howev er, a copy or move constructor invocation is often optimized away by constructing the object used to initialize right in the target object. For example:
> 赋值操作可以使用拷贝或移动赋值运算符
> 原则上，其他情况使用拷贝或者移动构造函数
> 拷贝或移动构造函数的调用会常常被优化掉，即直接在目标对象中构建所使用的对象，例如：

```cpp
X make(Sometype);
X x = make(value);
```

Here, a compiler will typically construct the X from make() directly in $x$ ; thus eliminating (‘‘eliding’’) a copy.
> 对于 `x = make(value)` 这类的赋值语句，编译器通常会直接在变量 `x` 的位置构建 `make() ` 函数返回的对象（而不是先构建一个临时对象，然后再通过拷贝/移动构造函数构造 ` x `），从而消除了一个拷贝操作（“拷贝消除”）
> 这等价于直接声明 `X x(value)`

In addition to the initialization of named objects and of objects on the free store, constructors are used to initialize temporary objects and to implement explicit type conversion.
> 构造函数除了用于命名对象的初始化和堆中对象的初始化，还会用于暂时对象的初始化和实现显式的类型转换

Except for the ‘‘ordinary constructor,’’ these special member functions will be generated by the compiler as needed. If you want to be explicit about generating default implementations, you can:
> 除了普通的构造函数之外，这类较特殊的成员函数（如拷贝构造函数、移动构造函数、拷贝赋值运算符、移动赋值运算符和析构函数）默认情况下会由编译器根据需要自动生成
> 如果我们希望显式地生成默认实现，可以在类定义中使用 `= default` 标记

```cpp
class Y {
public:
    Y(Sometype);
    Y(const Y&) = default; // I really do want the default copy constructor
    Y(Y&&) = default; // and the default move constructor
    // ...
}
```

If you are explicit about some defaults, other default definitions will not be generated.
> 需要注意的是，一旦我们在类定义中显式声明了一些默认成员函数，编译器将不会为我们生成其他的默认成员函数
> 例如如果我们显式声明了一个默认拷贝构造函数，编译器就不会为我们生成默认的拷贝赋值运算符或其他默认构造函数
> 同样，如果你显式声明了一个默认移动构造函数，那么默认的移动赋值运算符也不会被生成

When a class has a pointer member, it is usually a good idea to be explicit about copy and move operations. The reason is that a pointer may point to something that the class needs to delete , in which case the default memberwise copy would be wrong. Alternatively, it might point to some- thing that the class must not delete . In either case, a reader of the code would like to know. For an example, see $\S5.2.1$ .
> 如果一个类有指针成员，我们最好显式指定拷贝和移动操作，因为该指针可能指向某个需要类去删除的对象，在这种情况下，默认的逐成员拷贝（是浅拷贝）是错误的
> 或者，指针可能指向某个类不应该删除的对象
> 在这两种情况下，代码的阅读者都应该清楚这一点

A good rule of thumb (sometimes called the rule of zero ) is to either define all of the essential operations or none (using the default for all). For example:
> 一个较好的经验法则是要么定义所有的必要操作，要么不定义（所有的都是默认），例如：

```cpp
struct Z {
    Vector v ;
    string s;
};

Z z1; // default initialize z1.v and z1.s
Z z2 = z1; // default copy z1.v and z1.s
```

Here, the compiler will synthesize memberwise default construction, copy, move, and destructor as needed, and all with the correct semantics.
> 本例中，编译器会根据需要自动构造逐成员的默认构造、拷贝、移动和析构

To complement $=$ default , we hav e =delete to indicate that an operation is not to be generated. A base class in a class hierarchy is the classical example where we don’t want to allow a memberwise copy. For example:
> 和 `=default` 对应的是 `=delete` ，它用于指定某个操作不应该被生成
> 经典的例子就是类层次结构中的基类中，我们不允许逐成员的拷贝：

```cpp
class Shape {
public:
    Shape(const Shape&) = delete; // no copy operations
    Shape& operator=(const Shape&) = delete;
    // ...
}
void copy(Shape& s1, const Shape& s2)
{
    s1 = s2; // error: Shape copy is deleted
}
```

A =delete makes an attempted use of the delete d function a compile-time error; =delete can be used to suppress any function, not just essential member functions.
> 尝试调用 deleted 函数时会发生编译时错误
> `=delete` 可以用于删除任意函数，并不仅限于必要的成员函数

### 5.1.2 Conversions
A constructor taking a single argument defines a conversion from its argument type. For example, complex (§4.2.1) provides a constructor from a double :
> 仅接受一个参数的构造函数实际上定义了对于其参数类型的一个隐式转换操作，例如：

```cpp
complex z1 = 3.14; // z1 becomes {3.14, 0.0}
complex z2 = z1 * 2;  // z2 becomes z1 * {2.0, 0} == {6.28, 0.0}
```

This implicit conversion is sometimes ideal, but not always. For example, Vector (§4.2.2) provides a constructor from an int :

```cpp
Vector v1 = 7; // OK: v1 has 7 elements
```

This is typically considered unfortunate, and the standard-library vector does not allow this int-to-vector ‘‘conversion.’’
> 标准库 `vector` 并没有提供 int-to-vector 的“转换”，即

The way to avoid this problem is to say that only explicit ‘‘conversion’’ is allowed; that is, we can define the constructor like this:
> 要避免这类语义不恰当的转换，我们需要声明仅允许显式的转换：

```cpp
class Vector {
public:
    explict Vector(int s); // no implicit conversion from int to Vector
    // ...
};
```

That gives us:

```cpp
Vector v1(7); // OK; v1 has 7 elements
Vector v2 = 7; // error: no implicit conversion from int to Vector
```

When it comes to conversions, more types are like Vector than are like complex , so use explicit for constructors that take a single argument unless there is a good reason not to.
> 使用 `explicit` 修饰的单参数构造函数将不会允许通过隐式类型转换被调用，而必须显式被调用
> 对于单参数的构造函数，尽量使用显式构造函数，除非有合适的理由

### 5.1.3 Member Initializers
When a data member of a class is defined, we can supply a default initializer called a default member initializer . Consider a revision of complex (§4.2.1):
> 当定义类的数据成员时，我们可以为它们提供一个默认初始化值，这被称为默认成员初始化

```cpp
class complex {
    double re = 0;
    double im = 0; // representation: two doubles with default value 0.0
public:
    complex(double r, double i): re{r}, im{i} {} // construct complex from two scalars: {r, i}
    complex(double r): re{r} {} // construct complex from one scalar: {r,0}
    complex() {} // default complex: {0.0}
};
```

The default value is used whenever a constructor doesn’t provide a value. This simplifies code and helps us to avoid accidentally leaving a member uninitialized.
> 初始值会在构造函数没有为类内数据成员提供值时使用，这可以简化代码，并避免未初始化的成员

## 5.2 Copy and Move
By default, objects can be copied. This is true for objects of user-defined types as well as for built- in types. The default meaning of copy is memberwise copy: copy each member. For example, using complex from $\S4.2.1$ :
> 默认情况下，用户定义的以及内建的类型的对象都是可以拷贝的
> 拷贝的默认含义就是逐成员的拷贝：拷贝每一个成员

```cpp
void test(complex z1)
{
    complex z2 {z1}; // copy initialization
    complex z3;
    z3 = z2; // copy assignment
}
```

Now z1 , z2 , and z3 have the same value because both the assignment and the initialization copied both members.
> 本例中，`z1,z2,z3` 最后会有相同的值，因为赋值和初始化都会拷贝类中的数据成员

When we design a class, we must always consider if and how an object might be copied. For simple concrete types, memberwise copy is often exactly the right semantics for copy. For some sophisticated concrete types, such as Vector , memberwise copy is not the right semantics for copy; for abstract types it almost never is.
> 设计一个类时，必须考虑其对象是否会被拷贝，以及如何被拷贝
> 对于简单的具体类型，逐成员的拷贝往往就是所需要的正确语义
> 对于复杂的具体类型，例如 `Vector` ，逐成员的拷贝就不是正确的语义
> 对于抽象类型，逐成员拷贝一定不是正确语义

### 5.2.1 Copying Containers
When a class is a resource handle – that is, when the class is responsible for an object accessed through a pointer – the default memberwise copy is typically a disaster. Memberwise copy would violate the resource handle’s inv ariant (§3.5.2). For example, the default copy would leave a copy of a Vector referring to the same elements as the original:
> 当类是作为资源句柄，也就是会通过指针访问一个对象，此时逐成员拷贝一般是错误的，它一般会违反资源句柄的不变式

```cpp
void bad_copy(Vector v1)
{
    Vector v2 = v1; // copy v1's representation into v2
    v1[0] = 2; // v2[0] is now also 2
    v2[1] = 3; // v1[1] is now also 3
}
```

Assuming that v1 has four elements, the result can be represented graphically like this:

Fortunately, the fact that Vector has a destructor is a strong hint that the default (memberwise) copy semantics is wrong and the compiler should at least warn against this example. We need to define better copy semantics.

Copying of an object of a class is defined by two members: a copy constructor and a copy assignment :
> 一个类通过拷贝构造函数和拷贝赋值函数来定义其对象的拷贝行为

```cpp
class Vector {
private:
    double* elem; // elem points to an array of sz doubles
    int sz;
public:
    Vector(int s); // constructor: establish invariant, acquire resources
    ~Vector() { delete[] elem; } // destructor: release resources 
    
    Vector(const Vector& a); // copy constructor
    Vector& operator=(const Vector& a); //copy assignment
    
    double& operator[](int i);
    const double& operator[](int i) const;

    int size() const;
};
```

A suitable definition of a copy constructor for Vector allocates the space for the required number of elements and then copies the elements into it so that after a copy each Vector has its own copy of the elements:
> Vector 的正确拷贝构造函数应该分配特定大小的空间，然后将元素逐个拷贝过去，使得每个 Vector 都有自己的元素拷贝

```cpp
Vector::Vector(const Vector& a) // copy constuctor
    :elem{new double[a.sz]}, //allocate space for elements
     sz{a.sz}
{
    for(int i = 0; i!=sz; i++)
        elem[i] = a.elem[i]; // copy elements
}
```

The result of the $\scriptstyle{\mathbf{v}}2={\mathbf{v}}7$ example can now be presented as:

Of course, we need a copy assignment in addition to the copy constructor:
> 还需要拷贝赋值函数

```cpp
Vector& Vector::operator=(const Vector& a) // copy assignment
{
    double* p = new double[a.sz];
    for(int i = 0; i != a.sz; i++)
        p[i] = a.elem[i];
    delete[] elem; // delete old elements
    elem = p;
    sz = a.sz;
    return *this;
}
```

The name `this` is predefined in a member function and points to the object for which the member function is called.
> `this` 为成员函数中预定义的名字，指向了调用成员函数的对象

### 5.2.2 Moving Containers
We can control copying by defining a copy constructor and a copy assignment, but copying can be costly for large containers. We avoid the cost of copying when we pass objects to a function by using references, but we can’t return a reference to a local object as the result (the local object would be destroyed by the time the caller got a chance to look at it). Consider:
> 拷贝是昂贵的，传递实参时，我们通过引用避免拷贝，但我们不能在函数中返回一个对局部对象的引用，局部对象在函数作用域结束后就会被摧毁
> 考虑：

```cpp
Vector operator+(const Vector& a, const Vector& b)
{
    if(a.size() != b.size())
        throw Vector_size_mismatch{};

    Vector res(a.size());

    for(int i = 0; i!=a.size(); i++)
        res[i] = a[i] + b[i];

    return res;
}
```

Returning from a $+$ involves copying the result out of the local variable res and into some place where the caller can access it. We might use this $^+$ like this:
> 上例定义的 `+` 中会在返回时将 `res` 拷贝到调用者可以访问的位置，也就是返回时会调用拷贝构造（默认的或者用户定义的）将返回对象拷贝到调用者栈帧中的接受返回值的对象

```cpp
void f(const Vector& x, const Vector& y, const Vector& z)
{
    Vector r; 
    r = x + y + z;
}
```

That would be copying a Vector at least twice (one for each use of the $^+$ operator). If a Vector is large, say, 10,000 double s, that could be embarrassing. 
> 上例中，两个 `+` 就会导致至少两次的 Vector 拷贝

The most embarrassing part is that res in operator+ $\cdot ()$ is never used again after the copy. We didn’t really want a copy; we just wanted to get the result out of a function: we wanted to move a Vector rather than copy it. Fortunately, we can state that intent:
> 关键问题在于 `operator+()` 中的 `res` 在拷贝之后是不会再被使用的
> 我们实际的目的不在于拷贝，而是想要将结果返回到函数外，也就是移动它

```cpp
class Vector {
    Vector(const Vector& a); // copy constructor
    Vector& operator=(const Vector& a); // copy assignment
    Vector(Vector&& a); // move constructor
    Vector& operator=(Vector&& a); // move assignment
};
```

Given that definition, the compiler will choose the move constructor to implement the transfer of the return value out of the function. This means that $\scriptstyle{\mathsf{r}}=\!\mathbf{x}+\mathbf{y}+\mathbf{z}$ will involve no copying of Vector s. Instead, Vector s are just moved.
> 定义好了移动构造函数之后，编译器就会使用移动构造函数来实现函数返回值的转移，也就是用移动构造函数构建调用者栈帧中的接受返回值的对象

As is typical, Vector ’s move constructor is trivial to define:

```cpp
Vector::Vector(Vector&& a)
    : elem{a.elem}, // grab the elements from a
     sz{a.sz}
{
    a.elem = nullptr; // now a has no elements
    a.sz = 0;
}
```

The && means ‘‘rvalue reference’’ and is a reference to which we can bind an rvalue. The word ‘‘rvalue’’ is intended to complement ‘‘lvalue,’’ which roughly means ‘‘something that can appear on the left-hand side of an assignment.’’ So an rvalue is – to a first approximation – a value that you can’t assign to, such as an integer returned by a function call. Thus, an rvalue reference is a refer- ence to something that nobody else can assign to, so we can safely ‘‘steal’’ its value. The res local variable in operator+() for Vector s is an example.
> `&&` 意思是右值引用，即可以绑定到右值的引用
> 右值的概念和左值的概念相对，右值不能出现在赋值语句左边，也就是不可赋值
> 本列中，我们的移动构造函数没有对数据进行拷贝，甚至移动，移动构造函数的语义就是将参数指向的右值对象的数据的所有权转移到自己身上

> [! 左值和右值]
> 左值 lvalue：指可以出现在赋值语句左边的值，左值通常代表一个标识符，它可以指向一个已命名的存储位置，这个位置在程序运行期间一直存在，其特点有：
>   - 可寻址性：左值可以取地址 (`&x`)，因此可以出现在赋值语句的左侧
>   - 持久性：左值通常对应于存储在内存中的持久对象
>  右值 rvalue：指只能出现在赋值语句右边的值，右值通常是临时的，不具有持久性，通常代表一个匿名的、临时存在的值或对象，其特点有：
>    - 不可寻址性：右值通常不可取地址 (`&10` 是非法的)
>    - 临时性：右值通常代表临时的对象或值，这些对象或值在表达式计算完成后即被销毁


> [! 右值引用类型]
> 右值引用类型允许引用右值，其语法是在类型后加上 `&&` ，表示一个绑定到右值的引用，其特点有：
>   - 临时对象引用：绑定到临时对象
>   - 移动语义：支持将临时对象的资源“移动”到另一个对象中，而不是拷贝
>   - 完美转发：用于模板函数中的完美转发，确保传入的参数以最自然的形式传递给另一个函数或构造函数

A move constructor does not take a const argument: after all, a move constructor is supposed to remove the value from its argument. A move assignment is defined similarly.
> 移动构造函数不能接受 `const` 参数，因为它需要“移除”其参数中的值
> 移动拷贝函数的定义是类似的

A move operation is applied when an rvalue reference is used as an initializer or as the right- hand side of an assignment.
> 当一个右值引用被作为初始化值时，或者作为赋值语句的右边时，移动操作就会被调用（移动构造和移动赋值）

After a move, a moved-from object should be in a state that allows a destructor to be run. Typi- cally, we also allow assignment to a moved-from object. The standard-library algorithms (Chapter 12) assumes that. Our Vector does that.
> 当一个对象被移动之后，被移动的对象应该处于一种可以安全析构的状态（当然它的数据被“清空”了）
> 此外，通常我们也允许对被移动后的对象进行赋值，标准库算法就假定被移动的对象仍然可以被使用，至少在该对象析构之前是如此（注意右值可不一定都是临时对象）

Where the programmer knows that a value will not be used again, but the compiler can’t be expected to be smart enough to figure that out, the programmer can be specific:
> 如果程序员明确一个值不会再被使用，可以明确使用 `std::move` 来调用移动赋值/构造函数

```cpp
Vector f()
{
    Vector x(1000);
    Vector y(2000);
    Vector z(3000);
    z = x; // we get a copy (x might be used later in f())
    y = std::move(x); // we get a move (move assignmet)
    // better not use x here
    return z; // we get a move
}
```

The standard-library function move () doesn’t actually move anything. Instead, it returns a reference to its argument from which we may move – an rvalue reference ; it is a kind of cast (§4.2.3).
> 标准库函数 `move()` 并不会移动任何东西，而是返回一个指向其参数的右值引用，可以将它视作一种类型转化

Just before the return we have:

When we return from $\mathfrak{f}()$ , z is destroyed after its elements has been moved out of $\mathsf{f}()$ by the return . However, y ’s destructor will delete[] its elements.
> `f()` 返回之后，`z` 会在它的所有权被  `return` 调用的移动构造移交之后被摧毁，`y` 的析构函数会 `delete[]` 它的所有元素

The compiler is obliged (by the $\mathrm{C++}$ standard) to eliminate most copies associated with initial- ization, so move constructors are not invoked as often as you might imagine. This copy elision eliminates even the very minor overhead of a move. On the other hand, it is typically not possible to implicitly eliminate copy or move operations from assignments, so move assignments can be critical for performance.
> C++的编译器被要求消除大多数和初始化相关的拷贝，因此移动构造函数并不是被频繁调用的（有时候编译器会直接优化拷贝构造，即拷贝省略），编译器的拷贝省略甚至开销比移动构造还小
> 但让编译器隐式从赋值操作中消除拷贝或移动是不可能的，因此显式使用移动赋值有助于提升性能

## 5.3 Resource Management
By defining constructors, copy operations, move operations, and a destructor, a programmer can provide complete control of the lifetime of a contained resource (such as the elements of a con- tainer). 
> 通过构造函数、拷贝操作、移动操作、析构函数，我们已经对容器内资源的生命周期具有了完全的控制

Furthermore, a move constructor allows an object to move simply and cheaply from one scope to another. That way, objects that we cannot or would not want to copy out of a scope can be simply and cheaply moved out instead. Consider a standard-library thread representing a concurrent activity (§15.2) and a Vector of a million double s. We can’t copy the former and don’t want to copy the latter.
> 对于不能拷贝以及拷贝开销过大的对象，我们就可以使用移动构造

```cpp
Vector init(int n)
{
    thread t {heartbeat}; // run heartbeat concurrently(int a seperate thread)
    my_threads.push_back(std::move(t)); // move t into my_threads
    // more initialization
    Vector vec(n);
    for(int i = 0; i != vec.size(); i++)
        vec[i] = 777;
    return vec; // move vec out of init()
}
auto v = init(1'000'000); // start heartbeat and initialize v
```

Resource handles, such as Vector and thread , are superior alternatives to direct use of built-in point- ers in many cases. In fact, the standard-library ‘‘smart pointers,’’ such as unique_ptr , are themselves resource handles (§13.2.1).
> 资源句柄（管理资源的对象），例如 `Vector` 和 `thread` 比直接使用内建指针的优先级更高
> 标准库 `unique_ptr` 等智能指针本身就是资源句柄

I used the standard-library vector to hold the thread s because we don’t get to parameterize our simple Vector with an element type until $\S6.2$ .

In very much the same way that new and delete disappear from application code, we can make pointers disappear into resource handles. In both cases, the result is simpler and more maintainable code, without added overhead. In particular, we can achieve strong resource safety ; that is, we can eliminate resource leaks for a general notion of a resource. Examples are vector s holding memory, thread s holding system threads, and fstream s holding file handles.
> 让 `new/delete` 从应用代码中消失，以及利用资源句柄让指针从应用代码中消失，可以让我们的应用更简洁并且更可维护，我们的资源也更加安全，即为一种通用的资源概念消除了资源泄露的可能性
> 例如 vector 持有内存，thread 持有系统线程，fstream 持有文件句柄

In many languages, resource management is primarily delegated to a garbage collector. $\mathrm{C++}$ also offers a garbage collection interface so that you can plug in a garbage collector. Howev er, I consider garbage collection the last choice after cleaner, more general, and better localized alterna- tives to resource management have been exhausted. My ideal is not to create any garbage, thus eliminating the need for a garbage collector: Do not litter!
> 许多语言使用垃圾收集器管理资源
> C++也提供了 GC 接口，便于我们插入 GC，但 C++ 更偏好的思想更好的资源管理以不产生任何垃圾，就不需要 GC

Garbage collection is fundamentally a global memory management scheme. Clever implemen- tations can compensate, but as systems are getting more distributed (think caches, multicores, and clusters), locality is more important than ever.
> GC 本质是全局的内存管理方法，但随着系统变得更加分布，局部性就更重要

Also, memory is not the only resource. A resource is anything that has to be acquired and (explicitly or implicitly) released after use. Examples are memory, locks, sockets, file handles, and thread handles. Unsurprisingly, a resource that is not just memory is called a non-memory resource . A good resource management system handles all kinds of resources. Leaks must be avoided in any long-running system, but excessive resource retention can be almost as bad as a leak. For example, if a system holds on to memory, locks, files, etc. for twice as long, the system needs to be provisioned with potentially twice as many resources.
> 内存也不是唯一的资源
> 需要被获取，并在使用后释放的都是资源，例如内存、锁、套接字、文件句柄、线程句柄
> 资源泄露以及资源持有过久都不利于长期运行的系统稳定

Before resorting to garbage collection, systematically use resource handles: let each resource have an owner in some scope and by default be released at the end of its owners scope. In $\mathrm{C++}$ , this is known as RAII ( Resource Acquisition Is Initialization ) and is integrated with error handling in the form of exceptions. Resources can be moved from scope to scope using move semantics or ‘‘smart pointers,’’ and shared ownership can be represented by ‘‘shared pointers’’ (§13.2.1).
> 系统地使用资源句柄：让每个资源在某个作用域都有拥有者，并且默认在其拥有者的作用域结束后被释放
> 这就是 RAII，它以异常的形式和错误处理集成
> 资源可以使用移动语义或智能指针从作用域移动到作用域
> 所有权的共享可以由共享指针表示

In the $\mathrm{C++}$ standard library, RAII is pervasive: for example, memory ( string , vector , map , unordered_map , etc.), files ( ifstream , ofstream , etc.), threads ( thread ), locks ( lock_guard , unique_lock , etc.), and general objects (through unique_ptr and shared_ptr ). The result is implicit resource man- agement that is invisible in common use and leads to low resource retention durations.
> C++标准库中，RAII 无处不在
> 内存：`string/vector/map/unordered_map`
> 文件：`ifstream/ofstream`
> 线程：`thread`
> 锁：`lock_guard/unique_lock`
> 通用对象：`unique_ptr/shared_ptr`
> 这使得资源管理是隐式的，并且资源持有也不会持久

## 5.4 Conventional Operations
Some operations have conventional meanings when defined for a type. These conventional mean- ings are often assumed by programmers and libraries (notably, the standard library), so it is wise to conform to them when designing new types for which the operations make sense.
> 一些操作具有常规的语义，设计新类型时，最好遵循它们

• Comparisons: $==,\,!=,<,<=,>,$ , and $>=(\S5.4.1)$ 
• Container operations: siz e () , begin () , and end () (§5.4.2) 
• Input and output operations: ${}>>{}$ and $<<(\S5.4.3)$ 
• User-defined literals (§5.4.4) • swap () (§5.4.5) • Hash functions: hash $<>$ (§5.4.6)

### 5.4.1 Comparisons
The meaning of the equality comparisons ( $\acute{=}$ and $!=$ ) is closely related to copying. After a copy, the copies should compare equal:
> 等价比较的意义和拷贝紧密相关，在拷贝之后，拷贝件和原件应该在比较下相等

```cpp
X a = something;
X b = a;
assert(a == b); // if a != b here, somthing is very odd
```

When defining $==$ , also define $!=$ and make sure that ${\mathsf{a l}}{\mathsf{=}}{\mathsf{b}}$ means ! $(a{=}b)$ .
> 定义 `==` 时，记得用 `!(a==b)` 定义 `!=`

Similarly, if you define $<.$ , also define $<=,>,$ , and make sure that the usual equivalences hold: 
• $\scriptstyle{a<=b}$ means (a<b)|| $(a\!\!=\!\!=\!\! b)$ ) and ! $({\tt b}\!<\!{\tt a})$ . 
• a>b means b<a \. 
• $a>=b$ means $(\mathsf{a}\!\!>\!\!\mathsf{b})||(\mathsf{a}\!\!=\!\!\mathsf{b})$ and !(a<b) .

> 类似地：
> 定义了 `<` 之后，记得定义
> `a<=b` : `(a < b) || (a == b)` 或者 `!(b < a)`
> `a>b` : `b<a`
> `a>=b` : `(a>b) || (a==b)` 或者 `!(a<b)`

To giv e identical treatment to both operands of a binary operator, such as $==,$ , it is best defined as a free-standing function in the namespace of its class. For example:
> 对于对称的运算符例如 `==` ，最好直接在其类的所处的命名空间中定义

```cpp
namespace NX {
class X {
    // ...
};
bool operator=(const X&, const X&);
};
```

### 5.4.2 Container Operations
Unless there is a really good reason not to, design containers in the style of the standard-library containers (Chapter 11). In particular, make the container resource safe by implementing it as a handle with appropriate essential operations (§5.1.1, §5.2).
> 以标准库容器的风格设计容器，为其实现合适的必要操作，是容器可以作为资源句柄

The standard-library containers all know their number of elements and we can obtain it by calling size() . For example:
> 所有的标准库容器都通过 `size()` 获取元素数量

```cpp
for(size_t i = 0; i < c.size(); i++) // size_t is the name of the type returned by a standard-library size()
    c[i] = 0;
```

However, rather than traversing containers using indices from 0 to size() , the standard algorithms (Chapter 12) rely on the notion of sequence s delimited by pairs of iterator s:
> 标准算法依赖于迭代器的概念来遍历容器，使得算法可以独立于容器的具体类型工作

```cpp
for(auto p = c.begin(); p != c.end(); p++)
    *p = 0;
```

Here, c.begin () is an iterator pointing to the first element of c and c.end () points one-beyond-the-last element of c . Like pointers, iterators support $^{++}$ to move to the next element and $^*$ to access the value of the element pointed to. This iterator model (§12.3) allows for great generality and efficiency. Iterators are used to pass sequences to standard-library algorithms. For example:
> `c.begin()` 指向 `c` 的第一个元素，`c.end()` 指向 `c` 的最后一个元素的后一个元素
> 迭代器支持 `++` 以移动到下一个元素，支持 `*` 进行来访问所指向的元素
> 迭代器用于向标准库算法传递序列（表示序列的起始），例如：

```cpp
sort(v.begin(), v.end());
```

For details and more container operations, see Chapter 11 and Chapter 12.

Another way of using the number of elements implicitly is a range- for loop:
> range-for-loop 是隐式使用容器元素数量的另一种方式

```cpp
for(auto& x : c)
    x = 0;
```

This uses c.begin () and c.end () implicitly and is roughly equivalent to the more explicit loop.
> range-for-loop 实际上隐式使用了 `c.begin()` 和 `c.end()` 

### 5.4.3 Input and Output Operations
For pairs of integers, $<<$ means left-shift and ${}>>{}$ means right-shift. However, for iostreams , they are the output and input operator, respectively (§1.8, Chapter 10). For details and more I/O operations, see Chapter 10.
> 操作数都是整形时，`<</>>` 意思是左移和右移
> 对于 `iostream` ，`<</>>` 是输入和输出操作符

### 5.4.4 User-Defined Literals
One purpose of classes was to enable the programmer to design and implement types to closely mimic built-in types. Constructors provide initialization that equals or exceeds the ﬂexibility and efficiency of built-in type initialization, but for built-in types, we have literals:

• `123` is an `int` . 
• `0xFF00u` is an `unsigned int` . 
• `123.456` is a `double` . 
• `"Surprise!"` is a `const char[10]` .

It can be useful to provide such literals for a user-defined type also. This is done by defining the meaning of a suitable suffix to a literal, so we can get
> 除了基本类型以外，一些用户定义类型也有字面值，它们通过后缀指定

• `"Surprise!"s` is a `std:: string` . 
• `123s` is `second` s. 
• `12.7i` is `imaginary` so that `12.7i+47` is a `complex` number (i.e., `{47,12.7}` ).

In particular, we can get these examples from the standard library by using suitable headers and namespaces:
> 在合适的头文件和命名空间下，以上几个例子就是成立的：

Standard-Library Suffixes for Literals

|   `<chrono>`   | `std::literals::chrono_literals`  | `h,min,s,ms,us,ns` |
| :------------: | :-------------------------------: | :----------------: |
|   `<string>`   | `std::literals::string_literals`  |        `s`         |
| `<string_new>` | `std::literals::string_literals`  |        `sv`        |
|  `<complex>`   | `std::literals::complex_literals` |     `i,il,if`      |

Unsurprisingly, literals with user-defined suffixes are called user-defined literals or UDL s. Such lit- erals are defined using literal operators . A literal operator converts a literal of its argument type, followed by a subscript, into its return type. For example, the i for imaginar y suffix might be imple- mented like this:
> 具有用户定义后缀的字面值称为用户定义的字面值
> 这类字面值通过字面值运算符定义
> 字面值运算符将 `<a literal of its argument type><subscript>` 模式的字面值转化为其返回类型
> 例如，`imaginary` 的后缀 `i` 可以实现为：

```cpp
constexpr complex<double> operator""i(long double arg) // imaginary literal
{
    return {0, arg};
}
```

Here

• The `operator""` indicates that we are defining a literal operator. 
• The `i` after the ‘‘literal indicator’’ `""` is the suffix to which the operator gives a meaning. 
• The argument type, `long double` , indicates that the suffix ( `i` ) is being defined for a ﬂoating- point literal. 
• The return type, `complex<double>` , specifies the type of the resulting literal.

Given that, we can write

```cpp
complex<double> z = 2.7182818+6.283185i;
```

### 5.4.5 `swap()`
Many algorithms, most notably sort() , use a swap() function that exchanges the values of two objects. Such algorithms generally assume that swap () is very fast and doesn’t throw an exception. The standard-library provides a std:: swap (a, b) implemented as three move operations: (tmp=a, $\scriptstyle{\mathfrak{a}}={\mathfrak{b}},$ , $\mathtt{b=t m p})$ . If you design a type that is expensive to copy and could plausibly be swapped (e.g., by a sort function), then give it move operations or a swap () or both. Note that the standard-library con- tainers (Chapter 11) and string (§9.2.1) have fast move operations.
> `sort()` 使用 `swap()` 交换对象
> 标准库的 `std::swap(a,b)` 实现为三个移动操作 `{tmp = a, a = b, b = tmp}`
> 如果我们定义的类型拷贝较昂贵，我们需要自己重定义 `swap` 或移动运算符
> 标准库容器和 `string` 都有快速的移动运算符

### 5.4.6 `hash<>`
The standard-library `unordered_map<K, V>` is a hash table with `K` as the key type and `V` as the value type (§11.5). To use a type `X` as a key, we must define `hash<X>` . The standard library does that for us for common types, such as `std:: string` .
> 标准库 `unordered_map<K, V>` 是以 `K` 为键类型，以 `V` 为值类型的哈希表
> 要使用类型 `X` 为键，必须定义 `hash<X>` ，标准库为常用类型定义了 `hash<X>` ，例如 `std::string`

## 5.5 Advice
[1] Control construction, copy, move, and destruction of objects; $\S5.1.1$ ; [CG: R.1].
[2] Design constructors, assignments, and the destructor as a matched set of operations; §5.1.1; [CG: C.22].
[3] Define all essential operations or none; $\S5.1.1$ ; [CG: C.21].
[4] If a default constructor, assignment, or destructor is appropriate, let the compiler generate it (don’t rewrite it yourself); $\S5.1.1$ ; [CG: C.20].
[5] If a class has a pointer member, consider if it needs a user-defined or deleted destructor, copy and move; $\S5.1.1$ ; [CG: C.32] [CG: C.33].
[6] If a class has a user-defined destructor, it probably needs user-defined or deleted copy and move; $\S5.2.1$ .
[7] By default, declare single-argument constructors explicit ; $\S5.1.1$ ; [CG: C.46].
[8] If a class member has a reasonable default value, provide it as a data member initializer; $\S5.1.3$ ; [CG: C.48].
[9] Redefine or prohibit copying if the default is not appropriate for a type; §5.2.1, $\S4.6.5$ ; [CG: C.61].
[10] Return containers by value (relying on move for efficiency); §5.2.2; [CG: F.20].
[11] For large operands, use const reference argument types; $\S5.2.2$ ; [CG: F.16].
[12] Provide strong resource safety; that is, never leak anything that you think of as a resource; $\S5.3$ ; [CG: R.1].
[13] If a class is a resource handle, it needs a user-defined constructor, a destructor, and non- default copy operations; $\S5.3$ ; [CG: R.1].
[14] Overload operations to mimic conventional usage; $\S5.4$ ; [CG: C.160].
[15] Follow the standard-library container design; $\S5.4.2$ ; [CG: C.100].
> 1. 通过定义构造、拷贝、移动、析构函数来控制对象的构造、拷贝、移动、析构
> 2. 将构造、赋值、析构设计为配套的操作集合
> 3. 定义所有的必要操作或者不定义
> 4. 如果默认构造、赋值、析构满足要求，就让编译器生成，不需要重写
> 5. 如果类有指针成员，考虑是否需要用户定义的或删除的析构、拷贝、移动函数
> 6. 如果类由用户定义的析构函数，类可能还需要用户定义的或删除的拷贝和移动函数
> 7. 默认情况下，显式声明单参数的构造函数
> 8. 如果类成员由合理的默认值，将其作为数据成员初始化值
> 9. 如果默认拷贝对于该类型不合适，重定义或禁止其拷贝函数
> 10. 按值返回容器（使用移动语义提高效率）
> 11. 多使用常引用实参类型
> 12. 不要泄露我们认为是资源的任何东西
> 13. 若类是一个资源句柄，它需要定义构造函数、析构函数、非默认拷贝函数
> 14. 重载运算以模仿惯例的用法
> 15. 遵循标准库容器设计

# 6 Templates
## 6.1 Introduction
Someone who wants a vector is unlikely always to want a vector of double s. A vector is a general concept, independent of the notion of a ﬂoating-point number. Consequently, the element type of a vector ought to be represented independently. A template is a class or a function that we parame- terize with a set of types or values. We use templates to represent ideas that are best understood as something general from which we can generate specific types and functions by specifying argu- ments, such as the vector ’s element type double .
> 模板是一类函数或类，我们用一系列值或者类型来参数化模板

## 6.2 Parameterized Types
We can generalize our vector-of-doubles type to a vector-of-anything type by making it a template and replacing the specific type double with a type parameter. For example:
> 我们使用模板来抽象化 vector，使用一个类型参数来替代具体类型

```cpp
template<typename T>
class Vector {
private:
    T* elem; // elem points to an array of sz elements of type T
    int sz;
public:
    explicit Vector(int s); // constructor: establish invariant, acquire resources
    ~Vector() { delete[] elem; } // destructor: release resources
    // .. copy and move operationis ..
    T& operator[](int i); // for non-const Vectors
    const T& operator[](int i) const; // for const Vectors
    int size() const { return sz; }
};
```

The template<typename $\mathsf{T}\!>$ prefix makes T a parameter of the declaration it prefixes. It is $\mathbf{C++}$ ’s ver- sion of the mathematical ‘‘for all T’’ or more precisely ‘‘for all types T.’’ If you want the mathe- matical ‘‘for all T, such that P (T),’’ you need concepts $(\S6.2.1,\,\S7.2)$ . Using class to introduce a type parameter is equivalent to using typename , and in older code we often see template<class $\mathsf{T}\!>$ as the prefix.
> `template<typename T>` 的前缀使得 `T` 作为它之后的声明的一个形参，可以理解其语义为 "for all T" 或 “for all types T , such that P(T)”
> `template<class T>` 和 `template<typename T>` 等价

The member functions might be defined similarly: 
> 成员函数的定义和类定义类似：

```cpp
template<typename T>
Vector<T>::Vector(int s)
{
    if (s < 0)
        throw Negative_size{};
    elem = new T[s];
    sz = s;
}

template<typename T>
const T& Vector<T>::operator[](int i) const
{
    if ( i < 0 || size <= i)
        throw out_of_range{"Vector::operator[]"};
    return elem[i];
}
```

Given these definitions, we can define Vector s like this:

```cpp
Vector<char> vc(200); // vector of 200 characters
Vector<string> vs(17); // vector of 17 strings
Vector<list<int>> vli(45); // vector of 45 lists of integers
```

The ${}>>{}$ in `Vector<list<int>>` terminates the nested template arguments; it is not a misplaced input operator.

We can use Vector s like this:

```cpp
void write(const Vector<string>& vx) // Vector of some strings
{
    for(int i = 0; i != vs.size(); i++)
        cout << vx[i];
}
```

To support the range-for loop for our Vector , we must define suitable begin () and end () functions:
> 为了支持使用 range-for 来遍历 `Vector` ，我们需要定义合适的 `begin()` 和 `end()` 函数

```cpp
template<typename T>
T* begin(Vector<T>& x)
{
    return x.size() ? &x[0] : nullptr; // pointer to first element or nullptr
}

template<typename T>
T* end(Vector<T>& x)
{
    return x.size() ? &x[0] + x.size() : nullptr; // pointer to one-past-last element
}
```

Given those, we can write:

```cpp
void f2(Vector<sting>& vs) // Vector of some strings
{
    for(auto& s : vs)
        cout << s;
}
```

Similarly, we can define lists, vectors, maps (that is, associative arrays), unordered maps (that is, hash tables), etc., as templates (Chapter 11).

Templates are a compile-time mechanism, so their use incurs no run-time overhead compared to hand-crafted code. In fact, the code generated for `Vector<double>` is identical to the code generated for the version of Vector from Chapter 4. Furthermore, the code generated for the standard-library `vector<double>` is likely to be better (because more effort has gone into its implementation).
> 模板是编译时机制，因此使用模板不会引起运行时开销
> 为 `Vector<double>` 生成的代码和我们对应手写的代码是一样的

A template plus a set of template arguments is called an instantiation or a specialization . Late in the compilation process, at instantiation time , code is generated for each instantiation used in a program $(\S7.5)$ . The code generated is type checked so that the generated code is as type safe as handwritten code. Unfortunately, that type check often occurs late in the compilation process, at instantiation time. > 模板加上一组模板参数称为实例化或特化 > 在编译过程中的实例化事件，编译器会为程序中用到的实例生成代码，生成的代码也会进行类型检查，因此生成的代码是类型安全的

### 6.2.1 Constrained Template Arguments (C++20)
Most often, a template will make sense only for template arguments that meet certain criteria. For example, a Vector typically offers a copy operation, and if it does, it must require that its elements must be copyable. That is, we must require that Vector ’s template argument is not just a typename but an Element where ‘‘ Element ’’ specifies the requirements of a type that can be an element:
> 许多情况下，模板仅在模板参数满足特定准则下才有意义
> 例如，Vector 通常提供拷贝操作，故要求其元素是可拷贝的
> 因此，模板的实参应不仅仅是一个类型名称，而应该是一个“元素”
> `Element` 指定了一个类型要成为元素的要求

```cpp
template<Element T>
class Vector {
private:
    T* elem; // elem points to an array of sz elements of type T
    int sz;
}
```

This `template<Element T>` prefix is $\mathbf{C++}^{\dagger}$ ’s version of mathematic’s ‘‘for all T such that Element (T) ’’; that is, Element is a predicate that checks whether T has all the properties that a Vector requires. Such a predicate is called a concept $(\S7.2)$ . A template argument for which a concept is specified is called a constrained argument and a template for which an argument is constrained is called a constrained template .
> `template<Element T>` 可以解释为：对于所有的 `T` ，使得 `Element(T)`
> 也就是说，`Element` 是一个谓词，它检查 `T` 是否具有 Vector 要求的所有性质
> 这类谓词被称为 concept
> 一个指定了 concept 的模板实参被称为受限制的实参，实参是受限制实参的模板也称为受限制的模板

It is a compile-time error to try to instantiate a template with a type that does not meet its requirements. For example:
> 尝试不满足需求的类型实例化模板会导致编译时错误

```cpp
Vector<int> v1; // OK: we can copy and int
Vector<thread> v2; // error: we can't copy a standard thread
```

Since $\mathrm{C++}$ does not officially support concepts before C++20 , older code uses unconstrained template arguments and leaves requirements to documentation.
> C++20开始支持 concept，旧的代码使用无限制模板，将模板参数要求在文档中注明

### 6.2.2 Value Template Arguments
In addition to type arguments, a template can take value arguments. For example:
> 模板除了支持类型参数，还支持值参数

```cpp
template<typename T, int N>
struct Buffer {
    using value_type = T;
    constexpr int size() { return N; }
    T[N];
};
```

The alias ( `value_type` ) and the constexpr function are provided to allow users (read-only) access to the template arguments. 
> `using value_type = T`  定义了一个类型别名，也就是定义了 `value_type` 为模板参数 T 的同义词，之后，在类的其他部分中引用 `T` 时，可以使用 `value_type` 来提高代码的可读性
> `constexpr int size()` 为常量表达式函数，用于返回模板的值参数（也为常量表达式）

Value arguments are useful in many contexts. For example, Buffer allows us to create arbitrarily sized buffers with no use of the free store (dynamic memory): 

```cpp
Buffer<char, 1024> glob; // glob buffer of characters (statically allocated)

viod fct() 
{
    Buffer<int, 10> buf; // local buffer of integers (on the stack)
}
```

A template value argument must be a constant expression.
> 模板的值参数必须是常量表达式 ( `constexpr` )

### 6.2.3 Template Argument Deduction
Consider using the standard-library template pair :
> 标准库的模板 `pair` 

```cpp
pair<int, double> p = {1, 5.2};
```

Many have found the need to specify the template argument types tedious, so the standard library offers a function, `make_pair()` , that deduces the template arguments of the `pair` it returns from its function arguments:
> 标准库提供了函数 `make_pair()` ，该函数返回实例化的 `pair` ，它从传入的函数实参中推导 `pair` 的模板类型参数

```cpp
auot p = make_pair(1, 5.2); // p is a pair<int, double>
```

This leads to the obvious question ‘‘Why can’t we just deduce template parameters from constructor arguments?’’ So, in C++17, we can. That is:
> C++17开始，支持直接从构造函数的参数推导模板类型参数

```cpp
pair p = {1, 5.2}; // p is a pair<int, double>
```

This is not just a problem with `pair` ; `make_` functions are very common. Consider a simple example:
> 标准库中其他的类型也有 `make_xxx` 函数

```cpp
template<typename T>
class Vector {
public:
    Vector(int);
    Vector(initializer_list<T>); // initializer-list constructor
};

Vector v1{1, 2, 3}; // deduce v1's element type from the initializer element type
Vector v2 = v1; // deduce v2's element type from v1's element type

auto p = new Vector{1, 2, 3}; // p points to a Vector<int>
Vector<int> v3(1); // here we need to be explicit about the element type(no element type is mentioned)
```

Clearly, this simplifies notation and can eliminate annoyances caused by mistyping redundant template argument types. However, it is not a panacea. Deduction can cause surprises (both for `make_ ` functions and constructors). Consider:

```cpp
Vector<string> vs1{"Hello", "World"}; 
Vector vs{"Hello", "World"}; // deduces to Vector<const char*>
Vector vs2{"Hello"s, "World"s};  // deduces to Vector<string>
Vector vs3{"Hello"s, "World"}; // error: the initializer list is not homogenous
```

The type of a C-style string literal is const char ∗ (§1.7.1). If that was not what was intended, use the s suffix to make it a proper string (§9.2). If elements of an initializer list have differing types, we cannot deduce a unique element type, so we get an error.
> 不添加 `s` 后缀时，自动推导出的是 C 风格的字符串字面值类型，即 `const char*`
> 如果初始化列表中的元素有不同的自动推导的类型，则会编译出错

When a template argument cannot be deduced from the constructor arguments, we can help by providing a deduction guide . Consider:
> 如果模板类型参数不能从构造函数实参推导时，我们可以提供引导帮助编译器推导

```cpp
template<typename T>
class Vector2 {
public:
    using value_type = T;
    // ...
    Vector2 (initializer_list<T>); // initializer-list constructor
    // ...
    template<typename Iter>
        Vector2(Iter b, Iter e); // [b:e) range constructor
    // ...
};

Vector2 v1{1, 2, 3}; // element type is int
Vector2 v2(v1.begin(), v1.begin() + 2);

```

Obviously, v2 should be a `Vector2<int>` , but without help, the compiler cannot deduce that. The code only states that there is a constructor from a pair of values of the same type. Without language support for concepts $(\S7.2)$ , the compiler cannot assume anything about that type. 
> 实例代码中通过 `template<typename Iter> Vector2(Iter b, Iter e)` 定义了模板构造函数，但该定义的语义仅仅是类存在一个接受两个相同类型参数的构造函数，在没有 concept 支持的情况下，编译器不能通过该构造函数的模板参数类型 `Iter` 推导出类模板参数的类型 `T`

To allow deduction, we can add a deduction guide after the declaration of Vector2 :
> 为了允许推导，我们在该构造函数的声明之后添加推导引导

```cpp
template<typename Iter>
Vector2(Iter, Iter) -> Vector2<typename Iter::value_type>;
```

That is, if we see a Vector2 initialized by a pair of iterators, we should deduce `Vector2::value_type` to be the iterator’s value type.
> 此时，编译器看到 `Vector2 v2(v1.begin(), v2.begin() + 2)` 调用时，它做的推导是：`v1.begin()` 的类型是 `std::vector<int>::iterator` ，故模板参数 `Iter` 就是 `std::vector<int>::iterator` ，而 v2 的模板参数就是 `std::vector<int>::iterator::value_type` 即 `int`

The effects of deduction guides are often subtle, so it is best to design class templates so that deduction guides are not needed. However, the standard library is full of classes that don’t (yet) use concept s $(\S7.2)$ and have such ambiguities, so it uses quite a few deduction guides.
> 最好设计不需要推导引导的类模板
> 但标准库有许多类尚未使用 concepts，故使用了一些推导引导

## 6.3 Parameterized Operations
Templates have many more uses than simply parameterizing a container with an element type. In particular, they are extensively used for parameter iz ation of both types and algorithms in the stan- dard library $(\S11.6,\S12.6)$ .
> 除了用于参数化容器的元素类型以外，在标准库中，模板还用于参数化类型和算法

There are three ways of expressing an operation parameterized by types or values: 

- A function template 
- A function object: an object that can carry data and be called like a function 
- A lambda expression: a shorthand notation for a function object

> 用类型或值来参数化某个操作的三种方法：
> 函数模板
> 函数对象（可以被调用的对象）
> lambda 表达式（函数对象的简写）

### 6.3.1 Function Templates
We can write a function that calculates the sum of the element values of any sequence that a range- for can traverse (e.g., a container) like this:
> 一个计算支持 range-for 遍历的序列中的所有值之和的函数定义如下：

```cpp
template<typename Sequence, typename Value>
Value sum(const Sequence& s, Value v)
{
    for(auto x : s)
        v += x;
    return v;
}
```

The Value template argument and the function argument v are there to allow the caller to specify the type and initial value of the accumulator (the variable in which to accumulate the sum):
> 其中模板参数 `Value` 和函数实参 `v` 用于允许调用者指定累加器的类型和初始值

```cpp
void user(Vector<int>& vi, list<double>& ld, vector<complex<double>>& vc)
{
    int x = sum(vi, 0); // the sum of a vector of ints (add ints)
    double d = sum(vi, 0.0); // the sum of a vector of ints (add doubles)
    double dd = sum(ld, 0.0); // the sum of a list of doubles
    auto z = sum(vc, complex{0.0, 0.0}); // the sum of a vector of complex<double> s
}
```

The point of adding int s in a double would be to gracefully handle a number larger than the largest int . 

Note how the types of the template arguments for `sum<Sequence ,Value>` are deduced from the function arguments. Fortunately, we do not need to explicitly specify those types.
> 函数模板的类型参数会从传入给函数的实参的类型中推导

This sum() is a simplified version of the standard-library accumulate() (§14.3).
> 示例的 `sum()` 函数就是标准库函数 `accumulate()` 函数的简化版本

A function template can be a member function, but not a virtual member. The compiler would not know all instantiations of such a template in a program, so it could not generate a vtbl (§4.4).
> 函数模板可以是成员函数，但不能是虚函数
> (函数模板在编译时根据静态推导的类型实例化为多个函数，在运行时，实际上并没有“模板”这一概念。虚函数表的构建同样在编译时完成，编译器为每个类添加一个表，该表用虚函数名称作为键，用指向该类的虚函数实现的指针作为值。
> 在运行时，程序动态判断调用虚函数的对象的实际类型，找到对应的虚函数表，索引到指向实际虚函数实现的指针，进而调用实际的虚函数实现。
> 一般的编译器实现都限制在一个虚函数仅占用一个表项，故虚函数表的大小实际上可以随着类型的声明就固定下来。
> 但如果允许函数模板为虚函数，由于函数模板实际实例化的数量需要在编译器完整阅读整个程序之后才能确定，则虚函数表的大小需要在编译器阅读整个程序之后才能确定，这样的实现较为麻烦，故 C++ 不支持函数模板为虚函数)

### 6.3.2 Function Objects
One particularly useful kind of template is the function object (sometimes called a functor ), which is used to define objects that can be called like functions. For example:
> 函数对象/仿函数也是常用的模板，函数对象即可以像函数一样被调用的对象

```cpp
template<typename T>
class Less_than {
    const T val; // value to compare againts
public:
    Less_than(const T& v): val{v} {}
    bool operator()(const T& x) const { return x < val; } // call operator
};
```

The function called `operator()` implements the ‘‘function call,’’‘‘call,’’ or ‘‘application’’ operator () . We can define named variables of type Less_than for some argument type:

```cpp
Less_than lti {42}; // lti(i) will compare i to 42 using < (i<42)
Less_than lts {"Backus"s}; // lts(s) will compare s to "Backus" using < (s<"Backus")
Less_than<string> lts2 {"Naur"}; // "Naur" is a C-style string, so we need <string> to get the right <
```

We can call such an object, just as we call a function:

```cpp
void fct(int n, const string& s)
{
    bool b1 = lti(n); // true if n < 42
    bool b2 = lts(s); // true if s < "Backus"
}
```

Such function objects are widely used as arguments to algorithms. For example, we can count the occurrences of values for which a predicate returns true :
> 函数对象常用作算法的参数，例如计数一个序列内谓词范围为真的值的数量：

```cpp
template<typename C, typename P>
    // requires Sequence<C> && Callable<P, Value_type<P>>
int count(const C& c, P pred)
{
    int cnt = 0;
    for(const auto& x : c)
        if(pred(x))
            ++cnt;
    return cnt;
}
```

A predicate is something that we can invoke to return true or false . For example:

```cpp
void f(const Vector<int>& vec, const list<string>& lst,int x, const string& s)
{
    cout<<"number of values less than"<< x <<":"<< count(vec, Less_than{x}) << '\n';
    cout<<"number of values less than"<< s <<":"<< count(lst, Less_than{s}) << '\n';
}
```

Here, `Less_than{x}` constructs an object of type `Less_than<int>` , for which the call operator compares to the `int` called `x` ; `Less_than{s}` constructs an object that compares to the `string` called `s` . The beauty of these function objects is that they carry the value to be compared against with them. We don’t hav e to write a separate function for each value (and each type), and we don’t hav e to intro- duce nasty global variables to hold values. Also, for a simple function object like `Less_than` , inlining is simple, so a call of `Less_than` is far more efficient than an indirect function call. The ability to carry data plus their efficiency makes function objects particularly useful as arguments to algorithms.
> 本例中，`Less_than{x}` 会通过模板参数推导自动构建 `Less_than<int>` 类的对象，类似地，`Less_than{s}` 会通过模板参数推导自动构建 `Less_than<string>` 类的对象

Function objects used to specify the meaning of key operations of a general algorithm (such as `Less_than` for `count()` ) are often referred to as policy objects .
> 函数对象常用于指定通用算法的关键操作的具体含义，例如 `count()` 算法中的 `Less_than` ，故也常被称为策略对象

### 6.3.3 Lambda Expressions
In $\S6.3.2$ , we defined Less_than separately from its use. That can be inconvenient. Consequently, there is a notation for implicitly generating function objects:
> lambda 表达式用于隐式地生成函数对象

```cpp
void f(const Vector<int>& vec, const list<string>& lst, int x, const string& s)
{
    cout << "number of values less than " << x 
         << ":" << count(vec, [&](int a){ return a<x;});
         << "\n";

    cout << "number of values less than " << s 
         << ":" << count(lst, [&](const string& a){ return a<s;});
         << "\n";
}
```

The notation `[&](int a){ return a<x; }` is called a lambda expression . It generates a function object exactly like `Less_than<int>{x}` . The `[&]` is a capture list specifying that all local names used in the lambda body (such as $\pmb{x}$ ) will be accessed through references. Had we wanted to ‘‘capture’’ only ${\bf x},$ , we could have said so: `[&x]` . Had we wanted to give the generated object a copy of $\mathbf{x}$ , we could have said so: `[=x]` . Capture nothing is `[ ]`  , capture all local names used by reference is `[&]` , and capture all local names used by value is `[=]`
> `[&](int a){ return a<x; }` 称为 lambda 表达式，它生成的函数对象实际和 `Less_than<int>{x}` 是一样的
> `[&]` 是捕获列表，指定需要在 lambda 表达式函数体内使用的局部名字，`&` 指定了通过引用捕获，也就是借由引用访问局部变量
> 如果只想捕获 `x` ，可以写为 `[&x]`
> 如果需要拷贝，可以写为 `[=x]`
> 不捕获，则写为 `[]`
> 引用捕获全部局部名字，写为 `[&]`，值捕获全部局部名字，写为 `[=]`

Using lambdas can be convenient and terse, but also obscure. For nontrivial actions (say, more than a simple expression), I prefer to name the operation so as to more clearly state its purpose and to make it available for use in several places in a program.

In $\S4.5.3$ , we noted the annoyance of having to write many functions to perform operations on elements of `vector` s of pointers and `unique_ptr` s, such as draw_all() and rotate_all() . Function objects (in particular, lambdas) can help by allowing us to separate the traversal of the container from the specification of what is to be done with each element.
> 函数对象可以用于分离对容器的遍历和对每个元素的具体操作

First, we need a function that applies an operation to each object pointed to by the elements of a container of pointers:
> 先定义一个函数，它遍历容器，为每个元素应用一个操作

```cpp
template<typename C, typename Oper>
void for_all(C& c, Oper op) // assume that C is a container of pointers
    // requires Sequence<C> && Callable<Oper, Value_type<C>>
{
    for(auto& x : c)
        op(x); // pass op() a reference to each element pointed to
}
```

Now, we can write a version of user() from $\S4.5$ without writing a set of `_all` functions:
> 然后将需要应用的操作定义为函数对象，作为实参传入该遍历函数即可

```cpp
void user2()
{
    vector<unique_ptr<Shape>> v;
    while(cin)
        v.push_back(read_shape(cin));
    for_all(v, [](unique_ptr<Shape>& ps) { ps->draw(); }); // draw_all()
    for_all(v, [](unique_ptr<Shape>& ps) { ps->rotate(45); }) // rotate_all(45)
}
```

I pass a `unique_ptr<Shape>&` to a lambda so that `for_all()` doesn’t have to care exactly how the objects are stored. In particular, those `for_all()` calls do not affect the lifetime of the `Shape` s passed and the bodies of the lambdas use the argument just as if they had been a plain-old pointers.
> `for_all` 遍历的是指针容器，因此并不需要关心对象实际是如何存储的
> `unique_ptr<Shape>` 的使用和 `Shape*` 的使用并无二至

Like a function, a lambda can be generic. For example:

```cpp
template<class S>
void rotate_and_draw(vector<S>& v, int r)
{
    for_all(v, [](auto& s){ s->rotate(r); s->draw();});
}
```

Here, like in variable declarations, auto means that any type is accepted as an initializer (an argument is considered to initialize the formal parameter in a call). This makes a lambda with an auto parameter a template, a generic lambda . For reasons lost in standards committee politics, this use of auto is not currently allowed for function arguments.
> lambda 表达式允许将形参类型写为 `auto` ，意为接受任意类型作为初始值 (函数调用时，实参会作为形参的初始值)
> 因此，接受 `auto` 形参的 lambda 表达式就是一个模板/泛型 lambda
> 目前不允许函数形参类型为 `auto`

We can call this generic `rotate_and_draw()` with any container of objects that you can `draw()` and `rotate()` . For example:
> 此时的 `rotate_and_draw()` 函数接受 `vector<S>& v` 指针容器，`S` 允许指向任意类型的指针，其中的 lambda 表达式定义的函数对象接受任意类型的实参
> 在 `for_all` 中，我们传递给它的是容器 `v` 中的元素，也就是指向任意类型对象的指针（只要定义了 `rotate(), draw()` 函数就是合法的）

```cpp
void user4()
{
    vector<unique_ptr<Shape>> v1;
    vector<Shape*> v2;
    // ...
    rotate_and_draw(v1, 45);
    rotate_and_draw(v2, 90);
}
```

Using a lambda, we can turn any statement into an expression. This is mostly used to provide an operation to compute a value as an argument value, but the ability is general. 
> lambda 表达式可以将任意的语句集合在一个表达式里，除了提供函数对象作为实参以外，它也有更广泛的用法

Consider a complicated initialization:

```cpp
enum class init_mode { zero, seq, cpy, patrn }; // initializer alternatives

// messy code

// int n, Init_mode m, vector<int>& arg, and iterators p and q are defined somewhere 

vector<int> v;

switch(m) {
case zero:
    v = vector<int>(n); // n elements initialized to 0
    break;
case cpy:
    v = arg;
    break;
};

// ...

if(m == seq)
    v.assign(p, q); // copy from sequence [p:q)

// ...
```

This is a stylized example, but unfortunately not atypical. We need to select among a set of alternatives for initializing a data structure (here v ) and we need to do different computations for different alternatives. Such code is often messy, deemed essential ‘‘for efficiency,’’ and a source of bugs:

- The variable could be used before it gets its intended value. 
- The ‘‘initialization code’’ could be mixed with other code, making it hard to comprehend.
- When ‘‘initialization code’’ is mixed with other code it is easier to forget a case. 
- This isn’t initialization, it’s assignment. 

Instead, we could convert it to a lambda used as an initializer:

```cpp
// int n, Init_mod m, vector<int>& arg, and iterators p and q are defined somewhere
vector<int> v = [&] {
    case zero:
        return vector<int>(n); // n elements initialized to 0
    case seq:
        return vector<int>{p, q}; // copy from sequence [p:q)
    case cpy:
        return arg;
}();
```

I still ‘‘forgot’’ a `case` , but now that’s easily spotted.
> 例如将多个条件初始化语句放在一个 lambda 表达式内，使得代码更加紧凑清晰

## 6.4 Template Mechanisms
To define good templates, we need some supporting language facilities:

- Values dependent on a type: variable templates (§6.4.1). 
- Aliases for types and templates: alias templates (§6.4.2). 
- A compile-time selection mechanism: `if constexpr` (§6.4.3). 
- A compile-time mechanism to inquire about properties of types and expressions: `requires` - expressions $(\S7.2.3)$ .

In addition, `constexpr` functions (§1.6) and `static_asserts` (§3.5.5) often take part in template design and use.

These basic mechanisms are primarily tools for building general, foundational abstractions.

### 6.4.1 Variable Templates
When we use a type, we often want constants and values of that type. This is of course also the case when we use a class template: when we define a `C<T>` , we often want constants and variables of type T and other types depending on T . Here is an example from a ﬂuid dynamic simulation [Garcia, 2015]:
> 我们使用一个类型时，常常想要该类型的常数和值
> 使用类模板时也是如此：我们定义 `C<T>` 时，常常想要类型 `T` 的常数和变量以及其他依赖于 `T` 的类型的常数和变量，例如：

> 变量模板用于帮助创建不同类型的编译时常量表达式，变量模板必须初始化为常量表达式，这个常量值可以借助模板特性被作为多个类型的值使用

```cpp
template<class T>
    constexpr T viscosity = 0.4;

template<class T>
    constexpr space_vector<T> external_acceleration = { T{}, T{-9.8}, T{}};

auto vis2 = 2 * viscosity<double>;
auto acc = external_accleration<float>;
```

Here, `space_vector` is a three-dimensional vector.

Naturally, we can use arbitrary expressions of suitable type as initializers. Consider:
> 可以使用任意合适类型的表达式作为变量模板的初始化值

```cpp
template<typename T, typename T2>
constexpr bool Assignable = is_assignable<T&, T2>::value; // is_assignable is a type trait (§13.9.1)

template<typename T>
void testing()
{
    static_assert(Assignable<T&, double>, "can't assign a double");
    static_assert(Assignable<T&, string>, "can't assign a string");
}
```

After some significant mutations, this idea becomes the heart of concept definitions $(\S7.2)$ .

### 6.4.2 Aliases
Surprisingly often, it is useful to introduce a synonym for a type or a template. For example, the standard header `<cstddef>` contains a definition of the alias `size_t` , maybe:
> 我们可以为一个类型或模板引入别名
> 例如标准头文件 `<cstddef>` 包含了别名 `size_t` ，定义如下：

```cpp
using size_t = unsigned int;
```

The actual type named `size_t` is implementation-dependent, so in another implementation `size_t` may be an `unsigned long` . Having the alias `size_t` allows the programmer to write portable code.
> `size_t` 实际的定义取决于实现，也有可能是 `unsigned long`

It is very common for a parameterized type to provide an alias for types related to their template arguments. For example: 
> 一个参数化的类型为它们的模板参数提供别名是非常常见的

```cpp
template<typename T>
class Vector {
public:
    using value_type = T;
}
```

In fact, every standard-library container provides `value_type` as the name of its value type (Chapter 11). This allows us to write code that will work for every container that follows this convention. For example:
> 所有的标准库容器都提供了别名 `value_type`

```cpp
template<typename C>
using Value_type = typename C::value_type; // the type of C's elements

template<typename Container>
void algo(Container& c)
{
    Vector<Value_type<Container>> vec; // keep results here
}
```

> 本例中
```cpp
template<typename C>
using Value_type = typename C::value_type;
```
> 定义了一个模板别名，模板别名也称为类型别名或者模板类型别名，模板别名的作用就是为复杂的模板类型表达式起一个别名
> `typename C::value_type` 中的 `typename` 关键字用于指示 `C::value_type` 是一个类型名，避免编译器将其解析为标识符
> 本例中的 `Value_type<Container>` 实际上等价于 `Container::value_type`

The aliasing mechanism can be used to define a new template by binding some or all template arguments. For example:
> 通过绑定部分或全部的模板参数，模板别名还可以用于定义新的模板

```cpp
template<typename Key, typename Value>
class Map {
    // ...
};

template<typename Value>
using String_map = Map<string, Value>;

String_map<int> m; // m is a Map<string, int>
```

### 6.4.3 Compile-Time if
Consider writing an operation that can use one of two operations `slow_and_safe(T)` or `simple_and_fast(T)` . Such problems abound in foundational code where generality and optimal perfor- mance are essential. The traditional solution is to write a pair of overloaded functions and select the most appropriate based on a trait (§13.9.1), such as the standard-library `is_pod` . If a class hierarchy is inv olved, a base class can provide the `slow_and_safe` general operation and a derived class can override with a `simple_and_fast` implementation. 
>考虑编写一个操作，该操作可以选择使用 `slow_and_safe(T)` 或 `simple_and_fast(T)` 中的一个
>这类问题在基础代码中比比皆是，因为在这些场景中通用性和最优性能都是至关重要的
>传统的解决方案是编写一对重载函数，并基于某个特征（如标准库中的 `is_pod`）选择最合适的一个
>如果涉及类层次结构，基类可以提供通用的 `slow_and_safe` 操作，而派生类可以用 `simple_and_fast` 实现来覆盖它

> 编译时条件判断通过 `constexpr if` 语句实现，该语句允许我们在编译时根据一个常量表达式的值来选择不同的代码路径

In C++17 , we can use a compile-time if :

```cpp
template<typename T>
void update(T& target) 
{
    // ...
    if constexpr(is_pod<T>::value)
        simple_and_fast(target); // for "plain old data"
    else
        slow_and_safe(target);
}
```

The `is_pod<T>` is a type trait (§13.9.1) that tells us whether a type can be trivially copied. 
> 其中的 `is_pod<T>` 是一个类型特性，用于确认一个类型是否可以被简单拷贝

Only the selected branch of an `if constexpr` is instantiated. This solution offers optimal performance and locality of the optimization. 
> 只有被选择的 `if constexpr`  路径会被实例化

Importantly, an `if constexpr` is not a text-manipulation mechanism and cannot be used to break the usual rules of grammar, type, and scope. 
> `if constexpr` 不能用于破坏通常的语法规则、类型、作用域

For example: 

```cpp
template<typename T>
void bad(T arg) {
    if constexpr (Something<T>::value)
        try { // syntax error
    g(arg);

    if constexpr(Something<T>::value)
        } catch(...) {/*...*/} // syntax error
}
```

Allowing such text manipulation could seriously compromise readability of code and create problems for tools relying on modern program representation techniques (such as ‘‘abstract syntax trees’’).

## 6.5 Advice
[1] Use templates to express algorithms that apply to many argument types; §6.1; [CG: T.2].
[2] Use templates to express containers; $\S6.2$ ; [CG: T.3].
[3] Use templates to raise the level of abstraction of code; $\S6.2$ ; [CG: T.1].
[4] Templates are type safe, but checking happens too late; $\S6.2$ .
[5] Let constructors or function templates deduce class template argument types; $\S6.2.3$ .
[6] Use function objects as arguments to algorithms; $\S6.3.2$ ; [CG: T.40].
[7] Use a lambda if you need a simple function object in one place only; $\S6.3.2$ .
[8] A virtual function member cannot be a template member function; $\S6.3.1$ .
[9] Use template aliases to simplify notation and hide implementation details; $\S6.4.2$ .
[10] To use a template, make sure its definition (not just its declaration) is in scope; $\S7.5$ .
[11] Templates offer compile-time ‘‘duck typing’’; $\S7.5$ .
[12] There is no separate compilation of templates: `#include` template definitions in every translation unit that uses them.
> 1. 使用模板来表示应用于多个实参类型的算法
> 2. 使用模板来表示容器
> 3. 使用模板来提高代码的抽象层次
> 4. 模板是类型安全的（有类型检查机制），但类型检查发生较晚（在编译时才进行类型检查；`concept` 支持在定义时就进行类型检查）
> 5. 让构造函数或函数模板推导类模板实参类型
> 6. 使用函数对象作为算法的实参
> 7. 如果仅需要在一个地方使用简单的函数对象，使用 lambda
> 8. 虚函数不能是模板成员函数
> 9. 使用模板别名以简化标记和隐藏实现细节
> 10. 使用模板时，保证其定义（不仅仅是声明）在作用域内
> 11. 模板提供了编译时的“duck tying”
> 12. 模板不存在分离编译，需要在每个编译单元 `#include` 模板定义以使用它 (普通的函数和类可以在头文件中放声明，在源文件中放定义，而模板则需要将声明和定义都放在头文件中，并在需要使用模板的编译单元中 `#include` 该头文件，这是为了保证使用模板的编译单元可以访问模板的全部定义，编译器需要根据实际对模板的使用方式来将模板的各个定义实例化为普通的函数)

# 7 Concepts and Generic Programming
## 7.1 Introduction
What are templates for? In other words, what programming techniques are effective when you use templates? Templates offer:

- The ability to pass types (as well as values and templates) as arguments without loss of information. This implies excellent opportunities for inlining, of which current implementa- tions take great advantage. 
- Opportunities to weave together information from different contexts at instantiation time. This implies optimization opportunities. 
- The ability to pass constant values as arguments. This implies the ability to do compile-time computation.

In other words, templates provide a powerful mechanism for compile-time computation and type manipulation that can lead to very compact and efficient code. Remember that types (classes) can contain both code (§6.3.2) and values (§6.2.2).

The first and most common use of templates is to support generic programming , that is, programming focused on the design, implementation, and use of general algorithms. Here, ‘‘general’’ means that an algorithm can be designed to accept a wide variety of types as long as they meet the algorithm’s requirements on its arguments. Together with concepts, the template is C++ main support for generic programming. Templates provide (compile-time) parametric polymorphism.
> 模板的最常见的用法就是支持泛型编程，也就是聚焦于针对通用算法的设计、实现、使用来编程
> “通用”意味着算法被设计为可以接受多种类型作为实参，只要它们满足要求
> 模板和概念是 C++ 为泛型编程的主要支持，其中模板提供的是编译时的参数化多态

## 7.2 Concepts (C++20)
Consider the sum () from $\S6.3.1$ :

```cpp
template<typename Seq, typename Num>
Num sum(Seq s, Num v)
{
    for(const auto& x : s)
        v += s;
    return v;
}
```

It can be invoked for any data structure that supports begin () and end () so that the range- for will work. Such structures include the standard-library vector , list , and map. Furthermore, the element type of the data structure is limited only by its use: it must be a type that we can add to the Value argument. Examples are int s, double s, and Matrix es (for any reasonable definition of Matrix ). 
> 该模板化的 `sum()` 函数可以接受任意支持 `begin()/end()` （即支持 range-for 遍历）的数据结构
> 这类结构包括标准库的 `vector/list/map`  等
> 另外，该函数有一个限制就是容器的元素类型应该是支持 `+` 运算符的，例如 `int/double/Matrix` 等

We could say that the `sum()` algorithm is generic in two dimensions: the type of the data structure used to store elements (‘‘the sequence’’) and the type of elements.
> `sum()` 函数在两个维度上是泛用的：用于存储元素的数据结构（即序列）的类型、元素的类型

So, `sum()` requires that its first template argument is some kind of sequence and its second tem- plate argument is some kind of number. We call such requirements concepts .
> 因此 `sum()` 要求第一个模板实参是某一类序列，第二个模板实参是某类数
> 我们称这类要求为概念

Language support for concepts is not yet ISO $\mathrm{C++}$ , but it is an ISO Technical Specification [ConceptsTS]. Implementations are in use, so I risk recommending it here even though details are likely to change and it may be years before everybody can use it in production code.
> 对于概念的支持尚且不是 ISO C++

### 7.2.1 Use of Concepts
Most template arguments must meet specific requirements for the template to compile properly and for the generated code to work properly. That is, most templates must be constrained templates (§6.2.1). The type-name introducer `typename` is the least constraining, requiring only that the argu- ment be a type. Usually, we can do better than that. Consider that sum() again:
> 大多数的模板应该是受限制模板，也就是模板实参需要满足特定要求，模板才可以正常编译和使用
> `typename` 的约束是最少的，仅要求模板实参是一个类型

```cpp
template<Sequence Seq, Number Num>
Num sum(Seq s, Num v)
{
    for(const auto& x : s)
        v += s;
    return v;
}
```

That’s much clearer. Once we have defined what the concepts `Sequence` and `Number` mean, the compiler can reject bad calls by looking at `sum()` ’s interface only, rather than looking at its implementation. This improves error reporting.
> 参考上例，如果我们定义好了 `Sequence/Number` 的概念，则编译器可以通过 `sum()` 的接口就可以判断对于它的某个特定调用是否合法，而不需要查看 `sum()` 的实现

However, the specification of `sum()` ’s interface is not complete: I ‘‘forgot’’ to say that we should be able to add elements of a `Sequence` to a `Number` . We can do that:
> 另外，我们需要指定：我们可以将 `Sequence` 中的元素加到 `Number` 上

```cpp
template<Sequeuce Seq, Number Num> 
    requires Arithmetic<Value_type<Seq>, Num>
Num sum(Seq s, Num n);
```

The `Value_type` of a sequence is the type of the elements in the sequence. `Arithmetic<X, Y>`  is a concept specifying that we can do arithmetic with numbers of types `X`  and `Y` . This saves us from accidentally trying to calculate the `sum()` of a `vector<string>` or a `vector<int*>` while still accepting `vector<int>` and `vector<complex<double>>` .
> `Value_type<Seq>` 表示 `Seq` 的元素类型
> `Arithmetic<X, Y>` 是一个概念，指定了我们需要可以对类型 `X/Y` 做算数运算
> 指定了该概念之后，`sum()` 应该仍然可以接受 `vector<int>/vector<complex<double>>` ，但不能接受 `vector<string>/vector<int*>`

In this example, we needed only += , but for simplicity and ﬂexibility, we should not constrain our template argument too tightly. In particular, we might someday want to express sum () in terms of $^+$ and $=$ rather than $+=$ , and then we’d be happy that we used a general concept (here, Arithmetic ) rather than a narrow requirement to ‘‘have += .’’
> 要求宽泛的概念（例如 `Arithmetic`）显然比要求 “have +=” 更具灵活性

Partial specifications, as in the first `sum()` using concepts, can be very useful. Unless the specification is complete, some errors will not be found until instantiation time. However, partial specifications can help a lot, express intent, and are essential for smooth incremental development where we don’t initially recognize all the requirements we need. With mature libraries of concepts, initial specifications will be close to perfect.
> 部分指定是指在使用概念时，先定义一部分约束条件，然后随着时间的推移不断完善这些约束
> 这种方法在开发过程中非常有用，尤其是在逐步开发和完善代码库的过程中
> 在成熟的库中，概念通常已经经过了广泛的测试和优化，初始的规范通常会接近完美，这意味着在使用这些库时，部分指定的概念可以快速地达到预期的效果，而不需要进行大量的调整和修正

Unsurprisingly, `requires Arithmetic<Value_type<Seq>, Num>` is called a requirements-clause. The `template<Sequence Seq>` notation is simply a shorthand for an explicit use of `requires Sequence<Seq>` . If I liked verbosity, I could equivalently have written
> `requires Arithmetic<Value_type<Seq>, Num>` 称为 requirements 子句
> `template<Seqeuce Seq>` 实际上就是对于显式使用 `requires Sequence<Seq>` 的简写，上例实际等价于：

```cpp
template<typename Seq, typename Num>
    requires Sequence<Seq> && Number<Num> && Arithmetic<Value_type<Seq>, Num>
Num sum(Seq s, Num n);
```

On the other hand, we could also use the equivalence between the two notations to write:
> 也等价于：

```cpp
template<Sequence Seq, Arithmetic<Value_type<Seq>> Num>
Num sum(Seq s, Num n);
```

Where we cannot yet use concepts, we have to make do with naming conventions and comments, such as:
> 如果编译器尚且不支持概念，则可以使用注释写明 requirements 子句

```cpp
template<typename Sequence, typename Number>
    // requires Arithmetic<Value_type<Sequence>, Number>
Number sum(Sequence s, Number n);
```

Whatever notation we chose, it is important to design a template with semantically meaningful constraints on its arguments $(\S7.2.4)$ .

### 7.2.2 Concept-based Overloading
Once we have properly specified templates with their interfaces, we can overload based on their properties, much as we do for functions. 
> 指定好模板的接口后，我们可以基于其属性进行重载，就和函数类似

Consider a slightly simplified standard-library function `advance()` that advances an iterator (§12.3):
> 例如重载模板函数 `advance()`：

```cpp
template<Forward_iterator Iter>
void advance(Iter p, int n)    // move p n elements forward
{
    while(n--)
        ++p;    // a forward iterator has ++, but not + or +=
}

template<Random_access_iterator Iter>
void advance(Iter p, int n)    // move p n elements forward
{
    p += n;   // a random-access iterator has +=
}
```

The compiler will select the template with the strongest requirements met by the arguments. 
> 编译器会根据实参类型，选择实参可以满足的要求最强的模板

In this case, a list only supplies forward iterators, but a vector offers random-access iterators, so we get:
> 例如 `list` 仅支持前向迭代器，而 `vector` 支持随机访问迭代器，则 `advance()` 对于 `vector` 的迭代器就会使用要求更强的快速版本的 `advance()`，而对于 `list` 的迭代器就会使用要求更弱的慢速版本的 `advance()`

```cpp
void user(vector<int>::iterator vip, list<string>::iterator lsp)
{
    advance(vip, 10);    // use the fast advance()
    advance(lsp, 10);    // use the slow advance()
}
```

Like other overloading, this is a compile-time mechanism implying no run-time cost, and where the compiler does not find a best choice, it gives an ambiguity error. 
> 利用概念进行模板重载和其他类型的重载一样，也是编译时的机制，不会有运行时开销
> 当编译器找不到最优的选择时，会报错

The rules for concept-based over-loading are far simpler than the rules for general overloading (§1.3). Consider first a single argument for several alternative functions:

- If the argument doesn’t match the concept, that alternative cannot be chosen. 
- If the argument matches the concept for just one alternative, that alternative is chosen. 
- If arguments from two alternatives are equally good matches for a concept, we have an ambiguity. 
- If arguments from two alternatives match a concept and one is stricter than the other (match all the requirements of the other and more), that alternative is chosen.

> 基于概念的重载的规则要比通用的重载的规则更加简单
> 考虑单个实参有多个候选函数的情况，选择函数的规则如下：
> - 实参不匹配概念的候选不选择
> - 如果仅有一个候选的概念匹配，就选择它
> - 如果有两个候选的匹配一样好，就出现歧义
> - 如果有两个候选的概念匹配，但其中一个候选的概念更加严格 (意思是该候选的 requirements 包括了另一个候选的 requirements，并且有自己额外的要求)，则选择更加严格的候选

For an alternative to be chosen it has to be

- a match for all of its arguments, and 
- at least an equally good match for all arguments as other alternatives, and
- a better match for at least one argument.

> 一个候选要被选择，它需要：
> - 匹配所有的实参
> - 至少是和其他候选一样对于所有实参是一样好的匹配
> - 对于至少一个实参是一个更好的匹配

### 7.2.3 Valid Code
The question of whether a set of template arguments offers what a template requires of its template parameters ultimately boils down to whether some expressions are valid.
> 检查模板实参是否满足模板形参的要求本质上是检查某些表达式是否有效的问题

Using a `requires` -expression, we can check if a set of expressions is valid. For example:
> 我们使用 `requires` 表达式来检查一组表达式是否有效：

```cpp
template<Forward_iterator Iter>
void advance(Iter p, int n) // move p n elements forward
{
    while(n--)
        ++p;    // a forward iterator has ++, but not + or +=
}

template<Forward_iterator Iter>
    requires requires(Iter p, int i) {p[i]; p+i;} // Iter has subscripting and addition
void advance(Iter p, int n) // move p n elements forward
{
    p += n;    // a random-access iterator has +=
}
```

No, that `requires requires` is not a typo. The first `requires` starts the `requirements` -clause and the second `requires` starts the `requires` − expression
> `requires requires` 中，第一个 `requires` 表示 `requirements` 子句的开始，也就是开始阐述概念，第二个 `requires` 表示 `requires` 表达式的开始

```cpp
requires(Iter p, int i) { p[i]; p+i; }
```

A `requires` −expression is a predicate that is true if the statements in it are valid code and false if they are not.
> `requires` 表达式是一个谓词，如果表达式中的语句是有效的代码，它返回 true，否则返回 false

> 上例中第二个 `template` 的定义实际上等价于：

```cpp
template<typename Iter>
    requires Forward_iterator<Iter> && requires(Iter p, int i) { p[i]; p+i; }
void advance(Iter p, int n)
{
    p += n;
}
```

I consider `requires` -expressions the assembly code of generic programming. Like ordinary assembly code, `requires` -expressions are extremely ﬂexible and impose no programming discipline. In some form or other, they are at the bottom of most interesting generic code, just as assembly code is at the bottom of most interesting ordinary code. Like assembly code, `requires` -expressions should not be seen in ‘‘ordinary code.’’ If you see `requires requires` in your code, it is probably too low level.
>"我"认为 `requires` 表达式是泛型编程的汇编代码，和普通的汇编代码一样，`requires` 表达式极其灵活，并且不需要特定的编程规范
>它们是大多数有趣的泛型代码的基础，就像汇编代码是大多数有趣的一般代码的基础一样
>像汇编代码一样，`requires` 表达式不应该出现在“普通代码”中，如果你在代码中看到 `requires requires`，那可能是太底层了

The use of `requires requires` in advance () is deliberately inelegant and hackish. Note that I ‘‘forgot’’ to specify $\mathrel{+{=}}$ and the required return types for the operations. You have been warned! Prefer named concepts for which the name indicates its semantic meaning.
>在 advance() 中使用 `requires requires` 是故意设计得不够优雅且有些 hacky（黑客风格），请注意，我“忘记”指定了 `+=` 和操作所需的返回类型（这里的意思是使用 `requires` 表达式时常常容易导致我们忘记指定所需的要求，因此，要）更倾向于使用名称明确的概念，其名称可以表明其语义意义

Prefer use of properly named concepts with well-specified semantics $(\S7.2.4)$ and use `requires` - expressions in the definition of those.
>优先使用具有明确语义定义的适当命名概念（参见§7.2.4）
>这些概念的定义则使用 `requires` 表达式实现

### 7.2.4 Definition of Concepts
Eventually, we expect to find useful concepts, such as `Sequence` and `Arithmetic` in libraries, including the standard library. The Ranges Technical Specification [RangesTS] already offers a set for constraining standard-library algorithms (§12.7). However, simple concepts are not hard to define.
> 标准库提供了 `Sequence` , `Arithmetic` 等有用的概念，Ranges Techinical Sepcification 提供了一组用于约束标准库算法的工具

A concept is a compile-time predicate specifying how one or more types can be used. Consider first one of the simplest examples:
> 概念是一种在编译时用于指定一个或多个类型如何被使用的谓词
> 首先考虑其中一个最简单的例子：

```cpp
template<typename T>
concept Equality_comparable = 
    requires(T a, T b) {
        {a == b} -> bool; // compare Ts with ==
        {a != b} -> bool; // compare Ts with !=
    };
```

`Equality comparable` is the concept we use to ensure that we can compare values of a type equal and non-equal. 
> 本例中，概念 `Equality comparable` 的语义就是保证类型 `T` 是可以用 `==` 和 `!=` 比较的，并且比较的结果可以被转化为 `bool` 类型
> 本例中，概念 `Equality comparable` 使用一个 `requires` 表达式来初始化

We simply say that, given two values of the type, they must be comparable using == and != and the result of those operations must be convertible to bool. For example:

```cpp
static_assert(Equality_comparable<int>); // succeeds

struct S {int a;};

static_assert(Equality_comparable<S>); // fails because structs don't automatically get == and !=
```

The definition of the concept `Equality_comparable` is exactly equivalent to the English description and no longer. The value of a `concept` is always `bool` .
> 概念的值总是 `bool` 值 (`requires` 表达式的值就是 `bool` 值)

Defining `Equality_comparable` to handle nonhomogeneous comparisons is almost as easy:

```cpp
template<typename T, typename T2 = T>
concept Equality_comparable = 
    requires(T a, T2 b) {
        { a == b } -> bool; // compare a T to a T2 with ==
        { a != b } -> bool; // compare a T to a T2 with !=
        { b == a } -> bool; // compare a T2 to a T with ==
        { b != a } -> bool; // compare a T2 to a T with !=
    };
```

The typename ${\mathsf{T}}{\mathsf{z}}={\mathsf{T}}$ says that if we don’t specify a second template argument, T2 will be the same as T ; T is a *default template argument* .
> 其中的 `typename T2 = T` 的意思是如果不指定第二个模板实参，模板实参 `T2` 就设定为 `T` ，`T` 作为它的默认模板实参

We can test `Equality_comparable` like this:

```cpp
static_assert(Equality_comparable<int, double>); // succeeds
static_assert(Equality_comparable<int>); // succeeds (T2 is defaulted to int)
static_assert(Equality_comparable<int, string>); // fails
```

For a more complex example, consider a sequence:

```cpp
template<typename S>
concept Sequence = requries(S a) {
    typename Value_type<S>; // S must have a value type
    typename Iterator_type<S>; // S must have a iterator type

    {begin(a)} -> Iterator_type<S>; // begin(a) must return an iterator
    {end(a)} -> Iterator_type<S>; // end(a) must return an iterator

    requires Same_type<Value_type<S>, Value_type<Iterator_type<S>>>;
    requires Input_iterator<Iterator_type<S>>;
}
```

For a type `S` to be a `Sequence` , it must provide a `Value_type` (the type of its elements) and an Iterator_type (the type of its iterators; see $\S12.1$ ). It must also ensure that there exist `begin()` and `end()` functions that return iterators, as is idiomatic for standard-library containers $(\S11.3)$ . Finally, the Iterator_type really must be an input_iterator with elements of the same type as the elements of S .
> 类型 `S` 要满足 `Sequence` ，它需要提供 `Value_type` 和 `Iterator_type` ，同时需要保证存在返回 `Iterator_type` 的 `begin()/end()` 函数，最后，`Iterator_type<S>` 应该是 `Input_iterator` ，且其 `Value_type` 和 `S` 的 `Value_type` 相同

The hardest concepts to define are the ones that represent fundamental language concepts. Consequently, it is best to use a set from an established library. For a useful collection, see $\S12.7$ .
>最难定义的概念是那些代表基本语言概念的概念，因此，最好使用来自已经建立好的库的一组概念，详见 $\S12.7$

## 7.3 Generic Programming
The form of generic programming supported by $\mathrm{C++}$ centers around the idea of abstracting from concrete, efficient algorithms to obtain generic algorithms that can be combined with different data representations to produce a wide variety of useful software [Stepanov, 2009]. 
>C++ 支持的泛型编程的形式主要围绕从具体的、高效的算法抽象出通用算法的思想，这些通用算法可以与不同的数据表示相结合，从而产生各种有用的软件 [Stepanov, 2009]

The abstractions representing the fundamental operations and data structures are called concepts ; they appear as requirements for template parameters.
>表示基本操作和数据结构的抽象称为概念；它们作为模板参数的要求出现

### 7.3.1 Use of Concepts
Good, useful concepts are fundamental and are discovered more than they are designed. Examples are integer and ﬂoating-point number (as defined even in Classic C), sequence, and more general mathematical concepts, such as field and vector space. They represent the fundamental concepts of a field of application. That is why they are called ‘‘concepts.’’ Identifying and formalizing concepts to the degree necessary for effective generic programming can be a challenge.
>好的，有用的概念是基础性的，并且更多的是被发现而不是被设计出来的。例如，整数和浮点数（即使在经典的C语言中也有定义）、序列，以及更广泛的数学概念，如域和向量空间。它们代表了应用领域中的基本概念，这就是为什么它们被称为“概念”
>识别并形式化这写概念到足以进行有效泛型编程所需的程度的概念可能会是一项挑战。

For basic use, consider the concept `Regular` $(\S12.7)$ . A type is regular when it behaves much like an `int` or a `vector` . 
> 对于基础的使用，考虑概念 `Regular` ，当一个类型的行为类似于 `int` 或 `vector` 时，该类型就是规则的

An object of a regular type

- can be default constructed. 
- can be copied (with the usual semantics of copy, yielding two objects that are independent and compare equal) using a constructor or an assignment. 
- can be compared using `==` and `!=`. 
- doesn’t suffer technical problems from overly clever programming tricks.

> 规则类型的对象满足：
>- 可以进行默认构造
>- 可以通过构造函数或赋值操作进行拷贝（遵循常规的拷贝语义，生成两个独立且相等的对象）
>- 可以使用 `==` 和 `!=` 进行比较
>- 不会因为过于聪明的编程技巧而产生技术问题

A `string` is another example of a regular type. Like `int` , `string` is also `StrictTotallyOrdered` (§12.7). That is, two strings can be compared using < , $<=,\,>,$ , and $>=$ with the appropriate semantics.
> 例如，`string` 就是规则类型，并且和 `int` 一样，`string` 是 `StrictTotallyOrdered` ，即两个 `string` 对象可以根据恰当的语义，使用 `</>/<=/>=` 进行比较

A concept is not just a syntactic notion, it is fundamentally about semantics. For example, don’t define `+` to divide; that would not match the requirements for any reasonable number. Unfortunately, we do not yet have any language support for expressing semantics, so we have to rely on expert knowledge and common sense to get semantically meaningful concepts. Do not define semantically meaningless concepts, such as `Addable` and `Subtractable` . Instead, rely on domain knowledge to define concepts that match fundamental concepts in an application domain.
>concept 不仅仅是语法上的概念，它本质上是关于语义的
>例如，不要将加法运算符 `+`  定义为除法，这不符合任何合理数字的要求
>不幸的是，我们目前还没有语言支持来表达语义，因此我们必须依靠专家知识和常识来获得具有语义意义的概念
>不要定义语义上无意义的概念，如 ` Addable ` 和 ` Subtractable `，相反，应依靠领域知识来定义与应用程序领域中的基本概念相匹配的概念

### 7.3.2 Abstraction Using Templates
Good abstractions are carefully grown from concrete examples. It is not a good idea to try to ‘‘abstract’’ by trying to prepare for every conceivable need and technique; in that direction lies inel- eg ance and code bloat. Instead, start with one – and preferably more – concrete examples from real use and try to eliminate inessential details. Consider:
>好的抽象是通过对具体实例精心提炼出来的，试图通过准备应对所有可预见的需求和技术来“抽象化”并不是一个好主意；那样做会导致僵化和代码膨胀
>相反，应该从一个——最好是多个——实际使用的具体例子开始，并尝试消除非本质的细节，考虑：

```cpp
double sum(const vector<int>& v)
{
    double res = 0;
    for(auto x : v)
        res += x;
    return res;
}
```

This is obviously one of many ways to compute the sum of a sequence of numbers.

Consider what makes the code less general than it needs to be:

- Why just `int` s? 
- Why just `vector` s? 
- Why accumulate in a `double` ? 
- Why start at `0` ? 
- Why add?

Answering the first four questions by making the concrete types into template arguments, we get the simplest form of the standard-library `accumulate` algorithm: 
> 通过让具体类型成为模板参数，我们可以回答前4个问题
> 由此，我们得到了标准库 `accumulate` 算法的最简单形式

```cpp
template<typename Iter, typename Val>
Val accumulate(Iter first, Iter last, Val res) {
    for(auto p = first; p != last; ++p)
        res += *p;
    return res;
}
```

Here, we have:

- The data structure to be traversed has been abstracted into a pair of iterators representing a sequence $(\S12.1)$ . 
- The type of the accumulator has been made into a parameter. 
- The initial value is now an input; the type of the accumulator is the type of this initial value.

> 本例中：
> - 需要遍历的数据结构被抽象为了一对表示一个序列的迭代器
> - 累加值的类型被变为模板形参
> - 初始的累加值需要被输入，累加值的类型就是输入的初始累加值的类型

A quick examination or – even better – measurement will show that the code generated for calls with a variety of data structures is identical to what you get from the hand-coded original example. For example:

```cpp
void use(const vector<int>& vec, const list<double>& lst)
{
    auto sum = accumulate(begin(vec), end(vec), 0.0); // accumulate in a double
    auto sum2 = accumulate(begin(lst), end(lst), sum);
}
```

The process of generalizing from a concrete piece of code (and preferably from several) while preserving performance is called lifting . Conversely, the best way to develop a template is often to

- first, write a concrete version 
- then, debug, test, and measure it 
- finally, replace the concrete types with template arguments.

>从具体的代码片段（最好是多个片段）中进行泛化，并在此过程中保持性能的过程称为提升，而开发模板的最佳方式通常就是：
>- 首先，编写一个具体的版本
>- 然后，对其进行调试、测试和测量
>- 最终，用模板参数替换具体的类型

Naturally, the repetition of begin() and end() is tedious, so we can simplify the user interface a bit: 

```cpp
template<Range R, Number Val>    // a Range is something with begin() and end()
Val accumulate(const R& r, Val res = 0)
{
    for(auto p = begin(r); p != end(r); p++)
        res += *p;
    return res;
}
```

For full generality, we can abstract the `+=` operation also; see $\S14.3$ .

## 7.4 Variadic Templates
A template can be defined to accept an arbitrary number of arguments of arbitrary types. Such a template is called a variadic template . 
> 模板可以被定义为接受任意数量的任意类型的实参，这类模板成为可变参数模板

Consider a simple function to write out values of any type that has a `<<` operator:
> 考虑一个简单的函数 `print` ，它接受任意数量的值，仅要求值的类型具有 `<<` 运算符

```cpp
void user()
{
    print("first: ", 1, 2.2, "hello\n"s); // first: 1 2.2 hello
    print("\nsecond: ", 0.2, 'c', 'yuck's, 0, 1, 2, '\n'); // second: 0.2 c yuck 0 1 2
}
```

Traditionally, implementing a variadic template has been to separate the first argument from the rest and then recursively call the variadic template for the tail of the arguments:
>传统上，实现变参模板的方法是将第一个参数与其余参数分开，然后递归调用变参模板来处理参数的尾部：

```cpp
void print()
{
    // what we do for no arguments: nothing
}

template<typename T, typename... Tail>
void print(T head, Tail... tail)
{
    // what we do for each arguments, e.g.
    cout<<head<<' ';
    print(tail...);
}
```

The `typename...` indicates that `Tail` is a sequence of types. The `Tail...` indicates that `tail` is a sequence of values of the types in Tail . A parameter declared with a ... is called a parameter pack . Here, `tail` is a (function argument) parameter pack where the elements are of the types found in the (template argument) parameter pack `Tail` . So, `print()` can take any number of arguments of any types.
>  本例中：
>  模板形参中的 `typename...` 表示 `Tail` 是一系列类型
>  函数形参中的 `Tail...` 表示 `tail` 是 `Tail` 中类型的一系列值
>  使用 `...` 声明的参数称为参数包（parameter pack）
>  本例中，`tail` 是一个（函数参数）参数包，其中的元素的类型来自（模板参数）参数包 `Tail` ，因此，`print()` 可以接受任意数量和任意类型的参数

A call of `print()` separates the arguments into a head (the first) and a tail (the rest). The head is printed and then `print()` is called for the tail. Eventually, of course, tail will become empty, so we need the no-argument version of `print()` to deal with that.
> 调用 `print()` 后，第一个实参作为 head，其余的都作为 tail，head 会被打印出来，然后 tail 再作为实参传入给递归调用的 `print()` ，最后直到 tail 变为空，嗲用空的 `print()` ，结束递归

If we don’t want to allow the zero-argument case, we can eliminate that `print()` using a compile-time if :

```cpp
template<typename T, typename... Tail>
void print(T head, Tail... tail)
{
    cout<<head<<' ';
    if constexpr(sizeof...(tail) > 0)
        print(tail...);
}
```

I used a compile-time `if` (§6.4.3), rather than a plain run-time `if` to avoid a final, never called, call `print()` from being generated.
> 当然我们可以用 `if` 来判断 `tail` 中剩余的参数数量是否还大于零，如果大于零，就继续递归，否则就不调用
> 而普通的运行时 `if` 可以优化为编译时 `if` ，因为 `tail` 的长度是可以在编译时确定的，这样可以避免生成一句额外的不会被调用的 `print()` 调用

The strength of variadic templates (sometimes just called variadics ) is that they can accept any arguments you care to give them. 
>变参模板（有时简称为变参）的优势在于它们可以接受任意你想要提供的参数

Weaknesses include

- The recursive implementations can be tricky to get right. 
- The recursive implementations can be surprisingly expensive in compile time. 
- The type checking of the interface is a possibly elaborate template program.

Because of their ﬂexibility, variadic templates are widely used in the standard library, and occasionally wildly overused.
>其劣势包括
>- 递归实现可能会难以正确实现。
>- 递归实现可能会在编译时消耗大量时间。
>- 接口的类型检查可能是相当复杂的模板程序。
>由于其灵活性，变参模板在标准库中被广泛使用，但有时也被过度使用

### 7.4.1 Fold Expressions
To simplify the implementation of simple variadic templates, C++17 offers a limited form of iteration over elements of a parameter pack. For example:
> C++17 提供了对参数包内的元素进行迭代的一种有限的形式，来简化简单的可变模板函数的实现，例如：

```cpp
template<Number... T>
int sum(T... v)
{
    return (v + ... + 0); // add all elements of v starting with 0
}
```

Here, `sum()` can take any number of arguments of any types. Assuming that `sum()` really adds its arguments, we get:
> 本例中，`sum()` 是一个可变模板函数，它接受任意数量的 `Number` 实参

```cpp
int x = sum(1, 2, 3, 4, 5); // x becomes 15
int y = sum('a', 2.4, x); // y becomes 114 (2.4 is truncated and the value of 'a' is 97)
```

The body of sum uses a fold expression:
> `sum()` 函数体使用了折叠表达式：

```cpp
return (v + ... + 0); // add all elements of v to 0
```

Here, `(v+...+0)` means add all the elements of `v` starting with the initial value 0 . The first element to be added is the ‘‘rightmost’’ (the one with the highest index): `(v[0]+(v[1]+(v[2]+(v[3]+(v[4]+0)))))` . That is, starting from the right where the 0 is. It is called a right fold . 
> 折叠表达式中的 `(v+...+0)` 表示将 `v` 中的所有元素依次相加到初始值 `0` 中
> 第一个被加上的元素是“最右边”的元素，即索引最大的元素，因此这也被称为右折叠

Alternatively, we could have used a left fold :

```cpp
template<Number... T>
int sum2(T... v)
{
    return (0+...+v); // add all elements of v to 0
}
```

Now, the first element to be added is the ‘‘leftmost’’ (the one with the lowest index): `(((((0+v[0])+v[1])+v[2])+v[3])+v[4])` . That is, starting from the left where the 0 is.
> `(0+...+v)` 则表示左折叠，即第一个被加上的元素是“最左边的元素”，即索引最小的元素

Fold is a very powerful abstraction, clearly related to the standard-library `accumulate()` , with a variety of names in different languages and communities. In C++ , the fold expressions are currently restricted to simplify the implementation of variadic templates. A fold does not have to perform numeric computations. Consider a famous example:
>折叠是一种非常强大的抽象概念，与标准库中的 `accumulate()` 明显相关，在不同的语言和社区中有各种各样的名称
>在 C++ 中，折叠表达式目前被限制仅用于简化可变参数模板的实现，折叠并不限于进行数值计算，考虑一个著名的例子：

```cpp
template<typename... T>
void print(T&& args)
{
    (std::cout<<...<<args) << '\n'; // print all arguments
}

print("Hello!"s,'', "World ", 2017); // (((((std::cout << "Hello!"s) << ’ ’) << "Wor ld ") << 2017) << ’\n’);)
```

Many use cases simply involve a set of values that can be converted to a common type. In such cases, simply copying the arguments into a vector or the desired type often simplifies further use:
>许多用例仅涉及一组可以转换为公共类型的值，在这种情况下，将参数复制到一个向量或所需类型中通常会简化后续使用：

```cpp
template<typename Res, typename... Ts>
vector<Res> to_vector(Ts&&... ts)
{
    vector<Res> res;
    (res.push_back(ts)...); // no initial value needed
    return res;
}
```

We can use `to_vector` like this:

```cpp
auto x = to_vector<double>(1, 2, 4.5, 'a');
template<typename... Ts>
int fct(Ts&&... ts)
{
    auto args = to_vector<string>(ts...); // args[i] is the ith argument
    // ... use args here ...
}
int y = fct("foo", "bar", s);
```

### 7.4.2 Forwarding Arguments
Passing arguments unchanged through an interface is an important use of variadic templates. Con- sider a notion of a network input channel for which the actual method of moving values is a param- eter. Different transport mechanisms have different sets of constructor parameters:

template<typename Transpor t> requires concepts:: InputTranspor t<Transpor t> class InputChannel { public: // ... InputChannel (Transpor tArgs&&... transportArgs) : \_transpor t (std::forward<Transpor tArgs>(transpor tArgs)...) {} // ... Transpor t \_transpor t; };

The standard-library function forward () (§13.2.2) is used to move the arguments unchanged from the InputChannel constructor to the Transpor constructor.

The point here is that the writer of InputChannel can construct an object of type Transpor with- out having to know what arguments are required to construct a particular Transpor . The imple- menter of InputChannel needs only to know the common user interface for all Transpor objects.

Forwarding is very common in foundational libraries where generality and low run-time over- head are necessary and very general interfaces are common.

# 7.5 Template Compilation Model

Assuming concepts $(\S7.2)$ , the arguments for a template are checked against its concepts. Errors found here will be reported and the programmer has to fix the problems. What cannot be checked at this point, such as arguments for unconstrained template arguments, is postponed until code is generated for the template and a set of template arguments: ‘‘at template instantiation time.’’ For pre-concept code, this is where all type checking happens. When using concepts, we get here only after concept checking succeeded.

An unfortunate side effect of instantiation-time (late) type checking is that a type error can be found uncomfortably late and can result in spectacularly bad error messages because the compiler found the problem only after combining information from several places in the program.

The instantiation-time type checking provided for templates checks the use of arguments in the template definition. This provides a compile-time variant of what is often called duck typing (‘‘If it walks like a duck and it quacks like a duck, it’s a duck’’). Or – using more technical terminology – we operate on values, and the presence and meaning of an operation depend solely on its operand values. This differs from the alternative view that objects have types, which determine the presence and meaning of operations. Values ‘‘live’’ in objects. This is the way objects (e.g., variables) work in $\mathrm{C++}$ , and only values that meet an object’s requirements can be put into it. What is done at com- pile time using templates mostly does not involve objects, only values. The exception is local vari- ables in a constexpr function (§1.6) that are used as objects inside the compiler.

To use an unconstrained template, its definition (not just its declaration) must be in scope at its point of use. For example, the standard header <vector> holds the definition of vector . In practice, this means that template definitions are typically found in header files, rather than .cpp files. This changes when we start to use modules (§3.3). Using modules, the source code is organized in the same way for ordinary functions and template functions. In both cases, definitions will be pro- tected against the problems of textual inclusion.

# 7.6 Advice

[1] Templates provide a general mechanism for compile-time programming; $\S7.1$ .

[2] When designing a template, carefully consider the concepts (requirements) assumed for its template arguments; $\S7.3.2$ .

[3] When designing a template, use a concrete version for initial implementation, debugging, and measurement; $\S7.3.2$ .

[4] Use concepts as a design tool; $\S7.2.1$ .

[5] Specify concepts for all template arguments; $\S7.2$ ; [CG: T.10].

[6] Whenever possible use standard concepts (e.g., the Ranges concepts); $\S7.2.4$ ; [CG: T.11].

[7] Use a lambda if you need a simple function object in one place only; $\S6.3.2$ .

[8] There is no separate compilation of templates: #include template definitions in every transla- tion unit that uses them.

[9] Use templates to express containers and ranges; $\S7.3.2$ ; [CG: T.3].

[10] Avoid ‘‘concepts’’ without meaningful semantics; $\S7.2$ ; [CG: T.20].

[11] Require a complete set of operations for a concept; $\S7.2$ ; [CG: T.21].

[12] Use variadic templates when you need a function that takes a variable number of arguments of a variety of types; $\S7.4$ .

[13] Don’t use variadic templates for homogeneous argument lists (prefer initializer lists for that); $\S7.4$ .

[14] To use a template, make sure its definition (not just its declaration) is in scope; $\S7.5$ .

[15] Templates offer compile-time ‘‘duck typing’’; $\S7.5$ .

# Library Overview

Why waste time learning when ignorance is instantaneous? – Hobbes

• Introduction • Standard-Library Components • Standard-Library Headers and Namespace • Advice

# 8.1 Introduction

No significant program is written in just a bare programming language. First, a set of libraries is developed. These then form the basis for further work. Most programs are tedious to write in the bare language, whereas just about any task can be rendered simple by the use of good libraries.

Continuing from Chapters 1–7, Chapters 9–15 give a quick tour of key standard-library facili- ties. I very brieﬂy present useful standard-library types, such as string , ostream , variant , vector , map , path , unique_ptr , thread , reg ex , and complex , as well as the most common ways of using them.

As in Chapters 1–7, you are strongly encouraged not to be distracted or discouraged by an incomplete understanding of details. The purpose of this chapter is to convey a basic understanding of the most useful library facilities.

The specification of the standard library is over two thirds of the ISO $\mathrm{C++}$ standard. Explore it, and prefer it to home-made alternatives. Much thought has gone into its design, more still into its implementations, and much effort will go into its maintenance and extension.

The standard-library facilities described in this book are part of every complete $\mathrm{C++}$ implemen- tation. In addition to the standard-library components, most implementations offer ‘‘graphical user interface’’ systems (GUIs), Web interfaces, database interfaces, etc. Similarly, most application- development environments provide ‘‘foundation libraries’’ for corporate or industrial ‘‘standard’’ development and/or execution environments. Here, I do not describe such systems and libraries.

The intent is to provide a self-contained description of $\mathrm{C++}$ as defined by the standard and to keep the examples portable. Naturally, a programmer is encouraged to explore the more extensive facili- ties available on most systems.

# 8.2 Standard-Library Components

The facilities provided by the standard library can be classified like this:

• Run-time language support (e.g., for allocation and run-time type information). • The C standard library (with very minor modifications to minimize violations of the type system). • Strings (with support for international character sets, localization, and read-only views of substrings); see $\S9.2$ . • Support for regular expression matching; see $\S9.4$ . • I/O streams is an extensible framework for input and output to which users can add their own types, streams, buffering strategies, locales, and character sets (Chapter 10). There is also a library for manipulating file systems in a portable manner (§10.10). • A framework of containers (such as vector and map ) and algorithms (such as find () , sor t () , and merge () ); see Chapter 11 and Chapter 12. This framework, conventionally called the STL [Stepanov, 1994], is extensible so users can add their own containers and algorithms. • Support for numerical computation (such as standard mathematical functions, complex numbers, vectors with arithmetic operations, and random number generators); see $\S4.2.1$ and Chapter 14. • Support for concurrent programming, including thread s and locks; see Chapter 15. The con- currency support is foundational so that users can add support for new models of concur- rency as libraries. • Parallel versions of most STL algorithms and some numerical algorithms (e.g., sor t () and reduce () ); see $\S12.9$ and $\S14.3.1$ . • Utilities to support template metaprogramming (e.g., type traits; $\S13.9$ ), STL-style generic programming (e.g., pair ; $\S13.4.3)$ , general programming (e.g., variant and optional; $\S13.5.1$ , $\S13.5.2)$ , and clock (§13.7). • Support for efficient and safe management of general resources, plus an interface to optional garbage collectors (§5.3). • ‘‘Smart pointers’’ for resource management (e.g., unique_ptr and shared_ptr ; $\S13.2.1$ ). • Special-purpose containers, such as array (§13.4.1), bitset $(\S13.4.2)$ , and tuple (§13.4.3). • Suffixes for popular units, such as ms for milliseconds and i for imaginary (§5.4.4). The main criteria for including a class in the library were that: • it could be helpful to almost every $\mathrm{C++}$ programmer (both novices and experts), • it could be provided in a general form that did not add significant overhead compared to a simpler version of the same facility, and • simple uses should be easy to learn (relative to the inherent complexity of their task).

Essentially, the $\mathrm{C++}$ standard library provides the most common fundamental data structures together with the fundamental algorithms used on them.

# 8.3 Standard-Library Headers and Namespace

Every standard-library facility is provided through some standard header. For example:

#include<string> #include<list>

This makes the standard string and list available. The standard library is defined in a namespace (§3.4) called std . To use standard-library facili- ties, the std:: prefix can be used:

std:: string sheep {"Four legs Good; two legs Baaad!"}; std::list<std::string> slogans {"War is Peace", "Freedom is Slaver y", "Ignorance is Strength"};

For simplicity, I will rarely use the std:: prefix explicitly in examples. Neither will I always #include the necessary headers explicitly. To compile and run the program fragments here, you must #include the appropriate headers and make the names they declare accessible. For example:

#include<string> // make the standard string facilities accessible using namespace std; // make std names available without std:: prefix

string s $\{"{\mathbb{C}}_{++}$ is a general−purpose programming language"}; // OK: string is std::string

It is generally in poor taste to dump every name from a namespace into the global namespace. However, in this book, I use the standard library exclusively and it is good to know what it offers. Here is a selection of standard-library headers, all supplying declarations in namespace std :

![](images/c7bc87d0fe6380e5051533da2d982f80f1b271924d8a75dc3e0f666cd286cc5b.jpg)

![](images/e0ed7e0b99eea8b7c8c0deb1b8103092801246f525f29c9876988e221344c373.jpg) 
This listing is far from complete. Headers from the C standard library, such as <stdlib. h> are provided. For each such header there is also a version with its name prefixed by c and the .h removed. This version, such as <cstdlib> places its declarations in the std namespace.

# 8.4 Advice

[1] Don’t reinvent the wheel; use libraries; $\S8.1$ ; [CG: SL. 1.]

[2] When you have a choice, prefer the standard library over other libraries; $\S8.1$ ; [CG: SL. 2].

[3] Do not think that the standard library is ideal for everything; $\S8.1$ .

[4] Remember to #include the headers for the facilities you use; $\S8.3$ .

[5] Remember that standard-library facilities are defined in namespace std ; $\S8.3$ ; [CG: SL. 3].

# Strings and Regular Expressions

Prefer the standard to the offbeat. – Strunk & White

• Introduction • Strings string Implementation; • String Views • Regular Expressions Searching; Regular Expression Notation; Iterators • Advice

# 9.1 Introduction

Te xt manipulation is a major part of most programs. The $\mathrm{C++}$ standard library offers a string type to save most users from C-style manipulation of arrays of characters through pointers. A string_view type allows us to manipulate sequences of characters however they may be stored (e.g., in a std:: string or a char[] ). In addition, regular expression matching is offered to help find patterns in text. The regular expressions are provided in a form similar to what is common in most modern languages. Both string s and reg ex objects can use a variety of character types (e.g., Unicode).

# 9.2 Strings

The standard library provides a string type to complement the string literals (§1.2.1); string is a Reg- ular type $(\S7.2,\S12.7)$ for owning and manipulating a sequence of characters of various character types. The string type provides a variety of useful string operations, such as concatenation. For example:

string compose (const string& name, const string& domain) { return name $+\ ^{\intercal}@^{\intercal}+$ domain; }

auto addr $=$ compose ("dmr","bell−labs. com");

Here, addr is initialized to the character sequence dmr@bell−labs. com . ‘‘Addition’’ of string s means concatenation. You can concatenate a string , a string literal, a C-style string, or a character to a string . The standard string has a move constructor, so returning even long string s by value is effi- cient (§5.2.2).

In many applications, the most common form of concatenation is adding something to the end of a string . This is directly supported by the $\mathrel{+{=}}$ operation. For example:

void m2 (string& s1, string& s2) { ${\mathfrak{s}}1={\mathfrak{s}}1+\mathbb{N}\mathfrak{n}^{\prime}$ ; // append newline ${\mathfrak{s}}{\mathfrak{z}}+={\mathfrak{w}}{\mathfrak{n}}^{\mathfrak{r}}$ ; // append newline }

The two ways of adding to the end of a string are semantically equivalent, but I prefer the latter because it is more explicit about what it does, more concise, and possibly more efficient.

A string is mutable. In addition to $=$ and $+=$ , subscripting (using [ ] ) and substring operations are supported. For example:

string name $=$ "Niels Stroustrup";

{ string $\mathbf{s}=$ name.substr (6,10); // $s=$ "Stroustr up" name .replace (0,5,"nicholas"); // name becomes "nicholas Stroustrup" name[0] $=$ toupper (name[0]); // name becomes "Nicholas Stroustrup" }

The substr () operation returns a string that is a copy of the substring indicated by its arguments. The first argument is an index into the string (a position), and the second is the length of the desired substring. Since indexing starts from 0 , s gets the value Stroustrup .

The replace () operation replaces a substring with a value. In this case, the substring starting at 0 with length 5 is Niels ; it is replaced by nicholas . Finally, I replace the initial character with its uppercase equivalent. Thus, the final value of name is Nicholas Stroustrup . Note that the replace- ment string need not be the same size as the substring that it is replacing.

Among the many useful string operations are assignment (using $\fallingdotseq$ ), subscripting (using [ ] or at () as for vector ; $\S11.2.2)$ , comparison $(\mathrm{using}=\mathrm{and}\mathrel{\mathop{}\!:}=)$ , and lexicographical ordering (using $<,\,<=,\,>$ , and $>=$ ), iteration (using iterators as for vector ; $\S12.2)$ ), input (§10.3), and streaming (§10.8).

Naturally, string s can be compared against each other, against $\mathrm{C}.$ -style strings $\S1.7.1)$ , and against string literals. For example:

string incantation; void respond (const string& answer) { if (answer $==$ incantation) { // perfor m magic } else if (answer $==$ "yes") { // ... } // ... }

If you need a C-style string (a zero-terminated array of char ), string offers read-only access to its contained characters. For example:

void print (const string& s) { printf ("For people who like printf: %s\n",s.c_str ()); // s.c_str () returns a pointer to s’ characters cout $<<$ "For people who like streams: " $<<\mathsf{s}<<\mathsf{v n}^{\prime}$ ; }

A string literal is by definition a const char ∗ . To get a literal of type std:: string use a s suffix. For example:

auto s $=$ "Cat"s; // a std:: str ing auto $\mathsf{p}="\mathsf{D o g}"$ ; // a C-style string: a const char\*

To use the s suffix, you need to use the namespace std::literals:: string_literals (§5.4.4).

# 9.2.1 string Implementation

Implementing a string class is a popular and useful exercise. However, for general-purpose use, our carefully crafted first attempts rarely match the standard string in convenience or performance. These days, string is usually implemented using the short-string optimization . That is, short string values are kept in the string object itself and only longer strings are placed on free store. Consider:

string s1 {"Annemarie"}; // shor t str ing string s2 {"Annemarie Stroustrup"}; // long string

The memory layout will be something like this:

![](images/707a9393b17e35c04c14dddadbb909433385bd415491cc15f7a492333c923efa.jpg)

When a string ’s value changes from a short to a long string (and vice versa) its representation adjusts appropriately. How many characters can a ‘‘short’’ string have? That’s implementation defined, but ‘‘about 14 characters’’ isn’t a bad guess.

The actual performance of string s can depend critically on the run-time environment. In partic- ular, in multi-threaded implementations, memory allocation can be relatively costly. Also, when lots of strings of differing lengths are used, memory fragmentation can result. These are the main reasons that the short-string optimization has become ubiquitous.

To handle multiple character sets, string is really an alias for a general template basic_string with the character type char :

template<typename Char> class basic_string { // ... string of Char ... };

using string $=$ basic_string<char>;

A user can define strings of arbitrary character types. For example, assuming we have a Japanese character type Jchar , we can write:

using Jstring $=$ basic_string<Jchar>;

Now we can do all the usual string operations on Jstring , a string of Japanese characters.

# 9.3 String Views

The most common use of a sequence of characters is to pass it to some function to read. This can be achieved by passing a string by value, a reference to a string, or a C-style string. In many sys- tems there are further alternatives, such as string types not offered by the standard. In all of these cases, there are extra complexities when we want to pass a substring. To address this, the standard library offers string_view ; a string_view is basically a (pointer, length) pair denoting a sequence of characters:

string_view: { begin () , siz e () } characters: P i e t H e i n

A string_view gives access to a contiguous sequence of characters. The characters can be stored in many possible ways, including in a string and in a C-style string. A string_view is like a pointer or a reference in that it does not own the characters it points to. In that, it resembles an STL pair of iter- ators (§12.3).

Consider a simple function concatenating two strings:

{ string res (sv1.length ()+sv2.length ()); char ∗ ${\mathsf p}=$ for (char c : sv1) // one way to copy ${}^{*}\mathsf{p}_{++}=\mathsf{c},$ copy (sv2.begin (), sv2.end (), p); // another way return res; }

We can call this cat () :

string king $=$ "Harold"; auto s1 $=$ cat (king,"William"); // str ing and const char\* auto ${\mathfrak{s z}}=$ cat (king, king); // str ing and string auto ${\mathfrak{s}}{\mathfrak{z}}=$ cat ("Edward","Stephen"sv); // const char \* and string_view auto ${\mathfrak{s}}4=$ cat ("Canute"sv, king); auto ${\tt s5}={\tt}$ cat ({&king[0], 2},"Henry"sv); // HaHenr y auto ${\mathfrak{s6}}=$ cat ({&king[0], 2},{&king[2], 4}); // Harold

This cat () has three advantages over the compose () that takes const string& arguments (§9.2):

• It can be used for character sequences managed in many different ways. • No temporary string arguments are created for C-style string arguments. • We can easily pass substrings.

Note the use of the sv (‘‘string view’’) suffix. To use that we need to using namespace std::literals:: string view literals; // §5.4.4

Why bother? The reason is that when we pass "Edward" we need to construct a string_view from a const char ∗ and that requires counting the characters. For "Stephen"sv the length is computed at compile time. When returning a string_view , remember that it is very much like a pointer; it needs to point to something: string_view bad (){ string s $=$ "Once upon a time"; return {&s[5], 4}; // bad: returning a pointer to a local }

We are returning a pointer to characters of a string that will be destroyed before we can use them.

One significant restriction of string_view is that it is a read-only view of its characters. For example, you cannot use a string_view to pass characters to a function that modifies its argument to lowercase. For that, you might consider using a gsl:: span or gsl:: string_span (§13.3). The behavior of out-of-range access to a string_view is unspecified. If you want guaranteed range checking, use at () , which throws out_of_rang e for attempted out-of-range access, use a gsl:: string_span (§13.3), or ‘‘just be careful.’’

# 9.4 Regular Expressions

Regular expressions are a powerful tool for text processing. They provide a way to simply and tersely describe patterns in text (e.g., a U.S. postal code such as TX 77845 , or an ISO-style date, such as 2009−06−07 ) and to efficiently find such patterns. In <reg ex> , the standard library provides support for regular expressions in the form of the std:: reg ex class and its supporting functions. To give a taste of the style of the reg ex library, let us define and print a pattern:

reg ex pat $\{{\sf R}" ({\sf N w}\{2\}{\sf s}*{\sf N d}\{5\}(-{\sf N d}\{4\})?)"\};$ // U.S. postal code pattern: XXddddd-dddd and var iants

People who have used regular expressions in just about any language will find $\mathsf{W}\{2\}\backslash\mathsf{s}\ast\mathsf{V d}\{5\}(-\mathsf{V d}\{4\})?$ familiar. It specifies a pattern starting with two letters \w{2} optionally followed by some space $\backslash\mathbf{s}^{*}$ followed by five digits \d{5} and optionally followed by a dash and four digits −\d{4} . If you are not familiar with regular expressions, this may be a good time to learn about them ([Stroustrup, 2009], [Maddock, 2009], [Friedl, 1997]).

To express the pattern, I use a raw string literal starting with $\mathsf{R}" ($ and terminated by $)"$ . This allows backslashes and quotes to be used directly in the string. Raw strings are particularly suitable for regular expressions because they tend to contain a lot of backslashes. Had I used a conventional string, the pattern definition would have been:

reg ex pat {"\\w{2}\\s ∗ \\d{5}(−\\d{4})?"}; // U.S. postal code pattern In <reg ex> , the standard library provides support for regular expressions:

• reg ex_match () : Match a regular expression against a string (of known size) (§9.4.2). • reg ex_search () : Search for a string that matches a regular expression in an (arbitrarily long) stream of data (§9.4.1). • reg ex_replace () : Search for strings that match a regular expression in an (arbitrarily long) stream of data and replace them. • reg ex_iterator : Iterate over matches and submatches (§9.4.3). • reg ex token iterator : Iterate over non-matches.

# 9.4.1 Searching

The simplest way of using a pattern is to search for it in a stream:

int lineno ${\bf\Gamma}={\bf0}$ ; for (string line; getline (cin, line); ) { // read into line buffer $^{++}$ lineno; smatch matches; // matched strings go here if (regex_search (line ,matches, pat)) // search for pat in line cout$<<$ lineno$<<"$: "$<<$ matches[0]$<<\mathfrak{w}$;}

The reg ex_search (line ,matches, pat) searches the line for anything that matches the regular expression stored in pat and if it finds any matches, it stores them in matches . If no match was found, reg ex_search (line ,matches, pat) returns false . The matches variable is of type smatch . The ‘‘s’’ stands for ‘‘sub’’ or ‘‘string,’’ and an smatch is a vector of submatches of type string . The first ele- ment, here matches[0] , is the complete match. The result of a reg ex_search () is a collection of matches, typically represented as an smatch :

void use ()

ifstream in ("file. txt"); // input file if (! in) // check that the file was opened cerr $<<$ "no file\n"; reg ex pat {R" (\w{2}\s ∗ \d{5}(−\d{4})?)"}; // U.S. postal code pattern int lineno ${\bf\Gamma}={\bf0}$ ; for (string line; getline (in, line); ) { $^{++}$ lineno; smatch matches; // matched strings go here if (regex_search (line , matches, pat)) { cout $<<$ lineno $<<"$ : " $<<$ matches[0] $<<\mathfrak{w}$ ; // the complete match if (1<matches. siz e () && matches[1]. matched) // if there is a sub-pattern // and if it is matched cout << "\t: " << matches[1] $<<\mathfrak{w}$ ; // submatch } } }

This function reads a file looking for U.S. postal codes, such as TX77845 and DC 20500−0001 . An smatch type is a container of regex results. Here, matches[0] is the whole pattern and matches[1] is the optional four-digit subpattern.

The newline character, $\mathtt{\backslash n}$ , can be part of a pattern, so we can search for multiline patterns. Obviously, we shouldn’t read one line at a time if we want to do that.

The regular expression syntax and semantics are designed so that regular expressions can be compiled into state machines for efficient execution [Cox, 2007]. The reg ex type performs this compilation at run time.

# 9.4.2 Regular Expression Notation

The reg ex library can recognize several variants of the notation for regular expressions. Here, I use the default notation, a variant of the ECMA standard used for ECMAScript (more commonly known as JavaScript).

The syntax of regular expressions is based on characters with special meaning:

![](images/bf0f4b79b0545352a2e472d841b029ccc3142781005af5ed3ba7b57c53c3368f.jpg) 
For example, we can specify a line starting with zero or more A s followed by one or more B s

followed by an optional C like this:

ˆA ∗ B+C?\$

Examples that match:

AAAAAA AAAAAA BBBB BBBB BC BC B

Examples that do not match:

AAAAA // no B AAAABC // initial space AABBCC // too many Cs

A part of a pattern is considered a subpattern (which can be extracted separately from an smatch ) if it is enclosed in parentheses. For example:

$\forall\mathsf{d}+-\backslash\mathsf{d}+$ // no subpatterns $\mathsf{\backslash d+}(-\mathsf{\backslash d+})$ // one subpattern $(\mathsf{u d}+)(-\mathsf{u d}+)$ // two subpatter ns

A pattern can be optional or repeated (the default is exactly once) by adding a suffix:

![](images/7eb1bfd30f0d16dc0811b8e0aeb9530684bc027ff3789a7b5018653166b8ab23.jpg)

For example:

A{3}B{2,4}C ∗

Examples that match:

AAABBC AAABBB

Examples that do not match:

AABBC // too few As AAABC // too few Bs AAABBBBBCCC // too many Bs

A suffix ? after any of the repetition notations $(?,*,+,$ , and $\{\}$ ) makes the pattern matcher ‘‘lazy’’ or ‘‘non-greedy.’’ That is, when looking for a pattern, it will look for the shortest match rather than the longest. By default, the pattern matcher always looks for the longest match; this is known as the Max Munch rule . Consider:

ababab

The pattern $(\mathsf{a b})+$ matches all of ababab . Howev er, $\scriptstyle (a b)+?$ matches only the first ab . The most common character classifications have names:

![](images/89b96c202769c6e81f3f2067542f8305cf88bd412f0c1cd2fd1a41bfdb62e93d.jpg)

In a regular expression, a character class name must be bracketed by [: :] . For example, [:digit:] matches a decimal digit. Furthermore, they must be used within a [ ] pair defining a character class. Several character classes are supported by shorthand notation:

![](images/5918b16c4f993b419d912fb8fe68bbbc6e7a17f883d45c55ba8087503636a146.jpg)

In addition, languages supporting regular expressions often provide:

![](images/f46ea6dc1e2a5a7d210de658c959b0d119a0a2b3ffca26fba10c345a83416157.jpg)

For full portability, use the character class names rather than these abbreviations.

As an example, consider writing a pattern that describes $\mathrm{C++}$ identifiers: an underscore or a let- ter followed by a possibly empty sequence of letters, digits, or underscores. To illustrate the sub- tleties involved, I include a few false attempts:

[:alpha:][:alnum:] ∗ // wrong: characters from the set ": alpha" followed by ... [[:alpha:]] [[:alnum:]] ∗ // wrong: doesn’t accept underscore $'-'$ is not alpha) ([[:alpha:]] |_) [[:alnum:]] ∗ // wrong: underscore is not part of alnum either ([[:alpha:]] |_)([[:alnum:]] |_) ∗ // OK, but clumsy [[:alpha:]_][[:alnum:]_] ∗ // OK: include the underscore in the character classes [_[:alpha:]][_[:alnum:]] ∗ // also OK [_[:alpha:]]\w ∗ // \w is equivalent to [\_[:alnum:]]

Finally, here is a function that uses the simplest version of reg ex_match () (§9.4.1) to test whether a string is an identifier:

{ reg ex pat {"[\_[:alpha:]]\\w ∗ "}; // underscore or letter // followed by zero or more underscores, letters, or digits return regex_match (s, pat); }

Note the doubling of the backslash to include a backslash in an ordinary string literal. Use raw string literals to alleviate problems with special characters. For example:

bool is*identifier (const string& s) { reg ex pat {R" ([*[:alpha:]]\w ∗ )"}; return regex_match (s, pat); }

Here are some examples of patterns:

Ax∗// A, Ax, AxxxxAx+ // Ax, Axxx Not A \d−?\d // 1-2, 12 Not 1--2 \w{2}−\d{4,5} // Ab-1234, XX-54321, 22-5432 Digits are in \w (\d ∗ :)? (\d+) // 12:3, 1:23, 123, : 123 Not 123: (bs|BS) // bs, BS Not bS [aeiouy] // a, o, u An English vow el, not x [ˆaeiouy] // x, k Not an English vow el, not e [aˆeiouy] // a, ˆ, o, u An English vow el or ˆ

A group (a subpattern) potentially to be represented by a sub_match is delimited by parentheses. If you need parentheses that should not define a subpattern, use (?: rather than plain ( . For example:

$(\backslash\mathsf{s}|\!:\!|,)^{\ast}(\mathsf{d}\mathsf{d}^{\ast})$ // optional spaces, colons, and/or commas followed by an optional number

Assuming that we were not interested in the characters before the number (presumably separators), we could write:

$(?:\backslash{\mathsf{s}}|::|,)^{*}({\mathsf{u d}}*)$ // optional spaces, colons, and/or commas followed by an optional number

This would save the regular expression engine from having to store the first characters: the (?: vari- ant has only one subpattern.

![](images/9f8bb020a6013d7d8cbb82a9c6034abca562f487826daee095a3521fe9d2177c.jpg)

That last pattern is useful for parsing XML. It finds tag/end-of-tag markers. Note that I used a non-greedy match (a lazy match ), $\pmb{\cdot}^{*2}$ , for the subpattern between the tag and the end tag. Had I used plain $\cdot^{*}$ , this input would have caused a problem:

Always look on the <b>bright</b> side of <b>life</b>.

A greedy match for the first subpattern would match the first $<$ with the last $>$ . That would be cor- rect behavior, but unlikely what the programmer wanted. For a more exhaustive presentation of regular expressions, see [Friedl, 1997].

# 9.4.3 Iterators

We can define a reg ex_iterator for iterating over a sequence of characters finding matches for a pat- tern. For example, we can use a sreg ex_iterator (a reg ex_iterator<string>) to output all whitespace-separated words in a string :

void test (){ string input $=$ "aa as; asd $^{++}$ eˆasdf asdfg"; reg ex pat $\{\mathsf{R}" (\mathsf{A}\mathsf{s}+(\mathsf{W}\mathsf{+}))"\}$ ; for (sreg ex_iterator p (input.begin (), input.end (), pat); p!=sregex_iterator{}; ${\mathbf{++}}{\mathbf{p}})$ ) cout $<<({\boldsymbol{*}}{\boldsymbol{\mathsf{p}}})[1]<<"\backslash{\boldsymbol{\mathsf{n}}}"$ ; }

This outputs:

as asd asdfg

We missed the first word, aa , because it has no preceding whitespace. If we simplify the pattern to $\mathsf{R}" ((\mathsf{W}+))"$ , we get

aa as asd e asdf asdfg

A reg ex_iterator is a bidirectional iterator, so we cannot directly iterate over an istream (which offers only an input iterator). Also, we cannot write through a reg ex_iterator , and the default reg ex_iterator ( reg ex_iterator{} ) is the only possible end-of-sequence.

# 9.5 Advice

[1] Use std:: string to own character sequences; $\S9.2$ ; [CG: SL. str. 1].

[2] Prefer string operations to C-style string functions; $\S9.1$ .

[3] Use string to declare variables and members rather than as a base class; $\S9.2$ .

[4] Return string s by value (rely on move semantics); §9.2, $\S9.2.1$ .

[5] Directly or indirectly, use substr () to read substrings and replace () to write substrings; $\S9.2$ .

[6] A string can grow and shrink, as needed; $\S9.2$ .

[7] Use at () rather than iterators or [ ] when you want range checking; $\S9.2$ .

[8] Use iterators and [ ] rather than at () when you want to optimize speed; $\S9.2$ .

[9] string input doesn’t overﬂow; §9.2, $\S10.3$ .

[10] Use c_str () to produce a C-style string representation of a string (only) when you have to; $\S9.2$ .

[11] Use a stringstream or a generic value extraction function (such as to $\scriptscriptstyle<\!\!\mathsf{X}\!\!>$ ) for numeric conver- sion of strings; $\S10.8$ .

[12] A basic_string can be used to make strings of characters on any type; $\S9.2.1$ .

[13] Use the s suffix for string literals meant to be standard-library string s; $\S9.3$ [CG: SL. str. 12].

[14] Use string_view as an argument of functions that needs to read character sequences stored in various ways; $\S9.3$ [CG: SL. str. 2].

[15] Use gsl:: string_span as an argument of functions that needs to write character sequences stored in various ways; $\S9.3$ . [CG: SL. str. 2] [CG: SL. str. 11].

[16] Think of a string_view as a kind of pointer with a size attached; it does not own its characters; $\S9.3$ .

[17] Use the sv suffix for string literals meant to be standard-library string_view s; $\S9.3$ .

[18] Use reg ex for most conventional uses of regular expressions; $\S9.4$ .

[19] Prefer raw string literals for expressing all but the simplest patterns; $\S9.4$ .

[20] Use reg ex_match () to match a complete input; §9.4, $\S9.4.2$ .

[21] Use reg ex_search () to search for a pattern in an input stream; $\S9.4.1$ .

[22] The regular expression notation can be adjusted to match various standards; $\S9.4.2$ .

[23] The default regular expression notation is that of ECMAScript; $\S9.4.2$ .

[24] Be restrained; regular expressions can easily become a write-only language; $\S9.4.2$ .

[25] Note that $\backslash$ allows you to express a subpattern in terms of a previous subpattern; $\S9.4.2$ .

[26] Use ? to make patterns ‘‘lazy’’; $\S9.4.2$ .

[27] Use reg ex_iterator s for iterating over a stream looking for a pattern; $\S9.4.3$ .

# Input and Output

What you see is all you get. – Brian W. Kernighan

• Introduction • Output • Input • I/O State • I/O of User-Defined Types • Formatting • File Streams • String Streams • C-style I/O • File System • Advice

# 10.1 Introduction

The I/O stream library provides formatted and unformatted buffered I/O of text and numeric values. An ostream converts typed objects to a stream of characters (bytes):

![](images/9ad611b652de80736acea8d244276429ddb6d4444aa705425b3697237489e1d4.jpg)

An istream converts a stream of characters (bytes) to typed objects:

![](images/e09cda666f037f4fb04c58b06d42f9e1de04d4b860a5139910ff12fcaa4333fe.jpg)

The operations on istream s and ostream s are described in $\S10.2$ and $\S10.3$ . The operations are type- safe, type-sensitive, and extensible to handle user-defined types $(\S10.5)$ .

Other forms of user interaction, such as graphical I/O, are handled through libraries that are not part of the ISO standard and therefore not described here.

These streams can be used for binary I/O, be used for a variety of character types, be locale spe- cific, and use advanced buffering strategies, but these topics are beyond the scope of this book.

The streams can be used for input into and output from std:: string s $(\S10.3)$ , for formatting into string buffers (§10.8), and for file I/O $(\S10.10)$ .

The I/O stream classes all have destructors that free all resources owned (such as buffers and file handles). That is, they are examples of "Resource Acquisition Is Initialization" (RAII; §5.3).

# 10.2 Output

In <ostream> , the I/O stream library defines output for every built-in type. Further, it is easy to define output of a user-defined type (§10.5). The operator $<<$ (‘‘put to’’) is used as an output opera- tor on objects of type ostream ; cout is the standard output stream and cerr is the standard stream for reporting errors. By default, values written to cout are converted to a sequence of characters. For example, to output the decimal number 10 , we can write:

$$
\begin{array}{r l}&{\mathsf{v o i d\f (f)}}\\ &{\{\qquad}\\ &{\mathsf{c o u t<<10};}\\ &{\}}\end{array}
$$

This places the character 1 followed by the character 0 on the standard output stream. Equivalently, we could write: void g (){ int x {10}; cout $<<\pmb{x}$ ; }

Output of different types can be combined in the obvious way:

void h (int i) { cout $<<$ "the value of i is "; cout $<<$ i; cout $<<\mathfrak{w}$ ; }

For h (10) , the output will be:

the value of i is 10

People soon tire of repeating the name of the output stream when outputting several related items. Fortunately, the result of an output expression can itself be used for further output. For example:

void h2 (int i) { cout $<<$ "the value of i is " $<<\mathsf{i}<<\mathsf{\backslash n}^{\mathsf{r}}$ ; }

This h2 () produces the same output as ${\mathfrak{h}}()$ .

A character constant is a character enclosed in single quotes. Note that a character is output as a character rather than as a numerical value. For example:

void k (){ int $\mathbf{b}="\mathbf{b}"$ ; // note: char implicitly converted to int char ${\mathfrak{c}}={}^{1}{\mathfrak{c}}^{1}$ ; cout $<<\mathsf{a}^{\prime}<<\mathsf{b}<<\mathsf{c};$ }

The integer value of the character $\mathbf{\ddot{b}}$ is 98 (in the ASCII encoding used on the $\mathrm{C++}$ implementation that I used), so this will output a98c .

# 10.3 Input

In <istream> , the standard library offers istream s for input. Like ostream s, istream s deal with char- acter string representations of built-in types and can easily be extended to cope with user-defined types.

The operator ${}>>{}$ (‘‘get from’’) is used as an input operator; cin is the standard input stream. The type of the right-hand operand of ${}>>{}$ determines what input is accepted and what is the target of the input operation. For example:

void f (){ int i; cin ${}>>{}$ i; // read an integer into i double d; cin >> d; // read a double-precision ﬂoating-point number into d }

This reads a number, such as 1234 , from the standard input into the integer variable i and a ﬂoating- point number, such as 12.34e5 , into the double-precision ﬂoating-point variable d .

Like output operations, input operations can be chained, so I could equivalently have written:

void f (){ int i; double d; cin >> i >> d; // read into i and d }

In both cases, the read of the integer is terminated by any character that is not a digit. By default, ${}>>{}$ skips initial whitespace, so a suitable complete input sequence would be

1234 12.34e5

Often, we want to read a sequence of characters. A convenient way of doing that is to read into a string . For example:

void hello (){ cout $<<$ "Please enter your name\n"; string str; cin ${}>>{}$ str; cout $<<$ "Hello, " << str << "!\n"; }

If you type in Eric the response is:

Hello, Eric!

By default, a whitespace character, such as a space or a newline, terminates the read, so if you enter Eric Bloodaxe pretending to be the ill-fated king of York, the response is still:

Hello, Eric!

You can read a whole line using the getline () function. For example:

void hello_line (){ cout $<<$ "Please enter your name\n"; string str; getline (cin, str); cout $<<$ "Hello, " << str << "!\n"; }

With this program, the input Eric Bloodaxe yields the desired output:

Hello, Eric Bloodaxe!

The newline that terminated the line is discarded, so cin is ready for the next input line.

Using the formatted I/O operations is usually less error-prone, more efficient, and less code than manipulating characters one by one. In particular, istream s take care of memory management and range checking. We can do formatting to and from memory using stringstream s (§10.8).

The standard strings have the nice property of expanding to hold what you put in them; you don’t hav e to pre-calculate a maximum size. So, if you enter a couple of megabytes of semicolons, the program will echo pages of semicolons back at you.

# 10.4 I/O State

An iostream has a state that we can examine to determine whether an operation succeeded. The most common use is to read a sequence of values:

vector<int> read_ints (istream& is) { vector<int> res; for (int i; is>>i; ) res. push_back (i); return res; }

This reads from is until something that is not an integer is encountered. That something will typi- cally be the end of input. What is happening here is that the operation $\mathrm{i}\!\!\! s\!\!>>\!\!\!\mathrm{i}$ returns a reference to is , and testing an iostream yields true if the stream is ready for another operation.

In general, the I/O state holds all the information needed to read or write, such as formatting information (§10.6), error state (e.g., has end-of-input been reached?), and what kind of buffering is used. In particular, a user can set the state to reﬂect that an error has occurred $(\S10.5)$ and clear the state if an error wasn’t serious. For example, we could imagine a version of read_ints () that accepted a terminating string:

vector<int> read_ints (istream& is, const string& terminator) { vector<int> res; for (int i; is ${}>>{}$ i; ) res. push_back (i); if (is.eof ()) // fine: end of file return res; if (is.fail ()) { // we failed to read an int; was it the terminator? is.clear (); // reset the state to good () is. ung et (); // put the non-digit back into the stream string s; if (cin>>s && $\==$ terminator) return res; cin.setstate (ios_base::failbit); // add fail () to cin’s state } return res; }

auto $\mathbf{v}=$ read_ints (cin,"stop");

# 10.5 I/O of User-Defined Types

In addition to the I/O of built-in types and standard string s, the iostream library allows programmers to define I/O for their own types. For example, consider a simple type Entr y that we might use to represent entries in a telephone book:

struct Entry { string name; int number; };

We can define a simple output operator to write an Entr y using a {"name", number} format similar to the one we use for initialization in code:

ostream& operator<<(ostream& os, const Entry& e) { return os << "{\"" $<<$ e.name $<<"\"$ , " << e.number $<<$ "}"; }

A user-defined output operator takes its output stream (by reference) as its first argument and returns it as its result.

The corresponding input operator is more complicated because it has to check for correct for- matting and deal with errors:

istream& operator>>(istream& is, Entry& e) // read { "name" , number } pair. Note: for matted with { " " , and } { char c, c2; if (is>>c && ${\mathfrak{c}}{=}{\mathfrak{c}}^{\prime}$ && is>>c2 && c2 $==$ "') { // star t with a { " string name; // the default value of a string is the empty string: "" while (is.get (c) && c!='"') // anything before a " is part of the name name $\scriptstyle{+=6}$ ; if (is>>c && ${\mathfrak{c}}{\mathrel{=}}{\mathfrak{i}}$ ,') { int number ${\bf\Gamma}={\bf0}$ ; if ( $\mathrm{i}\mathbf{s}>>$ number>>c && ${\mathfrak{c}}{=}{\mathfrak{i}}$ ) { // read the number and a } $\mathbf{e}=$ {name ,number}; // assign to the entry return is; } } } is.setstate (ios_base::failbit); // register the failure in the stream return is; }

An input operation returns a reference to its istream that can be used to test if the operation suc- ceeded. For example, when used as a condition, $\mathrm{i}\mathfrak{s}{>}\mathfrak{c}$ means ‘‘Did we succeed at reading a char from is into c ?’’

The $\mathrm{i}\mathfrak{s}{>}\mathfrak{c}$ skips whitespace by default, but is. g et (c) does not, so this Entr y -input operator ignores (skips) whitespace outside the name string, but not within it. For example:

{ "John Marwood Cleese", 123456 {"Michael Edward Palin", 987654}

We can read such a pair of values from input into an Entr y like this:

for (Entr y ee; cin>>ee; ) // read from cin into ee cout << ee $<<\mathfrak{w}$ ; // wr ite ee to cout

The output is:

{"John Marwood Cleese", 123456} {"Michael Edward Palin", 987654}

See $\S9.4$ for a more systematic technique for recognizing patterns in streams of characters (regular expression matching).

# 10.6 Formatting

The iostream library provides a large set of operations for controlling the format of input and out- put. The simplest formatting controls are called manipulators and are found in <ios> , <istream> , <ostream> , and <iomanip> (for manipulators that take arguments). For example, we can output inte- gers as decimal (the default), octal, or hexadecimal numbers:

cout $<<1234<<$ ',' << hex << 1234 << ',' << oct << 1234 << '\n'; // pr int 1234,4d2,2322 We can explicitly set the output format for ﬂoating-point numbers:

# constexpr double ${\mathsf{d}}=123.456$ ;

cout $<<\mathsf{d}<<"~;$ ; " // use the default for mat for d $<<$ scientific << d << "; " // use 1.123e2 style for mat for d $<<$ hexﬂoat << d << "; " // use hexadecimal notation for d << fixed << d << "; " // use 123.456 style for mat for d << defaultﬂoat $<<\mathsf{d}<<\mathsf{\backslash n}!$ '; // use the default for mat for d

This produces:

123.456; 1.234560e+002; 0x1. edd2f2p+6; 123.456000; 123.456

Precision is an integer that determines the number of digits used to display a ﬂoating-point number: • The general format ( defaultﬂoat ) lets the implementation choose a format that presents a value in the style that best preserves the value in the space available. The precision specifies the maximum number of digits. • The scientific format ( scientific ) presents a value with one digit before a decimal point and an exponent. The precision specifies the maximum number of digits after the decimal point. • The fixed format ( fixed ) presents a value as an integer part followed by a decimal point and a fractional part. The precision specifies the maximum number of digits after the decimal point.

Floating-point values are rounded rather than just truncated, and precision () doesn’t affect integer output. For example:

cout.precision (8); cout << 1234.56789 << ' ' << 1234.56789 << ' ' << 123456 << '\n';

cout.precision (4); cout $<<1234.56789<<1`<<1234.56789<<1`<<123456<<19`;$ cout $<<1234.56789<<1\backslash\mathfrak{n}^{\prime}$ ;

This produces:

1234.5679 1234.5679 123456 1235 1235 123456 1235

These ﬂoating-point manipulators are ‘‘sticky’’; that is, their effects persist for subsequent ﬂoating- point operations.

# 10.7 File Streams

In <fstream> , the standard library provides streams to and from a file:

• ifstream s for reading from a file • ofstream s for writing to a file • fstream s for reading from and writing to a file

For example:

ofstream ofs {"target"}; // ‘‘o’’ for ‘‘output’’ if (! ofs) error ("couldn't open 'target' for writing");

Testing that a file stream has been properly opened is usually done by checking its state.

ifstream ifs {"source"}; // ‘‘i’’ for ‘‘input’’ if (! ifs) error ("couldn't open 'source' for reading");

Assuming that the tests succeeded, ofs can be used as an ordinary ostream (just like cout ) and ifs can be used as an ordinary istream (just like cin ). File positioning and more detailed control of the way a file is opened is possible, but beyond the scope of this book. For the composition of file names and file system manipulation, see $\S10.10$ .

# 10.8 String Streams

In <sstream> , the standard library provides streams to and from a string :

• istringstream s for reading from a string • ostringstream s for writing to a string • stringstream s for reading from and writing to a string .

For example:

void test ()

{ ostringstream oss; oss $<<$ "{temperature," $<<$ scientific << 123.4567890 << "}"; cout $<<$ oss.str () $<<\mathfrak{w}$ ; }

The result from an ostringstream can be read using str () . One common use of an ostringstream is to format before giving the resulting string to a GUI. Similarly, a string received from a GUI can be read using formatted input operations (§10.3) by putting it into an istringstream .

A stringstream can be used for both reading and writing. For example, we can define an opera- tion that can convert any type with a string representation into another that can also be represented as a string :

template<typename Target $=$ string, typename Source =string> Targ et to (Source arg) // convert Source to Target { stringstream interpreter; Targ et result; if (! (interpreter $<<$ arg) // wr ite arg into stream || !(interpreter ${}>>{}$ result) // read result from stream $||$ !(interpreter >> std::ws). eof ()) // stuff left in stream? throw runtime_error{"to ${<}0$ failed"}; return result; }

A function template argument needs to be explicitly mentioned only if it cannot be deduced or if there is no default (§7.2.4), so we can write:

auto $\mathbf{x}1=$ to<string, double>(1.2); // very explicit (and verbose) auto $\pmb{x2=}$ to<string>(1.2); // Source is deduced to double auto ${\tt x3}={\tt t o}{<}{\tt>}({\tt^{\prime}}$ 1.2); // Target is defaulted to string; Source is deduced to double auto ${\pmb x}{\pmb4}=$ to (1.2); // the $<>$ is redundant; // Target is defaulted to string; Source is deduced to double

If all function template arguments are defaulted, the $<>$ can be left out.

I consider this a good example of the generality and ease of use that can be achieved by a com- bination of language features and standard-library facilities.

# 10.9 C-style I/O

The $\mathrm{C++}$ standard library also supports the C standard-library I/O, including printf () and scanf () . Many uses of this library are unsafe from a type and security point-of-view, so I don’t recommend its use. In particular, it can be difficult to use for safe and convenient input. It does not support user-defined types. If you don’t use C-style I/O and care about I/O performance, call

ios_base:: sync_with_stdio (false); // avoid significant overhead Without that call, iostream s can be significantly slowed down to be compatible with the C-style I/O.

# 10.10 File System

Most systems have a notion of a file system providing access to permanent information stored as files . Unfortunately, the properties of file systems and the ways of manipulating them vary greatly. To deal with that, the file system library in <filesystem> offers a uniform interface to most facilities of most file systems. Using <filesystem> , we can portably

• express file system paths and navigate through a file system • examine file types and the permissions associated with them The filesystem library can handle unicode, but explaining how is beyond the scope of this book. I recommend the cppreference [Cppreference] and the Boost filesystem documentation [Boost] for detailed information.

Consider an example:

path ${\mathsf{f}}=$ "dir/hypothetical. cpp"; // naming a file

asser t (exists (f)); // f must exist

if (is*regular*file (f)) // is f an ordinary file? cout $<<\mathsf{f}<<"$ is a file; its size is " $<<$ file_siz e (f) $<<\mathfrak{w}$ ;

Note that a program manipulating a file system is usually running on a computer together with other programs. Thus, the contents of a file system can change between two commands. For exam- ple, even though we first of all carefully asserted that f existed, that may no longer be true when on the next line, we ask if f is a regular file. A path is quite a complicated class, capable of handling the native character sets and conven- tions of many operating systems. In particular, it can handle file names from command lines as presented by main () ; for example: int main (int argc, char ∗ argv[]) { if (argc $<z$ ) { cerr $<<$ "arguments expected\n"; return 1; } path p {argv[1]}; // create a path from the command line cout << p << " " << exists (p) $<<\mathfrak{w}$ ; // note: a path can be printed like a str ing // ... }

A path is not checked for validity until it is used. Even then, its validity depends on the conven- tions of the system on which the program runs.

Naturally, a path can be used to open a file

void use (path p) { ofstream f {p}; if (! f) error ("bad file name: ", p); f << "Hello, file!"; }

In addition to path , <filesystem> offers types for traversing directories and inquiring about the prop- erties of the files found:

![](images/734c172612750240d7963e97345a1f56598d8cd954fd5e033b4d1be0f44dbfc4.jpg)

Consider a simple, but not completely unrealistic, example:

void print_directory (path p) tr y { if (is_directory (p)) { cout $<<\mathsf{p}<<"\!:\!\mathsf{N}^{\dprime}$ ; for (const directory_entr y& $\mathbf{x}:$ director y_iterator{p}) cout << " " << x.path () $<<\mathfrak{w}$ ; } } catch (const filesystem_error& ex) { cerr << ex.what () $<<\mathfrak{w}$ ; }

A string can be implicitly converted to a path so we can exercise print_director y like this:

void use () { print_director y ("."); // current directory print_director y (".."); // parent directory print_director y ("/"); // Unix root directory print_director y ("c: "); // Windows volume C for (string s; cin>>s; ) print_director y (s); }

Had I wanted to list subdirectories also, I would have said recursive director y_iterator{p} . Had I wanted to print entries in lexicographical order, I would have copied the path s into a vector and sorted that before printing.

Class path offers many common and useful operations:

![](images/1fc314cb903c5e5926db8d983315328e56b2d5b13cb113cc7829541ccd51e34e.jpg)

For example:

void test (path p)

if (is*directory (p)) { cout $<<\mathsf{p}<<"\!:\!\mathsf{N n}"$ ; for (const directory_entr y& $\mathbf{x}:$ director y_iterator (p)) { const path& ${\bf f}={\bf x}$ ; // refer to the path part of a director y entr y if (f.extension () $==$ ". exe") cout $<<$ f.stem $10<<"$ is a Windows executable\n"; else { string $\mathsf{n}=$ f.extension (). string (); if ( ${\bf\ddot{n}}=={\bf\ddot{n}}$ .cpp" $||{\textsf{n}}=={\"}.{\mathbb{C}}"\ ||{\textsf{n}}=={\"}.{\mathbb{c}}x x")$ cout $<<$ f.stem () $<<$ " is a $\mathtt{c}*{++}$ source file\n"; } } }

We use a path as a string (e.g., f.extension ) and we can extract strings of various types from a path (e.g., f.extension (). string () ).

Note that naming conventions, natural languages, and string encodings are rich in complexity. The filesystem-library abstractions offer portability and great simplification.

![](images/20b10c6844941230d97286abee9fdacd8b6a1c781befecd78bedfddcbdb83704.jpg)

Many operations have overloads that take extra arguments, such as operating systems permissions. The handling of such is far beyond the scope of this book, so look them up if you need them.

Like copy () , all operations come in two versions: • The basic version as listed in the table, e.g., exists (p) . The function will throw filesys- tem_error if the operation failed. • A version with an extra error_code argument, e.g., exists (p, e) . Check e to see if the opera- tions succeeded.

We use the error codes when operations are expected to fail frequently in normal use and the throwing operations when an error is considered exceptional.

Often, using an inquiry function is the simplest and most straightforward approach to examin- ing the properties of a file. The <filesystem> library knows about a few common kinds of files and classifies the rest as ‘‘other’’:

![](images/9786b4ee428fa586773c16908050edf86765b02171a2d82ab674150f5d8402a4.jpg)

# 10.11 Advice

[1] iostream s are type-safe, type-sensitive, and extensible; $\S10.1$ .

[2] Use character-level input only when you have to; $\S10.3$ ; [CG: SL. io. 1].

[3] When reading, always consider ill-formed input; $\S10.3$ ; [CG: SL. io. 2].

[4] Avoid endl (if you don’t know what endl is, you haven’t missed anything); [CG: SL. io. 50].

[5] Define $<<$ and ${}>>{}$ for user-defined types with values that have meaningful textual representa- tions; $\S10.1$ , $\S10.2$ , $\S10.3$ .

[6] Use cout for normal output and cerr for errors; $\S10.1$ .

[7] There are iostream s for ordinary characters and wide characters, and you can define an iostream for any kind of character; $\S10.1$ .

[8] Binary I/O is supported; $\S10.1$ .

[9] There are standard iostream s for standard I/O streams, files, and string s; $\S10.2,\,\S10.3,\,\S10.7,$ , $\S10.8$ .

[10] Chain $<<$ operations for a terser notation; $\S10.2$ .

[11] Chain ${}>>{}$ operations for a terser notation; $\S10.3$ .

[12] Input into string s does not overﬂow; $\S10.3$ .

[13] By default ${}>>{}$ skips initial whitespace; $\S10.3$ .

[14] Use the stream state fail to handle potentially recoverable I/O errors; $\S10.4$ .

[15] You can define $<<$ and ${}>>{}$ operators for your own types; $\S10.5$ .

[16] You don’t need to modify istream or ostream to add new $<<$ and ${}>>{}$ operators; $\S10.5$ .

[17] Use manipulators to control formatting; $\S10.6$ .

[18] precision () specifications apply to all following ﬂoating-point output operations; $\S10.6$ .

[19] Floating-point format specifications (e.g., scientific ) apply to all following ﬂoating-point out- put operations; $\S10.6$ .

[20] #include <ios> when using standard manipulators; $\S10.6$ .

[21] #include <iomanip> when using standard manipulators taking arguments; $\S10.6$ .

[22] Don’t try to copy a file stream.

[23] Remember to check that a file stream is attached to a file before using it; $\S10.7$ .

[24] Use stringstream s for in-memory formatting; $\S10.8$ .

[25] You can define conversions between any two types that both have string representation; $\S10.8$ .

[26] C-style I/O is not type-safe; $\S10.9$ .

[27] Unless you use printf-family functions call ios_base:: sync_with_stdio (false) ; $\S10.9$ ; [CG: SL. io. 10].

[28] Prefer <filesystem> to direct use of a specific operating system interfaces; $\S10.10$ .

# Containers

It was new. It was singular. It was simple. It must succeed! – H. Nelson

• Introduction • vector Elements; Range Checking • list • map • unordered_map • Container Overview • Advice

# 11.1 Introduction

Most computing involves creating collections of values and then manipulating such collections. Reading characters into a string and printing out the string is a simple example. A class with the main purpose of holding objects is commonly called a container . Providing suitable containers for a giv en task and supporting them with useful fundamental operations are important steps in the construction of any program.

To illustrate the standard-library containers, consider a simple program for keeping names and telephone numbers. This is the kind of program for which different approaches appear ‘‘simple and obvious’’ to people of different backgrounds. The Entr y class from $\S10.5$ can be used to hold a simple phone book entry. Here, we deliberately ignore many real-world complexities, such as the fact that many phone numbers do not have a simple representation as a 32-bit int .

# 11.2 vector

The most useful standard-library container is vector . A vector is a sequence of elements of a given type. The elements are stored contiguously in memory. A typical implementation of vector $(\S4.2.2,\,\S5.2)$ will consist of a handle holding pointers to the first element, one-past-the-last ele- ment, and one-past-the-last allocated space $(\S12.1)$ (or the equivalent information represented as a pointer plus offsets):

![](images/bbb50a8d8dbb1740de62b98983eb0bfbbffb1400541f5132fa5057161c3aafde.jpg)

In addition, it holds an allocator (here, alloc ), from which the vector can acquire memory for its ele- ments. The default allocator uses new and delete to acquire and release memory (§13.6).

We can initialize a vector with a set of values of its element type:

vector<Entr y> phone_book $=\left\{\begin{array}{r l}\end{array}\right.$ {"David Hume", 123456}, {"Karl Popper", 234567}, {"Ber trand Ar thur William Russell", 345678} };

Elements can be accessed through subscripting. So, assuming that we have defined $<<$ for Entr y , we can write:

void print_book (const vector<Entry>& book){ for (int $\mathbf{i}=\mathbf{0}$ ; i!=book.size (); ${++\mathrm{i}}$ ) cout $<<$ book[i] $<<\mathfrak{w}$ ; }

As usual, indexing starts at 0 so that book[0] holds the entry for David Hume . The vector member function siz e () gives the number of elements.

The elements of a vector constitute a range, so we can use a range- for loop (§1.7): void print_book (const vector<Entry>& book){ for (const auto& x : book) // for "auto" see §1.4 cout $<<\mathtt{x}<<\mathtt{"}\mathtt{N}\mathtt{"}$ ; }

When we define a vector , we giv e it an initial size (initial number of elements):

vector<int> $\mathsf{v}1=\{1,2,3,$ 4}; // size is 4 vector<string> v2; // size is 0 vector<Shape ∗ > v3 (23); // size is 23; initial element value: nullptr vector<double> v4 (32,9.9); // size is 32; initial element value: 9.9

An explicit size is enclosed in ordinary parentheses, for example, (23) , and by default, the elements are initialized to the element type’s default value (e.g., nullptr for pointers and 0 for numbers). If you don’t want the default value, you can specify one as a second argu- ment (e.g., 9.9 for the 32 elements of v4 ).

The initial size can be changed. One of the most useful operations on a vector is push_back () , which adds a new element at the end of a vector , increasing its size by one. For example, assuming that we have defined ${}>>{}$ for Entr y , we can write:

void input (){ for (Entr y e; cin>>e; ) phone_book. push_back (e); }

This reads Entr y s from the standard input into phone_book until either the end-of-input (e.g., the end of a file) is reached or the input operation encounters a format error.

The standard-library vector is implemented so that growing a vector by repeated push_back () s is efficient. To show how, consider an elaboration of the simple Vector from (Chapter 4 and Chapter 6) using the representation indicated in the diagram above:

template<typename T> class Vector { T ∗ elem; // pointer to first element $\mathsf{T}\ast$ space; // pointer to first unused (and uninitialized) slot $\mathsf{T}\ast$ last; // pointer to last slot public: // ... int size (); // number of elements (space-elem) int capacity (); // number of slots available for elements (last-elem) // ... void reserve (int newsz); // increase capacity () to newsz // ... void push_back (const T& t); // copy t into Vector void push_back (T&& t); // move t into Vector };

The standard-library vector has members capacity () , reser ve () , and push_back () . The reser ve () is used by users of vector and other vector members to make room for more elements. It may have to allocate new memory and when it does, it moves the elements to the new allocation.

Given capacity () and reser ve () , implementing push_back () is trivial:

template<typename T> void Vector<T>:: push_back (const T& t) { if (capacity ()<size ()+1) // make sure we have space for t reser ve (siz e ( $\mathrel{\mathop:}=$ 0?8:2 ∗ siz e ()); // double the capacity new (space) T{t}; // initialize \*space to t ++space; }

Now allocation and relocation of elements happens only infrequently. I used to use reser ve () to try to improve performance, but that turned out to be a waste of effort: the heuristic used by vector is on average better than my guesses, so now I only explicitly use reser ve () to avoid reallocation of elements when I want to use pointers to elements.

A vector can be copied in assignments and initializations. For example:

vector<Entr y> book2 $=$ phone_book;

Copying and moving of vector s are implemented by constructors and assignment operators as described in $\S5.2$ . Assigning a vector involves copying its elements. Thus, after the initialization of book2 , book2 and phone_book hold separate copies of every Entr y in the phone book. When a vector holds many elements, such innocent-looking assignments and initializations can be expen- sive. Where copying is undesirable, references or pointers $(\S1.7)$ or move operations (§5.2.2) should be used.

The standard-library vector is very ﬂexible and efficient. Use it as your default container; that is, use it unless you have a solid reason to use some other container. If you avoid vector because of concerns about ‘‘efficiency,’’ measure. Our intuition is most fallible in matters of the performance of container uses.

# 11.2.1 Elements

Like all standard-library containers, vector is a container of elements of some type T , that is, a vector<T> . Just about any type qualifies as an element type: built-in numeric types (such as char , int , and double ), user-defined types (such as string , Entr y , list<int> , and Matrix<double ,2> ), and point- ers (such as const char ∗ , Shape ∗ , and double ∗ ). When you insert a new element, its value is copied into the container. For example, when you put an integer with the value 7 into a container, the resulting element really has the value 7 . The element is not a reference or a pointer to some object containing 7 . This makes for nice, compact containers with fast access. For people who care about memory sizes and run-time performance this is critical.

If you have a class hierarchy (§4.5) that relies on vir tual functions to get polymorphic behavior, do not store objects directly in a container. Instead store a pointer (or a smart pointer; $\S13.2.1)$ . For example:

vector<Shape> vs; vector<Shape ∗ > vps; vector<unique_ptr<Shape>> vups;

// No, don’t - there is no room for a Circle or a Smiley

// better, but see §4.5.3

// OK

# 11.2.2 Range Checking

The standard-library vector does not guarantee range checking. For example:

void silly (vector<Entr y>& book) { int $=$ book[book.size ()]. number; // book.size () is out of range // ... }

That initialization is likely to place some random value in i rather than giving an error. This is undesirable, and out-of-range errors are a common problem. Consequently, I often use a simple range-checking adaptation of vector :

template<typename T> class Vec : public std::vector<T> { public: using vector<T>:: vector; // use the constructors from vector (under the name Vec) T& operator[](int i) // range check { return vector<T>:: at (i); } const T& operator[](int i) const // range check const objects; §4.2.1 { return vector<T>:: at (i); } };

Vec inherits everything from vector except for the subscript operations that it redefines to do range checking. The at () operation is a vector subscript operation that throws an exception of type out_of_rang e if its argument is out of the vector ’s range (§3.5.1).

For Vec , an out-of-range access will throw an exception that the user can catch. For example: void checked (Vec<Entr y>& book) { tr y { book[book. siz e ()] $=$ {"Joe", 999999}; // will throw an exception // ... } catch (out_of_rang e&) { cerr $<<$ "range error\n"; } }

The exception will be thrown, and then caught (§3.5.1). If the user doesn’t catch an exception, the program will terminate in a well-defined manner rather than proceeding or failing in an undefined manner. One way to minimize surprises from uncaught exceptions is to use a main () with a tr y - block as its body. For example:

int main () tr y { // your code } catch (out_of_rang e&) { cerr $<<$ "range error\n"; } catch (...) { cerr $<<$ "unknown exception thrown\n"; }

This provides default exception handlers so that if we fail to catch some exception, an error mes- sage is printed on the standard error-diagnostic output stream cerr (§10.2).

Why doesn’t the standard guarantee range checking? Many performance-critical applications use vector s and checking all subscripting implies a cost on the order of $10\%$ . Obviously, that cost can vary dramatically depending on hardware, optimizers, and an application’s use of subscripting. However, experience shows that such overhead can lead people to prefer the far more unsafe built- in arrays. Even the mere fear of such overhead can lead to disuse. At least vector is easily range checked at debug time and we can build checked versions on top of the unchecked default. Some implementations save you the bother of defining Vec (or equivalent) by providing a range-checked version of vector (e.g., as a compiler option).

A range- for avoids range errors at no cost by accessing elements through iterators in the range [ begin () : end () ). As long as their iterator arguments are valid, the standard-library algorithms do the same to ensure the absence of range errors.

If you can use vector:: at () directly in your code, you don’t need my Vec workaround. Further- more, some standard libraries have range-checked vector implementations that offer more complete checking than Vec .

# 11.3 list

The standard library offers a doubly-linked list called list :

![](images/39abe368e809796188dcdb0bf89ba53e0e11f945875955704284934141e4ec1f.jpg)

We use a list for sequences where we want to insert and delete elements without moving other ele- ments. Insertion and deletion of phone book entries could be common, so a list could be appropri- ate for representing a simple phone book. For example:

list<Entr y> phone_book $=\left\{\begin{array}{r l}\end{array}\right.$ {"David Hume", 123456}, {"Karl Popper", 234567}, {"Ber trand Ar thur William Russell", 345678} };

When we use a linked list, we tend not to access elements using subscripting the way we com- monly do for vectors. Instead, we might search the list looking for an element with a given value. To do this, we take advantage of the fact that a list is a sequence as described in Chapter 12:

int get_number (const string& s) { for (const auto& x : phone_book) if (x.name $\mathrel{\mathop:}=\mathrel{\mathop:}\mathsf{s},$ ) return x.number; return 0; // use 0 to represent "number not found" }

The search for s starts at the beginning of the list and proceeds until s is found or the end of phone_book is reached.

Sometimes, we need to identify an element in a list . For example, we may want to delete an element or insert a new element before it. To do that we use an iterator : a list iterator identifies an element of a list and can be used to iterate through a list (hence its name). Every standard-library container provides the functions begin () and end () , which return an iterator to the first and to one- past-the-last element, respectively (Chapter 12). Using iterators explicitly, we can – less elegantly – write the get_number () function like this:

int get_number (const string& s)

{ for (auto ${\mathsf p}=$ phone*book.begin (); p!=phone_book.end (); ${\mathrel{+{+}}}{\mathsf{p}},$ ) if (p−>name $\mathrel{\mathop:}=\mathrel{\mathop:}\mathfrak{s}*{i}$ ) return p−>number; return 0; // use 0 to represent "number not found" }

In fact, this is roughly the way the terser and less error-prone range- for loop is implemented by the compiler. Giv en an iterator p , ${}^{*}{\mathfrak{p}}$ is the element to which it refers, ${\bf++}{\bf p}$ advances p to refer to the next element, and when p refers to a class with a member m , then $\mathsf{p}\mathrm{--}\mathsf{s m}$ is equivalent to $({}^{*}{\mathsf{p}}).{\mathsf{m}}$ .

Adding elements to a list and removing elements from a list is easy:

void f (const Entry& ee, list<Entr y>:: iterator p, list<Entry>:: iterator q) { phone_book. inser t (p, ee); // add ee before the element referred to by $p$ phone_book.erase (q); // remove the element referred to by $q$ }

For a list , inser t (p, elem) inserts an element with a copy of the value elem before the element pointed to by p . Here, p may be an iterator pointing one-beyond-the-end of the list . Conversely, erase (p) removes the element pointed to by p and destroys it.

These list examples could be written identically using vector and (surprisingly, unless you understand machine architecture) perform better with a small vector than with a small list . When all we want is a sequence of elements, we have a choice between using a vector and a list . Unless you have a reason not to, use a vector . A vector performs better for traversal (e.g., find () and count () ) and for sorting and searching (e.g., sor t () and equal_rang e () ; $\S12.6$ , $\S13.4.3)$ ).

The standard library also offers a singly-linked list called forward_list :

![](images/5674852f61094ba09159891422dcb8d04446afbaed1794b1532e93e8565d1606.jpg)

A forward_list differs from a list by only allowing forward iteration. The point of that is to save space. There is no need to keep a predecessor pointer in each link and the size of an empty for- ward_list is just one pointer. A forward_list doesn’t even keep its number of elements. If you need the element count, count. If you can’t afford to count, you probably shouldn’t use a forward_list .

# 11.4 map

Writing code to look up a name in a list of (name, number) pairs is quite tedious. In addition, a lin- ear search is inefficient for all but the shortest lists. The standard library offers a balanced binary search tree (usually, a red-black tree) called map :

![](images/48dd46024dabe14127891b593b1c0f4949d558018dfbd218f5cdf490b5be2059.jpg)

In other contexts, a map is known as an associative array or a dictionary. It is implemented as a bal- anced binary tree.

The standard-library map is a container of pairs of values optimized for lookup. We can use the same initializer as for vector and list (§11.2, §11.3):

map<string, int> phone_book { {"David Hume", 123456}, {"Karl Popper", 234567}, {"Ber trand Ar thur William Russell", 345678} };

When indexed by a value of its first type (called the key ), a map returns the corresponding value of the second type (called the value or the mapped type ). For example:

int get_number (const string& s) { return phone_book[s];}

In other words, subscripting a map is essentially the lookup we called get_number () . If a key isn’t found, it is entered into the map with a default value for its value . The default value for an integer type is 0 ; the value I just happened to choose represents an invalid telephone number.

If we wanted to avoid entering invalid numbers into our phone book, we could use find () and inser t () instead of [ ] .

# 11.5 unordered_map

The cost of a map lookup is O (log (n)) where n is the number of elements in the map . That’s pretty good. For example, for a map with 1,000,000 elements, we perform only about 20 comparisons and indirections to find an element. However, in many cases, we can do better by using a hashed lookup rather than a comparison using an ordering function, such as $<$ . The standard-library hashed containers are referred to as ‘‘unordered’’ because they don’t require an ordering function:

![](images/cfb92cdc8982b696367b481106452b820c7b438e04cca7bd03a26fbfe5f034b6.jpg)

For example, we can use an unordered_map from <unordered_map> for our phone book:

unordered_map<string, int> phone_book { {"David Hume", 123456}, {"Karl Popper", 234567}, {"Ber trand Ar thur William Russell", 345678} };

Like for a map , we can subscript an unordered_map :

int get_number (const string& s) { return phone_book[s];}

The standard library provides a default hash function for string s as well as for other built-in and standard-library types. If necessary, you can provide your own (§5.4.6). Possibly, the most com- mon need for a ‘‘custom’’ hash function comes when we want an unordered container of one of our own types. A hash function is often provided as a function object (§6.3.2). For example:

struct Record { string name; int product_code; // ... }; struct Rhash { // a hash function for Record siz e_t operator ()(const Record& r) const{ return hash<string>()(r.name) ˆ hash<int>()(r.product_code); } };

unordered_set<Record, Rhash> my_set; // set of Records using Rhash for lookup

Designing good hash functions is an art and sometimes requires knowledge of the data to which it will be applied. Creating a new hash function by combining existing hash functions using exclu- sive-or $\mathrm{()}$ is simple and often very effective.

We can avoid explicitly passing the hash operation by defining it as a specialization of the stan- dard-library hash :

namespace std { // make a hash function for Record template $<>$ struct hash<Record> { using argument_type $=$ Record; using result_type $=$ std:: size_t; siz e_t operator ()(const Record& r) const{ return hash<string>()(r.name) ˆ hash<int>()(r.product_code); } }; }

Note the differences between a map and an unordered_map :

• A map requires an ordering function (the default is $\nleq$ ) and yields an ordered sequence. • A unordered_map requires and an equality function (the default is $==$ ); it does not maintain an order among its elements.

Given a good hash function, an unordered_map is much faster than a map for large containers. However, the worst-case behavior of an unordered_map with a poor hash function is far worse than that of a map .

# 11.6 Container Overview

The standard library provides some of the most general and useful container types to allow the pro- grammer to select a container that best serves the needs of an application:

![](images/65e52a57ee97fb5b8b76aed54d43452683ef7a2270edd8c4fd2a157811bd2b99.jpg)

The unordered containers are optimized for lookup with a key (often a string); in other words, they are implemented using hash tables.

The containers are defined in namespace std and presented in headers <vector> , <list> , <map> , etc. (§8.3). In addition, the standard library provides container adaptors queue<T> , stack<T> , and priority_queue<T> . Look them up if you need them. The standard library also provides more specialized container-like types, such as array $\tt<T, N>$ (§13.4.1) and bitset<N> (§13.4.2).

The standard containers and their basic operations are designed to be similar from a notational point of view. Furthermore, the meanings of the operations are equivalent for the various contain- ers. Basic operations apply to every kind of container for which they make sense and can be effi- ciently implemented:

![](images/05d1ff13fb5a376ad699003252da305156d642042abecce31e83a149b7953b62.jpg)

This notational and semantic uniformity enables programmers to provide new container types that can be used in a very similar manner to the standard ones. The range-checked vector, Vector (§3.5.2, Chapter 4), is an example of that. The uniformity of container interfaces allows us to spec- ify algorithms independently of individual container types. However, each has strengths and weak- nesses. For example, subscripting and traversing a vector is cheap and easy. On the other hand, vector elements are moved when we insert or remove elements; list has exactly the opposite proper- ties. Please note that a vector is usually more efficient than a list for short sequences of small ele- ments (even for inser t () and erase () ). I recommend the standard-library vector as the default type for sequences of elements: you need a reason to choose another.

Consider the singly-linked list, forward_list , a container optimized for the empty sequence (§11.3). An empty forward_list occupies just one word, whereas an empty vector occupy three. Empty sequences, and sequences with only an element or two, are surprisingly common and useful.

An emplace operation, such as emplace_back () takes arguments for an element’s constructor and builds the object in a newly allocated space in the container, rather than copying an object into the container. For example, for a vector<pair<int, string>> we could write:

v.push_back (pair{1,"copy or move")); // make a pair and move it into v v.emplace_back (1,"build in place"); // buid a pair in v

# 11.7 Advice

[1] An STL container defines a sequence; $\S11.2$ .

[2] STL containers are resource handles; $\S11.2$ , $\S11.3$ , §11.4, $\S11.5$ .

[3] Use vector as your default container; $\S11.2$ , $\S11.6$ ; [CG: SL. con. 2].

[4] For simple traversals of a container, use a range- for loop or a begin/end pair of iterators; $\S11.2,\S11.3$ .

[5] Use reser ve () to avoid invalidating pointers and iterators to elements; $\S11.2$ .

[6] Don’t assume performance benefits from reser ve () without measurement; $\S11.2$ .

[7] Use push_back () or resiz e () on a container rather than realloc () on an array; $\S11.2$ .

[8] Don’t use iterators into a resized vector ; $\S11.2$ .

[9] Do not assume that [ ] range checks; $\S11.2$ .

[10] Use at () when you need guaranteed range checks; $\S11.2$ ; [CG: SL. con. 3].

[11] Use range- for and standard-library algorithms for cost-free avoidance of range errors; $\S11.2.2$ .

[12] Elements are copied into a container; $\S11.2.1$ .

[13] To preserve polymorphic behavior of elements, store pointers; $\S11.2.1$ .

[14] Insertion operations, such as inser t () and push_back () , are often surprisingly efficient on a vector ; $\S11.3$ .

[15] Use forward_list for sequences that are usually empty; $\S11.6$ .

[16] When it comes to performance, don’t trust your intuition: measure; $\S11.2$ .

[17] A map is usually implemented as a red-black tree; $\S11.4$ .

[18] An unordered_map is a hash table; $\S11.5$ .

[19] Pass a container by reference and return a container by value; $\S11.2$ .

[20] For a container, use the () -initializer syntax for sizes and the $\{\}$ -initializer syntax for lists of elements; $\S4.2.3$ , $\S11.2$ .

[21] Prefer compact and contiguous data structures; $\S11.3$ .

[22] A list is relatively expensive to traverse; $\S11.3$ .

[23] Use unordered containers if you need fast lookup for large amounts of data; $\S11.5$ .

[24] Use ordered associative containers (e.g., map and set ) if you need to iterate over their ele- ments in order; $\S11.4$ .

[25] Use unordered containers for element types with no natural order (e.g., no reasonable $\nleq$ ); $\S11.4$ .

[26] Experiment to check that you have an acceptable hash function; $\S11.5$ .

[27] A hash function obtained by combining standard hash functions for elements using the exclu- sive-or operator ( ˆ ) is often good; $\S11.5$ .

[28] Know your standard-library containers and prefer them to handcrafted data structures; $\S11.6$ .

# Algorithms

Do not multiply entities beyond necessity. – William Occam

• Introduction • Use of Iterators • Iterator Types • Stream Iterators • Predicates • Algorithm Overview • Concepts • Container Algorithms • Parallel Algorithms • Advice

# 12.1 Introduction

A data structure, such as a list or a vector, is not very useful on its own. To use one, we need opera- tions for basic access such as adding and removing elements (as is provided for list and vector ). Furthermore, we rarely just store objects in a container. We sort them, print them, extract subsets, remove elements, search for objects, etc. Consequently, the standard library provides the most common algorithms for containers in addition to providing the most common container types. For example, we can simply and efficiently sort a vector of Entr y s and place a copy of each unique vector element on a list :

void f (vector<Entry>& vec, list<Entry>& lst)

{ sor t (vec.begin (), vec.end ()); // use $<$ for order unique_copy (vec.begin (), vec.end (), lst.begin ()); // don’t copy adjacent equal elements

For this to work, less than $(<)$ and equal $(==)$ must be defined for Entr y s. For example:

bool operator<(const Entry& x, const Entry& y) // less than { return x.name<y.name; // order Entries by their names }

A standard algorithm is expressed in terms of (half-open) sequences of elements. A sequence is represented by a pair of iterators specifying the first element and the one-beyond-the-last element:

![](images/7e89bb903207401cd253a72172097bca77b6e9ba22e471466ffe47ecc5b094af.jpg)

In the example, sor t () sorts the sequence defined by the pair of iterators vec.begin () and vec.end () , which just happens to be all the elements of a vector . For writing (output), you need only to specify the first element to be written. If more than one element is written, the elements following that ini- tial element will be overwritten. Thus, to avoid errors, lst must have at least as many elements as there are unique values in vec .

If we wanted to place the unique elements in a new container, we could have written:

list<Entr y> f (vector<Entr y>& vec) { list<Entr y> res; sor t (vec.begin (), vec.end ()); unique_copy (vec.begin (), vec.end (), back_inser ter (res)); // append to res return res; }

The call back_inser ter (res) constructs an iterator for res that adds elements at the end of a container, extending the container to make room for them. This saves us from first having to allocate a fixed amount of space and then filling it. Thus, the standard containers plus back_inser ter () s eliminate the need to use error-prone, explicit C-style memory management using realloc () . The standard-library list has a move constructor (§5.2.2) that makes returning res by value efficient (even for list s of thousands of elements).

If you find the pair-of-iterators style of code, such as sor t (vec.begin (), vec.end ()) , tedious, you can define container versions of the algorithms and write sor t (vec) (§12.8).

# 12.2 Use of Iterators

For a container, a few iterators referring to useful elements can be obtained; begin () and end () are the best examples of this. In addition, many algorithms return iterators. For example, the standard algorithm find looks for a value in a sequence and returns an iterator to the element found:

bool has_c (const string& s, char c) // does s contain the character c? { auto ${\mathsf p}=$ find (s.begin (),s.end (), c); if $\scriptstyle{\mathfrak{p}}!=\!\mathfrak{s}.$. end ()) return true; else return false; }

Like many standard-library search algorithms, find returns end () to indicate ‘‘not found.’’ An equiv- alent, shorter, definition of has_c () is:

bool has_c (const string& s, char c) // does s contain the character c? { return find (s.begin (),s.end (), c)!=s.end (); }

A more interesting exercise would be to find the location of all occurrences of a character in a string. We can return the set of occurrences as a vector of string iterators. Returning a vector is efficient because vector provides move semantics (§5.2.1). Assuming that we would like to modify the locations found, we pass a non- const string:

vector<string::iterator> find_all (string& s, char c) // find all occurrences of c in s { vector<string::iterator> res; for (auto ${\mathsf p}=$ s.begin (); p!=s.end (); ${\mathbf{++}}{\mathbf{p}})$ ) if $({}^{*}\mathsf{p}\mathrm{==}\mathsf{c})$ res. push_back (p); return res; }

We iterate through the string using a conventional loop, moving the iterator p forward one element at a time using $^{++}$ and looking at the elements using the dereference operator $^*$ . We could test find_all () like this:

void test (){ string m {"Mary had a little lamb"}; for (auto p : find_all (m,'a')) if $({}{^\ast}{\mathsf{p}}!\!\equiv\!\!{\mathsf{a}}!)$ cerr << "a bug!\n"; }

That call of find_all () could be graphically represented like this:

![](images/ac24ddcd424592d5e57e4ef8134d5c9010b4ffb3c6283c8a0d0afe5e6d7d2f4f.jpg)

Iterators and standard algorithms work equivalently on every standard container for which their use makes sense. Consequently, we could generalize find_all () :

template<typename C, typename $\lor>$ vector<typename C::iterator> find_all (C& c, V v) // find all occurrences of v in c { vector<typename C::iterator> res; for (auto ${\mathsf p}=$ c.begin (); p!=c.end (); ${\bf++}{\bf p})$ ) if $({}^{*}\mathsf{p}\mathrm{==}\mathsf{v})$ ) res. push_back (p); return res; }

The typename is needed to inform the compiler that C ’s iterator is supposed to be a type and not a value of some type, say, the integer 7 . We can hide this implementation detail by introducing a type alias (§6.4.2) for Iterator :

template<typename T> using Iterator $=$ typename T:: iterator; // T’s iterator template<typename C, typename $\lor>$ vector<Iterator<C>> find_all (C& c, V v) // find all occurrences of v in c { vector<Iterator<C>> res; for (auto ${\mathsf p}=$ c.begin (); p!=c.end (); ${\mathfrak{++}}{\mathfrak{p}}$ ) if $({}^{*}\mathsf{p}\mathrm{==}\mathsf{v})$ res. push_back (p); return res; }

We can now write:

void test (){ string m {"Mary had a little lamb"}; for (auto p : find_all (m,'a')) // p is a str ing:: iterator if $({}^{*}{\mathsf{p}}!\!=\!\!{\mathsf{a}}^{\mathsf{i}})$ cerr $<<$ "string bug!\n"; list<double> ld {1.1, 2.2, 3.3, 1.1}; for (auto p : find_all (ld, 1.1)) // p is a list<double>:: iterator if $({\tt s p l}{=}1.1)$ ) cerr $<<$ "list bug!\n"; vector<string> vs { "red", "blue", "green", "green", "orange", "green" }; for (auto p : find_all (vs,"red")) // p is a vector<str ing>:: iterator if $[{}^{*}{\mathsf{p}}!=$ "red") cerr $<<$ "vector bug!\n";

for (auto p : find_all (vs,"green")) ${}^{*}{\mathsf{p}}={}$ "ver t";

Iterators are used to separate algorithms and containers. An algorithm operates on its data through iterators and knows nothing about the container in which the elements are stored. Conversely, a container knows nothing about the algorithms operating on its elements; all it does is to supply iter- ators upon request (e.g., begin () and end () ). This model of separation between data storage and algorithm delivers very general and ﬂexible software.

# 12.3 Iterator Types

What are iterators really? Any particular iterator is an object of some type. There are, however, many different iterator types, because an iterator needs to hold the information necessary for doing its job for a particular container type. These iterator types can be as different as the containers and the specialized needs they serve. For example, a vector ’s iterator could be an ordinary pointer, because a pointer is quite a reasonable way of referring to an element of a vector :

![](images/481d4759eb34c31f7bb2640f59d78d8cb951011b7b08dc32f39be1d5755d7a8d.jpg)

Alternatively, a vector iterator could be implemented as a pointer to the vector plus an index:

![](images/feb36cd1dfa18c2aac5fcfd4567be9482751fbf37132a02659820415db157462.jpg)

Using such an iterator would allow range checking.

A list iterator must be something more complicated than a simple pointer to an element because an element of a list in general does not know where the next element of that list is. Thus, a list iter- ator might be a pointer to a link:

![](images/1889d2624d1c46a5f9dbf86cfa1cdcf781eef93374d7aa62eec5dc9b3a418da7.jpg)

What is common for all iterators is their semantics and the naming of their operations. For exam- ple, applying $^{++}$ to any iterator yields an iterator that refers to the next element. Similarly, $^*$ yields the element to which the iterator refers. In fact, any object that obeys a few simple rules like these is an iterator – Iterator is a concept $(\S7.2,\S12.7)$ . Furthermore, users rarely need to know the type of a specific iterator; each container ‘‘knows’’ its iterator types and makes them available under the conventional names iterator and const_iterator . For example, list<Entr y>:: iterator is the general itera- tor type for list<Entr y> . We rarely have to worry about the details of how that type is defined.

# 12.4 Stream Iterators

Iterators are a general and useful concept for dealing with sequences of elements in containers. However, containers are not the only place where we find sequences of elements. For example, an input stream produces a sequence of values, and we write a sequence of values to an output stream. Consequently, the notion of iterators can be usefully applied to input and output.

To make an o stream iterator , we need to specify which stream will be used and the type of objects written to it. For example:

o stream iterator<string> oo {cout}; // wr ite str ings to cout The effect of assigning to ∗ oo is to write the assigned value to cout. For example:

{ ∗ oo $=$ "Hello, "; // meaning cout<<"Hello, " ${\bf++o o}$ ; ∗ oo $=$ "world!\n"; // meaning cout<<"wor ld!\n" }

This is yet another way of writing the canonical message to standard output. The ${\bf++o o}$ is done to mimic writing into an array through a pointer. Similarly, an i stream iterator is something that allows us to treat an input stream as a read-only container. Again, we must specify the stream to be used and the type of values expected:

i stream iterator<string> ii {cin};

Input iterators are used in pairs representing a sequence, so we must provide an i stream iterator to indicate the end of input. This is the default i stream iterator :

i stream iterator<string> eos $\{\}$ ;

Typically, i stream iterator s and o stream iterator s are not used directly. Instead, they are provided as arguments to algorithms. For example, we can write a simple program to read a file, sort the words read, eliminate duplicates, and write the result to another file:

int main ()

{ string from, to; cin ${}>>{}$ from >> to; // get source and target file names ifstream is {from}; // input stream for file "from" i stream iterator<string> ii {is}; // input iterator for stream i stream iterator<string> eos {}; // input sentinel

ofstream os {to}; // output stream for file "to" o stream iterator<string> oo {os,"\n"}; // output iterator for stream vector<string> b {ii, eos}; // b is a vector initialized from input sor t (b.begin (),b.end ()); // sor t the buffer unique_copy (b.begin (),b.end (), oo); // copy buffer to output, discard replicated values return !is.eof () || !os; // retur n error state (§1.2.1, §10.4)

An ifstream is an istream that can be attached to a file, and an ofstream is an ostream that can be attached to a file (§10.7). The o stream iterator ’s second argument is used to delimit output values.

Actually, this program is longer than it needs to be. We read the strings into a vector , then we sor t () them, and then we write them out, eliminating duplicates. A more elegant solution is not to store duplicates at all. This can be done by keeping the string s in a set , which does not keep dupli- cates and keeps its elements in order (§11.4). That way, we could replace the two lines using a vector with one using a set and replace unique_copy () with the simpler copy () :

set<string> b {ii, eos}; // collect strings from input copy (b.begin (),b.end (), oo); // copy buffer to output

We used the names ii , eos , and oo only once, so we could further reduce the size of the program:

int main () { string from, to; cin ${}>>{}$ from >> to; // get source and target file names ifstream is {from}; // input stream for file "from" ofstream os {to}; // output stream for file "to" set<string> b {i stream iterator<string>{is}, i stream iterator<string>{}}; // read input copy (b.begin (),b.end (), o stream iterator<string>{os,"\n"}); // copy to output return !is.eof () || !os; // retur n error state (§1.2.1, §10.4) }

It is a matter of taste and experience whether or not this last simplification improves readability.

# 12.5 Predicates

In the examples so far, the algorithms have simply ‘‘built in’’ the action to be done for each element of a sequence. However, we often want to make that action a parameter to the algorithm. For example, the find algorithm $(\S12.2,\ \S12.6)$ provides a convenient way of looking for a specific value. A more general variant looks for an element that fulfills a specified requirement, a predicate . For example, we might want to search a map for the first value larger than 42 . A map allows us to access its elements as a sequence of (key, value) pairs, so we can search a map<string, int> ’s sequence for a pair<const string, int> where the int is greater than 42 :

void f (map<string, int>& m) { auto ${\mathsf p}=$ find_if (m.begin (),m.end (), Greater_than{42}); // ... } Here, Greater_than is a function object (§6.3.2) holding the value ( 42 ) to be compared against: struct Greater_than { int val; Greater_than (int v) $:$ val{v} { } bool operator ()(const pair<string, int>& r) const { return r.second>val; } }; Alternatively, we could use a lambda expression (§6.3.2): auto${\mathsf p}=$ find_if (m.begin (), m.end (), [](const auto& r) { return r.second>42; });

A predicate should not modify the elements to which it is applied.

# 12.6 Algorithm Overview

A general definition of an algorithm is ‘‘a finite set of rules which gives a sequence of operations for solving a specific set of problems [and] has five important features: Finiteness ... Definiteness ... Input ... Output ... Effectiveness’’ [Knuth, 1968,§1.1]. In the context of the $\mathrm{C++}$ standard library, an algorithm is a function template operating on sequences of elements.

The standard library provides dozens of algorithms. The algorithms are defined in namespace std and presented in the <algorithm $\mid>$ header. These standard-library algorithms all take sequences as inputs. A half-open sequence from b to e is referred to as [ b : e ). Here are a few examples:

![](images/0ee8b880098f2f324ca73f292dc6f4b5cff93ffb6fd754e3215650ea9f62926c.jpg)

![](images/72524fdc76eccb951c7768fa8e3985bfc80639df5b8461fa9af13da0c7c877d1.jpg)

These algorithms, and many more (e.g., $\S14.3)$ , can be applied to elements of containers, string s, and built-in arrays.

Some algorithms, such as replace () and sor t (), modify element values, but no algorithm adds or subtracts elements of a container. The reason is that a sequence does not identify the container that holds the elements of the sequence. To add or delete elements, you need something that knows about the container (e.g., a back_inser ter ; $\S12.1\bigcirc$ ) or directly refers to the container itself (e.g., push_back () or erase () ; $\S11.2)$ ).

Lambdas are very common as operations passed as arguments. For example:

vector<int> $\mathsf{v}=\{\boldsymbol{0}, 1,\! 2,\! 3,\! 4,\! 5\}$; for_each (v.begin (),v.end (),[](int& x){ $\scriptstyle{\pmb{\chi}}={\pmb{\chi}}^{*}{\pmb{\chi}}$ ; }); // v=={0,1,4,9,16,25}

The standard-library algorithms tend to be more carefully designed, specified, and implemented than the average hand-crafted loop, so know them and use them in preference to code written in the bare language.

# 12.7 Concepts $\mathbf{(C++20)}$

In $_{\mathrm{C++20}}$ , the standard-library algorithms will be specified using concepts (Chapter 7). The pre- liminary work on this can be found in the Ranges Technical Specification [RangesTS]. Implemen- tations can be found on the Web. For $_{\mathrm{C++20}}$ , the ranges concepts are defined in <rang es> .

Rang e is a generalization of the $_{\textrm{C++98}}$ sequences defined by begin () / end () pairs. Rang e is a con- cept specifying what it takes to be a sequence of elements. It can be defined by

• A {begin, end} pair of iterators • A {begin, n} pair, where begin is an iterator and n is the number of elements • A {begin, pred} pair, where begin is an iterator and pred is a predicate; if pred (p) is true for the iterator p , we hav e reached the end of the sequence. This allows us to have infinite sequences and sequences that are generated ‘‘on the ﬂy.’’

This Rang e concept is what allows us to say sor t (v) rather than sor t (v.begin (),v.end ()) as we had to using the STL since 1994. For example:

template<BoundedRang e R> requires Sortable<R> void sort (R& r){ return sort (begin (r), end (r));

The relation for Sor table is defaulted to less . In general, where a standard-library algorithm requires a sequence defined by a pair of iterators, $\mathrm{C++20}$ will allow a Rang e as a notationally simpler alternative.

In addition to Rang e , $_{\mathrm{C++20}}$ offers many useful concepts. These concepts are found in the headers <rang es> , <iterator> , and <concepts> .

![](images/7285e86da0d2cb242476da23d312fec35625096915d4d2b2a67bd5c453cc16c6.jpg)

Common is important for specifying algorithms that should work with a variety of related types while still being mathematically sound. Common<T, U> is a type C that we can use for comparing a T with a U by first converting both to C s. For example, we would like to compare a std:: string with a C-style string (a char ∗ ) and an int with a double but not a std:: string with an int . To ensure that we specialize common_type_t , used in the definition of Common , suitably:

using common_type_t<std:: string, char ∗ > $=$ std:: string; using common_type_t<double ,int> $=$ double;

The definition of Common is a bit tricky but solves a hard fundamental problem. Fortunately, we don’t need to define a common_type_t specialization unless we want to use operations on mixes of types for which a library doesn’t (yet) have suitable definitions. Common or CommonReference is used in the definitions of most concepts and algorithms that can compare values of different types.

The concepts related to comparison are strongly inﬂuenced by [Stepanov, 2009].

![](images/75a38d0e53225a9fc417ef4fe7d80c097cafb1025ed55202be1533893b60fb68.jpg) 
The use of both Weakly Equality Comparable With and Weakly Equality Comparable shows a (so far) missed opportunity to overload.

![](images/fe999c7081c88189b008834e3f8f7dbf375d77a23aadf29c5e34f56e0b3001cd.jpg) 
Regular is the ideal for types. A Regular type works roughly like an int and simplifies much of our thinking about how to use a type $(\S7.2)$ . The lack of default $==$ for classes means that most classes start out as SemiRegular ev en though most could and should be Regular.

![](images/63c49c58698e955eacd12575de830c54c5a1e4bf02b440aefaaef40bcb15706e.jpg) 
A function f () is equality preserving if $\pmb{x}{=}\mathbf{y}$ implies that ${\mathfrak{f}}(\mathsf{x}){=}{\mathfrak{f}}(\mathsf{y})$ . Strict weak ordering is what the standard library usually assumes for comparisons, such as $<$ ; look it up if you feel the need to know. Relation and StrictWeakOrder differ only in semantics. We can’t (currently) represent that in code so the names simply express our intent.

![](images/ea2afc8172a836195d0064069c77ac5307ca16115a3406dc75b3a0786385a15d.jpg)

The different kinds (categories) of iterators are used to select the best algorithm for a given algo- rithm; see $\S7.2.2$ and $\S13.9.1$ . For an example of an InputIterator , see $\S12.4$ .

The basic idea of a sentinel is that we can iterate over a range starting at an iterator until the predicate becomes true for an element. That way, an iterator $\mathsf{p}$ and a sentinel s define a range $\left[{\mathfrak{p}}{\cdot}{\mathfrak{s}}({\ast}{\mathfrak{p}})\right)$ . For example, we could define a predicate for a sentinel for traversing a C-style string using a pointer as the iterator:

[](const char∗p) {return$\scriptstyle{*\!\mathsf{p}}=0$; }

The summary of Mergeable and Sor table are simplified relative to their definition in $_{\mathrm{C++20}}$ .

![](images/92ce3523184c2b2ee565d58a1331b8857e7840f1c4252d6290cf777026f7cfeb.jpg)

There are a few more concepts in <rang es> , but this set is a good start.

# 12.8 Container Algorithms

When we can’t wait for Ranges, we can define our own simple range algorithms. For example, we can easily provide the shorthand to say just sor t (v) instead of sor t (v.begin (),v.end ()) :

namespace Estd { using namespace std; template<typename C> void sort (C& c){ sor t (c.begin (),c.end ()); } template<typename C, typename Pred> void sort (C& c, Pred p) { sor t (c.begin (),c.end (), p); } // ... }

I put the container versions of sor t () (and other algorithms) into their own namespace Estd (‘‘extended std ’’) to avoid interfering with other programmers’ uses of namespace std and also to make it easier to replace this stopgap with Rang e s.

# 12.9 Parallel Algorithms

When the same task is to be done to many data items, we can execute it in parallel on each data item provided the computations on different data items are independent:

• parallel execution : tasks are done on multiple threads (often running on several processor cores) • vectorized execution : tasks are done on a single thread using vectorization, also known as SIMD (‘‘Single Instruction, Multiple Data’’). The standard library offers support for both and we can be specific about wanting sequential execu- tion; in <execution> , we find: • seq : sequential execution • par : parallel execution (if feasible) • par_unseq : parallel and/or unsequenced (vectorized) execution (if feasible). Consider std:: sor t () : sor t (v.begin (),v.end ()); // sequential

sequential (same as the default) sor t (par,v.begin (),v.end ()); // parallel sor t (par_unseq,v.begin (),v.end ()); // parallel and/or vector ized

Whether it is worthwhile to parallelize and/or vectorize depends on the algorithm, the number of elements in the sequence, the hardware, and the utilization of that hardware by programs running on it. Consequently, the execution policy indicators are just hints. A compiler and/or run-time scheduler will decide how much concurrency to use. This is all nontrivial and the rule against mak- ing statements about efficiency without measurement is very important here.

Most standard-library algorithms, including all in the table in $\S12.6$ except equal_rang e , can be requested to be parallelized and vectorized using par and par_unseq as for sor t () . Why not

equal_rang e () ? Because so far nobody has come up with a worthwhile parallel algorithm for that. Many parallel algorithms are used primarily for numeric data; see $\S14.3.1$ . When requesting parallel execution, be sure to avoid data races (§15.2) and deadlock (§15.5).

# 12.10 Advice

[1] An STL algorithm operates on one or more sequences; $\S12.1$ .

[2] An input sequence is half-open and defined by a pair of iterators; $\S12.1$ .

[3] When searching, an algorithm usually returns the end of the input sequence to indicate ‘‘not found’’; $\S12.2$ .

[4] Algorithms do not directly add or subtract elements from their argument sequences; $\S12.2$ , $\S12.6$ .

[5] When writing a loop, consider whether it could be expressed as a general algorithm; $\S12.2$ .

[6] Use predicates and other function objects to give standard algorithms a wider range of mean- ings; $\S12.5$ , $\S12.6$ .

[7] A predicate must not modify its argument; $\S12.5$ .

[8] Know your standard-library algorithms and prefer them to hand-crafted loops; $\S12.6$ .

[9] When the pair-of-iterators style becomes tedious, introduce a container/range algorithm; $\S12.8$ .

# Utilities

The time you enjoy wasting is not wasted time. – Bertrand Russell

• Introduction • Resource Management unique_ptr and shared_ptr; move () and forward () • Range Checking: span • Specialized Containers array ; bitset ; pair and tuple • Alternatives variant ; optional ; any • Time • Function Adaption Lambdas as Adaptors; mem_fn () ; function • Allocators • Type Functions iterator_traits ; Type Predicates; enable_if • Advice

# 13.1 Introduction

Not all standard-library components come as part of obviously labeled facilities, such as ‘‘contain- ers’’ or ‘‘I/O.’’ This section gives a few examples of small, widely useful components. Such com- ponents (classes and templates) are often called vocabulary types because they are part of the com- mon vocabulary we use to describe our designs and programs. Such library components often act as building blocks for more powerful library facilities, including other components of the standard library. A function or a type need not be complicated or closely tied to a mass of other functions and types to be useful.

# 13.2 Resource Management

One of the key tasks of any nontrivial program is to manage resources. A resource is something that must be acquired and later (explicitly or implicitly) released. Examples are memory, locks, sockets, thread handles, and file handles. For a long-running program, failing to release a resource in a timely manner (‘‘a leak’’) can cause serious performance degradation and possibly even a mis- erable crash. Even for short programs, a leak can become an embarrassment, say by a resource shortage increasing the run time by orders of magnitude.

The standard library components are designed not to leak resources. To do this, they rely on the basic language support for resource management using constructor/destructor pairs to ensure that a resource doesn’t outlive an object responsible for it. The use of a constructor/destructor pair in Vector to manage the lifetime of its elements is an example (§4.2.2) and all standard-library con- tainers are implemented in similar ways. Importantly, this approach interacts correctly with error handling using exceptions. For example, this technique is used for the standard-library lock classes:

mutex m; // used to protect access to shared data

// ... void f (){ scoped_lock<mutex> lck {m}; // acquire the mutex m // ... manipulate shared data ... }

A thread will not proceed until lck ’s constructor has acquired the mutex (§15.5). The corresponding destructor releases the resources. So, in this example, scoped_lock ’s destructor releases the mutex when the thread of control leaves f () (through a return , by ‘‘falling off the end of the function,’’ or through an exception throw).

This is an application of RAII (the ‘‘Resource Acquisition Is Initialization’’ technique; $\S4.2.2)$ . RAII is fundamental to the idiomatic handling of resources in $\mathrm{C++}$ . Containers (such as vector and map , string , and iostream ) manage their resources (such as file handles and buffers) similarly.

# 13.2.1 unique_ptr and shared_ptr

The examples so far take care of objects defined in a scope, releasing the resources they acquire at the exit from the scope, but what about objects allocated on the free store? In <memor y> , the stan- dard library provides two ‘‘smart pointers’’ to help manage objects on the free store:

[1] unique_ptr to represent unique ownership [2] shared_ptr to represent shared ownership The most basic use of these ‘‘smart pointers’’ is to prevent memory leaks caused by careless pro- gramming. For example:

void f (int i, int j) // X\* vs. unique_ptr<X>

$\mathsf{X}\ast\mathsf{p}=\mathsf{n e w}\,\mathsf{X};$ // allocate a new X unique_ptr<X> sp {new X}; // allocate a new X and give its pointer to unique_ptr // ...

if (i<99) throw Z{}; // may throw an exception if $(\mathsf{j}\!<\! 77)$ return; // may retur n "ear ly" // ... use p and sp .. delete p; // destroy \*p }

Here, we ‘‘forgot’’ to delete p if $_{\mathrm{i}<\mathfrak{g g}}$ or if $\mathsf{j}\!<\! 77$ . On the other hand, unique_ptr ensures that its object is properly destroyed whichever way we exit f () (by throwing an exception, by executing return , or by ‘‘falling off the end’’). Ironically, we could have solved the problem simply by not using a pointer and not using new :

void f (int i, int j)// use a local var iable{ X x; // ... }

Unfortunately, overuse of new (and of pointers and references) seems to be an increasing problem.

However, when you really need the semantics of pointers, unique_ptr is a very lightweight mechanism with no space or time overhead compared to correct use of a built-in pointer. Its further uses include passing free-store allocated objects in and out of functions:

unique_ptr<X> make_X (int i) // make an X and immediately give it to a unique_ptr { // ... check i, etc. ... return unique_ptr<X>{new X{i}}; }

A unique_ptr is a handle to an individual object (or an array) in much the same way that a vector is a handle to a sequence of objects. Both control the lifetime of other objects (using RAII) and both rely on move semantics to make return simple and efficient.

The shared_ptr is similar to unique_ptr except that shared_ptr s are copied rather than moved. The shared_ptr s for an object share ownership of an object; that object is destroyed when the last of its shared_ptr s is destroyed. For example:

void f (shared*ptr<fstream>); void g (shared_ptr<fstream>); void user (const string& name, ios_base:: openmode mode) { shared_ptr<fstream> fp {new fstream (name ,mode)}; if (! ∗ fp) // make sure the file was properly opened throw No*file{}; f (fp); g (fp); // ... }

Now, the file opened by fp ’s constructor will be closed by the last function to (explicitly or implic- itly) destroy a copy of fp . Note that $\mathfrak{f}()$ or ${\mathfrak{g}}()$ may spawn a task holding a copy of fp or in some other way store a copy that outlives user () . Thus, shared_ptr provides a form of garbage collection that respects the destructor-based resource management of the memory-managed objects. This is neither cost free nor exorbitantly expensive, but it does make the lifetime of the shared object hard to predict. Use shared_ptr only if you actually need shared ownership.

Creating an object on the free store and then passing the pointer to it to a smart pointer is a bit verbose. It also allows for mistakes, such as forgetting to pass a pointer to a unique_ptr or giving a pointer to something that is not on the free store to a shared_ptr . To avoid such problems, the stan- dard library (in <memor y> ) provides functions for constructing an object and returning an appropri- ate smart pointer, make_shared () and make_unique () . For example:

struct S { int i; string s; double d; // ... };

auto p1 $=$ make_shared<S>(1,"Ankh Morpork", 4.65); // p1 is a shared_ptr<S> auto ${\mathfrak{p}}{\mathfrak{Z}}=$ make_unique<S $\mathrm{i}>$ // p2 is a unique_ptr<S>

Now, p2 is a unique_ptr<S> pointing to a free-store-allocated object of type S with the value {2,"Oz"s, 7.62} .

Using make_shared () is not just more convenient than separately making an object using new and then passing it to a shared_ptr , it is also notably more efficient because it does not need a sepa- rate allocation for the use count that is essential in the implementation of a shared_ptr .

Given unique_ptr and shared_ptr , we can implement a complete ‘‘no naked new ’’ policy (§4.2.2) for many programs. However, these ‘‘smart pointers’’ are still conceptually pointers and therefore only my second choice for resource management – after containers and other types that manage their resources at a higher conceptual level. In particular, shared_ptr s do not in themselves provide any rules for which of their owners can read and/or write the shared object. Data races $(\S15.7)$ and other forms of confusion are not addressed simply by eliminating the resource management issues.

Where do we use ‘‘smart pointers’’ (such as unique_ptr ) rather than resource handles with oper- ations designed specifically for the resource (such as vector or thread )? Unsurprisingly, the answer is ‘‘when we need pointer semantics.’’

• When we share an object, we need pointers (or references) to refer to the shared object, so a shared_ptr becomes the obvious choice (unless there is an obvious single owner). • When we refer to a polymorphic object in classical object-oriented code (§4.5), we need a pointer (or a reference) because we don’t know the exact type of the object referred to (or ev en its size), so a unique_ptr becomes the obvious choice. • A shared polymorphic object typically requires shared_ptr s.

We do not need to use a pointer to return a collection of objects from a function; a container that is a resource handle will do that simply and efficiently (§5.2.2).

# 13.2.2 move () and forward ()

The choice between moving and copying is mostly implicit (§3.6). A compiler will prefer to move when an object is about to be destroyed (as in a return ) because that’s assumed to be the simpler and more efficient operation. However, sometimes we must be explicit. For example, a unique_ptr is the sole owner of an object. Consequently, it cannot be copied:

void f1 () { auto ${\mathsf p}=$ make_unique<int>(2); auto ${\mathfrak{q}}={\mathfrak{p}}$ ; // error : we can’t copy a unique_ptr // ... }

If you want a unique_ptr elsewhere, you must move it. For example:

void f1 () { auto ${\mathsf p}=$ make_unique<int>(2); auto ${\sf q}={\sf}$ move (p); // p now holds nullptr // ... }

Confusingly, std:: move () doesn’t move anything. Instead, it casts its argument to an rvalue refer- ence, thereby saying that its argument will not be used again and therefore may be moved (§5.2.2). It should have been called something like rvalue_cast . Like other casts, it’s error-prone and best avoided. It exists to serve a few essential cases. Consider a simple swap:

template <typename T> void swap (T& a, T& b) { T tmp {move (a)}; // the T constructor sees an rvalue and moves a $=$ move (b); // the T assignment sees an rvalue and moves b = move (tmp); // the T assignment sees an rvalue and moves }

We don’t want to repeatedly copy potentially large objects, so we request moves using std:: move () . Like for other casts, there are tempting, but dangerous, uses of std:: move () . Consider:

string s1 $=$ "Hello"; string ${\mathfrak{s z}}=$ "World"; vector<string> v; v.push_back (s1); // use a "const string&" argument; push_back () will copy v.push_back (move (s2)); // use a move constr uctor

Here s1 is copied (by push_back () ) whereas s2 is moved. This sometimes (only sometimes) makes the push_back () of s2 cheaper. The problem is that a moved-from object is left behind. If we use s2 again, we have a problem:

cout $<<$ s1[2]; // wr ite ’l’ cout $<<$ s2[2]; // crash?

I consider this use of std:: move () to be too error-prone for widespread use. Don’t use it unless you can demonstrate significant and necessary performance improvement. Later maintenance may acci- dentally lead to unanticipated use of the moved-from object.

The state of a moved-from object is in general unspecified, but all standard-library types leave a moved-from object in a state where it can be destroyed and assigned to. It would be unwise not to follow that lead. For a container (e.g., vector or string ), the moved-from state will be ‘‘empty.’’ For many types, the default value is a good empty state: meaningful and cheap to establish.

Forwarding arguments is an important use case that requires moves $(\S7.4.2)$ . We sometimes want to transmit a set of arguments on to another function without changing anything (to achieve ‘‘perfect forwarding’’):

template<typename T, typename ... Args> unique_ptr<T> make_unique (Args&&... args)

return unique_ptr<T>{new T{std::forward<Args>(args)...}}; // forward each argument }

The standard-library forward () differs from the simpler std:: move () by correctly handling subtleties to do with lvalue and rvalue (§5.2.2). Use std:: forward () exclusively for forwarding and don’t for- ward () something twice; once you have forwarded an object, it’s not yours to use anymore.

# 13.3 Range Checking: gsl::span

Traditionally, range errors have been a major source of serious errors in C and $\mathrm{C++}$ programs. The use of containers (Chapter 11), algorithms (Chapter 12), and range- for has significantly reduced this problem, but more can be done. A key source of range errors is that people pass pointers (raw or smart) and then rely on convention to know the number of elements pointed to. The best advice for code outside resource handles is to assume that at most one object is pointed to [CG: F.22], but without support that advice is unmanageable. The standard-library string_view (§9.3) can help, but that is read-only and for characters only. Most programmers need more.

The Core Guidelines [Stroustrup, 2015] offer guidelines and a small Guidelines Support Library [GSL], including a span type for referring to a range of elements. This span is being proposed for the standard, but for now it is just something you can download if needed.

A string_span is basically a (pointer, length) pair denoting a sequence of elements:

span<int> : { begin () , siz e () }

integers: 1 2 3 5 8 13 21 34 55

A span gives access to a contiguous sequence of elements. The elements can be stored in many ways, including in vector s and built-in arrays. Like a pointer, a span does not own the characters it points to. In that, it resembles a string_view (§9.3) and an STL pair of iterators $(\S12.3)$ .

Consider a common interface style:

void fpn (int ∗ p, int n) { for (int $\mathbf{i}={\mathbf{0}}$ ; i<n; ${\bf++i}$ ) p[i] ${\bf\Gamma}={\bf0}$ ; }

We assume that p points to n integers. Unfortunately, this assumption is simply a convention, so we can’t use it to write a range- for loop and the compiler cannot implement cheap and effective range checking. Also, our assumption can be wrong:

void use (int x) { int a[100]; fpn (a, 100); // OK fpn (a, 1000); // oops, my finger slipped! (range error in fpn) fpn (a+10,100); // range error in fpn fpn (a, x); // suspect, but looks innocent }

We can do better using a span :

void fs (span<int> p) { for (int& x : p) $\pmb{\chi}=\pmb{0}$ ; }

We can use fs like this:

void use (int x) { int a[100]; fs (a); // implicitly creates a span<int>{a, 100} fs (a, 1000); // error : span expected fs ({a+10,100}); // a range error in fs fs ({a, x}); // obviously suspect }

That is, the common case, creating a span directly from an array, is now safe (the compiler com- putes the element count) and notationally simple. For other cases, the probability of mistakes is lowered because the programmer has to explicitly compose a span.

The common case where a span is passed along from function to function is simpler than for (pointer, count) interfaces and obviously doesn’t require extra checking:

void f1 (span<int> p); void f2 (span<int> p) { // ... f1 (p); }

When used for subscripting (e.g., r[i] ), range checking is done and a gsl:: fail_fast is thrown in case of a range error. Range checks can be suppressed for performance critical code. When span makes it into the standard, I expect that std:: span will use contracts [Garcia, 2016] [Garcia, 2018] to control responses to range violation.

Note that just a single range check is needed for the loop. Thus, for the common case where the body of a function using a span is a loop over the span , range checking is almost free.

A span of characters is supported directly and called gsl:: string_span .

# 13.4 Specialized Containers

The standard library provides several containers that don’t fit perfectly into the STL framework (Chapter 11, Chapter 12). Examples are built-in arrays, array , and string . I sometimes refer to those as ‘‘almost containers,’’ but that is not quite fair: they hold elements, so they are containers, but each has restrictions or added facilities that make them awkward in the context of the STL. Describing them separately also simplifies the description of the STL.

![](images/cac6128d9e237f7e89e70ae6ef886589116fc4b54e248121787a92a60a82704c.jpg)

Why does the standard library provide so many containers? They serve common but different (often overlapping) needs. If the standard library didn’t provide them, many people would have to design and implement their own. For example:

• pair and tuple are heterogeneous; all other containers are homogeneous (all elements are of the same type). • array , vector , and tuple elements are contiguously allocated; forward_list and map are linked structures. • bitset and vector<bool> hold bits and access them through proxy objects; all other standard- library containers can hold a variety of types and access elements directly. • basic_string requires its elements to be some form of character and to provide string manip- ulation, such as concatenation and locale-sensitive operations. • valarray requires its elements to be numbers and to provide numerical operations.

All of these containers can be seen as providing specialized services needed by large communities of programmers. No single container could serve all of these needs because some needs are contra- dictory, for example, ‘‘ability to grow’’ vs. ‘‘guaranteed to be allocated in a fixed location,’’ and ‘‘elements do not move when elements are added’’ vs. ‘‘contiguously allocated.’’

# 13.4.1 array

An array , defined in <array> , is a fixed-size sequence of elements of a given type where the number of elements is specified at compile time. Thus, an array can be allocated with its elements on the stack, in an object, or in static storage. The elements are allocated in the scope where the array is defined. An array is best understood as a built-in array with its size firmly attached, without implicit, potentially surprising conversions to pointer types, and with a few convenience functions provided. There is no overhead (time or space) involved in using an array compared to using a built-in array. An array does not follow the ‘‘handle to elements’’ model of STL containers. Instead, an array directly contains its elements.

An array can be initialized by an initializer list:

array<int, $_{3>}$ a1 $=\{1,\! 2,\! 3\}$ ;

The number of elements in the initializer must be equal to or less than the number of elements specified for the array .

The element count is not optional:

array<int>$\mathsf{a x}=\{1,\! 2,\! 3\}$;// error size not specified

The element count must be a constant expression:

void f (int n){ array<string, n> aa $=$ {"John's", "Queens' "}; // error : size not a constant expression // }

If you need the element count to be a variable, use vector

When necessary, an array can be explicitly passed to a C-style function that expects a pointer. For example:

void f (int ∗ p, int sz); // C-style interface void g (){ array<int, 10> a; f (a,a.siz e ()); // error : no conversion f (&a[0],a.siz e ()); // $C$ -style use f (a.data (),a.siz e ()); // C-style use auto ${\mathsf p}=$ find (a.begin (),a.end (), 777); // $C{\mathrm{++}}/S T L$ -style use // ... }

Why would we use an array when vector is so much more ﬂexible? An array is less ﬂexible so it is simpler. Occasionally, there is a significant performance advantage to be had by directly accessing elements allocated on the stack rather than allocating elements on the free store, accessing them indirectly through the vector (a handle), and then deallocating them. On the other hand, the stack is a limited resource (especially on some embedded systems), and stack overﬂow is nasty.

Why would we use an array when we could use a built-in array? An array knows its size, so it is easy to use with standard-library algorithms, and it can be copied using $=$ . Howev er, my main rea- son to prefer array is that it saves me from surprising and nasty conversions to pointers. Consider:

#

{ Circle a1[10]; array<Circle, $10\!\!>$ a2; // ... Shape ∗ ${\mathfrak{p}}1={\mathfrak{a}}1$ ; // OK: disaster waiting to happen Shape ∗ ${\mathfrak{p}}{\mathfrak{z}}={\mathfrak{a}}{\mathfrak{z}}$ ; // error : no conversion of array<Circle, 10> to Shape\* p1[3]. draw (); // disaster }

The ‘‘disaster’’ comment assumes that siz eof (Shape)<siz eof (Circle) , so subscripting a Circle[] through a Shape ∗ gives a wrong offset. All standard containers provide this advantage over built-in arrays.

# 13.4.2 bitset

Aspects of a system, such as the state of an input stream, are often represented as a set of ﬂags indi- cating binary conditions such as good/bad, true/false, and on/off. $\mathrm{C++}$ supports the notion of small sets of ﬂags efficiently through bitwise operations on integers (§1.4). Class bitset<N> generalizes this notion by providing operations on a sequence of N bits [ 0 : N ), where N is known at compile time. For sets of bits that don’t fit into a long long int , using a bitset is much more convenient than using integers directly. For smaller sets, bitset is usually optimized. If you want to name the bits, rather than numbering them, you can use a set (§11.4) or an enumeration (§2.5).

A bitset can be initialized with an integer or a string:

bitset<9> bs1 {"110001111"}; bitset<9> bs2 {0b1'1000'1111}; // binar y literal using digit separators (§1.4)

The usual bitwise operators (§1.4) and the left- and right-shift operators $<<\mathrm{and}>>$ ) can be applied:

bitset<9> bs3 $=$ ˜bs1; // complement: $b s3{=}"{0}{01110000}"$ bitset<9> bs4 $=$ bs1&bs3; // all zeros bitset<9> bs5 $=$ bs1<<2; // shift left: $b s5="O O O11111O O"$

The shift operators (here, $<<$ ) ‘‘shift in’’ zeros. The operations to_ullong () and to_string () provide the inverse operations to the constructors. For example, we could write out the binary representation of an int :

void binary (int i) { bitset<8 ∗ siz eof (int)> ${\sf b}={\sf i}$ ; // assume 8-bit byte (see also §14.7) cout $<<$ b.to_string () $<<\mathfrak{w}$ '; // wr ite out the bits of i }

This prints the bits represented as 1 s and 0 s from left to right, with the most significant bit leftmost, so that argument 123 would give the output

00000000000000000000000001111011

For this example, it is simpler to directly use the bitset output operator:

void binary2 (int i) { bitset<8 ∗ siz eof (int)> ${\mathsf b}={\mathsf i}$ ; // assume 8-bit byte (see also §14.7) cout $<<\mathsf{b}<<\mathsf{v n}^{\prime}$ ; // wr ite out the bits of i }

# 13.4.3 pair and tuple

Often, we need some data that is just data; that is, a collection of values, rather than an object of a class with well-defined semantics and an invariant for its value (§3.5.2). In such cases, a simple struct with an appropriate set of appropriately named members is often ideal. Alternatively, we could let the standard library write the definition for us. For example, the standard-library algo- rithm equal_rang e returns a pair of iterators specifying a subsequence meeting a predicate:

template<typename Forward_iterator, typename T, typename Compare> pair<Forward iterator, Forward iterator> equal_rang e (Forward iterator first, Forward iterator last, const T& val, Compare cmp);

Given a sorted sequence [ first : last ), equal_rang e () will return the pair representing the subsequence that matches the predicate cmp . We can use that to search in a sorted sequence of Record s:

auto less$=$ [](const Record& r1, const Record& r2) { return r1. name<r2. name;};// compare namesvoid f (const vector<Record>& v) // assume that v is sorted on its "name" field { auto er $=$ equal_range (v.begin (),v.end (), Record{"Reg"}, less); for (auto ${\mathsf p}=$ er.first; p!=er. second; ${\mathbf{++}}{\mathbf{p}})$ // pr int all equal records cout $<<\boldsymbol{\mathsf{p}}$ ; // assume that $<<$ is defined for Record }

The first member of a pair is called first and the second member is called second . This naming is not particularly creative and may look a bit odd at first, but such consistent naming is a boon when we want to write generic code. Where the names first and second are too generic, we can use struc- tured binding (§3.6.3):

void f2 (const vector<Record>& v) // assume that v is sorted on its "name" field { auto [first, last] $=$ equal_range (v.begin (),v.end (), Record{"Reg"}, less); for (auto ${\mathsf p}=$ first; p!=last; ${\mathfrak{++}}{\mathfrak{p}},$ ) // pr int all equal records cout $<<\boldsymbol{\mathsf{p}}$ ; // assume that $<<$ is defined for Record }

The standard-library pair (from <utility> ) is quite frequently used in the standard library and else- where. A pair provides operators, such a $;=,==,$ , and $<$ , if its elements do. Type deduction makes it easy to create a pair without explicitly mentioning its type. For example:

void f (vector<string>& v)

{ pair p1 {v.begin (), 2}; // one way auto ${\mathfrak{p}}{\mathfrak{2}}=$ make_pair (v.begin (), 2); // another way // ... }

Both p1 and p2 are of type pair<vector<string>:: iterator, int>

If you need more than two elements (or less), you can use tuple (from <utility> ). A tuple is a het- erogeneous sequence of elements; for example:

tuple<string, int, double> t1 {"Shark", 123,3.14}; // the type is explicitly specified auto ${\bf t2=}$ make_tuple (string{"Herring"}, 10,1.23); // the type is deduced to tuple<string, int, double> tuple t3 {"Cod"s, 20,9.99}; // the type is deduced to tuple<string, int, double>

Older code tends to use make_tuple () because template argument type deduction from constructor arguments is $\mathrm{C++17}$ .

Access to tuple members is through a get function template:

string ${\sf s}={\sf g e t}\!<\! 0\!>\! ({\sf t}^{\cdot}$ 1); // get the first element: "Shark" int $\pmb{x}=\mathsf{g e t}\!<\! 1\!>\! ($ t1); // get the second element: 123 double d $=$ get<2>(t1); // get the third element: 3.14

The elements of a tuple are numbered (starting with zero) and the indices must be constants.

Accessing members of a tuple by their index is general, ugly, and somewhat error-prone. Fortu- nately, an element of a tuple with a unique type in that tuple can be ‘‘named’’ by its type:

auto $\mathbf{s}=$ get<string>(t1); // get the string: "Shark" auto $\pmb{x}=$ get<int>(t1); // get the int: 123 auto d $=$ get<double>(t1); // get the double: 3.14

We can use get $<>$ for writing also:

get<string>(t1) $=$ "Tuna"; // wr ite to the string get<int>(t1) ${\o}=7$ ; // wr ite to the int get<double> $\ngtr ({\sf t1})=312$ ; // wr ite to the double

Like pair s, tuple s can be assigned and compared if their elements can be. Like tuple elements, pair elements can be accessed using get<>() .

Like for pair , structured binding (§3.6.3) can be used for tuple . Howev er, when code doesn’t need to be generic, a simple struct with named members often leads to more maintainable code.

# 13.5 Alternatives

The standard library offers three types to express alternatives:

• variant to represent one of a specified set of alternatives (in <variant> ) • optional to represent a value of a specified type or no value (in <optional> ) • any to represent one of an unbounded set of alternative types (in <any> )

These three types offer related functionality to the user. Unfortunately, they don’t offer a unified interface.

# Section 13.5.1

# 13.5.1 variant

A variant $\scriptstyle{<\!\!\mathsf{A},\mathsf{B},\mathsf{C}\!\!>}$ is often a safer and more convenient alternative to explicitly using a union (§2.4). Possibly the simplest example is to return either a value or an error code:

variant<string, int> compose_message (istream& s) { string mess; // ... read from s and compose message ... if (no_problems) return mess; // retur n a str ing else return error_number; // retur n an int }

When you assign or initialize a variant with a value, it remembers the type of that value. Later, we can inquire what type the variant holds and extract the value. For example:

auto $\mathfrak{m}=$ compose_message (cin)); if (holds alternative<string>(m)) { cout $<<$ m.get<string>(); } else { int err $=$ m.get<int>(); // ... handle error ... }

This style appeals to some people who dislike exceptions (see $\S3.5.3)$ , but there are more interest- ing uses. For example, a simple compiler may need to distinguish between different kind of nodes with different representations:

using Node $=$ variant<Expression, Statement, Declaration, Type>; void check (Node∗p){ if (holds alternative<Expression> $({}^{*}{\mathfrak{p}})$ ) { Expression& $\mathbf{e}=$ get<Expression> $({}^{*}{\mathfrak{p}})$ ; // ... } else if (holds alternative<Statement> $({}^{*}{\mathfrak{p}})$ ) { Statement&$\mathbf{s}=$ get<Statement>$\boldsymbol{\cdot}({\ast}\mathfrak{p})$;// ... } // ... Declaration and Type ... }

This pattern of checking alternatives to decide on the appropriate action is so common and rela- tively inefficient that it deserves direct support:

void check (Node∗p){ visit (overloaded { [](Expression& e) $\{/^{\star}\dots{}^{\star}/\},$ [](Statement& s) $\{/^{\star}\dots{}^{\star}/\}$ , // ... Declaration and Type ... }, ∗ p); }

This is basically equivalent to a virtual function call, but potentially faster. As with all claims of performance, this ‘‘potentially faster’’ should be verified by measurements when performance is critical. For most uses, the difference in performance is insignificant. Unfortunately, the overloaded is necessary and not standard. It’s a ‘‘piece of magic’’ that builds an overload set from a set of arguments (usually lambdas): template<class... Ts> struct overloaded : Ts... { using Ts:: operator ()...; }; template<class... Ts> overloaded (Ts...) $->$ overloaded<Ts...>; // deduction guide The ‘‘visitor’’ visit then applies () to the overload , which selects the most appropriate lambda to call according to the overload rules. A deduction guide is a mechanism for resolving subtle ambiguities, primarily for constructors of class templates in foundation libraries (§6.2.3). If we try to access a variant holding a different type than the expected one, bad variant access is thrown.

# 13.5.2 optional

An optional<A> can be seen as a special kind of variant (like a variant<A, nothing> ) or as a generaliza- tion of the idea of an $\pmb{\mathsf{A}}^{*}$ either pointing to an object or being nullptr . An optional can be useful for functions that may or may not return an object: optional<string> compose_message (istream& s) { string mess; // ... read from s and compose message ... if (no_problems) return mess; return {}; // the empty optional }

Given that, we can write

if (auto $\mathsf{m}=$ compose_message (cin)) cout $<<\mathsf{\Omega}$ ; // note the dereference (\*) else { // ... handle error ... }

This appeals to some people who dislike exceptions (see $\S3.5.3)$ . Note the curious use of ∗ . An optional is treated as a pointer to its object rather than the object itself. The optional equivalent to nullptr is the empty object, $\{\}$ . For example: int cat (optional<int> a, optional<int> b) { int res ${\bf\Gamma}={\bf0}$ ; if (a) res+= ∗ a; if (b) res $\scriptstyle+=*\,\mathbf{b}$ ; return res; } int $\mathbf{x}=$ cat (17,19); int $\mathsf{y}=$ cat (17,{}); int ${\pmb z}={\mathsf{c a t}}(\{\},\!\{\})$ ;

If we try to access an optional that does not hold a value, the result is undefined; an exception is not thrown. Thus, optional is not guaranteed type safe.

# 13.5.3 any

An any can hold an arbitrary type and know which type (if any) it holds. It is basically an uncon- strained version of variant:

any compose_message (istream& s) { string mess; // ... read from s and compose message ... if (no_problems) return mess; // retur n a str ing else return error_number; // retur n an int }

When you assign or initialize an any with a value, it remembers the type of that value. Later, we can inquire what type the any holds and extract the value. For example:

auto $\mathfrak{m}=$ compose_message (cin)); string& $\mathbf{s}=$ any_cast<string>(m); cout $<<\mathfrak{s}$ ;

If we try to access an any holding a different type than the expected one, bad_any_access is thrown. There are also ways of accessing an any that do not rely on exceptions.

# 13.6 Allocators

By default, standard-library containers allocate space using new . Operators new and delete provide a general free store (also called dynamic memory or heap) that can hold objects of arbitrary size and user-controlled lifetime. This implies time and space overheads that can be eliminated in many special cases. Therefore, the standard-library containers offer the opportunity to install allocators with specific semantics where needed. This has been used to address a wide variety of concerns related to performance (e.g., pool allocators), security (allocators that clean-up memory as part of deletion), per-thread allocation, and non-uniform memory architectures (allocating in specific memories with pointer types to match). This is not the place to discuss these important, but very specialized and often advanced techniques. However, I will give one example motivated by a real- world problem for which a pool allocator was the solution.

An important, long-running system used an event queue (see $\S15.6)$ using vector s as events that were passed as shared_ptr s. That way, the last user of an event implicitly deleted it:

struct Event { vector<int> data $=$ vector<int>(512); }; list<shared_ptr<Event>> q; void producer () { for (int ${\boldsymbol{\mathsf{n}}}={\boldsymbol{\mathsf{0}}}$ ; n!=LOTS; ${\bf++}{\bf n}$ ) { lock_guard lk $\{{\mathfrak{m}}\}$ ; // m is a mutex (§15.5) q.push_back (make_shared<Event>()); cv. notify_one (); } }

From a logical point of view this worked nicely. It is logically simple, so the code is robust and maintainable. Unfortunately, this led to massive fragmentation. After 100,000 events had been passed among 16 producers and 4 consumers, more than 6GB memory had been consumed.

The traditional solution to fragmentation problems is to rewrite the code to use a pool allocator. A pool allocator is an allocator that manages objects of a single fixed size and allocates space for many objects at a time, rather than using individual allocations. Fortunately, $_{\mathrm{C++17}}$ offers direct support for that. The pool allocator is defined in the pmr (‘‘polymorphic memory resource’’) sub- namespace of std :

pmr:: synchroniz ed pool resource pool; // make a pool struct Event { vector<int> data $=$ vector<int>{512,&pool}; // let Events use the pool }; list<shared_ptr<Event>> q {&pool}; // let q use the pool

void producer () { for (int ${\boldsymbol{\mathsf{n}}}={\boldsymbol{\mathsf{0}}}$ ; n!=LOTS; ${\mathfrak{++}}{\mathfrak{n}}$ ) { scoped_lock lk {m}; // m is a mutex (§15.5) q.push_back (allocate_shared<Event, pmr:: polymorphic al locator<Event>>{&pool}); cv. notify_one (); } }

Now, after 100,000 events had been passed among 16 producers and 4 consumers, less than 3MB memory had been consumed. That’s about a 2000-fold improvement! Naturally, the amount of memory actually in use (as opposed to memory wasted to fragmentation) is unchanged. After elim- inating fragmentation, memory use was stable over time so the system could run for months.

Techniques like this have been applied with good effects from the earliest days of $\mathrm{C++}$ , but gen- erally they required code to be rewritten to use specialized containers. Now, the standard contain- ers optionally take allocator arguments. The default is for the containers to use new and delete .

# 13.7 Time

In <chrono> , the standard library provides facilities for dealing with time. For example, here is the basic way of timing something:

using namespace std:: chrono; // in sub-namespace std:: chrono; see §3.4

auto ${\bf t0}=$ high resolution clock:: now (); do_work (); auto t1 $=$ high resolution clock:: now (); cout $<<$ duration_cast<milliseconds>(t1−t0). count () $<<$ "msec\n";

The clock returns a time_point (a point in time). Subtracting two time_point s giv es a duration (a period of time). Various clocks give their results in various units of time (the clock I used measures nanoseconds ), so it is usually a good idea to convert a duration into a known unit. That’s what dura- tion_cast does. Don’t make statements about ‘‘efficiency’’ of code without first doing time measurements. Guesses about performance are most unreliable. To simplify notation and minimize errors, <chrono> offers time-unit suffixes (§5.4.4). For example:

this_thread:: sleep (10ms+33us); // wait for 10 milliseconds and 33 microseconds The chrono suffixes are defined in namespace std:: chrono_literals .

An elegant and efficient extension to <chrono> , supporting longer time intervals (e.g., years and months), calendars, and time zones, is being added to the standard for $_{\mathrm{C++20}}$ . It is currently avail- able and in wide production use [Hinnant, 2018] [Hinnant, 2018b]. You can say things like

auto spring_day $=$ apr/7/2018; cout $<<$ weekday (spring_day) $<<\mathfrak{w}$ '; // Saturday

It even handles leap seconds.

# 13.8 Function Adaption

When passing a function as a function argument, the type of the argument must exactly match the expectations expressed in the called function’s declaration. If the intended argument ‘‘almost matches expectations,’’ we hav e three good alternatives:

• Use a lambda (§13.8.1). • Use std:: mem_fn () to make a function object from a member function (§13.8.2). • Define the function to accept a std:: function (§13.8.3).

There are many other ways, but usually one of these three ways works best.

# 13.8.1 Lambdas as Adaptors

Consider the classical ‘‘draw all shapes’’ example:

void draw_all (vector<Shape ∗ >& v) { for_each (v.begin (),v.end (),[](Shape ∗ p) { p−>draw (); }); }

Like all standard-library algorithms, for_each () calls its argument using the traditional function call syntax ${\mathfrak{f}}({\mathsf{x}})$ , but Shape's draw () uses the conventional OO notation ${\tt x}{->}{\tt f}(0)$ . A lambda easily mediates between the two notations.

# 13.8.2 mem_fn ()

Given a member function, the function adaptor mem_fn (mf) produces a function object that can be called as a nonmember function. For example:

void draw_all (vector<Shape ∗ >& v) { for_each (v.begin (),v.end (), mem_fn (&Shape::draw)); }

Before the introduction of lambdas in $_{\mathrm{C++11}}$ , mem_fn () and equivalents were the main way to map from the object-oriented calling style to the functional one.

# 13.8.3 function

The standard-library function is a type that can hold any object you can invoke using the call opera- tor () . That is, an object of type function is a function object (§6.3.2). For example:

int f1 (double); function<int (double)> fct1 {f1}; // initialize to f1 int f2 (string); function fct2 {f2}; // fct2’s type is function<int (string)>

function fct3 $=$ [](Shape ∗ p) { p−>draw (); }; // fct3’s type is function<void (Shape\*)> For fct2 , I let the type of the function be deduced from the initializer: int (string) .

Obviously, function s are useful for callbacks, for passing operations as arguments, for passing function objects, etc. However, it may introduce some run-time overhead compared to direct calls, and a function , being an object, does not participate in overloading. If you need to overload func- tion objects (including lambdas), consider overloaded (§13.5.1).

# 13.9 Type Functions

A type function is a function that is evaluated at compile time given a type as its argument or returning a type. The standard library provides a variety of type functions to help library imple- menters (and programmers in general) to write code that takes advantage of aspects of the lan- guage, the standard library, and code in general.

For numerical types, numeric_limits from <limits> presents a variety of useful information (§14.7). For example:

constexpr ﬂoat min $=$ numeric_limits<ﬂoat>:: min (); // smallest positive ﬂoat Similarly, object sizes can be found by the built-in siz eof operator $(\S1.4)$ . For example: constexpr int szi $=$ sizeof (int); // the number of bytes in an int

Such type functions are part of $\mathbf{C++}$ ’s mechanisms for compile-time computation that allow tighter type checking and better performance than would otherwise have been possible. Use of such fea- tures is often called metaprogramming or (when templates are involved) template metaprogram- ming . Here, I just present the use of two facilities provided by the standard library: iterator_traits $(\S13.9.1)$ and type predicates (§13.9.2). Concepts $(\S7.2)$ make some of these techniques redundant and simplify many of the rest, but concepts are still not standard or universally available, so the techniques presented here are in wide use.

# 13.9.1 iterator_traits

The standard-library sor t () takes a pair of iterators supposed to define a sequence (Chapter 12). Furthermore, those iterators must offer random access to that sequence, that is, they must be ran- dom-access iterators . Some containers, such as forward_list , do not offer that. In particular, a for- ward_list is a singly-linked list so subscripting would be expensive and there is no reasonable way to refer back to a previous element. However, like most containers, forward_list offers forward iter- ators that can be used to traverse the sequence by algorithms and for -statements $(\S6.2)$ .

The standard library provides a mechanism, iterator_traits , that allows us to check which kind of iterator is provided. Given that, we can improve the range sor t () from $\S12.8$ to accept either a vector or a forward_list . For example:

void test (vector<string>& v, forward_list<int>& lst) { sor t (v); // sor t the vector sor t (lst); // sor t the singly-linked list }

The techniques needed to make that work are generally useful.

First, I write two helper functions that take an extra argument indicating whether they are to be used for random-access iterators or forward iterators. The version taking random-access iterator arguments is trivial:

template<typename Ran> // for random-access iterators void sort_helper (Ran beg, Ran end, random_access_iterator_tag) // we can subscript into [beg:end) { sor t (beg, end); // just sort it }

The version for forward iterators simply copies the list into a vector , sorts, and copies back:

template<typename For> // for forward iterators void sort_helper (For beg, For end, forward_iterator_tag) // we can traverse [beg:end) { vector<Value_type<For>> v {beg, end}; // initialize a vector from [beg:end) sor t (v.begin (),v.end ()); // use the random access sort copy (v.begin (),v.end (), beg); // copy the elements back }

Value_type<For> is the type of For ’s elements, called it’s value type . Every standard-library iterator has a member value_type . I get the Value_type<For> notation by defining a type alias (§6.4.2):

template<typename C> using Value_type $=$ typename C:: value_type; // C’s value type Thus, for a vector<X> , Value_type $\tt<\!\!\mathbf{X}\!\!>$ is X . The real ‘‘type magic’’ is in the selection of helper functions: template<typename C> void sort (C& c){ using Iter $=$ Iterator_type<C>; sor t_helper (c.begin (),c.end (), Iterator category<Iter>{}); }

Here, I use two type functions: Iterator_type $\tt<\tt C>$ returns the iterator type of C (that is, C:: iterator ) and then Iterator_categor y<Iter> $\cdot\}$ constructs a ‘‘tag’’ value indicating the kind of iterator provided:

• std:: random access iterator tag if C ’s iterator supports random access • std:: forward_iterator_tag if C ’s iterator supports forward iteration Given that, we can select between the two sorting algorithms at compile time. This technique, called tag dispatch , is one of several used in the standard library and elsewhere to improve ﬂexibil- ity and performance. We could define Iterator_type like this:

template<typename C> using Iterator_type $=$ typename C:: iterator; // C’s iterator type

However, to extend this idea to types without member types, such as pointers, the standard-library support for tag dispatch comes in the form of a class template iterator_traits from <iterator> . The specialization for pointers looks like this:

template<class T> struct iterator_traits $\begin{array}{r l}{\rvert}&{{}<\!\mathsf{T}^{*}\!\!>\!\left\{\begin{array}{l l}\end{array}\right.}\end{array}$ using difference_type $=$ ptrdiff_t; using value_type $={\mathsf{T}}$ ; using pointer $=\mathsf{T}^{*}$ ; using reference $=$ T&; using iterator category $=$ random access iterator tag; };

We can now write:

template<typename Iter>

Now an int ∗ can be used as a random-access iterator despite not having a member type; Iterator_cat- egor y<int $^{\ast>}$ is random access iterator tag . Many traits and traits-based techniques will be made redundant by concepts (§7.2). Consider the concepts version of the sor t () example: template<Random Access Iterator Iter> void sort (Iter p, Iter q); // use for std:: vector and other types supporting random access template<ForwardIterator Iter> void sort (Iter p, Iter q) // use for std:: list and other types supporting just forward traversal { vector<Value_type<Iter>> v {p, q}; sor t (v); // use the random-access sort copy (v.begin (),v.end (), p); } template<Rang e R> void sort (R& r){ sor t (r.begin (),r.end ()); // use the appropriate sort }

Progress happens.

# 13.9.2 Type Predicates

In <type_traits> , the standard library offers simple type functions, called type predicates that answers a fundamental question about types. For example:

bool b1 $=$ std::is_arithmetic<int>(); // yes, int is an arithmetic type bool b2 $=$ std::is_arithmetic<string>(); // no, std:: str ing is not an arithmetic type

Other examples are is_class , is_pod , is_literal_type , has_vir tual_destructor , and is_base_of . They are most useful when we write templates. For example:

template<typename Scalar> class complex { Scalar re, im; public: static_asser t (is_arithmetic<Scalar>(), "Sorr y, I only suppor t complex of arithmetic types"); // ... };

To improve readability, the standard library defines template aliases. For example:

template<typename T> constexpr bool is_arithmetic_v $=$ std::is_arithmetic<T>();

I’m no great fan of the $\_{\lor}$ suffix notation, but the technique for defining aliases is immensely useful. For example, the standard library defines the concept Regular (§12.7) like this:

template<class T> concept Regular $=$ Semiregular<T> && EqualityComparable<T>;

# 13.9.3 enable_if

Obvious ways of using type predicates includes conditions for static_asser s, compile-time if s, and enable_if s. The standard-library enable_if is a widely used mechanism for conditonally introducing definitions. Consider defining a ‘‘smart pointer’’:

template<typename T> class Smart_pointer { // ... T& operator $^{*}\! ()$ ; T& operator− ${\cdot>}0$ ; $/\!/\!-\!>$ should wor k if and only if T is a class };

The $->$ should be defined if and only if T is a class type. For example, Smar t_pointer<vector<T>> should have $->$ , but Smar t_pointer<int> should not. We cannot use a compile-time if because we are not inside a function. Instead, we write

template<typename T> class Smart_pointer { // ... T& operator ∗ (); std::enable_if<is_class $\scriptstyle<\!\!\mathsf{T}\!>\!\! (),\mathsf{T}\!\&>$ operator− ${\cdot}{>}0$ ; // -> is defined if and only if T is a class };

If is_class $\mathord{<}\overline{{\Pi}}\mathord{>}\!\left (\begin{array}{l}{0}\\ {\overline{{\Pi}}\mathord{>}\!\left (\begin{array}{l}{0}\\ {\overline{{\Pi}}\mathord{>}\!\left (\begin{array}{l}{0}\\ {\overline{{\Pi}}\mathord{>}\!\left (\begin{array}{l}{0}\\ {\overline{{\Pi}}\mathord{>}\!\left (\begin{array}{l}{0}\\ {\overline{{\Pi}}\mathord{>}\!\left (\begin{array}{l}{0}\\ {0}\end{array}\right)}\end{array}\right)}}}\end{array}\right)$ is true , the return type of operator− ${\cdot}{>}0$ is T& ; otherwise, the definition of operator− ${\cdot}{>}0$ is ignored.

The syntax of enable_if is odd, awkward to use, and will in many cases be rendered redundant by concepts (§7.2). However, enable_if is the basis for much current template metaprogramming and for many standard-library components. It relies on a subtle language feature called SFINAE (‘‘Substitution Failure Is Not An Error’’).

# 13.10 Advice

[1] A library doesn’t hav e to be large or complicated to be useful; $\S13.1$ .

[2] A resource is anything that has to be acquired and (explicitly or implicitly) released; $\S13.2$ .

[3] Use resource handles to manage resources (RAII); $\S13.2$ ; [CG: R.1].

[4] Use unique_ptr to refer to objects of polymorphic type; $\S13.2.1$ ; [CG: R.20].

[5] Use shared_ptr to refer to shared objects (only); $\S13.2.1$ ; [CG: R.20].

[6] Prefer resource handles with specific semantics to smart pointers; $\S13.2.1$ .

[7] Prefer unique_ptr to shared_ptr ; $\S5.3$ , $\S13.2.1$ .

[8] Use make_unique () to construct unique_ptr s; $\S13.2.1$ ; [CG: R.22].

[9] Use make_shared () to construct shared_ptr s; $\S13.2.1$ ; [CG: R.23].

[10] Prefer smart pointers to garbage collection; $\S5.3,\S13.2.1$ .

[11] Don’t use std:: move (); $\S13.2.2$ ; [CG: ES. 56].

[12] Use std:: forward () exclusively for forwarding; $\S13.2.2$ .

[13] Never read from an object after std:: move () ing or std:: forward () ing it; $\S13.2.2$ .

[14] Prefer span s to pointer-plus-count interfaces; $\S13.3$ ; [CG: F.24].

[15] Use array where you need a sequence with a constexpr size; $\S13.4.1$ .

[16] Prefer array over built-in arrays; $\S13.4.1$ ; [CG: SL. con. 2].

[17] Use bitset if you need N bits and N is not necessarily the number of bits in a built-in integer type; $\S13.4.2$ .

[18] Don’t overuse pair and tuple ; named struct s often lead to more readable code; $\S13.4.3$ .

[19] When using pair , use template argument deduction or make_pair () to avoid redundant type specification; $\S13.4.3$.

[20] When using tuple , use template argument deduction and make_tuple () to avoid redundant type specification; $\S13.4.3$ ; [CG: T.44].

[21] Prefer variant to explicit use of union s; $\S13.5.1$ ; [CG: C.181].

[22] Use allocators to prevent memory fragmentation; $\S13.6$ .

[23] Time your programs before making claims about efficiency; $\S13.7$ .

[24] Use duration_cast to report time measurements with proper units; $\S13.7$ .

[25] When specifying a duration , use proper units; $\S13.7$ .

[26] Use mem_fn () or a lambda to create function objects that can invoke a member function when called using the traditional function call notation; $\S13.8.2$ .

[27] Use function when you need to store something that can be called; $\S13.8.3$ .

[28] You can write code to explicitly depend on properties of types; $\S13.9$ .

[29] Prefer concepts over traits and enable_if whenever you can; $\S13.9$ .

[30] Use aliases and type predicates to simplify notation; $\S13.9.1$ , $\S13.9.2$ .

# This page intentionally left blank

# Numerics

The purpose of computing is insight, not numbers. – R. W. Hamming

... but for the student, numbers are often the best road to insight. – A. Ralston

• Introduction • Mathematical Functions • Numerical Algorithms Parallel Numerical Algorithms • Complex Numbers • Random Numbers • Vector Arithmetic • Numeric Limits • Advice

# 14.1 Introduction

$\mathrm{C++}$ was not designed primarily with numeric computation in mind. However, numeric computa- tion typically occurs in the context of other work – such as database access, networking, instrument control, graphics, simulation, and financial analysis – so $\mathrm{C++}$ becomes an attractive vehicle for computations that are part of a larger system. Furthermore, numeric methods have come a long way from being simple loops over vectors of ﬂoating-point numbers. Where more complex data structures are needed as part of a computation, $C++^{\prime}\mathrm{s}$ strengths become relevant. The net effect is that $\mathrm{C++}$ is widely used for scientific, engineering, financial, and other computation involving sophisticated numerics. Consequently, facilities and techniques supporting such computation have emerged. This chapter describes the parts of the standard library that support numerics.

# 14.2 Mathematical Functions

In <cmath> , we find the standard mathematical functions , such as sqr t () , log () , and sin () for argu- ments of type ﬂoat , double , and long double :

![](images/f62872fabd061701cbcdb13dff2b42bcb24e32fe87b3328dace82d94f42cf344.jpg)

The versions for complex (§14.4) are found in <complex> . For each function, the return type is the same as the argument type.

Errors are reported by setting errno from <cerrno> to EDOM for a domain error and to ERANGE for a range error. For example:

void f (){ errno ${\bf\Gamma}={\bf0}$ ; // clear old error state sqr t (−1); if (errno $==$ EDOM) cerr $<<$ "sqrt () not defined for negative argument"; errno ${\bf\Gamma}={\bf0}$ ; // clear old error state pow (numeric_limits<double>:: max (), 2); if (errno $==$ ERANGE) cerr $<<$ "result of pow () too large to represent as a double"; }

A few more mathematical functions are found in <cstdlib> and the so-called special mathematical functions , such as beta () , rieman_z eta () , and sph_bessel () , are also in <cmath> .

# 14.3 Numerical Algorithms

In <numeric> , we find a small set of generalized numerical algorithms, such as accumulate () .

![](images/50e0635aaa80bb3cd18e5725b7b3c8dbd21d9b7d0be900ae59814b727a66846d.jpg)

These algorithms generalize common operations such as computing a sum by letting them apply to all kinds of sequences. They also make the operation applied to elements of those sequences a parameter. For each algorithm, the general version is supplemented by a version applying the most common operator for that algorithm. For example:

list<double> lst {1, 2, 3, 4, 5, 9999.99999}; auto $\mathbf{s}=$ accumulate (lst.begin (), lst.end (), 0.0); // calculate the sum: 10014.9999

These algorithms work for every standard-library sequence and can have operations supplied as arguments (§14.3).

# 14.3.1 Parallel Algorithms

In <numeric> , the numerical algorithms have parallel versions $(\S12.9)$ that are slightly different:

![](images/73e66beb37a72698eac546126d50d26d45196e7aa6fb61e3806c0efe82ef5bc5.jpg)

![](images/24e5be0622716343769ca8bbaedd951130a21184104302a596134cbd8deb0297.jpg)

For simplicity, I left out the versions of these algorithms that take functor arguments, rather than just using $^+$ and $=$ . Except for reduce () , I also left out the versions with default policy (sequential) and default value. Just as for the parallel algorithms in <algorithm $\mid>$ (§12.9), we can specify an execution policy: vector<double> v {1, 2, 3, 4, 5, 9999.99999}; auto $\mathbf{s}=$ reduce (v.begin (),v.end ()); // calculate the sum using a double as the accumulator vector<double> large; // ... fill large with lots of values ... auto ${\mathfrak{s z}}=$ reduce (par_unseq, large .begin (), large .end ()); // calculate the sum using available parallelism

The parallel algorithms (e.g., reduce () ) differ from the sequentional ones (e.g., accumulate () ) by allowing operations on elements in unspecified order.

# 14.4 Complex Numbers

The standard library supports a family of complex number types along the lines of the complex class described in $\S4.2.1$ . To support complex numbers where the scalars are single-precision ﬂoat- ing-point numbers ( ﬂoat s), double-precision ﬂoating-point numbers ( double s), etc., the standard library complex is a template: template<typename Scalar> class complex { public: complex (const Scalar& re $\scriptstyle=\left\{\begin{array}{l l}{\begin{array}{r l}\end{array}}\end{array}\right.$ , const Scalar& im $\mathbf{\beta=}\mathbf{\beta}$ ); // default function arguments; see §3.6.1 // ... };

The usual arithmetic operations and the most common mathematical functions are supported for complex numbers. For example:

void f (complex<ﬂoat> ﬂ, complex<double> db) { complex<long double> ld {ﬂ+sqrt (db)}; db $\scriptstyle+=\mathbf{f}\mathbf{f}*\mathbf{3}$ ; ﬂ $=$ pow (1/ﬂ, 2); // ... }

The sqr t () and pow () (exponentiation) functions are among the usual mathematical functions defined in <complex> (§14.2).

# Section 14.5

# 14.5 Random Numbers

Random numbers are useful in many contexts, such as testing, games, simulation, and security. The diversity of application areas is reﬂected in the wide selection of random number generators provided by the standard library in <random> . A random number generator consists of two parts:

[1] An engine that produces a sequence of random or pseudo-random values [2] A distribution that maps those values into a mathematical distribution in a range Examples of distributions are uniform in t distribution (where all integers produced are equally likely), normal distribution (‘‘the bell curve’’), and exponential distribution (exponential growth); each for some specified range. For example:

using my_engine $=$ default random engine; // type of engine using my_distribution $=$ uniform in t distribution<>; // type of distribution

my_engine re $\{\}$ ; my_distribution one_to_six {1,6}; auto die $=\left[\begin{array}{l l}\end{array}\right]\langle\ell\rangle$ return one_to_six (re); }

int $\begin{array}{r}{\pmb{x}=\mathsf{d i e}();}\end{array}$ ;

// the default engine

// distr ibution that maps to the ints 1.. 6

// make a generator

// roll the die: x becomes a value in [1:6] Thanks to its uncompromising attention to generality and performance, one expert has deemed the standard-library random number component ‘‘what every random number library wants to be when it grows up.’’ Howev er, it can hardly be deemed ‘‘novice friendly.’’ The using statements and the lambda make what is being done a bit more obvious.

For novices (of any background) the fully general interface to the random number library can be a serious obstacle. A simple uniform random number generator is often sufficient to get started. For example:

Rand_int rnd {1,10}; // make a random number generator for [1:10] int ${\pmb x}={\mathsf{r n d}}()$ ; // x is a number in [1:10]

So, how could we get that? We hav e to get something that, like die () , combines an engine with a distribution inside a class Rand_int :

class Rand_int { public: Rand_int (int low, int high) : dist{low, high} { } int operator ()() { return dist (re); } // draw an int void seed (int s) { re.seed (s); } // choose new random engine seed private: default random engine re; uniform in t distribution<> dist; };

That definition is still ‘‘expert level,’’ but the use of Rand_int () is manageable in the first week of a $\mathrm{C++}$ course for novices. For example:

# int main ()

{ constexpr int max ${\boldsymbol{\mathbf{\mathit{\sigma}}}}={\boldsymbol{\mathbf{\mathit{\Theta}}}}{\boldsymbol{\mathbf{\mathit{\sigma}}}}$ ; Rand_int rnd {0, max}; // make a unifor m random number generator vector<int> histogram (max+1); // make a vector of appropriate size for (int i=0; i!=200; ${\bf++i}$ ) $^{++}$ histogram[rnd ()]; // fill histogram with the frequencies of numbers [0: max] for (int $\mathbf{i}=\mathbf{0}$ ; i!=histogram.size (); ${\bf++i}$ ) { // wr ite out a bar graph cout $<<\mathsf{i}<<\mathsf{i}$ ; for (int $\mathsf{j}\mathsf{=}\mathsf{0}$ ; j!=histogram[i]; ${\bf++j}$ ) cout $<<"$ ; cout $<<$ endl; } }

The output is a (reassuringly boring) uniform distribution (with reasonable statistical variation):

0 ∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗ 1 ∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗ 2 ∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗ 3 ∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗ 4 ∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗ 5 ∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗ 6 ∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗ 7 ∗∗∗∗∗∗∗∗∗∗∗ 8 ∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗ 9 ∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗

There is no standard graphics library for $\mathrm{C++}$ , so I use ‘‘ASCII graphics.’’ Obviously, there are lots of open source and commercial graphics and GUI libraries for $\mathrm{C++}$ , but in this book I restrict myself to ISO standard facilities.

# 14.6 Vector Arithmetic

The vector described in $\S11.2$ was designed to be a general mechanism for holding values, to be ﬂexible, and to fit into the architecture of containers, iterators, and algorithms. However, it does not support mathematical vector operations. Adding such operations to vector would be easy, but its generality and ﬂexibility preclude optimizations that are often considered essential for serious numerical work. Consequently, the standard library provides (in <valarray> ) a vector -like template, called valarray , that is less general and more amenable to optimization for numerical computation:

template<typename T> class valarray { // ... };

The usual arithmetic operations and the most common mathematical functions are supported for valarray s. For example:

void f (valarray<double>& a1, valarray<double>& a2) { valarray<double> a $=$ a1 ∗ 3.14+a2/a1; // numer ic array operators $^{*}\!,\,+,\,/,\, a n d=$ $\mathsf{a2}+=\mathsf{a1}*\mathsf{3}. 14$ ; a $=$ abs (a); double ${\mathsf{d}}=$ a2[7]; // ... }

In addition to arithmetic operations, valarray offers stride access to help implement multidimen- sional computations.

# 14.7 Numeric Limits

In <limits> , the standard library provides classes that describe the properties of built-in types – such as the maximum exponent of a ﬂoat or the number of bytes in an int . For example, we can assert that a char is signed:

static_asser t (numeric_limits<char>:: is_signed,"unsigned characters!"); static_asser t (100000<numeric_limits<int>:: max (),"small ints!");

Note that the second assert (only) works because numeric_limits<int>:: max () is a constexpr function

(§1.6).

# 14.8 Advice

[1] Numerical problems are often subtle. If you are not $100\%$ certain about the mathematical aspects of a numerical problem, either take expert advice, experiment, or do both; $\S14.1$ .

[2] Don’t try to do serious numeric computation using only the bare language; use libraries; $\S14.1$ .

[3] Consider accumulate () , inner_product () , par tial_sum () , and adjacent difference () before you write a loop to compute a value from a sequence; $\S14.3$ .

[4] Use std:: complex for complex arithmetic; $\S14.4$ .

[5] Bind an engine to a distribution to get a random number generator; $\S14.5$ .

[6] Be careful that your random numbers are sufficiently random; $\S14.5$ .

[7] Don’t use the C standard-library rand () ; it isn’t insufficiently random for real uses; $\S14.5$ .

[8] Use valarray for numeric computation when run-time efficiency is more important than ﬂexi- bility with respect to operations and element types; $\S14.6$ .

[9] Properties of numeric types are accessible through numeric_limits ; $\S14.7$ .

[10] Use numeric_limits to check that the numeric types are adequate for their use; $\S14.7$ .

# This page intentionally left blank

# Concurrency

Keep it simple: as simple as possible, but no simpler. – A. Einstein

• Introduction • Tasks and thread s • Passing Arguments • Returning Results • Sharing Data • Waiting for Events • Communicating Tasks future and promise ; packaged_task ; async () • Advice

# 15.1 Introduction

Concurrency – the execution of several tasks simultaneously – is widely used to improve through- put (by using several processors for a single computation) or to improve responsiveness (by allow- ing one part of a program to progress while another is waiting for a response). All modern pro- gramming languages provide support for this. The support provided by the $\mathrm{C++}$ standard library is a portable and type-safe variant of what has been used in $\mathrm{C++}$ for more than 20 years and is almost universally supported by modern hardware. The standard-library support is primarily aimed at sup- porting systems-level concurrency rather than directly providing sophisticated higher-level concur- rency models; those can be supplied as libraries built using the standard-library facilities.

The standard library directly supports concurrent execution of multiple threads in a single address space. To allow that, $\mathrm{C++}$ provides a suitable memory model and a set of atomic opera- tions. The atomic operations allow lock-free programming [Dechev, 2010]. The memory model ensures that as long as a programmer avoids data races (uncontrolled concurrent access to mutable data), everything works as one would naively expect. However, most users will see concurrency only in terms of the standard library and libraries built on top of that. This section brieﬂy gives examples of the main standard-library concurrency support facilities: thread s, mutex es, lock () opera- tions, packaged_task s, and future s. These features are built directly upon what operating systems offer and do not incur performance penalties compared with those. Neither do they guarantee sig- nificant performance improvements compared to what the operating system offers.

Do not consider concurrency a panacea. If a task can be done sequentially, it is often simpler and faster to do so. As an alternative to using explicit concurrency features, we can often use a parallel algorithm to exploit multiple execution engines for better performance $(\S12.9,\S14.3.1)$ .

# 15.2 Tasks and thread s

We call a computation that can potentially be executed concurrently with other computations a task . A thread is the system-level representation of a task in a program. A task to be executed concur- rently with other tasks is launched by constructing a std:: thread (found in <thread> ) with the task as its argument. A task is a function or a function object:

void f (); // function struct F { // function object void operator ()(); // F’s call operator (§6.3.2) }; void user () { thread t1 {f}; // f () executes in separate thread thread t2 {F ()}; // F ()() executes in separate thread t1.join (); // wait for t1 t2.join (); // wait for t2 }

The join () s ensure that we don’t exit user () until the threads have completed. To ‘‘join’’ a thread means to ‘‘wait for the thread to terminate.’’

Threads of a program share a single address space. In this, threads differ from processes, which generally do not directly share data. Since threads share an address space, they can communicate through shared objects (§15.5). Such communication is typically controlled by locks or other mechanisms to prevent data races (uncontrolled concurrent access to a variable).

Programming concurrent tasks can be very tricky. Consider possible implementations of the tasks f (a function) and F (a function object):

void f (){ cout $<<$ "Hello "; }

struct F { void operator ()() { cout $<<$ "Parallel World!\n"; } };

This is an example of a bad error: here, f and F () each use the object cout without any form of syn- chronization. The resulting output would be unpredictable and could vary between different execu- tions of the program because the order of execution of the individual operations in the two tasks is not defined. The program may produce ‘‘odd’’ output, such as

PaHerallllel o World!

Only a specific guarantee in the standard saves us from a data race within the definition of ostream that could lead to a crash.

When defining tasks of a concurrent program, our aim is to keep tasks completely separate except where they communicate in simple and obvious ways. The simplest way of thinking of a concurrent task is as a function that happens to run concurrently with its caller. For that to work, we just have to pass arguments, get a result back, and make sure that there is no use of shared data in between (no data races).

# 15.3 Passing Arguments

Typically, a task needs data to work upon. We can easily pass data (or pointers or references to the data) as arguments. Consider:

struct F { // function object: do something with v vector<double>& v; F (vector<double>& vv) : v{vv} { } void operator ()(); // application operator ; §6.3.2 }; int main () { vector<double> some_vec {1,2,3,4,5,6,7,8,9}; vector<double> vec2 {10,11,12,13,14}; thread t1 {f,ref (some_vec)}; // f (some_vec) executes in a separate thread thread t2 {F{vec2}}; // F (vec2)() executes in a separate thread t1.join (); t2.join (); }

Obviously, F{vec2} saves a reference to the argument vector in F . F can now use that vector and hopefully no other task accesses vec2 while F is executing. Passing vec2 by value would eliminate that risk.

The initialization with {f,ref (some_vec)} uses a thread variadic template constructor that can accept an arbitrary sequence of arguments $(\S7.4)$ . The ref () is a type function from <functional> that unfortunately is needed to tell the variadic template to treat some_vec as a reference, rather than as an object. Without that ref () , some_vec would be passed by value. The compiler checks that the first argument can be invoked giv en the following arguments and builds the necessary function object to pass to the thread. Thus, if F:: operator ()() and f () perform the same algorithm, the handling of the two tasks are roughly equivalent: in both cases, a function object is constructed for the thread to execute.

# 15.4 Returning Results

In the example in $\S15.3$ , I pass the arguments by non- const reference. I only do that if I expect the task to modify the value of the data referred to (§1.7). That’s a somewhat sneaky, but not uncom- mon, way of returning a result. A less obscure technique is to pass the input data by const refer- ence and to pass the location of a place to deposit the result as a separate argument:

void f (const vector<double>& v, double ∗ res); // take input from v; place result in \*res

class F { public: F (const vector<double>& vv, double ∗ p) : v{vv}, res{p} { } void operator ()(); // place result in \*res private: const vector<double>& v; // source of input double ∗ res; // target for output

double g (const vector<double>&); // use return value void user (vector<double>& vec1, vector<double> vec2, vector<double> vec3) { double res1; double res2; double res3; thread t1 {f,cref (vec1),&res1}; // f (vec1,&res1) executes in a separate thread thread t2 {F{vec2,&res2}}; // F{vec2,&res2}() executes in a separate thread thread t3 { [&](){ res3 $=$ g (vec3); } }; // capture local var iables by reference t1.join (); t2.join (); t3.join (); cout $<<$ res1 << ' ' << res2 << ' ' << res3 << '\n'; }

This works and the technique is very common, but I don’t consider returning results through refer- ences particularly elegant, so I return to this topic in $\S15.7.1$ .

# Section 15.5

# 15.5 Sharing Data

Sometimes tasks need to share data. In that case, the access has to be synchronized so that at most one task at a time has access. Experienced programmers will recognize this as a simplification (e.g., there is no problem with many tasks simultaneously reading immutable data), but consider how to ensure that at most one task at a time has access to a given set of objects.

The fundamental element of the solution is a mutex , a ‘‘mutual exclusion object.’’ A thread acquires a mutex using a lock () operation:

mutex m; // controlling mutex int sh; // shared data

void f ()

{ scoped_lock lck {m}; // acquire mutex sh $\mathrel{+{=}}7$ ; // manipulate shared data } // release mutex implicitly

The type of lck is deduced to be scoped_lock<mutex> (§6.2.3). The scoped_lock ’s constructor acquires the mutex (through a call m.lock () ). If another thread has already acquired the mutex, the thread waits (‘‘blocks’’) until the other thread completes its access. Once a thread has completed its access to the shared data, the scoped_lock releases the mutex (with a call m.unlock () ). When a mutex is released, thread s waiting for it resume executing (‘‘are woken up’’). The mutual exclusion and locking facilities are found in <mutex> .

Note the use of RAII (§5.3). Use of resource handles, such as scoped_lock and unique_lock (§15.6), is simpler and far safer than explicitly locking and unlocking mutex es.

The correspondence between the shared data and a mutex is conventional: the programmer sim- ply has to know which mutex is supposed to correspond to which data. Obviously, this is error- prone, and equally obviously we try to make the correspondence clear through various language means. For example:

class Record { public: mutex rm; // ... };

It doesn’t take a genius to guess that for a Record called rec , you are supposed to acquire rec. rm before accessing the rest of rec , though a comment or a better name might have helped the reader.

It is not uncommon to need to simultaneously access several resources to perform some action. This can lead to deadlock. For example, if thread1 acquires mutex1 and then tries to acquire mutex2 while thread2 acquires mutex2 and then tries to acquire mutex1 , then neither task will ever proceed further. The scoped_lock helps by enabling us to acquire several locks simultaneously:

void f (){ scoped_lock lck {mutex1, mutex2, mutex3}; // acquire all three locks // ... manipulate shared data ... } // implicitly release all mutexes

This scoped_lock will proceed only after acquiring all its mutex es arguments and will never block (‘‘go to sleep’’) while holding a mutex . The destructor for scoped_lock ensures that the mutex es are released when a thread leaves the scope.

Communicating through shared data is pretty low lev el. In particular, the programmer has to devise ways of knowing what work has and has not been done by various tasks. In that regard, use of shared data is inferior to the notion of call and return. On the other hand, some people are con- vinced that sharing must be more efficient than copying arguments and returns. That can indeed be so when large amounts of data are involved, but locking and unlocking are relatively expensive operations. On the other hand, modern machines are very good at copying data, especially compact data, such as vector elements. So don’t choose shared data for communication because of ‘‘effi- ciency’’ without thought and preferably not without measurement.

The basic mutex allows one thread at a time to access data. One of the most common ways of sharing data is among many readers and a single writer. This ‘‘reader-writer lock’’ idiom is sup- ported be shared_mutex . A reader will acquire the mutex ‘‘shared’’ so that other readers can still gain access, whereas a writer will demand exclusive access. For example:

void reader (){ shared_lock lck {mx}; // willing to share access with other readers // ... read ... } void writer () { unique_lock lck {mx}; // needs exclusive (unique) access // ... write ... }

# 15.6 Waiting for Events

Sometimes, a thread needs to wait for some kind of external event, such as another thread complet- ing a task or a certain amount of time having passed. The simplest ‘‘event’’ is simply time passing. Using the time facilities found in <chrono> I can write:

using namespace std:: chrono; // see §13.7

auto $\mathbf{\deltat0}=$ high resolution clock:: now (); this_thread:: sleep_for (milliseconds{20}); auto t1 $=$ high resolution clock:: now ();

cout $<<$ duration_cast<nanoseconds>(t1−t0). count () << " nanoseconds passed\n";

Note that I didn’t even hav e to launch a thread ; by default, this_thread refers to the one and only thread.

I used duration_cast to adjust the clock’s units to the nanoseconds I wanted. The basic support for communicating using external events is provided by condition variable s found in <condition variable> . A condition variable is a mechanism allowing one thread to wait for another. In particular, it allows a thread to wait for some condition (often called an event ) to occur as the result of work done by other thread s.

Using condition variable s supports many forms of elegant and efficient sharing but can be rather tricky. Consider the classical example of two thread s communicating by passing messages through a queue . For simplicity, I declare the queue and the mechanism for avoiding race conditions on that queue global to the producer and consumer:

class Message { // object to be communicated // ...

queue<Message> mqueue; // the queue of messages condition variable mcond; // the var iable communicating events mutex mmutex; // for synchronizing access to mcond

The types queue , condition variable , and mutex are provided by the standard library. The consumer () reads and processes Message s:

void consumer ()

{ while (true) { unique_lock lck {mmutex}; // acquire mmutex mcond.wait (lck,[] { return !mqueue.empty (); }); // release lck and wait; // re-acquire lck upon wakeup // don’t wake up unless mqueue is non-empty auto $\mathsf{m}=$ mqueue.front (); // get the message mqueue .pop (); lck.unlock (); // release lck // ... process m ... } }

Here, I explicitly protect the operations on the queue and on the condition variable with a unique_lock on the mutex . Waiting on condition variable releases its lock argument until the wait is over (so that the queue is non-empty) and then reacquires it. The explicit check of the condition, here !mqueue .empty () , protects against waking up just to find that some other task has ‘‘gotten there first’’ so that the condition no longer holds.

I used a unique_lock rather than a scoped_lock for two reasons: • We need to pass the lock to the condition variable ’s wait () . A scoped_lock cannot be copied, but a unique_lock can be. • We want to unlock the mutex protecting the condition variable before processing the mes- sage. A unique_lock offers operations, such as lock () and unlock () , for low-level control of synchronization. On the other hand, unique_lock can only handle a single mutex .

The corresponding producer looks like this:

void producer () { while (true) { Message m; // ... fill the message ... scoped_lock lck {mmutex}; // protect operations mqueue .push (m); mcond. notify_one (); // notify } // release lock (at end of scope) }

# 15.7 Communicating Tasks

The standard library provides a few facilities to allow programmers to operate at the conceptual level of tasks (work to potentially be done concurrently) rather than directly at the lower level of threads and locks:

• future and promise for returning a value from a task spawned on a separate thread • packaged_task to help launch tasks and connect up the mechanisms for returning a result • async () for launching of a task in a manner very similar to calling a function

These facilities are found in <future> .

# 15.7.1 future and promise

The important point about future and promise is that they enable a transfer of a value between two tasks without explicit use of a lock; ‘‘the system’’ implements the transfer efficiently. The basic idea is simple: when a task wants to pass a value to another, it puts the value into a promise . Some- how, the implementation makes that value appear in the corresponding future , from which it can be read (typically by the launcher of the task). We can represent this graphically:

![](images/bcb4e5c4456a8349326ce310ee4fc80540b9927eaeb38f0b8bbc02382b9c3ad9.jpg)

If we have a future<X> called fx , we can get () a value of type X from it: X v = fx. g et (); // if necessary, wait for the value to get computed

If the value isn’t there yet, our thread is blocked until it arrives. If the value couldn’t be computed, get () might throw an exception (from the system or transmitted from the task from which we were trying to get () the value).

The main purpose of a promise is to provide simple ‘‘put’’ operations (called set_value () and set_exception () ) to match future ’s get () . The names ‘‘future’’ and ‘‘promise’’ are historical; please don’t blame or credit me. They are yet another fertile source of puns.

If you have a promise and need to send a result of type X to a future , you can do one of two things: pass a value or pass an exception. For example:

void f (promise<X>& px) // a task: place the result in px

{ // ... tr y { X res; // ... compute a value for res ... px. set_value (res); } catch (...) { // oops: couldn’t compute res px. set_exception (current exception ()); // pass the exception to the future’s thread } }

The current exception () refers to the caught exception.

To deal with an exception transmitted through a future , the caller of get () must be prepared to catch it somewhere. For example:

void g (future<X>& fx) // a task: get the result from fx

{ // ... tr y { X $\mathbf{v}=$ fx. g et (); // if necessary, wait for the value to get computed // ... use v ... } catch (...) { // oops: someone couldn’t compute v // ... handle error ... } }

If the error doesn’t need to be handled by g () itself, the code reduces to the minimal:

void g (future<X>& fx) // a task: get the result from fx { // ... X v = fx. g et (); // if necessary, wait for the value to get computed // ... use v ... }

# 15.7.2 packaged_task

How do we get a future into the task that needs a result and the corresponding promise into the thread that should produce that result? The packaged_task type is provided to simplify setting up tasks connected with future s and promise s to be run on thread s. A packaged_task provides wrapper code to put the return value or exception from the task into a promise (like the code shown in $\S15.7.1)$ ). If you ask it by calling get_future , a packaged_task will give you the future corresponding to its promise . For example, we can set up two tasks to each add half of the elements of a vector<double> using the standard-library accumulate () (§14.3):

double accum (double ∗ beg, double ∗ end, double init) // compute the sum of [beg:end) starting with the initial value init { return accumulate (beg, end, init); } double comp2 (vector<double>& v) { using Task_type $=$ double (double ∗ ,double ∗ ,double); // type of task packaged_task<Task_type> pt0 {accum}; // package the task (i.e., accum) packaged_task<Task_type> pt1 {accum}; future<double> f0 {pt0. get_future ()}; // get hold of pt0’s future future<double> f1 {pt1. get_future ()}; // get hold of pt1’s future double ∗ first $=$ thread t1 {move (pt0),first,first+v.siz e ()/2,0}; // star t a thread for pt0 thread t2 {move (pt1),first+v.siz e ()/2,first+v.siz e (), 0}; // star t a thread for pt1 // ... return f0.get ()+f1. g et (); // get the results }

The packaged_task template takes the type of the task as its template argument (here Task_type , an alias for double (double ∗ ,double ∗ ,double) ) and the task as its constructor argument (here, accum ). The move () operations are needed because a packaged_task cannot be copied. The reason that a packaged_task cannot be copied is that it is a resource handle: it owns its promise and is (indirectly) responsible for whatever resources its task may own.

Please note the absence of explicit mention of locks in this code: we are able to concentrate on tasks to be done, rather than on the mechanisms used to manage their communication. The two tasks will be run on separate threads and thus potentially in parallel.

# 15.7.3 async ()

The line of thinking I have pursued in this chapter is the one I believe to be the simplest yet still among the most powerful: treat a task as a function that may happen to run concurrently with other tasks. It is far from the only model supported by the $\mathrm{C++}$ standard library, but it serves well for a wide range of needs. More subtle and tricky models (e.g., styles of programming relying on shared memory), can be used as needed.

To launch tasks to potentially run asynchronously, we can use async () :

double comp4 (vector<double>& v) // spawn many tasks if v is large enough { if (v.siz e ()<10000) // is it wor th using concurrency? return accum (v.begin (),v.end (), 0.0); auto $\mathsf{v0}=\&\mathsf{v}[0]$ ; auto sz $=$ v.siz e (); auto $\mathbf{f0}=$ async (accum, v0, v0+sz/4,0.0); // first quarter auto f1 $=$ async (accum, v0+sz/4, v0+sz/2,0.0); // second quarter auto ${\bf f}2=$ ∗ // third quarter auto f3 $=$ ∗ // four th quar ter return f0.get ()+f1. g et ()+f2. g et ()+f3. g et (); // collect and combine the results }

Basically, async () separates the ‘‘call part’’ of a function call from the ‘‘get the result part’’ and sep- arates both from the actual execution of the task. Using async () , you don’t hav e to think about threads and locks. Instead, you think just in terms of tasks that potentially compute their results asynchronously. There is an obvious limitation: don’t even think of using async () for tasks that share resources needing locking. With async () you don’t even know how many thread s will be used because that’s up to async () to decide based on what it knows about the system resources available at the time of a call. For example, async () may check whether any idle cores (processors) are avail- able before deciding how many thread s to use.

Using a guess about the cost of computation relative to the cost of launching a thread , such as $\mathsf{v.s i z e ()}\!\!<\!\! 10000$ , is very primitive and prone to gross mistakes about performance. However, this is not the place for a proper discussion about how to manage thread s. Don’t take this estimate as more than a simple and probably poor guess.

It is rarely necessary to manually parallelize a standard-library algorithm, such as accumulate () , because the parallel algorithms, such as reduce (par_unseq,/ ∗ ... ∗ /) , usually do a better job at that $(\S14.3.1)$ . However, the technique is general.

Please note that async () is not just a mechanism specialized for parallel computation for increased performance. For example, it can also be used to spawn a task for getting information from a user, leaving the ‘‘main program’’ active with something else (§15.7.3).

# 15.8 Advice

[1] Use concurrency to improve responsiveness or to improve throughput; $\S15.1$ .

[2] Work at the highest level of abstraction that you can afford; $\S15.1$ .

[3] Consider processes as an alternative to threads; $\S15.1$ .

[4] The standard-library concurrency facilities are type safe; $\S15.1$ .

[5] The memory model exists to save most programmers from having to think about the machine architecture level of computers; $\S15.1$ .

[6] The memory model makes memory appear roughly as naively expected; $\S15.1$ .

[7] Atomics allow for lock-free programming; $\S15.1$ .

[8] Leave lock-free programming to experts; $\S15.1$ .

[9] Sometimes, a sequential solution is simpler and faster than a concurrent solution; $\S15.1$ .

[10] Avoid data races; $\S15.1$ , $\S15.2$ .

[11] Prefer parallel algorithms to direct use of concurrency; $\S15.1,\S15.7.3$ .

[12] A thread is a type-safe interface to a system thread; $\S15.2$ .

[13] Use join () to wait for a thread to complete; $\S15.2$ .

[14] Avoid explicitly shared data whenever you can; $\S15.2$ .

[15] Prefer RAII to explicit lock/unlock; $\S15.5$ ; [CG: CP. 20].

[16] Use scoped_lock to manage mutex es; $\S15.5$ .

[17] Use scoped_lock to acquire multiple locks; $\S15.5$ ; [CG: CP. 21].

[18] Use shared_lock to implement reader-write locks; $\S15.5$ ;

[19] Define a mutex together with the data it protects; $\S15.5$ ; [CG: CP. 50].

[20] Use condition variable s to manage communication among thread s; $\S15.6$ .

[21] Use unique_lock (rather than scoped_lock ) when you need to copy a lock or need lower-level manipulation of synchronization; $\S15.6$ .

[22] Use unique_lock (rather than scoped_lock ) with condition variable s; $\S15.6$ .

[23] Don’t wait without a condition; $\S15.6$ ; [CG: CP. 42].

[24] Minimize time spent in a critical section; $\S15.6$ [CG: CP. 43].

[25] Think in terms of tasks that can be executed concurrently, rather than directly in terms of thread s; $\S15.7$ .

[26] Value simplicity; $\S15.7$ .

[27] Prefer packaged_task and future s over direct use of thread s and mutex es; $\S15.7$ .

[28] Return a result using a promise and get a result from a future ; $\S15.7.1$ ; [CG: CP. 60].

[29] Use packaged_task s to handle exceptions thrown by tasks and to arrange for value return; $\S15.7.2$ .

[30] Use a packaged_task and a future to express a request to an external service and wait for its response; $\S15.7.2$ .

[31] Use async () to launch simple tasks; $\S15.7.3$ ; [CG: CP. 61].

# History and Compatibility

Hurry Slowly (festina lente). – Octavius, Caesar Augustus

• History Timeline; The Early Years; The ISO $\mathrm{C++}$ Standards; Standards and Programming Style $\mathrm{C++}$ Use • $\mathrm{C++}$ Feature Evolution $_{\mathrm{C++11}}$ Language Features; $_{\textrm{C++14}}$ Language Features; $_{\mathrm{C++17}}$ Language Features; $_{\mathrm{C++11}}$ Standard-Library Components; $_{\textrm{C++14}}$ Standard-Library Components; $_{\mathrm{C++17}}$ Standard-Library Components; Removed and Deprecated Features • $\mathrm{C/C++}$ Compatibility C and $\mathrm{C++}$ Are Siblings; Compatibility Problems • Bibliography • Advice

# 16.1 History

I inv ented $\mathrm{C++}$ , wrote its early definitions, and produced its first implementation. I chose and for- mulated the design criteria for $\mathrm{C++}$ , designed its major language features, developed or helped to develop many of the early libraries, and for 25 years was responsible for the processing of exten- sion proposals in the $\mathrm{C++}$ standards committee.

$\mathrm{C++}$ was designed to provide Simula’s facilities for program organization [Dahl, 1970] together with C’s efficiency and ﬂexibility for systems programming [Kernighan, 1978]. Simula was the ini- tial source of $C++^{\ast}\mathrm{s}$ abstraction mechanisms. The class concept (with derived classes and virtual functions) was borrowed from it. However, templates and exceptions came to $\mathrm{C++}$ later with dif- ferent sources of inspiration.

The evolution of $\mathrm{C++}$ was always in the context of its use. I spent a lot of time listening to users and seeking out the opinions of experienced programmers. In particular, my colleagues at AT&T Bell Laboratories were essential for the growth of $\mathrm{C++}$ during its first decade.

This section is a brief overview; it does not try to mention every language feature and library component. Furthermore, it does not go into details. For more information, and in particular for more names of people who contributed, see my two papers from the ACM History of Programming Languages conferences [Stroustrup, 1993] [Stroustrup, 2007] and my Design and Evolution of $C++$ book (known as ‘‘D&E’’) [Stroustrup, 1994]. They describe the design and evolution of $\mathrm{C++}$ in detail and document inﬂuences from other programming languages.

Most of the documents produced as part of the ISO $\mathrm{C++}$ standards effort are available online [WG21]. In my FAQ, I try to maintain a connection between the standard facilities and the people who proposed and refined those facilities [Stroustrup, 2010]. $\mathrm{C++}$ is not the work of a faceless, anonymous committee or of a supposedly omnipotent ‘‘dictator for life’’; it is the work of many dedicated, experienced, hard-working individuals.

# 16.1.1 Timeline

The work that led to $\mathrm{C++}$ started in the fall of 1979 under the name ‘‘C with Classes.’’ Here is a simplified timeline:

1979 Work on $^{\backprime}\mathbf{C}$ with Classes’’ started. The initial feature set included classes and derived classes, public/private access control, constructors and destructors, and function declara- tions with argument checking. The first library supported non-preemptive concurrent tasks and random number generators. 1984 ‘‘C with Classes’’ was renamed to $\mathrm{C++}$ . By then, $\mathrm{C++}$ had acquired virtual functions, function and operator overloading, references, and the I/O stream and complex number libraries. 1985 First commercial release of $\mathrm{C++}$ (October 14). The library included I/O streams, com- plex numbers, and tasks (non-preemptive scheduling). 1985 The $C++$ Programming Language ( $\mathrm{{}^{\ast\!}T C++P L}$ ,’’ October 14) [Stroustrup, 1986]. 1989 The Annotated $C++R$ eference Manual (‘‘the ARM’’) [Ellis, 1989]. 1991 The $C++$ Programming Language, Second Edition [Stroustrup, 1991], presenting generic programming using templates and error handling based on exceptions, including the ‘‘Resource Acquisition Is Initialization’’ (RAII) general resource-management idiom. 1997 The $C++$ Programming Language, Third Edition [Stroustrup, 1997] introduced ISO $\mathbf{C++}$ , including namespaces, dynamic*cast , and many refinements of templates. The standard library added the STL framework of generic containers and algorithms. 1998 ISO $\mathrm{C++}$ standard $[C++, 1998]$ . 2002 Work on a revised standard, colloquially named $\mathrm{C++0x}$ , started. 2003 A ‘‘bug fix’’ revision of the ISO $\mathrm{C++}$ standard was issued. A $\mathrm{C++}$ Technical Report introduced new standard-library components, such as regular expressions, unordered con- tainers (hash tables), and resource management pointers, which later became part of $*{\mathrm{C++11}}$ . 2006 An ISO $\mathrm{C++}$ Technical Report on Performance addressed questions of cost, predictability, and techniques, mostly related to embedded systems programming $[C++, 2004]$ .

2011 ISO $_{\mathrm{C++11}}$ standard $[C++, 2011]$ ]. It provided uniform initialization, move semantics, types deduced from initializers ( auto ), range- for , variadic template arguments, lambda expressions, type aliases, a memory model suitable for concurrency, and much more. The standard library added several components, including threads, locks, and most of the components from the 2003 Technical Report. 2013 The first complete $_{\mathrm{C++11}}$ implementations emerged. 2013 The $C++$ Programming Language, Fourth Edition introduced $_{\mathrm{C++11}}$ . 2014 ISO $_{\textrm{C++14}}$ standard $[\mathrm{C++,}2014]$ completing $_{\mathrm{C++11}}$ with variable templates, digit sepa- rators, generic lambdas, and a few standard-library improvements. The first $_{\textrm{C++14}}$ implementations were completed. 2015 The $\mathrm{C++}$ Core Guidelines projects started [Stroustrup, 2015]. 2015 The concepts TS was approved. 2017 ISO $_{\mathrm{C++17}}$ standard $[C++, 2017]$ offering a diverse set of new features, including order of evaluation guarantees, structured bindings, fold expressions, a file system library, parallel algorithms, and variant and optional types. The first $_{\mathrm{C++17}}$ implementations were com- pleted. 2017 The modules TS and the Ranges TS were approved. 2020 ISO $_{\mathrm{C++20}}$ standard (scheduled).

During development, $_{\mathrm{C++11}}$ was known as $\mathrm{C++0x}$ . As is not uncommon in large projects, we were overly optimistic about the completion date. To wards the end, we joked that the ’x’ in $\mathrm{C++0x}$ was hexadecimal so that $\mathrm{C++0x}$ became $\mathbf{C}{+}{+}0\mathbf{B}$ . On the other hand, the committee shipped $_{\textrm{C++14}}$ and $\mathrm{C++17}$ on time, as did the major compiler providers.

# 16.1.2 The Early Years

I originally designed and implemented the language because I wanted to distribute the services of a UNIX kernel across multiprocessors and local-area networks (what are now known as multicores and clusters). For that, I needed to precisely specify parts of a system and how they communicated. Simula [Dahl, 1970] would have been ideal for that, except for performance considerations. I also needed to deal directly with hardware and provide high-performance concurrent programming mechanisms for which C would have been ideal, except for its weak support for modularity and type checking. The result of adding Simula-style classes to C (Classic C; $\S16.3.1)$ , ‘‘C with Classes,’’ was used for major projects in which its facilities for writing programs that use minimal time and space were severely tested. It lacked operator overloading, references, virtual functions, templates, exceptions, and many, many details [Stroustrup, 1982]. The first use of $\mathrm{C++}$ outside a research organization started in July 1983.

The name $\mathrm{C++}$ (pronounced ‘‘see plus plus’’) was coined by Rick Mascitti in the summer of 1983 and chosen as the replacement for $^{\backprime}\mathbf{C}$ with Classes’’ by me. The name signifies the evolu- tionary nature of the changes from C; $^{**}$ is the C increment operator. The slightly shorter name $^{**}\mathbf{C}+^{\mathbf{\Lambda}}$ is a syntax error; it had also been used as the name of an unrelated language. Connoisseurs of C semantics find $\mathrm{C++}$ inferior to ${\mathrel{+{+}}}\mathbf{C}$ . The language was not called D, because it was an exten- sion of C, because it did not attempt to remedy problems by removing features, and because there already existed several would-be C successors named D. For yet another interpretation of the name $\mathrm{C++}$ , see the appendix of [Orwell, 1949].

$\mathrm{C++}$ was designed primarily so that my friends and I would not have to program in assembler, C, or various then-fashionable high-level languages. Its main purpose was to make writing good programs easier and more pleasant for the individual programmer. In the early years, there was no $\mathrm{C++}$ paper design; design, documentation, and implementation went on simultaneously. There was no $^{**}\mathbf{C}^{++}$ project’’ either, or a $^{**}\mathrm{C}^{++}$ design committee.’’ Throughout, $\mathrm{C++}$ evolved to cope with problems encountered by users and as a result of discussions among my friends, my colleagues, and me.

The very first design of $\mathrm{C++}$ (then called ‘‘C with Classes’’) included function declarations with argument type checking and implicit conversions, classes with the public / private distinction between the interface and the implementation, derived classes, and constructors and destructors. I used macros to provide primitive parameter iz ation [Stroustrup, 1982]. This was in non-experimental use by mid-1980. Late that year, I was able to present a set of language facilities supporting a coherent set of programming styles. In retrospect, I consider the introduction of constructors and destructors most significant. In the terminology of the time [Stroustrup, 1979]:

A ‘‘new function’’ creates the execution environment for the member functions and the ‘‘delete function’’ reverses that.

Soon after, ‘‘new function’ and ‘‘delete function’ were renamed ‘‘constructor’’ and ‘‘destructor.’’ Here is the root of $\mathrm{C++}^{\ast}$ s strategies for resource management (causing a demand for exceptions) and the key to many techniques for making user code short and clear. If there were other languages at the time that supported multiple constructors capable of executing general code, I didn’t (and don’t) know of them. Destructors were new in $\mathrm{C++}$ .

$\mathrm{C++}$ was released commercially in October 1985. By then, I had added inlining $(\S1.3,\S4.2.1)$ , const s (§1.6), function overloading (§1.3), references (§1.7), operator overloading $(\S4.2.1)$ , and vir- tual functions (§4.4). Of these features, support for run-time polymorphism in the form of virtual functions was by far the most controversial. I knew its worth from Simula but found it impossible to convince most people in the systems programming world of its value. Systems programmers tended to view indirect function calls with suspicion, and people acquainted with other languages supporting object-oriented programming had a hard time believing that vir tual functions could be fast enough to be useful in systems code. Conversely, many programmers with an object-oriented background had (and many still have) a hard time getting used to the idea that you use virtual func- tion calls only to express a choice that must be made at run time. The resistance to virtual func- tions may be related to a resistance to the idea that you can get better systems through more regular structure of code supported by a programming language. Many C programmers seem convinced that what really matters is complete ﬂexibility and careful individual crafting of every detail of a program. My view was (and is) that we need every bit of help we can get from languages and tools: the inherent complexity of the systems we are trying to build is always at the edge of what we can express.

Early documents (e.g., [Stroustrup, 1985] and [Stroustrup, 1994]) described $\mathrm{C++}$ like this: $C++$ is a general-purpose programming language that • is a better $C$ • supports data abstraction • supports object-oriented programming

Note not $^{**}\mathbf{C}^{++}$ is an object-oriented programming language.’’ Here, ‘‘supports data abstraction’’ refers to information hiding, classes that are not part of class hierarchies, and generic programming.

Initially, generic programming was poorly supported through the use of macros [Stroustrup, 1982]. Templates and concepts came much later.

Much of the design of $\mathrm{C++}$ was done on the blackboards of my colleagues. In the early years, the feedback from Stu Feldman, Alexander Fraser, Steve Johnson, Brian Kernighan, Doug McIlroy, and Dennis Ritchie was invaluable.

In the second half of the 1980s, I continued to add language features in response to user com- ments. The most important of those were templates [Stroustrup, 1988] and exception handling [Koenig, 1990], which were considered experimental at the time the standards effort started. In the design of templates, I was forced to decide among ﬂexibility, efficiency, and early type checking. At the time, nobody knew how to simultaneously get all three. To compete with C-style code for demanding systems applications, I felt that I had to choose the first two properties. In retrospect, I think the choice was the correct one, and the search for better type checking of templates continues [DosReis, 2006] [Gregor, 2006] [Sutton, 2011] [Stroustrup, 2012a]. The design of exceptions focused on multilevel propagation of exceptions, the passing of arbitrary information to an error handler, and the integration between exceptions and resource management by using local objects with destructors to represent and release resources. I clumsily named that critical technique Resource Acquisition Is Initialization and others soon reduced that to the acronym RAII (§4.2.2).

I generalized $\textstyle\mathbf{C++}^{\prime};$ s inheritance mechanisms to support multiple base classes [Stroustrup, 1987]. This was called multiple inheritance and was considered difficult and controversial. I considered it far less important than templates or exceptions. Multiple inheritance of abstract classes (often called interfaces ) is now universal in languages supporting static type checking and object-oriented programming.

The $\mathrm{C++}$ language evolved hand-in-hand with some of the key library facilities. For example, I designed the complex [Stroustrup, 1984], vector, stack, and (I/O) stream classes [Stroustrup, 1985] together with the operator overloading mechanisms. The first string and list classes were developed by Jonathan Shopiro and me as part of the same effort. Jonathan’s string and list classes were the first to see extensive use as part of a library. The string class from the standard $\mathrm{C++}$ library has its roots in these early efforts. The task library described in [Stroustrup, 1987b] was part of the first $^{\circ}\mathrm{C}$ with Classes’’ program ever written in 1980. It provided coroutines and a scheduler. I wrote it and its associated classes to support Simula-style simulations. Unfortunately, we had to wait until 2011 (30 years!) to get concurrency support standardized and universally available (Chapter 15). Corou- tines are likely to be part of $_{\mathrm{C++20}}$ [CoroutinesTS]. The development of the template facility was inﬂuenced by a variety of vector , map , list , and sor templates devised by Andrew Koenig, Alex Stepanov, me, and others.

The most important innovation in the 1998 standard library was the STL, a framework of algo- rithms and containers (Chapter 11, Chapter 12). It was the work of Alex Stepanov (with Dave Musser, Meng Lee, and others) based on more than a decade’s work on generic programming. The STL has been massively inﬂuential within the $\mathrm{C++}$ community and beyond.

$\mathrm{C++}$ grew up in an environment with a multitude of established and experimental programming languages (e.g., Ada [Ichbiah, 1979], Algol 68 [Woodward, 1974], and ML [Paulson, 1996]). At the time, I was comfortable in about 25 languages, and their inﬂuences on $\mathrm{C++}$ are documented in [Stroustrup, 1994] and [Stroustrup, 2007]. However, the determining inﬂuences always came from the applications I encountered. It was a deliberate policy to hav e the development of $\mathrm{C++}$ ‘‘prob- lem driven’’ rather than imitative.

# 16.1.3 The ISO $\mathbf{C++}$ Standards

The explosive growth of $\mathrm{C++}$ use caused some changes. Sometime during 1987, it became clear that formal standardization of $\mathrm{C++}$ was inevitable and that we needed to start preparing the ground for a standardization effort [Stroustrup, 1994]. The result was a conscious effort to maintain contact between implementers of $\mathrm{C++}$ compilers and their major users. This was done through paper and electronic mail and through face-to-face meetings at $\mathrm{C++}$ conferences and elsewhere.

AT&T Bell Labs made a major contribution to $\mathrm{C++}$ and its wider community by allowing me to share drafts of revised versions of the $\mathrm{C++}$ reference manual with implementers and users. Because many of those people worked for companies that could be seen as competing with AT&T, the significance of this contribution should not be underestimated. A less enlightened company could have caused major problems of language fragmentation simply by doing nothing. As it hap- pened, about a hundred individuals from dozens of organizations read and commented on what became the generally accepted reference manual and the base document for the ANSI $\mathrm{C++}$ stan- dardization effort. Their names can be found in The Annotated $C++$ Reference Manual (‘‘the ARM’’) [Ellis, 1989]. The X3J16 committee of ANSI was convened in December 1989 at the ini- tiative of Hewlett-Packard. In June 1991, this ANSI (American national) standardization of $\mathrm{C++}$ became part of an ISO (international) standardization effort for $\mathrm{C++}$ . The ISO $\mathrm{C++}$ committee is called WG21. From 1990, these joint $\mathrm{C++}$ standards committees have been the main forum for the ev olution of $\mathrm{C++}$ and the refinement of its definition. I served on these committees throughout. In particular, as the chairman of the working group for extensions (later called the evolution group) from 1990 to 2014, I was directly responsible for handling proposals for major changes to $\mathrm{C++}$ and the addition of new language features. An initial draft standard for public review was produced in April 1995. The first ISO $\mathrm{C++}$ standard (ISO/IEC 14882-1998) $[\mathrm{C++}$ ,1998] was ratified by a 22-0 national vote in 1998. A ‘‘bug fix release’’ of this standard was issued in 2003, so you sometimes hear people refer to $_{\textrm C++03}$ , but that is essentially the same language as $_{\textrm{C++98}}$ .

$_{\mathrm{C++11}}$ , known for years as $\mathrm{C++0x}$ , is the work of the members of WG21. The committee worked under increasingly onerous self-imposed processes and procedures. These processes prob- ably led to a better (and more rigorous) specification, but they also limited innovation [Strous- trup, 2007]. An initial draft standard for public review was produced in 2009. The second ISO $\mathrm{C++}$ standard (ISO/IEC 14882-2011) $[C++, 2011]$ was ratified by a 21-0 national vote in August 2011.

One reason for the long gap between the two standards is that most members of the committee (including me) were under the mistaken impression that the ISO rules required a ‘‘waiting period’’ after a standard was issued before starting work on new features. Consequently, serious work on new language features did not start until 2002. Other reasons included the increased size of modern languages and their foundation libraries. In terms of pages of standards text, the language grew by about $30\%$ and the standard library by about $100\%$ . Much of the increase was due to more detailed specification, rather than new functionality. Also, the work on a new $\mathrm{C++}$ standard obviously had to take great care not to compromise older code through incompatible changes. There are billions of lines of $\mathrm{C++}$ code in use that the committee must not break. Stability over decades is an essen- tial ‘‘feature.’’

$_{\mathrm{C++11}}$ added massively to the standard library and pushed to complete the feature set needed for a programming style that is a synthesis of the ‘‘paradigms’’ and idioms that had proven success- ful with $\mathrm{C++98}$ .

The overall aims for the $_{\mathrm{C++11}}$ effort were:

• Make $\mathrm{C++}$ a better language for systems programming and library building. • Make $\mathrm{C++}$ easier to teach and learn.

The aims are documented and detailed in [Stroustrup, 2007].

A major effort was made to make concurrent systems programming type-safe and portable. This involved a memory model (§15.1) and support for lock-free programming, This was the work of Hans Boehm, Brian McKnight, and others in the concurrency working group. On top of that, we added the thread s library.

After $_{\mathrm{C++11}}$ , there was wide agreement that 13 years between standards were far too many. Herb Sutter proposed that the committee adopt a policy of shipping on time at fixed intervals, the ‘‘train model.’’ I argued strongly for a short interval between standards to minimize the chance of delays because someone insisted on extra time to allow inclusion of ‘‘just one more essential fea- ture.’’ We agreed on an ambitious 3-year schedule with the idea that we should alternate between minor and major releases.

$\mathrm{C++14}$ was deliberately a minor release aiming at ‘‘completing $_{\mathrm{C++11}}$ .’’ This reﬂects the real- ity that with a fixed release date, there will be features that we know we want, but can’t deliver on time. Also, once in widespread use, gaps in the feature set will inevitably be discovered.

To allow work to progress faster, to allow parallel development of independent features, and to better utilize the enthusiasm and skills of the many volunteers, the committee makes use of the ISO mechanisms of developing and publishing ‘‘Technical Specifications’’ (TSs). That seems to work well for standard-library components, though it can lead to more stages in the development process, and thus delays. For language features, TSs seems to work less well. Possibly the reason is that few significant language features are truly independent, because the work of crafting standards wording isn’t all that different between a standard and a TS, and because fewer people can experi- ment with compiler implementations.

$\mathrm{C++17}$ was meant to be a major release. By ‘‘major,’’ I mean containing features that will change the way we think about design and structure our software. By this definition, $_{\mathrm{C++17}}$ was at best a medium release. It included a lot of minor extensions, but the features that would have made dramatic changes (e.g., concepts, modules, and coroutines) were either not ready or became mired in controversy and lack of design direction. As a result, $_{\mathrm{C++17}}$ includes a little bit for everyone, but nothing that will significantly change the life of a $\mathrm{C++}$ programmer who has already absorbed the lessons of $_{\mathrm{C++11}}$ and $_{\mathrm{C++14}}$ . I hope that $_{\mathrm{C++20}}$ will be the promised and much-needed major revision, and that the major new features will become widely available well before 2020. The dan- gers are ‘‘Design by committee,’’ feature bloat, lack of consistent style, and short-sighted decisions. In a committee with well over 100 members present at each meeting and more participating on- line, such undesirable phenomena are almost unavoidable. Making progress toward a simpler-to- use and more coherent language is very hard.

# 16.1.4 Standards and Style

A standard says what will work, and how. It does not say what constitutes good and effective use. There are significant differences between understanding the technical details of programming lan- guage features and using them effectively in combination with other features, libraries, and tools to produce better software. By ‘‘better’’ I mean ‘‘more maintainable, less error-prone, and faster.’’ We need to develop, popularize, and support coherent programming styles. Further, we must sup- port the evolution of older code to these more modern, effective, and coherent styles.

With the growth of the language and its standard library, the problem of popularizing effective programming styles became critical. It is extremely difficult to make large groups of programmers depart from something that works for something better. There are still people who see $\mathrm{C++}$ as a few minor additions to C and people who consider 1980s Object-Oriented programming styles based on massive class hierarchies the pinnacle of development. Many are struggling to use $_{\mathrm{C++11}}$ well in environments with lots of old $\mathrm{C++}$ code. On the other hand, there are also many who enthu- siastically overuse novel facilities. For example, some programmers are convinced that only code using massive amounts of template metaprogramming is true $\mathrm{C++}$ .

What is Modern $C{\mathrel{+{+}}?}$ In 2015, I set out to answer this question by developing a set of coding guidelines supported by articulated rationales. I soon found that I was not alone in grappling with that problem and together with people from many parts of the world, notably from Microsoft, Red Hat, and Facebook, we started the $^{**}\mathbf{C}^{++}$ Core Guidelines’’ project [Stroustrup, 2015]. This is an ambitious project aiming at complete type-safety and complete resource-safety as a base for sim- pler, faster, and more maintainable code [Stroustrup, 2015b]. In addition to specific coding rules with rationales, we back up the guidelines with static analysis tools and a tiny support library. I see something like that as necessary for moving the $\mathrm{C++}$ community at large forward to benefit from the improvements in language features, libraries, and supporting tools.

# 16.1.5 $\mathbf{C++}$ Use

$\mathrm{C++}$ is now a very widely used programming language. Its user populations grew quickly from one in 1979 to about 400,000 in 1991; that is, the number of users doubled about every 7.5 months for more than a decade. Naturally, the growth rate slowed since that initial growth spurt, but my best estimate is that there are about 4.5 million $\mathrm{C++}$ programmers in 2018 [Kazakova, 2015]. Much of that growth happened after 2005 when the exponential explosion of processor speed stopped so that language performance grew in importance. This growth was achieved without formal marketing or an organized user community.

$\mathrm{C++}$ is primarily an industrial language; that is, it is more prominent in industry than in educa- tion or programming language research. It grew up in Bell Labs inspired by the varied and strin- gent needs of telecommunications and of systems programming (including device drivers, network- ing, and embedded systems). From there, $\mathrm{C++}$ use has spread into essentially every industry: microelectronics, Web applications and infrastructure, operating systems, financial, medical, auto- mobile, aerospace, high-energy physics, biology, energy production, machine learning, video games, graphics, animation, virtual reality, and much more. It is primarily used where problems require $\mathbf{C++}^{\dagger}$ ’s combination of the ability to use hardware effectively and to manage complexity. This seems to be a continuously expanding set of applications [Stroustrup, 1993] [Stroustrup, 2014].

# 16.2 $\mathbf{C++}$ Feature Evolution

Here, I list the language features and standard-library components that have been added to $\mathrm{C++}$ for the $_{\mathrm{C++11}}$ , $_{\mathrm{C++14}}$ , and $_{\mathrm{C++17}}$ standards.

# 16.2.1 $\mathbf{C++11}$ Language Features

Looking at a list of language features can be quite bewildering. Remember that a language feature is not meant to be used in isolation. In particular, most features that are new in $_{\mathrm{C++11}}$ make no sense in isolation from the framework provided by older features.

[1] Uniform and general initialization using $\{\}$ -lists (§1.4, §4.2.3) [2] Type deduction from initializer: auto (§1.4) [3] Prevention of narrowing (§1.4) [4] Generalized and guaranteed constant expressions: constexpr (§1.6) [5] Range- for -statement (§1.7) [6] Null pointer keyword: nullptr (§1.7) [7] Scoped and strongly typed enums : enum class (§2.5) [8] Compile-time assertions: static*asser (§3.5.5) [9] Language mapping of $\{\}$ -list to std:: initializ er_list (§4.2.3) [10] Rvalue references, enabling move semantics (§5.2.2) [11] Nested template arguments ending with ${}>>{}$ (no space between the ${\bf\rho}*{>S}$ ) [12] Lambdas (§6.3.2) [13] Variadic templates $(\S7.4)$ [14] Type and template aliases (§6.4.2) [15] Unicode characters [16] long long integer type [17] Alignment controls: alignas and alignof [18] The ability to use the type of an expression as a type in a declaration: decltype [19] Raw string literals (§9.4) [20] Generalized POD (‘‘Plain Old Data’’) [21] Generalized union s [22] Local classes as template arguments [23] Suffix return type syntax [24] A syntax for attributes and two standard attributes: [[carries dependency]] and [[noreturn]] [25] Preventing exception propagation: the noexcept specifier (§3.5.1) [26] Testing for the possibility of a throw in an expression: the noexcept operator. [27] C99 features: extended integral types (i.e., rules for optional longer integer types); con- catenation of narrow/wide strings; **STDC_HOSTED** ; \_Pragma (X) ; vararg macros and empty macro arguments [28] **func** as the name of a string holding the name of the current function [29] inline namespaces [30] Delegating constructors [31] In-class member initializers (§5.1.3) [32] Control of defaults: default and delete (§5.1.1) [33] Explicit conversion operators [34] User-defined literals (§5.4.4) [35] More explicit control of template instantiation: extern template s [36] Default template arguments for function templates

[37] Inheriting constructors [38] Override controls: override and final (§4.5.1) [39] A simpler and more general SFINAE (Substitution Failure Is Not An Error) rule [40] Memory model (§15.1) [41] Thread-local storage: thread_local

For a more complete description of the changes to in , see [Stroustrup, 2013].

# 16.2.2 $\mathbf{C++14}$ Language Features

[1] Function return-type deduction; $\S3.6.2$ [2] Improved constexpr functions, e.g., for -loops allowed (§1.6) [3] Variable templates (§6.4.1) [4] Binary literals (§1.4) [5] Digit separators (§1.4) [6] Generic lambdas (§6.3.3) [7] More general lambda capture [8] [[deprecated]] attribute [9] A few more minor extensions

# 16.2.3 $\mathbf{C++17}$ Language Features

[1] Guaranteed copy elision (§5.2.2) [2] Dynamic allocation of over-aligned types [3] Stricter order of evaluation (§1.4) [4] UTF-8 literals ( u8 ) [5] Hexadecimal ﬂoating-point literals [6] Fold expressions (§7.4.1) [7] Generic value template arguments ( auto template parameters) [8] Class template argument type deduction (§6.2.3) [9] Compile-time if (§6.4.3) [10] Selection statements with initializers (§1.8) [11] constexpr lambdas [12] inline variables [13] Structured bindings (§3.6.3) [14] New standard attributes: [[fallthrough]] , [[nodiscard]] , and [[maybe_unused]] [15] std:: byte type [16] Initialization of an enum by a value of its underlying type (§2.5) [17] A few more minor extensions

# 16.2.4 $\mathbf{C++11}$ Standard-Library Components

The $_{\mathrm{C++11}}$ additions to the standard library come in two forms: new components (such as the regu- lar expression matching library) and improvements to $_{\textrm{C++98}}$ components (such as move construc- tors for containers).

[1] initializ er_list constructors for containers (§4.2.3)

[2] Move semantics for containers (§5.2.2, $\S11.2)

$ [3] A singly-linked list: forward_list (§11.6)

[4] Hash containers: unordered_map , unordered_multimap , unordered_set , and unordered_mul- tiset $(\S11.6,\S11.5)

$ [5] Resource management pointers: unique_ptr , shared_ptr , and weak_ptr (§13.2.1)

[6] Concurrency support: thread (§15.2), mutexes (§15.5), locks (§15.5), and condition vari- ables (§15.6)

[7] Higher-level concurrency support: packaged_thread , future , promise , and async () (§15.7)

[8] tuple s (§13.4.3)

[9] Regular expressions: reg ex (§9.4)

[10] Random numbers: distributions and engines (§14.5)

[11] Integer type names, such as int16_t , uint32_t , and int_fast64_t

[12] A fixed-sized contiguous sequence container: array (§13.4.1)

[13] Copying and rethrowing exceptions (§15.7.1)

[14] Error reporting using error codes: system_error

[15] emplace () operations for containers (§11.6)

[16] Wide use of constexpr functions

[17] Systematic use of noexcept functions

[18] Improved function adaptors: function and bind () (§13.8)

[19] string to numeric value conversions

[20] Scoped allocators

[21] Type traits, such as is_integral and is_base_of (§13.9.2)

[22] Time utilities: duration and time_point (§13.7)

[23] Compile-time rational arithmetic: ratio

[24] Abandoning a process: quick_exit

[25] More algorithms, such as move () , copy_if () , and is_sor ted () (Chapter 12)

[26] Garbage collection API (§5.3)

[27] Low-level concurrency support: atomic s

# 16.2.5 $\mathbf{C++14}$ Standard-Library Components

[1] shared_mutex (§15.5)

[2] User-defined literals (§5.4.4)

[3] Tuple addressing by type (§13.4.3)

[4] Associative container heterogenous lookup

[5] A few more minor features

# 16.2.6 $\mathbf{C++17}$ Standard-Library Components

[1] File system (§10.10)

[2] Parallel algorithms $(\S12.9,\S14.3.1)

$ [3] Mathematical special functions (§14.2)

[4] string_view (§9.3) [5] any (§13.5.3) [6] variant (§13.5.1) [7] optional (§13.5.2) [8] invoke () [9] Elementary string conversions: to_chars and from_chars [10] Polymorphic allocator (§13.6) [11] A few more minor extensions

# 16.2.7 Removed and Deprecated Features

There are billions of lines of $\mathrm{C++}$ ‘‘out there’’ and nobody knows exactly what features are in criti- cal use. Consequently, the ISO committee removes older features only reluctantly and after years of warning. However, sometimes troublesome features are removed:

• $_{\mathrm{C++17}}$ finally removed exceptions specifications: void f () throw (X, Y); // $C{\mathrel{+}}{+}98,$ ; now an error The support facilities for exception specifications, unexcepted_handler , set_unexpected () , get_unexpected () , and unexpected () , are similarly removed. Instead, use noexcept (§3.5.1). • Trigraphs are no longer supported. • The auto_ptr is deprecated. Instead, use unique_ptr (§13.2.1). • The use of the storage specifier register is removed. • The use of $^{++}$ on a bool is removed. • The $\mathrm{C++98}$ expor feature was removed because it was complex and not shipped by the major vendors. Instead, expor is used as a keyword for modules (§3.3). • Generation of copy operations is deprecated for a class with a destructor (§5.1.1). • Assignment of a string literal to a char ∗ is removed. Instead use const char ∗ or auto . • Some $\mathrm{C++}$ standard-library function objects and associated functions are deprecated. Most relate to argument binding. Instead use lambdas and function (§13.8).

By deprecating a feature, the standards committee expresses the wish that the feature will go away. However, the committee does not have a mandate to immediately remove a heavily used feature – however redundant or dangerous it may be. Thus, a deprecation is a strong hint to avoid the fea- ture. It may disappear in the future. Compilers are likely to issue warnings for uses of deprecated features. However, deprecated features are part of the standard and history shows that they tend to remain supported ‘‘forever’’ for reasons of compatibility.

# 16.3 $\mathbf{C/C++}$ Compatibility

With minor exceptions, $\mathrm{C++}$ is a superset of C (meaning C11; [C, 2011]). Most differences stem from $\mathbf{C++}^{\dagger}$ ’s greater emphasis on type checking. Well-written C programs tend to be $\mathrm{C++}$ programs as well. A compiler can diagnose every difference between $\mathrm{C++}$ and C. The $\mathrm{C99/C++11}$ incom- patibilities are listed in Appendix C of the standard.

# 16.3.1 C and $\mathbf{C++}$ Are Siblings

Classic C has two main descendants: ISO C and ISO $\mathrm{C++}$ . Over the years, these languages have ev olved at different paces and in different directions. One result of this is that each language pro- vides support for traditional C-style programming in slightly different ways. The resulting incom- patibilities can make life miserable for people who use both C and $\mathrm{C++}$ , for people who write in one language using libraries implemented in the other, and for implementers of libraries and tools for C and $\mathrm{C++}$ .

How can I call C and $\mathrm{C++}$ siblings? Look at a simplified family tree:

![](images/dadc880f9aeb8dc7ff04ab798a206294f36adaeaaefc8fd63291529d47e5074e.jpg)

A solid line means a massive inheritance of features, a dashed line a borrowing of major features, and a dotted line a borrowing of minor features. From this, ISO C and ISO $\mathrm{C++}$ emerge as the two major descendants of K&R C [Kernighan, 1978], and as siblings. Each carries with it the key aspects of Classic C, and neither is $100\%$ compatible with Classic C. I picked the term ‘‘Classic C’’ from a sticker that used to be affixed to Dennis Ritchie’s terminal. It is K&R C plus enumera- tions and struct assignment. BCPL is defined by [Richards, 1980] and C89 by [C1990].

Note that differences between C and $\mathrm{C++}$ are not necessarily the result of changes to C made in $\mathrm{C++}$ . In several cases, the incompatibilities arise from features adopted incompatibly into C long after they were common in $\mathrm{C++}$ . Examples are the ability to assign a $^{\mathsf{T}*}$ to a void ∗ and the linkage of global const s [Stroustrup, 2002]. Sometimes, a feature was even incompatibly adopted into C after it was part of the ISO $\mathrm{C++}$ standard, such as details of the meaning of inline .

# 16.3.2 Compatibility Problems

There are many minor incompatibilities between C and $\mathrm{C++}$ . All can cause problems for a pro- grammer, but all can be coped with in the context of $\mathrm{C++}$ . If nothing else, C code fragments can be compiled as C and linked to using the extern "C" mechanism.

The major problems for converting a C program to $\mathrm{C++}$ are likely to be:

• Suboptimal design and programming style. • A void ∗ implicitly converted to a $\mathsf{T}*$ (that is, converted without a cast). • $\mathrm{C++}$ keywords, such as class and private , used as identifiers in C code. • Incompatible linkage of code fragments compiled as C and fragments compiled as $\mathrm{C++}$ .

# 16.3.2.1 Style Problems

Naturally, a C program is written in a C style, such as the style used in K&R [Kernighan, 1988]. This implies widespread use of pointers and arrays, and probably many macros. These facilities are hard to use reliably in a large program. Resource management and error handling are often ad hoc, documented (rather than language and tool supported), and often incompletely documented and adhered to. A simple line-for-line conversion of a C program into a $\mathrm{C++}$ program yields a program that is often a bit better checked. In fact, I have nev er converted a C program into $\mathrm{C++}$ without finding some bug. However, the fundamental structure is unchanged, and so are the fundamental sources of errors. If you had incomplete error handling, resource leaks, or buffer overﬂows in the original C program, they will still be there in the $\mathrm{C++}$ version. To obtain major benefits, you must make changes to the fundamental structure of the code:

[1] Don’t think of $\mathrm{C++}$ as C with a few features added. $\mathrm{C++}$ can be used that way, but only suboptimally. To get really major advantages from $\mathrm{C++}$ as compared to C, you need to apply different design and implementation styles. [2] Use the $\mathrm{C++}$ standard library as a teacher of new techniques and programming styles. Note the difference from the C standard library (e.g., $=$ rather than strcpy () for copying and $==$ rather than strcmp () for comparing). [3] Macro substitution is almost never necessary in $\mathrm{C++}$ . Use const (§1.6), constexpr (§1.6), enum or enum class (§2.5) to define manifest constants, inline $(\S4.2.1)$ to avoid function- calling overhead, template s (Chapter 6) to specify families of functions and types, and namespace s (§3.4) to avoid name clashes. [4] Don’t declare a variable before you need it and initialize it immediately. A declaration can occur anywhere a statement can (§1.8), in for -statement initializers (§1.7), and in con- ditions (§4.5.2). [5] Don’t use malloc () . The new operator (§4.2.2) does the same job better, and instead of realloc () , try a vector $(\S4.2.3,\S12.1)$ . Don’t just replace malloc () and free () with ‘‘naked’’ new and delete (§4.2.2).

[6] Avoid void ∗ , unions, and casts, except deep within the implementation of some function or class. Their use limits the support you can get from the type system and can harm per- formance. In most cases, a cast is an indication of a design error. [7] If you must use an explicit type conversion, use an appropriate named cast (e.g., static_cast ; $\S16.2.7)$ for a more precise statement of what you are trying to do. [8] Minimize the use of arrays and C-style strings. $\mathrm{C++}$ standard-library string s (§9.2), array s (§13.4.1), and vector s (§11.2) can often be used to write simpler and more maintainable code compared to the traditional C style. In general, try not to build yourself what has already been provided by the standard library. [9] Avoid pointer arithmetic except in very specialized code (such as a memory manager) and for simple array traversal (e.g., ${\mathsf{++}}{\mathsf{p}})$ ). [10] Do not assume that something laboriously written in C style (avoiding $\mathrm{C++}$ features such as classes, templates, and exceptions) is more efficient than a shorter alternative (e.g., using standard-library facilities). Often (but of course not always), the opposite is true.

# 16.3.2.2 void ∗

In C, a void $^*$ may be used as the right-hand operand of an assignment to or initialization of a vari- able of any pointer type; in $\mathrm{C++}$ it may not. For example:

void f (int n){ int ∗ p = malloc (n ∗ siz eof (int)); $/{}^{*}$ not $C{\mathrel{+{+}}};$ in $C++,$ , allocate using ‘‘new’’ \*/ // ... }

This is probably the single most difficult incompatibility to deal with. Note that the implicit con- version of a void ∗ to a different pointer type is not in general harmless:

char ch; void ∗ $\mathtt{p v}=\mathtt{\&c h}$ ; ${\mathsf{i n t}}\!*{\mathsf{p i}}={\mathsf{p v}}$ ; // not $C++$ $\mathbf{\boldsymbol{*}p i=666;}$ ; // overwr ite ch and other bytes near ch

In both languages, cast the result of malloc () to the right type. If you use only $\mathrm{C++}$ , avoid malloc () .

# 16.3.2.3 Linkage

C and $\mathrm{C++}$ can (and often are) implemented to use different linkage conventions. The most basic reason for that is $C++^{\ast}\mathrm{s}$ greater emphasis on type checking. A practical reason is that $\mathrm{C++}$ supports overloading, so there can be two global functions called open () . This has to be reﬂected in the way the linker works. To giv e a $\mathrm{C++}$ function C linkage (so that it can be called from a C program fragment) or to allow a C function to be called from a $\mathrm{C++}$ program fragment, declare it extern "C" . For example: extern "C" double sqrt (double);

Now sqr t (double) can be called from a C or a $\mathrm{C++}$ code fragment. The definition of sqr t (double) can also be compiled as a C function or as a $\mathrm{C++}$ function.

Only one function of a given name in a scope can have C linkage (because C doesn’t allow function overloading). A linkage specification does not affect type checking, so the $\mathrm{C++}$ rules for function calls and argument checking still apply to a function declared extern "C" .

# 16.4 Bibliography

[Boost]

[C, 1990]

[C, 1999]

[C, 2011] $I C++, I998J$ $I C++, 2O O4J

$ [C++Math, 2010]

[C++, 2011]

[C++, 2014]

[C++, 2017]

[ConceptsTS]

[CoroutinesTS]

[Cppreference]

[Cox, 2007]

[Dahl, 1970]

[Dechev, 2010]

[DosReis, 2006]

[Ellis, 1989]

The Boost Libraries: free peer-reviewed portable $C++$ source libraries . www. boost. org. X3 Secretariat: Standard – The C Language . X3J11/90-013. ISO Standard ISO/IEC 9899-1990. Computer and Business Equipment Manufacturers Association. Washington, DC. ISO/IEC 9899. Standard – The C Language . X3J11/90-013-1999. ISO/IEC 9899. Standard – The C Language . X3J11/90-013-2011. ISO/IEC JTC1/SC22/WG21 (editor: Andrew Koenig): International Stan- dard – The $C++$ Language . ISO/IEC 14882:1998. ISO/IEC JTC1/SC22/WG21 (editor: Lois Goldtwaite): Technical Report on $C++$ Performance. ISO/IEC TR 18015:2004 (E) International Standard – Extensions to the $C++$ Library to Support Mathe- matical Special Functions . ISO/IEC 29124:2010. ISO/IEC JTC1/SC22/WG21 (editor: Pete Becker): International Standard – The $C++$ Language . ISO/IEC 14882:2011. ISO/IEC JTC1/SC22/WG21 (editor: Stefanus du Toit): International Stan- dard – The $C++$ Language . ISO/IEC 14882:2014. ISO/IEC JTC1/SC22/WG21 (editor: Richard Smith): International Standard – The $C++$ Language . ISO/IEC 14882:2017. ISO/IEC JTC1/SC22/WG21 (editor: Gabriel Dos Reis): Technical Specifica- tion: $C++$ Extensions for Concepts . ISO/IEC TS 19217:2015. ISO/IEC JTC1/SC22/WG21 (editor: Gor Nishanov): Technical Specification: $C++$ Extensions for Coroutines . ISO/IEC TS 22277:2017. Online source for $C++$ language and standard library facilities . www. cppreference. com. Russ Cox: Regular Expression Matching Can Be Simple And Fast . January 2007. swtch. com/˜rsc/regexp/regexp1. html. O-J. Dahl, B. Myrhaug, and K. Nygaard: SIMULA Common Base Language. Norwegian Computing Center S-22. Oslo, Norway. 1970. D. Dechev, P. Pirkelbauer, and B. Stroustrup: Understanding and Effectively Preventing the ABA Problem in Descriptor-based Lock-free Designs . 13th IEEE Computer Society ISORC 2010 Symposium. May 2010. Gabriel Dos Reis and Bjarne Stroustrup: Specifying $C++$ Concepts . POPL06. January 2006. Margaret A. Ellis and Bjarne Stroustrup: The Annotated $C++$ Reference Manual . Addison-Wesley. Reading, Massachusetts. 1990. ISBN 0-201-51459-1.

[Garcia, 2015]

[Garcia, 2016]

[Garcia, 2018]

[Friedl, 1997]

[GSL]

[Gregor, 2006]

[Hinnant, 2018]

[Hinnant, 2018b]

[Ichbiah, 1979]

[Kazakova, 2015]

[Kernighan, 1978]

[Kernighan, 1988]

[Knuth, 1968]

[Koenig, 1990]

[Maddock, 2009]

[ModulesTS]

[Orwell, 1949]

[Paulson, 1996]

[RangesTS]

[Richards, 1980] J. Daniel Garcia and B. Stroustrup: Improving performance and maintain- ability through refactoring in $C{\mathrel{+{+}}}I I$ . Isocpp. org. August 2015. http://www. stroustrup. com/improving_garcia_stroustrup_2015. pdf. G. Dos Reis, J. D. Garcia, J. Lakos, A. Meredith, N. Myers, B. Stroustrup: A Contract Design . P0380R1. 2016-7-11. G. Dos Reis, J. D. Garcia, J. Lakos, A. Meredith, N. Myers, B. Stroustrup: Support for contract based programming in $C++$ . P0542R4. 2018-4-2. Jeffrey E. F. Friedl: Mastering Regular Expressions . O’Reilly Media. Sebastopol, California. 1997. ISBN 978-1565922570. N. MacIntosh (Editor): Guidelines Support Library . https://github. com/mi- crosoft/gsl. Douglas Gregor et al.: Concepts: Linguistic Support for Generic Program- ming in $C++$ . OOPSLA’06. Howard Hinnant: Date . https://howardhinnant. github. io/date/date. html. Github. 2018. Howard Hinnant: Timezones . https://howardhinnant. github. io/date/tz. html. Github. 2018. Jean D. Ichbiah et al.: Rationale for the Design of the ADA Pro gramming Language . SIGPLAN Notices. Vol. 14, No. 6. June 1979. Anastasia Kazakova: Infographic: $C/C++$ facts . https://blog. jetbrains. com/clion/2015/07/infographics-cpp-facts-before-clion/ July 2015. Brian W. Kernighan and Dennis M. Ritchie: The C Programming Language. Prentice Hall. Englewood Cliffs, New Jersey. 1978. Brian W. Kernighan and Dennis M. Ritchie: The C Programming Language, Second Edition. Prentice-Hall. Englewood Cliffs, New Jersey. 1988. ISBN 0-13-110362-8. Donald E. Knuth: The Art of Computer Programming . Addison-Wesley. Reading, Massachusetts. 1968. A. R. Koenig and B. Stroustrup: Exception Handling for $C++$ (revised) . Proc USENIX $\mathrm{C++}$ Conference. April 1990. John Maddock: Boost. Regex . www. boost. org. 2009. 2017. ISO/IEC JTC1/SC22/WG21 (editor: Gabriel Dos Reis): Technical Specifica- tion: $C++$ Extensions for Modules . ISO/IEC TS 21544:2018. George Orwell: 1984. Secker and Warburg. London. 1949. Larry C. Paulson: ML for the Working Programmer . Cambridge University Press. Cambridge. 1996. ISO/IEC JTC1/SC22/WG21 (editor: Eric Niebler): Technical Specification: $C++$ Extensions for Ranges . ISO/IEC TS 21425:2017. ISBN 0-521-56543-X. Martin Richards and Colin Whitby-Strevens: BCPL – The Language and Its Compiler. Cambridge University Press. Cambridge. 1980. ISBN 0-521-21965-5.

Alexander Stepanov and Meng Lee: The Standard Template Library . HP Labs Technical Report HPL-94-34 (R. 1). 1994. Alexander Stepanov and Paul McJones: Elements of Programming . Addi- son-Wesley. 2009. ISBN 978-0-321-63537-2. Personal lab notes. B. Stroustrup: Classes: An Abstract Data Type Facility for the C Language . Sigplan Notices. January 1982. The first public description of ‘‘C with Classes.’’ B. Stroustrup: Operator Overloading in $C++$ . Proc. IFIP WG2.4 Confer- ence on System Implementation Languages: Experience & Assessment. September 1984. B. Stroustrup: An Extensible I/O Facility for $C++$ . Proc. Summer 1985 USENIX Conference. B. Stroustrup: The $C++$ Programming Language . Addison-Wesley. Read- ing, Massachusetts. 1986. ISBN 0-201-12078-X. B. Stroustrup: Multiple Inheritance for $C++$ . Proc. EUUG Spring Confer- ence. May 1987. B. Stroustrup and J. Shopiro: A Set of $C$ Classes for Co-Routine Style Pro- gramming . Proc. USENIX $\mathrm{C++}$ Conference. Santa Fe, New Mexico. No- vember 1987. B. Stroustrup: Parameterized Types for $C++$ . Proc. USENIX $\mathrm{C++}$ Confer- ence, Denver, Colorado. 1988. B. Stroustrup: The $C++$ Programming Language (Second Edition) . Addi- son-Wesley. Reading, Massachusetts. 1991. ISBN 0-201-53992-6. B. Stroustrup: A History of $C++$ : 1979–1991 . Proc. ACM History of Pro- gramming Languages Conference (HOPL-2). ACM Sigplan Notices. Vol 28, No 3. 1993. B. Stroustrup: The Design and Evolution of $C++$ . Addison-Wesley. Read- ing, Massachusetts. 1994. ISBN 0-201-54330-3. B. Stroustrup: The $C++$ Programming Language, Third Edition . Addison- Wesley. Reading, Massachusetts. 1997. ISBN 0-201-88954-4. Hardcover (‘‘Special’’) Edition. 2000. ISBN 0-201-70073-5. B. Stroustrup: $C$ and $C++$ : Siblings , $C$ and $C{+}{+}{\cdot}\, A$ Case for Compatibility , and $C$ and $C++$ : Case Studies in Compatibility . The $\mathrm{C}/\mathrm{C++}$ Users Journal. July-September 2002. www. stroustrup. com/papers. html. B. Stroustrup: Evolving a language in and for the real world: $C++$ 1991-2006 . ACM HOPL-III. June 2007. B. Stroustrup: Programming – Principles and Practice Using $C++$ . Addi- son-Wesley. 2009. ISBN 0-321-54372-6. B. Stroustrup: The $\; C++I l\; F A Q$ . www. stroustrup. com $/\mathbf{C}_{++1}$ 1FAQ. html. B. Stroustrup and A. Sutton: A Concept Design for the STL . WG21 Techni- cal Report $\Nu3351{=}12{\cdot}0041$ . January 2012. B. Stroustrup: Software Development for Infrastructure . Computer. January 2012. doi: 10.1109/MC. 2011.353.

[Stroustrup, 2013] B. Stroustrup: The $C++$ Programming Language (Fourth Edition) . Addison- Wesley. 2013. ISBN 0-321-56384-0.

[Stroustrup, 2014] B. Stroustrup: $\mathrm{C++}$ Applications. http://www. stroustrup. com/applica- tions. html.

[Stroustrup, 2015] B. Stroustrup and H. Sutter: $C++$ Core Guidelines . https://github. com/isocpp/Cpp Core Guidelines/blob/master/CppCoreGuide- lines. md.

[Stroustrup, 2015b] B. Stroustrup, H. Sutter, and G. Dos Reis: A brief introduction to $C++, s$ model for type- and resource-safety . Isocpp. org. October 2015. Revised December 2015. http://www. stroustrup. com/resource-model. pdf.

[Sutton, 2011] A. Sutton and B. Stroustrup: Design of Concept Libraries for $C++$ . Proc. SLE 2011 (International Conference on Software Language Engineering). July 2011.

[WG21] ISO SC22/WG21 The $\mathrm{C++}$ Programming Language Standards Committee: Document Archive . www. open-std. org/jtc1/sc22/wg21.

[Williams, 2012] Anthony Williams: $C++$ Concurrency in Action – Practical Multithreading . Manning Publications Co. ISBN 978-1933988771.

[Woodward, 1974] P. M. Woodward and S. G. Bond: Algol 68-R Users Guide. Her Majesty’s Stationery Office. London. 1974.

# 16.5 Advice

[1] The ISO $\mathrm{C++}$ standard $[C++, 2017]$ ] defines $\mathrm{C++}$ .

[2] When chosing a style for a new project or when modernizing a code base, rely on the $\mathrm{C++}$ Core Guidelines; $\S16.1.4$ .

[3] When learning $\mathrm{C++}$ , don’t focus on language features in isolation; $\S16.2.1$ .

[4] Don’t get stuck with decades-old language-feature sets and design techniques; $\S16.1.4$ .

[5] Before using a new feature in production code, try it out by writing small programs to test the standards conformance and performance of the implementations you plan to use.

[6] For learning $\mathbf{C++}$ , use the most up-to-date and complete implementation of Standard $\mathrm{C++}$ that you can get access to.

[7] The common subset of C and $\mathrm{C++}$ is not the best initial subset of $\mathrm{C++}$ to learn; $\S16.3.2.1$ .

[8] Prefer named casts, such as static_cast over C-style casts; $\S16.2.7$ .

[9] When converting a C program to $\mathbf{C++}$ , first make sure that function declarations (prototypes) and standard headers are used consistently; $\S16.3.2$ .

[10] When converting a C program to $\mathrm{C++}$ , rename variables that are $\mathrm{C++}$ keywords; $\S16.3.2$ .

[11] For portability and type safety, if you must use C, write in the common subset of C and $\mathrm{C++}$ ; $\S16.3.2.1$ .

[12] When converting a C program to $\mathrm{C++}$ , cast the result of malloc () to the proper type or change all uses of malloc () to uses of new ; $\S16.3.2.2$ .

[13] When converting from malloc () and free () to new and delete , consider using vector , push_back () , and reser ve () instead of realloc () ; $\S16.3.2.1$ .

[14] In $\mathrm{C++}$ , there are no implicit conversions from int s to enumerations; use explicit type conver- sion where necessary.

[15] For each standard C header $<\!\!\mathbf{X}.\mathbf{h}\!\!>$ that places names in the global namespace, the header <cX> places the names in namespace std .

[16] Use extern $"{\mathfrak{C}}"$ when declaring C functions; $\S16.3.2.3$ .

[17] Prefer string over C-style strings (direct manipulation of zero-terminated arrays of char ).

[18] Prefer iostream s over stdio .

[19] Prefer containers (e.g., vector ) over built-in arrays.

# Index

Knowledge is of two kinds. We know a subject ourselves, or we know where we can find information on it. – Samuel Johnson

# Token

!= container 147 not-equal operator 6

" , string literal 3

$\S$ , regex 117

$\%$ modulus operator 6 remainder operator 6

$\%=$ , operator 7

$\&$ address-of operator 11 reference to 12

&& , rvalue reference 71

( , regex 117

() , call operator 85

(?: pattern 120

) , regex 117

∗ contents-of operator 11 multiply operator 6 pointer to 11 regex 117

$^{\ast=}$ , scaling operator 7

∗ ? lazy 118

- plus operator 6 regex 117 str ing concatenation 111

$^{++}$ , increment operator 7

$+=$ operator 7 str ing append 112

$+?$ lazy 118

- , minus operator 6

-- , decrement operator 7

. , regex 117

/ , divide operator 6

// comment 2

$/=$ , scaling operator 7

: public 55

<< 75 output operator 3

$<=$ container 147 less-than-or-equal operator 6

$<$ container 147 less-than operator 6

= 0 54 and == 7 assignment 16 auto 8 container 147 initializer 7 initializer narrowing 8 str ing assignment 112

== $=$ and 7 container 147 equal operator 6 str ing 112

> container 147 greater-than operator 6

> = container 147 greater-than-or-equal operator 6

> > 75 template arguments 215

? , regex 117

?? lazy 118

[ , regex 117

[] array 171 array of 11 str ing 112

\ , backslash 3

] , regex 117 ˆ , regex 117

{ , regex 117

{} grouping 2 initializer 8

{}? lazy 118

| , regex 117

} , regex 117

˜ , destructor 51 0 $=\quad54$ nullptr NULL 13

# A

abs () 188 abstract class54type 54 accumulate () 189 acquisition RAII, resource 164 adaptor, lambda as 180 address, memory 16 address-of operator & 11 adjacent difference () 189 aims, $_{\mathrm{C++11}}$ 213 algorithm 149 container 150, 160

lifting 100 numerical 189 parallel 161 standard library 156 <algor ithm> 109, 156 alias template 184 using 90 alignas 215 alignof 215 allocation 51 allocator new , container 178 almost container 170 alnum , regex 119 alpha , regex 119 [[:alpha:]] letter 119 ANSI $\mathrm{C++}$ 212 any 177 append $+=$ , str ing 112 argument constrained 81 constrained template 82 default function 42 default template 98 function 41 passing, function 66 type 82 value 82 arithmetic conversions, usual 7 operator 6 vector 192 ARM 212 array array vs. 172 of [] 11 array 171 [] 171 data () 171 initialize 171 size () 171 vs. array 172 vs. vector 171 <array> 109 asin () 188 assembler 210 asser t () 40 assertion static_asser 40 Assignable , concept 158 assignment $=\quad16$ $=$ , str ing 112 copy 66, 69 initialization and 18 move 66, 72 associative array – see map

async () launch 204 at () 141 atan () 188 atan2 () 188 AT&T Bell Laboratories 212 auto $=$ 8 auto_ptr , deprecated 218

# B

back_inser ter () 150 backslash \ 3 bad_var iant_access 176 base and derived class 55 basic_str ing 114 BCPL 219 begin () 75, 143, 147, 150 beginner, book for 1 Bell Laboratories, AT&T 212 beta () 188 bibliography 222 Bidirectional Iterator , concept 159 Bidirectional Range , concept 160 binary search 156 binding, structured 45 bit-field, bitset and 172 bitset 172 and bit-field 172 and enum 172 blank , regex 119 block as function body, tr y 141 tr y 36 body, function 2 book for beginner 1 bool 5 Boolean , concept 158 BoundedRange , concept 160 break 15

# C

C 209 and $\mathrm{C++}$ compatibility 218 Classic 219 difference from 218 K&R 219 void $^*$ assignment, difference from 221 with Classes 208 with Classes language features 210 with Classes standard library 211 $\mathrm{C++}$ ANSI 212 compatibility, C and 218 Core Guidelines 214 core language 2 history 207 ISO 212 meaning 209 modern 214 pronunciation 209 standard, ISO 2 standard library 2 standardization 212 timeline 208

$_{C++03}$ 212

$_{\mathrm{C++0x}}$ , $_{\mathrm{C++11}}$ 209, 212

$_{\mathrm{C++11}}$ aims 213 $\mathrm{C++0x}$ 209, 212 language features 215 library components 216

$_{\mathrm{C++14}}$ 213 language features 216 library components 217

$_{\mathrm{C++17}}$ 1, 213 language features 216 library components 217

$_{\mathrm{C++20}}$ 1, 157, 213 concepts 94 contracts 40 modules 32 $C{+}{+}98$ 212 standard library 211 C11 218 C89 and C99 218 C99, C89 and 218 call operator () 85 callback 181 capacity () 139, 147 capture list 87 carr ies_dependency 215 cast 53 catch clause 36 ev ery exception 141 catch (...) 141 ceil () 188 char 5 character sets, multiple 114 check compile-time 40 run-time 40 checking, cost of range 142 chrono , namespace 179 <chrono> 109, 179, 200 class 48 abstract 54 base and derived 55 concrete 48 hierarchy 57

# 230 Index

scope 9 template 79 Classic C 219 C-library header 110 clock timing 200 <cmath> 109, 188 cntr , regex 119 code complexity, function and 4 comment, // 2 Common , concept 158 CommonReference , concept 158 common*type_t 158 communication, task 202 comparison 74 operator 6, 74 compatibility, C and $\mathrm{C++}$ 218 compilation model, template 104 separate 30 compiler 2 compile-time check 40 computation 181 evaluation 10 complete encapsulation 66 complex 49, 190 <complex> 109, 188, 190 complexity, function and code 4 components $*{\mathrm{C++11}}$ library 216 $_{\mathrm{C++14}}$ library 217 $_{\mathrm{C++17}}$ library 217 computation, compile-time 181 concatenation $^+$ , str ing 111 concept 81, 94 Assignable 158 based overloading 95 Bidirectional Iterator 159 Bidirectional Range 160 Boolean 158 BoundedRange 160 Common 158 CommonReference 158 Constr uctible 158 ConvertibleTo 158 Copyable 158 CopyConstr uctible 158 DefaultConstr uctible 158 Der ivedFrom 158 Destr uctible 158 Equality Comparable 158 ForwardIterator 159 ForwardRange 160 InputIterator 159 InputRange 160 Integral 158

Invocable 159 In voc able Regular 159 Iterator 159 Mergeable 159 Movable 158 MoveConstr uctible 158 OutputIterator 159 OutputRange 160 Permutable 159 Predicate 159 Random Access Iterator 159 Random Access Range 160 Range 157 Range 160 Regular 158 Relation 159 Same 158 Semiregular 158 Sentinel 159 SignedIntegral 158 SizedRange 160 SizedSentinel 159 Sor table 159 Str ictTotallyOrdered 158 Str ictWeakOrder 159 support 94 Swappable 158 SwappableWith 158 Unsigned Integral 158 use 94 View 160 Weakly Equality Comparable 158 concepts $_{\mathrm{C++20}}$ 94 definition of 97 in <concepts> 158 in <iterator> 158 in <ranges> 158 <concepts> , concepts in 158 concrete class 48 type 48 concurrency 195 condition, declaration in 61 condition_var iable 201 notify_one () 202 wait () 201 <condition_var iable> 201 const immutability 9 member function 50 constant expression 10 const_cast 53 constexpr function 10 immutability 9

const*iterator 154 constrained argument 81 template 82 template argument 82 Constr uctible , concept 158 constructor and destructor 210 copy 66, 69 default 50 delegating 215 explicit 67 inheriting 216 initializer-list 52 invariant and 37 move 66, 71 container 51, 79, 137 $>=$ 147 $>$ 147 = 147 == 147 < 147 <= 147 $\circeq$ 147 algorithm 150, 160 allocator new 178 almost 170 object in 140 overview 146 retur n 151 sor t () 181 specialized 170 standard library 146 contents-of operator $^*$ 11 contract 40 contracts, $*{\mathrm{C++20}}$ 40 conversion 67 explicit type 53 narrowing 8 conversions, usual arithmetic 7 ConvertibleTo , concept 158 copy 68 assignment 66, 69 constructor 66, 69 cost of 70 elision 72 elision 66 memberwise 66 copy () 156 Copyable , concept 158 CopyConstr uctible , concept 158 copy_if () 156 Core Guidelines, $\mathrm{C++}$ 214 core language, $\mathrm{C++}$ 2 coroutine 211cos () 188

cosh () 188 cost of copy 70 of range checking 142 count () 156 count_if () 155–156 cout , output 3 <cstdlib> 110 C-style error handling 188 string 13

# D

\D , regex 119

\d , regex 119 d , regex 119 data race 196 data () , array 171 D&E 208 deadlock 199 deallocation 51 debugging template 100 declaration 5 function 4 in condition 61 interface 29 -declaration, using 34 declarator operator 12 decltype 215 decrement operator --7deduction guide 83, 176 retur n -type 44 default constructor 50 function argument 42 member initializer 68 operations 66 template argument 98$=$ default 66 DefaultConstr uctible , concept 158 definition implementation 30 of concepts 97 delegating constructor 215 $=$ delete 67 delete naked 52 operator 51 deprecated auto_ptr 218 feature 218 deque 146 derived class , base and 55 Der ivedFrom , concept 158

Destr uctible , concept 158 destructor 51, 66 ˜ 51 constructor and 210 vir tual 59 dictionary – see map difference from C 218 from C void $^*$ assignment 221 digit, [[:digit:]] 119 digit , regex 119

[[:digit:]] digit 119

-directive, using 35 dispatch, tag 181 distribution, random 191 divide operator / 6 domain error 188 double 5 duck typing 104 duration 179 duration_cast 179 dynamic store 51 dynamic_cast 61 is instance of 62 is kind of 62

# E

EDOM 188 element requirements 140 elision, copy 66 emplace_back () 147 empty () 147 enable_if 184 encapsulation, complete 66 end () 75, 143, 147, 150 engine, random 191 enum , bitset and 172 equal operator $==$ 6 equality preserving 159 Equality Comparable , concept 158 equal_range () 156, 173 ERANGE 188 erase () 143, 147 err no 188 error domain 188 handling 35 handling, C-style 188 range 188 recovery 38 run-time 35 error-code, exception vs 38 essential operations 66 evaluation compile-time 10

order of 7 example find_all () 151 Hello, Wor ld! 2 Rand_int 191 Vec 141 exception 35 and main () 141 catch ev ery 141 specification, removed 218 vs error-code 38 exclusive_scan () 189 execution policy 161 explicit type conversion 53 explicit constructor 67 exponential d is tr ibution 191 expor removed 218 expr () 188 expression constant 10 lambda 87 requires] 96 exter n template 215

# F

fabs () 188 facilities, standard library 108 fail*fast 170 feature, deprecated 218 features C with Classes language 210 $*{\mathrm{C++11}}$ language 215 $_{\mathrm{C++14}}$ language 216 $_{\mathrm{C++17}}$ language 216 file, header 31 final 216 find () 150, 156 find_all () example 151 find_if () 155–156 first , pair member 173 ﬂoor () 188 fmod () 188 for statement 11 statement, range 11 forward () 167 forwarding, perfect 168 ForwardIterator , concept 159 forward_list 146 singly-linked list 143 <forward_list> 109 ForwardRange , concept 160 free store 51 frexp () 188 <fstream> 109

**func** 215 function 2 and code complexity 4 argument 41 argument, default 42 argument passing 66 body 2 body, tr y block as 141 const member 50 constexpr 10 declaration 4 implementation of vir tual 56 mathematical 188 object 85 overloading 4 return value 41 template 84 type 181 value return 66 function 180 and nullptr 180 fundamental type 5 future and promise 202 member get () 202 <future> 109, 202

# G

garbage collection 73 generic programming 93, 210 get<>() by index 174 by type 174 get () , future member 202 graph , regex 119 greater-than operator $>\ensuremath{\mathrm{~\textrm~{~6~}~}}$ greater-than-or-equal operator $>=$ 6 greedy match 118, 121 grouping, {} 2 gsl namespace 168 span 168 Guidelines, $\mathrm{C++}$ Core 214

# H

half-open sequence 156 handle 52 resource 69, 165 hardware, mapping to 16 hash table 144 hash $<>$ , unordered_map 76 header C-library 110

file 31 standard library 109 heap 51 Hello, Wor ld! example 2 hierarchy class 57 navigation 61 history, $\mathrm{C++}$ 207 HOPL 208

# I

if statement 14 immutability const 9 constexpr 9 implementation definition 30 inheritance 60 iterator 153 of vir tual function 56 str ing 113 in-class member initialization 215 #include 31 inclusive_scan () 189 increment operator $^{++}\quad^{7}$ index, $\mathsf{g e t}\!>>\!\!\left (\right)$ by 174 inheritance 55 implementation 60 interface 60 multiple 211inheriting constructor 216 initialization and assignment 18 in-class member 215 initialize 52 array 171 initializer $=\quad7\quad$ {} 8 default member 68 narrowing, $\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\$ initializer-list constructor 52 initialize r list 52 inline 49 namespace 215 inlining 49 inner_product () 189 InputIterator , concept 159 InputRange , concept 160 inser t () 143, 147 instantiation 81 instruction, machine 16 int 5 output bits of 172 Integral , concept 158

interface declaration 29 inheritance 60 invariant 37 and constructor 37 Invocable , concept 159 In voc able Regular , concept 159 I/O, iterator and 154

<ios> 109

<iostream> 3, 109 iota () 189 is instance of, dynamic_cast 62 kind of, dynamic_cast 62 ISO $\mathrm{C++}$ 212 $\mathrm{C++}$ standard 2 ISO-14882 212 i stream iterator 154 iterator 75, 150 and I/O 154 implementation 153 Iterator , concept 159 iterator 143, 154 <iterator> 182 concepts in 158 iterator_categor y 182 iterator_traits 181–182 iterator_type 182

# J

join () , thread 196

# K

key and value 144 K&R C 219

# L

\L , regex 119

\l , regex 119 lambda as adaptor 180 expression 87 language and library 107 features, C with Classes 210 features, $_{\mathrm{C++11}}$ 215 features, $_{\textrm{C++14}}$ 216 features, $C{+}{+}17$ 216 launch, async () 204 lazy $+?$ 118

?? 118 {}? 118 ∗ ? 118 match 118, 121 ldexp () 188 leak, resource 62, 72, 164 less-than operator $<\quad^{6}$ less-than-or-equal operator $<=$ 6 letter, [[:alpha:]] 119 library algorithm, standard 156 C with Classes standard 211 $_{\textrm{C++98}}$ standard 211 components, $_{\mathrm{C++11}}$ 216 components, $_{\textrm{C++14}}$ 217 components, $C{+}{+}{1}{7}$ 217 container, standard 146 facilities, standard 108 language and 107 non-standard 107 standard 107 lifetime, scope and 9 lifting algorithm 100 <limits> 181, 193 linker 2 list capture 87 forward_list singly-linked 143 list 142, 146 literal " , string 3 raw string 116 suffix, s 113 suffix, sv 115 type of string 113 user-defined 75, 215 literals str ing_literals 113 str ing_view_literals 115 local scope 9 lock, reader-writer 200 log () 188 log10 () 188 long long 215 lower , regex 119

# M

machine instruction 16 main () 2 exception and 141 make_pair () 173 make_shared () 166 make_tuple () 174 make_unique () 166 management, resource 72, 164

map 144, 146 and unordered*map 146 <map> 109 mapped type, value 144 mapping to hardware 16 match greedy 118, 121 lazy 118, 121 mathematical function 188 functions, special 188 functions, standard 188 <math. h> 188 Max Munch rule 118 meaning, $\mathrm{C++}$ 209 member function, const 50 initialization, in-class 215 initializer, default 68 memberwise copy 66 mem_fn () 180 memory 73 address 16 <memor y> 109, 164, 166 merge () 156 Mergeable , concept 159 minus operator - 6 model, template compilation 104 modern $\mathrm{C++}$ 214 modf () 188 modularity 29 module suport 32 modules, $*{\mathrm{C++20}}$ 32 modulus operator $\%$ 6 Movable , concept 158 move 71 assignment 66, 72 constructor 66, 71 move () 72, 156, 167MoveConstr uctible , concept 158 moved-from object 72 state 168 move-only type 167 multi-line pattern 117 multimap 146 multiple character sets 114 inheritance 211return-values 44 multiply operator $^*$ 6 multiset 146 mutex 199 <mutex> 199

# N

$\backslash\mathsf{n}$ , newline 3 naked delete 52 new 52 namespace scope 9 namespace 34 chrono 179 gsl 168 inline 215 pmr 178 std 3, 35, 109 narrowing $=$ initializer 8 conversion 8 navigation, hierarchy 61 new container allocator 178 naked 52 operator 51 newline $\backslash\mathsf{n}$ 3 noexcept 37 noexcept () 215 non-memory resource 73 non-standard library 107 noretur n 215 nor mal distribution 191 notation, regular expression 117 not-equal operator $!=\quad6$ notify_one () , condition_var iable 202 NULL 0 , nullptr 13 nullptr 13 function and 180 NULL 0 13 number, random 191 <numer ic> 189 numerical algorithm 189 numer ic_limits 193

# O

object 5 function 85 in container 140 moved-from 72 object-oriented programming 57, 210 operations default 66 essential 66 operator $\%=\mathrm{~\ensuremath~{~7~}~}$ $\mathrel{+{=}}\phantom{-}7$ & , address-of 11 () , call 85 ∗ , contents-of 11

-- , decrement 7 / , divide 6 $==.$ , equal 6 $>$ , greater-than 6 $>=$ , greater-than-or-equal 6 $^{++}$ , increment 7 $<$ , less-than 6 $<=$ , less-than-or-equal 6 - , minus 6 $\%$ , modulus 6 ∗ , multiply 6 $\stackrel{!=}{}$ , not-equal 6 $<<$ , output 3 $^+$ , plus 6 $\%$ , remainder 6 $/=$ , scaling 7 $^{*}\!=$ , scaling 7 arithmetic 6 comparison 6, 74 declarator 12 delete 51 new 51 overloaded 51 user-defined 51 optimization, short-string 113 optional 176 order of evaluation 7 o stream iterator 154 out_of_range 141 output bits of int 172 cout 3 operator $<<\ensuremath{\mathrm{~\textrm~{~\,~}~}}3$ OutputIterator , concept 159 OutputRange , concept 160 overloaded operator 51 overloading concept based 95 function 4 overr ide 55 overview, container 146 ownership 164

# P

packaged_task thread 203 pair 173 and structured binding 174 member first 173 member second 173 par 161 parallel algorithm 161 parameterized type 79 par tial_sum () 189 par_unseq161passing data to task 197

pattern 116 (?: 120 multi-line 117 perfect forwarding 168 Permutable , concept 159 phone_book example 138 plus operator $^+$ 6 pmr , namespace 178 pointer 17 smart 164 to $^*$ 11 policy, execution 161 polymorphic type 54 pow () 188 precondition 37 predicate 86, 155 type 183 Predicate , concept 159 pr int , regex 119 procedural programming 2 program 2 programming generic 93, 210 object-oriented 57, 210 procedural 2 promise future and 202 member set_exception () 202 member set_value () 202 pronunciation, $\mathrm{C++}$ 209 punct , regex 119 pure vir tual 54 purpose, template 93 push_back () 52, 139, 143, 147 push_front () 143

# R

R" 116 race, data 196 RAII and resource management 36 and tr y -block 40 and tr y -statement 36 resource acquisition 164 scoped_lock and 199–200 RAII 52 Rand_int example 191 random number 191 random distribution 191 engine 191 <random> 109, 191 Random Access Iterator , concept 159 Random Access Range , concept 160 range

checking, cost of 142 checking Vec 140 error 188 for statement 11 Range concept 157 concept 160

<ranges> 157

<ranges> , concepts in 158 raw string literal 116 reader-writer lock 200 recovery, error 38 reduce () 189 reference 17 && , rvalue 71 rvalue 72 to & 12 regex ∗ 117 } 117 { 117 ) 117 | 117 ] 117 [ 117 ˆ 117 ? 117 . 117 $\S$ 117 + 117 ( 117 alnum 119 alpha 119 blank 119 cntr 119 d 119 \d 119 \D 119 digit119graph 119 \l 119 \L 119 lower 119 pr int 119 punct 119 regular expression 116 repetition 118 s 119 $\backslash\mathbb{S}$ 119 $\backslash\mathbb{S}$ 119 space 119 \U 119 \u 119 upper 119 w 119 \W 119

\w 119 xdigit 119 <regex> 109, 116 regular expression 116 regex_iterator 121 regex_search 116 regular expression notation 117 expression regex 116 expression <regex> 116 Regular , concept 158 reinter pret_cast 53 Relation , concept 159 remainder operator $\%$ 6 removed exception specification218expor 218 repetition, regex 118 replace () 156 str ing 112 replace_if () 156 requirement, template 94 requirements, element 140 requires] expression 96 reser ve () 139, 147 resize () 147 resource acquisition RAII 164 handle 69, 165 leak 62, 72, 164 management 72, 164 management, RAII and 36 non-memory 73 retention 73 safety 72 rethrow 38 return function value 66 type, suffix 215 value, function 41 retur n container 151 type, void 3 returning results from task 198 retur n -type deduction 44 return-values, multiple 44 riemanzeta () 188 rule Max Munch 118 of zero 67 run-time check 40 error 35 rvalue reference 72 reference && 71

# S

s literal suffix 113 \s , regex 119 s , regex 119 \S , regex 119 safety, resource 72 Same , concept 158 scaling operator $/=$ 7 operator $^{*}\!\underline{{=}}$ 7 scope and lifetime 9 class 9 local 9 namespace 9 scoped_lock 164 and RAII 199–200 unique_lock and 201 scoped_lock () 199 search, binary 156 second , pair member 173 Semiregular , concept 158 Sentinel , concept 159 separate compilation 30 sequence 150 half-open 156 set 146 <set> 109 set_exception () , promise member 202 set_value () , promise member 202 shared_lock 200 shared_mutex 200 shared_ptr 164 sharing data task 199 short-string optimization 113 SignedIntegral , concept 158 SIMD 161 Simula 207 sin () 188 singly-linked list, forward_list 143 sinh () 188 size of type 6 size () 75, 147 array 171 SizedRange , concept 160 SizedSentinel , concept 159 sizeof 6 sizeof () 181 size_t 90 smart pointer 164 smatch 116 sor t () 149, 156 container 181 Sor table , concept 159 space , regex 119 span

gsl 168 str ing*view and 168 special mathematical functions 188 specialized container 170 sphbessel () 188 sqr t () 188 <sstream> 109 standard ISO $\mathrm{C++}$ 2 library 107 library algorithm 156 library, $\mathrm{C++}$ 2 library, C with Classes 211 library, $*{\textrm{C++98}}$ 211 library container 146 library facilities 108 library header 109 library std 109 mathematical functions 188 standardization, $\mathrm{C++}$ 212 state, moved-from 168 statement for 11 if 14 range for 11 switch 14 while 14 static_asser 193 assertion 40 static_cast 53 std namespace 3, 35, 109 standard library 109 <stdexcept> 109 STL 211 store dynamic 51 free 51 Str ictTotallyOrdered , concept 158 Str ictWeakOrder , concept 159 string C-style 13 literal " 3 literal, raw 116 literal, type of 113 Unicode 114 str ing 111 [] 112 $==\ \ \ \ 112$ append $+=$ 112 assignment $=$ 112 concatenation $^+$ 111 implementation 113 replace () 112 substr () 112 <str ing> 109, 111

str ing_literals , literals 113 str ing_span 170 str ing_view 114 and span 168 str ing_view_literals , literals 115 structured binding 45 binding, pair and 174 binding, tuple and 174 subclass, superclass and 55 [] subscripting 147 substr () , str ing 112 suffix 75 return type 215 s literal 113 sv literal 115 superclass and subclass 55 suport, module 32 support, concept 94 sv literal suffix 115 sw ap () 76 Swappable , concept 158 SwappableWith , concept 158 switch statement 14 synchronized pool resource 178

# T

table, hash 144 tag dispatch 181 tanh () 188 task and thread 196 communication 202 passing data to 197 returning results from 198 sharing data 199 $\mathrm{TC++PL}$ 208 template 79 alias 184 argument, constrained 82 argument, default 98arguments, ${}>>{}$ 215 class 79 compilation model 104 constrained 82 debugging 100 exter n 215 function 84 purpose 93 requirement 94 variadic 100 this 70 thread join () 196 packaged_task 203

task and 196 <thread> 109, 196 thread_local 216 time 179 timeline, $\mathrm{C++}$ 208 time_point 179 timing, clock 200 to hardware, mapping 16 transfor m_reduce () 189 translation unit 32 tr y block 36 block as function body 141 tr y -block, RAII and 40 tr y -statement, RAII and 36 tuple 174 and structured binding 174 type 5 abstract 54 argument 82 concrete 48 conversion, explicit 53 function 181 fundamental 5 get<>() by 174 move-only 167 of string literal 113 parameterized 79 polymorphic 54 predicate 183 size of 6 typename 79, 152 <type_traits> 183 typing, duck 104

# U

\U , regex 119

\u , regex 119 udl 75 Unicode string 114 unifor m in t distribution 191 uninitialized 8 unique_copy () 149, 156 unique_lock 200–201 and scoped_lock 201 unique_ptr 62, 164 unordered_map 144, 146 hash<> 76 map and 146 <unordered_map> 109 unordered_multimap 146 unordered_multiset 146 unordered_set 146 unsigned 5 Unsigned Integral , concept 158

# 240 Index

upper , regex 119 use, concept 94 user-defined literal 75, 215 operator 51 using alias 90 -declaration 34 -directive 35 usual arithmetic conversions 7 <utility> 109, 173–174

while statement 14

X X3J16 212 xdigit , regex 119

Z zero, rule of 67

# V

valarray 192 <valarray>192value 5 argument 82 key and 144 mapped type 144 return, function 66 valuetype 147 value_type 90 variable 5 variadic template 100 variant 175 Vec example 141 range checking 140 vector arithmetic 192 vector 138, 146 array vs. 171 <vector> 109 vector<bool> 170 vectorized 161 View , concept 160 vir tual 54 destructor 59 function, implementation of 56 function table vtbl 56 pure 54 void ∗ 221 $^*$ assignment, difference from C 221 retur n type 3 vtbl , vir tual function table 56

# W

w , regex 119

\w , regex 119

\W , regex 119 wait () , condition_var iable 201 Weakly Equality Comparable , concept 158 WG21 208

# This page intentionally left blank

# More Guidance from the Inventor of C++

![](images/07d1afde091bee54ea32004b66f3e8580b284ba48abf8bd0f09147bac81c9840.jpg)

![](images/7d89ed228304fb76abdb574b5e31968f31a5d6fac5f2d2705a9a1d43719db13f.jpg)

The $c{\mathrel{+{+}}}$ Programming Language, Fourth Edition, delivers meticulous, richly explained, and integrated coverage of the entire language—its facilities, abstraction mechanisms, standard libraries, and key design techniques. Throughout, Stroustrup presents concise, “pure $\mathsf{C}_{\mathrm{++}}\uparrow\uparrow\prime$ examples, which have been carefully crafted to clarify both usage and program design.

Available in soft cover, hard cover, and eBook formats.

Programming: Principles and Practice Using $c{++,}$ , Second Edition, is a general introduction to programming, including object-oriented and generic programming, and a solid introduction to the $\mathsf{C}^{++}$ language. Stroustrup presents modern $\mathsf{C}^{++}$ techniques from the start, introducing the $\mathsf{C}^{++}$ standard library to simplify programming tasks.

Available in soft cover with lay-flat spine and eBook formats.

# informit. com/stroustrup
