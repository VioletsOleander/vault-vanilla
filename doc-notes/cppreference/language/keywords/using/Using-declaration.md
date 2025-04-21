---
version: 2024.8.12
completed:
---
>  using-声明适用于类成员

Introduces a name that is defined elsewhere into the declarative region where this using-declaration appears. 
>  using-声明用于将定义于其他地方的名字引入 using-声明所处的声明区中

See [using enum](https://en.cppreference.com/w/cpp/language/enum#Using-enum-declaration "cpp/language/enum") and (since C++20) [using namespace](https://en.cppreference.com/w/cpp/language/namespace#Using-directives "cpp/language/namespace") for other related declarations.

`using typename(optional) nested-name-specifier unqualified-id;` (until C++17)
`using declarator-list;` (since C++17)

- `typename` : the keyword typename may be used as necessary to resolve [dependent names](https://en.cppreference.com/w/cpp/language/dependent_name "cpp/language/dependent name"), when the using-declaration introduces a member type from a base class into a class template
- `nested-name-specifier` : a sequence of names and scope resolution operators `::`, ending with a scope resolution operator. A single `::` refers to the global namespace.
>  嵌套名说明符：名字与作用域解析运算符 `::` 的序列，单个 `::` 指全局命名空间
>  (例如 `std::`)

- `unqualified-id` : an [id-expression](https://en.cppreference.com/w/cpp/language/identifiers "cpp/language/identifiers")
>  无限定标识：一个标识表达式
>  (例如 `string`)

- `declarator-list` : comma-separated list of one or more declarators of the `typename` (optional) nested-name-specifier unqualified-id. Some or all of the declarators may be followed by an ellipsis ... to indicate [pack expansion](https://en.cppreference.com/w/cpp/language/parameter_pack "cpp/language/parameter pack")

### Explanation
Using-declarations can be used to introduce namespace members into other namespaces and block scopes, or to introduce base class members into derived class definitions, or to introduce [enumerators](https://en.cppreference.com/w/cpp/language/enum "cpp/language/enum") into namespaces, block, and class scopes(since C++20).
>  using-声明将命名空间成员引入另一个命名空间和块作用域，
>  或者将基类成员引入衍生类定义，
>  或者将枚举项引入命名空间、块、或类作用域中 (since C++20)

A using-declaration with more than one using-declarator is equivalent to a corresponding sequence of using-declarations with one using-declarator. (since C++17)

#### In namespace and block scope
[Using-declarations](https://en.cppreference.com/w/cpp/language/namespace#Using-declarations "cpp/language/namespace") introduce a member of another namespace into the current namespace or block scope.
>  using-声明可以将另一个命名空间的成员引入当前命名空间或块作用域

```cpp
#include <iostream>
#include <string>

using std::string;

int main()
{
    string str = "Example";
    using std::cout;
    cout<<str;
}
```

See [namespace](https://en.cppreference.com/w/cpp/language/namespace "cpp/language/namespace") for details.

#### In class definition
Using-declaration introduces a member of a base class into the derived class definition, such as to expose a protected member of base as public member of derived. In this case, nested-name-specifier must name a base class of the one being defined. If the name is the name of an overloaded member function of the base class, all base class member functions with that name are introduced. If the derived class already has a member with the same name, parameter list, and qualifications, the derived class member hides or overrides (doesn't conflict with) the member that is introduced from the base class.
>  using-声明可以将基类的成员引入到派生类定义中，例如将基类的保护成员暴露为派生类的公开成员
>  此时，nested-name-specifier 必须是所定义的类的某个基类
>  如果引入的名字是基类的某个重载的成员函数的名字，则具有该名字的所有基类成员函数都被引入 (名字相同的所有重载成员函数都被引入)
>  如果派生类已经包含了相同名字的成员、参数列表、限定的成员，派生类成员会隐藏或覆盖从基类引入的名字 (也就是引入无效)

```cpp
#include <iostream>
 
struct B
{
    virtual void f(int) { std::cout << "B::f\n"; }
    void g(char)        { std::cout << "B::g\n"; }
    void h(int)         { std::cout << "B::h\n"; }
protected:
    int m; // B::m is protected
    typedef int value_type;
};
 
struct D : B
{
    using B::m;          // D::m is public
    using B::value_type; // D::value_type is public

    using B::f;
    void f(int) override { std::cout << "D::f\n"; } // D::f(int) overrides B::f(int)

    using B::g;
    void g(int) { std::cout << "D::g\n"; } // both g(int) and g(char) are visible

    using B::h;
    void h(int) { std::cout << "D::h\n"; } // D::h(int) hides B::h(int)
};
 
int main()
{
    D d;
    B& b = d;
 
//  b.m = 2;  // Error: B::m is protected
    d.m = 1;  // protected B::m is accessible as public D::m
    
    b.f(1);   // calls derived f()
    d.f(1);   // calls derived f()
    std::cout << "----------\n";
    
    d.g(1);   // calls derived g(int)
    d.g('a'); // calls base g(char), exposed via using B::g;
    std::cout << "----------\n";
    
    b.h(1);   // calls base h()
    d.h(1);   // calls derived h()
}
```

#### Inheriting constructors
If the _using-declaration_ refers to a constructor of a direct base of the class being defined (e.g. using Base:: Base;), all constructors of that base (ignoring member access) are made visible to overload resolution when initializing the derived class.
>  如果 using-声明指代了当前类的某个直接基类的构造函数，则在初始化派生类时 (创建派生类对象)，基类的所有构造函数 (忽略成员访问) 均对重载决议可见 
>  (也就是此时不管私有公有还是保护的基类构造函数都可以被用于构造派生类)

> [!info]
> 重载解析/决议是指调用函数或构造函数时，选择最合适的重载版本的过程

If overload resolution selects an inherited constructor, it is accessible if it would be accessible when used to construct an object of the corresponding base class: the accessibility of the using-declaration that introduced it is ignored.

If overload resolution selects one of the inherited constructors when initializing an object of such derived class, then the `Base` subobject from which the constructor was inherited is initialized using the inherited constructor, and all other bases and members of `Derived` are initialized as if by the defaulted default constructor (default member initializers are used if provided, otherwise default initialization takes place). The entire initialization is treated as a single function call: initialization of the parameters of the inherited constructor is [sequenced before](https://en.cppreference.com/w/cpp/language/eval_order "cpp/language/eval order") initialization of any base or member of the derived object.

```cpp
struct B1 { B1(int, ...) {} };
struct B2 { B2(double)   {} };
 
int get();
 
struct D1 : B1
{
    using B1::B1; // inherits B1(int, ...)
    int x;
    int y = get();
};
 
void test()
{
    D1 d(2, 3, 4); // OK: B1 is initialized by calling B1(2, 3, 4),
                   // then d.x is default-initialized (no initialization is performed),
                   // then d.y is initialized by calling get()
 
    D1 e;          // Error: D1 has no default constructor
}
 
struct D2 : B2
{
    using B2::B2; // inherits B2(double)
    B1 b;
};
 
D2 f(1.0); // error: B1 has no default constructor

struct W { W(int); };
 
struct X : virtual W
{
    using W::W; // inherits W(int)
    X() = delete;
};
 
struct Y : X
{
    using X::X;
};
 
struct Z : Y, virtual W
{
    using Y::Y;
};
 
Z z(0); // OK: initialization of Y does not invoke default constructor of X
```

If the `Base` base class subobject is not to be initialized as part of the `Derived` object (i.e., `Base` is a [virtual base class](https://en.cppreference.com/w/cpp/language/derived_class#Virtual_base_classes "cpp/language/derived class") of `Derived`, and the `Derived` object is not the [most derived object](https://en.cppreference.com/w/cpp/language/object#Subobjects "cpp/language/object")), the invocation of the inherited constructor, including the evaluation of any arguments, is omitted:

```
struct V
{
    V() = default;
    V(int);
};
 
struct Q { Q(); };
 
struct A : virtual V, Q
{
    using V::V;
    A() = delete;
};
 
int bar() { return 42; }
 
struct B : A
{
    B() : A(bar()) {} // OK
};
 
struct C : B {};
 
void foo()
{
    C c; // “bar” is not invoked, because the V subobject
         // is not initialized as part of B
         // (the V subobject is initialized as part of C,
         //  because “c” is the most derived object)
}
```

If the constructor was inherited from multiple base class subobjects of type `Base`, the program is ill-formed, similar to multiply-inherited non-static member functions:

```
struct A { A(int); };
struct B : A { using A::A; };
struct C1 : B { using B::B; };
struct C2 : B { using B::B; };
 
struct D1 : C1, C2
{
    using C1::C1;
    using C2::C2;
};
D1 d1(0); // ill-formed: constructor inherited from different B base subobjects
 
struct V1 : virtual B { using B::B; };
struct V2 : virtual B { using B::B; };
 
struct D2 : V1, V2
{
    using V1::V1;
    using V2::V2;
};
D2 d2(0); // OK: there is only one B subobject.
          // This initializes the virtual B base class,
          //   which initializes the A base class
          // then initializes the V1 and V2 base classes
          //   as if by a defaulted default constructor
```

As with using-declarations for any other non-static member functions, if an inherited constructor matches the signature of one of the constructors of `Derived`, it is hidden from lookup by the version found in `Derived`. If one of the inherited constructors of `Base` happens to have the signature that matches a copy/move constructor of the `Derived`, it does not prevent implicit generation of `Derived` copy/move constructor (which then hides the inherited version, similar to `using operator=`).

```
struct B1 { B1(int); };
struct B2 { B2(int); };
 
struct D2 : B1, B2
{
    using B1::B1;
    using B2::B2;
 
    D2(int); // OK: D2::D2(int) hides both B1::B1(int) and B2::B2(int)
};
D2 d2(0);    // calls D2::D2(int)
```

Within a [templated class](https://en.cppreference.com/w/cpp/language/templates "cpp/language/templates"), if a using-declaration refers to a [dependent name](https://en.cppreference.com/w/cpp/language/dependent_name "cpp/language/dependent name"), it is considered to name a constructor if the nested-name-specifier has a terminal name that is the same as the unqualified-id.

```
template<class T>
struct A : T
{
    using T::T; // OK, inherits constructors of T
};
 
template<class T, class U>
struct B : T, A<U>
{
    using A<U>::A; // OK, inherits constructors of A<U>
    using T::A;    // does not inherit constructor of T
                   // even though T may be a specialization of A<>
};
```

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| #### Introducing scoped enumerators<br><br>In addition to members of another namespace and members of base classes, using-declaration can also introduce enumerators of [enumerations](https://en.cppreference.com/w/cpp/language/enum "cpp/language/enum") into namespace, block, and class scopes.<br><br>A using-declaration can also be used with unscoped enumerators.<br><br>enum class button { up, down };<br> <br>struct S<br>{<br>    using button::up;<br>    button b = up; // OK<br>};<br> <br>using button::down;<br>constexpr button non_up = down; // OK<br> <br>constexpr auto get_button(bool is_up)<br>{<br>    using button::up, button::down;<br>    return is_up ? up : down; // OK<br>}<br> <br>enum unscoped { val };<br>using unscoped::val; // OK, though needless | (since C++20) |

### Notes

Only the name explicitly mentioned in the using-declaration is transferred into the declarative scope: in particular, enumerators are not transferred when the enumeration type name is using-declared.

A using-declaration cannot refer to a namespace, to a scoped enumerator(until C++20), to a destructor of a base class or to a specialization of a member template for a user-defined conversion function.

A using-declaration cannot name a member template specialization ([template-id](https://en.cppreference.com/w/cpp/language/templates#template-id "cpp/language/templates") is not permitted by the grammar):

struct B
{
    template<class T>
    void f();
};
 
struct D : B
{
    using B::f;      // OK: names a template
//  using B::f<int>; // Error: names a template specialization
 
    void g() { f<int>(); }
};

A using-declaration also can't be used to introduce the name of a dependent member template as a _template-name_ (the `template` disambiguator for [dependent names](https://en.cppreference.com/w/cpp/language/dependent_name "cpp/language/dependent name") is not permitted).

template<class X>
struct B
{
    template<class T>
    void f(T);
};
 
template<class Y>
struct D : B<Y>
{
//  using B<Y>::template f; // Error: disambiguator not allowed
    using B<Y>::f;          // compiles, but f is not a template-name
 
    void g()
    {
//      f<int>(0);          // Error: f is not known to be a template name,
                            // so < does not start a template argument list
        f(0);               // OK
    }   
};

If a using-declaration brings the base class assignment operator into derived class, whose signature happens to match the derived class's copy-assignment or move-assignment operator, that operator is hidden by the implicitly-declared copy/move assignment operator of the derived class. Same applies to a using-declaration that inherits a base class constructor that happens to match the derived class copy/move constructor(since C++11).

|   |   |
|---|---|
|The semantics of inheriting constructors were retroactively changed by a [defect report against C++11](https://en.cppreference.com/w/cpp/language/using_declaration#Defect_reports). Previously, an inheriting constructor declaration caused a set of synthesized constructor declarations to be injected into the derived class, which caused redundant argument copies/moves, had problematic interactions with some forms of SFINAE, and in some cases can be unimplementable on major ABIs. Older compilers may still implement the previous semantics.<br><br>\|[[Expand](https://en.cppreference.com/w/cpp/language/using_declaration#)] Old inheriting constructor semantics\|<br>\|---\||(since C++11)|

|   |   |
|---|---|
|[Pack expansions](https://en.cppreference.com/w/cpp/language/parameter_pack "cpp/language/parameter pack") in using-declarations make it possible to form a class that exposes overloaded members of variadic bases without recursion:<br><br>template<typename... Ts><br>struct Overloader : Ts...<br>{<br>    using Ts::operator()...; // exposes operator() from every base<br>};<br> <br>template<typename... T><br>Overloader(T...) -> Overloader<T...>; // C++17 deduction guide, not needed in C++20<br> <br>int main()<br>{<br>    auto o = Overloader{ [] (auto const& a) {[std::cout](http://en.cppreference.com/w/cpp/io/cout) << a;},<br>                         [] (float f) {[std::cout](http://en.cppreference.com/w/cpp/io/cout) << [std::setprecision](http://en.cppreference.com/w/cpp/io/manip/setprecision)(3) << f;} };<br>}|(since C++17)|

|Feature-test macro|Value|Std|Feature|
|---|---|---|---|
|[`__cpp_inheriting_constructors`](https://en.cppreference.com/w/cpp/feature_test#cpp_inheriting_constructors "cpp/feature test")|[`200802L`](https://en.cppreference.com/w/cpp/compiler_support/11#cpp_inheriting_constructors_200802L "cpp/compiler support/11")|(C++11)|[Inheriting constructors](https://en.cppreference.com/w/cpp/language/using_declaration#Inheriting_constructors)|
|[`201511L`](https://en.cppreference.com/w/cpp/compiler_support/11#cpp_inheriting_constructors_201511L "cpp/compiler support/11")|(C++11)  <br>(DR)|Rewording inheriting constructors|
|[`__cpp_variadic_using`](https://en.cppreference.com/w/cpp/feature_test#cpp_variadic_using "cpp/feature test")|[`201611L`](https://en.cppreference.com/w/cpp/compiler_support/17#cpp_variadic_using_201611L "cpp/compiler support/17")|(C++17)|[Pack expansions](https://en.cppreference.com/w/cpp/language/parameter_pack "cpp/language/parameter pack") in `using`-declarations|

### Keywords

[using](https://en.cppreference.com/w/cpp/keyword/using "cpp/keyword/using")

### Defect reports

The following behavior-changing defect reports were applied retroactively to previously published C++ standards.

|DR|Applied to|Behavior as published|Correct behavior|
|---|---|---|---|
|[CWG 258](https://cplusplus.github.io/CWG/issues/258.html)|C++98|a non-const member function of a derived class can  <br>override and/or hide a const member function of its base|overriding and hiding also require  <br>cv-qualifications to be the same|
|[CWG 1738](https://cplusplus.github.io/CWG/issues/1738.html)|C++11|it was not clear whether it is permitted to  <br>explicitly instantiate or explicitly specialize  <br>specializations of inheriting constructor templates|prohibited|
|[CWG 2504](https://cplusplus.github.io/CWG/issues/2504.html)|C++11|the behavior of inheriting constructors  <br>from virtual base classes was unclear|made clear|
|[P0136R1](https://wg21.link/P0136R1)|C++11|inheriting constructor declaration injects  <br>additional constructors in the derived class|causes base class constructors  <br>to be found by name lookup|

1. References

### References

- C++23 standard (ISO/IEC 14882:2024):

- 9.9 The `using` declaration [namespace.udecl]

- C++20 standard (ISO/IEC 14882:2020):

- 9.9 The `using` declaration [namespace.udecl]

- C++17 standard (ISO/IEC 14882:2017):

- 10.3.3 The `using` declaration [namespace.udecl]

- C++14 standard (ISO/IEC 14882:2014):

- 7.3.3 The `using` declaration [namespace.udecl]

- C++11 standard (ISO/IEC 14882:2011):

- 7.3.3 The `using` declaration [namespace.udecl]

- C++03 standard (ISO/IEC 14882:2003):

- 7.3.3 The `using` declaration [namespace.udecl]

- C++98 standard (ISO/IEC 14882:1998):

- 7.3.3 The `using` declaration [namespace.udecl]