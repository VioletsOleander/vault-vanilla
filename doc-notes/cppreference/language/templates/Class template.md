A class template defines a family of classes.

### Syntax
`template<parameter-list> class-declaration` (1)
`template<parameter-list> requires constraint class-declaration` (2) (since C++20)
`export template<parameter-list> class-declaration` (3) (removed in C++11)

### Explanation
`class-declaration` -  a [class declaration](https://en.cppreference.com/w/cpp/language/class "cpp/language/class"). The class name declared becomes a template name.
>  `class-declaration` 定义了模板名称

`parameter-list` -  a non-empty comma-separated list of the [template parameters](https://en.cppreference.com/w/cpp/language/template_parameters "cpp/language/template parameters"), each of which is either a [constant parameter](https://en.cppreference.com/w/cpp/language/template_parameters#Constant_template_parameter "cpp/language/template parameters"), a [type parameter](https://en.cppreference.com/w/cpp/language/template_parameters#Type_template_parameter "cpp/language/template parameters"), a [template parameter](https://en.cppreference.com/w/cpp/language/template_parameters#Template_template_parameter "cpp/language/template parameters"), or a [parameter pack](https://en.cppreference.com/w/cpp/language/parameter_pack "cpp/language/parameter pack") of any of those.
>  `parameter-list` 是一个非空的模板参数列表
>  列表中的参数可以是常量参数、类型参数、模板参数或者以上参数的参数包

`constraint` -  a [constraint expression](https://en.cppreference.com/w/cpp/language/constraints "cpp/language/constraints") which restricts the template parameters accepted by this class template
>  `constraint` 是约束表达式，用于约束该类模板能够接收的模板参数

`export` was an optional modifier which declared the template as _exported_ (when used with a class template, it declared all of its members exported as well). Files that instantiated exported templates did not need to include their definitions: the declaration was sufficient. Implementations of `export` were rare and disagreed with each other on details. (until C++11)

### Class template instantiation
A class template by itself is not a type, or an object, or any other entity. No code is generated from a source file that contains only template definitions. In order for any code to appear, a template must be instantiated: the template arguments must be provided so that the compiler can generate an actual class (or function, from a function template).
>  类模板本身不是类型、对象或其他实体
>  仅包含模板定义的源文件不会生成任何代码，要生成代码，模板必须被实例化
>  要实例化模板，必须提供模板参数，以便编译器可以生成实际的类

#### Explicit instantiation
`template class-key template-name<argument-list>` (1)
`extern template class-key template-name<argument-list>` (2) (since C++11)

`class-key` - `class` , `struct` or `union`

1) Explicit instantiation definition
2) Explicit instantiation declaration

>  语法 (1) 称为显式实例化定义
>  语法 (2) 称为显式实例化声明

An explicit instantiation definition forces instantiation of the class, struct, or union they refer to. It may appear in the program anywhere after the template definition, and for a given argument-list, is only allowed to appear once in the entire program, no diagnostic required.
>  显式实例化定义将实例化所引用的类、结构体、联合体
>  在模板定义之后的任何地方都可以使用显式实例化定义，但不允许使用重复的参数列表，即对于每个给定的参数列表，对应的显式实例化定义只能在整个程序中出现一次
>  如果违反，编译器将报错，无需诊断

An explicit instantiation declaration (an extern template) skips implicit instantiation step: the code that would otherwise cause an implicit instantiation instead uses the explicit instantiation definition provided elsewhere (resulting in link errors if no such instantiation exists). This can be used to reduce compilation times by explicitly declaring a template instantiation in all but one of the source files using it, and explicitly defining it in the remaining file. (since C++11)

Classes, functions, variables (since C++14), and member template specializations can be explicitly instantiated from their templates. Member functions, member classes, and static data members of class templates can be explicitly instantiated from their member definitions.

Explicit instantiation can only appear in the enclosing namespace of the template, unless it uses qualified-id:

```cpp
namespace N
{
    template<class T>
    class Y // template definition
    {
        void mf() {}
    };
}
 
// template class Y<int>; // error: class template Y not visible in the global namespace
using N::Y;
// template class Y<int>; // error: explicit instantiation outside
                          // of the namespace of the template
template class N::Y<char*>;       // OK: explicit instantiation
template void N::Y<double>::mf(); // OK: explicit instantiation
```

Explicit instantiation has no effect if an [explicit specialization](https://en.cppreference.com/w/cpp/language/template_specialization "cpp/language/template specialization") appeared before for the same set of template arguments.

Only the declaration is required to be visible when explicitly instantiating a function template, a variable template(since C++14), a member function or static data member of a class template, or a member function template. The complete definition must appear before the explicit instantiation of a class template, a member class of a class template, or a member class template, unless an explicit specialization with the same template arguments appeared before.

If a function template, variable template(since C++14), member function template, or member function or static data member of a class template is explicitly instantiated with an explicit instantiation definition, the template definition must be present in the same translation unit.

When an explicit instantiation names a class template specialization, it serves as an explicit instantiation of the same kind (declaration or definition) of each of its non-inherited non-template members that has not been previously explicitly specialized in the translation unit. If this explicit instantiation is a definition, it is also an explicit instantiation definition only for the members that have been defined at this point.

Explicit instantiation definitions ignore member access specifiers: parameter types and return types may be private.

#### Implicit instantiation
When code refers to a template in context that requires a completely defined type, or when the completeness of the type affects the code, and this particular type has not been explicitly instantiated, implicit instantiation occurs. For example, when an object of this type is constructed, but not when a pointer to this type is constructed.

This applies to the members of the class template: unless the member is used in the program, it is not instantiated, and does not require a definition.

```cpp
template<class T>
struct Z // template definition
{
    void f() {}
    void g(); // never defined
};
 
template struct Z<double>; // explicit instantiation of Z<double>
Z<int> a;                  // implicit instantiation of Z<int>
Z<char>* p;                // nothing is instantiated here
 
p->f(); // implicit instantiation of Z<char> and Z<char>::f() occurs here.
        // Z<char>::g() is never needed and never instantiated:
        // it does not have to be defined
```

If a class template has been declared, but not defined, at the point of instantiation, the instantiation yields an incomplete class type:

```cpp
template<class T>
class X;    // declaration, not definition
 
X<char> ch; // error: incomplete type X<char>
```

|   |   |
|---|---|
|[Local classes](https://en.cppreference.com/w/cpp/language/class#Local_classes "cpp/language/class") and any templates used in their members are instantiated as part of the instantiation of the entity within which the local class or enumeration is declared.|(since C++17)|

### Keywords

[export](https://en.cppreference.com/w/cpp/keyword/export "cpp/keyword/export")(until C++11)[extern](https://en.cppreference.com/w/cpp/keyword/extern "cpp/keyword/extern")(since C++11)

### See also

- [template parameters and arguments](https://en.cppreference.com/w/cpp/language/template_parameters "cpp/language/template parameters") allow templates to be parameterized
- [function template declaration](https://en.cppreference.com/w/cpp/language/function_template "cpp/language/function template") declares a function template
- [template specialization](https://en.cppreference.com/w/cpp/language/template_specialization "cpp/language/template specialization") defines an existing template for a specific type
- [parameter packs](https://en.cppreference.com/w/cpp/language/parameter_pack "cpp/language/parameter pack") allows the use of lists of types in templates (since C++11)