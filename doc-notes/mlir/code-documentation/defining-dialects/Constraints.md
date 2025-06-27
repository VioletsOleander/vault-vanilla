---
completed: true
---
- [Attribute / Type Constraints](https://mlir.llvm.org/docs/DefiningDialects/Constraints/#attribute--type-constraints)

## Attribute / Type Constraints 
When defining the arguments of an operation in TableGen, users can specify either plain attributes/types or use attribute/type constraints to levy additional requirements on the attribute value or operand type.

```tablegen
def My_Type1 : MyDialect_Type<"Type1", "type1"> { ... }
def My_Type2 : MyDialect_Type<"Type2", "type2"> { ... }

// Plain type
let arguments = (ins MyType1:$val);
// Type constraint
let arguments = (ins AnyTypeOf<[MyType1, MyType2]>:$val);
```

>  在 TableGen 中定义 operation 的 argument 时，用户可以指定普通的属性/类型，也可以使用属性/类型约束来施加额外的要求

`AnyTypeOf` is an example for a type constraints. Many useful type constraints can be found in `mlir/IR/CommonTypeConstraints.td`. 
>  上述示例中的 `AnyTypeOf` 就是一个类型约束
>  `mlir/IR/CommonTypeConstraints.td` 中定义了许多类型约束

Additional verification code is generated for type/attribute constraints. Type constraints can not only be used when defining operation arguments, but also when defining type parameters.
>  MLIR 会为具有类型/属性约束的 operation arguments 生成额外的验证代码
>  类型约束除了在定义 operation arguments 时可以使用，也可以在定义类型参数时使用

Optionally, C++ functions can be generated, so that type/attribute constraints can be checked from C++. The name of the C++ function must be specified in the `cppFunctionName` field. If no function name is specified, no C++ function is emitted.

```tablegen
// Example: Element type constraint for VectorType
def Builtin_VectorTypeElementType : AnyTypeOf<[AnyInteger, Index, AnyFloat]> {
  let cppFunctionName = "isValidVectorTypeElementType";
}
```

The above example tranlates into the following C++ code:

```c++
bool isValidVectorTypeElementType(::mlir::Type type) {
  return (((::llvm::isa<::mlir::IntegerType>(type))) || ((::llvm::isa<::mlir::IndexType>(type))) || ((::llvm::isa<::mlir::FloatType>(type))));
}
```

>  指定了类型/属性约束后，MLIR 会在内部生成并集成类型验证逻辑
>  如果我们想要在其他地方调用 MLIR 生成的类型验证逻辑，可以指定 `cppFunctionName` 字段，让 MLIR 另外生成一个用于验证的 C++ 函数

An extra TableGen rule is needed to emit C++ code for type/attribute constraints. This will generate only the declarations/definitions of the type/attribute constaraints that are defined in the specified `.td` file, but not those that are in included `.td` files.

```cmake
mlir_tablegen(<Your Dialect>TypeConstraints.h.inc -gen-type-constraint-decls)
mlir_tablegen(<Your Dialect>TypeConstraints.cpp.inc -gen-type-constraint-defs)
mlir_tablegen(<Your Dialect>AttrConstraints.h.inc -gen-attr-constraint-decls)
mlir_tablegen(<Your Dialect>AttrConstraints.cpp.inc -gen-attr-constraint-defs)
```

The generated `<Your Dialect>TypeConstraints.h.inc` respectivelly `<Your Dialect>AttrConstraints.h.inc` will need to be included whereever you are referencing the type/attributes constraint in C++. Note that no C++ namespace will be emitted by the code generator. The `#include` statements of the `.h.inc`/`.cpp.inc` files should be wrapped in C++ namespaces by the user.