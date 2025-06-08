---
completed: true
---
## Introduction
TableGen’s purpose is to help a human develop and maintain records of domain-specific information. Because there may be a large number of these records, it is specifically designed to allow writing flexible descriptions and for common features of these records to be factored out. This reduces the amount of duplication in the description, reduces the chance of error, and makes it easier to structure domain specific information.
>  TableGen 的目的是帮助人类开发和维护领域特定的信息记录
>  因为记录的数量可能很多，故 TableGen 被设计为可以为这些记录抽象出的共同特征编写灵活的描述

The TableGen front end parses a file, instantiates the declarations, and hands the result off to a domain-specific [backend](https://llvm.org/docs/TableGen/index.html#backend) for processing. See the [TableGen Programmer’s Reference](https://llvm.org/docs/TableGen/ProgRef.html) for an in-depth description of TableGen. See [tblgen - Description to C++ Code](https://llvm.org/docs/CommandGuide/tblgen.html) for details on the `*-tblgen` commands that run the various flavors of TableGen.
>  TableGen 前端解析文件，生成声明，将结果交由领域特定的后端处理

The current major users of TableGen are [The LLVM Target-Independent Code Generator](https://llvm.org/docs/CodeGenerator.html) and the [Clang diagnostics and attributes](https://clang.llvm.org/docs/UsersManual.html#controlling-errors-and-warnings).

Note that if you work with TableGen frequently and use emacs or vim, you can find an emacs “TableGen mode” and a vim language file in the `llvm/utils/emacs` and `llvm/utils/vim` directories of your LLVM distribution, respectively.

## The TableGen program
TableGen files are interpreted by the TableGen program: `llvm-tblgen` available on your build directory under bin. It is not installed in the system (or where your sysroot is set to), since it has no use beyond LLVM’s build process.
>  TableGen 源文件由 `llvm-tblgen` 程序解析
>  `llvm-tblgen` 可以在 `build/bin` 下找到，该程序的使用范围不会超过 LLVM 的构建过程

### Running TableGen
TableGen runs just like any other LLVM tool. The first (optional) argument specifies the file to read. If a filename is not specified, `llvm-tblgen` reads from standard input.
>  和其他 LLVM 工具一样，`llvm-tblgen` 的第一个参数指定需要读取的文件，如果没有指定，则从标准输入中读取

To be useful, one of the [backends](https://llvm.org/docs/TableGen/index.html#backends) must be used. These backends are selectable on the command line (type ‘ `llvm-tblgen -help` ’ for a list). For example, to get a list of all of the definitions that subclass a particular type (which can be useful for building up an enum list of these records), use the `-print-enums` option:
>  要获取继承特定类型的所有定义和子类，可以使用 `-print-enums` 选项

```
$ llvm-tblgen X86.td -print-enums -class=Register
AH, AL, AX, BH, BL, BP, BPL, BX, CH, CL, CX, DH, DI, DIL, DL, DX, EAX, EBP, EBX,
ECX, EDI, EDX, EFLAGS, EIP, ESI, ESP, FP0, FP1, FP2, FP3, FP4, FP5, FP6, IP,
MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7, R10, R10B, R10D, R10W, R11, R11B, R11D,
R11W, R12, R12B, R12D, R12W, R13, R13B, R13D, R13W, R14, R14B, R14D, R14W, R15,
R15B, R15D, R15W, R8, R8B, R8D, R8W, R9, R9B, R9D, R9W, RAX, RBP, RBX, RCX, RDI,
RDX, RIP, RSI, RSP, SI, SIL, SP, SPL, ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7,
XMM0, XMM1, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, XMM2, XMM3, XMM4, XMM5,
XMM6, XMM7, XMM8, XMM9,
```

```
$ llvm-tblgen X86.td -print-enums -class=Instruction
ABS_F, ABS_Fp32, ABS_Fp64, ABS_Fp80, ADC32mi, ADC32mi8, ADC32mr, ADC32ri,
ADC32ri8, ADC32rm, ADC32rr, ADC64mi32, ADC64mi8, ADC64mr, ADC64ri32, ADC64ri8,
ADC64rm, ADC64rr, ADD16mi, ADD16mi8, ADD16mr, ADD16ri, ADD16ri8, ADD16rm,
ADD16rr, ADD32mi, ADD32mi8, ADD32mr, ADD32ri, ADD32ri8, ADD32rm, ADD32rr,
ADD64mi32, ADD64mi8, ADD64mr, ADD64ri32, ...
```

The default backend prints out all of the records. There is also a general backend which outputs all the records as a JSON data structure, enabled using the -dump-json option.
>  TableGen 的默认后端会打印出所有的 records
>  TableGen 的一个通用后端以 JSON 格式输出所有 records (`-dump-json` option)

If you plan to use TableGen, you will most likely have to write a [backend](https://llvm.org/docs/TableGen/index.html#backend) that extracts the information specific to what you need and formats it in the appropriate way. You can do this by extending TableGen itself in C++, or by writing a script in any language that can consume the JSON output.
>  要使用 TableGen，一般需要自己实现后端，该后端负责从源码提取所需的信息，并以相应的方式对其格式化
>  可以通过用 C++ 拓展 TableGen 本身来实现这一点，或者编写任何可以接收 JSON 输出的脚本来实现这一点

### Example
With no other arguments, `llvm-tblgen` parses the specified file and prints out all of the classes, then all of the definitions. This is a good way to see what the various definitions expand to fully. 
>  没有其他参数时，`llvm-tblgen` 会解析指定的文件，并输出所有的类，然后输出所有的定义
>  这可以用于查看各种定义完全展开的形式

Running this on the ` X86.td ` file prints this (at the time of this writing):
>  例如，`llvm-tblgen` 对 `x86.td` 文件的输出如下所示:

```
...
def ADD32rr {   // Instruction X86Inst I
  string Namespace = "X86";
  dag OutOperandList = (outs GR32:$dst);
  dag InOperandList = (ins GR32:$src1, GR32:$src2);
  string AsmString = "add{l}\t{$src2, $dst|$dst, $src2}";
  list<dag> Pattern = [(set GR32:$dst, (add GR32:$src1, GR32:$src2))];
  list<Register> Uses = [];
  list<Register> Defs = [EFLAGS];
  list<Predicate> Predicates = [];
  int CodeSize = 3;
  int AddedComplexity = 0;
  bit isReturn = 0;
  bit isBranch = 0;
  bit isIndirectBranch = 0;
  bit isBarrier = 0;
  bit isCall = 0;
  bit canFoldAsLoad = 0;
  bit mayLoad = 0;
  bit mayStore = 0;
  bit isImplicitDef = 0;
  bit isConvertibleToThreeAddress = 1;
  bit isCommutable = 1;
  bit isTerminator = 0;
  bit isReMaterializable = 0;
  bit isPredicable = 0;
  bit hasDelaySlot = 0;
  bit usesCustomInserter = 0;
  bit hasCtrlDep = 0;
  bit isNotDuplicable = 0;
  bit hasSideEffects = 0;
  InstrItinClass Itinerary = NoItinerary;
  string Constraints = "";
  string DisableEncoding = "";
  bits<8> Opcode = { 0, 0, 0, 0, 0, 0, 0, 1 };
  Format Form = MRMDestReg;
  bits<6> FormBits = { 0, 0, 0, 0, 1, 1 };
  ImmType ImmT = NoImm;
  bits<3> ImmTypeBits = { 0, 0, 0 };
  bit hasOpSizePrefix = 0;
  bit hasAdSizePrefix = 0;
  bits<4> Prefix = { 0, 0, 0, 0 };
  bit hasREX_WPrefix = 0;
  FPFormat FPForm = ?;
  bits<3> FPFormBits = { 0, 0, 0 };
}
...
```

This definition corresponds to the 32-bit register-register `add` instruction of the x86 architecture. `def ADD32rr` defines a record named `ADD32rr`, and the comment at the end of the line indicates the superclasses of the definition. 
>  上例的定义对应了 x86 架构中的 32 位寄存器-寄存器 `add` 指令
>  其中 `def ADD32rr` 定义了一个名为 `ADD32rr` 的记录，行末的注释指出了该定义的超类

The body of the record contains all of the data that TableGen assembled for the record, indicating that the instruction is part of the “X86” namespace, the pattern indicating how the instruction is selected by the code generator, that it is a two-address instruction, has a particular encoding, etc. The contents and semantics of the information in the record are specific to the needs of the X86 backend, and are only shown as an example.
>  record 的主体包含了 TableGen 为该 record 收集的所有数据: 表明了该指令属于 "X86" 命名空间，包含了指示代码生成器如何选择指令的模式，表明了该指令是双地址指令，表明了该指令具有特定的编码，等等
>  record 中的内容和语义是针对 X86 后端的需求设计的

As you can see, a lot of information is needed for every instruction supported by the code generator, and specifying it all manually would be unmaintainable, prone to bugs, and tiring to do in the first place. Because we are using TableGen, all of the information was derived from the following definition:
>  可以看到，代码生成器支持的每一条指令都需要大量信息，如果手动指定这些信息，将难以维护
>  而由于我们使用了 TableGen，故以上示例中的所有信息都来源于以下的定义：

```
let Defs = [EFLAGS],
    isCommutable = 1,                  // X = ADD Y,Z --> X = ADD Z,Y
    isConvertibleToThreeAddress = 1 in // Can transform into LEA.
def ADD32rr  : I<0x01, MRMDestReg, (outs GR32:$dst),
                                   (ins GR32:$src1, GR32:$src2),
                 "add{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (add GR32:$src1, GR32:$src2))]>;
```

This definition makes use of the custom class `I` (extended from the custom class `X86Inst`), which is defined in the X86-specific TableGen file, to factor out the common features that instructions of its class share. A key feature of TableGen is that it allows the end-user to define the abstractions they prefer to use when describing their information.
>  该定义利用了自定义类 `I` (继承自自定义类 `X86Inst`)，自定义类 `I` 在特定于 X86 的 TableGen 文件中定义，用于提取其类别中指令的共同特征
>  TableGen 的一个关键特性是：它允许用户自行定义他们在描述信息时个人偏好的抽象方式

## Syntax
TableGen has a syntax that is loosely based on C++ templates, with built-in types and specification. In addition, TableGen’s syntax introduces some automation concepts like multiclass, foreach, let, etc.
>  TableGen 的语法大致上基于 C++ 模板，具有内建类型和规范
>  此外，TableGen 的语法引入了一些自动化概念，例如 multiclass, foreach, let 等

### Basic concepts
TableGen files consist of two key parts: ‘classes’ and ‘definitions’, both of which are considered ‘records’.
>  TableGen 文件由两个关键部分组成：类和定义，二者都被视作 "record"

**TableGen records** have a unique name, a list of values, and a list of superclasses. The list of values is the main data that TableGen builds for each record; it is this that holds the domain specific information for the application. The interpretation of this data is left to a specific [backend](https://llvm.org/docs/TableGen/index.html#backend), but the structure and format rules are taken care of and are fixed by TableGen.
>  TableGen records
>  record 具有一个唯一的名称、一组值和一组超类
>  这组值是 TableGen 为该 record 构建的主要数据，它承载了应用的领域特定信息，对这组值的解释由特定的后端负责，但其结构和格式狗则由 TableGen 确定并固定下来

**TableGen definitions** are the concrete form of ‘records’. These generally do not have any undefined values, and are marked with the ‘ `def` ’ keyword.
>  TableGen definitions
>  TableGen definitions 是 “record” 的具体形式
>  这些定义中通常不会有未定义的值，定义以 `def` 关键字标记

```
def FeatureFPARMv8 : SubtargetFeature<"fp-armv8", "HasFPARMv8", "true", "Enable ARMv8 FP">;
```

In this example, FeatureFPARMv8 is `SubtargetFeature` record initialised with some values. 
>  上例中，`FeatureFPARMv8` 是一个初始化了一些值的 `SubtargetFeature` 记录

The names of the classes are defined via the keyword class either on the same file or some other included. Most target TableGen files include the generic ones in `include/llvm/Target`.
>  类的名称通过关键字 `class` 在同一文件或某些 included 的文件中定义 (类似头文件的用法)，大多数目标 TableGen 文件会 include 位于 `include/llvm/Target` 中的通用文件

**TableGen classes** are abstract records that are used to build and describe other records. These classes allow the end-user to build abstractions for either the domain they are targeting (such as “Register”, “RegisterClass”, and “Instruction” in the LLVM code generator) or for the implementor to help factor out common properties of records (such as “FPInst”, which is used to represent floating point instructions in the X86 backend). 
>  TableGen classes
>  TableGen 类是抽象的 record，用于构建和描述其他 records
>  类可以用于为目标领域 (例如 LLVM 代码生成器中的 “寄存器”，“寄存器类”，“指令”) 构建抽象，或者用于帮助实现者提取记录的通用属性 (例如 “FPInst”，用于表示 X86 后端中的浮点指令)

TableGen keeps track of all of the classes that are used to build up a definition, so the backend can find all definitions of a particular class, such as “Instruction”.
>  TableGen 跟踪所有用于构建一个定义的类，故后端可以找到一个特定的类的所有定义，例如 “Instruction”

```
class ProcNoItin<string Name, list<SubtargetFeature> Features>
      : Processor<Name, NoItineraries, Features>;
```

Here, the class ProcNoItin, receiving parameters Name of type string and a list of target features is specializing the class Processor by passing the arguments down as well as hard-coding NoItineraries.
>  上例中，类 `ProcNoltin` 接收类型为 `string` 的参数 `Name` ，以及一个目标特性列表，通过将参数传递下去并硬编码 `Noltineraries` 来特化类 `Processor`

**TableGen multiclasses** are groups of abstract records that are instantiated all at once. Each instantiation can result in multiple TableGen definitions. If a multiclass inherits from another multiclass, the definitions in the sub-multiclass become part of the current multiclass, as if they were declared in the current multiclass.
>  TableGen multiclasses
>  TableGen 多类是一组抽象记录的集合，它会会被一次性实例化，每次实例化可能会生成多个 TableGen 定义
>  如果一个多类继承自另一个多类，子多类中的定义会称为当前多类中的一部分，就好像它们声明在当前多类中一样

```
multiclass ro_signed_pats<string T, string Rm, dag Base, dag Offset, dag Extend, dag address, ValueType sty> {
def : Pat<(i32 (!cast<SDNode>("sextload" # sty) address)),
          (!cast<Instruction>("LDRS" # T # "w_" # Rm # "_RegOffset")
            Base, Offset, Extend)>;

def : Pat<(i64 (!cast<SDNode>("sextload" # sty) address)),
          (!cast<Instruction>("LDRS" # T # "x_" # Rm # "_RegOffset")
            Base, Offset, Extend)>;
}

defm : ro_signed_pats<"B", Rm, Base, Offset, Extend,
                      !foreach(decls.pattern, address,
                               !subst(SHIFT, imm_eq0, decls.pattern)),
                      i8>;
```

See the [TableGen Programmer’s Reference](https://llvm.org/docs/TableGen/ProgRef.html) for an in-depth description of TableGen.

## TableGen backends
TableGen files have no real meaning without a backend. The default operation when running `*-tblgen` is to print the information in a textual format, but that’s only useful for debugging the TableGen files themselves. 
>  TableGen 文件没有后端，就没有实际意义
>  运行 `*-tblgen` 的默认操作是将信息以文本格式输出，但这仅对调试 TableGen 文件本身有用

The power in TableGen is, however, to interpret the source files into an internal representation that can be generated into anything you want.
>  TableGen 的强大之处在于将其源文件解释为一个内部表现形式，然后可以生成我们想要的任何内容

Current usage of TableGen is to create huge include files with tables that you can either include directly (if the output is in the language you’re coding), or be used in pre-processing via macros surrounding the include of the file.
>  目前 TableGen 的主要用途是创建包含大量表格的大型头文件，这些表格可以直接 include (如果输出的语言是我们正在使用的语言)，或者通过围绕文件的 include 的宏在预处理阶段使用

Direct output can be used if the backend already prints a table in C format or if the output is just a list of strings (for error and warning messages). 
>  如果后端已经以 C 形式打印了表格，或者输出只是一个字符串列表 (用于错误和警告消息)，则可以使用直接输出

Pre-processed output should be used if the same information needs to be used in different contexts (like Instruction names), so your backend should print a meta-information list that can be shaped into different compile-time formats.
>  如果相同的信息需要在不同的上下文使用 (如指令名称)，则应使用预处理输出
>  因此，我们实现的后端应该打印一个元信息列表，该列表可以通过不同的编译时格式进行调整

See [TableGen BackEnds](https://llvm.org/docs/TableGen/BackEnds.html) for a list of available backends, and see the [TableGen Backend Developer’s Guide](https://llvm.org/docs/TableGen/BackGuide.html) for information on how to write and debug a new backend.

## Tools and Resources
In addition to this documentation, a list of tools and resources for TableGen can be found in TableGen’s [README](https://github.com/llvm/llvm-project/blob/main/llvm/utils/TableGen/README.md).

## TableGen Deficiencies
Despite being very generic, TableGen has some deficiencies that have been pointed out numerous times. The common theme is that, while TableGen allows you to build domain specific languages, the final languages that you create lack the power of other DSLs, which in turn increase considerably the size and complexity of TableGen files.
>  TableGen 非常通用，但也存在缺陷
>  TableGen 允许我们创建领域特定的语言，但最终创造出的语言缺乏其他 DSL 强大的功能，反过来增加了 TableGen 文件的规模和复杂性

At the same time, TableGen allows you to create virtually any meaning of the basic concepts via custom-made backends, which can pervert the original design and make it very hard for newcomers to understand the evil TableGen file.
>  TableGen 允许通过定制后端为基本概念赋予几乎任何含义，但这可能会扭曲原始设计，并使新用户难以理解那些复杂的 TableGen 文件

There are some in favor of extending the semantics even more, but making sure backends adhere to strict rules. Others are suggesting we should move to less, more powerful DSLs designed with specific purposes, or even reusing existing DSLs