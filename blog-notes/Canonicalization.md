---
completed: true
---
Site: https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html
Date: 22 Nov 2018

Canonicalization and canonical forms are one dimension of organizing the work of an optimizing compiler.
> 本文讨论编译器优化中的 "规范化" 和 "规范形式""

## Intro
A lot of code constructs can be written in multiple ways. For example:
>  许多的代码结构可以以多种形式表述，例如下例展示了 `x + 4` 的三种表示方式

```
   x + 4
   4 + x
   (x + 2) + 2
```

_Canonicalization_ means picking one of these forms to be the _canonical form_, and then going through the program and rewriting all constructs which are equivalent to the canonical form into the canonical form.
>  “规范化” 意为选择这些表示方式的其中一种形式作为 "规范形式"，然后遍历程序，将所有等价于规范形式的结构都重写为规范形式

In the case of add, it’s common to pick the form that has a constant on the right side, so we’d rewrite all these constructs to `x + 4`.
>  例如上例的加法运算中，我们一般选择的形式是常数位于右边的形式，故我们会将所有这样的结构重写为 `x + 4` 的形式

### Why is canonicalization useful?
The goal of canonicalization is _to make subsequent optimizations more effective_. This is a key point, and we’ll get into some of the subtleties below. But there are a lot of cases where it’s just obviously a good thing to do. It means that subsequent optimizations that look for specific patterns of code only have to look for the canonical forms, rather than all forms.
>  规范化的目标是让随后的优化更加高效
>  例如，一个显然的点是随后的查找特定代码模式的相关优化只需要查找规范形式，而不是其他形式

Another way of saying this is, having a canonicalization pass is a way of factoring out the parts in a compiler that know all the different forms `x + 4` could take, so that most optimization passes don’t have to worry about this. They don’t have to look for `4 + x`, because they can assume that looking for `x + 4` covers that. Handy!
>  换句话说，规范化 pass 是将编译器所了解的 `x + 4` 的所有不同形式分离出来，使得之后的优化 pass 不需要考虑这一点
>  它们不需要查找 `4 + x` 这样的形式，因为它们直接假定了查找 `x + 4` 可以覆盖所有情况

### How do we choose a canonical form?
Sometimes it’s easy. The canonical form for `2 + 3` is `5`, because that’s clearly simpler in every possible way.

Sometimes it’s an arbitrary choice. It often doesn’t matter than much whether one picks `4 + x` over `x + 4`, but it is helpful to pick one or the other, so sometimes it’s just human aesthetics.

It’s tempting to pick whatever form would be fastest, or _optimal_, on the target machine. And indeed, sometimes what’s fastest aligns with what’s simplest. `2 + 3` canonicalizing `5` is typically such a case. But sometimes it doesn’t.

>  通常我们倾向于选择在目标机器上最快或最优的形式为规范形式，并且有时最简单的方案就是最快的方案，例如 `2 + 3` 的最快形式就是 `5` ，当然有时也并非如此

### Yeah so what about `x * 2`…
`x * 2` is equivalent to `x + x`; which of these should be the canonical form? It might seem like we might want to say: pick whatever’s optimal for the target architecture. Addition is generally faster than multiplication, so that would suggest we pick `x + x` as the canonical form.
>  我们再考虑 `x * 2` ，它等价于 `x + x` ，那么我们应该选择哪一个作为规范形式
>  似乎较优的方式就是选择针对目标架构最优的形式，通常加法快于乘法，故我们可能会选择 `x + x` 作为规范形式

But, `x + x` can actually make things harder for subsequent optimizations, because it means that now `x` has multiple uses. Having multiple uses makes some optimizations more complex – in terms of the dependence graph, this is a DAG rather than a tree, and trees are generally simpler to work with. So maybe `x * 2` is actually a better canonical form, even if it’s a worse optimal form.
>  但 `x + x` 实际上会让后续的优化更加困难，因为它意味着此时 `x` 有了多次使用
>  多次使用 `x` 会让一些优化更加复杂——从依赖图的角度来看，依赖图将会成为一个有向无环图而不是树，而树通常更容易处理
>  因此，也许 `x * 2` 才是更好的规范形式，即使它并不是计算上最优的

That said, we might consider canonicalizing this to `x << 1`, which has only one use of `x`, and has the nice property of making it as obvious as possible that the least significant bit of the result is zero. That way, we don’t have to have as much random knowledge of multiplication by powers of two strewn throughout the compiler.
>  实际上，我们可以考虑规范形式为 `x << 1`，该形式仅使用一次 `x` ，并且可以从移位操作的语义中，很容易读出该计算的结果的最低有效位显然为 0

Efficiency on the target machine still matters, but we can defer thinking about that until codegen, where it’s no longer as important to enable subsequent optimizations. At that point, we’re going to start caring about picking between `+`, `*`, and `<<` based on which one executes fastest.
>  虽然我们之前从依赖图的角度出发，但目标机器上的运算效率仍然是一个重要考虑因素，但我们可以推迟到代码生成阶段再考虑这一点，在代码生成阶段，我们就不需要再考虑为后续优化提供便利，而是仅考虑 `+, *, <<` 哪种形式的执行更快

The basic philosophy of canonicalization says that canonical forms should be translated into optimal forms toward the back of the compiler, after all mid-level optimizations which benefit from canonical form are done. It’s also worth noting that codegen itself benefits from having the code coming into it be in canonical form, so that it doesn’t have to recognize all the ways to write `x << 1`, and can just recognize one pattern for that and emit the optimal code for it.
>  规范化的基本思想是：在所有从规范化受益的中间级别优化完成之后，规范化形式应该在编译器后端再被转化为最优形式
>  并且注意，代码生成本身在其输入形式为规范化形式时也可以受益，这使得代码生成不需要识别所有 `x << 1` 的其他形式，而只需识别一种模式，为其生成最优代码即可

This is often a source of confusion: Is canonicalization the same as optimization? It’s often done as part of the “optimizer”, and many of the things it does produce more optimal code directly. But ultimately, in its purest form, canonicalization just focuses on removing unnecessary variation so that subsequent optimizations can be simpler.
>  规范化和优化并不是一个概念，虽然规范化通常作为 “优化器” 的一部分执行，并且规范化做的许多事通常会直接生成更优的代码，但本质上，规范化仅专注于消除不必要的变化，使得后续的优化可以更加简单

> Canonical form, canonical form  
> Canonical form hates optimal form  
> They have a fight, canonical wins  
> Canonical form…

(sung to the tune of “Particle Man” by They Might Be Giants)

### Sometimes it’s ambiguous: redundancy elimination
Is redundancy elimination a canonicalization or an optimization?
>  冗余消除是规范化还是优化？

It’s certainly simpler to compute a given value once and reuse the value, rather than compute it twice. But does that aid subsequent optimizations? It depends.
>  冗余消除中，我们计算一个给定值一次，之后便复用这个值
>  这相较于重复计算显然更加简单，但这是否会帮助后续的优化，则视情况而定

A case where it does aid subsequent optimizations is when it eliminates redundant memory accesses. That way, it’s essentially saying that no later passes have to even ask what the dependencies are for a given memory access, because the memory access has been eliminated.
>  其帮助后续优化的一个情况是它消除了冗余的内存访问，这等价于说后续的 passes 甚至不需要去询问某个内存访问的依赖关系，因为这个内存访问已经被消除了

A case where it doesn’t is where it can take expression trees where everything has a single use and give some values multiple uses. Some kinds of optimization passes are harder to do on DAGs than on trees, so this can result in pessimizations in some cases, depending on what kinds of things the rest of the compiler is doing.
>  其不帮助后续优化的一个情况是：原来的表达式树中，每个值仅有一次使用，经过冗余消除后，某些值被多次使用
>  一些类型的优化 passes 在 DAG 上相较于在树上会更难实现，因此冗余消除也可能会降低性能

However, we typically do think of redundancy elimination as being a canonicalization. It’s trivial to convert multiple-use values into single-use values by duplicating code, while going the other direction requires some analysis.
>  我们一般会将冗余消除视作规范化，我们可以简单通过复制代码，将多用途值转化为单用途值

### Even more ambiguous: inlining
Inlining can act like canonicalization, especially in cases where the inlined function body can be optimized away. However, thinking of inlining in terms of canonicalization doesn’t lead to a natural threshold. Should we maximally inline everything as far as possible? Or should we do the reverse and maximally outline and deduplicate outlined functions? Neither extreme is particularly practical, so most compilers use heuristics in practice rather than having a rigid definition of canonical form relative to calls.
>  内联可以起到规范化的作用，尤其是在内联的函数体可以被优化掉的情况
>  但是，仅从规范化的角度看待内联，我们无法判断内联到何种程度：是尽可能多内联，还是尽可能多外联
>  两个极端都不是实际，编译器通常使用启发式方法，而不是根据某个相对于函数调用的规范形式的严格定义

That said, even though there’s no clear boundary, we can imagine a rough guideline. Think of old-school C code, written at a time when “C is a portable assembly language” was more true than it is today, where calls that were important to inline were written as macros. In many ways, one of the jobs of compilers for higher-level languages is to compile them down to roughly this level, so that they can be optimized using optimization techniques which work well at this level. In theory, we could define the task of a canonicalizing inliner to just be to inline code down to what it would have looked like if that same code had been written in old-school C (at least with respect to inlining). 
>  即便没有清晰的界限，我们也可以有粗略的指导原则
>  在老式的 C 代码中，函数内联通过宏实现，现今的许多高级语言编译器的任务就是将这些语言大致编译到这一级别，进而利用在这一级别上的优化技术
>  理论上，我们可以定义规范化内联器的任务它将代码内敛到使用老式 C 代码编写相同功能可能呈现的样子 (至少在内联方面)

That’s not very pure, and it’s difficult to precisely describe, but it’s an intuitive and relatively practical compromise between extremes.

### Canonical form isn’t just for arithmetic expressions!
Everything in an IR may be subjected to canonicalization.
>  IR 中的所有内容都可能需要规范化

For example, a control flow canonicalization might involve sorting the basic blocks of a program into Reverse Post-Order (RPO). Doing this is a simple way to ensure that the code is optimized the same regardless of how the user organizes the code inside their functions.
>  例如，控制流规范化可能涉及将程序的基本块排序为逆后序，这可以确保无论用户在函数内以何种方式组织 diamagnetic，优化都以相同方式进行

> [!info] Reverse Post-Order 逆后序
> 逆后序即后续遍历的逆序
> 后序遍历的访问顺序为：左 -> 右 -> 根
> 逆后序的访问顺序为：根 -> 右 -> 左

Dead code elimination is also a kind of canonicalization. The canonical form for dead code is no code.
>  死代码消除也是依赖规范化，死代码的规范形式就是没有代码

In aggressive loop-transforming compilers, another form of canonicalization is to maximally fission loops into as many parts as possible, and then assume subsequent passes will fuse them back together into optimal loops that effectively utilize available registers and cache space.
>  在激进的循环转化编译器中，规范化的另一种形式是尽可能将循环拆分成多个部分，然后假设后续的 passes 会将它们组合为最优的循环 (有效利用寄存器和缓存空间) 

### Canonicalization as compression
It’s often the case that smaller forms are preferred over longer forms, so canonicalization tends to make code smaller, making it a form of compression. Also, since it effectively reduces non-essential entropy, it can make subsequent general-purpose compression more effective as well.
>  通常，我们相对于更长的形式，偏好更短的形式，因此规范化倾向于让代码更短，因此规范化也可以视为一种形式的压缩
>  并且，因为规范化有效地减少了不必要的熵 (不必要的信息复杂度)，故规范化也可以使得后续的通用目的压缩更加高效

If we could conceptually perform all theoretically possible canonicalizations on a program, we’d end up with something related to its [Kolmogorov Complexity](https://en.wikipedia.org/wiki/Kolmogorov_complexity) (it may not be identical, since canonicalization puts the needs of subsequent optimizations first, rather than absolute compression). Maximal canonicalization is frequently impossible in practice, because of the halting problem, but also because even in cases where it’s theoretically possible, it can require impractical amounts of computation.
>  如果我们在概念上可以为一个程序执行所有可能的规范化，我们最终得到的结果将和它的 Kolmogorov 复杂性有关
>  在实际中，最大程度的规范化通常不可能，原因包括了停机问题和计算资源需求量过大的问题

> [!info] Kolmogorov Complexity
>  在算法信息论中，一个对象 (例如一段文本) 的 Kolmogorov complexity，是指生成该对象的最短计算机程序的长度 (在一个预定义的编程语言中)
>  Kolmogorov Complexity 度量了描述一个对象所需的计算资源

That said, it is pretty fun to think that for any given program, there is a theoretical “maximally canonical form” for that program, that all equivalent ways of writing that program could be reduced to. It is tempting to think of this as a kind of pure essence of the program.
>  我们可以认为任意给定程序都存在一个理论上的 “最大规范形式”，所有描写该程序的等价方式都可以简化为该 “最大规范形式”，不妨把它视作该程序的本质

### Excessive canonicalization
Canonicalization discards inessential information. However, sometimes that information can be useful to preserve.
>  规范化丢弃不必要的信息，但有时这些信息有保存的价值

An example arises in instruction scheduling: Say a user writes code like

```
  x = a + b;
  y = c * d;
```

Assuming there’s no aliasing going on here, there’s no reason why one of these statements has to be ordered before the other. Their order in the user’s source code is inessential information. “Sea of nodes” style compilers may canonicalize to the point where there is no inherent ordering between these two statements.

>  一个例子是指令调度，考虑以上例子的代码
>  假设不存在别名，则上述两个语句的顺序并没有意义，因此它们在用户源码中的顺序是不必要的信息
>  "Sea of nodes" 风格的编译器可能会将这段代码规范化到两个语句之间不存在固有的执行顺序的程度

The compiler backend ultimately has to produce machine code, which on conventional architectures requires the compiler to pick _some_ ordering. Compilers can be pretty smart, and can take into consideration many things, such as available execution resources before and after these statements to know what order a CPU would prefer to see them in. However, as smart as they can be, compilers can’t always find the optimal answers. Optimal instruction scheduling is NP complete, but also, it may come down to runtime factors that ahead-of-time compilers don’t have. And CPU hardware performance characteristics aren’t always fully documented.
>  但编译器后端最终需要生成机器码，故对于常规的机器，编译器最终还是要选择某个顺序
>  编译器可以考虑许多事情，例如考虑这些语句之前和之后的可用执行资源，以确定 CPU 偏好的顺序
>  但编译器也不能总是找到最优答案，最优指令调度是 NP 完全问题，并且可能取决于编译器无法知道的运行时因素
>  此外，CPU 的硬件特性也不会总是完全公开

#### “Do No Harm”?
Most software has no idea how it’ll be mapped on to CPU pipelines. But some does. And it’s these cases where getting the mapping right is most important.
>  大多数软件并不知道它将如何被映射到 CPU 流水线，但有些软件知道，在这种情况下，保持映射正确就十分重要

On such software, optimizations that make actual improvements are fine, however it can be important that compilers not make anything _worse_ than if they had translated the code naively. Aggressively canonicalizing compilers have a risk that they will throw away information and at the end reconstruct a form which is worse than if they had just simply translated the code as it was written.
>  对于这类软件，编译要注意它的优化不应该让最终代码的表现比简单翻译源码的情况更差
>  过于激进的规范化编译器存在这样的风险，因为它们可能会丢弃信息

This, along with the ambiguous cases above, suggests that canonicalization be used in practical rather than rigid ways.
>  这也表明了规范化的应用应该具有灵活性而非固化

---

## The theoretical shape of optimization.
A compiler typically starts with human-written source code.
>  编译器从人类编写的源码开始

Typical human-written code follows various human-oriented sensibilities. Different humans may have different aesthetic sensibilities, and this can lead to writing the same code in different ways. But of course this isn’t interesting to optimizers.
>  人类编写的源码遵循不同人的观念，故相同的代码会被按照不同的方式编写

So the first then we typically do when the code hits the optimizers is to start canonicalizing. Throw away useless fluff that humans imbue their code with.
>  这些方式的差异对于优化器来说应该是无关的，故在将代码交给优化器之前，第一件事就是执行规范化，去掉人类代码中的无用的多余部分

Canonicalizing optimizations are cascading; often doing one canonicalization will enable more. Folding an expression to a constant may allow other expressions to be folded to constants. Replacing a load by forwarding a value through a prior store may create a direct edge between two expression trees and introduce opportunities to simplify further.
>  规范化优化是级联的，通常执行一个规范化会引发更多的规范化
>  例如，将一个表达式折叠为常量会使得其他表达式也被折叠为常量，通过向前传递之前 store 的值来替换 load 可以创建两个表达式树之间的一条边，引入进一步简化的机会

So ignoring the practical concerns we mentioned above, we can imagine a theoretical compiler that does all the canonicalization it knows how to do up front. And while we don’t realistically approach Kolmogorov Complexity levels, we do lift the program closer to what we might think of as its essence, the simplest form that does what it needs to do.
>  编译器通过规范化，将程序转化为更接近它 “本质” 的更简单形式

And then, the compiler can begin to optimize, rewriting canonical forms into optimal forms, as it lowers the code all the way down to assembly code.
>  之后，编译器进行优化，将规范形式重写为最优形式，然后将代码下降为汇编

For practical reasons, it isn’t always possible to build compilers in terms of a purely canonicalizing phase and a purely optimizing phase, however this can be a useful reference point for understanding compilers in practice.