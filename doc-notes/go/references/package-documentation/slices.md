---
version: 1.24.3
---
### Overview 
Package `slices` defines various functions useful with slices of any type.
### Functions
#### `func SortFunc`

```go
func SortFunc[S ~[]E, E any](x S, cmp func(a, b E) int)
```

`SortFunc` sorts the slice x in ascending order as determined by the `cmp` function. This sort is not guaranteed to be stable. `cmp(a, b)` should return a negative number when a < b, a positive number when a > b and zero when a == b or a and b are incomparable in the sense of a strict weak ordering.
>  `SortFunc` 根据 `cmp` 决定的顺序将切片 `x` 按升序排序，排序不保证稳定
>  `cmp(a, b)` 应该在 `a<b` 时返回负数，在 `a>b` 时返回正数，在 `a==b` 时或者在严格弱序的意义上 `a,b` 不可比时，返回零

`SortFunc` requires that `cmp` is a strict weak ordering. See [https://en.wikipedia.org/wiki/Weak_ordering#Strict_weak_orderings](https://en.wikipedia.org/wiki/Weak_ordering#Strict_weak_orderings). The function should return 0 for incomparable items.
>  `SortFunc` 要求 `cmp` 是一个严格的弱序关系，`cmp` 对于不可比较的元素应该返回零

Example (CaseInsensitive)

```go
package main

import (
    "fmt"
    "slices"
    "strings"
)

func main() {
    names := []string{"Bob", "alice", "VERA"}
    slices.SortFunc(names, func(a, b string) int {
        return strings.Compare(strings.ToLower(a), strings.ToLower(b))
    })
    fmt.Println(names)
}
```

Example (MultiField)

```go
package main

import (
    "cmp"
    "fmt"
    "slices"
    "strings"
)

func main() {
    type Person struct {
        Name string
        Age  int
    }
    people := []Person{
        {"Gopher", 13},
        {"Alice", 55},
        {"Bob", 24},
        {"Alice", 20},
    }
    slices.SortFunc(people, func(a, b Person) int {
        if n := strings.Compare(a.Name, b.Name); n != 0 {
            return n
        }
        // If names are equal, order by age
        return cmp.Compare(a.Age, b.Age)
    })
    fmt.Println(people)
}
```
