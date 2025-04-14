>  Revision 1.2.2 (2021-10-01)

Copyright presently by YAML Language Development Team[1](https://yaml.org/spec/1.2.2/#fn:team)  
Copyright 2001-2009 by Oren Ben-Kiki, Clark Evans, Ingy döt Net

This document may be freely copied, provided it is not modified.

**Status of this Document**
This is the **YAML specification v1.2.2**. It defines the **YAML 1.2 data language**. There are no normative changes from the **YAML specification v1.2**. The primary objectives of this revision are to correct errors and add clarity.

This revision also strives to make the YAML language development process more open, more transparent and easier for people to contribute to. The input format is now Markdown instead of DocBook, and the images are made from plain text LaTeX files rather than proprietary drawing software. All the source content for the specification is publicly hosted[2](https://yaml.org/spec/1.2.2/#fn:spec-repo).

The previous YAML specification[3](https://yaml.org/spec/1.2.2/#fn:1-2-spec) was published 12 years ago. In that time span, YAML’s popularity has grown significantly. Efforts are ongoing to improve the language and grow it to meet the needs and expectations of its users. While this revision of the specification makes no actual changes to YAML, it begins a process by which the language intends to evolve and stay modern.

The YAML specification is often seen as overly complicated for something which appears to be so simple. Even though YAML often is used for software configuration, it has always been and will continue to be a complete data serialization language. Future YAML plans are focused on making the language and ecosystem more powerful and reliable while simultaneously simplifying the development process for implementers.

While this revision of the specification is limiting itself to informational changes only, there is companion documentation intended to guide YAML framework implementers and YAML language users. This documentation can continue to evolve and expand continually between published revisions of this specification.

See:

- [YAML Resources Index](https://yaml.org/spec/1.2.2/ext/resources)
- [YAML Vocabulary Glossary](https://yaml.org/spec/1.2.2/ext/glossary)
- [YAML Specification Changes](https://yaml.org/spec/1.2.2/ext/changes)
- [YAML Specification Errata](https://yaml.org/spec/1.2.2/ext/errata)

**Abstract**
YAML™ (rhymes with “camel”) is a human-friendly, cross language, Unicode based data serialization language designed around the common native data types of dynamic programming languages. It is broadly useful for programming needs ranging from configuration files to internet messaging to object persistence to data auditing and visualization. Together with the Unicode standard for characters [4](https://yaml.org/spec/1.2.2/#fn:unicode), this specification provides all the information necessary to understand YAML version 1.2 and to create programs that process YAML information.

# Chapter 1. Introduction to YAML
YAML (a recursive acronym for “YAML Ain’t Markup Language”) is a data serialization language designed to be human-friendly and work well with modern programming languages for common everyday tasks. This specification is both an introduction to the YAML language and the concepts supporting it. It is also a complete specification of the information needed to develop [applications](https://yaml.org/spec/1.2.2/#processes-and-models) for processing YAML.

Open, interoperable and readily understandable tools have advanced computing immensely. YAML was designed from the start to be useful and friendly to people working with data. It uses Unicode [printable](https://yaml.org/spec/1.2.2/#character-set) characters, [some](https://yaml.org/spec/1.2.2/#indicator-characters) of which provide structural information and the rest containing the data itself. YAML achieves a unique cleanness by minimizing the amount of structural characters and allowing the data to show itself in a natural and meaningful way. For example, [indentation](https://yaml.org/spec/1.2.2/#indentation-spaces) may be used for structure, [colons](https://yaml.org/spec/1.2.2/#flow-mappings) separate [key/value pairs](https://yaml.org/spec/1.2.2/#mapping) and [dashes](https://yaml.org/spec/1.2.2/#block-sequences) are used to create “bulleted” [lists](https://yaml.org/spec/1.2.2/#sequence).

There are many kinds of [data structures](https://yaml.org/spec/1.2.2/#dump), but they can all be adequately [represented](https://yaml.org/spec/1.2.2/#representation-graph) with three basic primitives: [mappings](https://yaml.org/spec/1.2.2/#mapping) (hashes/dictionaries), [sequences](https://yaml.org/spec/1.2.2/#sequence) (arrays/lists) and [scalars](https://yaml.org/spec/1.2.2/#scalars) (strings/numbers). YAML leverages these primitives and adds a simple typing system and [aliasing](https://yaml.org/spec/1.2.2/#anchors-and-aliases) mechanism to form a complete language for [serializing](https://yaml.org/spec/1.2.2/#serializing-the-representation-graph) any [native data structure](https://yaml.org/spec/1.2.2/#representing-native-data-structures). While most programming languages can use YAML for data serialization, YAML excels in working with those languages that are fundamentally built around the three basic primitives. These include common dynamic languages such as JavaScript, Perl, PHP, Python and Ruby.
>  有许多种类的数据结构，但它们都可以被三种基本原语: **映射、序列、标量**充分表示
>  YAML 基于这三个原语，并添加了类型系统和别名机制，形成了一个完整的语言，用于序列化任意本地数据结构
>  YAML 与本质上也基于这三个原语的语言协作时表现出色，例如 JS, Python 等动态语言

There are hundreds of different languages for programming, but only a handful of languages for storing and transferring data. Even though its potential is virtually boundless, YAML was specifically created to work well for common use cases such as: configuration files, log files, interprocess messaging, cross-language data sharing, object persistence and debugging of complex data structures. When data is easy to view and understand, programming becomes a simpler task.

## 1.1. Goals
The design goals for YAML are, in decreasing priority:

1. YAML should be easily readable by humans.
2. YAML data should be portable between programming languages.
3. YAML should match the [native data structures](https://yaml.org/spec/1.2.2/#representing-native-data-structures) of dynamic languages.
4. YAML should have a consistent model to support generic tools.
5. YAML should support one-pass processing.
6. YAML should be expressive and extensible.
7. YAML should be easy to implement and use.

## 1.2. YAML History
The YAML 1.0 specification was published in early 2004 by by Clark Evans, Oren Ben-Kiki, and Ingy döt Net after 3 years of collaborative design work through the yaml-core mailing list [5](https://yaml.org/spec/1.2.2/#fn:yaml-core). The project was initially rooted in Clark and Oren’s work on the SML-DEV [6](https://yaml.org/spec/1.2.2/#fn:sml-dev) mailing list (for simplifying XML) and Ingy’s plain text serialization module [7](https://yaml.org/spec/1.2.2/#fn:denter) for Perl. The language took a lot of inspiration from many other technologies and formats that preceded it.

The first YAML framework was written in Perl in 2001 and Ruby was the first language to ship a YAML framework as part of its core language distribution in 2003.

The YAML 1.1[8](https://yaml.org/spec/1.2.2/#fn:1-1-spec) specification was published in 2005. Around this time, the developers became aware of JSON[9](https://yaml.org/spec/1.2.2/#fn:json). By sheer coincidence, JSON was almost a complete subset of YAML (both syntactically and semantically).

In 2006, Kyrylo Simonov produced PyYAML[10](https://yaml.org/spec/1.2.2/#fn:pyyaml) and LibYAML[11](https://yaml.org/spec/1.2.2/#fn:libyaml). A lot of the YAML frameworks in various programming languages are built over LibYAML and many others have looked to PyYAML as a solid reference for their implementations.

The YAML 1.2[3](https://yaml.org/spec/1.2.2/#fn:1-2-spec) specification was published in 2009. Its primary focus was making YAML a strict superset of JSON. It also removed many of the problematic implicit typing recommendations.

Since the release of the 1.2 specification, YAML adoption has continued to grow, and many large-scale projects use it as their primary interface language. In 2020, the new [YAML language design team](https://yaml.org/spec/1.2.2/ext/team) began meeting regularly to discuss improvements to the YAML language and specification; to better meet the needs and expectations of its users and use cases.

This YAML 1.2.2 specification, published in October 2021, is the first step in YAML’s rejuvenated development journey. YAML is now more popular than it has ever been, but there is a long list of things that need to be addressed for it to reach its full potential. The YAML design team is focused on making YAML as good as possible.

## 1.3. Terminology
The key words “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHALL NOT”, “SHOULD”, “SHOULD NOT”, “RECOMMENDED”, “MAY”, and “OPTIONAL” in this document are to be interpreted as described in RFC 2119 [12](https://yaml.org/spec/1.2.2/#fn:rfc-2119).

The rest of this document is arranged as follows. Chapter [2](https://yaml.org/spec/1.2.2/#language-overview) provides a short preview of the main YAML features. Chapter [3](https://yaml.org/spec/1.2.2/#processes-and-models) describes the YAML information model and the processes for converting from and to this model and the YAML text format. The bulk of the document, chapters [4](https://yaml.org/spec/1.2.2/#syntax-conventions), [5](https://yaml.org/spec/1.2.2/#character-productions), [6](https://yaml.org/spec/1.2.2/#structural-productions), [7](https://yaml.org/spec/1.2.2/#flow-style-productions), [8](https://yaml.org/spec/1.2.2/#block-style-productions) and [9](https://yaml.org/spec/1.2.2/#document-stream-productions), formally define this text format. Finally, chapter [10](https://yaml.org/spec/1.2.2/#recommended-schemas) recommends basic YAML schemas.

# Chapter 2. Language Overview
This section provides a quick glimpse into the expressive power of YAML. It is not expected that the first-time reader grok all of the examples. Rather, these selections are used as motivation for the remainder of the specification.

## 2.1. Collections
YAML’s [block collections](https://yaml.org/spec/1.2.2/#block-collection-styles) use [indentation](https://yaml.org/spec/1.2.2/#indentation-spaces) for scope and begin each entry on its own line. [Block sequences](https://yaml.org/spec/1.2.2/#block-sequences) indicate each entry with a dash and space (“`-` ”). [Mappings](https://yaml.org/spec/1.2.2/#mapping) use a colon and space (“`:` ”) to mark each [key/value pair](https://yaml.org/spec/1.2.2/#mapping). [Comments](https://yaml.org/spec/1.2.2/#comments) begin with an octothorpe (also called a “hash”, “sharp”, “pound” or “number sign” - “`#`”).
>  YAML 的块集合使用缩进表示范围，且每个条目独占一行
>  块序列使用 `- ` 表示每个条目
>  映射使用 `: ` 表示每个键值对
>  注释以 `#` 开头

**Example 2.1 Sequence of Scalars (ball players)**

```yml
- Mark McGwire
- Sammy Sosa
- Ken Griffey
```

**Example 2.2 Mapping Scalars to Scalars (player statistics)**

```yml
hr:  65    # Home runs
avg: 0.278 # Batting average
rbi: 147   # Runs Batted In
```

**Example 2.3 Mapping Scalars to Sequences (ball clubs in each league)**

```yml
american:
- Boston Red Sox
- Detroit Tigers
- New York Yankees
national:
- New York Mets
- Chicago Cubs
- Atlanta Braves
```

**Example 2.4 Sequence of Mappings (players’ statistics)**

```yml
-
  name: Mark McGwire
  hr:   65
  avg:  0.278
-
  name: Sammy Sosa
  hr:   63
  avg:  0.288
```

YAML also has [flow styles](https://yaml.org/spec/1.2.2/#flow-style-productions), using explicit [indicators](https://yaml.org/spec/1.2.2/#indicator-characters) rather than [indentation](https://yaml.org/spec/1.2.2/#indentation-spaces) to denote scope. The [flow sequence](https://yaml.org/spec/1.2.2/#flow-sequences) is written as a [comma](https://yaml.org/spec/1.2.2/#flow-collection-styles) separated list within [square](https://yaml.org/spec/1.2.2/#flow-sequences) [brackets](https://yaml.org/spec/1.2.2/#flow-sequences). In a similar manner, the [flow mapping](https://yaml.org/spec/1.2.2/#flow-mappings) uses [curly](https://yaml.org/spec/1.2.2/#flow-mappings) [braces](https://yaml.org/spec/1.2.2/#flow-mappings).
>  YAML 也有流样式，使用显示的指示符而不是缩进表示范围
>  流序列以逗号分割的形式写在 `[]` 内
>  流映射以逗号分割的形式写在 `{}` 内

**Example 2.5 Sequence of Sequences**

```yml
- [name        , hr, avg  ]
- [Mark McGwire, 65, 0.278]
- [Sammy Sosa  , 63, 0.288]
```

**Example 2.6 Mapping of Mappings**

```yml
Mark McGwire: {hr: 65, avg: 0.278}
Sammy Sosa: {
    hr: 63,
    avg: 0.288,
 }
```

## 2.2. Structures
YAML uses three dashes (“`---`”) to separate [directives](https://yaml.org/spec/1.2.2/#directives) from [document](https://yaml.org/spec/1.2.2/#documents) [content](https://yaml.org/spec/1.2.2/#nodes). This also serves to signal the start of a document if no [directives](https://yaml.org/spec/1.2.2/#directives) are present. Three dots ( “`...`”) indicate the end of a document without starting a new one, for use in communication channels.
>  YAML 使用 `---` 分割指令和文档内容
>  如果不存在指令，这也标志着文档的开始
>  `...` 用于表示文档的结束而不启动新的文档，适用于通信通道中

**Example 2.7 Two Documents in a Stream (each with a leading comment)**

```yml
# Ranking of 1998 home runs
---
- Mark McGwire
- Sammy Sosa
- Ken Griffey

# Team ranking
---
- Chicago Cubs
- St Louis Cardinals
```

**Example 2.8 Play by Play Feed from a Game**

```yml
---
time: 20:03:20
player: Sammy Sosa
action: strike (miss)
...
---
time: 20:03:47
player: Sammy Sosa
action: grand slam
...
```

Repeated [nodes](https://yaml.org/spec/1.2.2/#nodes) (objects) are first [identified](https://yaml.org/spec/1.2.2/#anchors-and-aliases) by an [anchor](https://yaml.org/spec/1.2.2/#anchors-and-aliases) (marked with the ampersand - “`&`”) and are then [aliased](https://yaml.org/spec/1.2.2/#anchors-and-aliases) (referenced with an asterisk - “`*`”) thereafter.
>  重复的节点 (对象) 首先通过一个锚点 (`&`) 标记，然后通过别名 (`*`) 在之后进行引用

**Example 2.9 Single Document with Two Comments**

```yml
---
hr: # 1998 hr ranking
- Mark McGwire
- Sammy Sosa
# 1998 rbi ranking
rbi:
- Sammy Sosa
- Ken Griffey
```

**Example 2.10 Node for “`Sammy Sosa`” appears twice in this document**

```yml
---
hr:
- Mark McGwire
# Following node labeled SS
- &SS Sammy Sosa
rbi:
- *SS # Subsequent occurrence
- Ken Griffey
```

A question mark and space (“`?` ”) indicate a complex [mapping](https://yaml.org/spec/1.2.2/#mapping) [key](https://yaml.org/spec/1.2.2/#nodes). Within a [block collection](https://yaml.org/spec/1.2.2/#block-collection-styles), [key/value pairs](https://yaml.org/spec/1.2.2/#mapping) can start immediately following the [dash](https://yaml.org/spec/1.2.2/#block-sequences), [colon](https://yaml.org/spec/1.2.2/#flow-mappings) or [question mark](https://yaml.org/spec/1.2.2/#flow-mappings).
>  `? ` 表示一个复杂的映射键
>  在块集合中，键值对可以紧跟着 `-,:,?` 后开始

**Example 2.11 Mapping between Sequences**

```yml
? - Detroit Tigers
  - Chicago cubs
: - 2001-07-23

? [ New York Yankees,
    Atlanta Braves ]
: [ 2001-07-02, 2001-08-12,
    2001-08-14 ]
```

**Example 2.12 Compact Nested Mapping**

```yml
---
# Products purchased
- item    : Super Hoop
  quantity: 1
- item    : Basketball
  quantity: 4
- item    : Big Shoes
  quantity: 1
```

## 2.3. Scalars
[Scalar content](https://yaml.org/spec/1.2.2/#scalar) can be written in [block](https://yaml.org/spec/1.2.2/#scalars) notation, using a [literal style](https://yaml.org/spec/1.2.2/#literal-style) (indicated by “`|`”) where all [line breaks](https://yaml.org/spec/1.2.2/#line-break-characters) are significant. Alternatively, they can be written with the [folded style](https://yaml.org/spec/1.2.2/#folded-style) (denoted by “`>`”) where each [line break](https://yaml.org/spec/1.2.2/#line-break-characters) is [folded](https://yaml.org/spec/1.2.2/#line-folding) to a [space](https://yaml.org/spec/1.2.2/#white-space-characters) unless it ends an [empty](https://yaml.org/spec/1.2.2/#empty-lines) or a [more-indented](https://yaml.org/spec/1.2.2/#example-more-indented-lines) line.
>  标量内容可以用块表示法书写，使用字面风格 (`|`) 来保留所有换行符
>  另一种方式是使用折叠风格 (`>`)，其中每个换行符都会被折叠为一个空格，除非它出现在一个空行或缩进更深的行中

**Example 2.13 In literals, newlines are preserved**

```yml
# ASCII Art
--- |
  \//||\/||
  // ||  ||__
```

**Example 2.14 In the folded scalars, newlines become spaces**

```yml
--- >
  Mark McGwire's
  year was crippled
  by a knee injury.
```

**Example 2.15 Folded newlines are preserved for “more indented” and blank lines**

```yml
--- >
 Sammy Sosa completed another
 fine season with great stats.

   63 Home Runs
   0.288 Batting Average

 What a year!
```

**Example 2.16 Indentation determines scope**

```yml
name: Mark McGwire
accomplishment: >
  Mark set a major league
  home run record in 1998.
stats: |
  65 Home Runs
  0.278 Batting Average
```

YAML’s [flow scalars](https://yaml.org/spec/1.2.2/#flow-scalar-styles) include the [plain style](https://yaml.org/spec/1.2.2/#plain-style) (most examples thus far) and two quoted styles. The [double-quoted style](https://yaml.org/spec/1.2.2/#double-quoted-style) provides [escape sequences](https://yaml.org/spec/1.2.2/#escaped-characters). The [single-quoted style](https://yaml.org/spec/1.2.2/#single-quoted-style) is useful when [escaping](https://yaml.org/spec/1.2.2/#escaped-characters) is not needed. All [flow scalars](https://yaml.org/spec/1.2.2/#flow-scalar-styles) can span multiple lines; [line breaks](https://yaml.org/spec/1.2.2/#line-break-characters) are always [folded](https://yaml.org/spec/1.2.2/#line-folding).
>  YAML 的流标量包括普通风格 (目前为止的大多数示例) 和双引号风格
>  双引号风格提供了转义序列，单引号风格在不需要转义时很有用
>  所有的流标量可以跨越多行，换行符会被折叠

**Example 2.17 Quoted Scalars**

```yml
unicode: "Sosa did fine.\u263A"
control: "\b1998\t1999\t2000\n"
hex esc: "\x0d\x0a is \r\n"

single: '"Howdy!" he cried.'
quoted: ' # Not a ''comment''.'
tie-fighter: '|\-*-/|'
```

**Example 2.18 Multi-line Flow Scalars**

```yml
plain:
  This unquoted scalar
  spans many lines.

quoted: "So does this
  quoted scalar.\n"
```

## 2.4. Tags
In YAML, [untagged nodes](https://yaml.org/spec/1.2.2/#resolved-tags) are given a type depending on the [application](https://yaml.org/spec/1.2.2/#processes-and-models). The examples in this specification generally use the `seq`, `map` and `str` types from the [fail safe schema](https://yaml.org/spec/1.2.2/#failsafe-schema). A few examples also use the `int`, `float` and `null` types from the [JSON schema](https://yaml.org/spec/1.2.2/#json-schema).
>  YAML 中，未标记节点的类型取决于应用程序
>  本规范中的示例通常使用安全模式架构中的 `seq, map, str` 类型，还有一些示例使用了 JSON 模式中的 `int, float, null` 类型

**Example 2.19 Integers**

```yml
canonical: 12345
decimal: +12345
octal: 0o14
hexadecimal: 0xC
```

**Example 2.20 Floating Point**

```
canonical: 1.23015e+3
exponential: 12.3015e+02
fixed: 1230.15
negative infinity: -.inf
not a number: .nan
```

**Example 2.21 Miscellaneous**

```
null:
booleans: [ true, false ]
string: '012345'
```

**Example 2.22 Timestamps**

```
canonical: 2001-12-15T02:59:43.1Z
iso8601: 2001-12-14t21:59:43.10-05:00
spaced: 2001-12-14 21:59:43.10 -5
date: 2002-12-14
```

Explicit typing is denoted with a [tag](https://yaml.org/spec/1.2.2/#tags) using the exclamation point (“`!`”) symbol. [Global tags](https://yaml.org/spec/1.2.2/#tags) are URIs and may be specified in a [tag shorthand](https://yaml.org/spec/1.2.2/#tag-shorthands) notation using a [handle](https://yaml.org/spec/1.2.2/#tag-handles). [Application](https://yaml.org/spec/1.2.2/#processes-and-models) -specific [local tags](https://yaml.org/spec/1.2.2/#tags) may also be used.
>  显式的类型通过标签表示 (`!`)
>  全局标签是 URL，可以通过使用句柄的标签简写标记指定，应用特定的本地标签也可以使用

**Example 2.23 Various Explicit Tags**

```yml
---
not-date: !!str 2002-04-28

picture: !!binary |
 R0lGODlhDAAMAIQAAP//9/X
 17unp5WZmZgAAAOfn515eXv
 Pz7Y6OjuDg4J+fn5OTk6enp
 56enmleECcgggoBADs=

application specific tag: !something |
 The semantics of the tag
 above may be different for
 different documents.
```

**Example 2.24 Global Tags**

```yml
%TAG ! tag:clarkevans.com,2002:
--- !shape
  # Use the ! handle for presenting
  # tag:clarkevans.com,2002:circle
- !circle
  center: &ORIGIN {x: 73, y: 129}
  radius: 7
- !line
  start: *ORIGIN
  finish: { x: 89, y: 102 }
- !label
  start: *ORIGIN
  color: 0xFFEEBB
  text: Pretty vector drawing.
```

**Example 2.25 Unordered Sets**

```yml
# Sets are represented as a
# Mapping where each key is
# associated with a null value
--- !!set
? Mark McGwire
? Sammy Sosa
? Ken Griffey
```

**Example 2.26 Ordered Mappings**

```yml
# Ordered maps are represented as
# A sequence of mappings, with
# each mapping having one key
--- !!omap
- Mark McGwire: 65
- Sammy Sosa: 63
- Ken Griffey: 58
```

## 2.5. Full Length Example
Below are two full-length examples of YAML. The first is a sample invoice; the second is a sample log file.

**Example 2.27 Invoice**

```
--- !<tag:clarkevans.com,2002:invoice>
invoice: 34843
date   : 2001-01-23
bill-to: &id001
  given  : Chris
  family : Dumars
  address:
    lines: |
      458 Walkman Dr.
      Suite #292
    city    : Royal Oak
    state   : MI
    postal  : 48046
ship-to: *id001
product:
- sku         : BL394D
  quantity    : 4
  description : Basketball
  price       : 450.00
- sku         : BL4438H
  quantity    : 1
  description : Super Hoop
  price       : 2392.00
tax  : 251.42
total: 4443.52
comments:
  Late afternoon is best.
  Backup contact is Nancy
  Billsmer @ 338-4338.
```

**Example 2.28 Log File**

```
---
Time: 2001-11-23 15:01:42 -5
User: ed
Warning:
  This is an error message
  for the log file
---
Time: 2001-11-23 15:02:31 -5
User: ed
Warning:
  A slightly different error
  message.
---
Date: 2001-11-23 15:03:17 -5
User: ed
Fatal:
  Unknown variable "bar"
Stack:
- file: TopClass.py
  line: 23
  code: |
    x = MoreObject("345\n")
- file: MoreClass.py
  line: 58
  code: |-
    foo = bar
```