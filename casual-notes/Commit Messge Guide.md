# Style 1 Conventional Commit Message
## Angular Commit Message Guidelines
We have very precise rules over how our git commit messages can be formatted. This leads to **more readable messages** that are easy to follow when looking through the **project history**. But also, we use the git commit messages to **generate the Angular change log**.
### Commit Message Format
Each commit message consists of a **header**, a **body** and a **footer**. The header has a special format that includes a **type**, a **scope** and a **subject**:
> commit message 包括三部分：header、body、footer
> header 包括三部分：type、scope、subject

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```
The **header** is mandatory and the **scope** of the header is optional.
> header 中的 scope 为 optional

Any line of the commit message cannot be longer 100 characters! This allows the message to be easier to read on GitHub as well as in various git tools.

The footer should contain a [closing reference to an issue](https://help.github.com/articles/closing-issues-via-commit-messages/) if any.

Samples: (even more [samples](https://github.com/angular/angular/commits/master))
```
docs(changelog): update changelog to beta.5
```

```
fix(release): need to depend on latest rxjs and zone.js

The version in our package.json gets copied to the one we publish, and users need the latest of these.
```
### Header

#### Type
> header 的 type 必须是以下类型的其中之一

Must be one of the following:
- **build**: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)
- **ci**: Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **test**: Adding missing tests or correcting existing tests
#### Scope
> header 的 scope 是所影响的 npm package 的名称

The scope should be the name of the npm package affected (as perceived by the person reading the changelog generated from commit messages.

The following is the list of supported scopes:
- **animations**
- **common**
- **compiler**
- **compiler-cli**
- **core**
- **elements**
- **forms**
- **http**
- **language-service**
- **platform-browser**
- **platform-browser-dynamic**
- **platform-server**
- **platform-webworker**
- **platform-webworker-dynamic**
- **router**
- **service-worker**
- **upgrade**

There are currently a few exceptions to the "use package name" rule:
- **packaging**: used for changes that change the npm package layout in all of our packages, e.g. public path changes, package.json changes done to all packages, d.ts file/format changes, changes to bundles, etc.
- **changelog**: used for updating the release notes in CHANGELOG.md
- **aio**: used for docs-app (angular.io) related changes within the /aio directory of the repo
- none/empty string: useful for `style`, `test` and `refactor` changes that are done across all packages (e.g. `style: add missing semicolons`)
#### Subject
> header 的 subject 首字母不需要大写

The subject contains a succinct description of the change:
- use the imperative, present tense: "change" not "changed" nor "changes"
- don't capitalize the first letter
- no dot (.) at the end
#### Revert
If the commit reverts a previous commit, it should begin with `revert:` , followed by the header of the reverted commit. In the body it should say: `This reverts commit <hash>.`, where the hash is the SHA of the commit being reverted.
### Body
> body 使用使用祈使语气，现在时态
> body 包含 change 的 motivation 以及和之前行为的对比

Just as in the **subject**, use the imperative, present tense: "change" not "changed" nor "changes". The body should include the motivation for the change and contrast this with previous behavior.
### Footer
> footer 用于 refer to 解决的 issues 以及 breaking changes 块

The footer should contain any information about **Breaking Changes** and is also the place to reference GitHub issues that this commit **Closes**.

**Breaking Changes** should start with the word `BREAKING CHANGE:` with a space or two newlines. The rest of the commit message is then used for this.

A detailed explanation can be found in this [document](https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit#).
## Conventional Commits 1.0.0
### Summary
The Conventional Commits specification is a lightweight convention on top of commit messages. It provides an easy set of rules for creating an explicit commit history; which makes it easier to write automated tools on top of. This convention dovetails with [SemVer](http://semver.org/), by describing the features, fixes, and breaking changes made in commit messages.

The commit message should be structured as follows:

---

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

---

The commit contains the following structural elements, to communicate intent to the consumers of your library:
1. **fix:** a commit of the _type_ `fix` patches a bug in your codebase (this correlates with [`PATCH`](http://semver.org/#summary) in Semantic Versioning).
2. **feat:** a commit of the _type_ `feat` introduces a new feature to the codebase (this correlates with [`MINOR`](http://semver.org/#summary) in Semantic Versioning).
3. **BREAKING CHANGE:** a commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope, introduces a breaking API change (correlating with [`MAJOR`](http://semver.org/#summary) in Semantic Versioning). A BREAKING CHANGE can be part of commits of any _type_.
4. _types_ other than `fix:` and `feat:` are allowed, for example [@commitlint/config-conventional](https://github.com/conventional-changelog/commitlint/tree/master/%40commitlint/config-conventional) (based on the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)) recommends `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others.
5. _footers_ other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to [git trailer format](https://git-scm.com/docs/git-interpret-trailers).

Additional types are not mandated by the Conventional Commits specification, and have no implicit effect in Semantic Versioning (unless they include a BREAKING CHANGE). A scope may be provided to a commit’s type, to provide additional contextual information and is contained within parenthesis, e.g., `feat(parser): add ability to parse arrays`.
### Examples
#### Commit message with description and breaking change footer
```
feat: allow provided config object to extend other configs

BREAKING CHANGE: `extends` key in config file is now used for extending other config files
```
#### Commit message with `!` to draw attention to breaking change
> breaking change 在 type/scope 可以添加 `!`
```
feat!: send an email to the customer when a product is shipped
```
#### Commit message with scope and `!` to draw attention to breaking change
```
feat(api)!: send an email to the customer when a product is shipped
```
#### Commit message with both `!` and BREAKING CHANGE footer
```
chore!: drop support for Node 6

BREAKING CHANGE: use JavaScript features not available in Node 6.
```
#### Commit message with no body
```
docs: correct spelling of CHANGELOG
```
#### Commit message with scope
```
feat(lang): add Polish language
```
#### Commit message with multi-paragraph body and multiple footers
```
fix: prevent racing of requests

Introduce a request id and a reference to latest request. Dismiss
incoming responses other than from latest request.

Remove timeouts which were used to mitigate the racing issue but are
obsolete now.

Reviewed-by: Z
Refs: #123
```
### Specification
> commit 必须有类型，需要冒号加空格 `:<space>` ，
> `feat` 在添加新特性时使用
> `fix` 在修复 bug 时使用
> scope 可选，必须为名词，描述代码库的某个范围
> subject 是对 code changes 的简短总结
> footer 的格式为 `<word token>:<space><string value>`  或 `<word token><space>#<string value>` 
> footer token 使用 `-` 连接，不使用空格，除了 `BREAKING CHANGE`

The key words “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHALL NOT”, “SHOULD”, “SHOULD NOT”, “RECOMMENDED”, “MAY”, and “OPTIONAL” in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).
1. Commits MUST be prefixed with a type, which consists of a noun, `feat`, `fix`, etc., followed by the OPTIONAL scope, OPTIONAL `!`, and REQUIRED terminal colon and space.
2. The type `feat` MUST be used when a commit adds a new feature to your application or library.
3. The type `fix` MUST be used when a commit represents a bug fix for your application.
4. A scope MAY be provided after a type. A scope MUST consist of a noun describing a section of the codebase surrounded by parenthesis, e.g., `fix(parser):`
5. A description MUST immediately follow the colon and space after the type/scope prefix. The description is a short summary of the code changes, e.g., _fix: array parsing issue when multiple spaces were contained in string_.
6. A longer commit body MAY be provided after the short description, providing additional contextual information about the code changes. The body MUST begin one blank line after the description.
7. A commit body is free-form and MAY consist of any number of newline separated paragraphs.
8. One or more footers MAY be provided one blank line after the body. Each footer MUST consist of a word token, followed by either a `:<space>` or `<space>#` separator, followed by a string value (this is inspired by the [git trailer convention](https://git-scm.com/docs/git-interpret-trailers)).
9. A footer’s token MUST use `-` in place of whitespace characters, e.g., `Acked-by` (this helps differentiate the footer section from a multi-paragraph body). An exception is made for `BREAKING CHANGE`, which MAY also be used as a token. 
10. A footer’s value MAY contain spaces and newlines, and parsing MUST terminate when the next valid footer token/separator pair is observed.
11. Breaking changes MUST be indicated in the type/scope prefix of a commit, or as an entry in the footer. (Breaking change 要么在 header 中添加上 `!` ，要么在 footer 中添加 BREAKING CHANGE 块)
12. If included as a footer, a breaking change MUST consist of the uppercase text BREAKING CHANGE, followed by a colon, space, and description, e.g., _BREAKING CHANGE: environment variables now take precedence over config files_.
13. If included in the type/scope prefix, breaking changes MUST be indicated by a `!` immediately before the `:`. If `!` is used, `BREAKING CHANGE:` MAY be omitted from the footer section, and the commit description SHALL be used to describe the breaking change.
14. Types other than `feat` and `fix` MAY be used in your commit messages, e.g., _docs: update ref docs._
15. The units of information that make up Conventional Commits MUST NOT be treated as case sensitive by implementors, with the exception of BREAKING CHANGE which MUST be uppercase.
16. BREAKING-CHANGE MUST be synonymous with BREAKING CHANGE, when used as a token in a footer.
### Why Use Conventional Commits
- Automatically generating CHANGELOGs.
- Automatically determining a semantic version bump (based on the types of commits landed).
- Communicating the nature of changes to teammates, the public, and other stakeholders.
- Triggering build and publish processes.
- Making it easier for people to contribute to your projects, by allowing them to explore a more structured commit history.
## Semantic Versioning 2.0.0
### Summary
Given a version number MAJOR.MINOR.PATCH, increment the:
1. MAJOR version when you make incompatible API changes
2. MINOR version when you add functionality in a backward compatible manner
3. PATCH version when you make backward compatible bug fixes
Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.
> 版本号格式：主版本号. 次版本号. 修订号
> 主版本号在做了不兼容的 API 修改时使用
> 次版本号在做了新的向下兼容的功能时使用
> 修订号在做了向下兼容的 bug fix 时使用
> 先行版本号和构建元数据可以添加到版本号的后面
### Introduction
In the world of software management there exists a dreaded place called “dependency hell.” The bigger your system grows and the more packages you integrate into your software, the more likely you are to find yourself, one day, in this pit of despair.

In systems with many dependencies, releasing new package versions can quickly become a nightmare. If the dependency specifications are too tight, you are in danger of version lock (the inability to upgrade a package without having to release new versions of every dependent package). If dependencies are specified too loosely, you will inevitably be bitten by version promiscuity (assuming compatibility with more future versions than is reasonable). Dependency hell is where you are when version lock and/or version promiscuity prevent you from easily and safely moving your project forward.

As a solution to this problem, we propose a simple set of rules and requirements that dictate how version numbers are assigned and incremented. These rules are based on but not necessarily limited to pre-existing widespread common practices in use in both closed and open-source software. For this system to work, you first need to declare a public API. This may consist of documentation or be enforced by the code itself. Regardless, it is important that this API be clear and precise. Once you identify your public API, you communicate changes to it with specific increments to your version number. Consider a version format of X.Y.Z (Major.Minor.Patch). Bug fixes not affecting the API increment the patch version, backward compatible API additions/changes increment the minor version, and backward incompatible API changes increment the major version.
> 我们的软件定义了公共 API，我们通过版本号向大家说明我们对于 API 的修改
> 不会影响 API 的 bug 修复增加 PATCH 版本号
> API 保持向下兼容的新增和修改增加 MINOR 版本号
> 向后不兼容的修改增加 MAJOR 版本号

We call this system “Semantic Versioning.” Under this scheme, version numbers and the way they change convey meaning about the underlying code and what has been modified from one version to the next.
> 我们通过版本号的修改表达了我们的代码修改的信息
### Semantic Versioning Specification (SemVer)
The key words “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHALL NOT”, “SHOULD”, “SHOULD NOT”, “RECOMMENDED”, “MAY”, and “OPTIONAL” in this document are to be interpreted as described in [RFC 2119](https://tools.ietf.org/html/rfc2119).
> 使用语义化版本控制的软件必须定义公共 API
> 版本号不能包括先导零
> 标记版本号的软件发布后，该版本的内容就不能再修改，任意的修改需要以新版本的形式发布
> 版本号 1.0.0 用于表示公共 API 的形成
> 次版本号必须在出现向下兼容的新功能时递增，在公共 API 的任何功能被标记为弃用时，次版本号也需要递增，其中可以包含 PATCH 级别的修改
> 在有任何不向下兼容的修改被加入 API 时，主版本号必须递增，其中可以包括 MINOR 和 PATCH 级别的修改
 >先行版本号可以加在 PATCH 之后，先加上一个 `-` ，然后是一系列 `.` 分割的标识符，标识符的范围是 `[0-9A-Za-z-]` ，不能有空格，数字标识符不能有先导零；先行版本的优先级低于对应的标准版本，先行版本表示该发布可能不稳定，且不一定满足对应的兼容性标准
> 版本编译信息可以被标注在先行版本号之后，先加上一个 `+` ，然后是一系列 `.` 分割的标识符，判断版本的优先级时，版本编译信息可以被忽略，因此仅在编译信息有差别的版本优先级相同
> 版本的优先级即不同的版本在排序时如何比较

1.  Software using Semantic Versioning MUST declare a public API. This API could be declared in the code itself or exist strictly in documentation. However it is done, it SHOULD be precise and comprehensive.
2.  A normal version number MUST take the form X.Y.Z where X, Y, and Z are non-negative integers, and MUST NOT contain leading zeroes. X is the major version, Y is the minor version, and Z is the patch version. Each element MUST increase numerically. For instance: 1.9.0 -> 1.10.0 -> 1.11.0.
3.  Once a versioned package has been released, the contents of that version MUST NOT be modified. Any modifications MUST be released as a new version.
4.  Major version zero (0.y.z) is for initial development. Anything MAY change at any time. The public API SHOULD NOT be considered stable.
5. Version 1.0.0 defines the public API. The way in which the version number is incremented after this release is dependent on this public API and how it changes.
6.  Patch version Z (x.y.Z | x > 0) MUST be incremented if only backward compatible bug fixes are introduced. A bug fix is defined as an internal change that fixes incorrect behavior.
7.  Minor version Y (x.Y.z | x > 0) MUST be incremented if new, backward compatible functionality is introduced to the public API. It MUST be incremented if any public API functionality is marked as deprecated. It MAY be incremented if substantial new functionality or improvements are introduced within the private code. It MAY include patch level changes. Patch version MUST be reset to 0 when minor version is incremented.
8.  Major version X (X.y.z | X > 0) MUST be incremented if any backward incompatible changes are introduced to the public API. It MAY also include minor and patch level changes. Patch and minor versions MUST be reset to 0 when major version is incremented.
9.  A pre-release version MAY be denoted by appending a hyphen and a series of dot separated identifiers immediately following the patch version. Identifiers MUST comprise only ASCII alphanumerics and hyphens [0-9A-Za-z-]. Identifiers MUST NOT be empty. Numeric identifiers MUST NOT include leading zeroes. Pre-release versions have a lower precedence than the associated normal version. A pre-release version indicates that the version is unstable and might not satisfy the intended compatibility requirements as denoted by its associated normal version. Examples: `1.0.0-alpha, 1.0.0-alpha.1, 1.0.0-0.3.7, 1.0.0-x.7.z.92, 1.0.0-x-y-z.--.`
10.  Build metadata MAY be denoted by appending a plus sign and a series of dot separated identifiers immediately following the patch or pre-release version. Identifiers MUST comprise only ASCII alphanumerics and hyphens [0-9A-Za-z-]. Identifiers MUST NOT be empty. Build metadata MUST be ignored when determining version precedence. Thus two versions that differ only in the build metadata, have the same precedence. Examples: `1.0.0-alpha+001, 1.0.0+20130313144700, 1.0.0-beta+exp.sha.5114f85, 1.0.0+21AF26D3----117B344092BD`.
11.  Precedence refers to how versions are compared to each other when ordered.
    1. Precedence MUST be calculated by separating the version into major, minor, patch and pre-release identifiers in that order (Build metadata does not figure into precedence).
    2. Precedence is determined by the first difference when comparing each of these identifiers from left to right as follows: Major, minor, and patch versions are always compared numerically. Example: 1.0.0 < 2.0.0 < 2.1.0 < 2.1.1.
    3. When major, minor, and patch are equal, a pre-release version has lower precedence than a normal version: Example: 1.0.0-alpha < 1.0.0.
    4. Precedence for two pre-release versions with the same major, minor, and patch version MUST be determined by comparing each dot separated identifier from left to right until a difference is found as follows:
        1. Identifiers consisting of only digits are compared numerically.
        2. Identifiers with letters or hyphens are compared lexically in ASCII sort order.
        3. Numeric identifiers always have lower precedence than non-numeric identifiers.
        4. A larger set of pre-release fields has a higher precedence than a smaller set, if all of the preceding identifiers are equal. Example: 1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-alpha.beta < 1.0.0-beta < 1.0.0-beta.2 < 1.0.0-beta.11 < 1.0.0-rc.1 < 1.0.0.
# Style 2 Old Style
## Templates
### Template 1
This is a template, [written originally by Tim Pope](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html), which appears in the [_Pro Git Book_](https://git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project).
```
Summarize changes in around 50 characters or less
# 50以内字符描述更改

More detailed explanatory text, if necessary. Wrap it to about 72
characters or so. In some contexts, the first line is treated as the
subject of the commit and the rest of the text as the body. The
blank line separating the summary from the body is critical (unless
you omit the body entirely); various tools like `log`, `shortlog`
and `rebase` can get confused if you run the two together.
# 提供更加详细的补充说明

Explain the problem that this commit is solving. Focus on why you
are making this change as opposed to how (the code explains that).
Are there side effects or other unintuitive consequences of this
change? Here's the place to explain them.
# 解释当前 commit 所解决的问题
# 聚焦于产生此更改的原因，而非手段
# 如果有，解释该 change 是否有副作用或其他不直观的影响

Further paragraphs come after blank lines.

 - Bullet points are okay, too

 - Typically a hyphen or asterisk is used for the bullet, preceded
   by a single space, with blank lines in between, but conventions
   vary here
# 可以使用要点 (bullet points)

If you use an issue tracker, put references to them at the bottom,
like this:

Resolves: #123
See also: #456, #789
# 使用 issue tracker 的话，将对 issue 的引用放在底部
# 例如
# Resolves: #123 
# See also: #456, #789
```
### Template 2
```
##################################################
# Write a title summarizing what this commit does.
# Start with an uppercase imperative verb, such as
# Add, Drop, Fix, Refactor, Bump; see ideas below.
# Think of your title as akin to an email subject,
# so you don't need to end with a sentence period.
# Use 50 char maximum, which is this line's width.
##################################################
Add your title here

########################################################################
# Why is this change happening?
# Describe the purpose, such as a goal, or use case, or user story, etc.
# For every line, use 72 char maximum width, which is this line's width.
########################################################################
Why:

########################################################################
# How is this change happening?
# Describe any relevant algorithms, protocols, implementation spec, etc.
# For every line, use 72 char maximum width, which is this line's width.
########################################################################
How:

########################################################################
# Add any tags you want, such as search text, hashtags, keywords, codes.
# For every line, use 72 char maximum width, which is this line's width.
########################################################################
Tags:

########################################################################
#
# ## Help ##
#
# Subject line imperative uppercase verbs:
#
#   Add = Create a capability e.g. feature, test, dependency.
#   Drop = Delete a capability e.g. feature, test, dependency.
#   Fix = Fix an issue e.g. bug, typo, accident, misstatement.
#   Bump = Increase the version of something e.g. a dependency.
#   Make = Change the build process, or tools, or infrastructure.
#   Start = Begin doing something; e.g. enable a toggle, feature flag, etc.
#   Stop = End doing something; e.g. disable a toggle, feature flag, etc.
#   Optimize = A change that MUST be just about performance, e.g. speed up code.
#   Document = A change that MUST be only in the documentation, e.g. help files.
#   Refactor = A change that MUST be just refactoring.
#   Reformat = A change that MUST be just format, e.g. indent line, trim space, etc.
#   Rephrase = A change that MUST be just textual, e.g. edit a comment, doc, etc.
#
# For the subject line:
#
#   * Use 50 characters maximum.
#
#   * Do not use a sentence-ending period.
#
# For the body text:
#
#   * Use as many lines as you like.
#
#   * Use 72 characters maximum per line for typical word wrap text.
#
#
# ## Trailers ##
#
# Trailers suitable for tracking and also for `git interpret-trailers`.
#
# Example of "See: " trailers that mean "see this additional information"
# and links to relevant web pages, issue trackers, blog posts, etc.:
#
#     See: https://example. com/
#     See: Issue #123 <https://example. com/issue/123>
#
# We like to use the "See: " trailers to link to issue trackers (e.g. Jira, 
# Asana, Basecamp, Trello), document files and folders (e.g. Box, Dropbox),
# UI/UX designs (e.g. Figma, Lucidchart), reference pages (e.g. Wikipedia,
# internet RFCs, IEEE standards), and web posts (e.g. StackOverflow, HN).
#
# Example of "Co-authored-by: " trailers that list each author's name
# and their preferred commit message email address or contact URL:
#
#     Co-authored-by: Alice Adams <alice@example. com>
#     Co-authored-by: Bob Brown <https://bob. example. com>
#
# We like to use the "Co-authored-by: " trailers when we pair program,
# triple program, and group program. These are parsed automatically by
# some version control services (e.g. GitHub, GitLab) and will link
# to the authors' accounts and show up on the authors' commit history.
#
# Example of "Sponsored-by: " trailers that list each sponsor's name,
# which could be a person's or organization's, and contact email or URL:
#
#     Sponsored-by: Adam Anderson <adam@example. com>
#     Sponsored-by: Bravo Organization <https://bravo. example. com>
#
# The git tools require trailers to be last in a commit message,
# and must be one trailer per line, and with no extra lines between.
#
#
# ## Usage ##
#
# Put the template file here:
#
#     ~/. git-commit-template. txt
#
# Configure git to use the template file by running:
#
#     git config --global commit. template ~/. git-commit-template. txt
#
# Add the template file to the ~/. gitconfig file:
#
#     [commit]
#       template = ~/. git-commit-template. txt
#
# If you prefer other file locations or ways of working,
# you can freely adjust the usage as you like.
#
#
# ## Usage needs commit. cleanup strip ##
#
# This template intends for the commit to strip the comments.
#
# To strip the comments, your git `commit. cleanup` config must be `strip`.
#
# If you don't use `strip`, then these commit comments won't be deleted.
#
#
# ## More ideas ##
#
# Some teams like to add a git commit message verification processes,
# such as a git pre-commit hook that runs a linter on the message text.
# 
# In our experience, this can be helpful especially if the linter can
# provide advice that explains how to make the message better.
#
#
# ## Tracking ##
#
# * Package: git-commit-template
# * Version: 7.2.0
# * Updated: 2022-11-22T00:55:28Z
# * Licence: GPL-2.0-or-later or contact us for custom license.
# * Contact: Joel Parker Henderson (http://joelparkerhenderson. com)
#
########################################################################

### GIT TRAILERS -- THESE MUST BE LAST IN THE COMMIT MESSAGE ###

# Git trailers are optional. Use them if you want, how you want.
# The trailers below are provided as examples that you can customize.

# For example, you can add any relevant links to a blog post, or graphic
# design images, or industry publications, specifications, tickets, etc.
#See: Description <https://example. com/...>
#See: Description <https://example. com/...>

# If the commit is written by multiple people, then use the git trailers
# to thank each person as a co-author; various git tools can track this.
#Co-authored-by: Name <name@example. com>
#Co-authored-by: Name <name@example. com>

# If the commit is sponsored by a person or company, then add them here.
# This kind of trailer is more-frequent in open source funding projects.
#Sponsored-by: Name <name@example. com>
#Sponsored-by: Name <name@example. com>
```

## Commit Message Header
### Use imperative form
> 祈使句

```
# Good
Use InventoryBackendPool to retrieve inventory backend
```

```
# Bad
Used InventoryBackendPool to retrieve inventory backend
```

_But why use the imperative form?_

A commit message describes what the referenced change actually **does**, its effects, not what was done.
### Capitalize the first letter
> 首字母大写

```
# Good
Add `use` method to Credit model
```

```
# Bad
add `use` method to Credit model
```

The reason that the first letter should be capitalized is to follow the grammar rule of using capital letters at the beginning of sentences.

The use of this practice may vary from person to person, team to team, or even from language to language. Capitalized or not, an important point is to stick to a single standard and follow it.
### Try to communicate what the change does without having to look at the source code
> 尝试让 commit message header 在不需要 viewer 查看源码的情况下表达出该 commit 做了哪些改变 (change)

```
# Good
Add `use` method to Credit model
```

```
# Bad
Add `use` method
```

```
# Good
Increase left padding between textbox and layout frame
```

```
# Bad
Adjust css
```

It is useful in many scenarios (e.g. multiple commits, several changes and refactors) to help reviewers understand what the committer was thinking.
## Commit Message Body
### Use the message body to explain "why", "for what", "how" and additional details
> Commit message body 用于解释该 commit 的原因、目的、手段以及其他细节

```
# Good
Fix method name of InventoryBackend child classes

Classes derived from InventoryBackend were not
respecting the base class interface.

It worked because the cart was calling the backend implementation
incorrectly.
```

```
# Good
Serialize and deserialize credits to json in Cart

Convert the Credit instances to dict for two main reasons:

  - Pickle relies on file path for classes and we do not want to break up
    everything if a refactor is needed
  - Dict and built-in types are pickleable by default
```

```
# Good
Add `use` method to Credit

Change from namedtuple to class because we need to
setup a new attribute (in_use_amount) with a new value
```

The subject and the body of the messages are separated by a blank line. Additional blank lines are considered as a part of the message body.

Characters like `-`, `*` and ` are elements that improve readability.
### Avoid generic messages or messages without any context
> 避免使用无上下文的信息
 
```
# Bad
Fix this

Fix stuff

It should work now

Change stuff

Adjust css
```
### Limit the number of characters
> header 不超过50个字符，body 每一行不超过 72字符

[It's recommended](https://git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project#_commit_guidelines) to use a maximum of 50 characters for the subject and 72 for the body.
### Keep language consistency
> 保持用同一个语言 (如英语)

For project owners: Choose a language and write all commit messages using that language. Ideally, it should match the code comments, default translation locale (for localized projects), etc.

For contributors: Write your commit messages using the same language as the existing commit history.

```
# Good
ababab Add `use` method to Credit model
efefef Use InventoryBackendPool to retrieve inventory backend
bebebe Fix method name of InventoryBackend child classes
```

```
# Good (Portuguese example)
ababab Adiciona o método `use` ao model Credit
efefef Usa o InventoryBackendPool para recuperar o backend de estoque
bebebe Corrige nome de método na classe InventoryBackend
```

```
# Bad (mixes English and Portuguese)
ababab Usa o InventoryBackendPool para recuperar o backend de estoque
efefef Add `use` method to Credit model
cdcdcd Agora vai
```

# Appendix
## Commit Message Guide for Obsidian Vault
**Template:**

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

types:
- note
- log
- idea
- chore: about obsidian settings, plugins etc...

scopes:
- note
    - paper
    - book
    - lecture
    - doc
    - course
    - causal
    - homework

**Rules:**
1. General guideline refers to the angular style.
2. Log commit should be done daily, and the commit message should summarize the daily progress.
3. Log commit should in principle be the last commit of the day, before log commit, other types of commit related to the daily progress should be done.

