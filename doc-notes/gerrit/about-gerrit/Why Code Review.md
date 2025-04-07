---
completed: true
---
# Use Gerrit to Be a Rockstar Programmer
>  version v3.11.2

## Overview
The term _rockstar_ is often used to describe those talented programmers who seem to work faster and better than everyone else, much like a composer who seems to effortlessly churn out fantastic music. However, just as the spontaneity of masterful music is a fantasy, so is the development of exceptional code.

The process of composing and then recording music is painstaking — the artist records portions of a composition over and over, changing each take until one song is completed by combining those many takes into a cohesive whole. The end result is the recording of the best performance of the best version of the song.

Consider Queen’s six-minute long Bohemian Rhapsody, which took three weeks to record. Some segments were overdubbed 180 times!

Software engineering is much the same. Changes that seem logical and straightforward in retrospect actually required many revisions and many hours of work before they were ready to be merged into a code base. A single conceptual code change (_fix bug 123_) often requires numerous iterations before it can be finalized. Programmers typically:

- Fix compilation errors
- Factor out a method, to avoid duplicate code
- Use a better algorithm, to make it faster
- Handle error conditions, to make it more robust
- Add tests, to prevent a bug from regressing
- Adapt tests, to reflect changed behavior
- Polish code, to make it easier to read
- Improve the commit message, to explain why a change was made

In fact, first drafts of code changes are best kept out of project history. Not just because rockstar programmers want to hide sloppy first attempts at making something work. It’s more that keeping intermediate states hampers effective use of version control. Git works best when one commit corresponds to one functional change, as is required for:

- git revert
- git cherry-pick
- [git bisect](https://www.kernel.org/pub/software/scm/git/docs/git-bisect-lk2009.html)

## Amending commits
Git provides a mechanism to continually update a commit until it’s perfect: use `git commit --amend` to remake (re-record) a code change. After you update a commit in this way, your branch then points to the new commit. However, the older (imperfect) revision is not lost. It can be found via the `git reflog`.
>  `git commit --amend` 用于持续更新某个提交，实际上被更新的旧提交没有消失，可以通过 `git reflog` 找到

## Code review
At least two well-known open source projects insist on these practices:

- [Git](http://git-scm.com/)
- [Linux Kernel](http://www.kernel.org/category/about.html)

However, contributors to these projects don’t refine and polish their changes in private until they’re perfect. Instead, polishing code is part of a review process — the contributor offers their change to the project for other developers to evaluate and critique. 
>  贡献者将更改提交给项目，供其他开发者评估，该过程即 code review

This process is called _code review_ and results in numerous benefits:

- Code reviews mean that every change effectively has shared authorship
- Developers share knowledge in two directions: Reviewers learn from the patch author how the new code they will have to maintain works, and the patch author learns from reviewers about best practices used in the project.
- Code review encourages more people to read the code contained in a given change. As a result, there are more opportunities to find bugs and suggest improvements.
- The more people who read the code, the more bugs can be identified. Since code review occurs before code is submitted, bugs are squashed during the earliest stage of the software development lifecycle.
- The review process provides a mechanism to enforce team and company policies. For example, _all tests shall pass on all platforms_ or _at least two people shall sign off on code in production_.

>  code review 的好处有
>  - 意味着每个更改都具有共同的作者身份
>  - 开发人员双向地共享知识: Reviewers <-> Author
>  - 更有可能发现 bug 或提出改进
>  - 因为 code review 在 code 被提交之前执行，故可以在软件开发周期的最早阶段消除 bug
>  - 审查过程提供了一种机制来执行团队和公司的政策，例如，“所有测试必须在所有平台上通过”或“至少两个人必须对生产环境中的代码进行确认”

Many successful software companies, including Google, use code review as a standard, integral stage in the software development process.

## Web-based code review
To review work, the Git and Linux Kernel projects send patches via email.

Code Review (Gerrit) adds a modern web interface to this workflow. Rather than send patches and comments via email, Gerrit users push commits to Gerrit where diffs are displayed on a web page. Reviewers can post comments directly on the diff. If a change must be reworked, users can push a new, amended revision of the same change. Reviewers can then check if the new revision addresses the original concerns. If not, the process is repeated.
>  Gerrit 为 Code Review 实现 web interface
>  Gerrit 用户将 commits 推送到 Gerrit, Gerrit 会直接在网页上显示 diffs, Reviewers 可以直接在 diff 上评论
>  如果需要重新修改变更，用户可以推送一个新的 amended revision, Reviewers 继而检查新 revision 是否解决原始问题，如果没有，流程将重复进行

## Gerrit’s magic
When you push a change to Gerrit, how does Gerrit detect that the commit amends a previous change? Gerrit can’t use the SHA-1, since that value changes when `git commit --amend` is called. 
>  Gerrit 不使用 SHA-1 来检测新的 commit 是否 amend 了之前的更改，因为在 `git commit --amend` 调用后，SHA-1 值一定会变更

>  这里要解决的是判断概念性更改的问题，即我们可能会在 Code Review 的过程中多次 amend 某个 commit，这些 commits 都指向同一个概念性更改

Fortunately, upon amending a commit, the commit message is retained by default. This is where Gerrit’s solution lies: Gerrit identifies a conceptual  change with a footer in the commit message. Each commit message footer contains a Change-Id message hook, which uniquely identifies a change across all its drafts. For example:

`Change-Id: I9e29f5469142cc7fce9e90b0b09f5d2186ff0990`

>  Gerrit 利用 `git commit --amend` 默认保留 commit message 的机制
>  Gerrit 会通过 commit message 中的 footer 来识别概念性更改，每个 commit message footer 都包含一个 Change-Id message hook, 该 hook 会唯一地标识一次更改，Change-Id message hook 的示例如上

Thus, if the Change-Id remains the same as commits are amended, Gerrit detects that each new version refers to the same conceptual change. The Gerrit web interface groups versions so that reviewers can see how your change evolves during the code review.
>  如果 Change-Id 在 commit 被 amend 之后保持不变, Gerrit 会检测到新的 commit 仍然指向相同的概念性更改
>  Gerrit 的 web interface 会按照概念性更改给 commits 分组，故 reviewers 可以审查一个概念性更改的演化历史

To Gerrit, the identifier can be random.
>  Gerrit 会随机为概念性更改赋予 Change-Id

Gerrit provides a client-side [message hook](https://gerrit-documentation.storage.googleapis.com/Documentation/3.11.2/cmd-hook-commit-msg.html) to automatically add to commit messages when necessary.
>  Gerrit 提供了 client-side 的 message hook ，可以在必要时添加到 commit message 

