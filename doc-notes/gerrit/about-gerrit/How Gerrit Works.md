>  version v3.11.2

To learn how Gerrit fits into and complements the developer workflow, consider a typical project. The following project contains a central source repository (_Authoritative Repository_) that serves as the authoritative version of the project’s contents.
>  考虑一个典型的项目，它包含了中心的源码仓库 (权威仓库)，作为该项目内容的权威版本

![Authoritative Source Repository](https://gerrit-documentation.storage.googleapis.com/Documentation/3.11.2/images/intro-quick-central-repo.png)

Figure 1. Central Source Repository

When implemented, Gerrit becomes the central source repository and introduces an additional concept: a store of _Pending Changes_.
>  实现了 Gerrit 后，Gerrit 会成为中心源码仓库
>  Gerrit 引入一层额外概念: 待处理更改仓库

![Gerrit as the Central Repository](https://gerrit-documentation.storage.googleapis.com/Documentation/3.11.2/images/intro-quick-central-gerrit.png)

Figure 2. Gerrit as the Central Repository

When Gerrit is configured as the central source repository, all code changes are sent to Pending Changes for others to review and discuss. When enough reviewers have approved a code change, you can submit the change to the code base.
>  当 Gerrit 被配置为中心源码仓库后，所有的代码更改都会发送到待处理更改仓库中，供审查和讨论
>  当一个更改被足够多的 reviewer 批准后，更改可以提交到 code base

In addition to the store of Pending Changes, Gerrit captures notes and comments made about each change. This enables you to review changes at your convenience or when a conversation about a change can’t happen in person. In addition, notes and comments provide a history of each change (what was changed and why and who reviewed the change).
>  Gerrit 除了存储待处理更改外，还会捕获关于每个更改的注释和评论
>  注释和评论为每个更改提供了历史记录 (更改了什么、什么原因、谁审查了该更改)

Like any repository hosting product, Gerrit provides a powerful [access control model](https://gerrit-documentation.storage.googleapis.com/Documentation/3.11.2/access-control.html), which enables you to fine-tune access to your repository.
>  Gerrit 允许自行调整对仓库的访问权限

---