# Getting Started
## About collaborative development models
The way you use pull requests depends on the type of development model you use in your project. You can use the fork and pull model or the shared repository model.
>  使用 pull request 的方式取决于我们项目采用的开发模型
>  开发模型有: fork and pull model， shared repository model

### Fork and pull model
In the fork and pull model, anyone can fork an existing ("upstream") repository to which they have read access and the owner of the upstream repository allows it. Be aware that a fork and its upstream share the same git data. This means that all content uploaded to a fork is accessible from the upstream and all other forks of that upstream. 
>  fork and pull model 中，任何人只要对 upstream repo 有读取权限，就可以 fork upstream repo
>  fork 和其 upstream 共享相同的 git 数据，这意味着所有上传到 fork 的数据都可以从 upstream 访问，也可以从 upstream 的所有其他 forks 访问

You do not need permission from the upstream repository to push to a fork of it you created. You can optionally allow anyone with push access to the upstream repository to make changes to your pull request branch. 
>  push 到 fork 不需要 upstream 的写入权限，我们可以选择性地允许任何对 upstream 有 push 权限的用户更改我们的 pull request branch

This model is popular with open-source projects as it reduces the amount of friction for new contributors and allows people to work independently without upfront coordination.
>  该模型常见于开源项目

Tip
For more information on open source, specifically how to create and grow an open source project, we've created [Open Source Guides](https://opensource.guide/) that will help you foster a healthy open source community. You can also take a free [GitHub Skills](https://skills.github.com/) course on maintaining open source communities.

### Shared repository model
In the shared repository model, collaborators are granted push access to a single shared repository and topic branches are created when changes need to be made. 
>  shared repository model 中，协作者被直接赋予对单个共享 repo 的 push 权限，当需要做出更改时，会创建 topic branches

Pull requests are useful in this model as they initiate code review and general discussion about a set of changes before the changes are merged into the main development branch. 
>  在该模型中，pull requests 也非常有用，因为它们可以在一系列更高被合并到 main branch 之前，发起 code review 和讨论

This model is more prevalent with small teams and organizations collaborating on private projects.
>  该模型常见于小团队和组织协作的私有项目

### Further reading
- [About pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
- [Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
- [Allowing changes to a pull request branch created from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork)

## Helping others review your changes
You can use pull requests to provide clear context for your changes and keep your team informed, improving collaboration and the quality of reviews.

When you create a pull request, you’re asking your team to review your changes and provide feedback. 
>  当我们创建一个 pull request 时，我们是在向团队请求审查我们的更改并提供反馈

This guide provides best practices for creating pull requests that are easy to review and keep your team informed, so that you can improve collaboration and the quality of reviews.

### Making your changes easy to review
Clear context in your pull requests helps reviewers quickly see what you’ve changed and why it matters. This makes the review process faster and smoother, with less back-and-forth, and helps your team give better feedback and make confident decisions. For information on creating a pull request, see [Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

#### Write small pull requests
Aim to create small, focused pull requests that fulfill a single purpose. Smaller pull requests are easier and faster to review and merge, leave less room to introduce bugs, and provide a clearer history of changes.
>  力求创建小型的，集中的 pull requests, pull requests 应该仅实现单一的目的
>  较小的 pull requests 更容易和更快进行审查和合并，更不容易引入 bug，并能提供更清晰的变更历史记录

#### Provide context and guidance
Write clear titles and descriptions for your pull requests so that reviewers can quickly understand what the pull request does. 
>  为 pr 创建清晰的标题和描述

In the pull request body, include:

- The purpose of the pull request
- An overview of what changed
- Links to any additional context such as tracking issues or previous conversations

>  pr 的内容体中，需要包含
>  - pr 的目的
>  - 对改变内容的纵览
>  - 对任意额外上下文的链接，例如 issues 或之前的讨论

To help reviewers, share the type of feedback you need. For example, do you need a quick look or a deeper critique? Additionally, you can use GitHub Copilot to generate a summary of your pull request. See [Use GitHub Copilot to generate pull request summaries](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes#use-github-copilot-to-generate-pull-request-summaries), later in this article.
>  在 pr 中写出所需要的反馈类型
>  例如，是需要快速浏览还是深入的讨论
>  此外，还可以用 Copilot 生成 pr 的摘要

If your pull request consists of changes to multiple files, provide guidance to reviewers about the order in which to review the files. Recommend where to start and how to proceed with the review.
>  如果 pr 包含多个文件，提供一个指引，说明审查顺序

#### Review your own pull request first
Review, build, and test your own pull request before submitting it. This will allow you to catch errors or typos that you may have missed, before others start reviewing.
>  在提交 pr 之前，自己先审查、构建、测试

#### Review for security
There are various tools available that can help you review your pull request for potential security issues before others review it. Reviewing for security helps to catch and resolve security issues early, and lets you highlight unresolved risks for others to review and advise on. For example, you can:

- Check the dependency diff to see if your pull request is introducing vulnerable dependencies. See [Reviewing dependency changes in a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/reviewing-dependency-changes-in-a-pull-request).
- Check the GitHub Advisory Database to find additional context and information on vulnerable dependencies.
- Investigate and resolve any failing security checks or workflows, such as the dependency review action or the code scanning results check. See [About dependency review](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/about-dependency-review#about-the-dependency-review-action) and [Triaging code scanning alerts in pull requests](https://docs.github.com/en/code-security/code-scanning/managing-code-scanning-alerts/triaging-code-scanning-alerts-in-pull-requests#about-code-scanning-as-a-pull-request-check).
- If your repository has set up code scanning as a pull request check, use GitHub Copilot Autofix to suggest fixes for security vulnerabilities in your code. See [Triaging code scanning alerts in pull requests](https://docs.github.com/en/code-security/code-scanning/managing-code-scanning-alerts/triaging-code-scanning-alerts-in-pull-requests#working-with-copilot-autofix-suggestions-for-alerts-on-a-pull-request).

### Keeping your team informed
Pull requests can do more than just document code changes—they’re also a powerful way to keep your team and manager informed about the status of your work. By making your progress visible in your pull requests, you can reduce the need for separate updates and ensure everyone stays aligned.

#### Use GitHub Copilot to generate pull request summaries
Note
You'll need access to GitHub Copilot. For more information, see [What is GitHub Copilot?](https://docs.github.com/en/copilot/about-github-copilot/what-is-github-copilot#getting-access-to-copilot).

You can use Copilot to generate a summary of a pull request on GitHub. You can use the summary to help reviewers understand your changes.

1. On GitHub, create a pull request or navigate to an existing pull request.
    
    Note
    Copilot does not take into account any existing content in the pull request description, so it is best to start with a blank description.
    
2. Navigate to the text field where you want to add the pull request summary.
    
    - If you're creating a new pull request, use the "Add a description" field.
    - If you're adding a description to an existing pull request, edit the opening comment.
    - If you're adding a summary as a comment, navigate to the "Add a comment" section at the bottom of the pull request page.
3. In the header of the text field, select , then click **Summary**.
    
    ![Screenshot of the form for creating a pull request. A Copilot icon is highlighted, and a box appears with the "Summary" command.](https://docs.github.com/assets/cb-42638/images/help/copilot/copilot-description-suggestion.png)
    
4. Wait for Copilot to produce the summary, then check over the results carefully.
5. Add any additional context that will help people viewing your pull request.
6. When you're happy with the description, click **Create pull request** on a new pull request, or **Update comment** if you're editing an existing description.

Tip

You can also use Copilot Chat to turn your work into a discussion or blog post. See [Writing discussions or blog posts](https://docs.github.com/en/copilot/copilot-chat-cookbook/documenting-code/writing-discussions-or-blog-posts).

#### Link to related issues or projects
Connect your pull request to relevant issues or project boards to show how your work fits into the larger project.

- Add keywords like `Closes ISSUE-LINK` in your description to automatically link and close the issue when the pull request is merged.
- Use Projects to track your work and link to the project from your pull request, making progress easy to track in one place. See [About Projects](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects).

>  在 pr 描述中添加关键词例如 `Closes ISSUE-LINK`，以便在 merge pr 时自动关闭 issue

#### Highlight the status with labels
Add a status label to your pull request to show whether it’s ready for review, blocked, or in progress. This helps reviewers understand the state of your work at a glance. For more information, see [Managing labels](https://docs.github.com/en/issues/using-labels-and-milestones-to-track-work/managing-labels).
>  为 pr 添加状态标签，显示它是否准备被审查、受阻还是进行中

## Managing and standardizing pull requests
Use these steps to manage and standardize the pull requests that contributors create in your repository.

If you are a repository maintainer, there are several ways that you can manage and standardize the pull requests that contributors create in your repository. These steps can help you ensure that pull requests are reviewed by the right people, and that they meet your repository's standards.

### Using pull request templates
Pull request templates let you customize and standardize the information you'd like to be included when someone creates a pull request in your repository. When you add a pull request template to your repository, project contributors will automatically see the template's contents in the pull request body. For more information, see [Creating a pull request template for your repository](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository).
>  pr templates 可以标准化 pr 中需要包含的信息

You can use pull request templates to standardize the review process for your repository. For example, you can include a list of tasks that you would like authors to complete before merging their pull requests, by adding a task list to the template. For more information, see [About task lists](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/about-task-lists).
>  pr template 可以用于标准化 pr 的审查过程，例如可以添加一个希望作者在 merge 它们的 pr 之前要完成的任务列表

You can request that contributors include an issue reference in their pull request body, so that merging the pull request will automatically close the issue. For more information, see [Linking a pull request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).

### Defining code owners
You may want to make sure that specific individuals always review changes to certain code or files in your repository. For example, you may want to ensure that a member of the security team always reviews changes to your `SECURITY.md` file or `dependabot.yml` file.

You can define individuals or teams that you consider responsible for code or files in a repository to be code owners. Code owners will automatically be requested for review when someone opens a pull request that modifies the files that they own. You can define code owners for specific types of files or directories, as well as for different branches in a repository. For more information, see [About code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners).
>  可以定义特定的用户作为特定文件或目录，以及不同的分支的 code owner
>  code owner 在相关的文件被修改的 pr 被提出时，会被自动提醒

### Using protected branches
You can use protected branches to prevent pull requests from being merged into important branches, such as `main`, until certain conditions are met. For example, you can require an approving review, or require that all status checks are passing. See [About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
>  可以使用受保护的 branch 特性，确保 pr 在满足特定条件以前，不会被 merge 到特定 branch
>  例如，可以要求 pr 需要获得审查批准，或者要求所有状态检查都通过

### Using rulesets
Working alongside protected branches, rulesets let you enforce policies across your repository, such as requiring status checks or workflows to pass before a pull request can be merged.

Rulesets are especially useful for maintaining repository security when combined with other automated security checks. For example:

- You can use rulesets to enforce the dependency review action, a workflow that blocks pull requests that are introducing vulnerable dependencies into your codebase. See [Enforcing dependency review across an organization](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/enforcing-dependency-review-across-an-organization).
- If your repository is configured with code scanning, you can use rulesets to set code scanning merge protection, which prevents pull requests from being merged if there is a code scanning alert of a certain severity, or if a code scanning analysis is still in progress. See [Set code scanning merge protection](https://docs.github.com/en/code-security/code-scanning/managing-your-code-scanning-configuration/set-code-scanning-merge-protection).

>  ruleset 和其他的自动安全检查结合时，对于维护 repo 安全性特别有用，例如
>  - 可以用 ruleset 强制执行依赖项审查操作，这是一个阻止引入易受攻击的依赖项到代码库的 pr 的工作流
>  - 如果 repo 配置了 code scanning，我们可以用 ruleset 设置 code scanning 合并保护，它会避免合并 code scanning 报告了特定严重级别的 pr

### Using push rulesets
With push rulesets, you can block pushes to a private or internal repository and that repository's entire fork network based on file extensions, file path lengths, file and folder paths, and file sizes.
>  push ruleset 基于文件拓展名、文件路径长度、文件和目录路径、文件大小，来阻止向私有或内部 repo 的 push

Push rules do not require any branch targeting because they apply to every push to the repository.
>  push rule 不需要指定分支目标，它们应用于所有对 repo 的 push

Push rulesets allow you to:

- **Restrict file paths:** Prevent commits that include changes in specified file paths from being pushed.
    
    You can use `fnmatch` syntax for this. For example, a restriction targeting `test/demo/**/*` prevents any pushes to files or folders in the `test/demo/` directory. A restriction targeting `test/docs/pushrules.md` prevents pushes specifically to the `pushrules.md` file in the `test/docs/` directory. For more information, see [Creating rulesets for a repository](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository#using-fnmatch-syntax).
    
- **Restrict file path length:** Prevent commits that include file paths that exceed a specified character limit from being pushed.
- **Restrict file extensions:** Prevent commits that include files with specified file extensions from being pushed.
- **Restrict file size:** Prevent commits that exceed a specified file size limit from being pushed.

>  push rules 可以用于
>  - 限制文件路径: 阻止包含指定文件路径的更改被 push
>  - 限制文件路径长度: 避免包含了路径长度超过了自定字符数的文件被 push
>  - 限制文件拓展: 避免包含了特定拓展名的文件被 push
>  - 限制文件大小: 避免包含了超过特定大小的文件被 push

#### About push rulesets for forked repositories
Push rules apply to the entire fork network for a repository, ensuring every entry point to the repository is protected. For example, if you fork a repository that has push rulesets enabled, the same push rulesets will also apply to your forked repository.

For a forked repository, the only people who have bypass permissions for a push rule are the people who have bypass permissions in the root repository.

>  push rules 适用于 repo 的整个 fork 网络，确保 repo 的所有入口点都受保护

For more information, see [About rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets#push-rulesets).

### Using automated tools to review code styling
Use automated tools, such as linters, in your repository's pull requests to maintain consistent styling and make code more understandable. Using automated tools to catch smaller problems like typos or styling leaves more time for reviewers to focus on the substance of a pull request.
>  可以在 pr 中使用自动化工具，例如 linters 来保持代码一致的风格，以及发现 typos 等

For example, you can use GitHub Actions to set up code linters that can run on pull requests as part of your continuous integration (CI) workflow. For more information, see [About continuous integration with GitHub Actions](https://docs.github.com/en/actions/automating-builds-and-tests/about-continuous-integration).
>  例如，可以使用 GitHub Actions 来设置 code linters，使其作为 CI 工作流的一部分在 pr 上运行

