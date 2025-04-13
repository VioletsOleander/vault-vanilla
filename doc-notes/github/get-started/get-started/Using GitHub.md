## GitHub flow
### [Introduction](https://docs.github.com/en/get-started/using-github/github-flow#introduction)
GitHub flow is a lightweight, branch-based workflow. The GitHub flow is useful for everyone, not just developers. For example, here at GitHub, we use GitHub flow for our [site policy](https://github.com/github/site-policy), [documentation](https://github.com/github/docs), and [roadmap](https://github.com/github/roadmap).
### [Prerequisites](https://docs.github.com/en/get-started/using-github/github-flow#prerequisites)
To follow GitHub flow, you will need a GitHub account and a repository. For information on how to create an account, see "[Creating an account on GitHub](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github)." For information on how to create a repository, see "[Quickstart for repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories)." For information on how to find an existing repository to contribute to, see "[Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)."
### [Following GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow#following-github-flow)
**Tip:** You can complete all steps of GitHub flow through the GitHub web interface, command line and [GitHub CLI](https://cli.github.com/), or [GitHub Desktop](https://docs.github.com/en/desktop). For more information about the tools you can use to connect to GitHub, see "[Connecting to GitHub](https://docs.github.com/en/get-started/using-github/connecting-to-github)."
#### [Create a branch](https://docs.github.com/en/get-started/using-github/github-flow#create-a-branch)
Create a branch in your repository. A short, descriptive branch name enables your collaborators to see ongoing work at a glance. For example, `increase-test-timeout` or `add-code-of-conduct`. For more information, see "[Creating and deleting branches within your repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository)."

By creating a branch, you create a space to work without affecting the default branch. Additionally, you give collaborators a chance to review your work.
#### [Make changes](https://docs.github.com/en/get-started/using-github/github-flow#make-changes)
On your branch, make any desired changes to the repository. For more information, see "[Creating new files](https://docs.github.com/en/repositories/working-with-files/managing-files/creating-new-files)", "[Editing files](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files)", "[Renaming a file](https://docs.github.com/en/repositories/working-with-files/managing-files/renaming-a-file)", "[Moving a file to a new location](https://docs.github.com/en/repositories/working-with-files/managing-files/moving-a-file-to-a-new-location)", or "[Deleting files in a repository](https://docs.github.com/en/repositories/working-with-files/managing-files/deleting-files-in-a-repository)."

Your branch is a safe place to make changes. If you make a mistake, you can revert your changes or push additional changes to fix the mistake. Your changes will not end up on the default branch until you merge your branch.

Commit and push your changes to your branch. Give each commit a descriptive message to help you and future contributors understand what changes the commit contains. For example, `fix typo` or `increase rate limit`.

Ideally, each commit contains an isolated, complete change. This makes it easy to revert your changes if you decide to take a different approach. For example, if you want to rename a variable and add some tests, put the variable rename in one commit and the tests in another commit. Later, if you want to keep the tests but revert the variable rename, you can revert the specific commit that contained the variable rename. If you put the variable rename and tests in the same commit or spread the variable rename across multiple commits, you would spend more effort reverting your changes.
> 理想情况下，每个提交都是独立、完整的改变，这使得我们之后想要 revert 到应用某个改变之前仅需要 revert 某个特定的提交即可

By committing and pushing your changes, you back up your work to remote storage. This means that you can access your work from any device. It also means that your collaborators can see your work, answer questions, and make suggestions or contributions.

Continue to make, commit, and push changes to your branch until you are ready to ask for feedback.

**Tip:** Make a separate branch for each set of unrelated changes. This makes it easier for reviewers to give feedback. It also makes it easier for you and future collaborators to understand the changes and to revert or build on them. Additionally, if there is a delay in one set of changes, your other changes aren't also delayed.
> 提示：对于不相关的一组修改，最好在不同的分支中实现
#### [Create a pull request](https://docs.github.com/en/get-started/using-github/github-flow#create-a-pull-request)
Create a pull request to ask collaborators for feedback on your changes. Pull request review is so valuable that some repositories require an approving review before pull requests can be merged. If you want early feedback or advice before you complete your changes, you can mark your pull request as a draft. For more information, see "[Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)."

When you create a pull request, include a summary of the changes and what problem they solve. You can include images, links, and tables to help convey this information. If your pull request addresses an issue, link the issue so that issue stakeholders are aware of the pull request and vice versa. If you link with a keyword, the issue will close automatically when the pull request merges. For more information, see "[Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)" and "[Linking a pull request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)."

In addition to filling out the body of the pull request, you can add comments to specific lines of the pull request to explicitly point something out to the reviewers.

Your repository may be configured to automatically request a review from specific teams or users when a pull request is created. You can also manually @mention or request a review from specific people or teams.

If your repository has checks configured to run on pull requests, you will see any checks that failed on your pull request. This helps you catch errors before merging your branch. For more information, see "[About status checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks)."
#### [Address review comments](https://docs.github.com/en/get-started/using-github/github-flow#address-review-comments)
Reviewers should leave questions, comments, and suggestions. Reviewers can comment on the whole pull request or add comments to specific lines or files. You and reviewers can insert images or code suggestions to clarify comments. For more information, see "[Reviewing changes in pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests)."

You can continue to commit and push changes in response to the reviews. Your pull request will update automatically.
#### [Merge your pull request](https://docs.github.com/en/get-started/using-github/github-flow#merge-your-pull-request)
Once your pull request is approved, merge your pull request. This will automatically merge your branch so that your changes appear on the default branch. GitHub retains the history of comments and commits in the pull request to help future contributors understand your changes. For more information, see "[Merging a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/merging-a-pull-request)."

GitHub will tell you if your pull request has conflicts that must be resolved before merging. For more information, see "[Addressing merge conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts)."

Branch protection settings may block merging if your pull request does not meet certain requirements. For example, you need a certain number of approving reviews or an approving review from a specific team. For more information, see "[About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)."
#### [Delete your branch](https://docs.github.com/en/get-started/using-github/github-flow#delete-your-branch)
After you merge your pull request, delete your branch. This indicates that the work on the branch is complete and prevents you or others from accidentally using old branches. For more information, see "[Deleting and restoring branches in a pull request](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/deleting-and-restoring-branches-in-a-pull-request)."

Don't worry about losing information. Your pull request and commit history will not be deleted. You can always restore your deleted branch or revert your pull request if needed.
## Connecting to GitHub
### [Introduction](https://docs.github.com/en/get-started/using-github/connecting-to-github#introduction)
GitHub is a web-based app that lets you host files in repositories, collaborate on work, and track changes to files over time. Version tracking on GitHub is powered by the open source software Git. Whenever you update a repository on GitHub, Git tracks the changes you make.

There are many ways to work with GitHub, and you can choose a method that suits your level of experience, personal preferences, and the repositories you work with. For example, you can choose whether you want to work in the browser or from your desktop, how you want to use Git, and what capabilities you need from your editor and other software. You may choose to work with different repositories in different ways.

If you're new to GitHub, a good way to start contributing is to make changes in the browser on GitHub.com. As you become more familiar with GitHub and start contributing larger changes, you may want to start working with other tools. This article explains how to progress through these stages and helps you choose the best tool for your requirements at each stage. To quickly compare all the tools available for working with GitHub, see "[Comparison of tools for connecting to GitHub](https://docs.github.com/en/get-started/using-github/connecting-to-github#comparison-of-tools-for-connecting-to-github)."
### [Getting started](https://docs.github.com/en/get-started/using-github/connecting-to-github#getting-started)
In the user interface on GitHub.com, you can perform the whole "GitHub flow" for contributing to a repository, including creating a branch or fork, editing and previewing files, committing your changes, and creating a pull request. You can also upload files from your computer or download them from the repository. For more information, see "[GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow)."

Working directly on GitHub.com is often the quickest way to contribute to a repository, for the following reasons.

- You're working directly with the repository hosted on GitHub, so you don't have to download a copy of the repository to your computer and keep this copy in sync.
- If you're already signed in to GitHub, you have access to any repository where you have the necessary permissions, so you don't need to set up any additional authentication on your computer.
- You can commit changes in the user interface, so you don't need to use the command line or memorize any Git commands.

For a tutorial to help you get started with making changes in the browser, see "[Hello World](https://docs.github.com/en/get-started/start-your-journey/hello-world)."
### [Making more complex changes in the browser](https://docs.github.com/en/get-started/using-github/connecting-to-github#making-more-complex-changes-in-the-browser)
Working directly on GitHub.com is best for small, simple changes, often targeting a single file in a repository. If you want to work in the browser but need to make more complex changes, such as moving content between files, you can choose from the following tools to open a repository in a dedicated editor.

- If you want an editor where you can quickly open or create files, you can press the `.` key in any repository to open the github.dev editor. This is a lightweight web-based editor that includes many of the features of Visual Studio Code, such as a search bar and buttons for Git commands. For more information, see "[The github.dev web-based editor](https://docs.github.com/en/codespaces/the-githubdev-web-based-editor)."
- If you want to stay in the browser but need to do things like run commands, create a test build of your project, or install dependencies, you can open a repository in a codespace. A codespace is a remote development environment with storage and compute power. It includes an editor and integrated terminal, and comes preinstalled with common tools you may need to work with a project, including Git. For more information, see "[GitHub Codespaces overview](https://docs.github.com/en/codespaces/overview)."

Alternatively, you can connect to GitHub from your desktop, and work with a local copy of the repository.
### [Working from the desktop](https://docs.github.com/en/get-started/using-github/connecting-to-github#working-from-the-desktop)
To work with a repository from your desktop, you'll need to download (or "clone") a copy of the repository to your computer, then push any changes you make to GitHub. Working from your desktop can have several advantages over working in the browser.

- You can work with all your local files and tools.
- You have access to compute power. For example, you might need to run a script to create a local preview of a site, so you can test the changes you're making.
- You don't need an Internet connection to work on a project.

If you haven't worked with a GitHub repository from your desktop before, you'll need to authenticate to GitHub from your computer, so you can access the repositories you need. You may also need to set up your working environment with the tools you need to contribute, such as Git, an editor, and dependencies for a project. For these reasons, it can take longer to get started if you want to work from your desktop, compared to working in the browser.

There are several tools you can use to connect to GitHub from your desktop. These tools allow you to authenticate to GitHub, clone a repository, track your changes, and push the changes to GitHub.

- If you want a lot of control and flexibility, you can use the command line. You'll need to install Git and be familiar with some basic Git commands. You can also install GitHub CLI, a command-line interface that lets you perform many actions on GitHub, such as creating a pull request or forking a repository. For more information, see "[Set up Git](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git)" and "[About GitHub CLI](https://docs.github.com/en/github-cli/github-cli/about-github-cli)."
- If you'd prefer to use a visual interface, you can use a visual Git client such as GitHub Desktop. With GitHub Desktop, you can visualize the changes you're making and access most Git commands through a visual interface, so you don't need to memorize any commands. For more information, see "[About GitHub Desktop](https://docs.github.com/en/desktop/overview/about-github-desktop)."
- If you want to work in one place, you can often do most things from your editor. An editor such as VS Code includes an integrated terminal and buttons for common Git commands, so you can edit files and push your changes to GitHub all from one place. You can also install an extension to work directly with pull requests and issues on GitHub. To get started, see [Download VS Code](https://code.visualstudio.com/download).
### [Comparison of tools for connecting to GitHub](https://docs.github.com/en/get-started/using-github/connecting-to-github#comparison-of-tools-for-connecting-to-github)
The following table provides a comparison between the tools you can use to work with repositories on GitHub, both in your browser and from your desktop.

You can perform the whole GitHub flow with any of the tools described here. Each tool includes access not only to Git commands for tracking the changes you've made, but also to GitHub-specific features, so you can create a pull request with your proposed changes from wherever you're working.

For more information about how to authenticate to GitHub with each of these tools, see "[About authentication to GitHub](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github)."

| Tool                                | Use case                                                                                                                                                                                                                                                                                                                                                                                      | Browser or desktop |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **On GitHub.com**                   | You want a visual interface and need to make quick, simple changes, typically involving a single commit. For an introduction, see "[Hello World](https://docs.github.com/en/get-started/start-your-journey/hello-world)."                                                                                                                                                                     | Browser            |
| **github.dev**                      | You want to make more complex changes to a repository than is possible on GitHub.com, but don't need to work with a terminal or tools you have installed on your computer. For more information, see "[The github.dev web-based editor](https://docs.github.com/en/codespaces/the-githubdev-web-based-editor#opening-the-githubdev-editor)."                                                  | Browser            |
| **GitHub Codespaces**               | You need the resources of a computer to do things like run scripts, create a test build of your project, or install dependencies, and you want to get started quickly by working in a cloud-based environment. For more information, see "[GitHub Codespaces overview](https://docs.github.com/en/codespaces/overview)."                                                                      | Browser or desktop |
| **GitHub Desktop**                  | You want to work with files locally, and would prefer a visual interface to use Git, visualize changes, and interact with GitHub. For more information, see "[About GitHub Desktop](https://docs.github.com/en/desktop/overview/about-github-desktop)."                                                                                                                                       | Desktop            |
| **IDE or text editor**              | You're working with more complex files and projects and want everything in one place.                                                                                                                                                                                                                                                                                                         | Desktop            |
| **Command-line Git and GitHub CLI** | You're used to working from the command line and want to avoid switching context, or you need to access a complex Git command that isn't integrated into visual interfaces. For more information, see "[Set up Git](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git)" and "[About GitHub CLI](https://docs.github.com/en/github-cli/github-cli/about-github-cli)." | Desktop            |
| **GitHub API**                      | You want to automate common tasks such as backing up your data, or create integrations that extend GitHub. For more information, see "[Comparing GitHub's REST API and GraphQL API](https://docs.github.com/en/rest/about-the-rest-api/comparing-githubs-rest-api-and-graphql-api)."                                                                                                          | Browser or desktop |

## Communicating on GitHub
### [Introduction](https://docs.github.com/en/get-started/using-github/communicating-on-github#introduction)
GitHub provides built-in collaborative communication tools allowing you to interact closely with your community. This quickstart guide will show you how to pick the right tool for your needs.

You can create and participate in issues, pull requests, and GitHub Discussions, depending on the type of conversation you'd like to have.
#### [GitHub Issues](https://docs.github.com/en/get-started/using-github/communicating-on-github#github-issues)

- Are useful for discussing specific details of a project such as bug reports, planned improvements and feedback
- Are specific to a repository, and usually have a clear owner
- Are often referred to as GitHub's bug-tracking system

#### [Pull requests](https://docs.github.com/en/get-started/using-github/communicating-on-github#pull-requests)

- Allow you to propose specific changes
- Allow you to comment directly on proposed changes suggested by others
- Are specific to a repository

#### [GitHub Discussions](https://docs.github.com/en/get-started/using-github/communicating-on-github#github-discussions)

- Are like a forum, and are best used for open-form ideas and discussions where collaboration is important
- May span many repositories
- Provide a collaborative experience outside the codebase, allowing the brainstorming of ideas, and the creation of a community knowledge base
- Often don’t have a clear owner
- Often do not result in an actionable task

### [Which discussion tool should I use?](https://docs.github.com/en/get-started/using-github/communicating-on-github#which-discussion-tool-should-i-use)
#### [Scenarios for issues](https://docs.github.com/en/get-started/using-github/communicating-on-github#scenarios-for-issues)

- I want to keep track of tasks, enhancements and bugs.
- I want to file a bug report.
- I want to share feedback about a specific feature.
- I want to ask a question about files in the repository.

##### [Issue example](https://docs.github.com/en/get-started/using-github/communicating-on-github#issue-example)

This example illustrates how a GitHub user created an issue in our documentation open source repository to make us aware of a bug, and discuss a fix.

![Screenshot of an issue, with the title "Blue link text in notices is unreadable due to blue background."](https://docs.github.com/assets/cb-435865/images/help/issues/issue-example.png)

- A user noticed that the blue color of the banner at the top of the page in the Chinese version of the GitHub Docs makes the text in the banner unreadable.
- The user created an issue in the repository, stating the problem and suggesting a fix (which is, use a different background color for the banner).
- A discussion ensues, and eventually, a consensus will be reached about the fix to apply.
- A contributor can then create a pull request with the fix.

#### [Scenarios for pull requests](https://docs.github.com/en/get-started/using-github/communicating-on-github#scenarios-for-pull-requests)

- I want to fix a typo in a repository.
- I want to make changes to a repository.
- I want to make changes to fix an issue.
- I want to comment on changes suggested by others.

##### [Pull request example](https://docs.github.com/en/get-started/using-github/communicating-on-github#pull-request-example)

This example illustrates how a GitHub user created a pull request in our documentation open source repository to fix a typo.

In the **Conversation** tab of the pull request, the author explains why they created the pull request.

![Screenshot of the "Conversation" tab of a pull request.](https://docs.github.com/assets/cb-56735/images/help/pull_requests/pr-conversation-example.png)

The **Files changed** tab of the pull request shows the implemented fix.

![Screenshot of the "Files changed" tab of a pull request.](https://docs.github.com/assets/cb-43403/images/help/pull_requests/pr-files-changed-example.png)

- This contributor notices a typo in the repository.
- The user creates a pull request with the fix.
- A repository maintainer reviews the pull request, comments on it, and merges it.

#### [Scenarios for GitHub Discussions](https://docs.github.com/en/get-started/using-github/communicating-on-github#scenarios-for-github-discussions)

- I have a question that's not necessarily related to specific files in the repository.
- I want to share news with my collaborators, or my team.
- I want to start or participate in an open-ended conversation.
- I want to make an announcement to my community.

##### [GitHub Discussions example](https://docs.github.com/en/get-started/using-github/communicating-on-github#github-discussions-example)

This example shows the GitHub Discussions welcome post for the GitHub Docs open source repository, and illustrates how the team wants to collaborate with their community.

![Screenshot of an example of a discussion, with the title "Welcome to GitHub Docs Discussions."](https://docs.github.com/assets/cb-194708/images/help/discussions/github-discussions-example.png)

This community maintainer started a discussion to welcome the community, and to ask members to introduce themselves. This post fosters an inviting atmosphere for visitors and contributors. The post also clarifies that the team's happy to help with contributions to the repository.