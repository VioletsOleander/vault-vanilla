# Understanding GitHub Actions
Learn the basics of GitHub Actions, including core concepts and essential terminology.

## Overview
GitHub Actions is a continuous integration and continuous delivery (CI/CD) platform that allows you to automate your build, test, and deployment pipeline. You can create workflows that build and test every pull request to your repository, or deploy merged pull requests to production.
>  GitHub Actions 是一个持续集成和持续交付 (CI/CD) 平台，允许我们自动化构建、测试和部署流水线
>  我们可以创建工作流，对 repo 的每个 pull request 执行构建和测试，或者将 repo 合并的 pull request 部署到生产环境

GitHub Actions goes beyond just DevOps and lets you run workflows when other events happen in your repository. For example, you can run a workflow to automatically add the appropriate labels whenever someone creates a new issue in your repository.
>  GitHub Actions 不局限于 DevOps，在我们的 repo 发生其他事件时，也可以运行工作流
>  例如，可以运行一个工作流，在有人创建 issue 时自动为其添加标签

GitHub provides Linux, Windows, and macOS virtual machines to run your workflows, or you can host your own self-hosted runners in your own data center or cloud infrastructure.
>  GitHub 提供了多平台的虚拟机以运行我们的工作流，也可以将工作流的运行托管在我们自己的数据中心或云平台

## The components of GitHub Actions
You can configure a GitHub Actions **workflow** to be triggered when an **event** occurs in your repository, such as a pull request being opened or an issue being created. Your workflow contains one or more **jobs** which can run in sequential order or in parallel. Each job will run inside its own virtual machine **runner**, or inside a container, and has one or more **steps** that either run a script that you define or run an **action**, which is a reusable extension that can simplify your workflow.
>  GitHub Actions workflow 在我们的仓库中发生 event 时被触发
>  event 可以是 pull request 被 open, issue 被创建等
>  workflow 包含一个或多个 jobs, jobs 可以串行也可以并行运行，每个 job 在自己的虚拟机 runner 中 (或者在容器中) 运行
>  每个 job 包含一个或多个 steps，每个 step 可以运行一个自定义脚本，或运行一个 action, action 是可复用的

![Diagram of an event triggering Runner 1 to run Job 1, which triggers Runner 2 to run Job 2. Each of the jobs is broken into multiple steps.](https://docs.github.com/assets/cb-25535/images/help/actions/overview-actions-simple.png)

### Workflows
A **workflow** is a configurable automated process that will run one or more jobs. Workflows are defined by a YAML file checked in to your repository and will run when triggered by an event in your repository, or they can be triggered manually, or at a defined schedule.
>  workflow 是一个可配置的自动化流程，workflow 运行一个或多个 job
>  workflow 由 repo 中检入的一个 YAML 文件定义，当 repo 中的 event 发生，workflow 会被触发
>  workflow 也可以定时或人为触发

Workflows are defined in the `.github/workflows` directory in a repository. A repository can have multiple workflows, each of which can perform a different set of tasks such as:

- Building and testing pull requests
- Deploying your application every time a release is created
- Adding a label whenever a new issue is opened

>  workflow 定义在 repo 中的 `.github/workflow` 中，一个 repo 可以有多个 workflow，每个 workflow 执行特定的任务，例如
>  - 构建和测试 pull request
>  - 在 release 被创建时，部署应用
>  - 新 issues 被 open 时，添加标签

You can reference a workflow within another workflow. For more information, see [Reusing workflows](https://docs.github.com/en/actions/using-workflows/reusing-workflows).

For more information, see [Writing workflows](https://docs.github.com/en/actions/using-workflows).

### Events
An **event** is a specific activity in a repository that triggers a **workflow** run. For example, an activity can originate from GitHub when someone creates a pull request, opens an issue, or pushes a commit to a repository. You can also trigger a workflow to run on a [schedule](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule), by [posting to a REST API](https://docs.github.com/en/rest/repos/repos#create-a-repository-dispatch-event), or manually.
>  event 是 repo 中触发 workflow 运行的特殊活动，例如 pull request 被创建, issue 被 open, commit 被 push

For a complete list of events that can be used to trigger workflows, see [Events that trigger workflows](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows).

### Jobs
A **job** is a set of **steps** in a workflow that is executed on the same **runner**. Each step is either a shell script that will be executed, or an **action** that will be run. Steps are executed in order and are dependent on each other. Since each step is executed on the same runner, you can share data from one step to another. For example, you can have a step that builds your application followed by a step that tests the application that was built.
>  job 是 workflow 中的一组执行在相同 runner 上的 steps，每个 step 要么是一个脚本，要么是一个 action
>  steps 按序执行，且相互依赖，因为 steps 在同一 runner 上执行，故可以在 steps 之间共享数据，例如，可以让一个 step 构建应用，下一个 step 测试应用

You can configure a job's dependencies with other jobs; by default, jobs have no dependencies and run in parallel. When a job takes a dependency on another job, it waits for the dependent job to complete before running.
>  jobs 之前的依赖可以配置，默认情况下 jobs 之间无依赖，并行运行
>  存在依赖时 jobs 会按序运行

For example, you might configure multiple build jobs for different architectures without any job dependencies and a packaging job that depends on those builds. The build jobs run in parallel, and once they complete successfully, the packaging job runs.
>  例如，可以配置针对多个架构的并行构建 jobs，以及依赖于这些 jobs 的打包 job

For more information, see [Choosing what your workflow does](https://docs.github.com/en/actions/using-jobs).

### Actions
An **action** is a custom application for the GitHub Actions platform that performs a complex but frequently repeated task. Use an action to help reduce the amount of repetitive code that you write in your **workflow** files. An action can pull your Git repository from GitHub, set up the correct toolchain for your build environment, or set up the authentication to your cloud provider.
>  action 是 GitHub Actions 平台行的自定义应用程序，一般执行复杂且经常重复的任务
>  action 可以从 GitHub 拉取我们的 repo，为我们的构建环境设定工具链，或为我们的云提供商设定验证

You can write your own actions, or you can find actions to use in your workflows in the GitHub Marketplace.

For more information on actions, see [Sharing automations](https://docs.github.com/en/actions/creating-actions).

### Runners
A **runner** is a server that runs your workflows when they're triggered. Each runner can run a single **job** at a time. GitHub provides Ubuntu Linux, Microsoft Windows, and macOS runners to run your **workflows**. Each workflow run executes in a fresh, newly-provisioned virtual machine.
>  runner 是运行 workflow 的服务器，每个 runner 一次运行一个 job
>  每个 workflow 都会在全新的虚拟机上执行

GitHub also offers larger runners, which are available in larger configurations. For more information, see [Using larger runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners).

If you need a different operating system or require a specific hardware configuration, you can host your own runners.

For more information about self-hosted runners, see [Hosting your own runners](https://docs.github.com/en/actions/hosting-your-own-runners).

## Next steps
GitHub Actions can help you automate nearly every aspect of your application development processes. Ready to get started? Here are some helpful resources for taking your next steps with GitHub Actions:

- To create a GitHub Actions workflow, see [Using workflow templates](https://docs.github.com/en/actions/learn-github-actions/using-starter-workflows).
- For continuous integration (CI) workflows, see [Building and testing](https://docs.github.com/en/actions/automating-builds-and-tests).
- For building and publishing packages, see [Publishing packages](https://docs.github.com/en/actions/publishing-packages).
- For deploying projects, see [Use cases and examples](https://docs.github.com/en/actions/deployment).
- For automating tasks and processes on GitHub, see [Managing projects](https://docs.github.com/en/actions/managing-issues-and-pull-requests).
- For examples that demonstrate more complex features of GitHub Actions, see [Use cases and examples](https://docs.github.com/en/actions/examples). These detailed examples explain how to test your code on a runner, access the GitHub CLI, and use advanced features such as concurrency and test matrices.
- To certify your proficiency in automating workflows and accelerating development with GitHub Actions, earn a GitHub Actions certificate with GitHub Certifications. For more information, see [About GitHub Certifications](https://docs.github.com/en/get-started/showcase-your-expertise-with-github-certifications/about-github-certifications).

# About continuous integration with GitHub Actions
You can create custom continuous integration (CI) workflows directly in your GitHub repository with GitHub Actions.

## About continuous integration
Continuous integration (CI) is a software practice that requires frequently committing code to a shared repository. Committing code more often detects errors sooner and reduces the amount of code a developer needs to debug when finding the source of an error. Frequent code updates also make it easier to merge changes from different members of a software development team. This is great for developers, who can spend more time writing code and less time debugging errors or resolving merge conflicts.
>  持续集成是一种软件实践，要求开发人员频繁地将代码提交到共享 repo 中
>  更频繁地提交代码可以更快检测到错误，减少开发人员查找错误来源时需要 debug 的代码量
>  频繁的代码更新也使得合并不同成员的贡献更加容易

When you commit code to your repository, you can continuously build and test the code to make sure that the commit doesn't introduce errors. Your tests can include code linters (which check style formatting), security checks, code coverage, functional tests, and other custom checks.
>  代码提交后，可以持续地构建并测试代码，确保 commit 不会引入错误
>  测试可以包括代码格式检查、安全检查、代码覆盖率、功能测试等

Building and testing your code requires a server. You can build and test updates locally before pushing code to a repository, or you can use a CI server that checks for new code commits in a repository.
>  可以在随送代码之前本地构建并测试，也可以在推送后在 CI 服务器测试

## About continuous integration using GitHub Actions
CI using GitHub Actions offers workflows that can build the code in your repository and run your tests. Workflows can run on GitHub-hosted virtual machines, or on machines that you host yourself. For more information, see [Using GitHub-hosted runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners) and [About self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners).

You can configure your CI workflow to run when a GitHub event occurs (for example, when new code is pushed to your repository), on a set schedule, or when an external event occurs using the repository dispatch webhook.

GitHub runs your CI tests and provides the results of each test in the pull request, so you can see whether the change in your branch introduces an error. When all CI tests in a workflow pass, the changes you pushed are ready to be reviewed by a team member or merged. When a test fails, one of your changes may have caused the failure.
>  GitHub 会运行 CI workflow，提供 pull request 在每个测试上的结果
>  如果 CI 测试通过，则我们推送的修改可以被合并或者被进一步审查

When you set up CI in your repository, GitHub analyzes the code in your repository and recommends CI workflows based on the language and framework in your repository. For example, if you use [Node.js](https://nodejs.org/en/), GitHub will suggest a workflow template that installs your Node.js packages and runs your tests. You can use the CI workflow template suggested by GitHub, customize the suggested workflow template, or create your own custom workflow file to run your CI tests.
>  GitHub 提供了 CI workflow 模板

In addition to helping you set up CI workflows for your project, you can use GitHub Actions to create workflows across the full software development life cycle. For example, you can use actions to deploy, package, or release your project. For more information, see [Writing workflows](https://docs.github.com/en/actions/learn-github-actions).
>  除了 CI，整个软件开发流程的 workflow 例如部署、打包、发布都可以通过 GitHub Actions 实现

For a definition of common terms, see [Understanding GitHub Actions](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions).

## Workflow templates
GitHub offers CI workflow templates for a variety of languages and frameworks.

Browse the complete list of CI workflow templates offered by GitHub in the [actions/starter-workflows](https://github.com/actions/starter-workflows/tree/main/ci) repository.

## Further reading

- [Building and testing](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing)
- [Managing billing for GitHub Actions](https://docs.github.com/en/billing/managing-billing-for-github-actions)
