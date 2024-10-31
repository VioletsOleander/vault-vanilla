Models, Spaces, and Datasets are hosted on the Hugging Face Hub as [Git repositories](https://git-scm.com/about), which means that version control and collaboration are core elements of the Hub. In a nutshell, a repository (also known as a **repo**) is a place where code and assets can be stored to back up your work, share it with the community, and work in a team.
> Hugging Face Hub 用 Git 仓库托管 Models、Spaces、Datasets

# Getting Started with Repositories
This beginner-friendly guide will help you get the basic skills you need to create and manage your repository on the Hub. Each section builds on the previous one, so feel free to choose where to start!

## Requirements
This document shows how to handle repositories through the web interface as well as through the terminal. There are no requirements if working with the UI. If you want to work with the terminal, please follow these installation instructions.

If you do not have `git` available as a CLI command yet, you will need to [install Git](https://git-scm.com/downloads) for your platform. You will also need to [install Git LFS](https://git-lfs.github.com/), which will be used to handle large files such as images and model weights.

To be able to push your code to the Hub, you’ll need to authenticate somehow. The easiest way to do this is by installing the [`huggingface_hub` CLI](https://huggingface.co/docs/huggingface_hub/index) and running the login command:

```
python -m pip install huggingface_hub
huggingface-cli login
```

## Creating a repository
Using the Hub’s web interface you can easily create repositories, add files (even large ones!), explore models, visualize diffs, and much more. There are three kinds of repositories on the Hub, and in this guide you’ll be creating a **model repository** for demonstration purposes. For information on creating and managing models, datasets, and Spaces, refer to their respective documentation.

1. To create a new repository, visit [huggingface.co/new](http://huggingface.co/new):

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/new_repo.png)

2. Specify the owner of the repository: this can be either you or any of the organizations you’re affiliated with.
3. Enter your model’s name. This will also be the name of the repository.
4. Specify whether you want your model to be public or private.
5. Specify the license. You can leave the _License_ field blank for now. To learn about licenses, visit the [**Licenses**](https://huggingface.co/docs/hub/repositories-licenses) documentation.

After creating your model repository, you should see a page like this:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/empty_repo.png)

Note that the Hub prompts you to create a _Model Card_, which you can learn about in the [**Model Cards documentation**](https://huggingface.co/docs/hub/model-cards). Including a Model Card in your model repo is best practice, but since we’re only making a test repo at the moment we can skip this.

## Adding files to a repository (Web UI)
To add files to your repository via the web UI, start by selecting the **Files** tab, navigating to the desired directory, and then clicking **Add file**. You’ll be given the option to create a new file or upload a file directly from your computer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/repositories-add_file.png)

### Creating a new file
Choosing to create a new file will take you to the following editor screen, where you can choose a name for your file, add content, and save your file with a message that summarizes your changes. Instead of directly committing the new file to your repo’s `main` branch, you can select `Open as a pull request` to create a [Pull Request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions).

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/repositories-create_file.png)

### Uploading a file
If you choose _Upload file_ you’ll be able to choose a local file to upload, along with a message summarizing your changes to the repo.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/repositories-upload_file.png)

As with creating new files, you can select `Open as a pull request` to create a [Pull Request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) instead of adding your changes directly to the `main` branch of your repo.

## Adding files to a repository (terminal)
### Cloning repositories
Downloading repositories to your local machine is called _cloning_. You can use the following commands to load your repo and navigate to it:

```
git clone https://huggingface.co/<your-username>/<your-model-name>
cd <your-model-name>
```

You can clone over SSH with the following command:

```
git clone git@hf.co:<your-username>/<your-model-name>
cd <your-model-name>
```

You’ll need to add your SSH public key to [your user settings](https://huggingface.co/settings/keys) to push changes or access private repositories.

### Set up
Now’s the time, you can add any files you want to the repository! 🔥

Do you have files larger than 10MB? Those files should be tracked with `git-lfs`, which you can initialize with:
> 大于 10MB 的文件需要用 `git-lfs` 追踪

```
git lfs install
```

Note that if your files are larger than **5GB** you’ll also need to run:

```
huggingface-cli lfs-enable-largefiles .
```

When you use Hugging Face to create a repository, Hugging Face automatically provides a list of common file extensions for common Machine Learning large files in the `.gitattributes` file, which `git-lfs` uses to efficiently track changes to your large files. However, you might need to add new extensions if your file types are not already handled. You can do so with `git lfs track "*.your_extension"`.
> `git-lfs` 根据 `.gitattributes` 中指定的后缀追踪相应的文件
> Huggingface 会自动提供常见的大文件后缀在 `.gitattributes` 中
> 可以手动 `git lfs track "*.your_extension"` 来添加让 `git-lfs` 追踪的后缀

### Pushing files
You can use Git to save new files and any changes to already existing files as a bundle of changes called a _commit_, which can be thought of as a “revision” to your project. To create a commit, you have to `add` the files to let Git know that we’re planning on saving the changes and then `commit` those changes. In order to sync the new commit with the Hugging Face Hub, you then `push` the commit to the Hub.

```
# Create any files you like! Then...
git add .
git commit -m "First model version"  # You can choose any descriptive message
git push
```

And you’re done! You can check your repository on Hugging Face with all the recently added files. For example, in the screenshot below the user added a number of files. Note that some files in this example have a size of `1.04 GB`, so the repo uses Git LFS to track it.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/repo_with_files.png)

If you cloned the repository with HTTP, you might be asked to fill your username and password on every push operation. The simplest way to avoid repetition is to [switch to SSH](https://huggingface.co/docs/hub/repositories-getting-started#cloning-repositories), instead of HTTP. Alternatively, if you have to use HTTP, you might find it helpful to setup a [git credential helper](https://git-scm.com/docs/gitcredentials#_avoiding_repetition) to autofill your username and password.

## Viewing a repo’s history
Every time you go through the `add` - `commit` - `push` cycle, the repo will keep track of every change you’ve made to your files. The UI allows you to explore the model files and commits and to see the difference (also known as _diff_) introduced by each commit. To see the history, you can click on the **History: X commits** link.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/repo_history.png)

You can click on an individual commit to see what changes that commit introduced:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/explore_history.gif)

# Repository Settings

## Private repositories
You can choose a repository’s visibility when you create it, and any repository that you own can have its visibility toggled between _public_ and _private_ in the **Settings** tab. Unless your repository is owned by an [organization](https://huggingface.co/docs/hub/organizations), you are the only user that can make changes to your repo or upload any code. Setting your visibility to _private_ will:

- Ensure your repo does not show up in other users’ search results.
- Other users who visit the URL of your private repo will receive a `404 - Repo not found` error.
- Other users will not be able to clone your repo.

## Renaming or transferring a repo
If you own a repository, you will be able to visit the **Settings** tab to manage the name and ownership. Note that there are certain limitations in terms of use cases.

Moving can be used in these use cases ✅

- Renaming a repository within same user.
- Renaming a repository within same organization. The user must be part of the organization and have “write” or “admin” rights in the organization.
- Transferring repository from user to an organization. The user must be part of the organization and have “write” or “admin” rights in the organization.
- Transferring a repository from an organization to yourself. You must be part of the organization, and have “admin” rights in the organization.
- Transferring a repository from a source organization to another target organization. The user must have “admin” rights in the source organization **and** either “write” or “admin” rights in the target organization.

Moving does not work for ❌

- Transferring a repository from an organization to another user who is not yourself.
- Transferring a repository from a source organization to another target organization if the user does not have both “admin” rights in the source organization **and** either “write” or “admin” rights in the target organization.
- Transferring a repository from user A to user B.

If these are use cases you need help with, please send us an email at **website at huggingface.co**.

## Disabling Discussions / Pull Requests
You can disable all discussions and Pull Requests. Once disabled, all community and contribution features won’t be available anymore. This action can be reverted without losing any previous discussions or Pull Requests.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-settings-disable.png)

# Pull requests and Discussions
Hub Pull requests and Discussions allow users to do community contributions to repositories. Pull requests and discussions work the same for all the repo types.

At a high level, the aim is to build a simpler version of other git hosts’ (like GitHub’s) PRs and Issues:

- no forks are involved: contributors push to a special `ref` branch directly on the source repo.
- there’s no hard distinction between discussions and PRs: they are essentially the same so they are displayed in the same lists.
- they are streamlined for ML (i.e. models/datasets/spaces repos), not arbitrary repos.

_Note, Pull Requests and discussions can be enabled or disabled from the [repository settings](https://huggingface.co/docs/hub/repositories-settings#disabling-discussions--pull-requests) _

> Huggingface 的 PR 相较于 Github 更加简化：
> PR 不需要 fork，贡献者直接 push 到源 repo 的 `ref` 分支
> PR 和 discussion 没有明显差别，二者本质上等价
> PR 针对 ML 的流程

## List
By going to the community tab in any repository, you can see all Discussions and Pull requests. You can also filter to only see the ones that are open.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-list.png)

## View
The Discussion page allows you to see the comments from different users. If it’s a Pull Request, you can see all the changes by going to the Files changed tab.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-view.png)

## Editing a Discussion / Pull request title
If you opened a PR or discussion, are the author of the repository, or have write access to it, you can edit the discussion title by clicking on the pencil button.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-edit-title.PNG)

## Pin a Discussion / Pull Request
If you have write access to a repository, you can pin discussions and Pull Requests. Pinned discussions appear at the top of all the discussions.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-pin.png)

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-pinned.png)

## Lock a Discussion / Pull Request
If you have write access to a repository, you can lock discussions or Pull Requests. Once a discussion is locked, previous comments are still visible and users won’t be able to add new comments.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-lock.png)
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-locked.png)

## Comment edition and moderation
If you wrote a comment or have write access to the repository, you can edit the content of the comment from the contextual menu in the top-right corner of the comment box.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-comment-menu.png)

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-comment-menu-edit.png)

Once the comment has been edited, a new link will appear above the comment. This link shows the edit history.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-comment-edit-link.png)

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-comment-edit-history.png)

You can also hide a comment. Hiding a comment is irreversible, and nobody will be able to see its content nor edit it anymore.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/discussions-comment-hidden.png)

Read also [moderation](https://huggingface.co/docs/hub/moderation) to see how to report an abusive comment.

## [](https://huggingface.co/docs/hub/repositories-pull-requests-discussions#can-i-use-markdown-and-latex-in-my-comments-and-discussions)Can I use Markdown and LaTeX in my comments and discussions?

Yes! You can use Markdown to add formatting to your comments. Additionally, you can utilize LaTeX for mathematical typesetting, your formulas will be rendered with [KaTeX](https://katex.org/) before being parsed in Markdown.

For LaTeX equations, you have to use the following delimiters:

- `$$ ... $$` for display mode
- `\\(...\\)` for inline mode (no space between the slashes and the parenthesis).

## [](https://huggingface.co/docs/hub/repositories-pull-requests-discussions#how-do-i-manage-pull-requests-locally)How do I manage Pull requests locally?

Let’s assume your PR number is 42.

Copied

git fetch origin refs/pr/42:pr/42
git checkout pr/42
# Do your changes
git add .
git commit -m "Add your change"
git push origin pr/42:refs/pr/42

### [](https://huggingface.co/docs/hub/repositories-pull-requests-discussions#draft-mode)Draft mode

Draft mode is the default status when opening a new Pull request from scratch in “Advanced mode”. With this status, other contributors know that your Pull request is under work and it cannot be merged. When your branch is ready, just hit the “Publish” button to change the status of the Pull request to “Open”. Note that once published you cannot go back to draft mode.

## [](https://huggingface.co/docs/hub/repositories-pull-requests-discussions#pull-requests-advanced-usage)Pull requests advanced usage

### [](https://huggingface.co/docs/hub/repositories-pull-requests-discussions#where-in-the-git-repo-are-changes-stored)Where in the git repo are changes stored?

Our Pull requests do not use forks and branches, but instead custom “branches” called `refs` that are stored directly on the source repo.

[Git References](https://git-scm.com/book/en/v2/Git-Internals-Git-References) are the internal machinery of git which already stores tags and branches.

The advantage of using custom refs (like `refs/pr/42` for instance) instead of branches is that they’re not fetched (by default) by people (including the repo “owner”) cloning the repo, but they can still be fetched on demand.

### [](https://huggingface.co/docs/hub/repositories-pull-requests-discussions#fetching-all-pull-requests-for-git-magicians-)Fetching all Pull requests: for git magicians 🧙‍♀️

You can tweak your local **refspec** to fetch all Pull requests:

1. Fetch

Copied

git fetch origin refs/pr/*:refs/remotes/origin/pr/*

2. create a local branch tracking the ref

Copied

git checkout pr/{PR_NUMBER}
# for example: git checkout pr/42

3. IF you make local changes, to push to the PR ref:

Copied

git push origin pr/{PR_NUMBER}:refs/pr/{PR_NUMBER}
# for example: git push origin pr/42:refs/pr/42

[<>Update on GitHub](https://github.com/huggingface/hub-docs/blob/main/docs/hub/repositories-pull-requests-discussions.md)