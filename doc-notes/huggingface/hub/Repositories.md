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

## Can I use Markdown and LaTeX in my comments and discussions?
Yes! You can use Markdown to add formatting to your comments. Additionally, you can utilize LaTeX for mathematical typesetting, your formulas will be rendered with [KaTeX](https://katex.org/) before being parsed in Markdown.

For LaTeX equations, you have to use the following delimiters:

- `$$ ... $$` for display mode
- `\\(...\\)` for inline mode (no space between the slashes and the parenthesis).

## How do I manage Pull requests locally?
Let’s assume your PR number is 42.

```
git fetch origin refs/pr/42:pr/42
git checkout pr/42
# Do your changes
git add .
git commit -m "Add your change"
git push origin pr/42:refs/pr/42
```

### Draft mode
Draft mode is the default status when opening a new Pull request from scratch in “Advanced mode”. With this status, other contributors know that your Pull request is under work and it cannot be merged. When your branch is ready, just hit the “Publish” button to change the status of the Pull request to “Open”. Note that once published you cannot go back to draft mode.
> Draft mode 的 PR 不能被 merge，将 Draft mode 的 PR publish 为 Open mode 之后就可以被 merge，PR 变为 Open mode 之后不能再回到 Draft mode

## Pull requests advanced usage
### Where in the git repo are changes stored?
Our Pull requests do not use forks and branches, but instead custom “branches” called `refs` that are stored directly on the source repo.

[Git References](https://git-scm.com/book/en/v2/Git-Internals-Git-References) are the internal machinery of git which already stores tags and branches.

The advantage of using custom refs (like `refs/pr/42` for instance) instead of branches is that they’re not fetched (by default) by people (including the repo “owner”) cloning the repo, but they can still be fetched on demand.
> Huggingface 的 PR 不使用 fork 和 branch，PR 统一由源 repo 中的 `refs` 管理
> `refs` 相较于传统分支的优势在于 `refs`  默认不会被 fetch，可以按需显式指定 fetch

### Fetching all Pull requests: for git magicians 🧙‍♀️
You can tweak your local **refspec** to fetch all Pull requests:

1. Fetch

```
git fetch origin refs/pr/*:refs/remotes/origin/pr/*
```

2. create a local branch tracking the ref

```
git checkout pr/{PR_NUMBER}
# for example: git checkout pr/42
```

3. IF you make local changes, to push to the PR ref:

```
git push origin pr/{PR_NUMBER}:refs/pr/{PR_NUMBER}
# for example: git push origin pr/42:refs/pr/42
```

# Notifications
Notifications allow you to know when new activities (**Pull Requests or discussions**) happen on models, datasets, and Spaces belonging to users or organizations you are watching.

By default, you’ll receive a notification if:

- Someone mentions you in a discussion/PR.
- A new comment is posted in a discussion/PR you participated in.
- A new discussion/PR or comment is posted in one of the repositories of an organization or user you are watching.

![Notifications page](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/notifications-page.png)

You’ll get new notifications by email and [directly on the website](https://huggingface.co/notifications), you can change this in your [notifications settings](https://huggingface.co/docs/hub/notifications#notifications-settings).

## Filtering and managing notifications
On the [notifications page](https://huggingface.co/notifications), you have several options for filtering and managing your notifications more effectively:

- Filter by Repository: Choose to display notifications from a specific repository only.
- Filter by Read Status: Display only unread notifications or all notifications.
- Filter by Participation: Show notifications you have participated in or those which you have been directly mentioned.

Additionally, you can take the following actions to manage your notifications:

- Mark as Read/Unread: Change the status of notifications to mark them as read or unread.
- Mark as Done: Once marked as done, notifications will no longer appear in the notification center (they are deleted).

By default, changes made to notifications will only apply to the selected notifications on the screen. However, you can also apply changes to all matching notifications (like in Gmail for instance) for greater convenience.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/notifications-select-all.png)

## Watching users and organizations
By default, you’ll be watching all the organizations you are a member of and will be notified of any new activity on those.

You can also choose to get notified on arbitrary users or organizations. To do so, use the “Watch repos” button on their HF profiles. Note that you can also quickly watch/unwatch users and organizations directly from your [notifications settings](https://huggingface.co/docs/hub/notifications#notifications-settings).

_Unlike GitHub or similar services, you cannot watch a specific repository. You must watch users/organizations to get notified about any new activity on any of their repositories. The goal is to simplify this functionality for users as much as possible and to make sure you don’t miss anything you might be interested in._
> Huggingface 不支持 watch 特定 repo，仅支持 watch 特定 user 或 organization

## Notifications settings
In your [notifications settings](https://huggingface.co/settings/notifications) page, you can choose specific channels to get notified on depending on the type of activity, for example, receiving an email for direct mentions but only a web notification for new activity on watched users and organizations. By default, you’ll get an email and a web notification for any new activity but feel free to adjust your settings depending on your needs.

_Note that clicking the unsubscribe link in an email will unsubscribe you for the type of activity, eg direct mentions._

![Notifications settings page](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/notifications-settings.png)

You can quickly add any user/organization to your watch list by searching them by name using the dedicated search bar. Unsubscribe from a specific user/organization simply by unticking the corresponding checkbox.

## Mute notifications for a specific repository
It’s possible to mute notifications for a particular repository by using the “Mute notifications” action in the repository’s contextual menu. This will prevent you from receiving any new notifications for that particular repository. You can unmute the repository at any time by clicking the “Unmute notifications” action in the same repository menu.

![mute notification menu](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/notifications-mute-menu.png)

_Note, if a repository is muted, you won’t receive any new notification unless you’re directly mentioned or participating to a discussion._

The list of muted repositories is available from the notifications settings page:

![Notifications settings page muted repositories](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/notifications-settings-muted.png)

# Collections
Use Collections to group repositories from the Hub (Models, Datasets, Spaces and Papers) on a dedicated page.

![Collection page](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-intro.webp)

Collections have many use cases:

- Highlight specific repositories on your personal or organizational profile.
- Separate key repositories from others for your profile visitors.
- Showcase and share a complete project with its paper(s), dataset(s), model(s) and Space(s).
- Bookmark things you find on the Hub in categories.
- Have a dedicated page of curated things to share with others.

This is just a list of possible uses, but remember that collections are just a way of grouping things, so use them in the way that best fits your use case.

## Creating a new collection
There are several ways to create a collection:

- For personal collections: Use the **+ New** button on your logged-in homepage (1).
- For organization collections: Use the **+ New** button available on organizations page (2).

![New collection](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-new.webp)

It’s also possible to create a collection on the fly when adding the first item from a repository page, select **+ Create new collection** from the dropdown menu. You’ll need to enter a title and short description for your collection to be created.

## Adding items to a collection
There are 2 ways to add items to a collection:

- From any repository page: Use the context menu available on any repository page then select **Add to collection** to add it to a collection (1).
- From the collection page: If you know the name of the repository you want to add, use the **+ add to collection** option in the right-hand menu (2).

![Add items to collections](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-add.webp)

It’s possible to add external repositories to your collections, not just your own.

## Collaborating on collections
Organization collections are a great way to build collections together. Any member of the organization can add, edit and remove items from the collection. Use the **history feature** to keep track of who has edited the collection.

![Collection history](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-history.webp)

## Collection options
### Collection visibility

![Collections on profiles](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-profile.webp)

**Public** collections appear at the top of your profile or organization page and can be viewed by anyone. The first 3 items in each collection are visible directly in the collection preview (1). To see more, the user must click to go to the collection page.

Set your collection to **private** if you don’t want it to be accessible via its URL (it will not be displayed on your profile/organization page). For organizations, private collections are only available to members of the organization.

### Ordering your collections and their items
You can use the drag and drop handles in the collections list (on the left side of your collections page) to change the order of your collections (1). The first two collections will be directly visible on your profile/organization pages.

You can also sort repositories within a collection by dragging the handles next to each item (2).

![Collections sort](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-sort.webp)

### Deleting items from a collection
To delete an item from a collection, click the trash icon in the menu that shows up on the right when you hover over an item (1). To delete the whole collection, click delete on the right-hand menu (2) - you’ll need to confirm this action.

![Collection delete](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-delete.webp)

### Adding notes to collection’s items
It’s possible to add a note to any item in a collection to give it more context (for others, or as a reminder to yourself). You can add notes by clicking the pencil icon when you hover over an item with your mouse. Notes are plain text and don’t support markdown, to keep things clean and simple. URLs in notes are converted into clickable links.

![Collection note](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-note.webp)

### Adding images to a collection item
Similarily, you can attach images to a collection item. This is useful for showcasing the output of a model, the content of a dataset, attaching an infographic for context, etc.

To start adding images to your collection, you can click on the image icon in the contextual menu of an item. The menu shows up when you hover over an item with your mouse.

![Collection image icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collections-image-button.webp)

Then, add images by dragging and dropping images from your computer. You can also click on the gray zone to select image files from your computer’s file system.

![Collection image drop zone with images](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collections-image-gallery.webp)

You can re-order images by drag-and-dropping them. Clicking on an image will open it in full-screen mode.

![Collection image viewer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collections-image-viewer.webp)

# Webhooks
Webhooks are now publicly available!

Webhooks are a foundation for MLOps-related features. They allow you to listen for new changes on specific repos or to all repos belonging to particular set of users/organizations (not just your repos, but any repo).

You can use them to auto-convert models, build community bots, or build CI/CD for your models, datasets, and Spaces (and much more!).

The documentation for Webhooks is below – or you can also browse our **guides** showcasing a few possible use cases of Webhooks:

- [Fine-tune a new model whenever a dataset gets updated (Python)](https://huggingface.co/docs/hub/webhooks-guide-auto-retrain)
- [Create a discussion bot on the Hub, using a LLM API (NodeJS)](https://huggingface.co/docs/hub/webhooks-guide-discussion-bot)
- [Create metadata quality reports (Python)](https://huggingface.co/docs/hub/webhooks-guide-metadata-review)
- and more to come…

## Create your Webhook
You can create new Webhooks and edit existing ones in your Webhooks [settings](https://huggingface.co/settings/webhooks):

![Settings of an individual webhook](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhook-settings.png)

Webhooks can watch for repos updates, Pull Requests, discussions, and new comments. It’s even possible to create a Space to react to your Webhooks!

## Webhook Payloads
After registering a Webhook, you will be notified of new events via an `HTTP POST` call on the specified target URL. The payload is encoded in JSON.

You can view the history of payloads sent in the activity tab of the webhook settings page, it’s also possible to replay past webhooks for easier debugging:

![image.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhook-activity.png)

As an example, here is the full payload when a Pull Request is opened:

```json
{
  "event": {
    "action": "create",
    "scope": "discussion"
  },
  "repo": {
    "type": "model",
    "name": "openai-community/gpt2",
    "id": "621ffdc036468d709f17434d",
    "private": false,
    "url": {
      "web": "https://huggingface.co/openai-community/gpt2",
      "api": "https://huggingface.co/api/models/openai-community/gpt2"
    },
    "owner": {
      "id": "628b753283ef59b5be89e937"
    }
  },
  "discussion": {
    "id": "6399f58518721fdd27fc9ca9",
    "title": "Update co2 emissions",
    "url": {
      "web": "https://huggingface.co/openai-community/gpt2/discussions/19",
      "api": "https://huggingface.co/api/models/openai-community/gpt2/discussions/19"
    },
    "status": "open",
    "author": {
      "id": "61d2f90c3c2083e1c08af22d"
    },
    "num": 19,
    "isPullRequest": true,
    "changes": {
      "base": "refs/heads/main"
    }
  },
  "comment": {
    "id": "6399f58518721fdd27fc9caa",
    "author": {
      "id": "61d2f90c3c2083e1c08af22d"
    },
    "content": "Add co2 emissions information to the model card",
    "hidden": false,
    "url": {
      "web": "https://huggingface.co/openai-community/gpt2/discussions/19#6399f58518721fdd27fc9caa"
    }
  },
  "webhook": {
    "id": "6390e855e30d9209411de93b",
    "version": 3
  }
}
```

### Event
The top-level properties `event` is always specified and used to determine the nature of the event.

It has two sub-properties: `event.action` and `event.scope`.

`event.scope` will be one of the following values:

- `"repo"` - Global events on repos. Possible values for the associated `action`: `"create"`, `"delete"`, `"update"`, `"move"`.
- `"repo.content"` - Events on the repo’s content, such as new commits or tags. It triggers on new Pull Requests as well due to the newly created reference/commit. The associated `action` is always `"update"`.
- `"repo.config"` - Events on the config: update Space secrets, update settings, update DOIs, disabled or not, etc. The associated `action` is always `"update"`.
- `"discussion"` - Creating a discussion or Pull Request, updating the title or status, and merging. Possible values for the associated `action`: `"create"`, `"delete"`, `"update"`.
- `"discussion.comment"` - Creating, updating, and hiding a comment. Possible values for the associated `action`: `"create"`, `"update"`.

More scopes can be added in the future. To handle unknown events, your webhook handler can consider any action on a narrowed scope to be an `"update"` action on the broader scope.

For example, if the `"repo.config.dois"` scope is added in the future, any event with that scope can be considered by your webhook handler as an `"update"` action on the `"repo.config"` scope.

### Repo
In the current version of webhooks, the top-level property `repo` is always specified, as events can always be associated with a repo. For example, consider the following value:

```
"repo": {
	"type": "model",
	"name": "some-user/some-repo",
	"id": "6366c000a2abcdf2fd69a080",
	"private": false,
	"url": {
		"web": "https://huggingface.co/some-user/some-repo",
		"api": "https://huggingface.co/api/models/some-user/some-repo"
	},
	"headSha": "c379e821c9c95d613899e8c4343e4bfee2b0c600",
	"tags": [
		"license:other",
		"has_space"
	],
	"owner": {
		"id": "61d2000c3c2083e1c08af22d"
	}
}
```

`repo.headSha` is the sha of the latest commit on the repo’s `main` branch. It is only sent when `event.scope` starts with `"repo"`, not on community events like discussions and comments.

### Code changes
On code changes, the top-level property `updatedRefs` is specified on repo events. It is an array of references that have been updated. Here is an example value:

```
"updatedRefs": [
  {
    "ref": "refs/heads/main",
    "oldSha": "ce9a4674fa833a68d5a73ec355f0ea95eedd60b7",
    "newSha": "575db8b7a51b6f85eb06eee540738584589f131c"
  },
  {
    "ref": "refs/tags/test",
    "oldSha": null,
    "newSha": "575db8b7a51b6f85eb06eee540738584589f131c"
  }
]
```

Newly created references will have `oldSha` set to `null`. Deleted references will have `newSha` set to `null`.

You can react to new commits on specific pull requests, new tags, or new branches.

### Discussions and Pull Requests
The top-level property `discussion` is specified on community events (discussions and Pull Requests). The `discussion.isPullRequest` property is a boolean indicating if the discussion is also a Pull Request (on the Hub, a PR is a special type of discussion). Here is an example value:

```
"discussion": {
	"id": "639885d811ae2bad2b7ba461",
	"title": "Hello!",
	"url": {
		"web": "https://huggingface.co/some-user/some-repo/discussions/3",
		"api": "https://huggingface.co/api/models/some-user/some-repo/discussions/3"
	},
	"status": "open",
	"author": {
		"id": "61d2000c3c2083e1c08af22d"
	},
	"isPullRequest": true,
	"changes": {
		"base": "refs/heads/main"
	}
	"num": 3
}
```

### Comment
The top level property `comment` is specified when a comment is created (including on discussion creation) or updated. Here is an example value:

```
"comment": {
	"id": "6398872887bfcfb93a306f18",
	"author": {
		"id": "61d2000c3c2083e1c08af22d"
	},
	"content": "This adds an env key",
	"hidden": false,
	"url": {
		"web": "https://huggingface.co/some-user/some-repo/discussions/4#6398872887bfcfb93a306f18"
	}
}
```

## Webhook secret
Setting a Webhook secret is useful to make sure payloads sent to your Webhook handler URL are actually from Hugging Face.

If you set a secret for your Webhook, it will be sent along as an `X-Webhook-Secret` HTTP header on every request. Only ASCII characters are supported.

It's also possible to add the secret directly in the handler URL. For example, setting it as a query parameter: https://example.com/webhook?secret=XXX.

This can be helpful if accessing the HTTP headers of the request is complicated for your Webhook handler.

## Rate limiting
Each Webhook is limited to 1,000 triggers per 24 hours. You can view your usage in the Webhook settings page in the “Activity” tab.

If you need to increase the number of triggers for your Webhook, contact us at [website@huggingface.co](mailto:website@huggingface.co).

## Developing your Webhooks
If you do not have an HTTPS endpoint/URL, you can try out public tools for webhook testing. These tools act as catch-all (capture all requests) sent to them and give 200 OK status code. [Beeceptor](https://beeceptor.com/) is one tool you can use to create a temporary HTTP endpoint and review the incoming payload. Another such tool is [Webhook.site](https://webhook.site/).

Additionally, you can route a real Webhook payload to the code running locally on your machine during development. This is a great way to test and debug for faster integrations. You can do this by exposing your localhost port to the Internet. To be able to go this path, you can use [ngrok](https://ngrok.com/) or [localtunnel](https://theboroer.github.io/localtunnel-www/).

## Debugging Webhooks
You can easily find recently generated events for your webhooks. Open the activity tab for your webhook. There you will see the list of recent events.

![image.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhook-payload.png)

Here you can review the HTTP status code and the payload of the generated events. Additionally, you can replay these events by clicking on the `Replay` button!

Note: When changing the target URL or secret of a Webhook, replaying an event will send the payload to the updated URL.

## FAQ
##### Can I define webhooks on my organization vs my user account?
No, this is not currently supported.

##### How can I subscribe to events on all repos (or across a whole repo type, like on all models)?
This is not currently exposed to end users but we can toggle this for you if you send an email to [website@huggingface.co](mailto:website@huggingface.co).

# Jupyter Notebooks on the Hugging Face Hub
[Jupyter notebooks](https://jupyter.org/) are a very popular format for sharing code and data analysis for machine learning and data science. They are interactive documents that can contain code, visualizations, and text.

## Rendering Jupyter notebooks on the Hub
Under the hood, Jupyter Notebook files (usually shared with a `.ipynb` extension) are JSON files. While viewing these files directly is possible, it’s not a format intended to be read by humans. The Hub has rendering support for notebooks hosted on the Hub. This means that notebooks are displayed in a human-readable format.
> Jupyter Notebook (`.ipynb`) 本质上是 JSON 文件，Huggingface Hub 会自动将该文件渲染为人类可读的格式

![Before and after notebook rendering](https://huggingface.co/blog/assets/135_notebooks-hub/before_after_notebook_rendering.png)

Notebooks will be rendered when included in any type of repository on the Hub. This includes models, datasets, and Spaces.

## Launch in Google Colab
[Google Colab](https://colab.google/) is a free Jupyter Notebook environment that requires no setup and runs entirely in the cloud. It’s a great way to run Jupyter Notebooks without having to install anything on your local machine. Notebooks hosted on the Hub are automatically given a “launch in Google Colab” button. This allows you to open the notebook in Colab with a single click.

# Repository limitations and recommendations
There are some limitations to be aware of when dealing with a large amount of data in your repo. Given the time it takes to stream the data, getting an upload/push to fail at the end of the process or encountering a degraded experience, be it on hf.co or when working locally, can be very annoying.

## Recommendations
We gathered a list of tips and recommendations for structuring your repo. If you are looking for more practical tips, check out [this guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#tips-and-tricks-for-large-uploads) on how to upload large amount of data using the Python library.

|Characteristic|Recommended|Tips|
|---|---|---|
|Repo size|-|contact us for large repos (TBs of data)|
|Files per repo|<100k|merge data into fewer files|
|Entries per folder|<10k|use subdirectories in repo|
|File size|<20GB|split data into chunked files|
|Commit size|<100 files*|upload files in multiple commits|
|Commits per repo|-|upload multiple files per commit and/or squash history|

_* Not relevant when using `git` CLI directly_

Please read the next section to understand better those limits and how to deal with them.

## Explanations
What are we talking about when we say “large uploads”, and what are their associated limitations? Large uploads can be very diverse, from repositories with a few huge files (e.g. model weights) to repositories with thousands of small files (e.g. an image dataset).

Under the hood, the Hub uses Git to version the data, which has structural implications on what you can do in your repo. If your repo is crossing some of the numbers mentioned in the previous section, **we strongly encourage you to check out [`git-sizer`](https://github.com/github/git-sizer)**, which has very detailed documentation about the different factors that will impact your experience. Here is a TL;DR of factors to consider:

- **Repository size**: The total size of the data you’re planning to upload. We generally support repositories up to 300GB. If you would like to upload more than 300 GBs (or even TBs) of data, you will need to ask us to grant more storage. To do that, please send an email with details of your project to [datasets@huggingface.co](mailto:datasets@huggingface.co).
- **Number of files**:
    - For optimal experience, we recommend keeping the total number of files under 100k. Try merging the data into fewer files if you have more. For example, json files can be merged into a single jsonl file, or large datasets can be exported as Parquet files or in [WebDataset](https://github.com/webdataset/webdataset) format.
    - The maximum number of files per folder cannot exceed 10k files per folder. A simple solution is to create a repository structure that uses subdirectories. For example, a repo with 1k folders from `000/` to `999/`, each containing at most 1000 files, is already enough.
- **File size**: In the case of uploading large files (e.g. model weights), we strongly recommend splitting them **into chunks of around 20GB each**. There are a few reasons for this:
    - Uploading and downloading smaller files is much easier both for you and the other users. Connection issues can always happen when streaming data and smaller files avoid resuming from the beginning in case of errors.
    - Files are served to the users using CloudFront. From our experience, huge files are not cached by this service leading to a slower download speed. In all cases no single LFS file will be able to be >50GB. I.e. 50GB is the hard limit for single file size.
- **Number of commits**: There is no hard limit for the total number of commits on your repo history. However, from our experience, the user experience on the Hub starts to degrade after a few thousand commits. We are constantly working to improve the service, but one must always remember that a git repository is not meant to work as a database with a lot of writes. If your repo’s history gets very large, it is always possible to squash all the commits to get a fresh start using `huggingface_hub`’s [`super_squash_history`](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.super_squash_history). Be aware that this is a non-revertible operation.
- **Number of operations per commit**: Once again, there is no hard limit here. When a commit is uploaded on the Hub, each git operation (addition or delete) is checked by the server. When a hundred LFS files are committed at once, each file is checked individually to ensure it’s been correctly uploaded. When pushing data through HTTP, a timeout of 60s is set on the request, meaning that if the process takes more time, an error is raised. However, it can happen (in rare cases) that even if the timeout is raised client-side, the process is still completed server-side. This can be checked manually by browsing the repo on the Hub. To prevent this timeout, we recommend adding around 50-100 files per commit.

## Sharing large datasets on the Hub
One key way Hugging Face supports the machine learning ecosystem is by hosting datasets on the Hub, including very large ones. However, if your dataset is bigger than 300GB, you will need to ask us to grant more storage.

In this case, to ensure we can effectively support the open-source ecosystem, we require you to let us know via [datasets@huggingface.co](mailto:datasets@huggingface.co).

When you get in touch with us, please let us know:

- What is the dataset, and who/what is it likely to be useful for?
- The size of the dataset.
- The format you plan to use for sharing your dataset.

For hosting large datasets on the Hub, we require the following for your dataset:

- A dataset card: we want to ensure that your dataset can be used effectively by the community and one of the key ways of enabling this is via a dataset card. This [guidance](https://huggingface.co/docs/hub/datasets-cards.md) provides an overview of how to write a dataset card.
- You are sharing the dataset to enable community reuse. If you plan to upload a dataset you anticipate won’t have any further reuse, other platforms are likely more suitable.
- You must follow the repository limitations outlined above.
- Using file formats that are well integrated with the Hugging Face ecosystem. We have good support for [Parquet](https://huggingface.co/docs/datasets/v2.19.0/en/loading#parquet) and [WebDataset](https://huggingface.co/docs/datasets/v2.19.0/en/loading#webdataset) formats, which are often good options for sharing large datasets efficiently. This will also ensure the dataset viewer works for your dataset.
- Avoid the use of custom loading scripts when using datasets. In our experience, datasets that require custom code to use often end up with limited reuse.

Please get in touch with us if any of these requirements are difficult for you to meet because of the type of data or domain you are working in.

# Next Steps
These next sections highlight features and additional information that you may find useful to make the most out of the Git repositories on the Hugging Face Hub.

## How to programmatically manage repositories
Hugging Face supports accessing repos with Python via the [`huggingface_hub` library](https://huggingface.co/docs/huggingface_hub/index). The operations that we’ve explored, such as downloading repositories and uploading files, are available through the library, as well as other useful functions!
> Huggingface 支持通过 `huggingface_hub` 库访问 repo，执行下载、上传文件等操作

If you prefer to use git directly, please read the sections below.

## Learning more about Git
A good place to visit if you want to continue learning about Git is [this Git tutorial](https://learngitbranching.js.org/). For even more background on Git, you can take a look at [GitHub’s Git Guides](https://github.com/git-guides).

## How to use branches
To effectively use Git repos collaboratively and to work on features without releasing premature code you can use **branches**. Branches allow you to separate your “work in progress” code from your “production-ready” code, with the additional benefit of letting multiple people work on a project without frequently conflicting with each others’ contributions. You can use branches to isolate experiments in their own branch, and even [adopt team-wide practices for managing branches](https://ericmjl.github.io/essays-on-data-science/workflow/gitflow/).

To learn about Git branching, you can try out the [Learn Git Branching interactive tutorial](https://learngitbranching.js.org/).

## Using tags
Git allows you to _tag_ commits so that you can easily note milestones in your project. As such, you can use tags to mark commits in your Hub repos! To learn about using tags, you can visit [this DevConnected post](https://devconnected.com/how-to-create-git-tags/).

Beyond making it easy to identify important commits in your repo’s history, using Git tags also allows you to do A/B testing, [clone a repository at a specific tag](https://www.techiedelight.com/clone-specific-tag-with-git/), and more! The `huggingface_hub` library also supports working with tags, such as [downloading files from a specific tagged commit](https://huggingface.co/docs/huggingface_hub/main/en/how-to-downstream#hfhuburl).

## How to duplicate or fork a repo (including LFS pointers)
If you’d like to copy a repository, depending on whether you want to preserve the Git history there are two options.

### Duplicating without Git history
In many scenarios, if you want your own copy of a particular codebase you might not be concerned about the previous Git history. In this case, you can quickly duplicate a repo with the handy [Repo Duplicator](https://huggingface.co/spaces/huggingface-projects/repo_duplicator)! You’ll have to create a User Access Token, which you can read more about in the [security documentation](https://huggingface.co/docs/hub/security-tokens).

### Duplicating with the Git history (Fork)
A duplicate of a repository with the commit history preserved is called a _fork_. You may choose to fork one of your own repos, but it also common to fork other people’s projects if you would like to tinker with them.

**Note that you will need to [install Git LFS](https://git-lfs.github.com/) and the [`huggingface_hub` CLI](https://huggingface.co/docs/huggingface_hub/index) to follow this process**. When you want to fork or [rebase](https://git-scm.com/docs/git-rebase) a repository with LFS files you cannot use the usual Git approach that you might be familiar with since you need to be careful to not break the LFS pointers. Forking can take time depending on your bandwidth because you will have to fetch and re-upload all the LFS files in your fork.

For example, say you have an upstream repository, **upstream**, and you just created your own repository on the Hub which is **myfork** in this example.

1. Create a destination repository (e.g. **myfork**) in [https://huggingface.co](https://huggingface.co/)
2. Clone your fork repository:

```
git clone git@hf.co:me/myfork
```

3. Fetch non-LFS files:

```
cd myfork
git lfs install --skip-smudge --local # affects only this clone
git remote add upstream git@hf.co:friend/upstream
git fetch upstream
```

4. Fetch large files. This can take some time depending on your download bandwidth:

```
git lfs fetch --all upstream # this can take time depending on your download bandwidth
```

4.a. If you want to completely override the fork history (which should only have an initial commit), run:

```
git reset --hard upstream/main
```

4.b. If you want to rebase instead of overriding, run the following command and resolve any conflicts:

```
git rebase upstream/main
```

5. Prepare your LFS files to push:

```
git lfs install --force --local # this reinstalls the LFS hooks
huggingface-cli lfs-enable-largefiles . # needed if some files are bigger than 5GB
```

6. And finally push:

```
git push --force origin main # this can take time depending on your upload bandwidth
```

Now you have your own fork or rebased repo in the Hub!

# Licenses
You are able to add a license to any repo that you create on the Hugging Face Hub to let other users know about the permissions that you want to attribute to your code or data. The license can be specified in your repository’s `README.md` file, known as a _card_ on the Hub, in the card’s metadata section. Remember to seek out and respect a project’s license if you’re considering using their code or data.

A full list of the available licenses is available here:

|Fullname|License identifier (to use in repo card)|
|---|---|
|Apache license 2.0|`apache-2.0`|
|MIT|`mit`|
|OpenRAIL license family|`openrail`|
|BigScience OpenRAIL-M|`bigscience-openrail-m`|
|CreativeML OpenRAIL-M|`creativeml-openrail-m`|
|BigScience BLOOM RAIL 1.0|`bigscience-bloom-rail-1.0`|
|BigCode Open RAIL-M v1|`bigcode-openrail-m`|
|Academic Free License v3.0|`afl-3.0`|
|Artistic license 2.0|`artistic-2.0`|
|Boost Software License 1.0|`bsl-1.0`|
|BSD license family|`bsd`|
|BSD 2-clause “Simplified” license|`bsd-2-clause`|
|BSD 3-clause “New” or “Revised” license|`bsd-3-clause`|
|BSD 3-clause Clear license|`bsd-3-clause-clear`|
|Computational Use of Data Agreement|`c-uda`|
|Creative Commons license family|`cc`|
|Creative Commons Zero v1.0 Universal|`cc0-1.0`|
|Creative Commons Attribution 2.0|`cc-by-2.0`|
|Creative Commons Attribution 2.5|`cc-by-2.5`|
|Creative Commons Attribution 3.0|`cc-by-3.0`|
|Creative Commons Attribution 4.0|`cc-by-4.0`|
|Creative Commons Attribution Share Alike 3.0|`cc-by-sa-3.0`|
|Creative Commons Attribution Share Alike 4.0|`cc-by-sa-4.0`|
|Creative Commons Attribution Non Commercial 2.0|`cc-by-nc-2.0`|
|Creative Commons Attribution Non Commercial 3.0|`cc-by-nc-3.0`|
|Creative Commons Attribution Non Commercial 4.0|`cc-by-nc-4.0`|
|Creative Commons Attribution No Derivatives 4.0|`cc-by-nd-4.0`|
|Creative Commons Attribution Non Commercial No Derivatives 3.0|`cc-by-nc-nd-3.0`|
|Creative Commons Attribution Non Commercial No Derivatives 4.0|`cc-by-nc-nd-4.0`|
|Creative Commons Attribution Non Commercial Share Alike 2.0|`cc-by-nc-sa-2.0`|
|Creative Commons Attribution Non Commercial Share Alike 3.0|`cc-by-nc-sa-3.0`|
|Creative Commons Attribution Non Commercial Share Alike 4.0|`cc-by-nc-sa-4.0`|
|Community Data License Agreement – Sharing, Version 1.0|`cdla-sharing-1.0`|
|Community Data License Agreement – Permissive, Version 1.0|`cdla-permissive-1.0`|
|Community Data License Agreement – Permissive, Version 2.0|`cdla-permissive-2.0`|
|Do What The F*ck You Want To Public License|`wtfpl`|
|Educational Community License v2.0|`ecl-2.0`|
|Eclipse Public License 1.0|`epl-1.0`|
|Eclipse Public License 2.0|`epl-2.0`|
|Etalab Open License 2.0|`etalab-2.0`|
|European Union Public License 1.1|`eupl-1.1`|
|GNU Affero General Public License v3.0|`agpl-3.0`|
|GNU Free Documentation License family|`gfdl`|
|GNU General Public License family|`gpl`|
|GNU General Public License v2.0|`gpl-2.0`|
|GNU General Public License v3.0|`gpl-3.0`|
|GNU Lesser General Public License family|`lgpl`|
|GNU Lesser General Public License v2.1|`lgpl-2.1`|
|GNU Lesser General Public License v3.0|`lgpl-3.0`|
|ISC|`isc`|
|LaTeX Project Public License v1.3c|`lppl-1.3c`|
|Microsoft Public License|`ms-pl`|
|Apple Sample Code license|`apple-ascl`|
|Mozilla Public License 2.0|`mpl-2.0`|
|Open Data Commons License Attribution family|`odc-by`|
|Open Database License family|`odbl`|
|Open Rail++-M License|`openrail++`|
|Open Software License 3.0|`osl-3.0`|
|PostgreSQL License|`postgresql`|
|SIL Open Font License 1.1|`ofl-1.1`|
|University of Illinois/NCSA Open Source License|`ncsa`|
|The Unlicense|`unlicense`|
|zLib License|`zlib`|
|Open Data Commons Public Domain Dedication and License|`pddl`|
|Lesser General Public License For Linguistic Resources|`lgpl-lr`|
|DeepFloyd IF Research License Agreement|`deepfloyd-if-license`|
|Llama 2 Community License Agreement|`llama2`|
|Llama 3 Community License Agreement|`llama3`|
|Llama 3.1 Community License Agreement|`llama3.1`|
|Llama 3.2 Community License Agreement|`llama3.2`|
|Gemma Terms of Use|`gemma`|
|Unknown|`unknown`|
|Other|`other`|

In case of `license: other` please add the license’s text to a `LICENSE` file inside your repo (or contact us to add the license you use to this list), and set a name for it in `license_name`.