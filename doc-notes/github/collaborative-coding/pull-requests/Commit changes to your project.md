---
completed:
---
# Create & edit commits
## About commits
You can save small groups of meaningful changes as commits.

### About commits
Similar to saving a file that's been edited, a commit records changes to one or more files in your branch. Git assigns each commit a unique ID, called a SHA or hash, that identifies:

- The specific changes
- When the changes were made
- Who created the changes

When you make a commit, you must include a commit message that briefly describes the changes.

If the repository you are committing to has compulsory commit signoffs enabled, and you are committing via the web interface, you will automatically sign off on the commit as part of the commit process. For more information, see [Managing the commit signoff policy for your repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-the-commit-signoff-policy-for-your-repository).

You can add a co-author on any commits you collaborate on. For more information, see [Creating a commit with multiple authors](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors).

You can also create a commit on behalf of an organization. For more information, see [Creating a commit on behalf of an organization](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-on-behalf-of-an-organization).

Rebasing allows you to change a series of commits and can modify the order of the commits in your timeline. For more information, see [About Git rebase](https://docs.github.com/en/get-started/using-git/about-git-rebase).

### About commit branches and tag labels
You can see which branch a commit is on by looking at the labels beneath the commit on the commit page.

1. On GitHub, navigate to the main page of the repository.
2. On the main page of the repository, above the file list, click  **commits**.
    
    ![Screenshot of the main page for a repository. A clock icon and "178 commits" is highlighted with an orange outline.](https://docs.github.com/assets/cb-48469/images/help/commits/commits-page.png)
    
3. To navigate to a specific commit, click the commit message for that commit.
    
    ![Screenshot of a commit in the commit list for a repository. "Update README.md" is highlighted with an orange outline.](https://docs.github.com/assets/cb-17733/images/help/commits/commit-message-link.png)
    
4. To see what branch the commit is on, check the label below the commit message.
    
    ![Screenshot of a commit summary. A branch icon and "main" are highlighted with an orange outline.](https://docs.github.com/assets/cb-15839/images/help/commits/commit-branch-indicator.png)
    

If your commit is not on the default branch (`main`), the label will show the branches which contain the commit. If the commit is part of an unmerged pull request, you can click the link to go to the pull request.

Once the commit is on the default branch, any tags that contain the commit will be shown and the default branch will be the only branch listed. For more information on tags, see [Git Basics - Tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging) in the Git documentation.

![Screenshot of a commit summary. The tag icon and "v2.3.4" are highlighted with an orange outline.](https://docs.github.com/assets/cb-17593/images/help/commits/commit-tag-label.png)

### Using the file tree
You can use the file tree to navigate between files in a commit.

1. On GitHub, navigate to the main page of the repository.
2. On the main page of the repository, above the file list, click  **commits**.
    
    ![Screenshot of the main page for a repository. A clock icon and "178 commits" is highlighted with an orange outline.](https://docs.github.com/assets/cb-48469/images/help/commits/commits-page.png)
    
3. To navigate to a specific commit, click the commit message for that commit.
    
    ![Screenshot of a commit in the commit list for a repository. "Update README.md" is highlighted with an orange outline.](https://docs.github.com/assets/cb-17733/images/help/commits/commit-message-link.png)
    
4. Click on a file in the file tree to view the corresponding file diff. If the file tree is hidden, click  to display the file tree.
    
    Note
    
    The file tree will not display if your screen width is too narrow or if the commit only includes one file.
    
    ![Screenshot of the "Files changed" tab of a pull request. In the left sidebar, the file tree is outlined in dark orange.](https://docs.github.com/assets/cb-140879/images/help/repository/file-tree.png)
    
5. To filter by file path, enter part or all of the file path in the **Filter changed files** search box.

### Further reading
- [Committing and reviewing changes to your project in GitHub Desktop](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop#about-commits) on GitHub Desktop

## Creating a commit with multiple authors
You can attribute a commit to more than one author by adding one or more `Co-authored-by` trailers to the commit's message. Co-authored commits are visible on GitHub.

### Required co-author information
Before you can add a co-author to a commit, you must know the appropriate email to use for each co-author. For the co-author's commit to count as a contribution, you must use the email associated with their account on GitHub.com.

If a person chooses to keep their email address private, you should use their GitHub-provided `no-reply` email to protect their privacy. Otherwise, the co-author's email will be available to the public in the commit message. If you want to keep your email private, you can choose to use a GitHub-provided `no-reply` email for Git operations and ask other co-authors to list your `no-reply` email in commit trailers.
>  如果一个人选择将他们的电子邮件地址保密，你应该使用他们由 GitHub 提供的 `no-reply` 电子邮件来保护他们的隐私。否则，合著者的电子邮件将在提交信息中对公众可见。如果你想保持你的电子邮件私密，你可以选择在 Git 操作中使用 GitHub 提供的 `no-reply` 电子邮件，并要求其他合著者在提交尾注中列出你的 `no-reply` 电子邮件。

For more information, see [Setting your commit email address](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address).

Tip
You can help a co-author find their preferred email address by sharing this information:

- To find your GitHub-provided `no-reply` email, navigate to your email settings page under "Keep my email address private."
- To find the email you used to configure Git on your computer, run `git config user.email` on the command line.

### Creating co-authored commits using GitHub Desktop
You can use GitHub Desktop to create a commit with a co-author. For more information, see [Committing and reviewing changes to your project in GitHub Desktop](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop#write-a-commit-message-and-push-your-changes) and [GitHub Desktop](https://desktop.github.com/).

### Creating co-authored commits on the command line
1. Collect the name and email address for each co-author. If a person chooses to keep their email address private, you should use their GitHub-provided `no-reply` email to protect their privacy.
2. Type your commit message and a short, meaningful description of your changes. After your commit description, instead of a closing quotation, add an empty line.
    
```shell
$ git commit -m "Refactor usability tests.
>
>
```
    
Tip
If you're using a text editor on the command line to type your commit message, ensure there is **a blank line** (two consecutive newlines) between the end of your commit description and the `Co-authored-by:` **commit trailer**.
    
3. On the next line of the commit message, type `Co-authored-by: name <name@example.com>` with specific information for each co-author. After the co-author information, add a closing quotation mark.
    
    If you're adding multiple co-authors, give each co-author their own line and `Co-authored-by:` commit trailer. Do not add blank lines between each co-author line.
    
    ```shell
    $ git commit -m "Refactor usability tests.
    >
    > Co-authored-by: NAME <NAME@EXAMPLE.COM>
    > Co-authored-by: ANOTHER-NAME <ANOTHER-NAME@EXAMPLE.COM>"
    ```
    

The new commit and message will appear on GitHub.com the next time you push. For more information, see [Pushing commits to a remote repository](https://docs.github.com/en/get-started/using-git/pushing-commits-to-a-remote-repository).

>  commit 的 coauthor 需要在 commit message 的 trailer 中显式写出，格式如上所示

### Creating co-authored commits on GitHub
After you've made changes in a file using the web editor on GitHub, you can create a co-authored commit by adding a `Co-authored-by:` trailer to the commit's message.

1. Collect the name and email address for each co-author. If a person chooses to keep their email address private, you should use their GitHub-provided `no-reply` email to protect their privacy.
2. Click **Commit changes...**
3. In the "Commit message" field, type a short, meaningful commit message that describes the changes you made.
4. In the text box below your commit message, add `Co-authored-by: name <name@example.com>` with specific information for each co-author. If you're adding multiple co-authors, give each co-author their own line and `Co-authored-by:` commit trailer.
5. Click **Commit changes** or **Propose changes**.

The new commit and message will appear on GitHub.com.

### Further reading
- [Viewing a project's contributors](https://docs.github.com/en/repositories/viewing-activity-and-data-for-your-repository/viewing-a-projects-contributors)
- [Changing a commit message](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/changing-a-commit-message)
- [Committing and reviewing changes to your project in GitHub Desktop](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop#4-write-a-commit-message-and-push-your-changes) in the GitHub Desktop documentation

