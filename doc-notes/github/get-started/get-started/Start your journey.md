## About GitHub and Git
You can use GitHub and Git to collaborate on work.
### [About GitHub](https://docs.github.com/en/get-started/start-your-journey/about-github-and-git#about-github)
GitHub is a cloud-based platform where you can store, share, and work together with others to write code.

Storing your code in a "repository" on GitHub allows you to:

- **Showcase or share** your work.
- **Track and manage** changes to your code over time.
- Let others **review** your code, and make suggestions to improve it.
- **Collaborate** on a shared project, without worrying that your changes will impact the work of your collaborators before you're ready to integrate them.

Collaborative working, one of GitHub’s fundamental features, is made possible by the open-source software, Git, upon which GitHub is built.
### [About Git](https://docs.github.com/en/get-started/start-your-journey/about-github-and-git#about-git)
Git is a version control system that intelligently tracks changes in files. Git is particularly useful when you and a group of people are all making changes to the same files at the same time.

Typically, to do this in a Git-based workflow, you would:

- **Create a branch** off from the main copy of files that you (and your collaborators) are working on.
- **Make edits** to the files independently and safely on your own personal branch.
- Let Git intelligently **merge** your specific changes back into the main copy of files, so that your changes don't impact other people's updates.
- Let Git **keep track** of your and other people's changes, so you all stay working on the most up-to-date version of the project.

If you want to learn more about Git, see "[About Git](https://docs.github.com/en/get-started/using-git/about-git)."
#### [How do Git and GitHub work together?](https://docs.github.com/en/get-started/start-your-journey/about-github-and-git#how-do-git-and-github-work-together)
When you upload files to GitHub, you'll store them in a "Git repository." This means that when you make changes (or "commits") to your files in GitHub, Git will automatically start to track and manage your changes.

There are plenty of Git-related actions that you can complete on GitHub directly in your browser, such as creating a Git repository, creating branches, and uploading and editing files.

However, most people work on their files locally (on their own computer), then continually sync these local changes—and all the related Git data—with the central "remote" repository on GitHub. There are plenty of tools that you can use to do this, such as GitHub Desktop.

Once you start to collaborate with others and all need to work on the same repository at the same time, you’ll continually:

- **Pull** all the latest changes made by your collaborators from the remote repository on GitHub.
- **Push** back your own changes to the same remote repository on GitHub.

Git figures out how to intelligently merge this flow of changes, and GitHub helps you manage the flow through features such as "pull requests."
### [Where do I start?](https://docs.github.com/en/get-started/start-your-journey/about-github-and-git#where-do-i-start)
If you're new to GitHub, and unfamiliar with Git, we recommend working through the articles in the "[Start your journey](https://docs.github.com/en/get-started/start-your-journey)" category. The articles focus on tasks you can complete directly in your browser on GitHub and will help you to:

- **Create an account** on GitHub.
- **Learn the "GitHub Flow"**, and the key principles of collaborative working (branches, commits, pull requests, merges).
- **Personalise your profile** to share your interests and skills.
- **Explore GitHub** to find inspiration for your own projects and connect with others.
- Learn how to **download** interesting code for your own use.
- Learn how to **upload** something you're working on to a GitHub repository.
## Create an account
### [About your personal account on GitHub.com](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github#about-your-personal-account-on-githubcom)
To get started with GitHub, you'll need to create a free personal account on GitHub.com and verify your email address.

Every person who uses GitHub signs in to a user account. Your user account is your identity on GitHub and has a username and profile. For example, see [@octocat's profile](https://github.com/octocat).

Later, you can explore the different types of accounts that GitHub offers, and decide if you need a billing plan. For more information, see "[Types of GitHub accounts](https://docs.github.com/en/get-started/learning-about-github/types-of-github-accounts)" and "[GitHub’s plans](https://docs.github.com/en/get-started/learning-about-github/githubs-plans)."

Note that the steps in this article don't apply to Enterprise Managed Users. If your GitHub account has been created for you by your company, you can skip this article and continue to "[Hello World](https://docs.github.com/en/get-started/start-your-journey/hello-world)."
### [Signing up for a new personal account](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github#signing-up-for-a-new-personal-account)

1. Navigate to [https://github.com/](https://github.com/).
2. Click **Sign up**.
3. Follow the prompts to create your personal account.

During sign up, you'll be asked to verify your email address. Without a verified email address, you won't be able to complete some basic GitHub tasks, such as creating a repository.

If you're having problems verifying your email address, there are some troubleshooting steps you can take. For more information, see "[Verifying your email address](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/verifying-your-email-address#troubleshooting-email-verification)."
### [Next steps](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github#next-steps)

- Now that you've created your personal account, we'll start to explore the basics of GitHub. In the next tutorial, "[Hello World](https://docs.github.com/en/get-started/start-your-journey/hello-world)," you'll learn about repositories and how to create one, and you'll be introduced to concepts such as branching, commits, and pull requests.
- We strongly recommend that you configure 2FA for your account. 2FA is an extra layer of security that can help keep your account secure. For more information, see "[Configuring two-factor authentication](https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa/configuring-two-factor-authentication)."
## Hello World
### [Introduction](https://docs.github.com/en/get-started/start-your-journey/hello-world#introduction)
This tutorial teaches you GitHub essentials like repositories, branches, commits, and pull requests. You'll create your own Hello World repository and learn GitHub's pull request workflow, a popular way to create and review code.

In this quickstart guide, you will:

- Create and use a repository.
- Start and manage a new branch.
- Make changes to a file and push them to GitHub as commits.
- Open and merge a pull request.

#### [Prerequisites](https://docs.github.com/en/get-started/start-your-journey/hello-world#prerequisites)

- You must have a GitHub account. For more information, see "[Creating an account on GitHub](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github)."
- You don't need to know how to code, use the command line, or install Git (the version control software that GitHub is built on).

### [Step 1: Create a repository](https://docs.github.com/en/get-started/start-your-journey/hello-world#step-1-create-a-repository)
The first thing we'll do is create a repository. You can think of a repository as a folder that contains related items, such as files, images, videos, or even other folders. A repository usually groups together items that belong to the same "project" or thing you're working on.

Often, repositories include a README file, a file with information about your project. README files are written in Markdown, which is an easy-to-read, easy-to-write language for formatting plain text. We'll learn more about Markdown in the next tutorial, "[Setting up your profile](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile)."

GitHub lets you add a README file at the same time you create your new repository. GitHub also offers other common options such as a license file, but you do not have to select any of them now.

Your `hello-world` repository can be a place where you store ideas, resources, or even share and discuss things with others.

1. In the upper-right corner of any page, select , then click **New repository**.
    
    ![Screenshot of a GitHub dropdown menu showing options to create new items. The menu item "New repository" is outlined in dark orange.](https://docs.github.com/assets/cb-29762/images/help/repository/repo-create-global-nav-update.png)
    
2. In the "Repository name" box, type `hello-world`.
3. In the "Description" box, type a short description. For example, type "This repository is for practicing the GitHub Flow."
4. Select whether your repository will be **Public** or **Private**.
5. Select **Add a README file**.
6. Click **Create repository**.

### [Step 2: Create a branch](https://docs.github.com/en/get-started/start-your-journey/hello-world#step-2-create-a-branch)
Branching lets you have different versions of a repository at one time.

By default, your repository has one branch named `main` that is considered to be the definitive branch. You can create additional branches off of `main` in your repository.

Branching is helpful when you want to add new features to a project without changing the main source of code. The work done on different branches will not show up on the main branch until you merge it, which we will cover later in this guide. You can use branches to experiment and make edits before committing them to `main`.

When you create a branch off the `main` branch, you're making a copy, or snapshot, of `main` as it was at that point in time. If someone else made changes to the `main` branch while you were working on your branch, you could pull in those updates.

This diagram shows:

- The `main` branch
- A new branch called `feature`
- The journey that `feature` takes before it's merged into `main`

![Diagram of the two branches. The "feature" branch diverges from the "main" branch, goes through stages for "Commit changes," "Submit pull request," and "Discuss proposed changes," and is then merged back into main.](https://docs.github.com/assets/cb-23923/images/help/repository/branching.png)

#### [Creating a branch](https://docs.github.com/en/get-started/start-your-journey/hello-world#creating-a-branch)

1. Click the **Code** tab of your `hello-world` repository.
2. Above the file list, click the dropdown menu that says **main**.
    
    ![Screenshot of the repository page. A dropdown menu, labeled with a branch icon and "main", is highlighted with an orange outline.](https://docs.github.com/assets/cb-16584/images/help/branches/branch-selection-dropdown-global-nav-update.png)
    
3. Type a branch name, `readme-edits`, into the text box.
4. Click **Create branch: readme-edits from main**.
    
    ![Screenshot of the branch dropdown for a repository. "Create branch: readme-edits from 'main'" is outlined in dark orange.](https://docs.github.com/assets/cb-31023/images/help/repository/new-branch.png)
    

Now you have two branches, `main` and `readme-edits`. Right now, they look exactly the same. Next you'll add changes to the new `readme-edits` branch.
### [Step 3: Make and commit changes](https://docs.github.com/en/get-started/start-your-journey/hello-world#step-3-make-and-commit-changes)
When you created a new branch in the previous step, GitHub brought you to the code page for your new `readme-edits` branch, which is a copy of `main`.

You can make and save changes to the files in your repository. On GitHub, saved changes are called commits. Each commit has an associated commit message, which is a description explaining why a particular change was made. Commit messages capture the history of your changes so that other contributors can understand what you’ve done and why.

1. Under the `readme-edits` branch you created, click the `README.md` file.
2. To edit the file, click .
3. In the editor, write a bit about yourself.
4. Click **Commit changes**.
5. In the "Commit changes" box, write a commit message that describes your changes.
6. Click **Commit changes**.

These changes will be made only to the README file on your `readme-edits` branch, so now this branch contains content that's different from `main`.
### [Step 4: Open a pull request](https://docs.github.com/en/get-started/start-your-journey/hello-world#step-4-open-a-pull-request)
Now that you have changes in a branch off of `main`, you can open a pull request.

Pull requests are the heart of collaboration on GitHub. When you open a pull request, you're proposing your changes and requesting that someone review and pull in your contribution and merge them into their branch. Pull requests show diffs, or differences, of the content from both branches. The changes, additions, and subtractions are shown in different colors.

As soon as you make a commit, you can open a pull request and start a discussion, even before the code is finished.

In this step, you'll open a pull request in your own repository and then merge it yourself. It's a great way to practise the GitHub flow before working on larger projects.

1. Click the **Pull requests** tab of your `hello-world` repository.
2. Click **New pull request**.
3. In the **Example Comparisons** box, select the branch you made, `readme-edits`, to compare with `main` (the original).
4. Look over your changes in the diffs on the Compare page, make sure they're what you want to submit.
    
    ![Screenshot of a diff for the README.md file. 3 red lines list the text that's being removed, and 3 green lines list the text being added.](https://docs.github.com/assets/cb-32937/images/help/repository/diffs.png)
    
5. Click **Create pull request**.
6. Give your pull request a title and write a brief description of your changes. You can include emojis and drag and drop images and gifs.
7. Click **Create pull request**.

#### [Reviewing a pull request](https://docs.github.com/en/get-started/start-your-journey/hello-world#reviewing-a-pull-request)
When you start collaborating with others, this is the time you'd ask for their review. This allows your collaborators to comment on, or propose changes to, your pull request before you merge the changes into the `main` branch.

We won't cover reviewing pull requests in this tutorial, but if you're interested in learning more, see "[About pull request reviews](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews)." Alternatively, try the [GitHub Skills](https://skills.github.com/) "Reviewing pull requests" course.
### [Step 5: Merge your pull request](https://docs.github.com/en/get-started/start-your-journey/hello-world#step-5-merge-your-pull-request)
In this final step, you will merge your `readme-edits` branch into the `main` branch. After you merge your pull request, the changes on your `readme-edits` branch will be incorporated into `main`.

Sometimes, a pull request may introduce changes to code that conflict with the existing code on `main`. If there are any conflicts, GitHub will alert you about the conflicting code and prevent merging until the conflicts are resolved. You can make a commit that resolves the conflicts or use comments in the pull request to discuss the conflicts with your team members.

In this walk-through, you should not have any conflicts, so you are ready to merge your branch into the main branch.

1. At the bottom of the pull request, click **Merge pull request** to merge the changes into `main`.
2. Click **Confirm merge**. You will receive a message that the request was successfully merged and the request was closed.
3. Click **Delete branch**. Now that your pull request is merged and your changes are on `main`, you can safely delete the `readme-edits` branch. If you want to make more changes to your project, you can always create a new branch and repeat this process.
4. Click back to the **Code** tab of your `hello-world` repository to see your published changes on `main`.

### [Conclusion](https://docs.github.com/en/get-started/start-your-journey/hello-world#conclusion)
By completing this tutorial, you've learned to create a project and make a pull request on GitHub.

As part of that, we've learned how to:

- Create a repository.
- Start and manage a new branch.
- Change a file and commit those changes to GitHub.
- Open and merge a pull request.

## Setting up your profile
### [About your profile](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile#about-your-profile)
Your profile page on GitHub is a place where people can find out more about you. You can use your profile to:

- **Share** your interests and skills.
- **Showcase** your projects and contributions.
- **Express** your identity and show the GitHub community who you are.

In this tutorial, you'll learn how to personalize your profile by adding a profile picture, bio, and a profile README.

You'll also learn the basics of Markdown syntax, which is what you'll use to format any writing you do on GitHub.

#### [Prerequisites](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile#prerequisites)

- You must have a GitHub account. For more information, see "[Creating an account on GitHub](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github)."

### [Adding a profile picture and bio](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile#adding-a-profile-picture-and-bio)
First, we'll add a picture to your profile. Your profile picture helps identify you across GitHub.
#### [Adding a profile picture](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile#adding-a-profile-picture)

1. In the upper-right corner of any page, click your existing profile avatar, then, from the dropdown menu, click **Settings**.
2. Under "Profile Picture", select  **Edit**, then click **Upload a photo...**.
    
    ![Screenshot of the "Public profile" section of a user account's settings. A button, labeled with a pencil icon and "Edit", is outlined in dark orange.](https://docs.github.com/assets/cb-72839/images/help/profile/edit-profile-photo.png)
    
3. Select an image, then click **Upload**.
4. Crop your picture.
5. Click **Set new profile picture**.

Next, we'll add some basic information about yourself to share with other GitHub users. This information will display below your profile picture on your profile page.
#### [Adding a bio](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile#adding-a-bio)

1. On your profile page, under your profile picture, click **Edit profile**.
2. Under "Bio", write one or two sentences about yourself, such as who you are and what you do.
    **Note:** Keep the bio short; we'll add a longer description of your interests in your profile README in the section below.
3. To add an emoji to your bio, visit "[Emoji cheat sheet](https://www.webfx.com/tools/emoji-cheat-sheet/)" and copy and paste an emoji into the "Bio" dialog box.
4. Optionally, add your preferred pronouns, workplace, location and timezone, and any links to your personal website and social accounts. Your pronouns will only be visible to users that are signed in to GitHub.
5. Click **Save**.

### [Adding a profile README](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile#adding-a-profile-readme)
Next, we'll create a special repository and README file that will be displayed directly on your profile page.

Your profile README contains information such as your interests, skills, and background, and it can be a great way to introduce yourself to other people on GitHub and showcase your work.

As we learned in the "[Hello World](https://docs.github.com/en/get-started/start-your-journey/hello-world)" tutorial, `README.md` files are written using Markdown syntax (note the `.md` file extension), which is just a way to format plain text.

In the following steps, we'll create and edit your profile README.
#### [Step 1: Create a new repository for your profile README](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile#step-1-create-a-new-repository-for-your-profile-readme)

1. In the upper-right corner of any page, select , then click **New repository**.
    
    ![Screenshot of a GitHub dropdown menu showing options to create new items. The menu item "New repository" is outlined in dark orange.](https://docs.github.com/assets/cb-29762/images/help/repository/repo-create-global-nav-update.png)
    
2. Under "Repository name", type a repository name that matches your GitHub username. For example, if your username is "octocat", the repository name must be "octocat."
3. Optionally, in the "Description" field, type a description of your repository. For example, "My personal repository."
4. Select **Public**.
5. Select **Initialize this repository with a README**.
6. Click **Create repository**.

#### [Step 2: Edit the `README.md` file](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile#step-2-edit-the-readmemd-file)

1. Click the  next to your profile README.
    
    ![Screenshot of @octocat's profile README. A pencil icon is outlined in dark orange.](https://docs.github.com/assets/cb-11970/images/help/profile/edit-profile-readme.png)
    
2. In the "Edit" view, you'll see some pre-populated text to get you started. On line 1, delete the text that says `#### Hi there` and type `## About me`.
    
    - In Markdown syntax, `###` renders the plain text as a small ("third-level") heading, while `##` or `#` renders a second- and first-level heading respectively.
    
3. Toggle to "Preview" to see how the plain text now renders. You should see the new text displayed as a much larger heading.
4. Toggle back to the "Edit" view.
5. Delete line 3 and line 16.
    
    - This HTML syntax (e.g. `<!--`) is keeping the other lines hidden when you toggle to "Preview".
    
6. Complete some of the prompts on lines 8 to 15, and delete any lines you don't want. For example, add your interests, skills, hobbies, or a fun fact about yourself.
7. Now, toggle to "Preview". You should see your completed prompts render as a bulleted list.
8. Toggle back to "Edit" and remove any other lines of text that you don't want displayed on your profile.
9. Keep customizing and editing your profile README.
    
    - Use the "[Emoji cheat sheet](https://www.webfx.com/tools/emoji-cheat-sheet/)" to add emojis.
    - Use the "[Markdown cheat sheet](https://www.markdownguide.org/cheat-sheet/)" to experiment with additional Markdown formatting.

#### [Step 3: Publish your changes to your profile](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile#step-3-publish-your-changes-to-your-profile)

1. When you're happy with how your profile README looks in "Preview", and you're ready to publish it, click **Commit changes..**.
2. In the open dialog box, simply click again **Commit changes**.
3. Navigate back to your profile page. You will see your new profile README displayed on your profile.

### [Next steps](https://docs.github.com/en/get-started/start-your-journey/setting-up-your-profile#next-steps)

- If you want to learn more Markdown syntax and add more sophisticated formatting to your profile README, see "[Quickstart for writing on GitHub](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github)."
- Alternatively, try the [GitHub Skills](https://skills.github.com/) "Communicate using Markdown" course.
- In the next tutorial, "[Finding inspiration on GitHub](https://docs.github.com/en/get-started/start-your-journey/finding-inspiration-on-github)," we'll look at ways you can explore GitHub to find projects and people that interest you.
## Find Inspiration
### [Introduction](https://docs.github.com/en/get-started/start-your-journey/finding-inspiration-on-github#introduction)
GitHub is a vast open-source community. You can explore GitHub to find interesting repositories, topics, code, people, and organizations that can inspire your own work, or support your own learning.

Once you've found something that interests you, you can:

- **Star** the repository or topic, so you can easily find it again later.
- **Follow** people or organizations, so you can stay updated on their activities.
- **Download** useful repositories or code, and customize it for your own use.
- **Contribute** to another user's project, by opening a pull request.

Once you star repositories or follow people, you will see updates on their activities on your [personal dashboard](https://github.com/dashboard).
### [Visit Explore GitHub](https://docs.github.com/en/get-started/start-your-journey/finding-inspiration-on-github#visit-explore-github)

1. Navigate to [Explore GitHub](https://github.com/explore).
2. Browse popular repositories and topics.
3. Click  **Star** next to repositories and topics that interest you, so you can easily find them again later.
4. Navigate to your [stars page](https://github.com/stars) to see all your starred repositories and topics.

### [Search for a topic or project on GitHub](https://docs.github.com/en/get-started/start-your-journey/finding-inspiration-on-github#search-for-a-topic-or-project-on-github)

1. Navigate to [https://github.com/search](https://github.com/search).
2. Type a keyword or query into the search bar. For example, try "how to build a webpage", or "skills-course". For more detailed information on how to search GitHub for specific topics, repositories, or code, see "[About searching on GitHub](https://docs.github.com/en/search-github/getting-started-with-searching-on-github/about-searching-on-github)."
3. Use the left sidebar to filter the results. For example, to browse all repositories in the "skills-course" topic, search "skills-course", then filter by "Topic".
4. Star the repositories that match your interests.

### [Following people and organizations on GitHub](https://docs.github.com/en/get-started/start-your-journey/finding-inspiration-on-github#following-people-and-organizations-on-github)
Following people and organizations on GitHub is another good way to stay updated on projects and topics that interest you.
#### [Following people](https://docs.github.com/en/get-started/start-your-journey/finding-inspiration-on-github#following-people)

1. Navigate to the user's profile page.
2. Under the user's profile picture, click **Follow**.
3. Optionally, to unfollow a user, click **Unfollow**.

#### [Following organizations](https://docs.github.com/en/get-started/start-your-journey/finding-inspiration-on-github#following-organizations)

1. Navigate to the organization page you want to follow.
2. In the top-right corner, click **Follow**.
    
    ![Screenshot of @octo-org's profile page. A button, labeled "Follow", is outlined in dark orange.](https://docs.github.com/assets/cb-32862/images/help/profile/organization-profile-following.png)
    
3. Optionally, to unfollow an organization, click **Unfollow**.
## Download files
### [Introduction](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github#introduction)
GitHub.com is home to millions of open-source software projects, that you can copy, customize, and use for your own purposes.

There are different ways to get a copy of a repository's files on GitHub. You can:

- **Download** a snapshot of a repository's files as a zip file to your own (local) computer.
- **Clone** a repository to your local computer using Git.
- **Fork** a repository to create a new repository on GitHub.

Each of these methods has its own use case, which we'll explain in the next section.

This tutorial focuses on downloading a repository's files to your local computer. For example, if you've found some interesting content in a repository on GitHub, downloading is a simple way to get a copy of the content, without using Git or applying version control.
#### [Understanding the differences between downloading, cloning, and forking](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github#understanding-the-differences-between-downloading-cloning-and-forking)

|Term|Definition|Use case|
|---|---|---|
|Download|To save a snapshot of a repository's files to your local computer.|You want to use or customize the content of the files, but you're not interested in applying version control.|
|Clone|To make a full copy of a repository's data, including all versions of every file and folder.|You want to work on a full copy of the repository on your local computer, using Git to track and manage your changes. You likely intend to sync these locally-made changes with the GitHub-hosted repository. For more information, see "[Cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)."|
|Fork|To create a new repository on GitHub, linked to your personal account, that shares code and visibility settings with the original ("upstream") repository.|You want to use the original repository's data as a basis for your own project on GitHub. Or, you want to use the fork to propose changes to the original ("upstream") repository. After forking the repository, you still might want to clone the repository, so that you can work on the changes on your local computer. For more information, see "[Fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)."|

### [Prerequisites](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github#prerequisites)

- You must have a GitHub account.

### [Downloading a repository's files](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github#downloading-a-repositorys-files)
For the tutorial, we'll use a demo repository ([octocat/Spoon-Knife](https://github.com/octocat/Spoon-Knife)).

1. Navigate to [octocat/Spoon-Knife](https://github.com/octocat/Spoon-Knife).
2. Above the list of files, click  **Code**.
    
    ![Screenshot of the list of files on the landing page of a repository. The "Code" button is highlighted with a dark orange outline.](https://docs.github.com/assets/cb-13128/images/help/repository/code-button.png)
    
3. Click  **Download ZIP**.

### [Conclusion](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github#conclusion)
You now have a copy of the repository's files saved as a zip file on your local computer. You can edit and customize the files for your own purposes.
## Upload a project
### [Introduction](https://docs.github.com/en/get-started/start-your-journey/uploading-a-project-to-github#introduction)
This tutorial will show you how to upload a group of files to a GitHub repository.

Uploading your files to a GitHub repository lets you:

- **Apply version control** when you make edits to the files, so your project's history is protected and manageable.
- **Back up** your work, because your files are now stored in the cloud.
- **Pin** the repository to your personal profile, so that others can see your work.
- **Share** and discuss your work with others, either publicly or privately.

If you're already familiar with Git, and you're looking for information on how to upload a locally-stored Git repository to GitHub, see "[Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github#adding-a-local-repository-to-github-using-git)."
### [Prerequisites](https://docs.github.com/en/get-started/start-your-journey/uploading-a-project-to-github#prerequisites)

- You must have a GitHub account. For more information, see "[Creating an account on GitHub](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github)."
- You should have a group of files you'd like to upload.

### [Step 1: Create a new repository for your project](https://docs.github.com/en/get-started/start-your-journey/uploading-a-project-to-github#step-1-create-a-new-repository-for-your-project)
It's a good idea to create a new repository for each individual project you're working on. If you're writing a software project, grouping all the related files in a new repository makes it easier to maintain and manage the codebase over time.

1. In the upper-right corner of any page, select , then click **New repository**.
    
    ![Screenshot of a GitHub dropdown menu showing options to create new items. The menu item "New repository" is outlined in dark orange.](https://docs.github.com/assets/cb-29762/images/help/repository/repo-create-global-nav-update.png)
    
2. In the "Repository name" box, type a name for your project. For example, type "my-first-project."
3. In the "Description" box, type a short description. For example, type "This is my first project on GitHub."
4. Select whether your repository will be **Public** or **Private**. Select "Public" if you want others to be able to see your project.
5. Select **Add a README file**. You will edit this file in a later step.
6. Click **Create repository**.

### [Step 2: Upload files to your project's repository](https://docs.github.com/en/get-started/start-your-journey/uploading-a-project-to-github#step-2-upload-files-to-your-projects-repository)
So far, you should only see one file listed in the repository, the `README.md` file you created when you initialized the repository. Now, we'll upload some of your own files.

1. To the right of the page, select the **Add file** dropdown menu.
2. From the dropdown menu, click **Upload files**.
3. On your computer, open the folder containing your work, then drag and drop all files and folders into the browser.
4. At the bottom of the page, under "Commit changes", select "Commit directly to the `main` branch, then click **Commit changes**.

### [Step 3: Edit the README file for your project's repository](https://docs.github.com/en/get-started/start-your-journey/uploading-a-project-to-github#step-3-edit-the-readme-file-for-your-projects-repository)
Your repository's README file is typically the first item someone will see when visiting your repository. It usually contains information on what your project is about and why your project is useful.

As we learned in the "[Hello World](https://docs.github.com/en/get-started/start-your-journey/hello-world)" tutorial, the README file (`README.md`) is written in Markdown syntax. Markdown is an easy-to-read, easy-to-write language for formatting plain text.

In this step, we'll edit your project's `README.md` using Markdown so that it includes some basic information about your project.

1. From the list of files, click `README.md` to view the file.
2. In the upper right corner of the file view, click  to open the file editor.
    
    - You will see that some information about your project has been pre-filled for you. For example, you should see the repository name and repository description you completed in Step 1 displayed on line 1 and line 2.
3. Delete the existing text apart from `#`, then type a proper title for your project.
    
    - Example: `## About my first project on GitHub`.
4. Next, add some information about your project, such as a description of the project's purpose or its main features.
    
    **Note:** If you're not sure what to write, take a look at other repositories on GitHub to see how other people describe their projects.
    
    To apply more sophisticated formatting, such as adding images, links, and footnotes, see "[Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)."
    
5. Above the new content, click **Preview**.
    
    ![Screenshot of a file in edit mode. Above the file's contents, a tab labeled "Preview" is outlined in dark orange.](https://docs.github.com/assets/cb-35434/images/help/repository/edit-readme-preview-changes.png)
    
6. Take a look at how the file will render once we save our changes, then toggle back to "Edit".
7. Continue to edit and preview the text until you're happy with the content of your README.
8. In the top right, click **Commit changes**.
9. In the dialog box that opens, a commit message has been pre-filled for you ("Update README.md") and, by default, the option to "Commit directly to the `main` branch" has been selected. Leave these options as they are and go ahead and click **Commit changes**.

### [Conclusion](https://docs.github.com/en/get-started/start-your-journey/uploading-a-project-to-github#conclusion)
You have now created a new repository, uploaded some files to it, and added a project README.

If you set your repository visibility to "Public," the repository will be displayed on your personal profile and you can share the URL of your repository with others.

As you add, edit or delete files directly in the browser on GitHub, GitHub will track these changes ("commits"), so you can start to manage your project's history and evolution.

When making changes, remember that you can create a new branch from the `main` branch of your repository, so that you can experiment without affecting the main copy of files. Then, when you're happy with a set of a changes, open a pull request to merge the changes into your `main` branch. For a reminder of how to do this, see "[Hello World](https://docs.github.com/en/get-started/start-your-journey/hello-world)."
## Learning resources
### [Using GitHub](https://docs.github.com/en/get-started/start-your-journey/git-and-github-learning-resources#using-github)
Become better acquainted with GitHub through our "[Using GitHub](https://docs.github.com/en/get-started/using-github)" articles:

- To review the fundamentals of a GitHub workflow, see "[GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow)."
- To learn about the various tools for working with repositories hosted on GitHub, and how to choose a tool that best suits your needs, see "[Connecting to GitHub](https://docs.github.com/en/get-started/using-github/connecting-to-github)."
- To understand the different communication tools on GitHub, such as GitHub Issues, GitHub Discussions, and pull requests, see "[Communicating on GitHub](https://docs.github.com/en/get-started/using-github/communicating-on-github)."

### [Using Git](https://docs.github.com/en/get-started/start-your-journey/git-and-github-learning-resources#using-git)
Familiarize yourself with Git through our series of articles:

- "[Getting started with Git](https://docs.github.com/en/get-started/getting-started-with-git)."
- "[Using Git](https://docs.github.com/en/get-started/using-git)."

There are also lots of other online reading resources to help you learn Git:

- [Official Git project site](https://git-scm.com/).
- [ProGit book](http://git-scm.com/book).
- [Git command list](https://git-scm.com/docs).

### [Online courses](https://docs.github.com/en/get-started/start-your-journey/git-and-github-learning-resources#online-courses)

- GitHub Skills offers free interactive courses that are built into GitHub with instant automated feedback and help. Learn to open your first pull request, make your first open source contribution, create a GitHub Pages site, and more. For more information about course offerings, see [GitHub Skills](https://skills.github.com/).
- [Git branching](http://learngitbranching.js.org/) is a free interactive tool for learning and practising Git concepts.
- An interactive [online Git course](https://www.pluralsight.com/courses/code-school-git-real) from [Pluralsight](https://www.pluralsight.com/codeschool) can also teach you the basics of Git.

### [Training](https://docs.github.com/en/get-started/start-your-journey/git-and-github-learning-resources#training)

#### [GitHub's web-based educational programs](https://docs.github.com/en/get-started/start-your-journey/git-and-github-learning-resources#githubs-web-based-educational-programs)
GitHub offers live [trainings](https://services.github.com/#upcoming-events) with a hands-on, project-based approach for those who love the command line and those who don't.
#### [Training for your company](https://docs.github.com/en/get-started/start-your-journey/git-and-github-learning-resources#training-for-your-company)
GitHub offers [in-person classes](https://services.github.com/#offerings) taught by our highly-experienced educators. [Contact us](https://services.github.com/#contact) to ask your training-related questions.
### [Community](https://docs.github.com/en/get-started/start-your-journey/git-and-github-learning-resources#community)
You can connect with developers around the world to ask and answer questions, learn, and interact directly with GitHub staff. To get the conversation started, see "[GitHub Community Support](https://github.com/orgs/community/discussions/)."