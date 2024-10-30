# Get Started
# Start your journey
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
# On boarding
### Getting started with your Github account
#### [Part 1: Configuring your GitHub account](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#part-1-configuring-your-github-account)
The first steps in starting with GitHub are to create an account, choose a product that fits your needs best, verify your email, set up two-factor authentication, and view your profile.

There are several types of accounts on GitHub. Every person who uses GitHub has their own personal account, which can be part of multiple organizations and teams. Your personal account is your identity on GitHub.com and represents you as an individual.
##### [1. Creating an account](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#1-creating-an-account)
To sign up for an account, navigate to [https://github.com/](https://github.com/) and follow the prompts.

To keep your GitHub account secure you should use a strong and unique password. For more information, see "[Creating a strong password](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-strong-password)."
##### [2. Choosing your GitHub product](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#2-choosing-your-github-product)
You can choose GitHub Free or GitHub Pro to get access to different features for your personal account. You can upgrade at any time if you are unsure at first which product you want.

For more information on all of GitHub's plans, see "[GitHub’s plans](https://docs.github.com/en/get-started/learning-about-github/githubs-plans)."
##### [3. Verifying your email address](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#3-verifying-your-email-address)
To ensure you can use all the features in your GitHub plan, verify your email address after signing up for a new account. For more information, see "[Verifying your email address](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/verifying-your-email-address)."
##### [4. Configuring two-factor authentication](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#4-configuring-two-factor-authentication)
Two-factor authentication, or 2FA, is an extra layer of security used when logging into websites or apps. We strongly urge you to configure 2FA for the safety of your account. For more information, see "[About two-factor authentication](https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa/about-two-factor-authentication)."

Optionally, after you have configured 2FA, add a passkey to your account to enable a secure, passwordless login. See "[Managing your passkeys](https://docs.github.com/en/authentication/authenticating-with-a-passkey/managing-your-passkeys)."
##### [5. Viewing your GitHub profile and contribution graph](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#5-viewing-your-github-profile-and-contribution-graph)
Your GitHub profile tells people the story of your work through the repositories and gists you've pinned, the organization memberships you've chosen to publicize, the contributions you've made, and the projects you've created. For more information, see "[About your profile](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/about-your-profile)" and "[Viewing contributions on your profile](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/managing-contribution-settings-on-your-profile/viewing-contributions-on-your-profile)."
#### [Part 2: Using GitHub's tools and processes](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#part-2-using-githubs-tools-and-processes)
To best use GitHub, you'll need to set up Git. Git is responsible for everything GitHub-related that happens locally on your computer. To effectively collaborate on GitHub, you'll write in issues and pull requests using GitHub Flavored Markdown.
##### [1. Learning Git](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#1-learning-git)
GitHub's collaborative approach to development depends on publishing commits from your local repository to GitHub for other people to view, fetch, and update using Git. For more information about Git, see the "[Git Handbook](https://guides.github.com/introduction/git-handbook/)" guide. For more information about how Git is used on GitHub, see "[GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow)."
##### [2. Setting up Git](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#2-setting-up-git)
If you plan to use Git locally on your computer, whether through the command line, an IDE or text editor, you will need to install and set up Git. For more information, see "[Set up Git](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git)."

If you prefer to use a visual interface, you can download and use GitHub Desktop. GitHub Desktop comes packaged with Git, so there is no need to install Git separately. For more information, see "[Getting started with GitHub Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop)."

Once you install Git, you can connect to GitHub repositories from your local computer, whether your own repository or another user's fork. When you connect to a repository on GitHub.com from Git, you'll need to authenticate with GitHub using either HTTPS or SSH. For more information, see "[About remote repositories](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories)."
##### [3. Choosing how to interact with GitHub](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#3-choosing-how-to-interact-with-github)
Everyone has their own unique workflow for interacting with GitHub; the interfaces and methods you use depend on your preference and what works best for your needs.

For more information about the different approaches for interacting with GitHub, and a comparison of the tools you can use, see "[Connecting to GitHub](https://docs.github.com/en/get-started/using-github/connecting-to-github)."
##### [4. Writing on GitHub](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#4-writing-on-github)
To make your communication clear and organized in issues and pull requests, you can use GitHub Flavored Markdown for formatting, which combines an easy-to-read, easy-to-write syntax with some custom functionality. For more information, see "[About writing and formatting on GitHub](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/about-writing-and-formatting-on-github)."

You can learn GitHub Flavored Markdown with the "[Communicate using Markdown](https://github.com/skills/communicate-using-markdown)" course on GitHub Skills.
##### [5. Searching on GitHub](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#5-searching-on-github)
Our integrated search allows you to find what you are looking for among the many repositories, users and lines of code on GitHub. You can search globally across all of GitHub or limit your search to a particular repository or organization. For more information about the types of searches you can do on GitHub, see "[About searching on GitHub](https://docs.github.com/en/search-github/getting-started-with-searching-on-github/about-searching-on-github)."

Our search syntax allows you to construct queries using qualifiers to specify what you want to search for. For more information on the search syntax to use in search, see "[Searching on GitHub](https://docs.github.com/en/search-github/searching-on-github)."
##### [6. Managing files on GitHub](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#6-managing-files-on-github)
With GitHub, you can create, edit, move and delete files in your repository or any repository you have write access to. You can also track the history of changes in a file line by line. For more information, see "[Managing files](https://docs.github.com/en/repositories/working-with-files/managing-files)."
#### [Part 3: Collaborating on GitHub](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#part-3-collaborating-on-github)
Any number of people can work together in repositories across GitHub. You can configure settings, create projects, and manage your notifications to encourage effective collaboration.
##### [1. Working with repositories](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#1-working-with-repositories)
###### [Creating a repository](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#creating-a-repository)
A repository is like a folder for your project. You can have any number of public and private repositories in your personal account. Repositories can contain folders and files, images, videos, spreadsheets, and data sets, as well as the revision history for all files in the repository. For more information, see "[About repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories)."

When you create a new repository, you should initialize the repository with a README file to let people know about your project. For more information, see "[Creating a new repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)."
###### [Cloning a repository](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#cloning-a-repository)
You can clone an existing repository from GitHub to your local computer, making it easier to add or remove files, fix merge conflicts, or make complex commits. Cloning a repository pulls down a full copy of all the repository data that GitHub has at that point in time, including all versions of every file and folder for the project. For more information, see "[Cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)."
###### [Forking a repository](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#forking-a-repository)
A fork is a copy of a repository that you manage, where any changes you make will not affect the original repository unless you submit a pull request to the project owner. Most commonly, forks are used to either propose changes to someone else's project or to use someone else's project as a starting point for your own idea. For more information, see "[Working with forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks)."
##### [2. Importing your projects](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#2-importing-your-projects)
If you have existing projects you'd like to move over to GitHub you can import projects using the GitHub Importer, the command line, or external migration tools. For more information, see "[Importing source code](https://docs.github.com/en/migrations/importing-source-code)."
##### [3. Managing collaborators and permissions](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#3-managing-collaborators-and-permissions)
You can collaborate on your project with others using your repository's issues, pull requests, and projects. You can invite other people to your repository as collaborators from the **Collaborators** tab in the repository settings. For more information, see "[Inviting collaborators to a personal repository](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-access-to-your-personal-repositories/inviting-collaborators-to-a-personal-repository)."

You are the owner of any repository you create in your personal account and have full control of the repository. Collaborators have write access to your repository, limiting what they have permission to do. For more information, see "[Permission levels for a personal account repository](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-user-account-settings/permission-levels-for-a-personal-account-repository)."
##### [4. Managing repository settings](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#4-managing-repository-settings)
As the owner of a repository you can configure several settings, including the repository's visibility, topics, and social media preview. For more information, see "[Managing your repository’s settings and features](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features)."
##### [5. Setting up your project for healthy contributions](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#5-setting-up-your-project-for-healthy-contributions)
To encourage collaborators in your repository, you need a community that encourages people to use, contribute to, and evangelize your project. For more information, see "[Building Welcoming Communities](https://opensource.guide/building-community/)" in the Open Source Guides.

By adding files like contributing guidelines, a code of conduct, and a license to your repository you can create an environment where it's easier for collaborators to make meaningful, useful contributions. For more information, see "[Setting up your project for healthy contributions](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions)."
##### [6. Using GitHub Issues and Projects](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#6-using-github-issues-and-projects)
You can use GitHub Issues to organize your work with issues and pull requests and manage your workflow with Projects. For more information, see "[About issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues)" and "[About Projects](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)."
##### [7. Managing notifications](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#7-managing-notifications)
Notifications provide updates about the activity on GitHub you've subscribed to or participated in. If you're no longer interested in a conversation, you can unsubscribe, unwatch, or customize the types of notifications you'll receive in the future. For more information, see "[About notifications](https://docs.github.com/en/account-and-profile/managing-subscriptions-and-notifications-on-github/setting-up-notifications/about-notifications)."
##### [8. Working with GitHub Pages](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#8-working-with-github-pages)
You can use GitHub Pages to create and host a website directly from a repository on GitHub.com. For more information, see "[About GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages)."
##### [9. Using GitHub Discussions](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#9-using-github-discussions)
You can enable GitHub Discussions for your repository to help build a community around your project. Maintainers, contributors and visitors can use discussions to share announcements, ask and answer questions, and participate in conversations around goals. For more information, see "[About discussions](https://docs.github.com/en/discussions/collaborating-with-your-community-using-discussions/about-discussions)."
#### [Part 4: Customizing and automating your work on GitHub](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#part-4-customizing-and-automating-your-work-on-github)
You can use tools from the GitHub Marketplace, the GitHub API, and existing GitHub features to customize and automate your work.
##### [1. Using GitHub Marketplace](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#1-using-github-marketplace)
GitHub Marketplace contains integrations that add functionality and improve your workflow. You can discover, browse, and install free and paid tools, including GitHub Apps, OAuth apps, and GitHub Actions, in [GitHub Marketplace](https://github.com/marketplace).
##### [2. Using the GitHub API](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#2-using-the-github-api)
There are two versions of the GitHub API: the REST API and the GraphQL API. You can use the GitHub APIs to automate common tasks, [back up your data](https://docs.github.com/en/repositories/archiving-a-github-repository/backing-up-a-repository), or [create integrations](https://docs.github.com/en/get-started/exploring-integrations/about-integrations) that extend GitHub. For more information, see "[Comparing GitHub's REST API and GraphQL API](https://docs.github.com/en/rest/overview/about-githubs-apis)."
##### [3. Building GitHub Actions](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#3-building-github-actions)
With GitHub Actions, you can automate and customize GitHub.com's development workflow on GitHub. You can create your own actions, and use and customize actions shared by the GitHub community. For more information, see "[Writing workflows](https://docs.github.com/en/actions/learn-github-actions)."
##### [4. Publishing and managing GitHub Packages](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#4-publishing-and-managing-github-packages)
GitHub Packages is a software package hosting service that allows you to host your software packages privately or publicly and use packages as dependencies in your projects. For more information, see "[Introduction to GitHub Packages](https://docs.github.com/en/packages/learn-github-packages/introduction-to-github-packages)."
#### [Part 5: Building securely on GitHub](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#part-5-building-securely-on-github)
GitHub has a variety of security features that help keep code and secrets secure in repositories. Some features are available for all repositories, while others are only available for public repositories and repositories with a GitHub Advanced Security license. For an overview of GitHub security features, see "[GitHub security features](https://docs.github.com/en/code-security/getting-started/github-security-features)."
##### [1. Securing your repository](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#1-securing-your-repository)
As a repository administrator, you can secure your repositories by configuring repository security settings. These include managing access to your repository, setting a security policy, and managing dependencies. For public repositories, and for private repositories owned by organizations where GitHub Advanced Security is enabled, you can also configure code and secret scanning to automatically identify vulnerabilities and ensure tokens and keys are not exposed.

For more information on steps you can take to secure your repositories, see "[Quickstart for securing your repository](https://docs.github.com/en/code-security/getting-started/quickstart-for-securing-your-repository)."
##### [2. Managing your dependencies](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#2-managing-your-dependencies)
A large part of building securely is maintaining your project's dependencies to ensure that all packages and applications you depend on are updated and secure. You can manage your repository's dependencies on GitHub by exploring the dependency graph for your repository, using Dependabot to automatically raise pull requests to keep your dependencies up-to-date, and receiving Dependabot alerts and security updates for vulnerable dependencies.

For more information, see "[Securing your software supply chain](https://docs.github.com/en/code-security/supply-chain-security)."
#### [Part 6: Participating in GitHub's community](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#part-6-participating-in-githubs-community)
There are many ways to participate in the GitHub community. You can contribute to open source projects, interact with people in the GitHub Community Support, or learn with GitHub Skills.
##### [1. Contributing to open source projects](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#1-contributing-to-open-source-projects)
Contributing to open source projects on GitHub can be a rewarding way to learn, teach, and build experience in just about any skill you can imagine. For more information, see "[How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)" in the Open Source Guides.

You can find personalized recommendations for projects and good first issues based on your past contributions, stars, and other activities in [Explore GitHub](https://github.com/explore). For more information, see "[Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)."
##### [2. Interacting with GitHub Community Support](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#2-interacting-with-github-community-support)
You can connect with developers around the world to ask and answer questions, learn, and interact directly with GitHub staff. To get the conversation started, see "[GitHub Community Support](https://github.com/orgs/community/discussions/)."
##### [3. Reading about GitHub on GitHub Docs](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#3-reading-about-github-on-github-docs)
You can read documentation that reflects the features available to you on GitHub. For more information, see "[About versions of GitHub Docs](https://docs.github.com/en/get-started/learning-about-github/about-versions-of-github-docs)."
##### [4. Learning with GitHub Skills](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#4-learning-with-github-skills)
You can learn new skills by completing fun, realistic projects in your very own GitHub repository with [GitHub Skills](https://skills.github.com/). Each course is a hands-on lesson created by the GitHub community and taught by a friendly bot.

For more information, see "[Git and GitHub learning resources](https://docs.github.com/en/get-started/start-your-journey/git-and-github-learning-resources)."
##### [5. Supporting the open source community](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#5-supporting-the-open-source-community)
GitHub Sponsors allows you to make a monthly recurring payment to a developer or organization who designs, creates, or maintains open source projects you depend on. For more information, see "[About GitHub Sponsors](https://docs.github.com/en/sponsors/getting-started-with-github-sponsors/about-github-sponsors)."
##### [6. Contacting GitHub Support](https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account#6-contacting-github-support)
GitHub Support can help you troubleshoot issues you run into while using GitHub. For more information, see "[About GitHub Support](https://docs.github.com/en/support/learning-about-github-support/about-github-support)."
# Using Github
## Github flow
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