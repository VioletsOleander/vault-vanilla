# The Basics
## What is a container?
### Explanation
Imagine you're developing a killer web app that has three main components - a React frontend, a Python API, and a PostgreSQL database. If you wanted to work on this project, you'd have to install Node, Python, and PostgreSQL.

How do you make sure you have the same versions as the other developers on your team? Or your CI/CD system? Or what's used in production?

How do you ensure the version of Python (or Node or the database) your app needs isn't affected by what's already on your machine? How do you manage potential conflicts?

Enter containers!

What is a container? Simply put, containers are isolated processes for each of your app's components. Each component - the frontend React app, the Python API engine, and the database - runs in its own isolated environment, completely isolated from everything else on your machine.
> 容器是运行于主机上的一个沙盒进程，与其他进程隔离
> 应用程序的各个组件运行于各自的隔离环境中

Here's what makes them awesome. Containers are:

- Self-contained. Each container has everything it needs to function with no reliance on any pre-installed dependencies on the host machine.
- Isolated. Since containers are run in isolation, they have minimal influence on the host and other containers, increasing the security of your applications.
- Independent. Each container is independently managed. Deleting one container won't affect any others.
- Portable. Containers can run anywhere! The container that runs on your development machine will work the same way in a data center or anywhere in the cloud!

>- 容器是自洽的，容器自身拥有需要运行的所有依赖，不需要宿主机上安装依赖
>- 容器是隔离的，容器对于主机和其他容器有最小的映像，故提高了应用的安全性
>- 容器是独立的，容器各自独立管理，删除一个容器不影响其他
>- 容器是可移植的，容器可以运行于任何地方

#### Containers versus virtual machines (VMs)
Without getting too deep, a VM is an entire operating system with its own kernel, hardware drivers, programs, and applications. Spinning up a VM only to isolate a single application is a lot of overhead.

A container is simply an isolated process with all of the files it needs to run. If you run multiple containers, they all share the same kernel, allowing you to run more applications on less infrastructure.
> 虚拟机是完整的 OS，有自己的内核、硬盘驱动、程序、应用
> 容器是一个隔离的进程，带有它运行所需要的文件
> 运行多个容器时，它们共享相同的内核

> **Using VMs and containers together**
> 
> Quite often, you will see containers and VMs used together. As an example, in a cloud environment, the provisioned machines are typically VMs. However, instead of provisioning one machine to run one application, a VM with a container runtime can run multiple containerized applications, increasing resource utilization and reducing costs.

## What is an image?
### Explanation
Seeing a [container](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/) is an isolated process, where does it get its files and configuration? How do you share those environments?

That's where container images come in. A container image is a standardized package that includes all of the files, binaries, libraries, and configurations to run a container.
> 容器映像是一个标准化的包，它包含了运行容器所需要的所有文件、二进制文件、库和配置

For a [PostgreSQL](https://hub.docker.com/_/postgres) image, that image will package the database binaries, config files, and other dependencies. For a Python web app, it'll include the Python runtime, your app code, and all of its dependencies.
> 例如 PostgreSQL 映像会打包数据库二进制文件和配置文件以及一些依赖，Python 应用的映像会打包 Python 运行时、应用代码以及其依赖

There are two important principles of images:

1. Images are immutable. Once an image is created, it can't be modified. You can only make a new image or add changes on top of it.
2. Container images are composed of layers. Each layer represents a set of file system changes that add, remove, or modify files.

> 映像的两个规则：
> - 映像是不可变的，映像被创建后就不可修改，需要对其修改时，需要创建新的映像
> - 映像由层组成，每一层都表示一组文件系统修改，包括添加、移除、修改文件等

These two principles let you to extend or add to existing images. For example, if you are building a Python app, you can start from the [Python image](https://hub.docker.com/_/python) and add additional layers to install your app's dependencies and add your code. This lets you focus on your app, rather than Python itself.
> 映像的这两个规则让我们专注于对现有的映像进行拓展
> 例如，构建 Python 应用时，我们从 Python 映像开始，添加用于安装应用的依赖的层和安装应用代码的层，专注于应用而不是 Python 本身

#### Finding images
[Docker Hub](https://hub.docker.com/) is the default global marketplace for storing and distributing images. It has over 100,000 images created by developers that you can run locally. You can search for Docker Hub images and run them directly from Docker Desktop.

Docker Hub provides a variety of Docker-supported and endorsed images known as Docker Trusted Content. These provide fully managed services or great starters for your own images. These include:

- [Docker Official Images](https://hub.docker.com/search?q=&type=image&image_filter=official) - a curated set of Docker repositories, serve as the starting point for the majority of users, and are some of the most secure on Docker Hub
- [Docker Verified Publishers](https://hub.docker.com/search?q=&image_filter=store) - high-quality images from commercial publishers verified by Docker
- [Docker-Sponsored Open Source](https://hub.docker.com/search?q=&image_filter=open_source) - images published and maintained by open-source projects sponsored by Docker through Docker's open source program

> Docker Hub 中 Docker 支持且认同的映像称为 Docker Trusted Content，包括了
> - Docker 官方映像
> - Docker 验证的发布者的映像
> - Docker 赞助的开源映像

For example, [Redis](https://hub.docker.com/_/redis) and [Memcached](https://hub.docker.com/_/memcached) are a few popular ready-to-go Docker Official Images. You can download these images and have these services up and running in a matter of seconds. There are also base images, like the [Node.js](https://hub.docker.com/_/node) Docker image, that you can use as a starting point and add your own files and configurations.

## What is a registry?
### Explanation
Now that you know what a container image is and how it works, you might wonder - where do you store these images?

Well, you can store your container images on your computer system, but what if you want to share them with your friends or use them on another machine? That's where the image registry comes in.

An image registry is a centralized location for storing and sharing your container images. It can be either public or private. [Docker Hub](https://hub.docker.com/) is a public registry that anyone can use and is the default registry.
> 映像注册表就是存储和共享映像的中心化位置
> Docker Hub 就是最大的公有映像注册表

While Docker Hub is a popular option, there are many other available container registries available today, including [Amazon Elastic Container Registry(ECR)](https://aws.amazon.com/ecr/), [Azure Container Registry (ACR)](https://azure.microsoft.com/en-in/products/container-registry), and [Google Container Registry (GCR)](https://cloud.google.com/artifact-registry). You can even run your private registry on your local system or inside your organization. For example, Harbor, JFrog Artifactory, GitLab Container registry etc.

#### Registry vs. repository
While you're working with registries, you might hear the terms _registry_ and _repository_ as if they're interchangeable. Even though they're related, they're not quite the same thing.

A _registry_ is a centralized location that stores and manages container images, whereas a _repository_ is a collection of related container images within a registry. Think of it as a folder where you organize your images based on projects. Each repository contains one or more container images.
> 注册表是存储大量映像的中心位置
> 仓库是注册表中一组相关映像的集合，例如我们根据项目来将相关的映像组织到同一个目录下

> **Note**
> 
> You can create one private repository and unlimited public repositories using the free version of Docker Hub. For more information, visit the [Docker Hub subscription page](https://www.docker.com/pricing/).

## What is Docker Compose?
### Explanation
If you've been following the guides so far, you've been working with single container applications. But, now you're wanting to do something more complicated - run databases, message queues, caches, or a variety of other services. Do you install everything in a single container? Run multiple containers? If you run multiple, how do you connect them all together?

One best practice for containers is that each container should do one thing and do it well. While there are exceptions to this rule, avoid the tendency to have one container do multiple things.
> 容器的一个最佳实践是保持每个容器仅做一件事

You can use multiple `docker run` commands to start multiple containers. But, you'll soon realize you'll need to manage networks, all of the flags needed to connect containers to those networks, and more. And when you're done, cleanup is a little more complicated.

With Docker Compose, you can define all of your containers and their configurations in a single YAML file. If you include this file in your code repository, anyone that clones your repository can get up and running with a single command.
> Docker Compose 使用单个 yaml 文件定义和我们应用相关的所有容器各自的配置
> 在仓库中添加该 yaml 文件后，我们可以仅用单个命令启动应用 (启动所有的容器)

It's important to understand that Compose is a declarative tool - you simply define it and go. You don't always need to recreate everything from scratch. If you make a change, run `docker compose up` again and Compose will reconcile the changes in your file and apply them intelligently.
> Docker Compose 是声明式工具，仅提供定义
> `docker compose up` 用于让 Compose 识别配置文件中的修改并重新应用它们

> **Dockerfile versus Compose file**
> 
> A Dockerfile provides instructions to build a container image while a Compose file defines your running containers. Quite often, a Compose file references a Dockerfile to build an image to use for a particular service.

> Dockerfile 用于提供构建映像所需要的指令
> Composefile 用于定义一组容器是如何运行的
> Composefile 往往会引用 Dockerfile 来构建特定映像并使用它

