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

# Building Images
## Understanding the image layers
### Explanation
As you learned in [What is an image?](https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-an-image/), container images are composed of layers. And each of these layers, once created, are immutable. But, what does that actually mean? And how are those layers used to create the filesystem a container can use?
> 我们知道容器映像由层组成，这些层一旦被创建就是不可变的

#### Image layers
Each layer in an image contains a set of filesystem changes - additions, deletions, or modifications. 
> 映像中的每一层都包含一组文件系统更改 - 添加、删除或修改

Let’s look at a theoretical image:

1. The first layer adds basic commands and a package manager, such as apt.
2. The second layer installs a Python runtime and pip for dependency management.
3. The third layer copies in an application’s specific requirements.txt file.
4. The fourth layer installs that application’s specific dependencies.
5. The fifth layer copies in the actual source code of the application.

> 考虑一个映像：
> 第一层添加基本的命令和包管理器，例如 apt
> 第二层安装 Python 运行时和 pip 用于依赖管理
> 第三层将应用的 `requirements.txt` 文件拷贝进来
> 第四层安装应用特定的依赖
> 第五层将应用的源代码拷贝进来

This example might look like:

![screenshot of the flowchart showing the concept of the image layers](https://docs.docker.com/get-started/docker-concepts/building-images/images/container_image_layers.webp)

This is beneficial because it allows layers to be reused between images. For example, imagine you wanted to create another Python application. Due to layering, you can leverage the same Python base. This will make builds faster and reduce the amount of storage and bandwidth required to distribute the images. The image layering might look similar to the following:
> 这样做的好处在于映像之间可以复用部分层
> 例如，需要创建另一个 Python 程序时，我们只需要在第三层之后修改即可，复用了同样的 Python 基础环境
> 这使得构建更快，并且减少了分发映像所需要的存储量和带宽
> 例如：

![screenshot of the flowchart showing the benefits of the image layering](https://docs.docker.com/get-started/docker-concepts/building-images/images/container_image_layer_reuse.webp)

Layers let you extend images of others by reusing their base layers, allowing you to add only the data that your application needs.
> 我们通过复用其他映像的基础层来拓展出自己的映像，因此仅需要添加我们的应用所需要的数据

#### Stacking the layers
Layering is made possible by content-addressable storage and union filesystems. While this will get technical, here’s how it works:

1. After each layer is downloaded, it is extracted into its own directory on the host filesystem.
2. When you run a container from an image, a union filesystem is created where layers are stacked on top of each other, creating a new and unified view.
3. When the container starts, its root directory is set to the location of this unified directory, using `chroot`.

> 层是通过内容可寻址的存储和联合文件系统实现的，其工作机理为：
> 1. 在每一层被下载后，该层会被提取到宿主机文件系统中的独立目录中
> 2. 从一个映像运行容器时，一个联合的文件系统会被创建，在该文件系统中，一个个层堆叠起来，构建一个新的统一视图
> 3. 容器启动时，其根目录就通过 `chroot`  被设定为该统一目录的位置

> [! 内容可寻址存储]
> 内容可寻址存储允许根据存储内容的哈希值来引用文件

> [! 联合文件系统]
> 联合文件系统允许将多个目录合并为一个单一的视图，使得容器的文件系统看起来像一个整体，实际上仍由多个分层 (独立目录) 组成

> [! `chroot`]
> `chroot` 命令将进程的根目录更改到指定目录，则在该进程看来，指定的目录就是系统的根目录，使得进程认为自己运行在一个独立的环境中

When the union filesystem is created, in addition to the image layers, a directory is created specifically for the running container. This allows the container to make filesystem changes while allowing the original image layers to remain untouched. This enables you to run multiple containers from the same underlying image.
> 当联合文件系统被创建时，除了映像的层以外，Docker 会为运行的容器专门创建一个目录
> 这使得容器可以做出文件系统修改的同时保持原来的映像层不变，因此可以从相同的映像中运行多个容器

## Writing a Dockerfile
### Explanation
A Dockerfile is a text-based document that's used to create a container image. It provides instructions to the image builder on the commands to run, files to copy, startup command, and more.
> Dockerfile 为纯文本，用于创建容器映像
> Dockerfile 为映像构建程序提供指令，指示应该运行的命令、应该拷贝的文件、启动命令等

As an example, the following Dockerfile would produce a ready-to-run Python application:

```dockerfile
FROM python:3.12
WORKDIR /usr/local/app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY src ./src
EXPOSE 5000

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Common instructions
Some of the most common instructions in a `Dockerfile` include:

- `FROM <image>` - this specifies the base image that the build will extend.
- `WORKDIR <path>` - this instruction specifies the "working directory" or the path in the image where files will be copied and commands will be executed.
- `COPY <host-path> <image-path>` - this instruction tells the builder to copy files from the host and put them into the container image.
- `RUN <command>` - this instruction tells the builder to run the specified command.
- `ENV <name> <value>` - this instruction sets an environment variable that a running container will use.
- `EXPOSE <port-number>` - this instruction sets configuration on the image that indicates a port the image would like to expose.
- `USER <user-or-uid>` - this instruction sets the default user for all subsequent instructions.
- `CMD ["<command>", "<arg1>"]` - this instruction sets the default command a container using this image will run.

> Dockerfile 中常用的指令包括：
> `FROM <image>` 指定该映像所基于的基础映像
> `WORKDIR <path>` 指定映像中工作目录的路径，即文件将被拷贝到的目录和命令执行所在的目录
> `COPY <host-path> <image-path>` 告诉映像构建程序将主机中的文件拷贝到映像的指定目录中
> `RUN <command>` 告诉映像构建程序需要运行的命令
> `ENV <name> <value>` 为运行容器设定环境变量
> `EXPOSE <port-number>` 设定映像应该暴露的接口
> `USER <user-or-uid>` 设定之后所有命令的默认用户
> `CMD ["<command>", "<arg1>"]` 设定映像启动容器时默认运行的命令

To read through all of the instructions or go into greater detail, check out the [Dockerfile reference](https://docs.docker.com/engine/reference/builder/).

A Dockerfile typically follows these steps:

1. Determine your base image
2. Install application dependencies
3. Copy in any relevant source code and/or binaries
4. Configure the final image

> Dockerfile 一般遵循以下步骤：
> 1. 决定基础映像
> 2. 安装应用依赖
> 3. 将相关的源码、二进制文件拷贝进来
> 4. 配置最终映像

## Build, tag, and publish an image
### Explanation
In this guide, you will learn the following:

- Building images - the process of building an image based on a `Dockerfile`
- Tagging images - the process of giving an image a name, which also determines where the image can be distributed
- Publishing images - the process to distribute or share the newly created image using a container registry

#### Building images
Most often, images are built using a Dockerfile. The most basic `docker build` command might look like the following:

```bash
docker build .
```

The final `.` in the command provides the path or URL to the [build context](https://docs.docker.com/build/concepts/context/#what-is-a-build-context). At this location, the builder will find the `Dockerfile` and other referenced files.

When you run a build, the builder pulls the base image, if needed, and then runs the instructions specified in the Dockerfile.

> 映像最常使用 Dockerfile 构建
> 最基本的 `docker build` 命令为 `docker build .` ，该命令的 `.` 即为构建上下文提供的路径或 URL，构建程序在该位置下找到 Dockerfile 以及其他参考文件
> 运行 `docker build` 之后，构建程序拉去基础映像，如果有的话，还有运行 Dockerfile 中指定的命令

With the previous command, the image will have no name, but the output will provide the ID of the image. As an example, the previous command might produce the following output:

> 直接使用 `docker bulid .` 构建出来的映像没有名字，但有独立的 ID

```console
$ docker build .
[+] Building 3.5s (11/11) FINISHED                                              docker:desktop-linux
 => [internal] load build definition from Dockerfile                                            0.0s
 => => transferring dockerfile: 308B                                                            0.0s
 => [internal] load metadata for docker.io/library/python:3.12                                  0.0s
 => [internal] load .dockerignore                                                               0.0s
 => => transferring context: 2B                                                                 0.0s
 => [1/6] FROM docker.io/library/python:3.12                                                    0.0s
 => [internal] load build context                                                               0.0s
 => => transferring context: 123B                                                               0.0s
 => [2/6] WORKDIR /usr/local/app                                                                0.0s
 => [3/6] RUN useradd app                                                                       0.1s
 => [4/6] COPY ./requirements.txt ./requirements.txt                                            0.0s
 => [5/6] RUN pip install --no-cache-dir --upgrade -r requirements.txt                          3.2s
 => [6/6] COPY ./app ./app                                                                      0.0s
 => exporting to image                                                                          0.1s
 => => exporting layers                                                                         0.1s
 => => writing image sha256:9924dfd9350407b3df01d1a0e1033b1e543523ce7d5d5e2c83a724480ebe8f00    0.0s
```

With the previous output, you could start a container by using the referenced image:
> 可以通过 ID 启动映像

```console
docker run sha256:9924dfd9350407b3df01d1a0e1033b1e543523ce7d5d5e2c83a724480ebe8f00
```

That name certainly isn't memorable, which is where tagging becomes useful.

#### Tagging images
Tagging images is the method to provide an image with a memorable name. However, there is a structure to the name of an image. A full image name has the following structure:

```text
[HOST[:PORT_NUMBER]/]PATH[:TAG]
```

- `HOST`: The optional registry hostname where the image is located. If no host is specified, Docker's public registry at `docker.io` is used by default.
- `PORT_NUMBER`: The registry port number if a hostname is provided
- `PATH`: The path of the image, consisting of slash-separated components. For Docker Hub, the format follows `[NAMESPACE/]REPOSITORY`, where namespace is either a user's or organization's name. If no namespace is specified, `library` is used, which is the namespace for Docker Official Images.
- `TAG`: A custom, human-readable identifier that's typically used to identify different versions or variants of an image. If no tag is specified, `latest` is used by default.

> 可以给映像做标记
> 一个完整的映像名称的结构为 `[HOST[:PORT_NUMBER]/]PATH[:TAG]`
> `HOST` : 映像所在的注册表主机名，如果没有指定，默认使用 Docker 在 `docker.io` 的公共注册表
> `PORT_NUMBER` : 如果主机名提供，可以进一步提供注册表端口号
> `PATH` : 映像的路径，Docker Hub 中映像的路径遵循的格式为 `[NAMESPACE/]REPOSITORY` ，其中 `NAMESPACE` 为用户或者组织名称，如果没有指定，使用 `library` ，即 Docker 官方映像的命名空间
> `TAG` : 自定义的标识符，用于表示映像的不同版本或变体，如果没有提供，默认使用 `latest`

Some examples of image names include:

- `nginx`, equivalent to `docker.io/library/nginx:latest`: this pulls an image from the `docker.io` registry, the `library` namespace, the `nginx` image repository, and the `latest` tag.
- `docker/welcome-to-docker`, equivalent to `docker.io/docker/welcome-to-docker:latest`: this pulls an image from the `docker.io` registry, the `docker` namespace, the `welcome-to-docker` image repository, and the `latest` tag
- `ghcr.io/dockersamples/example-voting-app-vote:pr-311`: this pulls an image from the GitHub Container Registry, the `dockersamples` namespace, the `example-voting-app-vote` image repository, and the `pr-311` tag

To tag an image during a build, add the `-t` or `--tag` flag:
> 在构建时使用 `-t/--tag` 为映像添加标记

```console
docker build -t my-username/my-image .
```

If you've already built an image, you can add another tag to the image by using the [`docker image tag`](https://docs.docker.com/engine/reference/commandline/image_tag/) command:
> 使用 `docker image tag` 可以为已经构建的映像添加另一个标签

```console
docker image tag my-username/my-image another-username/another-image:v1
```

#### Publishing images
Once you have an image built and tagged, you're ready to push it to a registry. To do so, use the [`docker push`](https://docs.docker.com/engine/reference/commandline/image_push/) command:
> 构建好并标记好了映像后，可以 `docker push` 将其 push 到注册表

```console
docker push my-username/my-image
```

Within a few seconds, all of the layers for your image will be pushed to the registry.

> **Requiring authentication**
> 
> Before you're able to push an image to a repository, you will need to be authenticated. To do so, simply use the [docker login](https://docs.docker.com/engine/reference/commandline/login/) command.

## Using the build cache
### Explanation
Consider the following Dockerfile that you created for the [getting-started](https://docs.docker.com/get-started/docker-concepts/building-images/writing-a-dockerfile/) app.

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN yarn install --production
CMD ["node", "./src/index.js"]
```

When you run the `docker build` command to create a new image, Docker executes each instruction in your Dockerfile, creating a layer for each command and in the order specified. For each instruction, Docker checks whether it can reuse the instruction from a previous build. If it finds that you've already executed a similar instruction before, Docker doesn't need to redo it. Instead, it’ll use the cached result. This way, your build process becomes faster and more efficient, saving you valuable time and resources.
> 当我们运行 `docker build` 创建新映像时，Docker 执行 Dockerfile 中的每个指令按照顺序为每个指令创建一个层
> 对于每个指令，Docker 会检查它是否可以从之间的构建中复用它，如果 Docker 发现我们之前执行过相似的指令，Docker 不会重新执行该指令，而是直接使用缓存的结果

Using the build cache effectively lets you achieve faster builds by reusing results from previous builds and skipping unnecessary work. In order to maximize cache usage and avoid resource-intensive and time-consuming rebuilds, it's important to understand how cache invalidation works. Here are a few examples of situations that can cause cache to be invalidated:

- Any changes to the command of a `RUN` instruction invalidates that layer. Docker detects the change and invalidates the build cache if there's any modification to a `RUN` command in your Dockerfile.
- Any changes to files copied into the image with the `COPY` or `ADD` instructions. Docker keeps an eye on any alterations to files within your project directory. Whether it's a change in content or properties like permissions, Docker considers these modifications as triggers to invalidate the cache.
- Once one layer is invalidated, all following layers are also invalidated. If any previous layer, including the base image or intermediary layers, has been invalidated due to changes, Docker ensures that subsequent layers relying on it are also invalidated. This keeps the build process synchronized and prevents inconsistencies.

When you're writing or editing a Dockerfile, keep an eye out for unnecessary cache misses to ensure that builds run as fast and efficiently as possible.

> 以下提供一些示例帮助理解缓存是如何失效的：
> - 任意对 `RUN` 指令中的命令进行的修改都会使该层的构建缓存失效
> - 任意对被 `COPY` 或 `ADD` 指令拷贝到映像中的文件的修改都会使该层的构建缓存失效，Docker 会检查项目目录中对文件的任意修改，无论是对内容的修改还是对属性的修改 (例如访问权限)
> - 一旦一层失效，它随后的层都会失效，这也适用于基础映像中的层以及其他中间层
> 在编辑 Dockerfile 时需要考虑到构建缓存的失效规则以写出能尽量保证构建高效的 Dockerfile

## Multi-stage builds
### Explanation
In a traditional build, all build instructions are executed in sequence, and in a single build container: downloading dependencies, compiling code, and packaging the application. All those layers end up in your final image. This approach works, but it leads to bulky images carrying unnecessary weight and increasing your security risks. This is where multi-stage builds come in.
> 传统构建中，所有的构建指令按照顺序在单个构建容器中执行：下载依赖、编译代码、打包应用程序
> 这些指令得到的所有层都会包含在我们的最终镜像中，这仍以导致镜像体积过大，并且增加安全风险

Multi-stage builds introduce multiple stages in your Dockerfile, each with a specific purpose. Think of it like the ability to run different parts of a build in multiple different environments, concurrently. By separating the build environment from the final runtime environment, you can significantly reduce the image size and attack surface. This is especially beneficial for applications with large build dependencies.
> 多阶段构建在 Dockerfile 中引入了多个阶段，每个阶段都有特定的目的
> 这类似于在多个不同环境中并发地运行构建的不同部分
> 通过将构建环境和最终的运行时环境分离，我们可以显著减少镜像的大小和攻击面，这对于具有大的构建依赖的程序十分有益

Multi-stage builds are recommended for all types of applications.

- For interpreted languages, like JavaScript or Ruby or Python, you can build and minify your code in one stage, and copy the production-ready files to a smaller runtime image. This optimizes your image for deployment.
- For compiled languages, like C or Go or Rust, multi-stage builds let you compile in one stage and copy the compiled binaries into a final runtime image. No need to bundle the entire compiler in your final image.

> 对于所有类型的应用，都推荐多阶段构建
> - 对于解释型语言，可以在一个阶段构建和压缩代码，然后再将生产就绪的文件复制到一个更小的运行时镜像，以优化镜像的部署
> - 对于编译型语言，可以再一个阶段编译代码，然后再将编译好的二进制文件拷贝到最终的运行时镜像，而不需要在最终镜像中捆绑整个编译器

Here's a simplified example of a multi-stage build structure using pseudo-code. Notice there are multiple `FROM` statements and a new `AS <stage-name>`. In addition, the `COPY` statement in the second stage is copying `--from` the previous stage.

```dockerfile
# Stage 1: Build Environment
FROM builder-image AS build-stage 
# Install build tools (e.g., Maven, Gradle)
# Copy source code
# Build commands (e.g., compile, package)

# Stage 2: Runtime environment
FROM runtime-image AS final-stage  
#  Copy application artifacts from the build stage (e.g., JAR file)
COPY --from=build-stage /path/in/build/stage /path/to/place/in/final/stage
# Define runtime configuration (e.g., CMD, ENTRYPOINT) 
```

This Dockerfile uses two stages:

- The build stage uses a base image containing build tools needed to compile your application. It includes commands to install build tools, copy source code, and execute build commands.
- The final stage uses a smaller base image suitable for running your application. It copies the compiled artifacts (a JAR file, for example) from the build stage. Finally, it defines the runtime configuration (using `CMD` or `ENTRYPOINT`) for starting your application.

> 该例展示了使用二阶段构建的 Dockerfile
> 其中 `FROM <image-name> AS <stage-name>` 用于指定构建阶段，`COPY` 语句中的 `--from=<stage-name>` 用于从特定的之前阶段拷贝文件
> 该 Dockerfile 中，构建阶段使用包含了需要用于编译应用的构建工具的基础映像，它包含了用于安装构建工具、拷贝源代码、执行构建命令的指令；最终阶段使用更小的映像作为运行应用的基础映像，它从构建阶段拷贝编译产物，最后定义启动应用时的运行时配置 (使用 `CMD` 或 `ENTRYPOINT` )

> 对于多阶段构建的 Dockerfile，`docker build` 默认构建最后的阶段，我们可以通过 `--target` 选择需要构建的特定阶段
