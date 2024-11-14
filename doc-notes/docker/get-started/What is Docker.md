Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. With Docker, you can manage your infrastructure in the same ways you manage your applications. By taking advantage of Docker's methodologies for shipping, testing, and deploying code, you can significantly reduce the delay between writing code and running it in production.

## The Docker platform
Docker provides the ability to package and run an application in a loosely isolated environment called a container. The isolation and security lets you run many containers simultaneously on a given host. Containers are lightweight and contain everything needed to run the application, so you don't need to rely on what's installed on the host. You can share containers while you work, and be sure that everyone you share with gets the same container that works in the same way.
>Docker 提供了一个在称为容器的松散隔离环境中打包和运行应用程序的能力
>容器是轻量级的，包含了应用程序运行所要求的一切
>容器也可以共享

Docker provides tooling and a platform to manage the lifecycle of your containers:

- Develop your application and its supporting components using containers.
- The container becomes the unit for distributing and testing your application.
- When you're ready, deploy your application into your production environment, as a container or an orchestrated service. This works the same whether your production environment is a local data center, a cloud provider, or a hybrid of the two.

>Docker 提供了管理容器的生命周期的工具和平台
>- 使用容器开发应用和其支持组件
>- 容器会成为发布和测试应用的单元
>- 利用容器将应用直接布置到生产环境

## What can I use Docker for?
### Fast, consistent delivery of your applications
Docker streamlines the development lifecycle by allowing developers to work in standardized environments using local containers which provide your applications and services. Containers are great for continuous integration and continuous delivery (CI/CD) workflows.
> 开发者在本地容器提供的标准环境中开发，使用容器进行持续集成和持续发布

Consider the following example scenario:

- Your developers write code locally and share their work with their colleagues using Docker containers.
- They use Docker to push their applications into a test environment and run automated and manual tests.
- When developers find bugs, they can fix them in the development environment and redeploy them to the test environment for testing and validation.
- When testing is complete, getting the fix to the customer is as simple as pushing the updated image to the production environment.

> 例如以下工作流：
> - 使用容器共享开发的工作
> - 使用容器将应用推送到测试环境，运行测试
> - 根据测试结果，在开发环境修 bug，并重新部署到测试环境进行验证和测试
> - 测试通过后，将更新的映像推送到生产环境

### Responsive deployment and scaling
Docker's container-based platform allows for highly portable workloads. Docker containers can run on a developer's local laptop, on physical or virtual machines in a data center, on cloud providers, or in a mixture of environments.
> 基于容器的平台使得我们的 workload 高度可移植，因为容器可以运行在大量平台和环境上

Docker's portability and lightweight nature also make it easy to dynamically manage workloads, scaling up or tearing down applications and services as business needs dictate, in near real time.
> 容器的轻量特性使得 workload 易于动态管理，进而易于实时将应用和服务拓展或拆分

### Running more workloads on the same hardware
Docker is lightweight and fast. It provides a viable, cost-effective alternative to hypervisor-based virtual machines, so you can use more of your server capacity to achieve your business goals. Docker is perfect for high density environments and for small and medium deployments where you need to do more with fewer resources.
> 容器比基于 hypervisor 的虚拟机更加轻量，故占用更少资源

## Docker architecture
Docker uses a client-server architecture. The Docker client talks to the Docker daemon, which does the heavy lifting of building, running, and distributing your Docker containers. The Docker client and daemon can run on the same system, or you can connect a Docker client to a remote Docker daemon. The Docker client and daemon communicate using a REST API, over UNIX sockets or a network interface. Another Docker client is Docker Compose, that lets you work with applications consisting of a set of containers.
> Docker 使用客户端-服务器结构
> Docker 客户端和 Docker 守护进程进行通信，Docker 守护进程构建、运行、发布容器的工作
> Docker 客户端运行于本地，Docker 守护进程可以运行于本地或远程主机
> Docker 客户端和 Docker 守护进程通过 REST API 通信，基于 UNIX 套接字或网络接口
> 另一种 Docker 客户端称为 Docker Compose，Docker Compose 允许我们使用由一组容器构成的应用程序

![Docker Architecture diagram](https://docs.docker.com/get-started/images/docker-architecture.webp)

### The Docker daemon
The Docker daemon (`dockerd`) listens for Docker API requests and manages Docker objects such as images, containers, networks, and volumes. A daemon can also communicate with other daemons to manage Docker services.
> Docker 守护进程 (`dockerd`) 监听 Docker 客户端的 Docker API 请求，管理各类 Docker 对象，如映像、容器、网络、卷
> 守护进程之间也可以通信来管理 Docker 服务

### The Docker client
The Docker client (`docker`) is the primary way that many Docker users interact with Docker. When you use commands such as `docker run`, the client sends these commands to `dockerd`, which carries them out. The `docker` command uses the Docker API. The Docker client can communicate with more than one daemon.
> 用户通过 Docker 客户端 (`docker`) 与 Docker 交互，用户输入的指令，如 `docker run` 会由客户端发送给守护进程执行
> 用户通过使用 `docker` 命令以使用 Docker API
> Docker 客户端可以与多个守护进程通信

### Docker Desktop
Docker Desktop is an easy-to-install application for your Mac, Windows or Linux environment that enables you to build and share containerized applications and microservices. Docker Desktop includes the Docker daemon (`dockerd`), the Docker client (`docker`), Docker Compose, Docker Content Trust, Kubernetes, and Credential Helper. For more information, see [Docker Desktop](https://docs.docker.com/desktop/).
> Docker 桌面是 Docker 的桌面应用，包含了 Docker 守护进程 (`dockerd`)、Docker 客户端 (`docker`)、 Docker Compose、 Docker Content Trust,、Kubernetes、Credential Helper

### Docker registries
A Docker registry stores Docker images. Docker Hub is a public registry that anyone can use, and Docker looks for images on Docker Hub by default. You can even run your own private registry.
> Docker 注册表存储 Docker 映像
> Docker Hub 是任何人可以使用的公共注册表，Docker 默认在 Docker Hub 寻找 Docker 映像
> 用户也可以使用私人注册表

When you use the `docker pull` or `docker run` commands, Docker pulls the required images from your configured registry. When you use the `docker push` command, Docker pushes your image to your configured registry.
> 使用 `docker pull` 或 `docker run` 命令时，Docker 将从我们设定的注册表中拉取所要求的映像
> 使用 `docker push` 命令时，Docker 将映像推送到所设定的注册表上

### Docker objects
When you use Docker, you are creating and using images, containers, networks, volumes, plugins, and other objects. This section is a brief overview of some of those objects.
> Docker 对象包括映像、容器、网络、卷、插件等 

#### Images
An image is a read-only template with instructions for creating a Docker container. Often, an image is based on another image, with some additional customization. For example, you may build an image which is based on the `ubuntu` image, but installs the Apache web server and your application, as well as the configuration details needed to make your application run.
> 映像是一个只读的指令模板，用于创建容器
> 一个映像一般基于另一个映像，在其上增加了部分改动
> 例如，在 `ubnutu` 映像上添加 Apache web server 和一些应用，并修改一些配置来构建自己的映像

You might create your own images or you might only use those created by others and published in a registry. To build your own image, you create a Dockerfile with a simple syntax for defining the steps needed to create the image and run it. Each instruction in a Dockerfile creates a layer in the image. When you change the Dockerfile and rebuild the image, only those layers which have changed are rebuilt. This is part of what makes images so lightweight, small, and fast, when compared to other virtualization technologies.
> 可以使用自己创建的映像或他人创建的映像
> 自己创建映像时需要建立 Dockerfile，Dockerfile 定义了创建映像和运行映像的所需步骤
> Dockefile 中的每个指令会在映像中创建一层，当我们修改 Dockerfile 重建映像时，只有修改过的层会重建，这使得映像相较于其他虚拟化技术更加轻量和迅速

#### Containers
A container is a runnable instance of an image. You can create, start, stop, move, or delete a container using the Docker API or CLI. You can connect a container to one or more networks, attach storage to it, or even create a new image based on its current state.
> 容器是映像的可运行实例
> 我们可以使用 Docker API 或 CLI 创建、开始、停止、移动、删除容器
> 我们可以将容器连接到一个或多个网络，将存储附加到容器，甚至根据容器当前的状态创建一个新的映像

By default, a container is relatively well isolated from other containers and its host machine. You can control how isolated a container's network, storage, or other underlying subsystems are from other containers or from the host machine.
> 容器默认情况下和其他容器以及宿主机互相隔离
> 我们可以控制容器的网络、存储或其他底层子系统与其他容器或主机的隔离程度

A container is defined by its image as well as any configuration options you provide to it when you create or start it. When a container is removed, any changes to its state that aren't stored in persistent storage disappear.
> 容器由其映像以及创建或启动时提供给它的任何配置选项定义
> 删除容器后，未存储在永久存储中的对其状态的任何更改都将消失

##### Example `docker run` command
The following command runs an `ubuntu` container, attaches interactively to your local command-line session, and runs `/bin/bash`.
> 以下命令运行一个 ubuntu 容器，并将该容器交互式连接/附加到我们的本地命令行会话，然后运行 `/bin/bash`

```console
$ docker run -i -t ubuntu /bin/bash
```

When you run this command, the following happens (assuming you are using the default registry configuration):

1. If you don't have the `ubuntu` image locally, Docker pulls it from your configured registry, as though you had run `docker pull ubuntu` manually.
2. Docker creates a new container, as though you had run a `docker container create` command manually.
3. Docker allocates a read-write filesystem to the container, as its final layer. This allows a running container to create or modify files and directories in its local filesystem.
4. Docker creates a network interface to connect the container to the default network, since you didn't specify any networking options. This includes assigning an IP address to the container. By default, containers can connect to external networks using the host machine's network connection.
5. Docker starts the container and executes `/bin/bash`. Because the container is running interactively and attached to your terminal (due to the `-i` and `-t` flags), you can provide input using your keyboard while Docker logs the output to your terminal.
6. When you run `exit` to terminate the `/bin/bash` command, the container stops but isn't removed. You can start it again or remove it.

>运行此命令时，会发生以下情况 (假设使用的是默认注册表配置)：
>1. 如果本地没有 `ubuntu` 映像, Docker 会从配置的注册表拉取该映像，等价于运行命令 `docker pull ubuntu`
>2. Docker 创建一个新的容器, 等价于运行命令 `docker container create`
>3. Docker 对容器分配一个读写文件系统作为它的最后一层，该层允许运行中的容器在本地文件系统创建或修改文件和目录
>4. Docker 创建一个网络接口以将容器连接到默认网络，包括为容器分配一个 IP 地址，默认情况下，容器可以使用主机的网络连接以连接外部网络
>5. Docker 启动容器并执行 `/bin/bash`，因为容器是交互式运行，并且连接到了我们的终端 (因为 `-i` 和 `-t` 标志)，因此 Docker 会将输出记录到我们的终端，我们也可以通过终端进行输入
>6. 当我们运行 `exit` 来终止 `/bin/bash` 命令时，容器会停止，但不会被移除，我们可以选择重新启动它或移除它

## The underlying technology
Docker is written in the [Go programming language](https://golang.org/) and takes advantage of several features of the Linux kernel to deliver its functionality. Docker uses a technology called `namespaces` to provide the isolated workspace called the container. When you run a container, Docker creates a set of namespaces for that container.

These namespaces provide a layer of isolation. Each aspect of a container runs in a separate namespace and its access is limited to that namespace.
> Docker 由 Go 编写，利用了 Linux 内核的部分特性
> Docker 使用了命名空间技术隔离容器的工作空间，当我们运行一个容器时Docker 会为容器创建一组命名空间，这些命名空间提供了一层隔离，容器的每个方面都在一个单独的命名空间中运行，其访问权限仅限于该命名空间

