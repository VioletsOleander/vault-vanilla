# 1 Docker Overview
## 1.1 The Docker platform
Docker提供了一个在称为容器的松散隔离环境中打包和运行应用程序的能力，容器是轻量级的，包含了应用程序运行所要求的一切，容器也可以共享
Docker提供了工具和平台帮助我们管理容器的生命周期，容器会成为应用程序发布和测试的单元，利用容器可以方便应用直接布置到生产环境

## 1.2 What can I use Docker for
- 快速持续地交付应用
- 响应式的部署和拓展
- 在同一硬件上运行更多工作(比虚拟机更轻量)

## 1.3 Docker architecture
客户端-服务器结构

Docker客户端和Docker守护进程进行通信，构建、运行、发布容器的工作都由守护进程完成
Docker守护进程可以运行于本地或远程主机
Docker客户端和Docker守护进程通过REST API通信，基于UNIX套接字或网络接口

另一种Docker客户端称为Docker Compose，它允许我们使用由一组容器构成的应用程序

**The Docker daemon**
Docker守护进程(`dockerd`)监听Docker API请求，管理各类Docker对象如映像、容器、网络、卷，守护进程之间也可以通信

**The Docker client**
Docker用于通过Docker客户端 (`docker`)与Docker交互，用户输入的指令如 `docker run` 会由客户端发送给守护进程执行，用户通过使用 `docker` 命令以使用Docker API，Docker客户端可以与多个守护进程通信

**Docker Desktop**
Docker桌面是Docker的桌面应用，包含了Docker守护进程(`dockerd`)、Docker客户端 (`docker`)、 Docker Compose、 Docker Content Trust,、Kubernetes、Credential Helper

**Docker registries**
Docker注册表存储Docker映像，Docker Hub是任何人可以使用的公共注册表，Docker默认在Docker Hub寻找Docker映像
用户可以使用私人注册表

当我们使用 `docker pull` 或 `docker run` 命令时，Docker将从我们设定的注册表中拉取所要求的映像，当我们使用 `docker push` 命令时，Docker将映像推送到所设定的注册表上

**Docker objects**
Docker对象包括映像、容器、网络、卷、插件等 

(1) Images
映像是一个只读的指令模板，用于创建容器，一个映像一般基于另一个映像，在其上增加了部分改动
比如我们可以基于 `ubuntu` 映像建立自己的映像，然后在其中下载Apache网络服务器和我们的应用及其依赖项

可以使用自己创建的映像或他人创建的映像，自己创建映像时需要建立Dockerfile，Dockefile中的每个指令会在映像中创建一层，当我们修改Dockerfile重建映像时，只有修改过的层会重建，这个技术使得映像更加轻量和迅速

(2) Containers
实例是映像的可运行实例，我们可以使用Docker API或CLI创建、开始、停止、移动、删除容器，我们可以将容器连接到一个或多个网络，将存储附加到容器，甚至根据其当前的状态创建一个新的映像

我们可以控制容器的网络、存储或其他底层子系统与其他容器或主机的隔离程度

容器由其映像以及创建或启动时提供给它的任何配置选项定义，删除容器后，未存储在永久存储中的对其状态的任何更改都将消失

Example docker run command：
以下命令运行一个ubuntu容器，以交互方式连接到本地命令行会话，并运行/bin/bash：
`docker run -i -t ubuntu /bin/bash`

运行此命令时，会发生以下情况(假设使用的是默认注册表配置)：
1. 如果本地没有 `ubuntu` 映像, Docker会从配置的注册表拉取该映像，等价于运行命令 `docker pull ubuntu`
2. Docker创建一个新的容器, 等价于运行命令 `docker container create`
3. Docker对容器分配一个读写文件系统作为它的最后一层，该层允许运行中的容器在本地文件系统创建或修改文件和目录
4. Docker创建一个网络接口以将容器连接到默认网络，包括为容器分配一个IP地址，默认情况下，容器可以使用主机的网络连接以连接外部网络
5. Docker启动容器并执行 `/bin/bash`，因为容器是交互式运行，并且连接到了我们的终端(因为 `-i` 和 `-t` 标志)，因此Docker会将输出记录到我们的终端，我们也可以通过终端进行输入
6. 当我们运行 `exit` 来终止 `/bin/bash` 命令时，容器会停止，但不会被移除，我们这是可以选择重新启动它或移除它

## 1.4 The Underlying Technology
Docker由Go编写，利用了Linux内核的部分特性，Docker使用了命名空间技术隔离容器的工作空间，当我们运行一个容器时，Docker会为容器创建一组命名空间，这些命名空间提供了一层隔离，容器的每个方面都在一个单独的命名空间中运行，其访问权限仅限于该命名空间

# 2 Get Started
## 2.1 Overview
### 2.1.1 What is a container
容器是运行于主机上的一个沙盒进程，与其他进程隔离
- 容器是映像的可运行实例，我们借助Docker API或CLI创建、启动、停止、移动或删除容器
- 容器可以运行于本地，或运行于虚拟机，或部署到云上
- 容器是可移植的，可以在任何OS上运行
- 容器与其他容器隔离，运行自己的软件、二进制文件、配置等
### 2.1.2 What is a image
一个运行中的容器需要使用一个隔离的文件系统，而这个文件系统由映像提供，映像必须包含运行一个程序所需的一切，包括依赖项、配置、脚本、二进制文件等，映像也包含了容器的配置项，包括环境变量、用于运行的默认命令、其他元数据等

## 2.2 Containerize an application
### 2.2.1 Build the app's image
建立映像需要用到Dockerfile，Dockerfile是基于文本的文件，没有文件拓展名，Docker根据Dockerfile来构建一个容器映像
1. 创建空文件Dockerfile
	`type nul > Dockerfile`
2. 修改Dockerfile
	```dockerfile
	# syntax=docker/dockerfile:1
	
	FROM node:18-alpine
	WORKDIR /app
	COPY . .
	RUN yarn install --production
	CMD ["node", "src/index.js"]
	EXPOSE 3000
	```
3. 构建映像
	`docker build -t getting-started .`

`doker build` 命令根据Dockerfile构建映像，我们指定了要从 `node:18-alpine` 映像开始构建，Docker会首先下载这个映像

`WORKDIR` 指令指定了在容器的文件系统中， `RUN`, `CMD`, `ENTRYPOINT`, `COPY`,  `ADD` 这些指令的工作目录

`COPY` 指令将本地文件系统中 `<src>` 路径下的文件拷贝到容器的文件系统中的 `<dest>` 路径下，`<dest>` 路径使用绝对路径，即从根目录 `/` 开始，否则默认是从 `WORKDIR` 开始的相对路径

`CMD` 指令指定从此映像启动容器时要运行的默认命令

`-t` 标志用于标记映像，我们将映像命名为 `getting-started`

`.` 告诉Docker在当前目录寻找Dockerfile

### 2.2.2 Start an app container
创建映像后，可以用 `docker run` 命令创建一个新的容器运行应用：
1. 使用docker Run命令运行容器，并指定映像名称
	`docker run -dp 127.0.0.1:3000:3000 getting-started`
	`-d` 标志( `--detach` 的缩写)在后台运行容器
	`-p` 标志( `--publish` 的缩写)创建主机和容器之间的端口映射，`-p` 标志采用 `HOST:CONTAINER` 格式的字符串值，其中 `HOST`是主机上的地址，`CONTAINER` 是容器上的端口，该命令将容器的端口 `3000` 发布到主机上的`127.0.0.1:3000`(`localhost:3000`)，如果没有端口映射，将无法从主机访问应用程序
2. 打开web浏览器，访问 `http://localhost:3000` ，可以查看应用程序正常运行，这是一个todo list管理程序

可以使用CLI或Docker Desktop的图形界面查看容器，我们可以看到有一个容器正在运行，该容器正在使用 `getting-started` 映像并位于端口`3000`上

可以用`docker ps` 列出我们的容器

## 2.3 Update the application

### 2.3.1 Update the source code
1. 修改 `src/static/js/app.js` 文件
2. 用 `docker build` 命令构建更新版本的映像
	`docker build -t getting-started .`
3. 启动一个容器
	`docker run -dp 127.0.0.1:3000:3000 getting-started`
由于旧容器已经使用了主机上的端口 `3000` ，而一个端口只能由一个程序监听，因此我们无法启动新的容器，需要先移除旧的容器

### 2.3.2 Remove the old container
1. 通过 `docker ps` 得到容器ID
2. 通过 `docker stop <the-container-id>` 停止容器
3. 通过 `docker rm <the-container-id>` 移除已停止的容器

`docker rm -f <the-container-id>` 可以同时完成停止容器和移除容器

现在可以正常启动新的容器了

## 2.4 Share the application
我们使用Docker注册表分享映像，默认的注册表是Docker Hub

Docker Hub是世界最大的映像库和分享社区，访问Docker Hub需要Docker ID

拥有Docker ID后，可以通过 `docker login -u USER-NAME` 登录

### 2.4.1 Create a repositry
要推送一个映像，首先要在Docker Hub中创建一个仓库
如果是要分享的仓库，可见性需要设置为公有

一个仓库可以以标签(tags)的形式存储多个映像，一般是存储一个映像的多个版本，用标签区分版本号

### 2.4.2 Push the image
`docker push` 命令用于将映像推送到Docker Hub，如：
`docker push vioser/getting-started`
该命令将寻找一个名为 `vioser/getting-started` 的映像
用`docker image ls` 查看，显然我们只有一个名为 `getting-started` 的映像，而没有名为 `vioser/getting-started` 的映像，因此命令执行不成功

要将本地的映像推送到Docker Hub，我们首先要做的是将本地映像命名为Docker Hub用户名+仓库名

使用 `docker tag` 可以给映像取一个别名，如：
`docker tag getting-started vioser/gettting-started`

现在使用 `docker push vioser/getting-started:tagname` 可以成功将 `vioser/getting-started` 映像推送到Docker Hub，仓库中的映像以标签名辨识
直接用 `docker push vioser/getting-started` ，则Docker默认为映像添加的标签名是 `latest`

### 2.4.3 Run the image on a new instance
将一个映像推送到Docker Hub之后，我们可以在另一个环境中拉取这个映像，并运行实例(Docker是支持创建多平台映像的，因此平台不同也有解决方案)

我们在浏览器中用Docker Hub账号登录Play with Docker以模拟另一个环境
点击 `ADD NEW INSTANCE` 就可以创建一个沙盒环境
在环境中运行 `docker run -dp 0.0.0.0:3000:3000 YOUR-USER-NAME/getting-started` 利用我们上传的映像创建一个容器

可以看到映像被拉取并启动

注意，将容器的端口绑定到IP地址 `127.0.0.1` 时，容器的端口实际上只对本机开放，而将容器的端口绑定到IP地址 `0.0.0.0` 时，容器的端口实际上对外部网络也开放

Additional：
我们可以用 `docker search keyword` 搜索Docker Hub中的映像，`keyword` 可以是映像名、用户名、或是描述
我们可以用 `docker pull <image-name>` 从Docker Hub中下载/拉取映像

## 2.5 Persist the DB
### 2.5.1 The container's filesystem
当容器运行时，它会将映像的多个层用于自己的文件系统，每个容器也拥有自己的“scratch space“(暂存空间)来创建/更新/删除文件，这些操作只对当前容器有效，对于由同个映像创建出的其他容器不可见

### 2.5.2 Container volumes
每个容器都从映像定义开始启动，而当我们删除容器时，我们在容器中所做的更改如文件创建、更新、删除等就会丢失


利用卷可以将容器的文件系统和主机的文件系统进行连接，如果我们在容器中挂载了一个主机的目录，那么我们可以在主机上也看到该目录中的更改，在容器重新启动时挂载上这个目录，就可以在容器中看到相同的文件，通过这种方式可以保存在容器中的修改

### 2.5.3 Persist the todo data
todo程序默认将数据存储于SQLite数据库，在容器的文件系统中，数据库文件位于 `/etc/todos/todo.db` 

我们只要将数据库文件保存于主机之上，就可以使其可用于下一个容器
创建一个卷并将其挂载到容器中存储数据的目录后，当容器写入todo.db文件时，数据会保存到主机上的卷中

卷的具体操作由Docker完全管理

**Create a volume and start the container**
`docker volume create` 用于创造一个卷，如：
`docker volume create todo-db`

在 `docker run` 命令中添加 `--mount` 选项，指明挂载的类型为卷，指明卷的名称为刚才创建的 `todo-db` ，指明卷要挂载到路径 `/etc/todos` 下(即卷会保存该路径下所有的文件)：
```powershell
docker run -dp 127.0.0.1:3000:3000 --mount type=volume,src=todo-db,target=/etc/todos getting-started
```

**Verify that the data persists**
在todo app中添加一些项，然后用 `docker rm -f <id>` 将容器停止并移除

然后用 `docker run -dp 127.0.0.1:3000:3000 --mount type=volume,src=todo-db,target=/etc/todos getting-started` 启动一个新的容器

可以发现即使我们启动了新的容器，之前的项还在

### 2.5.4 Dive into the volume
`docker volume inspect` 命令用于查看卷的详细信息：
`docker volume inspect todo-db`

其中的 `Mountpoint` 告诉了我们本地磁盘中数据的实际位置

## 2.6 Use bind mounts
绑定挂载是除了卷挂载以外另一种类型的挂载，绑定挂载允许我们将主机文件系统上的一个目录与容器进行共享
我们常常用绑定挂载将源代码目录挂载到容器上，当我们在主机上保存文件时，容器就可以看到文件中的更改(这意味着我们可以在容器中运行程序监视主机文件系统的更改并作出响应，比如在本地IDE上修改代码，在容器中运行代码)

### 2.6.1 Quick volume type comparisons
卷挂载和绑定挂载的区别：

|           | 卷挂载      | 绑定挂载 |
| --------- | -------- | ---- |
| 本地位置      | Docker决定 | 用户决定 |
| 用容器内容填充新卷 | 是        | 否    |
| 支持卷驱动     | 是        | 否    |

### 2.6.2 Trying out bind mounts
切换当前目录到 `getting-started-app`

在绑定挂载了当前目录的 `ubuntu` 容器中启动 `bash` ：
`docker run -it --mount "type=bind,src=%cd%,target=/src" ubuntu bash`

`--mount` 选项指定了将本机的 `getting-started-app` 目录绑定挂载到容器的 `/src` 目录

该命令让Docker启动了一个交互式的 `bash` 会话，初始位于容器文件系统的根目录下

`cd /src` ，我们可以发现因为这个目录和本机的 `getting-started-app` 目录进行了绑定挂载，这个目录的文件内容和本机的 `getting-started-app` 目录下的内容完全一样

在容器中用 `touch myfile.txt` 创建一个文件，可以发现本机上也出现了这个文件
在本机上删除这个文件，可以发现容器中这个文件也消失了
可以发现任何一方的改变都是双方可见的

`CTRL + D` 以停止容器和交互式会话

### 2.6.2 Development containers
**Run your app in a development container**
利用绑定挂载，可以
1. 将源代码挂载到容器中
2. 下载所有的依赖
3. 启动 `nodemon` 监视文件系统的变化
以运行一个开发容器(即利用绑定挂载进行开发)

在PowerShell中运行 
```powershell
docker run -dp 127.0.0.1:3000:3000 `
-w /app --mount "type=bind,src=$pwd,target=/app" `
node:18-alpine `
sh -c "yarn install && yarn run dev"
```
解释：
`-dp 127.0.0.1:3000:3000` 在 `detached` 模式(后台模式)运行，创建端口映射
`-w /app` 设置工作目录
`--mount "type=bind,src=$pwd,target=/app"` 将当前目录和容器下的 `/app` 目录绑定挂载
`node:18-alpine` 使用的映像 
`sh -c "yarn install && yarn run dev"` 要在工作目录执行的命令，该命令用 `sh` 启动了shell，然后运行 `yarn install` 下载包，运行 `yarn run` 启动开发服务器( `dev` 脚本启动了 `nodemon`)

使用 ` docker logs <container-id>` 可以查看日志

**Develop your app with the development container**
现在我们可以在本机上修改源代码，如修改 `src/static/js/app.js`

可以发现应用虽然在运行中，但是前端的内容也立即发生了改变，因为 `nodemon` 检测到了代码的改变，然后重启了Node服务器

每当我们改变了源代码，绑定挂载帮助我们将修改反映到容器的文件系统，而 `nodemon` 检测到改变时，就会自动重启app

因为我们改变了源代码，所以当我们退出时，如果希望下次正常启动容器这个改变可以被反映，我们需要重建映像 `docker build -t getting-started .`
绑定挂载就是帮助了我们在开发时不需要重建映像就可以改变容器内文件

## 2.7 Multi container apps
目前为止，我们使用的都是单容器app
在构建应用程序栈的时候，往往需要多容器，这帮助我们将API和前端与后端数据库分离(我们不希望在生产中将数据库引擎和应用一起部署)，另外，在一个容器中运行多进程需要进程管理，因为一个容器在一个时间只能运行一个进程，此外，多容器可以方便我们分离当前版和更新的版本

我们希望将我们的todo app运行于多容器中，其中todo app一个容器，MySQL数据库一个容器

### 2.7.1 Container networking
默认情况下，容器是独立运行的，同一主机上运行的各个容器相互隔离，但将两个容器置于同一个网络中，可以让容器之间相互通信

### 2.7.2 Start MySQL
将容器附加到一个网络中有两种方法：
- 在启动容器时为容器分配网络
- 在容器运行时将其连接到网络

`docker network create` 命令用于创建网络，如：
`docker network create todo-app` 

在容器启动时就可以将其附加到网络：
```powershell
docker run -d `
    --network todo-app --network-alias mysql `
    -v todo-mysql-data:/var/lib/mysql `
    -e MYSQL_ROOT_PASSWORD=secret `
    -e MYSQL_DATABASE=todos `
    mysql:8.0
```
该命令启动了一个MySQL容器其中：
`-d` 标志用于指定容器后台运行，
`--network` 标志用于指定要连接的网络，
`--network-alias` 标志用于指定该容器在网络中的别名，
`-v` 标志用于卷挂载，本例中将卷 `todo-mysql-data` (如果这个卷不存在，Docker会自动创建)挂载到 `/var/lib/mysql` 目录(这个目录是MySQL存储数据的目录)，
`-e` 标志用于额外设置容器的环境变量，本例中 `MYSQL_ROOT_PASSWORD=secret` 将MySQL `root` 账户的密码设置为 `secret` (注意这个环境变量对于MySQL容器是必须的)，`MYSQL_DATABASE=todos` 将数据库的名称设置为 `todos`

`docker exec` 命令用于在容器中执行指令，启动后，我们用 `docker exec` 命令登录 `root` 账户：
`docker exec -it <mysql-container-id> mysql -u root -p`
输入密码即可登录

登录后，输入 `SHOW DATABASE` 命令就可以看到有 `todos` 数据库存在，再输入 `exit` 就可以退出MySQL的shell回到本机的shell

### 2.7.3 Connect to MySQL
Docker Hub中的 `nicolaka/netshoot` 映像附带了许多工具，用于解决和调试网络问题

我们利用 `nicolaka/netshoot` 创建一个新的容器，注意要将它与我们的数据库容器(MySQL)连接到同一个网络中：
`docker run -it --network todo-app nicolaka/netshoot`

在 `nicolaka/netshoot` 创建的容器中， `dig` 命令可以根据容器在网络中的名称查找容器在网络中的IP地址，可以认为这是一个有用的DNS工具，可以查找主机名对应的IP地址：
`dig mysql` (注意我们之前通过 `network-alias` 指定了容器在网络中的别名)

在指令结果的“ANSWER SECTION”中，我们可以看到关于 `mysql` 的一个 `A` 类型的记录，记录了它的IP地址
事实上 `mysql` 被我们用作了数据库容器在网络中的主机名，我们的app容器可以直接通过这个主机名与它连接

### 2.7.4 Run your app with MySQL
todo小程序也支持一些环境变量的设置，用于和MySQL数据库连接，包括：
- `MYSQL_HOST` 
	MySQL服务器的主机名
- `MYSQL_USER` 
	连接中的用户名
- `MYSQL_PASSWORD`
	连接的密码
- `MYSQL_DB`
	要连接到的数据库名

> 注意：
> 利用环境变量进行连接设定的方式常在开发中使用，但是不推荐在实际将程序运行于生产中使用
> 一个更安全的机制是使用容器编排框架(orchesetration framework)提供的机密支持(secret support)，在大多数情况下，这些机密(secret)都以文件的形式文件挂载在运行容器中
> 许多应用程序(包括MySQL映像和todo应用程序)都支持带有 `_FILE` 后缀的环境变量，即 `varname_FILE` 这样的形式，`varname_FILE` 用于指向包含了 `varname` 这个环境变量的文件
> 例如，设置了环境变量 `MYSQL_PASSWORD_FILE` 为相应的文件名 ，应用程序就会从这个文件中查找连接的密码，而不是直接从环境变量 `MYSQL_PASSWORD` 中获取连接的密码，Docker没有做任何事支持这类 `_FILE` 环境变量，我们需要自己设定逻辑，让我们的应用程序根据 `_FILE` 环境变量查找文件并获得具体变量的值

我们用 `docker run` 启动todo-app容器，并设置好环境变量，同时注意将它连接到网络中：
```powershell
docker run -dp 127.0.0.1:3000:3000 `
  -w /app -v "$(pwd):/app" `
  --network todo-app `
  -e MYSQL_HOST=mysql `
  -e MYSQL_USER=root `
  -e MYSQL_PASSWORD=secret `
  -e MYSQL_DB=todos `
  node:18-alpine `
  sh -c "yarn install && yarn run dev"
```

我们可以用 `docker logs <container-id>` 查看日志，可以发现日志中有容器连接到数据库的记录

我们可以在app中加入一些item，然后连接到MySQL数据库中，查看这些item是否写入了数据库中：
`docker exec -it <mysql-container-id> mysql -p todos`
`select * from todo_items;`
可以发现我们的items已经写入了数据库中

## 2.8 Use Docker Compose
Docker Compose是一个帮助我们定义和共享多容器应用的工具

使用Docker Compose的优势在于我们可以在一个位于项目仓库根目录下的文件中定义应用程序栈，并且方便其他贡献者对项目进行贡献，Compose用户允许克隆仓库，以使用app

### 2.8.1 Create the Compose file
在 `getting-started-app` 目录中，创建一个名为 `compose.yaml` 的文件
```text
├── getting-started-app/
│ ├── Dockerfile
│ ├── compose.yaml
│ ├── node_modules/
│ ├── package.json
│ ├── spec/
│ ├── src/
│ └── yarn.lock
```

### 2.8.2 Define the app service
我们曾用该命令启动我们的app服务：
```powershell
docker run -dp 127.0.0.1:3000:3000 \
  -w /app -v "$(pwd):/app" \
  --network todo-app \
  -e MYSQL_HOST=mysql \
  -e MYSQL_USER=root \
  -e MYSQL_PASSWORD=secret \
  -e MYSQL_DB=todos \
  node:18-alpine \
  sh -c "yarn install && yarn run dev"
```

我们也可以在 `compose.yaml` 中定义我们的服务：
1. 在 `compose.yaml` 中首先定义我们第一个服务(或容器)的名称和映像，这里我们定义的名称也会是容器在网络中的别名
	```yaml
	services:
	  app:    
		  image: node:18-alpine
	```
2. 在 `command` 部分写上要执行的命令
    ```yaml
    services:
      app:
        image: node:18-alpine
        command: sh -c "yarn install && yarn run dev"
    ```
3.  `ports` 部分写上服务要映射的端口 
    ```yaml
    services:
      app:
        image: node:18-alpine
        command: sh -c "yarn install && yarn run dev"
        ports:
          - 127.0.0.1:3000:3000
    ```
4. 在 `working_dir` 部分写上工作目录，在 `volumes` 部分写上绑定挂载的目录，可以使用相对路径
    ```yaml
    services:
      app:
        image: node:18-alpine
        command: sh -c "yarn install && yarn run dev"
        ports:
          - 127.0.0.1:3000:3000
        working_dir: /app
        volumes:
          - ./:/app
    ```
5. 最后在 `environment` 部分写上环境变量
    ```yaml
    services:
      app:
        image: node:18-alpine
        command: sh -c "yarn install && yarn run dev"
        ports:
          - 127.0.0.1:3000:3000
        working_dir: /app
        volumes:
          - ./:/app
        environment:
          MYSQL_HOST: mysql
          MYSQL_USER: root
          MYSQL_PASSWORD: secret
          MYSQL_DB: todos
    ```

### 2.8.3 Define the MySQL service
同样，对于MySQL服务，我们之前使用的是命令启动：
```powershell
docker run -d \
  --network todo-app --network-alias mysql \
  -v todo-mysql-data:/var/lib/mysql \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=todos \
  mysql:8.0
```

现在可以在 `compose.yaml` 中定义：
1. 首先定义新服务名称为 `mysql` ，注意它在网络中的别称也将是 `mysql` ，同时指定它要使用的映像
    ```yaml
    services:
      app:
        # The app service definition
      mysql:
        image: mysql:8.0
    ```
2. 如果需要卷挂载，我们需要在顶级的 `volumes` 部分写上卷名称，如果该卷不存在，Docker会自动创建(根据默认配置)，然后再 `services` 下 `volumes` 部分指明挂载的目录
    ```yaml
    services:
      app:
        # The app service definition
      mysql:
        image: mysql:8.0
        volumes:
          - todo-mysql-data:/var/lib/mysql
    
    volumes:
      todo-mysql-data:
    ```
3. 最后指明环境变量
    ```yaml
    services:
      app:
        # The app service definition
      mysql:
        image: mysql:8.0
        volumes:
          - todo-mysql-data:/var/lib/mysql
        environment:
          MYSQL_ROOT_PASSWORD: secret
          MYSQL_DATABASE: todos
    
    volumes:
      todo-mysql-data:
    ```

全部完整的 `compose.yaml` 应该是：
```yaml
services:
  app:
    image: node:18-alpine
    command: sh -c "yarn install && yarn run dev"
    ports:
      - 127.0.0.1:3000:3000
    working_dir: /app
    volumes:
      - ./:/app
    environment:
      MYSQL_HOST: mysql
      MYSQL_USER: root
      MYSQL_PASSWORD: secret
      MYSQL_DB: todos

  mysql:
    image: mysql:8.0
    volumes:
      - todo-mysql-data:/var/lib/mysql
    environment:
      MYSQL_ROOT_PASSWORD: secret
      MYSQL_DATABASE: todos

volumes:
  todo-mysql-data:
```

### 2.8.4 Run the application stack
使用 `docker compose up` 命令启动整个应用栈，`-d` 标志使它们运行于后台： `docker compose up`

可以发现Docker创建了网络、卷，同时启动了容器，默认情况下，Docker Compose会为我们定义的应用程序栈创建它专用的网络，因此我们不需要在 `compose.yaml` 中显示创建网络

使用 `docker compose logs` 可以查看日志，可以看到各个服务的消息，如果想看特定服务的消息，可以使用 `docker compose logs <service-name>` ，如 `docker compose logs app`

在Docker Dashboard，我们可以看到一个名为 `getting-started-app` 的容器组，包含了应用栈中的容器，容器组的名称一般直接以 `compose.yaml` 位于的文件夹名称命名
容器组下的容器命名一般遵循 `<service-name>-<replica-number>` 的格式

使用 `docker compose down` 可以关闭应用栈，删除容器
默认情况下，在 `compose.yaml` 中指名的卷，不会被移除，如果我们需要将它们也移除的话，可以加上 `--volumes` 标志

## 2.9 Image-building best practices
### 2.9.1 Image layering
`docker image history` 命令用于查看一个映像的各层是如何被创建的，如：
`docker image history getting-started`

输出的每一行代表一个层，其中最底层位于底端，最高层位于顶端

如果希望得到完整的输出，可以加上 `--no-trunc` 标志：
`docker image history --no-trunc getting-started`

### 2.9.2 Layer caching
注意当一层改变了，所有在该层之下的层也要被改变

我们用Dockerfile创建了 `getting-started` 映像：
```dockerfile
# syntax=docker/dockerfile:1
FROM node:18-alpine
WORKDIR /app
COPY . .
RUN yarn install --production
CMD ["node", "src/index.js"]
```

可以看到，在 `docker image history` 中，Dockerfile中的每一行都成为了映像中的一层

每次我们修改了文件，就需要重建映像，而如果不希望每次重建映像都要运行 `yarn install --production` 重新下载依赖，可以利用层缓存

为此，需要重构Dockerfile，对于基于Node的应用，它的依赖项一般都定义于 `package.json` 文件中：
```dockerfile
# syntax=docker/dockerfile:1
FROM node:18-alpine
WORKDIR /app
COPY package.json yarn.lock ./
RUN yarn install --production
COPY . .
CMD ["node", "src/index.js"]
```
我们先将 `package.json` 复制，然后下载依赖，在将其他文件复制，这样，只有在我们修改了 `package.json` 文件时，重建映像时才会重新运行 `yarn install --production`  以重新下载依赖

之后我们在与Dockerfile的同一个文件夹内创建一个名为 `.dockerignore` 的文件，文件内容为：`node_modules` 
利用`.dockerignore` 文件，我们可以选择性地只复制映像有关的文件，在本例中，根据 `.dockerignore` 文件， `node_modules` 目录将在第二次的 `COPY` 中被忽略(之所以需要忽略该文件夹，是因为它可能会覆盖 `RUN` 指令创建的文件)

然后我们创建映像：
`docker build -t getting-started .`

之后我们可以对一些文件进行修改，如修改 `src/static/index.html` ，将 `<title>` 改为 "The Awesome Todo App"

然后 `docker build -t getting-started .` 重建映像，观察变化：
```text
 => CACHED [2/5] WORKDIR /app                                 0.0s
 => CACHED [3/5] COPY package.json yarn.lock ./               0.0s
 => CACHED [4/5] RUN yarn install --production                0.0s
 => [5/5] COPY . .
```
可以发现这次构建的速度更快了，因为一些指令直接使用了之前缓存的层，不需要再重建新层，使用缓存，不但重构映像会变快，拉取和推送映像也会更快

### 2.9.3 Multi-stage builds
多阶段构建：
- 分离构建时依赖和运行时依赖
- 通过仅装在程序需要运行的内从，减少总体映像大小

**Maven/Tomcat example**
构建基于Java的应用时，需要JDK以编译源码和字节码，但JDK在生产部署时是不需要的，并且，我们如果使用Maven或Gradle这样的工具来帮助构建应用时，我们的最终映像也是不需要它们的

我们可以使用多阶段构建：
```dockerfile
# syntax=docker/dockerfile:1
FROM maven AS build
WORKDIR /app
COPY . .
RUN mvn package

FROM tomcat
COPY --from=build /app/target/file.war /usr/local/tomcat/webapps 
```
我们在第一个阶段，称其为 `build` 阶段，使用Maven构建Java，在第二个阶段(从 `FROM tomcat` 开始) 将 `build` 阶段的文件拷贝，最后得到的映像仅仅只是第二个阶段创建的映像，我们也可以使用 `--target` 标志选择要构建的映像是来源于哪个阶段

**React example**
在构建React应用程序时，需要一个Node环境来将JSX代码、SASS样式表等编译为静态HTML、JS和CSS
但如果不进行服务器端渲染，则生产构建不需要Node环境，只需要在静态nginx容器运行中静态资源即可，因此只需要运输静态资源
```dockerfile
# syntax=docker/dockerfile:1
FROM node:18 AS build
WORKDIR /app
COPY package* yarn.lock ./
RUN yarn install
COPY public ./public
COPY src ./src
RUN yarn run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
```
我们使用 `node:18` 映像进行构建，然后直接将它的输出静态资源拷贝至一个 nginx容器

# 3 Build with Docker
## 3.1 Introduction
### 3.1.1 The application
这是一个客户端-服务器应用
```text
.
├── Dockerfile
├── cmd
│   ├── client
│   │   ├── main.go
│   │   ├── request.go
│   │   └── ui.go
│   └── server
│       ├── main.go
│       └── translate.go
├── go.mod
└── go.sum
```
`cmd` 目录包含了应用的客户端和服务器的代码
客户端作为用户接口，服务器接收客户端的信息，进行翻译，然后返回给客户端

### 3.1.2 The Dockerfile

# 4 Develop with Docker
## 4.1 Docker development best practices
### 4.1.1 How to keep your images small
- 从合适的基映像开始
	比如我们在需要JDK时，考虑直接将我们的映像基于带有JDK的映像，如 `elipse-temurin` ，而不是从零开始
- 使用多阶段构建
	比如先使用 `maven` 映像构建Java应用，然后构建 `tomcat` 映像以部署应用，将`maven` 映像的静态产物复制到合适的位置即可，最终得到的映像只包含了运行时需要的环境
	如果Docker版本不支持多阶段构建，合并 `RUN` 指令可以减少映像的层数，比如将 
	`RUN apt-get -y update`
	`RUN apt-get install -y python`
	改为
	`RUN apt-get -y update && apt-get install -y python`
- 如果有多个映像共享的部分，考虑将这个部分创建为一个基映像，将其他映像基于这个基映像，这样在主机上，Docker只需要加载并缓存基映像的部分一次即可供公共使用
- 如果生产映像不需要debug工具，可以将生产映像作为基映像，debug映像基于生产映像，将测试和debug工具置于debug映像
- 构建映像时，给他们合适的标签，标签应包含：版本信息、目的(生产 `prod` 或测试 `test` )、稳定性以及在不同环境中部署时有用的其他信息，不要依赖于自动创建的 `latest` 标签

### 4.1.2 Where and how to persist application data
