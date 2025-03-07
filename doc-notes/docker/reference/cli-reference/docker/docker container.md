# docker container commit
## Description
It can be useful to commit a container's file changes or settings into a new image. This lets you debug a container by running an interactive shell, or export a working dataset to another server.

Commits do not include any data contained in mounted volumes.

By default, the container being committed and its processes will be paused while the image is committed. This reduces the likelihood of encountering data corruption during the process of creating the commit. If this behavior is undesired, set the `--pause` option to false.

The `--change` option will apply `Dockerfile` instructions to the image that's created. Supported `Dockerfile` instructions: `CMD` | `ENTRYPOINT` | `ENV` | `EXPOSE` | `LABEL` | `ONBUILD` | `USER` | `VOLUME` | `WORKDIR`

>  `docker commit` 的提交不会包含任意挂载卷中的数据
>  默认情况下，commit 时，正在被提交的容器及其进程会被暂停，避免提交时数据损坏 (即默认 `--pause true` )
>  `--change` 会将 `Dockerfile` 指令应用于它所创建的镜像，支持的 `Dockerfile` 指令如上


