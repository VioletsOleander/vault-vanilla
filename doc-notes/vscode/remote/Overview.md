---
completed: true
---
# VS Code Remote Development
**Visual Studio Code Remote Development** allows you to use a container, remote machine, or the [Windows Subsystem for Linux](https://learn.microsoft.com/windows/wsl) (WSL) as a full-featured development environment. You can:

- Develop on the **same operating system** you deploy to or use **larger or more specialized** hardware.
- **Separate** your development environment to avoid impacting your local **machine configuration**.
- Make it easy for new contributors to **get started** and keep everyone on a **consistent environment**.
- Use tools or runtimes **not available** on your local OS or manage **multiple versions** of them.
- Develop your Linux-deployed applications using the **Windows Subsystem for Linux**.
- Access an **existing** development environment from **multiple machines or locations**.
- Debug an **application running somewhere else** such as a customer site or in the cloud.

**No source code** needs to be on your local machine to get these benefits. Each extension in the [Remote Development extension pack](https://aka.ms/vscode-remote/download/extension) can run commands and other extensions directly inside a container, in WSL, or on a remote machine so that everything feels as it does when you run locally. The extensions install VS Code Server on the remote OS; the server is independent of any existing VS Code installation on the remote OS.
>  远程开发拓展包中的每个拓展都可以直接在容器, WSL, 远程机器上运行命令或其他拓展，看起来就像一切都在本地运行
>  远程开发拓展包会在远程 OS 上安装 VS Code Server，该 Server 独立于远程机器上任意现存的 VS Code 安装

![Architecture](https://code.visualstudio.com/assets/docs/remote/remote-overview/architecture.png)

## Getting started
### Remote Development extension pack
The [Remote Development extension pack](https://aka.ms/vscode-remote/download/extension) includes four extensions. See the following articles to get started with each of them:

- [Remote - SSH](https://code.visualstudio.com/docs/remote/ssh) - Connect to any location by opening folders on a remote machine/VM using SSH.
- [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) - Work with a separate toolchain or container-based application inside (or mounted into) a container.
- [WSL](https://code.visualstudio.com/docs/remote/wsl) - Get a Linux-powered development experience in the Windows Subsystem for Linux.
- [Remote - Tunnels](https://code.visualstudio.com/docs/remote/tunnels) - Connect to a remote machine via a secure tunnel, without configuring SSH.

>  远程开发拓展包包含了四个拓展:
>  - Remote - SSH: 使用 SHH 连接到远程机器/VM 并打开文件夹
>  - Dev Containers: 在容器中工作
>  - WSL
>  - Remote - Tunnels: 通过安全隧道连接到远程机器，不需要配置 SSH

While most VS Code extensions should work unmodified in a remote environment, extension authors can learn more at [Supporting Remote Development](https://code.visualstudio.com/api/advanced-topics/remote-extensions).
>  大多数 VS Code 拓展在远程环境都能正常工作，无需修改

## Remote tutorials
The tutorials below will walk you through running Visual Studio Code with the Remote Development extensions.

| Tutorial                                                                                                                     | Description                                                             |
| ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [Remote via SSH](https://code.visualstudio.com/docs/remote/ssh-tutorial)                                                     | Connect to remote and virtual machines with Visual Studio Code via SSH. |
| [Work in WSL](https://code.visualstudio.com/docs/remote/wsl-tutorial)                                                        | Run Visual Studio Code in Windows Subsystem for Linux.                  |
| [Develop in Containers](https://code.visualstudio.com/docs/devcontainers/tutorial)                                           | Run Visual Studio Code in a Docker Container.                           |
| [GitHub Codespaces](https://docs.github.com/github/developing-online-with-codespaces/using-codespaces-in-visual-studio-code) | Connect to a codespace with Visual Studio Code.                         |

## GitHub Codespaces
[GitHub Codespaces](https://code.visualstudio.com/docs/remote/codespaces) provides remote development environments that are managed for you. You can configure and create a development environment hosted in the cloud, which is spun up and available when you need it.

## Questions or feedback
- See [Tips and Tricks](https://code.visualstudio.com/docs/remote/troubleshooting) or the [FAQ](https://code.visualstudio.com/docs/remote/faq).
- Search on [Stack Overflow](https://stackoverflow.com/questions/tagged/vscode-remote).
- Add a [feature request](https://aka.ms/vscode-remote/feature-requests) or [report a problem](https://aka.ms/vscode-remote/issues/new).
