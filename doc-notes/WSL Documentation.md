# Overview
## What is the Windows Subsystem for Linux?
Windows Subsystem for Linux (WSL) is a feature of Windows that allows you to run a Linux environment on your Windows machine, without the need for a separate virtual machine or dual booting. WSL is designed to provide a seamless and productive experience for developers who want to use both Windows and Linux at the same time.

- Use WSL to install and run various Linux distributions, such as Ubuntu, Debian, Kali, and more. [Install Linux distributions](https://learn.microsoft.com/en-us/windows/wsl/install) and receive automatic updates from the [Microsoft Store](https://learn.microsoft.com/en-us/windows/wsl/compare-versions#wsl-in-the-microsoft-store), [import Linux distributions not available in the Microsoft Store](https://learn.microsoft.com/en-us/windows/wsl/use-custom-distro), or [build your own custom Linux distribution](https://learn.microsoft.com/en-us/windows/wsl/build-custom-distro).
- Store files in an isolated Linux file system, specific to the installed distribution.
- Run command-line tools, such as BASH.
- Run common BASH command-line tools such as `grep`, `sed`, `awk`, or other ELF-64 binaries.
- Run Bash scripts and GNU/Linux command-line applications including:
    - Tools: vim, emacs, tmux
    - Languages: [NodeJS](https://learn.microsoft.com/en-us/windows/nodejs/setup-on-wsl2), JavaScript, [Python](https://learn.microsoft.com/en-us/windows/python/web-frameworks), Ruby, C/C++, C# & F#, Rust, Go, etc.
    - Services: SSHD, [MySQL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-database), Apache, lighttpd, [MongoDB](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-database), [PostgreSQL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-database).
- Install additional software using your own GNU/Linux distribution package manager.
- Invoke Windows applications using a Unix-like command-line shell.
- Invoke GNU/Linux applications on Windows.
- [Run GNU/Linux graphical applications](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) integrated directly to your Windows desktop
- Use your device [GPU to accelerate Machine Learning workloads running on Linux.](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)

### What is WSL 2?
WSL 2 is the default distro type when installing a Linux distribution. WSL 2 uses virtualization technology to run a Linux kernel inside of a lightweight utility virtual machine (VM). Linux distributions run as isolated containers inside of the WSL 2 managed VM. Linux distributions running via WSL 2 will share the same network namespace, device tree (other than `/dev/pts`), CPU/Kernel/Memory/Swap, `/init` binary, but have their own PID namespace, Mount namespace, User namespace, Cgroup namespace, and `init` process.
> WSL2是安装 Linux 发布时获得的默认发布类型，Linux 发布运行于在 WSL2 管理的 VM 中的隔离容器中
> 通过 WSL2 运行的 Linux 发布将共享相同的网络命名空间、设备树、CPU/内核/内存/交换区、`/init` binary，但具有独立的 PID 命名空间、挂载空间、用户命名空间、Cgroup 命名空间和 `init` 进程

WSL 2 **increases file system performance** and adds **full system call compatibility** in comparison to the WSL 1 architecture. Learn more about how [WSL 1 and WSL 2 compare](https://learn.microsoft.com/en-us/windows/wsl/compare-versions).

Individual Linux distributions can be run with either the WSL 1 or WSL 2 architecture. Each distribution can be upgraded or downgraded at any time and you can run WSL 1 and WSL 2 distributions side by side. See the [Set WSL version command](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#set-default-wsl-version).

### Microsoft Loves Linux
Learn more about [Linux resources at Microsoft](https://learn.microsoft.com/en-us/linux), including Microsoft tools that run on Linux, Linux training courses, Cloud Solution Architecture for Linux, and Microsoft + Linux news, events, and partnerships. **Microsoft Loves Linux!**

## Comparing WSL Versions
### Comparing WSL 1 and WSL 2
This guide will compare WSL 1 and WSL 2, including [exceptions for using WSL 1 rather than WSL 2](https://learn.microsoft.com/en-us/windows/wsl/compare-versions#exceptions-for-using-wsl-1-rather-than-wsl-2). The primary differences between WSL 1 and WSL 2 are the use of an actual Linux kernel inside a managed VM, support for full system call compatibility, and performance across the Linux and Windows operating systems. WSL 2 is the current default version when installing a Linux distribution and uses the latest and greatest in virtualization technology to run a Linux kernel inside of a lightweight utility virtual machine (VM). WSL2 runs Linux distributions as isolated containers inside the managed VM. If your distribution is currently running WSL 1 and you want to update to WSL 2, see [update from WSL 1 to WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install#upgrade-version-from-wsl-1-to-wsl-2).
> WSL2 开始在 VM 中运行真实的 Linux 内核，因此支持完整的系统调用兼容性
#### Comparing features

|Feature|WSL 1|WSL 2|
|---|---|---|
|Integration between Windows and Linux|✅|✅|
|Fast boot times|✅|✅|
|Small resource foot print compared to traditional Virtual Machines|✅|✅|
|Runs with current versions of VMware and VirtualBox|✅|❌|
|Managed VM|❌|✅|
|Full Linux Kernel|❌|✅|
|Full system call compatibility|❌|✅|
|Performance across OS file systems|✅|❌|
|systemd support|❌|✅|
|IPv6 support|✅|✅|

As you can tell from the comparison table above, the WSL 2 architecture outperforms WSL 1 in several ways, with the exception of performance across OS file systems, which can be addressed by storing your project files on the same operating system as the tools you are running to work on the project.

WSL 2 is only available in Windows 11 or Windows 10, Version 1903, Build 18362 or later. Check your Windows version by selecting the **Windows logo key + R**, type **winver**, select **OK**. (Or enter the `ver` command in Windows Command Prompt). You may need to [update to the latest Windows version](ms-settings:windowsupdate). For builds lower than 14393, WSL is not supported at all.

For more info on the latest WSL 2 updates, see the [Windows Command Line blog](https://devblogs.microsoft.com/commandline/), including [Systemd support is now available in WSL](https://devblogs.microsoft.com/commandline/systemd-support-is-now-available-in-wsl/) and [WSL September 2023 update](https://devblogs.microsoft.com/commandline/windows-subsystem-for-linux-september-2023-update/) for more info on IPv6 support.

> [!Note]
>WSL 2 will work with [VMware 15.5.5+](https://blogs.vmware.com/workstation/2020/05/vmware-workstation-now-supports-hyper-v-mode.html) and although [VirtualBox 6+](https://www.virtualbox.org/wiki/Changelog-6.0) states that there is WSL support, there are still significant challenges that make it unsupported. Learn more in our [FAQs.](https://learn.microsoft.com/en-us/windows/wsl/faq#will-i-be-able-to-run-wsl-2-and-other-3rd-party-virtualization-tools-such-as-vmware--or-virtualbox-)

### What's new in WSL 2
WSL 2 is a major overhaul of the underlying architecture and uses virtualization technology and a Linux kernel to enable new features. The primary goals of this update are to increase file system performance and add full system call compatibility.

- [WSL 2 system requirements](https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-2---check-requirements-for-running-wsl-2)
- [Set your Linux distribution version from WSL 1 to WSL 2](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#set-wsl-version-to-1-or-2)
- [Frequently Asked Questions about WSL 2](https://learn.microsoft.com/en-us/windows/wsl/faq)

#### WSL 2 architecture
A traditional VM experience can be slow to boot up, is isolated, consumes a lot of resources, and requires your time to manage it. WSL 2 does not have these attributes.

WSL 2 provides the benefits of WSL 1, including seamless integration between Windows and Linux, fast boot times, a small resource footprint, and requires no VM configuration or management. While WSL 2 does use a VM, it is managed and run behind the scenes, leaving you with the same user experience as WSL 1.

#### Full Linux kernel
The Linux kernel in WSL 2 is built by Microsoft from the latest stable branch, based on the source available at kernel.org. This kernel has been specially tuned for WSL 2, optimizing for size and performance to provide an amazing Linux experience on Windows. The kernel will be serviced by Windows updates, which means you will get the latest security fixes and kernel improvements without needing to manage it yourself.

The [WSL 2 Linux kernel is open source](https://github.com/microsoft/WSL2-Linux-Kernel). If you'd like to learn more, check out the blog post [Shipping a Linux Kernel with Windows](https://devblogs.microsoft.com/commandline/shipping-a-linux-kernel-with-windows/) written by the team that built it.

Learn more in the [Release Notes for Windows Subsystem for Linux kernel](https://learn.microsoft.com/en-us/windows/wsl/kernel-release-notes).

#### Increased file IO performance
File intensive operations like git clone, npm install, apt update, apt upgrade, and more are all noticeably faster with WSL 2.

The actual speed increase will depend on which app you're running and how it is interacting with the file system. Initial versions of WSL 2 run up to 20x faster compared to WSL 1 when unpacking a zipped tarball, and around 2-5x faster when using git clone, npm install and cmake on various projects.

#### Full system call compatibility
Linux binaries use system calls to perform functions such as accessing files, requesting memory, creating processes, and more. Whereas WSL 1 used a translation layer that was built by the WSL team, WSL 2 includes its own Linux kernel with full system call compatibility. Benefits include:

- A whole new set of apps that you can run inside of WSL, such as **[Docker](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers)** and more.
- Any updates to the Linux kernel are immediately ready for use. (You don't have to wait for the WSL team to implement updates and add the changes).

### Exceptions for using WSL 1 rather than WSL 2
We recommend that you use WSL 2 as it offers faster performance and 100% system call compatibility. However, there are a few specific scenarios where you might prefer using WSL 1. Consider using WSL 1 if:

- Your project files must be stored in the Windows file system. WSL 1 offers faster access to files mounted from Windows.
    - If you will be using your WSL Linux distribution to access project files on the Windows file system, and these files cannot be stored on the Linux file system, you will achieve faster performance across the OS files systems by using WSL 1.
- A project which requires cross-compilation using both Windows and Linux tools on the same files.
    - File performance across the Windows and Linux operating systems is faster in WSL 1 than WSL 2, so if you are using Windows applications to access Linux files, you will currently achieve faster performance with WSL 1.
- Your project needs access to a serial port or USB device. _However,_ USB device support is now available for WSL 2 via the USBIPD-WIN project. See [Connect USB devices](https://learn.microsoft.com/en-us/windows/wsl/connect-usb) for set up steps.
- WSL 2 does not include support for accessing serial ports. Learn more in the [FAQs](https://learn.microsoft.com/en-us/windows/wsl/faq#can-i-access-the-gpu-in-wsl-2--are-there-plans-to-increase-hardware-support-) or in [WSL GitHub repo issue on serial support](https://github.com/microsoft/WSL/issues/4322).
- You have strict memory requirements
    - WSL 2's memory usage grows and shrinks as you use it. When a process frees memory this is automatically returned to Windows. However, as of right now WSL 2 does not yet release cached pages in memory back to Windows until the WSL instance is shut down. If you have long running WSL sessions, or access a very large amount of files, this cache can take up memory on Windows. We are tracking the work to improve this experience on [the WSL GitHub repository issue 4166](https://github.com/microsoft/WSL/issues/4166).
- For those using VirtualBox, be sure to use the latest version of both VirtualBox and WSL 2. See the [related FAQ](https://learn.microsoft.com/en-us/windows/wsl/faq#will-i-be-able-to-run-wsl-2-and-other-3rd-party-virtualization-tools-such-as-vmware--or-virtualbox-).
- If you rely on a Linux distribution to have an IP address in the same network as your host machine, you may need to set up a workaround in order to run WSL 2. WSL 2 is running as a hyper-v virtual machine. This is a change from the bridged network adapter used in WSL 1, meaning that WSL 2 uses a Network Address Translation (NAT) service for its virtual network, instead of making it bridged to the host Network Interface Card (NIC) resulting in a unique IP address that will change on restart. To learn more about the issue and workaround that forwards TCP ports of WSL 2 services to the host OS, see [WSL GitHub repository issue 4150, NIC Bridge mode (TCP Workaround)](https://github.com/microsoft/WSL/issues/4150).
> WSL2 以 hyper-v 虚拟机运行，WSL1 使用的是桥接的网络适配器，即会桥接到主机的网卡，而 WSL2 使用的是网络地址转换，因此在重新启动时都会获得独立的 IP 地址

>[!Note]
>Consider trying the VS Code [Remote WSL Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) to enable you to store your project files on the Linux file system, using Linux command line tools, but also using VS Code on Windows to author, edit, debug, or run your project in an internet browser without any of the performance slow-downs associated with working across the Linux and Windows file systems. [Learn more](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode).

### WSL in the Microsoft Store
WSL has lifted the update functionality out of the Windows OS Image into a package that is available via the Microsoft Store. This means faster updates and servicing as soon as they're available, rather than needing to wait for an update of your Windows operating system.

WSL was originally included in the Windows operating system as an optional component that need to be enabled in order to install a Linux distribution. WSL in the Store has the same user experience, and is the same product, but receives updates and servicing as a store package, rather than as an entire OS update. Beginning in Windows version 19044 or higher, running the `wsl.exe --install` command will install the WSL servicing update from the Microsoft Store. ([See the blog post announcing this update](https://devblogs.microsoft.com/commandline/the-windows-subsystem-for-linux-in-the-microsoft-store-is-now-generally-available-on-windows-10-and-11/)). If you are already using WSL, you can update to ensure that you're receiving the latest WSL features and servicing from the store by running `wsl.exe --update`.

## Basic commands for WSL
The WSL commands below are listed in a format supported by PowerShell or Windows Command Prompt. To run these commands from a Bash / Linux distribution command line, you must replace `wsl` with `wsl.exe`. For a full list of commands, run `wsl --help`. If you have not yet done so, we recommend [updating to the version of WSL installed from Microsoft Store](https://apps.microsoft.com/detail/9P9TQF7MRM4R) in order to receive WSL updates as soon as they are available. ([Learn more about installing WSL via Microsoft Store.](https://devblogs.microsoft.com/commandline/the-windows-subsystem-for-linux-in-the-microsoft-store-is-now-generally-available-on-windows-10-and-11/)).

### Install

```
wsl --install
```

Install WSL and the default Ubuntu distribution of Linux. [Learn more](https://learn.microsoft.com/en-us/windows/wsl/install). You can also use this command to install additional Linux distributions by running `wsl --install <Distribution Name>`. For a valid list of distribution names, run `wsl --list --online`.

Options include:

- `--distribution`: Specify the Linux distribution to install. You can find available distributions by running `wsl --list --online`.
- `--no-launch`: Install the Linux distribution but do not launch it automatically.
- `--web-download`: Install from an online source rather than using the Microsoft Store.

When WSL is not installed options include:

- `--inbox`: Installs WSL using the Windows component instead of using the Microsoft Store. _(WSL updates will be received via Windows updates, rather than pushed out as-available via the store)._
- `--enable-wsl1`: Enables WSL 1 during the install of the Microsoft Store version of WSL by also enabling the "Windows Subsystem for Linux" optional component.
- `--no-distribution`: Do not install a distribution when installing WSL.

>[!Note]
>If you run WSL on Windows 10 or an older version, you may need to include the `-d` flag with the `--install` command to specify a distribution: `wsl --install -d <distribution name>`.

### List available Linux distributions

```
wsl --list --online
```

See a list of the Linux distributions available through the online store. This command can also be entered as: `wsl -l -o`.

### List installed Linux distributions

```
wsl --list --verbose
```

See a list of the Linux distributions installed on your Windows machine, including the state (whether the distribution is running or stopped) and the version of WSL running the distribution (WSL 1 or WSL 2). [Comparing WSL 1 and WSL 2](https://learn.microsoft.com/en-us/windows/wsl/compare-versions). This command can also be entered as: `wsl -l -v`. Additional options that can be used with the list command include: `--all` to list all distributions, `--running` to list only distributions that are currently running, or `--quiet` to only show distribution names.

### Set WSL version to 1 or 2

```
wsl --set-version <distribution name> <versionNumber>
```

To designate the version of WSL (1 or 2) that a Linux distribution is running on, replace `<distribution name>` with the name of the distribution and replace `<versionNumber>` with 1 or 2. [Comparing WSL 1 and WSL 2](https://learn.microsoft.com/en-us/windows/wsl/compare-versions). WSL 2 is only available in Windows 11 or Windows 10, Version 1903, Build 18362 or later.

>[!Warning]
>Switching between WSL 1 and WSL 2 can be time-consuming and result in failures due to the differences between the two architectures. For distributions with large projects, we recommend backing up files before attempting a conversion.

### Set default WSL version

```
wsl --set-default-version <Version>
```

To set a default version of WSL 1 or WSL 2, replace `<Version>` with either the number 1 or 2. For example, `wsl --set-default-version 2`. The number represents the version of WSL to default to for new Linux distribution installations. [Comparing WSL 1 and WSL 2](https://learn.microsoft.com/en-us/windows/wsl/compare-versions). WSL 2 is only available in Windows 11 or Windows 10, Version 1903, Build 18362 or later.

### Set default Linux distribution

```
wsl --set-default <Distribution Name>
```

To set the default Linux distribution that WSL commands will use to run, replace `<Distribution Name>` with the name of your preferred Linux distribution.

### Change directory to home

```
wsl ~
```

The `~` can be used with wsl to start in the user's home directory. To jump from any directory back to home from within a WSL command prompt, you can use the command: `cd ~`.

### Run a specific Linux distribution from PowerShell or CMD

```
wsl --distribution <Distribution Name> --user <User Name>
```

To run a specific Linux distribution with a specific user, replace `<Distribution Name>` with the name of your preferred Linux distribution (ie. Debian) and `<User Name>` with the name of an existing user (ie. root). If the user doesn't exist in the WSL distribution, you will receive an error. To print the current user name, use the command `whoami`.

### Update WSL

```
wsl --update
```

Update your WSL version to the latest version. Options include:

- `--web-download`: Download the latest update from the GitHub rather than the Microsoft Store.

### Check WSL status

```
wsl --status
```

See general information about your WSL configuration, such as default distribution type, default distribution, and kernel version.

### Check WSL version

```
wsl --version
```

Check the version information about WSL and its components.

### Help command

```
wsl --help
```

See a list of options and commands available with WSL.

### Run as a specific user

```
wsl --user <Username>
```

To run WSL as a specified user, replace `<Username>` with the name of a user that exists in the WSL distribution.

### Change the default user for a distribution

```
<DistributionName> config --default-user <Username>
```

Change the default user for your distribution log-in. The user has to already exist inside the distribution in order to become the default user.

For example: `ubuntu config --default-user johndoe` would change the default user for the Ubuntu distribution to the "johndoe" user.

>[!Note]
>If you are having trouble figuring out the name of your distribution, use the command `wsl -l`.

>[!Warning]
>This command will not work for imported distributions, because these distributions do not have an executable launcher. You can instead change the default user for imported distributions using the `/etc/wsl.conf` file. See the Automount options in the [Advanced Settings Configuration](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#user-settings) doc.

### Shutdown

```
wsl --shutdown
```

Immediately terminates all running distributions and the WSL 2 lightweight utility virtual machine. This command may be necessary in instances that require you to restart the WSL 2 virtual machine environment, such as [changing memory usage limits](https://learn.microsoft.com/en-us/windows/wsl/disk-space) or making a change to your [.wslconfig file](https://learn.microsoft.com/en-us/windows/wsl/manage).
> 终止所有运行的发布以及 WSL2 轻量级功能虚拟机
> 需要重启 WSL2 虚拟机环境时，例如改变内存使用限制以及修改 `.wslconfig` 时，需要使用该命令

### Terminate

```powershell
wsl --terminate <Distribution Name>
```

To terminate the specified distribution, or stop it from running, replace `<Distribution Name>` with the name of the targeted distribution.
> 停止指定的运行中的发布

### Identify IP address

- `wsl hostname -I`: Returns the IP address of your Linux distribution installed via WSL 2 (the WSL 2 VM address)
> 返回 WSL2 安装的 Linux 发布的 IP 地址（也就是 WSL2 VM 的地址）

- `ip route show | grep -i default | awk '{ print $3}'`: Returns the IP address of the Windows machine as seen from WSL 2 (the WSL 2 VM)
> 返回 WSL2 VM 视角下的 Windows 机器的 IP 地址

For a more detailed explanation, see [Accessing network applications with WSL: Identify IP Address](https://learn.microsoft.com/en-us/windows/wsl/networking#identify-ip-address).

### Export a distribution

```
wsl --export <Distribution Name> <FileName>
```

Exports a snapshot of the specified distribution as a new distribution file. Defaults to tar format. The filename can be `-` for standard input. Options include:
> 将指定的发布导出为一个新的发布文件
> 文件格式默认为 tar

- `--vhd`: Specifies the export distribution should be a .vhdx file instead of a tar file (this is only supported using WSL 2)

### Import a distribution

```
wsl --import <Distribution Name> <InstallLocation> <FileName>
```

Imports the specified tar file as a new distribution. The filename can be `-` for standard input. Options include:
> 将指定的 tar 文件导入为一个新的发布

- `--vhd`: Specifies the import distribution should be a .vhdx file instead of a tar file (this is only supported using WSL 2)
- `--version <1/2>`: Specifies whether to import the distribution as a WSL 1 or WSL 2 distribution

### Import a distribution in place

```
wsl --import-in-place <Distribution Name> <FileName>
```

Imports the specified .vhdx file as a new distribution. The virtual hard disk must be formatted in the ext4 filesystem type.

### Unregister or uninstall a Linux distribution
While Linux distributions can be installed through the Microsoft Store, they can't be uninstalled through the store.

To unregister and uninstall a WSL distribution:

```
wsl --unregister <DistributionName>
```

Replacing `<DistributionName>` with the name of your targeted Linux distribution will unregister that distribution from WSL so it can be reinstalled or cleaned up. **Caution:** Once unregistered, all data, settings, and software associated with that distribution will be permanently lost. Reinstalling from the store will install a clean copy of the distribution. For example, `wsl --unregister Ubuntu` would remove Ubuntu from the distributions available in WSL. Running `wsl --list` will reveal that it is no longer listed.

You can also uninstall the Linux distribution app on your Windows machine just like any other store application. To reinstall, find the distribution in the Microsoft Store and select "Launch".

### Mount a disk or device

```
wsl --mount <DiskPath>
```

Attach and mount a physical disk in all WSL2 distributions by replacing `<DiskPath>` with the directory\file path where the disk is located. See [Mount a Linux disk in WSL 2](https://learn.microsoft.com/en-us/windows/wsl/wsl2-mount-disk). Options include:

- `--vhd`: Specifies that `<Disk>` refers to a virtual hard disk.
- `--name`: Mount the disk using a custom name for the mountpoint
- `--bare`: Attach the disk to WSL2, but don't mount it.
- `--type <Filesystem>`: Filesystem type to use when mounting a disk, if not specified defaults to ext4. This command can also be entered as: `wsl --mount -t <Filesystem>`.You can detect the filesystem type using the command: `blkid <BlockDevice>`, for example: `blkid <dev/sdb1>`.
- `--partition <Partition Number>`: Index number of the partition to mount, if not specified defaults to the whole disk.
- `--options <MountOptions>`: There are some filesystem-specific options that can be included when mounting a disk. For example, [ext4 mount options](https://www.kernel.org/doc/Documentation/filesystems/ext4.txt) like: `wsl --mount -o "data-ordered"` or `wsl --mount -o "data=writeback`. However, only filesystem-specific options are supported at this time. Generic options, such as `ro`, `rw`, or `noatime`, are not supported.

>[!Note]
>If you're running a 32-bit process in order to access wsl.exe (a 64-bit tool), you may need to run the command in the following manner: `C:\Windows\Sysnative\wsl.exe --command`.

### Unmount disks

```
wsl --unmount <DiskPath>
```

Unmount a disk given at the disk path, if no disk path is given then this command will unmount and detach ALL mounted disks.
> 默认卸载所有挂载的磁盘

### Deprecated WSL commands

```
wslconfig.exe [Argument] [Options]
```

```
bash [Options]
```

```
lxrun /[Argument]
```

These commands were the original wsl syntax for configuring Linux distributions installed with WSL, but have been replaced with the `wsl` or `wsl.exe` command syntax.

# Install
## How to install Linux on Windows with WSL
Developers can access the power of both Windows and Linux at the same time on a Windows machine. The Windows Subsystem for Linux (WSL) lets developers install a Linux distribution (such as Ubuntu, OpenSUSE, Kali, Debian, Arch Linux, etc) and use Linux applications, utilities, and Bash command-line tools directly on Windows, unmodified, without the overhead of a traditional virtual machine or dualboot setup.

### Prerequisites
You must be running Windows 10 version 2004 and higher (Build 19041 and higher) or Windows 11 to use the commands below. If you are on earlier versions please see [the manual install page](https://learn.microsoft.com/en-us/windows/wsl/install-manual).

### Install WSL command
You can now install everything you need to run WSL with a single command. Open PowerShell or Windows Command Prompt in **administrator** mode by right-clicking and selecting "Run as administrator", enter the wsl --install command, then restart your machine.

```
wsl --install
```

This command will enable the features necessary to run WSL and install the Ubuntu distribution of Linux. ([This default distribution can be changed](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#install)).

If you're running an older build, or just prefer not to use the install command and would like step-by-step directions, see **[WSL manual installation steps for older versions](https://learn.microsoft.com/en-us/windows/wsl/install-manual)**.

The first time you launch a newly installed Linux distribution, a console window will open and you'll be asked to wait for files to de-compress and be stored on your machine. All future launches should take less than a second.

Note
The above command only works if WSL is not installed at all. If you run `wsl --install` and see the WSL help text, please try running `wsl --list --online` to see a list of available distros and run `wsl --install -d <DistroName>` to install a distro. To uninstall WSL, see [Uninstall legacy version of WSL](https://learn.microsoft.com/en-us/windows/wsl/troubleshooting#uninstall-legacy-version-of-wsl) or [unregister or uninstall a Linux distribution](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#unregister-or-uninstall-a-linux-distribution).

### Change the default Linux distribution installed
By default, the installed Linux distribution will be Ubuntu. This can be changed using the `-d` flag.

- To change the distribution installed, enter: `wsl --install -d <Distribution Name>`. Replace `<Distribution Name>` with the name of the distribution you would like to install.
- To see a list of available Linux distributions available for download through the online store, enter: `wsl --list --online` or `wsl -l -o`.
- To install additional Linux distributions after the initial install, you may also use the command: `wsl --install -d <Distribution Name>`.

Tip
If you want to install additional distributions from inside a Linux/Bash command line (rather than from PowerShell or Command Prompt), you must use .exe in the command: `wsl.exe --install -d <Distribution Name>` or to list available distributions: `wsl.exe -l -o`.
>如果要从 Linux/Bash 命令行安装额外的发行版 (而不是从 PowerShell 或命令提示符)，需要在命令中使用 `wsl.exe`：`wsl.exe --install -d <发行版名称>` 或者列出可用的发行版：`wsl.exe -l -o`

If you run into an issue during the install process, check the [installation section of the troubleshooting guide](https://learn.microsoft.com/en-us/windows/wsl/troubleshooting#installation-issues).

To install a Linux distribution that is not listed as available, you can [import any Linux distribution](https://learn.microsoft.com/en-us/windows/wsl/use-custom-distro) using a TAR file. Or in some cases, [as with Arch Linux](https://wsldl-pg.github.io/ArchW-docs/How-to-Setup/), you can install using an `.appx` file. You can also create your own [custom Linux distribution](https://learn.microsoft.com/en-us/windows/wsl/build-custom-distro) to use with WSL.

### Set up your Linux user info
Once you have installed WSL, you will need to create a user account and password for your newly installed Linux distribution. See the [Best practices for setting up a WSL development environment](https://learn.microsoft.com/en-us/windows/wsl/setup/environment#set-up-your-linux-username-and-password) guide to learn more.

### Set up and best practices
We recommend following our [Best practices for setting up a WSL development environment](https://learn.microsoft.com/en-us/windows/wsl/setup/environment) guide for a step-by-step walk-through of how to set up a user name and password for your installed Linux distribution(s), using basic WSL commands, installing and customizing Windows Terminal, set up for Git version control, code editing and debugging using the VS Code remote server, good practices for file storage, setting up a database, mounting an external drive, setting up GPU acceleration, and more.

### Check which version of WSL you are running
You can list your installed Linux distributions and check the version of WSL each is set to by entering the command: `wsl -l -v` in PowerShell or Windows Command Prompt.

To set the default version to WSL 1 or WSL 2 when a new Linux distribution is installed, use the command: `wsl --set-default-version <Version#>`, replacing `<Version#>` with either 1 or 2.

To set the default Linux distribution used with the `wsl` command, enter: `wsl -s <DistributionName>` or `wsl --set-default <DistributionName>`, replacing `<DistributionName>` with the name of the Linux distribution you would like to use. For example, from PowerShell/CMD, enter: `wsl -s Debian` to set the default distribution to Debian. Now running `wsl npm init` from Powershell will run the `npm init` command in Debian.

To run a specific wsl distribution from within PowerShell or Windows Command Prompt without changing your default distribution, use the command: `wsl -d <DistributionName>`, replacing `<DistributionName>` with the name of the distribution you want to use.

Learn more in the guide to [Basic commands for WSL](https://learn.microsoft.com/en-us/windows/wsl/basic-commands).

### Upgrade version from WSL 1 to WSL 2

New Linux installations, installed using the `wsl --install` command, will be set to WSL 2 by default.

The `wsl --set-version` command can be used to downgrade from WSL 2 to WSL 1 or to update previously installed Linux distributions from WSL 1 to WSL 2.

To see whether your Linux distribution is set to WSL 1 or WSL 2, use the command: `wsl -l -v`.

To change versions, use the command: `wsl --set-version <distro name> 2` replacing `<distro name>` with the name of the Linux distribution that you want to update. For example, `wsl --set-version Ubuntu-20.04 2` will set your Ubuntu 20.04 distribution to use WSL 2.

If you manually installed WSL prior to the `wsl --install` command being available, you may also need to [enable the virtual machine optional component](https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-3---enable-virtual-machine-feature) used by WSL 2 and [install the kernel package](https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package) if you haven't already done so.

To learn more, see the [Command reference for WSL](https://learn.microsoft.com/en-us/windows/wsl/basic-commands) for a list of WSL commands, [Comparing WSL 1 and WSL 2](https://learn.microsoft.com/en-us/windows/wsl/compare-versions) for guidance on which to use for your work scenario, or [Best practices for setting up a WSL development environment](https://learn.microsoft.com/en-us/windows/wsl/setup/environment) for general guidance on setting up a good development workflow with WSL.

### Ways to run multiple Linux distributions with WSL
WSL supports running as many different Linux distributions as you would like to install. This can include choosing distributions from the [Microsoft Store](https://aka.ms/wslstore), [importing a custom distribution](https://learn.microsoft.com/en-us/windows/wsl/use-custom-distro), or [building your own custom distribution](https://learn.microsoft.com/en-us/windows/wsl/build-custom-distro).

There are several ways to run your Linux distributions once installed:

- [Install Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/get-started) _**(Recommended)**_ Using Windows Terminal supports as many command lines as you would like to install and enables you to open them in multiple tabs or window panes and quickly switch between multiple Linux distributions or other command lines (PowerShell, Command Prompt, Azure CLI, etc). You can fully customize your terminal with unique color schemes, font styles, sizes, background images, and custom keyboard shortcuts. [Learn more.](https://learn.microsoft.com/en-us/windows/terminal)
- You can directly open your Linux distribution by visiting the Windows Start menu and typing the name of your installed distributions. For example: "Ubuntu". This will open Ubuntu in its own console window.
- From Windows Command Prompt or PowerShell, you can enter the name of your installed distribution. For example: `ubuntu`
- From Windows Command Prompt or PowerShell, you can open your default Linux distribution inside your current command line, by entering: `wsl.exe`.
- From Windows Command Prompt or PowerShell, you can use your default Linux distribution inside your current command line, without entering a new one, by entering:`wsl [command]`. Replacing `[command]` with a WSL command, such as: `wsl -l -v` to list installed distributions or `wsl pwd` to see where the current directory path is mounted in wsl. From PowerShell, the command `get-date` will provide the date from the Windows file system and `wsl date` will provide the date from the Linux file system.

The method you select should depend on what you're doing. If you've opened a WSL command line within a Windows Prompt or PowerShell window and want to exit, enter the command: `exit`.

### Want to try the latest WSL preview features?
Try the most recent features or updates to WSL by joining the [Windows Insiders Program](https://insider.windows.com/getting-started). Once you have joined Windows Insiders, you can choose the channel you would like to receive preview builds from inside the Windows settings menu to automatically receive any WSL updates or preview features associated with that build. You can choose from:

- Dev channel: Most recent updates, but low stability.
- Beta channel: Ideal for early adopters, more reliable builds than the Dev channel.
- Release Preview channel: Preview fixes and key features on the next version of Windows just before its available to the general public.

# Tutorials
## Set up a WSL development environment
A step-by-step guide to the best practices for setting up a WSL development environment. Learn how to run the command to install the default Bash shell that uses Ubuntu or can be set to install other Linux distributions, use basic WSL commands, set up Visual Studio Code or Visual Studio, Git, Windows Credential Manager, databases like MongoDB, Postgres, or MySQL, set up GPU acceleration, run GUI apps, and more.

### Get started
Windows Subsystem for Linux comes with the Windows operating system, but you must enable it and install a Linux distribution before you can begin using it.

To use the simplified --install command, you must be running a recent build of Windows (Build 20262+). To check your version and build number, select **Windows logo key + R**, type **winver**, select **OK**. You can update using the [Settings menu](ms-settings:windowsupdate) or [Windows Update Assistant](https://www.microsoft.com/software-download/).

If you prefer to install a Linux distribution other than Ubuntu, or would prefer to complete these steps manually, see the [WSL installation page](https://learn.microsoft.com/en-us/windows/wsl/install) for more details.

Open PowerShell (or Windows Command Prompt) and enter:

```
wsl --install
```

The --install command performs the following actions:

- Enables the optional WSL and Virtual Machine Platform components
- Downloads and installs the latest Linux kernel
- Sets WSL 2 as the default
- Downloads and installs the Ubuntu Linux distribution (reboot may be required)

You will need to restart your machine during this installation process.

![PowerShell command line running wsl --install](https://learn.microsoft.com/en-us/windows/wsl/media/wsl-install.png)

Check the [troubleshooting installation](https://learn.microsoft.com/en-us/windows/wsl/troubleshooting) article if you run into any issues.

### Set up your Linux username and password
Once the process of installing your Linux distribution with WSL is complete, open the distribution (Ubuntu by default) using the Start menu. You will be asked to create a **User Name** and **Password** for your Linux distribution.

- This **User Name** and **Password** is specific to each separate Linux distribution that you install and has no bearing on your Windows user name.
- Please note that whilst entering the **Password**, nothing will appear on screen. This is called blind typing. You won't see what you are typing, this is completely normal.
- Once you create a **User Name** and **Password**, the account will be your default user for the distribution and automatically sign-in on launch.
- This account will be considered the Linux administrator, with the ability to run `sudo` (Super User Do) administrative commands.
- Each Linux distribution running on WSL has its own Linux user accounts and passwords. You will have to configure a Linux user account every time you add a distribution, reinstall, or reset.

Note
Linux distributions installed with WSL are a per-user installation and can't be shared with other Windows user accounts. Encountering a username error? [StackExchange: What characters should I use or not use in usernames on Linux?](https://serverfault.com/questions/73084/what-characters-should-i-use-or-not-use-in-usernames-on-linux)

![Ubuntu command line enter UNIX username](https://learn.microsoft.com/en-us/windows/wsl/media/ubuntuinstall.png)

To change or reset your password, open the Linux distribution and enter the command: `passwd`. You will be asked to enter your current password, then asked to enter your new password, and then to confirm your new password.

If you forgot the password for your Linux distribution:

1. Open PowerShell and enter the root of your default WSL distribution using the command: `wsl -u root`
    > If you need to update the forgotten password on a distribution that is not your default, use the command: `wsl -d Debian -u root`, replacing `Debian` with the name of your targeted distribution.
2. Once your WSL distribution has been opened at the root level inside PowerShell, you can use this command to update your password: `passwd <username>` where `<username>` is the username of the account in the distribution whose password you've forgotten.
3. You will be prompted to enter a new UNIX password and then confirm that password. Once you're told that the password has updated successfully, close WSL inside of PowerShell using the command: `exit`.

> 最初创建的和平时直接登录 WSL 的都是管理员账户
> ` wsl -u root ` 用于登录根用户
> 登录根用户后可以用 `passwd <username>` 重置任意用户的密码

### Update and upgrade packages
We recommend that you regularly update and upgrade your packages using the preferred package manager for the distribution. For Ubuntu or Debian, use the command:

```
sudo apt update && sudo apt upgrade
```

Windows does not automatically update or upgrade your Linux distribution(s). This is a task that most Linux users prefer to control themselves.
> 建议规律性使用 Linux 发布的包管理程序更新并且升级包
> 对于 Ubuntu 或 Debian，就是使用 `sudo apt update && sudo apt upgrade`

### Add additional distributions
To add additional Linux distributions, you can install via the [Microsoft Store](https://aka.ms/wslstore), via the [--import command](https://learn.microsoft.com/en-us/windows/wsl/use-custom-distro), or by [sideloading your own custom distribution](https://learn.microsoft.com/en-us/windows/wsl/build-custom-distro). You may also want to [set up custom WSL images for distribution across your enterprise company](https://learn.microsoft.com/en-us/windows/wsl/enterprise).

### Set up Windows Terminal
Windows Terminal can run any application with a command line interface. Its main features include multiple tabs, panes, Unicode and UTF-8 character support, a GPU accelerated text rendering engine, and the ability to create your own themes and customize text, colors, backgrounds, and shortcuts.

Whenever a new WSL Linux distribution is installed, a new instance will be created for it inside the Windows Terminal that can be customized to your preferences.

We recommend using WSL with Windows Terminal, especially if you plan to work with multiple command lines. See the Windows Terminal docs for help with setting it up and customizing your preferences, including:

- [Install Windows Terminal or Windows Terminal (Preview)](https://learn.microsoft.com/en-us/windows/terminal/get-started) from the Microsoft Store
- [Use the Command Palette](https://learn.microsoft.com/en-us/windows/terminal/get-started#invoke-the-command-palette)
- Set up [custom actions](https://learn.microsoft.com/en-us/windows/terminal/#custom-actions) like keyboard shortcuts to make the terminal feel natural to your preferences
- Set up the [default startup profile](https://learn.microsoft.com/en-us/windows/terminal/customize-settings/startup)
- Customize the appearance: [theme](https://learn.microsoft.com/en-us/windows/terminal/customize-settings/appearance#theme), [color schemes](https://learn.microsoft.com/en-us/windows/terminal/customize-settings/color-schemes), [name and starting directory](https://learn.microsoft.com/en-us/windows/terminal/customize-settings/profile-general), [background image](https://learn.microsoft.com/en-us/windows/terminal/customize-settings/profile-appearance#background-image), etc.
- Learn how to use [command line arguments](https://learn.microsoft.com/en-us/windows/terminal/command-line-arguments?tabs=windows) like opening a terminal with multiple command lines split into window panes or tabs
- Learn about the [search feature](https://learn.microsoft.com/en-us/windows/terminal/search)
- Find [tips and tricks](https://learn.microsoft.com/en-us/windows/terminal/tips-and-tricks), like how to rename or color a tab, use mouse interactions, or enable "Quake mode"
- Find tutorials on how to set up [a customized command prompt](https://learn.microsoft.com/en-us/windows/terminal/tutorials/custom-prompt-setup), [SSH profiles](https://learn.microsoft.com/en-us/windows/terminal/tutorials/ssh), or [tab titles](https://learn.microsoft.com/en-us/windows/terminal/tutorials/tab-title)
- Find a [custom terminal gallery](https://learn.microsoft.com/en-us/windows/terminal/custom-terminal-gallery/custom-schemes) and a [troubleshooting guide](https://learn.microsoft.com/en-us/windows/terminal/troubleshooting)

![Windows Terminal screenshot](https://learn.microsoft.com/en-us/windows/wsl/media/terminal.png)

### File storage
- To open your WSL project in Windows File Explorer, enter: `explorer.exe .` 
    _Be sure to add the period at the end of the command to open the current directory._
    
- [Store your project files on the same operating system as the tools you plan to use](https://learn.microsoft.com/en-us/windows/wsl/filesystems#file-storage-and-performance-across-file-systems).  
    For the fastest performance speed, store your files in the WSL file system if you are working on them with Linux tools in a Linux command line (Ubuntu, OpenSUSE, etc). If you're working in a Windows command line (PowerShell, Command Prompt) with Windows tools, store your files in the Windows file system. Files can be accessed across the operating systems, but it may significantly slow down performance.

For example, when storing your WSL project files:

- Use the Linux file system root directory: `\\wsl$\<DistroName>\home\<UserName>\Project`
- Not the Windows file system root directory: `C:\Users\<UserName>\Project` or `/mnt/c/Users/<UserName>/Project$`

![Windows File Explorer displaying Linux storage](https://learn.microsoft.com/en-us/windows/wsl/media/windows-file-explorer.png)


### Set up your favorite code editor
We recommend using Visual Studio Code or Visual Studio, as they directly support remote development and debugging with WSL. Visual Studio Code allows you to use WSL as a full-featured development environment. Visual Studio offers native WSL support for C++ cross-platform development.

#### Use Visual Studio Code
Follow this step-by-step guide to [Get started using Visual Studio Code with WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode), which includes installing the [Remote Development extension pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack). This extension enables you to run WSL, SSH, or a development container for editing and debugging with the full set of Visual Studio Code features. Quickly swap between different, separate development environments and make updates without worrying about impacting your local machine.

Once VS Code is installed and set up, you can open your WSL project with a VS Code remote server by entering: `code .`

_Be sure to add the period at the end of the command to open the current directory._

![VS Code with WSL extensions displayed](https://learn.microsoft.com/en-us/windows/wsl/media/vscode-remote-wsl-extensions.png)

#### Use Visual Studio
Follow this step-by-step guide to [Get started using Visual Studio with WSL for C++ cross-platform development](https://learn.microsoft.com/en-us/cpp/build/walkthrough-build-debug-wsl2). Visual Studio 2022 enables you to build and debug CMake projects on Windows, WSL distributions, and SSH connections from the same instance of Visual Studio.

![Select a target system in Visual Studio 2022](https://learn.microsoft.com/en-us/windows/wsl/media/vs-target-system.png)

### Set up version management with Git
Follow this step-by-step guide to [Get started using Git on WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-git) and connect your project to the Git version control system, along with using the credential manager for authentication, using Git Ignore files, understanding Git line endings, and using the Git commands built-in to VS Code.

![Displaying git version in the command line](https://learn.microsoft.com/en-us/windows/wsl/media/git-versions.gif)

### Set up remote development containers with Docker
Follow this step-by-step guide to [Get started with Docker remote containers on WSL 2](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers) and connect your project to a remote development container with Docker Desktop for Windows.

![Docker Desktop screenshot](https://learn.microsoft.com/en-us/windows/wsl/media/docker-running.png)

### Set up a database
Follow this step-by-step guide to [Get started with databases on WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-database) and connect your project to a database in the WSL environment. Get started with MySQL, PostgreSQL, MongoDB, Redis, Microsoft SQL Server, or SQLite.

![Running MongoDB in Ubuntu via WSL](https://learn.microsoft.com/en-us/windows/wsl/media/mongodb.png)

### Set up GPU acceleration for faster performance
Follow this step-by-step guide to set up [GPU accelerated machine learning training in WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute) and leverage your computer's GPU (graphics processing unit) to accelerate performance heavy workloads.

![Running GPU acceleration with WSL](https://learn.microsoft.com/en-us/windows/wsl/media/gpu-acceleration.gif)

### Basic WSL commands
The Linux distributions that you install via WSL are best managed using PowerShell or Windows Command Prompt (CMD). See the [WSL command reference guide](https://learn.microsoft.com/en-us/windows/wsl/basic-commands) for a list of basic commands to be familiar with when using WSL.

In addition, many commands are interoperable between Windows and Linux. Here are a couple of examples:

- [Run Linux tools from a Windows command line](https://learn.microsoft.com/en-us/windows/wsl/filesystems#run-linux-tools-from-a-windows-command-line): Open PowerShell and display the directory contents of `C:\temp>` using the Linux `ls -la` command by entering: `wsl ls -la`
- [Mix Linux and Windows commands](https://learn.microsoft.com/en-us/windows/wsl/filesystems#mixing-linux-and-windows-commands): In this example, the Linux command `ls -la` is used to list files in the directory, then the PowerShell command `findstr` is used to filter the results for words containing "git": `wsl ls -la | findstr "git"`. This could also be done mixing the Windows `dir` command with the Linux `grep` command: `dir | wsl grep git`.
- [Run a Windows tool directly from the WSL command line](https://learn.microsoft.com/en-us/windows/wsl/filesystems#run-windows-tools-from-linux): `<tool-name>.exe` For example, to open your .bashrc file (the shell script that runs whenever your Linux command line is started), enter: `notepad.exe .bashrc`
- [Run the Windows ipconfig.exe tool with the Linux Grep tool](https://learn.microsoft.com/en-us/windows/wsl/filesystems#run-windows-tools-from-linux): From Bash enter the command `ipconfig.exe | grep IPv4 | cut -d: -f2` or from PowerShell enter `ipconfig.exe | wsl grep IPv4 | wsl cut -d: -f2` This example demonstrates the ipconfig tool on the Windows file system being used to display the current TCP/IP network configuration values and then being filtered to only the IPv4 result with grep, a Linux tool.

### Mount an external drive or USB
Follow this step-by-step guide to [Get started mounting a Linux disk in WSL 2](https://learn.microsoft.com/en-us/windows/wsl/wsl2-mount-disk).

![wsl mount command screenshot](https://learn.microsoft.com/en-us/windows/wsl/media/wslmountsimple.png)

### Run Linux GUI apps
Follow this tutorial to learn how to set up and [run Linux GUI apps on WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps).

