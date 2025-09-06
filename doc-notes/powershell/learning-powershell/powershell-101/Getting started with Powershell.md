---
completed: true
version: "7.5"
---
# Chapter 1 - Getting started with PowerShell

- 08/03/2024

This chapter focuses on finding and launching PowerShell and solving the initial pain points that new users experience with PowerShell. Follow along and walk through the examples in this chapter on your lab environment computer.

## What is PowerShell?
Windows PowerShell is an easy-to-use command-line shell and scripting environment for automating administrative tasks of Windows-based systems. Windows PowerShell is preinstalled on all modern versions of the Windows operating system.

## Where to find PowerShell
The easiest way to find PowerShell on Windows 11 is to type `PowerShell` into the search bar, as shown in Figure 1-1. Notice that there are four different shortcuts for Windows PowerShell.

![Figure 1-1 - Search for PowerShell.](https://learn.microsoft.com/en-us/powershell/docs-conceptual/learn/ps101/media/figure1-1.jpg?view=powershell-7.5)

Windows PowerShell shortcuts on a 64-bit version of Windows:

- Windows PowerShell
- Windows PowerShell ISE
- Windows PowerShell (x86)
- Windows PowerShell ISE (x86)

On a 64-bit version of Windows, you have a 64-bit version of the Windows PowerShell console and the Windows PowerShell Integrated Scripting Environment (ISE) and a 32-bit version of each one, as indicated by the (x86) suffix on the shortcuts.

>  Windows 自带了 PowerShell 控制台、PowerShell 集成脚本环境 (ISE)，以及它们的 32bit 版本 (后缀为 x86)

Note
Windows 11 only ships as a 64-bit operating system. There is no 32-bit version of Windows 11. However, Windows 11 includes 32-bit versions of Windows PowerShell and the Windows PowerShell ISE.

You only have two shortcuts if you're running an older 32-bit version of Windows. Those shortcuts don't have the (x86) suffix but are 32-bit versions.

I recommend using the 64-bit version of Windows PowerShell if you're running a 64-bit operating system unless you have a specific reason for using the 32-bit version.

Depending on what version of Windows 11 you're running, Windows PowerShell might open in [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/).

Microsoft no longer updates the PowerShell ISE. The ISE only works with Windows PowerShell 5.1. [Visual Studio Code](https://code.visualstudio.com/) (VS Code) with the [PowerShell extension](https://code.visualstudio.com/docs/languages/powershell) works with both versions of PowerShell. VS Code and the PowerShell extension don't ship in Windows. Install VS Code and the extension on the computer where you create PowerShell scripts. You don't need to install them on all the computers where you run PowerShell.
>  PowerShell ISE 不再更新

## How to launch PowerShell
I use three different Active Directory user accounts in the production environments I support. I mirrored those accounts in the lab environment used in this book. I sign into my Windows 11 computer as a domain user without domain or local administrator rights.

Launch the PowerShell console by clicking the **Windows PowerShell** shortcut, as shown in Figure 1-1. Notice that the title bar of the console says **Windows PowerShell**, as shown in Figure 1-2.

![Figure 1-2 - Title bar of PowerShell window.](https://learn.microsoft.com/en-us/powershell/docs-conceptual/learn/ps101/media/figure1-2.jpg?view=powershell-7.5)

Some commands run fine when you run PowerShell as an ordinary user. However, PowerShell doesn't participate in [User Access Control (UAC)](https://learn.microsoft.com/en-us/windows/security/application-security/application-control/user-account-control/). That means it's unable to prompt for elevation for tasks that require the approval of an administrator.
>  PowerShell 不参与用户访问控制，这意味着它无法在需要管理员批准的任务中提示提升权限

Note
UAC is a Windows security feature that helps prevent malicious code from running with elevated privileges.

When signed on as an ordinary user, PowerShell returns an error when you run a command that requires elevation. For example, stopping a Windows service:

PowerShell

```
Stop-Service -Name W32Time
```

Output

```
Stop-Service : Service 'Windows Time (W32Time)' cannot be stopped due to
the following error: Cannot open W32Time service on computer '.'.
At line:1 char:1
+ Stop-Service -Name W32Time
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : CloseError: (System.ServiceProcess.ServiceCon
   troller:ServiceController) [Stop-Service], ServiceCommandException
    + FullyQualifiedErrorId : CouldNotStopService,Microsoft.PowerShell.Comm
   ands.StopServiceCommand
```

The solution is to run PowerShell elevated as a user who is a local administrator. That's how I configured my second domain user account. Following the principle of least privilege, this account shouldn't be a domain administrator or have any elevated privileges in the domain.

To start PowerShell with elevated rights, right-click the **Windows PowerShell** shortcut and select **Run as administrator**, as shown in Figure 1-3.

![Figure 1-3 - Context menu - Run as administrator.](https://learn.microsoft.com/en-us/powershell/docs-conceptual/learn/ps101/media/figure1-3.jpg?view=powershell-7.5)

Windows prompts you for credentials because you logged into Windows as an ordinary user. Enter the credentials of your domain user who is a local administrator, as shown in Figure 1-4.

![Figure 1-4 - User account control - Enter credentials.](https://learn.microsoft.com/en-us/powershell/docs-conceptual/learn/ps101/media/figure1-4.png?view=powershell-7.5)

Notice that the title bar of the elevated console windows says **Administrator: Windows PowerShell**, as shown in Figure 1-5.

![Figure 1-5 - Title bar of elevated PowerShell window.](https://learn.microsoft.com/en-us/powershell/docs-conceptual/learn/ps101/media/figure1-5.jpg?view=powershell-7.5)

Now that you're running PowerShell elevated as an administrator, UAC is no longer a problem when you run a command that requires elevation.

Important
You should only run PowerShell elevated as an administrator when absolutely necessary.

When you target remote computers, there's no need to run PowerShell elevated. Running PowerShell elevated only affects commands that run against your local computer.

You can simplify finding and launching PowerShell. Pin the PowerShell or Windows Terminal shortcut to your taskbar. Search for PowerShell again, except this time right-click on it and select **Pin to taskbar** as shown in Figure 1-6.

![Figure 1-6 - Context menu - Pin to taskbar.](https://learn.microsoft.com/en-us/powershell/docs-conceptual/learn/ps101/media/figure1-6.jpg?view=powershell-7.5)

Important
The original version of this book, published in 2017, recommended pinning a shortcut to the taskbar to launch an elevated instance automatically every time you start PowerShell. However, due to potential security concerns, I no longer recommend it. Any applications you launch from an elevated instance of PowerShell also bypass UAC and run elevated. For example, if you launch a web browser from an elevated instance of PowerShell, any website you visit containing malicious code also runs elevated.
>  管理员运行的 PowerShell 发起的任意应用都会绕过 UAC

When you need to run PowerShell with elevated permissions, right-click the PowerShell shortcut pinned to your taskbar while pressing Shift. Select **Run as administrator**, as shown in Figure 1-7.

![Figure 1-7 - Context menu - Run as administrator.](https://learn.microsoft.com/en-us/powershell/docs-conceptual/learn/ps101/media/figure1-7.jpg?view=powershell-7.5)


## Determine your version of PowerShell
There are automatic variables in PowerShell that store state information. One of these variables is `$PSVersionTable`, which contains version information about your PowerShell session.

PowerShell

```
$PSVersionTable
```

Output

```
Name                           Value
----                           -----
PSVersion                      5.1.22621.2428
PSEdition                      Desktop
PSCompatibleVersions           {1.0, 2.0, 3.0, 4.0...}
BuildVersion                   10.0.22621.2428
CLRVersion                     4.0.30319.42000
WSManStackVersion              3.0
PSRemotingProtocolVersion      2.3
SerializationVersion           1.1.0.1
```

>  变量 `$PSVersionTable` 存储了 PowerShell 版本信息

If you're running a version of Windows PowerShell older than 5.1, you should update your version of Windows. Windows PowerShell 5.1 is preinstalled on the currently supported versions of Windows.

PowerShell version 7 isn't a replacement for Windows PowerShell 5.1; it installs side-by-side with Windows PowerShell. Windows PowerShell version 5.1 and PowerShell version 7 are two different products. For more information about the differences between Windows PowerShell version 5.1 and PowerShell version 7, see [Migrating from Windows PowerShell 5.1 to PowerShell 7](https://learn.microsoft.com/en-us/powershell/scripting/whats-new/migrating-from-windows-powershell-51-to-powershell-7).
>  Windows PowerShell 5.1 和 PowerShell 7 是两个不同的产品
>  前者仅针对 Windows，后者是在前者经验上开发的跨平台，更强的工具

Tip
PowerShell version 6, formerly known as PowerShell Core, is no longer supported.

## Execution policy
PowerShell execution policy controls the conditions under which you can run PowerShell scripts. The execution policy in PowerShell is a safety feature designed to help prevent the unintentional execution of malicious scripts. However, it's not a security boundary because it can't stop determined users from deliberately running scripts. A determined user can bypass the execution policy in PowerShell.
>  PowerShell 执行策略控制了在哪些情况下可以运行 PowerShell 脚本

You can set an execution policy for the local computer, current user, or a PowerShell session. You can also set execution policies for users and computers with Group Policy.

The following table shows the default execution policy for current Windows operating systems.

|Windows Operating System Version|Default Execution Policy|
|---|---|
|Windows Server 2022|Remote Signed|
|Windows Server 2019|Remote Signed|
|Windows Server 2016|Remote Signed|
|Windows 11|Restricted|
|Windows 10|Restricted|

Regardless of the execution policy setting, you can run any PowerShell command interactively. The execution policy only affects commands running in a script. Use the `Get-ExecutionPolicy` cmdlet to determine the current execution policy setting.
>  `Get-ExecutionPolicy` 获取当前执行策略设置

Check the execution policy setting on your computer.

PowerShell

```
Get-ExecutionPolicy
```

Output

```
Restricted
```

List the execution policy settings for all scopes.

PowerShell

```
Get-ExecutionPolicy -List
```

Output

```
        Scope ExecutionPolicy
        ----- ---------------
MachinePolicy       Undefined
   UserPolicy       Undefined
      Process       Undefined
  CurrentUser       Undefined
 LocalMachine       Undefined
```

All Windows client operating systems have the default execution policy setting of `Restricted`. You can't run PowerShell scripts using the `Restricted` execution policy setting. To test the execution policy, save the following code as a `.ps1` file named `Get-TimeService.ps1`.

Tip
A PowerShell script is a plaintext file that contains the commands you want to run. PowerShell script files use the `.ps1` file extension. To create a PowerShell script, use a code editor like Visual Studio Code (VS Code) or any text editor such as Notepad.
>  PowerShell 脚本为纯文本文件，后缀为 `.ps1`

When you run the following command interactively, it completes without error.

PowerShell

```
Get-Service -Name W32Time
```

However, PowerShell returns an error when you run the same command from a script.

PowerShell

```
.\Get-TimeService.ps1
```

OutputCopy

```
.\Get-TimeService.ps1 : File C:\tmp\Get-TimeService.ps1 cannot be loaded
because running scripts is disabled on this system. For more information,
see about_Execution_Policies at
https:/go.microsoft.com/fwlink/?LinkID=135170.
At line:1 char:1
+ .\Get-TimeService.ps1
+ ~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : SecurityError: (:) [], PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
```

When you run a command in PowerShell that generates an error, read the error message before retrying the command. Notice the error message tells you why the command failed:

> _... running scripts is disabled on this system._

To enable the execution of scripts, change the execution policy with the `Set-ExecutionPolicy` cmdlet. `LocalMachine` is the default scope when you don't specify the **Scope** parameter. You must run PowerShell elevated as an administrator to change the execution policy for the local machine. Unless you're signing your scripts, I recommend using the `RemoteSigned` execution policy. `RemoteSigned` prevents you from running downloaded scripts that aren't signed by a trusted publisher.
>  `Set-ExecutionPolicy` 改变执行策略
>  默认的作用域是 `LocalMachine`
>  推荐使用 `RemoteSigned` 执行策略，这避免我们运行由不是信任的发布者发布，并被我们下载下来的脚本

Before you change the execution policy, read the [about_Execution_Policies](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies) help article to understand the security implications.

Change the execution policy setting on your computer to `RemoteSigned`.

PowerShell

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

If you have successfully changed the execution policy, PowerShell displays the following warning:

Output

```
Execution Policy Change
The execution policy helps protect you from scripts that you do not trust.
Changing the execution policy might expose you to the security risks
described in the about_Execution_Policies help topic at
https:/go.microsoft.com/fwlink/?LinkID=135170. Do you want to change the
execution policy?
[Y] Yes  [A] Yes to All  [N] No  [L] No to All  [S] Suspend  [?] Help
(default is "N"):y
```

If you're not running PowerShell elevated as an administrator, PowerShell returns the following error message:

Output

```
Set-ExecutionPolicy : Access to the registry key 'HKEY_LOCAL_MACHINE\SOFTWAR
E\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell' is denied. To
change the execution policy for the default (LocalMachine) scope, start
Windows PowerShell with the "Run as administrator" option. To change the
execution policy for the current user, run "Set-ExecutionPolicy -Scope
CurrentUser".
At line:1 char:1
+ Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : PermissionDenied: (:) [Set-ExecutionPolicy],
   UnauthorizedAccessException
    + FullyQualifiedErrorId : System.UnauthorizedAccessException,Microsoft.
   PowerShell.Commands.SetExecutionPolicyCommand
```

It's also possible to change the execution policy for the current user without requiring you to run PowerShell elevated as an administrator. This step is unnecessary if you successfully set the execution policy for the local machine to `RemoteSigned`.

PowerShell

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

With the execution policy set to `RemoteSigned`, the `Get-TimeService.ps1` script runs successfully.

PowerShell

```
.\Get-TimeService.ps1
```

Output

```
Status   Name               DisplayName
------   ----               -----------
Running  W32Time            Windows Time
```

## Summary
In this chapter, you learned where to find and how to launch PowerShell. You also learned how to determine the version of PowerShell and the purpose of execution policies.

## Review

1. How do you determine what PowerShell version a computer is running?
2. When should you launch PowerShell elevated as an administrator?
3. What's the default execution policy on Windows client computers, and what does it prevent you from doing?
4. How do you determine the current PowerShell execution policy setting?
5. How do you change the PowerShell execution policy?

## References
To learn more about the concepts covered in this chapter, read the following PowerShell help articles.

- [about_Automatic_Variables](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_automatic_variables)
- [about_Execution_Policies](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies)

## Next steps
In the next chapter, you'll learn about the discoverability of commands in PowerShell. You'll also learn how to download PowerShell's help files so you can view the help in your PowerShell session.

# Chapter 2 - The Help system

- 06/28/2024

In an experiment designed to assess proficiency in PowerShell, two distinct groups of IT professionals — beginners and experts — were first given a written examination without access to a computer. Surprisingly, the test scores indicated comparable skills across both groups. A subsequent test was then administered, mirroring the first but with one key difference: participants had access to an offline computer equipped with PowerShell. The results revealed a significant skills gap between the two groups this time.

What factors contributed to the outcomes observed between the two assessments?

> Experts don't always know the answers, but they know how to figure out the answers.

>  专家并不总是知道答案，但是他们知道如何找到答案

The outcomes observed in the results of the two tests were because experts don't memorize thousands of PowerShell commands. Instead, they excel at using the Help system within PowerShell, enabling them to discover and learn how to use commands when necessary.

> Becoming proficient with the Help system is the key to success with PowerShell.

>  要熟练 PowerShell，关键就是熟练它的帮助系统

I heard Jeffrey Snover, the creator of PowerShell, share a similar story on multiple occasions.

## Discoverability
Compiled commands in PowerShell are known as cmdlets, pronounced as _"command-let"_, not _"CMD-let"_. The naming convention for cmdlets follows a singular **Verb-Noun** format to make them easily discoverable. For instance, `Get-Process` is the cmdlet to determine what processes are running, and `Get-Service` is the cmdlet to retrieve a list of services. Functions, also known as script cmdlets, and aliases are other types of PowerShell commands that are discussed later in this book. The term _"PowerShell command"_ describes any command in PowerShell, regardless of whether it's a cmdlet, function, or alias.
>  PowerShell 中编译好的命令叫做 cmdlet
>  cmdlet 的命名遵循动词-名词形式，例如 `Get-Process` 是用于查找当前运行进程的 cmdlet，`Get-Services` 是用户获取服务列表的 cmdlet
>  函数 (也称为脚本 cmdlet)，以及别名，是 PowerShell 中的其他命令类型
>  "PowerShell Command"代指 PowerShell 中的任何命令，无论是 cmdlet、函数还是别名

You can also run operating system native commands from PowerShell, such as traditional command-line programs like `ping.exe` and `ipconfig.exe`.
>  PowerShell 也可以运行 OS 原生的命令，例如传统的命令行程序 `ping.exe, ipconfig.exe`

## The three core cmdlets in PowerShell

- `Get-Help`
- `Get-Command`
- `Get-Member` (covered in chapter 3)

I'm often asked: _"How do you figure out what the commands are in PowerShell?"_. Both `Get-Help` and `Get-Command` are invaluable resources for discovering and understanding commands in PowerShell.

>  三个核心 cmdlet:
>  - `Get-Help`
>  - `Get-Command`
>  - `Get-Member`

## Get-Help
The first thing you need to know about the Help system in PowerShell is how to use the `Get-Help` cmdlet.

`Get-Help` is a multipurpose command that helps you learn how to use commands once you find them. You can also use `Get-Help` to locate commands, but in a different and more indirect way when compared to `Get-Command`.
>  `Get-Help` 用于获取命令的帮助，也可以用于找到命令，但是寻找命令更直接的方式是 `Get-Command`

When using `Get-Help` to locate commands, it initially performs a wildcard search for command names based on your input. If that doesn't find any matches, it conducts a comprehensive full-text search across all PowerShell help articles on your system. If that also fails to find any results, it returns an error.
>  使用 `Get-Help` 定位命令时，它首先基于输入对所有的命令名称执行一次通配符搜索，如果没有找到，对系统中所有的 PowerShell 帮助文章执行一次全文本搜素哦，如果没有找到，返回错误

Here's how to use `Get-Help` to view the help content for the `Get-Help` cmdlet.

PowerShell

```
Get-Help -Name Get-Help
```

Beginning with PowerShell version 3.0, the help content doesn't ship preinstalled with the operating system. When you run `Get-Help` for the first time, a message asks if you want to download the PowerShell help files to your computer.

Answering **Yes** by pressing Y executes the `Update-Help` cmdlet, downloading the help content.

Output

```
Do you want to run Update-Help?
The Update-Help cmdlet downloads the most current Help files for Windows
PowerShell modules, and installs them on your computer. For more information
about the Update-Help cmdlet, see
https:/go.microsoft.com/fwlink/?LinkId=210614.
[Y] Yes  [N] No  [S] Suspend  [?] Help (default is "Y"):
```

If you don't receive this message, run `Update-Help` from an elevated PowerShell session running as an administrator.

Once the update is complete, the help article is displayed.

Take a moment to run the example on your computer, review the output, and observe how the help system organizes the information.

- NAME
- SYNOPSIS
- SYNTAX
- DESCRIPTION
- RELATED LINKS
- REMARKS

As you review the output, keep in mind that help articles often contain a vast amount of information, and what you see by default isn't the entire help article.

>  帮助系统按照以下部分来组织信息:
>  - 名称
>  - 概要
>  - 语法
>  - 详细描述
>  - 相关链接
>  - 备注

### Parameters
When you run a command in PowerShell, you might need to provide additional information or input to the command. Parameters allow you to specify options and arguments that change the behavior of a command. The **SYNTAX** section of each help article outlines the available parameters for the command.

`Get-Help` has several parameters that you can specify to return the entire help article or a subset for a command. To view all the available parameters for `Get-Help`, see the **SYNTAX** section of its help article, as shown in the following example.

>  SYNTAX 部分展示了命令接收哪些参数

Output

```
...
SYNTAX
    Get-Help [[-Name] <System.String>] [-Category {Alias | Cmdlet | Provider
    | General | FAQ | Glossary | HelpFile | ScriptCommand | Function |
    Filter | ExternalScript | All | DefaultHelp | Workflow | DscResource |
    Class | Configuration}] [-Component <System.String[]>] [-Full]
    [-Functionality <System.String[]>] [-Path <System.String>] [-Role
    <System.String[]>] [<CommonParameters>]

    Get-Help [[-Name] <System.String>] [-Category {Alias | Cmdlet | Provider
    | General | FAQ | Glossary | HelpFile | ScriptCommand | Function |
    Filter | ExternalScript | All | DefaultHelp | Workflow | DscResource |
    Class | Configuration}] [-Component <System.String[]>] -Detailed
    [-Functionality <System.String[]>] [-Path <System.String>] [-Role
    <System.String[]>] [<CommonParameters>]

    Get-Help [[-Name] <System.String>] [-Category {Alias | Cmdlet | Provider
    | General | FAQ | Glossary | HelpFile | ScriptCommand | Function |
    Filter | ExternalScript | All | DefaultHelp | Workflow | DscResource |
    Class | Configuration}] [-Component <System.String[]>] -Examples
    [-Functionality <System.String[]>] [-Path <System.String>] [-Role
    <System.String[]>] [<CommonParameters>]

    Get-Help [[-Name] <System.String>] [-Category {Alias | Cmdlet | Provider
    | General | FAQ | Glossary | HelpFile | ScriptCommand | Function |
    Filter | ExternalScript | All | DefaultHelp | Workflow | DscResource |
    Class | Configuration}] [-Component <System.String[]>] [-Functionality
    <System.String[]>] -Online [-Path <System.String>] [-Role
    <System.String[]>] [<CommonParameters>]

    Get-Help [[-Name] <System.String>] [-Category {Alias | Cmdlet | Provider
    | General | FAQ | Glossary | HelpFile | ScriptCommand | Function |
    Filter | ExternalScript | All | DefaultHelp | Workflow | DscResource |
    Class | Configuration}] [-Component <System.String[]>] [-Functionality
    <System.String[]>] -Parameter <System.String> [-Path <System.String>]
    [-Role <System.String[]>] [<CommonParameters>]

    Get-Help [[-Name] <System.String>] [-Category {Alias | Cmdlet | Provider
    | General | FAQ | Glossary | HelpFile | ScriptCommand | Function |
    Filter | ExternalScript | All | DefaultHelp | Workflow | DscResource |
    Class | Configuration}] [-Component <System.String[]>] [-Functionality
    <System.String[]>] [-Path <System.String>] [-Role <System.String[]>]
    -ShowWindow [<CommonParameters>]
...
```

### Parameter sets
When you review the **SYNTAX** section for `Get-Help`, notice that the information appears to be repeated six times. Each of those blocks is an individual parameter set, indicating the `Get-Help` cmdlet features six distinct sets of parameters. A closer look reveals each parameter set contains at least one unique parameter, making it different from the others.
>  SYNTAX 重复了六个 block，意味着 `Get-Help` 有六个独立的参数组，每个参数组至少包含一个独一无二的参数，使得参数组之间可以相互区分

Parameter sets are mutually exclusive. Once you specify a unique parameter that only exists in one parameter set, PowerShell limits you to using the parameters contained within that parameter set. For instance, you can't use the **Full** and **Detailed** parameters of `Get-Help` together because they belong to different parameter sets.
>  参数组之间是互斥的

Each of the following parameters belongs to a different parameter set for the `Get-Help` cmdlet.

- Full
- Detailed
- Examples
- Online
- Parameter
- ShowWindow

### The command syntax
If you're new to PowerShell, comprehending the cryptic information — characterized by square and angle brackets — in the **SYNTAX** section might seem overwhelming. However, learning these syntax elements is essential to becoming proficient with PowerShell. The more frequently you use the PowerShell Help system, the easier it becomes to remember all the nuances.

View the syntax of the `Get-EventLog` cmdlet.

PowerShell

```
Get-Help Get-EventLog
```

The following output shows the relevant portion of the help article.

OutputCopy

```
...
SYNTAX
    Get-EventLog [-LogName] <System.String> [[-InstanceId]
    <System.Int64[]>] [-After <System.DateTime>] [-AsBaseObject] [-Before
    <System.DateTime>] [-ComputerName <System.String[]>] [-EntryType {Error
    | Information | FailureAudit | SuccessAudit | Warning}] [-Index
    <System.Int32[]>] [-Message <System.String>] [-Newest <System.Int32>]
    [-Source <System.String[]>] [-UserName <System.String[]>]
    [<CommonParameters>]

    Get-EventLog [-AsString] [-ComputerName <System.String[]>] [-List]
    [<CommonParameters>]
...
```

The syntax information includes pairs of square brackets (`[]`). Depending on their usage, these square brackets serve two different purposes.

- Elements enclosed in square brackets are optional.
- An empty set of square brackets following a datatype, such as `<string[]>`, indicates that the parameter can accept multiple values passed as an array or a collection object.

>  `[]` 中的内容是可选的
>  `<string[]>` 这样的形式表示该参数可以接收多个值，以数组形式存储

### Positional parameters
Some cmdlets are designed to accept positional parameters. Positional parameters allow you to provide a value without specifying the name of the parameter. When using a parameter positionally, you must specify its value in the correct position on the command line. You can find the positional information for a parameter in the **PARAMETERS** section of a command's help article. When you explicitly specify parameter names, you can use the parameters in any order.
>  PARAMETER 部分包含了位置参数的信息

For the `Get-EventLog` cmdlet, the first parameter in the first parameter set is **LogName**. **LogName** is enclosed in square brackets, indicating it's a positional parameter.

>  使用 `[]` 包围表示 `-LogName` 是位置参数，我们可以通过名称指定，也可以直接在直接写下参数的值

Syntax

```
Get-EventLog [-LogName] <System.String>
```

Since **LogName** is a positional parameter, you can specify it by either name or position. According to the angle brackets following the parameter name, the value for **LogName** must be a single string. The absence of square brackets enclosing both the parameter name and datatype indicates that **LogName** is a required parameter within this particular parameter set.
>  根据后面的 `<>`，可以知道 `LogName` 的值是单个字符串，并且可以知道在这个参数组下，参数 `LogName` 是必须的

The second parameter in that parameter set is **InstanceId**. Both the parameter name and datatype are entirely enclosed in square brackets, signifying that **InstanceId** is an optional parameter.
>  如果参数名和 datatype 都在 `[]` 内，就表示该参数完全可选

SyntaxCopy

```
[[-InstanceId] <System.Int64[]>]
```

Furthermore, **InstanceId** has its own pair of square brackets, indicating that it's a positional parameter similar to the **LogName** parameter. Following the datatype, an empty set of square brackets implies that **InstanceId** can accept multiple values.
>  此外，`[-InstanceId]` 的 `[]` 也表示它可以作为位置参数

### Switch parameters
A parameter that doesn't require a value is called a switch parameter. You can easily identify switch parameters because there's no datatype following the parameter name. When you specify a switch parameter, its value is `true`. When you don't specify a switch parameter, its value is `false`.
>  不要求值的参数称为开关参数，可以看到它后面没有跟着数据类型，它的值只会是 `true, false`

The second parameter set includes a **List** parameter, which is a switch parameter. When you specify the **List** parameter, it returns a list of event logs on the local computer.

Syntax

```
[-List]
```

### A simplified approach to syntax
There's a more user-friendly method to obtain the same information as the cryptic command syntax for some commands, except in plain English. PowerShell returns the complete help article when using `Get-Help` with the **Full** parameter, making it easier to understand a command's usage.
>  `Get-Help -Full` 可以获取完整的帮助文档

PowerShell

```
Get-Help -Name Get-Help -Full
```

Take a moment to run the example on your computer, review the output, and observe how the help system organizes the information.

- NAME
- SYNOPSIS
- SYNTAX
- DESCRIPTION
- PARAMETERS
- INPUTS
- OUTPUTS
- NOTES
- EXAMPLES
- RELATED LINKS

By specifying the **Full** parameter with the `Get-Help` cmdlet, the output includes several extra sections. Among these sections, **PARAMETERS** often provides a detailed explanation for each parameter. However, the extent of this information varies depending on the specific command you're investigating.
>  `Get-Help -Full` 会输出更多 sections
>  其中 PARAMETER 会为每个参数提供详细解释

Output

```
...
    -Detailed <System.Management.Automation.SwitchParameter>
        Adds parameter descriptions and examples to the basic help display.
        This parameter is effective only when the help files are installed
        on the computer. It has no effect on displays of conceptual ( About_
        ) help.

        Required?                    true
        Position?                    named
        Default value                False
        Accept pipeline input?       False
        Accept wildcard characters?  false

    -Examples <System.Management.Automation.SwitchParameter>
        Displays only the name, synopsis, and examples. This parameter is
        effective only when the help files are installed on the computer. It
        has no effect on displays of conceptual ( About_ ) help.

        Required?                    true
        Position?                    named
        Default value                False
        Accept pipeline input?       False
        Accept wildcard characters?  false

    -Full <System.Management.Automation.SwitchParameter>
        Displays the entire help article for a cmdlet. Full includes
        parameter descriptions and attributes, examples, input and output
        object types, and additional notes.

        This parameter is effective only when the help files are installed
        on the computer. It has no effect on displays of conceptual ( About_
        ) help.

        Required?                    false
        Position?                    named
        Default value                False
        Accept pipeline input?       False
        Accept wildcard characters?  false
...
```

When you ran the previous command to display the help for the `Get-Help` command, you probably noticed the output scrolled by too quickly to read it.

If you're using the PowerShell console, Windows Terminal, or VS Code and need to view a help article, the `help` function can be useful. It pipes the output of `Get-Help` to `more.com`, displaying one page of help content at a time. I recommend using the `help` function instead of the `Get-Help` cmdlet because it provides a better user experience and it's less to type.
>  `help` 函数会将 `Get-Help` 的输出交给 `more.com`，一次展示一页内容
>  推荐使用 `help`

Note
The ISE doesn't support using `more.com`, so running `help` works the same way as `Get-Help`.

Run each of the following commands in PowerShell on your computer.

PowerShell

```
Get-Help -Name Get-Help -Full
help -Name Get-Help -Full
help Get-Help -Full
```

Did you observe any variations in the output when you ran the previous commands?

In the previous example, the first line uses the `Get-Help` cmdlet, the second uses the `help` function, and the third line omits the **Name** parameter while using the `help` function. Since **Name** is a positional parameter, the third example takes advantage of its position instead of explicitly stating the parameter's name.

The difference is that the last two commands display their output one page at a time. When using the `help` function, press the Spacebar to display the next page of content or Q to quit. If you need to terminate any command running interactively in PowerShell, press Ctrl+C.

To quickly find information about a specific parameter, use the **Parameter** parameter. This approach returns content containing only the parameter-specific information, rather than the entire help article. This is the easiest way to find information about a specific parameter.

The following example uses the `help` function with the **Parameter** parameter to return information from the help article for the **Name** parameter of `Get-Help`.

>  使用 `-Parameter` 获取关于特定参数的信息

PowerShell

```
help Get-Help -Parameter Name
```

The help information shows that the **Name** parameter is positional and must be specified in the first position (position zero) when used positionally.

OutputCopy

```
-Name <System.String>
    Gets help about the specified command or concept. Enter the name of a cmdlet, function, provider, script, or workflow, such as `Get-Member`, a conceptual article name, such as `about_Objects`, or an alias, such as `ls`. Wildcard characters are permitted in cmdlet and provider names, but you can't use wildcard characters to find the names of function help and script help articles.

    To get help for a script that isn't located in a path that's listed in the `$env:Path` environment variable, type the script's path and file name.

    If you enter the exact name of a help article, `Get-Help` displays the article contents.

    If you enter a word or word pattern that appears in several help
    article titles, `Get-Help` displays a list of the matching titles.

    If you enter any text that doesn't match any help article titles, `Get-Help` displays a list of articles that include that text in their contents.

    The names of conceptual articles, such as `about_Objects`, must be
    entered in English, even in non-English versions of PowerShell.

    Required?                    false
    Position?                    0
    Default value                None
    Accept pipeline input?       True (ByPropertyName)
    Accept wildcard characters?  true
```

The **Name** parameter expects a string value as identified by the `<String>` datatype next to the parameter name.

There are several other parameters you can specify with `Get-Help` to return a subset of a help article. To see how they work, run the following commands on your computer.

PowerShell

```
Get-Help -Name Get-Command -Full
Get-Help -Name Get-Command -Detailed
Get-Help -Name Get-Command -Examples
Get-Help -Name Get-Command -Online
Get-Help -Name Get-Command -Parameter Noun
Get-Help -Name Get-Command -ShowWindow
```

I typically use `help <command name>` with the **Full** or **Online** parameter. If you only have an interest in the examples, use the **Examples** parameter. If you only have an interest in a specific parameter, use the **Parameter** parameter.

When you use the **ShowWindow** parameter, it displays the help content in a separate searchable window. You can move that window to a different monitor if you have multiple monitors. However, the **ShowWindow** parameter has a known bug that might prevent it from displaying the entire help article. The **ShowWindow** parameter also requires an operating system with a Graphical User Interface (GUI). It returns an error when you attempt to use it on Windows Server Core.
>  `ShowWindow` 在一个额外的窗口展示帮助内容

If you have internet access, you can use the **Online** parameter instead. The **Online** parameter opens the help article in your default web browser. The online content is the most up-to-date content. The browser allows you to search the help content and view other related help articles.

Note
The **Online** parameter isn't supported for **About** articles.

PowerShell

```
help Get-Command -Online
```

### Finding commands with Get-Help
To find commands with `Get-Help`, specify a search term surrounded by asterisk (`*`) wildcard characters for the value of the **Name** parameter. The following example uses the **Name** parameter positionally.
>  为 `Name` 参数指定通配符，可以借助 `Get-Help` 搜素命令

PowerShell

```
help *process*
```

Output

```
Name                              Category  Module                    Synops
----                              --------  ------                    ------
Enter-PSHostProcess               Cmdlet    Microsoft.PowerShell.Core Con...
Exit-PSHostProcess                Cmdlet    Microsoft.PowerShell.Core Clo...
Get-PSHostProcessInfo             Cmdlet    Microsoft.PowerShell.Core Get...
Debug-Process                     Cmdlet    Microsoft.PowerShell.M... Deb...
Get-Process                       Cmdlet    Microsoft.PowerShell.M... Get...
Start-Process                     Cmdlet    Microsoft.PowerShell.M... Sta...
Stop-Process                      Cmdlet    Microsoft.PowerShell.M... Sto...
Wait-Process                      Cmdlet    Microsoft.PowerShell.M... Wai...
Invoke-LapsPolicyProcessing       Cmdlet    LAPS                      Inv...
ConvertTo-ProcessMitigationPolicy Cmdlet    ProcessMitigations        Con...
Get-ProcessMitigation             Cmdlet    ProcessMitigations        Get...
Set-ProcessMitigation             Cmdlet    ProcessMitigations        Set...
```

In this scenario, you aren't required to add the `*` wildcard characters. If `Get-Help` can't find a command matching the value you provided, it does a full-text search for that value. The following example produces the same results as specifying the `*` wildcard character on each end of `process`.
>  也可以不指定通配符，如果 `get-help` 找不到匹配给定名字的命令，会执行全文搜素，结果和指定 `*process*` 是一样的

PowerShell

```
help process
```

When you specify a wildcard character within the value, `Get-Help` only searches for commands that match the pattern you provided. It doesn't perform a full-text search. The following command doesn't return any results.

PowerShell

```
help pr*cess
```

PowerShell generates an error if you specify a value that begins with a dash without enclosing it in quotes because it interprets it as a parameter name. No such parameter name exists for the `Get-Help` cmdlet.

PowerShell

```
help -process
```

If you're attempting to search for commands that end with `-process`, you must add an `*` to the beginning of the value.

PowerShell

```
help *-process
```

When you search for PowerShell commands with `Get-Help`, it's better to be vague rather than too specific.
>  使用 `get-help` 搜索命令时，提供模糊的描述比提供精确的描述更好

When you searched for `process` earlier, the results only returned commands that included `process` in their name. But if you search for `processes`, it doesn't find any matches for command names. As previously stated, when help doesn't find any matches, it performs a comprehensive full-text search of every help article on your system and returns those results. This type of search often produces more results than expected, including information not relevant to you.

PowerShell

```
help processes
```

Output

```
Name                              Category  Module                    Synops
----                              --------  ------                    ------
Disconnect-PSSession              Cmdlet    Microsoft.PowerShell.Core Dis...
Enter-PSHostProcess               Cmdlet    Microsoft.PowerShell.Core Con...
ForEach-Object                    Cmdlet    Microsoft.PowerShell.Core Per...
Get-PSHostProcessInfo             Cmdlet    Microsoft.PowerShell.Core Get...
Get-PSSessionConfiguration        Cmdlet    Microsoft.PowerShell.Core Get...
New-PSSessionOption               Cmdlet    Microsoft.PowerShell.Core Cre...
New-PSTransportOption             Cmdlet    Microsoft.PowerShell.Core Cre...
Out-Host                          Cmdlet    Microsoft.PowerShell.Core Sen...
Start-Job                         Cmdlet    Microsoft.PowerShell.Core Sta...
Where-Object                      Cmdlet    Microsoft.PowerShell.Core Sel...
Debug-Process                     Cmdlet    Microsoft.PowerShell.M... Deb...
Get-Process                       Cmdlet    Microsoft.PowerShell.M... Get...
Get-WmiObject                     Cmdlet    Microsoft.PowerShell.M... Get...
Start-Process                     Cmdlet    Microsoft.PowerShell.M... Sta...
Stop-Process                      Cmdlet    Microsoft.PowerShell.M... Sto...
Wait-Process                      Cmdlet    Microsoft.PowerShell.M... Wai...
Clear-Variable                    Cmdlet    Microsoft.PowerShell.U... Del...
Convert-String                    Cmdlet    Microsoft.PowerShell.U... For...
ConvertFrom-Csv                   Cmdlet    Microsoft.PowerShell.U... Con...
ConvertFrom-Json                  Cmdlet    Microsoft.PowerShell.U... Con...
ConvertTo-Html                    Cmdlet    Microsoft.PowerShell.U... Con...
ConvertTo-Xml                     Cmdlet    Microsoft.PowerShell.U... Cre...
Debug-Runspace                    Cmdlet    Microsoft.PowerShell.U... Sta...
Export-Csv                        Cmdlet    Microsoft.PowerShell.U... Con...
Export-FormatData                 Cmdlet    Microsoft.PowerShell.U... Sav...
Format-List                       Cmdlet    Microsoft.PowerShell.U... For...
Format-Table                      Cmdlet    Microsoft.PowerShell.U... For...
Get-Unique                        Cmdlet    Microsoft.PowerShell.U... Ret...
Group-Object                      Cmdlet    Microsoft.PowerShell.U... Gro...
Import-Clixml                     Cmdlet    Microsoft.PowerShell.U... Imp...
Import-Csv                        Cmdlet    Microsoft.PowerShell.U... Cre...
Measure-Object                    Cmdlet    Microsoft.PowerShell.U... Cal...
Out-File                          Cmdlet    Microsoft.PowerShell.U... Sen...
Out-GridView                      Cmdlet    Microsoft.PowerShell.U... Sen...
Select-Object                     Cmdlet    Microsoft.PowerShell.U... Sel...
Set-Variable                      Cmdlet    Microsoft.PowerShell.U... Set...
Sort-Object                       Cmdlet    Microsoft.PowerShell.U... Sor...
Tee-Object                        Cmdlet    Microsoft.PowerShell.U... Sav...
Trace-Command                     Cmdlet    Microsoft.PowerShell.U... Con...
Write-Information                 Cmdlet    Microsoft.PowerShell.U... Spe...
Export-BinaryMiLog                Cmdlet    CimCmdlets                Cre...
Get-CimAssociatedInstance         Cmdlet    CimCmdlets                Ret...
Get-CimInstance                   Cmdlet    CimCmdlets                Get...
Import-BinaryMiLog                Cmdlet    CimCmdlets                Use...
Invoke-CimMethod                  Cmdlet    CimCmdlets                Inv...
New-CimInstance                   Cmdlet    CimCmdlets                Cre...
Remove-CimInstance                Cmdlet    CimCmdlets                Rem...
Set-CimInstance                   Cmdlet    CimCmdlets                Mod...
Compress-Archive                  Function  Microsoft.PowerShell.A... Cre...
Get-Counter                       Cmdlet    Microsoft.PowerShell.D... Get...
Invoke-WSManAction                Cmdlet    Microsoft.WSMan.Manage... Inv...
Remove-WSManInstance              Cmdlet    Microsoft.WSMan.Manage... Del...
Get-WSManInstance                 Cmdlet    Microsoft.WSMan.Manage... Dis...
New-WSManInstance                 Cmdlet    Microsoft.WSMan.Manage... Cre...
Set-WSManInstance                 Cmdlet    Microsoft.WSMan.Manage... Mod...
about_Arithmetic_Operators        HelpFile
about_Arrays                      HelpFile
about_Environment_Variables       HelpFile
about_Execution_Policies          HelpFile
about_Functions                   HelpFile
about_Jobs                        HelpFile
about_Logging                     HelpFile
about_Methods                     HelpFile
about_Objects                     HelpFile
about_Pipelines                   HelpFile
about_Preference_Variables        HelpFile
about_Remote                      HelpFile
about_Remote_Jobs                 HelpFile
about_Session_Configuration_Files HelpFile
about_Simplified_Syntax           HelpFile
about_Switch                      HelpFile
about_Variables                   HelpFile
about_Variable_Provider           HelpFile
about_Windows_PowerShell_5.1      HelpFile
about_WQL                         HelpFile
about_WS-Management_Cmdlets       HelpFile
about_Foreach-Parallel            HelpFile
about_Parallel                    HelpFile
about_Sequence                    HelpFile
```

When you searched for `process`, it returned 12 results. However, when searching for `processes`, it produced 78 results. If your search only finds one match, `Get-Help` displays the help content instead of listing the search results.
>  如果搜索仅找到一个匹配，`get-help` 就会直接展示帮助内容

PowerShell

```
help *hotfix*
```

Output

```
NAME
    Get-HotFix

SYNOPSIS
    Gets the hotfixes that are installed on local or remote computers.


SYNTAX
    Get-HotFix [-ComputerName <System.String[]>] [-Credential
    <System.Management.Automation.PSCredential>] [-Description
    <System.String[]>] [<CommonParameters>]

    Get-HotFix [[-Id] <System.String[]>] [-ComputerName <System.String[]>]
    [-Credential <System.Management.Automation.PSCredential>]
    [<CommonParameters>]


DESCRIPTION
    > This cmdlet is only available on the Windows platform. The
    `Get-HotFix` cmdlet uses the Win32_QuickFixEngineering WMI class to
    list hotfixes that are installed on the local computer or specified
    remote computers.


RELATED LINKS
    Online Version: https://learn.microsoft.com/powershell/module/microsoft.
    powershell.management/get-hotfix?view=powershell-5.1&WT.mc_id=ps-gethelp
    about_Arrays
    Add-Content
    Get-ComputerRestorePoint
    Get-Credential
    Win32_QuickFixEngineering class

REMARKS
    To see the examples, type: "Get-Help Get-HotFix -Examples".
    For more information, type: "Get-Help Get-HotFix -Detailed".
    For technical information, type: "Get-Help Get-HotFix -Full".
    For online help, type: "Get-Help Get-HotFix -Online"
```

You can also find commands that lack help articles with `Get-Help`, although this capability isn't commonly known. The `more` function is one of the commands that doesn't have a help article. To confirm that you can find commands with `Get-Help` that don't include help articles, use the `help` function to find `more`.

PowerShell

```
help *more*
```

The search only found one match, so it returned the basic syntax information you see when a command doesn't have a help article.

OutputCopy

```
NAME
    more

SYNTAX
    more [[-paths] <string[]>]

ALIASES
    None

REMARKS
    None
```

The PowerShell help system also contains conceptual **About** help articles. You must update the help content on your system to get the **About** articles. For more information, see the [Updating help](https://learn.microsoft.com/en-us/powershell/scripting/learn/ps101/02-help-system?view=powershell-7.5#updating-help) section of this chapter.
>  help system 还包含了 About 帮助文章，获取相关命令的更多信息

Use the following command to return a list of all **About** help articles on your system.

PowerShell

```
help About_*
```

When you limit the results to one **About** help article, `Get-Help` displays the content of that article.

PowerShellCopy

```
help about_Updatable_Help
```

## Updating help
Earlier in this chapter, you updated the PowerShell help articles on your computer the first time you ran the `Get-Help` cmdlet. You should periodically run the `Update-Help` cmdlet on your computer to obtain any updates to the help content.

Important
In Windows PowerShell 5.1, you must run `Update-Help` as an administrator in an elevated PowerShell session.

In the following example, `Update-Help` downloads the PowerShell help content for all modules installed on your computer. You should use the **Force** parameter to ensure that you download the latest version of the help content.

PowerShell

```
Update-Help -Force
```

As shown in the following results, a module returned an error. Errors aren't uncommon and usually occur when the module's author doesn't configure updatable help correctly.

Output

```
Update-Help : Failed to update Help for the module(s) 'BitsTransfer' with UI culture(s) {en-US} : Unable to retrieve the HelpInfo XML file for UI culture en-US. Make sure the HelpInfoUri property in the module manifest is valid or check your network connection and then try the command again.
At line:1 char:1
+ Update-Help
+ ~~~~~~~~~~~
    + CategoryInfo          : ResourceUnavailable: (:) [Update-Help], Except
   ion
    + FullyQualifiedErrorId : UnableToRetrieveHelpInfoXml,Microsoft.PowerShe
   ll.Commands.UpdateHelpCommand
```

`Update-Help` requires internet access to download the help content. If your computer doesn't have internet access, use the `Save-Help` cmdlet on a computer with internet access to download and save the updated help content. Then, use the **SourcePath** parameter of `Update-Help` to specify the location of the saved updated help content.

## Get-Command
`Get-Command` is another multipurpose command that helps you find commands. When you run `Get-Command` without any parameters, it returns a list of all PowerShell commands on your system. You can also use `Get-Command` to get command syntax similar to `Get-Help`.
>  `get-command` 也用于搜索命令
>  不加参数时，会返回系统中的所有 PowerShell 命令
>  `get-command` 的搜索语法类似于 `get-help`

How do you determine the syntax for `Get-Command`? You could use `Get-Help` to display the help article for `Get-Command`, as shown in the [Get-Help](https://learn.microsoft.com/en-us/powershell/scripting/learn/ps101/02-help-system?view=powershell-7.5#get-help) section of this chapter. You can also use `Get-Command` with the **Syntax** parameter to view the syntax for any command. This shortcut helps you quickly determine how to use a command without navigating through its help content.
>  要知道一个命令的语法，就可以使用 `get-command -syntax`，而不必要使用 `get-help` 获得太多的帮助信息

PowerShell

```
Get-Command -Name Get-Command -Syntax
```

Using `Get-Command` with the **Syntax** parameter provides a more concise view of the syntax that shows the parameters and their value types, without listing the specific allowable values like `Get-Help` shows.
>  `get-command -syntax` 获取了命令的参数和他们的值类型，但不会像 `get-help` 一样列出具体的可允许的值

Output

```
Get-Command [[-ArgumentList] <Object[]>] [-Verb <string[]>]
[-Noun <string[]>] [-Module <string[]>]
[-FullyQualifiedModule <ModuleSpecification[]>] [-TotalCount <int>]
[-Syntax] [-ShowCommandInfo] [-All] [-ListImported]
[-ParameterName <string[]>] [-ParameterType <PSTypeName[]>]
[<CommonParameters>]

Get-Command [[-Name] <string[]>] [[-ArgumentList] <Object[]>]
[-Module <string[]>] [-FullyQualifiedModule <ModuleSpecification[]>]
[-CommandType <CommandTypes>] [-TotalCount <int>] [-Syntax]
[-ShowCommandInfo] [-All] [-ListImported] [-ParameterName <string[]>]
[-ParameterType <PSTypeName[]>] [<CommonParameters>]
```

If you need more detailed information about how to use a command, use `Get-Help`.
>  如果需要更详细的信息，就使用 `get-hlep`

PowerShell

```
help Get-Command -Full
```

The **SYNTAX** section of `Get-Help` provides a more user-friendly display by expanding enumerated values for parameters. It shows you the actual values you can use, making it easier to understand the available options.
>  `get-help` 的 SYNTAX 部分会更加详细，会展开可列举的参数值

Output

```
...
    Get-Command [[-Name] <System.String[]>] [[-ArgumentList]
    <System.Object[]>] [-All] [-CommandType {Alias | Function | Filter |
    Cmdlet | ExternalScript | Application | Script | Workflow |
    Configuration | All}] [-FullyQualifiedModule
    <Microsoft.PowerShell.Commands.ModuleSpecification[]>] [-ListImported]
    [-Module <System.String[]>] [-ParameterName <System.String[]>]
    [-ParameterType <System.Management.Automation.PSTypeName[]>]
    [-ShowCommandInfo] [-Syntax] [-TotalCount <System.Int32>]
    [<CommonParameters>]

    Get-Command [[-ArgumentList] <System.Object[]>] [-All]
    [-FullyQualifiedModule
    <Microsoft.PowerShell.Commands.ModuleSpecification[]>] [-ListImported]
    [-Module <System.String[]>] [-Noun <System.String[]>] [-ParameterName
    <System.String[]>] [-ParameterType
    <System.Management.Automation.PSTypeName[]>] [-ShowCommandInfo]
    [-Syntax] [-TotalCount <System.Int32>] [-Verb <System.String[]>]
    [<CommonParameters>]
...
```

The **PARAMETERS** section of the help for `Get-Command` reveals that the **Name**, **Noun**, and **Verb** parameters accept wildcard characters.
>  `get-command` 的 Name, Noun, Verb 参数接收通配符

OutputCopy

```
...
    -Name <System.String[]>
        Specifies an array of names. This cmdlet gets only commands that have the specified name. Enter a name or name pattern. Wildcard characters are permitted.

        To get commands that have the same name, use the All parameter. When two commands have the same name, by default, `Get-Command` gets the command that runs when you type the command name.

        Required?                    false
        Position?                    0
        Default value                None
        Accept pipeline input?       True (ByPropertyName, ByValue)
        Accept wildcard characters?  true

    -Noun <System.String[]>
        Specifies an array of command nouns. This cmdlet gets commands, which include cmdlets, functions, and aliases, that have names that include the specified noun. Enter one or more nouns or noun patterns. Wildcard characters are permitted.

        Required?                    false
        Position?                    named
        Default value                None
        Accept pipeline input?       True (ByPropertyName)
        Accept wildcard characters?  true
    -Verb <System.String[]>
        Specifies an array of command verbs. This cmdlet gets commands, which include cmdlets, functions, and aliases, that have names that include the specified verb. Enter one or more verbs or verb patterns. Wildcard characters are permitted.

        Required?                    false
        Position?                    named
        Default value                None
        Accept pipeline input?       True (ByPropertyName)
        Accept wildcard characters?  true
...
```

The following example uses the `*` wildcard character with the value for the **Name** parameter of `Get-Command`.

PowerShell

```
Get-Command -Name *service*
```

When you use wildcard characters with the **Name** parameter of `Get-Command`, it returns PowerShell commands and native commands, as shown in the following results.

Output

```

CommandType     Name                                               Version
-----------     ----                                               -------
Function        Get-NetFirewallServiceFilter                       2.0.0.0
Function        Set-NetFirewallServiceFilter                       2.0.0.0
Cmdlet          Get-Service                                        3.1.0.0
Cmdlet          New-Service                                        3.1.0.0
Cmdlet          New-WebServiceProxy                                3.1.0.0
Cmdlet          Restart-Service                                    3.1.0.0
Cmdlet          Resume-Service                                     3.1.0.0
Cmdlet          Set-Service                                        3.1.0.0
Cmdlet          Start-Service                                      3.1.0.0
Cmdlet          Stop-Service                                       3.1.0.0
Cmdlet          Suspend-Service                                    3.1.0.0
Application     SecurityHealthService.exe                          10.0.2...
Application     SensorDataService.exe                              10.0.2...
Application     services.exe                                       10.0.2...
Application     services.msc                                       0.0.0.0
Application     TieringEngineService.exe                           10.0.2...
Application     Windows.WARP.JITService.exe                        10.0.2...
```

You can limit the results of `Get-Command` to PowerShell commands using the **CommandType** parameter.

PowerShell

```
Get-Command -Name *service* -CommandType Cmdlet, Function, Alias, Script
```

>  `-CommandType` 可以限制展示的命令类型

Another option might be to use either the **Verb** or **Noun** parameter or both since only PowerShell commands have verbs and nouns.

The following example uses `Get-Command` to find commands on your computer that work with processes. Use the **Noun** parameter and specify `Process` as its value.

PowerShell

```
Get-Command -Noun Process
```

Output

```
CommandType     Name                                               Version
-----------     ----                                               -------
Cmdlet          Debug-Process                                      3.1.0.0
Cmdlet          Get-Process                                        3.1.0.0
Cmdlet          Start-Process                                      3.1.0.0
Cmdlet          Stop-Process                                       3.1.0.0
Cmdlet          Wait-Process                                       3.1.0.0
```

## Summary
In this chapter, you learned how to find commands with `Get-Help` and `Get-Command`. You also learned how to use the help system to understand how to use commands once you find them. In addition, you learned how to update the help system on your computer when new help content is available.

## Review

1. Is the **DisplayName** parameter of `Get-Service` positional?
2. How many parameter sets does the `Get-Process` cmdlet have?
3. What PowerShell commands exist for working with event logs?
4. What's the PowerShell command for returning a list of PowerShell processes running on your computer?
5. How do you update the PowerShell help content stored on your computer?

## References
To learn more about the concepts covered in this chapter, read the following PowerShell help articles.

- [Get-Help](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/get-help)
- [Get-Command](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/get-command)
- [Update-Help](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/update-help)
- [Save-Help](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/save-help)
- [about_Updatable_Help](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_updatable_help)
- [about_Command_Syntax](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_command_syntax)

## Next steps
In the next chapter, you'll learn about objects, properties, methods, and the `Get-Member` cmdlet.

