---
completed: true
---
>  https://en.wikipedia.org/wiki/Zero_trust_architecture

**Zero trust architecture** (**ZTA**) or **perimeterless security** is a design and implementation strategy of [IT systems](https://en.wikipedia.org/wiki/IT_system "IT system"). The principle is that users and devices should not be trusted by default, even if they are connected to a [privileged](https://en.wikipedia.org/wiki/Privilege_\(computing\) "Privilege (computing)") network such as a corporate [LAN](https://en.wikipedia.org/wiki/Local_area_network "Local area network") and even if they were previously verified.
>  零信任架构 (或无边界安全) 是一种 IT 系统的设计和实施策略，其原则是用户和设备默认不应该互相信任，即便二者都连接到了一个特权网络 (例如企业居于哇嘎)，即便二者之前已经被验证过

ZTA is implemented by establishing identity verification, validating device compliance prior to granting access, and ensuring least privilege access to only explicitly-authorized resources. Most modern corporate networks consist of many interconnected zones, [cloud services](https://en.wikipedia.org/wiki/Cloud_computing "Cloud computing") and infrastructure, connections to remote and mobile environments, and connections to non-conventional IT, such as [IoT](https://en.wikipedia.org/wiki/Internet_of_things "Internet of things") devices.
>  零信任架构通过构建身份验证实现，通过身份验证，在授予设备访问权限之前先验证设备的合规性，并确保仅对明确授权的资源赋予最低的访问权限
>  大多数现代企业网络由许多相互连接的区域、云服务和基础设施，与远程和移动环境的连接，以及与传统非 IT 设备 (例如 IoT 设备) 的连接组成

The traditional approach by trusting users and devices within a notional "corporate perimeter" or via a [VPN](https://en.wikipedia.org/wiki/Virtual_private_network "Virtual private network") connection is commonly not sufficient in the complex environment of a corporate network. The zero trust approach advocates [mutual authentication](https://en.wikipedia.org/wiki/Mutual_authentication "Mutual authentication"), including checking the identity and integrity of users and devices without respect to location, and providing access to applications and services based on the confidence of user and device identity and device status in combination with user [authentication](https://en.wikipedia.org/wiki/Authentication "Authentication"). [1](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-1) The zero trust architecture has been proposed for use in specific areas such as supply chains. [2](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-2) [3](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-3)
>  传统的零信任方法依赖于信任位于概念上 “企业边界” 的用户和设备或通过 VPN 连接，这些方法在复杂的公司网络环境中通常不足以确保安全
>  零信任方法提倡双向认证，检查用户和设备的身份与完整性，并基于对用户身份和设备身份的信任度以及设备状态来提供对应用和服务的访问
>  零信任架构已经被提议用于特定领域，例如供应链

The principles of zero trust can be applied to data access, and to the management of data. This brings about zero trust [data security](https://en.wikipedia.org/wiki/Data_security "Data security") where every request to access the data needs to be authenticated dynamically and ensure [least privileged access](https://en.wikipedia.org/wiki/Principle_of_least_privilege "Principle of least privilege") to resources. In order to determine if access can be granted, policies can be applied based on the attributes of the data, who the user is, and the type of environment using [Attribute-Based Access Control (ABAC)](https://en.wikipedia.org/wiki/Attribute-based_access_control "Attribute-based access control"). This zero-trust data security approach can protect access to the data. [4](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-4)
>  零信任原则可以用于数据访问和数据管理，以确保零信任数据安全
>  零信任数据安全中，每次访问数据的请求都需要动态认证，并确保对资源的最小权限访问
>  为了确定是否可以授予访问权限，可以应用基于数据属性、用户身份和环境类型的策略，使用基于属性的访问控制
>  这种零信任的数据安全方法可以保护对数据的访问

## History
In April 1994, the term "zero trust" was coined by Stephen Paul Marsh in his doctoral thesis on computer security at the [University of Stirling](https://en.wikipedia.org/wiki/University_of_Stirling "University of Stirling"). Marsh's work studied trust as something finite that can be described mathematically, asserting that the concept of trust transcends human factors such as [morality](https://en.wikipedia.org/wiki/Morality "Morality"), [ethics](https://en.wikipedia.org/wiki/Ethics "Ethics"), [lawfulness](https://en.wikipedia.org/wiki/Lawful "Lawful"), [justice](https://en.wikipedia.org/wiki/Justice "Justice"), and [judgement](https://en.wikipedia.org/wiki/Judgement "Judgement"). [5](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-5)
>  1994 年 4 月， Marsh 在其关于计算机安全的博士论文中首次提出了“零信任”这一术，Marsh 的研究将信任视为一种可以数学描述的有限量，并断言信任的概念超越了人类因素，如道德、伦理、合法性、正义、判断

The problems of the Smartie or M&M model of the network (the precursor description of [de-perimeterisation](https://en.wikipedia.org/wiki/De-perimeterisation "De-perimeterisation")) was described by a [Sun Microsystems](https://en.wikipedia.org/wiki/Sun_Microsystems "Sun Microsystems") engineer in a [Network World](https://en.wikipedia.org/wiki/Network_World "Network World") article in May 1994, who described firewalls' perimeter defense, as a hard shell around a soft center, like a Cadbury Egg. [6](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-6) 
>  Sun Microsystems 的一位工程师在 1994 年 5 月的《Network World》杂志文章中描述了智能豆（Smarties 或 M&M）网络模型的问题（这是去边界化概念的前身描述）
>  文中将防火墙的边界防御比喻为像 Cadbury 巧克力蛋那样的硬壳包裹着柔软的核心，即“硬壳软心”模式。

In 2001 the first version of the OSSTMM (Open Source Security Testing Methodology Manual) was released and this had some focus on trust. Version 3 which came out around 2007 has a whole chapter on Trust which says "Trust is a Vulnerability" and talks about how to apply the OSSTMM 10 controls based on Trust levels.
>  2001 年，OSSTMM (开源安全测试方法手册) 的第一个版本发布，其中对“信任”有一定的关注
>  2007 年左右发布的版本 3 中专门有一章关于“信任”，这一章提到“信任是一种漏洞”，并讨论了如何根据信任级别应用 OSSTMM 的 10 项控制措施。

In 2003 the challenges of defining the perimeter to an organization's IT systems was highlighted by the [Jericho Forum](https://en.wikipedia.org/wiki/Jericho_Forum "Jericho Forum") of this year, discussing the trend of what was then given the name "[de-perimeterisation](https://en.wikipedia.org/wiki/De-perimeterisation "De-perimeterisation")".
>  2003 年，[杰里科论坛](https://en.wikipedia.org/wiki/Jericho_Forum)强调了定义组织IT系统边界的挑战，并讨论了当时被称为“去边界化”的趋势

In response to [Operation Aurora](https://en.wikipedia.org/wiki/Operation_Aurora "Operation Aurora"), a Chinese APT attack throughout 2009, [Google](https://en.wikipedia.org/wiki/Google "Google") started to implement a zero-trust architecture referred to as [BeyondCorp](https://en.wikipedia.org/wiki/BeyondCorp "BeyondCorp").
>  作为对贯穿 2009 年的[极光行动](https://en.wikipedia.org/wiki/Operation_Aurora)的回应——这是一次由中国APT组织发起的攻击，Google 开始实施一种名为 BeyondCorp 的零信任架构。

In 2010 the term zero trust model was used by analyst John Kindervag of [Forrester Research](https://en.wikipedia.org/wiki/Forrester_Research "Forrester Research") to denote stricter cybersecurity programs and access control within corporations. [7](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-:1-7) [8](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-:2-8) [9](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-9) However, it would take almost a decade for zero trust architectures to become prevalent, driven in part by increased adoption of mobile and cloud services.
>  2010 年，Forrester Research 的分析师约翰·金德瓦格（John Kindervag）使用了“零信任模型”这一术语，用来指代企业内部更为严格的网络安全计划和访问控制
>  然而，零信任架构真正普及起来却花了将近十年的时间，部分原因是移动设备和云服务的广泛采用。

In 2018, work undertaken in the [United States](https://en.wikipedia.org/wiki/United_States "United States") by cybersecurity researchers at [NIST](https://en.wikipedia.org/wiki/NIST "NIST") and [NCCoE](https://en.wikipedia.org/wiki/National_Cybersecurity_Center_of_Excellence "National Cybersecurity Center of Excellence") led to the publication of NIST SP 800-207 – Zero Trust Architecture. [10](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-:3-10) [11](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-:4-11) The publication defines zero trust (ZT) as a collection of concepts and ideas designed to reduce the uncertainty in enforcing accurate, per-request access decisions in information systems and services in the face of a network viewed as compromised. A zero trust architecture (ZTA) is an enterprise's cyber security plan that utilizes zero trust concepts and encompasses component relationships, workflow planning, and access policies. Therefore, a zero trust enterprise is the network infrastructure (physical and virtual) and operational policies that are in place for an enterprise as a product of a zero trust architecture plan.
>  2018年，在美国，由NIST（美国国家标准与技术研究院）和NCCoE（国家网络安全卓越中心）的研究人员开展的工作导致了《NIST SP 800-207 - 零信任架构》的发布
>  该文件将零信任定义为一组概念和理念，旨在减少在被视为已被攻破的网络中执行准确、每次请求的访问决策时的不确定性
>  零信任架构是利用零信任概念的企业网络安全计划，包括组件关系、工作流程规划和访问策略
>  因此，零信任企业是作为零信任架构计划结果而存在的企业的网络基础设施（物理和虚拟）以及运营政策

There are several ways to implement all the tenets of ZT; a full ZTA solution will include elements of all three:

- Using enhanced identity governance and policy-based access controls.
- Using micro-segmentation
- Using overlay networks or software-defined perimeters

>  有多种方法可以实现零信任的所有原则；一个完整的零信任架构解决方案将包含以下三种方法的元素：
>  - 使用增强的身份治理和基于策略的访问控制
>  - 使用微分段
>  - 使用覆盖网络或软件定义的边界

In 2019 the United Kingdom [National Cyber Security Centre (NCSC)](https://en.wikipedia.org/wiki/National_Cyber_Security_Centre_\(United_Kingdom\) "National Cyber Security Centre (United Kingdom)") recommended that network architects consider a zero trust approach for new IT deployments, particularly where significant use of cloud services is planned. [12](https://en.wikipedia.org/wiki/Zero_trust_architecture#cite_note-:0-12) An alternative but consistent approach is taken by [NCSC](https://en.wikipedia.org/wiki/National_Cyber_Security_Centre_\(United_Kingdom\) "National Cyber Security Centre (United Kingdom)"), in identifying the key principles behind zero trust architectures:

- Single strong source of user identity
- User authentication
- Machine authentication
- Additional context, such as policy compliance and device health
- Authorization policies to access an application
- Access control policies within an application

>  在 2019 年，英国国家网络安全中心（NCSC）建议网络架构师在新的 IT 部署中考虑采用零信任方法，特别是在计划大量使用云服务的情况下
>  另一种方法由英国国家网络安全中心（NCSC）提出，用于识别零信任架构背后的关键原则：
>  - 用户身份的单一强认证来源。
>  - 用户身份验证。
>  - 设备身份验证。
>  - 额外的上下文信息，例如策略合规性和设备健康状况。
>  - 访问应用程序的授权策略。
>  - 应用程序内的访问控制策略。

> This page was last edited on 5 April 2025, at 18:52 (UTC). 