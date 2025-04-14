>  https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-NIST-1

**Virtual private network** (**VPN**) is a [network architecture](https://en.wikipedia.org/wiki/Network_architecture "Network architecture") for virtually extending a [private network](https://en.wikipedia.org/wiki/Private_network "Private network") (i.e. any [computer network](https://en.wikipedia.org/wiki/Computer_network "Computer network") which is not the public [Internet](https://en.wikipedia.org/wiki/Internet "Internet")) across one or multiple other networks which are either untrusted (as they are not controlled by the entity aiming to implement the VPN) or need to be isolated (thus making the lower network invisible or not directly usable). [1](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-NIST-1)
>  VPN 是一种网络架构，用于在虚拟层面上将一个私有网络 (即任意不是公共互联网的网络) 拓展到一个或多个其他网络中，这些网络可能是不受信任的 (因为它们不在希望实施 VPN 的实体控制之下) 或者需要被隔离 (从而使得底层网络不可见或者无法直接使用)

>  例如，公司的私有网络可以通过 VPN 将其拓展到互联网 (Internet)

A VPN can extend access to a private network to users who do not have direct access to it, such as an office network allowing secure access from off-site over the Internet. [2](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-Cisco-2) This is achieved by creating a link between [computing devices](https://en.wikipedia.org/wiki/Computing_device "Computing device") and computer networks by the use of network [tunneling protocols](https://en.wikipedia.org/wiki/Tunneling_protocol "Tunneling protocol").
>  一个 VPN 可以将私有网络的访问权限拓展到没有直接访问权限的用户，例如允许远程办公人员通过 Internet 安全访问，这是通过使用网络隧道协议在计算设备和网络之间建立连接实现的

It is possible to make a VPN secure to use on top of insecure communication medium (such as the public internet) by choosing a tunneling protocol that implements [encryption](https://en.wikipedia.org/wiki/Encryption "Encryption"). This kind of VPN implementation has the benefit of reduced costs and greater flexibility, with respect to dedicated communication lines, for [remote workers](https://en.wikipedia.org/wiki/Remote_work "Remote work"). [3](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-3)
>  通过选择一个实现了加密的隧道协议，可以在不安全的通道介质上 (例如公共互联网) 建立安全的 VPN
>  这类 VPN 实现相对于专用通信线路而言具有更低成本，并且对远程工作者具有更大的灵活性

The term _VPN_ is also used to refer to [VPN services](https://en.wikipedia.org/wiki/VPN_service "VPN service") which sell access to their own private networks for internet access by connecting their customers using VPN tunneling protocols.
>  VPN 也指代 VPN 服务，VPN 服务通过使用 VPN 隧道协议，将客户连接到它们自己的私有网络，从而为客户提供互联网接入

## Motivation
The goal of a virtual private network is to allow [network hosts](https://en.wikipedia.org/wiki/Network_host "Network host") to exchange network messages across another network to access private content, as if they were part of the same network. This is done in a way that makes crossing the intermediate network transparent to network applications. Users of a network connectivity service may consider such an intermediate network to be untrusted, since it is controlled by a third-party, and might prefer a VPN implemented via protocols that protect the privacy of their communication.
>  一个 VPN 的目标是允许网络主机通过另一个网络和私有网络交换消息，以访问私有内容，仿佛它们属于同一个网络
>  对于网络应用程序来说，这层跨越中间网络的传输是透明的
>  使用 VPN 网络连接服务的用户可能认为这样的中间网络不可信，因为它由第三方控制，故倾向于使用通过保护通信隐私的协议实现的 VPN

In the case of a [Provider-provisioned VPN](https://en.wikipedia.org/wiki/Provider-provisioned_VPN "Provider-provisioned VPN"), the goal is not to protect against untrusted networks, but to isolate parts of the provider's own network infrastructure in virtual segments, in ways that make the contents of each segment private with respect to the others. This situation makes many other tunneling protocols suitable for building PPVPNs, even with weak or no security features (like in [VLAN](https://en.wikipedia.org/wiki/VLAN "VLAN")).
>  供应商提供的 VPN 的目标不是保护信息免受不信任网络的影响，而是隔离供应商自身网络基础设施的虚拟部分，使得每个 virtual segment 的内容相对于其他 virtual segments 是私有的
>  在这种需求下，许多其他隧道协议也可以用于构建 PPVPN，即便它们只有弱安全功能或没有安全功能

## VPN general working
How a VPN works depends on which technologies and protocols the VPN is built upon. A [tunneling protocol](https://en.wikipedia.org/wiki/Tunneling_protocol "Tunneling protocol") is used to transfer the network messages from one side to the other. The goal is to take network messages from applications on one side of the tunnel and replay them on the other side. Applications do not need to be modified to let their messages pass through the VPN, because the virtual network or link is made available to the OS.
>  VPN 的工作原理取决于它基于的技术和协议
>  VPN 使用隧道协议将网络消息从一端传输到另一端，VPN 提取来自隧道中应用端的网络消息，然后在另一端重新发送它们
>  应用无需进行修改即可让其消息通过 VPN，因为虚拟网络或链路对 OS 可用

Applications that do implement tunneling or [proxying](https://en.wikipedia.org/wiki/Proxy_pattern "Proxy pattern") features for themselves without making such features available as a network interface, are not to be considered VPN implementations but may achieve the same or similar end-user goal of exchanging private contents with a remote network.
>  自行实现了隧道或代理功能，但没有将这些功能作为网络接口使用的应用不应该被认为是 VPN 实现，但这些应用本身可以实现终端用户远程网络交换私有内容的相似的目标

## VPN topology configurations

[![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/VPN_classification-en.svg/330px-VPN_classification-en.svg.png)](https://en.wikipedia.org/wiki/File:VPN_classification-en.svg)

VPN classification tree based on the topology first, then on the technology used

[![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Virtual_Private_Network_overview.svg/330px-Virtual_Private_Network_overview.svg.png)](https://en.wikipedia.org/wiki/File:Virtual_Private_Network_overview.svg)

VPN connectivity overview, showing intranet site-to-site and remote-work configurations used together

Virtual private networks configurations can be classified depending on the purpose of the virtual extension, which makes different tunneling strategies appropriate for different topologies:
>  VPN 配置可以按照虚拟拓展的目的分类
>  不同目的的虚拟拓展使得不同的隧道策略适用于不同的拓扑结构

**Remote access**
A _host-to-network_ configuration is analogous to joining one or more computers to a network to which they cannot be directly connected. This type of extension provides that computer access to [local area network](https://en.wikipedia.org/wiki/Local_area_network "Local area network") of a remote site, or any wider enterprise networks, such as an [intranet](https://en.wikipedia.org/wiki/Intranet "Intranet"). Each computer is in charge of activating its own tunnel towards the network it wants to join. The joined network is only aware of a single remote host for each tunnel. 
>  Remote access
>  host-to-network 配置类似于将一台或者多台计算机连接到它们无法直接连接到的网络
>  这种类型的虚拟拓展为计算机提供了对远程站点的局域网或者范围更大的公司网络例如内联网的访问权限
>  每台计算机负责激活其自身通往目标网络的隧道，目标网络中的每个隧道仅对应单个远程主机

This may be employed for [remote workers](https://en.wikipedia.org/wiki/Remote_work "Remote work"), or to enable people accessing their private home or company resources without exposing them on the public Internet. Remote access tunnels can be either on-demand or always-on. Because the remote host location is usually unknown to the central network until the former tries to reach it, proper implementations of this configuration require the remote host to initiate the communication towards the central network it is accessing.
>  这类配置可以用于远程工作者，或者允许用户在不将其私有资源或公司资源暴露给公共互联网的情况下对其进行访问
>  远程访问隧道可以按需启动，也可以始终处于开启状态
>  由于远程主机的位置在它访问目标网络之前对于目标网络是未知的，故该配置的正确实现要求远程主机向目标网络发起通信

**Site-to-site**
A _site-to-site_ configuration connects two networks. This configuration expands a network across geographically disparate locations. Tunneling is only done between gateway devices located at each network location. These devices then make the tunnel available to other local network hosts that aim to reach any host on the other side. 
>  Site-to-site
>  site-to-site 配置将两个网络连接
>  该配置将网络拓展到地理位置不同的多个地点，隧道仅在各个网络的网关设备间建立
>  这些网关设备会让隧道对于其本地网络的其他主机可用，以便它们访问另一侧的主机

This is useful to keep sites connected to each other in a stable manner, like office networks to their headquarters or datacenter. In this case, any side may be configured to initiate the communication as long as it knows how to reach the other.
>  这类配置有助于以稳定的方式保持各个站点之间相互连接，例如保持办公室网络和其总部或数据中心的网络连接
>  在这类配置下，只要知道如何到达对方，任意一方都可以发起通信

In the context of site-to-site configurations, the terms _[intranet](https://en.wikipedia.org/wiki/Intranet "Intranet")_ and _[extranet](https://en.wikipedia.org/wiki/Extranet "Extranet")_ are used to describe two different use cases. [4](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-4) An _intranet_ site-to-site VPN describes a configuration where the sites connected by the VPN belong to the same organization, whereas an _extranet_ site-to-site VPN joins sites belonging to multiple organizations.
>  在 site-to-site 配置中，“内联网” 和 “外联网” 被用于描述两类不同用例
>  intranet site-to-site VPN 指将两个属于相同组织站点通过 VPN 连接，extranet site-to-site VPN 将属于不同组织站点通过 VPN 连接

Typically, individuals interact with remote access VPNs, whereas businesses tend to make use of site-to-site connections for [business-to-business](https://en.wikipedia.org/wiki/Business-to-business "Business-to-business"), cloud computing, and [branch office](https://en.wikipedia.org/wiki/Branch_office "Branch office") scenarios. However, these technologies are not mutually exclusive and, in a significantly complex business network, may be combined.
>  通常，个人用户使用 remote access VPN，而企业则倾向于在企业对企业、云计算、分支办公室等场景中使用 site-to-site VPN
>  remote access 和 site-to-site 并非互斥，可以结合使用

Apart from the general topology configuration, a VPN may also be characterized by:

- the tunneling protocol used to [tunnel](https://en.wikipedia.org/wiki/IP_tunnel "IP tunnel") the traffic,
- the tunnel's termination point location, e.g., on the customer [edge](https://en.wikipedia.org/wiki/Edge_device "Edge device") or network-provider edge,
- the security features provided,
- the [OSI layer](https://en.wikipedia.org/wiki/OSI_model "OSI model") they present to the connecting network, such as [Layer 2](https://en.wikipedia.org/wiki/Layer_2 "Layer 2") link/circuit or [Layer 3](https://en.wikipedia.org/wiki/Layer_3 "Layer 3") network connectivity,
- the number of simultaneous allowed tunnels,
- the relationship between the actor implementing the VPN and the network infrastructure provider, and whether the former trusts the medium of the former or not

>  除了上述的拓扑结构配置外，VPN 的区分点还在于
>  - 用于隧道化流量的隧道协议
>  - 隧道终止点的位置，例如是在客户边缘设备上或在网络供应商边缘设备上
>  - 提供的安全功能
>  - 向连接网络呈现的 OSI 层，例如第二层链路层或第三层网络层
>  - 允许同时建立的隧道数量
>  - 实施 VPN 的实体和网络基础设施供应商的关系，以及前者是否信任后者的传输介质

A variety of VPN technics exist to adapt to the above characteristics, each providing different network tunneling capabilities and different security model coverage or interpretation.

## VPN native and third-party support
[Operating systems](https://en.wikipedia.org/wiki/Operating_system "Operating system") vendors and developers do typically offer native support to a selection of VPN protocols which is subject to change over the years, as some have been proven to be unsecure with respect to modern requirements and expectations, and some others emerged.
>  OS 供应商和开发商通常会提供对一系列 VPN 协议的原生支持
>  支持列表会随着时间变化，例如一些协议不再安全，以及有新的协议出现

### VPN support in consumer operating systems
Desktop, smartphone and other end-user device operating systems do usually support configuring remote access VPN from their [graphical](https://en.wikipedia.org/wiki/Graphical_user_interface "Graphical user interface") or [command-line](https://en.wikipedia.org/wiki/Command-line_interface "Command-line interface") tools. [5](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-5) [6](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-6) [7](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-7) However, due to the variety of, often non standard, VPN protocols there exists many third-party applications that implement additional protocols not yet or no more natively supported by the OS.
>  各类用户设备的 OS 通常支持通过图形页面或命令行工具配置 remote access VPN
>  由于有许多非标准的 VPN 协议，许多第三方应用会实现 OS 原生不支持的额外协议

For instance, [Android](https://en.wikipedia.org/wiki/Android_\(operating_system\) "Android (operating system)") lacked native IPsec IKEv2 support until version 11, [8](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-8) and people needed to install third-party apps in order to connect that kind of VPNs, while [Microsoft Windows](https://en.wikipedia.org/wiki/Microsoft_Windows "Microsoft Windows"), [BlackBerry OS](https://en.wikipedia.org/wiki/BlackBerry_OS "BlackBerry OS") and others got it supported in the past.

Conversely, Windows does not support plain IPsec IKEv1 remote access native VPN configuration (commonly used by [Cisco](https://en.wikipedia.org/wiki/Cisco_Systems_VPN_Client "Cisco Systems VPN Client") and [Fritz!Box](https://en.wikipedia.org/wiki/Fritz!Box "Fritz!Box") VPN solutions) which makes the use of third-party applications mandatory for people and companies relying on such VPN protocol.

### VPN support in network devices
Network appliances, such as firewalls, do often include VPN gateway functionality for either remote access or site-to-site configurations. Their administration interfaces do often facilitate setting up virtual private networks with a selection of supported protocols which have been integrated for an easy out-of-box setup.
>  网络应用，例如防火墙，通常会包含 remote access 或 site-to-site VPN 的网关功能
>  其管理界面可以方便使用一系列已集成的支持的协议设置虚拟私人网络

In some cases, like in the open source operating systems devoted to firewalls and network devices (like [OpenWrt](https://en.wikipedia.org/wiki/OpenWrt "OpenWrt"), [IPFire](https://en.wikipedia.org/wiki/IPFire "IPFire"), [PfSense](https://en.wikipedia.org/wiki/PfSense "PfSense") or [OPNsense](https://en.wikipedia.org/wiki/OPNsense "OPNsense")) it is possible to add support for additional VPN protocols by installing missing software components or third-party apps.
>  一些专用于防火墙或网络设备的开源 OS 可以通过安装额外的软件组件添加对额外 VPN 协议的支持

Similarly, it is possible to get additional VPN configurations working, even if the OS does not facilitate the setup of that particular configuration, by manually editing internal configurations of by modifying the open source code of the OS itself. 
>  并且通过修改代码本身，可以提供对额外 VPN 配置的支持

For instance, pfSense does not support remote access VPN configurations through its user interface where the OS runs on the remote host, while provides comprehensive support for configuring it as the central VPN gateway of such remote-access configuration scenario.

Otherwise, commercial appliances with VPN features based on proprietary hardware/software platforms, usually support a consistent VPN protocol across their products but do not open up for customizations outside the use cases they intended to implement. This is often the case for appliances that rely on hardware acceleration of VPNs to provide higher throughput or support a larger amount of simultaneously connected users.
>  但基于专有硬件/软件平台的商业应用的 VPN 功能通常仅支持某个一致的 VPN 协议，并且不开放自定义功能

## Security mechanisms
Whenever a VPN is intended to virtually extend a private network over a third-party untrusted medium, it is desirable that the chosen protocols match the following security model:

- [confidentiality](https://en.wikipedia.org/wiki/Information_security#Confidentiality "Information security") to prevent disclosure of private information or [data sniffing](https://en.wikipedia.org/wiki/Packet_analyzer "Packet analyzer"), such that even if the network traffic is sniffed at the packet level (see network sniffer or [deep packet inspection](https://en.wikipedia.org/wiki/Deep_packet_inspection "Deep packet inspection")), an attacker would see only [encrypted data](https://en.wikipedia.org/wiki/Encryption "Encryption"), not the raw data
- message [integrity](https://en.wikipedia.org/wiki/Data_integrity "Data integrity") to detect and reject any instances of tampering with transmitted messages, [data packets](https://en.wikipedia.org/wiki/Network_packet "Network packet") are secured by [tamper proofing](https://en.wikipedia.org/wiki/Tamperproofing "Tamperproofing") via a [message authentication code](https://en.wikipedia.org/wiki/Message_authentication_code "Message authentication code") (MAC), which prevents the message from being altered or [tampered](https://en.wikipedia.org/wiki/Tamper-evident_technology "Tamper-evident technology") without being rejected due to the MAC not matching with the altered data packet.

>  当 VPN 被用于将私有网络在第三方不可信介质上拓展时，则选择的协议就需要满足以下安全模型
>  - 保密性，防止私人信息泄露或数据嗅探，即便网络流量在包级别被嗅探，攻击者也只能看到加密数据而非原数据
>  - 消息完整性，用于检测并拒绝任何篡改了原消息的行为，消息验证码为数据包实现防篡改，如果包被修改，其消息验证码将不匹配，进而被拒绝

VPN are not intended to make connecting users anonymous or unidentifiable from the untrusted medium network provider perspective. If the VPN makes use of protocols that do provide those confidentiality features, their usage can increase user [privacy](https://en.wikipedia.org/wiki/Privacy "Privacy") by making the untrusted medium owner unable to access the private data exchanged across the VPN.
>  VPN 并非旨在让连接用户对于不可信介质的网络服务提供商变得匿名或不可识别
>  如果 VPN 使用的协议提供了上述安全性质，不可信介质的网络服务提供商将无法访问用户通过 VPN 交换的私有数据

### Authentication
In order to prevent unauthorized users from accessing the VPN, most protocols can be implemented in ways that also enable [authentication](https://en.wikipedia.org/wiki/Authentication "Authentication") of connecting parties. This secures the joined remote network confidentiality, integrity and availability.
>  为了防止未经授权的用户访问 VPN，大多数 VPN 协议都支持连接方认证功能

Tunnel endpoints can be authenticated in various ways during the VPN access initiation. Authentication can happen immediately on VPN initiation (e.g. by simple whitelisting of endpoint IP address), or very lately after actual tunnels are already active (e.g. with a [web captive portal](https://en.wikipedia.org/wiki/Captive_portal "Captive portal")).
>  在 VPN 访问启动过程中，隧道端点可以通过各种方式验证
>  验证可以在 VPN 启动时立即进行 (例如通过端点 IP 地址白名单)，也可以在实际的隧道被激活后才进行 (例如通过网络门户)

Remote-access VPNs, which are typically user-initiated, may use [passwords](https://en.wikipedia.org/wiki/Passwords "Passwords"), [biometrics](https://en.wikipedia.org/wiki/Biometrics "Biometrics"), [two-factor authentication](https://en.wikipedia.org/wiki/Two-factor_authentication "Two-factor authentication"), or other [cryptographic](https://en.wikipedia.org/wiki/Cryptographic "Cryptographic") methods. People initiating this kind of VPN from unknown arbitrary network locations are also called "road-warriors". In such cases, it is not possible to use originating network properties (e.g. IP addresses) as secure authentication factors, and stronger methods are needed.
>  Remote-access VPN 通常是用户发起的，其验证方式可能会使用密码、生物识别、双因素认证或其他加密方法
>  从任意未知网络发起 Remote-access VPN 的用户称为 road-warriors，在这类情况下，无法使用来源网络属性 (例如 IP 地址) 作为身份验证因素

>  Site-to-site VPN 通常使用密码 (预先共享的 keys) 或数字证书进行验证，取决于 VPN 协议，互相连接的 sites 可以存储 key 用于自动建立 VPN 隧道，无需管理员干预

## VPN protocols to highlight
A virtual private network is based on a tunneling protocol, and may be possibly combined with other network or application protocols providing extra capabilities and different security model coverage.
>  VPN 基于隧道协议，并且可以与其他网络协议或应用协议结合，提供额外的功能和安全模型

- [Internet Protocol Security](https://en.wikipedia.org/wiki/Internet_Protocol_Security "Internet Protocol Security") ([IPsec](https://en.wikipedia.org/wiki/Internet_Protocol_Security "Internet Protocol Security")) was initially developed by the [Internet Engineering Task Force](https://en.wikipedia.org/wiki/Internet_Engineering_Task_Force "Internet Engineering Task Force") (IETF) for [IPv6](https://en.wikipedia.org/wiki/IPv6 "IPv6"), and was required in all standards-compliant implementations of IPv6 before [RFC](https://en.wikipedia.org/wiki/RFC_\(identifier\) "RFC (identifier)") [6434](https://datatracker.ietf.org/doc/html/rfc6434) made it only a recommendation. [9](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-rfc6434-9) This standards-based security protocol is also widely used with [IPv4](https://en.wikipedia.org/wiki/IPv4 "IPv4"). Its design meets most security goals: [availability, integrity, and confidentiality](https://en.wikipedia.org/wiki/Information_security#Key_concepts "Information security"). IPsec uses encryption, [encapsulating](https://en.wikipedia.org/wiki/Encapsulation_\(networking\) "Encapsulation (networking)") an IP packet inside an IPsec packet. De-encapsulation happens at the end of the tunnel, where the original IP packet is decrypted and forwarded to its intended destination. IPsec tunnels are set up by [Internet Key Exchange (IKE)](https://en.wikipedia.org/wiki/Internet_Key_Exchange "Internet Key Exchange") protocol. IPsec tunnels made with IKE version 1 (also known as IKEv1 tunnels, or often just "IPsec tunnels") can be used alone to provide VPN, but have been often combined to the [Layer 2 Tunneling Protocol (L2TP)](https://en.wikipedia.org/wiki/Layer_2_Tunneling_Protocol "Layer 2 Tunneling Protocol"). Their combination made possible to reuse existing L2TP-related implementations for more flexible authentication features (e.g. [Xauth](https://en.wikipedia.org/w/index.php?title=XAUTH&action=edit&redlink=1 "XAUTH (page does not exist)")), desirable for remote-access configurations. IKE version 2, which was created by Microsoft and Cisco, can be used alone to provide IPsec VPN functionality. Its primary advantages are the native support for authenticating via the [Extensible Authentication Protocol (EAP)](https://en.wikipedia.org/wiki/Extensible_Authentication_Protocol "Extensible Authentication Protocol") and that the tunnel can be seamlessly restored when the IP address of the associated host is changing, which is typical of a roaming mobile device, whether on [3G](https://en.wikipedia.org/wiki/3G "3G") or [4G](https://en.wikipedia.org/wiki/4G "4G") [LTE](https://en.wikipedia.org/wiki/LTE_\(telecommunication\) "LTE (telecommunication)") networks. IPsec is also often supported by network hardware accelerators, [10](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-10) which makes IPsec VPN desirable for low-power scenarios, like always-on remote access VPN configurations. [11](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-11) [12](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-12)
>  Internet Protocol Security (IPsec)
>  IPsec 最初由 IETF 为 IPv6 开发，并且在所有符合标准的 IPv6 实现中都包含
>  该安全协议也广泛用于 IPv4，其设计满足了大多数安全目标: 可用性、完整性、保密性
>  IPsec 使用加密技术，将一个 IP 包封装在一个 IPsec 包内，隧道的末端对 IPsec 包解封，并解码出原 IP 包，然后转发到其预定目的地
>  IPsec 隧道通过 IKE 协议建立，使用 IKE version 1 就可以单独提供 VPN 功能，但通常回合 L2TP 结合使用，以复用现存的 L2TP 相关实现，进行更灵活的身份认证，这对于 remote-access VPN 非常有用
>  IKE version 2 可以单独提供 IPsec VPN 功能，其主要优势在于原生支持通过 EAP 进行身份认证，并且当相关主机的 IP 地址发生变化时，可以无缝恢复 (这类场景在漫游移动设备上很常见)
>  IPsec 也常由网络硬件加速器支持，故 IPsec VPN 在低功耗场景下 (例如始终在线的 remote-access VPN) 是理想选择

[![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/IPSec_VPN-en.svg/330px-IPSec_VPN-en.svg.png)](https://en.wikipedia.org/wiki/File:IPSec_VPN-en.svg)

The life cycle phases of an IPSec tunnel in a virtual private network

- [Transport Layer Security](https://en.wikipedia.org/wiki/Transport_Layer_Security "Transport Layer Security") ([SSL/TLS](https://en.wikipedia.org/wiki/Transport_Layer_Security "Transport Layer Security")) can tunnel an entire network's traffic (as it does in the [OpenVPN](https://en.wikipedia.org/wiki/OpenVPN "OpenVPN") project and [SoftEther VPN](https://en.wikipedia.org/wiki/SoftEther_VPN "SoftEther VPN") project [13](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-13)) or secure an individual connection. A number of vendors provide remote-access VPN capabilities through TLS. A VPN based on TLS can connect from locations where the usual TLS web navigation ([HTTPS](https://en.wikipedia.org/wiki/HTTPS "HTTPS")) is supported without special extra configurations,
>  Transport Layer Security (SSL/TLS)
>  TLS 可以对整个网络的流量进行隧道传输，也可以仅仅保护单个连接
>  许多供应商通过 TLS 提供 remote-access VPN 功能
>  基于 TLS 的 VPN 可以在通常支持 TLS 网页浏览 (HTTPS) 的地方连接，无需进行额外配置

- [Datagram Transport Layer Security](https://en.wikipedia.org/wiki/Datagram_Transport_Layer_Security "Datagram Transport Layer Security") ([DTLS](https://en.wikipedia.org/wiki/Datagram_Transport_Layer_Security "Datagram Transport Layer Security")) – used in Cisco [AnyConnect](https://en.wikipedia.org/wiki/AnyConnect "AnyConnect") VPN and in [OpenConnect](https://en.wikipedia.org/wiki/OpenConnect "OpenConnect") VPN [14](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-14) to solve the issues [TLS](https://en.wikipedia.org/wiki/Transport_Layer_Security "Transport Layer Security") has with tunneling over [TCP](https://en.wikipedia.org/wiki/Transmission_Control_Protocol "Transmission Control Protocol") (SSL/TLS are TCP-based, and tunneling TCP over TCP can lead to big delays and connection aborts [15](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-15)).
>  Datagram Transport Layer Security (DTLS)
>  DTLS 用于解决 TLS 在 TCP 隧道上的问题 (SSL/TLS 都基于 TCP，而将 TCP 隧道封装在 TCP 中可能导致大量延迟和连接中断)

- [Microsoft Point-to-Point Encryption](https://en.wikipedia.org/wiki/Microsoft_Point-to-Point_Encryption "Microsoft Point-to-Point Encryption") ([MPPE](https://en.wikipedia.org/wiki/Microsoft_Point-to-Point_Encryption "Microsoft Point-to-Point Encryption")) works with the [Point-to-Point Tunneling Protocol](https://en.wikipedia.org/wiki/Point-to-Point_Tunneling_Protocol "Point-to-Point Tunneling Protocol") and in several compatible implementations on other platforms.
- Microsoft [Secure Socket Tunneling Protocol](https://en.wikipedia.org/wiki/Secure_Socket_Tunneling_Protocol "Secure Socket Tunneling Protocol") ([SSTP](https://en.wikipedia.org/wiki/Secure_Socket_Tunneling_Protocol "Secure Socket Tunneling Protocol")) tunnels [Point-to-Point Protocol](https://en.wikipedia.org/wiki/Point-to-Point_Protocol "Point-to-Point Protocol") (PPP) or Layer 2 Tunneling Protocol traffic through an [SSL/TLS](https://en.wikipedia.org/wiki/Transport_Layer_Security "Transport Layer Security") channel (SSTP was introduced in [Windows Server 2008](https://en.wikipedia.org/wiki/Windows_Server_2008 "Windows Server 2008") and in [Windows Vista](https://en.wikipedia.org/wiki/Windows_Vista "Windows Vista") Service Pack 1).
- Multi Path Virtual Private Network (MPVPN). Regula Systems Development Company owns the registered [trademark](https://en.wikipedia.org/wiki/Trademark "Trademark") "MPVPN".[relevant?](https://en.wikipedia.org/wiki/Wikipedia:Writing_better_articles#Stay_on_topic "Wikipedia: Writing better articles")_ [16](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-16)

- Secure Shell (SSH) VPN – [OpenSSH](https://en.wikipedia.org/wiki/OpenSSH "OpenSSH") offers VPN tunneling (distinct from [port forwarding](https://en.wikipedia.org/wiki/Port_forwarding "Port forwarding")) to secure remote connections to a network, inter-network links, and remote systems. OpenSSH server provides a limited number of concurrent tunnels. The VPN feature itself does not support personal authentication. [17](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-17) SSH is more often used to remotely connect to machines or networks instead of a site to site VPN connection.
>  Secure Shell
>  OpenSSH 提供了 VPN 隧道功能 (与端口转发不同) 以保护到某个网络的远程连接、网络间链路和远程系统
>  OpenSSH 服务器仅提供有限数量的并发隧道，其 VPN 特性不支持个人认证
>  SSH 更多用于远程连接到某个机器或网络而非到 site-to-site VPN 连接

- [WireGuard](https://en.wikipedia.org/wiki/WireGuard "WireGuard") is a protocol. In 2020, WireGuard support was added to both the Linux [18](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-18) and Android [19](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-19) kernels, opening it up to adoption by VPN providers. By default, WireGuard utilizes the [Curve25519](https://en.wikipedia.org/wiki/Curve25519 "Curve25519") protocol for [key exchange](https://en.wikipedia.org/wiki/Key_exchange "Key exchange") and [ChaCha20-Poly1305](https://en.wikipedia.org/wiki/ChaCha20-Poly1305 "ChaCha20-Poly1305") for encryption and message authentication, but also includes the ability to pre-share a symmetric key between the client and server. [20](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-20)

- [OpenVPN](https://en.wikipedia.org/wiki/OpenVPN "OpenVPN") is a [free and open-source](https://en.wikipedia.org/wiki/Free_and_open-source_software "Free and open-source software") VPN protocol based on the TLS protocol. It supports perfect [forward-secrecy](https://en.wikipedia.org/wiki/Forward_secrecy "Forward secrecy"), and most modern secure cipher suites, like [AES](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard "Advanced Encryption Standard"), [Serpent](https://en.wikipedia.org/wiki/Serpent_\(cipher\) "Serpent (cipher)"), [TwoFish](https://en.wikipedia.org/wiki/Twofish "Twofish"), etc. It is currently being developed and updated by OpenVPN Inc., a [non-profit](https://en.wikipedia.org/wiki/Nonprofit_organization "Nonprofit organization") providing secure VPN technologies.
>  OpenVPN
>  OpenVPN 是免费且开源的 VPN 协议，基于 TLS 协议
>  它支持完美的前向保密，以及大多数现代安全加密套件，例如 AES, Serpent, TwoFish

- Crypto IP Encapsulation (CIPE) is a free and open-source VPN implementation for tunneling [IPv4 packets](https://en.wikipedia.org/wiki/IPv4_packet "IPv4 packet") over [UDP](https://en.wikipedia.org/wiki/User_Datagram_Protocol "User Datagram Protocol") via [encapsulation](https://en.wikipedia.org/wiki/Encapsulation_\(networking\) "Encapsulation (networking)"). [21](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-21) CIPE was developed for [Linux](https://en.wikipedia.org/wiki/GNU/Linux "GNU/Linux") operating systems by Olaf Titz, with a [Windows](https://en.wikipedia.org/wiki/Windows_2000 "Windows 2000") [port](https://en.wikipedia.org/wiki/Port_\(software\) "Port (software)") implemented by Damion K. Wilson. [22](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-22) Development for CIPE ended in 2002. [23](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-23)
>  Crypto IP Encapsulation (CIPE)
>  CIPE 是免费且开源的 VPN 协议，该协议在 UDP 上传输封装后的 IPv4 包
>  CIPE 针对 Linux 开发

## Trusted delivery networks
Trusted VPNs do not use cryptographic tunneling; instead, they rely on the security of a single provider's network to protect the traffic. [24](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-24)
>  受信任的 VPN 不使用加密通道，而是依赖于单个供应商网络的安全性来保护流量

- [Multiprotocol Label Switching](https://en.wikipedia.org/wiki/Multiprotocol_Label_Switching "Multiprotocol Label Switching") (MPLS) often overlays VPNs, often with quality-of-service control over a trusted delivery network.
- L2TP [25](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-25) which is a standards-based replacement, and a compromise taking the good features from each, for two proprietary VPN protocols: Cisco's [Layer 2 Forwarding (L2F)](https://en.wikipedia.org/wiki/L2F "L2F") [26](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-26) (obsolete as of 2009) and Microsoft's [Point-to-Point Tunneling Protocol (PPTP)](https://en.wikipedia.org/wiki/Point-to-Point_Tunneling_Protocol "Point-to-Point Tunneling Protocol"). [27](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-27)

From a security standpoint, a VPN must either trust the underlying delivery network or enforce security with a mechanism in the VPN itself. Unless the trusted delivery network runs among physically secure sites only, both trusted and secure models need an authentication mechanism for users to gain access to the VPN.
>  从安全性的角度来看，VPN 要么信任底层传输网络，要么在 VPN 本身中采用额外机制来确保安全
>  除非受信任的传输网络仅在物理单圈的站点间运行，否则无论是受信任模型还是安全模型都需要一种身份验证机制，验证访问 VPN 用户的身份

## VPNs in mobile environments
[Mobile virtual private networks](https://en.wikipedia.org/wiki/Mobile_virtual_private_network "Mobile virtual private network") are used in settings where an endpoint of the VPN is not fixed to a single [IP address](https://en.wikipedia.org/wiki/IP_address_spoofing "IP address spoofing"), but instead roams across various networks such as data networks from cellular carriers or between multiple [Wi-Fi](https://en.wikipedia.org/wiki/Wi-Fi "Wi-Fi") access points without dropping the secure VPN session or losing application sessions. [28](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-Phifer-28) 
>  移动 VPN 用于 VPN 的终端地址不固定于单一 IP 地址的场景，例如移动终端的 IP 地址可能会从移动运营商的数据网络切换到不同的 WIFI 接入点，而 VPN 会话和应用程序会话需要保持而不断开

Mobile VPNs are widely used in [public safety](https://en.wikipedia.org/wiki/Public_safety "Public safety") where they give law-enforcement officers access to applications such as [computer-assisted dispatch](https://en.wikipedia.org/wiki/Computer-assisted_dispatch "Computer-assisted dispatch") and criminal databases, [29](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-29) and in other organizations with similar requirements such as [field service management](https://en.wikipedia.org/wiki/Field_service_management "Field service management") and healthcare. [30](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-Cheng-30)
## Networking limitations
A limitation of traditional VPNs is that they are point-to-point connections and do not tend to support [broadcast domains](https://en.wikipedia.org/wiki/Broadcast_domain "Broadcast domain"); therefore, communication, software, and networking, which are based on [layer 2](https://en.wikipedia.org/wiki/OSI_layer "OSI layer") and broadcast [packets](https://en.wikipedia.org/wiki/Network_packet "Network packet"), such as [NetBIOS](https://en.wikipedia.org/wiki/NetBIOS "NetBIOS") used in [Windows networking](https://en.wikipedia.org/wiki/My_Network_Places "My Network Places"), may not be fully supported as on a [local area network](https://en.wikipedia.org/wiki/Local_area_network "Local area network"). Variants on VPN such as [Virtual Private LAN Service](https://en.wikipedia.org/wiki/Virtual_Private_LAN_Service "Virtual Private LAN Service") (VPLS) and layer 2 tunneling protocols are designed to overcome this limitation. [31](https://en.wikipedia.org/wiki/Virtual_private_network#cite_note-31)
>  传统 VPN 的限制是它们是点对点连接，通常不支持广播域
>  因此，基于第二层和广播包的通讯可能不受支持
>  VPN 的变体例如 VPLS 和 layer 2 tunneling 协议被设计于克服该限制

>  This page was last edited on 28 March 2025, at 15:40 (UTC).