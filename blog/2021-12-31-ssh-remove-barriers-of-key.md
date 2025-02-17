---
title: 干掉口令卡 (关于 ssh 配置)
slug: ssh-remove-barriers-of-key
author: genshen
# authorTitle: Front End Engineer @ Facebook
authorURL: https://github.com/genshen
authorImageURL: https://avatars3.githubusercontent.com/u/11265498?s=460&v=4
tags: [ssh, sftp]
---

## 背景
实验室开通了一项计算服务，其登录节点采用 ssh 的方式登录，但是密码是采用 “固定密码+动态口令” 形式组合构成的。
主要的问题是，这个动态口令太不方便了。
它是一个动态口令卡，一般一个人拿了该口令卡，其他人的登录的时候就麻烦得很：
1. 每次登录必须输入密码和口令；
2. 如果自己不持有口令卡，得向其他人询问动态口令。
> 注意，我们是多人共享一个账号（即linux 用户），所有人都用这个 linux 用户操作。

现在就想能否在保证一定安全性的前提下，减少这些麻烦。
说到这，必须得继续吐槽下，这个实体的口令卡确实不像 google 验证器这类的软件那么方便，如果采用 google 验证器，
也就没有本文的这些 “旁门左道”了。

## 第一次尝试：ssh 多路复用
所谓 ssh 多路复用，它可以让你 ssh 连接了远程服务器后，第二次及以后的连接就可以复用第一次的连接。
这样带来的好处是，我们可以仅在第一次连接的时候输入密码+口令，以后的连接就不用输入了。  

当然，这里还有一个小问题。如果每人都在自己的电脑上配置多路复用，只能解决以上的问题1，而不能解决问题2。
为此，得让多路复用的连接**共享**起来。
于是，想起实验室的计算集群可以完成这个任务，然后就拿这个开始了配置~~（有管理员权限就是不一样）~~。
为了共享 ssh 连接，我们先在计算集群创建一个公用的用户，就叫 pub_test 吧。
然后修改 pub_test 用户的 ssh 配置。

<!--truncate-->

```bash
# cat .ssh/config
Host remote
    Hostname remote.com
    Port 65010
    User remote-user
    UserKnownHostsFile /dev/null
    ControlMaster auto
    ControlPath ~/.ssh/%r@%h:%p.socket
    ControlPersist 100h
```
这用，我们原本我们每次需要通过密码+口令连接到的远程（ssh -p 22 remote-user@remote.com）,
现在就可以通过多路复用连接了（ssh remote）。
当然，第一次 ssh remote 命令进行连接时，还是需要输入密码+口令，以后就不用了。
另外，这里，我将 `ControlPersist` 参数设置为 100小时，即 100小时没人用，连接就断开了。
后面如果过来 100小时，还想继续用，就得重新输入密码+口令进行认证了。
其中，这里的核心配置是 `ControlMaster` 选项，关于多路复用的进一步的用法，可以参考 [这里](https://vqiu.cn/ssh-multiplexing/)。

这样，每一个能以 pub_test 身份登录进实验室集群的人，都可以免密码地连接到远程计算服务了。
具体为：每个人用自己的账号登录实验室集群，然后通过 pub_test 用户的密码 su 到这个叫 pub_test 公用用户
（因为考虑安全性，实验室集群不支持 ssh 密码登录，所以必须这么转一下），最后用这个 pub_test 用户的身份执行 `ssh remote`命令即可。

## 问题
上述方案在 ssh 命令行下工作得很舒服，但是有时候我们还有一些额外需求。
例如，我们想开个 vs code 连接到远程计算服务 remote 上写代码，这个需求目前基本就没法实现
（如果要用 vs code 只能回退到远古时代的密码+口令的方式了）。

更要命的是，上传/下载文件很麻烦。
例如下载：我们先得要在实验室集群 pub_test 用户下，通过 sftp 连接到远程计算服务 remote 上，来下载文件。
再将其拷贝到实验室集群的自己的账号下，
最后自己的电脑需要从实验室集群的自己账号下面将文件拷贝到自己的电脑 （~~简直一遍一遍的套娃🪆过程~~）。

## 改进的方案
### 远程转发
然后，就想着能否用 ssh 的端口转发功能来搞事情。
例如下面的本地端口转发命令：
```bash
A: ssh -N user@B -L 3380:C:3389
```
如果我们在 A 机器上执行该 ssh 端口转发命令（假设 B 节点可以直接访问 C 节点），那么，访问 A 机器的 3380 端口，等效于访问 C 机器的 3389 端口。

这里，A 节点就是实验室的集群（我们称A节点为 node01 吧），然后 B 节点就是远程的计算服务的登录节点（我们称为 remote 吧）。
然后，C 节点可以继续是 remote 本身。
差不多的命令长这样：
```bash
node01: ssh -N remote -L 2222:login09:22
```
<!-- （这里，C节点写为了 localhost，即 remote节点本身。但是是通过 localhost 来访问的，而不是通过 remote 节点的公用 ip 地址访问的。
因为，如果这里继续按照 ip 地址访问，可能就用不了远程计算服务的 22 端口了。
） -->
这里，C 节点写为了 login09。
主要是由于远程的计算服务有两个特点：1）其暴露了 65010 端口供外网的用户 ssh 登录进来，该登录仅支持密码+口令的方式（即前文所述的传统的登录模式）。
另外，其还有一个端口是 22，这个端口可以通过私钥无密码登录（假设私钥文件为 id_rsa_remote ），但是该端口不会暴露到外网。
2）远程的计算服务包含多个进行负载均衡的登录节点，可以在一个登录节点 ssh 到其他的登录节点上（通过 22 端口登录）。

这里的端口转发，就相当于先从 65010 端口，登录到 remote 的计算服务的某个登录节点，
然后再通过 22 端口 ssh 到 login09 节点上。

这样以后，我们就可以在 node01上，使用下面的命令，无密码地访问远程的计算服务啦:
```bash
node01: ssh remote-user@localhost -p 2222 -i ~/.ssh/id_rsa_remote
```
这里，使用的私钥文件是 ~/.ssh/id_rsa_remote，也是用于从远程计算服务的某个登录节点无密码地登录到 login09 节点上的私钥文件。
我们需要从远程计算服务上，将该私钥文件下载到 node01 中。

### 开放端口
目前，只有用户登录的 node01 上，才能通过 node01 的 2222 端口无密码地访问远程的计算服务。
如果我们想让用户从自己电脑上直接访问远程计算服务，我们可以将 2222 端口开放。
例如，centos 7 上，我们可以用以下的命令开放 2222 端口（仅 TCP 协议就行）：
   ```bash
   sudo firewall-cmd --zone=public --permanent --add-port=2222/tcp
   sudo firewall-cmd --reload
   ```

另外，还有一个小细节。
我们用 `netstat -lnpt` 命令进程监听的端口，发现 ssh 端口转发所监听的地址是 127.0.0.1，而非 0.0.0.0。
```log
netstat -lnpt
(Not all processes could be identified, non-owned process info
 will not be shown, you would have to be root to see it all.)
Active Internet connections (only servers)
Proto Recv-Q Send-Q Local Address           Foreign Address         State       PID/Program name    
tcp        0      0 127.0.0.1:2222          0.0.0.0:*               LISTEN      4460/ssh: /home/gen 
```

这样似乎就没法让外面的用户用自己的电脑连接到 node01 的 2222 端口了。
解决方案在这里：https://stackoverflow.com/q/23781488。加上个 `GatewayPorts` 参数就行。

## 小结
总结一下配置的过程。  

### 持久化连接的配置（管理员）
对于 node01 上的任意某一个用户 G，在 node01的自己账号下：
1. 配置其 ssh 的 config 文件如下：
    ```bash
    # cat .ssh/config
    Host remote
        Hostname remote.com
        Port 65010
        User remote-user
        UserKnownHostsFile /dev/null
        ControlMaster auto
        ControlPath ~/.ssh/%r@%h:%p.socket
        ControlPersist 100h
    ```
2. 执行端口转发命令：
   ```bash
   ssh -o GatewayPorts=true -N remote -L 2222:login09:22
   # 可以再加上 -f 参数，以后台运行。
   ```
### 开放端口（管理员）
通过root权限，开放 node01 的 2222 端口，这样实验室的任何一用户都可以从自己的电脑上访问 2222 端口。
（如果不这样，用户还得登录到 node01 上，才能使用远程的计算服务，还是略麻烦）

   ```bash
   # Centos 7
   sudo firewall-cmd --zone=public --permanent --add-port=2222/tcp
   # sudo firewall-cmd --zone=public --permanent --remove-port=2222/tcp
   sudo firewall-cmd --reload
   ```

### 普通用户的配置
对于需要使用远程计算服务资源的人，采用如下配置：

1. 配置 ssh 无密码认证：生成一对公钥和密钥，私钥存在自己电脑上（如位置：`~/.ssh/id_rsa_remote`），在远程的计算服务的登录节点上配置公钥（公钥保存到`authorized_keys`中）。

2. 在**自己电脑**上，直接用上面开放的端口、地址及用户名，执行ssh连接：
    ```bash
    ssh -p 2222 remote-user@node01 -i ~/.ssh/id_rsa_remote  # 这里写成自己电脑访问 node01 的 ip 或者域名，私钥采用前一步生成的。
    ```
   或者采用**另一种方案**，是将配置写进 ssh 的 config 文件中，

    ```bash
    Host remote-forward
        Hostname node01 # 这里写成自己电脑访问node01 的 ip 或者域名
        Port 2222
        User remote-user
        IdentityFile ~/.ssh/id_rsa_remote
    ```

    然后执行 `ssh remote-forward` 即可。

---

至此，终于干掉了口令卡（别长时间不用口令卡，到时候找不着口令卡放哪了😂）。  
当然，现在的方案也还有一些小小的问题，例如每次都用的是远程计算服务的 login09 节点，没有充分利用负载均衡。
