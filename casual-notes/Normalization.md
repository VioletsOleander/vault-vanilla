# Four Normalizations
![[Four Normalizations.png| Four Normalizations]]
- Batch Normalization
- Layer Normalization
- Instance Normalization
- Group Normalization
## 1. Batch Normalization
![[Batch Norm.png | Batch Normalization]]
$$\begin{aligned}
x &\in \mathbb R^{N\times C\times H\times W} \\ \\
u_C & = \frac 1 {NWH}\sum_{i=1}^N\sum_{j=1}^W\sum_{k=1}^H x_{iCjk} \\ \\
\sigma_C^2 &= \frac 1 {NWH}\sum_{i=1}^N\sum_{j=1}^W\sum_{k=1}^H (x_{iCjk}-u_C)^2 \\ \\
\hat x &= \frac {x-u_C}{\sqrt{\sigma_C^2+\epsilon}}
\end{aligned}$$
$u_c$是channel c上的所有pixel的平均值
$\sigma_c^2$是channel c上的所有pixel的方差
每个channel有自己独立的Normalization Factor
一共C个channel
C个channel的Normalization Factor一起用向量$u_C$和$\sigma_C^2$表示
## 2. Layer Normalization
![[Layer Norm.png| Layer Norm]]
$$\begin{aligned}
x &\in \mathbb R^{N\times C\times H\times W} \\ \\
u_N & = \frac 1 {CWH}\sum_{i=1}^C\sum_{j=1}^W\sum_{k=1}^H x_{Nijk} \\ \\
\sigma_N^2 &= \frac 1 {CWH}\sum_{i=1}^C\sum_{j=1}^W\sum_{k=1}^H (x_{Nijk}-u_N)^2 \\ \\
\hat x &= \frac {x-u_N}{\sqrt{\sigma_N^2+\epsilon}}
\end{aligned}$$
$u_n$是样本n的所有pixel的平均值
$\sigma_n^2$是样本n的所有pixel的方差
每个样本有自己独立的Normalization Factor
一共N个样本
N个样本的Normalization Factor一起用向量$u_N$和$\sigma_N^2$表示
## 3. Instance Normalization
![[Instance Norm.png|Instance Norm]]
$$\begin{aligned}
x &\in \mathbb R^{N\times C\times H\times W} \\ \\
u_{NC} & = \frac 1 {WH}\sum_{j=1}^W\sum_{k=1}^H x_{NCjk} \\ \\
\sigma_{NC}^2 &= \frac 1 {WH}\sum_{j=1}^W\sum_{k=1}^H (x_{NCjk}-u_{NC})^2 \\ \\
\hat x &= \frac {x-u_{NC}}{\sqrt{\sigma_{NC}^2+\epsilon}}
\end{aligned}$$
$u_{nc}$是样本n的channel c上的所有pixel的平均值
$\sigma_{nc}^2$是样本n的channel c上的所有pixel的方差
每个样本的每个channel有自己独立的Normalization Factor
一共N个样本，每个样本有C个channel
每个样本的每个channel的Normalization Factor，一共N$\times C$个Normalization Factor用二维矩阵$u_{NC}$和$\sigma_{NC}^2$表示
## 4. Group Normalization
![[Group Norm.png|Group Norm]]
$$\begin{aligned}
x \in \mathbb R^{N\times C\times H\times W} &\rightarrow\ \ x\in \mathbb R^{N\times G\times C'\times H\times W}\quad C=G\times C' \\ \\
G &= number\  of\  groups \\\\
C'&= number\ of\ channel\ per\ group\\\\
u_{NG} & = \frac 1 {C'WH}\sum_{i=1}^C'\sum_{j=1}^W\sum_{k=1}^H x_{NGjk} \\ \\
\sigma_{NG}^2 &= \frac 1 {C'WH}\sum_{i=1}^C'\sum_{j=1}^W\sum_{k=1}^H (x_{NGjk}-u_{NG})^2 \\ \\
\hat x &= \frac {x-u_{NG}}{\sqrt{\sigma_{NG}^2+\epsilon}}
\end{aligned}$$
$u_{ng}$是样本n的group channel g(一个group channel由C个channel组成)上的pixel的平均值
$\sigma_{ng}^2$是样本n的group channel g(一个group channel由C'个channel组成)上的所有pixel的方差
每个样本的每个group channel有自己独立的Normalization Factor
一共N个样本，每个样本有G个group channel，每个group channel中包含了C‘个channel
每个样本的每个group channel的Normalization Factor，一共N$\times G$个Normalization Factor用二维矩阵$u_{NG}$和$\sigma_{NG}^2$表示