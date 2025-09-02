# Abstract
不对称比率相似度矩阵中，正负权重的绝对值是不同的
# 1 Introduction
常规的哈希方法中，目标函数的形式是
$$
\begin{align}
&L = \min_B \|K\cdot S-BB^T\|_F^2\\
&s.t. B\in\{-1,1\}^{N\times K}
\end{align}\tag{1}
$$
$K$为哈希码长度，$S = \{s_{ij}\}_{i,j=1}^N$为相似度矩阵
现存的哈希方法中，相似度矩阵$S$是对称比率矩阵

在零样本学习场景中，采用对称比率相似度矩阵容易过度学习可见类之间的相似度信息，而采用不对称比率相似度矩阵可以让损失函数强调不同类之间哈希码的差异性

重定义相似度矩阵，当样本不同类，$s_{ij} = r^d$，当样本同类，$s_{ij} = r^s$，定义相似度权重比率为$r^s/r^d$
在AWA2上的实验发现如果$r^s > r^d$，mAP会提高，一个可能的原因可能是对称的相似度矩阵会导致在可见类上的过拟合，因此损害了模型迁移知识的能力，而不对称的相似度矩阵可以消除这一影响

一些常规的哈希方法也利用了非对称比率相似度矩阵，例如HashNet，但这些方法意在处理不平衡样本类别的问题，和零样本哈希的问题不同
我们的方法意图解决零样本哈希方法中的过拟合的问题
# 2 Related Work
## 2.1 Hashing
## 2.2 Zero-Shot Hashing
本项工作只利用了类别标签的监督信息，不包括任意特殊的监督信息，例如语义或属性信息

# 3 Methodology
## 3.1 Problem Definition
数据集$\mathbf X \in \mathbb R^{N\times D}$，标签矩阵$\mathbf Y \in \{0,1\}^{N\times C}$
## 3.2 Analyses of Asymmetric Ratio Similarity Matrix to Hash Learning
> 对该部分的分析持怀疑态度

不可见类上的测试结果相对于可见类上的测试结果的对比表明不可见类对相似度矩阵的变化更为敏感

如果样本类别相同，我们希望它们哈希码的内积是$K$，否则是$-K$，正好对应$r^s/r^d = 1/-1$，但在零样本场景下，这容易导致不同类别间不足的差异性(insufficient distinction)，因为对称的相似度矩阵对两边的惩罚是一致的，这使得模型难以很好地辨别不可见类之间的差异，而可见类则处理地较好，我们认为这是一个过拟合现象
在零样本场景下，损失函数应该对不同类别之间的差异性更加注意，以区分出样本是否属于已知类

通常情况下$K\cdot S$的元素的范围是$[-K,K]$，如果$r^s/r^d = n/-1,n>1$，则$K\cdot S$的元素的范围是$[-K,nK]$，此时损失项中的$\|K\cdot S - BB^T\|_F^2$一定不会为零，因此，可以避免过度学习训练数据

当比率是$n/-1$时，项$\|K\cdot S - BB^T\|_F^2$可以让哈希码满足$K\cdot S$中的下界$-K$，但上界$nK$则永远不可能达到，上述分析说明上下界不能同时达到
因为损失函数可以满足$K\cdot S$中的下界，不同类别的样本的哈希码的内积在优化过程中会逐渐接近$-K$，因此损失函数会对不同类别样本的哈希码的区别性更加注意(优化过程中，不同类别的样本之间的哈希码会被优化，而相似类别样本之间的哈希码的优化则不明显)
当比率是$1/-n$时，损失函数则会对相同类别的哈希码的统一性更加注意

> 考虑往哪边优化对降低损失最有利
> 都从哈希码内积是$0$开始，$-K$对损失的贡献是$K^2$，$nK$对损失的贡献是$n^2K^2$
> 不相似优化到极限，哈希码内积为$-K$，此时$-K$对损失的贡献是$0$，因此损失降低了$K^2 - 0 = K^2$
> 相似优化到极限，哈希码内积为$K$，此时$nK$对损失的贡献是$(n-1)^2K^2$，因此损失降低$n^2K^2 - (n-1)^2K^2 = (2n-1)K^2$
> 显然当$n>1$时，$(2n-1)K^2 > K^2$，即相似优化到极限对降低损失更有利

零样本场景要求哈希函数区分不同的类别，当比率是$n/-1$，损失函数会对不同类别的哈希码的差异更加注意，相反，当比率时$1/-n$，损失函数会对相同类别的哈希码的统一性更加注意，因此在比率是从$1/-n$变化到$n/-1$时，准确率增加
## 3.3 Zero-Shot Hashing via Asymmetric Ratio Similarity Matrix
目标问题
$$\begin{align}
&\min_B \|K\cdot S-BB^T\|_F^2\\
&s.t.\begin{cases}B\in\{\pm 1\}^{N\times K}\\B^T1_N = 0_K,B^TB = N\cdot E_K\end{cases}
\end{align}\tag{5}$$
为了完全利用标签信息，引入从标签到哈希码的映射，重写目标
$$\begin{align}
&\min_{B,P,W} \|K\cdot S-BB^T\|_F^2+\lambda\|XP-B\|_F^2+\beta\|P\|_F^2\\
&+\gamma\|YW-B\|_F^2+\mu\|W\|_F^2\\
&s.t.\begin{cases}B\in\{\pm 1\}^{N\times K}\\B^T1_N = 0_K,B^TB = N\cdot E_K\end{cases}
\end{align}\tag{6}$$
引入$B$的连续近似$Z$
$$\begin{align}
&\min_{B,P,W,Z} \|K\cdot S-BZ^T\|_F^2+\lambda\|XP-B\|_F^2+\alpha\|B-Z\|_F^2\\
&+\beta\|P\|_F^2+\gamma\|YW-B\|_F^2+\mu\|W\|_F^2\\
&s.t.\begin{cases}B\in\{\pm 1\}^{N\times K}\\B^T1_N = 0_K,B^TB = N\cdot E_K\end{cases}
\end{align}\tag{7}$$
为了实现非线性哈希，引入核函数
$$\Phi(X) = [\Phi(\symbfit x_1),\Phi(\symbfit x_2),\cdots,\Phi(\symbfit x_n)]^T\tag{8}$$
其中$\Phi:\symbfit x \in \mathbb R^D \rightarrow \Phi(\symbfit x)\in \mathbb R^{D'}$
故得到
$$\begin{align}
&\min_{B,P,W,Z} \|K\cdot S-BZ^T\|_F^2+\lambda\|\Phi(X)P-B\|_F^2+\alpha\|B-Z\|_F^2\\
&+\beta\|P\|_F^2+\gamma\|YW-B\|_F^2+\mu\|W\|_F^2\\
&s.t.\begin{cases}B\in\{\pm 1\}^{N\times K}\\B^T1_N = 0_K,B^TB = N\cdot E_K\end{cases}
\end{align}\tag{9}$$
## 3.4 Optimization
