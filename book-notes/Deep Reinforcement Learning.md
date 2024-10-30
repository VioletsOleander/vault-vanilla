# Part1 基础知识
## 1 深度学习基础
### 1.1 线性模型
### 1.2 神经网络
### 1.3 反向传播和梯度下降
## 2 概率论基础与蒙特卡洛
### 2.1 概率论基础
随机变量$X$是一个不确定量，它的值取决于一个随机事件的结果，注意随机变量的观测值$x$只是数字而已，没有随机性

概率质量函数(Probability Mass Function)和概率密度函数(Probability Density Function/PDF)都用于表示随机变量$X$在确定的取值点$x$的可能性
- PMF描述离散概率分布，即$X$的取值范围$\mathcal X$是一个离散集合
	PMF满足$\sum_{x\in \mathcal X}p(x) = 1$
- PDF描述连续概率分布，$X$是连续随机变量，正态分布是最常见的连续概率分布，$X$的取值范围$\mathcal X$是$\mathbb R$
	PDF满足$\int_{\mathcal X}p(x)dx = 1$

设$p(X)$是随机变量$X$的PMF或PDF，函数$f(X)$的期望定义为
$$\mathbb E_{X\sim P(\cdot)}[f(X)] = \sum_{x\in\mathcal X}p(x)f(x)$$
或
$$\mathbb E_{X\sim P(\cdot)}[f(X)] = \int_{x\in\mathcal X}p(x)f(x)dx$$

设$g(X,Y)$为二元函数，如果对$g(X,Y)$关于随机变量$X$求期望，那么会消掉$X$，得到的结果是$Y$的函数

随机变量通过不断的随机抽样，得到多个观测值
### 2.2 蒙特卡洛
蒙特卡洛(Monte Carlo)是一大类随机算法(Randomized Algorithms)的总称，它们通过随机样本来估算真实值
#### 2.2.1 例一：近似$\pi$值
考虑用蒙特卡洛方法近似估计$\pi$值

假设有一个(伪)随机数生成器，可以均匀生成$-1$到$+1$之间的数
每次生成两个随机数，一个作为$x$，另一个作为$y$，就生成了一个平面坐标系中的点$(x, y)$
因为$x,y$服从$[-1,1]$上的均匀分布，所以$[-1, 1]\times[-1, 1]$这个正方形内的所有点每次被抽到的概率是相同的，重复抽样$n$，得到$n$个正方形内的点

正方形包含了圆心是$(0,0)$，半径是$1$的圆，
考虑点落在圆内的概率，点落在正方形内的概率是$1$，而由于抽样是均匀的，点落在正方形内每个点的概率都相同，因此点落在圆内的概率就是圆的面积和正方形面积的比值，即$p = \pi / 4$

设随机抽样$n$个点，落在圆内的点的数量是随机变量$M$，则$M$的期望是
$$\mathbb E[M] = pn = \pi n/ 4$$
在$n$很大的时候，统计落在圆内的点的数量$m$(随机变量$M$的真实观测值)，$m$会渐进地逼近$\mathbb E[M]$，因此$m \approx \mathbb E[M] = \pi n/4$

故$$\pi \approx4m / n$$
大数定律保证了蒙特卡洛的正确性：当$n$趋于无穷， $4m/n$趋于$\pi$
#### 2.2.2 例二：估算阴影部分面积
思路依旧是用概率联系面积，用面积(含未知数)计算点落在阴影部分的概率，用概率计算落在阴影部分点的数量的期望值，然后用统计值替代期望值，反推回未知数
#### 2.2.3 例三：近似定积分
近似求积分是蒙特卡洛最重要的应用之一，有很多科学和工程问题需要计算定积分，而函数$f(x)$可能很复杂，求定积分会很困难，甚至有可能不存在解析解
如果求解析解很困难，或者解析解不存在，则可以用蒙特卡洛近似计算数值解

一元函数的定积分：给定一元函数$f(x)$，$x$是标量，求函数在$a$到$b$区间上的定积分
$$I = \int_a^b f(x)dx$$
蒙特卡洛方法通过下面的步骤近似定积分：
1. 在区间$[a,b]$上随机抽样，得到$n$个样本$x_1,\cdots,x_n$，$n$越大，近似越准确
2. 对函数值$f(x_1),\cdots, f(x_n)$求平均，再乘以区间长度，即$$q_n = (b-a)\frac 1 n\sum_{i=1}^nf(x_i)$$
3. 返回$q_n$作为定积分$I$的估计值

多元函数的定积分：给定多元函数$f:\mathbb R^d \rightarrow \mathbb R$，$\symbfit x$是$d$维向量，求函数集合$\Omega$上的定积分
$$I = \int_{\Omega} f(\symbfit x)d\symbfit x$$

蒙特卡洛方法通过下面的步骤近似定积分：
1. 在集合$\Omega$上随机抽样，得到$n$个样本，$n$越大，近似越准确
2. 计算集合$\Omega$的体积$$v = \int_{\Omega}d\symbfit x$$
3. 对函数值$f(\symbfit x_1),\cdots, f(\symbfit x_n)$求平均，再乘以集合体积，即$$q_n = v\frac 1 n\sum_{i=1}^nf(\symbfit x_i)$$
4. 返回$q_n$作为定积分$I$的估计值

注意，算法第二步需要求$\Omega$的体积。如果集合是长方体、球体等规则形状，那么可以解析地算出体积，如果是不规则形状，那么就需要定积分求体积，这是比较困难的，可以用类似于上一小节“求阴影部分面积”的方法近似计算体积
#### 2.2.4 例四：近似期望
设$f:\Omega \rightarrow \mathbb R$是任意的多元函数，它关于变量$X$的期望是：
$$\mathbb E_{X\sim p(\cdot)}[f(X)] = \int_{\Omega}p(\symbfit x)\cdot f(\symbfit x)d\symbfit x$$
由于期望是定积分，所以可以按照上一小节的方法，用蒙特卡洛求定积分
上一小节在集合$\Omega$上做均匀抽样，用得到的样本近似上面的定积分

下面介绍一种更好的算法，既然我们知道概率密度函数，我们最好是按照概率密度函数做非均匀抽样，而不是均匀抽样，按照概率密度函数做非均匀抽样，可以比均匀抽样有更快的收敛，具体步骤如下：
1. 按照概率密度函数$p(\symbfit x)$，在集合上做非均匀随机抽样，得到$n$个样本，记作向量$\symbfit x_1, \cdots , \symbfit x_n \sim p(\cdot)$
2. 对函数值$f(\symbfit x_1),\cdots, f(\symbfit x_n)$求平均：$$q_n=\frac 1n\sum_{i=1}^nf(\symbfit x_i)$$
3. 返回$q_n$作为期望的估计值

实现时，可以用Robbins-Monro算法减少内存开销，具体地说，即不需要全部存储函数值$f(\symbfit x_1),\cdots , f(\symbfit x_n)$，而是初始化$q_0 = 0$，对$t = 1,\cdots, n$，计算
$$q_t = (1-\frac 1 t)q_{t-1} + \frac 1 tf(\symbfit x_t)\tag{2.6}$$
#### 2.2.5 例五：随机梯度
蒙特卡洛近似期望在机器学习中的一个应用是随机梯度
神经网络的训练定义为该优化问题
$$\min_{\symbfit w}\mathbb E_{X\sim p(\cdot)}[L(X;\symbfit w)]\tag{2.7}$$
目标函数$\mathbb E_{X\sim p(\cdot)}[L(X;\symbfit w)]$关于$\symbfit w$的梯度是
$$\symbfit g = \nabla_{\symbfit w}\mathbb E_{X\sim p(\cdot)}[L(X;\symbfit w)]=\mathbb E_{X\sim p(\cdot)}[\nabla_{\symbfit w}L(X;\symbfit w)]$$
要最小化目标函数，梯度下降更新$\symbfit w$
$$\symbfit w \leftarrow \symbfit w - \alpha \symbfit g$$
而直接计算梯度$\symbfit g$会比较慢，因此可以对梯度作蒙特卡洛近似(注意梯度是一个期望值)，把得到的近似梯度$\tilde {\symbfit g}$称为随机梯度(Stochastic Gradient)用近似梯度代替原来的梯度

步骤和用蒙特卡洛近似期望的步骤一致
1. 按照概率密度函数$p(\symbfit x)$，在集合上做非均匀随机抽样，得到$b$个样本，记作向量$\symbfit x_1, \cdots , \symbfit x_b \sim p(\cdot)$
2. 计算梯度$\nabla_{\symbfit w}L(\symbfit x_i; \symbfit w)$，求平均$$\tilde {\symbfit g}=\frac 1 b\sum_{i=1}^b\nabla_{\symbfit w}L(\symbfit x_i; \symbfit w)$$
3. 做随机梯度下降更新$\symbfit w$$$\symbfit w \leftarrow \symbfit w - \alpha \tilde{\symbfit g}$$

蒙特卡洛抽取的样本数量$b$称作批量大小(Batch Size)，通常是一个比较小的整数，比如 1、8、 16、 32

实际情况下，随机变量$X$定义为服从概率质量函数
$$p(\symbfit x_i) = \mathbb P(X=\symbfit x_i)=\frac 1n$$
即随机变量$X$的取值是$n$个数据点中的一个，概率都是$1/n$
## 3 马尔可夫决策过程(MDP)
### 3.1 基本概念
马尔可夫决策过程(Markov Decision Process/MDP)通常由状态空间、动作空间、状态转移矩阵、奖励函数和折扣因子组成
RL是一个序贯决策过程，视图找到一个决策规则/策略，以使得系统获得最大的累计奖励值/最大价值

**状态(State)** 是对当前环境的一个概括，状态是做决策的唯一依据
**状态空间(State Space)** 指所有可能存在状态的集合，记作$\mathcal S$，状态空间可以是有限/无限集合
**动作(Action)** 指作出的决策
**动作空间(Action Space)** 指所有可能动作的集合，记作$\mathcal A$
**智能体(Agent)** 指做动作的主体
**策略函数(Policy Function)** 是根据观测到的状态做出决策，控制智能体的动作

把状态记为$S$或$s$，把动作记为$A$或$a$，策略函数$\pi: \mathcal S \times \mathcal A \rightarrow [0,1]$是一个条件概率密度函数：
$$\pi(a|s)=\mathbb P(A=a|S=s)$$
策略函数的输入是状态$s$和动作$a$，输出一个概率值

RL的学习对象就是策略函数$\pi$

**奖励(Reward)** 是在智能体执行一个动作之后，环境返回给智能体的一个数值，奖励往往由我们自己来定义，奖励定义得好坏非常影响RL的结果
**状态转移(State Transition)** 指当前状态$s$变为新的状态$s'$，给定状态$s$，智能体执行动作$a$，环境根据状态转移函数，将$(s,a)$映射到$s'$，给出下个状态$s'$
**环境(Environment)** 谁能生成新的状态，谁就是环境
**状态转移函数(State-Transition Function)** 即环境用于生成新的状态时所生成的函数
状态转移函数可以是确定的，没有随机性，也可能是随机的(即使当前状态$s$和智能体的动作$a$确定了，也无法确定下一个状态$s'$)
我们通常认为状态转移是随机的，该随机性来自于环境

**随机状态转移函数** 记作$p(s'|s,a)$，它是一个条件概率密度函数：
$$p(s'|s,a) = \mathbb P(S' = s'|S = s, A=a)$$
本书只考虑随机状态转移，因为确定状态转移是随机状态转移的一个特例，即条件密度都集中在一个状态$s'$上

**智能体与环境交互(Agent Environment Interaction)** 是指智能体观测到环境的状态$s$，做出动作$a$，动作会改变环境的状态，环境反馈给智能体奖励$r$以及新的状态$s'$
### 3.2 随机性的来源
随机性有两个来源：策略函数与状态转移函数

动作的随机性来自于策略函数，给定状态$s$，策略函数$\pi(a | s)$会计算出动作空间$\mathcal A$中每个动作$a$的概率值，智能体执行的动作则按照概率值进行随机抽样，带有随机性(对于确定的状态，智能体执行的动作是不确定的)

状态的随机性来自于状态转移函数，当状态$s$和动作$a$都确定时，下一个状态仍带有随机性，环境用状态转移函数$p(s'|s,a)$计算所有可能状态的概率，然后按照概率值进行随机抽样，得到新的状态

奖励可以看作动作和状态的函数，给定当前状态$s_t$和动作$a_t$，则奖励$r_t$是唯一确定的
若给定当前状态$s_t$，但智能体尚未决策，即动作还未知，应记作随机变量$A_t$，则奖励也未知，应记作随机变量$R_t$，因此奖励的随机性从未知的动作$A_t\sim \pi(\cdot|s_t)$中来

在很多应用中，奖励$r_t$取决于$s_t,a_t,s_{t+1}$，因此给定当前状态和动作，奖励$R_t$依旧是未知的变量，它的随机性从未知的新状态$s_{t+1}\sim p(\cdot |s_t,a_t)$中来

**轨迹(Trajectory)** 指一回合(Episode)游戏中，智能体观测到的所有状态、动作、奖励：$$s_1,a_1,r_1,\ s_2,a_2,r_2,\ s_3,a_3,r_3,\cdots$$在$t$时刻，给定状态$S = s_t$，以下都是观测到的值：
$$s_1,a_1,r_1,\ s_2,a_2,r_2,\ s_3,a_3,r_3,\cdots,s_{t-1},a_{t-1},r_{t-1},\ s_t$$
而尚未被观测到的值都是随机变量：
$$A_t,R_t,\ S_{t+1},A_{t+1},R_{t+1},\ S_{t+2},A_{t+2},R_{t+2}\cdots$$
### 3.3 回报与折扣回报
#### 3.3.1 回报
**回报(Return)** 是从当前时刻开始到一回合结束所有奖励的总和，因此回报也叫累计奖励(Cumulative Future Reward)
把$t$时刻的回报记为$U_t$，如果一局游戏结束，已经观察到所有的奖励，则把回报记为$u_t$，回报的定义是：
$$U_t = R_t + R_{t+1}+R_{t+2}+R_{t+3}+\cdots$$

回报是未来获得的奖励总和，因此智能体的目标就是最大化回报，
RL的目标就是寻找一个策略，可以最大化回报的期望

注意RL的目标是最大化回报而不是最大化当前的奖励
#### 3.3.2 折扣回报
因为未来的不确定性很大，因此未来的奖励换算到现在都要乘上折扣率，例如如果现在的80元和一年后的100元是同样好的，就意味着一年后的奖励的重要性只有今天的$\gamma =0.8$倍，这里的$\gamma = 0.8$就是**折扣率(Discount Factor)**

RL中，通常使用**折扣回报(Discounted Return)** 给未来的奖励做折扣，折扣回报定义为：
$$U_t = R_t + \gamma R_{t+1}+\gamma^2R_{t+2}+\gamma^3R_{t+3}+\cdots$$
其中$\gamma \in [0,1]$是折扣率，对待越久远的未来，奖励的折扣越大
折扣率是超参数，折扣率的设置会影响RL的结果

回报就是折扣回报在折扣率为1时候的特例
因此之后提到的回报都指折扣回报
#### 3.3.3 回报中的随机性
假设一回合游戏一共有$n$步，完成这一回合之后，我们观测到的所有的奖励是$r_1,r_2,\cdots, r_n$，此时这些奖励不是随机变量，而是观测到的值，我们可以实际计算出折扣回报：
$$u_t = r_t + \gamma r_{t+1}+\gamma^2r_{t+2}+\gamma^3r_{t+3}+\cdots+\gamma^{n-t}r_n,
\quad \forall t=1,\cdots ,n$$
此时的折扣回报是实际观测到的数值，不具有随机性

假设我们此时在第$t$时刻，则我们只观测到$s_t$及其之前的动作、状态、奖励
$$s_1,a_1,r_1,\ s_2,a_2,r_2,\ s_3,a_3,r_3,\cdots,s_{t-1},a_{t-1},r_{t-1},\ s_t$$
而以下尚未被观测到的值都是随机变量：
$$A_t,R_t,\ S_{t+1},A_{t+1},R_{t+1},\ S_{t+2},A_{t+2},R_{t+2},\ \cdots,\ S_n,A_n,R_n$$

而在$t$时刻，回报$U_t$依赖于奖励$R_t,R_{t+1},\cdots,R_n$，这些奖励都是未知的随机变量，因此$U_t$也是未知的随机变量

考虑$U_t$随机性的来源
奖励$R_t$依赖于状态$s_t$(已观测到)和动作$A_t$(未知变量)，奖励$R_{t+1}$依赖于状态$S_{t+1}$和动作$A_{t+1}$(未知变量)，奖励$R_{t+2}$依赖于状态$S_{t+2}$和动作$A_{t+2}$(未知变量)，以此类推，故$U_t$的随机性来自于这些状态和动作：
$$A_t,\ S_{t+1},A_{t+1},\ S_{t+2},A_{t+2},\ \cdots,\ S_n,A_n$$
而前面已经讨论过，动作的随机性来自于策略函数，状态的随机性来自于状态转移函数
### 3.4 价值函数
#### 3.4.1 动作价值函数
我们知道回报$U_t$是$t$时刻后所有奖励的加权和，$U_t$的值可以直接告诉我们局势的好坏，但在$t$时刻，$U_t$尚且是一个随机变量，因此我们想要预判$U_t$的值，就用求期望的方法消除随机性

假设我们已经观测到状态$s_t$，并且做完决策，选中动作$a_t$，则$U_{t}$中的随机性来自于$t+1$时刻之后的状态和动作：
$$S_{t+1},A_{t+1},\ S_{t+2},A_{t+2},\ \cdots, S_{n},A_{n} $$
对$U_t$关于变量$S_{t+1},A_{t+1},\ \cdots, S_{n},A_{n}$求条件期望，得到：
$$Q_{\pi}(s_t,a_t)=\mathbb E_{S_{t+1},A_{t+1},\ \cdots, S_{n},A_{n} }[U_t|S_t=s_t,A_t=a_t]$$
期望中的$S_t = s_t, A_t = a_t$是条件，意思是已经观测到$S_t,A_t$的值
条件期望的结果$Q_{\pi}(s_t,a_t)$被称为**动作价值函数(Actoin-Value Function)**

动作价值函数$Q_{\pi}(s_t,a_t)$依赖于$s_t,a_t$，而不依赖于$t+1$时刻及其之后的状态和动作，这是因为随机变量$S_{t+1},A_{t+1},\cdots,S_{n},A_n$都被期望消除了

$Q_{\pi}(s_t,a_t)$实际上依赖于策略函数$\pi(a|s)$：
$$\begin{align}
&Q_{\pi}(s_t,a_t)=\mathbb E_{S_{t+1},A_{t+1},\ \cdots, S_{n},A_{n} }[U_t|S_t=s_t,A_t=a_t]\\
&=\int_{\mathcal S}ds_{t+1}\int_{\mathcal A}da_{t+1}\cdots\int_{\mathcal S}ds_n\int_{\mathcal A}da_n\left[\prod_{k=t+1}^np(s_k|s_{k-1},a_{k-1})\cdot\pi(a_k|s_k)\right]\cdot U_t
\end{align}$$
公式中的$\pi$是动作的概率密度函数，用不同的$\pi$，结果就会不同

$t$时刻的动作价值函数$Q_{\pi}(s_t,a_t)$依赖于以下三个因素：
- 当前状态$s_t$，当前状态越好，那么价值$Q_{\pi}(s_t,a_t)$越大，也就是说回报的期望值越大
- 当前动作$a_t$，智能体执行的动作越好，那么价值$Q_{\pi}(s_t,a_t)$越大
- 策略函数$\pi$，策略决定未来的动作$A_{t+1},A_{t+2},\cdots,A_n$的好坏，策略越好，那么$Q_{\pi}(s_t,a_t)$就越大
#### 3.4.2 最优动作价值函数
如果想要排除掉策略$\pi$的影响，只评价当前状态和动作的好坏，就涉及到**最优动作价值函数(Optimal Action-Value Function)**：
$$Q_{*}(s_t,a_t) = \max_{\pi}Q_{\pi}(s_t,a_t)\quad \forall s_t\in\mathcal S,a_t\in \mathcal A$$
公式的意思是有很多种策略函数$\pi$可以选择，而我们选择最好的策略函数：
$$\pi^{*}=\arg\max_{\pi} Q_{\pi}(s_t,a_t)\quad \forall s_t\in \mathcal S, a_t\in \mathcal A$$
$Q_{*}$和$Q_{\pi^*}$指的都是最优动作价值函数，最优动作价值函数$Q_{*}(s_t,a_t)$只依赖于$s_t,a_t$，与策略$\pi$无关

已知最优动作价值函数，给定当前状态$s_t$，我们就可以遍历动作空间$\mathcal A$，计算每一对$Q_{*}(s_t,a_t)$，也就是计算了在状态$s_t$和动作$a_t$的条件下，动作价值函数的严格上界(不论如何选择策略$\pi$，回报$U_t$的期望都不会超过$Q_{*}(s_t,a_t)$)，因此最优动作价值函数可以直接指导智能体要执行什么动作
#### 3.4.3 状态价值函数
已知策略函数$\pi$和当前的状态$s_t$，想要评估当前状态的价值，则使用**状态价值函数(State-Value Function)**：
$$\begin{align}
V_{\pi}(s_t) &= \mathbb E_{\mathcal A_t\sim\pi(\cdot|s_t)}\left[Q_{\pi}(s_t,\mathcal A_t)\right]\\
&= \sum_{a\in \mathcal A}\pi(a|s_t)\cdot Q_{\pi}(s_t,a)
\end{align}$$
公式将动作$\mathcal A_t$视作随机变量，将动作价值函数$Q_{\pi}(s_t,\mathcal A_t)$对动作$\mathcal A_t$求期望，把$\mathcal A_t$消掉，就得到了仅依赖于策略$\pi$和当前状态$s_t$，，不依赖于动作的状态价值函数

状态价值函数实质上也是回报$U_t$的期望：
$$V_{\pi}(s_t) = \mathbb E_{A_t,S_{t+1},A_{t+1},\cdots,S_n,A_n}\left[U_t|S_t = s_t\right]$$
期望消掉了$U_t$依赖的随机变量$A_t,S_{t+1},A_{t+1},\cdots,S_n,A_n$，状态价值函数$V_{\pi}(s_t)$越大，则当前状态和策略函数下回报$U_t$的期望越大，因此状态价值函数用于衡量当且策略$\pi$和状态$s_t$的好坏
### 3.5 策略学习和价值学习
解决实际问题中，我们首先要自己定义奖励，而目标就定义为最大化(折扣)回报，也就是最大化奖励的(加权)总和，之后考虑采取哪一种RL方法来实现目标

强化学习方法通常分为两类：**基于模型的方法(Model-Based)**、**无模型方法(Model-Free)**
本书主要介绍无模型方法，无模型方法又可以分为**价值学习**和**策略学习**

**价值学习(Value-Based Learning)** 指学习最优(动作)价值函数$Q_*(s,a)$，
智能体可以根据学习到的$Q_*(s,a)$进行决策，选出最好的动作，每次观测到一个状态$s_t$，我们可以遍历动作空间$\mathcal A$，计算每一对$Q_{*}(s_t,a_t)$，也就是计算了在状态$s_t$和动作$a_t$的条件下，动作价值函数的严格上界，这些$Q$值量化了每个动作的好坏，智能体应执行$Q$值最大的动作

智能体的决策可以写为：
$$a_t = \arg\max_{a\in \mathcal A}Q_*(s_t,a)$$

我们需要用智能体收集到的状态、动作、奖励作为训练数据，学习一个表格或神经网络，用于近似$Q_*$，最著名的价值学习方法是深度Q网络 DQN

**策略学习(Policy-Based Learning)** 指学习策略函数$\pi(a | s)$，
给定一个状态，我们可以根据策略函数计算所有动作的概率值，然后随机抽样选择一个动作让智能体执行
学习策略函数的方法有策略梯度等
### 3.6 实验环境
比较和评价RL算法最常用的是OpenAI Gym，它相当于深度学习中的ImageNet
Gym有几大类控制问题，比如经典控制问题、Atari游戏、机器人

Gym中的第一类问题是经典控制问题，都是小规模的简单问题，例如Cart Pole和Pendulum，
Cart Ple要求给小车向左或向右的力，移动小车，让上面的杆子能竖起来
Pendulum要求给钟摆一个力，让钟摆恰好能竖起来

第二类问题是Atari游戏，包括Pong、Space Invader、Breakout等，
Pong中的智能体是乒乓球拍，球拍可以上下运动，目的是借助对手的球并尽量让对手接不到球(打乒乓球)
Space Invader中的智能体是小飞机，可以左右移动，发射炮弹(飞机大战)
Breakout中的智能体是下面的球拍，可以左右移动，目的是接住球，并把上面的砖块都打掉(打砖块)

第三类问题是机器人连续的控制问题，例如在MuJoCo物理模拟器(模拟重力等物理量)中控制蚂蚁、人、猎豹等机器人站立、走路，机器人就是智能体

使用Gym标准库的一个例子
```python
import gym

env = gym.make('CartPole-v0') # 生成环境
state = env.reset() # 重置环境，让小车回到起点，并返回初始状态

for t in range(100):
	env.render() # 弹出窗口，把游戏中发生的显示到屏幕上
	print(state)

	# 采样动作
	action = env.action_space.sample()

	# 执行动作，环境更新状态，反馈一个奖励
	state, reward, done, info = env.step(action) 

	if done:
		print('Finished')
		break

env.close()
```
# Part 2 价值学习
## 4 DQN与Q学习
### 4.1 DQN
**最优动作价值函数的用途**：
最优动作价值函数$Q_*$可以让我们在$t$时刻就预见$t$到$n$时刻之间的累计奖励的期望，如果我们知道$Q_*$，$Q_*$就是我们做出动作的指导(用$Q_*$选出针对当前状态最优的动作)，我们需要进行重复的训练，积累足够多的“经验”，以得到最优动作价值函数

**最优动作价值函数的近似**：
在实践中，近似学习$Q_*$的最有效的办法是深度Q网络(Deep Q Network DQN)，记作$Q(s,a;\symbfit w)$，其中$\symbfit w$表示网络的参数，我们在起始时随机初始化$\symbfit w$，随后用“经验”去学习$\symbfit w$，学习的目标是：对于所有的$s$和$a$，DQN的预测$Q(s,a;\symbfit w)$，尽量接近$Q_*(s,a)$

DQN的输入是状态$s$，输出是离散动作空间$\mathcal A$上每个动作的$Q$值(即一个$|\mathcal A|$维的向量$\hat {\symbfit q}$)，换句话说DQN会给每个动作评分，分数越高动作越好，
我们常用的符号$Q(s,a,;\symbfit w)$是标量，指动作$a$对应的动作价值

**DQN的梯度**：
训练DQN时，对参数$\symbfit w$求梯度：
$$\nabla_{\symbfit w}Q(s,a;\symbfit w) = \frac {\partial Q(s,a;\symbfit w)}{\partial \symbfit w}$$
此即函数值$Q(s,a;\symbfit w)$关于参数$\symbfit w$的梯度，因为函数值是一个实数，故梯度的形状与$\symbfit w$的形状完全相同，因此只要给定观测值$s,a$，就可以求出梯度
### 4.2 时间差分(TD)算法
训练DQN最常用的算法是时间差分算法(Temporal Difference TD)
#### 4.2.1 驾车预测时间的例子
假设我们有一个模型$Q(s,d;\symbfit w)$，其中$s$是起点，$d$是终点，$\symbfit w$是参数，模型$Q$可以预测开车出行的时间开销，模型一开始不准确，甚至是纯随机的，但随着越来越多人使用这个模型，模型得到更多数据和训练，模型就会越来越准

在用户出发前，用户告诉模型起点$s$和终点$d$，模型做一次预测$\hat q = Q(s,d;\symbfit w)$，当用户结束形成，把实际驾车时间$y$反馈给模型，模型利用二者之差$\hat q - y$进行修正，这就是模型的训练过程

模型的修正就是用梯度下降对参数进行一次更新，对于一组训练数据$s,d, y$，我们希望估计值$\hat q = Q(s,d;\symbfit w)$尽量接近真实观测到的$y$，采用二者的平方差作为损失函数：
$$L(\symbfit w) = \frac 1 2 \left [Q(s,d;\symbfit w)-y\right]^2$$
用链式法则计算$\symbfit w$相对于损失函数$L(\symbfit w)$的梯度：
$$\nabla_{\symbfit w}L(\symbfit w) = (\hat q - y)\cdot \nabla_{\symbfit w} Q(s,d;\symbfit w)$$
然后做一次梯度下降更新模型参数$\symbfit w$：
$$\symbfit w \leftarrow \symbfit w - \alpha\cdot \nabla_{\symbfit w}L(\symbfit w)$$
#### 4.2.2 TD算法
接着上文驾车时间的例子，假设我们的出发点是北京，目标是上海，出发前模型估计全程时间是$\hat q =Q(\text{“北京”},\text{“上海”};\symbfit w) = 14$小时，模型建议的路径会途径济南，
从北京出发，过了$r = 4.5$个小时，我们到达了济南，此时让模型再做一次预测，模型算出$\hat q' \triangleq Q(\text{“济南”},\text{“上海”};\symbfit w) = 11$小时

如果我们不得不在济南取消此次行程，没有完成旅途，则这组数据($\hat q,\hat q', r)$可以利用时间差分算法帮助训练模型

根据模型的最新估计，到达济南时，整个旅程的总时间为：
$$\hat y \triangleq r + \hat q' = 4.5 + 11 = 15.5$$
TD算法把$\hat y = 15.5$称为TD目标(TD Target)，它比最初预测的$\hat q = 14$更可靠，因为TD目标中含有事实的成分，即其中的$r = 4.5$时实际的观测
因为$\hat y$比$\hat q$更可靠，因此可以使用$\hat y$对模型进行“修正”，即我们希望估计值$\hat q$尽量接近TD目标$\hat y$，故用二者的平方差作为损失函数：
$$L(\symbfit w) = \frac 1 2\left[Q(\text{“北京”},\text{“上海”};\symbfit w) -\hat y\right]^2$$
我们知道$\hat y = r + \hat q'$，而$\hat q'$依赖于$\symbfit w$，因此$\hat y$实际也依赖于$\symbfit w$，换句话或，就是也是$\symbfit w$的函数，而TD算法会忽视这一点，在实际求关于$\symbfit w$的梯度时，会将$\hat y$视为常数，因此此处计算$L(\symbfit w)$关于$\symbfit w$的梯度得到：
$$\begin{align}
\nabla_{\symbfit w}L(\symbfit w) &=\frac 1 2\nabla_{\symbfit w}(\hat q - \hat y)^2\\
&= (\hat q - \hat y)\cdot\nabla_{\symbfit w} \hat q\\
&=\delta\cdot\nabla_{\symbfit w} Q(\text{“北京”},\text{“上海”};\symbfit w)
\end{align}$$
此处$\delta = \hat q - \hat y = 14-15.5$称为TD误差(TD Error)
我们做一次梯度下降更新模型参数$\symbfit w$：
$$\symbfit w \leftarrow \symbfit w - \alpha\cdot\delta\cdot \nabla_{\symbfit w} Q(\text{“北京”},\text{“上海”};\symbfit w)$$
TD算法根据该公式更新参数

可以换个角度思考TD算法，模型估计从北京到上海需要$\hat q$小时，从济南到上海需要$\hat q'$小时，这相当于模型做了这样的估计：从北京到济南需要$\hat q - \hat q'$小时
而我们从北京到济南真实花费的时间是$r$小时，因此模型的估计与真实观测之差就是$\delta = (\hat q - \hat q') - r = \hat q-(\hat q' + r) = \hat q - \hat y$，也就是TD误差

因此TD误差就是模型估计与真实观测之差，TD算法的目的就是通过梯度下降更新参数$\symbfit w$使得目标函数$L(\symbfit w) = \frac 1 2 \delta^2$减小
### 4.3 用TD训练DQN
本节推导的是最原始的TD算法，在实践中效果不佳，实际训练DQN时，应使用第六章介绍的高级技巧
### 4.3.1 算法推导
下面推导DQN的TD算法，严格地说，是“Q学习算法”，它属于TD算法的一种

回忆一下回报的定义：$U_t = \sum_{k=t}^n \gamma^{k-t}\cdot R_k$，$U_{t+1}=\sum_{k=t+1}^n \gamma^{k-t-1}\cdot R_k$，
由此可得：
$$U_t = R_t + \gamma\cdot\sum_{k=t+1}^n \gamma^{k-t-1}\cdot R_k=R_t + \gamma\cdot U_{t+1}$$
回忆一下，最优动作价值函数可以写为：
$$Q_*(s_t,a_t) = \max_{\pi}\mathbb E[U_t|S_t = s_t, A_t = a_t]$$
# 附录A 贝尔曼方程
**定理A.1：贝尔曼方程(将$Q_{\pi}$表示成$Q_{\pi}$)**
假设$R_t$是$S_t,A_t,S_{t+1}$的函数，那么
$$Q_{\pi}(s_t,a_t) = \mathbb E_{S_{t+1},A_{t+1}}\left[R_t + \gamma\cdot Q_{\pi}(S_{t+1},A_{t+1}) | S_t = s_t,A_t = a_t\right]$$
证明：
根据回报的定义$U_t = \sum_{k=t}^n\gamma^{k-t}\cdot R_k$，可以知道：
$$U_t = R_t + \gamma \cdot U_{t+1}$$
用符号$\mathcal S_{t+1}:= \{S_{t+1},S_{t+2},\cdots\},\mathcal A_{t+1}:= \{A_{t+1},A_{t+2},\cdots\}$表示从$t+1$时刻起所有的状态和动作随机变量，回忆动作价值函数$Q_{\pi}$的定义：
$$Q_{\pi}(s_t,a_t) = \mathbb E_{\mathcal S_{t+1},\mathcal A_{t+1}}\left[U_t|S_t = s_t,A_t = a_t\right]$$
将其中的$U_t$替换称$R_t + \gamma \cdot U_{t+1}$，则：
$$\begin{align}
Q_{\pi}(s_t,a_t) &= \mathbb E_{\mathcal S_{t+1},\mathcal A_{t+1}}\left[R_t + \gamma\cdot U_{t+1}|S_t = s_t,A_t = a_t\right]\\
&=\mathbb E_{\mathcal S_{t+1},\mathcal A_{t+1}}[R_t |S_t = s_t,A_t = a_t]+\\
&\quad\gamma \cdot \mathbb E_{\mathcal S_{t+1},\mathcal A_{t+1}}[U_{t+1}|S_t = s_t,A_t = a_t]
\end{align}\tag{A.1}$$
假设$R_t$是$S_t,A_t,S_{t+1}$的函数，那么，给定$s_t$和$a_t$，$R_t$随机性的唯一来源就是$S_{t+1}$，因此：
$$\mathbb E_{\mathcal S_{t+1},\mathcal A_{t+1}}[R_t | S_t = s_t,A_t = a_t] = \mathbb E_{S_{t+1}}[R_t| S_t = s_t,A_t = a_t]\tag{A.2}$$
等式A.1右边$U_{t+1}$的期望可以写为：
$$\begin{align}
&\mathbb E_{\mathcal S_{t+1},\mathcal A_{t+1}}\left[U_{t+1}|S_t = s_t,A_t = a_t\right]\\
=&\mathbb E_{S_{t+1}, A_{t+1}}\left[\mathbb E_{\mathcal S_{t+2},\mathcal A_{t+2}}\left[U_{t+1}|S_{t+1},A_{t+1}\right]\ |\ S_t = s_t,A_t = a_t\right]\\
=&\mathbb E_{S_{t+1},A_{t+1}}[Q_{\pi}(S_{t+1},A_{t+1})|S_t = s_t,A_t= a_t]
\end{align}\tag{A.3}$$
结合公式A.1，A.2，A.3，可得：
$$\begin{align}
Q_{\pi}(s_t,a_t) =& \mathbb E_{S_{t+1}}[R_t|S_t = s_t,A_t = a_t]\\+&
\gamma\cdot\mathbb E_{S_{t+1},A_{t+1}}[Q_{\pi}(S_{t+1},A_{t+1})|S_t = s_t,A_t = a_t]\\
=&\mathbb E_{S_{t+1},A_{t+1}}\left[R_t + \gamma\cdot Q_{\pi}(S_{t+1},A_{t+1}) | S_t = s_t,A_t = a_t\right]
\end{align}$$
证毕

**定理A.2：贝尔曼方程(将$Q_{\pi}$表示成$V_{\pi}$)**
假设$R_t$是$S_t,A_t,S_{t+1}$的函数，那么
$$Q_{\pi}(s_t,a_t) = \mathbb E_{S_{t+1}}\left[R_t + \gamma\cdot V_{\pi}(S_{t+1})|S_t = s_t, A_t = a_t\right]$$
证明：
由于$V_{\pi}(S_{t+1}) = \mathbb E_{A_{t+1}}[Q(S_{t+1},A_{t+1})]$，由定理A.1得到A.2
证毕

**定理A.3：贝尔曼方程(将$V_{\pi}$表示成$V_{\pi}$)**
假设$R_t$是$S_t,A_t,S_{t+1}$的函数，那么
$$V_{\pi}(s_t) = \mathbb E_{A_t,S_{t+1}}\left[R_t + \gamma \cdot V_{\pi}(S_{t+1})|S_t = s_t\right]$$
证明：
由于$V_{\pi}(S_t) = \mathbb E_{A_t}\left[ Q(S_t,A_t)\right]$，由定理A.2可得定理A.3
证毕

**定理A.4：最优贝尔曼方程**
假设$R_t$是$S_t,A_t,S_{t+1}$的函数，那么
$$Q_*(s_t,a_t) = \mathbb E_{S_{t+1}\sim p(\cdot|s_t,a_t)}\left[R_t + \gamma \cdot \max_{A\in \mathcal A}Q_*(S_{t+1},A) | S_t = s_t,A_t = a_t\right]$$
证明：
设最优策略函数$\pi^* = \arg\max_{\pi}Q_{\pi}(s,a),\forall s\in \mathcal S, a\in \mathcal A$，由贝尔曼方程可得：
$$Q_{\pi^*}(s_t,a_t) = \mathbb E_{S_{t+1},A_{t+1}}\left[R_t + \gamma\cdot Q_{\pi^*}(S_{t+1},A_{t+1})|S_t = s_t,A_t = a_t\right]$$
根据定义，最优动作价值函数是：
$$Q_*(s,a):=\max_{\pi} Q_{\pi}(s,a),\quad \forall s\in \mathcal S, a\in \mathcal A$$
也就是说$Q_{\pi^*}(s,a)$就是$Q_*(s,a)$，因此：
$$Q_{*}(s_t,a_t) = \mathbb E_{S_{t+1},A_{t+1}}\left[R_t + \gamma\cdot Q_{*}(S_{t+1},A_{t+1})|S_t = s_t,A_t = a_t\right]$$
因为动作$A_{t+1} = \arg\max_A Q_*(S_{t+1},A)$是状态$S_{t+1}$的确定性函数，所以：
$$Q_*(s_t,a_t) = \mathbb E_{S_{t+1}}\left[R_t + \gamma \cdot \max_{A\in \mathcal A} Q_*(S_{t+1},A)| S_t = s_t,A_t = a_t\right]$$




