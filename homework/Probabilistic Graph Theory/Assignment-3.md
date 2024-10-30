Student: 陈信杰
Student ID: 2024316427
# Problem 1
(1)

![[Prob Graph-3.1.1.png]]

$y_{ij} \sim \mathcal N (\mu_j, \frac 1 {\tau_j}),\quad i = 1,2,\dots,n_j;j=1,2,\dots,m$
$\mu_j \sim \Gamma (\alpha_\mu, \beta_\mu),\quad j=1,2,\dots,m$
$\tau_j \sim \Gamma (a, b),\quad j =1,2,\dots,m$
$\alpha_\mu \sim Possion (\lambda)$
$\beta_\mu \sim \Gamma (c, d)$
(2)

![[Prob Graph-3.1.2.png]]

$y_{ij}\sim \mathcal N (g (\pmb x_{ij}^T\beta_j),\frac 1 {\tau_j}),\quad i=1,\dots, n_j, j=1,\dots, m$
$\beta_j \sim \mathcal N (\pmb \mu, \pmb \Sigma),\quad j=1,\dots, m$
$\pmb \mu \sim\mathcal N(\pmb \xi, \pmb \Psi)$
$\tau_j \sim \Gamma (a, b),\quad j =1,\dots, m$

# Problem 2
(1) Define $\mathcal X_t = \{X_t, S_{t}^{(1)}, S_{t}^{(2)}, S_{t}^{(3)}, Y_t \}$，and $P (\mathcal X_0) = \pi$, then, we can derive:

$$
\begin{align}
P(\mathcal X_0,\cdots,\mathcal X_t,\cdots,\mathcal X_n) &=P(\mathcal X_0)\prod_{t=1}^n P(\mathcal X_{t}\mid \mathcal X_{t-1},\cdots,\mathcal X_0)\\
&=\pi\prod_{t=1}^nP(\mathcal X_t\mid \mathcal X_{t-1}).
\end{align}
$$

According to the figure:

$$
\begin{align}
P(\mathcal X_t\mid \mathcal X_{t-1})&=P(X_t,S_{t}^{(1)},S_t^{(2)},S_t^{(3)},Y_t\mid X_{t-1},S_{t-1}^{(1)},S_{t-1}^{(2)},S_{t-1}^{(3)},Y_{t-1})\\
&=P(X_t,S_{t}^{(1)},S_t^{(2)},S_t^{(3)},Y_t\mid S_{t-1}^{(1)},S_{t-1}^{(2)},S_{t-1}^{(3)})\\
&=P(X_t)P(S_t^{(1)}\mid S_{t-1}^{(1)},X_t)P(S_{t}^{(2)}\mid S_{t-1}^{(2)},S_t^{(1)},X_t)\\
&P(S_t^{(3)}\mid S_{t-1}^{(3)},S_t^{(2)},S_t^{(1)},X_t)P(Y_t\mid S_t^{(3)},S_t^{(2)},S_t^{(1)},X_t).\\
\end{align}
$$

Therefore, the joint distribution $P$ can be written as:

$$
\begin{align}
P &=\pi \prod_{t=1}^n P(X_t)P(S_t^{(1)}\mid S_{t-1}^{(1)},X_t)P(S_{t}^{(2)}\mid S_{t-1}^{(2)},S_t^{(1)},X_t)\\
&P(S_t^{(3)}\mid S_{t-1}^{(3)},S_t^{(2)},S_t^{(1)},X_t)P(Y_t\mid S_t^{(3)},S_t^{(2)},S_t^{(1)},X_t).
\end{align}
$$

(2) The Markov blanket of $X_t$ is $\{S_{t}^{(1)},S_t^{(2)},S_t^{(3)},S_{t-1}^{(1)},S_{t-1}^{(2)},S_{t-1}^{(3)},Y_t\}$. The Markov blanket of $S_t^{(2)}$ is $\{X_t,X_{t+1},S_{t}^{(1)},S_t^{(3)},S_{t+1}^{(1)},S_{t+1}^2,S_{t-1}^{(2)},S_{t-1}^{(3)},Y_t\}$



