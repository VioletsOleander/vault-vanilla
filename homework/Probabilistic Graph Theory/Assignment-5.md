# Problem 1
(1) According to the HC theorem, we can directly write

$$
\begin{align}
P &= \frac 1 Z \exp(-U),\\
U &= -\left(\sum_{i=1}^2 \alpha_iH_i + \sum_{j=1}^3\beta_jV_j + \sum_{i=1}^2\sum_{j=1}^3 w_{H_iV_j}H_iV_j\right),\\
Z&=\sum_{H_1,H_2,V_1,V_2,V_3}\exp(-U).
\end{align}
$$

Substitute the values of the coefficients in, we can get

$$
\begin{align}
P&= \frac 1 Z \exp\{0.1H_1 + 0.1H_2 + 0.5V_1 -0.3V_2 -0.5V_3\\
&\quad\quad\quad\quad-1(H_1V_1 + H_1V_2 + H_1V_3) + 0.2(H_2V_1+H_2V_2+H_2V_3)\}\\
&= \frac 1 Z \exp\{0.1(H_1 + H_2) + 0.5(V_1 - V_3) -0.3V_2\\
&\quad\quad\quad\quad-H_1(V_1 + V_2 + V_3) + 0.2H_2(V_1+V_2+V_3)\}\\
&= \frac 1 Z \exp\{0.1(H_1 + H_2) + 0.5(V_1 - V_3) -0.3V_2 +(-H_1 + 0.2H_2)(V_1 + V_2 + V_3)\},\\
\end{align}
$$

where

$$
\begin{align}
Z=\sum_{H_1, H_2, V_1, V_2, V_3}\exp\{0.1(H_1 + H_2) + 0.5(V_1 - V_3) -0.3V_2 \\+(-H_1 + 0.2H_2)(V_1 + V_2 + V_3)\}.
\end{align}
$$

(2) Construct the Bethe graph for this Markov network as follows.

![[homework/pics/probabilistic graph theory/pgm-5.1.1.png]]

The pseudo code is designed as follows:

**Require**: Cluster graph $\mathcal U$, factor set $\Phi$.
**Procedure** initialize\_cluster\_graph( $\mathcal U$ : cluster graph, $\Phi$ : factor set)
  1: **for** each cluster $\pmb C_i$ in $\mathcal U$:
  2: $\beta_i := 1$
  3:   **for** each factor $\phi$ in $\Phi$:
  4:     **if** $Scope[\phi] \subseteq Scope[\pmb C_i]$:
  5:      $\beta_i := \beta_i \cdot \phi$
  6:     **end if**
  7:   **end for**
  8: **end for**
  9: **for** each edge $(i-j)$ in $\mathcal U$:
10:   $\mu_{ij} = 1$
11: **end for**

**Procedure** belief\_propagate( $i$ : sending clique, $j$ : receiving clique )
 1: $\sigma_{i\rightarrow j} := \sum_{\pmb C_i - \pmb S_{ij}}\beta_i$
 2: $\beta_j := \beta_j\cdot \frac {\sigma_{i\rightarrow j}}{\mu_{i j}}$
 3: $\mu_{ij} = \sigma_{i\rightarrow j}$

**Procedure** calibrate\_cluster\_graph( $\mathcal U$ : cluster graph, $\Phi$ : factor set)
 1: initialize\_cluster\_graph( $\mathcal U$, $\Phi$ )
 2: **While** graph is not calibrated
 3:   Randomly select an edge $(i-j)$ in $\mathcal U$
 4:   belief\_propagate( $i$, $j$ )
 5: **Return** $\{\beta_i\}$

(3) The KL divergence between the original distribution $P$ and the proposal distribution $Q$ is

$$
\begin{align}
D_{KL}(Q||P) &= E_{\pmb X\sim Q}\left[\ln\frac {Q(\pmb X)}{ P(\pmb X)}\right]\\
&=E_{\pmb X\sim Q}[\ln Q(\pmb X)] - E_{\pmb X\sim Q}[\ln P(\pmb X)]\\
&=-E_{\pmb X\sim Q}\left[\ln\frac {1}{Q(\pmb X)}\right] - E_{\pmb X\sim Q}[\ln P(\pmb X)]\\
&=-E_{\pmb X\sim Q}[\ln P(\pmb X)] - H(Q).
\end{align}
$$

Minimizing $D_{KL}(Q||P)$ is equivalent to maximizing 

$$
\begin{align}
L(Q) &= -D_{KL}(Q||P)\\
&=E_{\pmb X\sim Q}[\ln P(\pmb X)] + H(Q).
\end{align}$$

The mean field variational method assumes a fully factorized representation of proposal distribution $Q$ as

$$
Q(\pmb X) = \prod _iQ_i(X_i).
$$

Therefore, $L(Q)$ can be written as

$$
\begin{align}
L(Q) &=  E_{\pmb X \sim Q}[\ln P(\pmb X)] + H(Q)\\
&=\sum_{\pmb x}Q(\pmb x)\ln P(\pmb x) + \sum_{\pmb x}Q(\pmb x)(-\ln Q(\pmb x))\\
&=
\sum_{\pmb x}\left[\prod_i Q_i(x_i)\right]\ln P(\pmb x) + \sum_{\pmb x}\left[\prod_i Q_i(x_i)\right]\left[-\ln \prod_i Q_i(x_i)\right]\\
&=\sum_{\pmb x}\left[\prod_i Q_i(x_i)\right]\ln P(\pmb x) + \sum_{\pmb x}\left[\prod_i Q_i(x_i)\right]\left[-\sum_i \ln Q_i(x_i)\right]\\
&=\sum_{\pmb x}\left[\prod_i Q_i(x_i)\right]\left[\ln P(\pmb x) -\sum_i \ln Q_i(x_i)\right].\\
\end{align}
$$

Considering using iterative optimization to maximize $L(Q)$, we assume all other variable $X_j$ 's marginal $Q_j$ fixed except $X_i$ 's, and then write $L(Q)$ in terms of $Q_i$ .

$$
\begin{align}
L(Q_i)&=\sum_{\pmb x}\left[\prod_k Q_k(x_k)\right]\left[\ln P(\pmb x) -\sum_i \ln Q_k(x_k)\right]\\
&=\sum_{x_i}\sum_{\pmb x_{-i}} Q_i(x_i)\left[\prod_{j\ne i} Q_j(x_j)\right]\left[\ln P(\pmb x) -\sum_k \ln Q_k(x_k)\right]\\
&=\sum_{x_i}Q_i(x_i)\sum_{\pmb x_{-i}}\left[\prod_{j\ne i} Q_j(x_j)\right]\left[\ln P(\pmb x) -\sum_k \ln Q_k(x_k)\right]\\
&=\sum_{x_i}Q_i(x_i)\sum_{\pmb x_{-i}}\left[\prod_{j\ne i} Q_j(x_j)\ln P(\pmb x) -\prod_{j\ne i}Q_j(x_j)\sum_k \ln Q_k(x_k)\right]\\
&=\sum_{x_i}Q_i(x_i)\sum_{\pmb x_{-i}}\prod_{j\ne i} Q_j(x_j)\ln P(\pmb x) -\sum_{x_i}Q_i(x_i)\sum_{\pmb x_{-i}}\prod_{j\ne i}Q_j(x_j)\sum_k \ln Q_k(x_k)\\
&=\sum_{x_i}Q_i(x_i)\sum_{\pmb x_{-i}}\prod_{j\ne i} Q_j(x_j)\ln P(\pmb x) -\sum_{x_i}Q_i(x_i)\sum_{\pmb x_{-i}}\prod_{j\ne i}Q_j(x_j)\left[\ln Q_i(x_i) + \sum_{k\ne i} \ln Q_k(x_k)\right]\\
&=\sum_{x_i}Q_i(x_i)\sum_{\pmb x_{-i}}\prod_{j\ne i} Q_j(x_j)\ln P(\pmb x) -\sum_{x_i}Q_i(x_i)\sum_{\pmb x_{-i}}\prod_{j\ne i}Q_j(x_j)\ln Q_i(x_i) +\prod_{j\ne i}Q_j(x_j) \sum_{k\ne i} \ln Q_k(x_k)\\
&=\sum_{x_i}Q_i(x_i)\sum_{\pmb x_{-i}}\prod_{j\ne i}Q_j(x_j)\ln P(\pmb x)  - \sum_{x_i}Q_i(x_i) \ln Q_i(x_i) + const\\
&=\sum_{x_i}Q_i(x_i)\ln \frac {\exp \left\{\sum_{\pmb x_{-i}}\prod_{j\ne i}Q_j(x_j)\ln P(\pmb x)\right\}}{Q_i(x_i)} + const\\
&=\sum_{x_i}Q_i(x_i)\ln \frac {\exp\left\{E_{\pmb X_{-i}\sim Q_{-i}}[\ln P(\pmb X)]\right\}}{Q_i(x_i)} + const.
\end{align}
$$

By observation, we can get

$$
\sum_{x_i}Q_i(x_i)\ln \frac {\exp\{E_{\pmb X_{-i}\sim Q_{-i}}[\ln P(\pmb X)]\}}{Q_i(x_i)} = -D_{KL}(Q_i(X_i)|| \exp \{E_{\pmb X_{-i}\sim Q_{-i}}[\ln P(\pmb X)]\}).
$$

Therefore, to maximize $L(Q_i)$ in terms of $Q_i$ , we expect to minimize $D_{KL}(Q_i(X_i)||\exp \{E_{\pmb X_{-i}}[\ln P (\pmb X)]\})$ , so $Q_i(X_i)$ is expected to satisfy

$$
Q_i(X_i) \propto \exp\{E_{\pmb X_{-i}\sim Q_{-i}}[\ln P(\pmb X)]\}.
$$

Because each variable is 0-1 binary, we further assumes each variable conforms to a binomial distribution. That is to say

$$
\begin{align}
Q(\pmb X;\pmb \theta) &=\prod_i Q_i(X_i;\theta_i),\\
&\text{where}\quad Q_i(X_i;\theta_i) =\begin{cases}
\theta_i &\text{if}\ X_i = 1,\\
1-\theta_i &\text{if}\ X_i = 0.
\end{cases}
\end{align}
$$

We further consider more derivation on $E_{\pmb X_{-i}\sim Q_{-i}}[\ln P(\pmb X)]$:

$$
\begin{align}
E_{\pmb X_{-i}\sim Q_{-i}}[\ln P(\pmb X)] &=E_{\pmb X_{-i}\sim Q_{-i}}\left[\ln\frac 1 Z\exp\{\sum_{j} w_jX_j + \sum_{k,l}w_{kl}X_kX_l\}\right]\\
&=E_{\pmb X_{-i}\sim Q_{-i}}\left[\sum_{j} w_jX_j + \sum_{k,l}w_{kl}X_kX_l-\ln Z\right]\\
&=E_{\pmb X_{-i}\sim Q_{-i}}\left[\sum_{j\ne i} w_jX_j + w_iX_i + \sum_{k,l}w_{kl}X_kX_l-\ln Z\right]\\
&=w_iX_i + E_{\pmb X_{-i}\sim Q_{-i}}\left[\sum_{j\ne i} w_jX_j  + \sum_{k,l}w_{kl}X_kX_l-\ln Z\right]\\
&=w_iX_i + E_{\pmb X_{-i}\sim Q_{-i}}\left[\sum_{l\in neighbor(i)}w_{il}X_iX_l\right] +const\\
&=w_iX_i + E_{\pmb X_{-i}\sim Q_{-i}}\left[\sum_{l\in neighbor(i)}w_{il}X_iX_l\right] +const\\
&=w_iX_i + X_i\sum_{l\in neighbor(i)}w_{il} E_{X_l\sim Q_l}[X_l] + const\\
&=w_iX_i + X_i\sum_{l\in neighbor (i)}w_{il}\theta_l + const.
\end{align}
$$

Thus we expect

$$
Q_i(X_i)\propto \exp\{w_iX_i + X_i\sum_{l\in neighbor(i)}w_{il}\theta_l + const\}.
$$

Assume $Q_i(X_i) = t\cdot \exp\{w_iX_i + X_i\sum_{l\in neighbor(i)}w_{il}\theta_l + const\},(t\ne 0)$, then

$$
\begin{cases}
Q_i(X_i = 1) =  t\cdot \exp\{w_i + \sum_{l\in neighbor(i)}w_{il}\theta_l + const\} = \theta_i,\\
Q_i(X_i = 0) = t\cdot\exp\{const\} = 1-\theta_i.
\end{cases}
$$

Therefore

$$
\begin{align}
\theta_i &= \frac {\theta_i}{1}\\
&=\frac {\theta_i}{1 -\theta_i + \theta_i}\\
&=\frac {t\cdot \exp\{w_i + \sum_{l\in neighbor(i)}w_{il}\theta_l + const\}}{t\cdot \exp\{w_i + \sum_{l\in neighbor(i)}w_{il}\theta_l + const\} + t\cdot \exp\{const\}}\\
&=\frac { \exp\{w_i + \sum_{l\in neighbor(i)}w_{il}\theta_l + const\}}{\exp\{w_i + \sum_{l\in neighbor(i)}w_{il}\theta_l + const\} + \exp\{ const\}}\\
&=\frac { \exp\{w_i + \sum_{l\in neighbor(i)}w_{il}\theta_l \}}{\exp\{w_i + \sum_{l\in neighbor(i)}w_{il}\theta_l \} + 1}.\\
\end{align}
$$

We adopt iterative optimization process, using the above equality to update $\theta_i$ in the $i$ -th iteration of each epoch.

There are five nodes in the graph, so there are five $\theta$ s in $Q$. We denote them as $\theta_{V_1}, \theta_{V_2}, \theta_{V_3}, \theta_{H_1}, \theta_{H_2}$ separately.

The $\theta$ s are all initialized to $0.5$. According to the requirements, we do optimization for $20$ epochs, which corresponds to $20\cdot 5 = 100$ iterations.

The belief updating process for each node in the first iteration is plotted as follows:

![[homework/pics/probabilistic graph theory/pgm-5.1.2.png]]

After 100 iterations, each node's marginal distribution is

$$
\begin{align}
\begin{cases}
P(V_1 = 1) = 0.597\\
P(V_1 = 0) = 0.403,
\end{cases}
\\\\
\begin{cases}
P(V_2 = 1) = 0.400\\
P(V_2 = 0) = 0.600,
\end{cases}
\\\\
\begin{cases}
P(V_3 = 1) = 0.353\\
P(V_1 = 0) = 0.647,
\end{cases}
\\\\
\begin{cases}
P(H_1 = 1) = 0.222\\
P(H_1 = 0) = 0.778,
\end{cases}
\\\\
\begin{cases}
P(H_2 = 1) = 0.591\\
P(H_2 = 0) = 0.409.
\end{cases}
\end{align}
$$








