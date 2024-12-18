# Problem 1
(1) According to the HC theorem, we know

$$
\begin{align}
P &= \frac 1 Z\exp\{-U \}\\
U&= -\left(\sum_{i=1}^2 H_i \psi_{H_i}(H_i) + \sum_{j=1}^3V_j \psi_{V_j}(V_j) +\sum_{i=1}^2\sum_{j=1}^3 \psi_{H_iV_j}(H_iV_j)\right)\\
Z& = \sum_{H_1, H_2, V_1, V_2, V_3}\exp(-U)
\end{align}
$$

According to the problem, the $\psi$ s are defined as a constant. Therefore, we have

$$
\begin{align}
\psi_{H_1}(H_1) &= \alpha_1 = a\\
\psi_{H_2}(H_2) &= \alpha_2 = a\\
\psi_{V_1}(V_1)&=\beta_1 = -a\\
\psi_{V_2}(V_2)&=\beta_2 = -a\\
\psi_{V_3}(V_3)&=\beta_3 = -a,\\
\end{align}
$$

and

$$
\begin{align}
\psi_{H_1V_1}(H_1,V_1) = \psi_{H_1V_2}(H_1,V_2) = \psi_{H_1V_3}(H_1V_3)=-w\\
\psi_{H_2V_1}(H_2,V_1) = \psi_{H_2V_2}(H_2,V_2) = \psi_{H_2V_3}(H_2V_3)=w.\\
\end{align}
$$

Therefore

$$
\begin{align}
P&= \frac 1 Z \exp\{aH_1 + aH_2 - aV_1 - aV_2 - aV_3\\
&\quad\quad\quad\quad -wH_1V_1 - wH_1V_2 -wH_1V_3 + wH_2V_1 + wH_2V_2 + w H_2V_3\}\\
&= \frac 1 Z \exp\{a(H_1 + H_2 - V_1 - V_2 - V_3)\\
&\quad\quad\quad\quad-wH_1(V_1 + V_2 + V_3) + wH_2(V_1+V_2+V_3)\}\\
&= \frac 1 Z \exp\{a(H_1 + H_2 - V_1 - V_ 2-V_3) + w(H_2 - H_1)(V_1 + V_2 + V_3)\},\\
\end{align}
$$

where

$$
\begin{align}
Z=\sum_{H_1, H_2, V_1, V_2, V_3}\exp\{a(H_1 + H_2 - V_1 - V_ 2-V_3) + w(H_2 - H_1)(V_1 + V_2 + V_3)\}.
\end{align}
$$

(2) We adopts the EM paradigm. Firstly, we derive the posterior of the hidden variables given the current parameter and observations:

Denoting the current parameter as $a^t$ and $w^t$, and the observation as $v_1, v_2, v_3$ , we have

$$
\begin{align}
P(H_1\mid v_1,v_2,v_3) &=\frac {P(H_1, v_1, v_2, v_3)}{\sum_{H_1}P(H_1, v_1, v_2,v_3)}\\
&=\frac {P(H_1, v_1, v_2, v_3)}{P(H_1 = 1, v_1, v_2, v_3)+P(H_1 = 0, v_1, v_2, v_3)}\\
&=\frac {\tilde P(H_1, v_1, v_2, v_3)}{\tilde P(H_1 = 1, v_1, v_2, v_3)+\tilde P(H_1 = 0, v_1, v_2, v_3)}\\
&=\frac {\exp\{a^tH_1 - a^t(v_1+v_2 +v_3)-w^tH_1(v_1 + v_2 + v_3)\}}{\exp\{a^t-a^t(v_1+v_2+v_3)-w^t(v_1 +v_2 +v_3)\}+\exp\{-a^t(v_1+v_2+v_3)\}}\\
&=\frac {\exp\{a^tH_1 - w^tH_1(v_1 + v_2 + v_3)\}}{\exp\{a^t-w^t(v_1+v_2+v_3)\} + 1}\\
&=\frac {\exp\{H_1[a^t - w^t(v_1 + v_2 + v_3)]\}}{\exp\{a^t-w^t(v_1+v_2+v_3)\} + 1}\\
\end{align}
$$

and

$$
\begin{align}
P(H_2\mid v_1,v_2,v_3) &=\frac {P(H_2, v_1, v_2, v_3)}{\sum_{H_2}P(H_2, v_1, v_2,v_3)}\\
&=\frac {P(H_2, v_1, v_2, v_3)}{P(H_2 = 1, v_1, v_2, v_3)+P(H_2 = 0, v_1, v_2, v_3)}\\
&=\frac {\tilde P(H_2, v_1, v_2, v_3)}{\tilde P(H_2 = 1, v_1, v_2, v_3)+\tilde P(H_2 = 0, v_1, v_2, v_3)}\\
&=\frac {\exp\{a^tH_2 - a^t(v_1+v_2 +v_3)+w^tH_2(v_1 + v_2 + v_3)\}}{\exp\{a^t-a^t(v_1+v_2+v_3)+w^t(v_1 +v_2 +v_3)\}+\exp\{-a^t(v_1+v_2+v_3)\}}\\
&=\frac {\exp\{a^tH_2 + w^tH_2(v_1 + v_2 + v_3)\}}{\exp\{a^t+w^t(v_1+v_2+v_3)\} + 1}\\
&=\frac {\exp\{H_2[a^t + w^t(v_1 + v_2 + v_3)]\}}{\exp\{a^t+w^t(v_1+v_2+v_3)\} + 1}.
\end{align}
$$

Because the hidden variables are binary, their expectation value will be:

$$
\begin{align}
E_{a^t, w^t}[H_1] = P(H_1 = 1\mid v_1, v_2, v_3) = \frac {\exp\{a^t-w(v_1 + v_2+v_3)\}}{\exp\{a^t -w(v_1 + v_2 + v_3)\} + 1},\tag{1}\\
E_{a^t, w^t}[H_2] = P(H_2 = 1\mid v_1, v_2, v_3) = \frac {\exp\{a^t+w(v_1 + v_2+v_3)\}}{\exp\{a^t +w(v_1 + v_2 + v_3)\} + 1}.\tag{2}\\
\end{align}
$$

We fill the hidden variables with their expectation values. We denote the filled value in the $m$ -th instance as $h_1[m], h_2[m]$. Thus the log-likelihood function will be

$$
\begin{align}
\ell(\mathcal D:\pmb \theta) &= \sum_{m=1}^N\ln P(h_1[m],h_2[m],v_1[m],v_2[m],v_3[m]\mid \pmb \theta)\\
&=\sum_{m=1}^M\ln\exp\{a(h_1[m] + h_2[m] - v_1[m] - v_2[m]-v_3[m])\\
&=\sum_{m=1}^M\{a(h_1[m] + h_2[m] - v_1[m] - v_2[m]-v_3[m])\\
&\quad+ w(h_2[m] - h_1[m])(v_1[m] + v_2[m] + v_3[m])\}-M\ln Z.
\end{align}
$$

Because the data set is fairly large, we consider stochastic gradient ascent method with minibatch. Therefore, we first restrict our attention to the log-likelihood of the $m$ -th instance:

$$
\begin{align}
\ell(\xi[m] :\pmb \theta)  
&=a(h_1[m] + h_2[m] - v_1[m] - v_2[m]-v_3[m])\\
&\quad+ w(h_2[m] - h_1[m])(v_1[m] + v_2[m] + v_3[m])-\ln Z.
\end{align}
$$

The gradient of $\ell(\xi[m]:\pmb \theta)$ with respect to $a$ is:

$$
\begin{align}
\frac {\partial }{\partial a}\ell(\xi[m]:\pmb \theta) &= h_1[m] + h_2[m]-v_1[m]-v_2[m]-v_3[m] - \frac {\partial }{\partial a}\ln Z,\tag{3}\\
\end{align}
$$

where the gradient of $\ln Z$ with respect to $a$ should be:

$$
\begin{align}
\frac {\partial }{\partial a}\ln Z&= E_{\pmb \theta}[f_a]\\
&=E_{\pmb \theta}[H_1 + H_2 - V_1 - V_2 - V_3].\tag{4}\\
\end{align}
$$

The gradient of the minibatch's likelihood will be the average of the samples' gradient in the minibatch.

To calculate the expectation, we need do inference for all the variables with respect to the current parameter $\pmb \theta$. 

Here, we use Gibbs sampling to draw samples from the current distribution $P_{\pmb \theta}$ and use Monte Carlo method to approximate the expectation above.

The gradient of $\ell(\xi[m]:\pmb \theta)$ with respect to $w$ is:

$$
\begin{align}
\frac {\partial }{\partial w}\ell(\xi[m]:\pmb \theta) &= (h_2[m]-h_1[m])(v_1[m] + v_2[m] + v_3[m]) - \frac {\partial }{\partial w}\ln Z,\tag{5}\\
\end{align}
$$

where the gradient of $\ln Z$ with respect to $w$ should be:

$$
\begin{align}
\frac {\partial }{\partial w}\ln Z&= E_{\pmb \theta}[f_w]\\
&=E_{\pmb \theta}[(H_2 - H_1)(V_1 + V_2 + V_3)].\tag{6}\\
\end{align}
$$

Here, we also use Gibbs sampling to draw samples from the current distribution $P_{\pmb \theta}$ and use Monte Carlo method to approximate the expectation above.

The whole algorithm is presented below:
**Input**: Data set $\mathcal D$ with $M$ IID samples, Epoch number $epochs$, Learning rate $\gamma$, Sample number $s$, Minibatch size $batchsize$
 1: Initialize $a^0, w^0$ with random values.
 2: **for** $t$ **in** $[1, epochs]$
 3:   Shuffle $\mathcal D$ .
 4:   **for** $batch$ **in** $[1,M/batchsize]$
 5:   Do inference for $H_1, H_2$, and calculate $h_1[m], h_2[m]$ according to equation (1), and equation (2) for all samples in the batch.
 6:   Do Gibbs sampling to draw $s$ samples from $P_{a^t, w^t}$, and use Monte Carlo method to approximate the expectations in equation (4) and equation (6).
 7:   Calculate the gradients $g_a := \frac {\partial }{\partial a}\ell (\xi[m]:a^t, w^t)$ and $g_w := \frac {\partial }{\partial w}\ell(\xi[m]:a^t,w^t)$ according to equation (3) and equation (5) for all samples in the batch.
 8:   Update $g_a, g_w$ with the average all samples' gradient.
 9:   Update $a^t$ as $a^{t+1} := a^t + \gamma g_a$.
10:   Update $w^t$ as $w^{t+1} := w^{t} + \gamma g_w$.
11:   **end for**
12:  **end for**
13: **Return** $a^{epochs + 1}, w^{epoches + 1}$.

(3) The learning process is plotted as follows:

![[homework/pics/probabilistic graph theory/pgm-7.1.1.png]]

We finally get $a = 0.384$ and $w=0.856$.

The inference process of hidden variables for each data instance is coded in the script `inference.py`, and the inference result is saved in `inference_result.csv`.