# Problem 1
(1) The mutilated network is as follows.

![[homework/pics/probabilistic graph theory/pgm-6.1.1.png]]

Using the distribution defined by the mutilated network as proposal distribution for importance sampling is equivalent to applying likelihood weighting sampling algorithm in the original BN.

The likelihood weighting sampling algorithm for the original BN is presented as follows.

**Requires**: Original Bayesian Network $\mathcal B$ over $\mathcal X$, the observation $\pmb Z = \pmb z$
 1: Let $X_1, \dots, X_n$ be a topological ordering of $\mathcal X$ in $\mathcal B$ 
 2: $w := 1$
 3: **for** $i=1, \dots, n$
 4:   Let $\pmb u_i$ be the sampled values of $\text{Pa}_{X_i}$
 5:   **if** $X_i \not \in \pmb Z$
 6:     Sample $X_i$ 's value $x_i$ from CPD $P(X_i \mid \pmb u_i)$
 7:   **else**
 8:     Set $x_i$ to the observed value of $X_i$ in $\pmb z$
 9:     $w:= w\cdot P(x_i \mid \pmb u_i)$
10:   **end if**
11: **end for**
12: **Return** sample $(x_1, \dots, x_n)$ and its weight $w$

(2)

**Requires**: Distribution $P_{\mathcal B}$ defined by the Bayesian network $\mathcal B$, variables $\pmb X$ to be sampled , initial sampling distribution $P^{(0)}(\pmb X)$, required time steps $T$
 1: $n := |\pmb X|$
 2: Sample $\pmb x^{(0)}$ from $P^{(0)}(\pmb X)$
 3: **for** $t = 1, \dots, T$
 4:   **for** $i = 1, \dots, n$
 5:     Sample $x_i^{(t)}$ from $P_{\mathcal B}(X_i \mid x_1^{(t)}, \dots, x_{i-1}^{(t)}, x_{i+1}^{(t-1)}, \dots, x_n^{(t-1)})$  
 6:    **end for**
 7: **end for**
 8: **Return** $\pmb x^{(T)}$ 

(3) The importance weight in this case is

$$
\frac {P_{\mathcal B}(\pmb X)}{Q(\pmb X)} = \frac {P_{\mathcal B}(\pmb X)}{\prod_{i}Q_i(X_i)}.
$$

The algorithm is presented below.

**Requires**: Proposal distribution $Q$ over $\pmb X$, observation $\pmb Z = \pmb z$
 1: $n: = |\pmb X|$
 2: $w:=1$
 3: **for** $i = 1, \dots, n$
 4:   **if** $X_i \not\in \pmb Z$
 5:     Sample $x_i$ from $Q_i(X_i)$
 6:     $w := \frac {w}{Q_i(x_i)}$
 6:   **else**
 7:      Set $x_i$ to the observed value of $X_i$ in $\pmb z$
 8:   **end if**
 9: **end for**
10: Let $\pmb x = (x_1, \dots, x_n)$
11: $w := w\cdot P_{\mathcal B}(\pmb x)$  
12: **Return** $\pmb x$, $w$

# Problem 2
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
\psi_{H_1}(H_1) &= \alpha_1 = 0.1\\
\psi_{H_2}(H_2) &= \alpha_2 = 0.1\\
\psi_{V_1}(V_1)&=\beta_1 = 0.5\\
\psi_{V_2}(V_2)&=\beta_2 = -0.3\\
\psi_{V_3}(V_3)&=\beta_3 = -0.5,\\
\end{align}
$$

and

$$
\begin{align}
\psi_{H_1V_1}(H_1,V_1) = \psi_{H_1V_2}(H_1,V_2) = \psi_{H_1V_3}(H_1V_3)=-1\\
\psi_{H_2V_1}(H_2,V_1) = \psi_{H_2V_2}(H_2,V_2) = \psi_{H_2V_3}(H_2V_3)=0.2.\\
\end{align}
$$

Therefore

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

(2) In Gibbs sampling, each transition kernel for variable $X_i$ is proved only involve the neighbors of $X_i$. Thus, we first consider each variable's transition kernel separately.

$H_1$ 's transition kernel is 

$$
\begin{align}
&P(H_1 \mid h_2, v_1, v_2, v_3)\\
=&\frac {P(H_1, h_2, v_1, v_2, v_3)}{\sum_{h_1 \in Val(H_1)}P(h_1, h_2, v_1, v_2, v_3)}\\
=&\frac {\phi_{H_1}(H_1)\phi_{H_1V_1}(H_1,v_1)\phi_{H_1V_2}(H_1,v_2)\phi_{H_1V_3}(H_1,v_3)}{\sum_{h_1\in Val(H_1)}\phi_{H_1}(h_1)\phi_{H_1V_1}(h_1,v_1)\phi_{H_1V_2}(h_1,v_2)\phi_{H_1V_3}(h_1,v_3)}\\
=&\frac {\exp(0.1H_1)\cdot\exp(-H_1v_1)\cdot\exp(-H_1v_2)\cdot\exp(H_1v_3)}{\sum_{h_1 \in Val(h_1)}\exp(-0.1h_1)\cdot \exp(-h_1v_1)\cdot \exp(-h_1v_2)\cdot \exp(h_1v_3)}\\
=&\frac {\exp(0.1H_1 -H_1v_1-H_1v_2-H_1v_3)}{\sum_{h_1 \in Val(H_1)}\exp(-0.1h_1-h_1v_1-h_1v_2-h_1v_3)}\\
=&\frac {\exp \{H_1(0.1 - v_1 -v_2 -v_3)\}}{\sum_{h_1\in Val(H_1)}\exp\{h_1(0.1-v_1-v_2-v_3)\}}\\
=&\frac {\exp \{H_1(0.1 - v_1 -v_2 -v_3)\}}{\exp\{0.1-v_1-v_2-v_3\} + \exp\{-0.1 + v_1 + v_2 + v_3\}}\\
\end{align}
$$

$H_2$ 's transition kernel is

$$
\begin{align}
&P(H_1\mid h_1, v_1, v_2, v_3)\\
=&\frac {\phi_{H_2}(H_2)\phi_{H_2V_1}(H_2,v_1)\phi_{H_2V_2}(H_2,v_2)\phi_{H_2V_3}(H_2,v_3)}{\sum_{h_2\in Val(H_2)}\phi_{H_2}(h_2)\phi_{H_2V_1}(h_2,v_1)\phi_{H_2V_2}(h_2,v_2)\phi_{H_2V_3}(h_2,v_3)}\\
=&\frac {\exp\{H_2(0.1+0.2v_1+0.2v_2+0.2v_3)\}}{\sum_{h_2 \in Val(H_2)}\exp\{H_2(0.1 + 0.2v_1 + 0.2v_2 + 0.2v_3)\}}\\
=&\frac {\exp\{H_2(0.1+0.2v_1+0.2v_2+0.2v_3)\}}{\exp\{0.1 + 0.2v_1 + 0.2v_2 + 0.2v_3\} + \exp\{-0.1 -0.2v_1-0.2v_2-0.2v_3\}}\\
\end{align}
$$

$V_1$ 's transition kernel is

$$
\begin{align}
&P(V_1 \mid h_1,h_2)\\
=&\frac {\phi_{V_1}(V_1)\phi_{H_1V_1}(h_1,V_1)\phi_{H_2V_1}(h_2,V_1)}{\sum_{v_1 \in Val(V_1)}\phi_{V_1}(v_1)\phi_{H_1V_1}(h_1,v_1)\phi_{H_2V_1}\phi(h_2,v_1)}\\
=&\frac {\exp\{0.5V_1 -h_1V_1 + 0.2h_2V_1\}}{\exp\{0.5V_1 -h_1V_1 + 0.2h_2V_1\} + \exp\{0.5V_1 -h_1V_1 + 0.2h_2V_1\}}\\
=&\frac {\exp\{V_1(0.5 -h_1 + 0.2h_2)\}}{\exp\{0.5 -h_1 + 0.2h_2\} + \exp\{-0.5+h_1 - 0.2h_2\}}\\
\end{align}
$$

$V_2$ 's transition kernel is

$$
\begin{align}
&P(V_2 \mid h_1,h_2)\\
=&\frac {\phi_{V_2}(V_2)\phi_{H_1V_2}(h_1,V_2)\phi_{H_2V_2}(h_2,V_2)}{\sum_{v_1 \in Val(V_2)}\phi_{V_2}(v_1)\phi_{H_1V_2}(h_1,v_1)\phi_{H_2V_2}\phi(h_2,v_1)}\\
=&\frac {\exp\{-0.3V_2 -h_1V_2 + 0.2h_2V_2\}}{\exp\{-0.3V_2 -h_1V_2 + 0.2h_2V_2\} + \exp\{-0.3V_2 -h_1V_2 + 0.2h_2V_2\}}\\
=&\frac {\exp\{V_2(-0.3 -h_1 + 0.2h_2)\}}{\exp\{-0.3 -h_1 + 0.2h_2\} + \exp\{0.3+h_1 - 0.2h_2\}}\\
\end{align}
$$

$V_3$ 's transition kernel is

$$
\begin{align}
&P(V_3 \mid h_1,h_2)\\
=&\frac {\phi_{V_3}(V_3)\phi_{H_1V_3}(h_1,V_3)\phi_{H_2V_3}(h_2,V_3)}{\sum_{v_1 \in Val(V_3)}\phi_{V_3}(v_1)\phi_{H_1V_3}(h_1,v_1)\phi_{H_2V_3}\phi(h_2,v_1)}\\
=&\frac {\exp\{-0.5V_3 -h_1V_3 + 0.2h_2V_3\}}{\exp\{-0.5V_3 -h_1V_3 + 0.2h_2V_3\} + \exp\{-0.5V_3 -h_1V_3 + 0.2h_2V_3\}}\\
=&\frac {\exp\{V_3(-0.5 -h_1 + 0.2h_2)\}}{\exp\{-0.5 -h_1 + 0.2h_2\} + \exp\{0.5+h_1 - 0.2h_2\}}\\
\end{align}
$$

Thus, the Gibbs sampling algorithm is simply iteratively apply these local transition kernels until convergence.

Following the requirement, the marginal distribution is estimated in a sliding window way, where the window size is 100 and the step size is 50. The estimated marginal distribution of the first 3000 samples (with 3,000,000 burn in steps) is plotted as follows:

![[homework/pics/probabilistic graph theory/pgm-6.2.1.png]]

(3) In the problem's setting, the proposal distribution in the MH sampling is completely factorized. Thus, the transition kernel $\mathcal T_i^{Q_i}$ for variable $X_i$ is completely irreverent with other variables.

In the problem's setting, each variable's proposal distribution defines transition kernel as a Bernoulli with parameter $0.5$, which is easy to implement.

According to the formula, the acceptance probability is defined as

$$
\begin{align}
\mathcal A (\pmb x_{-i}, x_i \rightarrow \pmb x_{-i}, x_i') &= \min \left[1, \frac {\pi (\pmb x_{-i}, x_i')\mathcal T_i^{Q_i}(\pmb x_{-i}, x_i'\rightarrow \pmb x_{-i}, x_i)}{\pi (\pmb x_{-i}, x_i)\mathcal T_i^{Q_i}(\pmb x_{-i}, x_i \rightarrow \pmb x_{-i}, x_i')}\right]\\
&=\min\left[1, \frac {P_\Phi (x_i',\pmb x_{-i})}{P_\Phi (x_i,\pmb x_{-i})}\frac {\mathcal T_i^{Q_i}(\pmb x_{-i}, x_i'\rightarrow \pmb x_{-i}, x_i)}{\mathcal T_i^{Q_i}(\pmb x_{-i}, x_i\rightarrow \pmb x_{-i}, x_i')}\right].
\end{align}
$$

In the problem's setting, is can be simplified as

$$
\begin{align}
\mathcal A (\pmb x_{-i}, x_i \rightarrow \pmb x_{-i}, x_i') 
&=\min\left[1, \frac {P_\Phi (x_i',\pmb x_{-i})}{P_\Phi (x_i,\pmb x_{-i})}\frac {0.5}{0.5}\right]\\
&=\min\left[1, \frac {P_\Phi (x_i',\pmb x_{-i})}{P_\Phi (x_i,\pmb x_{-i})}\right]\\
&=\min\left[1, \frac {P_\Phi (x_i'\mid\pmb x_{-i})}{P_\Phi (x_i\mid\pmb x_{-i})}\right].
\end{align}
$$

As previously mentioned, the conditional probability in the fraction can be further simplified to involve only $X_i$ 's Markov blanket. The specific form is similar to the previously stated one and is omitted here.

The estimated marginal distribution of the first 3000 samples (with 30,000,000 burn in steps) is plotted as follows:

![[homework/pics/probabilistic graph theory/pgm-6.2.2.png]]

which has little difference from the Gibbs sampling's result.

(4) Changing the possible values of variables to $(0, 1)$ , the plot of Gibbs sampling (with 10000 burn in steps) is:

![[homework/pics/probabilistic graph theory/pgm-6.2.3.png]]

The plot of MH sampling (with 30000 burn in steps) is:

![[homework/pics/probabilistic graph theory/pgm-6.2.4.png]]

which still has little difference from the result of Gibbs sampling.

Comparing the sampling methods with the variational method, it is obvious that the variational method is more stable. Two types of methods are both guaranteed to converge. The sampling method is guaranteed to converge to the accurate distribution (if the chain is regular), thus is better than the approximate one. However, the required number of steps to converge is unclear and usually very large. In optimization, both methods adopt a local calculation pattern (only involves neighbors) and take a iterative optimization strategy, but the sampling method's required number of steps to converge is too large, therefore the variational method is computationally more efficient.








