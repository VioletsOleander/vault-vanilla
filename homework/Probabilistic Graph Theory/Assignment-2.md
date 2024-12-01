Student: 陈信杰
Student ID:  2024316427

# Problem 1
According to the chain rule and factorization theorem, we can easily derive:

$$
\begin{align}
P(A,B,C,D,E)&=P(A,C,D,E\mid B)P(B)\\
&=P(C,D,E\mid A,B)P(A\mid B)P(B)\\
&=P(D,E\mid C,A,B)P(C\mid A,B)P(A\mid B)P(B)\\
&=P(E\mid D,C,A,C)P(D\mid C,A,B)P(C\mid A,B)P(A\mid B)P(B)\\
&=P(E\mid D)P(D\mid A,B)P(C\mid A)P(A)P(B).
\end{align}
$$

To derive a formula for $P (B, C)$, considering decompose the joint probability as following:

$$
\begin{align}
P(A,B,C,D,E)&=P(A,D,E\mid B,C)P(B,C)\\
&=P(D,E\mid A,B,C)P(A\mid B,C)P(B,C)\\
&=P(E\mid D,A,B,C)P(D\mid A,B,C)P(A\mid B,C)P(B,C)\\
&=P(E\mid D)P(D\mid A,B)P(A\mid B,C)P(B,C)\\
&=P(E\mid D)P(D\mid A,B)\frac {P(A,B\mid C)}{P(B\mid C)}P(B,C)\\
&=P(E\mid D)P(D\mid A,B)\frac {P(A\mid C)P(B\mid C)}{P(B\mid C)}P(B,C)\\
&=P(E\mid D)P(D\mid A,B)P(A\mid C)P(B,C)\\
\end{align}
$$

Thus, we can derive:

$$
\begin{align}
P(E\mid D)P(D\mid A,B)P(A\mid C)P(B,C) &= P(E\mid D)P(D\mid A, B)P(C\mid A)P(A)P(B)\\
P(A\mid C)P(B,C)&=P(C\mid A)P(B)\\
P(B,C)&=\frac {P(C\mid A)}{P(A\mid C)}P(B)\\
P(B,C)&=\frac {P(C)}{P(A)}P(B)\\
P(B,C)&=\frac {\sum_{a\in Val(A)}P(C, a)}{P(A)}P(B)\\
P(B,C)&=\frac {\sum_{a\in Val(A)} P(C\mid a)P(a)}{P(A)}P(B)
\end{align}
$$

As for $P (B, C\mid E)$, we can derive:

$$
\begin{align}
P(B,C\mid E) &= \frac {P(B,C,E)}{P(E)}\\
&=\frac {\sum_{d\in Val(D)}P(B,C,d,E)}{\sum_{d \in Val(D)}P(d,E)}\\
&=\frac {\sum_{d\in Val(D)}P(B,C,d,E)}{\sum_{d \in Val(D)}P(E\mid d)P(d)}.\\

\end{align}
$$

$P (d)$ in the denominator can be reformulated as:

$$
\begin{align}
P(d) &= \sum_{a\in Val(A),b\in Val(B)}P(d,a,b)\\
&=\sum_{a\in Val(A),b\in Val(B)}P(d\mid a,b)P(a,b)\\
&=\sum_{a\in Val(A),b\in Val(B)}P(d\mid a,b)P(a)P(b).\\
\end{align}
$$

The $P(B, C, d, E)$ in the numerator can be reformulated as:

$$
\begin{align}
P(B,C,d,E)&=\sum_{a\in Val(A)}P(a,B,C,d,E)\\
&=\sum_{a\in Val(A)}P(E\mid d)P(d\mid a,B)P(C\mid a)P(a)P(B).
\end{align}
$$

Substisute the terms into the original fraction, we derive:

$$
P(B,C\mid E) =\frac {\sum_{d\in Val(D)}\left(\sum_{a\in Val(A)}P(E\mid d)P(d\mid a,B)P(C\mid a)P(a)P(B)\right)}{\sum_{d \in Val(D)}P(E\mid d)\left(\sum_{a\in Val(A),b\in Val(B)}P(d\mid a,b)P(a)P(b)\right)}
$$

# Problem 2
(1) 

$$
\begin{align}
&P(A,B,C,D,E,F,G,H,I,J,K)\\
=&P(A)P(B)P(C\mid A,B)P(D\mid B)P(F\mid C)P(G\mid C,D)P(E)P(H\mid D,E)\\
&P(I\mid F)P(J\mid G,I)P(K\mid H)
\end{align}
$$

(2)
a): True
b): True
c): False
d): False
e): False
f): False

(3)
a): $S_1 = \{E\}$
b): $S_2 = \{K,A\}$
c): $S_3 = \{E\}$

# Problem 3
(1)
The prefect I-map is drawn as:

![[homework/pics/probabilistic graph theory/pgm-2.3.1.png]]

(2)
The mininal I-map is drawn as:

![[homework/pics/probabilistic graph theory/pgm-2.3.1.png]]
I could not find a perfect I-map for this case.

# Problem 4
(1) We define 4 random variables:
$L$: binary-valued, indicating whether the criminal is lying. 
If the criminal lying, then $L=1$, else $L=0$.

$E$: real-valued, acquired from the EEG
$F$: discrete-valued, acquired by categorizing facial expressions
$V$: real-valued, acquiare from the voice recorder

Note $E, F, V$ are observed variables which contain noise.

The Bayesian network is drawn as following:

![[homework/pics/probabilistic graph theory/pgm-2.4.1.png]]

(2) We need to consider the correlation between continuous time steps, therefore, we add suffix $t$ for each random variables to denote that the variable is sampled at time step $t$.

Thus, we get 4 random variables at a given time step $t$: $L_t, E_t, F_t, V_t$

Considering continous time steps, we should realize that previous time steps may affect the following time steps. To simplify the model, we only consider the effect that the previous time step brought to the next time step, which is drawn as:

![[homework/pics/probabilistic graph theory/pgm-2.4.2.png]]

# Problem 5
The k-th base PDF of gamma-distributed random variable $x$ with $\alpha_k, \beta_k>0$ is written as:

$$
p_k(x\mid \pmb \alpha,\pmb \beta) =p(x\mid z = k,\pmb \alpha,\pmb \beta)= \frac {\beta_k^{\alpha_k}}{\Gamma(\alpha_k)}x^{\alpha_k -1}e^{-\beta_k x},\quad x\ge0
$$

Thus, the Gamma Mixture Model with proportions $\pi_1,\dots,\pi_K$ is:

$$
\begin{align}
P(x\mid \pmb \alpha,\pmb \beta) &= \sum_{k=1}^KP(z = k)p_k(x\mid \pmb \alpha, \pmb \beta)\\
&=\sum_{k=1}^K\pi_k p_k(x\mid \pmb \alpha,\pmb \beta)\\
&=\sum_{k=1}^K\frac {\pi_k\beta_k^{\alpha_k}}{\Gamma(\alpha_k)}x^{\alpha_k -1}e^{-\beta_k x},\quad x\ge0
\end{align}
$$

The graphical representation is as follows:

![[homework/pics/probabilistic graph theory/pgm-2.5.1.png]]
(2) 

$$
\begin{align}
P(z = k\mid x) &= \frac {P(z = k, x)}{P(x)}\\
&=\frac {P(x\mid z = k)P(z = k)}{\sum_{k=1}^KP(x\mid z= k)P(z = k)}\\
&=\frac {\pi_kp_k(x\mid \pmb \alpha, \pmb \beta)}{\sum_{k=1}^K\pi_k p_k(x\mid \pmb \alpha, \pmb \beta)}
\end{align}
$$

# Problem 6
(1)
(a)

$p (\pmb z[n]) = \mathcal N (\pmb z[n]\mid \pmb 0, \pmb I)$
$p (\pmb x[n]\mid \pmb z[n]) = \mathcal N (\pmb x[n]\mid \pmb W\pmb z[n] + \pmb \mu + \pmb \Psi)$

![[homework/pics/probabilistic graph theory/pgm-6.1.1.png]]
(b)
For factor analysis model: Because $\pmb W \in R^{D\times L}, \pmb \mu \in R^D, \Psi \in R^{D\times D}$, therefore $\pmb W$ accouts for $D\times L$ parameters, and $\pmb \mu$ accounts for $D$ parameters. Note that $\Psi$ is forced to be diagonal, therefore $\Psi$ accounts for $D$ parameters.

Thus, factor analysis model totally accounts for $D\times L + D + D = D\times (L + 2)$ parameters

For general multivariate Gaussian distribution: $p (\pmb x[n]) = \mathcal N (\pmb x[n]\mid \pmb \mu', \Psi')$
Because $\pmb \mu' \in R^D,\Psi' \in R^{D\times D}$, therefore $\pmb \mu'$ accounts for $D$ parameters, and $\Psi'$ accounts for $D\times D$ paramers.

Thus, general multivariate Gaussian model accounts for $D\times D + D = D\times (D + 1)$ paramters

Because $D\gg L$, $D\times (D + 1)$ is obviously much larger than $D\times (L + 2)$.

(2)
The data can be considered as a mixture of 4 components, therefore, we can define a mixture of 4 factor analysis model to fit this data.

We define an additional latent random variable $t$, which takes value $1,2,3,4$ with probability $\pi_1, \pi_2, \pi_3, \pi_4 (\sum_{i=1}^4 \pi_i = 1)$.

Then, we define parameters $\pmb W_i, \pmb \mu_i, \Psi_i(i=1,2,3,4)$ seperately for each component.

Finally, we define a mixture factor analysis model as:

$$
\begin{align}
P(\pmb x) &= \sum_{i=1}^4 P(t = i)p_i(\pmb x \mid \pmb z)\\
&=\sum_{i=1}^4 \pi_i\mathcal N(\pmb x\mid \pmb W_i\pmb z + \pmb \mu_i,\Psi_i)
\end{align}
$$
