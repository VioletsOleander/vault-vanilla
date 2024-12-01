Student: 陈信杰
Student ID: 2024316427
# Problem 1

(1) Considering a simplest Bayesian network $\mathcal G$ which is only composed of a v-structure: $Y\rightarrow X \leftarrow Z$, we can easily derive that $\mathcal I (\mathcal G) = \{(Y\perp Z)\}$.

To transform $\mathcal G$ into a Markov network $\mathcal H$, we have several options:
1. $Y, X, Z$: $Y, X, Z$ all stands alone. Obviously, $(X\perp Z) \in \mathcal I (\mathcal H)$ holds for this case, thus we can easily derive that $\mathcal I (\mathcal G) \ne \mathcal I (\mathcal H)$.
2. $Y-X, Z$: $Y$ and $X$ are connected, and $Z$ stands alone. $(X\perp Z) \in\mathcal I (\mathcal H)$ holds for this case, so $\mathcal I (\mathcal G)\ne \mathcal I (\mathcal H)$.
3. $Y, X-Z$: $X$ and $Z$ are connected, and $Y$ stands alone. $(X\perp Y)\in \mathcal I (\mathcal H)$ holds for this case, so $\mathcal I (\mathcal G)\ne \mathcal I (\mathcal H)$.
4. $Y-X-Z$: $X$ and $Z$ are connected. $X$ and $Y$ are connected. $(Y\perp Z\mid X) \in \mathcal I (\mathcal H)$ holds for this case, so $\mathcal I (\mathcal G)\ne \mathcal I (\mathcal H)$.
5. $Y-X-Z, Y-Z$: All three nodes are connected with each other. $(Y\perp Z)\not\in \mathcal I (\mathcal H)$ holds for this case, so $\mathcal I (\mathcal G)\ne \mathcal I (\mathcal H)$.

We have enumerated all the possible construction for corresponding Markov network $\mathcal H$, but only to find that $\mathcal I (\mathcal G)\ne \mathcal I (\mathcal H)$ holds for all the cases. Therefore, no MN can perfectly represente a v-structure in BN.

(2) Consider a quadrilateral-shaped MN $\mathcal H$ with four nodes: $A, B, C, D$ and four edges: $A-B, A-C, B-D, C-D$. We can easily derive that $\mathcal I (\mathcal H) = \{(A\perp D\mid B, D), (B\perp C\mid A, D)\}$.

Consider construcing a BN $\mathcal G$ with $\mathcal I (\mathcal H)$ holds. 
Firstly, edge $A\rightarrow D$ or $A\leftarrow D$ should not be in $\mathcal G$ to avoid $A, D$ become directly dependent. Similarily, edge $B\rightarrow C$ or $B \leftarrow C$ should not be in $\mathcal G$ too.

Secondly, assuming there is no edge between $A, C$, $\mathcal G$ will encode the conditional independence between $A, C$ as $(A\perp C\mid \text{Pa}_A)$, which does not holds in $\mathcal I (\mathcal H)$. Therefore, we should guarantee $(A, C)$ in $\mathcal G$ is connected. For similar reason, we should guarantee the pairs $(A, B), (C, D), (B, D)$ is interally connected.

From (1), we konws that MN can not encode v-structure, therefore, there should not be v-structures in $\mathcal G$ either.

Now, options for the edges in $\mathcal G$ are the following two:
1. $A\rightarrow C \rightarrow D \rightarrow B \rightarrow A$
2. $A\rightarrow B \rightarrow D \rightarrow C \rightarrow A$

Both of them forms a cycle, which is illeagle for BN.

Therefore, no BN can perfectly represent a polygon.

# Problem 2
(1) According to the constructive solution by Julian Beseg, the $Q$ function is

$$
\begin{align}
Q(A, B, C, D, E, F) &= \alpha_AX_A + \alpha_BX_B + \alpha_CX_C + \alpha_DX_D + \alpha_E X_E\\
&+\beta_{A,B}X_AX_B + \beta_{A,E}X_AX_E+ \beta_{B,E}X_BX_E\\
&+\beta_{B,C}X_BX_C + \beta_{E,F}X_EX_F\\
&+\beta_{C,D}X_CX_D + \beta_{C,F}X_CX_F + \beta_{D,F}X_DX_F\\
&+\gamma_{A,B,E}X_AX_BX_E + \gamma_{C,D,F}X_CX_DX_F\\
&+\delta_{B,C,E,F}X_BX_CX_EX_F.\\
\end{align}
$$

Therefore, the Gibbs distribution is

$$
P(A,B,C,D,E,F) = \frac 1 Z\exp(-U),
$$

where

$$
\begin{align}
U &= -\left(Q(A,B,C,D,E,F)+\text{constant}\right),\\
Z&=\sum_{A,B,C,D,E,F}\exp(-U).
\end{align}
$$

(2) $\text{MB}_{X_A} = \{X_B, X_E\}$, $\text{MB}_{X_B} = \{X_A, X_E, X_C\}$, $\text{MB}_{X_C} = \{X_B, X_F, X_D\}$.

# Problem 3
(1) The Gibbs distribution is

$$
P(Y_1, \dots, Y_n, X) = \frac 1 Z \exp(-U),
$$

where

$$
\begin{align}
U &=-\Big(\sum_{i=1}^n \alpha_i Y_i + \alpha_{n+1}X \\
&+ \sum_{j=1}^{n-1}\beta_{j, j+1}Y_{j}Y_{j+1} + \sum_{k=1}^n\gamma_k Y_kX\\
&+\sum_{l=1}^{n-1}\delta_{l,l+1}Y_lY_{l+1}X\Big),\\
Z&=\sum_{Y_1,\dots, Y_n,X}\exp(-U).
\end{align}
$$

(2) The Gibbs distribution is

$$
P(Y_1, \dots, Y_n \mid X) = \frac 1 Z \exp(-U),
$$

where

$$
\begin{align}
U &=-\Big(\sum_{i=1}^n \alpha_i Y_i + \sum_{j=1}^{n-1}\beta_{j, j+1}Y_{j}Y_{j+1} \\&+ \sum_{k=1}^n\gamma_k Y_kX+\sum_{l=1}^{n-1}\delta_{l,l+1}Y_lY_{l+1}X\Big),\\
Z&=\sum_{Y_1,\dots, Y_n,X}\exp(-U).
\end{align}
$$

# Problem 4
(1) $p (\pmb H \mid \pmb V), p (H_1 \mid \pmb V), p (H_2 \mid \pmb V)$ can be writted as:

$$
\begin{align}
p(\pmb H \mid \pmb V)&=\frac {P(\pmb H, \pmb V)}{p(\pmb V)},\\
p(H_1 \mid \pmb V)&=\frac {\sum_{H_2}P(\pmb H, \pmb V)}{p(\pmb V)},\\
p(H_2 \mid \pmb V)&=\frac {\sum_{H_1}P(\pmb H, \pmb V)}{p(\pmb V)}.
\end{align}
$$

Using the above joint distribution form, $p (\pmb H \mid \pmb V)$ can be further wirtten as:

$$
\begin{align}
p(\pmb H\mid \pmb V) &= \frac {P(\pmb H, \pmb V)}{p(\pmb V)}\\
&=\frac {P(\pmb H, \pmb V)}{\sum_{\pmb H}P(\pmb H, \pmb V)}\\
&=\frac {\frac 1 Z e^{-E(\pmb H, \pmb V)}}{\sum_{\pmb H}\frac 1 Z e^{-E(\pmb H, \pmb V)}}\\
&=\frac {e^{-E(\pmb H, \pmb V)}}{\sum_{\pmb H}e^{-E(\pmb H, \pmb V)}}\\
&=\frac {e^{-E(\pmb H, \pmb V)}}{\sum_{H_1}\sum_{H_2}e^{-E(\pmb H, \pmb V)}}.\\
\end{align}
$$

Therefore, to prove $p (\pmb H \mid \pmb V) = \prod_{i=1}^2 p (H_i\mid \pmb V)$, we shall prove that

$$
\frac {e^{-E(\pmb H, \pmb V)}}{\sum_{H_1}\sum_{H_2}e^{-E(\pmb H, \pmb V)}}=\frac {\sum_{H_1}e^{-E(\pmb H, \pmb V)}}{\sum_{H_1}\sum_{H_2}e^{-E(\pmb H, \pmb V)}}\cdot\frac {\sum_{H_2}e^{-E(\pmb H, \pmb V)}}{\sum_{H_1}\sum_{H_2}e^{-E(\pmb H, \pmb V)}}
$$

holds, which is equivalent to prove

$$
e^{-E(\pmb H, \pmb V)} = \frac {\left(\sum_{H_1} e^{-E(\pmb H, \pmb V)}\right)\cdot\left(\sum_{H_2} e^{-E(\pmb H, \pmb V)}\right)}{\sum_{H_1}\sum_{H_2}e^{-E(\pmb H, \pmb V)}}
$$

holds.

Because

$$
\begin{align}
-E(\pmb H, \pmb V) 
&= \sum_{i=1}^2 a_i h_i + \sum_{j=1}^5b_j v_j + \sum_{ij}w_{ij}h_i v_j\\
&= a_1h_1 + a_2h_2 + \sum_{j=1}^5b_j v_j + \sum_i\sum_jw_{ij}h_i v_j\\
&= a_1h_1 + a_2h_2 + \sum_{j=1}^5b_j v_j + \sum_jw_{1j}h_1 v_j + \sum_j w_2jh_2 v_j\\
&= a_1h_1 + a_2h_2 + \sum_jw_{1j}h_1 v_j + \sum_j w_{2j}h_2 v_j +  \sum_{j=1}^5b_j v_j \\
&= a_1h_1 +\sum_jw_{1j}h_1v_j +  a_2h_2 + \sum_j w_{2j}h_2 v_j +  \sum_{j=1}^5b_j v_j, \\
\end{align}
$$

therefore

$$
\begin{align}
e^{-E(\pmb H, \pmb V)}&=e^{a_1h_1 +\sum_jw_{1j}h_1v_j}\cdot e^{a_2h_2 + \sum_j w_{2j}h_2 v_j}\cdot e^{\sum_{j=1}^5b_j v_j},
\end{align}
$$

and

$$
\begin{align}
&\sum_{H_1}\sum_{H_2}e^{-E(\pmb H, \pmb V)}\\
=& \sum_{H_1}\sum_{H_2}e^{a_1h_1 +\sum_jw_{1j}h_1v_j}\cdot e^{a_2h_2 + \sum_j w_{2j}h_2 v_j}\cdot e^{\sum_{j=1}^5b_j v_j}\\
=&e^{\sum_{j=1}^5b_j v_j}\cdot\left(\sum_{H_1}\sum_{H_2}e^{a_1h_1 +\sum_jw_{1j}h_1v_j}\cdot e^{a_2h_2 + \sum_j w_{2j}h_2 v_j}\right)\\
=&e^{\sum_{j=1}^5b_j v_j}\cdot\left(\sum_{H_1}e^{a_1h_1 +\sum_jw_{1j}h_1v_j}\right)\cdot\left(\sum_{H_2} e^{a_2h_2 + \sum_j w_{2j}h_2 v_j}\right),\\\\
&\sum_{H_1}e^{-E(\pmb H, \pmb V)}\\
=& \sum_{H_1}e^{a_1h_1 +\sum_jw_{1j}h_1v_j}\cdot e^{a_2h_2 + \sum_j w_{2j}h_2 v_j}\cdot e^{\sum_{j=1}^5b_j v_j}\\
=& e^{\sum_{j=1}^5b_j v_j}\cdot\left(\sum_{H_1}e^{a_1h_1 +\sum_jw_{1j}h_1v_j}\right)\cdot e^{a_2h_2 + \sum_j w_{2j}h_2 v_j},\\\\
&\sum_{H_2}e^{-E(\pmb H, \pmb V)}\\
=& \sum_{H_2}e^{a_1h_1 +\sum_jw_{1j}h_1v_j}\cdot e^{a_2h_2 + \sum_j w_{2j}h_2 v_j}\cdot e^{\sum_{j=1}^5b_j v_j}\\
=& e^{\sum_{j=1}^5b_j v_j}\cdot e^{a_2h_1 + \sum_j w_{2j}h_1 v_j}\cdot\left(\sum_{H_2}e^{a_1h_2 +\sum_jw_{1j}h_2v_j}\right).
\end{align}
$$

Therefore

$$
\begin{align}
&\frac {\left (\sum_{H_1} e^{-E (\pmb H, \pmb V)}\right)\cdot\left (\sum_{H_2} e^{-E (\pmb H, \pmb V)}\right)}{\sum_{H_1}\sum_{H_2}e^{-E (\pmb H, \pmb V)}}\\
=&\frac {(e^{\sum_{j=1}^5b_j v_j})^2\cdot e^{a_2h_1 + \sum_j w_{2j}h_1 v_j}\cdot e^{a_2h_2 + \sum_j w_{2j}h_2 v_j}\cdot\left(\sum_{H_2}e^{a_1h_2 +\sum_jw_{1j}h_2v_j}\right)\cdot\left(\sum_{H_1}e^{a_1h_1 +\sum_jw_{1j}h_1v_j}\right)}{e^{\sum_{j=1}^5b_j v_j}\cdot\left(\sum_{H_1}e^{a_1h_1 +\sum_jw_{1j}h_1v_j}\right)\cdot\left(\sum_{H_2} e^{a_2h_2 + \sum_j w_{2j}h_2 v_j}\right)}\\
=&{e^{\sum_{j=1}^5b_j v_j}\cdot e^{a_2h_1 + \sum_j w_{2j}h_1 v_j}\cdot e^{a_2h_2 + \sum_j w_{2j}h_2 v_j}\cdot}\\
=& e^{-E (\pmb H, \pmb V)}.
\end{align}$$

According to previsou analysis, we can conclude that

$$p (\pmb H \mid \pmb V) = \prod_{i=1}^2 p (H_i\mid \pmb V).$$

Similarily, $p (\pmb V \mid \pmb H), p (V_1 \mid \pmb H), p (V_2 \mid \pmb H), p(V_3 \mid \pmb H), p(V_4 \mid \pmb H), p(V_5\mid \pmb H)$ can be writted as:

$$
\begin{align}
p(\pmb V \mid \pmb H)&=\frac {P(\pmb H, \pmb V)}{p(\pmb H)},\\
p(V_1 \mid \pmb H)&=\frac {\sum_{V_2,V_3,V_4,V_5}P(\pmb H, \pmb V)}{p(\pmb H)},\\
p(V_2 \mid \pmb H)&=\frac {\sum_{V_1,V_3,V_4,V_5}P(\pmb H, \pmb V)}{p(\pmb H)},\\
p(V_3 \mid \pmb H)&=\frac {\sum_{V_1,V_2,V_4,V_5}P(\pmb H, \pmb V)}{p(\pmb H)},\\
p(V_4 \mid \pmb H)&=\frac {\sum_{V_1,V_2,V_3,V_5}P(\pmb H, \pmb V)}{p(\pmb H)}
\end{align}
$$

Using the above joint distribution form, $p (\pmb V \mid \pmb H)$ can be further wirtten as:

$$
\begin{align}
p(\pmb V\mid \pmb H) &= \frac {P(\pmb H, \pmb V)}{p(\pmb H)}\\
&=\frac {P(\pmb H, \pmb V)}{\sum_{\pmb V}P(\pmb H, \pmb V)}\\
&=\frac {\frac 1 Z e^{-E(\pmb H, \pmb V)}}{\sum_{\pmb V}\frac 1 Z e^{-E(\pmb H, \pmb V)}}\\
&=\frac {e^{-E(\pmb H, \pmb V)}}{\sum_{\pmb V}e^{-E(\pmb H, \pmb V)}}\\
&=\frac {e^{-E(\pmb H, \pmb V)}}{\sum_{V_1}\sum_{V_2}\sum_{V_3}\sum_{V_4}\sum_{V_5}e^{-E(\pmb H, \pmb V)}}.\\
\end{align}
$$

Therefore, to prove $p (\pmb V \mid \pmb H) = \prod_{j=1}^5 p (V_j\mid \pmb H)$, we shall prove that

$$
\frac {e^{-E(\pmb H, \pmb V)}}{\sum_{V_1,V_2,V_4,V_5,V_5}e^{-E(\pmb H, \pmb V)}}=\frac {\sum_{V_2,V_3,V_4,V_5}e^{-E(\pmb H, \pmb V)}\cdots\sum_{V_1,V_2,V_3,V_4}e^{-E(\pmb H, \pmb V)}}{(\sum_{V_1,V_2,V_3,V_4,V_5}e^{-E(\pmb H, \pmb V)})^5}
$$

holds, which is equivalent to prove

$$
e^{-E(\pmb H, \pmb V)} = \frac {\sum_{V_2,V_3,V_4,V_5}e^{-E(\pmb H, \pmb V)}\cdots\sum_{V_1,V_2,V_3,V_4}e^{-E(\pmb H, \pmb V)}}{(\sum_{V_1,V_2,V_3,V_4,V_5}e^{-E(\pmb H, \pmb V)})^4}
$$

holds.

Because

$$
\begin{align}
-E(\pmb H, \pmb V) 
&= \sum_{i=1}^2 a_i h_i + \sum_{j=1}^5b_j v_j + \sum_{ij}w_{ij}h_i v_j\\
&= \sum_{i=1}^2 a_i h_i + \sum_{j=1}^5(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)\\
\end{align}
$$

therefore

$$
\begin{align}
e^{-E(\pmb H, \pmb V)}&= e^{\sum_{i=1}^2 a_i h_i}\cdot \prod_{j=1}^5e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)}\\
\end{align}
$$

Similariliy, we can derive

$$
\begin{align}
&\sum_{V_1,V_2,V_3,V_4,V_5}e^{-E(\pmb H, \pmb V)}\\
=& \sum_{V_1,V_2,V_3,V_4,V_5}e^{\sum_{i=1}^2 a_i h_i}\cdot \prod_{j=1}^5e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)}\\
=& e^{\sum_{i=1}^2 a_i h_i}\cdot \sum_{V_1,V_2,V_3,V_4,V_5} \prod_{j=1}^5e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)}\\
=& e^{\sum_{i=1}^2 a_i h_i}\cdot \prod_{j=1}^5\sum_{V_j}e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)},\\\\

&\sum_{V_2,V_3,V_4,V_5}e^{-E(\pmb H, \pmb V)}\\
=& \sum_{V_2,V_3,V_4,V_5}e^{\sum_{i=1}^2 a_i h_i}\cdot \prod_{j=1}^5e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)}\\
=& e^{\sum_{i=1}^2 a_i h_i}\cdot e^{(b_5v_5 + \sum_{i=1}w_{ij}h_iv_5)} \prod_{j=2,3,4,5}\sum_{V_j} e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)},\\\\

&\vdots\\\\

&\sum_{V_1,V_2,V_3,V_4}e^{-E(\pmb H, \pmb V)}\\
=& \sum_{V_2,V_3,V_4,V_5}e^{\sum_{i=1}^2 a_i h_i}\cdot \prod_{j=1}^5e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)}\\
=& e^{\sum_{i=1}^2 a_i h_i}\cdot e^{(b_1v_1 + \sum_{i=1}w_{ij}h_iv_1)}\prod_{j=1,2,3,4}\sum_{V_j} e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)}.\\\\
\end{align}
$$

Therefore

$$
\begin{align}
&\frac {\sum_{V_2,V_3,V_4,V_5}e^{-E(\pmb H, \pmb V)}\cdots\sum_{V_1,V_2,V_3,V_4}e^{-E(\pmb H, \pmb V)}}{(\sum_{V_1,V_2,V_3,V_4,V_5}e^{-E(\pmb H, \pmb V)})^4}\\
=&\frac {e^{\sum_{i=1}^2 a_i h_i}\cdot e^{(b_5v_5 + \sum_{i=1}w_{ij}h_iv_5)} \prod_{j=2,3,4,5}\sum_{V_j} e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)}\cdots e^{\sum_{i=1}^2 a_i h_i}\cdot e^{(b_1v_1 + \sum_{i=1}w_{ij}h_iv_1)}\cdot\prod_{j=1,2,3,4}\sum_{V_j} e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)}}{(e^{\sum_{i=1}^2 a_i h_i}\cdot \prod_{j=1}^5\sum_{V_j}e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)})^4}\\
=&{e^{\sum_{i=1}^2 a_i h_i}\cdot e^{(b_5v_5 + \sum_{i=1}w_{ij}h_iv_5)} \cdots \cdot e^{(b_1v_1 + \sum_{i=1}w_{ij}h_iv_1)}\cdot}\\
=&e^{\sum_{i=1}^2 a_i h_i}\cdot \prod_{j=1}^5e^{(b_j v_j + \sum_{i=1} w_{ij}h_iv_j)}\\
=&e^{-E(\pmb H, \pmb V)}.
\end{align}$$

According to previsou analysis, we can conclude that

$$p (\pmb V \mid \pmb H) = \prod_{j=1}^5 p (V_j\mid \pmb H).$$

(2) No, $H_1, H_2$ are clearly not marginally independent according to the graph. There are five active paths between $H_1, H_2$ .

(3) The log ratio can be written as

$$
\begin{align}
&\log \frac {P(H_1 = 1 \mid \pmb V)}{P(H_1 = 0 \mid \pmb V)}\\
=&\log \frac {P(H_1=1,\pmb V)}{P(H_1 = 0,\pmb V)}\\
=&\log \frac {\sum_{H_2}P(H_1 = 1, H_2, \pmb V)} {\sum_{H_2}P(H_1 = 0,H_2, \pmb V)}\\
=&-\sum_{h_2}E(h_1 = 1, h_2,\pmb v ) + \sum_{h_2}E(h_1 = 0, h_2, \pmb v)\\
=&(a_1 + a_2 + \sum_{j}b_jv_j + \sum_{j}w_{1j}v_j + \sum_j w_{2j}v_j) -(a_2 + \sum_{j} b_jv_j + \sum_j
w_{2j}v_j)\\
=&(a_1 + \sum_j w_{1j}v_j).
\end{align}
$$

Therefore, the value of $P (H_1 \mid \pmb V)$ will increase $a_1 + \sum_j w_{1j}v_j$.

# Problem 5
(1) 

![[homework/pics/probabilistic graph theory/pgm-4.5.1.png]]

(2) According to the graph, 

$$
\begin{align}
&P(X_2, X_3 \mid X_1, X_4, X_5)\\
=&P(X_2 \mid X_1, X_4, X_5) P(X_3 \mid X_1, X_4, X_5)\\
=&P(X_2 \mid X_4, X_5)P(X_3\mid X_1, X_4).
\end{align}
$$

Denote $\Sigma^{-1}$ as $J$, then according to GMRF's basic property, we can derive

$$
\begin{align}
P(\pmb x) &\propto \exp\left(-\frac 1 2 \pmb x J\pmb x + (J\pmb \mu)^T\pmb x\right)\\
&=\exp\left(-\frac 1 2\sum_{i,j}J_{ij}x_ix_j + [\sum_j J_{1j}\mu_j, \dots,\sum_j J_{5j}\mu_j]\pmb x\right)\\
&=\exp\left(-\frac 1 2\sum_{i,j}J_{ij}x_ix_j + \sum_i x_i(\sum_{j}J_{ij}\mu_j)\right)\\
&=\exp\left(-\frac 1 2\sum_i J_{ii}x_i^2 -\sum_{i,j(i\ne j)}J_{ij}x_ix_j + \sum_i x_i (\sum_j J_{ij}u_j)\right)\\
&=\exp\left(-\frac 1 2\sum_i J_{ii}x_i^2 +\sum_i x_i J_{ii}u_i -\sum_{i,j(i\ne j)}J_{ij}x_ix_j + \sum_i x_i (\sum_{j(j\ne i)} J_{ij}u_j)\right)\\
&=\exp\left(\sum_i J_{ii}x_i(-\frac 1 2x_i + \mu_i) -\sum_{i,j(i\ne j)}J_{ij}x_ix_j + \sum_i x_i (\sum_{j(j\ne i)} J_{ij}u_j)\right)\\
&=\exp\left(\sum_i J_{ii}(-\frac 1 2x_i^2 + \mu_ix_i) -\sum_{i}x_i(\sum_{j(j\ne i)}J_{ij}x_j) + \sum_i x_i (\sum_{j(j\ne i)} J_{ij}u_j)\right)\\
&=\exp\left(\sum_i J_{ii}(-\frac 1 2x_i^2 + \mu_ix_i) -\sum_{i}x_i(\sum_{j(j\ne i)}J_{ij}x_j- J_{ij}u_j)\right)\\
&=\exp\left(\sum_i J_{ii}(-\frac 1 2x_i^2 + \mu_ix_i) -\sum_{i,j(i\ne j)}J_{ij}(x_ix_j- x_iu_j)\right).\\
\end{align}
$$

Therefore

$$
\begin{align}
P(X_2 \mid X_4, X_5) 
&\propto \exp\left(-\frac 1 2 J_{22}x_2^2 + J_{22}\mu_2x_2 -J_{24}x_2x_4-J_{25}x_2x_5 + J_{24}\mu_4x_2 + J_{25}\mu_5x_2\right)\\
&=\exp\left(-\frac 1 2h_1 x_2^2 + h_2\mu_2x_2 + h_{24}\mu_4x_2 + h_{25}\mu_5x_2-h_{25}x_2x_5-h_{24}x_2x_4\right),

\\\\

P(X_3 \mid X_1, X_4) 
&\propto \exp\left(-\frac 1 2 J_{33}x_3^2 + J_{33}\mu_3x_3 -J_{34}x_3x_4-J_{31}x_3x_1 + J_{34}\mu_4x_3 + J_{31}\mu_1x_3\right)\\
&=\exp\left(-\frac 1 2h_3 x_3^2 + h_3\mu_3x_3 + h_{34}\mu_4x_3 + h_{31}\mu_1x_3-h_{31}x_2x_1-h_{34}x_3x_4\right),\
\end{align}
$$

and

$$
\begin{align}
&P(X_2\mid X_4, X_5)P(X_3\mid X_1, X_4)\\
\propto& \exp\bigg(-\frac 1 2 J_{22}x_2^2 + (J_{22}\mu_2 -J_{24}x_4 -J_{25}x_5 + J_{24}\mu_4 + J_{25}\mu_5)x_2\\
&\quad\quad-\frac 1 2 J_{33}x_3^2 + (J_{33}\mu_3 - J_{34}x_4 - J_{31}x_1 + J_{34}\mu_4 + J_{31}\mu_1)x_3\bigg)\\
=&\exp\bigg(-\frac 1 2 J_{22}x_2^2 + \left(\mu_2 +\sum_{j=4,5}\frac {J_{2j}}{J_{22}}(\mu_j-x_j)\right)x_2\\
&\quad\quad-\frac 1 2 J_{33}x_3^2 + \left(\mu_3 + \sum_{j=1,4} \frac {J_{3j}}{J_{33}}(\mu_j-x_j)\right)x_3\bigg)\\
\end{align}
$$

Denote $\pmb x' = \begin{bmatrix}x_2, x_3\end{bmatrix}$, $J' = \begin{bmatrix} h_2 & 0 \\ 0 & h_3\end{bmatrix}$, $\pmb \mu' = \begin{bmatrix}\mu_2 +\sum_{j=4,5}\frac {J_{2j}}{J_{22}}(\mu_j-x_j) \\ \mu_3 + \sum_{j=1,4} \frac {J_{3j}}{J_{33}}(\mu_j-x_j)\end{bmatrix}=\begin{bmatrix}\mu_2 +\sum_{j=4,5}\frac {h_{2j}}{h_2}(\mu_j-x_j) \\ \mu_3 + \sum_{j=1,4} \frac {h_{3j}}{h_3}(\mu_j-x_j)\end{bmatrix}$, we can derive

$$
P (\pmb x') \propto \exp\left (-\frac 1 2 \pmb x' J'\pmb x' + (J'\pmb \mu')^T\pmb x'\right).
$$

Therefore

$$
P(X_2,X_3\mid X_1,X_4,X_5) \sim \mathcal N(\pmb \mu', J'^{-1}).
$$











