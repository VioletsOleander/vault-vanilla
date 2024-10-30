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
(1) 

$$
\begin{align}
p(\pmb H\mid \pmb V) &= \frac {P(\pmb H, \pmb V)}{p(\pmb V)}\\
&=\frac {P(\pmb H, \pmb V)}{\sum_{\pmb H}P(\pmb H, \pmb V)}\\
&=\frac {\frac 1 Z e^{-E(\pmb H, \pmb V)}}{\sum_{\pmb H}\frac 1 Z e^{-E(\pmb H, \pmb V)}}\\
&=\frac {e^{-E(\pmb H, \pmb V)}}{\sum_{\pmb H}e^{-E(\pmb H, \pmb V)}}\\
&=\frac {e^{-E(\pmb H, \pmb V)}}{\sum_{\pmb H}e^{-E(\pmb H, \pmb V)}}\\
&=\frac {e^{-E(\pmb H, \pmb V)}}{\sum_{H_1}\sum_{H_2}e^{-E(\pmb H, \pmb V)}}\\
\end{align}
$$

Because

$$
\begin{align}
-E(\pmb H, \pmb V) 
&= \sum_{i=1}^2 a_i h_i + \sum_{j=1}^5b_j v_j + \sum_{ij}w_{ij}h_i v_j\\
&= a_1h_1 + a_2h_2 + \sum_{j=1}^5b_j v_j + \sum_i\sum_jw_{ij}h_i v_j\\
&= a_1h_1 + a_2h_2 + \sum_{j=1}^5b_j v_j + \sum_jw_{1j}h_1 v_j + \sum_j w_2jh_2 v_j\\
&= a_1h_1 + a_2h_2 + \sum_jw_{1j}h_1 v_j + \sum_j w_{2j}h_2 v_j +  \sum_{j=1}^5b_j v_j \\
&= a_1h_1 +\sum_jw_{1,j}h_1v_j +  a_2h_2 + \sum_j w_{2j}h_2 v_j +  \sum_{j=1}^5b_j v_j \\
\end{align}
$$

Therefore

$$
\begin{align}
e^{-E(\pmb H, \pmb V)}&=e^{}
\end{align}
$$



