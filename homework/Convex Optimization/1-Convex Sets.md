# Problem 1
Suppose $T (x) : D \to \mathbb R^n$ is a function with $x \in D$, $D$ is a convex set in $\mathbb R^n$. The first and second order partial derivatives $T (x)$ are all continuous on $D$, and the corresponding Jacobian matrix $J (x)$ of $T (x)$ is always positive definite on $D$. Please prove that $T (x)$ is an injective function.

Solution:
$\forall x_1, x_2 \in D, x_1 \ne x_2$, suppose $T (x_1) = T (x_2)$

# Problem 2
Please prove or disprove that $\cos(x + y) \le x^2 + y^2, x, y \in \mathbb R$ formulates an ellipsoid.

Solution:
An ellopsiod is defined as 

$$
{\mathcal{E}}=\{x\mid(x-x_{c})^{T}P^{-1}(x-x_{c})\leq1\},
$$

where $P$ is symmetric and postive definite.

Define $v = [x, y]^T \in \mathbb R^2$, then reformulate $\cos (x+y)\le x^2 + y^2$ as

$$
\cos(x+y) \le v^Tv.
$$

# Problem 3
Which of the following sets $S$ are polyhedra? If possible, express $S$ in the form $S = \{\pmb x \mid A\pmb x \le b, F\pmb x = g\}$.

- $S = \{y_1a_1 + y_2 a_2 \mid -1\le y_1 \le 1, -1\le y_2 \le 1\}$, where $a_1,a_2 \in \mathbb R^n$.
- $S = \{\pmb x \in \mathbb R^n \mid \pmb x \ge \pmb 0, \pmb 1^T \pmb x = 1, \sum_{i=1}^n x_ia_i = b_1, \sum_{i=1}^n x_i a_i^2 = b_2\}$, where $a_1,\dots, a_n \in \mathbb R$ and $b_1, b_2 \in \mathbb R$.
- $S = \{\pmb x \in \mathbb R^n \mid \pmb x \ge \pmb 0, \pmb x^T\pmb y \le 1 \text{ for all }y\text{ with } |y|_2 = 1\}$.
- $S = \{\pmb x\in \mathbb R^n \mid \pmb x \ge \pmb 0, \pmb x^T \pmb y \le 1 \text{ for all }y\text{ with }\sum_{i=1}^n |y_i| = 1\}$.


Solution:

(1) False


(2) True
$S = \{\pmb x \in \mathbb R^n \mid \pmb x \ge \pmb 0, \pmb 1^T \pmb x = 1, \sum_{i=1}^n x_ia_i = b_1, \sum_{i=1}^n x_i a_i^2 = b_2\}$ can be reformulated as

$$
S = \{\pmb x \mid (-I)\pmb x \le \pmb 0, \pmb 1^T \pmb x = 1, \pmb v^T\pmb x = b_1, \pmb q^T\pmb x = b_2 \}
$$

where $\pmb v = [a_1,a_2,\dots, a_n]^T, a_1,\dots, a_n \in \mathbb R,$ and $\pmb q = [b_1, b_2]^T, b_1, b_2 \in \mathbb R$. $I$ is the identity matrix.

(3) True
$S = \{\pmb x \in \mathbb R^n \mid \pmb x \ge \pmb 0, \pmb x^T\pmb y \le 1 \text{ for all }y\text{ with } |y|_2 = 1\}$ can be reformulated as

$$
S = \{\pmb x \mid (-I)\pmb x \le \pmb 0, W \pmb x \le \pmb 1\},
$$

where $W = [\pmb y_1, \pmb y_2, \dots]^T$ which contains all $\pmb y_i$ with $|\pmb y_i|_2 = 1$. $I$ is the identity matrix.

(4) True
Similiar to (3), we reformulate $S = \{\pmb x\in \mathbb R^n \mid \pmb x \ge \pmb 0, \pmb x^T \pmb y \le 1 \text{ for all }y\text{ with }\sum_{i=1}^n |y_i| = 1\}$ as

$$
S = \{\pmb x \mid (-I)\pmb x \le \pmb 0, W \pmb x \le \pmb 1\},
$$

where $W = [\pmb y_1, \pmb y_2, \dots]^T$ which contains all $\pmb y_i$ with $|\pmb y_i|_1 = 1$. $I$ is the identity matrix.
# Problem 4
We associate with each $A\in \mathbb S_{++}^n$ an ellipsoid centered at the origin, given by $E_A = \{\pmb x \mid \pmb x^T A^{-1}\pmb x \le 1\}$ . Please prove that we have $A\preceq B$ if and only if $E_A \subseteq E_B$.

Proof.
Supppose $A \preceq B$
