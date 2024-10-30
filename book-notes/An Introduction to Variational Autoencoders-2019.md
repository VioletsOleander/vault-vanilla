# 1 Introduction
## 1.5 Directed Graphical Models and Neural Networks
$$\begin{align}
\boldsymbol \eta&=\text{NeuralNet}(Pa(\mathbf x))\tag{1.7}\\
p_{\boldsymbol \theta}(\mathbf x | Pa(\mathbf x))&=p_{\boldsymbol \theta}(\mathbf x|\boldsymbol \eta)\tag{1.8}
\end{align}$$
## 1.8 Intractabilities
$$p_{\boldsymbol \theta}(\mathbf z |\mathbf x) = \frac {p_{\boldsymbol \theta}(\mathbf x, \mathbf z)}{p_{\boldsymbol \theta}(\mathbf x)}\tag{1.19}$$
联合分布$p_{\boldsymbol \theta}(\mathbf x, \mathbf z)$是可解的，因此边际$p_{\boldsymbol \theta}(\mathbf x)$和后验$p_{\boldsymbol \theta}(\mathbf z | \mathbf x)$只要其中之一可解，其中的另一个就可解
# 2 Variational Autoencoders
## 2.4 Reparameterization Trick
### 2.4.4 Computation of $\log q_{\phi}(\mathbf z | \mathbf x)$
$$\log q_{\phi}(\mathbf z |\mathbf x) = \log p(\boldsymbol \epsilon) -\log d_{\phi}(\mathbf x, \boldsymbol \epsilon)\tag{2.33}$$
## 2.5 Factorized Gaussian posteriors
$q_{\phi}(\mathbf z | \mathbf x) = \mathcal N(\mathbf z; \boldsymbol \mu, \text{diag}(\boldsymbol \sigma^2))$：
$$\begin{align}
(\boldsymbol \mu, \log \boldsymbol \sigma) &= \text{EncoderNeuralNet}_{\phi}(\mathbf x)\tag{2.36}\\
q_{\phi}(\mathbf z | \mathbf x) &= \prod_i q_{\phi}(z_i|\mathbf x)=\prod_i\mathcal N(z_i;\mu_i,\sigma_i^2)\tag{2.37}
\end{align}$$
再参数化之后，写为：
$$\begin{align}
\boldsymbol \epsilon &\sim \mathcal N(0, \mathbf I)\tag{2.38}\\
(\boldsymbol \mu, \log \boldsymbol \sigma) &= \text{EncoderNeuralNet}_{\phi}(\mathbf x)\tag{2.39}\\
\mathbf z&=\boldsymbol \mu + \boldsymbol \sigma \odot\boldsymbol \epsilon\tag{2.40}
\end{align}$$
雅可比矩阵为：
$$\frac {\partial \mathbf z}{\partial \boldsymbol \epsilon} = \text{diag}(\boldsymbol \sigma)$$
雅可比矩阵的对数行列式：
$$\log d_{\phi}(\mathbf x, \boldsymbol \epsilon) = \log \left|\det\left(\frac {\partial \mathbf z}{\partial \boldsymbol \epsilon}\right)\right| = \sum_i \log \sigma_i$$
因此，后验密度(posterior density)为：
$$\begin{align}
\log q_{\phi}(\mathbf z |\mathbf x) &= \log p(\boldsymbol \epsilon) -\log d_{\phi}(\mathbf x, \boldsymbol \epsilon)\\
&=\sum_i\log p(\epsilon_i) -\sum_i \log \sigma_i\\
&=\sum_i \log \mathcal N(\epsilon_i;0,1) - \sum_i\log \sigma_i
\end{align}$$
## 2.7 Marginal Likelihood and ELBO as KL Divergence
$q_{\mathcal D, \phi}(\mathbf x, \mathbf z)$和$p_{\theta}(\mathbf x, \mathbf z)$之间的KL散度可以写为负ELBO加上一个常数：
$$\begin{align}
&D_{KL}(q_{\mathcal D,\phi}(\mathbf x, \mathbf z) || p_{\boldsymbol \theta}(\mathbf x, \mathbf z))\tag{2.63}\\
=&-\mathbb E_{q_{\mathcal D}(\mathbf x)}\left[\mathbb E_{q_{\phi}}(\mathbf z | \mathbf x)[\log p_{\boldsymbol \theta}(\mathbf x, \mathbf z) - \log q_{\phi}(\mathbf z | \mathbf x)]-\log q_{\mathcal D}(\mathbf x)\right]\tag{2.64}\\
=&-\mathcal L_{\boldsymbol \theta, \phi}(\mathcal D) + \text{constant}\tag{2.65}
\end{align}$$
其中常数$\text{constant} = -\mathcal H(q_{\mathcal D}(\mathbf x))$，可见最大化ELBO等价于最小化$q_{\mathcal D, \phi}(\mathbf x, \mathbf z)$和$p_{\theta}(\mathbf x, \mathbf z)$之间的KL散度
(
$$\begin{align}
&D_{KL}(q_{\mathcal D,\phi}(\mathbf x, \mathbf z) || p_{\boldsymbol \theta}(\mathbf x, \mathbf z))\\
=&\mathbb E_{q_{\mathcal D, \phi}(\mathbf x, \mathbf z)}[\log q_{\mathcal D, \phi}(\mathbf x, \mathbf z)- \log p_{\theta}(\mathbf x, \mathbf z)]\\
=&\mathbb E_{q_{\mathcal D}(\mathbf x)}\left[\mathbb E_{q_{\phi}(\mathbf z | \mathbf x)}[\log q_{\mathcal D, \phi}(\mathbf x, \mathbf z)- \log p_{\theta}(\mathbf x, \mathbf z)]\right]\\
=&-\mathbb E_{q_{\mathcal D}(\mathbf x)}\left[\mathbb E_{q_{\phi}(\mathbf z | \mathbf x)}[\log p_{\theta}(\mathbf x, \mathbf z)-\log q_{\mathcal D, \phi}(\mathbf x, \mathbf z)]\right]\\
=&-\mathbb E_{q_{\mathcal D}(\mathbf x)}\left[\mathbb E_{q_{\phi}(\mathbf z | \mathbf x)}[\log p_{\theta}(\mathbf x, \mathbf z)-\log q_{\mathcal D}(\mathbf x)q_{\phi}(\mathbf z|\mathbf x)]\right]\\
=&-\mathbb E_{q_{\mathcal D}(\mathbf x)}\left[\mathbb E_{q_{\phi}(\mathbf z | \mathbf x)}[\log p_{\theta}(\mathbf x, \mathbf z)-\log q_{\mathcal D}(\mathbf x)-\log q_{\phi}(\mathbf z|\mathbf x)]\right]\\
=&-\mathbb E_{q_{\mathcal D}(\mathbf x)}\left[\mathbb E_{q_{\phi}(\mathbf z | \mathbf x)}[\log p_{\theta}(\mathbf x, \mathbf z)-\log q_{\phi}(\mathbf z|\mathbf x)]-\log q_{\mathcal D}(\mathbf x)\right]\\
=&-\mathbb E_{q_{\mathcal D}(\mathbf x)}\left[\mathbb E_{q_{\phi}(\mathbf z | \mathbf x)}[\log p_{\theta}(\mathbf x, \mathbf z)-\log q_{\phi}(\mathbf z|\mathbf x)]\right]-\mathcal H(q_{\mathcal D}(\mathbf x))\\
=&-\mathbb E_{q_{\mathcal D}(\mathbf x)}[\mathcal L_{\theta,\phi}(\mathbf x)]-\mathcal H(q_{\mathcal D}(\mathbf x))\\
=&-\mathcal L_{\theta,\phi}(\mathcal D)-\mathcal H(q_{\mathcal D}(\mathbf x))\\
\end{align}$$
)
ML和ELBO目标之间的关系可以总结如下：
$$\begin{align}
&D_{KL}(q_{\mathcal D, \phi}(\mathbf x, \mathbf z) || p_{\theta}(\mathbf x, \mathbf z))\tag{2.65}\\
=&D_{KL}(q_{\mathcal D}(\mathbf x) || p_{\theta}(\mathbf x))+ \mathbb E_{q_{\mathcal D}(\mathbf x)}[D_{KL}(q_{\mathcal D,\phi}(\mathbf z | \mathbf x )|| p_{\theta}(\mathbf z | \mathbf x))]\tag{2.67}\\
\ge&D_{KL}(q_{\mathcal D}(\mathbf x)||p_{\theta}(\mathbf x))\tag{2.68}
\end{align}$$
注意到最大化边际似然(ML)等价于最小化$q_{\mathcal D}(\mathbf x)$和$p_{\theta}(\mathbf x)$之间的KL散度，而它的上界是$q_{\mathcal D, \phi}(\mathbf x, \mathbf z)$和$p_{\theta}(\mathbf x, \mathbf z)$之间的KL散度，该上界通过对ELBO的最大化而最小化
(
$$\begin{align}
&D_{KL}(q_{\mathcal D, \phi}(\mathbf x, \mathbf z) || p_{\theta}(\mathbf x, \mathbf z))\\
=&\mathbb E_{q_{\mathcal D, \phi}(\mathbf x, \mathbf z)}[\log q_{\mathcal D, \phi}(\mathbf x, \mathbf z)- \log p_{\theta}(\mathbf x, \mathbf z)]\\
=&\mathbb E_{q_{\mathcal D, \phi}(\mathbf x, \mathbf z)}[\log q_{\mathcal D}(\mathbf x)q_{\phi}(\mathbf z|\mathbf x)- \log p_{\theta}(\mathbf x)p_{\theta}(\mathbf z|\mathbf x)]\\
=&\mathbb E_{q_{\mathcal D, \phi}(\mathbf x, \mathbf z)}[\log q_{\mathcal D}(\mathbf x)- \log p_{\theta}(\mathbf x)] + \mathbb E_{q_{\mathcal D, \phi}(\mathbf x, \mathbf z)}[\log q_{\phi}(\mathbf z| \mathbf x)- \log p_{\theta}(\mathbf z| \mathbf x)]\\
=&D_{KL}(q_{\mathcal D}(\mathbf x) || p_{\theta}(\mathbf x))+ \mathbb E_{q_{\mathcal D}(\mathbf x)}[D_{KL}(q_{\phi}(\mathbf z | \mathbf x )|| p_{\theta}(\mathbf z | \mathbf x))]\\
\ge&D_{KL}(q_{\mathcal D}(\mathbf x)||p_{\theta}(\mathbf x))
\end{align}$$
)

ELBO目标可以视为在增强空间(augmented space)的一个极大似然(ML)目标，对于某个固定的编码器选择(fixed choice of encoder)$q_{\phi}(\mathbf z | \mathbf x)$，我们将联合分布$p_{\theta}(\mathbf x, \mathbf z)$视作在原始数据$\mathbf x$和与每个数据点相关的(随机的)辅助特征$\mathbf z$上的增强的经验分布(augmented empirical distribution)
![[An Intro to VAE-Fig2.4.png]]
## 2.8 Challenges
### 2.8.1 Optimization issues
### 2.8.2 Blurriness of generative model
# 3 Beyond Gaussian Posteriors
# 4 Deeper Generative Models

