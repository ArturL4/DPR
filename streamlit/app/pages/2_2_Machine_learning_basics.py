import streamlit as st
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os


image_dir = os.path.dirname(__file__)
image_dir = os.path.join(image_dir, "../images/basics")

#############################
st.subheader("Linear algebra")

# Vector norms
st.markdown("One quite important part especially in learning algorithms is of course the :red[vector norm].")
st.markdown("Vector norms are scalar representations of vectors/matrices/tensors which can be used to optimize given an objective.")
st.markdown(
            r"""
            We focus here on the most important norms which are:
            - $l_2-$norm or euclidean-norm: $||x||_2 = \sqrt{\sum_{i=1}^M x_i^2} = \sqrt{x^T x}$
            - $l_1-$norm: $||x||_1 = \sum_{i=1}^M |x_i|$
            - $l_0-$norm: $||x||_0 = \#$ non-zero elements
            """)


st.markdown("---")
############################
st.subheader("Random variables and probability distributions")
st.markdown(r"""
#### Objects and Concepts
1. Random Variable
2. Probability mass function 
   - (PMF, dicrete RV) $\sum_i p_i = 1$
3. Probability density function (PDF, continious RV) $\int p(x)dx = 1$
#### Operations
- Expectation
- $\mathbb{E}\left[g(x)\right] = \int g(x) * p(x) dx$
#### Important Distributions

Multivariate normal distribution

$X \in \R^d \sim \mathcal{N}(\mu, C)$ 

$p(x) = \frac{1}{(2\pi)^{d/2}|C|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T C^{-1}(x-\mu)\right)$

- Expectation: $\mathbb{E} \left[X\right] = \mu$
- Covariance: $Cov\left[X\right] = \mathbb{E} \left[XX^T\right] = C$


""")
            
sns.set_theme()
fig, ax = plt.subplots(figsize=(8,4))
np.random.seed(42)
x = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 1000)
pdf_x = stats.norm(0,1).pdf(x)
cdf_x = stats.norm(0,1).cdf(x)
sns.lineplot(x=x, y=pdf_x)
sns.lineplot(x=x, y=cdf_x)
plt.legend(["PDF","" , "CDF"])
plt.title("Normal distribution")
st.pyplot(fig)

# Laplace
st.markdown(
    r"""
Laplace distribution

$X \in \R \sim \text{Laplace}(\mu, b), b > 0$

$p(x) = \frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right)$

- Expectation: $\mathbb{E} \left[X\right] = \mu$
- Var: $Var\left[X\right] = \mathbb{E} \left[XX^T\right] = 2b$
""")

fig, ax = plt.subplots(figsize=(8,4))
x = np.linspace(stats.laplace.ppf(0.01), stats.laplace.ppf(0.99), 1000)
laplace_dis = stats.laplace.pdf(x)
laplace_cdf = stats.laplace.cdf(x)
sns.lineplot(x=x, y=laplace_dis)
sns.lineplot(x=x, y=laplace_cdf, alpha=0.5, c="r")
plt.legend(["PDF", "", "CDF"])
plt.title("Laplace distribution")
st.pyplot(fig)

# Bernoulli

# Categorical Distribution



st.markdown("---")
st.markdown(
    r"""
#### Math 

- product rule: $p(\underbar{x}, \underbar{y}) = p(\underbar{x}| \underbar{y}) p(\underbar{y}) = p(\underbar{y}| \underbar{x}) p(\underbar{x})$
- Bayes rule: $p(y | x) = p(x|y)\frac{p(y)}{p(x)}$
- Chain rule of probability: 

$p(x_1, x_2, ..., x_N) = p(x_1| \textcolor{red}{x_2}, ..., x_N)p(x_2 | \textcolor{red}{x_3}, ..., x_N) p(x_{N-1}|x_N)p(x_N)$ 

$\textcolor{red}{\text{The order can be arbitary}}$

#### Indepence and identical distributed (i.i.d.)

$p(x, y) = p(x)p(y) \Rightarrow p(x|y) = p(x)$

1. independent: $p(x_1, ..., x_N) = \prod_i^N p_i(x_i) , X_i \sim p_i(x_i)$
2. identical: $p_i(x_i) = p(x_i)$

$\Rightarrow p(x_1, ..., x_N) = \prod_i^N p(x_i)$
""")


st.markdown("---")
st.markdown(
    r"""
    #### Kernel-based density estimation

$\^{p}(x) = \frac{1}{N} \sum_{n = 1}^N k(x - x(n))$,

$k$ being the kernel function

$\textcolor{blue}{\text{estimation}~\^{p}(x)}$

$\textcolor{red}{\text{kernel}~k(x)}$

$\textcolor{green}{\text{samples}}$
""")

st.image(os.path.join(image_dir, "prob_estimation_kernel.jpg"))

st.markdown(r"""
kernel $k(x)$, like a PDF:
1. $k(x) \geq 0, \forall x$
2. $\int k(x)dx = 1$

#### Gaussian Kernel (smooth)
 
- $N(0, I): k(x) = \frac{1}{(2\pi)^{d/2}}\exp{-\frac{1}{2}||x||^2}$
- $N(0, \textcolor{red}{\sigma^2} I): k(x) = \frac{1}{(2\pi\textcolor{red}{\sigma^2})^{d/2}}\exp{-\frac{1}{2\textcolor{red}{\sigma^2}}||x||^2} \Rightarrow$ quite popular in detection and pattern recognition due variable and controlled width

#### Dirac kernel

$k(x) = \delta(x)$

1. $\delta(x) = \begin{cases}
   \inf &\text{if } x=0 \\
   0 &\text{if } x\neq 0
\end{cases}$

2. $\int \delta(x)dx = 1$
3. sampling property: $\int\delta(x - x_0) * f(x)dx = f(x_0)$
""")
            
# Example with dirac as kernel

np.random.seed(42)
samples_4 = np.random.normal(0,1, (4,))
samples_1000 = np.random.normal(0,1, (1000,))

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-5, 5, 0.001)
# Mean = 0, SD = 2.
fig, ax = plt.subplots(2,1)
ax[0].plot(x_axis, stats.norm.pdf(x_axis,0, 1), linestyle="dashed")
sns.ecdfplot(x=samples_4, ax=ax[0])
ax[0].vlines(samples_4, 0, 0.05, color="r")

ax[0].legend(["Real gaussian", "Estimated CDF", "Samples"])
ax[1].plot(x_axis, stats.norm.pdf(x_axis,0, 1), linestyle="dashed")
sns.ecdfplot(x=samples_1000, ax=ax[1])
ax[1].vlines(samples_1000, 0, 0.05, color="r")

st.pyplot(fig)


st.markdown("---")
############################
st.subheader("Kullback-Leibler divergence and cross entropy")
st.markdown(
    r"""
    Measure for dissimilarity between two distributions.

1. Continous-values RV, PDF
    - $X \sim p(x)$: true distribution of X
    - $q(x)$: approximation of $p(x)$ (provided by NN)

KLD between $p$ and $q$:

$D_{KL}(p || q) = \int p(x) * ln\frac{p(x)}{q(x)}dx = \mathbb{E}_{X\sim p} \left[ln\frac{p(x)}{q(x)}\right]$

2. Discrete-values RV, PMF
   - $X\sim P(x)$: true PMF of $X$
   - $Q(x)$: approximation of $P(x)$

$D_{KL}(P||Q) = \mathbb{E}_{X\sim P}\left[ln\frac{P(X)}{Q(x)}\right] = \sum_{i=1}^c P(x_i) ln\frac{P(x_i)}{Q(x_i)}$

Properties:
1. Nonnegative $D_{KL}(p||q) \geq 0, \forall p,q$
2. Equality $D_{KL}(p||q) = 0$ iff $p(x) = q(x), \forall x$
3. Asymmetry $D_{KL}(p||q) \neq D_{KL}(q || p)$
4. Additivity
5. Relation to Cross entropy

### TODO calculation of optimal KLD (see Lec3 - ~1:15h)

#### Cross Entropy
    """
)
st.markdown("---")
############################
st.subheader("Probabilistic framework of supervised learning")

st.markdown(
    r"""
    
Estimation is still underlying the bayes decision theorem. The main difference between Signal Models and machine learning is mainly, that the $\textcolor{red}{prior}$ $p(y)$ and the $\textcolor{red}{likelihood}$ $p(x|y)$ are unknown analytically. Therefore also the a $\textcolor{red}{posterior}$ is unknown.

With machine learning we're trying to approximate the posterior $p(y|x)$ by a parametric posterior $q(y|x; \theta)$, given by a DNN with parameter Vector $\theta$.

$ x \Rightarrow \left[\text{DNN}, \theta\right]$

function $q() \Leftarrow$ architecture of DNN, $\theta$ is unknown, learn $\theta$ from $D_{train} = \left\{x(n), y(n)\right\}$


Learning criterion 

$\min_\theta D_{KL}(p(x,y) || q(x, y; \theta))$, $p()$ is fixed and forward KL is used $\Rightarrow$ minimize Cross entropy.

since $q(x, y; \theta) = \underbrace{q(y|x;\theta)}_{\text{DNN}}*q(x)$
""")


st.latex(r"""


\begin{array}{cc}
H(p || q) = &-\int p(x, y) \ln q(y |x;\theta) dx dy \\ &\underbrace{- \int p \ln q(x) dx dy}_{\text{independent of $\theta \Rightarrow$ ignore during minimization}}

\end{array}
""")




st.latex(r" H(p || q)= \int\overbrace{ p(x,y)}^\text{estimate using kernel based estimation} \underbrace{-\ln q(y|x; \theta)}_{\text{negative log likelihood (NLL) loss $l(x, y; \theta)$}} + \text{const}")


st.latex(r"H(\textcolor{red}{\^{p}} || q) = \underbrace{\overbrace{\frac{1}{N} \sum_{n=1}^N }^\text{$\^{p}$ estimated with dirac}l(x(n), y(n); \theta)}_\text{$\mathcal{L}(\theta)$: cost function} + const")

""")
#### Role of DNN:

#- a computational model to approximate $p(y | x)$ by $q(y|x; \theta)
#- learn $\theta$ from $D_{train}$ by $\min L(\theta)$
#    """

st.markdown("---")
############################
