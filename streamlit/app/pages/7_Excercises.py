import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


st.header("Excercises")
st.markdown("---")
#############################
st.subheader("Bayes decision theory")
st.markdown("#### Fruit sorter")
st.markdown(
    r"""
    We got 3 classes:
    - $\omega_1:$ Banana $\Rightarrow P(\omega_1) = 0.3$
    - $\omega_2:$ Apple $\Rightarrow P(\omega_2) = 0.6$ 
    - $\omega_3:$ Lemon $\Rightarrow P(\omega_3) = 0.1$

    The feature that is available is the color of the fruit which can take the following values $x_1 =$ {green, yellow, red}.
    Empirically the following likelihoods (class-conditional probabilities) have been calculated.
""")


st.latex(
r"""

\begin {array}{ccc}
P("green"|\omega_1) = 0.4 & P("yellow"|\omega_1) = 0.6 & P("red"|\omega_1) = 0.0 \\
P("green"|\omega_2) = 0.3 & P("yellow"|\omega_2) = 0.3 & P("red"|\omega_2) = 0.4 \\
P("green"|\omega_3) = 0.3 & P("yellow"|\omega_3) = 0.7 & P("red"|\omega_3) = 0.0 \\

\end{array}
$$
""")

st.markdown("a) Compute the possible a posteriori values. Which fruit sort is detected by a MAP classifier if the color is green?")
st.markdown(r"""
Posterior is given with $P(\omega|x) = \frac{P(\omega)}{P(x)}P(x|\omega)$

Therefore we first compute the evidence for each feature 

$P(x = color) = \sum_i^3 P(x = color|\omega_i)~P(\omega_i)$
""")

# A Priori
p_prior = np.asarray([0.3, 0.6, 0.1])

# Likelihoods row -> class, column -> color
p_likelihoods = np.asarray([
    [0.4, 0.6, 0.0],
    [0.3, 0.3, 0.4],
    [0.3, 0.7, 0.0]
    ])

p_c = list()

# Calculate Evidence
for color in range(p_likelihoods.shape[0]):
    p = np.dot(p_prior, p_likelihoods[:,color])
    p_c.append(p)


st.code(
    """
# A Priori
p_prior = np.asarray([0.3, 0.6, 0.1])

# Likelihoods row -> class, column -> color
p_likelihoods = np.asarray([
    [0.4, 0.6, 0.0],
    [0.3, 0.3, 0.4],
    [0.3, 0.7, 0.0]
    ])

p_c = list()

# Calculate Evidence
for color in range(p_likelihoods.shape[0]):
    p = np.dot(p_prior, p_likelihoods[:,color])
    p_c.append(p)
""")

st.markdown(
    r"""
    Which results in the following values:
    $$
    \begin{array}{ccc}
    P(green) = 0.33 & P(yellow) = 0.43 & P(red) = 0.24
    \end{array}
    $$
""")

st.markdown("Now we can compute the posteriors.")
posterior = np.zeros_like(p_likelihoods)
for i, el in enumerate(p_likelihoods):
    posterior[i, :] = el * p_prior[i]

posterior = posterior / np.asarray(p_c)[None, :]

st.code(
    """
    posterior = np.zeros_like(p_likelihoods)
for i, el in enumerate(p_likelihoods):
    posterior[i, :] = el * p_prior[i]

posterior = posterior / np.asarray(p_c)[None, :]
""")

st.markdown("Resulting in:")
st.code(
r"""

[
    [0.36363636 0.41860465 0.        ]
    [0.54545455 0.41860465 1.        ]
    [0.09090909 0.1627907  0.        ]
    ]    

""")