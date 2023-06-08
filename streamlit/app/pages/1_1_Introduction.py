import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from utils.utils import confusion_matrix_plot

sns.set_theme()


st.subheader("1.2 Detection")

st.subheader("1.3 Pattern Recognition")

st.subheader("1.4 Confusion matrix")

st.markdown("The most fundamental performance measure in classification is the so called :red[Confusion matrix].")
st.markdown(
    r"A Confusion matrix $C \in \mathbb{N}^{c \times c}$ specifies the correctly/wrongly classified classes and comes with a specific layout."
)

st.markdown(
    """The regular confusion matrix comes with natural numbers, where simply is counted how often a class is predicted correctly/falsely. By using different normalization strategies we can however extract different information."""
)

st.markdown(
    r"""- Column sum: $n:j = \Sigma_{i=1}^c n_{ij} = \#(\omega = \omega_j)$
- Row sum: $n:i = \Sigma_{j=1}^c n_{ij} = \#(\^{\omega} = \omega_j)$
- Matrix sum: $N = \Sigma_{j,i=1}^c n_{ij} = \text{total}\#$

To have an illustration let's have a look onto the following case:
- c = 2 classes, $\omega_1 =$ salmon, $\omega_2 =$ sea bass.
"""
)
conf_fig, _ = confusion_matrix_plot(50, 5, 10, 35, "salmon", "sea bass", normalize=None)
st.write(conf_fig)
st.markdown(
    """
In the confusion matrix we can select different elements to identify how many classifications are correct or wrong:

- 50 fishes are salmons and are classified correctly as salmons
- 10 fishes are salmons and are classified as sea bass
- 5 fishes are sea bass and are classified as salmon
- 35 fishes are sea bass and are classified correctly as sea bass

Column sum: $50 + 10 = 60$ Overall salmons, $5 + 35 = 40$ overall sea bass.

Row Rum: $50 + 5 = 55$ classified salmons, $10 + 35 = 45$ classified sea bass

matrix sum: $50 + 10 + 5 + 35 = 100$ overall fishes"""
)


st.subheader("1.4.1 Normalization and abbrevation of Confusion matrix")

st.markdown(
    r"""We now had the first glance on what a confusion matrix is. But as you might already have thought of,
measuring the classifiers performance is not that easy given some absolute values. Especially if you think about extending to more than just 2 classes."""
)


st.markdown("To tackle this problem, we can simply normalize our existing confusion matrix.")

st.markdown(r"""The easiest way you can think of is probably the normalization over the whole count.

To mathematically express what i mean, given the absolute appearance of $N$ distinct objects/classifications we can express a relative appearance by simply dividing the different counts:
""")
            
col1, col2, col3 = st.columns(3)

with col2:
    st.write(r"$P_j = P(\omega = \omega_j) = \frac{n_{:j}}{N}$")

st.markdown("This gives us a the so called :red[marginal distribution].")
st.write(confusion_matrix_plot(50, 5, 10, 35, "salmon", "sea bass", normalize="all")[0])



st.markdown(
    r"""Now we can do much more with the confusion matrix. As we've seen we worked with absolute values for now. This might be not the most convinient way to measure performance of a classifier (e.g. accuracy is given with probabilities).

:red[Normalization of $n_{ij} \Rightarrow$ prob]:

1. Marginal probability: $P_j = P(\omega = \omega_j) = \frac{n_{:j}}{N}$
2. Joint Probability: by matrix sum normalization $P(\^{\omega} =\omega_i, \omega = \omega_j ) = \frac{n_{ij}}{N}$ 
3. Conditional probability: Column sum normalization. $P(\^{\omega} = \omega_i | \omega = \omega_j) = \frac{n_{ij}}{n_{:j}}$. This will lead to True positive, false positive, ... rates!
"""
)
