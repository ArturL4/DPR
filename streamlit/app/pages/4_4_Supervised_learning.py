import streamlit as st
from utils.datasets import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from utils.classifier.classifiers import eval_classifiers

sns.set_theme()




st.header("Supervised learnings")

st.markdown("---")

st.subheader("Datasets")

st.markdown("We want to evaluate different classifiers on different datasets. For this we first artificially generate some datasets.")
colors=["red", "blue"]
cmap_custom = ListedColormap(colors)


dataset = st.radio(
    r'**Choose a dataset**',
    options=["Two moons", "Four parallel distributions", "Four gaussian distributions", "Circular distribution"]
    )

if dataset == "Two moons":
    x_train, y_train , x_test, y_test = datasets.two_moons_dataset(1000, 0.9)
elif dataset == "Four parallel distributions":
    x_train, y_train , x_test, y_test = datasets.four_parallel_dataset(1000, 0.9)
elif dataset == "Four gaussian distributions":
    x_train, y_train , x_test, y_test = datasets.four_gaussian_dataset(1000, 0.9)
else:
    x_train, y_train , x_test, y_test = datasets.circular_dataset(1000, 0.9)


fig, ax = plt.subplots(figsize=(15, 8))
ax.set_title(dataset, size=20)
ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap_custom)
st.pyplot(fig)


st.markdown(
    """
    Ok now after we've generated our data, we want to evalaute 3 different classifiers on them.
    This shall give us an intuition where the limitations of the classifiers are.

    But let's start with some theory before applying those classifiers.
    """)

st.markdown("---")
st.markdown(r"**Nearest Mean Classifier**")
st.markdown("We start simple by taking a closer look onto the nearest mean classifier.")
st.markdown(
    r"""
    The name somehow already tells everything about the classifier.

    Assume we got dataset $d = \{x(n), y(n)\}$ with $n=1, 2, ..., N$ indexes the sample, $x$ describing the feature and $y$ being the label.

    For this example we stay in 1D feature space and only consider 2 classes (Just for simplicity, the concept however can be applied to higher dimensions and more classes easily).

    With a training dataset given, we can simply calculate the features mean for each individual class e.g.

    $$
    \mu_j = \frac{1}{N_j}\sum_{y_n = \omega_j} x_n \qquad 1 \leq j \leq C
    $$
    """)
st.markdown(
    r"""
    we got two mean values now, $\mu_1$ and $\mu_2$. And now all we need to do is when getting some new input, compute the distance of a new sample to both means, compare both and assign the label to it, which distance is smaller.

    $$
    d_j = D(x, \mu_j)
    $$
    """)
st.markdown(
    r"""
    For now we have only taken into account the mean. 
    The idea can be further extended to also include the Covariance matrix.

    $$
    C_j = \frac{1}{N_j} \sum_{y_n=\omega_j}(x_n - \mu_j)(x_n - \mu_j)^T
    $$
    """)

st.markdown(
    r"""
    We extend our classical euclidean distance to consider covariances as well, which results in the so called :red[Mahalanobis Distance]

    $$
    D_{Maha} = \sqrt{(x_n - \mu_j)^T C(x_n - \mu_j)^T}
    $$

    But let's just visualize a bit
    """)

@st.cache_data
def one_sample_creation(mu1, sigma1, mu2, sigma2, size=100):
    if not size % 2 == 0:
        size += 1
    x_sample_1 = np.random.normal(mu1, sigma1, size//2)
    x_sample_2 = np.random.normal(mu2, sigma2, size//2)
    y_sample = np.zeros(size)
    y_sample[size//2::] = 1
    x_sample = np.concatenate([x_sample_1, x_sample_2])
    return x_sample, y_sample

x_sample, y_sample = one_sample_creation(2, 1, 4, 1, 100)

fig, ax = plt.subplots(figsize=(15, 8))
ax.vlines(x_sample[y_sample == 0], ymin=-0.01, ymax=0.01, colors="r")
ax.vlines(x_sample[y_sample == 1], ymin=-0.01, ymax=0.01, colors="b")
st.pyplot(fig)



fig = eval_classifiers(x_train, y_train, x_test, y_test)
st.pyplot(fig)