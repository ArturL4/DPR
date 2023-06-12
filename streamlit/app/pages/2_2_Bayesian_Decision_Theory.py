import json
import math
from collections import namedtuple

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import seaborn as sns
from plotly.subplots import make_subplots
from scipy.stats import norm
from streamlit_lottie import st_lottie

from utils.utils import confusion_matrix_plot

import streamlit as st

sns.set_theme()
np.random.seed(42)


#######################################################################################
#######################################################################################
#######################################################################################


st.subheader("Bayesian Theorem")
st.markdown(
    """Now after we got small insights into performance measurement 
(and of course other basics :angel:) we want now to take a closer look into 
the most fundamental theorem in probability theory, which is used almost in 
every classifier."""
)

st.markdown("As you might have guessed, i'm talking about bayes theorem")

_, mid, _ = st.columns(3)

with mid:
    st.markdown(r"$P(\omega_i | x) = \frac{p(x|\omega_i)}{p(x)}P(\omega_i)$")

st.markdown(
    """ where:
1. $P(\omega = \omega_i) = P(\omega_i)$: a priori probability
2. $p(x|\omega=\omega_i) = p(x|\omega_i)$: class conditional PDF of x or likelihood
3. $p(x, \omega_i)$: joint PDF
4. $p(x)$: marginal PDF of x, called evidence
5. $P(\omega_i | x)$: a posterior probability, class probability after measurement of x
"""
)
st.markdown("---")

st.markdown("To get an idea what is even meant with the above abbrevation, let's have a look onto an example")
st.markdown(
    """Let's assume we got two classes of fish we want to distinguish. One shall be Salmon, the other Sea bass.
Furthermore we make the simplified assumption, that you can distinguish those fish with ONE single feature, e.g. the length."""
)


seabass_mean, seabass_var = 30, 10
salmon_mean, salmon_var = 80, 25
X_AXIS = np.arange(0, 150, 0.002)

seabass_samples = np.random.normal(seabass_mean, seabass_var, 100)
salmon_samples = np.random.normal(salmon_mean, salmon_var, 100)
seabass_dis = norm.pdf(X_AXIS, seabass_mean, seabass_var)
salmon_dis = norm.pdf(X_AXIS, salmon_mean, salmon_var)

fig, ax = plt.subplots()

ax.vlines(salmon_samples, -0.001, 0.001, color="r")
ax.vlines(seabass_samples, -0.001, 0.001, color="b")
ax.plot(X_AXIS, seabass_dis, color="b", linestyle="dashed")
ax.plot(X_AXIS, salmon_dis, color="r", linestyle="dashed")
ax.set_xlabel("Length in [cm]")
ax.set_ylabel(r"Likelihood $P(x | \omega)$")
plt.legend(["Salmon", "Sea bass"])
st.write(fig)

st.markdown(
    "Furthermore we say we are at a point on earth where the prior probability $P(\omega)$ is equal for both fish (how lucky...)"
)

st.markdown("""Enough theory, let's catch a fish and measure it!""")
col1, col2 = st.columns(2)
with col1:
    st.markdown("...")
with col2:
    fish_url = requests.get("https://assets7.lottiefiles.com/packages/lf20_6wvpi7jz.json")
    fish_json = dict()
    fish_json = fish_url.json()
    st_lottie(
        fish_json,
        height=200,
        width=400,
        # speed of animation
        speed=1,
        # means the animation will run forever like a gif, and not as a still image
        loop=True,
        # quality of elements used in the animation, other values are "low" and "medium"
        quality="high",
    )

st.markdown("Amazing!! The fish is 60cm long")
fig, ax = plt.subplots()

ax.vlines(salmon_samples, -0.001, 0.001, color="r")
ax.vlines(seabass_samples, -0.001, 0.001, color="b")
ax.vlines(60, -0.001, 0.015, color="g")
ax.plot(X_AXIS, seabass_dis, color="b", linestyle="dashed")
ax.plot(X_AXIS, salmon_dis, color="r", linestyle="dashed")
ax.set_xlabel("Length in [cm]")
ax.set_ylabel(r"Likelihood $P(x | \omega)$")
plt.legend(["Salmon", "Sea bass", "Caught fish"])
st.write(fig)

st.markdown(
    """Plotting the newly caught fish into our plot from beforehand, we probably directly see, that it might will be a salmon, but what is the probability given the length that it is going to be a salmon?
For this we calculate the so called :red[a posterior $P(\omega | x)$] using bayes theorem.
"""
)




st.markdown(
    r"""
    Since we're dealing with a continuous feature space, we need to adapt bayes theroem a little bit.

    $$
    P(\omega|x) = \frac{P(x)}{f(x)}f(x|\omega)
    $$
    where $f(x) = \sum_i^C f(x|\omega_i)$ is the :red[evidence] given as probability density function
    and $f(x|\omega)~\sim~\mathcal{N}(\mu_c, C_c)$ is the :red[class conditional likelihood].
    """)

st.markdown(
    """
    Now, since working with continuous probability density functions is quite unhandy, we're going to bin the given probability density functions.
    """)
seabass_mean, seabass_var = 30, 10
salmon_mean, salmon_var = 80, 25
X_AXIS = np.arange(0, 150, 0.002)

seabass_samples = np.random.normal(seabass_mean, seabass_var, 100)
salmon_samples = np.random.normal(salmon_mean, salmon_var, 100)
seabass_dis = norm.pdf(X_AXIS, seabass_mean, seabass_var)
salmon_dis = norm.pdf(X_AXIS, salmon_mean, salmon_var)

fig, ax = plt.subplots()

ax.set_xlabel("Length in [cm]")
ax.set_ylabel(r"Likelihood $P(x | \omega)$")

e  = sns.histplot(x=seabass_samples, stat="probability", kde=True, binwidth=5, color="b")
sns.histplot(x=salmon_samples, stat="probability", kde=True, binwidth=5, color="r")
plt.legend(["Sea bass", "Salmon"])
st.pyplot(fig)

st.info(
    r"""
    As you can probably see, the density functions are not perfectly gaussian even though the examples are sampled from a gaussian distribution.
    This is due not enough samples + quantization from the binning process.
    """)



st.markdown("---")

#######################################################################################
#######################################################################################
#######################################################################################


st.subheader("2.2. Minimum Bayesian risk decision")

st.markdown(
    r"""After we've learned about the confusion matrix we might be happy thinking ... 
Nice i can measure the performance of my classifier! 
But hold on... What if i got like 100 different classes i want to measure my classifier's performance? Do i really need to scan the 
$100\times 100$ matrix and checkout every cell, interprete and recognize other values?"""
)

st.markdown("""We rather want to have a scalar to measure the overall performance of the Classifier. In this case we can calculate the overall error rate.
This can be done by first computing the confusion matrix, use joint normalization""", 
help=r"""Do you remember? :angel:

It was simply $P_j = P(\omega = \omega_j) = \frac{n_{:j}}{N}$""")

st.markdown(
    r"""
And disentangle the overall sum of the probabilities. All diagonal elements represent correctly classified, while every other cell is some misclassification.
$$
\begin{array}{cccc}
   1 &= \sum_i \sum_j P_{ij} &= \sum_i P_{ii} &+ \sum \sum_{i\neq j} P_{ij} \\
    & &= \underbrace{P(\^{\omega} = \omega)}_{\text{recognition rate}} &+ \underbrace{P(\^{\omega} \neq \omega)}_\text{error rate (ER)} \\
\end{array}
$$"""
)

st.markdown(r" The error rate $0 \leq ER \leq 1$ is the average over all error.")

st.markdown("""Ok this is already much more convinient... But this comes with a price, we can't distinguish anymore,
 where the error actually comes from. Which misclassification is causing the huge impact at our error rate and also can't adjust the classification to our needs.

 That's where we introduce something new: :red[The loss term].

 But first a small example to get the idea why we want to include the loss.

 Assume we got a food shop. We are using a classifier to predict, whether our food is
 1. Still good
 2. Already becoming bad
""")
            


st.markdown("What would be more problematic when predicting wrong? Either Good food which was stated to be bad and put to the garbage, or bad food which is said to be good and sold to some costumer?")

col1, col2 = st.columns(2)
with col1:
    st.button("Wasting good food is terrible! :astonished:", key="button1")
with col2:
    st.button("Consuming bad food will probably lead to some stomach pain. :dizzy_face:", key="button2")

if "button1" not in st.session_state:
    st.session_state["button1"] = False

if "button2" not in st.session_state:
    st.session_state["button2"] = False


if st.session_state["button1"] or st.session_state["button2"]:
    if st.session_state["button1"]:
        st.warning("""
        Indeed wasting food is a big problem.
        But thinking about your financial situation, poison one of your
        costumers will probably ruin you. So i stick to option 2
        """)
    if st.session_state["button2"]:
        st.success("""
        Yes correctly. 
        If you think about poisoning one of your costumers,
        this will probably lead to financial ruin.
        """)
    
    st.markdown("""
    Ok now we know, making a prediction that bad food is good, 
    is actually worse than predicting good food to be gone bad.

    But what exactly does it have to do with loss?

    Recapping what our classifier need's to do, we try to minimize the Error Rate $ER$ from before.
    Now we got different cases, one which is of course not that good if a misclassification takes place
    and :red[one which will definitely ruin our career as shop owner].

    Now the $ER$ just gave us information regarding misclassification, but did not take into account the severity of a misclassification.

    That's where the loss get's it role. The loss is nothing but a weighting of the different terms added up during calculation of the $ER$.

    So constructing the loss we can start with creating a loss matrix
    """)

    st.latex(r"""
    
    l_{ij} = l(\^{\omega} = \omega_i , \omega = \omega_i) \geq 0 \\
    \left[l_{ij}\right] \in \R_+^{c\times c}
    
    """)

    st.markdown(r"""
    As you can see the dimension of the Loss matrix is the same as of our confusion matrix.
    Now what elements do we put onto our matrix?
    So first of all, the diagonal elements can have a 0. The reason is that correctly classified predictions are on the diagonal and we don't want to put penalty on this.
    But what about the other two elements?
    Now as already mentioned:
    - predicting good food as bad is not that harmful. We give it a loss of $l_{21} = 1$
    - predicting bad food as good however can ruin us. Let's put a huge penalty on this case $l_{12} = 10$

    Therefore we get the following loss matrix:
    """)

    st.latex(r"""
    \left[
    \begin{array}{cc}
    0 & 10 \\
    1 & 0

    \end{array}
    \right]
    """)

    st.markdown("""Easy, now the last step to calculate our weighted error rate is quite similar as before. Just iterate over all wrongly classified elements, multiply with the loss and add everything together""")
    st.latex(r"""
    BR = \sum \sum_{i\neq j} l_{ij} P_{ij}
    """)

    st.markdown("""Amazing! We came to the point where we did a small modification, which can have a huge impact on our carefree living as shop owner!
    This weighted loss by the way is called :red[Bayesian risk]""")

    st.markdown("---")



#######################################################################################
#######################################################################################
#######################################################################################

    '''


    
        
    X_AXIS = np.arange(-20, 20, 0.001)

    mean_1 = st.slider("Mean of Orange graph", 0.25, -5.0, 5.0)
    variance_1 = st.slider("Variance of orange graph", 2.0, 2.0, 6.0)
    h_1 = norm.pdf(X_AXIS, mean_1, variance_1)

    mean_fixed = 5.0
    variance_fixed = 1.5
    h_2 = norm.pdf(X_AXIS, mean_fixed, variance_fixed)

    fig, ax = plt.subplots()
    ax.plot(X_AXIS, h_1, c="orange")
    ax.plot(X_AXIS, h_2, c="b")
    idx = np.argwhere(np.diff(np.sign(h_2 - h_1))).flatten()
    if len(idx) <= 1:
        idx = np.append(idx, 31511)

    max_val = np.max(np.maximum(h_1, h_2))
    plt.fill_between(
        X_AXIS[idx[0] : idx[1]],
        h_1[idx[0] : idx[1]],
        h_2[idx[0] : idx[1]],
        hatch="\\",
        alpha=0.2,
        linewidth=0.1,
        facecolor="b",
        edgecolor="b",
    )
    plt.fill_between(
        X_AXIS[idx[0] : idx[1]],
        np.zeros_like(h_1[idx[0] : idx[1]]),
        h_1[idx[0] : idx[1]],
        hatch="\\",
        alpha=0.2,
        linewidth=0.1,
        facecolor="orange",
        edgecolor="orange",
    )

    plt.legend([r"$H_0$", r"$H_1$", r"$P_{D}$", r"$P_{FA}$"])
    ax.vlines(X_AXIS[idx], -0.01, max_val, color="r", linestyles="dashed", linewidth=0.3)
    st.write(fig)


    import numpy as np
    import plotly.graph_objects as go

    # Create figure
    fig = go.Figure()


    trace1 = go.Scatter(
        visible=True,
        line=dict(color="orange", width=3),
        name="Noise",
        x=np.arange(0, 10, 0.01),
        y=norm.pdf(np.arange(0, 10, 0.01), 4, 1.4),
    )

    trace2 = go.Scatter(
        name="Target", line=dict(width=3), x=np.arange(0, 10, 0.01), y=norm.pdf(np.arange(0, 10, 0.01), 7, 1)
    )

    fig.add_trace(trace1)

    fig.add_trace(trace2)

    """
    # Add traces, one for each slider step
    for step in np.arange(0, 5, 0.1):
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=3),
                name="𝜈 = " + str(step),
                x=np.arange(0, 10, 0.01),
                y=norm.pdf(np.arange(0, 10, 0.01), step, 1)))
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, 10, 0.01),
                y=norm.pdf(np.arange(0, 10, 0.01), 7, 1)))
            
        

    # Make 10th trace visible
    fig.data[10].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout()

    """
    fig.update_layout()
    st.write(fig)
    '''