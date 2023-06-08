from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import requests
import json
from streamlit_lottie import st_lottie

import plotly.express as px
from plotly.subplots import make_subplots

sns.set_theme()


st.subheader("Bayesian Theorem")
st.markdown(
    """Now after we got small insights into performance measurement 
(and of course other basics :angel:) we want now to take a closer look into 
the most fundamental theorem in probability theory, which is used almost in 
every classifier."""
)

st.markdown("As you might have guessed, i'm talking about bayes theorem")

_, mid, _ =  st.columns(3)

with mid:
    st.markdown(r"$P(\omega_i | x) = \frac{p(x|\omega_i)}{p(x)}P(\omega_i)$")

st.markdown(""" where:
1. $P(\omega = \omega_i) = P(\omega_i)$: a priori probability
2. $p(x|\omega=\omega_i) = p(x|\omega_i)$: class conditional PDF of x or likelihood
3. $p(x, \omega_i)$: joint PDF
4. $p(x)$: marginal PDF of x, called evidence
5. $P(\omega_i | x)$: a posterior probability, class probability after measurement of x
""")
st.markdown("---")

st.markdown("To get an idea what is even meant with the above abbrevation, let's have a look onto an example")
st.markdown("""Let's assume we got two classes of fish we want to distinguish. One shall be Salmon, the other Sea bass.
Furthermore we make the simplified assumption, that you can distinguish those fish with ONE single feature, e.g. the length.""")
            

seabass_mean, seabass_var = 30, 10
salmon_mean, salmon_var = 80, 25
X_AXIS = np.arange(0,150, 0.002)

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

st.markdown("Furthermore we say we are at a point on earth where the prior probability $P(\omega)$ is equal for both fish (how lucky...)")

st.markdown("""Enough theory, let's catch a fish and measure it!""")
col1, col2 = st.columns(2)
with col1:
    st.markdown("...")
with col2:
    fish_url = requests.get("https://assets7.lottiefiles.com/packages/lf20_6wvpi7jz.json")  
    fish_json = dict()
    fish_json = fish_url.json()
    st_lottie(fish_json,
              height=200,  
          width=400,
          # speed of animation
          speed=1,  
          # means the animation will run forever like a gif, and not as a still image
          loop=True,  
          # quality of elements used in the animation, other values are "low" and "medium"
          quality='high',)
    
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

st.markdown("""Plotting the newly caught fish into our plot from beforehand, we probably directly see, that it might will be a salmon, but what is the probability given the length that it is going to be a salmon?
For this we calculate the so called :red[a posterior $P(\omega | x)$] using bayes theorem.
""")


st.markdown("---")

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


import plotly.graph_objects as go
import numpy as np

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
