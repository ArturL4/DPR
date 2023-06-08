from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
from plotly.subplots import make_subplots

sns.set_theme()


X_AXIS = np.arange(-20,20, 0.001)

mean_1 = st.slider("Mean of Orange graph", 0.25, -5.0, 5.0)
variance_1 = st.slider("Variance of orange graph", 2.0, 2.0, 6.0)
h_1 = norm.pdf(X_AXIS,mean_1, variance_1)

mean_fixed = 5.0
variance_fixed = 1.5
h_2 = norm.pdf(X_AXIS, mean_fixed, variance_fixed)

fig, ax = plt.subplots()
ax.plot(X_AXIS,h_1, c="orange")
ax.plot(X_AXIS, h_2, c="b")
idx = np.argwhere(np.diff(np.sign(h_2 - h_1))).flatten()
if len(idx) <=1:
    idx = np.append(idx, 31511)

max_val = np.max(np.maximum(h_1, h_2))
plt.fill_between(X_AXIS[idx[0]:idx[1]],h_1[idx[0]:idx[1]], h_2[idx[0]:idx[1]], hatch='\\', alpha=0.2, linewidth=0.1, facecolor="b", edgecolor="b")
plt.fill_between(X_AXIS[idx[0]:idx[1]],np.zeros_like(h_1[idx[0]:idx[1]]), h_1[idx[0]:idx[1]], hatch='\\', alpha=0.2, linewidth=0.1, facecolor="orange", edgecolor="orange")

plt.legend([r"$H_0$", r"$H_1$", r"$P_{D}$", r"$P_{FA}$"])
ax.vlines(X_AXIS[idx], -.01, max_val, color="r", linestyles="dashed", linewidth=0.3)
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
            y=norm.pdf(np.arange(0, 10, 0.01), 4, 1.4))

trace2 = go.Scatter(
        name="Target",
        line=dict(width=3),
        x=np.arange(0, 10, 0.01),
        y=norm.pdf(np.arange(0, 10, 0.01), 7, 1))

fig.add_trace(trace1)
    
fig.add_trace(trace2)

'''
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

'''
fig.update_layout()
st.write(fig)

