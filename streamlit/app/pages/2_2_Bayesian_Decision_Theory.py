from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


X_AXIS = np.arange(-20,20, 0.001)
with st.echo(code_location='below'):
    mean_1 = st.slider("Mean of Orange graph", 0.25, -5.0, 5.0)
    variance_1 = st.slider("Variance of orange graph", 0.1, 1.0, 10.0)
    h_1 = norm.pdf(X_AXIS,mean_1, variance_1)

    mean_fixed = 5.0
    variance_fixed = 2.0
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
    ax.vlines(X_AXIS[idx], -.01, max_val, color="r", linestyles="dashed", linewidth=0.3)
    st.write(fig)

